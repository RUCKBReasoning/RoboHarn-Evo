# ruff: noqa: E402 -- direct script execution bootstraps the repository root.

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from roboharn_evo.agent.hpk.candidate_features import candidate_semantic_features
from roboharn_evo.resources import config_path
from roboharn_evo.agent.hpk.episode_finalizer import (
    HPKEpisodeFinalizer,
    SnapshotRef,
    TransitionStrategyContext,
)
from roboharn_evo.agent.hpk.policy_config import LoadedHPKPolicy, load_policy_config, evolving_geometry_policy_path
from roboharn_evo.agent.hpk.promotion import PromotionPolicyV1
from roboharn_evo.agent.hpk.retriever import HPKRetriever
from roboharn_evo.agent.hpk.rollout_importer import ROLLOUT_IMPORT_MANIFEST_V2_SCHEMA
from roboharn_evo.agent.hpk.runtime_binding import build_runtime_binding
from roboharn_evo.agent.hpk.schemas import (
    CONDITION_ABSTRACTION_VERSION,
    CONDITION_SCHEMA,
    TASK_STRATEGY_SCHEMA,
    ABSTRACT_EFFECT_SCHEMA,
    HPKUnresolved,
    AbstractEffectV1,
    ConditionV1,
    TaskStrategyV1,
    canonical_json_bytes,
    stable_content_id,
)
from roboharn_evo.agent.hpk.sequential_coordinator import SequentialHPKCoordinator
from roboharn_evo.agent.hpk.snapshot_publisher import (
    publish_child_snapshot,
    publish_directory_noreplace,
)
from roboharn_evo.agent.hpk.strategy_extractor import extract_baseline_strategy
from roboharn_evo.agent.hpk.transition_runtime import (
    ACTION_ATTEMPT_PREPARED_EVENT,
    ACTION_EFFECT_TRANSITION_EVENT,
    ACTION_TOOL_BOUNDARY_EVENT,
    ACTION_TOOL_RESULT_BUNDLE_EVENT,
    begin_evolving_action_transition,
)
from roboharn_evo.agent.hpk.updater import updater_policy_identity
from roboharn_evo.agent.operation_candidates import select_operation_pose_candidate
from roboharn_evo.agent.recovery.recovery_adapter import RecoveryExecutionResult
from roboharn_evo.agent.recovery.tool_dispatcher import RecoveryToolDispatcher
from roboharn_evo.agent.recovery.tool_specs import RecoveryToolCall, RecoveryToolResult


ARTIFACT_SCHEMA = "roboharn_evo/hpk/p0de_cpu_cross_rollout_artifact/v1"
LINEAGE_SCHEMA = "roboharn_evo/hpk/p0de_cpu_cross_rollout_lineage/v1"
DETERMINISM_SCHEMA = "roboharn_evo/hpk/p0de_cpu_determinism/v1"
RUNTIME_SCHEMA = "roboharn_evo/hpk/evolving_runtime/v1"
FROZEN_CREATED_AT_K0 = "2026-08-21T00:00:00Z"
FROZEN_CREATED_AT_K1 = "2026-08-21T00:01:00Z"
GEOMETRY_POLICY_RAW_SHA256 = (
    "61f18b6103dc4b89c38d83d852db2aac5b6585a02415988e73d81a08e119608d"
)
DEV_PROMOTION_POLICY_RAW_SHA256 = (
    "972449f06850a8dac415ac9ffa9c53ae5e4a91010082ed53fcda5c3acd7ee756"
)

DEFAULT_OUTPUT_DIR = _REPO_ROOT / "eval_result" / "hpk" / "p0_de" / "cpu_cross_rollout"


class CPUArtifactError(RuntimeError):
    """The deterministic artifact could not be built without weakening a gate."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_file_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(value)) + b"\n"


def _write_fsynced(path: Path, value: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(value)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    directories.extend(path for path in root.rglob("*") if path.is_dir())
    for directory in sorted(
        directories, key=lambda value: len(value.parts), reverse=True
    ):
        _fsync_directory(directory)


def _load_policies() -> tuple[LoadedHPKPolicy, LoadedHPKPolicy, PromotionPolicyV1]:
    geometry = load_policy_config(
        evolving_geometry_policy_path(),
        GEOMETRY_POLICY_RAW_SHA256,
        "geometry",
    )
    loaded_promotion = load_policy_config(
        config_path("hpk_promotion_policy_integration_dev_v1.yaml"),
        DEV_PROMOTION_POLICY_RAW_SHA256,
        "promotion",
    )
    promotion = PromotionPolicyV1.from_mapping(loaded_promotion.payload)
    if promotion["development_only"] is not True:
        raise CPUArtifactError("CPU artifact requires development_only=true")
    if promotion["formal_evaluation_eligible"] is not False:
        raise CPUArtifactError("CPU artifact requires formal_evaluation_eligible=false")
    return geometry, loaded_promotion, promotion


def _policy_refs(
    geometry: LoadedHPKPolicy,
    loaded_promotion: LoadedHPKPolicy,
    promotion: PromotionPolicyV1,
) -> dict[str, Any]:
    return {
        "updater": updater_policy_identity(promotion),
        "promotion": {
            **loaded_promotion.identity(),
            "payload": loaded_promotion.payload,
        },
        "geometry": geometry.identity(),
    }


def _runtime_source_identity() -> dict[str, Any]:
    config_sha = _sha256(b"generic-cpu-runtime-config/v1")
    model_sha = _sha256(b"no-model-cpu-fixture/v1")
    runtime_sha = _sha256(b"roboharn-hpk-cpu-cross-rollout-source/v1")
    provenance_schema = "roboharn_evo/runtime_provenance/content/v1"
    provenance_manifest_sha = _sha256(
        canonical_json_bytes(
            {
                "schema": provenance_schema,
                "effective_config_sha256": config_sha,
                "agent_service_identity_sha256": model_sha,
                "runtime_tree_sha256": runtime_sha,
            }
        )
    )
    return {
        "runtime_identity": {
            "schema": provenance_schema,
            "manifest_sha256": provenance_manifest_sha,
            "effective_config_sha256": config_sha,
            "agent_service_identity_sha256": model_sha,
            "component": "hpk_cpu_cross_rollout",
            "version": "v1",
        },
        "source_identity": {
            "runtime_tree_sha256": runtime_sha,
            "component": "roboharn_agent_hpk",
            "fixture": "generic_principal_axis_grasp",
        },
    }


def _semantics() -> tuple[ConditionV1, TaskStrategyV1, AbstractEffectV1]:
    condition = ConditionV1.from_dict(
        {
            "schema": CONDITION_SCHEMA,
            "task_family": "generic_object_manipulation",
            "manipulation_phase": "object_acquisition",
            "operation": "grasp",
            "manipulated_object": {
                "semantic_class": "rigid_object",
                "geometry_class": "compact_rigid_object",
                "role": "observed_manipulated_object",
                "held_state": "not_held",
            },
            "target": {
                "semantic_class": None,
                "geometry_class": None,
                "role": None,
                "relation": None,
            },
            "scene_predicates": [],
            "preconditions": ["manipulated_object_observed"],
            "abstraction_version": CONDITION_ABSTRACTION_VERSION,
        }
    )
    task = TaskStrategyV1.from_dict(
        {
            "schema": TASK_STRATEGY_SCHEMA,
            "operation": "grasp",
            "manipulated_role": "observed_manipulated_object",
            "target_role": None,
            "target_relation": None,
            "manipulation_phase": "object_acquisition",
            "subgoal_purpose": "establish a verified object attachment",
            "preferred_arm": "either",
            "source": {
                "planner_subtask_text": "grasp the observed rigid object",
                "selected_skill": "monitored-object-acquisition",
                "action_mode": "grasp",
                "normalization_version": "task_strategy_normalizer/v1",
            },
        }
    )
    effect = AbstractEffectV1.from_dict(
        {
            "schema": ABSTRACT_EFFECT_SCHEMA,
            "effect_type": "grasp",
            "expected_predicates": ["object_attached"],
            "verifiability": "verified",
        }
    )
    return condition, task, effect


def _candidate(*, source_index: int, priority: float) -> dict[str, Any]:
    return {
        "candidate_id": f"private-axis-{source_index}",
        "source_candidate_index": source_index,
        "action_mode": "grasp",
        "arm": "left",
        "ee_target_pose": [0.0, 0.0, 0.80, 1.0, 0.0, 0.0, 0.0],
        "approach_pose": [0.0, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0],
        "geometry_source": "rgbd_observed_volume_principal_axes",
        "observed_grasp_width_m": 0.04,
        "reach_distance_m": 0.25,
        "priority": priority,
    }


def _instance(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "instance_id": "private-runtime-object",
        "operation_pose_candidates": [dict(value) for value in candidates],
    }


class _SyntheticSnapshot:
    def __init__(self, step_count: int) -> None:
        self.step_count = step_count
        self.step_limit = 20
        self.eval_success = False
        self.check_success = False
        self.max_reward = 0.0
        self.joint_vector = (0.0, 1.0)
        self.left_endpose = (0.0, 0.0, 0.8)
        self.right_endpose = (0.1, 0.0, 0.8)


class _SyntheticExecutor:
    def execute(self, *, call, task_env, latest_snapshot):
        del task_env
        snapshot = _SyntheticSnapshot(int(latest_snapshot.step_count) + 1)
        return RecoveryExecutionResult(
            result=RecoveryToolResult(
                tool_name=str(call.tool_name),
                success=True,
                message="ok",
                details={
                    "target_reached": True,
                    "step_count": snapshot.step_count,
                },
            ),
            latest_snapshot=snapshot,
        )

    def capabilities(self, task_env):  # pragma: no cover - unused integration seam
        raise CPUArtifactError(f"unexpected capabilities request: {task_env!r}")

    def set_reobserve_scene_enabled(self, enabled):
        del enabled


class _SyntheticEvolvingRuntime:
    evolving_enabled = True

    def __init__(self, binding: Mapping[str, Any]) -> None:
        self._binding = json.loads(canonical_json_bytes(binding).decode("utf-8"))
        self.accepted_transition_id: str | None = None

    def prepare_action_transition(self, **payload):
        return {
            **json.loads(canonical_json_bytes(self._binding).decode("utf-8")),
            "action_attempt_nonce": payload["action_attempt_nonce"],
        }

    def record_dispatch_observation(self, payload):
        del payload

    def accept_action_transition(self, **payload):
        self.accepted_transition_id = payload["transition"].stable_id


def _support_transition(
    *,
    private_episode_id: int,
    candidate: Mapping[str, Any],
    geometry_policy: LoadedHPKPolicy,
    runtime_binding_id: str,
) -> tuple[
    Any,
    TransitionStrategyContext,
    tuple[tuple[str, dict[str, Any]], ...],
    tuple[tuple[str, dict[str, Any]], ...],
]:
    condition, task, effect = _semantics()
    features = candidate_semantic_features(
        _instance([candidate]), candidate, geometry_policy=geometry_policy
    )
    if isinstance(features, HPKUnresolved):
        raise CPUArtifactError(
            f"candidate feature extraction failed: {features.reason}"
        )
    extracted = extract_baseline_strategy(
        condition=condition,
        task_strategy=task,
        candidate_features=features,
        expected_effect=effect,
    )
    if isinstance(extracted, HPKUnresolved):
        raise CPUArtifactError(f"strategy extraction failed: {extracted.reason}")
    attempt_nonce = stable_content_id(
        "afkattempt", {"fixture": "generic_cpu_update_attempt"}
    )
    binding = {
        "condition_id": condition.stable_id,
        "task_strategy_id": task.stable_id,
        "retrieved_hpk_entry_ids": [],
        "selected_hpk_entry_id": None,
        # This is a baseline transition.  The finalizer may resolve only this
        # absent z from the independently supplied typed strategy context.
        "geometric_strategy_id": None,
        "operation": "grasp",
        "arm": "left",
        "target_role": None,
        "target_relation": None,
        "selected_candidate_private_ref": str(candidate["candidate_id"]),
        "candidate_geometry_features": features.to_dict(),
        "geometric_compliance": True,
        "realization_status": "satisfied",
        "target_identity_status": "bound",
        "expected_effect": effect.to_dict(),
        "verifier_conflicts": [],
        "oracle_derived": False,
        "expert_derived": False,
        "_hpk_runtime_binding_id": runtime_binding_id,
    }
    private_events: list[tuple[str, dict[str, Any]]] = []
    public_events: list[tuple[str, dict[str, Any]]] = []
    runtime = _SyntheticEvolvingRuntime(binding)
    calls = [
        RecoveryToolCall(
            "close_gripper",
            {"arm": "left", "action_mode": "grasp"},
        )
    ]
    snapshot_before = _SyntheticSnapshot(1)
    session = begin_evolving_action_transition(
        hpk_runtime=runtime,
        episode_id=private_episode_id,
        seed=-1,
        calls=calls,
        pre_effect_state={"object_attached": False},
        snapshot_before=snapshot_before,
        context={"subgoal_purpose": task["subgoal_purpose"]},
        private_sink=lambda event, payload: private_events.append(
            (event, json.loads(canonical_json_bytes(payload).decode("utf-8")))
        ),
        public_sink=lambda event, payload: public_events.append(
            (event, json.loads(canonical_json_bytes(payload).decode("utf-8")))
        ),
        nonce_factory=lambda: attempt_nonce,
    )
    if session is None:
        raise CPUArtifactError("synthetic physical transition session was not created")
    RecoveryToolDispatcher(executor=_SyntheticExecutor()).dispatch_batch(
        calls,
        task_env=object(),
        latest_snapshot=snapshot_before,
        attempt_observer=session.observe_dispatch,
    )
    transition = session.finalize(
        action_effect={
            "effect_verified": "true",
            "effect_type": "grasp",
            "runtime_grasp_validation": {
                "applicable": True,
                "verified": True,
                "attachment_verified": True,
            },
        },
        post_effect_state={"object_attached": True},
    )
    if runtime.accepted_transition_id != transition.stable_id:
        raise CPUArtifactError("synthetic runtime did not accept its exact transition")
    context = TransitionStrategyContext.from_values(
        condition=condition,
        task_strategy=task,
        scene_signature=_sha256(
            canonical_json_bytes(
                {
                    "operation": "grasp",
                    "scene": "single_observed_rigid_object",
                }
            )
        ),
    )
    return transition, context, tuple(private_events), tuple(public_events)


def _write_v2_rollout(
    root: Path,
    *,
    private_episode_id: int,
    transition: Any,
    private_events: Sequence[tuple[str, Mapping[str, Any]]],
    public_transition_events: Sequence[tuple[str, Mapping[str, Any]]],
    runtime_binding: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, str]]:
    expected_private_order = (
        ACTION_ATTEMPT_PREPARED_EVENT,
        ACTION_TOOL_BOUNDARY_EVENT,
        ACTION_TOOL_BOUNDARY_EVENT,
        ACTION_TOOL_RESULT_BUNDLE_EVENT,
        ACTION_EFFECT_TRANSITION_EVENT,
    )
    if tuple(event for event, _payload in private_events) != expected_private_order:
        raise CPUArtifactError("synthetic runtime emitted an incomplete causal chain")
    if tuple(event for event, _payload in public_transition_events) != (
        "hpk_action_effect_transition_public",
    ):
        raise CPUArtifactError("synthetic runtime emitted an invalid public projection")
    rollout = root / "rollout"
    rollout.mkdir(mode=0o700)
    public_path = rollout / "public_trace.jsonl"
    private_path = rollout / "private_transition_trace.jsonl"
    public_records = [
        {
            "event": "episode_start",
            "episode_id": private_episode_id,
            "timestamp": 0.0,
            "seed": -1,
            "env_step": -1,
            "runtime_binding": dict(runtime_binding),
        },
        *[
            {
                "event": event,
                "episode_id": private_episode_id,
                "timestamp": 0.25 + index * 0.01,
                "seed": -1,
                "env_step": transition["env_step_after"],
                **dict(payload),
            }
            for index, (event, payload) in enumerate(public_transition_events)
        ],
        {
            "event": "episode_end",
            "episode_id": private_episode_id,
            "timestamp": 1.0,
            "seed": -1,
            "env_step": 3,
            "result": "Success",
            "natural_episode_end": True,
            "episode_validity": {
                "label": "valid_success",
                "reason": "valid natural CPU integration attempt",
            },
        },
    ]
    public_bytes = b"".join(
        canonical_json_bytes(value) + b"\n" for value in public_records
    )

    def event_step(event: str, payload: Mapping[str, Any]) -> int:
        if event == ACTION_ATTEMPT_PREPARED_EVENT:
            snapshot = payload.get("snapshot_before")
            return int(snapshot["step_count"]) if isinstance(snapshot, Mapping) else 0
        if event == ACTION_TOOL_BOUNDARY_EVENT:
            snapshot = payload.get("snapshot_after") or payload.get("snapshot_before")
            return int(snapshot["step_count"]) if isinstance(snapshot, Mapping) else 0
        if event in {ACTION_TOOL_RESULT_BUNDLE_EVENT, ACTION_EFFECT_TRANSITION_EVENT}:
            return int(transition["env_step_after"])
        raise CPUArtifactError(f"unexpected private causal event: {event}")

    private_records = [
        {
            "event": event,
            "episode_id": private_episode_id,
            "timestamp": 0.10 + index * 0.01,
            "seed": -1,
            "env_step": event_step(event, payload),
            **dict(payload),
        }
        for index, (event, payload) in enumerate(private_events)
    ]
    private_bytes = b"".join(
        canonical_json_bytes(value) + b"\n" for value in private_records
    )
    _write_fsynced(public_path, public_bytes)
    _write_fsynced(private_path, private_bytes)
    manifest = {
        "schema": ROLLOUT_IMPORT_MANIFEST_V2_SCHEMA,
        "rollout_id": "generic-cpu-cross-rollout-update",
        "episode_id": private_episode_id,
        "public_trace": {
            "path": public_path.name,
            "sha256": _sha256(public_bytes),
        },
        "private_transition_trace": {
            "path": private_path.name,
            "sha256": _sha256(private_bytes),
        },
        "runtime_binding": dict(runtime_binding),
        "information_access": {
            "oracle_derived": False,
            "expert_derived": False,
        },
    }
    manifest_path = rollout / "rollout_manifest.json"
    manifest_bytes = _canonical_file_bytes(manifest)
    _write_fsynced(manifest_path, manifest_bytes)
    _fsync_directory(rollout)
    return (
        manifest_path,
        public_path,
        {
            "manifest_sha256": _sha256(manifest_bytes),
            "public_trace_sha256": _sha256(public_bytes),
            "private_transition_trace_sha256": _sha256(private_bytes),
            "runtime_binding_id": str(runtime_binding["binding_id"]),
        },
    )


def _capabilities() -> dict[str, Any]:
    return {
        "task_families": ["generic_object_manipulation"],
        "domain_ids": ["generic_cpu_domain"],
        "operations": ["grasp"],
        "strategy_families": ["observed_grasp_geometry"],
        "target_relations": [],
        "hard_constraints": [],
        "geometry_source_classes": ["rgbd_observed"],
        "semantic_part_observed": False,
    }


def _snapshot_member_report(snapshot_ref: SnapshotRef) -> dict[str, Any]:
    manifest = snapshot_ref.snapshot.manifest
    return {
        "snapshot_id": snapshot_ref.snapshot_id,
        "manifest_sha256": snapshot_ref.manifest_sha256,
        "members": {
            name: {
                "path": descriptor["path"],
                "sha256": descriptor["sha256"],
                "size_bytes": descriptor["size_bytes"],
                "record_count": descriptor["record_count"],
            }
            for name, descriptor in sorted(manifest["members"].items())
        },
    }


def _tree_inventory(
    root: Path, *, exclude: frozenset[str] = frozenset()
) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        result[relative] = _sha256(path.read_bytes())
    return result


def _build_deterministic_run(root: Path) -> dict[str, Any]:
    geometry, loaded_promotion, promotion = _load_policies()
    policy_refs = _policy_refs(geometry, loaded_promotion, promotion)
    identities = _runtime_source_identity()
    snapshot_root = (root / "snapshots").resolve()
    k0_published = publish_child_snapshot(
        parent=None,
        expected_parent_snapshot_id=None,
        expected_parent_manifest_sha256=None,
        entries=(),
        policy_refs=policy_refs,
        evidence_batch=None,
        source_episode_ids=(),
        created_at=FROZEN_CREATED_AT_K0,
        destination_root=snapshot_root,
        runtime_source_identity=identities,
        purpose="evolving_integration",
    )
    k0 = SnapshotRef.from_published(k0_published)
    finalizer = HPKEpisodeFinalizer(
        promotion_policy=promotion,
        policy_refs=policy_refs,
        destination_root=snapshot_root,
        runtime_source_identity=identities,
        domain_ids=["generic_cpu_domain"],
        config_sha256=identities["runtime_identity"]["effective_config_sha256"],
        model_identity_sha256=identities["runtime_identity"][
            "agent_service_identity_sha256"
        ],
        runtime_source_sha256=identities["source_identity"]["runtime_tree_sha256"],
        runtime_schema=RUNTIME_SCHEMA,
    )
    runtime_binding = build_runtime_binding(
        snapshot_id=k0.snapshot_id,
        snapshot_manifest_sha256=k0.manifest_sha256,
        policy_refs=policy_refs,
        provenance_schema=identities["runtime_identity"]["schema"],
        provenance_manifest_sha256=identities["runtime_identity"]["manifest_sha256"],
        config_sha256=identities["runtime_identity"]["effective_config_sha256"],
        model_identity_sha256=identities["runtime_identity"][
            "agent_service_identity_sha256"
        ],
        runtime_source_sha256=identities["source_identity"]["runtime_tree_sha256"],
        evidence_runtime_schema=RUNTIME_SCHEMA,
    )
    coordinator = SequentialHPKCoordinator(
        initial_snapshot_ref=k0,
        finalizer=finalizer,
    )

    private_episode_id = 0
    update_lease = coordinator.begin_episode(private_episode_id)
    if update_lease.snapshot_ref.identity() != k0.identity():
        raise CPUArtifactError("update episode did not pin K0")
    update_candidates = [
        _candidate(source_index=0, priority=0.01),
        _candidate(source_index=1, priority=0.03),
    ]
    baseline_update = select_operation_pose_candidate(
        _instance(update_candidates), arm="left", action_mode="grasp"
    )
    if baseline_update is None or baseline_update["source_candidate_index"] != 0:
        raise CPUArtifactError("frozen update baseline did not select axis 0")
    transition, context, private_events, public_transition_events = _support_transition(
        private_episode_id=private_episode_id,
        candidate=baseline_update,
        geometry_policy=geometry,
        runtime_binding_id=str(runtime_binding["binding_id"]),
    )
    manifest_path, public_trace_path, rollout_hashes = _write_v2_rollout(
        root,
        private_episode_id=private_episode_id,
        transition=transition,
        private_events=private_events,
        public_transition_events=public_transition_events,
        runtime_binding=runtime_binding,
    )
    finalized = coordinator.finalize_episode(
        update_lease,
        manifest_path=manifest_path,
        trace_path=public_trace_path,
        expected_manifest_sha256=rollout_hashes["manifest_sha256"],
        expected_trace_sha256=rollout_hashes["public_trace_sha256"],
        transition_contexts={transition.stable_id: context},
        created_at=FROZEN_CREATED_AT_K1,
    )
    if not finalized.advanced or finalized.finalization.child is None:
        raise CPUArtifactError(
            "valid update episode did not publish K1: "
            + json.dumps(
                finalized.finalization.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    k1 = finalized.next_snapshot_ref
    if len(k1.snapshot.accepted_entries) != 1:
        raise CPUArtifactError("development policy did not accept exactly one entry")
    learned_entry = k1.snapshot.accepted_entries[0]

    probe_private_episode_id = 1
    probe_lease = coordinator.begin_episode(probe_private_episode_id)
    if probe_lease.snapshot_ref.identity() != k1.identity():
        raise CPUArtifactError("next episode did not automatically pin K1")
    retriever = HPKRetriever(
        probe_lease.snapshot_ref.snapshot.accepted_entries,
        snapshot_capabilities=_capabilities(),
    )
    retrieval = retriever.retrieve_geometry(
        context.condition,
        context.task_strategy,
        current_capabilities=_capabilities(),
        target_bound=True,
        operation_hint="grasp",
    )
    if retrieval.selected_entry_id != learned_entry["entry_id"]:
        raise CPUArtifactError("probe did not retrieve the new K1 entry")
    if retrieval.geometric_strategy is None:
        raise CPUArtifactError("probe retrieval has no geometric strategy")

    probe_candidates = [
        _candidate(source_index=0, priority=0.03),
        _candidate(source_index=1, priority=0.01),
    ]
    probe_instance = _instance(probe_candidates)
    baseline_probe = select_operation_pose_candidate(
        probe_instance, arm="left", action_mode="grasp"
    )
    ranking_audit: dict[str, Any] = {}
    learned_probe = select_operation_pose_candidate(
        probe_instance,
        arm="left",
        action_mode="grasp",
        geometric_strategy=retrieval.geometric_strategy,
        geometry_policy=geometry,
        ranking_audit=ranking_audit,
    )
    if baseline_probe is None or learned_probe is None:
        raise CPUArtifactError(
            "probe candidate selection unexpectedly returned no action"
        )
    if baseline_probe not in probe_instance["operation_pose_candidates"]:
        raise CPUArtifactError("baseline probe selected an illegal candidate")
    if learned_probe not in probe_instance["operation_pose_candidates"]:
        raise CPUArtifactError("HPK probe selected an illegal candidate")
    if baseline_probe["source_candidate_index"] != 1:
        raise CPUArtifactError("frozen probe baseline did not select axis 1")
    if learned_probe["source_candidate_index"] != 0:
        raise CPUArtifactError("K1 HPK did not select its learned axis 0 candidate")
    if baseline_probe["candidate_id"] == learned_probe["candidate_id"]:
        raise CPUArtifactError("K1 did not change the legal candidate selection")

    lineage = {
        "schema": LINEAGE_SCHEMA,
        "formal_evaluation": False,
        "development_integration": True,
        "k0": _snapshot_member_report(k0),
        "update": {
            "rollout_manifest_sha256": rollout_hashes["manifest_sha256"],
            "public_trace_sha256": rollout_hashes["public_trace_sha256"],
            "private_transition_trace_sha256": rollout_hashes[
                "private_transition_trace_sha256"
            ],
            "runtime_binding_id": rollout_hashes["runtime_binding_id"],
            "public_episode_group_id": finalized.finalization.episode_id,
            "transition_id": transition.stable_id,
            "evidence_ids": list(finalized.finalization.evidence_ids),
            "decision_ids": list(finalized.finalization.decision_ids),
            "entry_ids": list(finalized.finalization.updated_entry_ids),
        },
        "k1": _snapshot_member_report(k1),
        "next_episode": {
            "loaded_snapshot_id": probe_lease.snapshot_ref.snapshot_id,
            "loaded_manifest_sha256": probe_lease.snapshot_ref.manifest_sha256,
            "retrieved_entry_id": retrieval.selected_entry_id,
            "selected_geometric_strategy_id": retrieval.geometric_strategy.stable_id,
            "baseline_choice_relation": "align_principal_axis_1",
            "hpk_choice_relation": "align_principal_axis_0",
            "baseline_candidate_legal": True,
            "hpk_candidate_legal": True,
            "candidate_selection_changed": True,
            "geometric_compliance": ranking_audit["geometric_compliance"],
        },
        "policy": {
            "refs": policy_refs,
            "development_only": promotion["development_only"],
            "formal_evaluation_eligible": promotion["formal_evaluation_eligible"],
        },
    }
    _write_fsynced(root / "lineage_report.json", _canonical_file_bytes(lineage))

    core_inventory = _tree_inventory(root)
    artifact_payload = {
        "schema": ARTIFACT_SCHEMA,
        "created_at": FROZEN_CREATED_AT_K1,
        "claim_scope": "deterministic_cpu_integration_only",
        "formal_evaluation": False,
        "development_integration": True,
        "member_sha256": core_inventory,
        "lineage_report_sha256": core_inventory["lineage_report.json"],
        "k0_snapshot_id": k0.snapshot_id,
        "k0_manifest_sha256": k0.manifest_sha256,
        "k1_snapshot_id": k1.snapshot_id,
        "k1_manifest_sha256": k1.manifest_sha256,
        "public_episode_group_id": finalized.finalization.episode_id,
        "learned_entry_id": learned_entry["entry_id"],
        "candidate_selection_changed": True,
        "policy_refs": policy_refs,
    }
    artifact_payload["artifact_content_sha256"] = _sha256(
        canonical_json_bytes(artifact_payload)
    )
    _write_fsynced(
        root / "artifact_manifest.json", _canonical_file_bytes(artifact_payload)
    )
    _fsync_tree(root)
    return artifact_payload


def build_cpu_cross_rollout_artifact(
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build twice, compare every deterministic byte, then atomically publish."""

    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        raise CPUArtifactError("output_dir must be explicit and absolute")
    if destination.exists() or destination.is_symlink():
        raise CPUArtifactError(f"refusing to clobber existing artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = destination.parent.resolve(strict=True)
    staging = Path(tempfile.mkdtemp(prefix=".hpk-p0de-primary-", dir=parent))
    verification = Path(tempfile.mkdtemp(prefix=".hpk-p0de-verify-", dir=parent))
    try:
        primary_manifest = _build_deterministic_run(staging)
        verification_manifest = _build_deterministic_run(verification)
        primary_inventory = _tree_inventory(staging)
        verification_inventory = _tree_inventory(verification)
        if primary_manifest != verification_manifest:
            raise CPUArtifactError("repeated builds changed artifact manifest content")
        if primary_inventory != verification_inventory:
            raise CPUArtifactError("repeated builds changed deterministic file hashes")
        inventory_sha256 = _sha256(canonical_json_bytes(primary_inventory))
        determinism = {
            "schema": DETERMINISM_SCHEMA,
            "build_count": 2,
            "byte_identical": True,
            "inventory_sha256": inventory_sha256,
            "file_count": len(primary_inventory),
        }
        _write_fsynced(
            staging / "determinism_report.json", _canonical_file_bytes(determinism)
        )
        _fsync_tree(staging)
        _fsync_directory(parent)
        published_tree_sha256 = publish_directory_noreplace(staging, destination)
        return {
            "output_dir": str(destination),
            "artifact_content_sha256": primary_manifest["artifact_content_sha256"],
            "deterministic_inventory_sha256": inventory_sha256,
            "published_tree_sha256": published_tree_sha256,
            "k0_snapshot_id": primary_manifest["k0_snapshot_id"],
            "k0_manifest_sha256": primary_manifest["k0_manifest_sha256"],
            "k1_snapshot_id": primary_manifest["k1_snapshot_id"],
            "k1_manifest_sha256": primary_manifest["k1_manifest_sha256"],
            "public_episode_group_id": primary_manifest["public_episode_group_id"],
            "learned_entry_id": primary_manifest["learned_entry_id"],
            "candidate_selection_changed": True,
            "formal_evaluation": False,
            "development_integration": True,
        }
    finally:
        if verification.exists():
            shutil.rmtree(verification, ignore_errors=True)
        if staging != destination and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic HPK P0-D/E CPU cross-rollout fixture.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="absolute no-clobber artifact directory",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = build_cpu_cross_rollout_artifact(args.output_dir.absolute())
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
