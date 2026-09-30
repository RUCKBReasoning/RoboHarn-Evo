"""Episode-pinned HPK runtime for the P0-D/E evolving loop.

The runtime deliberately separates three boundaries:

* an immutable accepted-entry projection is pinned before an episode;
* private, typed candidate/transition context lives only in memory and trace;
* the coordinator may publish a child only from an explicit episode finalizer.

No retrieval call reopens a snapshot and a published child cannot affect the
episode that produced it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.action_transition import (
    ActionEffectTransitionV1,
    public_transition_projection,
)
from roboharn_evo.agent.hpk.audit import PrivateRankingAuditSink
from roboharn_evo.agent.hpk.candidate_features import candidate_semantic_features
from roboharn_evo.agent.hpk.episode_finalizer import (
    HPKEpisodeFinalizer,
    SnapshotRef,
    TransitionStrategyContext,
)
from roboharn_evo.agent.hpk.evolving_store import (
    LoadedEvolvingSnapshot,
    promotion_policy_from_ref,
)
from roboharn_evo.agent.hpk.geometry_policy import rank_operation_pose_candidates
from roboharn_evo.agent.hpk.policy_config import (
    SAFE_EXPLORATION_GEOMETRY_POLICY_CONFIG_SHA256,
    LoadedHPKPolicy,
)
from roboharn_evo.agent.hpk.promotion import PromotionPolicyV1
from roboharn_evo.agent.hpk.retriever import HPKDomainCapabilities, retrieve_semantic_knowledge
from roboharn_evo.agent.hpk.rollout_importer import ROLLOUT_IMPORT_MANIFEST_V2_SCHEMA
from roboharn_evo.agent.hpk.runtime_binding import (
    build_runtime_binding,
    validate_runtime_binding,
    validate_sequential_run_binding,
)
from roboharn_evo.agent.hpk.runtime_policy import HPKRuntimePolicy
from roboharn_evo.agent.hpk.safe_exploration import (
    SafeExplorationDecision,
    choose_safe_baseline_alternative,
)
from roboharn_evo.agent.hpk.schemas import (
    ABSTRACT_EFFECT_SCHEMA,
    AbstractEffectV1,
    HPKUnresolved,
    CandidateGeometryFeaturesV1,
    ConditionV1,
    EntryV1,
    GeometricStrategyV1,
    TaskStrategyV1,
    canonical_json_bytes,
    stable_content_id,
)
from roboharn_evo.agent.hpk.semantic_finalizer import (
    SemanticEpisodeFinalizer,
    SemanticFinalizationResult,
)
from roboharn_evo.agent.hpk.semantic_knowledge import (
    semantic_geometry_to_v1,
    semantic_object_from_mapping,
)
from roboharn_evo.agent.hpk.semantic_store import load_semantic_knowledge
from roboharn_evo.agent.hpk.sequential_coordinator import (
    CoordinatedEpisodeResult,
    EpisodeLease,
    SequentialHPKCoordinator,
)
from roboharn_evo.agent.hpk.static_runtime import (
    HPKBoundGeometryQuery,
    HPKGeometryDecision,
    HPKStaticRuntime,
)
from roboharn_evo.agent.hpk.strategy_extractor import (
    ExtractedHPKStrategy,
    expected_effect_for_operation,
    extract_baseline_strategy,
    strategy_key_id_for,
)
from roboharn_evo.agent.hpk.strategy_proposal import (
    PROPOSAL_RANK_TRACE_SCHEMA,
    PROPOSAL_EVIDENCE_PACKET_SCHEMA,
    ProposalEvidencePacketV1,
    StrategyProposalV1,
    abstract_effect_v1_to_v2,
    apply_geometry_delta,
    condition_v1_to_v2,
    geometric_strategy_v1_to_v2,
    geometric_strategy_v2_to_v1,
    rank_frozen_candidates_with_proposal,
    task_strategy_v1_to_v2,
)
from roboharn_evo.agent.hpk.transition_runtime import (
    PHYSICAL_RECOVERY_TOOLS,
    prove_semantic_attempt_group,
)
from roboharn_evo.agent.hpk.vlm_strategy_proposer import VLMStrategyProposer
from roboharn_evo.agent.recovery.tool_specs import RecoveryToolCall

EVOLVING_RUNTIME_SCHEMA = "roboharn_evo/agent/hpk/evolving_runtime/v1"


class HPKEvolvingRuntimeError(RuntimeError):
    """An episode pin, private binding, or once-only lifecycle was violated."""


@dataclass(frozen=True, slots=True)
class PublishedRolloutImportManifestV2:
    path: Path
    sha256: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PublishedFinalizationReceiptV1:
    path: Path
    sha256: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EvolvingRuntimeProvenance:
    manifest_sha256: str
    effective_config_sha256: str
    agent_service_identity_sha256: str
    runtime_tree_sha256: str
    schema: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "EvolvingRuntimeProvenance":
        if not isinstance(value, Mapping) or value.get("recorded") is not True:
            raise HPKEvolvingRuntimeError(
                "evolving HPK requires recorded content runtime provenance"
            )
        hashes: dict[str, str] = {}
        for key in (
            "manifest_sha256",
            "effective_config_sha256",
            "agent_service_identity_sha256",
            "runtime_tree_sha256",
        ):
            raw = value.get(key)
            if (
                not isinstance(raw, str)
                or len(raw) != 64
                or any(character not in "0123456789abcdef" for character in raw)
            ):
                raise HPKEvolvingRuntimeError(
                    f"evolving HPK runtime provenance has invalid {key}"
                )
            hashes[key] = raw
        schema = str(value.get("schema", "") or "").strip()
        if not schema:
            raise HPKEvolvingRuntimeError(
                "evolving HPK runtime provenance requires its schema"
            )
        return cls(schema=schema, **hashes)

    def runtime_source_identity(self) -> dict[str, Any]:
        return {
            "runtime_identity": {
                "schema": self.schema,
                "manifest_sha256": self.manifest_sha256,
                "effective_config_sha256": self.effective_config_sha256,
                "agent_service_identity_sha256": (self.agent_service_identity_sha256),
            },
            "source_identity": {
                "component": "roboharn_evo_agent_hpk",
                "runtime_tree_sha256": self.runtime_tree_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class _CandidateSelection:
    token: str
    condition: ConditionV1 | None
    task_strategy: TaskStrategyV1 | None
    semantic_object: dict[str, str]
    reasoning: dict[str, Any]
    candidate_features: CandidateGeometryFeaturesV1 | None
    expected_effect: AbstractEffectV1 | None
    scene_signature: str
    binding: dict[str, Any]
    extracted_strategy: ExtractedHPKStrategy | None
    selected_entry: EntryV1 | None
    proposal_capabilities: dict[str, Any]
    proposal_id: str | None = None
    proposed_geometric_strategy: GeometricStrategyV1 | None = None
    proposed_expected_effect: AbstractEffectV1 | None = None


@dataclass(frozen=True, slots=True)
class V2ProposalCandidateDecision:
    proposal: StrategyProposalV1
    selected_candidate: dict[str, Any] | None
    proposed_geometric_strategy: Any
    expected_effect: AbstractEffectV1
    rank_trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SemanticGeometryMatch:
    knowledge: dict[str, Any]
    geometric_strategy: GeometricStrategyV1


@dataclass(slots=True)
class _PendingV2Proposal:
    proposal: StrategyProposalV1
    source_transition_id: str


@dataclass(slots=True)
class _PreparedAttempt:
    nonce: str
    selection: _CandidateSelection | None
    binding: dict[str, Any]
    dispatch_observations: list[dict[str, Any]]
    accepted_transition_id: str | None = None


@dataclass(slots=True)
class _PendingSemanticAttempt:
    """Owner-only continuation state for one bounded semantic action."""

    nonce: str
    selection: _CandidateSelection
    operation: str
    arm: str
    segments: list[Any]
    env_step_started: int
    env_step_latest: int
    post_effect_state: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class _PendingSemanticVerification:
    candidate: dict[str, Any]
    operation: str
    arm: str
    candidate_ref: str


def _snapshot_policy_refs(snapshot: LoadedEvolvingSnapshot) -> dict[str, Any]:
    return dict(snapshot.manifest["policy_refs"])


def _scene_signature(
    *,
    condition: ConditionV1 | None,
    task_strategy: TaskStrategyV1 | None,
    candidate_features: CandidateGeometryFeaturesV1 | None,
) -> str:
    """Hash only transferable semantic state, never a seed/candidate identity."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "condition": None if condition is None else condition.to_dict(),
                "task_strategy": (
                    None if task_strategy is None else task_strategy.transferable_dict()
                ),
                "candidate_features": (
                    None if candidate_features is None else candidate_features.to_dict()
                ),
            }
        )
    ).hexdigest()


def _private_candidate_ref(value: Mapping[str, Any]) -> str:
    ref = str(value.get("candidate_id", "") or "").strip()
    if not ref:
        raise HPKEvolvingRuntimeError(
            "selected operation candidate lacks a private candidate_id"
        )
    return ref


def _proposal_capabilities(
    candidates: Sequence[Mapping[str, Any]],
    *,
    scene_state: Mapping[str, Any],
    geometry_policy: LoadedHPKPolicy,
) -> dict[str, Any]:
    """Derive only alternatives that exist in the current legal candidate set."""

    observed: list[CandidateGeometryFeaturesV1] = []
    for candidate in candidates:
        features = candidate_semantic_features(
            scene_state,
            candidate,
            geometry_policy=geometry_policy,
        )
        if not isinstance(features, HPKUnresolved):
            observed.append(features)

    def values(field: str, *, exclude: set[Any] | None = None) -> list[str]:
        blocked = exclude or set()
        return sorted(
            {
                str(item[field])
                for item in observed
                if item[field] is not None and item[field] not in blocked
            }
        )

    geometry_fields: dict[str, list[str]] = {}
    candidates_by_field = {
        "approach_family": values("approach_family", exclude={"unknown"}),
        "approach_direction": values("approach_direction_bucket", exclude={"unknown"}),
        "orientation_relation": values("orientation_relation", exclude={"unknown"}),
        "grasp_region": values("grasp_region", exclude={"unknown"}),
        "semantic_part": values("semantic_part"),
        "placement_relation": values("target_relation"),
    }
    for field, field_values in candidates_by_field.items():
        if field_values:
            geometry_fields[field] = field_values
    hard_constraints: list[str] = []
    if any(item["support_valid"] is True for item in observed):
        hard_constraints.append("support_valid")
    if any(item["target_region_free"] is True for item in observed):
        hard_constraints.append("target_region_free")
    if hard_constraints:
        geometry_fields["add_hard_constraints"] = sorted(hard_constraints)
    geometry_fields["add_avoid"] = ["repeat_equivalent_failed_candidate"]
    return {"task_fields": {}, "geometry_fields": geometry_fields}


def _has_non_equivalent_candidate_capability(
    *,
    base_geometry: Mapping[str, Any],
    capabilities: Mapping[str, Any],
) -> bool:
    fields = capabilities.get("geometry_fields")
    if not isinstance(fields, Mapping):
        return False
    base_by_field = {
        "approach_family": base_geometry.get("approach_family"),
        "approach_direction": base_geometry.get("approach_direction"),
        "orientation_relation": base_geometry.get("orientation_relation"),
        "grasp_region": base_geometry.get("grasp_region"),
        "semantic_part": base_geometry.get("semantic_part"),
        "placement_relation": base_geometry.get("placement_relation"),
    }
    return any(
        isinstance(fields.get(field), list)
        and any(value != base for value in fields[field])
        for field, base in base_by_field.items()
    )


def _unbound_transition_binding(*, conflict: str) -> dict[str, Any]:
    return {
        "condition_id": None,
        "task_strategy_id": None,
        "retrieved_hpk_entry_ids": [],
        "selected_hpk_entry_id": None,
        "geometric_strategy_id": None,
        "operation": None,
        "arm": None,
        "target_role": None,
        "target_relation": None,
        "selected_candidate_private_ref": None,
        "candidate_geometry_features": None,
        "geometric_compliance": "unverified",
        "realization_status": "unknown",
        "target_identity_status": "unknown",
        "expected_effect": None,
        "verifier_conflicts": [conflict],
        "oracle_derived": False,
        "expert_derived": False,
        "semantic_attempt_group": {
            "grouped": False,
            "physical_dispatch_indices": [],
        },
    }


_SEMANTIC_ATTEMPT_TOOLS = {
    "grasp": frozenset(
        {
            "move_ee_to_grounded_instance",
            "move_ee_to_pose",
            "close_gripper",
            "lift_ee",
            "retreat_arm",
        }
    ),
    "place": frozenset(
        {
            "move_ee_to_grounded_instance",
            "move_ee_to_pose",
            "open_gripper",
            "lift_ee",
            "retreat_arm",
        }
    ),
    "contact": frozenset(
        {
            "move_ee_to_grounded_instance",
            "move_ee_to_pose",
            "contact_displace",
            "retreat_arm",
        }
    ),
}
_SEMANTIC_TERMINAL_TOOL = {
    "grasp": "close_gripper",
    "place": "open_gripper",
    "contact": "contact_displace",
}
_SEMANTIC_PHASES = {
    ("grasp", "move_ee_to_pose"): "grasp_diagnostic",
    ("grasp", "close_gripper"): "grasp_close",
    ("grasp", "lift_ee"): "grasp_lift",
    ("grasp", "retreat_arm"): "grasp_retreat",
    ("place", "open_gripper"): "place_release",
    ("place", "move_ee_to_pose"): "place_clearance",
    ("place", "lift_ee"): "place_settle",
    ("place", "retreat_arm"): "place_retreat",
    ("contact", "contact_displace"): "contact_displace",
    ("contact", "retreat_arm"): "contact_retreat",
}

_RUNTIME_PLACE_CONTINUATION_MARKERS = frozenset(
    {
        "_runtime_visual_clearance",
        "_runtime_action_geometry_repair_safe_motion",
        "_runtime_action_geometry_repair_retreat",
        "_runtime_post_contact_clearance",
    }
)


def _semantic_marker(
    *, selection: _CandidateSelection, tool_name: str, phase: str
) -> dict[str, Any]:
    payload = {
        "schema": "roboharn_evo/hpk/semantic_phase/v1",
        "selection_token": selection.token,
        "operation": selection.binding["operation"],
        "arm": selection.binding["arm"],
        "tool_name": tool_name,
        "phase": phase,
    }
    payload["marker_id"] = stable_content_id("afkphase", payload)
    return payload


def _valid_semantic_marker(
    value: Any,
    *,
    selection: _CandidateSelection,
    tool_name: str,
) -> str | None:
    if not isinstance(value, Mapping):
        return None
    payload = dict(value)
    expected_keys = {
        "schema",
        "marker_id",
        "selection_token",
        "operation",
        "arm",
        "tool_name",
        "phase",
    }
    if set(payload) != expected_keys:
        return None
    phase = _SEMANTIC_PHASES.get((str(selection.binding["operation"]), tool_name))
    if (
        phase is None
        or payload["schema"] not in {"roboharn_evo/hpk/semantic_phase/v1", "tcm/afk/semantic_phase/v1"}
        or payload["selection_token"] != selection.token
        or payload["operation"] != selection.binding["operation"]
        or payload["arm"] != selection.binding["arm"]
        or payload["tool_name"] != tool_name
        or payload["phase"] != phase
    ):
        return None
    identity = dict(payload)
    marker_id = identity.pop("marker_id")
    if marker_id != stable_content_id("afkphase", identity):
        return None
    return phase


def _semantic_attempt_group_is_proven(
    *,
    calls: Sequence[Any],
    physical_indices: Sequence[int],
    selection: _CandidateSelection,
) -> bool:
    """Prove a low-level chain realizes one task-neutral semantic operation."""

    del physical_indices
    return prove_semantic_attempt_group(
        calls=tuple(calls),
        binding=selection.binding,
    )


def _advance_semantic_stage(
    *, operation: str, stage: str, tool_name: str
) -> tuple[str, str] | None:
    transitions = {
        ("grasp", "candidate", "close_gripper"): ("closed", "grasp_close"),
        ("grasp", "closed", "lift_ee"): ("lifted", "grasp_lift"),
        ("grasp", "closed", "move_ee_to_pose"): (
            "diagnosed",
            "grasp_diagnostic",
        ),
        ("grasp", "diagnosed", "retreat_arm"): (
            "retreated",
            "grasp_retreat",
        ),
        ("grasp", "lifted", "retreat_arm"): ("retreated", "grasp_retreat"),
        ("grasp", "closed", "retreat_arm"): ("retreated", "grasp_retreat"),
        ("place", "candidate", "open_gripper"): ("released", "place_release"),
        ("place", "released", "lift_ee"): ("settled", "place_settle"),
        ("place", "released", "move_ee_to_pose"): (
            "cleared",
            "place_clearance",
        ),
        ("place", "cleared", "retreat_arm"): (
            "retreated",
            "place_retreat",
        ),
        ("place", "settled", "retreat_arm"): ("retreated", "place_retreat"),
        ("place", "released", "retreat_arm"): ("retreated", "place_retreat"),
        ("contact", "candidate", "contact_displace"): (
            "displaced",
            "contact_displace",
        ),
        ("contact", "displaced", "retreat_arm"): (
            "retreated",
            "contact_retreat",
        ),
    }
    return transitions.get((operation, stage, tool_name))


def _read_stable_regular_file(path: str | Path, *, label: str) -> tuple[Path, bytes]:
    raw_path = Path(path)
    if not raw_path.is_absolute():
        raise HPKEvolvingRuntimeError(f"{label} path must be absolute")
    try:
        metadata = raw_path.lstat()
    except OSError as exc:
        raise HPKEvolvingRuntimeError(f"{label} is unavailable: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise HPKEvolvingRuntimeError(f"{label} must be a non-symlink regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise HPKEvolvingRuntimeError(
            "rollout manifest publication requires O_NOFOLLOW"
        )
    flags |= os.O_NOFOLLOW
    descriptor = os.open(raw_path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise HPKEvolvingRuntimeError(f"{label} changed while being hashed")
        data = b"".join(chunks)
        if len(data) != after.st_size:
            raise HPKEvolvingRuntimeError(f"{label} changed size while being hashed")
    finally:
        os.close(descriptor)
    return raw_path.resolve(strict=True), data


def publish_rollout_import_manifest_v2(
    *,
    expected_episode_id: str | int,
    public_trace_path: str | Path,
    private_trace_path: str | Path,
    manifest_path: str | Path,
) -> PublishedRolloutImportManifestV2:
    """Atomically publish one explicit, hash-pinned V2 import manifest."""

    if isinstance(expected_episode_id, bool) or not (
        isinstance(expected_episode_id, int)
        and expected_episode_id >= 0
        or isinstance(expected_episode_id, str)
        and bool(expected_episode_id.strip())
    ):
        raise HPKEvolvingRuntimeError(
            "expected_episode_id must be non-negative int or non-empty string"
        )
    public_path, public_bytes = _read_stable_regular_file(
        public_trace_path, label="HPK public trace"
    )
    private_path, private_bytes = _read_stable_regular_file(
        private_trace_path, label="HPK private trace"
    )
    if public_path == private_path:
        raise HPKEvolvingRuntimeError("public and private HPK traces must differ")
    public_sha = hashlib.sha256(public_bytes).hexdigest()
    private_sha = hashlib.sha256(private_bytes).hexdigest()
    try:
        first_line = next(line for line in public_bytes.splitlines() if line.strip())
        start_record = json.loads(first_line.decode("utf-8"))
    except (StopIteration, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HPKEvolvingRuntimeError(
            "HPK public trace lacks a valid first episode_start"
        ) from exc
    if (
        not isinstance(start_record, Mapping)
        or start_record.get("event") != "episode_start"
        or type(start_record.get("episode_id")) is not type(expected_episode_id)
        or start_record.get("episode_id") != expected_episode_id
    ):
        raise HPKEvolvingRuntimeError(
            "HPK public trace first event does not bind the expected episode"
        )
    try:
        runtime_binding = validate_runtime_binding(start_record.get("runtime_binding"))
    except Exception as exc:
        raise HPKEvolvingRuntimeError(
            f"HPK public trace runtime binding is invalid: {exc}"
        ) from exc
    rollout_id = stable_content_id(
        "afkrollout",
        {
            "expected_episode_id_type": type(expected_episode_id).__name__,
            "expected_episode_id": expected_episode_id,
            "public_trace_sha256": public_sha,
            "private_transition_trace_sha256": private_sha,
            "runtime_binding_id": runtime_binding["binding_id"],
        },
    )
    payload = {
        "schema": ROLLOUT_IMPORT_MANIFEST_V2_SCHEMA,
        "rollout_id": rollout_id,
        "episode_id": expected_episode_id,
        "public_trace": {"path": str(public_path), "sha256": public_sha},
        "private_transition_trace": {
            "path": str(private_path),
            "sha256": private_sha,
        },
        "runtime_binding": runtime_binding,
        "information_access": {
            "oracle_derived": False,
            "expert_derived": False,
        },
    }
    encoded = canonical_json_bytes(payload) + b"\n"
    raw_destination = Path(manifest_path)
    if not raw_destination.is_absolute():
        raise HPKEvolvingRuntimeError("rollout import manifest path must be absolute")
    try:
        parent = raw_destination.parent.resolve(strict=True)
    except OSError as exc:
        raise HPKEvolvingRuntimeError(
            f"rollout import manifest parent is unavailable: {exc}"
        ) from exc
    destination = parent / raw_destination.name
    temporary = parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, destination, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise HPKEvolvingRuntimeError(
            f"rollout import manifest already exists: {destination}"
        ) from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return PublishedRolloutImportManifestV2(
        path=destination,
        sha256=hashlib.sha256(encoded).hexdigest(),
        payload=copy.deepcopy(payload),
    )


def publish_finalization_receipt_v1(
    *,
    rollout_manifest_path: str | Path,
    coordinated_result: CoordinatedEpisodeResult,
    receipt_path: str | Path,
) -> PublishedFinalizationReceiptV1:
    """Publish a durable public receipt without appending to pinned inputs."""

    manifest_file, manifest_bytes = _read_stable_regular_file(
        rollout_manifest_path,
        label="rollout import manifest",
    )
    payload = {
        "schema": "roboharn_evo/hpk/finalization_receipt/v1",
        "rollout_import_manifest": {
            "path": str(manifest_file),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "finalization": coordinated_result.finalization.to_dict(),
        "next_snapshot": coordinated_result.next_snapshot_ref.identity(),
        "advanced": coordinated_result.advanced,
    }
    payload["receipt_id"] = stable_content_id("afkfinal", payload)
    encoded = canonical_json_bytes(payload) + b"\n"
    raw_destination = Path(receipt_path)
    if not raw_destination.is_absolute():
        raise HPKEvolvingRuntimeError("finalization receipt path must be absolute")
    try:
        parent = raw_destination.parent.resolve(strict=True)
    except OSError as exc:
        raise HPKEvolvingRuntimeError(
            f"finalization receipt parent is unavailable: {exc}"
        ) from exc
    destination = parent / raw_destination.name
    if destination.exists() or destination.is_symlink():
        existing_path, existing = _read_stable_regular_file(
            destination,
            label="existing finalization receipt",
        )
        if existing != encoded:
            raise HPKEvolvingRuntimeError(
                "finalization receipt already exists with different content"
            )
        return PublishedFinalizationReceiptV1(
            path=existing_path,
            sha256=hashlib.sha256(existing).hexdigest(),
            payload=copy.deepcopy(payload),
        )
    temporary = parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, destination, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise HPKEvolvingRuntimeError(
            "finalization receipt was concurrently published"
        ) from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return PublishedFinalizationReceiptV1(
        path=destination,
        sha256=hashlib.sha256(encoded).hexdigest(),
        payload=copy.deepcopy(payload),
    )


class HPKEvolvingRuntime(HPKStaticRuntime):
    """One shared planner/ImgAgent runtime with sequential SnapshotV2 updates."""

    def __init__(
        self,
        *,
        policy: HPKRuntimePolicy,
        snapshot: LoadedEvolvingSnapshot,
        task_family: str,
        domain_id: str,
        current_capabilities: HPKDomainCapabilities | Mapping[str, Any] | None = None,
        geometry_policy: LoadedHPKPolicy | None = None,
        promotion_policy: LoadedHPKPolicy | None = None,
        runtime_provenance: Mapping[str, Any] | None = None,
        finalizer: Any | None = None,
        coordinator: SequentialHPKCoordinator | None = None,
        proposer: VLMStrategyProposer | None = None,
    ) -> None:
        if not policy.evolving_enabled:
            raise HPKEvolvingRuntimeError(
                "HPKEvolvingRuntime requires agent.hpk.mode=evolving"
            )
        loaded_geometry = geometry_policy or policy.load_geometry_policy()
        loaded_promotion = promotion_policy or policy.load_promotion_policy()
        provenance = EvolvingRuntimeProvenance.from_mapping(runtime_provenance)
        self._validate_policy_refs(
            snapshot,
            geometry_policy=loaded_geometry,
            promotion_policy=loaded_promotion,
            run_scope=policy.run_scope,
        )
        super().__init__(
            policy=policy,
            snapshot=snapshot,
            task_family=task_family,
            domain_id=domain_id,
            current_capabilities=current_capabilities,
            snapshot_capabilities=current_capabilities,
            geometry_policy=loaded_geometry,
            allow_empty_snapshot=True,
        )
        self.promotion_policy = loaded_promotion
        self.runtime_provenance = provenance
        if policy.proposer_enabled and proposer is None:
            raise HPKEvolvingRuntimeError(
                "enabled HPK v2 proposer requires an explicit existing backend"
            )
        if not policy.proposer_enabled and proposer is not None:
            raise HPKEvolvingRuntimeError(
                "a proposer backend was supplied while agent.hpk.proposer is disabled"
            )
        self._v2_proposer = proposer
        semantic_attempt_policy = loaded_geometry.payload.get("semantic_attempt")
        if not isinstance(semantic_attempt_policy, Mapping):
            raise HPKEvolvingRuntimeError(
                "evolving HPK requires geometry policy v2 semantic_attempt"
            )
        self._max_pending_semantic_step_gap = int(
            semantic_attempt_policy["max_pending_env_step_gap"]
        )
        self._max_semantic_attempt_segments = int(
            semantic_attempt_policy["max_segments"]
        )
        initial_ref = SnapshotRef.from_loaded(snapshot)
        self._semantic_write_enabled = coordinator is None and finalizer is None
        if self._semantic_write_enabled:
            self._disable_legacy_entry_retrieval()
        if coordinator is not None and finalizer is not None:
            raise HPKEvolvingRuntimeError(
                "supply either coordinator or finalizer, not both"
            )
        if coordinator is None:
            episode_finalizer = finalizer or HPKEpisodeFinalizer(
                promotion_policy=PromotionPolicyV1.from_mapping(
                    loaded_promotion.payload
                ),
                policy_refs=_snapshot_policy_refs(snapshot),
                destination_root=Path(policy.snapshot_output_root),
                runtime_source_identity=provenance.runtime_source_identity(),
                domain_ids=(domain_id,),
                config_sha256=provenance.effective_config_sha256,
                model_identity_sha256=(provenance.agent_service_identity_sha256),
                runtime_source_sha256=provenance.runtime_tree_sha256,
                runtime_schema=EVOLVING_RUNTIME_SCHEMA,
            )
            coordinator = SequentialHPKCoordinator(
                initial_snapshot_ref=initial_ref,
                finalizer=episode_finalizer,
            )
        self._coordinator = coordinator
        self._lease: EpisodeLease | None = None
        self._episode_finalized = False
        self._finalization_result: CoordinatedEpisodeResult | None = None
        self._finalized_manifest_path: Path | None = None
        self._selection_ordinal = 0
        self._selections: dict[str, _CandidateSelection] = {}
        self._prepared: dict[str, _PreparedAttempt] = {}
        self._transition_contexts: dict[str, TransitionStrategyContext] = {}
        self._semantic_knowledge_attempts: list[
            tuple[dict[str, Any], dict[str, Any]]
        ] = []
        self._pending_semantic_verifications: dict[
            tuple[str, str, str], _PendingSemanticVerification
        ] = {}
        self._semantic_knowledge_path = (
            Path(policy.snapshot_output_root) / "knowledge.jsonl"
        )
        self._semantic_finalizer = SemanticEpisodeFinalizer(
            knowledge_path=self._semantic_knowledge_path,
            promotion_policy=loaded_promotion.payload,
        )
        self._semantic_finalization_result: SemanticFinalizationResult | None = None
        self._semantic_knowledge_used = False
        self._semantic_knowledge = (
            load_semantic_knowledge(self._semantic_knowledge_path)
            if self._semantic_knowledge_path.exists()
            else ()
        )
        self._pending_semantic_attempt: _PendingSemanticAttempt | None = None
        self._safe_exploration_counts: dict[str, int] = {}
        self._safe_exploration_disabled_conditions: set[str] = set()
        self._sequential_run_binding: dict[str, Any] | None = None
        self._episode_runtime_binding: dict[str, Any] | None = None
        self._v2_proposal_call_used = False
        self._pending_v2_proposal: _PendingV2Proposal | None = None
        self._v2_proposal_events: list[tuple[str, dict[str, Any]]] = []
        self._lock = threading.RLock()

    @classmethod
    def from_policy(
        cls,
        policy: HPKRuntimePolicy,
        *,
        task_family: str,
        domain_id: str,
        current_capabilities: HPKDomainCapabilities | Mapping[str, Any] | None = None,
        runtime_provenance: Mapping[str, Any] | None = None,
        finalizer: Any | None = None,
        proposer: VLMStrategyProposer | None = None,
    ) -> "HPKEvolvingRuntime":
        return cls(
            policy=policy,
            snapshot=policy.load_evolving_snapshot(),
            task_family=task_family,
            domain_id=domain_id,
            current_capabilities=current_capabilities,
            runtime_provenance=runtime_provenance,
            finalizer=finalizer,
            proposer=proposer,
        )

    @staticmethod
    def _validate_policy_refs(
        snapshot: LoadedEvolvingSnapshot,
        *,
        geometry_policy: LoadedHPKPolicy,
        promotion_policy: LoadedHPKPolicy,
        run_scope: str,
    ) -> None:
        refs = _snapshot_policy_refs(snapshot)
        if refs.get("geometry") != geometry_policy.identity():
            raise HPKEvolvingRuntimeError(
                "SnapshotV2 geometry policy ref does not match the explicit policy"
            )
        promotion_ref = refs.get("promotion")
        try:
            embedded_promotion = promotion_policy_from_ref(promotion_ref)
        except Exception as exc:
            raise HPKEvolvingRuntimeError(
                "SnapshotV2 promotion policy ref is invalid"
            ) from exc
        if (
            embedded_promotion.policy_id != promotion_policy.policy_id
            or embedded_promotion.config_sha256 != promotion_policy.config_sha256
            or canonical_json_bytes(embedded_promotion.to_dict())
            != canonical_json_bytes(promotion_policy.payload)
        ):
            raise HPKEvolvingRuntimeError(
                "SnapshotV2 promotion policy ref does not match the explicit policy"
            )
        development_only = promotion_policy.payload.get("development_only")
        if development_only is True and run_scope != "integration":
            raise HPKEvolvingRuntimeError(
                "development promotion snapshots are restricted to integration scope"
            )
        if run_scope == "formal_no_prior" and development_only is not False:
            raise HPKEvolvingRuntimeError(
                "formal evolving runtime requires a formal promotion policy"
            )

    @property
    def evolving_enabled(self) -> bool:
        return True

    @property
    def semantic_runtime_enabled(self) -> bool:
        return self._semantic_write_enabled

    @property
    def episode_lease(self) -> EpisodeLease | None:
        return self._lease

    @property
    def pinned_snapshot_ref(self) -> SnapshotRef | None:
        if self._lease is None:
            return None
        return self._lease.snapshot_ref

    @property
    def transition_contexts(self) -> dict[str, TransitionStrategyContext]:
        return dict(self._transition_contexts)

    @property
    def finalization_result(self) -> CoordinatedEpisodeResult | None:
        return self._finalization_result

    @property
    def semantic_finalization_result(self) -> SemanticFinalizationResult | None:
        return self._semantic_finalization_result

    @property
    def semantic_knowledge_used(self) -> bool:
        return self._semantic_knowledge_used

    def episode_runtime_binding(self) -> dict[str, Any]:
        """Return the once-validated episode lease without re-hashing at actions."""

        with self._lock:
            if (
                self._lease is None
                or self._episode_finalized
                or self._episode_runtime_binding is None
            ):
                raise HPKEvolvingRuntimeError(
                    "runtime binding requires an active pinned episode"
                )
            return copy.deepcopy(self._episode_runtime_binding)

    def _freeze_episode_runtime_binding(self, pinned: SnapshotRef) -> None:
        """Build the artifact-integrity lease once, outside the action loop."""

        self._episode_runtime_binding = build_runtime_binding(
            snapshot_id=pinned.snapshot_id,
            snapshot_manifest_sha256=pinned.manifest_sha256,
            policy_refs=_snapshot_policy_refs(pinned.snapshot),
            provenance_schema=self.runtime_provenance.schema,
            provenance_manifest_sha256=self.runtime_provenance.manifest_sha256,
            config_sha256=self.runtime_provenance.effective_config_sha256,
            model_identity_sha256=(
                self.runtime_provenance.agent_service_identity_sha256
            ),
            runtime_source_sha256=self.runtime_provenance.runtime_tree_sha256,
            evidence_runtime_schema=EVOLVING_RUNTIME_SCHEMA,
            run_binding=self._sequential_run_binding,
        )

    def bind_sequential_run(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Bind one preregistered run before any episode lease or trace exists."""

        validated = validate_sequential_run_binding(value)
        with self._lock:
            if self._lease is not None or self._prepared or self._selections:
                raise HPKEvolvingRuntimeError(
                    "sequential run binding must be installed before episode start"
                )
            if (
                self._sequential_run_binding is not None
                and self._sequential_run_binding != validated
            ):
                raise HPKEvolvingRuntimeError(
                    "evolving runtime is already bound to another sequential run"
                )
            self._sequential_run_binding = copy.deepcopy(validated)
            return copy.deepcopy(validated)

    @property
    def finalized_manifest_path(self) -> Path | None:
        return self._finalized_manifest_path

    def _clear_episode_transient(self) -> None:
        self._selection_ordinal = 0
        self._selections.clear()
        self._prepared.clear()
        self._transition_contexts.clear()
        self._semantic_knowledge_attempts.clear()
        self._pending_semantic_verifications.clear()
        self._semantic_knowledge_used = False
        self._pending_semantic_attempt = None
        self._safe_exploration_counts.clear()
        self._safe_exploration_disabled_conditions.clear()
        self._v2_proposal_call_used = False
        self._pending_v2_proposal = None
        self._v2_proposal_events.clear()

    @property
    def v2_proposer_enabled(self) -> bool:
        return self._v2_proposer is not None

    def drain_v2_proposal_events(self) -> list[tuple[str, dict[str, Any]]]:
        """Return candidate-free proposal lifecycle records for the agent trace."""

        with self._lock:
            events = copy.deepcopy(self._v2_proposal_events)
            self._v2_proposal_events.clear()
            return events

    def reset_episode(self) -> None:
        """Clear only transient query/attempt state; never reload or advance head."""

        with self._lock:
            super().reset_episode()
            self._clear_episode_transient()
            if self._episode_finalized:
                self._lease = None
                self._episode_runtime_binding = None
                self._episode_finalized = False
                self._finalization_result = None
                self._semantic_finalization_result = None
                self._finalized_manifest_path = None

    def begin_episode(self, expected_episode_id: str | int) -> EpisodeLease:
        with self._lock:
            if self._lease is not None and not self._episode_finalized:
                lease = self._coordinator.begin_episode(expected_episode_id)
                if lease.token != self._lease.token:
                    raise HPKEvolvingRuntimeError(
                        "another evolving episode is already pinned"
                    )
                return self._lease
            if self._episode_finalized:
                self._lease = None
                self._episode_runtime_binding = None
                self._episode_finalized = False
                self._finalization_result = None
                self._semantic_finalization_result = None
                self._finalized_manifest_path = None
            lease = self._coordinator.begin_episode(expected_episode_id)
            pinned = self._coordinator.snapshot_for_episode(lease)
            self._validate_policy_refs(
                pinned.snapshot,
                geometry_policy=self.geometry_policy,
                promotion_policy=self.promotion_policy,
                run_scope=self.policy.run_scope,
            )
            self._install_snapshot_projection(
                pinned.snapshot,
                snapshot_capabilities=self.current_capabilities,
                allow_empty_snapshot=True,
            )
            if self._semantic_write_enabled:
                self._disable_legacy_entry_retrieval()
            self._lease = lease
            self._clear_episode_transient()
            self._semantic_knowledge = (
                load_semantic_knowledge(self._semantic_knowledge_path)
                if self._semantic_knowledge_path.exists()
                else ()
            )
            self._freeze_episode_runtime_binding(pinned)
            return lease

    def build_bound_geometry_query(self, **kwargs: Any) -> HPKBoundGeometryQuery:
        """Project execution-arm binding out of transferable task identity."""

        query = super().build_bound_geometry_query(**kwargs)
        if not isinstance(query.task_strategy, TaskStrategyV1):
            return query
        if query.task_strategy["preferred_arm"] == "either":
            return query
        payload = query.task_strategy.to_dict()
        payload["preferred_arm"] = "either"
        return HPKBoundGeometryQuery(
            condition=query.condition,
            task_strategy=TaskStrategyV1.from_dict(payload),
            target_bound=query.target_bound,
            operation=query.operation,
            semantic_object=query.semantic_object,
        )

    @property
    def safe_exploration_enabled(self) -> bool:
        profile = self.geometry_policy.payload.get("safe_exploration")
        return bool(
            not self.v2_proposer_enabled
            and self.geometry_policy.policy_id in {"hpk_geometry_policy/v3", "afk_geometry_policy/v3"}
            and self.geometry_policy.config_sha256
            == SAFE_EXPLORATION_GEOMETRY_POLICY_CONFIG_SHA256
            and isinstance(profile, Mapping)
        )

    def retrieve_geometry(
        self,
        query: HPKBoundGeometryQuery,
    ) -> HPKGeometryDecision:
        """检索已接受的 HPK，匹配后关闭探索。"""

        with self._lock:
            decision = super().retrieve_geometry(query)
            if (
                self.safe_exploration_enabled
                and decision.retrieval.selected_entry is not None
                and isinstance(query.condition, ConditionV1)
            ):
                self._safe_exploration_disabled_conditions.add(
                    query.condition.stable_id
                )
            return decision

    def retrieve_semantic_geometry(
        self,
        query: HPKBoundGeometryQuery,
    ) -> SemanticGeometryMatch | None:
        """Direct semantic lookup; no content identity participates in matching."""

        if (
            not self._semantic_knowledge
            or not isinstance(query.condition, ConditionV1)
            or not isinstance(query.task_strategy, TaskStrategyV1)
            or query.semantic_object is None
        ):
            return None
        expected = expected_effect_for_operation(query.operation)
        if isinstance(expected, HPKUnresolved):
            return None
        preferred_arm = str(query.task_strategy["preferred_arm"])
        semantic_query = {
            "object": query.semantic_object,
            "condition": {
                "task": str(query.condition["task_family"]).replace("_", " "),
                "phase": str(query.condition["manipulation_phase"]).replace(
                    "_", " "
                ),
            },
            "task_strategy": {
                "operation": query.operation,
                "preferred_arm": (
                    "either arm"
                    if preferred_arm == "either"
                    else f"{preferred_arm} arm"
                ),
            },
            "expected_effect": {
                str(predicate): True
                for predicate in expected["expected_predicates"]
            },
        }
        matches = retrieve_semantic_knowledge(
            self._semantic_knowledge,
            semantic_query,
        )
        if not matches:
            return None
        knowledge = matches[0]
        return SemanticGeometryMatch(
            knowledge=knowledge,
            geometric_strategy=semantic_geometry_to_v1(knowledge),
        )

    def semantic_planner_context(self) -> dict[str, Any] | None:
        """Return accepted task knowledge exactly as the VLM should read it."""

        task = self.task_family.replace("_", " ").lower()
        knowledge = [
            copy.deepcopy(value)
            for value in self._semantic_knowledge
            if value["status"] == "accepted"
            and value["condition"]["task"] == task
        ]
        if not knowledge:
            return None
        return {
            "schema": "roboharn_evo/hpk/semantic_context/v1",
            "knowledge": knowledge,
        }

    def select_pending_v2_proposal(
        self,
        *,
        query: HPKBoundGeometryQuery,
        eligible_candidates: Sequence[Mapping[str, Any]],
        scene_state: Mapping[str, Any],
    ) -> V2ProposalCandidateDecision | None:
        """Apply one top-1 pending proposal at its next exact condition match."""

        with self._lock:
            pending = self._pending_v2_proposal
            if pending is None:
                return None
            if not isinstance(query.condition, ConditionV1) or not isinstance(
                query.task_strategy, TaskStrategyV1
            ):
                return None
            proposal = pending.proposal
            if canonical_json_bytes(condition_v1_to_v2(query.condition)) != (
                canonical_json_bytes(proposal["condition"])
            ) or canonical_json_bytes(task_strategy_v1_to_v2(query.task_strategy)) != (
                canonical_json_bytes(proposal["base_task_strategy"])
            ):
                return None
            proposed_v2 = apply_geometry_delta(
                proposal["base_geometric_strategy"], proposal["delta"]
            )
            proposed_v1 = geometric_strategy_v2_to_v1(
                proposed_v2, proposal["base_task_strategy"]
            )
            ranking = rank_operation_pose_candidates(
                eligible_candidates,
                geometric_strategy=proposed_v1,
                geometry_policy=self.geometry_policy,
                scene_state=scene_state,
            )
            selected = ranking.select()
            trace = (
                rank_frozen_candidates_with_proposal(
                    proposal=proposal,
                    frozen_candidates=eligible_candidates,
                    scene_state=scene_state,
                ).to_dict()
                if selected is not None
                else {
                    "schema": PROPOSAL_RANK_TRACE_SCHEMA,
                    "proposal_id": proposal["proposal_id"],
                    "rank_before_count": len(eligible_candidates),
                    "rank_after": [],
                    "rank_changed": True,
                    "selection_changed": False,
                    "selected_candidate_satisfies_proposal": False,
                    "motion_executed": False,
                    "hpk_store_write_performed": False,
                }
            )
            expected_v1 = AbstractEffectV1.from_dict(
                {
                    "schema": ABSTRACT_EFFECT_SCHEMA,
                    "effect_type": proposal["expected_effect"]["effect_type"],
                    "expected_predicates": proposal["expected_effect"][
                        "expected_predicates"
                    ],
                    "verifiability": "unverified",
                }
            )
            self._pending_v2_proposal = None
            self._v2_proposal_events.append(
                (
                    "hpk_v2_proposal_scheduled",
                    {
                        "schema": "roboharn_evo/hpk/v2_proposal_scheduled/v1",
                        "proposal_id": proposal["proposal_id"],
                        "source_transition_id": pending.source_transition_id,
                        "condition": proposal["condition"],
                        "task_strategy": proposal["base_task_strategy"],
                        "proposed_geometric_strategy": proposed_v2,
                        "expected_effect": proposal["expected_effect"],
                        "rank_trace": trace,
                        "selected": selected is not None,
                    },
                )
            )
            return V2ProposalCandidateDecision(
                proposal=proposal,
                selected_candidate=(None if selected is None else dict(selected)),
                proposed_geometric_strategy=proposed_v1,
                expected_effect=expected_v1,
                rank_trace=trace,
            )

    def select_safe_exploration_candidate(
        self,
        *,
        query: HPKBoundGeometryQuery,
        decision: HPKGeometryDecision,
        eligible_candidates: Sequence[Mapping[str, Any]],
        requested_candidate_id: Any = None,
        guard_legal: bool,
    ) -> SafeExplorationDecision:
        """Choose v3's single alternate rank from a prefiltered baseline order.

        The method has no task/seed/arm/result input.  It is called only after
        the existing operation Guard, blocked-candidate lifecycle, target
        binding, and feasibility generation have produced ``eligible_candidates``.
        """

        with self._lock:
            if self._lease is None or self._episode_finalized:
                raise HPKEvolvingRuntimeError(
                    "safe exploration requires an active pinned episode"
                )
            if not self.safe_exploration_enabled:
                return SafeExplorationDecision(None, "geometry_policy_v3_disabled")
            if decision.retrieval.selected_entry is not None:
                if isinstance(query.condition, ConditionV1):
                    self._safe_exploration_disabled_conditions.add(
                        query.condition.stable_id
                    )
                return SafeExplorationDecision(None, "accepted_exact_match")
            if not isinstance(query.condition, ConditionV1):
                return SafeExplorationDecision(None, "condition_unresolved")
            if not isinstance(query.task_strategy, TaskStrategyV1):
                return SafeExplorationDecision(None, "task_strategy_unresolved")
            if "no_exact_match" not in decision.retrieval.rejection_reasons:
                return SafeExplorationDecision(None, "no_match_not_proven")
            effect = expected_effect_for_operation(query.operation)
            if isinstance(effect, HPKUnresolved):
                return SafeExplorationDecision(None, "expected_effect_unresolved")
            condition_id = query.condition.stable_id
            if condition_id in self._safe_exploration_disabled_conditions:
                return SafeExplorationDecision(
                    None, "disabled_after_accepted_exact_match"
                )
            prior = self._safe_exploration_counts.get(condition_id, 0)
            result = choose_safe_baseline_alternative(
                geometry_policy=self.geometry_policy,
                runtime_binding=self.episode_runtime_binding(),
                condition_id=condition_id,
                task_strategy_id=query.task_strategy.stable_id,
                expected_effect_id=effect.stable_id,
                eligible_candidates=eligible_candidates,
                accepted_exact_match=False,
                requested_candidate_id=requested_candidate_id,
                guard_legal=guard_legal,
                prior_selections_for_condition=prior,
            )
            if result.selected:
                self._safe_exploration_counts[condition_id] = prior + 1
            return result

    def reuse_safe_exploration_continuation(
        self,
        *,
        query: HPKBoundGeometryQuery,
        decision: HPKGeometryDecision | None,
        selected_candidate: Mapping[str, Any],
        scene_memory: Mapping[str, Any],
        arm: str,
        continuation_binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Reuse one runtime-owned exploration selection without re-ranking.

        Geometry freshness remains the downstream continuation Guard's job.
        This check only proves the same episode-pinned typed selection and
        opaque private candidate survived the existing eligibility filter.
        """

        del scene_memory
        with self._lock:
            if self._lease is None or self._episode_finalized:
                raise HPKEvolvingRuntimeError(
                    "safe exploration continuation requires an active episode"
                )
            semantic_continuation = bool(
                isinstance(continuation_binding, Mapping)
                and continuation_binding.get("semantic_knowledge_used") is True
            )
            if not self.safe_exploration_enabled and not semantic_continuation:
                raise HPKEvolvingRuntimeError(
                    "safe exploration continuation requires geometry policy v3"
                )
            if decision is not None and decision.retrieval.selected_entry is not None:
                raise HPKEvolvingRuntimeError(
                    "accepted exact match closes safe exploration continuation"
                )
            if not isinstance(query.condition, ConditionV1) or not isinstance(
                query.task_strategy, TaskStrategyV1
            ):
                raise HPKEvolvingRuntimeError(
                    "safe exploration continuation semantics are unresolved"
                )
            binding = copy.deepcopy(dict(continuation_binding))
            token = str(binding.get("_hpk_selection_token", "") or "")
            selection = self._selections.get(token)
            if selection is None:
                raise HPKEvolvingRuntimeError(
                    "safe exploration continuation selection token is unknown"
                )
            frozen = selection.binding
            if canonical_json_bytes(binding) != canonical_json_bytes(frozen):
                raise HPKEvolvingRuntimeError(
                    "safe exploration continuation rewrote its frozen binding"
                )
            candidate_ref = _private_candidate_ref(selected_candidate)
            normalized_arm = str(arm or "").strip().lower()
            if (
                frozen.get("condition_id") != query.condition.stable_id
                or frozen.get("task_strategy_id") != query.task_strategy.stable_id
                or frozen.get("operation") != query.operation
                or frozen.get("arm") != normalized_arm
                or frozen.get("selected_candidate_private_ref") != candidate_ref
                or (
                    not semantic_continuation
                    and frozen.get("safe_exploration_public_audit_id") is None
                )
                or (
                    not semantic_continuation
                    and frozen.get("safe_exploration_private_audit_id") is None
                )
                or frozen.get("_hpk_runtime_binding_id")
                != self.episode_runtime_binding()["binding_id"]
            ):
                raise HPKEvolvingRuntimeError(
                    "safe exploration continuation differs from the frozen selection"
                )
            return copy.deepcopy(frozen)

    def freeze_post_binding_candidate(
        self,
        *,
        query: HPKBoundGeometryQuery,
        decision: HPKGeometryDecision | None,
        selected_candidate: Mapping[str, Any],
        scene_memory: Mapping[str, Any],
        arm: str,
        object_semantics: Mapping[str, Any] | None = None,
        ranking_sink: PrivateRankingAuditSink | None = None,
        trace_binding: Mapping[str, Any] | None = None,
        eligible_candidates: Sequence[Mapping[str, Any]] | None = None,
        proposal_decision: V2ProposalCandidateDecision | None = None,
        semantic_match: SemanticGeometryMatch | None = None,
    ) -> dict[str, Any]:
        """Freeze the real post-binding candidate before it can reach motion."""

        with self._lock:
            if self._lease is None or self._episode_finalized:
                raise HPKEvolvingRuntimeError(
                    "candidate selection requires an active pinned episode"
                )
            condition = (
                query.condition if isinstance(query.condition, ConditionV1) else None
            )
            task_strategy = (
                query.task_strategy
                if isinstance(query.task_strategy, TaskStrategyV1)
                else None
            )
            semantic_source = dict(
                semantic_match.knowledge["object"]
                if semantic_match is not None
                else object_semantics or {}
            )
            if condition is not None:
                manipulated = condition["manipulated_object"]
                semantic_source.setdefault(
                    "semantic_class", manipulated["semantic_class"]
                )
                semantic_source.setdefault(
                    "geometry_class", manipulated["geometry_class"]
                )
                semantic_source.setdefault("role", manipulated["role"])
                semantic_source.setdefault("held_state", manipulated["held_state"])
            semantic_object = semantic_object_from_mapping(semantic_source)
            reasoning = (
                copy.deepcopy(semantic_match.knowledge["reasoning"])
                if semantic_match is not None
                else {
                "observed_problem": "not available",
                "failure_analysis": "not available",
                "strategy_rationale": (
                    "not available"
                    if proposal_decision is None
                    else str(proposal_decision.proposal["rationale"])
                ),
                "causal_hypothesis": "not available",
                "expected_observation": "not available",
                "failure_condition": "not available",
                "source": (
                    "runtime extraction"
                    if proposal_decision is None
                    else "VLM strategy proposal"
                ),
                "confidence": 0.0,
                }
            )
            features = candidate_semantic_features(
                scene_memory,
                selected_candidate,
                geometry_policy=self.geometry_policy,
            )
            typed_features = None if isinstance(features, HPKUnresolved) else features
            selected_entry = (
                None if decision is None else decision.retrieval.selected_entry
            )
            extracted: ExtractedHPKStrategy | None = None
            if semantic_match is not None:
                if proposal_decision is not None or selected_entry is not None:
                    raise HPKEvolvingRuntimeError(
                        "semantic knowledge cannot override another HPK selection"
                    )
                if condition is None or task_strategy is None:
                    raise HPKEvolvingRuntimeError(
                        "semantic knowledge requires resolved condition and task"
                    )
                geometric_strategy = semantic_match.geometric_strategy
                effect = expected_effect_for_operation(query.operation)
                if isinstance(effect, HPKUnresolved):
                    raise HPKEvolvingRuntimeError(
                        "semantic knowledge expected effect is unresolved"
                    )
                expected_effect = effect
                extracted = ExtractedHPKStrategy(
                    condition=condition,
                    task_strategy=task_strategy,
                    geometric_strategy=geometric_strategy,
                    expected_effect=expected_effect,
                    strategy_key_id=strategy_key_id_for(
                        condition=condition,
                        task_strategy=task_strategy,
                        geometric_strategy=geometric_strategy,
                        expected_effect=expected_effect,
                    ),
                )
                geometric_strategy_id = geometric_strategy.stable_id
                retrieved_ids = []
                selected_entry_id = None
                geometric_compliance = True
            elif proposal_decision is not None:
                if selected_entry is not None:
                    raise HPKEvolvingRuntimeError(
                        "a pending proposal cannot override an accepted HPK entry"
                    )
                if str(selected_candidate.get("candidate_id", "")) != str(
                    (proposal_decision.selected_candidate or {}).get("candidate_id", "")
                ):
                    raise HPKEvolvingRuntimeError(
                        "proposal selection changed before candidate freeze"
                    )
                if condition is None or task_strategy is None or typed_features is None:
                    raise HPKEvolvingRuntimeError(
                        "proposal execution requires resolved condition, task, and geometry"
                    )
                geometric_strategy = proposal_decision.proposed_geometric_strategy
                expected_effect = proposal_decision.expected_effect
                extracted = ExtractedHPKStrategy(
                    condition=condition,
                    task_strategy=task_strategy,
                    geometric_strategy=geometric_strategy,
                    expected_effect=expected_effect,
                    strategy_key_id=strategy_key_id_for(
                        condition=condition,
                        task_strategy=task_strategy,
                        geometric_strategy=geometric_strategy,
                        expected_effect=expected_effect,
                    ),
                )
                geometric_strategy_id = geometric_strategy.stable_id
                retrieved_ids = []
                selected_entry_id = None
                geometric_compliance = True
            elif selected_entry is None:
                effect = expected_effect_for_operation(query.operation)
                expected_effect = None if isinstance(effect, HPKUnresolved) else effect
                if (
                    condition is not None
                    and task_strategy is not None
                    and typed_features is not None
                    and expected_effect is not None
                ):
                    extracted_value = extract_baseline_strategy(
                        condition=condition,
                        task_strategy=task_strategy,
                        candidate_features=typed_features,
                        expected_effect=expected_effect,
                        selected_hpk_entry_id=None,
                        allow_oracle_geometry=False,
                    )
                    if not isinstance(extracted_value, HPKUnresolved):
                        extracted = extracted_value
                        expected_effect = extracted.expected_effect
                geometric_strategy_id = (
                    None
                    if extracted is None
                    else extracted.geometric_strategy.stable_id
                )
                retrieved_ids: list[str] = []
                selected_entry_id = None
                geometric_compliance: bool | str = (
                    True if extracted is not None else "unverified"
                )
            else:
                geometric_strategy = decision.retrieval.geometric_strategy
                if geometric_strategy is None or ranking_sink is None:
                    raise HPKEvolvingRuntimeError(
                        "selected HPK candidate lacks its typed ranking binding"
                    )
                geometric_strategy_id = geometric_strategy.stable_id
                expected_effect = AbstractEffectV1.from_dict(
                    selected_entry["expected_effect"]
                )
                selected_entry_id = str(selected_entry["entry_id"])
                retrieved_ids = [selected_entry_id]
                geometric_compliance = ranking_sink.geometric_compliance
            candidate_ref = _private_candidate_ref(selected_candidate)
            normalized_arm = str(arm or "").strip().lower()
            if normalized_arm not in {"left", "right"}:
                raise HPKEvolvingRuntimeError("candidate arm must be left or right")
            scene_signature = _scene_signature(
                condition=condition,
                task_strategy=task_strategy,
                candidate_features=typed_features,
            )
            self._selection_ordinal += 1
            token = stable_content_id(
                "afkselection",
                {
                    "episode_control_key": self._lease.episode_control_key,
                    "selection_ordinal": self._selection_ordinal,
                    "candidate_private_ref": candidate_ref,
                    "condition_id": None if condition is None else condition.stable_id,
                    "task_strategy_id": (
                        None if task_strategy is None else task_strategy.stable_id
                    ),
                    "geometric_strategy_id": geometric_strategy_id,
                },
            )
            binding = {
                "condition_id": None if condition is None else condition.stable_id,
                "task_strategy_id": (
                    None if task_strategy is None else task_strategy.stable_id
                ),
                "retrieved_hpk_entry_ids": retrieved_ids,
                "selected_hpk_entry_id": selected_entry_id,
                "geometric_strategy_id": geometric_strategy_id,
                "operation": query.operation,
                "arm": normalized_arm,
                "target_role": (
                    None if condition is None else condition["target"]["role"]
                ),
                "target_relation": (
                    None if condition is None else condition["target"]["relation"]
                ),
                "selected_candidate_private_ref": candidate_ref,
                "candidate_geometry_features": (
                    None if typed_features is None else typed_features.to_dict()
                ),
                "geometric_compliance": geometric_compliance,
                "realization_status": (
                    "satisfied"
                    if geometric_compliance is True
                    else "violated"
                    if geometric_compliance is False
                    else "unknown"
                ),
                "target_identity_status": "bound",
                "expected_effect": (
                    None if expected_effect is None else expected_effect.to_dict()
                ),
                "verifier_conflicts": [],
                "oracle_derived": False,
                "expert_derived": False,
                "_hpk_selection_token": token,
                "_hpk_runtime_binding_id": self.episode_runtime_binding()["binding_id"],
            }
            if proposal_decision is not None:
                binding["proposal_id"] = proposal_decision.proposal["proposal_id"]
            if semantic_match is not None:
                binding["semantic_knowledge_used"] = True
                self._semantic_knowledge_used = True
            if trace_binding is not None:
                for key in (
                    "public_usage_audit_id",
                    "private_ranking_audit_id",
                    "snapshot_id",
                    "snapshot_manifest_sha256",
                    "safe_exploration_public_audit_id",
                    "safe_exploration_private_audit_id",
                ):
                    if key in trace_binding:
                        binding[key] = copy.deepcopy(trace_binding[key])
            selection = _CandidateSelection(
                token=token,
                condition=condition,
                task_strategy=task_strategy,
                semantic_object=semantic_object,
                reasoning=reasoning,
                candidate_features=typed_features,
                expected_effect=expected_effect,
                scene_signature=scene_signature,
                binding=copy.deepcopy(binding),
                extracted_strategy=extracted,
                selected_entry=selected_entry,
                proposal_capabilities=_proposal_capabilities(
                    tuple(eligible_candidates or (selected_candidate,)),
                    scene_state=scene_memory,
                    geometry_policy=self.geometry_policy,
                ),
                proposal_id=(
                    None
                    if proposal_decision is None
                    else proposal_decision.proposal["proposal_id"]
                ),
                proposed_geometric_strategy=(
                    None
                    if proposal_decision is None
                    else proposal_decision.proposed_geometric_strategy
                ),
                proposed_expected_effect=(
                    None
                    if proposal_decision is None
                    else proposal_decision.expected_effect
                ),
            )
            self._selections[token] = selection
            return copy.deepcopy(binding)

    def annotate_guarded_semantic_calls(
        self,
        calls: Sequence[RecoveryToolCall],
    ) -> list[RecoveryToolCall]:
        """Mark only runtime-generated phases in a proven guarded FSM chain.

        Planner-provided ``_hpk_*`` fields have already been stripped by the
        caller.  Candidate-directed moves must already carry the exact runtime
        selection token; this method never lends a token to an unbound move.
        """

        with self._lock:
            active: _CandidateSelection | None = None
            stage = ""
            pending = self._pending_semantic_attempt
            annotated: list[RecoveryToolCall] = []
            candidate_moves = {
                "move_ee_to_grounded_instance",
                "move_ee_to_pose",
            }
            for call in calls:
                tool_name = str(call.tool_name)
                args = copy.deepcopy(dict(call.args or {}))
                raw_binding = args.get("_hpk_usage_binding")
                token = (
                    str(raw_binding.get("_hpk_selection_token", "") or "")
                    if isinstance(raw_binding, Mapping)
                    else ""
                )
                selected = self._selections.get(token) if token else None
                runtime_continuation = False
                if active is None and pending is not None:
                    if (
                        pending.operation == "grasp"
                        and tool_name == "move_ee_to_pose"
                        and args.get("_runtime_grasp_diagnostic_motion") is True
                        and str(args.get("arm", "") or "").strip().lower()
                        == pending.arm
                    ):
                        active = pending.selection
                        stage = "closed"
                        runtime_continuation = True
                    elif (
                        pending.operation == "place"
                        and tool_name in {"move_ee_to_pose", "lift_ee", "retreat_arm"}
                        and any(
                            args.get(marker) is True
                            for marker in _RUNTIME_PLACE_CONTINUATION_MARKERS
                        )
                        and str(args.get("arm", "") or "").strip().lower()
                        == pending.arm
                    ):
                        active = pending.selection
                        stage = "released"
                        runtime_continuation = True
                if tool_name in candidate_moves:
                    if runtime_continuation:
                        transition = _advance_semantic_stage(
                            operation=str(active.binding["operation"]),
                            stage=stage,
                            tool_name=tool_name,
                        )
                        if transition is None:
                            active = None
                            stage = ""
                        else:
                            stage, phase = transition
                            args["_hpk_usage_binding"] = copy.deepcopy(active.binding)
                            args["_hpk_semantic_phase"] = _semantic_marker(
                                selection=active,
                                tool_name=tool_name,
                                phase=phase,
                            )
                    elif selected is not None:
                        active = selected
                        stage = "candidate"
                    elif tool_name in PHYSICAL_RECOVERY_TOOLS:
                        active = None
                        stage = ""
                elif tool_name in PHYSICAL_RECOVERY_TOOLS:
                    transition = (
                        None
                        if active is None
                        else _advance_semantic_stage(
                            operation=str(active.binding["operation"]),
                            stage=stage,
                            tool_name=tool_name,
                        )
                    )
                    call_arm = str(args.get("arm", "") or "").strip().lower()
                    if (
                        active is None
                        or transition is None
                        or call_arm != active.binding["arm"]
                        or raw_binding is not None
                        and selected is not active
                    ):
                        active = None
                        stage = ""
                    else:
                        stage, phase = transition
                        args["_hpk_usage_binding"] = copy.deepcopy(active.binding)
                        args["_hpk_semantic_phase"] = _semantic_marker(
                            selection=active,
                            tool_name=tool_name,
                            phase=phase,
                        )
                annotated.append(RecoveryToolCall(tool_name=tool_name, args=args))
            return annotated

    def pending_action_continuation(
        self,
        *,
        calls: Sequence[RecoveryToolCall],
        episode_id: str | int,
        snapshot_before: Any | None,
    ) -> tuple[str | None, str | None]:
        """Return the pinned nonce only for a strict owner-generated continuation."""

        with self._lock:
            pending = self._pending_semantic_attempt
            if pending is None:
                return None, None
            if self._lease is None or (
                type(episode_id) is not type(self._lease.expected_episode_id)
                or episode_id != self._lease.expected_episode_id
            ):
                return None, "pending_semantic_episode_mismatch"
            raw_step = getattr(snapshot_before, "step_count", None)
            step = (
                raw_step
                if isinstance(raw_step, int)
                and not isinstance(raw_step, bool)
                and raw_step >= 0
                else pending.env_step_latest
            )
            if step < pending.env_step_latest:
                return None, "pending_semantic_attempt_step_regression"
            if step - pending.env_step_latest > self._max_pending_semantic_step_gap:
                return None, "pending_semantic_attempt_timeout"
            physical = [
                call for call in calls if str(call.tool_name) in PHYSICAL_RECOVERY_TOOLS
            ]
            if not physical:
                if (
                    pending.operation == "place"
                    and bool(calls)
                    and all(str(call.tool_name) == "reobserve_scene" for call in calls)
                    and any(
                        pending.arm
                        in list((call.args or {}).get("_release_verification_arms", []))
                        for call in calls
                        if isinstance(
                            (call.args or {}).get("_release_verification_arms"), list
                        )
                    )
                ):
                    return pending.nonce, None
                return None, "pending_semantic_continuation_unmarked"
            for call in physical:
                args = call.args or {}
                binding = args.get("_hpk_usage_binding")
                marker = args.get("_hpk_semantic_phase")
                if (
                    not isinstance(binding, Mapping)
                    or binding.get("_hpk_selection_token") != pending.selection.token
                    or str(args.get("arm", "") or "").strip().lower() != pending.arm
                    or _valid_semantic_marker(
                        marker,
                        selection=pending.selection,
                        tool_name=str(call.tool_name),
                    )
                    is None
                ):
                    return None, "pending_semantic_continuation_binding_mismatch"
            return pending.nonce, None

    def defer_action_transition(
        self,
        *,
        nonce: str,
        selection_token: str,
        segments: Sequence[Any],
        env_step_started: int,
        env_step_latest: int,
        post_effect_state: Mapping[str, Any] | None,
    ) -> None:
        with self._lock:
            selection = self._selections.get(selection_token)
            if selection is None:
                raise HPKEvolvingRuntimeError(
                    "pending semantic action lost its frozen selection"
                )
            if self._pending_semantic_attempt is not None:
                raise HPKEvolvingRuntimeError(
                    "only one pending semantic action is permitted"
                )
            self._pending_semantic_attempt = _PendingSemanticAttempt(
                nonce=nonce,
                selection=selection,
                operation=str(selection.binding["operation"]),
                arm=str(selection.binding["arm"]),
                segments=list(segments),
                env_step_started=int(env_step_started),
                env_step_latest=int(env_step_latest),
                post_effect_state=(
                    None
                    if post_effect_state is None
                    else copy.deepcopy(dict(post_effect_state))
                ),
            )

    def take_pending_action_transition(self) -> _PendingSemanticAttempt | None:
        with self._lock:
            pending = self._pending_semantic_attempt
            self._pending_semantic_attempt = None
            return pending

    @property
    def has_pending_action_transition(self) -> bool:
        with self._lock:
            return self._pending_semantic_attempt is not None

    def prove_semantic_attempt_group(
        self,
        *,
        calls: Sequence[RecoveryToolCall],
        selection_token: str,
    ) -> bool:
        with self._lock:
            selection = self._selections.get(selection_token)
            if selection is None:
                return False
            physical_indices = [
                index
                for index, call in enumerate(calls)
                if str(call.tool_name) in PHYSICAL_RECOVERY_TOOLS
            ]
            return _semantic_attempt_group_is_proven(
                calls=calls,
                physical_indices=physical_indices,
                selection=selection,
            )

    def allow_pending_semantic_defer(self, *, segment_count: int) -> bool:
        return bool(
            isinstance(segment_count, int)
            and not isinstance(segment_count, bool)
            and 0 < segment_count < self._max_semantic_attempt_segments
        )

    def prepare_action_transition(
        self,
        *,
        action_attempt_nonce: str,
        episode_id: str | int,
        seed: int,
        calls: Sequence[Any],
        pre_effect_state: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        del seed, pre_effect_state, context
        with self._lock:
            if self._lease is None or self._episode_finalized:
                raise HPKEvolvingRuntimeError(
                    "transition preparation requires an active episode"
                )
            if (
                type(episode_id) is not type(self._lease.expected_episode_id)
                or episode_id != self._lease.expected_episode_id
            ):
                raise HPKEvolvingRuntimeError(
                    "transition episode does not match the pinned raw episode"
                )
            nonce = str(action_attempt_nonce or "").strip()
            pending = self._pending_semantic_attempt
            continuation = pending is not None and pending.nonce == nonce
            if not nonce or nonce in self._prepared and not continuation:
                raise HPKEvolvingRuntimeError(
                    "action_attempt_nonce must be non-empty and unique"
                )
            physical_indices = [
                index
                for index, call in enumerate(calls)
                if str(getattr(call, "tool_name", "")) in PHYSICAL_RECOVERY_TOOLS
            ]
            tokens: list[str] = []
            malformed_binding = False
            for index in physical_indices:
                args = getattr(calls[index], "args", {})
                if not isinstance(args, Mapping):
                    continue
                raw_binding = args.get("_hpk_usage_binding")
                if raw_binding is None:
                    continue
                if not isinstance(raw_binding, Mapping):
                    malformed_binding = True
                    continue
                token = str(raw_binding.get("_hpk_selection_token", "") or "")
                if not token:
                    malformed_binding = True
                    continue
                tokens.append(token)
            unique_tokens = tuple(dict.fromkeys(tokens))
            selection = (
                self._selections.get(unique_tokens[0])
                if len(unique_tokens) == 1 and not malformed_binding
                else None
            )
            if continuation:
                if selection is None and not physical_indices:
                    selection = pending.selection
                if selection is not pending.selection:
                    raise HPKEvolvingRuntimeError(
                        "pending continuation changed its frozen selection"
                    )
            if selection is None:
                conflict = (
                    "semantic_attempt_binding_ambiguous"
                    if len(unique_tokens) > 1 or malformed_binding
                    else "semantic_attempt_binding_missing"
                )
                binding = _unbound_transition_binding(conflict=conflict)
            else:
                binding = copy.deepcopy(selection.binding)
                binding["semantic_attempt_group"] = {
                    "grouped": _semantic_attempt_group_is_proven(
                        calls=calls,
                        physical_indices=physical_indices,
                        selection=selection,
                    ),
                    "physical_dispatch_indices": physical_indices,
                }
            binding["action_attempt_nonce"] = nonce
            binding["_hpk_runtime_binding_id"] = self.episode_runtime_binding()[
                "binding_id"
            ]
            if not continuation:
                self._prepared[nonce] = _PreparedAttempt(
                    nonce=nonce,
                    selection=selection,
                    binding=copy.deepcopy(binding),
                    dispatch_observations=[],
                )
            return copy.deepcopy(binding)

    def record_dispatch_observation(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            nonce = str(payload.get("action_attempt_nonce", "") or "").strip()
            prepared = self._prepared.get(nonce)
            if prepared is None:
                raise HPKEvolvingRuntimeError(
                    "dispatch observation has no prepared action attempt"
                )
            prepared.dispatch_observations.append(copy.deepcopy(dict(payload)))

    def _maybe_propose_from_verified_oppose(
        self,
        *,
        transition: ActionEffectTransitionV1,
        public_record: Mapping[str, Any],
        selection: _CandidateSelection,
    ) -> None:
        proposer = self._v2_proposer
        extracted = selection.extracted_strategy
        if (
            proposer is None
            or self._v2_proposal_call_used
            or selection.proposal_id is not None
            or selection.selected_entry is not None
            or extracted is None
            or transition["evidence_verdict"] != "oppose"
            or transition["motion_status"] != "completed"
            or transition["realization_status"] != "satisfied"
            or transition["infrastructure_valid"] is not True
            or transition["oracle_derived"] is True
            or transition["expert_derived"] is True
        ):
            return
        base_geometry = geometric_strategy_v1_to_v2(extracted.geometric_strategy)
        capabilities = selection.proposal_capabilities
        if not _has_non_equivalent_candidate_capability(
            base_geometry=base_geometry,
            capabilities=capabilities,
        ):
            self._v2_proposal_events.append(
                (
                    "hpk_v2_proposal_abstention",
                    {
                        "schema": "roboharn_evo/hpk/v2_proposal_abstention/v1",
                        "trigger_transition_id": transition.stable_id,
                        "reason": "no_non_equivalent_groundable_candidate",
                    },
                )
            )
            return
        observed_effect = transition["observed_effect"]
        observed_predicates = []
        if isinstance(observed_effect, Mapping):
            raw_predicates = observed_effect.get(
                "observed_predicates", observed_effect.get("predicates", [])
            )
            if isinstance(raw_predicates, list):
                observed_predicates = [str(item) for item in raw_predicates]
        public_transition_id = str(
            public_record.get("public_transition_id", "") or ""
        ).strip()
        if not public_transition_id:
            raise HPKEvolvingRuntimeError(
                "verified oppose lacks its public transition reference"
            )
        packet = ProposalEvidencePacketV1(
            {
                "schema": PROPOSAL_EVIDENCE_PACKET_SCHEMA,
                "condition": condition_v1_to_v2(extracted.condition),
                "current_task_strategy": task_strategy_v1_to_v2(
                    extracted.task_strategy
                ),
                "current_geometric_strategy": base_geometry,
                "expected_effect": abstract_effect_v1_to_v2(extracted.expected_effect),
                "observed_effect": {"predicates": observed_predicates},
                "verdict": "oppose",
                "recent_evidence": [
                    {
                        "strategy_summary": {
                            "evidence_ref": public_transition_id,
                            "orientation_relation": base_geometry[
                                "orientation_relation"
                            ],
                            "approach_family": base_geometry["approach_family"],
                            "approach_direction": base_geometry["approach_direction"],
                            "grasp_region": base_geometry["grasp_region"],
                            "realization_status": "satisfied",
                            "observed_predicates": observed_predicates,
                        },
                        "verdict": "oppose",
                    }
                ],
                "available_capabilities": capabilities,
                "retrieved_hpk": [],
                "optional_observations": {
                    "before_image_ref": None,
                    "after_image_ref": None,
                },
            }
        )
        self._v2_proposal_call_used = True
        try:
            result = proposer.propose(packet)
        except Exception as exc:
            self._v2_proposal_events.append(
                (
                    "hpk_v2_proposal_failed",
                    {
                        "schema": "roboharn_evo/hpk/v2_proposal_failed/v1",
                        "trigger_transition_id": transition.stable_id,
                        "packet": packet.to_dict(),
                        "error_type": type(exc).__name__,
                        "retry_count": 0,
                    },
                )
            )
            return
        selected = result.proposals[0] if result.proposals else None
        self._v2_proposal_events.append(
            (
                "hpk_v2_strategy_proposal",
                {
                    "schema": "roboharn_evo/hpk/v2_strategy_proposal_event/v1",
                    "trigger_transition_id": transition.stable_id,
                    "packet": packet.to_dict(),
                    "backend_called": result.backend_called,
                    "backend_audit": result.backend_audit,
                    "validation_errors": list(result.validation_errors),
                    "proposal": None if selected is None else selected.to_dict(),
                    "scheduled": selected is not None,
                },
            )
        )
        if selected is not None:
            self._pending_v2_proposal = _PendingV2Proposal(
                proposal=selected,
                source_transition_id=transition.stable_id,
            )

    def accept_action_transition(
        self,
        *,
        transition: ActionEffectTransitionV1,
        public_record: Mapping[str, Any],
        dispatch_observations: Sequence[Mapping[str, Any]],
    ) -> None:
        with self._lock:
            nonce = str(transition["action_attempt_nonce"] or "").strip()
            prepared = self._prepared.get(nonce)
            if prepared is None:
                raise HPKEvolvingRuntimeError(
                    "accepted transition has no prepared action attempt"
                )
            if prepared.accepted_transition_id is not None:
                raise HPKEvolvingRuntimeError(
                    "an action attempt may accept exactly one transition"
                )
            expected_public = public_transition_projection(transition)
            if canonical_json_bytes(dict(public_record)) != canonical_json_bytes(
                expected_public
            ):
                raise HPKEvolvingRuntimeError(
                    "public transition projection does not match the private record"
                )
            closed_observations = [
                item
                for item in prepared.dispatch_observations
                if item.get("phase") in {"after", "skipped"}
            ]
            if len(dispatch_observations) != len(closed_observations):
                raise HPKEvolvingRuntimeError(
                    "accepted dispatch bundle does not match observed boundaries"
                )
            selection = prepared.selection
            if (
                selection is not None
                and selection.condition is not None
                and selection.task_strategy is not None
            ):
                if transition["condition_id"] != selection.condition.stable_id or (
                    transition["task_strategy_id"] != selection.task_strategy.stable_id
                ):
                    raise HPKEvolvingRuntimeError(
                        "transition strategy IDs differ from the frozen candidate"
                    )
                context_kwargs: dict[str, Any] = {
                    "condition": selection.condition,
                    "task_strategy": selection.task_strategy,
                    "semantic_object": selection.semantic_object,
                    "reasoning": selection.reasoning,
                    "scene_signature": selection.scene_signature,
                }
                if selection.proposal_id is not None:
                    context_kwargs.update(
                        {
                            "proposal_id": selection.proposal_id,
                            "proposed_geometric_strategy": (
                                selection.proposed_geometric_strategy
                            ),
                            "proposed_expected_effect": (
                                selection.proposed_expected_effect
                            ),
                        }
                    )
                self._transition_contexts[transition.stable_id] = (
                    TransitionStrategyContext.from_values(**context_kwargs)
                )
                from roboharn_evo.agent.hpk.semantic_finalizer import (
                    knowledge_attempt_from_transition,
                )

                semantic_context = self._transition_contexts[transition.stable_id]
                candidate, semantic_evidence = knowledge_attempt_from_transition(
                    transition,
                    semantic_context,
                    selected_entry=selection.selected_entry,
                )
                self._semantic_knowledge_attempts.append(
                    (candidate, semantic_evidence)
                )
                candidate_ref = str(
                    transition["selected_candidate_private_ref"] or ""
                ).strip()
                if (
                    transition["evidence_verdict"] == "unverified"
                    and transition["motion_status"] == "completed"
                    and transition["realization_status"] == "satisfied"
                    and transition["operation"] in {"grasp", "place"}
                    and transition["arm"] in {"left", "right"}
                    and candidate_ref
                ):
                    key = (
                        str(transition["operation"]),
                        str(transition["arm"]),
                        candidate_ref,
                    )
                    self._pending_semantic_verifications[key] = (
                        _PendingSemanticVerification(
                            candidate=copy.deepcopy(candidate),
                            operation=key[0],
                            arm=key[1],
                            candidate_ref=key[2],
                        )
                    )
                if selection.proposal_id is not None:
                    self._v2_proposal_events.append(
                        (
                            "hpk_v2_proposal_resolution",
                            {
                                "schema": "roboharn_evo/hpk/v2_proposal_resolution/v1",
                                "proposal_id": selection.proposal_id,
                                "transition_id": transition.stable_id,
                                "verdict": transition["evidence_verdict"],
                                "realized_task_strategy_id": (
                                    selection.task_strategy.stable_id
                                ),
                                "realized_geometric_strategy_id": transition[
                                    "geometric_strategy_id"
                                ],
                                "motion_status": transition["motion_status"],
                                "realization_status": transition["realization_status"],
                            },
                        )
                    )
                else:
                    self._maybe_propose_from_verified_oppose(
                        transition=transition,
                        public_record=public_record,
                        selection=selection,
                    )
            prepared.accepted_transition_id = transition.stable_id

    @property
    def semantic_knowledge_attempts(
        self,
    ) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
        with self._lock:
            return tuple(copy.deepcopy(self._semantic_knowledge_attempts))

    def accept_delayed_runtime_verification(
        self,
        action_effect: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Attach a later deterministic result to the matching completed action."""

        from roboharn_evo.agent.hpk.semantic_knowledge import (
            semantic_evidence_from_runtime_verification,
        )

        payload = dict(action_effect)
        operation = ""
        validation: Mapping[str, Any] | None = None
        candidate_ref = ""
        for candidate_operation, field, candidate_field in (
            ("grasp", "runtime_grasp_validation", "grasp_candidate_id"),
            ("place", "runtime_place_validation", "place_candidate_id"),
        ):
            value = payload.get(field)
            if isinstance(value, Mapping):
                operation = candidate_operation
                validation = value
                candidate_ref = str(value.get(candidate_field) or "").strip()
                break
        if validation is None:
            return None
        arm = str(validation.get("arm") or "").strip().lower()
        if not candidate_ref or arm not in {"left", "right"}:
            return None
        key = (operation, arm, candidate_ref)
        with self._lock:
            pending = self._pending_semantic_verifications.get(key)
            if pending is None:
                return None
            evidence = semantic_evidence_from_runtime_verification(
                payload,
                object_semantics=pending.candidate["object"],
                operation=operation,
                arm=arm,
            )
            self._semantic_knowledge_attempts.append(
                (copy.deepcopy(pending.candidate), evidence)
            )
            if evidence["verdict"] in {"support", "oppose"}:
                self._pending_semantic_verifications.pop(key, None)
            return copy.deepcopy(evidence)

    def finalize_episode(
        self,
        *,
        manifest_path: str | Path,
        trace_path: str | Path,
        expected_manifest_sha256: str,
        expected_trace_sha256: str,
        created_at: str,
    ) -> CoordinatedEpisodeResult:
        """Finalize the active lease once; a child is visible next episode only."""

        with self._lock:
            if self._lease is None:
                raise HPKEvolvingRuntimeError("there is no active episode to finalize")
            if self._episode_finalized:
                raise HPKEvolvingRuntimeError("episode finalization is strictly once")
            expected_runtime_binding = self.episode_runtime_binding()
            writeback_allowed = bool(
                not self._semantic_write_enabled
                and not (
                    self._sequential_run_binding is not None
                    and type(self._lease.expected_episode_id) is int
                    and self._lease.expected_episode_id == 1
                )
            )
            result = self._coordinator.finalize_episode(
                self._lease,
                manifest_path=manifest_path,
                trace_path=trace_path,
                expected_manifest_sha256=expected_manifest_sha256,
                expected_trace_sha256=expected_trace_sha256,
                transition_contexts=dict(self._transition_contexts),
                created_at=created_at,
                expected_runtime_binding=expected_runtime_binding,
                writeback_allowed=writeback_allowed,
            )
            self._episode_finalized = True
            self._finalization_result = result
            if self._semantic_write_enabled:
                valid_read_only_finalization = any(
                    abstention.code == "sequential_writeback_disabled"
                    for abstention in result.finalization.abstentions
                )
                if valid_read_only_finalization:
                    self._semantic_finalization_result = (
                        self._semantic_finalizer.finalize(
                            self._semantic_knowledge_attempts
                        )
                    )
                    self._semantic_knowledge = (
                        self._semantic_finalization_result.knowledge
                    )
                else:
                    self._semantic_finalization_result = SemanticFinalizationResult(
                        status="episode invalid; knowledge unchanged",
                        knowledge_path=self._semantic_knowledge_path,
                        knowledge=self._semantic_knowledge,
                        newly_accepted=False,
                    )
            self._finalized_manifest_path = Path(manifest_path).resolve(strict=True)
            return result


def build_evolving_hpk_runtime(
    policy: HPKRuntimePolicy,
    *,
    task_family: str,
    domain_id: str,
    current_capabilities: HPKDomainCapabilities | Mapping[str, Any] | None = None,
    runtime_provenance: Mapping[str, Any] | None = None,
    finalizer: Any | None = None,
    proposer: VLMStrategyProposer | None = None,
) -> HPKEvolvingRuntime | None:
    """Build evolving once; other modes perform no SnapshotV2 I/O."""

    if not policy.evolving_enabled:
        return None
    return HPKEvolvingRuntime.from_policy(
        policy,
        task_family=task_family,
        domain_id=domain_id,
        current_capabilities=current_capabilities,
        runtime_provenance=runtime_provenance,
        finalizer=finalizer,
        proposer=proposer,
    )


__all__ = [
    "EVOLVING_RUNTIME_SCHEMA",
    "HPKEvolvingRuntime",
    "HPKEvolvingRuntimeError",
    "EvolvingRuntimeProvenance",
    "PublishedFinalizationReceiptV1",
    "PublishedRolloutImportManifestV2",
    "V2ProposalCandidateDecision",
    "build_evolving_hpk_runtime",
    "publish_finalization_receipt_v1",
    "publish_rollout_import_manifest_v2",
]
