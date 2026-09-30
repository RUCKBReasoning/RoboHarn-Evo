#!/usr/bin/env python3
"""Freeze one no-oracle HPK update->probe preregistration without rollout I/O."""

# ruff: noqa: E402 -- direct script execution bootstraps the repository root.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from roboharn_evo.agent.hpk.evolving_runtime import EvolvingRuntimeProvenance
from roboharn_evo.agent.hpk.policy_config import (
    SAFE_EXPLORATION_GEOMETRY_POLICY_RAW_SHA256,
    load_policy_config,
    safe_exploration_geometry_policy_path,
)
from roboharn_evo.agent.hpk.promotion import PromotionPolicyV1
from roboharn_evo.agent.hpk.runtime_binding import build_runtime_binding
from roboharn_evo.agent.hpk.schemas import canonical_json_bytes, stable_content_id
from roboharn_evo.agent.hpk.sequential_experiment import (
    ArtifactRef,
    ResourceUsage,
    SnapshotState,
    build_preregistration,
    load_acceptance_profile,
    publish_preregistration,
)
from roboharn_evo.agent.hpk.snapshot_publisher import publish_child_snapshot
from roboharn_evo.agent.hpk.updater import updater_policy_identity


class PreregistrationBuildError(RuntimeError):
    """The explicit inputs could not be frozen without weakening a gate."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_pinned(path: Path, expected_sha256: str, *, label: str) -> bytes:
    absolute = path.expanduser().absolute()
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise PreregistrationBuildError(f"{label} is unavailable: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PreregistrationBuildError(f"{label} must be a non-symlink regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
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
    ) or len(raw) != after.st_size:
        raise PreregistrationBuildError(f"{label} changed while being read")
    if _sha(raw) != expected_sha256:
        raise PreregistrationBuildError(f"{label} SHA-256 mismatch")
    return raw


def _artifact_ref(path: Path, expected_sha256: str, *, label: str) -> ArtifactRef:
    _read_pinned(path, expected_sha256, label=label)
    return ArtifactRef(str(path.expanduser().absolute()), expected_sha256)


def _write_once(path: Path, raw: bytes, *, label: str) -> ArtifactRef:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise PreregistrationBuildError(f"{label} already exists") from exc
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return ArtifactRef(str(path), _sha(raw))


def _runtime_provenance(
    path: Path,
    expected_sha256: str,
) -> tuple[EvolvingRuntimeProvenance, dict[str, Any]]:
    raw = _read_pinned(
        path,
        expected_sha256,
        label="runtime provenance manifest",
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreregistrationBuildError(
            f"runtime provenance manifest is invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PreregistrationBuildError("runtime provenance manifest must be an object")
    effective = payload.get("effective_config")
    if not isinstance(effective, Mapping):
        raise PreregistrationBuildError(
            "runtime provenance manifest lacks effective_config"
        )
    effective_payload = json.loads(canonical_json_bytes(dict(effective)).decode())
    effective_sha = _sha(canonical_json_bytes(effective_payload))
    if payload.get("effective_config_sha256") != effective_sha:
        raise PreregistrationBuildError(
            "runtime provenance effective_config SHA-256 mismatch"
        )
    summary = {
        "recorded": True,
        "schema": payload.get("schema"),
        "manifest_sha256": expected_sha256,
        "effective_config_sha256": effective_sha,
        "agent_service_identity_sha256": payload.get("agent_service_identity_sha256"),
        "runtime_tree_sha256": payload.get("runtime_tree_sha256"),
    }
    try:
        return EvolvingRuntimeProvenance.from_mapping(summary), effective_payload
    except Exception as exc:
        raise PreregistrationBuildError(
            f"runtime provenance identity is incomplete: {exc}"
        ) from exc


def build_sequential_preregistration(
    *,
    preregistration_path: Path,
    snapshot_output_root: Path,
    segmentation_artifact_root: Path,
    run_output_root: Path,
    task_name: str,
    task_config: str,
    task_definition_path: Path,
    task_definition_sha256: str,
    instruction: str,
    instruction_source_path: Path,
    instruction_source_sha256: str,
    update_seed: int,
    probe_seed: int,
    runtime_provenance_path: Path,
    runtime_provenance_sha256: str,
    promotion_policy_path: Path,
    promotion_policy_sha256: str,
    agent_api_base_url: str,
    sam3_service_url: str,
    created_at: str,
    budgets: Mapping[str, int],
    planned_usage: ResourceUsage,
    enable_v2_proposer: bool = False,
    expected_agent_model: str = "gpt-5.5",
    expected_agent_api_mode: str = "responses_compat",
    expected_reasoning_effort: str = "xhigh",
    expected_response_storage: str = "account_default",
    ckpt_setting: str = "hpk_p0de_gpt55_xhigh",
) -> dict[str, Any]:
    """Create a dedicated empty K0 and publish one canonical preregistration."""

    for label, value in (
        ("preregistration_path", preregistration_path),
        ("snapshot_output_root", snapshot_output_root),
        ("segmentation_artifact_root", segmentation_artifact_root),
        ("run_output_root", run_output_root),
    ):
        if not value.is_absolute():
            raise PreregistrationBuildError(f"{label} must be absolute")
    if preregistration_path.exists() or preregistration_path.is_symlink():
        raise PreregistrationBuildError("preregistration path already exists")
    if snapshot_output_root.exists() or snapshot_output_root.is_symlink():
        raise PreregistrationBuildError("snapshot_output_root already exists")
    if run_output_root.exists() or run_output_root.is_symlink():
        raise PreregistrationBuildError("run_output_root must not exist")
    if (
        not segmentation_artifact_root.is_dir()
        or segmentation_artifact_root.is_symlink()
    ):
        raise PreregistrationBuildError(
            "segmentation_artifact_root must be an existing non-symlink directory"
        )
    if {path.name for path in segmentation_artifact_root.iterdir()} != {
        "inputs",
        "masks",
    }:
        raise PreregistrationBuildError(
            "segmentation_artifact_root must contain exactly empty inputs/masks"
        )
    if any(
        any((segmentation_artifact_root / name).iterdir())
        for name in ("inputs", "masks")
    ):
        raise PreregistrationBuildError(
            "segmentation_artifact_root inputs/masks must be empty"
        )
    for path in segmentation_artifact_root.rglob("*"):
        if path.is_symlink() or not path.is_dir():
            raise PreregistrationBuildError(
                "segmentation_artifact_root must be empty before preregistration"
            )
    if not preregistration_path.parent.is_dir() or not run_output_root.parent.is_dir():
        raise PreregistrationBuildError("output parents must already exist")
    if not instruction or update_seed == probe_seed:
        raise PreregistrationBuildError(
            "instruction must be non-empty and update/probe seeds must differ"
        )

    provenance, effective = _runtime_provenance(
        runtime_provenance_path,
        runtime_provenance_sha256,
    )
    expected_effective = {
        "task_name": task_name,
        "task_config": task_config,
        "policy_name": "policy.roboharn_evo.deploy_policy",
        "instruction_set": "rmbench_original",
        "eval_start_seed": update_seed,
        "eval_start_seeds": [update_seed],
        "n_per_worker": 2,
        "num_workers": 1,
        "agent_api_base_url": agent_api_base_url,
        "sam3_service_url": sam3_service_url,
        "perception_condition": "no_oracle",
        "oracle_objects_enabled": False,
        "expected_agent_model": expected_agent_model,
        "expected_agent_api_mode": expected_agent_api_mode,
        "expected_reasoning_effort": expected_reasoning_effort,
        "expected_response_storage": expected_response_storage,
        "expected_agent_max_concurrent_requests": 1,
        "max_rounds": int(budgets["max_semantic_rounds_per_episode"]),
        "max_control_turns": int(budgets["max_control_turns_per_episode"]),
        "max_no_progress_control_turns": int(
            budgets["max_no_progress_control_turns_per_episode"]
        ),
        "eval_step_limit": int(budgets["max_environment_actions_per_episode"]),
        "max_objects": 8,
        "retry_budget": 0,
        "backend_error_budget": 1,
        "planner_timeout_sec": 600.0,
        "recovery_timeout_sec": 600.0,
        "ood_timeout_sec": 300.0,
        "query_timeout_sec": 600.0,
        "non_formal_diagnostic": True,
        "formal_protocol": False,
        "formal_protocol_version": 0,
        "require_explicit_eval_start_seeds": True,
        "require_sam3_preflight": True,
        "skip_preflight": False,
        "skip_agent_identity_check": False,
        "require_agent_inference_preflight": False,
        "record_runtime_provenance": True,
    }
    mismatched = [
        key
        for key, expected in expected_effective.items()
        if effective.get(key) != expected
    ]
    if mismatched:
        raise PreregistrationBuildError(
            "runtime provenance differs from the sequential contract: "
            + ", ".join(sorted(mismatched))
        )

    geometry_path = safe_exploration_geometry_policy_path()
    geometry = load_policy_config(
        geometry_path,
        SAFE_EXPLORATION_GEOMETRY_POLICY_RAW_SHA256,
        "geometry",
    )
    loaded_promotion = load_policy_config(
        promotion_policy_path,
        promotion_policy_sha256,
        "promotion",
    )
    promotion = PromotionPolicyV1.from_mapping(loaded_promotion.payload)
    if (
        promotion["development_only"] is not True
        or promotion["formal_evaluation_eligible"] is not False
        or promotion["allow_oracle_evidence"] is not False
        or promotion["allow_expert_prior"] is not False
    ):
        raise PreregistrationBuildError(
            "real integration smoke requires the no-prior development policy"
        )
    policy_refs = {
        "geometry": geometry.identity(),
        "promotion": {
            **loaded_promotion.identity(),
            "payload": loaded_promotion.payload,
        },
        "updater": updater_policy_identity(promotion),
    }
    root = publish_child_snapshot(
        parent=None,
        expected_parent_snapshot_id=None,
        expected_parent_manifest_sha256=None,
        entries=(),
        policy_refs=policy_refs,
        evidence_batch=None,
        source_episode_ids=(),
        created_at=created_at,
        destination_root=snapshot_output_root,
        runtime_source_identity=provenance.runtime_source_identity(),
        purpose="evolving_integration",
    )
    initial = SnapshotState.from_loaded(root.snapshot)
    task_definition = _artifact_ref(
        task_definition_path,
        task_definition_sha256,
        label="task definition",
    )
    instruction_source = _artifact_ref(
        instruction_source_path,
        instruction_source_sha256,
        label="instruction source",
    )
    base_config_path = (
        _ROOT / "benchmarks" / "rmbench" / "policy" / "roboharn_evo" / "deploy_policy.yml"
    ).resolve()
    try:
        launch = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PreregistrationBuildError(
            f"cannot load base deploy config: {exc}"
        ) from exc
    if not isinstance(launch, dict):
        raise PreregistrationBuildError("base deploy config must be an object")
    launch.update(
        {
            "policy_name": "policy.roboharn_evo.deploy_policy",
            "task_name": task_name,
            "task_config": task_config,
            "task_config_path": str(task_definition_path),
            "ckpt_setting": ckpt_setting,
            "seed": update_seed,
            "instruction_type": "unseen",
            "instruction_set": "rmbench_original",
            "eval_video_log": False,
            "output_root": str(run_output_root),
            "domain_id": "rmbench",
        }
    )
    launch["eval"] = {
        "test_num": 2,
        "start_seed": update_seed,
        "step_limit": int(budgets["max_environment_actions_per_episode"]),
        "exact_seed_fail_closed": True,
    }
    agent = dict(launch.get("agent") or {})
    agent["enabled"] = True
    monitor = dict(agent.get("monitor") or {})
    monitor["max_retries_per_skill"] = 0
    agent["monitor"] = monitor
    agent["hpk"] = {
        "mode": "evolving",
        "snapshot_manifest": initial.manifest_path,
        "expected_manifest_sha256": initial.manifest_sha256,
        "run_scope": "integration",
        "max_prompt_chars": 4000,
        "allow_expert_prior": False,
        "allow_human_integration_prior": False,
        "allow_oracle_evidence": False,
        "all_hard_mismatch_behavior": "fail_closed",
        "geometry_policy_path": str(geometry_path),
        "expected_geometry_policy_sha256": (
            SAFE_EXPLORATION_GEOMETRY_POLICY_RAW_SHA256
        ),
        "promotion_policy_path": str(promotion_policy_path),
        "expected_promotion_policy_sha256": promotion_policy_sha256,
        "snapshot_output_root": str(snapshot_output_root),
    }
    if enable_v2_proposer:
        agent["hpk"]["proposer"] = {
            "enabled": True,
            "max_proposals": 1,
            "trigger": "verified_oppose",
            "scope": "geometry",
        }
    agent["procedure_experience"] = {"mode": "off"}
    recovery = dict(agent.get("recovery") or {})
    recovery.update(
        {
            "enable_retry": False,
            "enable_reobserve": True,
            "enable_release_guard": True,
            "grasp_transport_policy": "strict",
            "action_geometry_repair_pending_policy": "strict",
            "enable_executable_candidate_temporal_consistency": True,
            "allow_memory_valid_final_grounded_action": False,
            "enable_automatic_self_occlusion_visual_clearance": True,
            "max_recovery_attempts": 0,
        }
    )
    agent["recovery"] = recovery
    agent["pure_tool_control"] = {
        "enabled": True,
        "trigger_step": 0,
        "wait_for_scene_memory": True,
        "max_wait_steps": 2,
        "retry_budget": 0,
        "max_rounds": int(budgets["max_semantic_rounds_per_episode"]),
        "max_control_turns": int(budgets["max_control_turns_per_episode"]),
        "max_no_progress_control_turns": int(
            budgets["max_no_progress_control_turns_per_episode"]
        ),
        "backend_error_budget": 1,
        "empty_plan_replan_threshold": 2,
        "repeat_signal": True,
        "bootstrap_with_planner": True,
        "skip_vla_rollout": True,
    }
    preprocess = dict(agent.get("observation_preprocess") or {})
    preprocess.update(
        {
            "enabled": True,
            "auto_objects": True,
            "query_url": agent_api_base_url.rstrip("/") + "/perception_queries",
            "normalization_url": (
                agent_api_base_url.rstrip("/") + "/normalize_perception_queries"
            ),
            "query_timeout_sec": 600,
            "max_objects": 8,
        }
    )
    segmentation = dict(preprocess.get("segmentation") or {})
    segmentation.update({"backend": "sam3", "service_url": sam3_service_url})
    preprocess["segmentation"] = segmentation
    oracle = dict(preprocess.get("oracle_objects") or {})
    oracle["enabled"] = False
    preprocess["oracle_objects"] = oracle
    agent["observation_preprocess"] = preprocess
    rollout_dump = dict(agent.get("rollout_dump") or {})
    rollout_dump["enabled"] = False
    agent["rollout_dump"] = rollout_dump
    launch["agent"] = agent
    for backend_name, endpoint, timeout in (
        ("planner", "plan", 600),
        ("ood", "ood", 300),
        ("recovery", "recover", 600),
    ):
        backend = dict(launch.get(backend_name) or {})
        backend["backend"] = "agent_api"
        agent_api = dict(backend.get("agent_api") or {})
        agent_api.update(
            {
                "server_url": agent_api_base_url.rstrip("/") + f"/{endpoint}",
                "timeout_sec": timeout,
                "auth_token": "",
                "auth_header": "Authorization",
                "extra_headers": {},
                "extra_body": {},
            }
        )
        backend["agent_api"] = agent_api
        launch[backend_name] = backend
    launch_raw = canonical_json_bytes(launch) + b"\n"
    launch_config = _write_once(
        preregistration_path.with_name("launch_config.json"),
        launch_raw,
        label="launch config",
    )
    base_binding = build_runtime_binding(
        snapshot_id=initial.snapshot_id,
        snapshot_manifest_sha256=initial.manifest_sha256,
        policy_refs=policy_refs,
        provenance_schema=provenance.schema,
        provenance_manifest_sha256=provenance.manifest_sha256,
        config_sha256=provenance.effective_config_sha256,
        model_identity_sha256=provenance.agent_service_identity_sha256,
        runtime_source_sha256=provenance.runtime_tree_sha256,
        evidence_runtime_schema="roboharn_evo/agent/hpk/evolving_runtime/v1",
    )
    profile = load_acceptance_profile()
    run_id = stable_content_id(
        "afkrun",
        {
            "task": task_name,
            "task_config": task_config,
            "instruction_sha256": _sha(instruction.encode()),
            "episodes": [update_seed, probe_seed],
            "runtime_provenance_sha256": runtime_provenance_sha256,
            "snapshot_output_root": str(snapshot_output_root),
            "segmentation_artifact_root": str(segmentation_artifact_root),
            "run_output_root": str(run_output_root),
        },
    )
    payload = {
        "schema": "roboharn_evo/hpk/p0de_sequential_preregistration/v1",
        "acceptance_profile": profile.identity(),
        "task": {
            "task_name": task_name,
            "task_config": task_config,
            "task_definition": task_definition.to_dict(),
            "instruction": instruction,
            "instruction_sha256": _sha(instruction.encode()),
            "instruction_source": instruction_source.to_dict(),
        },
        "launch_config": launch_config.to_dict(),
        "episodes": [
            {"role": "update", "seed": update_seed},
            {"role": "probe", "seed": probe_seed},
        ],
        "execution": profile.schedule,
        "hpk": {
            "mode": "evolving",
            "scope": {
                "name": "integration_development",
                "development_only": True,
                "formal_evaluation": False,
            },
            "initial_snapshot": initial.to_dict(),
            "runtime_binding": base_binding,
            "procedure_experience_mode": "off",
            "allow_oracle_evidence": False,
            "allow_expert_prior": False,
        },
        "manipulation_policy": {
            "grasp_transport_policy": "strict",
            "release_guard_enabled": True,
            "reobserve_enabled": True,
            "action_geometry_repair_pending_policy": "strict",
        },
        "budgets": dict(budgets),
        "planned_usage": planned_usage.to_dict(),
        "run_id": run_id,
        "run_claim_path": str(run_output_root / "hpk_run_claim.json"),
        "output_root": str(run_output_root),
        "snapshot_output_root": str(snapshot_output_root),
        "segmentation_artifact_root": str(segmentation_artifact_root),
    }
    preregistration = build_preregistration(payload)
    published = publish_preregistration(preregistration_path, preregistration)
    return {
        "preregistration_path": str(published.path),
        "preregistration_sha256": published.sha256,
        "preregistration_id": preregistration.preregistration_id,
        "run_id": preregistration.run_id,
        "run_binding_id": preregistration.runtime_binding["run_binding"][
            "run_binding_id"
        ],
        "k0_snapshot_id": initial.snapshot_id,
        "k0_manifest_path": initial.manifest_path,
        "k0_manifest_sha256": initial.manifest_sha256,
        "snapshot_output_root": str(snapshot_output_root),
        "segmentation_artifact_root": str(segmentation_artifact_root),
        "run_output_root": str(run_output_root),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-path", type=Path, required=True)
    parser.add_argument("--snapshot-output-root", type=Path, required=True)
    parser.add_argument("--segmentation-artifact-root", type=Path, required=True)
    parser.add_argument("--run-output-root", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--task-definition-path", type=Path, required=True)
    parser.add_argument("--task-definition-sha256", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--instruction-source-path", type=Path, required=True)
    parser.add_argument("--instruction-source-sha256", required=True)
    parser.add_argument("--update-seed", type=int, required=True)
    parser.add_argument("--probe-seed", type=int, required=True)
    parser.add_argument("--runtime-provenance-path", type=Path, required=True)
    parser.add_argument("--runtime-provenance-sha256", required=True)
    parser.add_argument("--promotion-policy-path", type=Path, required=True)
    parser.add_argument("--promotion-policy-sha256", required=True)
    parser.add_argument("--agent-api-base-url", required=True)
    parser.add_argument("--sam3-service-url", required=True)
    parser.add_argument("--expected-agent-model", default="gpt-5.5")
    parser.add_argument("--expected-agent-api-mode", default="responses_compat")
    parser.add_argument("--expected-reasoning-effort", default="xhigh")
    parser.add_argument("--expected-response-storage", default="account_default")
    parser.add_argument("--ckpt-setting", default="hpk_p0de_gpt55_xhigh")
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--max-external-calls", type=int, required=True)
    parser.add_argument(
        "--max-agent-api-calls",
        type=int,
        required=True,
        help="Dedicated Agent API call cap; use 0 to disable this service-specific cap.",
    )
    parser.add_argument("--max-sam3-calls", type=int, required=True)
    parser.add_argument("--max-images", type=int, required=True)
    parser.add_argument("--max-images-per-request", type=int, required=True)
    parser.add_argument("--max-image-bytes", type=int, required=True)
    parser.add_argument("--max-image-bytes-per-request", type=int, required=True)
    parser.add_argument("--max-wall-time-seconds", type=int, required=True)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    parser.add_argument("--enable-v2-proposer", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    profile = load_acceptance_profile()
    budgets = {
        "max_episodes": 2,
        "max_workers": 1,
        "max_environment_actions_per_episode": profile.per_episode[
            "max_environment_actions"
        ],
        "max_control_turns_per_episode": profile.per_episode["max_control_turns"],
        "max_no_progress_control_turns_per_episode": profile.per_episode[
            "max_no_progress_control_turns"
        ],
        "max_semantic_rounds_per_episode": profile.per_episode["max_semantic_rounds"],
        "max_external_model_calls": args.max_external_calls,
        "max_agent_api_calls": args.max_agent_api_calls,
        "max_sam3_calls": args.max_sam3_calls,
        "max_images": args.max_images,
        "max_images_per_request": args.max_images_per_request,
        "max_image_bytes": args.max_image_bytes,
        "max_image_bytes_per_request": args.max_image_bytes_per_request,
        "max_wall_time_seconds": args.max_wall_time_seconds,
        "max_artifact_bytes": args.max_artifact_bytes,
    }
    result = build_sequential_preregistration(
        preregistration_path=args.preregistration_path.absolute(),
        snapshot_output_root=args.snapshot_output_root.absolute(),
        segmentation_artifact_root=args.segmentation_artifact_root.absolute(),
        run_output_root=args.run_output_root.absolute(),
        task_name=args.task_name,
        task_config=args.task_config,
        task_definition_path=args.task_definition_path.absolute(),
        task_definition_sha256=args.task_definition_sha256,
        instruction=args.instruction,
        instruction_source_path=args.instruction_source_path.absolute(),
        instruction_source_sha256=args.instruction_source_sha256,
        update_seed=args.update_seed,
        probe_seed=args.probe_seed,
        runtime_provenance_path=args.runtime_provenance_path.absolute(),
        runtime_provenance_sha256=args.runtime_provenance_sha256,
        promotion_policy_path=args.promotion_policy_path.absolute(),
        promotion_policy_sha256=args.promotion_policy_sha256,
        agent_api_base_url=args.agent_api_base_url,
        sam3_service_url=args.sam3_service_url,
        created_at=args.created_at,
        budgets=budgets,
        planned_usage=ResourceUsage(
            external_model_calls=args.max_external_calls,
            images=args.max_images,
            image_bytes=args.max_image_bytes,
            artifact_bytes=args.max_artifact_bytes,
        ),
        enable_v2_proposer=bool(args.enable_v2_proposer),
        expected_agent_model=args.expected_agent_model,
        expected_agent_api_mode=args.expected_agent_api_mode,
        expected_reasoning_effort=args.expected_reasoning_effort,
        expected_response_storage=args.expected_response_storage,
        ckpt_setting=args.ckpt_setting,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
