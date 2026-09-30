#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from roboharn_evo.agent.hpk.compatibility import normalize_hpk_experiment_metadata, normalize_v3_rollout_record
from roboharn_evo.agent.hpk.hierarchical_store import load_hierarchical_store  # noqa: E402
from roboharn_evo.agent.failure_boundary_replay import load_failure_boundary  # noqa: E402
from roboharn_evo.benchmark_adapters.rmbench.v31_boundary_bank import (  # noqa: E402
    classify_failure_boundary_v31,
    recoverable_failure_boundary_from_agent_payload_v31,
    validate_failure_category_coverage_v31,
)
from roboharn_evo.benchmark_adapters.rmbench.v31_goal_bridge import (  # noqa: E402
    RMBenchFrozenGoalContextV31,
    build_rmbench_v31_goal_context_from_knowledge,
)


MANIFEST_SCHEMA = "roboharn_evo/rmbench/v31/utility_gate_manifest"
CELL_SCHEMA = "roboharn_evo/rmbench/v31/utility_gate_cell"
USAGE_SCHEMA = "roboharn_evo/rmbench/v31/utility_gate_usage"
REPORT_SCHEMA = "roboharn_evo/rmbench/v31/utility_gate_report"
RUN_RESULT_SCHEMA = "roboharn_evo/rmbench/v31/utility_gate_run_result"
CONDITIONS = ("c0", "c1", "c2")
CATEGORIES = frozenset({"goal_inconsistency", "realization_failure"})


class UtilityGateEvaluationError(RuntimeError):
    """The preregistration or result tree is not an evaluable utility gate."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UtilityGateEvaluationError(f"cannot read JSON object {path}") from exc
    if not isinstance(value, dict):
        raise UtilityGateEvaluationError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise UtilityGateEvaluationError(f"cannot read JSONL {path}") from exc
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UtilityGateEvaluationError(
                f"{path}:{line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise UtilityGateEvaluationError(
                f"{path}:{line_number} must contain one JSON object"
            )
        records.append(normalize_v3_rollout_record(value))
    if not records:
        raise UtilityGateEvaluationError(f"{path} is empty")
    return records


def _one(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if len(matches) != 1:
        raise UtilityGateEvaluationError(
            f"{root} must contain exactly one {pattern}; found {len(matches)}"
        )
    return matches[0]


def _path(base: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise UtilityGateEvaluationError(f"{label} must be an explicit path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise UtilityGateEvaluationError(
            f"{label} does not exist: {candidate}"
        ) from exc


def _event(records: Sequence[Mapping[str, Any]], name: str) -> list[Mapping[str, Any]]:
    return [record for record in records if record.get("event") == name]


def _exact_restore(result: Mapping[str, Any]) -> bool:
    boundary = result.get("boundary")
    match = boundary.get("state_match") if isinstance(boundary, Mapping) else None
    correction = match.get("physics_correction") if isinstance(match, Mapping) else None
    error = (
        correction.get("post_correction_max_numeric_error")
        if isinstance(correction, Mapping)
        else None
    )
    return bool(
        isinstance(match, Mapping)
        and match.get("matches") is True
        and match.get("agent_state_matches") is True
        and match.get("mismatches") == []
        and isinstance(error, (int, float))
        and not isinstance(error, bool)
        and float(error) <= 1e-6
    )


def _timestamps(records: Sequence[Mapping[str, Any]]) -> float | None:
    starts = _event(records, "episode_start")
    ends = _event(records, "episode_end")
    if len(starts) != 1 or len(ends) != 1:
        return None
    start = starts[0].get("timestamp")
    end = ends[0].get("timestamp")
    if not isinstance(start, (int, float)) or isinstance(start, bool):
        return None
    if not isinstance(end, (int, float)) or isinstance(end, bool):
        return None
    return max(0.0, float(end) - float(start))


def _target_effect_verified(
    records: Sequence[Mapping[str, Any]],
    *,
    boundary_step: int,
    manipulated_ref: str,
    target_ref: str,
    operation: str,
) -> bool:
    if operation != "place":
        return False
    for record in _event(records, "runtime_release_verification_resolved"):
        step = record.get("env_step")
        validation = record.get("validation")
        if not isinstance(step, int) or step <= boundary_step:
            continue
        if (
            not isinstance(validation, Mapping)
            or validation.get("verified") is not True
        ):
            continue
        if str(record.get("held_instance_id", "") or "").strip() != manipulated_ref:
            continue
        if str(record.get("target_id", "") or "").strip() != target_ref:
            continue
        if (
            validation.get("object_position_fresh") is True
            and validation.get("object_at_target") is True
            and validation.get("detachment_verified") is True
            and validation.get("stable_across_fresh_observations") is True
        ):
            return True
    return False


def _goal_attempts(
    records: Sequence[Mapping[str, Any]],
    *,
    boundary_step: int,
    manipulated_ref: str,
    target_ref: str,
    operation: str,
) -> tuple[list[dict[str, Any]], int]:
    attempts: list[dict[str, Any]] = []
    substitutions = 0
    for record in _event(records, "operation_candidate_selected"):
        step = record.get("env_step")
        if not isinstance(step, int) or step < boundary_step:
            continue
        action = str(
            record.get("action_mode", record.get("operation", "")) or ""
        ).strip()
        if action != operation:
            continue
        instance_ref = str(
            record.get("instance_id", record.get("track_id", "")) or ""
        ).strip()
        selected_target = str(record.get("target_id", "") or "").strip()
        goal_consistent = bool(
            instance_ref == manipulated_ref and selected_target == target_ref
        )
        substitutions += int(not goal_consistent)
        attempts.append(
            {
                "env_step": step,
                "candidate_id": record.get("candidate_id"),
                "instance_ref": instance_ref,
                "target_ref": selected_target,
                "goal_consistent": goal_consistent,
            }
        )
    return attempts, substitutions


def _usage(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema") not in {USAGE_SCHEMA, "tcm/rmbench/v31/utility_gate_usage"}:
        raise UtilityGateEvaluationError("usage schema mismatch")
    result: dict[str, Any] = {}
    for key in (
        "agent_api_calls",
        "successful_calls",
        "failed_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "automatic_retries",
    ):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise UtilityGateEvaluationError(f"usage.{key} must be non-negative")
        result[key] = item
    usage_complete = value.get("usage_complete")
    if not isinstance(usage_complete, bool):
        raise UtilityGateEvaluationError("usage.usage_complete must be a boolean")
    result["usage_complete"] = usage_complete
    wall = value.get("wall_time_sec")
    if not isinstance(wall, (int, float)) or isinstance(wall, bool) or wall < 0:
        raise UtilityGateEvaluationError("usage.wall_time_sec must be non-negative")
    result["wall_time_sec"] = float(wall)
    return result


def _runtime_identity(
    root: Path,
    *,
    condition: str,
    expected_seed: int,
    common_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    service = _read_json(_one(root / "launcher", "agent_service_identity.json"))
    runtime = _read_json(_one(root / "run_output", "runtime_provenance.json"))
    if (
        service.get("status") != "ok"
        or service.get("model") != common_contract["agent_model"]
        or service.get("reasoning_effort") != common_contract["reasoning_effort"]
        or service.get("max_retries") != 0
        or service.get("sampling") != common_contract["decoding"]
    ):
        raise UtilityGateEvaluationError("Agent service identity differs from plan")
    effective = runtime.get("effective_config")
    if not isinstance(effective, Mapping):
        raise UtilityGateEvaluationError("runtime provenance has no effective config")
    expected_effective = {
        "task_name": common_contract["task"],
        "task_config": common_contract["task_config"],
        "instruction_set": common_contract["instruction_set"],
        "eval_start_seed": expected_seed,
        "eval_step_limit": common_contract["environment_action_budget"],
        "max_control_turns": common_contract["control_turn_budget"],
        "retry_budget": 0,
        "backend_error_budget": 0,
        "perception_condition": "no_oracle",
    }
    for key, expected in expected_effective.items():
        if effective.get(key) != expected:
            raise UtilityGateEvaluationError(
                f"runtime effective config differs for {condition}: {key}"
            )
    content = runtime.get("runtime_content_manifest")
    aggregate = (
        str(content.get("aggregate_sha256", "") or "").strip()
        if isinstance(content, Mapping)
        else ""
    )
    if not aggregate:
        raise UtilityGateEvaluationError("runtime content identity is missing")
    return service, aggregate


def _cell(
    root: Path,
    *,
    condition: str,
    boundary_id: str,
    boundary_path: Path,
    goal_context_path: Path,
    manipulated_ref: str,
    target_ref: str,
    operation: str,
    expected_seed: int,
    expected_instruction: str,
    common_contract: Mapping[str, Any],
) -> dict[str, Any]:
    metadata_path = _one(root, "condition.json")
    trace_path = _one(root, "episode_*_agent_trace.jsonl")
    result_path = _one(root, "failure_boundary_result.json")
    usage_path = _one(root, "usage.json")
    runner_result_path = _one(root, "runner_result.json")
    metadata = normalize_hpk_experiment_metadata(_read_json(metadata_path))
    records = _read_jsonl(trace_path)
    result = _read_json(result_path)
    usage = _usage(usage_path)
    runner_result = _read_json(runner_result_path)
    try:
        service_identity, runtime_content_identity = _runtime_identity(
            root,
            condition=condition,
            expected_seed=expected_seed,
            common_contract=common_contract,
        )
    except Exception as exc:  # noqa: BLE001 - classify as infrastructure invalid
        service_identity = {}
        runtime_content_identity = ""
        errors = [f"runtime identity invalid: {exc}"]
    else:
        errors = []
    if metadata.get("schema") not in {CELL_SCHEMA, "tcm/rmbench/v31/utility_gate_cell"}:
        errors.append("condition schema mismatch")
    if metadata.get("condition") != condition:
        errors.append("condition label mismatch")
    if metadata.get("boundary_id") != boundary_id:
        errors.append("boundary ID mismatch")
    for key, expected in common_contract.items():
        if metadata.get(key) != expected:
            errors.append(f"common experiment field mismatch: {key}")
    if Path(str(metadata.get("boundary_path", ""))).resolve() != boundary_path:
        errors.append("condition boundary path mismatch")
    if Path(str(metadata.get("goal_context_path", ""))).resolve() != goal_context_path:
        errors.append("condition Goal Context path mismatch")
    expected_mode = "off" if condition == "c0" else "full"
    expected_enabled = condition == "c2"
    if metadata.get("hpk_v3_mode") != expected_mode:
        errors.append("HPK v3 mode mismatch")
    if metadata.get("hpk_goal_consistency_enabled") is not expected_enabled:
        errors.append("goal-consistency switch mismatch")
    if metadata.get("retry_budget") != 0 or usage["automatic_retries"] != 0:
        errors.append("automatic retry is not zero")
    if usage["failed_calls"] != 0 or usage["usage_complete"] is not True:
        errors.append("Agent API usage window is incomplete or contains failures")
    if metadata.get("store_writeback_enabled") is not False:
        errors.append("utility-gate Store is not frozen read-only")
    if (
        runner_result.get("schema") not in {RUN_RESULT_SCHEMA, "tcm/rmbench/v31/utility_gate_run_result"}
        or runner_result.get("status") != "completed"
        or runner_result.get("automatic_retry") is not False
        or runner_result.get("store_bytes_unchanged") is not True
    ):
        errors.append("runner did not prove one read-only no-retry cell")
    if result.get("schema") not in {"roboharn_evo/rmbench_failure_boundary_result/v1", "tcm/rmbench_failure_boundary_result/v1"}:
        errors.append("failure-boundary result schema mismatch")
    boundary = result.get("boundary")
    outcome = result.get("outcome")
    if not isinstance(boundary, Mapping) or not isinstance(outcome, Mapping):
        errors.append("failure-boundary result is incomplete")
        boundary = {}
        outcome = {}
    if Path(str(boundary.get("path", ""))).resolve() != boundary_path:
        errors.append("restored boundary path mismatch")
    if result.get("seed") != expected_seed:
        errors.append("seed mismatch")
    if result.get("instruction") != expected_instruction:
        errors.append("instruction mismatch")
    starts = _event(records, "episode_start")
    ends = _event(records, "episode_end")
    if len(starts) != 1 or len(ends) != 1:
        errors.append("trace does not contain exactly one episode")
        episode_end: Mapping[str, Any] = {}
    else:
        episode_end = ends[0]
    if episode_end.get("step_limit") != common_contract["environment_action_budget"]:
        errors.append("environment action budget mismatch")
    if episode_end.get("max_control_turns") != common_contract["control_turn_budget"]:
        errors.append("control-turn budget mismatch")
    validity = episode_end.get("episode_validity")
    if (
        not isinstance(validity, Mapping)
        or validity.get("benchmark_denominator_eligible") is not True
    ):
        errors.append("episode is infrastructure-invalid")
    if not _exact_restore(result):
        errors.append("restored state is not exact")

    boundary_step = int(boundary.get("env_step", 0) or 0)
    attempts, substitutions = _goal_attempts(
        records,
        boundary_step=boundary_step,
        manipulated_ref=manipulated_ref,
        target_ref=target_ref,
        operation=operation,
    )
    verified = _target_effect_verified(
        records,
        boundary_step=boundary_step,
        manipulated_ref=manipulated_ref,
        target_ref=target_ref,
        operation=operation,
    )
    event_counts = Counter(str(record.get("event", "")) for record in records)
    goal_filter_count = event_counts.get("hpk_v31_goal_filter_result", 0)
    retained_count = event_counts.get("hpk_v31_read_only_store_retained", 0)
    if condition == "c2" and attempts and goal_filter_count == 0:
        errors.append("C2 physical attempt has no v3.1 goal-filter trace")
    if condition != "c2" and goal_filter_count:
        errors.append("goal-filter trace appeared outside C2")
    if condition == "c0" and retained_count:
        errors.append("HPK-off condition emitted a Store-retained event")
    if condition in {"c1", "c2"} and retained_count != 1:
        errors.append("full-mode utility cell did not prove read-only Store retention")
    trace_wall = _timestamps(records)
    if trace_wall is not None and abs(trace_wall - usage["wall_time_sec"]) > 5.0:
        errors.append("usage wall time disagrees with the episode trace")
    return {
        "root": str(root),
        "valid": not errors,
        "validation_errors": errors,
        "exact_state_restored": _exact_restore(result),
        "verified_target_effect": verified,
        "recovery_success": bool(outcome.get("success")),
        "additional_environment_actions": int(
            outcome.get("additional_environment_actions", 0) or 0
        ),
        "goal_realization_attempt_count": len(attempts),
        "goal_consistent_attempt_count": len(attempts) - substitutions,
        "goal_substitution_count": substitutions,
        "goal_attempts": attempts,
        "runtime_mappable": bool(
            attempts or event_counts.get("hpk_v31_staged_realization_mapped", 0)
        ),
        "hpk_behavior_changed_claim_count": sum(
            1
            for record in records
            if str(record.get("event", "")).startswith("hpk_v3_")
            and record.get("behavior_changed") is True
        ),
        "event_counts": dict(sorted(event_counts.items())),
        "usage": usage,
        "agent_service_identity": service_identity,
        "runtime_content_identity": runtime_content_identity,
    }


def _invalid_cell(root: Path, exc: Exception) -> dict[str, Any]:
    return {
        "root": str(root),
        "valid": False,
        "validation_errors": [f"{type(exc).__name__}: {exc}"],
        "exact_state_restored": False,
        "verified_target_effect": False,
        "recovery_success": False,
        "additional_environment_actions": 0,
        "goal_realization_attempt_count": 0,
        "goal_consistent_attempt_count": 0,
        "goal_substitution_count": 0,
        "goal_attempts": [],
        "runtime_mappable": False,
        "hpk_behavior_changed_claim_count": 0,
        "event_counts": {},
        "usage": {},
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean(values: Sequence[int | float]) -> float | None:
    return None if not values else float(statistics.fmean(values))


def _common_contract(manifest: Mapping[str, Any], base: Path) -> dict[str, Any]:
    required = (
        "shared_core_baseline",
        "runtime_commit",
        "store_root",
        "agent_model",
        "reasoning_effort",
        "decoding",
        "task_config",
        "instruction_set",
        "grasp_transport_policy",
        "environment_action_budget",
        "control_turn_budget",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise UtilityGateEvaluationError(
            f"manifest is missing common experiment fields: {missing}"
        )
    if str(manifest["shared_core_baseline"]).strip() != "8276000":
        raise UtilityGateEvaluationError("shared-core baseline must be 8276000")
    runtime_commit = str(manifest["runtime_commit"] or "").strip()
    agent_model = str(manifest["agent_model"] or "").strip()
    effort = str(manifest["reasoning_effort"] or "").strip()
    task_config = str(manifest["task_config"] or "").strip()
    instruction_set = str(manifest["instruction_set"] or "").strip()
    grasp_transport_policy = str(manifest["grasp_transport_policy"] or "").strip()
    if (
        not runtime_commit
        or not agent_model
        or not effort
        or not task_config
        or not instruction_set
        or grasp_transport_policy != "strict"
    ):
        raise UtilityGateEvaluationError(
            "runtime, model, task config, and instruction set must be explicit"
        )
    decoding = manifest["decoding"]
    if not isinstance(decoding, Mapping) or not decoding:
        raise UtilityGateEvaluationError("manifest.decoding must be non-empty")
    action_budget = manifest["environment_action_budget"]
    control_budget = manifest["control_turn_budget"]
    for label, value in (
        ("environment_action_budget", action_budget),
        ("control_turn_budget", control_budget),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise UtilityGateEvaluationError(f"manifest.{label} must be positive")
    store_root = _path(base, manifest["store_root"], label="store_root")
    return {
        "shared_core_baseline": "8276000",
        "runtime_commit": runtime_commit,
        "store_root": str(store_root),
        "agent_model": agent_model,
        "reasoning_effort": effort,
        "decoding": dict(decoding),
        "task_config": task_config,
        "instruction_set": instruction_set,
        "grasp_transport_policy": grasp_transport_policy,
        "environment_action_budget": action_budget,
        "control_turn_budget": control_budget,
    }


def evaluate(manifest_path: Path, *, require_complete: bool = True) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    base = manifest_path.parent
    manifest = _read_json(manifest_path)
    if manifest.get("schema") not in {MANIFEST_SCHEMA, "tcm/rmbench/v31/utility_gate_manifest"}:
        raise UtilityGateEvaluationError("utility-gate manifest schema mismatch")
    raw_boundaries = manifest.get("boundaries")
    if not isinstance(raw_boundaries, list) or not raw_boundaries:
        raise UtilityGateEvaluationError("manifest.boundaries must be non-empty")
    if len(raw_boundaries) > 20:
        raise UtilityGateEvaluationError("utility gate permits at most 20 boundaries")
    if require_complete and len(raw_boundaries) < 12:
        raise UtilityGateEvaluationError(
            "the final utility gate requires at least 12 preregistered boundaries"
        )
    common_contract = _common_contract(manifest, base)
    common_contract = {
        "task": str(manifest.get("task", "") or "").strip(),
        **common_contract,
    }
    if not common_contract["task"]:
        raise UtilityGateEvaluationError("manifest.task must be explicit")
    task_knowledge, action_knowledge = load_hierarchical_store(
        common_contract["store_root"]
    )

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_boundaries:
        if not isinstance(raw, Mapping):
            raise UtilityGateEvaluationError("each boundary must be an object")
        boundary_id = str(raw.get("boundary_id", "") or "").strip()
        if not boundary_id or boundary_id in seen_ids:
            raise UtilityGateEvaluationError(
                "boundary IDs must be unique and non-empty"
            )
        seen_ids.add(boundary_id)
        category = str(raw.get("category", "") or "").strip()
        if category not in CATEGORIES:
            raise UtilityGateEvaluationError(
                f"unsupported boundary category: {category}"
            )
        boundary_path = _path(base, raw.get("boundary_path"), label="boundary_path")
        goal_context_path = _path(
            base, raw.get("goal_context_path"), label="goal_context_path"
        )
        context = RMBenchFrozenGoalContextV31.load(goal_context_path)
        task_index = raw.get("task_knowledge_index")
        action_index = raw.get("action_knowledge_index")
        semantic_target_ref = str(
            raw.get("required_target_scene_ref", "") or ""
        ).strip()
        if (
            isinstance(task_index, bool)
            or not isinstance(task_index, int)
            or not 0 <= task_index < len(task_knowledge)
            or isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or not 0 <= action_index < len(action_knowledge)
            or not semantic_target_ref
        ):
            raise UtilityGateEvaluationError(
                "boundary knowledge provenance is missing or invalid"
            )
        manipulated_ref = context.manipulated_object_ref
        target_ref = str(raw.get("expected_runtime_target_ref", "") or "").strip()
        if not target_ref:
            raise UtilityGateEvaluationError(
                "expected_runtime_target_ref must be preregistered"
            )
        boundary_payload = _read_json(boundary_path)
        typed_boundary = load_failure_boundary(boundary_path)
        failure_boundary = recoverable_failure_boundary_from_agent_payload_v31(
            typed_boundary.agent
        )
        holding = None if failure_boundary is None else failure_boundary[0]
        memory_state = typed_boundary.agent.get("memory_state")
        working = (
            memory_state.get("working") if isinstance(memory_state, Mapping) else None
        )
        scene_memory = (
            working.get("scene_memory") if isinstance(working, Mapping) else None
        )
        if holding is None or not isinstance(scene_memory, Mapping):
            raise UtilityGateEvaluationError(
                "boundary no longer contains a recoverable strict-hold failure"
            )
        if (
            classify_failure_boundary_v31(
                typed_boundary.agent,
                held_instance_ref=holding.held_instance_ref,
                expected_target_ref=target_ref,
            )
            != category
        ):
            raise UtilityGateEvaluationError(
                "boundary category differs from captured Runtime failure facts"
            )
        try:
            expected_context = build_rmbench_v31_goal_context_from_knowledge(
                task_knowledge=task_knowledge[task_index],
                action_knowledge=action_knowledge[action_index],
                scene_memory=scene_memory,
                manipulated_object_ref=holding.held_instance_ref,
                required_target_scene_ref=semantic_target_ref,
                expected_runtime_target_ref=target_ref,
                local_trace_ref=holding.active_skill_ref,
            )
        except Exception as exc:  # noqa: BLE001 - invalidate tampered preregistration
            raise UtilityGateEvaluationError(
                "cannot reproduce Goal Context from frozen Store and boundary"
            ) from exc
        if expected_context.to_private_dict() != context.to_private_dict():
            raise UtilityGateEvaluationError(
                "Goal Context differs from frozen Store-derived knowledge"
            )
        expected_seed = boundary_payload.get("seed")
        expected_instruction = boundary_payload.get("instruction")
        if isinstance(expected_seed, bool) or not isinstance(expected_seed, int):
            raise UtilityGateEvaluationError("boundary seed must be an integer")
        if not isinstance(expected_instruction, str) or not expected_instruction:
            raise UtilityGateEvaluationError("boundary instruction must be non-empty")
        raw_cells = raw.get("cells")
        if not isinstance(raw_cells, Mapping) or set(raw_cells) != set(CONDITIONS):
            raise UtilityGateEvaluationError("each boundary requires exactly C0/C1/C2")
        cells: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            root = _path(base, raw_cells[condition], label=f"{condition} root")
            try:
                cells[condition] = _cell(
                    root,
                    condition=condition,
                    boundary_id=boundary_id,
                    boundary_path=boundary_path,
                    goal_context_path=goal_context_path,
                    manipulated_ref=manipulated_ref,
                    target_ref=target_ref,
                    operation=context.goal_contract["operation"],
                    expected_seed=expected_seed,
                    expected_instruction=expected_instruction,
                    common_contract=common_contract,
                )
            except Exception as exc:  # noqa: BLE001 - classify infrastructure invalid
                cells[condition] = _invalid_cell(root, exc)
        valid_pair = all(cells[name]["valid"] for name in CONDITIONS)
        c1_sequence = [
            (item["instance_ref"], item["target_ref"], item["candidate_id"])
            for item in cells["c1"]["goal_attempts"]
        ]
        c2_sequence = [
            (item["instance_ref"], item["target_ref"], item["candidate_id"])
            for item in cells["c2"]["goal_attempts"]
        ]
        rows.append(
            {
                "boundary_id": boundary_id,
                "category": category,
                "boundary_path": str(boundary_path),
                "goal_context_path": str(goal_context_path),
                "expected_manipulated_object_ref": manipulated_ref,
                "expected_runtime_target_ref": target_ref,
                "task_knowledge_index": task_index,
                "action_knowledge_index": action_index,
                "valid_pair": valid_pair,
                "recoverability_observed": any(
                    cells[name]["verified_target_effect"] for name in CONDITIONS
                ),
                "conditions": cells,
                "comparison": {
                    "behavior_changed_c2_vs_c1": c2_sequence != c1_sequence,
                    "non_equivalent_c2_vs_c1": c2_sequence != c1_sequence,
                },
            }
        )

    if require_complete:
        try:
            validate_failure_category_coverage_v31(row["category"] for row in rows)
        except ValueError as exc:
            raise UtilityGateEvaluationError(str(exc)) from exc

    runtime_identities = {
        cell["runtime_content_identity"]
        for row in rows
        for cell in row["conditions"].values()
        if cell.get("runtime_content_identity")
    }
    service_identities = {
        json.dumps(
            cell["agent_service_identity"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in rows
        for cell in row["conditions"].values()
        if cell.get("agent_service_identity")
    }
    if len(runtime_identities) != 1 or len(service_identities) != 1:
        for row in rows:
            row["valid_pair"] = False
            for cell in row["conditions"].values():
                cell["valid"] = False
                cell.setdefault("validation_errors", []).append(
                    "paired cells did not use one exact Runtime and Agent service"
                )
    valid_rows = [row for row in rows if row["valid_pair"]]
    invalid_rate = (len(rows) - len(valid_rows)) / len(rows)
    aggregates: dict[str, Any] = {}
    for condition in CONDITIONS:
        cells = [row["conditions"][condition] for row in valid_rows]
        verified_count = sum(cell["verified_target_effect"] for cell in cells)
        recovery_count = sum(cell["recovery_success"] for cell in cells)
        attempts = sum(cell["goal_realization_attempt_count"] for cell in cells)
        consistent = sum(cell["goal_consistent_attempt_count"] for cell in cells)
        substitutions = sum(cell["goal_substitution_count"] for cell in cells)
        aggregates[condition] = {
            "valid_boundary_count": len(cells),
            "verified_target_effect_count": verified_count,
            "verified_target_effect_rate": _ratio(verified_count, len(cells)),
            "recovery_success_count": recovery_count,
            "recovery_success_rate": _ratio(recovery_count, len(cells)),
            "goal_consistent_realization_rate": _ratio(consistent, attempts),
            "goal_substitution_rate": _ratio(substitutions, attempts),
            "additional_environment_actions_mean": _mean(
                [cell["additional_environment_actions"] for cell in cells]
            ),
            "runtime_mappable_count": sum(cell["runtime_mappable"] for cell in cells),
            "agent_api_calls": sum(
                int(cell["usage"].get("agent_api_calls", 0)) for cell in cells
            ),
            "total_tokens": sum(
                int(cell["usage"].get("total_tokens", 0)) for cell in cells
            ),
            "wall_time_sec": sum(
                float(cell["usage"].get("wall_time_sec", 0.0)) for cell in cells
            ),
        }
    c1_rate = aggregates["c1"]["verified_target_effect_rate"]
    c2_rate = aggregates["c2"]["verified_target_effect_rate"]
    delta = None if c1_rate is None or c2_rate is None else c2_rate - c1_rate
    c2_substitution = aggregates["c2"]["goal_substitution_rate"]
    go = bool(
        len(valid_rows) >= 12
        and invalid_rate < 0.05
        and c2_rate is not None
        and c2_rate >= 0.30
        and delta is not None
        and delta >= 0.20
        and c2_substitution == 0.0
    )
    no_go = bool(
        len(valid_rows) >= 20
        and all(row["recoverability_observed"] for row in valid_rows)
        and c2_rate is not None
        and c2_rate <= 0.10
    )
    decision = "GO" if go else "NO_GO" if no_go else "INCONCLUSIVE"
    return {
        "schema": REPORT_SCHEMA,
        "manifest_path": str(manifest_path),
        "common_experiment_contract": common_contract,
        "boundary_count": len(rows),
        "valid_paired_boundary_count": len(valid_rows),
        "observed_recoverable_boundary_count": sum(
            row["recoverability_observed"] for row in valid_rows
        ),
        "infrastructure_invalid_rate": invalid_rate,
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "conditions": aggregates,
        "comparison": {
            "c2_minus_c1_verified_target_effect_rate": delta,
            "behavior_changed_boundary_count": sum(
                row["comparison"]["behavior_changed_c2_vs_c1"] for row in valid_rows
            ),
            "non_equivalent_boundary_count": sum(
                row["comparison"]["non_equivalent_c2_vs_c1"] for row in valid_rows
            ),
        },
        "decision": decision,
        "go_gate_passed": go,
        "no_go_gate_triggered": no_go,
        "rows": rows,
        "scientific_boundary": (
            "Behavior-change claims are descriptive only. Positive utility requires "
            "fresh deterministic Runtime verification at the preregistered target."
        ),
    }


def render_summary(report: Mapping[str, Any]) -> str:
    lines = [
        "# HPK v3.1 RMBench Utility Gate",
        "",
        f"Decision: **{report['decision']}**",
        "",
        f"Valid paired boundaries: {report['valid_paired_boundary_count']} / {report['boundary_count']}",
        f"Infrastructure-invalid rate: {report['infrastructure_invalid_rate']:.1%}",
        "",
        "| Condition | VerifiedTargetEffect | RecoverySuccess | Goal-consistent realization | Goal substitution | Mean additional actions |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        value = report["conditions"][condition]

        def percent(item: Any) -> str:
            return "n/a" if item is None else f"{item:.1%}"

        mean_actions = value["additional_environment_actions_mean"]
        lines.append(
            "| "
            + " | ".join(
                (
                    condition.upper(),
                    percent(value["verified_target_effect_rate"]),
                    percent(value["recovery_success_rate"]),
                    percent(value["goal_consistent_realization_rate"]),
                    percent(value["goal_substitution_rate"]),
                    "n/a" if mean_actions is None else f"{mean_actions:.1f}",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "Behavior change is not counted as utility. A positive target effect "
            "requires fresh deterministic Runtime placement verification.",
            "",
        )
    )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir.absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise UtilityGateEvaluationError(f"refusing to overwrite {output_dir}")
    report = evaluate(
        args.manifest,
        require_complete=not args.allow_incomplete,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        render_summary(report),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
