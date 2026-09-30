
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.compatibility import normalize_hpk_runtime_provenance, normalize_v3_rollout_record


class LiberoP0FeasibilityError(ValueError):
    """The feasibility matrix is incomplete or not matched."""


CONDITIONS = (
    "native_pi05",
    "agent_off",
    "task_hpk",
    "action_hpk",
    "task_action_hpk",
)
HPK_CONDITIONS = CONDITIONS[2:]
_EXPECTED_LAYERS = {
    "agent_off": (False, False),
    "task_hpk": (True, False),
    "action_hpk": (False, True),
    "task_action_hpk": (True, True),
}
_MATCHED_RUN_FIELDS = (
    "suite",
    "task_id",
    "init_state_id",
    "seed",
    "horizon",
    "camera_height",
    "camera_width",
    "policy_replan_steps",
    "policy_settle_steps",
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiberoP0FeasibilityError(f"cannot read {path}") from exc
    if not isinstance(value, dict):
        raise LiberoP0FeasibilityError(f"{path} must contain one JSON object")
    return value


def _jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LiberoP0FeasibilityError(f"cannot read {path}") from exc
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LiberoP0FeasibilityError(
                f"{path} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise LiberoP0FeasibilityError(
                f"{path} line {line_number} must contain an object"
            )
        values.append(value)
    return tuple(values)


def _policy_actions(
    trace: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, ...], ...]:
    result: list[tuple[float, ...]] = []
    for event in trace:
        if event.get("source") != "policy" or "action" not in event:
            continue
        action = event["action"]
        if isinstance(action, (str, bytes)) or not isinstance(action, Sequence):
            raise LiberoP0FeasibilityError("policy action must be an array")
        values = tuple(float(item) for item in action)
        if len(values) != 7 or not all(math.isfinite(item) for item in values):
            raise LiberoP0FeasibilityError("policy action must be finite native 7D")
        result.append(values)
    return tuple(result)


def _route_observation(usage: Mapping[str, Any], routes: Counter[str]) -> None:
    path = usage.get("exhaustive_or_family_path")
    if isinstance(path, str) and path:
        routes[path] += 1


def _agent_summary(
    condition: str,
    provenance: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    provenance = normalize_hpk_runtime_provenance(provenance)
    trace = tuple(normalize_v3_rollout_record(record) for record in trace)
    expected_task, expected_action = _EXPECTED_LAYERS[condition]
    hpk = provenance.get("hpk")
    loop = provenance.get("simulator", {}).get("roboharn_agent_loop")
    if not isinstance(hpk, Mapping) or not isinstance(loop, Mapping):
        raise LiberoP0FeasibilityError(f"{condition} lacks HPK or loop provenance")
    if hpk.get("task_knowledge_enabled") is not expected_task:
        raise LiberoP0FeasibilityError(f"{condition} Task Knowledge mode mismatch")
    if hpk.get("action_knowledge_enabled") is not expected_action:
        raise LiberoP0FeasibilityError(f"{condition} Action Knowledge mode mismatch")
    if hpk.get("persistent_store_write") is not False:
        raise LiberoP0FeasibilityError(f"{condition} Store must remain read-only")

    plans = tuple(event for event in trace if event.get("event") == "planner_decision")
    requests = tuple(
        event for event in trace if event.get("event") == "executor_request"
    )
    ends = tuple(event for event in trace if event.get("event") == "episode_end")
    if len(ends) != 1 or len(plans) != len(requests):
        raise LiberoP0FeasibilityError(
            f"{condition} requires matched planner/executor events and one episode end"
        )
    revisions = {
        event.get("executor_call_index"): event.get("scene_memory", {}).get("revision")
        for event in trace
        if event.get("event") == "perception_memory_before_action"
        and isinstance(event.get("scene_memory"), Mapping)
    }

    task_opportunities = task_adopted = task_behavior_changes = 0
    action_opportunities = action_adopted = action_behavior_changes = 0
    same_state_repeated_adoptions = 0
    suppressions: Counter[str] = Counter()
    route_paths: Counter[str] = Counter()
    selected_families: Counter[str] = Counter()
    selected_atomic: Counter[str] = Counter()
    seen_action_state: set[tuple[str, str, Any]] = set()

    for plan in plans:
        usage = plan.get("hpk_v3_subtask_usage")
        if not expected_task:
            if usage is not None:
                raise LiberoP0FeasibilityError(
                    f"{condition} unexpectedly contains Task Knowledge usage"
                )
            continue
        if not isinstance(usage, Mapping):
            raise LiberoP0FeasibilityError(f"{condition} lacks Task usage audit")
        task_opportunities += 1
        _route_observation(usage, route_paths)
        for family in usage.get("selected_family_summaries", []):
            if isinstance(family, Mapping) and family.get("family_name"):
                selected_families[str(family["family_name"])] += 1
        atomic = usage.get("selected_atomic_knowledge")
        if isinstance(atomic, int) and not isinstance(atomic, bool):
            selected_atomic[f"task:{atomic}"] += 1
        if usage.get("knowledge_adopted") is True:
            task_adopted += 1
        if usage.get("subtask_before") != usage.get("subtask_after"):
            task_behavior_changes += 1

    for request in requests:
        usage = request.get("hpk_v3_action_usage")
        if not expected_action:
            if usage is not None:
                raise LiberoP0FeasibilityError(
                    f"{condition} unexpectedly contains Action Knowledge usage"
                )
            continue
        if not isinstance(usage, Mapping):
            raise LiberoP0FeasibilityError(f"{condition} lacks Action usage audit")
        action_opportunities += 1
        _route_observation(usage, route_paths)
        for family in usage.get("selected_family_summaries", []):
            if isinstance(family, Mapping) and family.get("family_name"):
                selected_families[str(family["family_name"])] += 1
        atomic = usage.get("selected_atomic_knowledge")
        action = str(usage.get("action") or "unknown")
        if isinstance(atomic, int) and not isinstance(atomic, bool):
            selected_atomic[f"action:{atomic}"] += 1
        if usage.get("knowledge_adopted") is True:
            action_adopted += 1
            call_index = request.get("executor_call_index")
            state_key = (action, str(atomic), revisions.get(call_index))
            if state_key in seen_action_state:
                same_state_repeated_adoptions += 1
            seen_action_state.add(state_key)
        if request.get("prompt") != request.get("prompt_before_action_hpk"):
            action_behavior_changes += 1
        reason = usage.get("suppression_reason")
        if isinstance(reason, str) and reason:
            suppressions[reason] += 1

    end = ends[0]
    if end.get("benchmark_success") is not loop.get("success"):
        raise LiberoP0FeasibilityError(f"{condition} outcome trace mismatch")
    return {
        "success": bool(loop.get("success")),
        "steps": int(loop.get("executed_actions", 0)),
        "planner_calls": int(loop.get("planner_calls", 0)),
        "recovery_boundaries": int(loop.get("recovery_boundaries", 0)),
        "task_knowledge": {
            "opportunities": task_opportunities,
            "adopted": task_adopted,
            "behavior_changes": task_behavior_changes,
        },
        "action_knowledge": {
            "opportunities": action_opportunities,
            "adopted": action_adopted,
            "behavior_changes": action_behavior_changes,
            "same_state_repeated_adoptions": same_state_repeated_adoptions,
            "suppressions": dict(sorted(suppressions.items())),
        },
        "family_route_paths": dict(sorted(route_paths.items())),
        "selected_families": dict(sorted(selected_families.items())),
        "selected_atomic_knowledge": dict(sorted(selected_atomic.items())),
        "policy_actions": _policy_actions(trace),
    }


def _native_summary(
    provenance: Mapping[str, Any], trace: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rollout = provenance.get("simulator", {}).get("native_policy_off")
    if not isinstance(rollout, Mapping):
        raise LiberoP0FeasibilityError("native condition lacks rollout provenance")
    return {
        "success": bool(rollout.get("success")),
        "steps": int(rollout.get("executed_actions", 0)),
        "planner_calls": 0,
        "recovery_boundaries": 0,
        "task_knowledge": {"opportunities": 0, "adopted": 0, "behavior_changes": 0},
        "action_knowledge": {
            "opportunities": 0,
            "adopted": 0,
            "behavior_changes": 0,
            "same_state_repeated_adoptions": 0,
            "suppressions": {},
        },
        "family_route_paths": {},
        "selected_families": {},
        "selected_atomic_knowledge": {},
        "policy_actions": _policy_actions(trace),
    }


def _action_delta(
    baseline: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]
) -> dict[str, int | bool]:
    aligned = min(len(baseline), len(candidate))
    changed = sum(
        tuple(baseline[index]) != tuple(candidate[index]) for index in range(aligned)
    )
    return {
        "aligned_policy_action_steps": aligned,
        "changed_aligned_action_steps": changed,
        "policy_action_count_delta": len(candidate) - len(baseline),
        "sequence_changed": changed > 0 or len(candidate) != len(baseline),
    }


def _utility_label(off: Mapping[str, Any], hpk: Mapping[str, Any]) -> str:
    behavior_changes = int(hpk["task_knowledge"]["behavior_changes"]) + int(
        hpk["action_knowledge"]["behavior_changes"]
    )
    if behavior_changes == 0:
        return "neutral_no_knowledge_behavior_change"
    if bool(hpk["success"]) != bool(off["success"]):
        return "helpful" if hpk["success"] else "harmful"
    if hpk["success"] and off["success"]:
        if int(hpk["steps"]) < int(off["steps"]):
            return "helpful"
        if int(hpk["steps"]) > int(off["steps"]):
            return "harmful"
    return "neutral"


def audit_p0_feasibility_matrix(root: str | Path) -> dict[str, Any]:
    """Validate and summarize one 3-init by 5-condition read-only matrix."""

    directory = Path(root).expanduser().resolve(strict=True)
    config = _json(directory / "run_config.json")
    init_states = config.get("init_states")
    if not isinstance(init_states, list) or len(init_states) != 3:
        raise LiberoP0FeasibilityError(
            "run_config must freeze exactly three init states"
        )
    store_check = _json(directory / "store_read_only_check.json")
    if store_check.get("unchanged") is not True:
        raise LiberoP0FeasibilityError("evaluation Store changed during the matrix")

    rows: list[dict[str, Any]] = []
    labels: Counter[str] = Counter()
    parity: list[dict[str, Any]] = []
    for init_state in init_states:
        init_root = directory / f"init_{int(init_state)}"
        legacy_names = {"task_hpk": "task_afk", "action_hpk": "action_afk", "task_action_hpk": "task_action_afk"}
        condition_paths = {}
        for condition in CONDITIONS:
            candidates = {init_root / condition, init_root / legacy_names.get(condition, condition)}
            existing = [path for path in candidates if path.is_dir()]
            if len(existing) != 1:
                raise LiberoP0FeasibilityError(f"{condition} requires exactly one condition directory")
            condition_paths[condition] = existing[0]
        provenances = {
            condition: _json(condition_paths[condition] / "provenance.json")
            for condition in CONDITIONS
        }
        traces = {
            condition: _jsonl(
                condition_paths[condition]
                / (
                    "native_policy_trace.jsonl"
                    if condition == "native_pi05"
                    else "agent_loop_trace.jsonl"
                )
            )
            for condition in CONDITIONS
        }
        baseline = provenances["agent_off"]
        for condition, provenance in provenances.items():
            for field in _MATCHED_RUN_FIELDS:
                if provenance.get("run", {}).get(field) != baseline.get("run", {}).get(
                    field
                ):
                    raise LiberoP0FeasibilityError(
                        f"init {init_state} {condition} mismatches run field {field}"
                    )
            if provenance.get("task") != baseline.get("task"):
                raise LiberoP0FeasibilityError(
                    f"init {init_state} {condition} task mismatch"
                )
            if provenance.get("policy") != baseline.get("policy"):
                raise LiberoP0FeasibilityError(
                    f"init {init_state} {condition} policy mismatch"
                )
        summaries = {
            "native_pi05": _native_summary(
                provenances["native_pi05"], traces["native_pi05"]
            ),
            **{
                condition: _agent_summary(
                    condition, provenances[condition], traces[condition]
                )
                for condition in CONDITIONS[1:]
            },
        }
        off = summaries["agent_off"]
        parity.append(
            {
                "init_state_id": init_state,
                "same_task": True,
                "same_native_7d_action_contract": True,
                "same_horizon_replan_settle": True,
                "official_success_checker_preserved": True,
                "native_success": summaries["native_pi05"]["success"],
                "agent_off_success": off["success"],
            }
        )
        for condition in CONDITIONS:
            summary = summaries[condition]
            row = {
                "init_state_id": init_state,
                "condition": condition,
                **{
                    key: value
                    for key, value in summary.items()
                    if key != "policy_actions"
                },
            }
            if condition != "agent_off":
                row["action_sequence_vs_agent_off"] = _action_delta(
                    off["policy_actions"], summary["policy_actions"]
                )
            if condition in HPK_CONDITIONS:
                label = _utility_label(off, summary)
                row["utility_vs_agent_off"] = label
                labels[label] += 1
            rows.append(row)

    hpk_rows = [row for row in rows if row["condition"] in HPK_CONDITIONS]
    behavior_gate = any(
        row["task_knowledge"]["behavior_changes"]
        or row["action_knowledge"]["behavior_changes"]
        for row in hpk_rows
    )
    repeated = sum(
        row["action_knowledge"]["same_state_repeated_adoptions"] for row in hpk_rows
    )
    helpful = labels["helpful"]
    harmful = labels["harmful"]
    capability_mapping = _json(directory / "capability_mapping.json")
    mappings = capability_mapping.get("mappings")
    mapping_predeclared = bool(mappings) and all(
        isinstance(value, Mapping) and value.get("defined_before_outcome") is True
        for value in mappings
    )
    gates = {
        "observable_behavior_change": behavior_gate,
        "helpful_greater_than_harmful": helpful > harmful,
        "no_same_state_repeated_action_adoption": repeated == 0,
        "capability_mapping_predeclared": mapping_predeclared,
        "native_agent_off_parity_explainable": all(
            value["same_native_7d_action_contract"]
            and value["official_success_checker_preserved"]
            for value in parity
        ),
    }
    return {
        "schema": "roboharn_evo/libero_p0_read_only_feasibility/v1",
        "status": "DEVELOPMENT",
        "run_root": str(directory),
        "producing_commit": config.get("producing_commit"),
        "working_tree_clean": config.get("working_tree_clean"),
        "rows": rows,
        "condition_utility_counts": dict(sorted(labels.items())),
        "parity": parity,
        "store_read_only": store_check,
        "gates": gates,
        "go_no_go": "GO" if all(gates.values()) else "NO_GO",
        "claim_scope": (
            "read-only LIBERO feasibility pilot; Action Knowledge is prompt-level "
            "native-policy guidance, not RMBench candidate-level 6D reranking"
        ),
    }


__all__ = [
    "HPK_CONDITIONS",
    "CONDITIONS",
    "LiberoP0FeasibilityError",
    "audit_p0_feasibility_matrix",
]
