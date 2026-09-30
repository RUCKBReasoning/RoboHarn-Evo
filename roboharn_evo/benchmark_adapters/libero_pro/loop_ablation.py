"""Audit the matched lite versus perception-memory LIBERO ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.compatibility import normalize_hpk_runtime_provenance, normalize_v3_rollout_record


class LiberoLoopAblationError(ValueError):
    """The four-condition loop-profile ablation is incomplete or unmatched."""


_CONDITIONS = {
    "A_lite_off": ("lite", False),
    "B_memory_off": ("perception_memory", False),
    "C_lite_rmbench_hpk": ("lite", True),
    "D_memory_rmbench_hpk": ("perception_memory", True),
}
_MATCHED_RUN_FIELDS = (
    "horizon",
    "seed",
    "task_id",
    "init_state_id",
    "camera_height",
    "camera_width",
    "policy_replan_steps",
    "policy_settle_steps",
    "agent_max_planner_calls",
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiberoLoopAblationError(f"cannot read {path}") from exc
    if not isinstance(value, dict):
        raise LiberoLoopAblationError(f"{path.name} must contain one object")
    return value


def _jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LiberoLoopAblationError(f"cannot read {path}") from exc
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LiberoLoopAblationError(
                f"{path.name} line {line_number} is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise LiberoLoopAblationError(
                f"{path.name} line {line_number} must contain an object"
            )
        result.append(value)
    return tuple(result)


def _summary(
    name: str,
    provenance: dict[str, Any],
    trace: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    provenance = normalize_hpk_runtime_provenance(provenance)
    trace = tuple(normalize_v3_rollout_record(record) for record in trace)
    expected_profile, hpk_enabled = _CONDITIONS[name]
    starts = [value for value in trace if value.get("event") == "episode_start"]
    ends = [value for value in trace if value.get("event") == "episode_end"]
    if len(starts) != 1 or len(ends) != 1:
        raise LiberoLoopAblationError(
            f"{name} must contain exactly one episode_start and episode_end"
        )
    hpk = provenance.get("hpk")
    loop = provenance.get("simulator", {}).get("roboharn_agent_loop")
    if not isinstance(hpk, dict) or not isinstance(loop, dict):
        raise LiberoLoopAblationError(f"{name} lacks HPK or Agent-loop provenance")
    if hpk.get("loop_profile") != expected_profile:
        raise LiberoLoopAblationError(f"{name} loop profile mismatch")
    if bool(hpk.get("perception_memory_enabled")) is not (
        expected_profile == "perception_memory"
    ):
        raise LiberoLoopAblationError(f"{name} perception mode mismatch")
    if bool(hpk.get("action_knowledge_enabled")) is not hpk_enabled:
        raise LiberoLoopAblationError(f"{name} Action Knowledge mode mismatch")
    if (
        starts[0].get("loop_profile") != expected_profile
        or ends[0].get("loop_profile") != expected_profile
    ):
        raise LiberoLoopAblationError(f"{name} trace profile mismatch")

    requests = [value for value in trace if value.get("event") == "executor_request"]
    before = [
        value
        for value in trace
        if value.get("event") == "perception_memory_before_action"
    ]
    after = [
        value
        for value in trace
        if value.get("event") == "perception_memory_after_action"
    ]
    stats = loop.get("perception_memory_stats")
    if expected_profile == "lite":
        if before or after or stats is not None:
            raise LiberoLoopAblationError(
                f"{name} lite profile unexpectedly used perception memory"
            )
        perception = None
    else:
        if len(before) != len(requests) or len(after) != len(requests):
            raise LiberoLoopAblationError(
                f"{name} perception events do not cover every executor call"
            )
        if not isinstance(stats, dict) or stats.get("profile") != expected_profile:
            raise LiberoLoopAblationError(f"{name} lacks perception statistics")
        perception = {
            "query_calls": int(stats.get("query_calls", 0)),
            "sam3_calls": int(stats.get("sam3_calls", 0)),
            "verdicts": dict(stats.get("verdicts", {})),
            "final_scene_memory": stats.get("final_scene_memory"),
        }

    action_adopted = 0
    action_grounding_fresh = 0
    action_grounding_reused = 0
    suppression_reasons: dict[str, int] = {}
    for request in requests:
        usage = request.get("hpk_v3_action_usage")
        if not hpk_enabled:
            if usage is not None:
                raise LiberoLoopAblationError(
                    f"{name} Off condition contains Action Knowledge usage"
                )
            continue
        if not isinstance(usage, dict):
            raise LiberoLoopAblationError(
                f"{name} HPK condition lacks Action Knowledge audit"
            )
        if usage.get("knowledge_adopted") is True:
            action_adopted += 1
            if usage.get("grounding_reused") is True:
                action_grounding_reused += 1
            else:
                action_grounding_fresh += 1
        reason = usage.get("suppression_reason")
        if reason:
            key = str(reason)
            suppression_reasons[key] = suppression_reasons.get(key, 0) + 1

    return {
        "profile": expected_profile,
        "hpk_enabled": hpk_enabled,
        "success": bool(loop.get("success")),
        "steps": int(loop.get("executed_actions", 0)),
        "planner_calls": int(loop.get("planner_calls", 0)),
        "recovery_boundaries": int(loop.get("recovery_boundaries", 0)),
        "action_knowledge_adopted": action_adopted,
        "action_grounding_fresh": action_grounding_fresh,
        "action_grounding_reused": action_grounding_reused,
        "action_knowledge_suppressed": suppression_reasons,
        "perception_memory": perception,
    }


def audit_loop_profile_ablation(root: str | Path) -> dict[str, Any]:
    """Validate four matched runs and return descriptive ablation deltas."""

    directory = Path(root).expanduser().resolve(strict=True)
    legacy_names = {
        "C_lite_rmbench_hpk": "C_lite_rmbench_afk",
        "D_memory_rmbench_hpk": "D_memory_rmbench_afk",
    }
    condition_paths = {}
    for name in _CONDITIONS:
        candidates = {directory / name, directory / legacy_names.get(name, name)}
        existing = [path for path in candidates if path.is_dir()]
        if len(existing) != 1:
            raise LiberoLoopAblationError(f"{name} requires exactly one condition directory")
        condition_paths[name] = existing[0]
    provenances = {
        name: _json(condition_paths[name] / "provenance.json") for name in _CONDITIONS
    }
    traces = {
        name: _jsonl(condition_paths[name] / "agent_loop_trace.jsonl")
        for name in _CONDITIONS
    }
    baseline = provenances["A_lite_off"]
    for name, provenance in provenances.items():
        if name == "A_lite_off":
            continue
        for field in _MATCHED_RUN_FIELDS:
            if baseline["run"].get(field) != provenance["run"].get(field):
                raise LiberoLoopAblationError(
                    f"{name} does not match A_lite_off field {field}"
                )
        if baseline.get("task") != provenance.get("task"):
            raise LiberoLoopAblationError(f"{name} task differs from A_lite_off")
        if baseline.get("policy") != provenance.get("policy"):
            raise LiberoLoopAblationError(f"{name} policy differs from A_lite_off")
    summaries = {
        name: _summary(name, provenances[name], traces[name]) for name in _CONDITIONS
    }

    def delta(left: str, right: str) -> dict[str, int]:
        return {
            "success_delta": int(summaries[right]["success"])
            - int(summaries[left]["success"]),
            "steps_delta": summaries[right]["steps"] - summaries[left]["steps"],
            "recovery_boundaries_delta": summaries[right]["recovery_boundaries"]
            - summaries[left]["recovery_boundaries"],
            "planner_calls_delta": summaries[right]["planner_calls"]
            - summaries[left]["planner_calls"],
        }

    return {
        "schema": "roboharn_evo/libero_loop_profile_ablation/v1",
        "conditions": summaries,
        "paired_deltas": {
            "memory_without_hpk": delta("A_lite_off", "B_memory_off"),
            "memory_with_hpk": delta("C_lite_rmbench_hpk", "D_memory_rmbench_hpk"),
            "hpk_in_lite": delta("A_lite_off", "C_lite_rmbench_hpk"),
            "hpk_with_memory": delta("B_memory_off", "D_memory_rmbench_hpk"),
        },
        "claim_scope": (
            "one matched four-condition ablation; repeat across preregistered tasks "
            "and seeds before claiming performance improvement"
        ),
    }


__all__ = ["LiberoLoopAblationError", "audit_loop_profile_ablation"]
