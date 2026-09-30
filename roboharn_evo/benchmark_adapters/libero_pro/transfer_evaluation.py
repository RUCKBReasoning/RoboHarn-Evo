
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.schemas import reject_private_transferable
from roboharn_evo.agent.hpk.compatibility import normalize_hpk_runtime_provenance, normalize_v3_rollout_record


class LiberoTransferEvaluationError(ValueError):
    """A four-condition transfer result is incomplete or internally inconsistent."""


_CONDITIONS = (
    "off",
    "rmbench_task",
    "rmbench_task_action",
    "libero_native",
)
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
        raise LiberoTransferEvaluationError(f"cannot read {path}") from exc
    if not isinstance(value, dict):
        raise LiberoTransferEvaluationError(f"{path.name} must contain one object")
    return value


def _jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LiberoTransferEvaluationError(f"cannot read {path}") from exc
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LiberoTransferEvaluationError(
                f"{path.name} line {line_number} is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise LiberoTransferEvaluationError(
                f"{path.name} line {line_number} must contain an object"
            )
        result.append(value)
    return tuple(result)


def _condition_summary(
    name: str,
    *,
    provenance: dict[str, Any],
    trace: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    provenance = normalize_hpk_runtime_provenance(provenance)
    trace = tuple(normalize_v3_rollout_record(record) for record in trace)
    starts = [value for value in trace if value.get("event") == "episode_start"]
    ends = [value for value in trace if value.get("event") == "episode_end"]
    if len(starts) != 1 or len(ends) != 1:
        raise LiberoTransferEvaluationError(
            f"{name} must contain exactly one episode_start and episode_end"
        )
    plans = tuple(value for value in trace if value.get("event") == "planner_decision")
    requests = tuple(
        value for value in trace if value.get("event") == "executor_request"
    )
    if not plans or len(plans) != len(requests):
        raise LiberoTransferEvaluationError(
            f"{name} planner and executor request counts do not match"
        )
    plans_by_index = {value.get("planner_call_index"): value for value in plans}
    if len(plans_by_index) != len(plans):
        raise LiberoTransferEvaluationError(f"{name} planner indices are not unique")
    task_enabled = name != "off"
    action_enabled = name in {"rmbench_task_action", "libero_native"}
    expected_source = (
        None
        if name == "off"
        else "RMBench"
        if name.startswith("rmbench")
        else "LIBERO-PRO"
    )
    hpk = provenance.get("hpk")
    if not isinstance(hpk, dict):
        raise LiberoTransferEvaluationError(f"{name} provenance lacks HPK mode")
    if hpk.get("task_knowledge_enabled") is not task_enabled:
        raise LiberoTransferEvaluationError(f"{name} Task Knowledge mode mismatch")
    if hpk.get("action_knowledge_enabled") is not action_enabled:
        raise LiberoTransferEvaluationError(f"{name} Action Knowledge mode mismatch")
    if hpk.get("source_domain") != expected_source:
        raise LiberoTransferEvaluationError(f"{name} source domain mismatch")

    task_adopted = 0
    task_opportunities = 0
    for plan in plans:
        usage = plan.get("hpk_v3_subtask_usage")
        if not task_enabled:
            if usage is not None:
                raise LiberoTransferEvaluationError(
                    "Off trace unexpectedly contains Task Knowledge usage"
                )
            continue
        if not isinstance(usage, dict):
            raise LiberoTransferEvaluationError(
                f"{name} planner decision lacks Task Knowledge audit"
            )
        task_opportunities += 1
        if usage.get("knowledge_adopted") is True:
            task_adopted += 1

    action_adopted = 0
    action_opportunities = 0
    action_by_type: dict[str, dict[str, int]] = {}
    for request in requests:
        planner_index = request.get("planner_call_index")
        plan = plans_by_index.get(planner_index)
        if plan is None:
            raise LiberoTransferEvaluationError(
                f"{name} executor request has no planner decision"
            )
        before = request.get("prompt_before_action_hpk")
        if before != plan.get("subtask_after"):
            raise LiberoTransferEvaluationError(
                f"{name} executor baseline prompt differs from planner subtask"
            )
        usage = request.get("hpk_v3_action_usage")
        if not action_enabled:
            if usage is not None or request.get("prompt") != before:
                raise LiberoTransferEvaluationError(
                    f"{name} unexpectedly changed an executor prompt with Action Knowledge"
                )
            continue
        if not isinstance(usage, dict):
            raise LiberoTransferEvaluationError(
                f"{name} executor request lacks Action Knowledge audit"
            )
        if usage.get("source_domain") != expected_source:
            raise LiberoTransferEvaluationError(
                f"{name} Action Knowledge source domain mismatch"
            )
        if request.get("prompt") != usage.get("prompt_after"):
            raise LiberoTransferEvaluationError(
                f"{name} executor prompt differs from grounded action prompt"
            )
        if usage.get("prompt_before") != before:
            raise LiberoTransferEvaluationError(
                f"{name} Action Knowledge baseline prompt mismatch"
            )
        action = str(usage.get("action") or "unsupported")
        counts = action_by_type.setdefault(action, {"opportunities": 0, "adopted": 0})
        counts["opportunities"] += 1
        action_opportunities += 1
        if usage.get("knowledge_adopted") is True:
            if usage.get("retrieved_knowledge") is None or before == request.get(
                "prompt"
            ):
                raise LiberoTransferEvaluationError(
                    f"{name} adopted Action Knowledge lacks a real prompt change"
                )
            reject_private_transferable(
                usage["retrieved_knowledge"],
                path=f"{name}.retrieved_knowledge",
            )
            reject_private_transferable(
                request["prompt"],
                path=f"{name}.executor_prompt",
            )
            counts["adopted"] += 1
            action_adopted += 1

    loop = provenance.get("simulator", {}).get("roboharn_agent_loop")
    if not isinstance(loop, dict):
        raise LiberoTransferEvaluationError(f"{name} lacks Agent loop result")
    end = ends[0]
    if end.get("benchmark_success") is not loop.get("success") or end.get(
        "step_count"
    ) != loop.get("executed_actions"):
        raise LiberoTransferEvaluationError(f"{name} trace/provenance outcome mismatch")
    return {
        "success": bool(loop["success"]),
        "steps": int(loop["executed_actions"]),
        "recovery_boundaries": int(loop["recovery_boundaries"]),
        "planner_calls": int(loop["planner_calls"]),
        "task_knowledge": {
            "opportunities": task_opportunities,
            "adopted": task_adopted,
        },
        "action_knowledge": {
            "opportunities": action_opportunities,
            "adopted": action_adopted,
            "by_action": action_by_type,
        },
    }


def audit_transfer_experiment(root: str | Path) -> dict[str, Any]:
    """Validate four matched results and return descriptive paired metrics."""

    directory = Path(root).expanduser().resolve(strict=True)
    provenances = {
        name: _json(directory / name / "provenance.json") for name in _CONDITIONS
    }
    traces = {
        name: _jsonl(directory / name / "agent_loop_trace.jsonl")
        for name in _CONDITIONS
    }
    baseline = provenances["off"]
    for name in _CONDITIONS[1:]:
        for field in _MATCHED_RUN_FIELDS:
            if baseline["run"].get(field) != provenances[name]["run"].get(field):
                raise LiberoTransferEvaluationError(
                    f"{name} does not match Off run field {field}"
                )
        if baseline.get("task") != provenances[name].get("task"):
            raise LiberoTransferEvaluationError(f"{name} task identity differs")
        if baseline.get("policy") != provenances[name].get("policy"):
            raise LiberoTransferEvaluationError(f"{name} policy identity differs")
    summaries = {
        name: _condition_summary(
            name,
            provenance=provenances[name],
            trace=traces[name],
        )
        for name in _CONDITIONS
    }
    off = summaries["off"]
    deltas = {
        name: {
            "success_minus_off": int(summary["success"]) - int(off["success"]),
            "steps_minus_off": summary["steps"] - off["steps"],
            "recovery_boundaries_minus_off": (
                summary["recovery_boundaries"] - off["recovery_boundaries"]
            ),
        }
        for name, summary in summaries.items()
        if name != "off"
    }
    return {
        "schema": "roboharn_evo/libero_cross_benchmark_transfer_audit/v1",
        "conditions": summaries,
        "paired_deltas": deltas,
        "claim_scope": (
            "one complete four-condition matched case; repeat across preregistered "
            "init and seed pairs before claiming a performance improvement"
        ),
    }


__all__ = ["LiberoTransferEvaluationError", "audit_transfer_experiment"]
