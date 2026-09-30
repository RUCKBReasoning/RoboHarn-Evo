#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.roboharn_evo.agent.experience import (
    fallback_semantic_tags,
    has_semantic_tags,
    merge_semantic_tags,
    normalize_ood_scenario,
    normalize_semantic_tags,
    stable_experience_id,
)


OOD_SIGNAL_NAMES = {
    "object_not_visible",
    "motion_blocked",
    "grasp_lost",
    "scene_drift_detected",
    "requires_replan",
    "needs_recovery_tools",
    "stall_detected",
    "step_budget_exhausted",
}

POST_RECOVERY_EVENT_WINDOW = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RoboHarn-Evo experience ledgers from ImgAgent trace/events JSONL.")
    parser.add_argument("--trace", type=Path, required=True, help="Agent trace JSONL or rollout events.jsonl.")
    parser.add_argument("--recovery-output", type=Path, default=Path("policy/roboharn_evo/skills/experience/raw-traces/recovery_trials.jsonl"))
    parser.add_argument("--ood-output", type=Path, default=Path("policy/roboharn_evo/skills/experience/raw-traces/ood_trials.jsonl"))
    parser.add_argument("--reentry-output", type=Path, default=Path("policy/roboharn_evo/skills/experience/raw-traces/reentry_trials.jsonl"))
    parser.add_argument("--append", action="store_true", help="Append to output ledgers instead of rewriting them.")
    parser.add_argument("--use-fallback-tags", action="store_true", help="Infer semantic tags from task/subtask text when trace tags are missing.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(payload, dict):
                payload["_line_no"] = line_no
                records.append(payload)
    return records


def append_jsonl(path: Path, records: list[dict[str, Any]], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def sanitize_tag(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def default_semantic_tags() -> dict[str, Any]:
    return normalize_semantic_tags(
        {
            "task_family": "other",
            "subtask_type": "other",
            "state_tags": {},
            "tag_source": "unknown",
        }
    )


def fill_required_task_tags(tags: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_semantic_tags(tags)
    if not str(normalized.get("task_family", "")).strip():
        normalized["task_family"] = "other"
    if not str(normalized.get("subtask_type", "")).strip():
        normalized["subtask_type"] = "other"
    return normalize_semantic_tags(normalized)


def select_semantic_tags(
    candidates: list[Any],
    *,
    global_task: str,
    subtask: str,
    signal_name: str,
    use_fallback_tags: bool,
) -> dict[str, Any]:
    tags = normalize_semantic_tags({})
    preferred_source = ""
    for candidate in candidates:
        candidate_tags = normalize_semantic_tags(candidate)
        if has_semantic_tags(candidate_tags):
            if not preferred_source and str(candidate_tags.get("tag_source", "unknown")) != "unknown":
                preferred_source = str(candidate_tags.get("tag_source", ""))
            tags = merge_semantic_tags(tags, candidate_tags, overwrite=False)
    if not has_semantic_tags(tags):
        if use_fallback_tags:
            return fallback_semantic_tags(global_task=global_task, current_subtask=subtask, signal_name=signal_name)
        return default_semantic_tags()
    if preferred_source:
        tags["tag_source"] = preferred_source
    return fill_required_task_tags(tags)


def retrieval_tags_from_semantics(
    *,
    scenario: str,
    semantic_tags: dict[str, Any],
    extra_tags: list[str] | None = None,
) -> list[str]:
    state_tags = dict(semantic_tags.get("state_tags", {}))
    values = {
        scenario,
        str(semantic_tags.get("task_family", "")),
        str(semantic_tags.get("subtask_type", "")),
        str(state_tags.get("object_state", "")),
        str(state_tags.get("visibility_state", "")),
        str(state_tags.get("gripper_state", "")),
        str(state_tags.get("motion_state", "")),
        *(extra_tags or []),
    }
    return sorted({tag for tag in values if tag})


def build_context(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    global_task = ""
    active_subtask = ""
    semantic_tags: dict[str, Any] = normalize_semantic_tags({})
    pending_semantic_tags: dict[str, Any] = normalize_semantic_tags({})
    contexts: list[dict[str, Any]] = []
    for record in records:
        event = str(record.get("event", ""))
        if event == "instruction_set":
            global_task = str(record.get("instruction", "")).strip()
            active_subtask = ""
            semantic_tags = normalize_semantic_tags({})
            pending_semantic_tags = normalize_semantic_tags({})
        elif event == "control_turn_result":
            candidate_tags = normalize_semantic_tags(record.get("semantic_tags"))
            pending_semantic_tags = candidate_tags if has_semantic_tags(candidate_tags) else normalize_semantic_tags({})
        elif event == "control_decision":
            active_subtask = str(record.get("rendered_instruction") or record.get("subtask_text") or active_subtask).strip()
            candidate_tags = normalize_semantic_tags(record.get("semantic_tags"))
            if has_semantic_tags(candidate_tags):
                semantic_tags = candidate_tags
            elif has_semantic_tags(pending_semantic_tags):
                semantic_tags = pending_semantic_tags
            else:
                semantic_tags = normalize_semantic_tags({})
            pending_semantic_tags = normalize_semantic_tags({})
        elif event == "vla_request":
            active_subtask = str(record.get("subtask") or active_subtask).strip()
        enriched = dict(record)
        enriched["_global_task"] = global_task
        enriched["_active_subtask"] = active_subtask
        enriched["_semantic_tags"] = semantic_tags
        contexts.append(enriched)
    return contexts


def find_previous(records: list[dict[str, Any]], start_index: int, event_name: str) -> dict[str, Any] | None:
    for index in range(start_index - 1, -1, -1):
        if records[index].get("event") == event_name:
            return records[index]
    return None


def find_next(records: list[dict[str, Any]], start_index: int, event_names: set[str]) -> dict[str, Any] | None:
    for index in range(start_index + 1, len(records)):
        if str(records[index].get("event", "")) in event_names:
            return records[index]
    return None


def find_next_with_index(records: list[dict[str, Any]], start_index: int, event_names: set[str]) -> tuple[int, dict[str, Any]] | tuple[None, None]:
    for index in range(start_index + 1, len(records)):
        if str(records[index].get("event", "")) in event_names:
            return index, records[index]
    return None, None


def events_after(records: list[dict[str, Any]], start_index: int, *, limit: int = POST_RECOVERY_EVENT_WINDOW) -> list[dict[str, Any]]:
    return records[start_index + 1 : start_index + 1 + max(0, limit)]


def select_robot_state(candidates: list[Any]) -> dict[str, Any]:
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def recovery_outcome(route: dict[str, Any], result: dict[str, Any] | None, policy: dict[str, Any] | None) -> str:
    if result is None:
        return "incomplete_trace"
    tool_results = result.get("results", [])
    if isinstance(tool_results, list) and tool_results and not all(bool(item.get("success")) for item in tool_results if isinstance(item, dict)):
        return "recovery_failed"
    intent = str(route.get("post_recovery_intent") or result.get("post_recovery_intent") or "").strip()
    policy_action = str((policy or {}).get("action", "")).strip()
    final_action = policy_action or intent
    if final_action == "retry":
        return "recovery_success_retry"
    if final_action == "replan":
        return "recovery_success_replan"
    if final_action == "abort":
        return "aborted"
    if not tool_results and intent == "replan":
        return "replan_without_tools"
    return "detect_only"


def infer_post_recovery_attribution(
    records: list[dict[str, Any]],
    *,
    router_index: int,
    result_index: int | None,
    policy: dict[str, Any] | None,
    route: dict[str, Any],
) -> dict[str, Any]:
    start_index = router_index if result_index is None else result_index
    window = events_after(records, start_index)
    policy_action = str((policy or {}).get("action", "")).strip()
    intended_action = str(route.get("post_recovery_intent", "")).strip()
    effective_action = policy_action or intended_action

    secondary_signal = ""
    retry_reentered_vla = False
    success_seen = False
    task_finished = False
    final_episode_status = "unknown"
    evidence_events: list[dict[str, Any]] = []

    for event in window:
        event_name = str(event.get("event", ""))
        compact = {"event": event_name, "line": event.get("_line_no"), "env_step": event.get("env_step", -1)}
        if event_name == "monitor_signal":
            signal = str(event.get("signal", "")).strip()
            compact["signal"] = signal
            compact["reason"] = str(event.get("reason", ""))
            if signal in OOD_SIGNAL_NAMES and not secondary_signal:
                secondary_signal = signal
            if signal in {"task_success", "subtask_success"}:
                success_seen = True
        elif event_name == "recovery_policy":
            compact["action"] = str(event.get("action", ""))
        elif event_name == "control_decision":
            compact["action_mode"] = str(event.get("action_mode", ""))
            if str(event.get("action_mode", "")) == "finish":
                success_seen = True
                task_finished = True
        elif event_name == "vla_request":
            retry_reentered_vla = True
            compact["subtask"] = str(event.get("subtask", ""))
        elif event_name == "run_step":
            compact["monitor_status"] = str(event.get("monitor_status", ""))
            compact["recovery_pending"] = str(event.get("recovery_pending", ""))
        if event_name in {"monitor_signal", "recovery_policy", "control_decision", "vla_request", "run_step"}:
            evidence_events.append(compact)

    if success_seen:
        post_window_outcome = "success"
        final_episode_status = "success" if task_finished else "progressed"
    elif secondary_signal:
        post_window_outcome = "secondary_signal"
        final_episode_status = "unstable"
    elif effective_action == "retry" and retry_reentered_vla:
        post_window_outcome = "retry_reentered_vla"
        final_episode_status = "in_progress"
    elif effective_action == "replan":
        post_window_outcome = "replan_requested"
        final_episode_status = "replan"
    elif effective_action == "abort":
        post_window_outcome = "abort_requested"
        final_episode_status = "aborted"
    else:
        post_window_outcome = "unknown"

    return {
        "post_window_outcome": post_window_outcome,
        "secondary_signal": secondary_signal,
        "retry_success_after_recovery": bool(success_seen and effective_action == "retry"),
        "final_episode_status": final_episode_status,
        "outcome_evidence": {
            "window_size": len(window),
            "policy_action": policy_action,
            "intended_action": intended_action,
            "effective_action": effective_action,
            "retry_reentered_vla": retry_reentered_vla,
            "success_seen": success_seen,
            "events": evidence_events[:12],
        },
    }


def extract_recovery_trials(records: list[dict[str, Any]], trace_path: Path, *, use_fallback_tags: bool = False) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if record.get("event") != "recovery_router" or record.get("route_found") is False:
            continue
        monitor = find_previous(records, index, "monitor_signal") or {}
        result_index, result = find_next_with_index(records, index, {"recovery_result"})
        policy = find_next(records, index, {"recovery_policy", "control_decision", "vla_request"})
        scenario = normalize_ood_scenario(record.get("ood_scenario") or monitor.get("signal") or record.get("signal_name"))
        global_task = str(record.get("_global_task", "")).strip()
        subtask = str(record.get("_active_subtask", "")).strip()
        semantic_tags = select_semantic_tags(
            [record.get("semantic_tags"), monitor.get("semantic_tags"), record.get("_semantic_tags"), monitor.get("_semantic_tags")],
            global_task=global_task,
            subtask=subtask,
            signal_name=str(record.get("signal_name") or monitor.get("signal") or scenario),
            use_fallback_tags=use_fallback_tags,
        )
        robot_state = select_robot_state([record.get("robot_state"), monitor.get("robot_state")])
        tool_names = list(record.get("tools", []) or [])
        tool_results = list((result or {}).get("results", []) or [])
        outcome = recovery_outcome(record, result, policy)
        attribution = infer_post_recovery_attribution(
            records,
            router_index=index,
            result_index=result_index,
            policy=policy,
            route=record,
        )
        base = {
            "trace_file": str(trace_path),
            "router_line": record.get("_line_no"),
            "episode_id": record.get("episode_id", -1),
            "seed": record.get("seed", -1),
            "step": record.get("env_step", -1),
            "workflow": record.get("workflow", ""),
            "scenario": scenario,
        }
        trial_id = stable_experience_id("recovery_trial", base)
        trials.append(
            {
                "trial_id": trial_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "episode_id": record.get("episode_id", -1),
                "seed": record.get("seed", -1),
                "task_family": semantic_tags["task_family"],
                "global_task": global_task,
                "subtask": subtask,
                "subtask_type": semantic_tags["subtask_type"],
                "semantic_tags": semantic_tags,
                "step": record.get("env_step", -1),
                "monitor_signal": monitor.get("signal", record.get("signal_name", "")),
                "OOD_scenario": scenario,
                "observation_summary": str(record.get("observation_summary") or monitor.get("observation_summary") or ""),
                "robot_state": robot_state,
                "available_tools": list(record.get("available_tools", []) or []),
                "recovery_workflow": str(record.get("workflow", "")),
                "tool_calls": [{"tool_name": name, "args": {}} for name in tool_names],
                "tool_results": tool_results,
                "post_recovery_intent": str(record.get("post_recovery_intent", "")),
                "post_recovery_decision": str((policy or {}).get("action", "")),
                "outcome": outcome,
                **attribution,
                "budget_used": {},
                "failure_reason": str(record.get("reason") or monitor.get("reason") or ""),
                "trace_file": str(trace_path),
                "event_window": {
                    "monitor_line": monitor.get("_line_no"),
                    "router_line": record.get("_line_no"),
                    "result_line": None if result is None else result.get("_line_no"),
                    "post_line": None if policy is None else policy.get("_line_no"),
                },
                "retrieval_tags": retrieval_tags_from_semantics(
                    scenario=scenario,
                    semantic_tags=semantic_tags,
                    extra_tags=[
                        "retry" if str(record.get("post_recovery_intent", "")) == "retry" else "",
                        "replan" if str(record.get("post_recovery_intent", "")) == "replan" else "",
                        *[sanitize_tag(name) for name in tool_names],
                    ],
                ),
                "notes": str(record.get("reason", "")),
            }
        )
    return trials


def extract_ood_trials(records: list[dict[str, Any]], trace_path: Path, *, use_fallback_tags: bool = False) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for record in records:
        if record.get("event") != "monitor_signal":
            continue
        signal = str(record.get("signal", "")).strip()
        if signal not in OOD_SIGNAL_NAMES and not record.get("handoff_target"):
            continue
        scenario = normalize_ood_scenario(signal)
        global_task = str(record.get("_global_task", "")).strip()
        subtask = str(record.get("_active_subtask", "")).strip()
        semantic_tags = select_semantic_tags(
            [record.get("semantic_tags"), record.get("_semantic_tags")],
            global_task=global_task,
            subtask=subtask,
            signal_name=signal,
            use_fallback_tags=use_fallback_tags,
        )
        robot_state = select_robot_state([record.get("robot_state")])
        base = {
            "trace_file": str(trace_path),
            "line": record.get("_line_no"),
            "episode_id": record.get("episode_id", -1),
            "signal": signal,
        }
        trials.append(
            {
                "trial_id": stable_experience_id("ood_trial", base),
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "episode_id": record.get("episode_id", -1),
                "seed": record.get("seed", -1),
                "task_family": semantic_tags["task_family"],
                "global_task": global_task,
                "subtask": subtask,
                "subtask_type": semantic_tags["subtask_type"],
                "semantic_tags": semantic_tags,
                "step": record.get("env_step", -1),
                "monitor_signal": signal,
                "OOD_scenario": scenario,
                "observation_summary": str(record.get("observation_summary", "")),
                "robot_state": robot_state,
                "confidence": record.get("progress_score", 0.0),
                "reason": str(record.get("reason", "")),
                "handoff_target": str(record.get("handoff_target", "")),
                "trace_file": str(trace_path),
                "event_window": {"monitor_line": record.get("_line_no")},
                "outcome": "detect_only",
            }
        )
    return trials


def extract_reentry_trials(records: list[dict[str, Any]], trace_path: Path, *, use_fallback_tags: bool = False) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if record.get("event") != "recovery_policy":
            continue
        next_event = find_next(records, index, {"control_decision", "vla_request", "run_step"})
        global_task = str(record.get("_global_task", "")).strip()
        subtask = str(record.get("_active_subtask", "")).strip()
        action = str(record.get("action", "")).strip()
        semantic_tags = select_semantic_tags(
            [record.get("semantic_tags"), record.get("_semantic_tags")],
            global_task=global_task,
            subtask=subtask,
            signal_name=action,
            use_fallback_tags=use_fallback_tags,
        )
        robot_state = select_robot_state([record.get("robot_state"), (next_event or {}).get("robot_state")])
        base = {
            "trace_file": str(trace_path),
            "line": record.get("_line_no"),
            "episode_id": record.get("episode_id", -1),
            "action": action,
        }
        trials.append(
            {
                "trial_id": stable_experience_id("reentry_trial", base),
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "episode_id": record.get("episode_id", -1),
                "seed": record.get("seed", -1),
                "task_family": semantic_tags["task_family"],
                "global_task": global_task,
                "subtask": subtask,
                "subtask_type": semantic_tags["subtask_type"],
                "semantic_tags": semantic_tags,
                "step": record.get("env_step", -1),
                "post_recovery_decision": action,
                "robot_state": robot_state,
                "reason": str(record.get("reason", "")),
                "next_event": "" if next_event is None else str(next_event.get("event", "")),
                "outcome": "incomplete_trace" if next_event is None else "detect_only",
                "trace_file": str(trace_path),
                "event_window": {
                    "policy_line": record.get("_line_no"),
                    "next_line": None if next_event is None else next_event.get("_line_no"),
                },
            }
        )
    return trials


def main() -> None:
    args = parse_args()
    trace_path = args.trace.resolve()
    records = build_context(read_jsonl(trace_path))
    recovery_trials = extract_recovery_trials(records, trace_path, use_fallback_tags=bool(args.use_fallback_tags))
    ood_trials = extract_ood_trials(records, trace_path, use_fallback_tags=bool(args.use_fallback_tags))
    reentry_trials = extract_reentry_trials(records, trace_path, use_fallback_tags=bool(args.use_fallback_tags))
    append_jsonl(args.recovery_output, recovery_trials, append=args.append)
    append_jsonl(args.ood_output, ood_trials, append=args.append)
    append_jsonl(args.reentry_output, reentry_trials, append=args.append)
    print(
        json.dumps(
            {
                "trace": str(trace_path),
                "recovery_trials": len(recovery_trials),
                "ood_trials": len(ood_trials),
                "reentry_trials": len(reentry_trials),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
