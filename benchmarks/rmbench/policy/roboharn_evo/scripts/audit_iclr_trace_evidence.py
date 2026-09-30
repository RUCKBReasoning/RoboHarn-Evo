#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PERCEPTION_EVENTS = {
    "observation_preprocess",
    "observation_preprocess_finalized",
    "scene_memory_update",
}

NORMAL_CONTROL_EVENTS = {
    "control_turn_start",
    "control_turn_result",
    "control_decision",
    "vla_request",
    "vla_response",
}

RECOVERY_EVENTS = {
    "recovery_router",
    "recovery_dispatch",
    "recovery_result",
    "action_effect_verification",
    "environment_success_effect_commit",
}

LEVEL0_RECOVERY_CHAIN = [
    "observation_preprocess",
    "scene_memory_update",
    "recovery_router",
    "recovery_dispatch",
    "action_effect_verification",
    "recovery_result",
]

LEVEL0_TERMINAL_RECOVERY_CHAIN = [
    "observation_preprocess",
    "scene_memory_update",
    "recovery_router",
    "recovery_dispatch",
    "environment_success_effect_commit",
    "recovery_result",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit RoboHarn-Evo traces for ICLR claim evidence."
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Trace JSONL, rollout events JSONL, eval run directory, or rollout directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        action="append",
        default=[],
        help="Alias for --input kept for compatibility with other RoboHarn-Evo scripts.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("research_iclr/results"))
    parser.add_argument("--name", default="trace_evidence_summary")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if a recovery trace is incomplete.")
    return parser.parse_args()


def discover_trace_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            files.append(path)
            continue
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {raw_path}")
        run_dir = path.parent if path.name.startswith("episode_") and path.name.endswith("_rollout") else path
        files.extend(sorted(run_dir.glob("episode_*_agent_trace.jsonl")))
        files.extend(sorted(run_dir.glob("*agent_trace.jsonl")))
        if path.name.startswith("episode_") and path.name.endswith("_rollout"):
            files.extend(sorted(path.glob("events.jsonl")))
        else:
            files.extend(sorted(run_dir.glob("episode_*_rollout/events.jsonl")))
            files.extend(sorted(run_dir.glob("events.jsonl")))
    return dedupe_paths(files)


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
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


def event_order(records: list[dict[str, Any]], event_names: list[str]) -> dict[str, Any]:
    positions: dict[str, int] = {}
    for index, record in enumerate(records):
        event = str(record.get("event", ""))
        if event in event_names and event not in positions:
            positions[event] = index
    present = [event for event in event_names if event in positions]
    missing = [event for event in event_names if event not in positions]
    in_order = all(positions[present[index]] <= positions[present[index + 1]] for index in range(len(present) - 1))
    return {
        "expected": event_names,
        "present": present,
        "missing": missing,
        "in_order_for_present": in_order,
        "complete": not missing and in_order,
    }


def nested_values(payload: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(payload, dict):
        for item_key, item_value in payload.items():
            if item_key == key:
                values.append(item_value)
            values.extend(nested_values(item_value, key))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(nested_values(item, key))
    return values


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def analyze_observation_preprocess(records: list[dict[str, Any]]) -> dict[str, Any]:
    entries = [record for record in records if record.get("event") == "observation_preprocess"]
    segmentation_count = 0
    detection_count = 0
    success_count = 0
    error_count = 0
    grounding_success_count = 0
    cameras: set[str] = set()
    object_ids: set[str] = set()
    errors: Counter[str] = Counter()
    latencies: list[float] = []

    for record in entries:
        latency = safe_float(record.get("latency_sec"))
        if latency is not None:
            latencies.append(latency)
        for item in record.get("segmentation", []) or []:
            if not isinstance(item, dict):
                continue
            segmentation_count += 1
            if item.get("success") is True:
                success_count += 1
            else:
                error_count += 1
                error = str(item.get("error", "unknown_error")).strip() or "unknown_error"
                errors[error[:160]] += 1
            camera = str(item.get("camera", "")).strip()
            if camera:
                cameras.add(camera)
            object_id = str(item.get("object_id", "")).strip()
            if object_id:
                object_ids.add(object_id)
            detections = item.get("detections", [])
            if isinstance(detections, list):
                detection_count += len([det for det in detections if isinstance(det, dict)])
                for det in detections:
                    if isinstance(det, dict):
                        det_camera = str(det.get("camera", "")).strip()
                        if det_camera:
                            cameras.add(det_camera)
            grounding_values = nested_values(item, "grounding_3d")
            if any(isinstance(value, dict) and value.get("success") is True for value in grounding_values):
                grounding_success_count += 1

    return {
        "events": len(entries),
        "segmentation_entries": segmentation_count,
        "segmentation_success": success_count,
        "segmentation_errors": error_count,
        "detections": detection_count,
        "grounding_success_entries": grounding_success_count,
        "cameras": sorted(cameras),
        "object_ids": sorted(object_ids),
        "latency_sec_total": round(sum(latencies), 4),
        "latency_sec_avg": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        "top_errors": [{"error": key, "count": value} for key, value in errors.most_common(5)],
    }


def analyze_scene_memory(records: list[dict[str, Any]]) -> dict[str, Any]:
    entries = [record for record in records if record.get("event") == "scene_memory_update"]
    max_instances = 0
    stable_instances = 0
    tracked_instances = 0
    temporal_memory_updates = 0
    focus_updates = 0
    track_ids: set[str] = set()
    cameras: set[str] = set()
    last_focus: dict[str, Any] = {}
    last_summary = ""

    for record in entries:
        scene = record.get("scene_memory", {})
        if not isinstance(scene, dict):
            continue
        instances = [item for item in scene.get("instances", []) or [] if isinstance(item, dict)]
        max_instances = max(max_instances, len(instances))
        for instance in instances:
            if instance.get("stability") == "stable":
                stable_instances += 1
            if instance.get("status") == "tracked":
                tracked_instances += 1
            track_id = str(instance.get("track_id", "")).strip()
            if track_id:
                track_ids.add(track_id)
            camera = str(instance.get("camera", "")).strip()
            if camera:
                cameras.add(camera)
        focus = scene.get("task_focus")
        if isinstance(focus, dict):
            if focus.get("target_instances") or focus.get("tool_instances"):
                focus_updates += 1
            last_focus = {
                "target_instances": list(focus.get("target_instances", []) or []),
                "tool_instances": list(focus.get("tool_instances", []) or []),
                "source": str(focus.get("source", "")),
            }
        temporal = scene.get("temporal_memory")
        if isinstance(temporal, dict) and temporal:
            temporal_memory_updates += 1
        last_summary = str(scene.get("summary", last_summary))

    return {
        "events": len(entries),
        "max_instances_per_update": max_instances,
        "stable_instance_observations": stable_instances,
        "tracked_instance_observations": tracked_instances,
        "unique_track_ids": len(track_ids),
        "temporal_memory_updates": temporal_memory_updates,
        "focus_updates": focus_updates,
        "cameras": sorted(cameras),
        "last_focus": last_focus,
        "last_summary": last_summary[:360],
    }


def analyze_recovery(records: list[dict[str, Any]]) -> dict[str, Any]:
    router_records = [record for record in records if record.get("event") == "recovery_router"]
    result_records = [record for record in records if record.get("event") == "recovery_result"]
    dispatch_records = [record for record in records if record.get("event") == "recovery_dispatch"]
    effect_records = [record for record in records if record.get("event") == "action_effect_verification"]
    environment_effect_records = [
        record for record in records if record.get("event") == "environment_success_effect_commit"
    ]

    invalid_tool_names: list[dict[str, Any]] = []
    route_missing = 0
    workflows: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    result_success = 0
    result_failure = 0
    result_terminal_skip = 0
    effect_counts: Counter[str] = Counter()
    effect_types: Counter[str] = Counter()
    effect_confidences: list[float] = []

    for record in router_records:
        if record.get("route_found") is False:
            route_missing += 1
        workflow = str(record.get("workflow", "")).strip()
        if workflow:
            workflows[workflow] += 1
        available = {str(item) for item in record.get("available_tools", []) or []}
        for tool in record.get("tools", []) or []:
            tool_name = str(tool).strip()
            if not tool_name:
                continue
            tool_names[tool_name] += 1
            if available and tool_name not in available:
                invalid_tool_names.append(
                    {
                        "line": record.get("_line_no"),
                        "tool": tool_name,
                        "available_tools": sorted(available),
                    }
                )

    for record in result_records:
        for item in record.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            if details.get("terminal_skip") is True:
                result_terminal_skip += 1
            elif item.get("success") is True:
                result_success += 1
            else:
                result_failure += 1
            tool_name = str(item.get("tool_name", "")).strip()
            if tool_name:
                tool_names[tool_name] += 1

    for record in effect_records:
        result = record.get("result", {})
        if not isinstance(result, dict):
            result = {}
        effect = str(result.get("effect_verified", "unverified")).strip().lower() or "unverified"
        effect_counts[effect] += 1
        effect_type = str(result.get("effect_type", "unknown")).strip().lower() or "unknown"
        effect_types[effect_type] += 1
        confidence = safe_float(result.get("confidence"))
        if confidence is not None:
            effect_confidences.append(confidence)

    chains = recovery_chains(records)
    memory_feedback_mentions = count_action_effect_memory_feedback(records)

    return {
        "router_events": len(router_records),
        "dispatch_events": len(dispatch_records),
        "result_events": len(result_records),
        "action_effect_events": len(effect_records),
        "environment_success_effect_events": len(environment_effect_records),
        "complete_chains": sum(1 for item in chains if item["complete"]),
        "chains": chains,
        "route_missing": route_missing,
        "invalid_tool_names": invalid_tool_names,
        "tool_result_successes": result_success,
        "tool_result_failures": result_failure,
        "tool_result_terminal_skips": result_terminal_skip,
        "workflows": dict(sorted(workflows.items())),
        "tools": dict(sorted(tool_names.items())),
        "action_effect_counts": dict(sorted(effect_counts.items())),
        "action_effect_types": dict(sorted(effect_types.items())),
        "action_effect_avg_confidence": round(sum(effect_confidences) / len(effect_confidences), 4)
        if effect_confidences
        else 0.0,
        "action_effect_memory_feedback_mentions": memory_feedback_mentions,
    }


def recovery_chains(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    router_indexes = [index for index, record in enumerate(records) if record.get("event") == "recovery_router"]
    for chain_index, router_index in enumerate(router_indexes):
        end_index = router_indexes[chain_index + 1] if chain_index + 1 < len(router_indexes) else len(records)
        window = records[router_index:end_index]
        events = [str(record.get("event", "")) for record in window]
        router = records[router_index]
        has_dispatch = "recovery_dispatch" in events
        has_result = "recovery_result" in events
        has_effect = "action_effect_verification" in events
        has_environment_success_effect = "environment_success_effect_commit" in events
        route_found = router.get("route_found") is not False
        chains.append(
            {
                "router_line": router.get("_line_no"),
                "env_step": router.get("env_step", ""),
                "workflow": str(router.get("workflow", "")),
                "tools": list(router.get("tools", []) or []),
                "route_found": route_found,
                "has_dispatch": has_dispatch,
                "has_result": has_result,
                "has_action_effect_verification": has_effect,
                "has_environment_success_effect_commit": has_environment_success_effect,
                "complete": (
                    route_found
                    and has_dispatch
                    and has_result
                    and (has_effect or has_environment_success_effect)
                ),
            }
        )
    return chains


def count_action_effect_memory_feedback(records: list[dict[str, Any]]) -> int:
    first_effect_index = next(
        (index for index, record in enumerate(records) if record.get("event") == "action_effect_verification"),
        None,
    )
    if first_effect_index is None:
        return 0
    count = 0
    for record in records[first_effect_index + 1 :]:
        if record.get("event") not in {"recovery_router", "recovery_dispatch", "control_turn_result"}:
            continue
        text = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        if "action_effect:" in text:
            count += 1
    return count


def analyze_control(records: list[dict[str, Any]]) -> dict[str, Any]:
    control_latencies: list[float] = []
    preprocess_latencies: list[float] = []
    subtask_changes = 0
    last_subtask = ""
    for record in records:
        if record.get("event") == "control_turn_result":
            latency = safe_float(record.get("latency_sec"))
            if latency is not None:
                control_latencies.append(latency)
            subtask = str(record.get("subtask_text", "")).strip()
            if subtask and subtask != last_subtask:
                subtask_changes += 1
                last_subtask = subtask
        if record.get("event") == "observation_preprocess":
            latency = safe_float(record.get("latency_sec"))
            if latency is not None:
                preprocess_latencies.append(latency)
    return {
        "control_turns": sum(1 for record in records if record.get("event") == "control_turn_result"),
        "vla_requests": sum(1 for record in records if record.get("event") == "vla_request"),
        "vla_responses": sum(1 for record in records if record.get("event") == "vla_response"),
        "subtask_changes": subtask_changes,
        "control_latency_sec_total": round(sum(control_latencies), 4),
        "control_latency_sec_avg": round(sum(control_latencies) / len(control_latencies), 4)
        if control_latencies
        else 0.0,
        "preprocess_latency_sec_total": round(sum(preprocess_latencies), 4),
        "preprocess_latency_sec_avg": round(sum(preprocess_latencies) / len(preprocess_latencies), 4)
        if preprocess_latencies
        else 0.0,
    }


def analyze_trace(path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    counts = Counter(str(record.get("event", "unknown")) for record in records)
    episode_start = next((record for record in records if record.get("event") == "episode_start"), {})
    episode_end = next((record for record in reversed(records) if record.get("event") == "episode_end"), {})
    event_names = set(counts)

    perception = analyze_observation_preprocess(records)
    scene_memory = analyze_scene_memory(records)
    recovery = analyze_recovery(records)
    control = analyze_control(records)
    verifier_level0 = event_order(records, LEVEL0_RECOVERY_CHAIN)
    terminal_level0 = event_order(records, LEVEL0_TERMINAL_RECOVERY_CHAIN)
    level0 = min(
        (verifier_level0, terminal_level0),
        key=lambda item: (not bool(item["complete"]), len(item["missing"])),
    )
    level0 = {
        **level0,
        "accepted_effect_authority": (
            "agent_api_verifier"
            if level0["expected"] == LEVEL0_RECOVERY_CHAIN
            else "environment_eval_success"
        ),
    }
    normal_control_present = bool(event_names & NORMAL_CONTROL_EVENTS)
    recovery_present = bool(event_names & RECOVERY_EVENTS)

    evidence_flags = {
        "has_perception_preprocess": bool(event_names & PERCEPTION_EVENTS),
        "has_scene_memory": counts.get("scene_memory_update", 0) > 0 and scene_memory["max_instances_per_update"] > 0,
        "has_temporal_scene_memory": scene_memory["temporal_memory_updates"] > 0,
        "has_normal_vla_control": normal_control_present,
        "has_recovery_control": recovery_present,
        "has_complete_recovery_chain": recovery["complete_chains"] > 0,
        "has_action_effect_verification": recovery["action_effect_events"] > 0,
        "has_environment_success_effect_commit": recovery["environment_success_effect_events"] > 0,
        "has_action_effect_memory_feedback": recovery["action_effect_memory_feedback_mentions"] > 0,
        "has_multi_view_perception_trace": len(set(perception["cameras"]) | set(scene_memory["cameras"])) >= 2,
    }

    return {
        "trace": str(path),
        "records": len(records),
        "event_counts": dict(sorted(counts.items())),
        "episode": {
            "task_name": episode_start.get("task_name", ""),
            "instruction": str(episode_start.get("instruction", ""))[:500],
            "seed": episode_start.get("seed", episode_end.get("seed", "")),
            "success": episode_end.get("success", ""),
            "result": episode_end.get("result", ""),
            "failure_reason": episode_end.get("failure_reason", ""),
            "total_steps": episode_end.get("total_steps", episode_end.get("step", "")),
        },
        "level0_recovery_chain": level0,
        "perception": perception,
        "scene_memory": scene_memory,
        "control": control,
        "recovery": recovery,
        "evidence_flags": evidence_flags,
        "warnings": trace_warnings(perception=perception, scene_memory=scene_memory, recovery=recovery, level0=level0),
    }


def trace_warnings(
    *,
    perception: dict[str, Any],
    scene_memory: dict[str, Any],
    recovery: dict[str, Any],
    level0: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if perception["events"] and perception["segmentation_success"] == 0:
        warnings.append("perception_preprocess_present_but_no_successful_segmentation")
    if perception["segmentation_errors"]:
        warnings.append("segmentation_errors_present")
    if scene_memory["events"] and scene_memory["max_instances_per_update"] == 0:
        warnings.append("scene_memory_present_but_empty_instances")
    if recovery["router_events"] and recovery["complete_chains"] == 0:
        warnings.append("recovery_router_present_but_no_complete_recovery_chain")
    if recovery["invalid_tool_names"]:
        warnings.append("recovery_router_returned_unavailable_tools")
    if recovery["action_effect_events"] and recovery["action_effect_memory_feedback_mentions"] == 0:
        warnings.append("action_effect_verification_present_but_no_later_memory_feedback_observed")
    if recovery["router_events"] and not level0["complete"]:
        warnings.append("level0_recovery_chain_incomplete")
    return warnings


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    for report in reports:
        event_counts.update(report.get("event_counts", {}))
    flag_counts: Counter[str] = Counter()
    for report in reports:
        for key, value in report.get("evidence_flags", {}).items():
            if value:
                flag_counts[key] += 1
    recovery_traces = [report for report in reports if report["recovery"]["router_events"] > 0]
    return {
        "trace_count": len(reports),
        "recovery_trace_count": len(recovery_traces),
        "event_counts": dict(sorted(event_counts.items())),
        "evidence_flag_counts": dict(sorted(flag_counts.items())),
        "complete_recovery_chain_traces": sum(
            1 for report in reports if report["evidence_flags"]["has_complete_recovery_chain"]
        ),
        "action_effect_traces": sum(
            1 for report in reports if report["evidence_flags"]["has_action_effect_verification"]
        ),
        "multi_view_perception_traces": sum(
            1 for report in reports if report["evidence_flags"]["has_multi_view_perception_trace"]
        ),
        "segmentation_error_traces": sum(1 for report in reports if report["perception"]["segmentation_errors"] > 0),
        "all_warnings": sorted({warning for report in reports for warning in report.get("warnings", [])}),
    }


def claim_evidence_from_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate = aggregate_reports(reports)

    def supported(flag: str) -> bool:
        return int(aggregate["evidence_flag_counts"].get(flag, 0)) > 0

    return [
        {
            "claim_id": "C1",
            "claim": "HAGM has an implemented hierarchical embodied runtime.",
            "trace_evidence": supported("has_normal_vla_control") or supported("has_recovery_control"),
            "status": "trace_supported" if supported("has_normal_vla_control") or supported("has_recovery_control") else "not_observed",
            "caution": "This is architectural evidence, not task-success evidence.",
        },
        {
            "claim_id": "C2",
            "claim": "Observation preprocessing is used during rollout.",
            "trace_evidence": supported("has_perception_preprocess"),
            "status": "trace_supported" if supported("has_perception_preprocess") else "not_observed",
            "caution": "Multi-view support requires traces with at least two observed cameras.",
        },
        {
            "claim_id": "C3",
            "claim": "Scene memory provides stable instance references.",
            "trace_evidence": supported("has_scene_memory"),
            "status": "trace_supported" if supported("has_scene_memory") else "not_observed",
            "caution": "This does not prove scene-memory improves task performance.",
        },
        {
            "claim_id": "C4",
            "claim": "Recovery planner receives memory and produces tool-level recovery actions.",
            "trace_evidence": supported("has_complete_recovery_chain"),
            "status": "trace_supported" if supported("has_complete_recovery_chain") else "not_observed",
            "caution": "This does not prove recovery tools improve success rate.",
        },
        {
            "claim_id": "C5",
            "claim": "Action-effect verification writes useful feedback for subsequent recovery.",
            "trace_evidence": supported("has_action_effect_memory_feedback"),
            "status": "preliminary_trace_supported" if supported("has_action_effect_memory_feedback") else "not_observed",
            "caution": "Requires verifier on/off ablation before causal claims.",
        },
    ]


def build_summary(trace_files: list[Path]) -> dict[str, Any]:
    reports = [analyze_trace(path) for path in trace_files]
    return {
        "inputs": [str(path) for path in trace_files],
        "aggregate": aggregate_reports(reports),
        "claim_evidence": claim_evidence_from_reports(reports),
        "traces": reports,
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    aggregate = summary["aggregate"]
    lines.append("# Trace evidence audit")
    lines.append("")
    lines.append("This report is generated from real RoboHarn-Evo trace JSONL files for implementation and mechanism auditing.")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Traces analyzed: `{aggregate['trace_count']}`")
    lines.append(f"- Recovery traces: `{aggregate['recovery_trace_count']}`")
    lines.append(f"- Traces with complete recovery chain: `{aggregate['complete_recovery_chain_traces']}`")
    lines.append(f"- Traces with action-effect verification: `{aggregate['action_effect_traces']}`")
    lines.append(f"- Traces with observed multi-view perception: `{aggregate['multi_view_perception_traces']}`")
    lines.append(f"- Traces with segmentation errors: `{aggregate['segmentation_error_traces']}`")
    if aggregate["all_warnings"]:
        lines.append(f"- Warnings: `{', '.join(aggregate['all_warnings'])}`")
    lines.append("")
    lines.append("## Claim Evidence")
    lines.append("")
    lines.append("| Claim | Status | Caution |")
    lines.append("| --- | --- | --- |")
    for claim in summary["claim_evidence"]:
        lines.append(
            f"| {claim['claim_id']}: {claim['claim']} | {claim['status']} | {claim['caution']} |"
        )
    lines.append("")
    lines.append("## Per-Trace Summary")
    lines.append("")
    for trace in summary["traces"]:
        lines.append(f"### `{Path(trace['trace']).name}`")
        lines.append("")
        lines.append(f"- Path: `{trace['trace']}`")
        lines.append(f"- Records: `{trace['records']}`")
        lines.append(f"- Episode success: `{trace['episode'].get('success')}`; result: `{trace['episode'].get('result')}`; failure: `{trace['episode'].get('failure_reason')}`")
        lines.append(f"- Level 0 recovery chain complete: `{trace['level0_recovery_chain']['complete']}`")
        lines.append(f"- Missing chain events: `{', '.join(trace['level0_recovery_chain']['missing']) or 'none'}`")
        lines.append(f"- Preprocess: events=`{trace['perception']['events']}`, success=`{trace['perception']['segmentation_success']}`, errors=`{trace['perception']['segmentation_errors']}`, cameras=`{trace['perception']['cameras']}`")
        lines.append(f"- Scene memory: updates=`{trace['scene_memory']['events']}`, max_instances=`{trace['scene_memory']['max_instances_per_update']}`, tracks=`{trace['scene_memory']['unique_track_ids']}`, temporal_updates=`{trace['scene_memory']['temporal_memory_updates']}`")
        lines.append(f"- Recovery: routers=`{trace['recovery']['router_events']}`, complete_chains=`{trace['recovery']['complete_chains']}`, tools=`{trace['recovery']['tools']}`")
        lines.append(f"- Action effect: events=`{trace['recovery']['action_effect_events']}`, counts=`{trace['recovery']['action_effect_counts']}`, memory_feedback_mentions=`{trace['recovery']['action_effect_memory_feedback_mentions']}`")
        if trace["warnings"]:
            lines.append(f"- Warnings: `{', '.join(trace['warnings'])}`")
        lines.append("")
    lines.append("## Event Counts")
    lines.append("")
    lines.append("| Event | Count |")
    lines.append("| --- | ---: |")
    for event, count in aggregate["event_counts"].items():
        lines.append(f"| `{event}` | {count} |")
    lines.append("")
    return "\n".join(lines)


def write_outputs(summary: dict[str, Any], output_dir: Path, name: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}.json"
    md_path = output_dir / f"{name}.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> None:
    args = parse_args()
    inputs = [*args.input, *args.trace]
    if not inputs:
        raise SystemExit("Pass at least one --input/--trace path.")
    trace_files = discover_trace_files(inputs)
    if not trace_files:
        raise SystemExit("No trace JSONL files found.")
    summary = build_summary(trace_files)
    outputs = write_outputs(summary, args.output_dir, args.name)
    print(json.dumps({"outputs": outputs, "aggregate": summary["aggregate"]}, ensure_ascii=False, sort_keys=True, indent=2))
    if args.strict:
        recovery_traces = [trace for trace in summary["traces"] if trace["recovery"]["router_events"] > 0]
        incomplete = [trace["trace"] for trace in recovery_traces if not trace["evidence_flags"]["has_complete_recovery_chain"]]
        if incomplete:
            raise SystemExit(f"Incomplete recovery traces: {incomplete}")


if __name__ == "__main__":
    main()
