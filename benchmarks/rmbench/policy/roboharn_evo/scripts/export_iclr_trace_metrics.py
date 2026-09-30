#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.roboharn_evo.scripts.audit_iclr_trace_evidence import analyze_trace, discover_trace_files


CSV_COLUMNS = [
    "trace",
    "records",
    "task_name",
    "seed",
    "success",
    "result",
    "failure_reason",
    "total_steps",
    "has_perception_preprocess",
    "has_scene_memory",
    "has_temporal_scene_memory",
    "has_normal_vla_control",
    "has_recovery_control",
    "has_complete_recovery_chain",
    "has_action_effect_verification",
    "has_environment_success_effect_commit",
    "has_action_effect_memory_feedback",
    "has_multi_view_perception_trace",
    "preprocess_events",
    "segmentation_entries",
    "segmentation_success",
    "segmentation_errors",
    "detections",
    "grounding_success_entries",
    "preprocess_latency_sec_total",
    "preprocess_latency_sec_avg",
    "perception_cameras",
    "perception_object_ids",
    "scene_memory_events",
    "max_instances_per_update",
    "unique_track_ids",
    "temporal_memory_updates",
    "focus_updates",
    "control_turns",
    "vla_requests",
    "vla_responses",
    "control_latency_sec_total",
    "control_latency_sec_avg",
    "recovery_router_events",
    "recovery_dispatch_events",
    "recovery_result_events",
    "complete_recovery_chains",
    "tool_result_successes",
    "tool_result_failures",
    "tool_result_terminal_skips",
    "action_effect_events",
    "environment_success_effect_events",
    "action_effect_false",
    "action_effect_true",
    "action_effect_unverified",
    "action_effect_avg_confidence",
    "action_effect_memory_feedback_mentions",
    "warnings",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ICLR-ready per-trace metrics from RoboHarn-Evo trace JSONL files.")
    parser.add_argument("--input", type=Path, action="append", default=[], help="Trace JSONL or run directory. Can be repeated.")
    parser.add_argument("--trace", type=Path, action="append", default=[], help="Alias for --input.")
    parser.add_argument("--output-dir", type=Path, default=Path("research_iclr/results"))
    parser.add_argument("--name", default="trace_metrics")
    return parser.parse_args()


def join_list(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return ";".join(str(value) for value in values)


def row_from_report(report: dict[str, Any]) -> dict[str, Any]:
    episode = dict(report.get("episode", {}) or {})
    flags = dict(report.get("evidence_flags", {}) or {})
    perception = dict(report.get("perception", {}) or {})
    scene_memory = dict(report.get("scene_memory", {}) or {})
    control = dict(report.get("control", {}) or {})
    recovery = dict(report.get("recovery", {}) or {})
    action_effect_counts = dict(recovery.get("action_effect_counts", {}) or {})
    return {
        "trace": report.get("trace", ""),
        "records": report.get("records", 0),
        "task_name": episode.get("task_name", ""),
        "seed": episode.get("seed", ""),
        "success": episode.get("success", ""),
        "result": episode.get("result", ""),
        "failure_reason": episode.get("failure_reason", ""),
        "total_steps": episode.get("total_steps", ""),
        "has_perception_preprocess": bool(flags.get("has_perception_preprocess", False)),
        "has_scene_memory": bool(flags.get("has_scene_memory", False)),
        "has_temporal_scene_memory": bool(flags.get("has_temporal_scene_memory", False)),
        "has_normal_vla_control": bool(flags.get("has_normal_vla_control", False)),
        "has_recovery_control": bool(flags.get("has_recovery_control", False)),
        "has_complete_recovery_chain": bool(flags.get("has_complete_recovery_chain", False)),
        "has_action_effect_verification": bool(flags.get("has_action_effect_verification", False)),
        "has_environment_success_effect_commit": bool(
            flags.get("has_environment_success_effect_commit", False)
        ),
        "has_action_effect_memory_feedback": bool(flags.get("has_action_effect_memory_feedback", False)),
        "has_multi_view_perception_trace": bool(flags.get("has_multi_view_perception_trace", False)),
        "preprocess_events": perception.get("events", 0),
        "segmentation_entries": perception.get("segmentation_entries", 0),
        "segmentation_success": perception.get("segmentation_success", 0),
        "segmentation_errors": perception.get("segmentation_errors", 0),
        "detections": perception.get("detections", 0),
        "grounding_success_entries": perception.get("grounding_success_entries", 0),
        "preprocess_latency_sec_total": perception.get("latency_sec_total", 0.0),
        "preprocess_latency_sec_avg": perception.get("latency_sec_avg", 0.0),
        "perception_cameras": join_list(perception.get("cameras", [])),
        "perception_object_ids": join_list(perception.get("object_ids", [])),
        "scene_memory_events": scene_memory.get("events", 0),
        "max_instances_per_update": scene_memory.get("max_instances_per_update", 0),
        "unique_track_ids": scene_memory.get("unique_track_ids", 0),
        "temporal_memory_updates": scene_memory.get("temporal_memory_updates", 0),
        "focus_updates": scene_memory.get("focus_updates", 0),
        "control_turns": control.get("control_turns", 0),
        "vla_requests": control.get("vla_requests", 0),
        "vla_responses": control.get("vla_responses", 0),
        "control_latency_sec_total": control.get("control_latency_sec_total", 0.0),
        "control_latency_sec_avg": control.get("control_latency_sec_avg", 0.0),
        "recovery_router_events": recovery.get("router_events", 0),
        "recovery_dispatch_events": recovery.get("dispatch_events", 0),
        "recovery_result_events": recovery.get("result_events", 0),
        "complete_recovery_chains": recovery.get("complete_chains", 0),
        "tool_result_successes": recovery.get("tool_result_successes", 0),
        "tool_result_failures": recovery.get("tool_result_failures", 0),
        "tool_result_terminal_skips": recovery.get("tool_result_terminal_skips", 0),
        "action_effect_events": recovery.get("action_effect_events", 0),
        "environment_success_effect_events": recovery.get("environment_success_effect_events", 0),
        "action_effect_false": action_effect_counts.get("false", 0),
        "action_effect_true": action_effect_counts.get("true", 0),
        "action_effect_unverified": action_effect_counts.get("unverified", 0),
        "action_effect_avg_confidence": recovery.get("action_effect_avg_confidence", 0.0),
        "action_effect_memory_feedback_mentions": recovery.get("action_effect_memory_feedback_mentions", 0),
        "warnings": join_list(report.get("warnings", [])),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_columns = [
        "records",
        "preprocess_events",
        "segmentation_entries",
        "segmentation_success",
        "segmentation_errors",
        "detections",
        "grounding_success_entries",
        "scene_memory_events",
        "max_instances_per_update",
        "unique_track_ids",
        "temporal_memory_updates",
        "focus_updates",
        "control_turns",
        "vla_requests",
        "vla_responses",
        "recovery_router_events",
        "recovery_dispatch_events",
        "recovery_result_events",
        "complete_recovery_chains",
        "tool_result_successes",
        "tool_result_failures",
        "tool_result_terminal_skips",
        "action_effect_events",
        "environment_success_effect_events",
        "action_effect_false",
        "action_effect_true",
        "action_effect_unverified",
        "action_effect_memory_feedback_mentions",
    ]
    flag_columns = [column for column in CSV_COLUMNS if column.startswith("has_")]
    return {
        "trace_count": len(rows),
        "success_count": sum(1 for row in rows if str(row.get("success", "")).lower() == "true"),
        "numeric_sums": {
            column: sum(safe_number(row.get(column, 0)) for row in rows)
            for column in numeric_columns
        },
        "flag_counts": {
            column: sum(1 for row in rows if bool(row.get(column, False)))
            for column in flag_columns
        },
        "warnings": sorted(
            {
                warning
                for row in rows
                for warning in str(row.get("warnings", "")).split(";")
                if warning
            }
        ),
    }


def safe_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def main() -> None:
    args = parse_args()
    inputs = [*args.input, *args.trace]
    if not inputs:
        raise SystemExit("Pass at least one --input/--trace path.")
    trace_files = discover_trace_files(inputs)
    if not trace_files:
        raise SystemExit("No trace JSONL files found.")
    reports = [analyze_trace(path) for path in trace_files]
    rows = [row_from_report(report) for report in reports]
    summary = {
        "inputs": [str(path) for path in trace_files],
        "rows": rows,
        "aggregate": aggregate_rows(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.name}.csv"
    json_path = args.output_dir / f"{args.name}.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "aggregate": summary["aggregate"]}, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
