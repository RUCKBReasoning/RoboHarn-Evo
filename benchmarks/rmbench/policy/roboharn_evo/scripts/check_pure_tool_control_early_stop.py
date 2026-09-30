#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BAD_CONTROL_REASONS = {
    "episode_agent_finished",
    "episode_end_failed_with_task_finished",
    "episode_exception",
    "episode_interrupted",
    "incomplete_trace",
    "malformed_jsonl",
    "pure_tool_control_abort_finished",
    "pure_tool_control_replan_finished",
    "pure_tool_control_finish_without_eval_success",
    "pure_tool_control_check_success_finished",
    "pure_tool_control_subtask_success_finished",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check pure tool-control traces for early episode termination regressions.")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Trace JSONL, eval run directory, ckpt directory, or eval_result subtree. Can be passed multiple times.",
    )
    parser.add_argument("--trace", type=Path, action="append", default=[], help="Alias for --input.")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when no trace files are found.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Do not fail traces that have episode_start but no terminal episode_end event.",
    )
    return parser.parse_args()


def discover_traces(inputs: list[Path]) -> list[Path]:
    traces: list[Path] = []
    for raw_path in inputs:
        path = raw_path.expanduser()
        if path.is_file():
            traces.append(path)
            continue
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {raw_path}")
        traces.extend(sorted(path.rglob("episode_*_agent_trace.jsonl")))
        traces.extend(sorted(path.rglob("*agent_trace.jsonl")))
    return dedupe(traces)


def dedupe(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def read_jsonl_with_diagnostics(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    records: list[dict[str, Any]] = []
    malformed_line_numbers: list[int] = []
    with path.open("rb") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8").strip()
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                malformed_line_numbers.append(line_no)
                continue
            if not isinstance(record, dict):
                malformed_line_numbers.append(line_no)
                continue
            record["_line_no"] = line_no
            records.append(record)
    return records, malformed_line_numbers


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records, _ = read_jsonl_with_diagnostics(path)
    return records


def bool_field(record: dict[str, Any], key: str) -> bool:
    return bool(record.get(key, False))


def compact_event(record: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "_line_no",
        "event",
        "timestamp",
        "step",
        "total_steps",
        "reason",
        "failure_reason",
        "interrupt_signal",
        "action_mode",
        "preferred_action",
        "rejected",
        "eval_success",
        "check_success",
        "success",
        "task_finished",
        "monitor_phase",
        "monitor_status",
    )
    return {key: record.get(key) for key in keep if key in record}


def analyze_trace(path: Path, *, allow_incomplete: bool = False) -> dict[str, Any]:
    records, malformed_line_numbers = read_jsonl_with_diagnostics(path)
    issues: list[dict[str, Any]] = []
    if malformed_line_numbers:
        issues.append(
            {
                "type": "malformed_jsonl",
                "count": len(malformed_line_numbers),
                "line_numbers": malformed_line_numbers[:20],
            }
        )
    last_preferred_action = ""
    last_control_decision: dict[str, Any] = {}
    last_monitor_signal = ""
    saw_episode_start = False
    saw_episode_end = False

    for record in records:
        event = str(record.get("event", ""))
        if event == "episode_start":
            saw_episode_start = True
        elif event == "debug_recovery_pure_control_round":
            last_preferred_action = str(record.get("preferred_action", "") or "")
        elif event == "control_decision":
            last_control_decision = record
            if (
                str(record.get("action_mode", "")) == "finish"
                and not bool_field(record, "rejected")
                and not bool_field(record, "eval_success")
            ):
                issues.append(
                    {
                        "type": "pure_tool_control_finish_without_eval_success",
                        "event": compact_event(record),
                    }
                )
        elif event == "monitor_signal":
            last_monitor_signal = str(record.get("signal", "") or "")
        elif event == "episode_agent_finished" and bool_field(record, "task_finished"):
            if last_preferred_action == "abort":
                issue_type = "pure_tool_control_abort_finished"
            elif last_preferred_action == "replan":
                issue_type = "pure_tool_control_replan_finished"
            elif str(last_control_decision.get("action_mode", "")) == "finish" and not bool_field(last_control_decision, "eval_success"):
                issue_type = "pure_tool_control_finish_without_eval_success"
            elif last_monitor_signal == "subtask_success":
                issue_type = "pure_tool_control_subtask_success_finished"
            else:
                issue_type = "episode_agent_finished"
            issues.append(
                {
                    "type": issue_type,
                    "event": compact_event(record),
                    "last_preferred_action": last_preferred_action,
                    "last_control_decision": compact_event(last_control_decision) if last_control_decision else {},
                    "last_monitor_signal": last_monitor_signal,
                }
            )
        elif event == "episode_end":
            saw_episode_end = True
            if bool_field(record, "task_finished") and not bool_field(record, "success"):
                issue_type = "episode_end_failed_with_task_finished"
                if last_preferred_action == "abort":
                    issue_type = "pure_tool_control_abort_finished"
                elif last_preferred_action == "replan":
                    issue_type = "pure_tool_control_replan_finished"
                elif last_monitor_signal == "subtask_success":
                    issue_type = "pure_tool_control_subtask_success_finished"
                issues.append(
                    {
                        "type": issue_type,
                        "event": compact_event(record),
                        "last_preferred_action": last_preferred_action,
                        "last_control_decision": compact_event(last_control_decision) if last_control_decision else {},
                        "last_monitor_signal": last_monitor_signal,
                    }
                )
        elif event == "episode_exception":
            issues.append(
                {
                    "type": "episode_exception",
                    "event": compact_event(record),
                    "error": str(record.get("error", "") or ""),
                }
            )
        elif event == "episode_interrupt":
            issues.append(
                {
                    "type": "episode_interrupted",
                    "event": compact_event(record),
                    "last_preferred_action": last_preferred_action,
                    "last_control_decision": compact_event(last_control_decision) if last_control_decision else {},
                    "last_monitor_signal": last_monitor_signal,
                }
            )

    if saw_episode_start and not saw_episode_end:
        issues.append(
            {
                "type": "incomplete_trace",
                "event": compact_event(records[-1]) if records else {},
                "last_preferred_action": last_preferred_action,
                "last_control_decision": compact_event(last_control_decision) if last_control_decision else {},
                "last_monitor_signal": last_monitor_signal,
            }
        )

    bad_reasons = BAD_CONTROL_REASONS - ({"incomplete_trace"} if allow_incomplete else set())
    return {
        "trace": str(path),
        "records": len(records),
        "malformed_jsonl_line_count": len(malformed_line_numbers),
        "malformed_jsonl_line_numbers": malformed_line_numbers[:20],
        "issues": issues,
        "bad_issues": [issue for issue in issues if str(issue.get("type", "")) in bad_reasons],
    }


def main() -> None:
    args = parse_args()
    inputs = [*args.input, *args.trace]
    if not inputs:
        raise SystemExit("Pass at least one --input/--trace path.")
    traces = discover_traces(inputs)
    if not traces and args.strict:
        raise SystemExit("No trace files found.")
    reports = [analyze_trace(path, allow_incomplete=bool(args.allow_incomplete)) for path in traces]
    bad_reports = [report for report in reports if report["bad_issues"]]
    summary = {
        "trace_count": len(reports),
        "bad_trace_count": len(bad_reports),
        "issue_count": sum(len(report["issues"]) for report in reports),
        "bad_issue_count": sum(len(report["bad_issues"]) for report in reports),
        "reports": reports,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for report in reports:
        bad_types = {str(issue.get("type", "")) for issue in report["bad_issues"]}
        status = "INCOMPLETE" if bad_types == {"incomplete_trace"} else ("FAIL" if report["bad_issues"] else "PASS")
        print(f"{status} {report['trace']} issues={len(report['issues'])} bad={len(report['bad_issues'])}")
        for issue in report["bad_issues"]:
            print("  " + json.dumps(issue, ensure_ascii=False, sort_keys=True))
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, ensure_ascii=False, sort_keys=True))
    raise SystemExit(1 if bad_reports else 0)


if __name__ == "__main__":
    main()
