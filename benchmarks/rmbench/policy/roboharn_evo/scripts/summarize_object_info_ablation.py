from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.roboharn_evo.scripts.check_pure_tool_control_early_stop import analyze_trace, read_jsonl


RESULT_RE = re.compile(
    r"episode_id=(?P<episode_id>\d+),\s+seed=(?P<seed>\d+),\s+instruction=(?P<instruction>.*?),\s+"
    r"result=(?P<result>[^,]+),\s+steps=(?P<steps>\d+),\s+reward=(?P<reward>[-+0-9.eE]+),\s+"
    r"failure_reason=(?P<failure_reason>.*)$"
)
EVAL_JSON_PREFIX = "[eval] "
RESULT_PATH_RE = re.compile(r"Data has been saved to (?P<path>.+/_result\.txt)\s*$")
EXPECTED_COVER_BLOCKS_PHASES = (
    "cover_left",
    "cover_middle",
    "cover_right",
    "uncover_red",
    "uncover_green",
    "uncover_blue",
)
EPISODE_VALIDITY_LABELS = (
    "valid_success",
    "valid_task_failure",
    "user_interrupted",
    "infrastructure_invalid",
    "incomplete_or_corrupt",
)
INFRASTRUCTURE_TERMINAL_EVENTS = {
    "pure_tool_control_control_backend_unavailable",
    "pure_tool_control_recovery_backend_unavailable",
}
INFRASTRUCTURE_TERMINAL_REASON_PREFIXES = (
    "pure_tool_control_control_backend_unavailable:",
    "pure_tool_control_recovery_backend_unavailable:",
)
CAPABILITY_TERMINAL_REASON_PREFIXES = (
    "pure_tool_control_max_control_turns_exhausted:",
    "pure_tool_control_max_no_progress_control_turns_exhausted:",
    "pure_tool_control_identity_binding_unresolved:",
)
FORMAL_MAX_SEMANTIC_ROUNDS = 10
CURRENT_FORMAL_PROTOCOL_VERSION = 3
FORMAL_PROTOCOL_BUDGETS = {
    1: {
        "max_control_turns": 8,
        "max_no_progress_control_turns": None,
    },
    2: {
        "max_control_turns": 64,
        "max_no_progress_control_turns": 4,
    },
    3: {
        "max_control_turns": 64,
        "max_no_progress_control_turns": 10,
    },
}
FORMAL_MAX_CONTROL_TURNS = int(
    FORMAL_PROTOCOL_BUDGETS[CURRENT_FORMAL_PROTOCOL_VERSION]["max_control_turns"]
)
FORMAL_MAX_NO_PROGRESS_CONTROL_TURNS = int(
    FORMAL_PROTOCOL_BUDGETS[CURRENT_FORMAL_PROTOCOL_VERSION][
        "max_no_progress_control_turns"
    ]
)
FORMAL_EPISODE_IDENTITY_FIELDS = (
    "task_name",
    "task_config",
    "policy_name",
    "ckpt_setting",
)
FORMAL_RUNTIME_PROVENANCE_FIELDS = (
    "manifest_sha256",
    "runtime_tree_sha256",
    "git_head",
    "tracked_runtime_diff_sha256",
    "archive_file",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GPT-5.5 object-info ablation outputs.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Log parent produced by run_gpt55_object_info_ablation_8way.sh.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary path.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional per-episode CSV path.")
    parser.add_argument(
        "--require-acceptance",
        action="store_true",
        help="Exit non-zero unless every discovered episode passes the P0 pure tool-control acceptance gates.",
    )
    return parser.parse_args()


def parse_result_txt(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"result_path": str(path)}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("Success Rate:"):
            result["success_rate"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Reward:"):
            result["mean_reward"] = float(line.split(":", 1)[1].strip())
    return result


def parse_eval_log(path: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RESULT_RE.search(raw_line.strip())
        if not match:
            continue
        item = match.groupdict()
        episodes.append(
            {
                "episode_id": int(item["episode_id"]),
                "seed": int(item["seed"]),
                "instruction": item["instruction"],
                "result": item["result"],
                "success": item["result"].strip().lower() == "success",
                "steps": int(item["steps"]),
                "reward": float(item["reward"]),
                "failure_reason": item["failure_reason"],
                "eval_log": str(path),
            }
        )
    return episodes


def parse_worker_log(path: Path) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    agent_finished: dict[tuple[int, int], dict[str, Any]] = {}
    result_path = ""
    run_dir = ""
    instruction_by_key: dict[tuple[int, int], str] = {}

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        result_match = RESULT_PATH_RE.search(line)
        if result_match:
            result_path = result_match.group("path").strip()
            run_dir = str(Path(result_path).parent)
            continue
        if not line.startswith(EVAL_JSON_PREFIX):
            continue
        payload_text = line[len(EVAL_JSON_PREFIX) :].strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        event = str(payload.get("event", ""))
        if event == "rollout_video_finalize" and payload.get("rollout_dir"):
            run_dir = str(Path(str(payload["rollout_dir"])).parent)
        elif event in {"rollout_report_finalize", "rollout_contact_sheet_finalize"} and payload.get("report"):
            run_dir = str(Path(str(payload["report"])).parent)
        episode_id = int(payload.get("episode_id", -1))
        seed = int(payload.get("seed", -1))
        key = (episode_id, seed)
        if event == "episode_start":
            instruction = str(payload.get("instruction", "") or "")
            if instruction:
                instruction_by_key[key] = instruction
        elif event == "episode_agent_finished":
            agent_finished[key] = dict(payload)
        elif event == "episode_end":
            finished = agent_finished.get(key, {})
            result = str(payload.get("result", ""))
            episodes.append(
                {
                    "episode_id": episode_id,
                    "seed": seed,
                    "instruction": instruction_by_key.get(key, ""),
                    "result": result,
                    "success": bool(payload.get("success", False)),
                    "steps": int(payload.get("total_steps", payload.get("step", 0)) or 0),
                    "reward": float(payload.get("max_reward", payload.get("reward", 0.0)) or 0.0),
                    "failure_reason": str(payload.get("failure_reason", "") or ""),
                    "worker_log": str(path),
                    "agent_finish_reason": str(finished.get("reason", "") or ""),
                    "monitor_phase": str(payload.get("monitor_phase", finished.get("monitor_phase", "")) or ""),
                    "monitor_status": str(payload.get("monitor_status", finished.get("monitor_status", "")) or ""),
                    "task_finished": bool(payload.get("task_finished", finished.get("task_finished", False))),
                }
            )

    return {
        "condition": condition_from_path(path),
        "run_dir": run_dir or str(path.parent),
        "worker_log": str(path),
        "result_path": result_path,
        "episodes": episodes,
    }


def condition_from_path(path: Path) -> str:
    parts = set(path.parts)
    if "no_oracle" in parts:
        return "no_oracle"
    if "oracle" in parts:
        return "oracle"
    text = str(path)
    if "no_oracle" in text:
        return "no_oracle"
    if "oracle" in text:
        return "oracle"
    return "unknown"


def classify_cover_blocks_subtask(text: str) -> str:
    normalized = " ".join(str(text or "").lower().split())
    if "no further action" in normalized or "no further manipulation" in normalized:
        return ""
    is_uncover = bool(re.search(r"\b(?:uncover|remove)\b", normalized)) or "place it away" in normalized
    if is_uncover:
        color_mentions = [
            (normalized.index(color), color)
            for color in ("red", "green", "blue")
            if color in normalized
        ]
        if color_mentions:
            _, color = min(color_mentions)
            return f"uncover_{color}"
        return ""
    if "lid" not in normalized and "cover" not in normalized:
        return ""
    for position in ("left", "middle", "right"):
        if position in normalized:
            return f"cover_{position}"
    return ""


def evaluate_cover_blocks_phase_order(subtasks: list[str]) -> dict[str, Any]:
    classified = [classify_cover_blocks_subtask(text) for text in subtasks]
    expected = list(EXPECTED_COVER_BLOCKS_PHASES)
    matched: list[str] = []
    unexpected: list[dict[str, Any]] = []
    cursor = 0
    for index, phase in enumerate(classified):
        if not phase:
            unexpected.append({"index": index, "phase": "", "subtask": subtasks[index]})
            continue
        if cursor < len(expected) and phase == expected[cursor]:
            matched.append(phase)
            cursor += 1
            continue
        if cursor > 0 and phase == expected[cursor - 1]:
            continue
        unexpected.append({"index": index, "phase": phase, "subtask": subtasks[index]})
    return {
        "expected": expected,
        "classified": classified,
        "matched": matched,
        "unexpected": unexpected,
        "passed": cursor == len(expected) and not unexpected,
        "next_expected_phase": expected[cursor] if cursor < len(expected) else "",
    }


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "sum_sec": 0.0, "mean_sec": 0.0, "median_sec": 0.0, "max_sec": 0.0}
    return {
        "count": len(values),
        "sum_sec": round(sum(values), 3),
        "mean_sec": round(statistics.mean(values), 3),
        "median_sec": round(statistics.median(values), 3),
        "max_sec": round(max(values), 3),
    }


def first_int(*values: Any, default: int = -1) -> int:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def formal_runtime_provenance_summary(value: Any) -> dict[str, str]:
    payload = value if isinstance(value, dict) else {}
    return {
        key: str(payload.get(key, "") or "")
        for key in FORMAL_RUNTIME_PROVENANCE_FIELDS
    }


def is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdefABCDEF" for character in text)


def classify_trace_episode_validity(
    *,
    event_counts: Counter[str],
    episode_start: dict[str, Any],
    episode_end: dict[str, Any],
    strict_bad_issue_count: int,
    malformed_jsonl_line_count: int = 0,
) -> dict[str, Any]:
    """Conservatively classify one attempted episode from structured trace state."""

    runtime_payload = episode_end.get("episode_validity", {})
    runtime_label_raw = ""
    runtime_label = ""
    if isinstance(runtime_payload, dict):
        runtime_label_raw = str(runtime_payload.get("label", "") or "")
        if runtime_label_raw in EPISODE_VALIDITY_LABELS:
            runtime_label = runtime_label_raw

    trace_complete = event_counts["episode_start"] == 1 and event_counts["episode_end"] == 1
    success = bool(episode_end.get("success", False))
    environment_success = bool(episode_end.get("environment_success", False))
    terminal_reason = str(episode_end.get("terminal_failure_reason", "") or "")
    infrastructure_terminal = bool(
        any(event_counts[event] for event in INFRASTRUCTURE_TERMINAL_EVENTS)
        or terminal_reason.startswith(INFRASTRUCTURE_TERMINAL_REASON_PREFIXES)
    )

    if event_counts["episode_interrupt"]:
        trace_audit_label = "user_interrupted"
        trace_audit_reason = "episode_interrupt event present"
    elif event_counts["episode_exception"]:
        trace_audit_label = "incomplete_or_corrupt"
        trace_audit_reason = "episode_exception event present"
    elif malformed_jsonl_line_count:
        trace_audit_label = "incomplete_or_corrupt"
        trace_audit_reason = (
            f"trace contains {malformed_jsonl_line_count} malformed non-empty JSONL line(s)"
        )
    elif success and environment_success and trace_complete and strict_bad_issue_count == 0:
        trace_audit_label = "valid_success"
        trace_audit_reason = "complete clean trace with authoritative environment success"
    elif infrastructure_terminal:
        trace_audit_label = "infrastructure_invalid"
        trace_audit_reason = terminal_reason or "typed backend-unavailable terminal event present"
    elif not trace_complete:
        trace_audit_label = "incomplete_or_corrupt"
        trace_audit_reason = "episode_start/episode_end cardinality is not exactly one"
    elif strict_bad_issue_count:
        trace_audit_label = "incomplete_or_corrupt"
        trace_audit_reason = "strict trace checker found a control-flow integrity issue"
    elif success and not environment_success:
        trace_audit_label = "incomplete_or_corrupt"
        trace_audit_reason = "success flag lacks authoritative environment_success"
    elif bool(episode_end.get("terminal_failure", False)) and not terminal_reason.startswith(
        CAPABILITY_TERMINAL_REASON_PREFIXES
    ):
        trace_audit_label = "incomplete_or_corrupt"
        trace_audit_reason = "unclassified runtime terminal failure requires artifact audit"
    else:
        trace_audit_label = "valid_task_failure"
        trace_audit_reason = "complete clean episode without environment success or infrastructure failure"

    runtime_label_present = bool(runtime_label)
    runtime_label_matches = runtime_label_present and runtime_label == trace_audit_label
    eligibility_exclusion_reasons: list[str] = []
    if trace_audit_label not in {"valid_success", "valid_task_failure"}:
        eligibility_exclusion_reasons.append(f"trace_audit_label:{trace_audit_label}")
    if not runtime_label_present:
        eligibility_exclusion_reasons.append("runtime_validity_label_missing_or_invalid")
        reason = (
            "Runtime episode_validity.label is missing or invalid; raw independent trace audit: "
            f"{trace_audit_label} ({trace_audit_reason})"
        )
    elif not runtime_label_matches:
        eligibility_exclusion_reasons.append(
            f"runtime_validity_label_mismatch:{runtime_label}!={trace_audit_label}"
        )
        reason = (
            f"Runtime validity label {runtime_label!r} conflicts with independent trace audit "
            f"{trace_audit_label!r}: {trace_audit_reason}"
        )
    else:
        reason = trace_audit_reason
    benchmark_denominator_eligible = bool(
        trace_audit_label in {"valid_success", "valid_task_failure"}
        and runtime_label_matches
    )

    return {
        "label": trace_audit_label,
        "trace_audit_label": trace_audit_label,
        "trace_audit_reason": trace_audit_reason,
        "benchmark_denominator_eligible": benchmark_denominator_eligible,
        "task_success_numerator": benchmark_denominator_eligible and trace_audit_label == "valid_success",
        "reason": reason,
        "runtime_label_raw": runtime_label_raw,
        "runtime_label": runtime_label,
        "runtime_label_present": runtime_label_present,
        "runtime_label_matches_trace_audit": runtime_label_matches,
        "eligibility_exclusion_reasons": eligibility_exclusion_reasons,
    }


def parse_agent_trace(path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    event_counts = Counter(str(record.get("event", "")) for record in records)
    episode_start = next((record for record in records if record.get("event") == "episode_start"), {})
    episode_end = next((record for record in reversed(records) if record.get("event") == "episode_end"), {})
    subtasks = [
        str(record.get("subtask_text", "") or "")
        for record in records
        if record.get("event") == "control_turn_result"
    ]
    tool_counts = Counter(
        str(result.get("tool_name", ""))
        for record in records
        if record.get("event") == "recovery_result"
        for result in record.get("results", [])
        if isinstance(result, dict) and result.get("tool_name")
    )
    terminal_tool_skip_count = sum(
        1
        for record in records
        if record.get("event") == "recovery_result"
        for result in record.get("results", [])
        if isinstance(result, dict)
        and isinstance(result.get("details"), dict)
        and result["details"].get("terminal_skip") is True
    )
    effect_verdicts = Counter(
        str(record.get("result", {}).get("effect_verified", ""))
        for record in records
        if record.get("event") == "action_effect_verification" and isinstance(record.get("result"), dict)
    )
    control_latencies = [
        float(record.get("latency_sec", 0.0))
        for record in records
        if record.get("event") == "control_turn_result" and isinstance(record.get("latency_sec"), (int, float))
    ]
    strict_report = analyze_trace(path)
    malformed_jsonl_line_count = int(strict_report.get("malformed_jsonl_line_count", 0) or 0)
    episode_identity: dict[str, dict[str, str]] = {
        field: {
            "episode_start": str(episode_start.get(field, "") or ""),
            "episode_end": str(episode_end.get(field, "") or ""),
        }
        for field in FORMAL_EPISODE_IDENTITY_FIELDS
    }
    task_name = (
        episode_identity["task_name"]["episode_start"]
        or episode_identity["task_name"]["episode_end"]
    ).strip()
    phase_order = evaluate_cover_blocks_phase_order(subtasks) if task_name == "cover_blocks" else {
        "expected": [],
        "classified": [],
        "matched": [],
        "unexpected": [],
        "passed": True,
        "next_expected_phase": "",
    }
    query_error_count = event_counts["observation_preprocess_query_error"]
    normalization_error_count = event_counts["observation_preprocess_normalization_error"]
    inferred_gpt_calls: int | None = None
    inferred_breakdown: dict[str, int] = {}
    if query_error_count == 0 and normalization_error_count == 0:
        preprocess_count = event_counts["observation_preprocess"]
        inferred_breakdown = {
            "perception_queries": preprocess_count,
            "normalize_perception_queries": preprocess_count,
            "plan": event_counts["control_turn_result"],
            "recover_plan": event_counts["recovery_result"],
            "effect_verification_via_recover": event_counts["action_effect_verification"],
        }
        inferred_gpt_calls = sum(inferred_breakdown.values())
    wall_duration_sec = 0.0
    if episode_start.get("timestamp") is not None and episode_end.get("timestamp") is not None:
        wall_duration_sec = max(0.0, float(episode_end["timestamp"]) - float(episode_start["timestamp"]))
    episode_start_present = bool(episode_start)
    episode_end_present = bool(episode_end)
    trace_complete = bool(
        event_counts["episode_start"] == 1
        and event_counts["episode_end"] == 1
        and malformed_jsonl_line_count == 0
    )
    episode_end_success = bool(episode_end.get("success", False))
    strict_issue_count = len(strict_report.get("issues", []))
    strict_bad_issue_count = len(strict_report.get("bad_issues", []))
    strict_passed = strict_issue_count == 0
    vla_request_count = event_counts["vla_request"]
    episode_validity = classify_trace_episode_validity(
        event_counts=event_counts,
        episode_start=episode_start,
        episode_end=episode_end,
        strict_bad_issue_count=strict_bad_issue_count,
        malformed_jsonl_line_count=malformed_jsonl_line_count,
    )
    acceptance_passed = bool(
        trace_complete
        and episode_end_success
        and strict_passed
        and episode_validity["benchmark_denominator_eligible"]
        and vla_request_count == 0
        and phase_order["passed"]
    )
    failure_stage = ""
    if not acceptance_passed:
        if not episode_start_present:
            failure_stage = "missing_episode_start"
        elif not episode_end_present or not trace_complete:
            failure_stage = "incomplete_trace"
        elif not episode_end_success:
            failure_reason = str(episode_end.get("failure_reason", "") or "")
            if event_counts["episode_interrupt"]:
                failure_stage = failure_reason or "episode_interrupted"
            elif event_counts["episode_exception"]:
                failure_stage = failure_reason or "episode_exception"
            else:
                failure_stage = failure_reason or str(
                    phase_order.get("next_expected_phase", "") or "environment_failure"
                )
        elif not strict_passed:
            failure_stage = "strict_trace_failure"
        elif not episode_validity["runtime_label_present"]:
            failure_stage = "runtime_validity_label_missing_or_invalid"
        elif not episode_validity["runtime_label_matches_trace_audit"]:
            failure_stage = "runtime_validity_label_mismatch"
        elif vla_request_count:
            failure_stage = "unexpected_vla_request"
        elif not phase_order["passed"]:
            failure_stage = str(phase_order.get("next_expected_phase", "") or "phase_order_failure")
    return {
        "trace": str(path),
        "episode_id": first_int(episode_end.get("episode_id"), episode_start.get("episode_id")),
        "seed": first_int(episode_end.get("seed"), episode_start.get("seed")),
        "episode_start_id": first_int(episode_start.get("episode_id")),
        "episode_end_id": first_int(episode_end.get("episode_id")),
        "episode_start_seed": first_int(episode_start.get("seed")),
        "episode_end_seed": first_int(episode_end.get("seed")),
        "task_name": task_name,
        "task_config": (
            episode_identity["task_config"]["episode_start"]
            or episode_identity["task_config"]["episode_end"]
        ).strip(),
        "policy_name": (
            episode_identity["policy_name"]["episode_start"]
            or episode_identity["policy_name"]["episode_end"]
        ).strip(),
        "ckpt_setting": (
            episode_identity["ckpt_setting"]["episode_start"]
            or episode_identity["ckpt_setting"]["episode_end"]
        ).strip(),
        "episode_identity": episode_identity,
        "records": len(records),
        "malformed_jsonl_line_count": malformed_jsonl_line_count,
        "malformed_jsonl_line_numbers": list(
            strict_report.get("malformed_jsonl_line_numbers", []) or []
        ),
        "event_counts": dict(event_counts),
        "episode_start_present": episode_start_present,
        "episode_end_present": episode_end_present,
        "trace_complete": trace_complete,
        "episode_end_success": episode_end_success,
        "environment_success": bool(episode_end.get("environment_success", False)),
        "episode_validity": episode_validity,
        "episode_start_formal_protocol": episode_start.get("formal_protocol") is True,
        "episode_end_formal_protocol": episode_end.get("formal_protocol") is True,
        "episode_start_formal_protocol_version": first_int(
            episode_start.get("formal_protocol_version")
        ),
        "episode_end_formal_protocol_version": first_int(
            episode_end.get("formal_protocol_version")
        ),
        "episode_start_formal_protocol_version_recorded": (
            "formal_protocol_version" in episode_start
        ),
        "episode_end_formal_protocol_version_recorded": (
            "formal_protocol_version" in episode_end
        ),
        "episode_start_runtime_provenance": formal_runtime_provenance_summary(
            episode_start.get("runtime_provenance")
        ),
        "episode_end_runtime_provenance": formal_runtime_provenance_summary(
            episode_end.get("runtime_provenance")
        ),
        "perception_condition": str(episode_end.get("perception_condition", "") or ""),
        "pure_tool_control": episode_end.get("pure_tool_control") is True,
        "semantic_round_index_at_end": first_int(episode_end.get("semantic_round_index")),
        "max_semantic_rounds_per_active_subtask": first_int(
            episode_end.get("max_semantic_rounds_per_active_subtask")
        ),
        "max_control_turns": first_int(episode_end.get("max_control_turns")),
        "max_no_progress_control_turns": first_int(
            episode_end.get("max_no_progress_control_turns")
        ),
        "backend_error_budget": first_int(episode_end.get("backend_error_budget")),
        "strict_passed": strict_passed,
        "strict_issue_count": strict_issue_count,
        "strict_bad_issue_count": strict_bad_issue_count,
        "vla_request_count": vla_request_count,
        "control_subtasks": subtasks,
        "phase_order": phase_order,
        "tool_call_count": sum(tool_counts.values()),
        "tool_counts": dict(tool_counts),
        "terminal_tool_skip_count": terminal_tool_skip_count,
        "effect_verdicts": dict(effect_verdicts),
        "environment_success_effect_count": event_counts["environment_success_effect_commit"],
        "gpt_endpoint_call_count_inferred": inferred_gpt_calls,
        "gpt_endpoint_call_breakdown_inferred": inferred_breakdown,
        "gpt_call_count_inference_valid": inferred_gpt_calls is not None,
        "query_error_count": query_error_count,
        "normalization_error_count": normalization_error_count,
        "control_turn_latency_sec": summarize_values(control_latencies),
        "wall_duration_sec": round(wall_duration_sec, 3),
        "failure_stage": failure_stage,
        "acceptance_passed": acceptance_passed,
    }


def build_formal_metadata(
    report: dict[str, Any],
    *,
    run_condition: str,
    run_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, error: str) -> bool:
        if not condition:
            errors.append(error)
        return condition

    episode_identity = report.get("episode_identity", {})
    if not isinstance(episode_identity, dict):
        episode_identity = {}
    identity_present: dict[str, bool] = {}
    identity_matches: dict[str, bool] = {}
    normalized_identity: dict[str, dict[str, str]] = {}
    for field in FORMAL_EPISODE_IDENTITY_FIELDS:
        raw_values = episode_identity.get(field, {})
        if not isinstance(raw_values, dict):
            raw_values = {}
        start_value = str(raw_values.get("episode_start", "") or "")
        end_value = str(raw_values.get("episode_end", "") or "")
        normalized_identity[field] = {
            "episode_start": start_value,
            "episode_end": end_value,
        }
        present = bool(start_value.strip() and end_value.strip())
        identity_present[field] = require(
            present,
            f"{field}_missing_at_episode_boundary",
        )
        matches = bool(present and start_value == end_value)
        identity_matches[field] = matches
        if present:
            require(matches, f"{field}_start_end_mismatch")
    episode_id_consistent = require(
        int(report.get("episode_start_id", -1)) >= 0
        and int(report.get("episode_end_id", -1)) >= 0
        and int(report.get("episode_start_id", -1)) == int(report.get("episode_end_id", -1)),
        "episode_id_missing_or_inconsistent",
    )
    seed_present_and_consistent = require(
        int(report.get("episode_start_seed", -1)) >= 0
        and int(report.get("episode_end_seed", -1)) >= 0
        and int(report.get("episode_start_seed", -1)) == int(report.get("episode_end_seed", -1)),
        "seed_missing_or_inconsistent",
    )
    condition_recognized = require(
        run_condition in {"oracle", "no_oracle"},
        "run_condition_missing_or_unknown",
    )
    trace_condition = str(report.get("perception_condition", "") or "")
    condition_recorded_in_trace = require(
        trace_condition in {"oracle", "no_oracle"},
        "perception_condition_missing_or_invalid",
    )
    condition_matches_trace = require(
        condition_recognized
        and condition_recorded_in_trace
        and trace_condition == run_condition,
        "perception_condition_mismatch",
    )
    pure_tool_control = require(
        report.get("pure_tool_control") is True,
        "pure_tool_control_not_true",
    )
    start_protocol_version = int(
        report.get("episode_start_formal_protocol_version", -1)
    )
    end_protocol_version = int(
        report.get("episode_end_formal_protocol_version", -1)
    )
    start_protocol_version_recorded = bool(
        report.get("episode_start_formal_protocol_version_recorded", False)
    )
    end_protocol_version_recorded = bool(
        report.get("episode_end_formal_protocol_version_recorded", False)
    )
    protocol_version = -1
    protocol_version_source = "invalid"
    protocol_version_boundary_present = False
    protocol_version_boundary_consistent = False
    if not start_protocol_version_recorded and not end_protocol_version_recorded:
        if int(report.get("max_control_turns", -1)) == 8:
            protocol_version = 1
            protocol_version_source = "legacy_inferred"
            protocol_version_boundary_consistent = True
        else:
            require(False, "formal_protocol_version_missing_for_nonlegacy_budget")
    else:
        protocol_version_boundary_present = require(
            start_protocol_version_recorded and end_protocol_version_recorded,
            "formal_protocol_version_missing_at_episode_boundary",
        )
        if protocol_version_boundary_present:
            protocol_version_values_valid = require(
                start_protocol_version > 0 and end_protocol_version > 0,
                "formal_protocol_version_must_be_positive_integer",
            )
            if protocol_version_values_valid:
                protocol_version_boundary_consistent = require(
                    start_protocol_version == end_protocol_version,
                    "formal_protocol_version_start_end_mismatch",
                )
        if protocol_version_boundary_consistent:
            protocol_version_supported = require(
                start_protocol_version in FORMAL_PROTOCOL_BUDGETS,
                f"formal_protocol_version_unsupported:{start_protocol_version}",
            )
            if protocol_version_supported:
                protocol_version = start_protocol_version
                protocol_version_source = "explicit"

    protocol_budget = FORMAL_PROTOCOL_BUDGETS.get(protocol_version)
    max_semantic_rounds_matches_protocol = require(
        int(report.get("max_semantic_rounds_per_active_subtask", -1))
        == FORMAL_MAX_SEMANTIC_ROUNDS,
        f"max_semantic_rounds_must_equal_{FORMAL_MAX_SEMANTIC_ROUNDS}",
    )
    max_control_turns_matches_protocol = False
    max_no_progress_control_turns_required = False
    max_no_progress_control_turns_matches_protocol = False
    if protocol_budget is not None:
        expected_max_control_turns = int(protocol_budget["max_control_turns"])
        max_control_turns_matches_protocol = require(
            int(report.get("max_control_turns", -1)) == expected_max_control_turns,
            f"max_control_turns_must_equal_{expected_max_control_turns}_for_protocol_v{protocol_version}",
        )
        expected_max_no_progress_control_turns = protocol_budget[
            "max_no_progress_control_turns"
        ]
        max_no_progress_control_turns_required = (
            expected_max_no_progress_control_turns is not None
        )
        if max_no_progress_control_turns_required:
            expected_no_progress = int(expected_max_no_progress_control_turns)
            max_no_progress_control_turns_matches_protocol = require(
                int(report.get("max_no_progress_control_turns", -1))
                == expected_no_progress,
                (
                    "max_no_progress_control_turns_must_equal_"
                    f"{expected_no_progress}_for_protocol_v{protocol_version}"
                ),
            )
        else:
            max_no_progress_control_turns_matches_protocol = True
    backend_error_budget_recorded = require(
        int(report.get("backend_error_budget", -1)) >= 0,
        "backend_error_budget_missing_or_invalid",
    )
    formal_protocol_start = require(
        report.get("episode_start_formal_protocol") is True,
        "episode_start_formal_protocol_not_true",
    )
    formal_protocol_end = require(
        report.get("episode_end_formal_protocol") is True,
        "episode_end_formal_protocol_not_true",
    )

    start_provenance = formal_runtime_provenance_summary(
        report.get("episode_start_runtime_provenance")
    )
    end_provenance = formal_runtime_provenance_summary(
        report.get("episode_end_runtime_provenance")
    )
    start_missing_fields = [key for key, value in start_provenance.items() if not value.strip()]
    end_missing_fields = [key for key, value in end_provenance.items() if not value.strip()]
    provenance_start_complete = require(
        not start_missing_fields,
        "episode_start_runtime_provenance_missing:" + ",".join(start_missing_fields),
    )
    provenance_end_complete = require(
        not end_missing_fields,
        "episode_end_runtime_provenance_missing:" + ",".join(end_missing_fields),
    )
    provenance_start_end_matches = require(
        provenance_start_complete
        and provenance_end_complete
        and start_provenance == end_provenance,
        "runtime_provenance_start_end_mismatch",
    )
    provenance_hash_validity = [
        require(
            is_sha256(start_provenance.get(key, "")),
            f"runtime_provenance_{key}_invalid",
        )
        for key in ("manifest_sha256", "runtime_tree_sha256", "tracked_runtime_diff_sha256")
    ]
    provenance_hashes_valid = all(provenance_hash_validity)

    archive_file = start_provenance.get("archive_file", "")
    archive_name = Path(archive_file)
    archive_file_safe = require(
        bool(archive_file)
        and not archive_name.is_absolute()
        and archive_name.name == archive_file
        and archive_file.strip() == archive_file
        and archive_file not in {".", ".."}
        and "\x00" not in archive_file
        and "/" not in archive_file
        and "\\" not in archive_file,
        "runtime_provenance_archive_file_is_not_a_safe_basename",
    )
    archive_path = run_dir / archive_file if archive_file_safe else run_dir / "__invalid_provenance__"
    archive_resolves_inside_run_dir = False
    if archive_file_safe:
        try:
            archive_resolves_inside_run_dir = archive_path.resolve().parent == run_dir.resolve()
        except (OSError, RuntimeError, ValueError):
            archive_resolves_inside_run_dir = False
    require(
        archive_resolves_inside_run_dir,
        "runtime_provenance_archive_escapes_run_dir",
    )
    archive_exists = require(
        archive_resolves_inside_run_dir and archive_path.is_file(),
        "runtime_provenance_archive_missing",
    )

    archive_payload: dict[str, Any] = {}
    archive_sha256 = ""
    archive_json_valid = False
    if archive_exists:
        try:
            archive_bytes = archive_path.read_bytes()
            archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
            decoded = json.loads(archive_bytes.decode("utf-8"))
            if isinstance(decoded, dict):
                archive_payload = decoded
                archive_json_valid = True
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            archive_json_valid = False
    require(archive_json_valid, "runtime_provenance_archive_invalid_json")
    archive_manifest_hash_matches = require(
        archive_json_valid
        and bool(start_provenance.get("manifest_sha256"))
        and archive_sha256 == start_provenance.get("manifest_sha256"),
        "runtime_provenance_manifest_sha256_mismatch",
    )
    archive_payload_matches_trace = require(
        archive_json_valid
        and all(
            str(archive_payload.get(key, "") or "") == start_provenance.get(key, "")
            for key in ("runtime_tree_sha256", "git_head", "tracked_runtime_diff_sha256")
        ),
        "runtime_provenance_archive_payload_mismatch",
    )
    archive_secrets_safe = require(
        archive_json_valid and archive_payload.get("secrets_recorded") is False,
        "runtime_provenance_archive_secrets_recorded_not_false",
    )

    return {
        "episode_identity": normalized_identity,
        "task_name_present": identity_present["task_name"],
        "task_name_consistent": identity_matches["task_name"],
        "task_config_present": identity_present["task_config"],
        "task_config_consistent": identity_matches["task_config"],
        "policy_name_present": identity_present["policy_name"],
        "policy_name_consistent": identity_matches["policy_name"],
        "ckpt_setting_present": identity_present["ckpt_setting"],
        "ckpt_setting_consistent": identity_matches["ckpt_setting"],
        "episode_id_consistent": episode_id_consistent,
        "seed_present": seed_present_and_consistent,
        "seed_consistent": seed_present_and_consistent,
        "condition": run_condition,
        "condition_recognized": condition_recognized,
        "condition_recorded_in_trace": condition_recorded_in_trace,
        "condition_matches_trace": condition_matches_trace,
        "pure_tool_control": pure_tool_control,
        "formal_protocol_version": protocol_version,
        "formal_protocol_version_source": protocol_version_source,
        "episode_start_formal_protocol_version": start_protocol_version,
        "episode_end_formal_protocol_version": end_protocol_version,
        "episode_start_formal_protocol_version_recorded": start_protocol_version_recorded,
        "episode_end_formal_protocol_version_recorded": end_protocol_version_recorded,
        "formal_protocol_version_boundary_present": protocol_version_boundary_present,
        "formal_protocol_version_boundary_consistent": protocol_version_boundary_consistent,
        "max_semantic_rounds_recorded": int(
            report.get("max_semantic_rounds_per_active_subtask", -1)
        ) > 0,
        "max_semantic_rounds_matches_protocol": max_semantic_rounds_matches_protocol,
        "max_control_turns_recorded": int(report.get("max_control_turns", -1)) >= 0,
        "max_control_turns_matches_protocol": max_control_turns_matches_protocol,
        "max_no_progress_control_turns_recorded": int(
            report.get("max_no_progress_control_turns", -1)
        ) >= 0,
        "max_no_progress_control_turns_required": max_no_progress_control_turns_required,
        "max_no_progress_control_turns_matches_protocol": (
            max_no_progress_control_turns_matches_protocol
        ),
        "backend_error_budget_recorded": backend_error_budget_recorded,
        "formal_protocol_start": formal_protocol_start,
        "formal_protocol_end": formal_protocol_end,
        "runtime_provenance_start": start_provenance,
        "runtime_provenance_end": end_provenance,
        "runtime_provenance_start_complete": provenance_start_complete,
        "runtime_provenance_end_complete": provenance_end_complete,
        "runtime_provenance_start_end_matches": provenance_start_end_matches,
        "runtime_provenance_hashes_valid": provenance_hashes_valid,
        "runtime_provenance_archive_file_safe": archive_file_safe,
        "runtime_provenance_archive_exists": archive_exists,
        "runtime_provenance_archive_json_valid": archive_json_valid,
        "runtime_provenance_manifest_sha256_matches": archive_manifest_hash_matches,
        "runtime_provenance_archive_payload_matches_trace": archive_payload_matches_trace,
        "runtime_provenance_archive_secrets_safe": archive_secrets_safe,
        "errors": errors,
        "complete": not errors,
    }


def enrich_run_with_traces(run: dict[str, Any], run_dir: Path) -> None:
    trace_paths = sorted(run_dir.glob("episode_*_agent_trace.jsonl"))
    trace_reports = [parse_agent_trace(path) for path in trace_paths]
    unmatched_reports = list(trace_reports)
    run["trace_reports"] = trace_reports
    episodes = run.get("episodes", [])
    if not isinstance(episodes, list):
        return
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        episode_id = int(episode.get("episode_id", -1))
        seed = int(episode.get("seed", -1))
        report = next(
            (
                item
                for item in unmatched_reports
                if int(item.get("episode_id", -1)) == episode_id and int(item.get("seed", -1)) == seed
            ),
            None,
        )
        if report is not None:
            run_condition = str(run.get("condition", "") or "")
            formal_metadata = build_formal_metadata(
                report,
                run_condition=run_condition,
                run_dir=run_dir,
            )
            report["formal_metadata"] = formal_metadata
            validity = dict(report.get("episode_validity", {}) or {})
            exclusion_reasons = [
                str(item)
                for item in validity.get("eligibility_exclusion_reasons", []) or []
                if str(item).strip()
            ]
            exclusion_reasons.extend(
                f"formal_metadata:{error}" for error in formal_metadata.get("errors", [])
            )
            exclusion_reasons = list(dict.fromkeys(exclusion_reasons))
            raw_eligible = bool(validity.get("benchmark_denominator_eligible", False))
            formal_eligible = bool(raw_eligible and formal_metadata.get("complete", False))
            validity["benchmark_denominator_eligible"] = formal_eligible
            validity["task_success_numerator"] = bool(
                formal_eligible and validity.get("trace_audit_label") == "valid_success"
            )
            validity["formal_metadata_complete"] = bool(formal_metadata.get("complete", False))
            validity["eligibility_exclusion_reasons"] = exclusion_reasons
            validity["formal_eligibility_reason"] = (
                "runtime validity, independent trace audit, and formal metadata agree"
                if formal_eligible
                else "; ".join(exclusion_reasons) or "formal capability eligibility failed closed"
            )
            report["episode_validity"] = validity
            report["formal_base_eligible"] = formal_eligible
            report["formal_base_acceptance_passed"] = bool(
                report.get("acceptance_passed", False) and formal_eligible
            )
            report["formal_acceptance_passed"] = report["formal_base_acceptance_passed"]
            if not formal_eligible and not str(report.get("failure_stage", "") or ""):
                report["failure_stage"] = "formal_metadata_incomplete"
            episode["trace_metrics"] = report
            episode["episode_validity"] = dict(validity)
            unmatched_reports.remove(report)


def apply_cross_episode_formal_gates(
    runs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fail closed on duplicate identities, mixed runtimes, or mixed protocol versions."""

    candidates_by_condition: dict[str, list[dict[str, Any]]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for run in runs:
        condition = str(run.get("condition", "unknown") or "unknown")
        diagnostics.setdefault(
            condition,
            {
                "duplicate_identity_keys": [],
                "duplicate_identity_episode_count": 0,
                "runtime_tree_sha256_counts": {},
                "mixed_runtime_tree_sha256": False,
                "formal_protocol_version_counts": {},
                "mixed_formal_protocol_version": False,
            },
        )
        episodes = run.get("episodes", [])
        if not isinstance(episodes, list):
            continue
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            report = episode.get("trace_metrics", {})
            if not isinstance(report, dict) or not report:
                continue
            validity = report.get("episode_validity", {})
            formal_metadata = report.get("formal_metadata", {})
            if not isinstance(validity, dict) or not isinstance(formal_metadata, dict):
                continue
            base_eligible = bool(
                report.get(
                    "formal_base_eligible",
                    validity.get("benchmark_denominator_eligible", False)
                    and formal_metadata.get("complete", False),
                )
            )
            report["formal_base_eligible"] = base_eligible
            report.setdefault(
                "formal_base_acceptance_passed",
                bool(report.get("formal_acceptance_passed", False) and base_eligible),
            )
            report["formal_cross_episode_eligible"] = False
            if not base_eligible:
                continue

            identity = formal_metadata.get("episode_identity", {})
            if not isinstance(identity, dict):
                identity = {}

            def identity_value(field: str) -> str:
                values = identity.get(field, {})
                if not isinstance(values, dict):
                    return ""
                return str(values.get("episode_start", "") or "").strip()

            seed = first_int(report.get("episode_start_seed"), episode.get("seed"))
            identity_key = (
                identity_value("task_name"),
                identity_value("task_config"),
                condition,
                seed,
            )
            provenance = formal_metadata.get("runtime_provenance_start", {})
            if not isinstance(provenance, dict):
                provenance = {}
            runtime_tree_sha256 = str(
                provenance.get("runtime_tree_sha256", "") or ""
            ).lower()
            formal_protocol_version = first_int(
                formal_metadata.get("formal_protocol_version")
            )
            candidates_by_condition.setdefault(condition, []).append(
                {
                    "episode": episode,
                    "report": report,
                    "identity_key": identity_key,
                    "runtime_tree_sha256": runtime_tree_sha256,
                    "formal_protocol_version": formal_protocol_version,
                }
            )

    for condition, candidates in candidates_by_condition.items():
        identity_counts = Counter(candidate["identity_key"] for candidate in candidates)
        duplicate_keys = {
            key for key, count in identity_counts.items() if count > 1
        }
        runtime_tree_counts = Counter(
            str(candidate["runtime_tree_sha256"] or "") for candidate in candidates
        )
        mixed_runtime_tree = len(runtime_tree_counts) > 1
        protocol_version_counts = Counter(
            int(candidate["formal_protocol_version"]) for candidate in candidates
        )
        mixed_protocol_version = len(protocol_version_counts) > 1
        diagnostics[condition] = {
            "duplicate_identity_keys": [
                {
                    "task_name": key[0],
                    "task_config": key[1],
                    "condition": key[2],
                    "seed": key[3],
                    "count": identity_counts[key],
                }
                for key in sorted(duplicate_keys)
            ],
            "duplicate_identity_episode_count": sum(
                identity_counts[key] for key in duplicate_keys
            ),
            "runtime_tree_sha256_counts": dict(sorted(runtime_tree_counts.items())),
            "mixed_runtime_tree_sha256": mixed_runtime_tree,
            "formal_protocol_version_counts": {
                str(version): count
                for version, count in sorted(protocol_version_counts.items())
            },
            "mixed_formal_protocol_version": mixed_protocol_version,
        }

        for candidate in candidates:
            report = candidate["report"]
            validity = dict(report.get("episode_validity", {}) or {})
            duplicate_identity = candidate["identity_key"] in duplicate_keys
            cross_eligible = (
                not duplicate_identity
                and not mixed_runtime_tree
                and not mixed_protocol_version
            )
            report["formal_duplicate_identity"] = duplicate_identity
            report["formal_mixed_runtime_tree_sha256"] = mixed_runtime_tree
            report["formal_mixed_protocol_version"] = mixed_protocol_version
            report["formal_cross_episode_eligible"] = cross_eligible
            exclusion_reasons = [
                str(item)
                for item in validity.get("eligibility_exclusion_reasons", []) or []
                if str(item).strip()
            ]
            if duplicate_identity:
                exclusion_reasons.append("duplicate_identity")
            if mixed_runtime_tree:
                exclusion_reasons.append("mixed_runtime_tree_sha256")
            if mixed_protocol_version:
                exclusion_reasons.append("mixed_formal_protocol_version")
            exclusion_reasons = list(dict.fromkeys(exclusion_reasons))
            validity["benchmark_denominator_eligible"] = cross_eligible
            validity["task_success_numerator"] = bool(
                cross_eligible and validity.get("trace_audit_label") == "valid_success"
            )
            validity["eligibility_exclusion_reasons"] = exclusion_reasons
            if not cross_eligible:
                validity["formal_eligibility_reason"] = "; ".join(exclusion_reasons)
            report["episode_validity"] = validity
            candidate["episode"]["episode_validity"] = dict(validity)
            report["formal_acceptance_passed"] = bool(
                report.get("formal_base_acceptance_passed", False) and cross_eligible
            )
            if duplicate_identity:
                report["failure_stage"] = "duplicate_identity"
            elif mixed_runtime_tree:
                report["failure_stage"] = "mixed_runtime_tree_sha256"
            elif mixed_protocol_version:
                report["failure_stage"] = "mixed_formal_protocol_version"

    return diagnostics


def acceptance_failure(report: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        **extra,
        "trace": report.get("trace", ""),
        "failure_stage": report.get("failure_stage", ""),
        "trace_complete": report.get("trace_complete", False),
        "episode_start_present": report.get("episode_start_present", False),
        "episode_end_present": report.get("episode_end_present", False),
        "episode_end_success": report.get("episode_end_success", False),
        "strict_passed": report.get("strict_passed", False),
        "strict_issue_count": report.get("strict_issue_count", 0),
        "vla_request_count": report.get("vla_request_count", 0),
        "phase_order": report.get("phase_order", {}),
        "episode_validity": report.get("episode_validity", {}),
        "formal_metadata": report.get("formal_metadata", {}),
    }


def build_acceptance_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    apply_cross_episode_formal_gates(runs)
    episode_count = 0
    matched_episode_count = 0
    passed_count = 0
    failed: list[dict[str, Any]] = []
    all_trace_reports: list[dict[str, Any]] = []
    matched_trace_paths: set[str] = set()

    for run in runs:
        run_dir = str(run.get("run_dir", "") or "")
        trace_reports = [report for report in run.get("trace_reports", []) if isinstance(report, dict)]
        all_trace_reports.extend(trace_reports)
        episodes = run.get("episodes", [])
        if not isinstance(episodes, list) or not episodes:
            failed.append(
                {
                    "run_dir": run_dir,
                    "episode_id": None,
                    "seed": None,
                    "trace": "",
                    "failure_stage": "missing_episode_result",
                }
            )
            continue

        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            episode_count += 1
            episode_id = first_int(episode.get("episode_id"))
            seed = first_int(episode.get("seed"))
            report = episode.get("trace_metrics", {})
            if not isinstance(report, dict) or not report:
                failed.append(
                    {
                        "run_dir": run_dir,
                        "episode_id": episode_id,
                        "seed": seed,
                        "trace": "",
                        "failure_stage": "missing_trace",
                        "eval_success": bool(episode.get("success", False)),
                    }
                )
                continue

            matched_episode_count += 1
            trace_path = str(report.get("trace", "") or "")
            if trace_path:
                matched_trace_paths.add(trace_path)
            eval_success = bool(episode.get("success", False))
            trace_passed = bool(report.get("formal_acceptance_passed", False))
            if eval_success and trace_passed:
                passed_count += 1
                continue
            failure = acceptance_failure(
                report,
                run_dir=run_dir,
                episode_id=episode_id,
                seed=seed,
                eval_success=eval_success,
            )
            if trace_passed and not eval_success:
                failure["failure_stage"] = "eval_result_failure"
            failed.append(failure)

    orphan_traces = [
        acceptance_failure(
            report,
            run_dir=str(Path(str(report.get("trace", "") or "")).parent),
            episode_id=first_int(report.get("episode_id")),
            seed=first_int(report.get("seed")),
        )
        for report in all_trace_reports
        if str(report.get("trace", "") or "") not in matched_trace_paths
    ]
    all_passed = bool(
        episode_count > 0
        and matched_episode_count == episode_count
        and passed_count == episode_count
        and not failed
        and not orphan_traces
    )
    return {
        "all_passed": all_passed,
        "episode_count": episode_count,
        "trace_count": len(all_trace_reports),
        "matched_episode_count": matched_episode_count,
        "passed_count": passed_count,
        "failed_count": len(failed) + len(orphan_traces),
        "failed": failed,
        "orphan_traces": orphan_traces,
    }


def collect_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seen_run_dirs: set[str] = set()
    for result_path in sorted(root.rglob("_result.txt")):
        run_dir = result_path.parent
        condition = condition_from_path(result_path)
        run = parse_result_txt(result_path)
        run.update(
            {
                "condition": condition,
                "run_dir": str(run_dir),
                "eval_log": str(run_dir / "eval_log.txt") if (run_dir / "eval_log.txt").exists() else "",
                "episodes": parse_eval_log(run_dir / "eval_log.txt") if (run_dir / "eval_log.txt").exists() else [],
            }
        )
        enrich_run_with_traces(run, run_dir)
        runs.append(run)
        seen_run_dirs.add(str(run_dir.resolve()))
    for worker_log in sorted(root.rglob("worker_*.log")):
        run = parse_worker_log(worker_log)
        result_path_text = str(run.get("result_path", "") or "")
        if result_path_text:
            result_path = Path(result_path_text)
            if not result_path.is_absolute():
                result_path = REPO_ROOT / result_path
            result_run_dir = result_path.parent
            if str(result_run_dir.resolve()) in seen_run_dirs:
                continue
            if result_path.exists():
                run.update(parse_result_txt(result_path))
            eval_log = result_run_dir / "eval_log.txt"
            if eval_log.exists() and not run.get("episodes"):
                run["episodes"] = parse_eval_log(eval_log)
                run["eval_log"] = str(eval_log)
            enrich_run_with_traces(run, result_run_dir)
        else:
            result_run_dir = Path(str(run.get("run_dir", worker_log.parent)))
            if not result_run_dir.is_absolute():
                result_run_dir = REPO_ROOT / result_run_dir
            eval_log = result_run_dir / "eval_log.txt"
            if eval_log.exists() and not run.get("episodes"):
                run["episodes"] = parse_eval_log(eval_log)
                run["eval_log"] = str(eval_log)
            enrich_run_with_traces(run, result_run_dir)
        if run.get("episodes"):
            runs.append(run)
            seen_run_dirs.add(str(result_run_dir.resolve()))
    return runs


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    cross_episode_diagnostics = apply_cross_episode_formal_gates(runs)
    by_condition: dict[str, dict[str, Any]] = {}
    for run in runs:
        condition = str(run.get("condition", "unknown"))
        bucket = by_condition.setdefault(
            condition,
            {
                "condition": condition,
                "num_runs": 0,
                "num_episodes": 0,
                "num_success": 0,
                "mean_reward": 0.0,
                "mean_steps": 0.0,
                "num_capability_eligible": 0,
                "num_formal_base_eligible": 0,
                "num_valid_success": 0,
                "capability_reward_sum": 0.0,
                "capability_steps_sum": 0.0,
                "validity_counts": {label: 0 for label in EPISODE_VALIDITY_LABELS},
                "trace_audit_validity_counts": {label: 0 for label in EPISODE_VALIDITY_LABELS},
                "runtime_validity_contract_counts": {"matched": 0, "missing": 0, "mismatch": 0},
                "formal_metadata_counts": {"complete": 0, "incomplete": 0},
                "formal_eligibility_counts": {"eligible": 0, "excluded": 0},
                "formal_exclusion_counts": {},
                "duplicate_identity_keys": [],
                "duplicate_identity_episode_count": 0,
                "runtime_tree_sha256_counts": {},
                "mixed_runtime_tree_sha256": False,
                "formal_protocol_version_counts": {},
                "mixed_formal_protocol_version": False,
                "failure_reasons": {},
                "failure_stages": {},
                "trace_metric_episode_count": 0,
                "num_acceptance_passed": 0,
                "num_strict_passed": 0,
                "num_without_vla_requests": 0,
                "num_ordered_phase_passed": 0,
                "gpt_call_count_inference_valid_episode_count": 0,
                "mean_gpt_endpoint_calls_inferred": 0.0,
                "mean_tool_calls": 0.0,
                "mean_wall_duration_sec": 0.0,
            },
        )
        bucket["num_runs"] += 1
        episodes = run.get("episodes", [])
        if isinstance(episodes, list) and episodes:
            for episode in episodes:
                bucket["num_episodes"] += 1
                bucket["num_success"] += int(bool(episode.get("success", False)))
                bucket["mean_reward"] += float(episode.get("reward", 0.0))
                bucket["mean_steps"] += float(episode.get("steps", 0.0))
                trace_metrics = episode.get("trace_metrics", {})
                if not isinstance(trace_metrics, dict):
                    trace_metrics = {}
                validity = episode.get("episode_validity", {})
                if not isinstance(validity, dict):
                    validity = {}
                validity_label = str(validity.get("label", "") or "")
                if validity_label not in EPISODE_VALIDITY_LABELS:
                    validity_label = "incomplete_or_corrupt"
                bucket["validity_counts"][validity_label] += 1
                trace_audit_label = str(validity.get("trace_audit_label", "") or "")
                if trace_audit_label not in EPISODE_VALIDITY_LABELS:
                    trace_audit_label = validity_label
                bucket["trace_audit_validity_counts"][trace_audit_label] += 1

                runtime_label_present = bool(validity.get("runtime_label_present", False))
                runtime_label_matches = bool(validity.get("runtime_label_matches_trace_audit", False))
                if not runtime_label_present:
                    runtime_contract_state = "missing"
                elif runtime_label_matches:
                    runtime_contract_state = "matched"
                else:
                    runtime_contract_state = "mismatch"
                bucket["runtime_validity_contract_counts"][runtime_contract_state] += 1

                formal_metadata = trace_metrics.get("formal_metadata", {})
                if not isinstance(formal_metadata, dict):
                    formal_metadata = {}
                formal_metadata_complete = bool(formal_metadata.get("complete", False))
                bucket["formal_metadata_counts"][
                    "complete" if formal_metadata_complete else "incomplete"
                ] += 1
                formal_eligible = bool(
                    trace_metrics
                    and validity_label in {"valid_success", "valid_task_failure"}
                    and bool(validity.get("benchmark_denominator_eligible", False))
                    and runtime_contract_state == "matched"
                    and formal_metadata_complete
                    and bool(trace_metrics.get("formal_cross_episode_eligible", False))
                )
                bucket["num_formal_base_eligible"] += int(
                    bool(trace_metrics.get("formal_base_eligible", False))
                )
                if formal_eligible:
                    bucket["formal_eligibility_counts"]["eligible"] += 1
                    bucket["num_capability_eligible"] += 1
                    bucket["num_valid_success"] += int(validity_label == "valid_success")
                    bucket["capability_reward_sum"] += float(episode.get("reward", 0.0))
                    bucket["capability_steps_sum"] += float(episode.get("steps", 0.0))
                else:
                    bucket["formal_eligibility_counts"]["excluded"] += 1
                    exclusion_reasons = [
                        str(item)
                        for item in validity.get("eligibility_exclusion_reasons", []) or []
                        if str(item).strip()
                    ]
                    if not trace_metrics:
                        exclusion_reasons.append("missing_trace")
                    if runtime_contract_state == "missing":
                        exclusion_reasons.append("runtime_validity_label_missing_or_invalid")
                    elif runtime_contract_state == "mismatch":
                        exclusion_reasons.append("runtime_validity_label_mismatch")
                    if not formal_metadata_complete:
                        metadata_errors = [
                            str(item)
                            for item in formal_metadata.get("errors", []) or []
                            if str(item).strip()
                        ]
                        exclusion_reasons.extend(
                            f"formal_metadata:{item}" for item in metadata_errors
                        )
                        if not metadata_errors:
                            exclusion_reasons.append("formal_metadata:missing_or_incomplete")
                    if trace_audit_label not in {"valid_success", "valid_task_failure"}:
                        exclusion_reasons.append(f"trace_audit_label:{trace_audit_label}")
                    for exclusion_reason in dict.fromkeys(exclusion_reasons):
                        bucket["formal_exclusion_counts"][exclusion_reason] = (
                            bucket["formal_exclusion_counts"].get(exclusion_reason, 0) + 1
                        )
                reason = str(episode.get("failure_reason", "") or "none")
                bucket["failure_reasons"][reason] = bucket["failure_reasons"].get(reason, 0) + 1
                if isinstance(trace_metrics, dict) and trace_metrics:
                    bucket["trace_metric_episode_count"] += 1
                    bucket["num_acceptance_passed"] += int(
                        bool(episode.get("success", False))
                        and bool(trace_metrics.get("formal_acceptance_passed", False))
                    )
                    bucket["num_strict_passed"] += int(bool(trace_metrics.get("strict_passed", False)))
                    bucket["num_without_vla_requests"] += int(int(trace_metrics.get("vla_request_count", 0) or 0) == 0)
                    phase_order = trace_metrics.get("phase_order", {})
                    bucket["num_ordered_phase_passed"] += int(
                        isinstance(phase_order, dict) and bool(phase_order.get("passed", False))
                    )
                    gpt_calls = trace_metrics.get("gpt_endpoint_call_count_inferred")
                    if isinstance(gpt_calls, int):
                        bucket["gpt_call_count_inference_valid_episode_count"] += 1
                        bucket["mean_gpt_endpoint_calls_inferred"] += float(gpt_calls)
                    bucket["mean_tool_calls"] += float(trace_metrics.get("tool_call_count", 0) or 0)
                    bucket["mean_wall_duration_sec"] += float(trace_metrics.get("wall_duration_sec", 0.0) or 0.0)
                    failure_stage = str(trace_metrics.get("failure_stage", "") or "")
                    if not failure_stage and not bool(episode.get("success", False)):
                        failure_stage = "eval_result_failure"
                else:
                    failure_stage = "missing_trace"
                failure_stage = failure_stage or "none"
                bucket["failure_stages"][failure_stage] = bucket["failure_stages"].get(failure_stage, 0) + 1
        else:
            success_rate = run.get("success_rate")
            mean_reward = run.get("mean_reward")
            if success_rate is not None:
                bucket["num_episodes"] += 1
                bucket["num_success"] += int(float(success_rate) > 0.0)
                bucket["validity_counts"]["incomplete_or_corrupt"] += 1
                bucket["trace_audit_validity_counts"]["incomplete_or_corrupt"] += 1
                bucket["runtime_validity_contract_counts"]["missing"] += 1
                bucket["formal_metadata_counts"]["incomplete"] += 1
                bucket["formal_eligibility_counts"]["excluded"] += 1
                bucket["formal_exclusion_counts"]["missing_episode_result"] = (
                    bucket["formal_exclusion_counts"].get("missing_episode_result", 0) + 1
                )
                bucket["failure_stages"]["missing_episode_result"] = (
                    bucket["failure_stages"].get("missing_episode_result", 0) + 1
                )
            if mean_reward is not None:
                bucket["mean_reward"] += float(mean_reward)
    for bucket in by_condition.values():
        cross_diagnostics = cross_episode_diagnostics.get(str(bucket["condition"]), {})
        bucket["duplicate_identity_keys"] = list(
            cross_diagnostics.get("duplicate_identity_keys", []) or []
        )
        bucket["duplicate_identity_episode_count"] = int(
            cross_diagnostics.get("duplicate_identity_episode_count", 0) or 0
        )
        bucket["runtime_tree_sha256_counts"] = dict(
            cross_diagnostics.get("runtime_tree_sha256_counts", {}) or {}
        )
        bucket["mixed_runtime_tree_sha256"] = bool(
            cross_diagnostics.get("mixed_runtime_tree_sha256", False)
        )
        bucket["formal_protocol_version_counts"] = dict(
            cross_diagnostics.get("formal_protocol_version_counts", {}) or {}
        )
        bucket["mixed_formal_protocol_version"] = bool(
            cross_diagnostics.get("mixed_formal_protocol_version", False)
        )
        num_episodes = max(1, int(bucket["num_episodes"]))
        bucket["attempted_success_rate"] = float(bucket["num_success"]) / float(num_episodes)
        bucket["attempted_mean_reward"] = float(bucket["mean_reward"]) / float(num_episodes)
        bucket["attempted_mean_steps"] = (
            float(bucket["mean_steps"]) / float(num_episodes) if bucket["mean_steps"] else 0.0
        )
        capability_denominator = int(bucket["num_capability_eligible"])
        if capability_denominator > 0:
            bucket["success_rate"] = float(bucket["num_valid_success"]) / float(capability_denominator)
            bucket["mean_reward"] = float(bucket.pop("capability_reward_sum")) / float(capability_denominator)
            bucket["mean_steps"] = float(bucket.pop("capability_steps_sum")) / float(capability_denominator)
        else:
            bucket["success_rate"] = None
            bucket["mean_reward"] = None
            bucket["mean_steps"] = None
            bucket.pop("capability_reward_sum")
            bucket.pop("capability_steps_sum")
        bucket["formal_aggregate_formed"] = bool(
            capability_denominator > 0
            and not bucket["mixed_runtime_tree_sha256"]
            and not bucket["mixed_formal_protocol_version"]
        )
        if bucket["mixed_runtime_tree_sha256"]:
            bucket["formal_aggregate_exclusion_reason"] = "mixed_runtime_tree_sha256"
        elif bucket["mixed_formal_protocol_version"]:
            bucket["formal_aggregate_exclusion_reason"] = "mixed_formal_protocol_version"
        elif capability_denominator == 0:
            bucket["formal_aggregate_exclusion_reason"] = "no_formal_eligible_episodes"
        else:
            bucket["formal_aggregate_exclusion_reason"] = ""
        trace_metric_episode_count = int(bucket["trace_metric_episode_count"])
        trace_denominator = max(1, trace_metric_episode_count)
        bucket["trace_coverage_rate"] = float(trace_metric_episode_count) / float(num_episodes)
        bucket["acceptance_rate"] = float(bucket["num_acceptance_passed"]) / float(num_episodes)
        bucket["strict_pass_rate"] = float(bucket["num_strict_passed"]) / float(num_episodes)
        bucket["no_vla_rate"] = float(bucket["num_without_vla_requests"]) / float(num_episodes)
        bucket["ordered_phase_pass_rate"] = float(bucket["num_ordered_phase_passed"]) / float(num_episodes)
        gpt_call_denominator = max(1, int(bucket["gpt_call_count_inference_valid_episode_count"]))
        bucket["mean_gpt_endpoint_calls_inferred"] = (
            float(bucket["mean_gpt_endpoint_calls_inferred"]) / float(gpt_call_denominator)
        )
        bucket["mean_tool_calls"] = float(bucket["mean_tool_calls"]) / float(trace_denominator)
        bucket["mean_wall_duration_sec"] = float(bucket["mean_wall_duration_sec"]) / float(trace_denominator)
    return by_condition


def write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    apply_cross_episode_formal_gates(runs)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run in runs:
        episodes = run.get("episodes", [])
        if not isinstance(episodes, list) or not episodes:
            rows.append(
                {
                    "condition": run.get("condition", ""),
                    "run_dir": run.get("run_dir", ""),
                    "worker_log": run.get("worker_log", ""),
                    "episode_id": "",
                    "seed": "",
                    "success": "",
                    "result": "",
                    "reward": run.get("mean_reward", ""),
                    "steps": "",
                    "failure_reason": "",
                    "episode_validity": "incomplete_or_corrupt",
                    "benchmark_denominator_eligible": False,
                    "task_success_numerator": False,
                    "trace_audit_label": "incomplete_or_corrupt",
                    "runtime_validity_label": "",
                    "runtime_validity_contract_state": "missing",
                    "runtime_validity_label_matches": "",
                    "formal_metadata_complete": False,
                    "formal_eligibility_reason": "missing_episode_result",
                    "formal_exclusion_reasons": "missing_episode_result",
                    "formal_metadata_errors": "missing_episode_result",
                    "perception_condition": "",
                    "pure_tool_control": "",
                    "max_semantic_rounds_per_active_subtask": "",
                    "max_control_turns": "",
                    "max_no_progress_control_turns": "",
                    "backend_error_budget": "",
                    "formal_protocol_start": "",
                    "formal_protocol_end": "",
                    "formal_protocol_version": "",
                    "formal_protocol_version_source": "",
                    "episode_start_formal_protocol_version": "",
                    "episode_end_formal_protocol_version": "",
                    "runtime_provenance_start_end_matches": "",
                    "runtime_provenance_archive_exists": "",
                    "runtime_provenance_manifest_sha256_matches": "",
                    "malformed_jsonl_line_count": "",
                    "trace": "",
                    "acceptance_passed": "",
                    "strict_passed": "",
                    "episode_end_present": "",
                    "episode_end_success": "",
                    "ordered_phase_passed": "",
                    "vla_request_count": "",
                    "gpt_endpoint_call_count_inferred": "",
                    "tool_call_count": "",
                    "wall_duration_sec": "",
                    "failure_stage": "missing_episode_result",
                }
            )
            continue
        for episode in episodes:
            trace_metrics = episode.get("trace_metrics", {})
            if not isinstance(trace_metrics, dict):
                trace_metrics = {}
            validity = episode.get("episode_validity", {})
            if not isinstance(validity, dict):
                validity = {}
            formal_metadata = trace_metrics.get("formal_metadata", {})
            if not isinstance(formal_metadata, dict):
                formal_metadata = {}
            runtime_label_present = bool(validity.get("runtime_label_present", False))
            runtime_label_matches = bool(validity.get("runtime_label_matches_trace_audit", False))
            runtime_contract_state = (
                "missing"
                if not runtime_label_present
                else "matched"
                if runtime_label_matches
                else "mismatch"
            )
            phase_order = trace_metrics.get("phase_order", {})
            episode_acceptance_passed: bool | str = ""
            failure_stage = str(trace_metrics.get("failure_stage", "") or "")
            if trace_metrics:
                episode_acceptance_passed = bool(episode.get("success", False)) and bool(
                    trace_metrics.get("formal_acceptance_passed", False)
                )
                if not failure_stage and not bool(episode.get("success", False)):
                    failure_stage = "eval_result_failure"
            else:
                failure_stage = "missing_trace"
            rows.append(
                {
                    "condition": run.get("condition", ""),
                    "run_dir": run.get("run_dir", ""),
                    "worker_log": episode.get("worker_log", run.get("worker_log", "")),
                    "episode_id": episode.get("episode_id", ""),
                    "seed": episode.get("seed", ""),
                    "success": episode.get("success", ""),
                    "result": episode.get("result", ""),
                    "reward": episode.get("reward", ""),
                    "steps": episode.get("steps", ""),
                    "failure_reason": episode.get("failure_reason", ""),
                    "episode_validity": validity.get("label", "incomplete_or_corrupt"),
                    "benchmark_denominator_eligible": validity.get("benchmark_denominator_eligible", False),
                    "task_success_numerator": validity.get("task_success_numerator", False),
                    "trace_audit_label": validity.get("trace_audit_label", ""),
                    "runtime_validity_label": validity.get("runtime_label", ""),
                    "runtime_validity_contract_state": runtime_contract_state,
                    "runtime_validity_label_matches": validity.get("runtime_label_matches_trace_audit", ""),
                    "formal_metadata_complete": formal_metadata.get("complete", False),
                    "formal_eligibility_reason": validity.get("formal_eligibility_reason", ""),
                    "formal_exclusion_reasons": " | ".join(
                        str(item) for item in validity.get("eligibility_exclusion_reasons", []) or []
                    ),
                    "formal_metadata_errors": " | ".join(
                        str(item) for item in formal_metadata.get("errors", []) or []
                    ),
                    "perception_condition": trace_metrics.get("perception_condition", ""),
                    "pure_tool_control": trace_metrics.get("pure_tool_control", ""),
                    "max_semantic_rounds_per_active_subtask": trace_metrics.get(
                        "max_semantic_rounds_per_active_subtask", ""
                    ),
                    "max_control_turns": trace_metrics.get("max_control_turns", ""),
                    "max_no_progress_control_turns": trace_metrics.get(
                        "max_no_progress_control_turns", ""
                    ),
                    "backend_error_budget": trace_metrics.get("backend_error_budget", ""),
                    "formal_protocol_start": formal_metadata.get("formal_protocol_start", False),
                    "formal_protocol_end": formal_metadata.get("formal_protocol_end", False),
                    "formal_protocol_version": formal_metadata.get(
                        "formal_protocol_version", ""
                    ),
                    "formal_protocol_version_source": formal_metadata.get(
                        "formal_protocol_version_source", ""
                    ),
                    "episode_start_formal_protocol_version": formal_metadata.get(
                        "episode_start_formal_protocol_version", ""
                    ),
                    "episode_end_formal_protocol_version": formal_metadata.get(
                        "episode_end_formal_protocol_version", ""
                    ),
                    "runtime_provenance_start_end_matches": formal_metadata.get(
                        "runtime_provenance_start_end_matches", False
                    ),
                    "runtime_provenance_archive_exists": formal_metadata.get(
                        "runtime_provenance_archive_exists", False
                    ),
                    "runtime_provenance_manifest_sha256_matches": formal_metadata.get(
                        "runtime_provenance_manifest_sha256_matches", False
                    ),
                    "malformed_jsonl_line_count": trace_metrics.get("malformed_jsonl_line_count", ""),
                    "trace": trace_metrics.get("trace", ""),
                    "acceptance_passed": episode_acceptance_passed,
                    "strict_passed": trace_metrics.get("strict_passed", ""),
                    "episode_end_present": trace_metrics.get("episode_end_present", ""),
                    "episode_end_success": trace_metrics.get("episode_end_success", ""),
                    "ordered_phase_passed": phase_order.get("passed", "") if isinstance(phase_order, dict) else "",
                    "vla_request_count": trace_metrics.get("vla_request_count", ""),
                    "gpt_endpoint_call_count_inferred": trace_metrics.get("gpt_endpoint_call_count_inferred", ""),
                    "tool_call_count": trace_metrics.get("tool_call_count", ""),
                    "wall_duration_sec": trace_metrics.get("wall_duration_sec", ""),
                    "failure_stage": failure_stage,
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition",
                "run_dir",
                "worker_log",
                "episode_id",
                "seed",
                "success",
                "result",
                "reward",
                "steps",
                "failure_reason",
                "episode_validity",
                "benchmark_denominator_eligible",
                "task_success_numerator",
                "trace_audit_label",
                "runtime_validity_label",
                "runtime_validity_contract_state",
                "runtime_validity_label_matches",
                "formal_metadata_complete",
                "formal_eligibility_reason",
                "formal_exclusion_reasons",
                "formal_metadata_errors",
                "perception_condition",
                "pure_tool_control",
                "max_semantic_rounds_per_active_subtask",
                "max_control_turns",
                "max_no_progress_control_turns",
                "backend_error_budget",
                "formal_protocol_start",
                "formal_protocol_end",
                "formal_protocol_version",
                "formal_protocol_version_source",
                "episode_start_formal_protocol_version",
                "episode_end_formal_protocol_version",
                "runtime_provenance_start_end_matches",
                "runtime_provenance_archive_exists",
                "runtime_provenance_manifest_sha256_matches",
                "malformed_jsonl_line_count",
                "trace",
                "acceptance_passed",
                "strict_passed",
                "episode_end_present",
                "episode_end_success",
                "ordered_phase_passed",
                "vla_request_count",
                "gpt_endpoint_call_count_inferred",
                "tool_call_count",
                "wall_duration_sec",
                "failure_stage",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.input.resolve()
    runs = collect_runs(root)
    acceptance = build_acceptance_summary(runs)
    summary = {
        "input": str(root),
        "num_runs": len(runs),
        "aggregate": aggregate(runs),
        "acceptance": acceptance,
        "runs": runs,
    }
    output_path = args.output or (root / "object_info_ablation_summary.json")
    csv_path = args.csv or (root / "object_info_ablation_episodes.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, runs)
    print(
        json.dumps(
            {"summary": str(output_path), "csv": str(csv_path), "aggregate": summary["aggregate"], "acceptance": acceptance},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_acceptance and not acceptance["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
