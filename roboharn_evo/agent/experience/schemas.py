from __future__ import annotations

import hashlib
import json
from typing import Any


OOD_SCENARIOS = {
    "none",
    "object_not_visible",
    "motion_blocked",
    "grasp_lost",
    "scene_drift_detected",
    "requires_replan",
}

OUTCOME_TAXONOMY = {
    "detect_only",
    "recovery_success_retry",
    "recovery_success_replan",
    "recovery_failed",
    "replan_without_tools",
    "aborted",
    "false_positive",
    "false_negative",
    "incomplete_trace",
}

ACCEPTED_LESSON_STATUS = "accepted"
LESSON_STATUSES = {"candidate", ACCEPTED_LESSON_STATUS, "deprecated"}

SUCCESS_OUTCOMES = {
    "recovery_success_retry",
    "recovery_success_replan",
}

FAILURE_OUTCOMES = {
    "recovery_failed",
    "aborted",
    "false_positive",
    "false_negative",
}

FORBIDDEN_RETRIEVAL_FIELDS = {
    "benchmark_label",
    "gold_ood_scenario",
    "target_error_category",
    "error_category",
    "s_stage",
    "d_stage",
    "annotation",
    "annotations",
    "label",
    "labels",
}

REQUIRED_RECOVERY_TRIAL_FIELDS = {
    "trial_id",
    "episode_id",
    "seed",
    "global_task",
    "subtask",
    "step",
    "monitor_signal",
    "OOD_scenario",
    "observation_summary",
    "available_tools",
    "recovery_workflow",
    "tool_calls",
    "tool_results",
    "post_recovery_intent",
    "post_recovery_decision",
    "outcome",
    "budget_used",
    "failure_reason",
    "trace_file",
    "event_window",
}

OPTIONAL_RECOVERY_TRIAL_FIELD_TYPES = {
    "post_window_outcome": str,
    "secondary_signal": str,
    "retry_success_after_recovery": bool,
    "final_episode_status": str,
    "outcome_evidence": dict,
    "robot_state": dict,
}

OPTIONAL_RETRIEVAL_INDEX_FIELD_TYPES = {
    "support_count": int,
    "opposing_count": int,
    "last_validated_at": str,
    "confidence": str,
    "avoid_pattern": bool,
    "failure_penalty": int,
    "grounding_summary": dict,
}


class ExperienceValidationError(ValueError):
    pass


def normalize_ood_scenario(value: Any) -> str:
    scenario = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return scenario if scenario in OOD_SCENARIOS else "none"


def stable_experience_id(prefix: str, payload: dict[str, Any], *, length: int = 10) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:length]
    cleaned_prefix = "_".join(part for part in str(prefix).strip().lower().replace("-", "_").split("_") if part)
    return f"{cleaned_prefix}_{digest}" if cleaned_prefix else digest


def find_forbidden_fields(payload: Any, *, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_str = str(key)
            current_path = f"{path}.{key_str}" if path else key_str
            if key_str in FORBIDDEN_RETRIEVAL_FIELDS:
                found.append(current_path)
            found.extend(find_forbidden_fields(value, path=current_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(find_forbidden_fields(value, path=f"{path}[{index}]"))
    return found


def validate_recovery_trial(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(field for field in REQUIRED_RECOVERY_TRIAL_FIELDS if field not in record)
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")

    scenario = normalize_ood_scenario(record.get("OOD_scenario", ""))
    if scenario != record.get("OOD_scenario"):
        errors.append(f"invalid OOD_scenario: {record.get('OOD_scenario')!r}")

    outcome = str(record.get("outcome", "")).strip()
    if outcome and outcome not in OUTCOME_TAXONOMY:
        errors.append(f"invalid outcome: {outcome!r}")

    if "available_tools" in record and not isinstance(record["available_tools"], list):
        errors.append("available_tools must be a list")
    if "tool_calls" in record and not isinstance(record["tool_calls"], list):
        errors.append("tool_calls must be a list")
    if "tool_results" in record and not isinstance(record["tool_results"], list):
        errors.append("tool_results must be a list")
    if "event_window" in record and not isinstance(record["event_window"], dict):
        errors.append("event_window must be a dict")
    if "budget_used" in record and not isinstance(record["budget_used"], dict):
        errors.append("budget_used must be a dict")
    for field, expected_type in OPTIONAL_RECOVERY_TRIAL_FIELD_TYPES.items():
        if field in record and not isinstance(record[field], expected_type):
            errors.append(f"{field} must be a {expected_type.__name__}")

    return errors


def validate_retrieval_index_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field, expected_type in OPTIONAL_RETRIEVAL_INDEX_FIELD_TYPES.items():
        if field in record and not isinstance(record[field], expected_type):
            errors.append(f"{field} must be a {expected_type.__name__}")
    confidence = str(record.get("confidence", "")).strip()
    if confidence and confidence not in {"low", "medium", "high"}:
        errors.append(f"invalid confidence: {confidence!r}")
    return errors


def validate_retrieval_payload(payload: dict[str, Any]) -> None:
    forbidden = find_forbidden_fields(payload)
    if forbidden:
        raise ExperienceValidationError(
            "retrieval payload contains forbidden benchmark/annotation fields: "
            + ", ".join(sorted(forbidden))
        )
