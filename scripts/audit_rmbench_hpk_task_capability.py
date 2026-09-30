from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "roboharn_evo/hpk/rmbench_task_capability_input/v1"
REPORT_SCHEMA = "roboharn_evo/hpk/rmbench_task_capability_report/v1"


class TaskCapabilityError(ValueError):
    """Raised when task capability evidence is incomplete or ambiguous."""


def _exact(value: Mapping[str, Any], keys: set[str], path: str) -> None:
    if set(value) != keys:
        raise TaskCapabilityError(f"{path} fields mismatch")


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise TaskCapabilityError(f"{path} must be a string")
    result = " ".join(value.split())
    if not result:
        raise TaskCapabilityError(f"{path} must be non-empty")
    return result


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TaskCapabilityError(f"{path} must be an integer >= {minimum}")
    return value


def _probability(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskCapabilityError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise TaskCapabilityError(f"{path} must be between 0 and 1")
    return result


def _strings(value: Any, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TaskCapabilityError(f"{path} must be an array")
    result = tuple(_text(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        raise TaskCapabilityError(f"{path} must not contain duplicates")
    return result


def audit_tasks(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return an auditable D0 gate report without choosing tasks implicitly."""

    _exact(raw, {"schema", "criteria", "required_core_coverage", "tasks"}, "input")
    if raw["schema"] not in {INPUT_SCHEMA, "tcm/afk/rmbench_task_capability_input/v1"}:
        raise TaskCapabilityError("input.schema is unsupported")
    criteria = raw["criteria"]
    if not isinstance(criteria, Mapping):
        raise TaskCapabilityError("criteria must be an object")
    _exact(
        criteria,
        {
            "minimum_demonstrations",
            "minimum_meaningful_subtasks",
            "minimum_action_families",
            "minimum_baseline_trials",
            "minimum_baseline_success_rate",
            "maximum_baseline_success_rate",
            "required_core_tasks",
            "required_reserve_tasks",
        },
        "criteria",
    )
    minimum_demonstrations = _integer(
        criteria["minimum_demonstrations"], "criteria.minimum_demonstrations", minimum=1
    )
    minimum_subtasks = _integer(
        criteria["minimum_meaningful_subtasks"],
        "criteria.minimum_meaningful_subtasks",
        minimum=1,
    )
    minimum_action_families = _integer(
        criteria["minimum_action_families"],
        "criteria.minimum_action_families",
        minimum=1,
    )
    minimum_baseline_trials = _integer(
        criteria["minimum_baseline_trials"],
        "criteria.minimum_baseline_trials",
        minimum=1,
    )
    minimum_success_rate = _probability(
        criteria["minimum_baseline_success_rate"],
        "criteria.minimum_baseline_success_rate",
    )
    maximum_success_rate = _probability(
        criteria["maximum_baseline_success_rate"],
        "criteria.maximum_baseline_success_rate",
    )
    if minimum_success_rate >= maximum_success_rate:
        raise TaskCapabilityError("baseline success-rate interval must be non-empty")
    required_core_tasks = _integer(
        criteria["required_core_tasks"], "criteria.required_core_tasks", minimum=1
    )
    required_reserve_tasks = _integer(
        criteria["required_reserve_tasks"], "criteria.required_reserve_tasks", minimum=0
    )
    required_coverage = _strings(
        raw["required_core_coverage"], "required_core_coverage"
    )
    tasks = raw["tasks"]
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence) or not tasks:
        raise TaskCapabilityError("tasks must be a non-empty array")

    reports: list[dict[str, Any]] = []
    names: list[str] = []
    for index, item in enumerate(tasks):
        path = f"tasks[{index}]"
        if not isinstance(item, Mapping):
            raise TaskCapabilityError(f"{path} must be an object")
        _exact(
            item,
            {
                "task",
                "proposed_role",
                "demonstrations",
                "meaningful_subtasks",
                "action_families",
                "coverage_tags",
                "distinct_geometric_strategies",
                "effect_verifier_observes_key_postconditions",
                "action_knowledge_changes_eligible_candidate",
                "baseline_trials",
                "baseline_successes",
                "evidence",
            },
            path,
        )
        task = _text(item["task"], f"{path}.task")
        names.append(task)
        role = _text(item["proposed_role"], f"{path}.proposed_role")
        if role not in {"core", "reserve", "none"}:
            raise TaskCapabilityError(f"{path}.proposed_role is unsupported")
        demonstrations = _integer(item["demonstrations"], f"{path}.demonstrations")
        meaningful_subtasks = _integer(
            item["meaningful_subtasks"], f"{path}.meaningful_subtasks"
        )
        action_families = _strings(item["action_families"], f"{path}.action_families")
        coverage_tags = _strings(item["coverage_tags"], f"{path}.coverage_tags")
        evidence = _strings(item["evidence"], f"{path}.evidence")
        for flag in (
            "distinct_geometric_strategies",
            "effect_verifier_observes_key_postconditions",
            "action_knowledge_changes_eligible_candidate",
        ):
            if not isinstance(item[flag], bool):
                raise TaskCapabilityError(f"{path}.{flag} must be a boolean")
        baseline_trials = _integer(item["baseline_trials"], f"{path}.baseline_trials")
        baseline_successes = _integer(
            item["baseline_successes"], f"{path}.baseline_successes"
        )
        if baseline_successes > baseline_trials:
            raise TaskCapabilityError(f"{path}.baseline_successes exceeds trials")
        baseline_rate = (
            None if baseline_trials == 0 else baseline_successes / baseline_trials
        )
        checks = {
            "demonstrations": demonstrations >= minimum_demonstrations,
            "meaningful_subtasks": meaningful_subtasks >= minimum_subtasks,
            "action_families": len(action_families) >= minimum_action_families,
            "distinct_geometric_strategies": item["distinct_geometric_strategies"],
            "effect_verifier": item["effect_verifier_observes_key_postconditions"],
            "candidate_behavior_change": item[
                "action_knowledge_changes_eligible_candidate"
            ],
            "baseline_trials": baseline_trials >= minimum_baseline_trials,
            "baseline_not_saturated": (
                baseline_rate is not None
                and minimum_success_rate <= baseline_rate <= maximum_success_rate
            ),
            "evidence_present": bool(evidence),
        }
        reports.append(
            {
                "task": task,
                "proposed_role": role,
                "ready": all(checks.values()),
                "baseline_trials": baseline_trials,
                "baseline_successes": baseline_successes,
                "baseline_success_rate": baseline_rate,
                "action_families": list(action_families),
                "coverage_tags": list(coverage_tags),
                "checks": checks,
                "missing": [name for name, passed in checks.items() if not passed],
                "evidence": list(evidence),
            }
        )
    if len(names) != len(set(names)):
        raise TaskCapabilityError("task names must be unique")

    ready_core = [
        item for item in reports if item["ready"] and item["proposed_role"] == "core"
    ]
    ready_reserve = [
        item for item in reports if item["ready"] and item["proposed_role"] == "reserve"
    ]
    observed_coverage = {tag for item in ready_core for tag in item["coverage_tags"]}
    missing_coverage = sorted(set(required_coverage) - observed_coverage)
    gate_checks = {
        "core_task_count": len(ready_core) >= required_core_tasks,
        "reserve_task_count": len(ready_reserve) >= required_reserve_tasks,
        "core_coverage": not missing_coverage,
    }
    return {
        "schema": REPORT_SCHEMA,
        "gate_passed": all(gate_checks.values()),
        "gate_checks": gate_checks,
        "ready_core_tasks": [item["task"] for item in ready_core],
        "ready_reserve_tasks": [item["task"] for item in ready_reserve],
        "missing_core_coverage": missing_coverage,
        "tasks": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TaskCapabilityError("input must be an object")
    report = audit_tasks(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
