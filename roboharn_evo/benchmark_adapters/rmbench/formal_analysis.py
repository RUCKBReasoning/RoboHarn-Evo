"""Aggregate preregistered RMBench Q1 cells into paper-facing tables."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from roboharn_evo.agent.hpk.compatibility import normalize_hpk_event_name


TASK_ORDER = (
    "rearrange_blocks",
    "swap_blocks",
    "press_button",
    "place_block_mat",
    "put_back_block",
    "cover_blocks",
)
METHOD_ORDER = ("off", "flat", "task", "action", "full")
CATEGORY = {
    "rearrange_blocks": "task_dominant",
    "swap_blocks": "task_dominant",
    "press_button": "action_dominant",
    "place_block_mat": "action_dominant",
    "put_back_block": "coupled",
    "cover_blocks": "coupled",
}


class FormalQ1AnalysisError(RuntimeError):
    """Formal Q1 artifacts are missing or internally inconsistent."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalQ1AnalysisError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise FormalQ1AnalysisError(f"{path} must contain one JSON object")
    return value


def _trace(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FormalQ1AnalysisError(
                f"invalid trace JSON at {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise FormalQ1AnalysisError(
                f"trace record is not an object at {path}:{line_number}"
            )
        records.append(value)
    return records


def _bool_count(records: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(record.get(field) is True for record in records)


def mechanism_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Extract only directly trace-supported mechanism counts."""

    by_event: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_event[normalize_hpk_event_name(str(record.get("event", "")))].append(record)
    task = by_event["hpk_v3_subtask_usage"]
    action = by_event["hpk_v3_action_retrieval"]
    feasibility = by_event["hpk_v31_runtime_feasibility"]
    evidence = by_event["hpk_v3_action_evidence"]
    verdicts = Counter(
        str(record.get("verdict", record.get("effect_verdict", ""))).lower()
        for record in evidence
    )
    unresolved = sum(
        str(record.get("status", "")).lower() in {"unresolved", "violated"}
        or str(record.get("realization_status", "")).lower()
        in {"unresolved", "violated"}
        for record in feasibility
    )
    return {
        "task_retrievals": len(task),
        "task_knowledge_adopted": _bool_count(task, "knowledge_adopted"),
        "task_behavior_changed": _bool_count(task, "behavior_changed"),
        "action_retrievals": len(action),
        "action_knowledge_adopted": _bool_count(action, "knowledge_adopted"),
        "action_behavior_changed": _bool_count(action, "behavior_changed"),
        "goal_feasibility_checks": len(feasibility),
        "goal_feasibility_unresolved": unresolved,
        "action_evidence_support": verdicts["support"],
        "action_evidence_oppose": verdicts["oppose"],
        "action_evidence_unverified": verdicts["unverified"],
        "candidate_selections": len(by_event["operation_candidate_selected"]),
        "effect_verifications": len(by_event["action_effect_verification"]),
        "completed_subtasks": len(by_event["subtask_transition_commit"]),
    }


def wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    centre = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def exact_mcnemar(first: Sequence[bool], second: Sequence[bool]) -> dict[str, Any]:
    if len(first) != len(second):
        raise FormalQ1AnalysisError("paired outcomes must have equal length")
    gained = sum(
        (not left) and right for left, right in zip(first, second, strict=True)
    )
    regressed = sum(
        left and (not right) for left, right in zip(first, second, strict=True)
    )
    discordant = gained + regressed
    if discordant == 0:
        probability = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(0, min(gained, regressed) + 1)
        ) / (2**discordant)
        probability = min(1.0, 2.0 * tail)
    return {
        "paired_count": len(first),
        "gained": gained,
        "regressed": regressed,
        "discordant": discordant,
        "exact_two_sided_p": probability,
    }


def hierarchical_bootstrap_delta(
    pairs: Mapping[str, Sequence[tuple[bool, bool]]],
    *,
    draws: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    tasks = sorted(pairs)
    if not tasks or any(not pairs[task] for task in tasks):
        return {"draws": 0, "mean_delta": None, "ci95_low": None, "ci95_high": None}
    observed = mean(
        mean(float(treatment) - float(control) for control, treatment in pairs[task])
        for task in tasks
    )
    generator = random.Random(seed)
    samples: list[float] = []
    for _ in range(draws):
        selected_tasks = [generator.choice(tasks) for _ in tasks]
        task_deltas = []
        for task in selected_tasks:
            values = pairs[task]
            resampled = [generator.choice(values) for _ in values]
            task_deltas.append(
                mean(
                    float(treatment) - float(control)
                    for control, treatment in resampled
                )
            )
        samples.append(mean(task_deltas))
    samples.sort()
    low = samples[int(0.025 * (draws - 1))]
    high = samples[int(0.975 * (draws - 1))]
    return {
        "draws": draws,
        "mean_delta": observed,
        "ci95_low": low,
        "ci95_high": high,
    }


def _usage(cell_root: Path) -> dict[str, Any]:
    window = cell_root / "agent_service_audit_window.log"
    if not window.is_file():
        return {}
    prefix = "[openai-planner-audit] "
    records = []
    for line in window.read_text(encoding="utf-8").splitlines():
        offset = line.find(prefix)
        if offset < 0:
            continue
        value = json.loads(line[offset + len(prefix) :])
        if (
            isinstance(value, Mapping)
            and value.get("event") == "agent_model_request_complete"
        ):
            records.append(value)
    usages = [record.get("usage", {}) for record in records]
    return {
        "agent_api_calls": len(records),
        "failed_agent_api_calls": sum(
            record.get("success") is not True for record in records
        ),
        "automatic_retries": sum(
            int(record.get("retry_count", 0)) for record in records
        ),
        "input_tokens": sum(int(value.get("input_tokens", 0)) for value in usages),
        "output_tokens": sum(int(value.get("output_tokens", 0)) for value in usages),
        "total_tokens": sum(int(value.get("total_tokens", 0)) for value in usages),
    }


def collect_cells(
    experiment_root: str | Path,
    matrix_path: str | Path,
) -> list[dict[str, Any]]:
    root = Path(experiment_root).resolve(strict=True)
    with (
        Path(matrix_path)
        .resolve(strict=True)
        .open(encoding="utf-8", newline="") as stream
    ):
        matrix = list(csv.DictReader(stream))
    if len(matrix) != 300 or len({row["cell_id"] for row in matrix}) != 300:
        raise FormalQ1AnalysisError("formal matrix must contain 300 unique cells")
    results: list[dict[str, Any]] = []
    for row in matrix:
        cell_root = root / "cells" / row["cell_id"]
        result_path = cell_root / "cell_result.json"
        if not result_path.is_file():
            results.append(
                {
                    **row,
                    "status": "missing",
                    "denominator_eligible": False,
                    "success": False,
                }
            )
            continue
        cell = _json(result_path)
        if (
            cell.get("cell_id") != row["cell_id"]
            or cell.get("task") != row["task"]
            or cell.get("method") != row["method"]
            or cell.get("evaluation_seed") != int(row["evaluation_seed"])
        ):
            raise FormalQ1AnalysisError(f"cell identity mismatch: {row['cell_id']}")
        validity = cell.get("episode_validity", {})
        trace_path = cell.get("trace_path")
        mechanism = (
            mechanism_metrics(_trace(Path(trace_path)))
            if isinstance(trace_path, str) and Path(trace_path).is_file()
            else mechanism_metrics(())
        )
        results.append(
            {
                **row,
                **cell,
                "denominator_eligible": isinstance(validity, Mapping)
                and validity.get("benchmark_denominator_eligible") is True,
                **_usage(cell_root),
                **mechanism,
            }
        )
    return results


def aggregate_cells(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        for method in METHOD_ORDER:
            group = [
                cell
                for cell in cells
                if cell.get("task") == task and cell.get("method") == method
            ]
            eligible = [
                cell for cell in group if cell.get("denominator_eligible") is True
            ]
            successes = sum(cell.get("success") is True for cell in eligible)
            low, high = wilson_interval(successes, len(eligible))
            numeric_progress = [
                float(cell["official_progress"])
                for cell in eligible
                if isinstance(cell.get("official_progress"), (int, float))
                and not isinstance(cell.get("official_progress"), bool)
            ]
            rows.append(
                {
                    "task": task,
                    "category": CATEGORY[task],
                    "method": method,
                    "planned": len(group),
                    "completed": sum(cell.get("status") != "missing" for cell in group),
                    "denominator": len(eligible),
                    "infrastructure_invalid": sum(
                        cell.get("status") == "infrastructure_invalid" for cell in group
                    ),
                    "successes": successes,
                    "success_rate": successes / len(eligible) if eligible else None,
                    "success_ci95_low": low,
                    "success_ci95_high": high,
                    "mean_official_progress": mean(numeric_progress)
                    if numeric_progress
                    else None,
                }
            )
    return rows


__all__ = [
    "CATEGORY",
    "METHOD_ORDER",
    "TASK_ORDER",
    "FormalQ1AnalysisError",
    "aggregate_cells",
    "collect_cells",
    "exact_mcnemar",
    "hierarchical_bootstrap_delta",
    "mechanism_metrics",
    "wilson_interval",
]
