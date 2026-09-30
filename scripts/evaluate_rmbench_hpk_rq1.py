from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


CASE_SCHEMA = "roboharn_evo/hpk/rq1_evaluation_case/v1"
REPORT_SCHEMA = "roboharn_evo/hpk/rq1_evaluation_report/v1"


class RQ1EvaluationError(ValueError):
    """Raised when a case cannot support an unambiguous RQ1 score."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise RQ1EvaluationError(f"{path} fields mismatch")


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise RQ1EvaluationError(f"{path} must be a string")
    result = " ".join(value.split())
    if not result:
        raise RQ1EvaluationError(f"{path} must be non-empty")
    return result


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RQ1EvaluationError(f"{path} must be an integer >= {minimum}")
    return value


def _probability(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RQ1EvaluationError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RQ1EvaluationError(f"{path} must be between 0 and 1")
    return result


def _bools(value: Any, path: str, expected_length: int) -> tuple[bool, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RQ1EvaluationError(f"{path} must be an array")
    result = tuple(value)
    if len(result) != expected_length or any(
        not isinstance(item, bool) for item in result
    ):
        raise RQ1EvaluationError(f"{path} must contain {expected_length} booleans")
    return result


def _probabilities(
    value: Any,
    path: str,
    expected_length: int,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RQ1EvaluationError(f"{path} must be an array")
    result = tuple(
        _probability(item, f"{path}[{index}]") for index, item in enumerate(value)
    )
    if len(result) != expected_length:
        raise RQ1EvaluationError(f"{path} must contain {expected_length} probabilities")
    return result


def _chunk_groups(
    value: Any,
    *,
    chunk_count: int,
    path: str,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RQ1EvaluationError(f"{path} must be an array of chunk groups")
    groups: list[tuple[int, ...]] = []
    observed: list[int] = []
    for group_index, raw_group in enumerate(value):
        group_path = f"{path}[{group_index}]"
        if isinstance(raw_group, (str, bytes)) or not isinstance(raw_group, Sequence):
            raise RQ1EvaluationError(f"{group_path} must be an array")
        group = tuple(
            _integer(item, f"{group_path}[{index}]")
            for index, item in enumerate(raw_group)
        )
        if not group or group != tuple(range(group[0], group[-1] + 1)):
            raise RQ1EvaluationError(f"{group_path} must be non-empty and contiguous")
        groups.append(group)
        observed.extend(group)
    if not groups or observed != list(range(chunk_count)):
        raise RQ1EvaluationError(
            f"{path} must cover every chunk exactly once in temporal order"
        )
    return tuple(groups)


def _labels(value: Any, *, chunk_count: int, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RQ1EvaluationError(f"{path} must be an array")
    result = tuple(_text(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(result) != chunk_count:
        raise RQ1EvaluationError(f"{path} must contain one label per chunk")
    return result


def _alignment(
    value: Any, *, predicted_count: int, reference_count: int
) -> tuple[int | None, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RQ1EvaluationError("prediction.subtask_alignment must be an array")
    result: list[int | None] = []
    matched: list[int] = []
    for index, item in enumerate(value):
        if item is None:
            result.append(None)
            continue
        matched_index = _integer(item, f"prediction.subtask_alignment[{index}]")
        if matched_index >= reference_count:
            raise RQ1EvaluationError(
                "prediction.subtask_alignment index is out of range"
            )
        matched.append(matched_index)
        result.append(matched_index)
    if len(result) != predicted_count or len(matched) != len(set(matched)):
        raise RQ1EvaluationError(
            "prediction.subtask_alignment must align each prediction at most once"
        )
    return tuple(result)


def _links(
    value: Any,
    path: str,
    *,
    action_count: int,
    chunk_count: int,
) -> frozenset[tuple[int, int]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RQ1EvaluationError(f"{path} must be an array")
    result: set[tuple[int, int]] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise RQ1EvaluationError(f"{path}[{index}] must be an object")
        _exact_keys(item, {"action_group", "evidence_chunk"}, f"{path}[{index}]")
        pair = (
            _integer(item["action_group"], f"{path}[{index}].action_group"),
            _integer(item["evidence_chunk"], f"{path}[{index}].evidence_chunk"),
        )
        if pair[0] >= action_count or pair[1] >= chunk_count:
            raise RQ1EvaluationError(f"{path}[{index}] index is out of range")
        if pair in result:
            raise RQ1EvaluationError(f"{path} contains duplicate links")
        result.add(pair)
    return frozenset(result)


def _f1(reference: Iterable[Any], prediction: Iterable[Any]) -> float:
    reference_set = set(reference)
    prediction_set = set(prediction)
    if not reference_set and not prediction_set:
        return 1.0
    if not reference_set or not prediction_set:
        return 0.0
    true_positive = len(reference_set & prediction_set)
    precision = true_positive / len(prediction_set)
    recall = true_positive / len(reference_set)
    return (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )


def _boundaries(groups: Sequence[Sequence[int]], chunk_count: int) -> frozenset[int]:
    return frozenset(group[-1] for group in groups if group[-1] != chunk_count - 1)


def _lcs_length(left: Sequence[int], right: Sequence[int]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_item == right_item
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def _mean(values: Sequence[float | bool]) -> float:
    return 0.0 if not values else sum(float(item) for item in values) / len(values)


def evaluate_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and score one blinded method/trajectory case."""

    _exact_keys(
        raw,
        {
            "schema",
            "case",
            "method",
            "chunk_count",
            "reference",
            "prediction",
            "blind_review",
        },
        "case",
    )
    if raw["schema"] not in {CASE_SCHEMA, "tcm/afk/rq1_evaluation_case/v1"}:
        raise RQ1EvaluationError("case.schema is unsupported")
    case_name = _text(raw["case"], "case.case")
    method = _text(raw["method"], "case.method")
    chunk_count = _integer(raw["chunk_count"], "case.chunk_count", minimum=1)
    reference = raw["reference"]
    prediction = raw["prediction"]
    review = raw["blind_review"]
    if not all(isinstance(item, Mapping) for item in (reference, prediction, review)):
        raise RQ1EvaluationError(
            "reference, prediction, and blind_review must be objects"
        )
    _exact_keys(
        reference,
        {
            "subtask_chunk_groups",
            "action_chunk_groups",
            "action_labels",
            "delayed_evidence_links",
        },
        "reference",
    )
    _exact_keys(
        prediction,
        {
            "subtask_chunk_groups",
            "subtask_alignment",
            "action_chunk_groups",
            "action_labels",
            "delayed_evidence_links",
        },
        "prediction",
    )
    reference_subtasks = _chunk_groups(
        reference["subtask_chunk_groups"],
        chunk_count=chunk_count,
        path="reference.subtask_chunk_groups",
    )
    predicted_subtasks = _chunk_groups(
        prediction["subtask_chunk_groups"],
        chunk_count=chunk_count,
        path="prediction.subtask_chunk_groups",
    )
    alignment = _alignment(
        prediction["subtask_alignment"],
        predicted_count=len(predicted_subtasks),
        reference_count=len(reference_subtasks),
    )
    reference_actions = _chunk_groups(
        reference["action_chunk_groups"],
        chunk_count=chunk_count,
        path="reference.action_chunk_groups",
    )
    predicted_actions = _chunk_groups(
        prediction["action_chunk_groups"],
        chunk_count=chunk_count,
        path="prediction.action_chunk_groups",
    )
    reference_labels = _labels(
        reference["action_labels"],
        chunk_count=chunk_count,
        path="reference.action_labels",
    )
    predicted_labels = _labels(
        prediction["action_labels"],
        chunk_count=chunk_count,
        path="prediction.action_labels",
    )
    reference_links = _links(
        reference["delayed_evidence_links"],
        "reference.delayed_evidence_links",
        action_count=len(reference_actions),
        chunk_count=chunk_count,
    )
    predicted_links = _links(
        prediction["delayed_evidence_links"],
        "prediction.delayed_evidence_links",
        action_count=len(predicted_actions),
        chunk_count=chunk_count,
    )

    _exact_keys(
        review,
        {
            "purpose_correct",
            "completion_correct",
            "observed_fact_precision",
            "rationale_supported",
            "temporal_state_correct",
            "geometry_source_supported",
            "geometry_specific",
            "meaningless_geometry",
            "runtime_mappable",
            "absolute_pose_present",
            "expected_effect_correct",
            "final_verdict_correct",
            "unsupported_physical_claim",
            "top_level_object_count",
            "dynamic_state_leakage_count",
            "total_evidence",
            "duplicate_evidence",
        },
        "blind_review",
    )
    subtask_count = len(reference_subtasks)
    action_count = len(reference_actions)
    purpose = _bools(
        review["purpose_correct"], "blind_review.purpose_correct", subtask_count
    )
    completion = _bools(
        review["completion_correct"], "blind_review.completion_correct", subtask_count
    )
    fact_precision = _probabilities(
        review["observed_fact_precision"],
        "blind_review.observed_fact_precision",
        subtask_count,
    )
    rationale = _bools(
        review["rationale_supported"], "blind_review.rationale_supported", subtask_count
    )
    temporal = _bools(
        review["temporal_state_correct"],
        "blind_review.temporal_state_correct",
        action_count,
    )
    geometry_source = _bools(
        review["geometry_source_supported"],
        "blind_review.geometry_source_supported",
        action_count,
    )
    geometry_specific = _bools(
        review["geometry_specific"], "blind_review.geometry_specific", action_count
    )
    meaningless_geometry = _bools(
        review["meaningless_geometry"],
        "blind_review.meaningless_geometry",
        action_count,
    )
    runtime_mappable = _bools(
        review["runtime_mappable"], "blind_review.runtime_mappable", action_count
    )
    absolute_pose = _bools(
        review["absolute_pose_present"],
        "blind_review.absolute_pose_present",
        action_count,
    )
    expected_effect = _bools(
        review["expected_effect_correct"],
        "blind_review.expected_effect_correct",
        action_count,
    )
    final_verdict = _bools(
        review["final_verdict_correct"],
        "blind_review.final_verdict_correct",
        action_count,
    )
    unsupported_claim = _bools(
        review["unsupported_physical_claim"],
        "blind_review.unsupported_physical_claim",
        action_count,
    )
    top_level_object_count = _integer(
        review["top_level_object_count"],
        "blind_review.top_level_object_count",
        minimum=1,
    )
    dynamic_state_leakage_count = _integer(
        review["dynamic_state_leakage_count"],
        "blind_review.dynamic_state_leakage_count",
    )
    if dynamic_state_leakage_count > top_level_object_count:
        raise RQ1EvaluationError(
            "dynamic_state_leakage_count cannot exceed top_level_object_count"
        )
    total_evidence = _integer(review["total_evidence"], "blind_review.total_evidence")
    duplicate_evidence = _integer(
        review["duplicate_evidence"], "blind_review.duplicate_evidence"
    )
    if duplicate_evidence > total_evidence:
        raise RQ1EvaluationError("duplicate_evidence cannot exceed total_evidence")

    delayed_action_indices = sorted({item[0] for item in reference_links})

    predicted_reference_by_chunk: list[int | None] = [None] * chunk_count
    for group, reference_index in zip(predicted_subtasks, alignment, strict=True):
        for chunk in group:
            predicted_reference_by_chunk[chunk] = reference_index
    reference_by_chunk = [0] * chunk_count
    for reference_index, group in enumerate(reference_subtasks):
        for chunk in group:
            reference_by_chunk[chunk] = reference_index

    aligned_order = [item for item in alignment if item is not None]
    canonical_order = list(range(len(reference_subtasks)))
    order_denominator = max(len(aligned_order), len(canonical_order), 1)
    subtask_assignment_correct = [
        predicted_reference_by_chunk[index] == reference_by_chunk[index]
        for index in range(chunk_count)
    ]
    action_label_correct = [
        left == right
        for left, right in zip(reference_labels, predicted_labels, strict=True)
    ]
    metrics = {
        "subtask_boundary_f1": _f1(
            _boundaries(reference_subtasks, chunk_count),
            _boundaries(predicted_subtasks, chunk_count),
        ),
        "subtask_order_accuracy": _lcs_length(aligned_order, canonical_order)
        / order_denominator,
        "subtask_assignment_accuracy": _mean(subtask_assignment_correct),
        "action_boundary_f1": _f1(
            _boundaries(reference_actions, chunk_count),
            _boundaries(predicted_actions, chunk_count),
        ),
        "action_label_accuracy": _mean(action_label_correct),
        "action_assignment_accuracy": _mean(
            [
                subtask_ok and action_ok
                for subtask_ok, action_ok in zip(
                    subtask_assignment_correct,
                    action_label_correct,
                    strict=True,
                )
            ]
        ),
        "purpose_correctness": _mean(purpose),
        "completion_correctness": _mean(completion),
        "observed_fact_precision": _mean(fact_precision),
        "rationale_consistency": _mean(rationale),
        "unsupported_rationale_rate": 1.0 - _mean(rationale),
        "temporal_state_accuracy": _mean(temporal),
        "geometry_source_support": _mean(geometry_source),
        "geometry_specificity": _mean(geometry_specific),
        "meaningless_geometry_rate": _mean(meaningless_geometry),
        "runtime_mappability": _mean(runtime_mappable),
        "absolute_pose_rate": _mean(absolute_pose),
        "expected_effect_correctness": _mean(expected_effect),
        "final_verdict_accuracy": _mean(final_verdict),
        "delayed_verdict_accuracy": _mean(
            [final_verdict[index] for index in delayed_action_indices]
        ),
        "dynamic_state_leakage_rate": (
            dynamic_state_leakage_count / top_level_object_count
        ),
        "unsupported_physical_claim_rate": _mean(unsupported_claim),
        "delayed_evidence_link_f1": _f1(reference_links, predicted_links),
        "duplicate_evidence_rate": (
            0.0 if total_evidence == 0 else duplicate_evidence / total_evidence
        ),
    }
    return {"case": case_name, "method": method, "metrics": metrics}


def evaluate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise RQ1EvaluationError("at least one evaluation case is required")
    cases = [evaluate_case(record) for record in records]
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for case in cases:
        grouped[case["method"]].append(case["metrics"])
    methods: dict[str, Any] = {}
    for method, values in sorted(grouped.items()):
        metric_names = tuple(values[0])
        methods[method] = {
            "cases": len(values),
            "macro_metrics": {
                metric: _mean([value[metric] for value in values])
                for metric in metric_names
            },
        }
    return {
        "schema": REPORT_SCHEMA,
        "case_count": len(cases),
        "methods": methods,
        "cases": cases,
    }


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise RQ1EvaluationError(f"line {line_number} must be an object")
        records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_records(_read_jsonl(args.cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
