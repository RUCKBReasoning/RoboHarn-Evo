
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CASE_SCHEMA = "roboharn_evo/hpk/controlled_fault_case/v1"
REPORT_SCHEMA = "roboharn_evo/hpk/controlled_fault_report/v1"
PROPOSAL_SCOPES = {"subtask strategy", "action geometry"}
EXPECTED_SCOPES = PROPOSAL_SCOPES | {"abstain"}
VERDICTS = {"support", "oppose", "unverified"}
OFFLINE_GEOMETRY_REPORT_SCHEMA = "roboharn_evo/hpk/offline_geometry_proposal_report/v1"


class ControlledFaultEvaluationError(ValueError):
    """Raised when a controlled case cannot be scored unambiguously."""


def _exact(value: Mapping[str, Any], keys: set[str], path: str) -> None:
    if set(value) != keys:
        raise ControlledFaultEvaluationError(f"{path} fields mismatch")


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ControlledFaultEvaluationError(f"{path} must be a string")
    result = " ".join(value.split())
    if not result:
        raise ControlledFaultEvaluationError(f"{path} must be non-empty")
    return result


def evaluate_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    _exact(
        raw,
        {
            "schema",
            "case",
            "fault_family",
            "expected_scope",
            "proposals",
            "tool_reported_success",
            "knowledge_update_performed",
        },
        "case",
    )
    if raw["schema"] not in {CASE_SCHEMA, "tcm/afk/controlled_fault_case/v1"}:
        raise ControlledFaultEvaluationError("case.schema is unsupported")
    case_name = _text(raw["case"], "case.case")
    family = _text(raw["fault_family"], "case.fault_family")
    expected_scope = _text(raw["expected_scope"], "case.expected_scope")
    if expected_scope not in EXPECTED_SCOPES:
        raise ControlledFaultEvaluationError("case.expected_scope is unsupported")
    for flag in ("tool_reported_success", "knowledge_update_performed"):
        if not isinstance(raw[flag], bool):
            raise ControlledFaultEvaluationError(f"case.{flag} must be a boolean")
    proposals_raw = raw["proposals"]
    if isinstance(proposals_raw, (str, bytes)) or not isinstance(
        proposals_raw, Sequence
    ):
        raise ControlledFaultEvaluationError("case.proposals must be an array")
    if len(proposals_raw) > 2:
        raise ControlledFaultEvaluationError("case.proposals exceeds the frozen cap")
    proposals: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposals_raw):
        path = f"case.proposals[{index}]"
        if not isinstance(proposal, Mapping):
            raise ControlledFaultEvaluationError(f"{path} must be an object")
        _exact(
            proposal,
            {
                "scope",
                "runtime_mappable",
                "non_equivalent",
                "behavior_changed",
                "physical_verdict",
            },
            path,
        )
        scope = _text(proposal["scope"], f"{path}.scope")
        verdict = _text(proposal["physical_verdict"], f"{path}.physical_verdict")
        if scope not in PROPOSAL_SCOPES or verdict not in VERDICTS:
            raise ControlledFaultEvaluationError(f"{path} has an unsupported value")
        for flag in ("runtime_mappable", "non_equivalent", "behavior_changed"):
            if not isinstance(proposal[flag], bool):
                raise ControlledFaultEvaluationError(f"{path}.{flag} must be boolean")
        proposals.append(
            {
                "scope": scope,
                "runtime_mappable": proposal["runtime_mappable"],
                "non_equivalent": proposal["non_equivalent"],
                "behavior_changed": proposal["behavior_changed"],
                "physical_verdict": verdict,
            }
        )

    expected_abstention = expected_scope == "abstain"
    correct_scope = (
        not proposals
        if expected_abstention
        else any(item["scope"] == expected_scope for item in proposals)
    )
    correct_proposals = (
        []
        if expected_abstention
        else [item for item in proposals if item["scope"] == expected_scope]
    )
    runtime_mappable = any(item["runtime_mappable"] for item in correct_proposals)
    non_equivalent = any(item["non_equivalent"] for item in correct_proposals)
    behavior_changed = any(item["behavior_changed"] for item in correct_proposals)
    verified_useful = any(
        item["runtime_mappable"]
        and item["non_equivalent"]
        and item["behavior_changed"]
        and item["physical_verdict"] == "support"
        for item in correct_proposals
    )
    any_support = any(item["physical_verdict"] == "support" for item in proposals)
    any_unverified = any(item["physical_verdict"] == "unverified" for item in proposals)
    false_positive_update = raw["knowledge_update_performed"] and not any_support
    unverified_update = raw["knowledge_update_performed"] and any_unverified
    tool_effect_disagreement = raw["tool_reported_success"] and not any_support
    return {
        "case": case_name,
        "fault_family": family,
        "expected_scope": expected_scope,
        "proposal_count": len(proposals),
        "metrics": {
            "correct_scope_at_k": correct_scope,
            "runtime_mappable_at_k": runtime_mappable,
            "non_equivalent_at_k": non_equivalent,
            "behavior_change_at_k": behavior_changed,
            "verified_useful_at_k": verified_useful,
            "correct_abstention": expected_abstention and not proposals,
            "false_positive_knowledge": false_positive_update,
            "unverified_update": unverified_update,
            "tool_effect_disagreement": tool_effect_disagreement,
        },
    }


def _aggregate(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = tuple(cases[0]["metrics"])
    proposal_only = {
        "runtime_mappable_at_k",
        "non_equivalent_at_k",
        "behavior_change_at_k",
        "verified_useful_at_k",
    }
    abstention_only = {"correct_abstention"}
    eligible: dict[str, list[Mapping[str, Any]]] = {}
    for name in metric_names:
        if name in proposal_only:
            eligible[name] = [
                case for case in cases if case["expected_scope"] != "abstain"
            ]
        elif name in abstention_only:
            eligible[name] = [
                case for case in cases if case["expected_scope"] == "abstain"
            ]
        else:
            eligible[name] = list(cases)
    return {
        "cases": len(cases),
        "metric_denominators": {name: len(eligible[name]) for name in metric_names},
        **{
            name: (
                None
                if not eligible[name]
                else sum(bool(case["metrics"][name]) for case in eligible[name])
                / len(eligible[name])
            )
            for name in metric_names
        },
    }


def evaluate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ControlledFaultEvaluationError("at least one case is required")
    cases = [evaluate_case(item) for item in records]
    if len({item["case"] for item in cases}) != len(cases):
        raise ControlledFaultEvaluationError("case names must be unique")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_family[case["fault_family"]].append(case)
    return {
        "schema": REPORT_SCHEMA,
        "overall": _aggregate(cases),
        "by_fault_family": {
            family: _aggregate(values) for family, values in sorted(by_family.items())
        },
        "cases": cases,
    }


def cases_from_offline_geometry_report(
    raw: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Adapt the existing simulator-free proposal report without inventing effects."""

    if raw.get("schema") not in {OFFLINE_GEOMETRY_REPORT_SCHEMA, "tcm/afk/offline_geometry_proposal_report/v1"}:
        raise ControlledFaultEvaluationError(
            "offline geometry proposal report schema is unsupported"
        )
    if raw.get("motion_executed") is not False:
        raise ControlledFaultEvaluationError(
            "offline geometry proposal report must not claim motion"
        )
    report_cases = raw.get("cases")
    if isinstance(report_cases, (str, bytes)) or not isinstance(report_cases, Sequence):
        raise ControlledFaultEvaluationError(
            "offline geometry proposal report cases must be an array"
        )
    if raw.get("case_count") != len(report_cases):
        raise ControlledFaultEvaluationError(
            "offline geometry proposal report case_count mismatch"
        )
    result: list[dict[str, Any]] = []
    for index, case in enumerate(report_cases):
        path = f"offline_report.cases[{index}]"
        if not isinstance(case, Mapping):
            raise ControlledFaultEvaluationError(f"{path} must be an object")
        case_name = _text(case.get("case_id"), f"{path}.case_id")
        proposal_count = case.get("proposal_count")
        if proposal_count not in {0, 1, 2}:
            raise ControlledFaultEvaluationError(
                f"{path}.proposal_count exceeds the frozen cap"
            )
        if proposal_count == 0:
            proposals: list[dict[str, Any]] = []
        else:
            required_flags = (
                "schema_valid",
                "rank_changed",
                "selection_changed",
                "selected_candidate_satisfies_proposal",
            )
            if any(not isinstance(case.get(flag), bool) for flag in required_flags):
                raise ControlledFaultEvaluationError(
                    f"{path} proposal flags must be booleans"
                )
            failed = _text(case.get("failed_orientation"), f"{path}.failed_orientation")
            proposed = _text(
                case.get("proposed_orientation"), f"{path}.proposed_orientation"
            )
            proposal = {
                "scope": "action geometry",
                "runtime_mappable": case["schema_valid"],
                "non_equivalent": proposed != failed,
                "behavior_changed": bool(
                    case["rank_changed"]
                    and case["selection_changed"]
                    and case["selected_candidate_satisfies_proposal"]
                ),
                # No motion was executed, so a physical verdict cannot be inferred.
                "physical_verdict": "unverified",
            }
            proposals = [dict(proposal) for _ in range(proposal_count)]
        result.append(
            {
                "schema": CASE_SCHEMA,
                "case": case_name,
                "fault_family": "action geometry fault",
                "expected_scope": "action geometry",
                "proposals": proposals,
                "tool_reported_success": False,
                "knowledge_update_performed": False,
            }
        )
    return result


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ControlledFaultEvaluationError(f"line {line_number} is not an object")
        result.append(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--cases", type=Path)
    inputs.add_argument("--offline-geometry-report", type=Path)
    parser.add_argument(
        "--normalized-cases-output",
        type=Path,
        help="optional JSONL copy of the strict cases actually scored",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cases is not None:
        records = _read_jsonl(args.cases)
    else:
        value = json.loads(args.offline_geometry_report.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ControlledFaultEvaluationError(
                "offline geometry proposal report must be an object"
            )
        records = cases_from_offline_geometry_report(value)
    if args.normalized_cases_output is not None:
        args.normalized_cases_output.parent.mkdir(parents=True, exist_ok=True)
        args.normalized_cases_output.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
    report = evaluate_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
