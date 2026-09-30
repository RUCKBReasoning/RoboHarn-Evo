#!/usr/bin/env python3
"""Build the 20-case HPK v2 offline proposal/rank-change report.

The default controlled backend is deterministic test evidence, not a claim
about live GPT quality.  It exercises the exact VLMStrategyProposer parsing,
validation, proposal construction, and existing candidate-ranking path without
motion, simulator, network, or HPK Store writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from roboharn_evo.agent.hpk.schemas import canonical_json_bytes
from roboharn_evo.agent.hpk.strategy_proposal import (
    TASK_STRATEGY_V2_SCHEMA,
    ABSTRACT_EFFECT_V2_SCHEMA,
    CONDITION_V2_SCHEMA,
    GEOMETRIC_STRATEGY_V2_SCHEMA,
    PROPOSAL_EVIDENCE_PACKET_SCHEMA,
    STRATEGY_DELTA_SCHEMA,
    ProposalEvidencePacketV1,
    rank_frozen_candidates_with_proposal,
)
from roboharn_evo.agent.hpk.vlm_strategy_proposer import (
    ProposalBackendCompletion,
    VLMStrategyProposer,
)


def controlled_packet(case_index: int, *, failed_axis: int) -> ProposalEvidencePacketV1:
    orientation = f"align_principal_axis_{failed_axis}"
    evidence_ref = f"e_case_{case_index:02d}_axis{failed_axis}_oppose"
    return ProposalEvidencePacketV1(
        {
            "schema": PROPOSAL_EVIDENCE_PACKET_SCHEMA,
            "condition": {
                "schema": CONDITION_V2_SCHEMA,
                "task_family": "controlled_grasp_geometry",
                "operation": "grasp",
                "manipulation_phase": "grasp_candidate",
                "manipulated_role": "current_target_object",
                "target_role": None,
                "held_state": "not_held",
                "scene_predicates": [],
                "abstraction_version": "afk_condition_builder/v2",
            },
            "current_task_strategy": {
                "schema": TASK_STRATEGY_V2_SCHEMA,
                "operation": "grasp",
                "manipulated_role": "current_target_object",
                "target_role": None,
                "target_relation": None,
                "manipulation_phase": "grasp_candidate",
                "subgoal_purpose": "establish_stable_attachment",
                "preferred_arm": "either",
            },
            "current_geometric_strategy": {
                "schema": GEOMETRIC_STRATEGY_V2_SCHEMA,
                "strategy_family": "observed_grasp_geometry",
                "reference_frame": "object_principal_axes",
                "approach_family": "principal_axis_relative",
                "approach_direction": "lateral",
                "orientation_relation": orientation,
                "grasp_region": "observed_surface",
                "semantic_part": None,
                "placement_relation": None,
                "hard_constraints": [],
                "soft_preferences": [],
                "avoid": [],
                "geometry_source_class": "rgbd_observed",
            },
            "expected_effect": {
                "schema": ABSTRACT_EFFECT_V2_SCHEMA,
                "effect_type": "grasp",
                "expected_predicates": ["object_attached"],
            },
            "observed_effect": {"predicates": []},
            "verdict": "oppose",
            "recent_evidence": [
                {
                    "strategy_summary": {
                        "evidence_ref": evidence_ref,
                        "orientation_relation": orientation,
                        "realization_status": "satisfied",
                        "observed_effect": "attachment_absent",
                    },
                    "verdict": "oppose",
                }
            ],
            "available_capabilities": {
                "task_fields": {},
                "geometry_fields": {
                    "orientation_relation": [
                        "align_principal_axis_0",
                        "align_principal_axis_1",
                    ],
                    "add_avoid": ["repeat_equivalent_failed_candidate"],
                },
            },
            "retrieved_hpk": [],
            "optional_observations": {
                "before_image_ref": None,
                "after_image_ref": None,
            },
        }
    )


def frozen_candidates(case_index: int, *, failed_axis: int) -> list[dict[str, Any]]:
    alternative = 1 - failed_axis

    def candidate(axis: int, baseline_rank: int) -> dict[str, Any]:
        return {
            "candidate_id": f"private-case-{case_index:02d}-axis-{axis}",
            "action_mode": "grasp",
            "arm": "left",
            "priority": baseline_rank,
            "ee_target_pose": [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
            "approach_pose": [0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0],
            "geometry_source": "rgbd_observed_volume_principal_axes",
            "source_candidate_index": axis,
            "orientation_relation": f"align_principal_axis_{axis}",
            "approach_family": "principal_axis_relative",
            "approach_direction_bucket": "lateral",
            "grasp_region": "observed_surface",
            "observed_grasp_width_m": 0.03 + 0.001 * (case_index % 5) + 0.005 * axis,
            "reach_distance_m": 0.18 + 0.01 * baseline_rank,
        }

    return [candidate(failed_axis, 0), candidate(alternative, 1)]


class ControlledGeometryProposalBackend:
    """Deterministic backend used only to test the proposer contract."""

    def complete(self, **kwargs: Any) -> ProposalBackendCompletion:
        packet = json.loads(kwargs["input_text"])
        failed = packet["current_geometric_strategy"]["orientation_relation"]
        alternative = (
            "align_principal_axis_0"
            if failed == "align_principal_axis_1"
            else "align_principal_axis_1"
        )
        evidence_ref = packet["recent_evidence"][0]["strategy_summary"]["evidence_ref"]
        return ProposalBackendCompletion(
            output={
                "proposals": [
                    {
                        "delta": {
                            "schema": STRATEGY_DELTA_SCHEMA,
                            "scope": "geometry",
                            "task_delta": None,
                            "geometry_delta": {
                                "approach_family": None,
                                "approach_direction": None,
                                "orientation_relation": alternative,
                                "grasp_region": None,
                                "semantic_part": None,
                                "placement_relation": None,
                                "add_hard_constraints": [],
                                "add_avoid": ["repeat_equivalent_failed_candidate"],
                            },
                        },
                        "expected_effect": packet["expected_effect"],
                        "evidence_refs": [evidence_ref],
                        "rationale": (
                            "Test the other supported principal-axis orientation "
                            "after verified non-attachment."
                        ),
                    }
                ]
            },
            audit={
                "backend": "controlled_geometry_proposal_fixture",
                "external_model_call": False,
            },
        )


def build_report(*, case_count: int = 20) -> dict[str, Any]:
    if case_count != 20:
        raise ValueError("the frozen offline report requires exactly 20 cases")
    proposer = VLMStrategyProposer(ControlledGeometryProposalBackend())
    cases: list[dict[str, Any]] = []
    for index in range(case_count):
        failed_axis = index % 2
        packet = controlled_packet(index, failed_axis=failed_axis)
        result = proposer.propose(packet)
        if len(result.proposals) != 1:
            raise RuntimeError(f"controlled case {index} did not yield one proposal")
        trace = rank_frozen_candidates_with_proposal(
            proposal=result.proposals[0],
            frozen_candidates=frozen_candidates(index, failed_axis=failed_axis),
        )
        trace_payload = trace.to_dict()
        cases.append(
            {
                "case_id": f"geometry_failure_{index:02d}",
                "failed_orientation": f"align_principal_axis_{failed_axis}",
                "proposed_orientation": result.proposals[0]["delta"]["geometry_delta"][
                    "orientation_relation"
                ],
                "schema_valid": not result.validation_errors,
                "proposal_count": len(result.proposals),
                "rank_changed": trace_payload["rank_changed"],
                "selection_changed": trace_payload["selection_changed"],
                "selected_candidate_satisfies_proposal": trace_payload[
                    "selected_candidate_satisfies_proposal"
                ],
                "proposal": result.proposals[0].to_dict(),
                "rank_trace": trace_payload,
            }
        )
    report = {
        "schema": "roboharn_evo/hpk/offline_geometry_proposal_report/v1",
        "case_count": len(cases),
        "backend": "controlled_geometry_proposal_fixture",
        "actual_gpt_calls": 0,
        "motion_executed": False,
        "hpk_store_write_performed": False,
        "metrics": {
            "schema_valid_cases": sum(case["schema_valid"] for case in cases),
            "rank_changed_cases": sum(case["rank_changed"] for case in cases),
            "selection_changed_cases": sum(case["selection_changed"] for case in cases),
            "proposal_compliant_selection_cases": sum(
                case["selected_candidate_satisfies_proposal"] for case in cases
            ),
        },
        "cases": cases,
    }
    serialized = canonical_json_bytes(report).decode("utf-8")
    if "candidate_id" in serialized or "private-case-" in serialized:
        raise RuntimeError("public report leaked private candidate identity")
    return report


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# HPK v2 Offline Geometry Proposal Report",
        "",
        (
            "This is a controlled contract/ranking report. It uses no external "
            "GPT call, robot action, simulator, or HPK Store write."
        ),
        "",
        f"- cases: {report['case_count']}",
        f"- schema valid: {metrics['schema_valid_cases']}/{report['case_count']}",
        f"- rank changed: {metrics['rank_changed_cases']}/{report['case_count']}",
        f"- selection changed: {metrics['selection_changed_cases']}/{report['case_count']}",
        (
            "- selected candidate satisfies proposal: "
            f"{metrics['proposal_compliant_selection_cases']}/{report['case_count']}"
        ),
        "",
        "| Case | Failed geometry | Proposed geometry | Rank changed | Compliant |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for case in report["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['failed_orientation']} | "
            f"{case['proposed_orientation']} | {str(case['rank_changed']).lower()} | "
            f"{str(case['selected_candidate_satisfies_proposal']).lower()} |"
        )
    return "\n".join(lines) + "\n"


def write_report(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    report = build_report()
    report_path = output_dir / "proposal_report.json"
    markdown_path = output_dir / "proposal_report.md"
    trace_path = output_dir / "rank_change_trace.json"
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    trace_path.write_bytes(
        canonical_json_bytes(report["cases"][0]["rank_trace"]) + b"\n"
    )
    return {
        "proposal_report": str(report_path),
        "proposal_report_markdown": str(markdown_path),
        "rank_change_trace": str(trace_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_report(args.output_dir.absolute()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
