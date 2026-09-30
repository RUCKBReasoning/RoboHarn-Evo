# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.agent.hpk.goal_consistency import (
    build_realization_hypothesis_v31,
    filter_goal_consistent_candidates_v31,
    map_staged_carry_realization_v31,
    unresolved_feasibility_report_v31,
)
from roboharn_evo.agent.hpk.hierarchical_knowledge import (
    ActionKnowledgeV3,
    SubtaskGoalContractV31,
)
from roboharn_evo.agent.hpk.hierarchical_retriever import (
    VLMActionKnowledgeRetriever,
    build_action_knowledge_query,
)
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import HierarchicalBackendCompletion
from roboharn_evo.agent.operation_candidates import select_operation_pose_candidate
from roboharn_evo.benchmark_adapters.rmbench.v31_goal_bridge import (
    RMBenchFrozenGoalContextV31,
    RMBenchGoalContextBridgeV31,
    map_staged_carry_to_rmbench_calls_v31,
)


class _FrozenActionDecision:
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = copy.deepcopy(output)
        self.calls = 0

    def complete(self, **_kwargs: Any) -> HierarchicalBackendCompletion:
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("fixture permits exactly one frozen Action decision")
        return HierarchicalBackendCompletion(output=copy.deepcopy(self.output))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _contract() -> SubtaskGoalContractV31:
    return SubtaskGoalContractV31(
        {
            "subtask": "place the held object at its intended destination",
            "purpose": "advance the current subtask without changing its target",
            "operation": "place",
            "manipulated_role": "currently held object",
            "required_target_role": "destination support",
            "required_target_relation": "center of",
            "expected_effect": "the object is stably supported at the intended destination",
            "completion_condition": "the object remains there after release",
        }
    )


def _context() -> RMBenchFrozenGoalContextV31:
    return RMBenchFrozenGoalContextV31(
        goal_contract=_contract(),
        manipulated_object_ref="held-object",
        required_target_scene_ref="support-a",
        local_trace_ref="fixture-attempt",
    )


def _target(target_ref: str, support_ref: str) -> dict[str, Any]:
    return {
        "target_id": target_ref,
        "action_mode": "place",
        "held_instance_id": "held-object",
        "target_kind": "object_top",
        "support_instance_id": support_ref,
        "support_valid": True,
        "free": True,
    }


def _candidate(
    candidate_ref: str,
    target_ref: str,
    *,
    priority: float,
    lateral: bool,
) -> dict[str, Any]:
    target_pose = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]
    approach_pose = (
        [0.1, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]
        if lateral
        else [0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]
    )
    return {
        "candidate_id": candidate_ref,
        "target_id": target_ref,
        "held_instance_id": "held-object",
        "action_mode": "place",
        "arm": "left",
        "ee_target_pose": target_pose,
        "approach_pose": approach_pose,
        "geometry_source": "runtime_dynamic_place_geometry",
        "attachment_transform_source": ("verifier_confirmed_tcp_local_attachment"),
        "holding_status": "verified",
        "grasp_transport_policy": "strict",
        "orientation_policy": "preserve_current_rigid_attachment",
        "approach_clearance_m": 0.05,
        "reach_distance_m": 0.20,
        "support_valid": True,
        "free": True,
        "reachable_estimate": True,
        "valid": True,
        "priority": priority,
    }


def _knowledge() -> ActionKnowledgeV3:
    return ActionKnowledgeV3(
        {
            "condition": {
                "action": "place",
                "object_description": "a currently held rigid object",
                "held_state": "selected arm is holding an object",
                "target_relation": "center of",
            },
            "geometric_strategy": {
                "approach_direction": "from the side at validated clearance",
                "placement_relation": "center of the intended support",
                "clearance_or_support_constraint": (
                    "use a free Runtime-validated support"
                ),
                "avoid": ["release outside the intended support"],
            },
            "expected_effect": {
                "physical_effect": "the object is released at the intended support",
                "verification_observation": "the object remains there after release",
            },
            "evidence_summary": {
                "support": 1,
                "oppose": 0,
                "unverified": 0,
                "independent_verified_trials": 1,
            },
            "status": "supported",
        }
    )


def _fixture_a() -> dict[str, Any]:
    scene = {
        "operation_targets": [
            _target("runtime-target-a", "support-a"),
            _target("runtime-target-b", "support-b"),
        ],
        "manipulation_state": {
            "left": {"held_instance_id": "held-object"},
        },
    }
    prepared = RMBenchGoalContextBridgeV31(_context()).prepare(
        scene_memory=scene,
        active_skill_ref="fixture-attempt",
    )
    candidates = [
        _candidate("a-one", "runtime-target-a", priority=0.0, lateral=False),
        _candidate("a-two", "runtime-target-a", priority=1.0, lateral=True),
        _candidate("b-one", "runtime-target-b", priority=-1.0, lateral=False),
    ]
    filtered = filter_goal_consistent_candidates_v31(
        candidates,
        contract=_contract(),
        runtime_binding=prepared.runtime_binding,
        target_records=prepared.scene_memory["operation_targets"],
    )
    instance = {
        "instance_id": "held-object",
        "role": "currently held object",
        "operation_pose_candidates": candidates,
    }
    query = build_action_knowledge_query(
        scene_memory=prepared.scene_memory,
        instance=instance,
        arm="left",
        action_mode="place",
        eligible_candidates=filtered.candidates,
        goal_contract=_contract(),
    )
    backend = _FrozenActionDecision(
        {
            "selected_knowledge_index": 0,
            "selected_candidate_geometry_index": 1,
            "reason": "the lateral candidate realizes the frozen supported geometry",
        }
    )
    match = VLMActionKnowledgeRetriever(backend).retrieve((_knowledge(),), query)
    if match is None:
        raise RuntimeError("frozen fixture Action decision unexpectedly abstained")
    usage: dict[str, Any] = {}
    selected = select_operation_pose_candidate(
        instance,
        arm="left",
        action_mode="place",
        eligible_candidate_ids=[
            str(item["candidate_id"]) for item in filtered.candidates
        ],
        v3_action_match=match,
        v3_usage_audit=usage,
        geometry_scene_state=prepared.scene_memory,
    )
    passed = bool(
        [item["candidate_id"] for item in filtered.candidates] == ["a-one", "a-two"]
        and selected is not None
        and selected["candidate_id"] == "a-two"
        and all(item["target_id"] == "runtime-target-a" for item in filtered.candidates)
    )
    return {
        "schema": "roboharn_evo/rmbench/v31/fixture_a",
        "goal_contract": _contract().to_dict(),
        "runtime_goal_binding": prepared.runtime_binding.to_private_dict(),
        "candidates_before_goal_filter": candidates,
        "candidates_after_goal_filter": list(filtered.candidates),
        "goal_filter_rejections": list(filtered.rejection_reasons),
        "action_rank_before": usage.get("rank_before", []),
        "action_rank_after": usage.get("rank_after", []),
        "selected_candidate": selected,
        "passed": passed,
    }


def _fixture_b() -> dict[str, Any]:
    scene = {"operation_targets": [_target("runtime-target-b", "support-b")]}
    prepared = RMBenchGoalContextBridgeV31(_context()).prepare(
        scene_memory=scene,
        active_skill_ref="fixture-attempt",
    )
    candidates = [
        _candidate("b-one", "runtime-target-b", priority=0.0, lateral=False),
        _candidate("b-two", "runtime-target-b", priority=1.0, lateral=True),
    ]
    filtered = filter_goal_consistent_candidates_v31(
        candidates,
        contract=_contract(),
        runtime_binding=prepared.runtime_binding,
        target_records=prepared.scene_memory["operation_targets"],
    )
    return {
        "schema": "roboharn_evo/rmbench/v31/fixture_b",
        "goal_contract": _contract().to_dict(),
        "runtime_goal_binding": prepared.runtime_binding.to_private_dict(),
        "candidates_before_goal_filter": candidates,
        "candidates_after_goal_filter": list(filtered.candidates),
        "runtime_feasibility_report": filtered.feasibility_report.to_dict(),
        "selected_candidate": None,
        "motion_executed": False,
        "target_substitution": False,
        "passed": bool(filtered.unresolved and not filtered.candidates),
    }


def _fixture_c() -> dict[str, Any]:
    report = unresolved_feasibility_report_v31(
        failed_stage="transport mapping",
        missing_prerequisite="one-shot transport is unavailable",
        available_realization_families=("staged carry",),
        repairable_variables=("transport realization",),
    )
    hypothesis = build_realization_hypothesis_v31(
        contract=_contract(),
        feasibility_report=report,
        observed_problem="one-shot transport is unavailable",
        change={"transport_realization": "staged carry"},
        runtime_realization="staged carry",
        support_condition=(
            "the same object reaches and remains at the intended support"
        ),
        oppose_condition="the object is lost or reaches another support",
        rationale="change transport only while preserving the frozen goal",
    )
    prepared = RMBenchGoalContextBridgeV31(_context()).prepare(
        scene_memory={"operation_targets": [_target("runtime-target-a", "support-a")]},
        active_skill_ref="fixture-attempt",
    )
    realization = map_staged_carry_realization_v31(
        hypothesis=hypothesis,
        contract=_contract(),
        runtime_binding=prepared.runtime_binding,
        verified_attached=True,
    )
    calls = map_staged_carry_to_rmbench_calls_v31(
        realization,
        arm="left",
        validated_lift_segments_m=(0.04, 0.03),
        bounded_translation_steps=4,
        max_translation_m=0.08,
    )
    before_step = 17
    after_step = 23
    fresh_effect_observation = {
        "env_step": after_step,
        "fresh": True,
        "manipulated_object_ref": "held-object",
        "observed_support_ref": "support-a",
        "observed_relation": "center of",
        "expected_effect_verified": True,
        "authority": "deterministic fixture target-effect verifier",
    }
    passed = bool(
        hypothesis["preserve"] == _contract().to_dict()
        and realization.runtime_binding == prepared.runtime_binding
        and calls[-1].tool_name == "reobserve_scene"
        and after_step > before_step
        and fresh_effect_observation["expected_effect_verified"] is True
    )
    return {
        "schema": "roboharn_evo/rmbench/v31/fixture_c",
        "one_shot_feasibility": report.to_dict(),
        "realization_hypothesis": hypothesis.to_dict(),
        "staged_realization": realization.to_public_dict(),
        "runtime_goal_binding": prepared.runtime_binding.to_private_dict(),
        "mapped_existing_tool_calls": [
            {"tool_name": call.tool_name, "args": call.args} for call in calls
        ],
        "effect_observation_step_before": before_step,
        "effect_observation_step_after": after_step,
        "fresh_effect_observation": fresh_effect_observation,
        "scientific_boundary": (
            "This is a deterministic Runtime contract fixture; real physical "
            "utility is measured only in the paired boundary experiment."
        ),
        "passed": passed,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixtures = {
        "fixture_a": _fixture_a(),
        "fixture_b": _fixture_b(),
        "fixture_c": _fixture_c(),
    }
    for name, payload in fixtures.items():
        _write_json(output / f"{name}.json", payload)
    summary = {
        "schema": "roboharn_evo/rmbench/v31/contract_fixture_summary",
        "fixture_count": len(fixtures),
        "passed": all(payload["passed"] for payload in fixtures.values()),
        "fixtures": {
            name: {"path": f"{name}.json", "passed": payload["passed"]}
            for name, payload in fixtures.items()
        },
        "model_calls": 0,
        "simulator_actions": 0,
        "scientific_boundary": (
            "Fixtures validate deterministic wiring only; they are not utility results."
        ),
    }
    _write_json(output / "summary.json", summary)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
