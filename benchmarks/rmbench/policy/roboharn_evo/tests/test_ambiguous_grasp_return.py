from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from policy.roboharn_evo.agent.environment import EnvSnapshot
from policy.roboharn_evo.agent.recovery.ambiguous_grasp_return import (
    AmbiguousGraspReturnLimits,
    is_runtime_ambiguous_grasp_return_call,
    plan_ambiguous_grasp_return,
    reduce_ambiguous_grasp_return_results,
)
from policy.roboharn_evo.agent.recovery.tool_specs import (
    RecoveryToolCall,
    RecoveryToolResult,
)


_ARM = "right"
_INSTANCE = "track_0042"
_NONCE = "right:track_0042:candidate:7:31"
_GRASP = [0.12, -0.08, 0.90]
_APPROACH = [0.12, -0.08, 0.98]
_CURRENT = [0.12, -0.08, 0.925]
_QUAT = [1.0, 0.0, 0.0, 0.0]


def _state() -> dict:
    return {
        _ARM: {
            "phase": "grasp_candidate",
            "held_instance_id": _INSTANCE,
            "grasp_attempt_nonce": _NONCE,
            "holding_confirmed": False,
            "transport_authorized": False,
            "grasp_ee_target_world_m": list(_GRASP),
            "grasp_approach_world_m": list(_APPROACH),
            "diagnostic_lift_evidence": {
                "observed_lift_m": 0.025,
                "prelift_ee_world_m": list(_GRASP),
                "postlift_ee_world_m": list(_CURRENT),
                "lift_env_step": 31,
                "positive_attachment_evidence": False,
                "negative_attachment_evidence": False,
                "transport_authorized": False,
                "runtime_grasp_validation": {
                    "applicable": True,
                    "verified": None,
                    "arm": _ARM,
                    "held_instance_id": _INSTANCE,
                    "grasp_attempt_nonce": _NONCE,
                },
            },
        }
    }


def _robot(
    *,
    xyz: list[float] | None = None,
    quat: list[float] | None = None,
    gripper: float = 0.0,
) -> dict:
    return {
        _ARM: {
            "xyz": list(_CURRENT if xyz is None else xyz),
            "quat_wxyz": list(_QUAT if quat is None else quat),
            "gripper": gripper,
        }
    }


def _decision() -> dict:
    return {
        "allow_reobserve": False,
        "next_information_action": "perform_controlled_return_and_regrasp",
        "exhausted_reason": (
            "same_state_ambiguous_grasp_reobserve_budget_exhausted"
        ),
        "evidence_situation": "ambiguous_grasp",
        "stationary_reobserves_used": 1,
        "stationary_reobserve_limit": 1,
        "state_key": {
            "track_id": _INSTANCE,
            "arm": _ARM,
            "manipulation_phase": "grasp_candidate",
            "grasp_attempt_nonce": _NONCE,
            "quantized_ee_pose": [],
            "camera_set": ["head", "third"],
        },
    }


def _successful_results(
    calls: tuple[RecoveryToolCall, ...],
) -> list[RecoveryToolResult]:
    results: list[RecoveryToolResult] = []
    step = 32
    for call in calls:
        details: dict = {"step_count": step}
        if call.tool_name == "move_ee_to_pose":
            details.update(
                {
                    "arm": _ARM,
                    "target_reached": True,
                    "observed_pose": list(call.args["target_pose"]),
                }
            )
        elif call.tool_name == "open_gripper":
            details.update(
                {
                    "arm": _ARM,
                    "gripper_value": 1.0,
                }
            )
        results.append(
            RecoveryToolResult(
                tool_name=call.tool_name,
                success=True,
                details=details,
            )
        )
        if call.tool_name != "reobserve_scene":
            step += 1
    return results


def _post_snapshot(*, step: int = 34):
    return SimpleNamespace(
        step_count=step,
        right_endpose=list([*_APPROACH, *_QUAT]),
        left_endpose=[0.0, 0.0, 0.0, *_QUAT],
        raw={"endpose": {"right_gripper": 1.0}},
    )


def _snapshot_at(
    xyz: list[float],
    *,
    gripper: float,
    step: int,
):
    return SimpleNamespace(
        step_count=step,
        right_endpose=[*xyz, *_QUAT],
        left_endpose=[0.0, 0.0, 0.0, *_QUAT],
        raw={"endpose": {"right_gripper": gripper}},
    )


def _real_env_snapshot(
    *,
    arm: str,
    raw_endpose: dict,
    raw_joint_action: dict | None = None,
) -> EnvSnapshot:
    approach_pose = np.asarray([*_APPROACH, *_QUAT], dtype=np.float32)
    idle_pose = np.asarray(
        [0.0, 0.0, 0.80, *_QUAT],
        dtype=np.float32,
    )
    return EnvSnapshot(
        raw={
            "endpose": dict(raw_endpose),
            "joint_action": dict(raw_joint_action or {}),
        },
        head_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        left_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        right_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        joint_vector=np.zeros(8, dtype=np.float32),
        left_endpose=(approach_pose if arm == "left" else idle_pose),
        right_endpose=(approach_pose if arm == "right" else idle_pose),
        step_count=34,
        step_limit=150,
        eval_success=False,
        check_success=False,
        max_reward=0.0,
        instruction="generic manipulation",
    )


def test_plan_builds_exact_bounded_reverse_path_without_mutating_inputs() -> None:
    state = _state()
    robot = _robot()
    decision = _decision()
    before = deepcopy((state, robot, decision))

    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=robot,
        evidence_decision=decision,
    )

    assert plan.applicable is True
    assert plan.authorized is True
    assert plan.reason == "ambiguous_grasp_return_ready"
    assert plan.arm == _ARM
    assert plan.held_instance_id == _INSTANCE
    assert plan.grasp_attempt_nonce == _NONCE
    assert [call.tool_name for call in plan.calls] == [
        "move_ee_to_pose",
        "open_gripper",
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    assert [
        call.args["_runtime_ambiguous_grasp_return"]
        for call in plan.calls
    ] == [
        "return_to_grasp",
        "open_at_grasp",
        "retreat_to_approach",
        "observe_after_return",
    ]
    assert plan.calls[0].args["target_pose"] == [*_GRASP, *_QUAT]
    assert plan.calls[2].args["target_pose"] == [*_APPROACH, *_QUAT]
    assert plan.calls[0].args["max_translation"] <= 0.02
    assert plan.calls[2].args["max_translation"] <= 0.02
    assert plan.calls[0].args["steps"] == 2
    assert plan.calls[2].args["steps"] == 4
    release = plan.calls[1].args
    assert release["arm"] == _ARM
    assert release["release_held_instance_id"] == _INSTANCE
    assert release["grasp_attempt_nonce"] == _NONCE
    assert all(
        call.args["grasp_attempt_nonce"] == _NONCE
        and call.args["held_instance_id"] == _INSTANCE
        for call in plan.calls
    )
    assert (state, robot, decision) == before


def test_optional_observation_marker_can_be_omitted() -> None:
    plan = plan_ambiguous_grasp_return(
        manipulation_state=_state(),
        robot_state=_robot(),
        evidence_decision=_decision(),
        include_reobserve=False,
    )

    assert plan.authorized is True
    assert [call.tool_name for call in plan.calls] == [
        "move_ee_to_pose",
        "open_gripper",
        "move_ee_to_pose",
    ]


@pytest.mark.parametrize(
    ("mutate", "reason_fragment"),
    [
        (
            lambda state, robot, decision: state[_ARM].pop(
                "held_instance_id"
            ),
            "held_instance_id",
        ),
        (
            lambda state, robot, decision: state[_ARM].pop(
                "grasp_attempt_nonce"
            ),
            "grasp_attempt_nonce",
        ),
        (
            lambda state, robot, decision: state[_ARM].pop(
                "grasp_ee_target_world_m"
            ),
            "grasp pose",
        ),
        (
            lambda state, robot, decision: state[_ARM].pop(
                "grasp_approach_world_m"
            ),
            "approach pose",
        ),
        (
            lambda state, robot, decision: state[_ARM].pop(
                "diagnostic_lift_evidence"
            ),
            "diagnostic lift",
        ),
        (
            lambda state, robot, decision: state[_ARM][
                "diagnostic_lift_evidence"
            ].pop("prelift_ee_world_m"),
            "diagnostic lift completion",
        ),
        (
            lambda state, robot, decision: state[_ARM][
                "diagnostic_lift_evidence"
            ].pop("lift_env_step"),
            "diagnostic lift completion",
        ),
        (
            lambda state, robot, decision: robot[_ARM].update(
                {"gripper": 1.0}
            ),
            "closed gripper",
        ),
        (
            lambda state, robot, decision: robot[_ARM].update(
                {"quat_wxyz": [0.0, 0.0, 0.0, 0.0]}
            ),
            "quaternion",
        ),
        (
            lambda state, robot, decision: robot[_ARM].update(
                {"xyz": [0.15, -0.08, 0.925]}
            ),
            "corridor",
        ),
        (
            lambda state, robot, decision: robot[_ARM].update(
                {"xyz": [0.12, -0.08, 1.04]}
            ),
            "segment",
        ),
        (
            lambda state, robot, decision: state[_ARM].update(
                {"grasp_approach_world_m": [0.12, -0.08, 1.20]}
            ),
            "distance limit",
        ),
    ],
)
def test_missing_or_unsafe_inputs_fail_closed(mutate, reason_fragment) -> None:
    state = _state()
    robot = _robot()
    decision = _decision()
    mutate(state, robot, decision)

    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=robot,
        evidence_decision=decision,
    )

    assert plan.authorized is False
    assert plan.calls == ()
    assert reason_fragment in plan.reason


@pytest.mark.parametrize(
    "mutation",
    [
        lambda decision: decision.update({"allow_reobserve": True}),
        lambda decision: decision.update({"evidence_situation": "occlusion"}),
        lambda decision: decision.update({"exhausted_reason": ""}),
        lambda decision: decision["state_key"].update(
            {"track_id": "track_other"}
        ),
        lambda decision: decision["state_key"].update({"arm": "left"}),
        lambda decision: decision["state_key"].update(
            {"grasp_attempt_nonce": "stale-attempt"}
        ),
    ],
)
def test_budget_decision_must_match_the_exact_attempt(mutation) -> None:
    decision = _decision()
    mutation(decision)

    plan = plan_ambiguous_grasp_return(
        manipulation_state=_state(),
        robot_state=_robot(),
        evidence_decision=decision,
    )

    assert plan.authorized is False
    assert plan.calls == ()


@pytest.mark.parametrize("verdict", [True, False])
def test_confirmed_positive_or_negative_lift_is_not_ambiguous(
    verdict: bool,
) -> None:
    state = _state()
    diagnostic = state[_ARM]["diagnostic_lift_evidence"]
    diagnostic["runtime_grasp_validation"]["verified"] = verdict
    diagnostic[
        "positive_attachment_evidence"
        if verdict
        else "negative_attachment_evidence"
    ] = True

    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )

    assert plan.applicable is False
    assert plan.authorized is False
    assert plan.calls == ()


def test_multiple_exact_pending_attempts_fail_closed() -> None:
    state = _state()
    state["left"] = {
        **deepcopy(state[_ARM]),
        "held_instance_id": "track_left",
        "grasp_attempt_nonce": "left:track_left:candidate:2:31",
    }

    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )

    assert plan.authorized is False
    assert plan.calls == ()
    assert "more than one grasp_candidate" in plan.reason


def test_marker_query_requires_the_expected_tool_for_each_phase() -> None:
    plan = plan_ambiguous_grasp_return(
        manipulation_state=_state(),
        robot_state=_robot(),
        evidence_decision=_decision(),
    )

    assert all(
        is_runtime_ambiguous_grasp_return_call(call)
        for call in plan.calls
    )
    spoof = RecoveryToolCall(
        tool_name="open_gripper",
        args={
            **plan.calls[0].args,
            "_runtime_ambiguous_grasp_return": "return_to_grasp",
        },
    )
    assert is_runtime_ambiguous_grasp_return_call(spoof) is False
    assert is_runtime_ambiguous_grasp_return_call(
        RecoveryToolCall(tool_name="reobserve_scene", args={})
    ) is False


def test_successful_physical_return_and_fresh_snapshot_end_attempt() -> None:
    state = _state()
    state["left"] = {"phase": "idle-note"}
    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )

    reduction = reduce_ambiguous_grasp_return_results(
        state,
        calls=plan.calls,
        results=_successful_results(plan.calls),
        post_return_snapshot=_post_snapshot(),
    )

    assert reduction.physical_return_succeeded is True
    assert reduction.fresh_snapshot_confirmed is True
    assert reduction.attempt_completed is True
    assert reduction.reason == "ambiguous_grasp_return_completed"
    assert _ARM not in reduction.manipulation_state
    assert reduction.manipulation_state["left"] == {"phase": "idle-note"}
    assert _ARM in state


def test_fresh_physical_snapshot_can_complete_plan_without_observe_marker() -> None:
    state = _state()
    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
        include_reobserve=False,
    )

    reduction = reduce_ambiguous_grasp_return_results(
        state,
        calls=plan.calls,
        results=_successful_results(plan.calls),
        post_return_snapshot=_post_snapshot(),
    )

    assert reduction.physical_return_succeeded is True
    assert reduction.fresh_snapshot_confirmed is True
    assert reduction.attempt_completed is True
    assert _ARM not in reduction.manipulation_state


def test_result_reduction_requires_every_physical_phase_to_succeed() -> None:
    state = _state()
    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )
    results = _successful_results(plan.calls)
    results[1] = RecoveryToolResult(
        tool_name="open_gripper",
        success=False,
        details={"step_count": 33},
    )

    reduction = reduce_ambiguous_grasp_return_results(
        state,
        calls=plan.calls,
        results=results,
        post_return_snapshot=_post_snapshot(),
    )

    assert reduction.physical_return_succeeded is False
    assert reduction.attempt_completed is False
    assert reduction.manipulation_state[_ARM]["phase"] == (
        "ambiguous_grasp_return_pending"
    )
    assert reduction.manipulation_state[_ARM][
        "ambiguous_grasp_return"
    ]["status"] == "at_grasp_closed"
    assert state[_ARM]["phase"] == "grasp_candidate"


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        _post_snapshot(step=31),
        SimpleNamespace(
            step_count=34,
            right_endpose=[0.12, -0.08, 0.95, *_QUAT],
            raw={"endpose": {"right_gripper": 1.0}},
        ),
        SimpleNamespace(
            step_count=34,
            right_endpose=[*_APPROACH, *_QUAT],
            raw={"endpose": {"right_gripper": 0.0}},
        ),
    ],
)
def test_no_fresh_verified_post_return_snapshot_never_clears_state(
    snapshot,
) -> None:
    state = _state()
    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )

    reduction = reduce_ambiguous_grasp_return_results(
        state,
        calls=plan.calls,
        results=_successful_results(plan.calls),
        post_return_snapshot=snapshot,
    )

    assert reduction.physical_return_succeeded is True
    assert reduction.fresh_snapshot_confirmed is False
    assert reduction.attempt_completed is False
    assert reduction.manipulation_state[_ARM]["phase"] == (
        "ambiguous_grasp_return_pending"
    )
    assert reduction.manipulation_state[_ARM][
        "ambiguous_grasp_return"
    ]["status"] == "approach_reached_confirmation_pending"
    assert state[_ARM]["phase"] == "grasp_candidate"


def test_tampered_marker_identity_cannot_clear_attempt() -> None:
    state = _state()
    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )
    calls = list(plan.calls)
    calls[1] = RecoveryToolCall(
        tool_name=calls[1].tool_name,
        args={**calls[1].args, "grasp_attempt_nonce": "stale-attempt"},
    )

    reduction = reduce_ambiguous_grasp_return_results(
        state,
        calls=tuple(calls),
        results=_successful_results(tuple(calls)),
        post_return_snapshot=_post_snapshot(),
    )

    assert reduction.attempt_completed is False
    assert reduction.manipulation_state == state


def test_custom_limits_reject_excessive_diagnostic_return_distance() -> None:
    plan = plan_ambiguous_grasp_return(
        manipulation_state=_state(),
        robot_state=_robot(),
        evidence_decision=_decision(),
        limits=AmbiguousGraspReturnLimits(
            maximum_return_to_grasp_m=0.02,
        ),
    )

    assert plan.authorized is False
    assert "distance limit" in plan.reason


def test_open_then_partial_retreat_persists_resumable_exact_transaction() -> None:
    state = _state()
    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )
    results = _successful_results(plan.calls)
    results[2] = RecoveryToolResult(
        tool_name="move_ee_to_pose",
        success=True,
        details={
            "arm": _ARM,
            "step_count": 34,
            "target_reached": False,
            "observed_pose": [0.12, -0.08, 0.95, *_QUAT],
        },
    )
    results[3] = RecoveryToolResult(
        tool_name="reobserve_scene",
        success=False,
        details={"skipped": True},
    )

    partial = reduce_ambiguous_grasp_return_results(
        state,
        calls=plan.calls,
        results=results,
        post_return_snapshot=_snapshot_at(
            [0.12, -0.08, 0.95],
            gripper=1.0,
            step=34,
        ),
    )

    pending = partial.manipulation_state[_ARM]
    assert partial.attempt_completed is False
    assert pending["phase"] == "ambiguous_grasp_return_pending"
    transaction = pending["ambiguous_grasp_return"]
    assert transaction["status"] == "gripper_opened_retreat_pending"
    assert transaction["arm"] == _ARM
    assert transaction["held_instance_id"] == _INSTANCE
    assert transaction["grasp_attempt_nonce"] == _NONCE
    assert transaction["grasp_world_m"] == _GRASP
    assert transaction["approach_world_m"] == _APPROACH
    assert transaction["quat_wxyz"] == _QUAT

    resumed = plan_ambiguous_grasp_return(
        manipulation_state=partial.manipulation_state,
        robot_state=_robot(
            xyz=[0.12, -0.08, 0.95],
            gripper=1.0,
        ),
        # Resumption uses the persisted runtime authority and must not depend
        # on a planner manufacturing another budget decision.
        evidence_decision=None,
    )

    assert resumed.authorized is True
    assert [call.tool_name for call in resumed.calls] == [
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    assert [
        call.args["_runtime_ambiguous_grasp_return"]
        for call in resumed.calls
    ] == ["retreat_to_approach", "observe_after_return"]
    assert all(call.tool_name != "open_gripper" for call in resumed.calls)
    assert resumed.calls[0].args["target_pose"] == [*_APPROACH, *_QUAT]


def test_resumed_open_retreat_never_repeats_lower_or_open_and_can_finish() -> None:
    state = _state()
    initial = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )
    results = _successful_results(initial.calls)
    results[2] = RecoveryToolResult(
        tool_name="move_ee_to_pose",
        success=False,
        details={
            "arm": _ARM,
            "step_count": 34,
            "target_reached": False,
        },
    )
    results[3] = RecoveryToolResult(
        tool_name="reobserve_scene",
        success=False,
        details={"skipped": True},
    )
    partial = reduce_ambiguous_grasp_return_results(
        state,
        calls=initial.calls,
        results=results,
        post_return_snapshot=_snapshot_at(
            [0.12, -0.08, 0.95],
            gripper=1.0,
            step=34,
        ),
    )
    resumed = plan_ambiguous_grasp_return(
        manipulation_state=partial.manipulation_state,
        robot_state=_robot(
            xyz=[0.12, -0.08, 0.95],
            gripper=1.0,
        ),
        evidence_decision=None,
    )
    resumed_results = _successful_results(resumed.calls)
    resumed_results[0].details["step_count"] = 35
    resumed_results[1].details["step_count"] = 35

    completed = reduce_ambiguous_grasp_return_results(
        partial.manipulation_state,
        calls=resumed.calls,
        results=resumed_results,
        post_return_snapshot=_post_snapshot(step=35),
    )

    assert completed.physical_return_succeeded is True
    assert completed.fresh_snapshot_confirmed is True
    assert completed.attempt_completed is True
    assert _ARM not in completed.manipulation_state


def test_reached_retreat_can_finish_despite_optional_reobserve_failure() -> None:
    state = _state()
    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )
    results = _successful_results(plan.calls)
    results[3] = RecoveryToolResult(
        tool_name="reobserve_scene",
        success=False,
        details={"step_count": 34},
    )

    completed = reduce_ambiguous_grasp_return_results(
        state,
        calls=plan.calls,
        results=results,
        post_return_snapshot=_post_snapshot(step=34),
    )

    assert completed.physical_return_succeeded is True
    assert completed.fresh_snapshot_confirmed is True
    assert completed.attempt_completed is True
    assert _ARM not in completed.manipulation_state


def test_reached_retreat_without_snapshot_resumes_with_confirmation_only() -> None:
    state = _state()
    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )
    results = _successful_results(plan.calls)
    results[3] = RecoveryToolResult(
        tool_name="reobserve_scene",
        success=False,
        details={"step_count": 34},
    )
    pending = reduce_ambiguous_grasp_return_results(
        state,
        calls=plan.calls,
        results=results,
        post_return_snapshot=None,
    )

    arm_state = pending.manipulation_state[_ARM]
    assert arm_state["phase"] == "ambiguous_grasp_return_pending"
    assert arm_state["ambiguous_grasp_return"]["status"] == (
        "approach_reached_confirmation_pending"
    )
    confirmation = plan_ambiguous_grasp_return(
        manipulation_state=pending.manipulation_state,
        robot_state=_robot(xyz=_APPROACH, gripper=1.0),
        evidence_decision=None,
    )
    assert [call.tool_name for call in confirmation.calls] == [
        "reobserve_scene"
    ]
    assert confirmation.calls[0].args[
        "_runtime_ambiguous_grasp_return"
    ] == "observe_after_return"

    completed = reduce_ambiguous_grasp_return_results(
        pending.manipulation_state,
        calls=confirmation.calls,
        results=[
            RecoveryToolResult(
                tool_name="reobserve_scene",
                success=True,
                details={"step_count": 34},
            )
        ],
        post_return_snapshot=_post_snapshot(step=34),
    )
    assert completed.attempt_completed is True
    assert _ARM not in completed.manipulation_state


def test_open_pending_state_with_observed_closed_gripper_fails_closed() -> None:
    state = _state()
    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )
    results = _successful_results(plan.calls)
    results[2] = RecoveryToolResult(
        tool_name="move_ee_to_pose",
        success=False,
        details={
            "arm": _ARM,
            "step_count": 34,
            "target_reached": False,
        },
    )
    results[3] = RecoveryToolResult(
        tool_name="reobserve_scene",
        success=False,
        details={},
    )
    pending = reduce_ambiguous_grasp_return_results(
        state,
        calls=plan.calls,
        results=results,
        post_return_snapshot=None,
    )

    blocked = plan_ambiguous_grasp_return(
        manipulation_state=pending.manipulation_state,
        robot_state=_robot(xyz=[0.12, -0.08, 0.95], gripper=0.0),
        evidence_decision=None,
    )

    assert blocked.authorized is False
    assert blocked.calls == ()
    assert "open gripper" in blocked.reason


def test_open_failure_resumes_at_grasp_without_repeating_lowering() -> None:
    state = _state()
    initial = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )
    results = _successful_results(initial.calls)
    results[1] = RecoveryToolResult(
        tool_name="open_gripper",
        success=False,
        details={"step_count": 33},
    )
    results[2] = RecoveryToolResult(
        tool_name="move_ee_to_pose",
        success=False,
        details={"skipped": True},
    )
    results[3] = RecoveryToolResult(
        tool_name="reobserve_scene",
        success=False,
        details={"skipped": True},
    )
    pending = reduce_ambiguous_grasp_return_results(
        state,
        calls=initial.calls,
        results=results,
        post_return_snapshot=None,
    )

    resumed = plan_ambiguous_grasp_return(
        manipulation_state=pending.manipulation_state,
        robot_state=_robot(xyz=_GRASP, gripper=0.0),
        evidence_decision=None,
    )

    assert [call.tool_name for call in resumed.calls] == [
        "open_gripper",
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    assert resumed.calls[0].args[
        "_runtime_ambiguous_grasp_return"
    ] == "open_at_grasp"
    assert all(
        call.args.get("_runtime_ambiguous_grasp_return")
        != "return_to_grasp"
        for call in resumed.calls
    )


def test_confirmation_pending_can_finish_from_snapshot_without_marker() -> None:
    state = _state()
    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )
    pending = reduce_ambiguous_grasp_return_results(
        state,
        calls=plan.calls,
        results=_successful_results(plan.calls),
        post_return_snapshot=None,
    )
    markerless = plan_ambiguous_grasp_return(
        manipulation_state=pending.manipulation_state,
        robot_state=_robot(xyz=_APPROACH, gripper=1.0),
        evidence_decision=None,
        include_reobserve=False,
    )
    assert markerless.authorized is True
    assert markerless.calls == ()

    completed = reduce_ambiguous_grasp_return_results(
        pending.manipulation_state,
        calls=(),
        results=(),
        post_return_snapshot=_post_snapshot(step=34),
    )
    assert completed.attempt_completed is True
    assert _ARM not in completed.manipulation_state


def test_tampered_saved_pending_geometry_fails_closed() -> None:
    state = _state()
    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )
    pending = reduce_ambiguous_grasp_return_results(
        state,
        calls=plan.calls,
        results=_successful_results(plan.calls),
        post_return_snapshot=None,
    )
    pending.manipulation_state[_ARM]["ambiguous_grasp_return"][
        "approach_world_m"
    ] = [0.2, 0.2, 0.2]

    blocked = plan_ambiguous_grasp_return(
        manipulation_state=pending.manipulation_state,
        robot_state=_robot(xyz=_APPROACH, gripper=1.0),
        evidence_decision=None,
    )

    assert blocked.authorized is False
    assert blocked.calls == ()
    assert "geometry" in blocked.reason


@pytest.mark.parametrize(
    ("raw_endpose", "raw_joint_action"),
    [
        ({"right_gripper": 1.0}, {}),
        ({}, {"right_gripper": 1.0}),
    ],
)
def test_real_env_snapshot_ndarray_and_canonical_right_gripper_complete(
    raw_endpose: dict,
    raw_joint_action: dict,
) -> None:
    state = _state()
    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=_robot(),
        evidence_decision=_decision(),
    )

    reduction = reduce_ambiguous_grasp_return_results(
        state,
        calls=plan.calls,
        results=_successful_results(plan.calls),
        post_return_snapshot=_real_env_snapshot(
            arm="right",
            raw_endpose=raw_endpose,
            raw_joint_action=raw_joint_action,
        ),
    )

    assert reduction.attempt_completed is True
    assert reduction.fresh_snapshot_confirmed is True
    assert "right" not in reduction.manipulation_state


def test_real_single_arm_snapshot_uses_endpose_gripper_alias() -> None:
    state = _state()
    arm_state = state.pop("right")
    nonce = _NONCE.replace("right:", "left:", 1)
    arm_state["grasp_attempt_nonce"] = nonce
    validation = arm_state["diagnostic_lift_evidence"][
        "runtime_grasp_validation"
    ]
    validation["arm"] = "left"
    validation["grasp_attempt_nonce"] = nonce
    state["left"] = arm_state
    decision = _decision()
    decision["state_key"]["arm"] = "left"
    decision["state_key"]["grasp_attempt_nonce"] = nonce
    robot = {
        "left": {
            "xyz": list(_CURRENT),
            "quat_wxyz": list(_QUAT),
            "gripper": 0.0,
        }
    }
    plan = plan_ambiguous_grasp_return(
        manipulation_state=state,
        robot_state=robot,
        evidence_decision=decision,
    )
    results = _successful_results(plan.calls)
    for result in results:
        if result.tool_name != "reobserve_scene":
            result.details["arm"] = "left"

    reduction = reduce_ambiguous_grasp_return_results(
        state,
        calls=plan.calls,
        results=results,
        post_return_snapshot=_real_env_snapshot(
            arm="left",
            raw_endpose={"gripper": 1.0},
        ),
    )

    assert reduction.attempt_completed is True
    assert reduction.fresh_snapshot_confirmed is True
    assert "left" not in reduction.manipulation_state
