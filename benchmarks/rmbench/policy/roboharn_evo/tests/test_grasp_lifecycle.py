from __future__ import annotations

import math

import pytest

from policy.roboharn_evo.agent.grasp_lifecycle import (
    apply_close_time_grasp_transport_policy,
    apply_grasp_close_boundary_effect,
    is_runtime_grasp_diagnostic_lift_call,
    stage_grasp_verification_boundary,
)
from policy.roboharn_evo.agent.recovery.tool_specs import (
    RecoveryToolCall,
    RecoveryToolResult,
)


def _setup() -> RecoveryToolCall:
    return RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "arm": "right",
            "point_key": "grasp_world_m",
            "_operation_action_mode": "grasp",
            "instance_ref": "track_0001",
        },
    )


def _pending_state(*, diagnostic: dict | None = None) -> dict:
    state = {
        "phase": "grasp_candidate",
        "held_instance_id": "track_0001",
        "grasp_candidate_id": "candidate:right:001",
        "grasp_attempt_step": 7,
        "grasp_attempt_nonce": (
            "right:track_0001:candidate:right:001:7"
        ),
        "operation_action_mode": "grasp",
        "grasp_ee_target_world_m": [0.10, -0.20, 0.80],
        "grasp_approach_world_m": [0.13, -0.16, 0.82],
        "held_object_to_tcp_attachment": {
            "object_proxy_frame": "tcp_aligned_at_capture",
            "object_centroid_to_tcp_translation_tcp_m": [0.0, 0.0, 0.04],
            "capture_tcp_pose": [
                0.10,
                -0.20,
                0.80,
                0.5,
                -0.5,
                0.5,
                -0.5,
            ],
            "grasp_candidate_id": "candidate:right:001",
            "grasp_attempt_nonce": (
                "right:track_0001:candidate:right:001:7"
            ),
            "capture_step": 7,
            "source": "pending_lift_verification",
        },
        "holding_confirmed": False,
        "transport_authorized": False,
    }
    if diagnostic is not None:
        state["diagnostic_lift_evidence"] = diagnostic
    return {"right": state}


def _robot_state(
    *,
    xyz: list[float] | None = None,
    quaternion: list[float] | None = None,
) -> dict:
    return {
        "right": {
            "xyz": list(xyz or [0.10, -0.20, 0.80]),
            "quat_wxyz": list(quaternion or [0.5, -0.5, 0.5, -0.5]),
            "gripper": 0.0,
        }
    }


def test_atomic_close_and_lift_batch_is_split_at_close_boundary() -> None:
    staged = stage_grasp_verification_boundary(
        [
            _setup(),
            RecoveryToolCall(
                tool_name="close_gripper",
                args={"arm": "right"},
            ),
            RecoveryToolCall(
                tool_name="lift_ee",
                args={"arm": "right", "distance": 0.04},
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ],
        manipulation_state={},
    )

    assert [call.tool_name for call in staged] == [
        "move_ee_to_grounded_instance",
        "close_gripper",
        "reobserve_scene",
    ]
    assert staged[1].args["_runtime_grasp_close_boundary"] is True


def test_close_and_reobserve_batch_still_marks_verification_boundary() -> None:
    staged = stage_grasp_verification_boundary(
        [
            _setup(),
            RecoveryToolCall(
                tool_name="close_gripper",
                args={"arm": "right"},
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ],
        manipulation_state={},
    )

    assert [call.tool_name for call in staged] == [
        "move_ee_to_grounded_instance",
        "close_gripper",
        "reobserve_scene",
    ]
    assert staged[1].args["_runtime_grasp_close_boundary"] is True


def test_real_trace_shape_preserves_approach_and_grasp_ingress() -> None:
    approach = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "arm": "right",
            "point_key": "approach_world_m",
            "_operation_action_mode": "grasp",
            "instance_ref": "track_0001",
        },
    )
    staged = stage_grasp_verification_boundary(
        [
            approach,
            _setup(),
            RecoveryToolCall(
                tool_name="close_gripper",
                args={"arm": "right"},
            ),
            RecoveryToolCall(
                tool_name="lift_ee",
                args={"arm": "right", "distance": 0.05},
            ),
        ],
        manipulation_state={},
    )

    assert [call.tool_name for call in staged] == [
        "move_ee_to_grounded_instance",
        "move_ee_to_grounded_instance",
        "close_gripper",
        "reobserve_scene",
    ]
    assert staged[2].args["_runtime_grasp_close_boundary"] is True


def test_same_arm_motion_invalidates_preceding_grasp_setup() -> None:
    staged = stage_grasp_verification_boundary(
        [
            _setup(),
            RecoveryToolCall(
                tool_name="move_ee_to_pose",
                args={"arm": "right", "pose": [0.0] * 7},
            ),
            RecoveryToolCall(
                tool_name="close_gripper",
                args={"arm": "right"},
            ),
        ],
        manipulation_state={},
    )

    assert [call.tool_name for call in staged] == [
        "move_ee_to_grounded_instance",
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    assert "invalidated" in staged[-1].args["_guard_reason"]


def test_other_arm_motion_does_not_invalidate_grasp_setup() -> None:
    staged = stage_grasp_verification_boundary(
        [
            _setup(),
            RecoveryToolCall(
                tool_name="move_ee_to_pose",
                args={"arm": "left", "pose": [0.0] * 7},
            ),
            RecoveryToolCall(
                tool_name="close_gripper",
                args={"arm": "right"},
            ),
        ],
        manipulation_state={},
    )

    assert [call.tool_name for call in staged] == [
        "move_ee_to_grounded_instance",
        "move_ee_to_pose",
        "close_gripper",
        "reobserve_scene",
    ]
    assert staged[2].args["_runtime_grasp_close_boundary"] is True


def test_observation_boundary_is_never_reordered_behind_close() -> None:
    staged = stage_grasp_verification_boundary(
        [
            _setup(),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
            RecoveryToolCall(
                tool_name="close_gripper",
                args={"arm": "right"},
            ),
        ],
        manipulation_state={},
    )

    assert [call.tool_name for call in staged] == [
        "move_ee_to_grounded_instance",
        "reobserve_scene",
    ]


def test_pending_grasp_allows_only_bounded_diagnostic_lift() -> None:
    staged = stage_grasp_verification_boundary(
        [
            RecoveryToolCall(
                tool_name="move_to_home",
                args={"arm": "right"},
            ),
            RecoveryToolCall(
                tool_name="lift_ee",
                args={"arm": "right", "distance": 0.08, "steps": 9},
            ),
        ],
        manipulation_state=_pending_state(),
        robot_state=_robot_state(),
    )

    assert [call.tool_name for call in staged] == [
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    start = [0.10, -0.20, 0.80]
    target = staged[0].args["target_pose"][:3]
    displacement = [end - begin for begin, end in zip(start, target)]
    reverse_ingress = [0.03, 0.04, 0.02]
    assert math.dist(start, target) == pytest.approx(0.03)
    assert displacement == pytest.approx(
        [
            component / math.sqrt(0.03**2 + 0.04**2 + 0.02**2) * 0.03
            for component in reverse_ingress
        ]
    )
    assert staged[0].args["target_pose"][3:] == [
        0.5,
        -0.5,
        0.5,
        -0.5,
    ]
    assert staged[0].args["steps"] == 3
    assert staged[0].args["max_translation"] == pytest.approx(0.01)
    assert staged[0].args["_runtime_grasp_diagnostic_motion"] is True
    assert is_runtime_grasp_diagnostic_lift_call(staged[0]) is True


def test_pending_grasp_diagnostic_motion_fails_closed_off_ingress() -> None:
    staged = stage_grasp_verification_boundary(
        [
            RecoveryToolCall(
                tool_name="lift_ee",
                args={"arm": "right", "distance": 0.02},
            )
        ],
        manipulation_state=_pending_state(),
        robot_state=_robot_state(xyz=[0.16, -0.20, 0.80]),
    )

    assert [call.tool_name for call in staged] == ["reobserve_scene"]
    assert "unsafe" in staged[0].args["_guard_reason"]


def test_pending_grasp_diagnostic_motion_requires_exact_attempt_attachment() -> None:
    state = _pending_state()
    state["right"]["grasp_attempt_nonce"] = "different-attempt"

    staged = stage_grasp_verification_boundary(
        [
            RecoveryToolCall(
                tool_name="lift_ee",
                args={"arm": "right", "distance": 0.02},
            )
        ],
        manipulation_state=state,
        robot_state=_robot_state(),
    )

    assert [call.tool_name for call in staged] == ["reobserve_scene"]
    assert "unavailable or unsafe" in staged[0].args["_guard_reason"]


def test_planner_pose_move_cannot_spoof_diagnostic_motion_marker() -> None:
    staged = stage_grasp_verification_boundary(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_pose",
                args={
                    "arm": "right",
                    "target_pose": [
                        0.10,
                        -0.20,
                        0.83,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "_runtime_grasp_diagnostic_motion": True,
                },
            )
        ],
        manipulation_state=_pending_state(),
        robot_state=_robot_state(),
    )

    assert [call.tool_name for call in staged] == ["reobserve_scene"]
    assert "attachment is pending" in staged[0].args["_guard_reason"]


def test_explicit_contact_state_is_not_portable_grasp() -> None:
    state = _pending_state()
    state["right"]["operation_action_mode"] = "contact"
    request = RecoveryToolCall(
        tool_name="lift_ee",
        args={"arm": "right", "distance": 0.02},
    )

    staged = stage_grasp_verification_boundary(
        [request],
        manipulation_state=state,
        robot_state=_robot_state(),
    )

    assert staged == [request]


def test_pending_grasp_blocks_unverified_transport_motion() -> None:
    staged = stage_grasp_verification_boundary(
        [
            RecoveryToolCall(
                tool_name="move_to_home",
                args={"arm": "right"},
            )
        ],
        manipulation_state=_pending_state(),
    )

    assert [call.tool_name for call in staged] == ["reobserve_scene"]
    assert "attachment is pending" in staged[0].args["_guard_reason"]


def test_disabled_release_guard_passes_explicit_open_at_pending_grasp() -> None:
    request = RecoveryToolCall(
        tool_name="open_gripper",
        args={"arm": "right"},
    )

    staged = stage_grasp_verification_boundary(
        [request],
        manipulation_state=_pending_state(),
        release_guard_enabled=False,
    )

    assert staged == [request]


def test_close_boundary_rejects_model_only_completed_grasp() -> None:
    calls = stage_grasp_verification_boundary(
        [
            _setup(),
            RecoveryToolCall(
                tool_name="close_gripper",
                args={"arm": "right"},
            ),
            RecoveryToolCall(
                tool_name="lift_ee",
                args={"arm": "right", "distance": 0.02},
            ),
        ],
        manipulation_state={},
    )
    effect = apply_grasp_close_boundary_effect(
        {
            "effect_verified": "true",
            "effect_type": "grasp",
            "subtask_status": "completed",
        },
        calls=calls,
    )

    assert effect["effect_verified"] == "unverified"
    assert effect["subtask_status"] == "in_progress"
    assert effect["recommended_control"] == "retry"


def test_evidence_only_keeps_close_and_following_transport_in_one_batch() -> None:
    calls = [
        _setup(),
        RecoveryToolCall(
            tool_name="close_gripper",
            args={"arm": "right"},
        ),
        RecoveryToolCall(
            tool_name="move_to_home",
            args={"arm": "right"},
        ),
    ]

    staged = stage_grasp_verification_boundary(
        calls,
        manipulation_state={},
        grasp_transport_policy="evidence_only",
    )

    assert staged == calls
    assert all(
        "_runtime_grasp_close_boundary" not in call.args
        and "_runtime_grasp_diagnostic_motion" not in call.args
        for call in staged
    )


def test_evidence_only_does_not_block_pending_same_arm_motion() -> None:
    request = RecoveryToolCall(
        tool_name="move_to_home",
        args={"arm": "right"},
    )

    staged = stage_grasp_verification_boundary(
        [request],
        manipulation_state=_pending_state(),
        robot_state=_robot_state(),
        grasp_transport_policy="evidence_only",
    )

    assert staged == [request]


def test_evidence_only_close_state_is_authorized_but_not_confirmed() -> None:
    strict_state = _pending_state()["right"]

    state = apply_close_time_grasp_transport_policy(
        strict_state,
        grasp_transport_policy="evidence_only",
    )

    assert state["phase"] == "holding_provisional"
    assert state["holding_confirmed"] is False
    assert state["transport_authorized"] is True
    assert state["attachment_evidence_status"] == "unknown"
    assert state["grasp_transport_policy"] == "evidence_only"
    assert state["held_object_to_tcp_attachment"]["source"] == (
        "runtime_close_time_grasp_attachment_assumption"
    )
    assert state["held_object_to_tcp_attachment"]["authority"] == (
        "runtime_evidence_only_grasp_transport_policy"
    )


def test_evidence_only_close_effect_does_not_claim_verified_attachment() -> None:
    calls = [
        _setup(),
        RecoveryToolCall(
            tool_name="close_gripper",
            args={"arm": "right"},
        ),
    ]

    effect = apply_grasp_close_boundary_effect(
        {
            "effect_verified": "true",
            "effect_type": "grasp",
            "subtask_status": "completed",
        },
        calls=calls,
        grasp_transport_policy="evidence_only",
    )

    assert effect["effect_verified"] == "unverified"
    assert effect["subtask_status"] == "in_progress"
    assert effect["recommended_control"] == "continue"
    assert effect["authority"] == (
        "runtime_evidence_only_grasp_transport_policy"
    )


def test_evidence_only_close_effect_requires_successful_reached_setup() -> None:
    calls = [
        _setup(),
        RecoveryToolCall(
            tool_name="close_gripper",
            args={"arm": "right"},
        ),
    ]
    model_effect = {
        "effect_verified": "true",
        "effect_type": "grasp",
        "subtask_status": "completed",
    }

    effect = apply_grasp_close_boundary_effect(
        model_effect,
        calls=calls,
        results=[
            RecoveryToolResult(
                tool_name="move_ee_to_grounded_instance",
                success=True,
                details={"target_reached": False},
            ),
            RecoveryToolResult(
                tool_name="close_gripper",
                success=True,
            ),
        ],
        grasp_transport_policy="evidence_only",
    )

    assert effect == model_effect
