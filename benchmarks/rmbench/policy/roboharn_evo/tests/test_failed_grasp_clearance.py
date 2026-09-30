from __future__ import annotations

from copy import deepcopy

import numpy as np

from policy.roboharn_evo.agent.core.img_agent import ImgAgent
from policy.roboharn_evo.agent.recovery.failed_grasp_clearance import (
    apply_failed_grasp_clearance_results,
    stage_failed_grasp_clearance,
)
from policy.roboharn_evo.agent.recovery.rmbench_recovery_adapter import (
    RMBenchRecoveryAdapter,
)
from policy.roboharn_evo.agent.recovery.tool_specs import (
    RecoveryToolCall,
    RecoveryToolResult,
)
from policy.roboharn_evo.tests.test_debug_recovery_pure_control import (
    DummyConfig,
    make_card,
    make_snapshot,
)
from policy.roboharn_evo.tests.test_grounded_recovery_tools import (
    FakeRMBenchEnv,
    snapshot_for,
)


_GRASP = [0.113073, -0.213587, 0.909201]
_APPROACH = [0.112277, -0.210576, 0.989141]
_CURRENT = [0.11238, -0.21430, 0.93108]
_QUATERNION = [0.29681, 0.64878, 0.26261, -0.64963]
_ATTEMPT_NONCE = "right:track_0001:rgbd_volume:grasp:right:001:12"


def _state(*, verified: bool | None = False) -> dict:
    return {
        "right": {
            "phase": "grasp_candidate",
            "held_instance_id": "track_0001",
            "holding_confirmed": False,
            "transport_authorized": False,
            "grasp_candidate_id": "rgbd_volume:grasp:right:001",
            "grasp_attempt_step": 12,
            "grasp_attempt_nonce": _ATTEMPT_NONCE,
            "grasp_ee_target_world_m": list(_GRASP),
            "grasp_approach_world_m": list(_APPROACH),
            "diagnostic_lift_evidence": {
                "runtime_grasp_validation": {
                    "applicable": True,
                    "verified": verified,
                    "arm": "right",
                    "held_instance_id": "track_0001",
                    "grasp_candidate_id": (
                        "rgbd_volume:grasp:right:001"
                    ),
                    "grasp_attempt_nonce": _ATTEMPT_NONCE,
                    **(
                        {
                            "negative_camera_count": 2,
                            "failure_kind": (
                                "object_stationary_during_lift"
                            ),
                        }
                        if verified is False
                        else {}
                    ),
                }
            },
        }
    }


def _robot(*, gripper: float = 0.0, xyz: list[float] | None = None) -> dict:
    return {
        "right": {
            "xyz": list(_CURRENT if xyz is None else xyz),
            "quat_wxyz": list(_QUATERNION),
            "gripper": gripper,
        }
    }


def _calls(*, release_target: bool = False) -> list[RecoveryToolCall]:
    open_args = {"arm": "right", "release_held_instance_id": "track_0001"}
    if release_target:
        open_args["release_target_id"] = "place:mat"
    return [
        RecoveryToolCall(tool_name="open_gripper", args=open_args),
        RecoveryToolCall(tool_name="reobserve_scene", args={}),
    ]


def test_v4_true_failed_grasp_reverses_verified_ingress_before_reobserve() -> None:
    calls = _calls()
    state = _state()
    robot = _robot()
    calls_copy = deepcopy(calls)
    state_copy = deepcopy(state)
    robot_copy = deepcopy(robot)

    staged = stage_failed_grasp_clearance(
        calls,
        manipulation_state=state,
        robot_state=robot,
    )

    assert [call.tool_name for call in staged] == [
        "open_gripper",
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    retreat = staged[1]
    assert retreat.args["arm"] == "right"
    assert retreat.args["_runtime_failed_grasp_clearance"] == "move"
    target = np.asarray(retreat.args["target_pose"][:3], dtype=np.float64)
    current = np.asarray(_CURRENT, dtype=np.float64)
    reverse_ingress = np.asarray(_APPROACH) - np.asarray(_GRASP)
    displacement = target - current
    cosine = float(
        np.dot(displacement, reverse_ingress)
        / (np.linalg.norm(displacement) * np.linalg.norm(reverse_ingress))
    )
    assert cosine > 0.999
    assert 0.049 <= np.linalg.norm(displacement) <= 0.050001
    np.testing.assert_allclose(
        retreat.args["target_pose"][3:7],
        _QUATERNION,
        atol=1e-8,
    )
    assert retreat.args["steps"] == 3
    assert retreat.args["max_translation"] == 0.016667
    assert calls == calls_copy
    assert state == state_copy
    assert robot == robot_copy


def test_evidence_only_never_converts_visual_failure_into_clearance() -> None:
    planner_open = RecoveryToolCall(
        tool_name="open_gripper",
        args={
            "arm": "right",
            "release_held_instance_id": "track_0001",
            "_runtime_failed_grasp_clearance": "spoofed",
        },
    )

    staged = stage_failed_grasp_clearance(
        [planner_open],
        manipulation_state=_state(),
        robot_state=_robot(),
        grasp_transport_policy="evidence_only",
    )

    assert [call.tool_name for call in staged] == ["open_gripper"]
    assert "_runtime_failed_grasp_clearance" not in staged[0].args


def test_disabled_release_guard_never_takes_over_explicit_open() -> None:
    calls = _calls()

    staged = stage_failed_grasp_clearance(
        calls,
        manipulation_state=_state(),
        robot_state=_robot(),
        grasp_transport_policy="strict",
        release_guard_enabled=False,
    )

    assert [call.tool_name for call in staged] == [
        "open_gripper",
        "reobserve_scene",
    ]
    assert staged[0].args == calls[0].args


def test_ambiguous_or_confirmed_grasp_never_stages_failed_clearance() -> None:
    for verified in (None, True):
        calls = _calls()
        staged = stage_failed_grasp_clearance(
            calls,
            manipulation_state=_state(verified=verified),
            robot_state=_robot(),
        )
        assert [call.tool_name for call in staged] == [
            "reobserve_scene"
        ]
        assert "runtime-proven two-view failure" in staged[0].args[
            "_guard_reason"
        ]


def test_failed_clearance_fails_closed_outside_recorded_ingress_corridor() -> None:
    calls = _calls()
    staged = stage_failed_grasp_clearance(
        calls,
        manipulation_state=_state(),
        robot_state=_robot(xyz=[0.20, -0.21430, 0.93108]),
    )
    assert [call.tool_name for call in staged] == ["reobserve_scene"]
    assert "recorded grasp-ingress corridor" in staged[0].args[
        "_guard_reason"
    ]


def test_planner_place_fields_cannot_bypass_pending_grasp_clearance() -> None:
    calls = _calls(release_target=True)
    staged = stage_failed_grasp_clearance(
        calls,
        manipulation_state=_state(),
        robot_state=_robot(),
    )
    assert [call.tool_name for call in staged] == [
        "open_gripper",
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    assert "release_target_id" not in staged[0].args


def test_failed_clearance_requires_closed_gripper_and_normalizes_batch() -> None:
    calls = _calls()
    assert stage_failed_grasp_clearance(
        calls,
        manipulation_state=_state(),
        robot_state=_robot(gripper=1.0),
    )[0].tool_name == "reobserve_scene"

    extended = [
        RecoveryToolCall(
            tool_name="lift_ee",
            args={"arm": "right", "distance": 0.02},
        ),
        *calls,
    ]
    normalized = stage_failed_grasp_clearance(
        extended,
        manipulation_state=_state(),
        robot_state=_robot(),
    )
    assert [call.tool_name for call in normalized] == [
        "open_gripper",
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    assert all(call.tool_name != "lift_ee" for call in normalized)


def test_runtime_negative_lift_evidence_stages_clearance_through_img_agent() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    snapshot = make_snapshot()
    snapshot.step_count = 12
    snapshot.right_endpose = np.asarray(
        [*_CURRENT, *_QUATERNION],
        dtype=np.float32,
    )
    snapshot.raw.setdefault("endpose", {})["right_gripper"] = 0.0
    agent.latest_snapshot = snapshot
    state = _state()["right"]
    state.update(
        {
            "operation_action_mode": "grasp",
            "held_object_to_tcp_attachment": {
                "object_proxy_frame": "tcp_aligned_at_capture",
                "object_centroid_to_tcp_translation_tcp_m": [
                    0.0,
                    0.0,
                    0.12,
                ],
            },
        }
    )
    state.pop("diagnostic_lift_evidence")
    agent.memory_store.state.working.manipulation_state = {"right": state}

    agent._update_manipulation_state_from_effect(
        calls=[
            RecoveryToolCall(
                tool_name="lift_ee",
                args={"arm": "right", "distance": 0.025},
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ],
        results=[
            RecoveryToolResult(
                tool_name="lift_ee",
                success=True,
                details={
                    "arm": "right",
                    "axis": "z",
                    "signed_distance": 0.025,
                    "observed_axis_displacement_m": 0.02167,
                    "observed_displacement_xyz": [
                        0.000229,
                        -0.000319,
                        0.021669,
                    ],
                    "observed_pose": [*_CURRENT, *_QUATERNION],
                },
            ),
            RecoveryToolResult(
                tool_name="reobserve_scene",
                success=True,
            ),
        ],
        action_effect={
            "effect_verified": "false",
            "effect_type": "grasp",
            "runtime_grasp_validation": {
                "applicable": True,
                "verified": False,
                "arm": "right",
                "held_instance_id": "track_0001",
                "grasp_candidate_id": "rgbd_volume:grasp:right:001",
                "grasp_attempt_nonce": _ATTEMPT_NONCE,
                "negative_camera_count": 2,
                "failure_kind": "object_stationary_during_lift",
            },
        },
    )

    pending = agent.memory_store.state.working.manipulation_state["right"]
    validation = pending["diagnostic_lift_evidence"][
        "runtime_grasp_validation"
    ]
    assert validation["verified"] is False
    assert validation["negative_camera_count"] == 2

    staged = agent._with_internal_recovery_context(_calls())

    assert [call.tool_name for call in staged] == [
        "open_gripper",
        "move_ee_to_pose",
        "reobserve_scene",
    ]
    assert staged[1].args["_runtime_failed_grasp_clearance"] == "move"


def test_planner_cannot_spoof_failed_grasp_clearance_marker() -> None:
    agent = ImgAgent(make_card(DummyConfig()))
    agent.latest_snapshot = make_snapshot()

    guarded = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="move_ee_to_pose",
                args={
                    "arm": "right",
                    "target_pose": [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
                    "_runtime_failed_grasp_clearance": True,
                },
            )
        ]
    )

    assert len(guarded) == 1
    assert "_runtime_failed_grasp_clearance" not in guarded[0].args


def test_open_success_move_failure_preserves_clearance_obligation() -> None:
    staged = stage_failed_grasp_clearance(
        _calls(),
        manipulation_state=_state(),
        robot_state=_robot(),
    )
    updated = apply_failed_grasp_clearance_results(
        _state(),
        calls=staged,
        results=[
            RecoveryToolResult(tool_name="open_gripper", success=True),
            RecoveryToolResult(
                tool_name="move_ee_to_pose",
                success=False,
                details={"target_reached": False},
            ),
            RecoveryToolResult(tool_name="reobserve_scene", success=True),
        ],
        env_step=13,
    )

    state = updated["right"]
    assert state["phase"] == "failed_grasp_clearance_pending"
    assert state["failed_grasp_clearance_status"] == "gripper_opened"
    assert state["holding_confirmed"] is False
    retry = stage_failed_grasp_clearance(
        [RecoveryToolCall(tool_name="reobserve_scene", args={})],
        manipulation_state=updated,
        robot_state=_robot(gripper=1.0),
    )
    assert [call.tool_name for call in retry] == [
        "move_ee_to_pose",
        "reobserve_scene",
    ]


def test_clearance_state_clears_only_after_reached_move_and_observation() -> None:
    staged = stage_failed_grasp_clearance(
        _calls(),
        manipulation_state=_state(),
        robot_state=_robot(),
    )
    after_move = apply_failed_grasp_clearance_results(
        _state(),
        calls=staged,
        results=[
            RecoveryToolResult(tool_name="open_gripper", success=True),
            RecoveryToolResult(
                tool_name="move_ee_to_pose",
                success=True,
                details={"target_reached": True},
            ),
            RecoveryToolResult(tool_name="reobserve_scene", success=False),
        ],
        env_step=13,
    )
    assert after_move["right"]["failed_grasp_clearance_status"] == (
        "clearance_reached"
    )
    observe_only = stage_failed_grasp_clearance(
        [RecoveryToolCall(tool_name="lift_ee", args={"arm": "left"})],
        manipulation_state=after_move,
        robot_state=_robot(gripper=1.0),
    )
    assert [call.tool_name for call in observe_only] == [
        "reobserve_scene"
    ]
    cleared = apply_failed_grasp_clearance_results(
        after_move,
        calls=observe_only,
        results=[
            RecoveryToolResult(tool_name="reobserve_scene", success=True)
        ],
        env_step=14,
    )
    assert "right" not in cleared


def test_reobserve_ablation_uses_fresh_clearance_motion_snapshot() -> None:
    staged = stage_failed_grasp_clearance(
        _calls(),
        manipulation_state=_state(),
        robot_state=_robot(),
    )
    physical_only = staged[:2]

    cleared = apply_failed_grasp_clearance_results(
        _state(),
        calls=physical_only,
        results=[
            RecoveryToolResult(tool_name="open_gripper", success=True),
            RecoveryToolResult(
                tool_name="move_ee_to_pose",
                success=True,
                details={"target_reached": True},
            ),
        ],
        env_step=13,
        fresh_physical_snapshot_satisfies_observation=True,
    )

    assert "right" not in cleared


def test_observed_closed_gripper_retries_release_before_clearance_motion() -> None:
    staged = stage_failed_grasp_clearance(
        _calls(),
        manipulation_state=_state(),
        robot_state=_robot(),
    )
    pending = apply_failed_grasp_clearance_results(
        _state(),
        calls=staged,
        results=[
            RecoveryToolResult(tool_name="open_gripper", success=True),
            RecoveryToolResult(
                tool_name="move_ee_to_pose",
                success=False,
                details={"target_reached": False},
            ),
            RecoveryToolResult(tool_name="reobserve_scene", success=True),
        ],
        env_step=13,
    )

    retry = stage_failed_grasp_clearance(
        [RecoveryToolCall(tool_name="reobserve_scene", args={})],
        manipulation_state=pending,
        robot_state=_robot(gripper=0.0),
    )

    assert [call.tool_name for call in retry] == [
        "open_gripper",
        "reobserve_scene",
    ]
    assert retry[0].args["_runtime_failed_grasp_clearance"] == (
        "release"
    )


def test_clearance_adapter_executes_bounded_reverse_ingress_steps() -> None:
    right_state = _state()["right"]
    left_state = {
        **right_state,
        "grasp_candidate_id": "rgbd_volume:grasp:left:001",
        "diagnostic_lift_evidence": {
            "runtime_grasp_validation": {
                **right_state["diagnostic_lift_evidence"][
                    "runtime_grasp_validation"
                ],
                "arm": "left",
                "grasp_candidate_id": "rgbd_volume:grasp:left:001",
            }
        },
    }
    staged = stage_failed_grasp_clearance(
        [
            RecoveryToolCall(
                tool_name="open_gripper",
                args={
                    "arm": "left",
                    "release_held_instance_id": "track_0001",
                },
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ],
        manipulation_state={"left": left_state},
        robot_state={
            "left": {
                "xyz": list(_CURRENT),
                "quat_wxyz": list(_QUATERNION),
                "gripper": 0.0,
            }
        },
    )
    clearance = staged[1]
    env = FakeRMBenchEnv()
    env.left_pose = np.asarray(
        [*_CURRENT, *_QUATERNION],
        dtype=np.float32,
    )
    adapter = RMBenchRecoveryAdapter(env)

    result = adapter.move_ee_to_pose(
        snapshot_for(env),
        clearance.args,
    )

    assert result.result.success is True
    assert len(env.actions) == 3
    positions = [
        np.asarray(item["action"][:3], dtype=np.float64)
        for item in env.actions
    ]
    distances = [
        float(np.linalg.norm(position - np.asarray(_CURRENT)))
        for position in positions
    ]
    np.testing.assert_allclose(
        distances,
        [0.05 / 3.0, 0.10 / 3.0, 0.05],
        atol=2e-5,
    )
    assert max(
        float(np.linalg.norm(positions[index] - positions[index - 1]))
        for index in range(1, len(positions))
    ) <= 0.020001
