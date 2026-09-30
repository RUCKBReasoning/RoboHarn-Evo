from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from policy.roboharn_evo.agent.core.img_agent import ImgAgent
from policy.roboharn_evo.agent.recovery.tool_specs import (
    RecoveryToolCall,
    RecoveryToolResult,
)
from policy.roboharn_evo.tests.test_debug_recovery_pure_control import (
    DummyConfig,
    make_card,
    make_snapshot,
)


_TRACE_HELD_WORLD_M = [0.102259, -0.104676, 0.759049]
_TRACE_HELD_EXTENT_M = [0.051997, 0.055568, 0.039987]
_TRACE_RELEASE_EE_POSE = [
    0.10339,
    -0.108535,
    0.896546,
    0.241169,
    0.662773,
    0.267203,
    -0.656637,
]


def _flat_instance(
    instance_id: str,
    world_m: list[float],
) -> dict:
    return {
        "instance_id": instance_id,
        "track_id": instance_id,
        "status": "visible",
        "stability": "stable",
        "position_state": "current_verified",
        "world_m": list(world_m),
        "latest_world_m": list(world_m),
        "first_observed_world_m": list(world_m),
        "top_surface_world_m": [world_m[0], world_m[1], 0.7428],
        "quality": {
            "world_extent_m": [0.079, 0.078, 0.0016],
            "world_z_max_m": 0.7428,
        },
    }


def _trace_scene() -> dict:
    return {
        "env_step": 27,
        "instances": [
            {
                "instance_id": "track_0001",
                "track_id": "track_0001",
                "status": "tracked",
                "stability": "missing_current_frame",
                "position_state": "memory_valid",
                "world_m": list(_TRACE_HELD_WORLD_M),
                "latest_world_m": list(_TRACE_HELD_WORLD_M),
                "first_observed_world_m": [
                    0.097478,
                    -0.203906,
                    0.775435,
                ],
                "quality": {
                    "world_extent_m": list(_TRACE_HELD_EXTENT_M),
                    "world_z_max_m": 0.782156,
                },
            },
            _flat_instance("mat_right", [0.201496, -0.098879, 0.742]),
            _flat_instance("mat_up", [0.101248, 0.000518, 0.742]),
            _flat_instance("mat_left", [0.001405, -0.098867, 0.742]),
            _flat_instance("mat_down", [0.10075, -0.227829, 0.742]),
        ],
    }


def _trace_agent(
    *,
    scene: dict | None = None,
    release_guard_enabled: bool = True,
) -> ImgAgent:
    agent = ImgAgent(
        make_card(
            DummyConfig(
                release_guard_enabled=release_guard_enabled,
            )
        )
    )
    snapshot = make_snapshot()
    snapshot.step_count = 27
    snapshot.right_endpose = np.asarray(
        _TRACE_RELEASE_EE_POSE,
        dtype=np.float32,
    )
    snapshot.raw.setdefault("endpose", {})
    snapshot.raw["endpose"]["right_gripper"] = 0.0
    snapshot.tcp_calibration_by_arm = {
        "right": {
            "translation_m": [0.12, 0.0, 0.0],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "source": "robot_kinematics",
        }
    }
    agent.latest_snapshot = snapshot
    agent.memory_store.reset(
        task="generic object relocation",
        control_model_name="test",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent.memory_store.state.working.manipulation_state = {
        "right": {
            "phase": "holding",
            "held_instance_id": "track_0001",
            "holding_confirmed": True,
            "transport_authorized": True,
            "operation_action_mode": "grasp",
            "grasp_candidate_id": "rgbd_volume:grasp:right:001",
            "grasp_attempt_step": 20,
            "grasp_attempt_nonce": (
                "right:track_0001:rgbd_volume:grasp:right:001:20"
            ),
            "held_object_to_tcp_attachment": {
                "object_proxy_frame": "tcp_aligned_at_capture",
                "object_centroid_to_tcp_translation_tcp_m": [
                    -0.017553,
                    -0.000672,
                    -0.001179,
                ],
                "capture_tcp_pose": [
                    0.097878,
                    -0.202645,
                    0.783381,
                    0.269261,
                    0.652902,
                    0.272071,
                    -0.653602,
                ],
                "grasp_candidate_id": "rgbd_volume:grasp:right:001",
                "grasp_attempt_nonce": (
                    "right:track_0001:rgbd_volume:grasp:right:001:20"
                ),
                "capture_step": 20,
                "source": "runtime_multiview_grasp_motion_verified",
                "authority": "runtime_multiview_grasp_motion",
            },
        }
    }
    agent.memory_store.record_scene_memory(scene or _trace_scene())
    return agent


def _settle_call(*, distance: float = 0.006) -> RecoveryToolCall:
    return RecoveryToolCall(
        tool_name="contact_displace",
        args={
            "arm": "right",
            "axis": "z",
            "direction": "negative",
            "distance": distance,
            "steps": 1,
            "gripper_precondition": "closed",
        },
    )


def _settle_result(**overrides) -> RecoveryToolResult:
    details = {
        "arm": "right",
        "axis": "z",
        "signed_distance": -0.006,
        "observed_displacement_xyz": [0.000347, -0.000723, 0.000115],
        "observed_displacement_m": 0.000810336,
        "observed_axis_displacement_m": 0.00011462,
        "observed_pose": list(_TRACE_RELEASE_EE_POSE),
    }
    details.update(overrides)
    return RecoveryToolResult(
        tool_name="contact_displace",
        success=True,
        details=details,
    )


def _record_settle(
    agent: ImgAgent,
    *,
    call: RecoveryToolCall | None = None,
    result: RecoveryToolResult | None = None,
) -> None:
    # Unit tests do not run the real SceneMemoryTracker.  Preserve the current
    # reconstructed scene across the runtime-event writeback that the tracker
    # would normally own during rollout.
    scene = deepcopy(agent.memory_store.state.working.scene_memory)
    agent._update_manipulation_state_from_effect(
        calls=[call or _settle_call()],
        results=[result or _settle_result()],
        action_effect={
            "effect_verified": "unverified",
            "effect_type": "place",
        },
    )
    agent.memory_store.record_scene_memory(
        agent._with_runtime_manipulation_state(scene)
    )


def test_trace_resisted_settle_authorizes_release_without_task_hardcoding() -> None:
    agent = _trace_agent()

    _record_settle(agent)

    state = agent.memory_store.state.working.manipulation_state["right"]
    assert state["release_ready"] is True
    assert state["support_contact_release"]["state"] == "ready"
    assert state["support_contact_release"]["candidate"]["target_kind"] == (
        "free_support"
    )
    assert state["pre_release_validated"] is True

    guarded = agent._with_internal_recovery_context(
        [
            _settle_call(),
            RecoveryToolCall(
                tool_name="open_gripper",
                args={
                    "arm": "right",
                    "release_held_instance_id": "track_0001",
                },
            ),
            RecoveryToolCall(
                tool_name="lift_ee",
                args={"arm": "right", "distance": 0.06, "steps": 2},
            ),
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
        ]
    )

    assert [call.tool_name for call in guarded] == [
        "open_gripper",
        "lift_ee",
        "reobserve_scene",
    ]
    assert guarded[0].args["_runtime_support_contact_release"] is True
    assert guarded[0].args["_release_place_validation"]["valid"] is True
    assert guarded[0].args["release_target_id"] == state["place_target_id"]

    agent._update_manipulation_state_from_effect(
        calls=guarded,
        results=[
            RecoveryToolResult(tool_name="open_gripper", success=True),
            RecoveryToolResult(tool_name="lift_ee", success=True),
            RecoveryToolResult(tool_name="reobserve_scene", success=True),
        ],
        action_effect={
            "effect_verified": "unverified",
            "effect_type": "place",
        },
    )
    pending = agent.memory_store.state.working.manipulation_state["right"]
    assert pending["phase"] == "release_pending_verification"
    assert "support_contact_release" not in pending


def test_disabled_release_guard_bare_open_preserves_pending_verification() -> None:
    agent = _trace_agent(release_guard_enabled=False)
    _record_settle(agent)

    ready = deepcopy(
        agent.memory_store.state.working.manipulation_state["right"]
    )
    assert ready["support_contact_release"]["state"] == "ready"
    bare_open = RecoveryToolCall(
        tool_name="open_gripper",
        args={"arm": "right"},
    )
    guarded = agent._with_internal_recovery_context([bare_open])
    assert guarded == [bare_open]
    assert guarded[0].args == {"arm": "right"}

    agent.latest_snapshot.raw["endpose"]["right_gripper"] = 1.0
    open_result = RecoveryToolResult(
        tool_name="open_gripper",
        success=True,
        details={"gripper_value": 1.0},
    )
    validation = agent._runtime_place_effect_validation(
        calls=guarded,
        results=[open_result],
    )
    assert validation["applicable"] is True
    assert validation["release_verification_required"] is True
    assert validation["held_instance_id"] == ready["held_instance_id"]
    assert validation["target_id"] == ready["place_target_id"]

    agent._update_manipulation_state_from_effect(
        calls=guarded,
        results=[open_result],
        action_effect={
            "effect_verified": "unverified",
            "effect_type": "release",
            "runtime_place_validation": validation,
        },
    )
    pending = deepcopy(
        agent.memory_store.state.working.manipulation_state["right"]
    )
    assert pending["phase"] == "release_pending_verification"
    assert pending["held_instance_id"] == ready["held_instance_id"]
    assert pending["place_target_id"] == ready["place_target_id"]
    assert pending["place_candidate_id"] == ready["place_candidate_id"]
    assert pending["held_object_target_world_m"] == ready[
        "held_object_target_world_m"
    ]
    assert "support_contact_release" not in pending

    repeated_validation = agent._runtime_place_effect_validation(
        calls=guarded,
        results=[open_result],
    )
    agent._update_manipulation_state_from_effect(
        calls=guarded,
        results=[open_result],
        action_effect={
            "effect_verified": "unverified",
            "effect_type": "release",
            "runtime_place_validation": repeated_validation,
        },
    )
    repeated = agent.memory_store.state.working.manipulation_state[
        "right"
    ]
    assert repeated["phase"] == "release_pending_verification"
    for key in (
        "held_instance_id",
        "place_target_id",
        "place_candidate_id",
        "held_object_target_world_m",
        "updated_step",
    ):
        assert repeated[key] == pending[key]


@pytest.mark.parametrize(
    ("scene_mutation", "result_overrides", "distance"),
    [
        (
            lambda scene: scene["instances"][0].update(
                {
                    "world_m": [0.102259, -0.104676, 0.80],
                    "latest_world_m": [0.102259, -0.104676, 0.80],
                }
            ),
            {"_remove_attachment": True},
            0.006,
        ),
        (
            lambda scene: scene["instances"].append(
                {
                    "instance_id": "occupied",
                    "track_id": "occupied",
                    "status": "visible",
                    "world_m": list(_TRACE_HELD_WORLD_M),
                    "latest_world_m": list(_TRACE_HELD_WORLD_M),
                    "first_observed_world_m": list(_TRACE_HELD_WORLD_M),
                    "quality": {
                        "world_extent_m": [0.04, 0.04, 0.04],
                        "world_z_max_m": 0.779,
                    },
                }
            ),
            {},
            0.006,
        ),
        (
            lambda scene: None,
            {
                "observed_displacement_xyz": [0.003, 0.0, 0.0],
                "observed_displacement_m": 0.003,
                "observed_axis_displacement_m": 0.0,
            },
            0.006,
        ),
        (
            lambda scene: None,
            {"collision": True},
            0.006,
        ),
        (
            lambda scene: None,
            {},
            0.04,
        ),
    ],
    ids=[
        "stalled_above_support_plane",
        "target_occupied",
        "lateral_motion_anomaly",
        "collision_reported",
        "unbounded_settle_command",
    ],
)
def test_release_ready_rejects_unsafe_or_unvalidated_settles(
    scene_mutation,
    result_overrides: dict,
    distance: float,
) -> None:
    scene = _trace_scene()
    scene_mutation(scene)
    agent = _trace_agent(scene=scene)
    result_overrides = dict(result_overrides)
    if result_overrides.pop("_remove_attachment", False):
        agent.memory_store.state.working.manipulation_state["right"].pop(
            "held_object_to_tcp_attachment",
            None,
        )
    result = _settle_result(**result_overrides)
    result.details["signed_distance"] = -distance

    _record_settle(
        agent,
        call=_settle_call(distance=distance),
        result=result,
    )

    state = agent.memory_store.state.working.manipulation_state["right"]
    assert state.get("release_ready") is not True
    assert "support_contact_release" not in state


def test_release_guard_revalidates_object_pose_after_ready_state() -> None:
    agent = _trace_agent()
    _record_settle(agent)
    scene = deepcopy(agent.memory_store.state.working.scene_memory)
    held = scene["instances"][0]
    held["world_m"] = [0.15, -0.104676, 0.759049]
    held["latest_world_m"] = [0.15, -0.104676, 0.759049]
    agent.memory_store.record_scene_memory(scene)
    moved_pose = np.asarray(_TRACE_RELEASE_EE_POSE, dtype=np.float32)
    moved_pose[0] += 0.05
    agent.latest_snapshot.right_endpose = moved_pose

    guarded = agent._with_internal_recovery_context(
        [
            RecoveryToolCall(
                tool_name="open_gripper",
                args={
                    "arm": "right",
                    "release_held_instance_id": "track_0001",
                },
            )
        ]
    )

    assert [call.tool_name for call in guarded] == ["reobserve_scene"]
    assert "same-batch final place_world_m" in guarded[0].args[
        "_guard_reason"
    ]


def test_supported_object_top_requires_verified_transport_attachment() -> None:
    scene = _trace_scene()
    support = {
        "instance_id": "support",
        "track_id": "support",
        "status": "visible",
        "stability": "stable",
        "position_state": "current_verified",
        "world_m": [0.1, -0.2, 0.01],
        "latest_world_m": [0.1, -0.2, 0.01],
        "first_observed_world_m": [0.1, -0.2, 0.01],
        "top_surface_world_m": [0.1, -0.2, 0.02],
        "quality": {
            "world_extent_m": [0.08, 0.08, 0.02],
            "world_z_max_m": 0.02,
        },
    }
    held = scene["instances"][0]
    held["world_m"] = [0.1, -0.2, 0.04]
    held["latest_world_m"] = [0.1, -0.2, 0.04]
    held["quality"]["world_extent_m"] = [0.04, 0.04, 0.04]
    scene["instances"].append(support)
    agent = _trace_agent(scene=scene)
    agent.memory_store.state.working.manipulation_state["right"].pop(
        "held_object_to_tcp_attachment",
        None,
    )

    _record_settle(agent)

    state = agent.memory_store.state.working.manipulation_state["right"]
    assert "support_contact_release" not in state
    assert state.get("release_ready") is not True
