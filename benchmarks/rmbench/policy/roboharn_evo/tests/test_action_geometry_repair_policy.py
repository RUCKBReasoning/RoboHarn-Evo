from __future__ import annotations

import pytest

from policy.roboharn_evo.agent.action_geometry_repair_policy import (
    normalize_action_geometry_repair_pending_policy,
    plan_repair_pending_safe_motion,
    repair_pending_retreat_is_safe,
)


def _instance(**updates):
    value = {
        "position_state": "current_verified",
        "latest_world_m": [0.10, 0.20, 0.75],
        "world_m": [0.09, 0.19, 0.75],
        "quality": {
            "world_z_max_m": 0.78,
            "world_extent_m": [0.04, 0.04, 0.06],
        },
        # These are deliberately nonsensical quarantined poses.  The safe
        # planner must never consult them.
        "approach_world_m": [9.0, 9.0, 9.0],
        "grasp_world_m": [8.0, 8.0, 8.0],
    }
    value.update(updates)
    return value


def test_policy_normalization_supports_three_modes_and_hyphen_alias() -> None:
    assert normalize_action_geometry_repair_pending_policy("strict") == (
        "strict"
    )
    assert normalize_action_geometry_repair_pending_policy("safe-motion") == (
        "safe_motion"
    )
    assert normalize_action_geometry_repair_pending_policy("disabled") == (
        "disabled"
    )
    with pytest.raises(ValueError):
        normalize_action_geometry_repair_pending_policy("unsafe")


def test_safe_motion_lifts_before_translating_and_ignores_stale_poses() -> None:
    plan = plan_repair_pending_safe_motion(
        instance=_instance(),
        current_pose=[0.0, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0],
        gripper_state="open",
        ee_to_tcp_m=0.12,
        approach_clearance_m=0.08,
    )

    assert plan is not None
    assert plan.phase == "vertical_clearance"
    assert plan.reference_source == "latest_world_m"
    assert plan.reference_world_m == (0.10, 0.20, 0.75)
    assert plan.safe_ee_z_m == pytest.approx(0.98)
    assert plan.target_pose[:3] == pytest.approx((0.0, 0.0, 0.98))
    assert max(abs(value) for value in plan.target_pose[:3]) < 2.0

    lateral = plan_repair_pending_safe_motion(
        instance=_instance(),
        current_pose=[0.0, 0.0, 1.00, 1.0, 0.0, 0.0, 0.0],
        gripper_state="open",
        ee_to_tcp_m=0.12,
        approach_clearance_m=0.08,
    )

    assert lateral is not None
    assert lateral.phase == "lateral_standoff"
    assert lateral.target_pose[2] == pytest.approx(1.00)
    assert lateral.translation_m == pytest.approx(0.08)


@pytest.mark.parametrize(
    "updates,gripper,current_z",
    [
        ({"position_state": "motion_uncertain"}, "open", 0.95),
        ({}, "closed", 0.95),
        ({"latest_world_m": None, "world_m": None}, "open", 0.95),
        ({}, "open", 0.85),
    ],
)
def test_safe_motion_fails_closed_without_executable_position_or_clearance(
    updates,
    gripper,
    current_z,
) -> None:
    plan = plan_repair_pending_safe_motion(
        instance=_instance(**updates),
        current_pose=[0.0, 0.0, current_z, 1.0, 0.0, 0.0, 0.0],
        gripper_state=gripper,
        ee_to_tcp_m=0.12,
        approach_clearance_m=0.08,
    )

    assert plan is None


def test_memory_valid_safe_motion_uses_last_verified_world_position() -> None:
    plan = plan_repair_pending_safe_motion(
        instance=_instance(
            position_state="memory_valid",
            latest_world_m=[0.90, 0.90, 0.75],
            last_verified_world_m=[0.11, 0.21, 0.75],
        ),
        current_pose=[0.0, 0.0, 1.00, 1.0, 0.0, 0.0, 0.0],
        gripper_state="open",
        ee_to_tcp_m=0.12,
        approach_clearance_m=0.08,
    )

    assert plan is not None
    assert plan.reference_source == "last_verified_world_m"
    assert plan.reference_world_m == (0.11, 0.21, 0.75)


def test_retreat_must_be_open_gripper_bounded_and_not_approach_object() -> None:
    instance = _instance()
    pose = [0.0, 0.20, 1.00, 1.0, 0.0, 0.0, 0.0]

    assert repair_pending_retreat_is_safe(
        instance=instance,
        current_pose=pose,
        gripper_state="open",
        axis="x",
        direction="negative",
        distance_m=0.03,
        maximum_distance_m=0.05,
    )
    assert not repair_pending_retreat_is_safe(
        instance=instance,
        current_pose=pose,
        gripper_state="open",
        axis="x",
        direction="positive",
        distance_m=0.03,
        maximum_distance_m=0.05,
    )
    assert not repair_pending_retreat_is_safe(
        instance=instance,
        current_pose=pose,
        gripper_state="closed",
        axis="x",
        direction="negative",
        distance_m=0.03,
        maximum_distance_m=0.05,
    )
    assert not repair_pending_retreat_is_safe(
        instance=instance,
        current_pose=pose,
        gripper_state="open",
        axis="z",
        direction="negative",
        distance_m=0.03,
        maximum_distance_m=0.05,
    )
