from __future__ import annotations

import copy

import numpy as np
import pytest

from policy.roboharn_evo.agent.grounded_target_pose_contract import (
    GroundedTargetPoseResolution,
    authorize_memory_valid_final_grounded_action,
    resolve_grounded_target_pose,
)


CURRENT_POSE = [0.4, -0.3, 0.9, 2.0, 0.0, 0.0, 0.0]


def _instance() -> dict[str, object]:
    return {
        "instance_id": "track_0001",
        "approach_world_m": [0.1, 0.2, 0.3],
        "approach_quat_wxyz": [0.0, 0.0, 0.0, 2.0],
        "grasp_world_m": [0.11, 0.21, 0.25],
        "grasp_quat_wxyz": [0.0, 2.0, 0.0, 0.0],
        "contact_world_m": [0.12, 0.22, 0.24],
        "contact_quat_wxyz": [0.0, 0.0, 2.0, 0.0],
        "place_world_m": [0.5, 0.4, 0.2],
        "place_quat_wxyz": [2.0, 0.0, 0.0, 0.0],
        "world_m": [0.13, 0.23, 0.23],
    }


def _assert_success(
    resolution: GroundedTargetPoseResolution,
) -> np.ndarray:
    assert resolution.success is True
    assert resolution.error is None
    assert resolution.target_pose is not None
    assert resolution.target_pose.dtype == np.float32
    assert resolution.target_pose.shape == (7,)
    return resolution.target_pose


def test_default_uses_normalized_grounded_quaternion_and_zero_offset() -> None:
    pose = _assert_success(
        resolve_grounded_target_pose(
            instance=_instance(),
            point_key="approach",
            current_pose=None,
        )
    )

    np.testing.assert_allclose(
        pose,
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        atol=1e-7,
    )


def test_point_alias_and_offset_match_adapter_semantics() -> None:
    pose = _assert_success(
        resolve_grounded_target_pose(
            instance=_instance(),
            point_key="grasp_point_world",
            offset_xyz=[0.01, -0.02, 0.03],
            current_pose=CURRENT_POSE,
        )
    )

    np.testing.assert_allclose(
        pose,
        [0.12, 0.19, 0.28, 0.0, 1.0, 0.0, 0.0],
        atol=1e-7,
    )


def test_offset_can_cancel_a_materialized_candidate_point_change() -> None:
    first_instance = _instance()
    second_instance = _instance()
    second_instance["grasp_world_m"] = [0.12, 0.21, 0.25]

    first = _assert_success(
        resolve_grounded_target_pose(
            instance=first_instance,
            point_key="grasp",
            offset_xyz=[0.02, 0.0, 0.0],
            target_quat_wxyz="current",
            current_pose=CURRENT_POSE,
        )
    )
    second = _assert_success(
        resolve_grounded_target_pose(
            instance=second_instance,
            point_key="grasp",
            offset_xyz=[0.01, 0.0, 0.0],
            target_quat_wxyz="current",
            current_pose=CURRENT_POSE,
        )
    )

    # Retry identity must compare the resolved target, not candidate and
    # modifier fields independently: these two commands execute identically.
    np.testing.assert_allclose(first, second, atol=1e-7)


@pytest.mark.parametrize(
    "point",
    ([0.1, 0.2], [0.1, 0.2, float("nan")], "not-a-vector"),
)
def test_point_must_be_finite_exactly_three_dimensional(point: object) -> None:
    instance = _instance()
    instance["approach_world_m"] = point

    resolution = resolve_grounded_target_pose(
        instance=instance,
        current_pose=CURRENT_POSE,
    )

    assert resolution.success is False
    assert resolution.target_pose is None
    assert resolution.error is not None
    assert resolution.error.code == "missing_grounded_point"
    assert resolution.error.details["point_key"] == "approach_world_m"


def test_missing_point_error_lists_materialized_instance_keys() -> None:
    resolution = resolve_grounded_target_pose(
        instance={"instance_id": "track_0007", "world_m": [0.0, 0.0, 0.0]},
        point_key="place",
        current_pose=CURRENT_POSE,
    )

    assert resolution.error is not None
    assert resolution.error.code == "missing_grounded_point"
    assert resolution.error.message == "scene instance has no finite place_world_m"
    assert resolution.error.details == {
        "instance_id": "track_0007",
        "point_key": "place_world_m",
        "available_keys": ["instance_id", "world_m"],
    }


@pytest.mark.parametrize(
    "offset",
    (
        None,
        [0.0, 0.0],
        [0.0, 0.0, float("inf")],
        [10**1000, 0.0, 0.0],
        "invalid",
    ),
)
def test_explicit_offset_must_be_finite_exactly_three_dimensional(
    offset: object,
) -> None:
    resolution = resolve_grounded_target_pose(
        instance=_instance(),
        offset_xyz=offset,
        current_pose=CURRENT_POSE,
    )

    assert resolution.error is not None
    assert resolution.error.code == "invalid_offset_xyz"
    assert resolution.error.message == "offset_xyz must be a finite 3D vector"


def test_preserve_height_uses_current_z_plus_offset_z() -> None:
    pose = _assert_success(
        resolve_grounded_target_pose(
            instance=_instance(),
            point_key="contact",
            offset_xyz=[0.01, -0.02, 0.04],
            preserve_height=True,
            target_quat_wxyz="grounded",
            current_pose=CURRENT_POSE,
        )
    )

    np.testing.assert_allclose(
        pose,
        [0.13, 0.20, 0.94, 0.0, 0.0, 1.0, 0.0],
        atol=1e-7,
    )


def test_preserve_height_requires_a_real_boolean() -> None:
    resolution = resolve_grounded_target_pose(
        instance=_instance(),
        preserve_height=1,
        current_pose=CURRENT_POSE,
    )

    assert resolution.error is not None
    assert resolution.error.code == "invalid_preserve_height"
    assert resolution.error.details == {"preserve_height": 1}


@pytest.mark.parametrize(
    "current_pose",
    (None, [0.0] * 6, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
)
def test_preserve_height_validates_the_complete_current_pose(
    current_pose: object,
) -> None:
    resolution = resolve_grounded_target_pose(
        instance=_instance(),
        preserve_height=True,
        target_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
        current_pose=current_pose,
    )

    assert resolution.error is not None
    assert resolution.error.code == "invalid_current_pose"


@pytest.mark.parametrize("mode", ("grounded", "scene", "affordance", "auto"))
def test_grounded_quaternion_modes_use_point_specific_orientation(
    mode: str,
) -> None:
    pose = _assert_success(
        resolve_grounded_target_pose(
            instance=_instance(),
            point_key="place",
            target_quat_wxyz=mode,
            current_pose=None,
        )
    )

    np.testing.assert_allclose(pose[3:7], [1.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("mode", ("grounded", "scene", "affordance", "auto"))
def test_forced_grounded_mode_reports_missing_or_invalid_quaternion(
    mode: str,
) -> None:
    instance = _instance()
    instance["grasp_quat_wxyz"] = [0.0, 0.0, 0.0, 0.0]

    resolution = resolve_grounded_target_pose(
        instance=instance,
        point_key="grasp",
        target_quat_wxyz=mode,
        current_pose=CURRENT_POSE,
    )

    assert resolution.error is not None
    assert resolution.error.code == "missing_grounded_quaternion"
    assert resolution.error.message == (
        "scene instance has no finite grasp_quat_wxyz"
    )
    assert resolution.error.details["point_key"] == "grasp_world_m"


@pytest.mark.parametrize("mode", ("", "preserve", "current"))
def test_current_modes_preserve_and_normalize_current_orientation(
    mode: str,
) -> None:
    pose = _assert_success(
        resolve_grounded_target_pose(
            instance=_instance(),
            target_quat_wxyz=mode,
            current_pose=CURRENT_POSE,
        )
    )

    np.testing.assert_allclose(pose[3:7], [1.0, 0.0, 0.0, 0.0])


def test_invalid_grounded_default_falls_back_to_current_orientation() -> None:
    instance = _instance()
    instance["approach_quat_wxyz"] = [0.0, 0.0, 0.0, 0.0]

    pose = _assert_success(
        resolve_grounded_target_pose(
            instance=instance,
            current_pose=CURRENT_POSE,
        )
    )

    np.testing.assert_allclose(pose[3:7], [1.0, 0.0, 0.0, 0.0])


def test_non_operation_point_defaults_to_current_orientation() -> None:
    pose = _assert_success(
        resolve_grounded_target_pose(
            instance=_instance(),
            point_key="world_m",
            current_pose=CURRENT_POSE,
        )
    )

    np.testing.assert_allclose(pose[:3], [0.13, 0.23, 0.23])
    np.testing.assert_allclose(pose[3:7], [1.0, 0.0, 0.0, 0.0])


def test_current_quaternion_mode_requires_finite_current_pose() -> None:
    resolution = resolve_grounded_target_pose(
        instance=_instance(),
        target_quat_wxyz="current",
        current_pose=[0.0, 0.0, 0.0, float("nan"), 0.0, 0.0, 1.0],
    )

    assert resolution.error is not None
    assert resolution.error.code == "invalid_current_pose"


def test_explicit_target_quaternion_is_normalized() -> None:
    pose = _assert_success(
        resolve_grounded_target_pose(
            instance=_instance(),
            target_quat_wxyz=[0.0, 3.0, 4.0, 0.0],
            current_pose=None,
        )
    )

    np.testing.assert_allclose(pose[3:7], [0.0, 0.6, 0.8, 0.0])


@pytest.mark.parametrize(
    "quaternion",
    ([0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0, float("nan")], "invalid"),
)
def test_explicit_target_quaternion_must_be_finite_nonzero_4d(
    quaternion: object,
) -> None:
    resolution = resolve_grounded_target_pose(
        instance=_instance(),
        target_quat_wxyz=quaternion,
        current_pose=CURRENT_POSE,
    )

    assert resolution.error is not None
    assert resolution.error.code == "invalid_target_quaternion"
    assert resolution.error.message == (
        "target_quat_wxyz must be a finite 4D quaternion"
    )


def test_quat_wxyz_alias_is_used_when_primary_argument_is_omitted() -> None:
    pose = _assert_success(
        resolve_grounded_target_pose(
            instance=_instance(),
            quat_wxyz=[0.0, 0.0, 2.0, 0.0],
            current_pose=None,
        )
    )

    np.testing.assert_allclose(pose[3:7], [0.0, 0.0, 1.0, 0.0])


def test_explicit_none_primary_quaternion_suppresses_alias_like_dict_get() -> None:
    pose = _assert_success(
        resolve_grounded_target_pose(
            instance=_instance(),
            target_quat_wxyz=None,
            quat_wxyz=[1.0, 0.0, 0.0, 0.0],
            current_pose=None,
        )
    )

    # Explicit target_quat_wxyz=None wins, so the grounded default is used.
    np.testing.assert_allclose(pose[3:7], [0.0, 0.0, 0.0, 1.0])


def test_resolution_does_not_mutate_instance_offset_or_current_pose() -> None:
    instance = _instance()
    offset = [0.01, 0.02, 0.03]
    current_pose = list(CURRENT_POSE)
    before = (copy.deepcopy(instance), list(offset), list(current_pose))

    _assert_success(
        resolve_grounded_target_pose(
            instance=instance,
            offset_xyz=offset,
            preserve_height=True,
            current_pose=current_pose,
        )
    )

    assert instance == before[0]
    assert offset == before[1]
    assert current_pose == before[2]


def test_non_mapping_instance_returns_structured_error() -> None:
    resolution = resolve_grounded_target_pose(  # type: ignore[arg-type]
        instance=None,
        current_pose=CURRENT_POSE,
    )

    assert resolution.error is not None
    assert resolution.error.code == "invalid_grounded_instance"
    assert resolution.error.details == {"instance_type": "NoneType"}


def test_memory_valid_final_action_requires_exact_private_candidate() -> None:
    candidate = {
        "candidate_id": "candidate:left:grasp:000",
        "arm": "left",
        "action_mode": "grasp",
        "approach_pose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        "ee_target_pose": [0.1, 0.2, 0.25, 1.0, 0.0, 0.0, 0.0],
    }
    raw = {
        "instance_id": "track_0001",
        "track_id": "track_0001",
        "operation_pose_candidates": [candidate],
    }
    materialized = {
        **raw,
        "position_state": "memory_valid",
        "position_source": "retained_verified_memory",
        "action_geometry_state": "verified",
        "grasp_world_m": [0.1, 0.2, 0.25],
        "selected_operation_candidate_id": candidate["candidate_id"],
    }

    authorization = authorize_memory_valid_final_grounded_action(
        enabled=True,
        raw_instance=raw,
        materialized_instance=materialized,
        arm="left",
        point_key="grasp_world_m",
        action_mode="grasp",
        candidate_id=candidate["candidate_id"],
        offset_xyz=[0.0, 0.0, 0.0],
    )

    assert authorization is not None
    assert authorization["candidate_id"] == candidate["candidate_id"]
    assert authorization["position_state"] == "memory_valid"


def test_memory_valid_final_action_does_not_relax_geometry_or_offset_identity() -> None:
    instance = {
        "instance_id": "track_0001",
        "position_state": "memory_valid",
        "action_geometry_state": "verified",
        "contact_world_m": [0.1, 0.2, 0.25],
    }

    assert authorize_memory_valid_final_grounded_action(
        enabled=False,
        raw_instance=instance,
        materialized_instance=instance,
        arm="left",
        point_key="contact_world_m",
        action_mode="contact",
    ) is None
    assert authorize_memory_valid_final_grounded_action(
        enabled=True,
        raw_instance=instance,
        materialized_instance=instance,
        arm="left",
        point_key="contact_world_m",
        action_mode="contact",
        offset_xyz=[0.01, 0.0, 0.0],
    ) is None
    assert authorize_memory_valid_final_grounded_action(
        enabled=True,
        raw_instance=instance,
        materialized_instance={
            **instance,
            "action_geometry_state": "unavailable",
        },
        arm="left",
        point_key="contact_world_m",
        action_mode="contact",
    ) is None
