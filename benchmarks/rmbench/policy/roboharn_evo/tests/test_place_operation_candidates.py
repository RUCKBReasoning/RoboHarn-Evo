from __future__ import annotations

import copy

import numpy as np

from policy.roboharn_evo.agent.operation_candidates import (
    capture_held_object_to_tcp_attachment,
    manipulation_state_allows_transport,
    operation_pose_candidates,
    propagate_held_object_world_m,
    select_operation_pose_candidate,
    spatial_state_from_scene,
    validate_place_candidate,
    with_dynamic_place_candidates,
)


def _instance(
    instance_id: str,
    xyz: list[float],
    *,
    first_xyz: list[float] | None = None,
) -> dict:
    return {
        "instance_id": instance_id,
        "track_id": instance_id,
        "status": "visible",
        "stability": "stable",
        "world_m": list(xyz),
        "latest_world_m": list(xyz),
        "first_observed_world_m": list(first_xyz or xyz),
        "top_surface_world_m": [xyz[0], xyz[1], xyz[2] + 0.02],
        "quality": {
            "actionable": True,
            "world_extent_m": [0.04, 0.04, 0.04],
            "world_z_max_m": xyz[2] + 0.02,
        },
        "operation_pose_candidates": [],
    }


def _held_scene() -> dict:
    return {
        "env_step": 12,
        "instances": [
            _instance(
                "held",
                [0.00, 0.00, 0.20],
                first_xyz=[0.00, 0.00, 0.02],
            ),
            _instance("anchor_a", [0.08, 0.00, 0.02]),
            _instance("anchor_b", [0.16, 0.00, 0.02]),
        ],
        "task_focus": {
            "target_instances": ["anchor_a", "anchor_b"],
            "tool_instances": ["held"],
        },
    }


def _verified_holding_state(
    attachment: dict | None = None,
) -> dict:
    candidate_id = "candidate:left:held:001"
    attempt_nonce = "left:held:candidate:left:held:001:12"
    if attachment is None:
        attachment = capture_held_object_to_tcp_attachment(
            object_world_m=[0.0, 0.0, 0.20],
            robot_arm_state={
                "xyz": [0.01, -0.01, 0.24],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        )
    assert attachment is not None
    return {
        "phase": "holding",
        "held_instance_id": "held",
        "holding_confirmed": True,
        "transport_authorized": True,
        "grasp_candidate_id": candidate_id,
        "grasp_attempt_step": 12,
        "grasp_attempt_nonce": attempt_nonce,
        "held_object_to_tcp_attachment": {
            **attachment,
            "grasp_candidate_id": candidate_id,
            "grasp_attempt_nonce": attempt_nonce,
            "capture_step": 12,
            "source": "runtime_multiview_grasp_motion_verified",
            "authority": "runtime_multiview_grasp_motion",
        },
    }


def _evidence_only_holding_state() -> dict:
    state = _verified_holding_state()
    state.update(
        {
            "phase": "holding_provisional",
            "holding_confirmed": False,
            "transport_authorized": True,
            "grasp_transport_policy": "evidence_only",
            "attachment_evidence_status": "unknown",
        }
    )
    state["held_object_to_tcp_attachment"].update(
        {
            "source": "runtime_close_time_grasp_attachment_assumption",
            "authority": "runtime_evidence_only_grasp_transport_policy",
        }
    )
    return state


def test_strong_single_fixed_camera_attachment_authorizes_transport() -> None:
    state = _verified_holding_state()
    attachment = state["held_object_to_tcp_attachment"]
    attachment["source"] = (
        "runtime_single_fixed_camera_grasp_motion_verified"
    )
    attachment["authority"] = (
        "runtime_single_fixed_camera_grasp_motion"
    )

    assert manipulation_state_allows_transport(state) is True


def test_exact_evidence_only_attachment_authorizes_transport_and_place() -> None:
    state = _evidence_only_holding_state()

    assert manipulation_state_allows_transport(state) is True
    scene = with_dynamic_place_candidates(
        _held_scene(),
        manipulation_state={"left": state},
        robot_state={
            "left": {
                "xyz": [0.01, -0.01, 0.24],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "gripper": 0.0,
            }
        },
    )

    assert scene["operation_targets"]
    assert all(
        target["holding_status"] == "provisional_evidence_only"
        and target["grasp_transport_policy"] == "evidence_only"
        for target in scene["operation_targets"]
    )
    held = next(
        item for item in scene["instances"] if item["instance_id"] == "held"
    )
    place_candidates = [
        item
        for item in operation_pose_candidates(held)
        if item["action_mode"] == "place"
    ]
    assert place_candidates
    assert all(
        item["holding_status"] == "provisional_evidence_only"
        for item in place_candidates
    )


def test_evidence_only_attachment_still_requires_exact_attempt_identity() -> None:
    state = _evidence_only_holding_state()
    state["grasp_attempt_nonce"] = "stale-attempt"

    assert manipulation_state_allows_transport(state) is False


def test_active_strict_policy_revokes_persisted_evidence_only_authority() -> None:
    state = _evidence_only_holding_state()

    assert manipulation_state_allows_transport(
        state,
        active_grasp_transport_policy="evidence_only",
    ) is True
    assert manipulation_state_allows_transport(
        state,
        active_grasp_transport_policy="strict",
    ) is False
    scene = with_dynamic_place_candidates(
        _held_scene(),
        manipulation_state={"left": state},
        robot_state={
            "left": {
                "xyz": [0.01, -0.01, 0.24],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "gripper": 0.0,
            }
        },
        active_grasp_transport_policy="strict",
    )

    assert scene["operation_targets"] == []
    held = next(
        item for item in scene["instances"] if item["instance_id"] == "held"
    )
    assert all(
        item.get("action_mode") != "place"
        for item in operation_pose_candidates(held)
    )


def test_dynamic_place_candidates_use_public_targets_and_private_arm_poses() -> None:
    scene = with_dynamic_place_candidates(
        _held_scene(),
        manipulation_state={"left": _verified_holding_state()},
        robot_state={
            "left": {
                "xyz": [0.01, -0.01, 0.24],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "gripper": 0.0,
            },
            "right": {
                "xyz": [0.30, 0.00, 0.25],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "gripper": 1.0,
            },
        },
    )

    public_targets = scene["operation_targets"]
    assert public_targets
    assert all("target_id" in item for item in public_targets)
    assert all("candidate_id" not in item for item in public_targets)
    assert {"vacated_pose", "free_support", "object_top"}.issubset(
        {item["target_kind"] for item in public_targets}
    )
    assert all(item["arm_options"] == ["left"] for item in public_targets)

    held = next(
        item for item in scene["instances"] if item["instance_id"] == "held"
    )
    candidates = [
        item
        for item in operation_pose_candidates(held)
        if item["action_mode"] == "place"
    ]
    assert candidates
    assert len({item["target_id"] for item in candidates}) == len(candidates)
    assert all(item["arm"] == "left" for item in candidates)
    assert all(item["valid"] is True for item in candidates)

    vacated_target = next(
        item["target_id"]
        for item in public_targets
        if item["target_kind"] == "vacated_pose"
    )
    selected = select_operation_pose_candidate(
        held,
        arm="left",
        action_mode="place",
        requested_target_id=vacated_target,
    )
    assert selected is not None
    assert selected["target_id"] == vacated_target

    # The command preserves the measured rigid attachment: moving the held
    # object by delta moves the action EE by the same delta.  It does not send
    # the EE to the object target center.
    current_object = np.asarray([0.00, 0.00, 0.20])
    current_ee = np.asarray([0.01, -0.01, 0.24])
    object_target = np.asarray(selected["held_object_target_world_m"])
    ee_target = np.asarray(selected["ee_target_pose"][:3])
    np.testing.assert_allclose(
        ee_target - current_ee,
        object_target - current_object,
        atol=1e-6,
    )
    assert not np.allclose(ee_target, object_target)


def test_provisional_lift_occlusion_hold_cannot_generate_place_targets() -> None:
    scene = with_dynamic_place_candidates(
        _held_scene(),
        manipulation_state={
            "left": {
                "phase": "holding_provisional",
                "held_instance_id": "held",
                "holding_confirmed": False,
                "transport_authorized": True,
                "held_object_to_tcp_attachment": {
                    "object_proxy_frame": "tcp_aligned_at_capture",
                    "object_centroid_to_tcp_translation_tcp_m": [
                        0.01,
                        -0.01,
                        0.04,
                    ],
                    "source": (
                        "reached_grasp_close_plus_bounded_lift_under_"
                        "proven_self_occlusion"
                    ),
                },
            }
        },
        robot_state={
            "left": {
                "xyz": [0.01, -0.01, 0.24],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "gripper": 0.0,
            }
        },
    )

    assert scene["operation_targets"] == []
    held = next(
        item for item in scene["instances"] if item["instance_id"] == "held"
    )
    place_candidates = [
        item
        for item in operation_pose_candidates(held)
        if item["action_mode"] == "place"
    ]
    assert place_candidates == []
    assert all(
        item["post_release_verification_required"] is True
        for item in place_candidates
    )


def test_place_candidate_revalidation_rejects_new_occupancy() -> None:
    scene = with_dynamic_place_candidates(
        _held_scene(),
        manipulation_state={"left": _verified_holding_state()},
        robot_state={
            "left": {
                "xyz": [0.01, -0.01, 0.24],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        },
    )
    held = next(
        item for item in scene["instances"] if item["instance_id"] == "held"
    )
    candidate = next(
        item
        for item in operation_pose_candidates(held)
        if item.get("target_kind") == "vacated_pose"
    )
    blocked_scene = copy.deepcopy(scene)
    target = candidate["held_object_target_world_m"]
    blocked_scene["instances"].append(_instance("new_occupant", list(target)))

    validation = validate_place_candidate(
        blocked_scene,
        held_instance=held,
        candidate=candidate,
    )

    assert validation["valid"] is False
    assert validation["free"] is False
    assert validation["occupied_by"] == ["new_occupant"]


def test_free_support_target_does_not_chase_the_held_object() -> None:
    state = {"left": _verified_holding_state()}
    robot = {
        "left": {
            "xyz": [0.01, -0.01, 0.24],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
    }
    first = with_dynamic_place_candidates(
        _held_scene(),
        manipulation_state=state,
        robot_state=robot,
    )
    moved_scene = _held_scene()
    moved_scene["instances"][0]["latest_world_m"] = [0.04, 0.03, 0.20]
    second = with_dynamic_place_candidates(
        moved_scene,
        manipulation_state=state,
        robot_state=robot,
    )
    first_free = {
        item["target_id"]: item["object_target_world_m"]
        for item in first["operation_targets"]
        if item["target_kind"] == "free_support"
    }
    second_free = {
        item["target_id"]: item["object_target_world_m"]
        for item in second["operation_targets"]
        if item["target_kind"] == "free_support"
    }

    assert first_free
    assert second_free == first_free


def test_place_candidate_propagates_verified_tcp_local_attachment_through_rotation() -> None:
    attachment = capture_held_object_to_tcp_attachment(
        object_world_m=[0.0, 0.0, 0.20],
        robot_arm_state={
            "xyz": [0.01, -0.01, 0.24],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    )
    assert attachment is not None
    scene = with_dynamic_place_candidates(
        _held_scene(),
        manipulation_state={
            "left": _verified_holding_state(attachment)
        },
        robot_state={
            "left": {
                "xyz": [0.10, 0.10, 0.30],
                "quat_wxyz": [0.0, 0.0, 0.0, 1.0],
            }
        },
    )
    held = next(
        item for item in scene["instances"] if item["instance_id"] == "held"
    )
    candidate = next(
        item
        for item in operation_pose_candidates(held)
        if item.get("target_kind") == "vacated_pose"
    )

    # 180 degrees around z rotates the captured local [+.01,-.01,+.04]
    # centroid-to-TCP offset to [-.01,+.01,+.04] in world coordinates.
    np.testing.assert_allclose(
        candidate["held_object_to_tcp_translation_world_m"],
        [-0.01, 0.01, 0.04],
        atol=1e-6,
    )
    assert (
        candidate["attachment_transform_source"]
        == "verifier_confirmed_tcp_local_attachment"
    )


def test_public_tcp_attachment_propagation_updates_held_object_world_position() -> None:
    attachment = capture_held_object_to_tcp_attachment(
        object_world_m=[0.0, 0.0, 0.20],
        robot_arm_state={
            "xyz": [0.01, -0.01, 0.24],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    )
    propagated = propagate_held_object_world_m(
        robot_arm_state={
            "xyz": [0.10, 0.10, 0.30],
            "quat_wxyz": [0.0, 0.0, 0.0, 1.0],
        },
        attachment=attachment,
    )

    np.testing.assert_allclose(
        propagated,
        [0.11, 0.09, 0.26],
        atol=1e-6,
    )


def test_spatial_state_signature_is_computed_from_actual_coordinates() -> None:
    scene = _held_scene()
    first = spatial_state_from_scene(scene)
    moved = copy.deepcopy(scene)
    moved["instances"][1]["latest_world_m"][0] += 0.03
    second = spatial_state_from_scene(moved)

    assert first["signature"] != second["signature"]
    assert first["order_by_x"] == ["held", "anchor_a", "anchor_b"]
    first_by_id = {item["instance_id"]: item for item in first["instances"]}
    second_by_id = {item["instance_id"]: item for item in second["instances"]}
    assert (
        second_by_id["anchor_a"]["world_m"]
        != first_by_id["anchor_a"]["world_m"]
    )
