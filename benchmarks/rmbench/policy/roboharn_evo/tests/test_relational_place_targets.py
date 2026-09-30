from __future__ import annotations

import copy

import numpy as np

from policy.roboharn_evo.agent.operation_candidates import (
    operation_pose_candidates,
    validate_place_candidate,
    with_dynamic_place_candidates,
)
from policy.roboharn_evo.agent.perception.scene_memory import SceneMemoryTracker
from policy.roboharn_evo.agent.relational_place_targets import (
    REFERENCE_REGIONS_KEY,
    ReferenceRegionTracker,
)


def _instance(
    instance_id: str,
    xyz: list[float],
    extent: list[float],
    *,
    class_name: str,
    first_xyz: list[float] | None = None,
) -> dict:
    return {
        "instance_id": instance_id,
        "track_id": instance_id,
        "class": class_name,
        "source_object_id": class_name,
        "status": "visible",
        "stability": "stable",
        "position_state": "current_verified",
        "world_m": list(xyz),
        "latest_world_m": list(xyz),
        "first_observed_world_m": list(first_xyz or xyz),
        "top_surface_world_m": [
            xyz[0],
            xyz[1],
            xyz[2] + extent[2] / 2.0,
        ],
        "quality": {
            "actionable": True,
            "world_extent_m": list(extent),
            "world_z_max_m": xyz[2] + extent[2] / 2.0,
        },
        "operation_pose_candidates": [],
    }


def _detection(
    rank: int,
    center: list[float],
    lower: list[float],
    upper: list[float],
    *,
    camera: str = "head",
    score: float = 0.8,
) -> dict:
    return {
        "rank": rank,
        "score": score,
        "camera": camera,
        "quality": {"actionable": True},
        "grounding_3d": {
            "success": True,
            "centroid_world": center,
            "bbox_world_min": lower,
            "bbox_world_max": upper,
        },
    }


def _reference_query(*, detections: list[dict]) -> dict:
    return {
        "success": True,
        "object_id": "mat",
        "text_prompt": "blue square mat",
        "query_role": "context",
        "query_instance_hint": "four mats defining the workspace center",
        "query_reason": "The set defines the requested placement center.",
        "entity_scope": "reference_set",
        "placement_relation": "center_of",
        "expected_count": 4,
        "detections": detections,
    }


def _reference_instances() -> list[dict]:
    return [
        _instance(
            "track_group",
            [0.1014, -0.0829, 0.7427],
            [0.2787, 0.2761, 0.0019],
            class_name="mat",
        ),
        _instance(
            "track_lower",
            [0.1014, -0.1995, 0.7429],
            [0.0793, 0.0762, 0.0016],
            class_name="mat",
        ),
        _instance(
            "track_upper",
            [0.1014, 0.0008, 0.7426],
            [0.0789, 0.0773, 0.0016],
            class_name="mat",
        ),
    ]


def _reference_detections() -> list[dict]:
    return [
        _detection(
            0,
            [0.1014, -0.0829, 0.7427],
            [-0.0380, -0.2380, 0.7417],
            [0.2407, 0.0381, 0.7436],
            score=0.91,
        ),
        _detection(
            1,
            [0.1014, -0.1995, 0.7429],
            [0.0617, -0.2376, 0.7421],
            [0.1410, -0.1614, 0.7437],
            score=0.83,
        ),
        _detection(
            2,
            [0.1014, 0.0008, 0.7426],
            [0.0619, -0.0379, 0.7418],
            [0.1408, 0.0394, 0.7434],
            score=0.79,
        ),
    ]


def test_group_mask_and_member_masks_form_one_reference_region() -> None:
    tracker = ReferenceRegionTracker(max_missing_steps=4)
    regions = tracker.update(
        segmentation=[
            _reference_query(detections=_reference_detections())
        ],
        instances=_reference_instances(),
        env_step=17,
        current_subtask="put the block at the center of the four mats",
    )

    assert len(regions) == 1
    region = regions[0]
    np.testing.assert_allclose(
        region["center_world_xy"],
        [0.10135, -0.09995],
        atol=1e-6,
    )
    assert region["evidence_status"] == (
        "grounded_aggregate_reference_footprint"
    )
    assert region["observed_member_count"] == 2
    assert region["reference_instance_ids"] == [
        "track_group",
        "track_lower",
        "track_upper",
    ]
    assert region["occupancy_exempt_instance_ids"] == ["track_group"]


def test_scene_memory_keeps_reference_set_separate_from_singleton_binding() -> None:
    scene = SceneMemoryTracker().update(
        segmentation=[
            _reference_query(detections=_reference_detections())
        ],
        env_step=17,
        global_task="put the block at the center of the four mats",
        current_subtask="put the block at the center",
    )

    assert len(scene[REFERENCE_REGIONS_KEY]) == 1
    assert scene["task_focus"]["identity_binding_errors"] == []
    assert scene["instances"] == []
    assert scene["task_focus"]["context_instances"] == []


def test_dominant_plane_recovers_live_aggregate_with_raised_object() -> None:
    aggregate = _detection(
        0,
        [0.094849, -0.208686, 0.743505],
        [-0.037620, -0.238948, 0.741088],
        [0.240927, 0.037159, 0.782834],
        camera="third",
        score=0.91,
    )
    aggregate["grounding_3d"]["dominant_plane_footprint"] = {
        "valid": True,
        "source": "dominant_horizontal_z_band",
        "plane_z_world_m": 0.7422,
        "point_count": 5000,
        "inlier_count": 4210,
        "inlier_ratio": 0.842,
        "band_width_m": 0.008,
        "bbox_world_min": [-0.037620, -0.238948, 0.741088],
        "bbox_world_max": [0.240927, 0.037159, 0.743012],
        "extent_m": [0.278547, 0.276107, 0.001924],
    }
    tracker = ReferenceRegionTracker(max_missing_steps=4)
    regions = tracker.update(
        segmentation=[_reference_query(detections=[aggregate])],
        instances=[],
        env_step=8,
        current_subtask="put the block at the center of the four mats",
    )

    assert len(regions) == 1
    region = regions[0]
    assert region["support_valid"] is True
    assert region["evidence_status"] == (
        "grounded_aggregate_reference_footprint"
    )
    np.testing.assert_allclose(
        region["center_world_xy"],
        [0.1016535, -0.1008945],
        atol=1e-6,
    )
    assert region["observed_member_count"] == 0


def test_multiview_duplicate_does_not_count_as_an_extra_member() -> None:
    duplicate = _detection(
        0,
        [0.1020, -0.1990, 0.7430],
        [0.0620, -0.2370, 0.7421],
        [0.1412, -0.1610, 0.7438],
        camera="third",
        score=0.88,
    )
    tracker = ReferenceRegionTracker(max_missing_steps=4)
    regions = tracker.update(
        segmentation=[
            _reference_query(
                detections=[*_reference_detections(), duplicate]
            )
        ],
        instances=_reference_instances(),
        env_step=17,
        current_subtask="put the block at the center of the four mats",
    )

    assert regions[0]["observation_count"] == 3
    assert regions[0]["observed_member_count"] == 2


def test_reference_region_retains_geometry_only_within_same_subtask() -> None:
    tracker = ReferenceRegionTracker(max_missing_steps=2)
    initial = tracker.update(
        segmentation=[
            _reference_query(detections=_reference_detections())
        ],
        instances=_reference_instances(),
        env_step=10,
        current_subtask="place at center",
    )
    retained = tracker.update(
        segmentation=[],
        instances=_reference_instances(),
        env_step=11,
        current_subtask="place at center",
    )
    changed_subtask = tracker.update(
        segmentation=[],
        instances=_reference_instances(),
        env_step=11,
        current_subtask="return to the original pose",
    )

    assert initial[0]["position_state"] == "current_verified"
    assert retained[0]["position_state"] == "memory_valid"
    assert changed_subtask == []


def test_reference_region_step_zero_is_not_treated_as_missing() -> None:
    tracker = ReferenceRegionTracker(max_missing_steps=2)
    initial = tracker.update(
        segmentation=[
            _reference_query(detections=_reference_detections())
        ],
        instances=_reference_instances(),
        env_step=0,
        current_subtask="place at center",
    )
    retained = tracker.update(
        segmentation=[],
        instances=_reference_instances(),
        env_step=1,
        current_subtask="place at center",
    )

    assert initial[0]["support_valid"] is True
    assert initial[0]["position_state"] == "current_verified"
    assert initial[0]["last_observed_step"] == 0
    assert initial[0]["missing_observation_steps"] == 0
    assert retained[0]["support_valid"] is True
    assert retained[0]["position_state"] == "memory_valid"
    assert retained[0]["missing_observation_steps"] == 1


def test_incomplete_member_set_cannot_overwrite_verified_group_footprint() -> None:
    tracker = ReferenceRegionTracker(max_missing_steps=4)
    initial = tracker.update(
        segmentation=[
            _reference_query(detections=_reference_detections())
        ],
        instances=_reference_instances(),
        env_step=17,
        current_subtask="place at center",
    )
    incomplete_query = _reference_query(
        detections=[
            _detection(
                0,
                [0.1013, 0.0011, 0.7426],
                [0.0620, -0.0401, 0.7411],
                [0.1407, 0.0372, 0.7427],
                score=0.126,
            ),
            _detection(
                1,
                [0.2012, -0.0991, 0.7427],
                [0.1618, -0.1399, 0.7412],
                [0.2409, -0.0621, 0.7428],
                score=0.125,
            ),
            _detection(
                2,
                [0.1014, -0.1994, 0.7429],
                [0.0619, -0.2389, 0.7414],
                [0.1409, -0.1628, 0.7429],
                score=0.125,
            ),
        ]
    )
    incomplete_query["query_instance_hint"] = (
        "new wording for the same complete reference set"
    )
    retained = tracker.update(
        segmentation=[incomplete_query],
        instances=_reference_instances(),
        env_step=18,
        current_subtask="move the held object into the center region",
    )

    assert retained[0]["reference_region_id"] == initial[0][
        "reference_region_id"
    ]
    assert retained[0]["position_state"] == "memory_valid"
    np.testing.assert_allclose(
        retained[0]["center_world_xy"],
        initial[0]["center_world_xy"],
        atol=1e-6,
    )


def test_occluded_reference_geometry_cannot_replace_complete_planar_memory() -> None:
    tracker = ReferenceRegionTracker(max_missing_steps=6)
    complete = [
        _detection(
            0,
            [0.100130, -0.1995, 0.7418],
            [0.0600, -0.238376, 0.740684],
            [0.1400, -0.1600, 0.742807],
            camera="head",
        ),
        _detection(
            1,
            [0.100130, 0.0008, 0.7418],
            [0.0600, -0.0380, 0.740684],
            [0.1400, 0.037159, 0.742807],
            camera="head",
        ),
        _detection(
            2,
            [-0.0001, -0.1006, 0.7418],
            [-0.040281, -0.1400, 0.740684],
            [0.0398, -0.0600, 0.742807],
            camera="third",
        ),
        _detection(
            3,
            [0.2002, -0.1006, 0.7418],
            [0.1600, -0.1400, 0.740684],
            [0.240540, -0.0600, 0.742807],
            camera="third",
        ),
    ]
    initial = tracker.update(
        segmentation=[_reference_query(detections=complete)],
        instances=[],
        env_step=5,
        current_subtask="place at the center of the reference set",
    )
    np.testing.assert_allclose(
        initial[0]["center_world_xy"],
        [0.1001295, -0.1006085],
        atol=1e-6,
    )
    assert initial[0]["evidence_status"] == (
        "grounded_reference_member_envelope"
    )

    arm_contaminated = copy.deepcopy(complete)
    arm_contaminated[0]["grounding_3d"]["bbox_world_min"][1] = -0.273152
    arm_contaminated[0]["grounding_3d"]["bbox_world_max"][2] = 0.876188
    contaminated = tracker.update(
        segmentation=[_reference_query(detections=arm_contaminated)],
        instances=[],
        env_step=7,
        current_subtask="place at the center of the reference set",
    )
    assert contaminated[0]["position_state"] == "memory_valid"
    assert contaminated[0]["evidence_status"] == (
        "retained_after_lower_quality_current_observation"
    )
    np.testing.assert_allclose(
        contaminated[0]["center_world_xy"],
        initial[0]["center_world_xy"],
        atol=1e-6,
    )

    partial_aggregate = _detection(
        0,
        [0.099049, -0.149504, 0.7445],
        [-0.040255, -0.238376, 0.740719],
        [0.238352, -0.060631, 0.748281],
        camera="head",
        score=0.95,
    )
    partial_members = [
        _detection(
            index + 1,
            center,
            [center[0] - 0.025, center[1] - 0.025, 0.7410],
            [center[0] + 0.025, center[1] + 0.025, 0.7430],
            camera=("head" if index < 2 else "third"),
            score=0.80 - 0.05 * index,
        )
        for index, center in enumerate(
            (
                [0.02, -0.20, 0.7420],
                [0.18, -0.20, 0.7420],
                [0.02, -0.10, 0.7420],
                [0.18, -0.10, 0.7420],
            )
        )
    ]
    partial = tracker.update(
        segmentation=[
            _reference_query(
                detections=[partial_aggregate, *partial_members]
            )
        ],
        instances=[],
        env_step=9,
        current_subtask="place at the center of the reference set",
    )
    assert partial[0]["position_state"] == "memory_valid"
    assert partial[0]["evidence_status"] == (
        "retained_after_lower_quality_current_observation"
    )
    np.testing.assert_allclose(
        partial[0]["center_world_xy"],
        initial[0]["center_world_xy"],
        atol=1e-6,
    )
    assert partial[0]["observed_member_count"] == 4

    expired = tracker.update(
        segmentation=[
            _reference_query(
                detections=[partial_aggregate, *partial_members]
            )
        ],
        instances=[],
        env_step=12,
        current_subtask="place at the center of the reference set",
    )
    assert expired[0]["support_valid"] is False
    assert expired[0]["position_state"] == "motion_uncertain"
    assert expired[0]["evidence_status"] == (
        "reference_region_continuity_unresolved"
    )

    still_unresolved = tracker.update(
        segmentation=[
            _reference_query(
                detections=[partial_aggregate, *partial_members]
            )
        ],
        instances=[],
        env_step=13,
        current_subtask="place at the center of the reference set",
    )
    assert still_unresolved[0]["support_valid"] is False
    assert still_unresolved[0]["position_state"] == "motion_uncertain"
    assert still_unresolved[0]["evidence_status"] == (
        "reference_region_continuity_unresolved"
    )

    recovered = tracker.update(
        segmentation=[_reference_query(detections=complete)],
        instances=[],
        env_step=14,
        current_subtask="place at the center of the reference set",
    )
    assert recovered[0]["support_valid"] is True
    assert recovered[0]["position_state"] == "current_verified"
    assert recovered[0]["evidence_status"] == (
        "grounded_reference_member_envelope"
    )


def test_reference_member_motion_event_invalidates_region() -> None:
    tracker = ReferenceRegionTracker(max_missing_steps=4)
    tracker.update(
        segmentation=[
            _reference_query(detections=_reference_detections())
        ],
        instances=_reference_instances(),
        env_step=10,
        current_subtask="place at center",
    )
    invalid = tracker.update(
        segmentation=[],
        instances=_reference_instances(),
        env_step=11,
        current_subtask="place at center",
        position_events=[
            {
                "instance_ref": "track_lower",
                "position_state": "motion_uncertain",
            }
        ],
    )

    assert invalid[0]["support_valid"] is False
    assert invalid[0]["position_state"] == "motion_uncertain"


def test_relation_target_uses_runtime_tcp_mapping_and_revalidates_occupancy() -> None:
    references = _reference_instances()
    tracker = ReferenceRegionTracker(max_missing_steps=4)
    regions = tracker.update(
        segmentation=[
            _reference_query(detections=_reference_detections())
        ],
        instances=references,
        env_step=17,
        current_subtask="put the block at the center of the four mats",
    )
    held = _instance(
        "held",
        [0.0975, -0.2039, 0.82],
        [0.0520, 0.0556, 0.0400],
        class_name="block",
        first_xyz=[0.0975, -0.2039, 0.7754],
    )
    scene = {
        "env_step": 17,
        "instances": [held, *references],
        REFERENCE_REGIONS_KEY: regions,
    }
    enriched = with_dynamic_place_candidates(
        scene,
        manipulation_state={
            "right": {
                    "phase": "holding",
                    "held_instance_id": "held",
                    "holding_confirmed": True,
                    "transport_authorized": True,
                    "grasp_candidate_id": "candidate:right:held:001",
                    "grasp_attempt_step": 16,
                    "grasp_attempt_nonce": (
                        "right:held:candidate:right:held:001:16"
                    ),
                    "held_object_to_tcp_attachment": {
                        "object_proxy_frame": "tcp_aligned_at_capture",
                        "object_centroid_to_tcp_translation_tcp_m": [
                            0.0,
                            0.0,
                            0.12,
                        ],
                        "capture_tcp_pose": [
                            0.0975,
                            -0.2039,
                            0.94,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        "grasp_candidate_id": (
                            "candidate:right:held:001"
                        ),
                        "grasp_attempt_nonce": (
                            "right:held:candidate:right:held:001:16"
                        ),
                        "capture_step": 16,
                        "source": (
                            "runtime_multiview_grasp_motion_verified"
                        ),
                        "authority": "runtime_multiview_grasp_motion",
                    },
            }
        },
        robot_state={
            "right": {
                "xyz": [0.0975, -0.2039, 0.94],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "gripper": 0.0,
            }
        },
    )

    public = next(
        item
        for item in enriched["operation_targets"]
        if item["target_kind"] == "reference_region"
    )
    assert public["placement_relation"] == "center_of"
    np.testing.assert_allclose(
        public["object_target_world_m"][:2],
        regions[0]["center_world_xy"],
        atol=1e-6,
    )
    enriched_held = next(
        item
        for item in enriched["instances"]
        if item["instance_id"] == "held"
    )
    candidate = next(
        item
        for item in operation_pose_candidates(enriched_held)
        if item.get("target_id") == public["target_id"]
    )
    object_delta = np.asarray(
        candidate["held_object_target_world_m"]
    ) - np.asarray(held["world_m"])
    ee_delta = np.asarray(candidate["ee_target_pose"][:3]) - np.asarray(
        [0.0975, -0.2039, 0.94]
    )
    np.testing.assert_allclose(ee_delta, object_delta, atol=1e-6)
    assert validate_place_candidate(
        enriched,
        held_instance=enriched_held,
        candidate=candidate,
    )["valid"] is True

    occupied = copy.deepcopy(enriched)
    occupied["instances"].append(
        _instance(
            "new_occupant",
            [
                *public["object_target_world_m"][:2],
                public["object_target_world_m"][2],
            ],
            [0.04, 0.04, 0.04],
            class_name="object",
        )
    )
    blocked = validate_place_candidate(
        occupied,
        held_instance=enriched_held,
        candidate=candidate,
    )
    assert blocked["support_valid"] is True
    assert blocked["free"] is False
    assert blocked["occupied_by"] == ["new_occupant"]
