from __future__ import annotations

import unittest

from policy.roboharn_evo.agent.perception.scene_memory import (
    SceneMemoryTracker,
    extract_scene_candidates,
)


def detection(rank: int, bbox: list[int], world: list[float], score: float = 0.9) -> dict:
    return {
        "rank": rank,
        "score": score,
        "bbox_xyxy": bbox,
        "centroid_px": [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
        "mask_path": f"/tmp/lid_{rank}.png",
        "grounding_3d": {
            "success": True,
            "centroid_world": world,
            "top_surface_world": [world[0], world[1], world[2] + 0.04],
            "approach_point_world": [world[0], world[1], world[2] + 0.08],
        },
    }


def detection_with_extent(rank: int, bbox: list[int], world: list[float], extent: list[float], score: float = 0.9) -> dict:
    return {
        "rank": rank,
        "score": score,
        "bbox_xyxy": bbox,
        "centroid_px": [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
        "mask_path": f"/tmp/lid_extent_{rank}.png",
        "grounding_3d": {
            "success": True,
            "centroid_world": world,
            "bbox_world_min": [world[0] - extent[0] / 2.0, world[1] - extent[1] / 2.0, world[2] - extent[2] / 2.0],
            "bbox_world_max": [world[0] + extent[0] / 2.0, world[1] + extent[1] / 2.0, world[2] + extent[2] / 2.0],
            "top_surface_world": [world[0], world[1], world[2] + extent[2] / 2.0],
            "approach_point_world": [world[0], world[1], world[2] + extent[2] / 2.0 + 0.08],
        },
    }


def detection_with_world_bounds(
    rank: int,
    bbox: list[int],
    world: list[float],
    lower: list[float],
    upper: list[float],
    *,
    top_surface: list[float],
    score: float = 0.9,
) -> dict:
    return {
        "rank": rank,
        "score": score,
        "bbox_xyxy": bbox,
        "centroid_px": [
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
        ],
        "mask_path": f"/tmp/world_bounds_{rank}.png",
        "grounding_3d": {
            "success": True,
            "centroid_world": world,
            "bbox_world_min": lower,
            "bbox_world_max": upper,
            "top_surface_world": top_surface,
            "approach_point_world": [
                top_surface[0],
                top_surface[1],
                top_surface[2] + 0.08,
            ],
        },
    }


def image_only_detection(rank: int, bbox: list[int], score: float = 0.9) -> dict:
    return {
        "rank": rank,
        "score": score,
        "bbox_xyxy": bbox,
        "centroid_px": [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
        "mask_path": f"/tmp/image_only_{rank}.png",
    }


def detection_with_action_geometry(
    rank: int,
    bbox: list[int],
    world: list[float],
    *,
    top_offset: list[float],
    approach_offset: list[float],
    contact_offset: list[float],
    score: float = 0.9,
) -> dict:
    item = detection(rank, bbox, world, score=score)
    grounding = item["grounding_3d"]
    top = [world[index] + top_offset[index] for index in range(3)]
    approach = [world[index] + approach_offset[index] for index in range(3)]
    contact = [world[index] + contact_offset[index] for index in range(3)]
    grounding.update(
        {
            "top_surface_world": top,
            "approach_point_world": approach,
            "approach_pose_world": [*approach, 1.0, 0.0, 0.0, 0.0],
            "contact_point_world": contact,
            "contact_pose_world": [*contact, 1.0, 0.0, 0.0, 0.0],
        }
    )
    return item


def verified_release_event(
    track_id: str,
    *,
    observed_world: list[float],
    target_world: list[float],
    tolerance_m: float,
    env_step: int,
) -> dict:
    return {
        "instance_ref": track_id,
        "position_state": "current_verified",
        "world_m": observed_world,
        "confidence": 1.0,
        "tolerance_m": tolerance_m,
        "operation_target_world_m": target_world,
        "operation_target_tolerance_m": tolerance_m,
        "operation_target_id": "operation-target:e2",
        "arm": "right",
        "source": "verified_post_release_position",
        "env_step": env_step,
    }


class SceneMemoryTest(unittest.TestCase):
    def test_release_relocation_lease_reanchors_only_unique_target_candidate(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "cube",
                    "text_prompt": "red cube",
                    "query_role": "target",
                    "detections": [
                        detection(
                            0,
                            [260, 138, 288, 158],
                            [0.27, -0.10, 0.77],
                        )
                    ],
                }
            ],
            env_step=0,
            global_task="move one selected object",
            current_subtask="move the selected object",
        )
        track_id = initial["instances"][0]["track_id"]
        relocation_lease = [
            {
                "instance_ref": track_id,
                "target_world_m": [0.10, 0.01, 0.77],
                "tolerance_m": 0.045,
                "arm": "right",
                "release_step": 4,
            }
        ]
        relocated = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "cube",
                    "text_prompt": "red cube",
                    "query_role": "context",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection(
                            0,
                            [198, 108, 214, 126],
                            [0.102, 0.008, 0.769],
                            score=0.8,
                        ),
                        detection(
                            1,
                            [223, 138, 242, 157],
                            [0.156, -0.106, 0.772],
                            score=0.95,
                        ),
                    ],
                }
            ],
            env_step=6,
            global_task="move one selected object",
            current_subtask="verify the released object",
            identity_relocation_leases=relocation_lease,
        )

        instance = next(
            item
            for item in relocated["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(instance["latest_world_m"], [0.102, 0.008, 0.769])
        self.assertEqual(instance["history_length"], 1)
        self.assertEqual(len(instance["identity_relocation_history"]), 1)
        outcome = relocated["temporal_memory"][
            "identity_binding_outcomes"
        ][0]
        self.assertEqual(
            outcome["matched_by"],
            "operation_target_relocation",
        )
        self.assertEqual(
            relocated["temporal_memory"]["num_visible_candidates"],
            1,
        )

        stable = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "cube",
                    "text_prompt": "red cube",
                    "query_role": "context",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection(
                            0,
                            [198, 108, 214, 126],
                            [0.101, 0.009, 0.770],
                            score=0.82,
                        )
                    ],
                }
            ],
            env_step=7,
            global_task="move one selected object",
            current_subtask="verify the released object",
            identity_relocation_leases=relocation_lease,
        )
        stable_instance = next(
            item
            for item in stable["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(stable_instance["history_length"], 2)
        self.assertEqual(
            len(stable_instance["identity_relocation_history"]),
            1,
        )
        self.assertEqual(
            stable["temporal_memory"]["identity_binding_outcomes"][0][
                "matched_by"
            ],
            "operation_target_continuity",
        )

    def test_release_lease_recovers_unique_off_target_trace_observation(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        initial_world = [0.102096, -0.114293, 0.771531]
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "text_prompt": "selected component",
                    "query_role": "target",
                    "detections": [
                        detection_with_extent(
                            0,
                            [126, 160, 138, 174],
                            initial_world,
                            [0.039063, 0.062372, 0.039967],
                            score=0.72,
                        )
                    ],
                }
            ],
            env_step=22,
            global_task="generic multi-stage manipulation",
            current_subtask="move the selected component",
        )
        track_id = initial["instances"][0]["track_id"]
        target_world = [0.098879, -0.201065, 0.793909]
        recovery_anchor = [0.093796, -0.130835, 0.797822]
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "text_prompt": "selected component",
                    "query_role": "context",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection_with_extent(
                            0,
                            [126, 160, 138, 174],
                            [0.102377, -0.113884, 0.771712],
                            [0.038657, 0.058686, 0.039961],
                            score=0.777344,
                        ),
                        detection_with_extent(
                            1,
                            [219, 162, 237, 179],
                            [-0.244555, -0.094243, 0.778213],
                            [0.119198, 0.094158, 0.168748],
                            score=0.621094,
                        ),
                        detection_with_extent(
                            2,
                            [127, 160, 138, 174],
                            [0.101836, -0.113577, 0.771952],
                            [0.037443, 0.054045, 0.039961],
                            score=0.148438,
                        ),
                    ],
                }
            ],
            env_step=38,
            global_task="generic multi-stage manipulation",
            current_subtask="verify the released component",
            identity_relocation_leases=[
                {
                    "instance_ref": track_id,
                    "target_world_m": target_world,
                    "tolerance_m": 0.03,
                    "validation_tolerance_m": 0.02,
                    "arm": "right",
                    "release_step": 36,
                    "source": (
                        "release_pending_verification_operation_target"
                    ),
                    "recovery_anchors": [
                        {
                            "world_m": target_world,
                            "tolerance_m": 0.04,
                            "source": "committed_release_target",
                        },
                        {
                            "world_m": recovery_anchor,
                            "tolerance_m": 0.04,
                            "source": "pregrasp_contact_position",
                        },
                    ],
                }
            ],
        )

        instance = next(
            item
            for item in scene["instances"]
            if item["track_id"] == track_id
        )
        self.assertLess(
            sum(
                (
                    instance["latest_world_m"][index]
                    - [0.102377, -0.113884, 0.771712][index]
                )
                ** 2
                for index in range(3)
            )
            ** 0.5,
            0.001,
        )
        outcome = scene["temporal_memory"][
            "identity_binding_outcomes"
        ][0]
        self.assertEqual(
            outcome["status"],
            "matched_recovery_observation",
        )
        self.assertEqual(
            outcome["matched_by"],
            "runtime_manipulation_history_anchor",
        )
        self.assertEqual(
            outcome["recovery_anchor_source"],
            "pregrasp_contact_position",
        )
        self.assertGreater(
            outcome["relocation_target_error_m"],
            0.08,
        )
        self.assertEqual(
            scene["temporal_memory"]["num_visible_candidates"],
            1,
        )

    def test_release_recovery_anchors_remain_ambiguous_for_two_objects(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "detections": [
                        detection_with_extent(
                            0,
                            [120, 150, 140, 170],
                            [0.10, -0.10, 0.77],
                            [0.04, 0.04, 0.04],
                        )
                    ],
                }
            ],
            env_step=0,
            global_task="generic manipulation",
            current_subtask="move one component",
        )
        track_id = initial["instances"][0]["track_id"]
        target_world = [0.10, -0.20, 0.77]
        ambiguous = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "context",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection_with_extent(
                            0,
                            [120, 150, 140, 170],
                            [0.10, -0.10, 0.77],
                            [0.04, 0.04, 0.04],
                            score=0.8,
                        ),
                        detection_with_extent(
                            1,
                            [180, 110, 200, 130],
                            [0.135, -0.20, 0.77],
                            [0.04, 0.04, 0.04],
                            score=0.85,
                        ),
                    ],
                }
            ],
            env_step=3,
            global_task="generic manipulation",
            current_subtask="verify release",
            identity_relocation_leases=[
                {
                    "instance_ref": track_id,
                    "target_world_m": target_world,
                    "tolerance_m": 0.03,
                    "recovery_anchors": [
                        {
                            "world_m": target_world,
                            "tolerance_m": 0.04,
                            "source": "committed_release_target",
                        },
                        {
                            "world_m": [0.10, -0.10, 0.77],
                            "tolerance_m": 0.04,
                            "source": "pregrasp_observed_position",
                        },
                    ],
                }
            ],
        )

        outcome = ambiguous["temporal_memory"][
            "identity_binding_outcomes"
        ][0]
        self.assertEqual(
            outcome["status"],
            "ambiguous_recovery_anchor_candidates",
        )
        self.assertEqual(
            ambiguous["temporal_memory"]["num_visible_candidates"],
            0,
        )
        self.assertEqual(
            len(ambiguous["temporal_memory"]["dropped_candidates"]),
            2,
        )

    def test_release_lease_keeps_position_when_continuous_action_geometry_is_polluted(
        self,
    ) -> None:
        """A release observation may verify position without exposing bad poses."""

        tracker = SceneMemoryTracker(
            max_missing_steps=20,
            temporal_action_geometry_distance_m=0.04,
        )
        target_world = [0.110159, -0.112888, 0.774796]
        stable_offsets = {
            "top_offset": [0.0, 0.0, 0.04],
            "approach_offset": [0.0, 0.0, 0.20],
            "contact_offset": [0.0, 0.0, 0.12],
        }

        def with_extent(item: dict, world: list[float]) -> dict:
            extent = [0.05, 0.05, 0.04]
            item["grounding_3d"]["bbox_world_min"] = [
                world[index] - extent[index] / 2.0
                for index in range(3)
            ]
            item["grounding_3d"]["bbox_world_max"] = [
                world[index] + extent[index] / 2.0
                for index in range(3)
            ]
            return item

        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "text_prompt": "selected component",
                    "query_role": "target",
                    "detections": [
                        with_extent(
                            detection_with_action_geometry(
                                0,
                                [180, 130, 204, 154],
                                target_world,
                                **stable_offsets,
                            ),
                            target_world,
                        )
                    ],
                }
            ],
            env_step=10,
            global_task="move one selected object",
            current_subtask="place the selected object",
            observation_capture_id=10,
        )
        track_id = initial["instances"][0]["track_id"]
        release_lease = [
            {
                "instance_ref": track_id,
                "target_world_m": target_world,
                "tolerance_m": 0.039,
                "validation_tolerance_m": 0.026,
                "arm": "right",
                "release_step": 12,
                "source": (
                    "release_pending_verification_operation_target"
                ),
                "allow_position_only_action_geometry_quarantine": True,
            }
        ]

        def polluted_observation(
            world: list[float],
            *,
            mask_path: str,
        ) -> dict:
            item = detection_with_action_geometry(
                0,
                [176, 132, 202, 158],
                world,
                top_offset=[0.0, 0.0, 0.04],
                approach_offset=[0.0, 0.0, 0.425],
                contact_offset=[0.0, 0.0, 0.345],
            )
            item["mask_path"] = mask_path
            item["grounding_3d"]["operation_pose_candidates"] = [
                {
                    "candidate_id": "rgbd_surface:grasp:right:000",
                    "source_candidate_index": 0,
                    "action_mode": "grasp",
                    "arm": "right",
                    "object_contact_pose": [
                        *world,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "tcp_pose": [*world, 1.0, 0.0, 0.0, 0.0],
                    "ee_target_pose": [
                        world[0],
                        world[1],
                        world[2] + 0.12,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "approach_pose": [
                        world[0],
                        world[1],
                        world[2] + 0.20,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "approach_direction": [0.0, 0.0, -1.0],
                    "geometry_source": (
                        "rgbd_surface_normal_principal_axes"
                    ),
                }
            ]
            return with_extent(item, world)

        first_world = [0.093469, -0.110745, 0.777810]
        first = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "text_prompt": "selected component",
                    "query_role": "context",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        polluted_observation(
                            first_world,
                            mask_path="/tmp/release_position_only_12.png",
                        )
                    ],
                }
            ],
            env_step=12,
            global_task="move one selected object",
            current_subtask="verify the released object",
            identity_relocation_leases=release_lease,
            observation_capture_id=12,
        )

        first_instance = next(
            item
            for item in first["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(first_instance["status"], "visible")
        self.assertEqual(
            first_instance["position_state"],
            "current_verified",
        )
        self.assertEqual(first_instance["last_verified_step"], 12)
        self.assertEqual(first_instance["latest_world_m"], first_world)
        self.assertEqual(
            first_instance["action_geometry_state"],
            "relocation_pending",
        )
        self.assertEqual(
            first_instance["stability"],
            "relocation_geometry_pending",
        )
        self.assertIsNone(first_instance["top_surface_world_m"])
        self.assertIsNone(first_instance["approach_world_m"])
        self.assertIsNone(first_instance["grasp_world_m"])
        self.assertIsNone(first_instance["contact_world_m"])
        self.assertEqual(
            first_instance["operation_pose_candidate_count"],
            0,
        )
        self.assertEqual(
            first["temporal_memory"]["num_visible_candidates"],
            1,
        )
        self.assertEqual(
            first["temporal_memory"]["dropped_candidates"],
            [],
        )
        self.assertEqual(
            first["temporal_memory"]["identity_binding_outcomes"][0][
                "matched_by"
            ],
            "release_position_only_geometry_quarantine",
        )
        self.assertEqual(
            first_instance["identity_relocation_history"][-1]["reason"],
            "release_position_verification_with_quarantined_action_geometry",
        )

        second_world = [0.094169, -0.110445, 0.777610]
        second = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "text_prompt": "selected component",
                    "query_role": "context",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        polluted_observation(
                            second_world,
                            mask_path="/tmp/release_position_only_13.png",
                        )
                    ],
                }
            ],
            env_step=13,
            global_task="move one selected object",
            current_subtask="verify the released object",
            identity_relocation_leases=release_lease,
            observation_capture_id=13,
        )
        second_instance = next(
            item
            for item in second["instances"]
            if item["track_id"] == track_id
        )
        temporal_track = next(
            item
            for item in second["temporal_memory"]["tracks"]
            if item["track_id"] == track_id
        )
        self.assertEqual(second_instance["latest_world_m"], second_world)
        self.assertEqual(second_instance["last_verified_step"], 13)
        self.assertEqual(second_instance["history_length"], 2)
        self.assertEqual(
            [item["env_step"] for item in temporal_track["history"][-2:]],
            [12, 13],
        )
        self.assertEqual(
            [
                item["observation_capture_id"]
                for item in temporal_track["history"][-2:]
            ],
            [12, 13],
        )
        self.assertEqual(
            second_instance["action_geometry_confirmation_count"],
            0,
        )
        self.assertEqual(
            len(second_instance["identity_relocation_history"]),
            1,
        )
        self.assertEqual(
            second["temporal_memory"]["identity_binding_outcomes"][0][
                "matched_by"
            ],
            "post_release_identity_repair_lease",
        )

    def test_release_relocation_quarantines_polluted_action_geometry_until_two_clean_observations(
        self,
    ) -> None:
        tracker = SceneMemoryTracker(
            max_missing_steps=20,
            temporal_action_geometry_distance_m=0.04,
        )
        stable_offsets = {
            "top_offset": [0.0, 0.0, 0.04],
            "approach_offset": [0.0, 0.0, 0.20],
            "contact_offset": [0.0, 0.0, 0.12],
        }
        def with_extent(
            item: dict,
            world: list[float],
            extent: list[float],
        ) -> dict:
            item["grounding_3d"]["bbox_world_min"] = [
                world[index] - extent[index] / 2.0
                for index in range(3)
            ]
            item["grounding_3d"]["bbox_world_max"] = [
                world[index] + extent[index] / 2.0
                for index in range(3)
            ]
            return item

        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "text_prompt": "selected component",
                    "query_role": "target",
                    "detections": [
                        with_extent(
                            detection_with_action_geometry(
                                0,
                                [260, 138, 288, 158],
                                [0.27, -0.10, 0.77],
                                **stable_offsets,
                            ),
                            [0.27, -0.10, 0.77],
                            [0.05, 0.05, 0.04],
                        )
                    ],
                }
            ],
            env_step=0,
            global_task="move one selected object",
            current_subtask="move the selected object",
        )
        track_id = initial["instances"][0]["track_id"]
        relocation_lease = [
            {
                "instance_ref": track_id,
                "target_world_m": [0.04, -0.21, 0.77],
                "tolerance_m": 0.045,
                "arm": "right",
                "release_step": 23,
            }
        ]

        polluted = with_extent(
            detection_with_action_geometry(
                0,
                [183, 181, 202, 198],
                [0.04, -0.21, 0.77],
                top_offset=[0.0, 0.0, 0.128],
                approach_offset=[0.0, 0.0, 0.288],
                contact_offset=[0.0, 0.0, 0.208],
                score=0.28,
            ),
            [0.04, -0.21, 0.77],
            [0.052, 0.051, 0.127],
        )
        polluted["mask_path"] = "/tmp/relocated_polluted_00.png"
        relocated = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "text_prompt": "selected component",
                    "query_role": "context",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [polluted],
                }
            ],
            env_step=23,
            global_task="move one selected object",
            current_subtask="verify the released object",
            identity_relocation_leases=relocation_lease,
        )
        relocated_instance = next(
            item
            for item in relocated["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(
            relocated_instance["latest_world_m"],
            [0.04, -0.21, 0.77],
        )
        self.assertEqual(
            relocated_instance["stability"],
            "relocation_geometry_pending",
        )
        self.assertEqual(
            relocated_instance["action_geometry_state"],
            "relocation_pending",
        )
        self.assertIsNone(relocated_instance["top_surface_world_m"])
        self.assertIsNone(relocated_instance["approach_world_m"])
        self.assertIsNone(relocated_instance["grasp_world_m"])
        self.assertEqual(
            relocated_instance["operation_pose_candidate_count"],
            0,
        )
        self.assertTrue(
            any(
                warning.startswith(
                    "relocation_action_geometry_quarantined:"
                )
                for warning in relocated_instance["quality_warnings"]
            )
        )

        polluted_again = dict(polluted)
        polluted_again["mask_path"] = (
            "/tmp/relocated_polluted_01.png"
        )
        still_quarantined = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "text_prompt": "selected component",
                    "query_role": "target",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [polluted_again],
                }
            ],
            env_step=23,
            global_task="move one selected object",
            current_subtask="verify the released object",
        )
        still_quarantined_instance = next(
            item
            for item in still_quarantined["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(
            still_quarantined_instance[
                "action_geometry_confirmation_count"
            ],
            0,
        )
        self.assertIsNone(
            still_quarantined_instance["approach_world_m"]
        )

        clean_first = with_extent(
            detection_with_action_geometry(
                0,
                [188, 171, 206, 192],
                [0.041, -0.209, 0.770],
                **stable_offsets,
            ),
            [0.041, -0.209, 0.770],
            [0.049, 0.058, 0.04],
        )
        clean_first["mask_path"] = "/tmp/relocated_clean_00.png"
        pending_confirmation = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "text_prompt": "selected component",
                    "query_role": "target",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [clean_first],
                }
            ],
            env_step=26,
            global_task="move one selected object",
            current_subtask="reacquire the selected object",
        )
        pending_instance = next(
            item
            for item in pending_confirmation["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(
            pending_instance["action_geometry_confirmation_count"],
            1,
        )
        self.assertEqual(
            pending_instance["stability"],
            "relocation_geometry_pending",
        )
        self.assertIsNone(pending_instance["approach_world_m"])

        clean_second = with_extent(
            detection_with_action_geometry(
                0,
                [188, 171, 206, 192],
                [0.042, -0.208, 0.771],
                **stable_offsets,
            ),
            [0.042, -0.208, 0.771],
            [0.049, 0.058, 0.04],
        )
        clean_second["mask_path"] = "/tmp/relocated_clean_01.png"
        repaired = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "text_prompt": "selected component",
                    "query_role": "target",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [clean_second],
                }
            ],
            env_step=27,
            global_task="move one selected object",
            current_subtask="reacquire the selected object",
        )
        repaired_instance = next(
            item
            for item in repaired["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(repaired_instance["track_id"], track_id)
        self.assertEqual(
            repaired_instance["action_geometry_state"],
            "verified",
        )
        self.assertEqual(
            repaired_instance["action_geometry_confirmation_count"],
            0,
        )
        self.assertEqual(repaired_instance["status"], "visible")
        self.assertEqual(repaired_instance["stability"], "stable")
        self.assertEqual(repaired_instance["history_length"], 2)
        self.assertAlmostEqual(
            repaired_instance["top_surface_world_m"][2]
            - repaired_instance["world_m"][2],
            0.04,
            places=5,
        )
        self.assertAlmostEqual(
            repaired_instance["approach_world_m"][2]
            - repaired_instance["world_m"][2],
            0.20,
            places=5,
        )
        self.assertFalse(
            any(
                warning.startswith(
                    "relocation_action_geometry_quarantined:"
                )
                for warning in repaired_instance[
                    "quality_warnings"
                ]
            )
        )

    def test_post_release_repair_lease_filters_far_semantic_false_positive_until_two_clean_observations(
        self,
    ) -> None:
        """Replay the v16 failure: true relocated object plus a semantic decoy."""
        tracker = SceneMemoryTracker(
            max_missing_steps=20,
            temporal_action_geometry_distance_m=0.04,
        )
        stable_offsets = {
            "top_offset": [0.0, 0.0, 0.04],
            "approach_offset": [0.0, 0.0, 0.20],
            "contact_offset": [0.0, 0.0, 0.12],
        }

        def with_extent(
            item: dict,
            world: list[float],
            extent: list[float],
        ) -> dict:
            item["grounding_3d"]["bbox_world_min"] = [
                world[index] - extent[index] / 2.0
                for index in range(3)
            ]
            item["grounding_3d"]["bbox_world_max"] = [
                world[index] + extent[index] / 2.0
                for index in range(3)
            ]
            return item

        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "selected_component",
                    "text_prompt": "selected component",
                    "query_role": "target",
                    "detections": [
                        with_extent(
                            detection_with_action_geometry(
                                0,
                                [223, 138, 242, 157],
                                [0.155996, -0.105825, 0.772082],
                                **stable_offsets,
                            ),
                            [0.155996, -0.105825, 0.772082],
                            [0.055865, 0.062725, 0.039988],
                        )
                    ],
                }
            ],
            env_step=16,
            global_task="move one selected object",
            current_subtask="move the selected object",
        )
        track_id = initial["instances"][0]["track_id"]
        target_world = [0.039454, -0.007076, 0.772605]
        relocation_tolerance = 0.041899

        polluted_world = [0.028504, -0.010821, 0.771351]
        polluted = with_extent(
            detection_with_action_geometry(
                0,
                [175, 109, 192, 126],
                polluted_world,
                top_offset=[0.0, 0.0, 0.098487],
                approach_offset=[0.0, 0.0, 0.298487],
                contact_offset=[0.0, 0.0, 0.218487],
                score=0.316406,
            ),
            polluted_world,
            [0.058138, 0.104276, 0.129131],
        )
        polluted["mask_path"] = "/tmp/repair_polluted_00.png"
        false_positive_world = [0.037693, -0.106545, 0.772959]
        false_positive = with_extent(
            detection_with_action_geometry(
                1,
                [179, 138, 195, 157],
                false_positive_world,
                **stable_offsets,
                score=0.15918,
            ),
            false_positive_world,
            [0.047592, 0.046262, 0.039988],
        )
        false_positive["mask_path"] = (
            "/tmp/repair_false_positive_00.png"
        )
        quarantined = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "selected_component",
                    "text_prompt": "selected component",
                    "query_role": "context",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        polluted,
                        false_positive,
                    ],
                }
            ],
            env_step=22,
            global_task="move one selected object",
            current_subtask="verify the released object",
            identity_relocation_leases=[
                {
                    "instance_ref": track_id,
                    "target_world_m": target_world,
                    "tolerance_m": relocation_tolerance,
                    "arm": "right",
                    "release_step": 22,
                    "source": (
                        "release_pending_verification_operation_target"
                    ),
                }
            ],
        )
        quarantined_instance = next(
            item
            for item in quarantined["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(
            quarantined_instance["action_geometry_state"],
            "relocation_pending",
        )
        self.assertEqual(
            quarantined_instance[
                "action_geometry_confirmation_count"
            ],
            0,
        )
        self.assertEqual(
            quarantined_instance["latest_world_m"],
            polluted_world,
        )
        self.assertEqual(
            quarantined_instance["identity_repair_lease"][
                "target_world_m"
            ],
            target_world,
        )

        clean_true_world = [0.019937, -0.016233, 0.757028]

        def clean_frame(token: int) -> list[dict]:
            clean_true = with_extent(
                detection_with_action_geometry(
                    0,
                    [170, 118, 185, 128],
                    clean_true_world,
                    **stable_offsets,
                    score=0.535156,
                ),
                clean_true_world,
                [0.044442, 0.024967, 0.039675],
            )
            clean_true["mask_path"] = (
                f"/tmp/repair_true_clean_{token:02d}.png"
            )
            decoy = with_extent(
                detection_with_action_geometry(
                    1,
                    [179, 138, 195, 157],
                    false_positive_world,
                    **stable_offsets,
                    score=0.202148,
                ),
                false_positive_world,
                [0.047328, 0.043023, 0.039988],
            )
            decoy["mask_path"] = (
                f"/tmp/repair_false_positive_{token:02d}.png"
            )
            return [clean_true, decoy]

        first_clean = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "selected_component",
                    "text_prompt": "selected component",
                    "query_role": "target",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": clean_frame(1),
                }
            ],
            env_step=24,
            global_task="move one selected object",
            current_subtask="reacquire the selected object",
        )
        first_instance = next(
            item
            for item in first_clean["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(
            first_instance["action_geometry_confirmation_count"],
            1,
        )
        self.assertEqual(
            first_clean["temporal_memory"][
                "identity_binding_outcomes"
            ][0]["matched_by"],
            "post_release_identity_repair_lease",
        )
        self.assertEqual(
            first_clean["temporal_memory"]["num_visible_candidates"],
            1,
        )
        self.assertTrue(
            any(
                (
                    "bound_identity_repair_outside_lease:"
                    f"{track_id}"
                )
                in warning
                for item in first_clean["temporal_memory"][
                    "dropped_candidates"
                ]
                for warning in item["quality"]["warnings"]
            )
        )

        second_clean = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "selected_component",
                    "text_prompt": "selected component",
                    "query_role": "target",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": clean_frame(2),
                }
            ],
            env_step=25,
            global_task="move one selected object",
            current_subtask="reacquire the selected object",
        )
        repaired_instance = next(
            item
            for item in second_clean["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(
            repaired_instance["action_geometry_state"],
            "verified",
        )
        self.assertEqual(
            repaired_instance["track_id"],
            track_id,
        )
        self.assertNotIn(
            "identity_repair_lease",
            repaired_instance,
        )

    def test_post_release_repair_lease_keeps_two_near_target_candidates_ambiguous(
        self,
    ) -> None:
        tracker = SceneMemoryTracker(
            temporal_action_geometry_distance_m=0.04,
        )
        stable_offsets = {
            "top_offset": [0.0, 0.0, 0.04],
            "approach_offset": [0.0, 0.0, 0.20],
            "contact_offset": [0.0, 0.0, 0.12],
        }
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "detections": [
                        detection_with_action_geometry(
                            0,
                            [220, 130, 242, 155],
                            [0.20, -0.10, 0.65],
                            **stable_offsets,
                        )
                    ],
                }
            ],
            env_step=0,
            global_task="move one object",
            current_subtask="move the selected object",
        )
        track_id = initial["instances"][0]["track_id"]
        polluted = detection_with_action_geometry(
            0,
            [170, 110, 190, 130],
            [0.0, 0.0, 0.65],
            top_offset=[0.0, 0.0, 0.12],
            approach_offset=[0.0, 0.0, 0.28],
            contact_offset=[0.0, 0.0, 0.20],
        )
        polluted["mask_path"] = "/tmp/ambiguous_polluted.png"
        tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "context",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [polluted],
                }
            ],
            env_step=1,
            global_task="move one object",
            current_subtask="verify the released object",
            identity_relocation_leases=[
                {
                    "instance_ref": track_id,
                    "target_world_m": [0.0, 0.0, 0.65],
                    "tolerance_m": 0.08,
                }
            ],
        )
        near_left = detection_with_action_geometry(
            0,
            [120, 100, 140, 120],
            [-0.06, 0.0, 0.65],
            **stable_offsets,
        )
        near_left["mask_path"] = "/tmp/ambiguous_left.png"
        near_right = detection_with_action_geometry(
            1,
            [220, 100, 240, 120],
            [0.06, 0.0, 0.65],
            **stable_offsets,
        )
        near_right["mask_path"] = "/tmp/ambiguous_right.png"
        near_center_other_view = detection_with_action_geometry(
            0,
            [165, 150, 185, 170],
            [0.0, 0.02, 0.65],
            **stable_offsets,
        )
        near_center_other_view["mask_path"] = (
            "/tmp/ambiguous_center_other_view.png"
        )

        ambiguous = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [near_left, near_right],
                },
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    # A unique observation from another query/view must not
                    # override an unresolved two-candidate ambiguity.
                    "detections": [near_center_other_view],
                },
            ],
            env_step=2,
            global_task="move one object",
            current_subtask="reacquire the selected object",
        )

        instance = next(
            item
            for item in ambiguous["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(
            instance["action_geometry_state"],
            "relocation_pending",
        )
        self.assertEqual(
            instance["action_geometry_confirmation_count"],
            0,
        )
        self.assertEqual(
            ambiguous["temporal_memory"][
                "identity_binding_outcomes"
            ][0]["status"],
            "ambiguous_relocation_repair_candidates",
        )
        self.assertEqual(
            ambiguous["temporal_memory"][
                "identity_binding_outcomes"
            ][0]["ambiguous_query_count"],
            1,
        )
        self.assertEqual(
            ambiguous["temporal_memory"][
                "identity_binding_outcomes"
            ][0]["compatible_query_count"],
            1,
        )
        self.assertEqual(
            ambiguous["temporal_memory"][
                "num_visible_candidates"
            ],
            0,
        )

    def test_post_release_repair_lease_expires_after_bounded_observation_attempts(
        self,
    ) -> None:
        tracker = SceneMemoryTracker(
            max_missing_steps=2,
            temporal_action_geometry_distance_m=0.04,
        )
        stable_offsets = {
            "top_offset": [0.0, 0.0, 0.04],
            "approach_offset": [0.0, 0.0, 0.20],
            "contact_offset": [0.0, 0.0, 0.12],
        }
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "detections": [
                        detection_with_action_geometry(
                            0,
                            [220, 130, 242, 155],
                            [0.20, -0.10, 0.65],
                            **stable_offsets,
                        )
                    ],
                }
            ],
            env_step=0,
            global_task="move one object",
            current_subtask="move the selected object",
        )
        track_id = initial["instances"][0]["track_id"]
        polluted = detection_with_action_geometry(
            0,
            [170, 110, 190, 130],
            [0.0, 0.0, 0.65],
            top_offset=[0.0, 0.0, 0.12],
            approach_offset=[0.0, 0.0, 0.28],
            contact_offset=[0.0, 0.0, 0.20],
        )
        polluted["mask_path"] = "/tmp/expiry_polluted.png"
        tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "context",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [polluted],
                }
            ],
            env_step=1,
            global_task="move one object",
            current_subtask="verify the released object",
            identity_relocation_leases=[
                {
                    "instance_ref": track_id,
                    "target_world_m": [0.0, 0.0, 0.65],
                    "tolerance_m": 0.08,
                }
            ],
        )

        def ambiguous_frame(token: int) -> list[dict]:
            items = [
                detection_with_action_geometry(
                    0,
                    [120, 100, 140, 120],
                    [-0.06, 0.0, 0.65],
                    **stable_offsets,
                ),
                detection_with_action_geometry(
                    1,
                    [220, 100, 240, 120],
                    [0.06, 0.0, 0.65],
                    **stable_offsets,
                ),
            ]
            for index, item in enumerate(items):
                item["mask_path"] = (
                    f"/tmp/expiry_{token}_{index}.png"
                )
            return items

        for token in (1, 2):
            tracker.update(
                segmentation=[
                    {
                        "success": True,
                        "object_id": "component",
                        "query_role": "target",
                        "instance_ref": track_id,
                        "identity_binding_required": True,
                        "detections": ambiguous_frame(token),
                    }
                ],
                env_step=1,
                global_task="move one object",
                current_subtask="reacquire the selected object",
            )

        final_candidate = detection_with_action_geometry(
            0,
            [170, 112, 190, 132],
            [0.0, 0.0, 0.65],
            **stable_offsets,
        )
        final_candidate["mask_path"] = (
            "/tmp/expiry_after_limit.png"
        )
        expired = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [final_candidate],
                }
            ],
            env_step=1,
            global_task="move one object",
            current_subtask="reacquire the selected object",
        )

        instance = next(
            item
            for item in expired["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(
            instance["action_geometry_state"],
            "identity_repair_expired",
        )
        self.assertEqual(
            instance["stability"],
            "identity_repair_expired",
        )
        self.assertNotIn("identity_repair_lease", instance)
        self.assertEqual(
            instance["identity_repair_expiration"]["reason"],
            "post_release_identity_repair_lease_expired",
        )
        self.assertEqual(
            expired["temporal_memory"][
                "identity_binding_outcomes"
            ][0]["status"],
            "identity_repair_expired",
        )

    def test_post_release_repair_lease_time_expiry_is_exposed_then_ages_out(
        self,
    ) -> None:
        tracker = SceneMemoryTracker(
            max_missing_steps=2,
            temporal_action_geometry_distance_m=0.04,
        )
        stable_offsets = {
            "top_offset": [0.0, 0.0, 0.04],
            "approach_offset": [0.0, 0.0, 0.20],
            "contact_offset": [0.0, 0.0, 0.12],
        }
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "detections": [
                        detection_with_action_geometry(
                            0,
                            [220, 130, 242, 155],
                            [0.20, -0.10, 0.65],
                            **stable_offsets,
                        )
                    ],
                }
            ],
            env_step=0,
            global_task="move one object",
            current_subtask="move the selected object",
        )
        track_id = initial["instances"][0]["track_id"]
        polluted = detection_with_action_geometry(
            0,
            [170, 110, 190, 130],
            [0.0, 0.0, 0.65],
            top_offset=[0.0, 0.0, 0.12],
            approach_offset=[0.0, 0.0, 0.28],
            contact_offset=[0.0, 0.0, 0.20],
        )
        polluted["mask_path"] = (
            "/tmp/time_expiry_polluted.png"
        )
        quarantined = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "context",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [polluted],
                }
            ],
            env_step=1,
            global_task="move one object",
            current_subtask="verify the released object",
            identity_relocation_leases=[
                {
                    "instance_ref": track_id,
                    "target_world_m": [0.0, 0.0, 0.65],
                    "tolerance_m": 0.08,
                }
            ],
        )
        self.assertEqual(
            quarantined["instances"][0][
                "identity_repair_lease"
            ]["expires_step"],
            3,
        )

        expired = tracker.update(
            segmentation=[],
            env_step=4,
            global_task="move one object",
            current_subtask="reacquire the selected object",
        )
        expired_instance = next(
            item
            for item in expired["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(
            expired_instance["action_geometry_state"],
            "identity_repair_expired",
        )
        self.assertNotIn(
            "identity_repair_lease",
            expired_instance,
        )
        self.assertTrue(
            any(
                "must replan" in item
                for item in expired["uncertainty"]
            )
        )

        aged_out = tracker.update(
            segmentation=[],
            env_step=7,
            global_task="move one object",
            current_subtask="reacquire the selected object",
        )
        self.assertFalse(
            any(
                item["track_id"] == track_id
                for item in aged_out["instances"]
            )
        )

    def test_scene_memory_preserves_structured_operation_pose_candidates(self) -> None:
        tracker = SceneMemoryTracker()
        item = detection(0, [150, 150, 190, 185], [0.0, -0.15, 0.70])
        candidate = {
            "candidate_id": "rgbd_surface:grasp:right:000",
            "source_candidate_index": 0,
            "action_mode": "grasp",
            "arm": "right",
            "object_contact_pose": [0.0, -0.15, 0.74, 1.0, 0.0, 0.0, 0.0],
            "tcp_pose": [0.0, -0.15, 0.74, 1.0, 0.0, 0.0, 0.0],
            "ee_target_pose": [0.0, -0.15, 0.86, 1.0, 0.0, 0.0, 0.0],
            "approach_pose": [0.0, -0.15, 0.94, 1.0, 0.0, 0.0, 0.0],
            "approach_direction": [0.0, 0.0, -1.0],
            "geometry_source": "rgbd_surface_normal_principal_axes",
        }
        item["grounding_3d"]["operation_pose_candidates"] = [candidate]

        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "detections": [item],
                }
            ],
            env_step=0,
            global_task="opaque instruction",
            current_subtask="operate on the selected component",
        )

        instance = scene["instances"][0]
        self.assertEqual(instance["operation_pose_candidate_count"], 1)
        self.assertEqual(instance["operation_pose_candidates"], [candidate])

    def test_bound_multiview_reacquire_prefers_clean_geometry_before_redundancy(self) -> None:
        tracker = SceneMemoryTracker(
            max_missing_steps=20,
            temporal_action_geometry_distance_m=0.04,
        )
        stable_offsets = {
            "top_offset": [0.0, 0.0, 0.04],
            "approach_offset": [0.0, 0.0, 0.20],
            "contact_offset": [0.0, 0.0, 0.12],
        }
        initial_detection = detection_with_action_geometry(
            0,
            [150, 150, 190, 185],
            [0.097478, -0.203906, 0.775435],
            **stable_offsets,
        )
        initial_detection["grounding_3d"].update(
            {
                "bbox_world_min": [0.071480, -0.231690, 0.755441],
                "bbox_world_max": [0.123477, -0.176122, 0.795429],
            }
        )
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "selected_component",
                    "camera": "head",
                    "query_role": "target",
                    "detections": [initial_detection],
                }
            ],
            env_step=0,
            global_task="reposition one selected object",
            current_subtask="acquire the selected object",
        )
        track_id = initial["instances"][0]["track_id"]

        contaminated_head = detection_with_action_geometry(
            0,
            [155, 150, 198, 188],
            [0.095360, -0.214045, 0.778300],
            top_offset=[0.0, 0.0, 0.04],
            approach_offset=[0.0, 0.073, 0.20],
            contact_offset=[0.0, 0.073, 0.12],
            score=0.91,
        )
        contaminated_head["camera"] = "head"
        contaminated_head["mask_path"] = "/tmp/reacquire_head_contaminated.png"
        contaminated_head["grounding_3d"].update(
            {
                "bbox_world_min": [0.0690, -0.2420, 0.7583],
                "bbox_world_max": [0.1210, -0.1860, 0.7983],
            }
        )
        clean_third = detection_with_action_geometry(
            0,
            [205, 148, 245, 185],
            [0.102513, -0.193762, 0.770866],
            **stable_offsets,
            score=0.82,
        )
        clean_third["camera"] = "third"
        clean_third["mask_path"] = "/tmp/reacquire_third_clean.png"
        clean_third["grounding_3d"].update(
            {
                "bbox_world_min": [0.0765, -0.2215, 0.7509],
                "bbox_world_max": [0.1285, -0.1660, 0.7909],
            }
        )

        reacquired = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "selected_component",
                    "camera": "head",
                    "query_role": "target",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [contaminated_head],
                },
                {
                    "success": True,
                    "object_id": "selected_component",
                    "camera": "third",
                    "query_role": "target",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [clean_third],
                },
            ],
            env_step=1,
            global_task="reposition one selected object",
            current_subtask="reacquire after a failed grasp",
            position_events=[
                {
                    "instance_ref": track_id,
                    "position_state": "motion_uncertain",
                    "source": "failed_grasp",
                }
            ],
        )

        instance = next(
            item
            for item in reacquired["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(instance["position_state"], "current_verified")
        self.assertEqual(instance["camera"], "third")
        self.assertEqual(
            instance["latest_world_m"],
            [0.102513, -0.193762, 0.770866],
        )
        self.assertEqual(
            reacquired["temporal_memory"]["num_visible_candidates"],
            1,
        )
        self.assertTrue(
            any(
                warning.startswith(
                    "temporal_action_geometry_inconsistent:"
                )
                for dropped in reacquired["temporal_memory"][
                    "dropped_candidates"
                ]
                for warning in dropped["quality"]["warnings"]
            )
        )

    def test_bound_multiview_early_merge_preserves_inconsistent_geometry_variants(self) -> None:
        stable_offsets = {
            "top_offset": [0.0, 0.0, 0.04],
            "approach_offset": [0.0, 0.0, 0.20],
            "contact_offset": [0.0, 0.0, 0.12],
        }
        contaminated = detection_with_action_geometry(
            0,
            [150, 150, 190, 185],
            [0.0, -0.15, 0.70],
            top_offset=[0.0, 0.0, 0.04],
            approach_offset=[0.0, 0.073, 0.20],
            contact_offset=[0.0, 0.073, 0.12],
            score=0.95,
        )
        contaminated["camera"] = "head"
        contaminated["mask_path"] = "/tmp/early_merge_head.png"
        clean = detection_with_action_geometry(
            0,
            [205, 148, 245, 185],
            [0.0, -0.135, 0.70],
            **stable_offsets,
            score=0.80,
        )
        clean["camera"] = "third"
        clean["mask_path"] = "/tmp/early_merge_third.png"

        queries = [
            {
                "success": True,
                "object_id": "selected_component",
                "camera": candidate["camera"],
                "instance_ref": "track_0001",
                "identity_binding_required": True,
                "detections": [candidate],
            }
            for candidate in (contaminated, clean)
        ]
        preserved = extract_scene_candidates(
            queries,
            multiview_action_geometry_distance_m=0.04,
        )
        self.assertEqual(len(preserved), 2)

        consistent = detection_with_action_geometry(
            0,
            [205, 148, 245, 185],
            [0.0, -0.135, 0.70],
            **stable_offsets,
            score=0.80,
        )
        consistent["camera"] = "third"
        consistent["mask_path"] = "/tmp/early_merge_consistent_third.png"
        consistent_head = detection_with_action_geometry(
            0,
            [150, 150, 190, 185],
            [0.0, -0.15, 0.70],
            **stable_offsets,
            score=0.95,
        )
        consistent_head["camera"] = "head"
        consistent_head["mask_path"] = (
            "/tmp/early_merge_consistent_head.png"
        )
        merged = extract_scene_candidates(
            [
                {**queries[0], "detections": [consistent_head]},
                {**queries[1], "detections": [consistent]},
            ],
            multiview_action_geometry_distance_m=0.04,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            merged[0]["_supporting_cameras"],
            ["head", "third"],
        )

    def test_temporal_action_geometry_rejects_occluded_fragment_and_recovers(self) -> None:
        tracker = SceneMemoryTracker(
            max_missing_steps=20,
            temporal_action_geometry_distance_m=0.04,
        )
        clean = detection_with_action_geometry(
            0,
            [150, 150, 190, 185],
            [0.0, -0.15, 0.70],
            top_offset=[0.0, 0.0, 0.04],
            approach_offset=[0.0, 0.0, 0.20],
            contact_offset=[0.0, 0.0, 0.12],
        )
        initial = tracker.update(
            segmentation=[{"success": True, "object_id": "component", "query_role": "target", "detections": [clean]}],
            env_step=0,
            global_task="opaque instruction",
            current_subtask="operate on the selected component",
        )
        initial_instance = initial["instances"][0]
        initial_track_id = initial_instance["track_id"]

        contaminated = detection_with_action_geometry(
            0,
            [161, 166, 184, 176],
            [0.002, -0.13, 0.70],
            top_offset=[0.0, 0.10, 0.14],
            approach_offset=[0.0, 0.10, 0.22],
            contact_offset=[0.0, 0.10, 0.14],
        )
        rejected = tracker.update(
            segmentation=[
                {"success": True, "object_id": "component", "query_role": "target", "detections": [contaminated]}
            ],
            env_step=1,
            global_task="opaque instruction",
            current_subtask="operate on the selected component",
        )

        self.assertEqual(len(rejected["instances"]), 1)
        rejected_instance = rejected["instances"][0]
        self.assertEqual(rejected_instance["track_id"], initial_track_id)
        self.assertEqual(rejected_instance["status"], "tracked")
        self.assertEqual(rejected_instance["stability"], "geometry_inconsistent_current_frame")
        self.assertEqual(
            rejected_instance["position_state"],
            "memory_valid",
        )
        self.assertEqual(
            rejected_instance["last_verified_world_m"],
            initial_instance["last_verified_world_m"],
        )
        self.assertEqual(rejected_instance["contact_world_m"], initial_instance["contact_world_m"])
        self.assertEqual(rejected_instance["history_length"], 1)
        self.assertTrue(any("inconsistent action geometry" in item for item in rejected["uncertainty"]))
        self.assertEqual(rejected["temporal_memory"]["num_visible_candidates"], 0)
        self.assertEqual(rejected["temporal_memory"]["frames"][-1]["candidates"], [])
        dropped = rejected["temporal_memory"]["dropped_candidates"]
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["track_id"], initial_track_id)
        self.assertTrue(
            any(
                warning.startswith("temporal_action_geometry_inconsistent:")
                for warning in dropped[0]["quality"]["warnings"]
            )
        )

        reacquired = tracker.update(
            segmentation=[{"success": True, "object_id": "component", "query_role": "target", "detections": [clean]}],
            env_step=2,
            global_task="opaque instruction",
            current_subtask="operate on the selected component",
        )
        reacquired_instance = reacquired["instances"][0]
        self.assertEqual(reacquired_instance["track_id"], initial_track_id)
        self.assertEqual(reacquired_instance["status"], "visible")
        self.assertEqual(reacquired_instance["stability"], "stable")
        self.assertEqual(reacquired_instance["history_length"], 2)
        self.assertEqual(reacquired["temporal_memory"]["dropped_candidates"], [])

    def test_explicit_binding_accepts_retained_geometry_inconsistent_track(self) -> None:
        tracker = SceneMemoryTracker(
            max_missing_steps=20,
            temporal_action_geometry_distance_m=0.04,
        )
        first_subtask = "operate on the first selected component"
        second_subtask = "operate on the next Agent-selected component"
        clean = detection_with_action_geometry(
            0,
            [150, 150, 190, 185],
            [0.0, -0.15, 0.70],
            top_offset=[0.0, 0.0, 0.04],
            approach_offset=[0.0, 0.0, 0.20],
            contact_offset=[0.0, 0.0, 0.12],
        )
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "detections": [clean],
                }
            ],
            env_step=0,
            global_task="opaque instruction",
            current_subtask=first_subtask,
        )
        contaminated = detection_with_action_geometry(
            0,
            [161, 166, 184, 176],
            [0.002, -0.13, 0.70],
            top_offset=[0.0, 0.10, 0.14],
            approach_offset=[0.0, 0.10, 0.22],
            contact_offset=[0.0, 0.10, 0.14],
        )
        rejected = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "instance_ref": "track_0001",
                    "identity_binding_required": True,
                    "detections": [contaminated],
                }
            ],
            env_step=1,
            global_task="opaque instruction",
            current_subtask=second_subtask,
        )

        target_id = rejected["task_focus"]["target_instances"][0]
        target = next(item for item in rejected["instances"] if item["instance_id"] == target_id)
        self.assertEqual(target["track_id"], initial["instances"][0]["track_id"])
        self.assertEqual(target["status"], "tracked")
        self.assertEqual(target["stability"], "geometry_inconsistent_current_frame")
        self.assertEqual(rejected["task_focus"]["identity_binding_errors"], [])
        self.assertNotIn("temporal_binding_retained", rejected["task_focus"])

    def test_temporal_action_geometry_accepts_rigid_translation(self) -> None:
        tracker = SceneMemoryTracker(temporal_action_geometry_distance_m=0.04)
        offsets = {
            "top_offset": [0.0, 0.0, 0.04],
            "approach_offset": [0.0, 0.0, 0.20],
            "contact_offset": [0.0, 0.0, 0.12],
        }
        first = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "detections": [
                        detection_with_action_geometry(0, [150, 150, 190, 185], [0.0, -0.15, 0.70], **offsets)
                    ],
                }
            ],
            env_step=0,
            global_task="opaque instruction",
            current_subtask="",
        )
        track_id = first["instances"][0]["track_id"]
        translated_world = [0.03, -0.12, 0.72]
        translated = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "detections": [
                        detection_with_action_geometry(0, [155, 145, 195, 180], translated_world, **offsets)
                    ],
                }
            ],
            env_step=1,
            global_task="opaque instruction",
            current_subtask="",
        )

        instance = translated["instances"][0]
        self.assertEqual(instance["track_id"], track_id)
        self.assertEqual(instance["status"], "visible")
        self.assertEqual(instance["latest_world_m"], translated_world)
        self.assertEqual(instance["history_length"], 2)
        self.assertEqual(translated["temporal_memory"]["dropped_candidates"], [])

    def test_instances_keep_track_identity_without_spatial_role_labels(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "lid",
                    "query_role": "context",
                    "detections": [
                        detection(0, [220, 100, 280, 140], [0.2, -0.05, 0.82], score=0.9),
                        detection(1, [70, 100, 120, 140], [-0.2, -0.05, 0.82], score=0.5),
                        detection(2, [145, 100, 200, 140], [0.0, -0.05, 0.82], score=0.4),
                    ],
                }
            ],
            env_step=0,
            global_task="cover the blocks from left to right",
            current_subtask="cover the left block",
        )

        ids = [item["instance_id"] for item in scene["instances"]]
        self.assertEqual(ids, ["track_0001", "track_0002", "track_0003"])
        self.assertEqual(ids, [item["track_id"] for item in scene["instances"]])
        self.assertEqual(
            sorted(item["world_cm"][0] for item in scene["instances"]),
            [-20.0, 0.0, 20.0],
        )
        for instance in scene["instances"]:
            self.assertNotIn("role", instance)
            self.assertNotIn("ordinal", instance)
            self.assertNotIn("count_in_class", instance)
        self.assertEqual(scene["task_focus"]["target_instances"], [])

    def test_missing_instance_is_retained_temporarily(self) -> None:
        tracker = SceneMemoryTracker(max_missing_steps=20)
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "lid",
                    "detections": [
                        detection(0, [70, 100, 120, 140], [-0.2, -0.05, 0.82]),
                        detection(1, [145, 100, 200, 140], [0.0, -0.05, 0.82]),
                        detection(2, [220, 100, 280, 140], [0.2, -0.05, 0.82]),
                    ],
                }
            ],
            env_step=0,
            global_task="cover the blocks from left to right",
            current_subtask="",
        )
        middle_track_id = next(
            item["track_id"]
            for item in initial["instances"]
            if item["world_m"][0] == 0.0
        )
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "lid",
                    "detections": [
                        detection(0, [70, 100, 120, 140], [-0.2, -0.05, 0.82]),
                        detection(1, [220, 100, 280, 140], [0.2, -0.05, 0.82]),
                    ],
                }
            ],
            env_step=5,
            global_task="cover the blocks from left to right",
            current_subtask="",
        )

        tracked = [item for item in scene["instances"] if item["status"] == "tracked"]
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0]["instance_id"], middle_track_id)
        self.assertEqual(tracked[0]["track_id"], middle_track_id)

    def test_legacy_role_hint_does_not_select_from_multiple_candidates(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "widget",
                    "query_role": "target",
                    "query_instance_hint": "right",
                    "query_reason": "VLM selected the right widget as the active target.",
                    "detections": [
                        detection(0, [70, 100, 120, 140], [-0.2, -0.05, 0.82]),
                        detection(1, [220, 100, 280, 140], [0.2, -0.05, 0.82]),
                    ],
                },
                {
                    "success": True,
                    "object_id": "handle",
                    "query_role": "tool",
                    "query_reason": "VLM selected the handle as the manipulation tool.",
                    "detections": [
                        detection(0, [135, 90, 160, 120], [0.1, -0.05, 0.82]),
                    ],
                },
            ],
            env_step=0,
            global_task="opaque instruction with no built-in object words",
            current_subtask="",
        )

        self.assertEqual(scene["task_focus"]["target_instances"], [])
        handle = next(item for item in scene["instances"] if item["class"] == "handle")
        self.assertEqual(scene["task_focus"]["tool_instances"], [handle["track_id"]])
        self.assertIn(
            "widget:ambiguous_instance_candidates:2",
            scene["task_focus"]["identity_binding_errors"],
        )
        self.assertTrue(scene["task_focus"]["identity_binding_required"])
        self.assertEqual(scene["task_focus"]["source"], "vlm_perception_queries")

    def test_free_form_spatial_hints_do_not_bypass_exact_binding(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "widget",
                    "query_role": "target",
                    "query_instance_hint": "middle widget between the leftmost and rightmost widgets",
                    "detections": [
                        detection(0, [70, 100, 120, 140], [-0.2, -0.05, 0.82]),
                        detection(1, [145, 100, 195, 140], [0.0, -0.05, 0.82]),
                        detection(2, [220, 100, 280, 140], [0.2, -0.05, 0.82]),
                    ],
                },
                {
                    "success": True,
                    "object_id": "handle",
                    "query_role": "tool",
                    "query_instance_hint": "middle handle; avoid the left handle already used",
                    "detections": [
                        detection(3, [70, 60, 120, 90], [-0.2, 0.05, 0.82]),
                        detection(4, [145, 60, 195, 90], [0.0, 0.05, 0.82]),
                        detection(5, [220, 60, 280, 90], [0.2, 0.05, 0.82]),
                    ],
                },
            ],
            env_step=0,
            global_task="opaque instruction",
            current_subtask="operate on the selected pair",
        )

        self.assertEqual(scene["task_focus"]["target_instances"], [])
        self.assertEqual(scene["task_focus"]["tool_instances"], [])
        self.assertEqual(
            scene["task_focus"]["identity_binding_errors"],
            [
                "widget:ambiguous_instance_candidates:3",
                "handle:ambiguous_instance_candidates:3",
            ],
        )

    def test_center_is_not_a_runtime_synonym_for_middle(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "widget",
                    "query_role": "target",
                    "query_instance_hint": "center widget selected for manipulation",
                    "detections": [
                        detection(0, [70, 100, 120, 140], [-0.2, -0.05, 0.82]),
                        detection(1, [145, 100, 195, 140], [0.0, -0.05, 0.82]),
                        detection(2, [220, 100, 280, 140], [0.2, -0.05, 0.82]),
                    ],
                }
            ],
            env_step=0,
            global_task="opaque instruction",
            current_subtask="operate on the selected instance",
        )

        self.assertEqual(scene["task_focus"]["target_instances"], [])
        self.assertEqual(
            scene["task_focus"]["identity_binding_errors"],
            ["widget:ambiguous_instance_candidates:3"],
        )

    def test_detection_candidate_inherits_parent_grounding(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "widget",
                    "query_role": "target",
                    "grounding_3d": {
                        "success": True,
                        "centroid_world": [0.1, 0.2, 0.3],
                        "top_surface_world": [0.1, 0.2, 0.34],
                        "approach_point_world": [0.1, 0.2, 0.38],
                    },
                    "detections": [
                        {
                            "rank": 0,
                            "score": 0.9,
                            "bbox_xyxy": [1, 2, 3, 4],
                            "centroid_px": [2.0, 3.0],
                            "grounding_3d": {
                                "success": False,
                                "error": "candidate grounding unavailable",
                            },
                        }
                    ],
                }
            ],
            env_step=0,
            global_task="opaque task",
            current_subtask="",
        )

        instance = scene["instances"][0]
        self.assertEqual(instance["instance_id"], "track_0001")
        self.assertEqual(instance["world_m"], [0.1, 0.2, 0.3])
        self.assertEqual(instance["approach_world_m"], [0.1, 0.2, 0.38])
        self.assertEqual(instance["top_surface_world_m"], [0.1, 0.2, 0.34])
        self.assertEqual(instance["grasp_world_m"], [0.1, 0.2, 0.36])
        self.assertEqual(instance["contact_world_m"], [0.1, 0.2, 0.34])

    def test_grounded_affordance_pose_propagates_position_and_quaternion(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "tool",
                    "query_role": "tool",
                    "grounding_3d": {
                        "success": True,
                        "centroid_world": [0.1, 0.2, 0.3],
                        "top_surface_world": [0.1, 0.2, 0.34],
                        "approach_point_world": [0.1, 0.2, 0.42],
                        "approach_pose_world": [0.1, 0.2, 0.55, 0.5, -0.5, 0.5, 0.5],
                        "grasp_pose_world": [0.1, 0.2, 0.47, 0.5, -0.5, 0.5, 0.5],
                    },
                }
            ],
            env_step=0,
            global_task="opaque task",
            current_subtask="use the tool",
        )

        instance = scene["instances"][0]
        self.assertEqual(instance["approach_world_m"], [0.1, 0.2, 0.55])
        self.assertEqual(instance["grasp_world_m"], [0.1, 0.2, 0.47])
        self.assertEqual(instance["approach_quat_wxyz"], [0.5, -0.5, 0.5, 0.5])
        self.assertEqual(instance["grasp_quat_wxyz"], [0.5, -0.5, 0.5, 0.5])

    def test_scene_memory_uses_vlm_normalized_object_class_without_semantic_rewrite(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "lid",
                    "query_role": "tool",
                    "query_instance_hint": "left brown",
                    "detections": [
                        detection(0, [80, 100, 125, 140], [-0.18, -0.05, 0.82]),
                    ],
                }
            ],
            env_step=0,
            global_task="opaque task",
            current_subtask="",
        )

        self.assertEqual(scene["instances"][0]["instance_id"], "track_0001")
        self.assertEqual(scene["instances"][0]["class"], "lid")
        self.assertEqual(scene["task_focus"]["tool_instances"], ["track_0001"])

    def test_scene_memory_does_not_rewrite_unormalized_object_class(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "left_brown_lid",
                    "query_role": "tool",
                    "detections": [
                        detection(0, [80, 100, 125, 140], [-0.18, -0.05, 0.82]),
                    ],
                }
            ],
            env_step=0,
            global_task="opaque task",
            current_subtask="",
        )

        self.assertEqual(scene["instances"][0]["instance_id"], "track_0001")
        self.assertEqual(scene["instances"][0]["class"], "left_brown_lid")

    def test_overlapping_prompt_variants_are_merged_by_geometry(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "lid",
                    "query_role": "tool",
                    "detections": [
                        detection(0, [80, 100, 125, 140], [-0.18, -0.05, 0.82], score=0.7),
                    ],
                },
                {
                    "success": True,
                    "object_id": "lid",
                    "query_role": "tool",
                    "query_instance_hint": "left brown",
                    "detections": [
                        detection(1, [82, 101, 126, 141], [-0.181, -0.051, 0.821], score=0.8),
                    ],
                },
            ],
            env_step=0,
            global_task="opaque task",
            current_subtask="",
        )

        self.assertEqual(len(scene["instances"]), 1)
        self.assertEqual(scene["instances"][0]["instance_id"], "track_0001")
        self.assertEqual(scene["instances"][0]["score"], 0.8)

    def test_temporal_tracker_keeps_track_id_when_score_order_changes(self) -> None:
        tracker = SceneMemoryTracker(temporal_window_size=4)
        first = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "lid",
                    "detections": [
                        detection(0, [70, 100, 120, 140], [-0.2, -0.05, 0.82], score=0.9),
                        detection(1, [145, 100, 200, 140], [0.0, -0.05, 0.82], score=0.8),
                        detection(2, [220, 100, 280, 140], [0.2, -0.05, 0.82], score=0.7),
                    ],
                }
            ],
            env_step=0,
            global_task="opaque task",
            current_subtask="",
        )
        first_track_by_x = {
            item["world_m"][0]: item["track_id"] for item in first["instances"]
        }

        second = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "lid",
                    "detections": [
                        detection(0, [220, 100, 280, 140], [0.2, -0.05, 0.82], score=0.95),
                        detection(1, [70, 100, 120, 140], [-0.201, -0.05, 0.82], score=0.4),
                        detection(2, [145, 100, 200, 140], [0.0, -0.05, 0.82], score=0.3),
                    ],
                }
            ],
            env_step=1,
            global_task="opaque task",
            current_subtask="",
        )

        self.assertEqual(
            [item["instance_id"] for item in second["instances"]],
            ["track_0001", "track_0002", "track_0003"],
        )
        second_track_by_x = {
            round(item["world_m"][0], 1): item["track_id"]
            for item in second["instances"]
        }
        self.assertEqual(second_track_by_x, first_track_by_x)
        self.assertTrue(all(item["stability"] == "stable" for item in second["instances"]))
        self.assertEqual(second["temporal_memory"]["window_size"], 4)
        self.assertEqual(len(second["temporal_memory"]["frames"]), 2)

    def test_empty_sam_success_does_not_create_visible_instance(self) -> None:
        tracker = SceneMemoryTracker(max_missing_steps=20)
        first = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "lid",
                    "query_role": "tool",
                    "detections": [detection(0, [70, 100, 120, 140], [-0.2, -0.05, 0.82])],
                }
            ],
            env_step=0,
            global_task="opaque task",
            current_subtask="",
        )
        self.assertEqual(first["instances"][0]["status"], "visible")

        second = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "lid",
                    "query_role": "tool",
                    "detections": [],
                }
            ],
            env_step=1,
            global_task="opaque task",
            current_subtask="",
        )

        self.assertEqual(len(second["instances"]), 1)
        self.assertEqual(second["instances"][0]["status"], "tracked")
        self.assertEqual(second["instances"][0]["world_m"], [-0.2, -0.05, 0.82])
        self.assertEqual(second["instances"][0]["missing_steps"], 1)
        self.assertEqual(second["temporal_memory"]["num_visible_candidates"], 0)

    def test_temporal_tracker_matches_prompt_aliases_by_geometry(self) -> None:
        tracker = SceneMemoryTracker()
        first = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "brown_box",
                    "query_role": "tool",
                    "detections": [detection(0, [70, 100, 120, 140], [-0.2, -0.05, 0.82])],
                }
            ],
            env_step=0,
            global_task="opaque task",
            current_subtask="",
        )
        track_id = first["instances"][0]["track_id"]

        second = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "lid",
                    "query_role": "tool",
                    "detections": [detection(0, [71, 101, 121, 141], [-0.201, -0.051, 0.821])],
                }
            ],
            env_step=1,
            global_task="opaque task",
            current_subtask="",
        )

        self.assertEqual(len(second["instances"]), 1)
        self.assertEqual(second["instances"][0]["track_id"], track_id)
        self.assertIn("brown_box", second["instances"][0]["class_aliases"])
        self.assertIn("lid", second["instances"][0]["class_aliases"])

    def test_role_conflicted_cross_class_geometry_does_not_pollute_or_autoselect(self) -> None:
        tracker = SceneMemoryTracker(max_missing_steps=20)
        first = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "block",
                    "query_role": "target",
                    "detections": [
                        detection(0, [70, 150, 120, 190], [-0.2, -0.18, 0.76]),
                        detection(1, [145, 150, 195, 190], [0.0, -0.18, 0.76]),
                        detection(2, [220, 150, 270, 190], [0.2, -0.18, 0.76]),
                    ],
                },
                {
                    "success": True,
                    "object_id": "lid",
                    "query_role": "tool",
                    "detections": [
                        detection(3, [70, 90, 120, 130], [-0.2, -0.05, 0.82]),
                        detection(4, [145, 90, 195, 130], [0.0, -0.05, 0.82]),
                        detection(5, [220, 90, 270, 130], [0.2, -0.05, 0.82]),
                    ],
                },
            ],
            env_step=0,
            global_task="cover the blocks from left to right with lids",
            current_subtask="cover the left block",
        )

        self.assertEqual(first["task_focus"]["target_instances"], [])
        self.assertEqual(first["task_focus"]["tool_instances"], [])
        self.assertEqual(
            first["task_focus"]["identity_binding_errors"],
            [
                "block:ambiguous_instance_candidates:3",
                "lid:ambiguous_instance_candidates:3",
            ],
        )
        block_track_id = next(
            item["track_id"]
            for item in first["instances"]
            if item["class"] == "block" and item["world_m"][0] == -0.2
        )
        lid_track_id = next(
            item["track_id"]
            for item in first["instances"]
            if item["class"] == "lid" and item["world_m"][0] == -0.2
        )

        second = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "block",
                    "query_role": "target",
                    "detections": [],
                },
                {
                    "success": True,
                    "object_id": "lid",
                    "query_role": "tool",
                    "detections": [
                        detection(0, [71, 151, 121, 191], [-0.201, -0.181, 0.761], score=0.95),
                    ],
                },
            ],
            env_step=4,
            global_task="cover the blocks from left to right with lids",
            current_subtask="cover the left block",
        )

        block_track = next(item for item in second["instances"] if item["track_id"] == block_track_id)
        lid_track = next(item for item in second["instances"] if item["track_id"] == lid_track_id)
        self.assertEqual(block_track["query_role"], "target")
        self.assertNotIn("lid", block_track["class_aliases"])
        self.assertEqual(lid_track["status"], "tracked")
        self.assertEqual(second["task_focus"]["target_instances"], [])
        self.assertEqual(second["task_focus"]["tool_instances"], [])
        self.assertTrue(
            any(
                "role_conflict_with_current_target" in warning
                for item in second["temporal_memory"]["dropped_candidates"]
                for warning in item["quality"]["warnings"]
            )
        )

    def test_cross_camera_image_geometry_does_not_merge_without_world(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "lid",
                    "camera": "head",
                    "detections": [image_only_detection(0, [80, 100, 125, 140])],
                },
                {
                    "success": True,
                    "object_id": "lid",
                    "camera": "left",
                    "detections": [image_only_detection(1, [80, 100, 125, 140], score=0.8)],
                },
            ],
            env_step=0,
            global_task="opaque task",
            current_subtask="",
        )

        self.assertEqual(len(scene["instances"]), 2)
        self.assertEqual({item["camera"] for item in scene["instances"]}, {"head", "left"})

    def test_cross_camera_world_geometry_can_merge(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "lid",
                    "camera": "head",
                    "detections": [detection(0, [80, 100, 125, 140], [-0.2, -0.05, 0.82], score=0.7)],
                },
                {
                    "success": True,
                    "object_id": "lid",
                    "camera": "left",
                    "detections": [detection(1, [10, 20, 40, 60], [-0.201, -0.049, 0.821], score=0.9)],
                },
            ],
            env_step=0,
            global_task="opaque task",
            current_subtask="",
        )

        self.assertEqual(len(scene["instances"]), 1)
        self.assertEqual(scene["instances"][0]["camera"], "left")
        self.assertEqual(scene["instances"][0]["score"], 0.9)

    def test_scene_focus_rejects_low_score_large_grounding_and_keeps_tracked_target(self) -> None:
        tracker = SceneMemoryTracker(max_missing_steps=20)
        first = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "block",
                    "query_role": "target",
                    "detections": [
                        detection_with_extent(0, [80, 160, 105, 185], [-0.2, -0.18, 0.77], [0.04, 0.04, 0.04]),
                    ],
                },
                {
                    "success": True,
                    "object_id": "lid",
                    "query_role": "tool",
                    "detections": [
                        detection_with_extent(0, [130, 105, 175, 130], [-0.05, -0.05, 0.82], [0.08, 0.08, 0.04]),
                    ],
                },
            ],
            env_step=0,
            global_task="cover the block",
            current_subtask="cover the left block",
        )
        target_track_id = next(item["track_id"] for item in first["instances"] if item["class"] == "block")
        tool_track_id = next(item["track_id"] for item in first["instances"] if item["class"] == "lid")
        self.assertEqual(first["task_focus"]["target_instances"], [target_track_id])
        self.assertEqual(first["task_focus"]["tool_instances"], [tool_track_id])

        second = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "block",
                    "query_role": "target",
                    "detections": [],
                },
                {
                    "success": True,
                    "object_id": "lid",
                    "query_role": "tool",
                    "detections": [
                        detection_with_extent(
                            0,
                            [131, 48, 173, 60],
                            [-0.049, -0.018, 1.003],
                            [0.10, 0.37, 0.27],
                            score=0.10,
                        ),
                    ],
                }
            ],
            env_step=12,
            global_task="cover the block",
            current_subtask="cover the left block",
        )

        self.assertEqual(second["task_focus"]["target_instances"], [target_track_id])
        self.assertEqual(second["task_focus"]["tool_instances"], [tool_track_id])
        dropped = second["temporal_memory"]["dropped_candidates"]
        self.assertEqual(len(dropped), 1)
        warnings = dropped[0]["quality"]["warnings"]
        self.assertTrue(any("sam_score_below_min" in item for item in warnings))
        self.assertTrue(any("z_extent_too_large" in item for item in warnings))
        self.assertTrue(any("xy_extent_too_large" in item for item in warnings))
        self.assertTrue(any("world_z_above_workspace" in item for item in warnings))

    def test_scene_focus_rejects_high_workspace_false_positive_even_with_small_extent(self) -> None:
        tracker = SceneMemoryTracker(max_missing_steps=20)
        first = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "block",
                    "query_role": "target",
                    "detections": [
                        detection_with_extent(0, [80, 160, 105, 185], [-0.2, -0.18, 0.77], [0.04, 0.04, 0.04]),
                    ],
                }
            ],
            env_step=0,
            global_task="cover the block",
            current_subtask="cover the left block",
        )
        target_track_id = first["instances"][0]["track_id"]
        self.assertEqual(first["task_focus"]["target_instances"], [target_track_id])

        second = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "block",
                    "query_role": "target",
                    "detections": [
                        detection_with_extent(
                            0,
                            [131, 48, 173, 60],
                            [-0.049, -0.018, 1.003],
                            [0.04, 0.04, 0.04],
                            score=0.8,
                        ),
                    ],
                }
            ],
            env_step=12,
            global_task="cover the block",
            current_subtask="cover the left block",
        )

        self.assertEqual(second["task_focus"]["target_instances"], [target_track_id])
        dropped = second["temporal_memory"]["dropped_candidates"]
        self.assertEqual(len(dropped), 1)
        warnings = dropped[0]["quality"]["warnings"]
        self.assertFalse(any("sam_score_below_min" in item for item in warnings))
        self.assertFalse(any("z_extent_too_large" in item for item in warnings))
        self.assertFalse(any("xy_extent_too_large" in item for item in warnings))
        self.assertTrue(any("world_z_above_workspace" in item for item in warnings))

    def test_scene_memory_rejects_robot_self_geometry_candidate(self) -> None:
        tracker = SceneMemoryTracker(
            robot_self_filter_radius_m=0.08,
            robot_self_filter_z_margin_m=0.08,
        )
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "target",
                    "query_role": "target",
                    "robot_state": {
                        "left": {"xyz": [0.1, -0.1, 0.92]},
                        "right": {"xyz": [0.3, -0.3, 0.94]},
                    },
                    "detections": [
                        detection(0, [120, 80, 170, 140], [0.105, -0.105, 0.925], score=0.9),
                    ],
                }
            ],
            env_step=0,
            global_task="opaque task",
            current_subtask="",
        )

        self.assertEqual(scene["instances"], [])
        dropped = scene["temporal_memory"]["dropped_candidates"]
        self.assertEqual(len(dropped), 1)
        warnings = dropped[0]["quality"]["warnings"]
        self.assertTrue(any("robot_self_geometry:left_ee_distance" in item for item in warnings))

    def test_oracle_simulator_candidates_bypass_sam_quality_rejection(self) -> None:
        tracker = SceneMemoryTracker(max_missing_steps=20)
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "cover",
                    "query_role": "tool",
                    "backend": "oracle_simulator",
                    "oracle_aliases": ["cover", "lid"],
                    "detections": [
                        {
                            **detection_with_extent(0, [0, 0, 1, 1], [-0.2, -0.05, 0.82], [0.12, 0.16, 0.20], score=1.0),
                            "camera": "oracle",
                        },
                    ],
                }
            ],
            env_step=0,
            global_task="cover the block",
            current_subtask="cover the block",
        )

        self.assertEqual(scene["task_focus"]["tool_instances"], ["track_0001"])
        self.assertEqual(scene["instances"][0]["class_aliases"], ["cover", "lid"])
        self.assertEqual(scene["temporal_memory"]["dropped_candidates"], [])

    def test_oracle_actor_identity_survives_large_motion_and_query_alias_change(self) -> None:
        tracker = SceneMemoryTracker(stable_distance_m=0.08)
        first = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "cover",
                    "query_role": "tool",
                    "backend": "oracle_simulator",
                    "oracle_id": "covers_0",
                    "oracle_source_path": "covers[0]",
                    "oracle_class_name": "cover",
                    "oracle_aliases": ["cover", "lid"],
                    "detections": [detection(0, [70, 100, 120, 140], [-0.2, -0.05, 0.82])],
                }
            ],
            env_step=0,
            global_task="opaque task",
            current_subtask="move the selected tool",
        )
        track_id = first["instances"][0]["track_id"]

        second = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "lid",
                    "query_role": "tool",
                    "backend": "oracle_simulator",
                    "oracle_id": "covers_0",
                    "oracle_source_path": "covers[0]",
                    "oracle_class_name": "cover",
                    "oracle_aliases": ["cover", "lid"],
                    "detections": [detection(0, [170, 100, 220, 140], [-0.2, -0.20, 0.86])],
                }
            ],
            env_step=1,
            global_task="opaque task",
            current_subtask="move the selected tool",
        )

        self.assertEqual(len(second["instances"]), 1)
        self.assertEqual(second["instances"][0]["track_id"], track_id)
        self.assertEqual(second["instances"][0]["class"], "cover")
        self.assertEqual(second["instances"][0]["first_observed_world_m"], [-0.2, -0.05, 0.82])
        self.assertEqual(second["instances"][0]["latest_world_m"], [-0.2, -0.2, 0.86])
        self.assertEqual(second["temporal_memory"]["num_active_tracks"], 1)

    def test_oracle_bound_query_keeps_actor_id_association_path(self) -> None:
        tracker = SceneMemoryTracker(stable_distance_m=0.08)
        first = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "cover",
                    "query_role": "tool",
                    "backend": "oracle_simulator",
                    "oracle_id": "covers_0",
                    "oracle_source_path": "covers[0]",
                    "oracle_class_name": "cover",
                    "oracle_aliases": ["cover", "lid"],
                    "detections": [
                        detection(0, [70, 100, 120, 140], [-0.2, -0.05, 0.82]),
                    ],
                }
            ],
            env_step=0,
            global_task="cover the block",
            current_subtask="select the cover",
        )
        track_id = first["instances"][0]["track_id"]

        second = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "lid",
                    "query_role": "tool",
                    "backend": "oracle_simulator",
                    "oracle_id": "covers_0",
                    "oracle_source_path": "covers[0]",
                    "oracle_class_name": "cover",
                    "oracle_aliases": ["cover", "lid"],
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection(0, [170, 100, 220, 140], [-0.2, -0.20, 0.86]),
                    ],
                }
            ],
            env_step=1,
            global_task="cover the block",
            current_subtask="move the selected cover",
        )

        self.assertEqual(len(second["instances"]), 1)
        self.assertEqual(second["instances"][0]["track_id"], track_id)
        self.assertEqual(second["instances"][0]["latest_world_m"], [-0.2, -0.2, 0.86])
        self.assertEqual(second["temporal_memory"]["identity_binding_outcomes"], [])
        self.assertEqual(second["temporal_memory"]["dropped_candidates"], [])

    def test_distinct_oracle_actor_ids_do_not_merge_at_nearby_geometry(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "part",
                    "query_role": "tool",
                    "backend": "oracle_simulator",
                    "oracle_id": "parts_0",
                    "oracle_source_path": "parts[0]",
                    "oracle_class_name": "part",
                    "detections": [detection(0, [70, 100, 120, 140], [-0.01, -0.05, 0.82])],
                },
                {
                    "success": True,
                    "object_id": "part",
                    "query_role": "tool",
                    "backend": "oracle_simulator",
                    "oracle_id": "parts_1",
                    "oracle_source_path": "parts[1]",
                    "oracle_class_name": "part",
                    "detections": [detection(1, [75, 100, 125, 140], [0.01, -0.05, 0.82])],
                },
            ],
            env_step=0,
            global_task="opaque task",
            current_subtask="",
        )

        self.assertEqual(len(scene["instances"]), 2)
        self.assertEqual(scene["temporal_memory"]["num_active_tracks"], 2)

    def test_release_verification_can_bind_exact_context_instance(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "cube",
                    "query_role": "context",
                    "detections": [
                        detection(
                            0,
                            [20, 30, 50, 70],
                            [0.12, -0.16, 0.76],
                        )
                    ],
                }
            ],
            env_step=0,
            global_task="generic manipulation",
            current_subtask="verify the released object",
        )
        track_id = scene["instances"][0]["track_id"]

        rebound = tracker.rebind_task_focus(
            scene,
            perception_queries=[
                {
                    "object_id": "cube",
                    "query_role": "context",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                }
            ],
            global_task="generic manipulation",
            current_subtask="verify the released object",
            identity_binding_required=True,
            allow_context_binding=True,
            previous_scene_memory=scene,
        )

        focus = rebound["task_focus"]
        self.assertEqual(focus["target_instances"], [])
        self.assertEqual(focus["tool_instances"], [])
        self.assertEqual(focus["context_instances"], [track_id])
        self.assertEqual(focus["identity_binding_roles"], ["context"])
        self.assertEqual(focus["identity_binding_errors"], [])

    def test_agent_instance_ref_selects_exact_stable_track(self) -> None:
        tracker = SceneMemoryTracker()
        tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "detections": [
                        detection(0, [20, 30, 50, 70], [0.13, -0.04, 0.81]),
                        detection(1, [170, 35, 205, 75], [-0.07, 0.09, 0.84]),
                    ],
                }
            ],
            env_step=0,
            global_task="opaque instruction",
            current_subtask="discover candidate components",
        )
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "query_instance_hint": "the first candidate",
                    "instance_ref": "track_0002",
                    "identity_binding_required": True,
                    "detections": [
                        detection(0, [20, 30, 50, 70], [0.13, -0.04, 0.81]),
                        detection(1, [170, 35, 205, 75], [-0.07, 0.09, 0.84]),
                    ],
                }
            ],
            env_step=1,
            global_task="opaque instruction",
            current_subtask="operate on the API-selected component",
        )

        selected_id = scene["task_focus"]["target_instances"][0]
        selected = next(item for item in scene["instances"] if item["instance_id"] == selected_id)
        self.assertEqual(selected["track_id"], "track_0002")
        self.assertEqual(scene["task_focus"]["identity_binding_errors"], [])
        self.assertEqual(scene["temporal_memory"]["num_active_tracks"], 2)
        outcome = scene["temporal_memory"]["identity_binding_outcomes"][0]
        self.assertEqual(outcome["instance_ref"], "track_0002")
        self.assertEqual(outcome["track_id"], "track_0002")
        self.assertEqual(outcome["status"], "matched")
        self.assertEqual(outcome["candidate_count"], 2)
        self.assertEqual(outcome["compatible_candidate_count"], 1)
        dropped = scene["temporal_memory"]["dropped_candidates"]
        self.assertEqual(len(dropped), 1)
        self.assertTrue(
            any(
                warning == "bound_identity_candidate_mismatch:track_0002"
                for warning in dropped[0]["quality"]["warnings"]
            )
        )

    def test_bound_query_false_positive_cannot_pollute_another_track(self) -> None:
        tracker = SceneMemoryTracker()
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "cube_row",
                    "query_role": "context",
                    "detections": [
                        detection(0, [200, 100, 240, 145], [0.274, -0.08, 0.77]),
                    ],
                },
                {
                    "success": True,
                    "object_id": "button",
                    "query_role": "context",
                    "detections": [
                        detection(1, [70, 95, 110, 140], [-0.199, -0.08, 0.77]),
                    ],
                },
            ],
            env_step=0,
            global_task="rank the cubes and confirm",
            current_subtask="inspect the scene",
        )
        red_track_id = next(
            item["track_id"]
            for item in initial["instances"]
            if item["class"] == "cube_row"
        )
        button_track_id = next(
            item["track_id"]
            for item in initial["instances"]
            if item["class"] == "button"
        )

        rebound = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "red_cube",
                    "query_role": "target",
                    "instance_ref": red_track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection(0, [201, 101, 241, 146], [0.275, -0.081, 0.771], score=0.84),
                        detection(1, [70, 95, 110, 140], [-0.199, -0.08, 0.77], score=0.35),
                    ],
                }
            ],
            env_step=1,
            global_task="rank the cubes and confirm",
            current_subtask="move the selected red cube",
        )

        by_track = {item["track_id"]: item for item in rebound["instances"]}
        self.assertEqual(rebound["temporal_memory"]["num_active_tracks"], 2)
        self.assertIn("red_cube", by_track[red_track_id]["class_aliases"])
        self.assertNotIn("red_cube", by_track[button_track_id]["class_aliases"])
        self.assertNotIn("button", by_track[red_track_id]["class_aliases"])
        self.assertEqual(by_track[button_track_id]["status"], "tracked")
        self.assertEqual(
            rebound["task_focus"]["target_instances"],
            [red_track_id],
        )
        dropped = rebound["temporal_memory"]["dropped_candidates"]
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["track_id"], red_track_id)
        self.assertEqual(dropped[0]["world_m"], [-0.199, -0.08, 0.77])
        self.assertIn(
            f"bound_identity_candidate_mismatch:{red_track_id}",
            dropped[0]["quality"]["warnings"],
        )

    def test_bound_query_refines_unique_member_from_aggregate_track(self) -> None:
        tracker = SceneMemoryTracker()
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "item_collection",
                    "query_role": "target",
                    "camera": "head",
                    "detections": [
                        detection_with_extent(
                            0,
                            [179, 138, 245, 157],
                            [0.100, -0.106, 0.773],
                            [0.176, 0.059, 0.040],
                        ),
                    ],
                }
            ],
            env_step=0,
            global_task="rearrange several generic items",
            current_subtask="inspect the item collection",
        )
        track_id = initial["instances"][0]["track_id"]

        refined = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "selected_item",
                    "query_role": "target",
                    "camera": "head",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection_with_extent(
                            0,
                            [179, 138, 195, 157],
                            [0.038, -0.107, 0.773],
                            [0.047, 0.043, 0.040],
                            score=0.89,
                        ),
                    ],
                }
            ],
            env_step=1,
            global_task="rearrange several generic items",
            current_subtask="move the selected item",
        )

        self.assertEqual(len(refined["instances"]), 1)
        instance = refined["instances"][0]
        self.assertEqual(instance["track_id"], track_id)
        self.assertEqual(instance["status"], "visible")
        self.assertEqual(instance["class"], "selected_item")
        self.assertEqual(instance["class_aliases"], ["selected_item"])
        self.assertEqual(instance["world_m"], [0.038, -0.107, 0.773])
        self.assertEqual(
            instance["latest_world_m"],
            [0.038, -0.107, 0.773],
        )
        self.assertEqual(
            instance["identity_granularity"],
            "refined_member",
        )
        self.assertEqual(instance["history_length"], 1)
        self.assertEqual(
            refined["task_focus"]["target_instances"],
            [track_id],
        )
        outcome = refined["temporal_memory"][
            "identity_binding_outcomes"
        ][0]
        self.assertEqual(outcome["status"], "matched")
        self.assertEqual(
            outcome["matched_by"],
            "aggregate_member_refinement",
        )
        self.assertGreaterEqual(
            outcome["refinement_evidence"]["candidate_containment"],
            0.85,
        )
        self.assertEqual(
            refined["temporal_memory"]["dropped_candidates"],
            [],
        )

    def test_bound_query_with_two_compatible_candidates_fails_closed(self) -> None:
        tracker = SceneMemoryTracker()
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "context",
                    "detections": [
                        detection(0, [140, 100, 180, 140], [0.0, -0.08, 0.77]),
                    ],
                }
            ],
            env_step=0,
            global_task="generic manipulation",
            current_subtask="inspect the scene",
        )
        track_id = initial["instances"][0]["track_id"]

        ambiguous = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "selected_component",
                    "query_role": "target",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection(0, [80, 100, 120, 140], [-0.03, -0.08, 0.77]),
                        detection(1, [200, 100, 240, 140], [0.03, -0.08, 0.77]),
                    ],
                }
            ],
            env_step=1,
            global_task="generic manipulation",
            current_subtask="operate on the selected component",
        )

        instance = ambiguous["instances"][0]
        self.assertEqual(instance["track_id"], track_id)
        self.assertEqual(instance["status"], "tracked")
        self.assertEqual(instance["history_length"], 1)
        self.assertNotIn("selected_component", instance["class_aliases"])
        self.assertEqual(ambiguous["temporal_memory"]["num_active_tracks"], 1)
        self.assertEqual(ambiguous["temporal_memory"]["num_visible_candidates"], 0)
        outcome = ambiguous["temporal_memory"]["identity_binding_outcomes"][0]
        self.assertEqual(outcome["status"], "ambiguous_candidates")
        self.assertEqual(outcome["compatible_candidate_count"], 2)
        self.assertEqual(len(ambiguous["temporal_memory"]["dropped_candidates"]), 2)

    def test_bound_query_accepts_one_unique_observation_from_multiple_cameras(self) -> None:
        tracker = SceneMemoryTracker()
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "context",
                    "camera": "head",
                    "detections": [
                        detection(0, [140, 100, 180, 140], [0.0, -0.08, 0.77]),
                    ],
                }
            ],
            env_step=0,
            global_task="generic manipulation",
            current_subtask="inspect the scene",
        )
        track_id = initial["instances"][0]["track_id"]

        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "selected_component",
                    "query_role": "target",
                    "camera": camera,
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection(
                            0,
                            bbox,
                            world,
                            score=score,
                        )
                    ],
                }
                for camera, bbox, world, score in (
                    ("head", [141, 101, 181, 141], [0.001, -0.081, 0.771], 0.90),
                    ("left", [80, 90, 125, 145], [-0.001, -0.079, 0.770], 0.85),
                    ("right", [210, 95, 250, 140], [0.002, -0.080, 0.769], 0.80),
                )
            ],
            env_step=1,
            global_task="generic manipulation",
            current_subtask="operate on the selected component",
        )

        instance = scene["instances"][0]
        self.assertEqual(instance["track_id"], track_id)
        self.assertEqual(instance["status"], "visible")
        self.assertEqual(instance["history_length"], 2)
        self.assertIn("selected_component", instance["class_aliases"])
        outcome = scene["temporal_memory"]["identity_binding_outcomes"][0]
        self.assertEqual(outcome["status"], "matched")
        self.assertEqual(outcome["query_group_count"], 1)
        self.assertEqual(outcome["compatible_query_count"], 1)
        self.assertEqual(
            instance["supporting_cameras"],
            ["head", "left", "right"],
        )
        self.assertEqual(instance["multiview_support_count"], 3)
        self.assertEqual(outcome["ambiguous_query_count"], 0)
        self.assertEqual(
            scene["temporal_memory"]["dropped_candidates"],
            [],
        )

    def test_unresolved_bound_reference_does_not_create_tracks(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "instance_ref": "track_9999",
                    "identity_binding_required": True,
                    "detections": [
                        detection(0, [20, 30, 50, 70], [0.13, -0.04, 0.81]),
                        detection(1, [170, 35, 205, 75], [-0.07, 0.09, 0.84]),
                    ],
                }
            ],
            env_step=0,
            global_task="opaque instruction",
            current_subtask="operate on the selected component",
        )

        self.assertEqual(scene["instances"], [])
        self.assertEqual(scene["temporal_memory"]["num_active_tracks"], 0)
        self.assertEqual(scene["temporal_memory"]["num_visible_candidates"], 0)
        self.assertEqual(len(scene["temporal_memory"]["dropped_candidates"]), 2)
        self.assertEqual(
            scene["temporal_memory"]["identity_binding_outcomes"][0]["status"],
            "unresolved_reference",
        )
        self.assertEqual(
            scene["task_focus"]["identity_binding_errors"],
            ["component:unresolved_instance_reference"],
        )

    def test_required_identity_binding_does_not_fall_back_to_first_candidate(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "identity_binding_required": True,
                    "detections": [
                        detection(0, [20, 30, 50, 70], [0.13, -0.04, 0.81]),
                        detection(1, [170, 35, 205, 75], [-0.07, 0.09, 0.84]),
                    ],
                }
            ],
            env_step=0,
            global_task="opaque instruction",
            current_subtask="operate on the API-selected component",
        )

        self.assertEqual(scene["task_focus"]["target_instances"], [])
        self.assertEqual(
            scene["task_focus"]["identity_binding_errors"],
            ["component:missing_instance_reference"],
        )

    def test_required_identity_binding_rejects_candidate_without_3d_grounding(self) -> None:
        tracker = SceneMemoryTracker()
        scene = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "instance_ref": "track_0001",
                    "identity_binding_required": True,
                    "detections": [image_only_detection(0, [20, 30, 50, 70])],
                }
            ],
            env_step=0,
            global_task="opaque instruction",
            current_subtask="operate on the API-selected component",
        )

        self.assertEqual(scene["task_focus"]["target_instances"], [])
        self.assertEqual(
            scene["task_focus"]["identity_binding_errors"],
            ["component:unresolved_instance_reference"],
        )

    def test_agent_committed_track_survives_temporary_occlusion_without_candidate_substitution(self) -> None:
        tracker = SceneMemoryTracker(max_missing_steps=20)
        subtask = "operate on the selected component"
        tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "detections": [
                        detection(0, [20, 30, 50, 70], [0.13, -0.04, 0.81]),
                        detection(1, [170, 35, 205, 75], [-0.07, 0.09, 0.84]),
                    ],
                }
            ],
            env_step=0,
            global_task="opaque instruction",
            current_subtask=subtask,
        )
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "instance_ref": "track_0002",
                    "identity_binding_required": True,
                    "detections": [
                        detection(0, [20, 30, 50, 70], [0.13, -0.04, 0.81]),
                        detection(1, [170, 35, 205, 75], [-0.07, 0.09, 0.84]),
                    ],
                }
            ],
            env_step=1,
            global_task="opaque instruction",
            current_subtask=subtask,
        )
        occluded = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "detections": [detection(1, [170, 35, 205, 75], [-0.07, 0.09, 0.84])],
                }
            ],
            env_step=2,
            global_task="opaque instruction",
            current_subtask=subtask,
        )

        rebound = tracker.rebind_task_focus(
            occluded,
            perception_queries=[],
            global_task="opaque instruction",
            current_subtask=subtask,
            identity_binding_required=True,
            previous_scene_memory=initial,
        )

        target_id = rebound["task_focus"]["target_instances"][0]
        target = next(item for item in rebound["instances"] if item["instance_id"] == target_id)
        self.assertEqual(target["track_id"], "track_0002")
        self.assertEqual(target["status"], "tracked")
        self.assertEqual(rebound["task_focus"]["identity_binding_errors"], [])
        self.assertTrue(rebound["task_focus"]["temporal_binding_retained"])
        self.assertNotIn("track_0001", rebound["task_focus"]["temporarily_occluded_tracks"])

        reacquired = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "instance_ref": "track_0002",
                    "identity_binding_required": True,
                    "detections": [
                        detection(0, [20, 30, 50, 70], [0.13, -0.04, 0.81]),
                        detection(1, [170, 35, 205, 75], [-0.07, 0.09, 0.84]),
                    ],
                }
            ],
            env_step=3,
            global_task="opaque instruction",
            current_subtask=subtask,
        )
        reacquired = tracker.rebind_task_focus(
            reacquired,
            perception_queries=[
                {
                    "object_id": "component",
                    "role": "target",
                    "instance_ref": "track_0002",
                    "identity_binding_required": True,
                }
            ],
            global_task="opaque instruction",
            current_subtask=subtask,
            identity_binding_required=True,
            previous_scene_memory=rebound,
        )
        reacquired_id = reacquired["task_focus"]["target_instances"][0]
        reacquired_target = next(item for item in reacquired["instances"] if item["instance_id"] == reacquired_id)
        self.assertEqual(reacquired_target["track_id"], "track_0002")
        self.assertEqual(reacquired_target["status"], "visible")
        self.assertNotIn("temporal_binding_retained", reacquired["task_focus"])

    def test_temporal_binding_does_not_cross_subtask_boundary(self) -> None:
        tracker = SceneMemoryTracker(max_missing_steps=20)
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "instance_ref": "track_0001",
                    "identity_binding_required": True,
                    "detections": [detection(0, [20, 30, 50, 70], [0.13, -0.04, 0.81])],
                }
            ],
            env_step=0,
            global_task="opaque instruction",
            current_subtask="first subtask",
        )
        occluded = tracker.update(
            segmentation=[],
            env_step=1,
            global_task="opaque instruction",
            current_subtask="second subtask",
        )

        rebound = tracker.rebind_task_focus(
            occluded,
            perception_queries=[],
            global_task="opaque instruction",
            current_subtask="second subtask",
            identity_binding_required=True,
            previous_scene_memory=initial,
        )

        self.assertEqual(rebound["task_focus"]["target_instances"], [])
        self.assertEqual(
            rebound["task_focus"]["identity_binding_errors"],
            ["agent_returned_no_target_or_tool_binding"],
        )

    def test_temporal_binding_expires_with_track_missing_budget(self) -> None:
        tracker = SceneMemoryTracker(max_missing_steps=1)
        subtask = "operate on the selected component"
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "query_role": "target",
                    "instance_ref": "track_0001",
                    "identity_binding_required": True,
                    "detections": [detection(0, [20, 30, 50, 70], [0.13, -0.04, 0.81])],
                }
            ],
            env_step=0,
            global_task="opaque instruction",
            current_subtask=subtask,
        )
        expired = tracker.update(
            segmentation=[],
            env_step=2,
            global_task="opaque instruction",
            current_subtask=subtask,
        )

        rebound = tracker.rebind_task_focus(
            expired,
            perception_queries=[],
            global_task="opaque instruction",
            current_subtask=subtask,
            identity_binding_required=True,
            previous_scene_memory=initial,
        )

        self.assertEqual(rebound["task_focus"]["target_instances"], [])
        self.assertEqual(
            rebound["task_focus"]["identity_binding_errors"],
            ["agent_returned_no_target_or_tool_binding"],
        )

    def test_occlusion_preserves_verified_world_position_until_physical_event(
        self,
    ) -> None:
        tracker = SceneMemoryTracker(max_missing_steps=1)
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "selected_item",
                    "camera": "head",
                    "detections": [
                        detection(
                            0,
                            [120, 90, 150, 125],
                            [0.0688, -0.1958, 0.7733],
                            score=0.91,
                        )
                    ],
                }
            ],
            env_step=0,
            global_task="move an item",
            current_subtask="observe the item",
        )
        track_id = initial["instances"][0]["track_id"]
        self.assertEqual(
            initial["instances"][0]["position_state"],
            "current_verified",
        )

        occluded = tracker.update(
            segmentation=[],
            env_step=3,
            global_task="move an item",
            current_subtask="approach the item",
        )
        remembered = occluded["instances"][0]
        self.assertEqual(
            remembered["position_state"],
            "memory_valid",
        )
        self.assertEqual(
            remembered["last_verified_world_m"],
            [0.0688, -0.1958, 0.7733],
        )
        self.assertEqual(
            remembered["world_m"],
            [0.0688, -0.1958, 0.7733],
        )

        invalidated = tracker.apply_runtime_events(
            occluded,
            events=[
                {
                    "instance_ref": track_id,
                    "position_state": "motion_uncertain",
                    "source": "unverified_contact",
                    "reason": "contact occurred without verification",
                }
            ],
            env_step=4,
        )
        invalidated_item = invalidated["instances"][0]
        self.assertEqual(
            invalidated_item["position_state"],
            "motion_uncertain",
        )
        self.assertEqual(
            invalidated_item["last_verified_world_m"],
            [0.0688, -0.1958, 0.7733],
        )
        self.assertEqual(
            invalidated_item["last_motion_event"]["source"],
            "unverified_contact",
        )

        attached = tracker.apply_runtime_events(
            invalidated,
            events=[
                {
                    "instance_ref": track_id,
                    "position_state": "current_verified",
                    "world_m": [0.20, 0.05, 0.88],
                    "confidence": 1.0,
                    "source": "verified_tcp_attachment",
                }
            ],
            env_step=5,
        )
        attached_item = attached["instances"][0]
        self.assertEqual(
            attached_item["world_m"],
            [0.20, 0.05, 0.88],
        )
        self.assertEqual(attached_item["history_length"], 0)
        self.assertIsNone(attached_item["approach_world_m"])

    def test_step32_world_anchor_rejects_far_same_semantic_false_positive(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "blue_cube",
                    "camera": "head",
                    "detections": [
                        detection(
                            0,
                            [170, 150, 198, 180],
                            [0.0688, -0.1958, 0.7733],
                            score=0.94,
                        )
                    ],
                }
            ],
            env_step=31,
            global_task="rank the blocks",
            current_subtask="reacquire the placed blue block",
        )
        track_id = initial["instances"][0]["track_id"]
        tracker.apply_runtime_events(
            initial,
            events=[
                {
                    "instance_ref": track_id,
                    "position_state": "current_verified",
                    "world_m": [0.0688, -0.1958, 0.7733],
                    "confidence": 1.0,
                    "tolerance_m": 0.028,
                    "source": "verified_post_release_position",
                }
            ],
            env_step=31,
        )
        rebound = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "blue_cube",
                    "camera": "head",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection(
                            0,
                            [171, 151, 199, 181],
                            [0.067641, -0.199028, 0.774154],
                            score=0.902,
                        ),
                        detection(
                            1,
                            [174, 152, 202, 182],
                            [0.037772, -0.106597, 0.773352],
                            score=0.211,
                        ),
                    ],
                }
            ],
            env_step=32,
            global_task="rank the blocks",
            current_subtask="reacquire the placed blue block",
        )

        instance = rebound["instances"][0]
        self.assertEqual(instance["track_id"], track_id)
        self.assertEqual(
            instance["latest_world_m"],
            [0.067641, -0.199028, 0.774154],
        )
        outcome = rebound["temporal_memory"][
            "identity_binding_outcomes"
        ][0]
        self.assertEqual(
            outcome["matched_by"],
            "verified_world_position",
        )
        self.assertEqual(
            outcome["verified_position_tolerance_m"],
            0.028,
        )
        self.assertAlmostEqual(
            outcome["matched_position_error_m"],
            0.0036,
            places=3,
        )
        warnings = [
            warning
            for item in rebound["temporal_memory"][
                "dropped_candidates"
            ]
            for warning in (
                (item.get("quality") or {}).get(
                    "warnings",
                    [],
                )
            )
        ]
        self.assertTrue(
            any(
                warning.startswith(
                    "bound_identity_outside_verified_position"
                )
                for warning in warnings
            )
        )

    def test_verified_operation_target_recovers_e2_candidate_outside_centroid_gate(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        observed_world = [0.096177, -0.119476, 0.773579]
        target_world = [0.099212, -0.100609, 0.776744]
        true_world = [0.091210, -0.094005, 0.772526]
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "block",
                    "camera": "head",
                    "detections": [
                        detection(
                            0,
                            [197, 143, 216, 164],
                            observed_world,
                            score=0.5,
                        )
                    ],
                }
            ],
            env_step=13,
            global_task="move an object through two placements",
            current_subtask="verify the first placement",
        )
        track_id = initial["instances"][0]["track_id"]
        released = tracker.apply_runtime_events(
            initial,
            events=[
                verified_release_event(
                    track_id,
                    observed_world=observed_world,
                    target_world=target_world,
                    tolerance_m=0.020695,
                    env_step=13,
                )
            ],
            env_step=13,
        )
        anchor = released["instances"][0][
            "verified_operation_target_anchor"
        ]
        self.assertEqual(anchor["world_m"], target_world)
        self.assertEqual(anchor["tolerance_m"], 0.020695)

        rebound = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "block",
                    "camera": "head",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection(
                            0,
                            [198, 133, 217, 154],
                            true_world,
                            score=0.61,
                        ),
                        detection(
                            1,
                            [66, 135, 89, 157],
                            [-0.244214, -0.106932, 0.775443],
                            score=0.28,
                        ),
                    ],
                }
            ],
            env_step=80,
            global_task="move an object through two placements",
            current_subtask="reacquire the placed object",
        )

        instance = next(
            item
            for item in rebound["instances"]
            if item["track_id"] == track_id
        )
        self.assertEqual(instance["latest_world_m"], true_world)
        outcome = rebound["temporal_memory"][
            "identity_binding_outcomes"
        ][0]
        self.assertEqual(
            outcome["matched_by"],
            "verified_operation_target",
        )
        self.assertEqual(
            outcome["verified_position_anchor_kind"],
            "verified_operation_target",
        )
        self.assertEqual(
            outcome["verified_position_tolerance_m"],
            0.020695,
        )
        self.assertAlmostEqual(
            outcome["matched_position_error_m"],
            0.0112,
            places=3,
        )
        self.assertEqual(
            rebound["temporal_memory"]["num_visible_candidates"],
            1,
        )

    def test_two_candidates_inside_verified_operation_target_remain_ambiguous(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        observed_world = [0.096177, -0.119476, 0.773579]
        target_world = [0.099212, -0.100609, 0.776744]
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "camera": "head",
                    "detections": [
                        detection(
                            0,
                            [190, 140, 215, 165],
                            observed_world,
                        )
                    ],
                }
            ],
            env_step=13,
            global_task="move a component",
            current_subtask="verify placement",
        )
        track_id = initial["instances"][0]["track_id"]
        tracker.apply_runtime_events(
            initial,
            events=[
                verified_release_event(
                    track_id,
                    observed_world=observed_world,
                    target_world=target_world,
                    tolerance_m=0.020695,
                    env_step=13,
                )
            ],
            env_step=13,
        )

        ambiguous = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "camera": "head",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection(
                            0,
                            [185, 130, 205, 150],
                            [0.099212, -0.085609, 0.776744],
                        ),
                        detection(
                            1,
                            [220, 160, 240, 180],
                            [0.099212, -0.115609, 0.776744],
                        ),
                    ],
                }
            ],
            env_step=14,
            global_task="move a component",
            current_subtask="reacquire",
        )

        outcome = ambiguous["temporal_memory"][
            "identity_binding_outcomes"
        ][0]
        self.assertEqual(outcome["status"], "ambiguous_candidates")
        self.assertEqual(ambiguous["instances"][0]["status"], "tracked")

    def test_motion_uncertain_invalidates_verified_operation_target_anchor(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        observed_world = [0.096177, -0.119476, 0.773579]
        target_world = [0.099212, -0.100609, 0.776744]
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "camera": "head",
                    "detections": [
                        detection(
                            0,
                            [190, 140, 215, 165],
                            observed_world,
                        )
                    ],
                }
            ],
            env_step=13,
            global_task="move a component",
            current_subtask="verify placement",
        )
        track_id = initial["instances"][0]["track_id"]
        released = tracker.apply_runtime_events(
            initial,
            events=[
                verified_release_event(
                    track_id,
                    observed_world=observed_world,
                    target_world=target_world,
                    tolerance_m=0.020695,
                    env_step=13,
                )
            ],
            env_step=13,
        )
        invalidated = tracker.apply_runtime_events(
            released,
            events=[
                {
                    "instance_ref": track_id,
                    "position_state": "motion_uncertain",
                    "source": "unverified_contact",
                    "reason": "contact may have moved the object",
                }
            ],
            env_step=14,
        )
        self.assertIsNone(
            invalidated["instances"][0][
                "verified_operation_target_anchor"
            ]
        )

        ambiguous = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "component",
                    "camera": "head",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection(
                            0,
                            [198, 133, 217, 154],
                            [0.091210, -0.094005, 0.772526],
                        ),
                        detection(
                            1,
                            [165, 170, 185, 190],
                            [0.092260, -0.143827, 0.769494],
                        ),
                    ],
                }
            ],
            env_step=15,
            global_task="move a component",
            current_subtask="reacquire after uncertain contact",
        )
        outcome = ambiguous["temporal_memory"][
            "identity_binding_outcomes"
        ][0]
        self.assertEqual(outcome["status"], "ambiguous_candidates")

    def test_e2_operation_target_rebuilds_changed_action_geometry_after_two_frames(
        self,
    ) -> None:
        tracker = SceneMemoryTracker(
            temporal_action_geometry_distance_m=0.04,
        )
        observed_world = [0.096177, -0.119476, 0.773579]
        target_world = [0.099212, -0.100609, 0.776744]
        old = detection_with_action_geometry(
            0,
            [197, 143, 216, 164],
            observed_world,
            top_offset=[0.0, 0.0, 0.080909],
            approach_offset=[0.000595, 0.131658, 0.231460],
            contact_offset=[0.000357, 0.078995, 0.171240],
            score=0.5,
        )
        old["grounding_3d"].update(
            {
                "bbox_world_min": [0.066177, -0.149476, 0.753579],
                "bbox_world_max": [0.126177, -0.089476, 0.793579],
            }
        )
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "block",
                    "camera": "head",
                    "detections": [old],
                }
            ],
            env_step=13,
            global_task="move an object through two placements",
            current_subtask="verify first placement",
        )
        track_id = initial["instances"][0]["track_id"]
        tracker.apply_runtime_events(
            initial,
            events=[
                verified_release_event(
                    track_id,
                    observed_world=observed_world,
                    target_world=target_world,
                    tolerance_m=0.020695,
                    env_step=13,
                )
            ],
            env_step=13,
        )

        new_world = [0.091210, -0.094005, 0.772526]

        def new_detection(mask_path: str) -> dict:
            item = detection_with_action_geometry(
                0,
                [198, 133, 217, 154],
                new_world,
                top_offset=[0.0, 0.0, 0.008204],
                approach_offset=[0.000001, -0.000125, 0.208204],
                contact_offset=[0.000001, -0.000075, 0.128204],
                score=0.61,
            )
            item["mask_path"] = mask_path
            item["grounding_3d"].update(
                {
                    "bbox_world_min": [
                        0.061210,
                        -0.124005,
                        0.752526,
                    ],
                    "bbox_world_max": [
                        0.121210,
                        -0.064005,
                        0.792526,
                    ],
                    "operation_pose_candidates": [
                        {
                            "candidate_id": (
                                "rgbd_volume:grasp:right:000"
                            ),
                            "source_candidate_index": 0,
                            "action_mode": "grasp",
                            "arm": "right",
                            "object_contact_pose": [
                                0.091210,
                                -0.093995,
                                0.763599,
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                            ],
                            "tcp_pose": [
                                0.091210,
                                -0.094005,
                                0.780730,
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                            ],
                            "ee_target_pose": [
                                0.091211,
                                -0.094080,
                                0.900730,
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                            ],
                            "approach_pose": [
                                0.091211,
                                -0.094130,
                                0.980730,
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                            ],
                            "approach_direction": [0.0, 0.0, -1.0],
                            "geometry_source": (
                                "rgbd_observed_volume_principal_axes"
                            ),
                        }
                    ],
                }
            )
            return item

        first = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "block",
                    "camera": "head",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [new_detection("/tmp/e2_new_80.png")],
                }
            ],
            env_step=80,
            global_task="move an object through two placements",
            current_subtask="reacquire for the next manipulation",
            observation_capture_id=80,
        )
        first_instance = first["instances"][0]
        self.assertEqual(
            first_instance["action_geometry_state"],
            "relocation_pending",
        )
        self.assertEqual(
            first_instance["action_geometry_confirmation_count"],
            1,
        )
        self.assertIsNone(first_instance["approach_world_m"])
        self.assertEqual(
            first_instance["operation_pose_candidate_count"],
            0,
        )

        second = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "block",
                    "camera": "head",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [new_detection("/tmp/e2_new_81.png")],
                }
            ],
            env_step=81,
            global_task="move an object through two placements",
            current_subtask="reacquire for the next manipulation",
            observation_capture_id=81,
        )
        second_instance = second["instances"][0]
        self.assertEqual(
            second_instance["action_geometry_state"],
            "verified",
        )
        self.assertEqual(second_instance["latest_world_m"], new_world)
        self.assertEqual(
            second_instance["approach_world_m"],
            [0.091211, -0.09413, 0.98073],
        )
        self.assertGreater(
            second_instance["operation_pose_candidate_count"],
            0,
        )
        candidate_id = (
            "rgbd_volume:grasp:right:000"
        )
        self.assertEqual(
            second_instance[
                "operation_pose_candidate_provenance"
            ][candidate_id],
            {
                "geometry_observation_env_step": 81,
                "geometry_observation_capture_id": 81,
            },
        )
        self.assertNotIn(
            "_geometry_observation_env_step",
            second_instance["operation_pose_candidates"][0],
        )

    def test_two_candidates_inside_verified_position_gate_remain_ambiguous(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "cube",
                    "camera": "head",
                    "detections": [
                        detection(
                            0,
                            [120, 100, 150, 130],
                            [0.10, -0.10, 0.77],
                        )
                    ],
                }
            ],
            env_step=0,
            global_task="move a cube",
            current_subtask="observe",
        )
        track_id = initial["instances"][0]["track_id"]
        ambiguous = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "cube",
                    "camera": "head",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection(
                            0,
                            [121, 101, 151, 131],
                            [0.105, -0.10, 0.77],
                        ),
                        detection(
                            1,
                            [126, 101, 156, 131],
                            [0.118, -0.10, 0.77],
                        ),
                    ],
                }
            ],
            env_step=1,
            global_task="move a cube",
            current_subtask="reacquire",
        )

        self.assertEqual(len(ambiguous["instances"]), 1)
        instance = ambiguous["instances"][0]
        self.assertEqual(instance["status"], "tracked")
        self.assertEqual(
            instance["position_state"],
            "memory_valid",
        )
        outcome = ambiguous["temporal_memory"][
            "identity_binding_outcomes"
        ][0]
        self.assertEqual(outcome["status"], "ambiguous_candidates")

    def test_nested_masks_at_verified_pose_are_one_observation_hypothesis(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        clean_world = [0.097636, -0.204033, 0.774917]
        clean_lower = [0.071438, -0.231817, 0.742167]
        clean_upper = [0.123835, -0.176249, 0.782156]
        top = [0.097636, -0.204033, 0.782156]
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "block",
                    "camera": "head",
                    "detections": [
                        detection_with_world_bounds(
                            0,
                            [130, 110, 170, 150],
                            clean_world,
                            clean_lower,
                            clean_upper,
                            top_surface=top,
                            score=0.91,
                        )
                    ],
                }
            ],
            env_step=0,
            global_task="move the selected block",
            current_subtask="observe the selected block",
        )
        track_id = initial["instances"][0]["track_id"]

        rebound = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "block",
                    "camera": "head",
                    "instance_ref": track_id,
                    "identity_binding_required": True,
                    "detections": [
                        detection_with_world_bounds(
                            0,
                            [130, 110, 170, 150],
                            clean_world,
                            clean_lower,
                            clean_upper,
                            top_surface=top,
                            score=0.89,
                        ),
                        detection_with_world_bounds(
                            1,
                            [112, 93, 188, 169],
                            [0.094849, -0.208686, 0.743505],
                            [0.054493, -0.247476, 0.740658],
                            [0.135206, -0.169896, 0.782156],
                            top_surface=[
                                0.094849,
                                -0.208686,
                                0.782156,
                            ],
                            score=0.82,
                        ),
                    ],
                }
            ],
            env_step=1,
            global_task="move the selected block",
            current_subtask="reacquire the selected block",
        )

        instance = rebound["instances"][0]
        self.assertEqual(instance["track_id"], track_id)
        self.assertEqual(instance["status"], "visible")
        self.assertEqual(instance["position_state"], "current_verified")
        self.assertEqual(instance["latest_world_m"], clean_world)
        outcome = rebound["temporal_memory"][
            "identity_binding_outcomes"
        ][0]
        self.assertEqual(outcome["status"], "matched")
        self.assertEqual(outcome["observation_hypothesis_count"], 1)
        self.assertEqual(
            outcome["redundant_observation_variant_count"],
            1,
        )
        warnings = [
            warning
            for item in rebound["temporal_memory"]["dropped_candidates"]
            for warning in (item.get("quality") or {}).get("warnings", [])
        ]
        self.assertTrue(
            any(
                warning.startswith(
                    "bound_identity_redundant_observation_variant"
                )
                for warning in warnings
            )
        )

    def test_verified_interaction_identity_blocks_semantic_relabel_at_same_pose(
        self,
    ) -> None:
        tracker = SceneMemoryTracker()
        initial = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "button",
                    "camera": "head",
                    "detections": [
                        detection(
                            0,
                            [80, 90, 115, 125],
                            [-0.199, -0.104, 0.773],
                        )
                    ],
                }
            ],
            env_step=0,
            global_task="interact and rearrange",
            current_subtask="press the control",
        )
        track_id = initial["instances"][0]["track_id"]
        locked = tracker.apply_runtime_events(
            initial,
            events=[
                {
                    "instance_ref": track_id,
                    "verified_role": "press",
                    "effect_type": "press",
                    "source": "verified_physical_effect",
                }
            ],
            env_step=1,
        )
        self.assertEqual(
            locked["instances"][0]["verified_roles"][0][
                "role"
            ],
            "press",
        )

        confused = tracker.update(
            segmentation=[
                {
                    "success": True,
                    "object_id": "red_cube",
                    "camera": "head",
                    "detections": [
                        detection(
                            0,
                            [81, 91, 116, 126],
                            [-0.198, -0.105, 0.773],
                        )
                    ],
                }
            ],
            env_step=2,
            global_task="interact and rearrange",
            current_subtask="find the red cube",
        )
        self.assertEqual(len(confused["instances"]), 1)
        instance = confused["instances"][0]
        self.assertEqual(instance["track_id"], track_id)
        self.assertEqual(instance["class"], "button")
        self.assertNotIn(
            "red_cube",
            instance["class_aliases"],
        )
        warnings = [
            warning
            for item in confused["temporal_memory"][
                "dropped_candidates"
            ]
            for warning in (
                (item.get("quality") or {}).get(
                    "warnings",
                    [],
                )
            )
        ]
        self.assertIn(
            f"verified_identity_conflict:{track_id}",
            warnings,
        )


if __name__ == "__main__":
    unittest.main()
