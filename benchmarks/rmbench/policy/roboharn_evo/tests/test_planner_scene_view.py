from __future__ import annotations

from copy import deepcopy
import json

import pytest

from policy.roboharn_evo.agent.planner_scene_view import (
    build_planner_scene_view,
    compact_finalized_observation_event,
    normalize_planner_context_mode,
)


def _candidate(mode: str, arm: str, index: int) -> dict:
    pose = [0.1 + index * 0.001, -0.1, 0.8, 1.0, 0.0, 0.0, 0.0]
    return {
        "candidate_id": f"private:{mode}:{arm}:{index}",
        "action_mode": mode,
        "arm": arm,
        "tcp_pose": pose,
        "ee_target_pose": pose,
        "approach_pose": pose,
        "object_contact_pose": pose,
        "calibration_source": "test",
    }


def _scene() -> dict:
    instances = []
    for index in range(10):
        instance_id = f"track_{index:04d}"
        instances.append(
            {
                "instance_id": instance_id,
                "track_id": instance_id,
                "class": "block" if index == 0 else "mat",
                "query_role": "target" if index == 0 else "context",
                "status": "visible",
                "stability": "stable",
                "position_state": "current_verified",
                "position_source": "multiview_world_observation",
                "world_m": [0.01 * index, -0.1, 0.77],
                "first_observed_world_m": [0.01 * index, -0.11, 0.77],
                "last_verified_score": 0.9 - index * 0.01,
                "last_verified_step": 12,
                "action_geometry_state": "verified",
                "actionable": True,
                "approach_world_m": [0.01 * index, -0.1, 0.9],
                "approach_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "grasp_world_m": [0.01 * index, -0.1, 0.8],
                "grasp_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "bbox_xyxy": [1, 2, 3, 4],
                "centroid_px": [2, 3],
                "operation_pose_candidates": [
                    _candidate("grasp", "left", index),
                    _candidate("grasp", "right", index),
                    _candidate("contact", "right", index),
                ],
                "verified_roles": [
                    {
                        "role": "placed_object",
                        "effect_type": "release",
                        "env_step": 10,
                        "private_note": "do not expose",
                    }
                ],
            }
        )
    return {
        "env_step": 12,
        "task_focus": {
            "current_subtask": "return the block",
            "target_instances": ["track_0000"],
            "tool_instances": [],
            "context_instances": ["track_0008"],
            "identity_binding_required": True,
            "identity_binding_errors": [],
        },
        "instances": instances,
        "operation_targets": [
            {
                "target_id": "place:track_0000:center",
                "candidate_id": "private-place-candidate",
                "action_mode": "place",
                "held_instance_id": "track_0000",
                "target_kind": "reference_region",
                "object_target_world_m": [0.1, -0.2, 0.78],
                "placement_relation": "center_of",
                "reference_region_id": "region:mats",
                "reference_instance_ids": [
                    "track_0001",
                    "track_0002",
                    "track_0003",
                    "track_0004",
                ],
                "support_valid": True,
                "free": True,
                "arm_options": ["left", "right"],
            }
        ],
        "reference_regions": [
            {
                "reference_region_id": "region:mats",
                "reference_object_id": "mat",
                "placement_relation": "center_of",
                "center_world_xy": [0.1, -0.2],
                "position_state": "current_verified",
                "support_valid": True,
                "evidence_status": "resolved",
                "expected_count": 4,
                "observed_member_count": 4,
                "reference_instance_ids": [
                    "track_0001",
                    "track_0002",
                    "track_0003",
                    "track_0004",
                ],
                "observations": [{"mask_path": "/private/mask.png"}],
            }
        ],
        "spatial_state": {
            "signature": "abc123",
            "order_by_x": ["track_0000", "track_0001"],
            "order_by_y": ["track_0001", "track_0000"],
            "horizontal_overlap_pairs": [["track_0000", "track_0008"]],
            "support_pairs": [["track_0008", "track_0000"]],
            "instances": [
                {"instance_id": "track_0000", "world_m": [0.0, -0.1, 0.77]}
            ],
        },
        "temporal_memory": {"private_padding": "x" * 100_000},
        "uncertainty": [
            "track_0009 has low confidence",
            "track_0009 has low confidence",
            "one reference member is intermittently occluded",
        ],
    }


def _manipulation_state() -> dict:
    return {
        "right": {
            "phase": "holding_provisional",
            "held_instance_id": "track_0000",
            "holding_confirmed": False,
            "transport_authorized": True,
            "attachment_evidence_status": "unknown",
            "grasp_transport_policy": "evidence_only",
            "grasp_candidate_id": "private-grasp-candidate",
            "grasp_ee_target_world_m": [0.0, -0.1, 0.9],
            "held_object_to_tcp_attachment": {
                "capture_tcp_pose": [0.0, -0.1, 0.8, 1.0, 0.0, 0.0, 0.0]
            },
            "updated_step": 12,
        }
    }


def _nested_keys(value) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_nested_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_nested_keys(item))
        return keys
    return set()


def test_planner_scene_view_keeps_public_capabilities_without_private_geometry() -> None:
    scene = _scene()
    original = deepcopy(scene)
    preprocess = {
        "env_step": 12,
        "observation_generation": 7,
        "observation_capture_id": 9,
        "segmentation": [
            {"object_id": "block", "camera": "head", "success": True},
            {"object_id": "block", "camera": "third", "success": True},
        ],
    }

    view = build_planner_scene_view(
        scene,
        manipulation_state=_manipulation_state(),
        observation_preprocess=preprocess,
        max_instances=8,
    )

    assert scene == original
    assert view["schema"] == "planner_scene_view/compact_v1"
    assert len(view["instances"]) == 8
    assert view["instances"][0]["instance_id"] == "track_0000"
    assert view["instances"][1]["instance_id"] == "track_0008"
    assert view["instances"][0]["operations"] == {
        "grasp": {
            "available": True,
            "arms": ["left", "right"],
            "point_keys": ["approach_world_m", "grasp_world_m"],
        },
        "contact": {
            "available": True,
            "arms": ["right"],
            "point_keys": ["approach_world_m", "contact_world_m"],
        },
    }
    assert view["operation_targets"][0]["target_id"] == "place:track_0000:center"
    assert view["reference_regions"][0]["observed_member_count"] == 4
    assert view["spatial_state"]["order_by_x"] == ["track_0000", "track_0001"]
    assert view["manipulation_state"]["right"]["transport_authorized"] is True
    assert view["perception_status"]["cameras"] == ["head", "third"]
    assert view["uncertainty"] == [
        "track_0009 has low confidence",
        "one reference member is intermittently occluded",
    ]

    forbidden = {
        "candidate_id",
        "operation_pose_candidates",
        "ee_target_pose",
        "approach_pose",
        "tcp_pose",
        "approach_quat_wxyz",
        "grasp_quat_wxyz",
        "bbox_xyxy",
        "centroid_px",
        "mask_path",
        "held_object_to_tcp_attachment",
        "capture_tcp_pose",
        "grasp_ee_target_world_m",
        "temporal_memory",
    }
    assert forbidden.isdisjoint(_nested_keys(view))
    serialized = json.dumps(view, ensure_ascii=False)
    assert "private:" not in serialized
    assert "private-place-candidate" not in serialized
    assert len(serialized.encode("utf-8")) < 12_000


def test_compact_finalized_event_references_one_scene_view() -> None:
    scene = _scene()
    payload = {
        "stage": "observation_preprocess",
        "latency_sec": 1.25,
        "observation_generation": 3,
        "observation_capture_id": 4,
        "perception_queries": [{"object_id": "block"}],
        "segmentation": [
            {
                "object_id": "block",
                "camera": "head",
                "success": True,
                "detections": [{"mask_path": "/private/mask.png"}],
            }
        ],
        "scene_memory": scene,
    }
    view = build_planner_scene_view(scene, max_instances=8)

    compact = compact_finalized_observation_event(
        payload,
        planner_scene_view=view,
    )

    assert compact["schema"] == (
        "trace/observation_preprocess_finalized/compact_v1"
    )
    assert compact["schema_version"] == 1
    assert compact["planner_scene_view"] == view
    assert compact["full_runtime_state_retained_in_working_memory"] is True
    assert compact["detection_count"] == 1
    assert "segmentation" not in compact
    assert "scene_memory" not in compact
    assert "temporal_memory" not in json.dumps(compact)


def test_planner_scene_view_prioritizes_held_object_outside_current_focus() -> None:
    scene = _scene()
    view = build_planner_scene_view(
        scene,
        manipulation_state={
            "left": {
                "phase": "holding_provisional",
                "held_instance_id": "track_0009",
                "transport_authorized": True,
            }
        },
        max_instances=3,
    )

    visible_ids = [item["instance_id"] for item in view["instances"]]
    assert visible_ids == ["track_0000", "track_0009", "track_0008"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("legacy", "legacy"), ("compact-v1", "compact_v1")],
)
def test_normalize_planner_context_mode(value: str, expected: str) -> None:
    assert normalize_planner_context_mode(value) == expected


def test_normalize_planner_context_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="planner_context_mode"):
        normalize_planner_context_mode("full_v2")
