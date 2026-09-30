from __future__ import annotations

from copy import deepcopy
import json

import pytest

from policy.roboharn_evo.agent.trace_compaction import (
    compact_jsonl_event,
    compact_operation_candidate_selected_event,
    compact_scene_memory_update_event,
    compact_trace_event,
    normalize_trace_payload_mode,
)


def _candidate(candidate_id: str, *, arm: str = "right") -> dict:
    pose = [0.1, -0.2, 0.8, 1.0, 0.0, 0.0, 0.0]
    return {
        "candidate_id": candidate_id,
        "source_candidate_index": 0,
        "action_mode": "grasp",
        "arm": arm,
        "priority": 0.25,
        "geometry_source": "rgbd_observed_volume_principal_axes",
        "object_contact_pose": pose,
        "tcp_pose": pose,
        "ee_target_pose": pose,
        "approach_pose": pose,
        "executor_private": {"calibration": "do-not-log"},
    }


def _robot_state() -> dict:
    return {
        "step": 7,
        "step_limit": 150,
        "joint_dim": 14,
        "joint_norm": 1.4,
        "left": {
            "xyz": [-0.2, -0.3, 0.9],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "rpy": [0.0, 0.0, 0.0],
            "gripper": 1.0,
            "private_joint_vector": list(range(100)),
        },
        "right": {
            "xyz": [0.2, -0.3, 0.9],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "rpy": [0.0, 0.0, 0.0],
            "gripper": 0.0,
        },
    }


def _nested_key_count(value, key: str) -> int:
    if isinstance(value, dict):
        return int(key in value) + sum(
            _nested_key_count(item, key) for item in value.values()
        )
    if isinstance(value, list):
        return sum(_nested_key_count(item, key) for item in value)
    return 0


def _nested_keys(value) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for item in value.values():
            result.update(_nested_keys(item))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(_nested_keys(item))
        return result
    return set()


def test_compact_observation_is_non_mutating_and_keeps_one_robot_state() -> None:
    robot = _robot_state()
    candidate = _candidate("rgbd:grasp:right:000")
    grounding = {
        "success": True,
        "valid_ratio": 0.98,
        "centroid_world": [0.1, -0.2, 0.77],
        "bbox_world_min": [0.08, -0.22, 0.74],
        "bbox_world_max": [0.12, -0.18, 0.79],
        "top_surface_world": [0.1, -0.2, 0.79],
        "principal_axes_world": [[1, 0, 0], [0, 1, 0]],
        "operation_pose_candidates": [candidate],
    }
    detection = {
        "rank": 0,
        "score": 0.91,
        "bbox_xyxy": [10, 20, 30, 40],
        "centroid_px": [20, 30],
        "area_px": 400,
        "mask_path": "/tmp/block.png",
        "robot_state": robot,
        "grounding_3d": grounding,
        "quality": {"actionable": True, "warnings": []},
    }
    payload = {
        "env_step": 7,
        "stage": "observation_preprocess",
        "latency_sec": 1.2,
        "observation_generation": 3,
        "observation_capture_id": 4,
        "perception_queries": [
            {
                "object_id": "block",
                "text_prompt": "red block",
                "role": "target",
                "reason": "needed for grasp",
            }
        ],
        "segmentation": [
            {
                "object_id": "block",
                "text_prompt": "red block",
                "query_role": "target",
                "camera": "head",
                "backend": "sam3",
                "success": True,
                "score": 0.91,
                "bbox_xyxy": [10, 20, 30, 40],
                "centroid_px": [20, 30],
                "mask_path": "/tmp/block.png",
                "robot_state": robot,
                "grounding_3d": deepcopy(grounding),
                "detections": [detection],
            }
        ],
    }
    original = deepcopy(payload)

    compact = compact_trace_event(
        "observation_preprocess", payload, mode="compact_v1"
    )

    assert payload == original
    assert compact["schema"] == "trace/observation_preprocess/compact_v1"
    assert compact["schema_version"] == 1
    assert compact["observation_generation"] == 3
    assert compact["observation_capture_id"] == 4
    assert compact["detection_count"] == 1
    assert compact["cameras"] == ["head"]
    assert _nested_key_count(compact, "robot_state") == 1
    assert compact["robot_state"]["right"]["gripper"] == 0.0
    assert "private_joint_vector" not in json.dumps(compact)

    result = compact["segmentation"][0]
    assert result["selected_detection_index"] == 0
    assert "grounding_3d" not in result
    compact_detection = result["detections"][0]
    assert compact_detection["grounding_3d"]["centroid_world"] == [
        0.1,
        -0.2,
        0.77,
    ]
    assert compact_detection["quality"] == {
        "actionable": True,
    }
    assert compact_detection["operation_candidate_count"] == 1
    assert "operation_candidate_index" not in compact_detection
    assert compact["operation_candidate_index"] == {
        "group_columns": [
            "result_index",
            "detection_index",
            "rows",
        ],
        "columns": [
            "candidate_id",
            "source_candidate_index",
            "arm",
            "action_mode",
            "priority",
            "geometry_source",
        ],
        "dictionaries": {
            "candidate_id": ["rgbd:grasp:right:000"],
            "arm": ["right"],
            "action_mode": ["grasp"],
            "geometry_source": ["rgbd_observed_volume_principal_axes"],
        },
        "groups": [
            [
                0,
                0,
                [[0, 0, 0, 0, 0.25, 0]],
            ]
        ],
    }
    forbidden = {
        "operation_pose_candidates",
        "object_contact_pose",
        "tcp_pose",
        "ee_target_pose",
        "approach_pose",
        "principal_axes_world",
        "executor_private",
    }
    assert forbidden.isdisjoint(_nested_keys(compact))


def test_compact_observation_keeps_top_level_failure_summary() -> None:
    compact = compact_trace_event(
        "observation_preprocess",
        {
            "perception_error": "agent returned no perception queries",
            "segmentation": [],
        },
        mode="compact_v1",
    )

    assert compact["failed_segmentation_count"] == 0
    assert compact["errors"] == [
        {
            "scope": "observation",
            "kind": "perception_error",
            "error": "agent returned no perception queries",
        }
    ]


def test_compact_scene_memory_keeps_public_identity_state_and_operations() -> None:
    candidate = _candidate("rgbd:grasp:right:000")
    scene = {
        "env_step": 7,
        "robot_state": _robot_state(),
        "instances": [
            {
                "instance_id": "track_0001",
                "track_id": "track_0001",
                "class": "block",
                "status": "visible",
                "position_state": "current_verified",
                "position_source": "multiview_world_observation",
                "world_m": [0.1, -0.2, 0.77],
                "first_observed_world_m": [0.11, -0.21, 0.77],
                "last_verified_score": 0.91,
                "score": 0.88,
                "bbox_xyxy": [10, 20, 30, 40],
                "camera": "head",
                "supporting_cameras": ["third", "head", "head"],
                "last_verified_step": 7,
                "action_geometry_state": "verified",
                "actionable": True,
                "operation_pose_candidates": [candidate],
                "operation_pose_candidate_count": 1,
                "history": [{"raw": "x" * 20_000}],
                "robot_state": _robot_state(),
            }
        ],
        "task_focus": {
            "current_subtask": "pick block",
            "target_instances": ["track_0001"],
            "source": "agent_api",
        },
        "operation_targets": [
            {
                "target_id": "place:center",
                "action_mode": "place",
                "held_instance_id": "track_0001",
                "target_kind": "reference_region",
                "object_target_world_m": [0.0, 0.0, 0.77],
                "reference_region_id": "region:mats",
                "support_valid": True,
                "free": True,
                "candidate_id": "place:right:0",
                "ee_target_pose": [0.0] * 7,
            }
        ],
        "reference_regions": [
            {
                "reference_region_id": "region:mats",
                "reference_object_id": "mat",
                "placement_relation": "center_of",
                "center_world_xy": [0.0, 0.0],
                "position_state": "current_verified",
                "support_valid": True,
                "evidence_status": "resolved",
                "expected_count": 4,
                "observed_member_count": 4,
                "reference_instance_ids": ["m1", "m2", "m3", "m4"],
                "observations": [{"raw_mask": "x" * 20_000}],
            }
        ],
        "manipulation_state": {
            "right": {
                "phase": "holding_confirmed",
                "held_instance_id": "track_0001",
                "holding_confirmed": True,
                "transport_authorized": True,
                "grasp_candidate_id": "private-grasp-id",
                "held_object_to_tcp_attachment": {"pose": [0.0] * 7},
            }
        },
        "uncertainty": ["one reference is briefly occluded"],
        "temporal_memory": {"observations": ["x" * 100_000]},
    }
    payload = {
        "env_step": 7,
        "observation_generation": 3,
        "observation_capture_id": 4,
        "scene_memory": scene,
    }
    original = deepcopy(payload)

    compact = compact_scene_memory_update_event(payload)

    assert payload == original
    assert compact["schema"] == "trace/scene_memory_update/compact_v1"
    assert compact["schema_version"] == 1
    assert compact["observation_generation"] == 3
    assert compact["observation_capture_id"] == 4
    assert _nested_key_count(compact, "robot_state") == 1
    compact_scene = compact["scene_memory"]
    instance = compact_scene["instances"][0]
    assert instance == {
        "instance_id": "track_0001",
        "track_id": "track_0001",
        "class": "block",
        "camera": "head",
        "status": "visible",
        "position_state": "current_verified",
        "position_source": "multiview_world_observation",
        "last_verified_step": 7,
        "action_geometry_state": "verified",
        "actionable": True,
        "world_m": [0.1, -0.2, 0.77],
        "original_world_m": [0.11, -0.21, 0.77],
        "confidence": 0.91,
        "score": 0.88,
        "bbox_xyxy": [10, 20, 30, 40],
        "supporting_cameras": ["head", "third"],
        "operations": {
            "grasp": {
                "available": True,
                "arms": ["right"],
                "candidate_ids": ["rgbd:grasp:right:000"],
                "candidate_count": 1,
            }
        },
        "operation_candidate_count": 1,
    }
    assert compact_scene["operation_targets"][0]["target_id"] == "place:center"
    assert compact_scene["reference_regions"][0]["expected_count"] == 4
    assert compact_scene["uncertainty"] == [
        "one reference is briefly occluded"
    ]
    assert compact_scene["task_focus"]["source"] == "agent_api"
    assert compact_scene["temporal_memory"] == {"present": True}
    assert compact_scene["manipulation_state"]["right"] == {
        "phase": "holding_confirmed",
        "held_instance_id": "track_0001",
        "holding_confirmed": True,
        "transport_authorized": True,
    }
    forbidden = {
        "operation_pose_candidates",
        "object_contact_pose",
        "tcp_pose",
        "ee_target_pose",
        "approach_pose",
        "history",
        "observations",
        "held_object_to_tcp_attachment",
        "grasp_candidate_id",
    }
    assert forbidden.isdisjoint(_nested_keys(compact))
    assert len(json.dumps(compact).encode("utf-8")) < 12 * 1024


def test_selected_candidate_event_keeps_one_complete_pose_bundle() -> None:
    candidate = _candidate("rgbd:grasp:right:000")
    candidate.update(
        {
            "instance_id": "track_0001",
            "target_id": "track_0001",
            "source_detection_index": 2,
            "source_camera": "third",
            "candidate_geometry_revision": 5,
            "supporting_cameras": ["head", "third"],
        }
    )
    original = deepcopy(candidate)

    compact = compact_operation_candidate_selected_event(
        {
            "env_step": 7,
            "observation_generation": 3,
            "observation_capture_id": 4,
            "dispatch_index": 1,
            "dispatch_env_step": 7,
            "result_env_step": 8,
            "candidate": candidate,
        }
    )

    assert candidate == original
    assert compact["candidate_id"] == "rgbd:grasp:right:000"
    assert compact["instance_id"] == "track_0001"
    assert compact["target_id"] == "track_0001"
    assert compact["source_camera"] == "third"
    assert compact["candidate_geometry_revision"] == 5
    assert compact["observation_generation"] == 3
    assert compact["observation_capture_id"] == 4
    assert compact["dispatch_index"] == 1
    assert compact["dispatch_env_step"] == 7
    assert compact["result_env_step"] == 8
    assert compact["supporting_cameras"] == ["head", "third"]
    for pose_key in (
        "object_contact_pose",
        "tcp_pose",
        "ee_target_pose",
        "approach_pose",
    ):
        assert compact[pose_key] == candidate[pose_key]
        assert _nested_key_count(compact, pose_key) == 1
    assert "executor_private" not in compact


def test_compact_recovery_result_removes_only_repeated_candidate_geometry() -> None:
    pose = [0.1, -0.2, 0.8, 1.0, 0.0, 0.0, 0.0]
    payload = {
        "workflow": "grounded",
        "results": [
            {
                "tool_name": "move_ee_to_grounded_instance",
                "success": True,
                "message": "ok",
                "details": {
                    "operation_candidate_id": "rgbd:grasp:right:000",
                    "operation_tcp_pose": pose,
                    "operation_object_contact_pose": pose,
                    "selected_operation_candidate": {
                        "candidate_id": "rgbd:grasp:right:000",
                        "object_contact_pose": pose,
                        "tcp_pose": pose,
                        "ee_target_pose": pose,
                        "approach_pose": pose,
                    },
                    "target_pose": pose,
                    "requested_target": pose,
                    "observed_pose": pose,
                    "executed_pose": pose,
                    "target_error_m": 0.001,
                    "target_reached": True,
                    "collision": False,
                    "stalled": False,
                    "skipped": False,
                    "batch_halted": False,
                },
            }
        ],
    }
    original = deepcopy(payload)

    compact = compact_trace_event("recovery_result", payload)

    assert payload == original
    details = compact["results"][0]["details"]
    assert details["operation_candidate_id"] == "rgbd:grasp:right:000"
    assert "operation_tcp_pose" not in details
    assert "operation_object_contact_pose" not in details
    assert details["selected_operation_candidate"] == {
        "candidate_id": "rgbd:grasp:right:000"
    }
    for key in (
        "target_pose",
        "requested_target",
        "observed_pose",
        "executed_pose",
        "target_error_m",
        "target_reached",
        "collision",
        "stalled",
        "skipped",
        "batch_halted",
    ):
        assert details[key] == payload["results"][0]["details"][key]


@pytest.mark.parametrize("mode", ["legacy", "raw"])
def test_legacy_and_raw_modes_return_deep_copies(mode: str) -> None:
    payload = {"segmentation": [{"private": [1, 2, 3]}]}

    result = compact_trace_event("observation_preprocess", payload, mode=mode)

    assert result == payload
    assert result is not payload
    assert result["segmentation"] is not payload["segmentation"]


def test_unknown_compact_event_is_preserved_as_a_deep_copy() -> None:
    payload = {"results": [{"value": 1}]}
    compact = compact_trace_event("unmigrated_event", payload)

    assert compact == payload
    assert compact is not payload
    assert compact["results"] is not payload["results"]


def test_compact_jsonl_event_is_an_offline_envelope_convenience() -> None:
    event = {
        "event": "observation_preprocess",
        "timestamp": 1.5,
        "segmentation": [],
    }
    compact = compact_jsonl_event(event)

    assert compact["event"] == "observation_preprocess"
    assert compact["timestamp"] == 1.5
    assert compact["schema"] == "trace/observation_preprocess/compact_v1"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("legacy", "legacy"),
        ("raw", "raw"),
        ("compact-v1", "compact_v1"),
    ],
)
def test_normalize_trace_payload_mode(value: str, expected: str) -> None:
    assert normalize_trace_payload_mode(value) == expected


def test_normalize_trace_payload_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="trace_payload_mode"):
        normalize_trace_payload_mode("compact_v2")
