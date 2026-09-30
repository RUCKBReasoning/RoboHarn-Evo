from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from policy.roboharn_evo.agent.core.img_agent import ImgAgent
from policy.roboharn_evo.agent.operation_candidate_lifecycle import (
    candidate_geometry,
    candidate_geometry_revision,
)
from policy.roboharn_evo.agent.recovery.tool_specs import (
    RecoveryToolCall,
    RecoveryToolResult,
)
from policy.roboharn_evo.agent.trace_compaction import compact_trace_event
from policy.roboharn_evo.scripts.audit_iclr_trace_evidence import (
    analyze_observation_preprocess,
    analyze_scene_memory,
)
from policy.roboharn_evo.scripts.visualize_rollout_contact_sheet import summarize_event
from policy.roboharn_evo.scripts.visualize_rollout_report import event_summary


def _bare_agent(
    *,
    mode: str = "compact_v1",
    raw_sidecar: bool = False,
) -> ImgAgent:
    agent = ImgAgent.__new__(ImgAgent)
    agent._agent_card = SimpleNamespace(
        config=SimpleNamespace(
            observation_trace_payload_mode=mode,
            raw_trace_sidecar_enabled=raw_sidecar,
        )
    )
    agent._current_episode_id = 3
    agent._current_seed = 100002
    agent._raw_perception_sidecar_file = None
    agent._raw_perception_sidecar_capture_keys = set()
    return agent


def _candidate() -> dict:
    return {
        "candidate_id": "rgbd:grasp:right:001",
        "source_candidate_index": 1,
        "arm": "right",
        "action_mode": "grasp",
        "priority": 1,
        "geometry_source": "rgbd_observed_volume",
        "object_contact_pose": [0.1, -0.2, 0.78, 1, 0, 0, 0],
        "tcp_pose": [0.1, -0.2, 0.79, 1, 0, 0, 0],
        "ee_target_pose": [0.1, -0.2, 0.80, 1, 0, 0, 0],
        "approach_pose": [0.1, -0.2, 0.88, 1, 0, 0, 0],
    }


def test_both_writers_receive_the_same_compact_copy_without_runtime_mutation() -> None:
    agent = _bare_agent()
    trace_records: list[tuple[str, dict]] = []
    rollout_records: list[tuple[str, dict]] = []
    agent._trace = lambda event, **payload: trace_records.append(
        (event, payload)
    )
    agent._dump_rollout_event = (
        lambda event, **payload: rollout_records.append((event, payload))
    )
    raw = {
        "env_step": 2,
        "segmentation": [
            {
                "object_id": "block",
                "camera": "head",
                "success": True,
                "robot_state": {
                    "right": {"xyz": [0, 0, 1], "gripper": 1}
                },
                "detections": [
                    {
                        "score": 0.9,
                        "grounding_3d": {
                            "success": True,
                            "centroid_world": [0.1, 0.2, 0.8],
                            "operation_pose_candidates": [_candidate()],
                        },
                    }
                ],
            }
        ],
    }
    before = deepcopy(raw)

    agent._record_trace_and_rollout_event("observation_preprocess", raw)

    assert raw == before
    assert trace_records == rollout_records
    compact = trace_records[0][1]
    assert compact["schema"] == "trace/observation_preprocess/compact_v1"
    assert "operation_pose_candidates" not in json.dumps(compact)
    assert "ee_target_pose" not in json.dumps(compact)


def test_compactor_failure_is_logged_without_raising() -> None:
    agent = _bare_agent()
    trace_records: list[tuple[str, dict]] = []
    agent._trace = lambda event, **payload: trace_records.append(
        (event, payload)
    )
    agent._dump_rollout_event = lambda event, **payload: None

    with mock.patch(
        "policy.roboharn_evo.agent.core.img_agent.compact_trace_event",
        side_effect=RuntimeError("synthetic compactor failure"),
    ):
        agent._record_trace_and_rollout_event(
            "observation_preprocess",
            {"runtime_payload": "must-not-escape"},
        )

    assert trace_records == [
        (
            "observation_preprocess",
            {
                "schema": "trace/compaction_error/v1",
                "schema_version": 1,
                "compaction_error": "RuntimeError",
                "payload_omitted": True,
            },
        )
    ]


def test_finalized_projection_failure_is_bounded() -> None:
    agent = _bare_agent()
    agent._observation_preprocess_finalized_trace_payload = mock.Mock(
        side_effect=RuntimeError("synthetic finalized projection failure")
    )

    result = agent._safe_observation_preprocess_finalized_trace_payload(
        {
            "observation_generation": 4,
            "observation_capture_id": 9,
            "private_runtime_payload": "must-not-escape",
        }
    )

    assert result == {
        "schema": "trace/observation_preprocess_finalized_error/v1",
        "schema_version": 1,
        "projection_error": "RuntimeError",
        "payload_omitted": True,
        "observation_generation": 4,
        "observation_capture_id": 9,
    }


def test_raw_sidecar_is_default_off_and_deduplicated_by_capture(
    tmp_path: Path,
) -> None:
    payload = {
        "env_step": 2,
        "observation_generation": 4,
        "observation_capture_id": 9,
        "segmentation": [{"raw_candidates": [_candidate()]}],
    }
    off = _bare_agent(raw_sidecar=False)
    off._raw_perception_sidecar_file = tmp_path / "off.jsonl"
    off._write_raw_perception_sidecar_once(payload)
    assert not off._raw_perception_sidecar_file.exists()

    enabled = _bare_agent(raw_sidecar=True)
    enabled._raw_perception_sidecar_file = tmp_path / "raw_perception.jsonl"
    enabled._write_raw_perception_sidecar_once(payload)
    enabled._write_raw_perception_sidecar_once(
        {**payload, "observation_generation": 5}
    )

    lines = enabled._raw_perception_sidecar_file.read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["schema"] == "roboharn_evo/raw_perception_v1"
    assert record["observation_capture_id"] == 9
    assert record["payload"] == payload


def test_existing_raw_sidecar_restores_capture_deduplication(
    tmp_path: Path,
) -> None:
    rollout_dir = tmp_path / "episode_0000_rollout"
    rollout_dir.mkdir()
    sidecar = rollout_dir / "raw_perception.jsonl"
    sidecar.write_text(
        json.dumps(
            {
                "event": "observation_preprocess_raw",
                "episode_id": 3,
                "seed": 100002,
                "observation_capture_id": 9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    agent = _bare_agent(raw_sidecar=True)
    agent._trace_file = None
    agent._rollout_dir = rollout_dir

    agent._configure_raw_perception_sidecar_file()
    agent._write_raw_perception_sidecar_once(
        {
            "env_step": 2,
            "observation_generation": 5,
            "observation_capture_id": 9,
            "segmentation": [{"new": "must-not-append"}],
        }
    )

    assert len(sidecar.read_text(encoding="utf-8").splitlines()) == 1


def test_raw_sidecar_write_failure_is_best_effort(tmp_path: Path) -> None:
    agent = _bare_agent(raw_sidecar=True)
    agent._raw_perception_sidecar_file = tmp_path  # opening a directory fails
    payload = {
        "env_step": 2,
        "observation_generation": 4,
        "observation_capture_id": 9,
        "segmentation": [],
    }

    agent._write_raw_perception_sidecar_once(payload)

    assert agent._raw_perception_sidecar_capture_keys == set()


def test_raw_sidecar_separates_a_truncated_tail(tmp_path: Path) -> None:
    agent = _bare_agent(raw_sidecar=True)
    sidecar = tmp_path / "raw_perception.jsonl"
    sidecar.write_text('{"truncated":', encoding="utf-8")
    agent._raw_perception_sidecar_file = sidecar

    agent._write_raw_perception_sidecar_once(
        {
            "env_step": 2,
            "observation_generation": 4,
            "observation_capture_id": 9,
            "segmentation": [],
        }
    )

    lines = sidecar.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["observation_capture_id"] == 9


def test_selected_candidate_is_recorded_once_per_geometry_in_a_batch() -> None:
    agent = _bare_agent()
    selected_events: list[tuple[str, dict]] = []
    agent._record_trace_and_rollout_event = (
        lambda event, payload: selected_events.append((event, payload))
    )
    candidate = _candidate()
    materialized = {
        "instance_id": "track_0001",
        "track_id": "track_0001",
        "source_object_id": "block",
        "source_rank": 0,
        "camera": "third",
        "supporting_cameras": ["head", "third"],
        "selected_operation_candidate": candidate,
    }
    agent._grounded_instance_for_call = lambda call: materialized
    agent._recovery_history_instance_id = (
        lambda call, details: "track_0001"
    )
    agent._single_arm_from_call = lambda call: "right"
    revision = candidate_geometry_revision(candidate_geometry(candidate))
    private_args = {
        "arm": "right",
        "instance_id": "track_0001",
        "target_id": "track_0001",
        "_operation_action_mode": "grasp",
        "_operation_candidate_id": candidate["candidate_id"],
        "_operation_candidate_attempt": {
            "candidate_id": candidate["candidate_id"],
            "candidate_geometry_revision": revision,
            "dispatch_env_step": 11,
            "observation_generation": 4,
            "observation_capture_id": 9,
        },
    }
    calls = [
        RecoveryToolCall(
            tool_name="move_ee_to_grounded_instance",
            args={**private_args, "point_key": "approach_world_m"},
        ),
        RecoveryToolCall(
            tool_name="move_ee_to_grounded_instance",
            args={**private_args, "point_key": "grasp_world_m"},
        ),
        RecoveryToolCall(
            tool_name="move_ee_to_grounded_instance",
            args={**private_args, "point_key": "grasp_world_m"},
        ),
    ]
    results = [
        RecoveryToolResult(
            tool_name=call.tool_name,
            success=True,
            message="ok",
            details={
                "instance_id": "track_0001",
                "operation_candidate_id": candidate["candidate_id"],
                "operation_action_mode": "grasp",
                "target_reached": True,
            },
        )
        for call in calls[:2]
    ]
    results.append(
        RecoveryToolResult(
            tool_name=calls[2].tool_name,
            success=False,
            message="skipped",
            details={"skipped": True, "batch_halted": True},
        )
    )

    agent._record_selected_operation_candidates(calls=calls, results=results)

    assert len(selected_events) == 1
    event, payload = selected_events[0]
    assert event == "operation_candidate_selected"
    assert payload["candidate_geometry_revision"] == revision
    assert payload["source_camera"] == "third"
    for key in (
        "object_contact_pose",
        "tcp_pose",
        "ee_target_pose",
        "approach_pose",
    ):
        assert payload[key] == candidate[key]


def test_candidate_identity_mismatch_emits_error_without_pose_bundle() -> None:
    agent = _bare_agent()
    events: list[tuple[str, dict]] = []
    agent._record_trace_and_rollout_event = (
        lambda event, payload: events.append((event, payload))
    )
    candidate = _candidate()
    agent._grounded_instance_for_call = lambda call: {
        "track_id": "track_0001",
        "selected_operation_candidate": candidate,
    }
    agent._recovery_history_instance_id = (
        lambda call, details: "track_0001"
    )
    agent._single_arm_from_call = lambda call: "right"
    call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "_operation_candidate_id": "different:candidate",
            "_operation_candidate_attempt": {
                "candidate_id": "different:attempted-candidate",
                "candidate_geometry_revision": candidate_geometry_revision(
                    candidate_geometry(candidate)
                )
            },
        },
    )
    result = RecoveryToolResult(
        tool_name=call.tool_name,
        success=True,
        message="ok",
        details={"operation_candidate_id": candidate["candidate_id"]},
    )

    agent._record_selected_operation_candidates(
        calls=[call], results=[result]
    )

    assert [event for event, _ in events] == [
        "operation_candidate_selection_trace_error"
    ]
    payload_text = json.dumps(events[0][1])
    assert "requested_candidate_id" in payload_text
    assert "attempted_candidate_id" in payload_text
    assert "object_contact_pose" not in payload_text


def test_malformed_candidate_does_not_suppress_later_selection() -> None:
    agent = _bare_agent()
    events: list[tuple[str, dict]] = []
    agent._record_trace_and_rollout_event = (
        lambda event, payload: events.append((event, payload))
    )
    valid_payload = {
        "candidate_id": "candidate:valid",
        "candidate_geometry_revision": "revision:valid",
        **{
            key: value
            for key, value in _candidate().items()
            if key.endswith("_pose")
        },
    }
    agent._selected_operation_candidate_trace_payload = mock.Mock(
        side_effect=[
            RuntimeError("malformed first candidate"),
            (
                (
                    "track_0001",
                    "candidate:valid",
                    "revision:valid",
                    "right",
                    "grasp",
                    "",
                ),
                valid_payload,
                None,
            ),
        ]
    )
    calls = [
        RecoveryToolCall(
            tool_name="move_ee_to_grounded_instance", args={}
        ),
        RecoveryToolCall(
            tool_name="move_ee_to_grounded_instance", args={}
        ),
    ]
    results = [
        RecoveryToolResult(
            tool_name=call.tool_name,
            success=True,
            message="ok",
            details={},
        )
        for call in calls
    ]

    agent._record_selected_operation_candidates(calls=calls, results=results)

    assert [event for event, _ in events] == [
        "operation_candidate_selection_trace_error",
        "operation_candidate_selected",
    ]
    assert events[1][1]["candidate_id"] == "candidate:valid"


@pytest.mark.parametrize(
    "invalid_pose",
    (
        [],
        [0.1, -0.2, 0.78, float("nan"), 0.0, 0.0, 1.0],
        [0.1, -0.2, 0.78, 0.0, 0.0, 0.0, 0.0],
    ),
    ids=("empty", "nonfinite", "zero_quaternion"),
)
def test_incomplete_candidate_geometry_emits_explicit_error(
    invalid_pose: list[float],
) -> None:
    agent = _bare_agent()
    events: list[tuple[str, dict]] = []
    agent._record_trace_and_rollout_event = (
        lambda event, payload: events.append((event, payload))
    )
    candidate = _candidate()
    candidate["object_contact_pose"] = invalid_pose
    revision = candidate_geometry_revision(candidate_geometry(candidate))
    agent._grounded_instance_for_call = lambda call: {
        "track_id": "track_0001",
        "selected_operation_candidate": candidate,
    }
    agent._recovery_history_instance_id = (
        lambda call, details: "track_0001"
    )
    agent._single_arm_from_call = lambda call: "right"
    call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "_operation_candidate_id": candidate["candidate_id"],
            "_operation_candidate_attempt": {
                "candidate_geometry_revision": revision
            },
        },
    )
    result = RecoveryToolResult(
        tool_name=call.tool_name,
        success=True,
        message="ok",
        details={"operation_candidate_id": candidate["candidate_id"]},
    )

    agent._record_selected_operation_candidates(
        calls=[call], results=[result]
    )

    assert [event for event, _ in events] == [
        "operation_candidate_selection_trace_error"
    ]
    assert events[0][1]["missing_pose_keys"] == ["object_contact_pose"]


def test_trace_result_adds_revision_without_mutating_runtime_result() -> None:
    agent = _bare_agent()
    call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "_operation_candidate_id": "candidate:1",
            "_operation_candidate_attempt": {
                "candidate_id": "candidate:1",
                "candidate_geometry_revision": "revision:2",
            },
        },
    )
    result = RecoveryToolResult(
        tool_name=call.tool_name,
        success=True,
        message="ok",
        details={
            "target_pose": [0.1, 0.2, 0.8, 1, 0, 0, 0],
            "operation_tcp_pose": [0.1, 0.2, 0.8, 1, 0, 0, 0],
        },
    )
    before = deepcopy(result.details)

    projected = agent._recovery_results_for_trace([call], [result])

    assert result.details == before
    assert projected[0]["details"]["operation_candidate_id"] == "candidate:1"
    assert (
        projected[0]["details"]["operation_candidate_geometry_revision"]
        == "revision:2"
    )


def test_compact_event_stream_has_one_complete_selected_pose_bundle() -> None:
    agent = _bare_agent()
    trace_records: list[tuple[str, dict]] = []
    agent._trace = lambda event, **payload: trace_records.append(
        (event, payload)
    )
    agent._dump_rollout_event = lambda event, **payload: None
    candidate = {
        **_candidate(),
        "candidate_geometry_revision": "revision:2",
        "instance_id": "track_0001",
    }
    agent._record_trace_and_rollout_event(
        "operation_candidate_selected",
        candidate,
    )
    agent._record_trace_and_rollout_event(
        "recovery_result",
        {
            "results": [
                {
                    "tool_name": "move_ee_to_grounded_instance",
                    "success": True,
                    "message": "ok",
                    "details": {
                        "operation_candidate_id": candidate["candidate_id"],
                        "operation_candidate_geometry_revision": "revision:2",
                        "operation_tcp_pose": candidate["tcp_pose"],
                        "operation_object_contact_pose": candidate[
                            "object_contact_pose"
                        ],
                        "target_pose": candidate["ee_target_pose"],
                        "observed_pose": candidate["ee_target_pose"],
                        "target_reached": True,
                    },
                }
            ]
        },
    )

    assert [event for event, _ in trace_records] == [
        "operation_candidate_selected",
        "recovery_result",
    ]
    selection = trace_records[0][1]
    recovery = trace_records[1][1]
    for key in (
        "object_contact_pose",
        "tcp_pose",
        "ee_target_pose",
        "approach_pose",
    ):
        assert selection[key] == candidate[key]
    recovery_text = json.dumps(recovery)
    assert "operation_tcp_pose" not in recovery_text
    assert "operation_object_contact_pose" not in recovery_text
    assert recovery["results"][0]["details"]["target_pose"] == candidate[
        "ee_target_pose"
    ]
    assert recovery["results"][0]["details"]["target_reached"] is True


def test_compact_events_remain_visible_to_audit_and_visualizer_consumers() -> None:
    observation = compact_trace_event(
        "observation_preprocess",
        {
            "event": "observation_preprocess",
            "latency_sec": 0.5,
            "segmentation": [
                {
                    "object_id": "block",
                    "camera": "third",
                    "success": True,
                    "detections": [
                        {
                            "score": 0.9,
                            "bbox_xyxy": [1, 2, 3, 4],
                            "grounding_3d": {
                                "success": True,
                                "centroid_world": [0.1, 0.2, 0.8],
                            },
                        }
                    ],
                }
            ],
        },
        mode="compact_v1",
    )
    perception = analyze_observation_preprocess([observation])
    assert perception["segmentation_success"] == 1
    assert perception["detections"] == 1
    assert perception["grounding_success_entries"] == 1
    assert perception["cameras"] == ["third"]

    scene_event = compact_trace_event(
        "scene_memory_update",
        {
            "event": "scene_memory_update",
            "scene_memory": {
                "instances": [
                    {
                        "instance_id": "track_0001",
                        "track_id": "track_0001",
                        "camera": "third",
                        "status": "tracked",
                        "stability": "stable",
                        "world_m": [0.1, 0.2, 0.8],
                    }
                ],
                "temporal_memory": {"tracks": [{"history": [1, 2]}]},
            },
        },
        mode="compact_v1",
    )
    scene = analyze_scene_memory([scene_event])
    assert scene["unique_track_ids"] == 1
    assert scene["temporal_memory_updates"] == 1
    assert scene["cameras"] == ["third"]

    selected = compact_trace_event(
        "operation_candidate_selected",
        {
            "candidate_id": "candidate:1",
            "candidate_geometry_revision": "revision:2",
            "instance_id": "track_0001",
            "arm": "right",
            "action_mode": "grasp",
            "source_camera": "third",
            **{key: value for key, value in _candidate().items() if key.endswith("_pose")},
        },
        mode="compact_v1",
    )
    selected["event"] = "operation_candidate_selected"
    lines = summarize_event(selected)
    assert any("candidate:1" in line for line in lines)
    assert any("right / grasp / third" in line for line in lines)
    html_summary = event_summary(selected)
    assert "candidate:1" in html_summary
    assert "revision:2" in html_summary
