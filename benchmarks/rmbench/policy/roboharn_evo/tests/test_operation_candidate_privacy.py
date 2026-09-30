from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import pytest

from policy.roboharn_evo.agent.core.img_agent import ImgAgent
from policy.roboharn_evo.agent.operation_candidate_lifecycle import (
    OperationCandidateLifecycle,
)
from policy.roboharn_evo.agent.planner_state_projection import (
    public_manipulation_state,
)
from policy.roboharn_evo.agent.recovery.tool_specs import RecoveryToolCall


_BASE_GEOMETRY = {
    "object_contact_pose": [0.10, -0.15, 0.80, 1.0, 0.0, 0.0, 0.0],
    "tcp_pose": [0.10, -0.15, 0.84, 1.0, 0.0, 0.0, 0.0],
    "ee_target_pose": [0.10, -0.15, 0.90, 1.0, 0.0, 0.0, 0.0],
    "approach_pose": [0.10, -0.15, 0.98, 1.0, 0.0, 0.0, 0.0],
    "approach_direction": [0.0, 0.0, -1.0],
}


def _candidate(
    action_mode: str,
    *,
    suffix: str,
    x_offset_m: float = 0.0,
    capture_id: int = 20,
    target_id: str = "",
) -> dict[str, Any]:
    geometry = deepcopy(_BASE_GEOMETRY)
    for key in (
        "object_contact_pose",
        "tcp_pose",
        "ee_target_pose",
        "approach_pose",
    ):
        geometry[key][0] += x_offset_m
    candidate: dict[str, Any] = {
        "candidate_id": f"runtime-private:{action_mode}:right:{suffix}",
        "arm": "right",
        "action_mode": action_mode,
        "geometry_source": "rgbd_observed_volume_principal_axes",
        "_geometry_observation_env_step": capture_id,
        "_geometry_observation_capture_id": capture_id,
        **geometry,
    }
    if action_mode == "place":
        candidate.update(
            {
                "target_id": target_id,
                "target_kind": "free_support",
                "geometry_source": "runtime_dynamic_place_geometry",
                "holding_status": "verified",
                "support_valid": True,
                "free": True,
                "valid": True,
                "reachable_estimate": True,
            }
        )
    return candidate


def _instance(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "instance_id": "track_public_0001",
        "status": "visible",
        "position_state": "current_verified",
        "action_geometry_state": "verified",
        "actionable": True,
        "operation_pose_candidates": candidates,
    }


def _block(
    lifecycle: OperationCandidateLifecycle,
    candidate: dict[str, Any],
) -> None:
    attempt = lifecycle.bind_attempt(
        instance_id="track_public_0001",
        candidate=candidate,
        dispatch_env_step=24,
    )
    assert attempt
    for error_m in (0.071, 0.070):
        lifecycle.record_failure(
            attempt,
            target_error_m=error_m,
            env_step=24,
        )


def _assert_no_private_candidate_ids(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert not str(key).lower().endswith("candidate_id")
            _assert_no_private_candidate_ids(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_private_candidate_ids(item)


@pytest.mark.parametrize("action_mode", ["grasp", "contact", "place"])
def test_lifecycle_planner_payload_is_public_and_reports_runtime_fallback(
    action_mode: str,
) -> None:
    lifecycle = OperationCandidateLifecycle()
    target_id = (
        "place:held:track_public_0001:free-support:01"
        if action_mode == "place"
        else ""
    )
    blocked = _candidate(
        action_mode,
        suffix="blocked",
        target_id=target_id,
    )
    available = _candidate(
        action_mode,
        suffix="available",
        x_offset_m=0.04,
        capture_id=21,
        target_id=target_id,
    )
    _block(lifecycle, blocked)

    payload = lifecycle.planner_payload(
        [_instance([blocked, available])]
    )

    assert len(payload) == 1
    public_scope = payload[0]
    _assert_no_private_candidate_ids(public_scope)
    assert "runtime-private:" not in json.dumps(public_scope, sort_keys=True)
    assert public_scope["instance_id"] == "track_public_0001"
    assert public_scope["arm"] == "right"
    assert public_scope["action_mode"] == action_mode
    assert public_scope["blocked_candidate_count"] == 1
    assert public_scope["available_candidate_count"] == 1
    assert public_scope["runtime_will_select_next"] is True
    if action_mode == "place":
        assert public_scope["target_id"] == target_id
    else:
        assert "target_id" not in public_scope


@pytest.mark.parametrize("action_mode", ["grasp", "contact", "place"])
def test_lifecycle_planner_payload_reports_when_no_runtime_fallback_exists(
    action_mode: str,
) -> None:
    lifecycle = OperationCandidateLifecycle()
    target_id = (
        "place:held:track_public_0001:free-support:01"
        if action_mode == "place"
        else ""
    )
    blocked = _candidate(
        action_mode,
        suffix="only",
        target_id=target_id,
    )
    _block(lifecycle, blocked)

    payload = lifecycle.planner_payload([_instance([blocked])])

    assert len(payload) == 1
    public_scope = payload[0]
    _assert_no_private_candidate_ids(public_scope)
    assert public_scope["blocked_candidate_count"] == 1
    assert public_scope["available_candidate_count"] == 0
    assert public_scope["runtime_will_select_next"] is False


def test_compact_result_details_omits_runtime_operation_candidate_id() -> None:
    agent = object.__new__(ImgAgent)
    private_id = "runtime-private:grasp:right:000"

    compact = agent._compact_result_details(
        {
            "arm": "right",
            "instance_id": "track_public_0001",
            "operation_action_mode": "grasp",
            "operation_candidate_id": private_id,
            "operation_target_id": "public-target-01",
            "target_reached": False,
            "implementation_debug": "not planner-visible",
        }
    )

    assert "operation_candidate_id" not in compact
    assert private_id not in json.dumps(compact, sort_keys=True)
    assert compact["operation_target_id"] == "public-target-01"
    assert compact["target_reached"] is False


@pytest.mark.parametrize(
    ("action_mode", "point_key"),
    [("grasp", "grasp_world_m"), ("contact", "contact_world_m")],
)
def test_recovery_history_does_not_expose_runtime_candidate(
    action_mode: str,
    point_key: str,
) -> None:
    agent = object.__new__(ImgAgent)
    private_id = f"runtime-private:{action_mode}:right:000"
    call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "arm": "right",
            "instance_id": "track_public_0001",
            "point_key": point_key,
            "_operation_action_mode": action_mode,
            "_operation_candidate_id": private_id,
        },
    )

    context = agent._recovery_tool_history_context(
        call=call,
        result_details={
            "arm": "right",
            "instance_id": "track_public_0001",
            "operation_action_mode": action_mode,
            "operation_candidate_id": private_id,
            "target_reached": True,
        },
        result_message="executed grounded move",
        result_success=True,
        last_grounded_point_by_arm={},
        last_grounded_reached_by_arm={},
    )

    rendered = ",".join(context)
    assert "candidate=" not in rendered
    assert private_id not in rendered
    assert f"mode={action_mode}" in context


def test_place_recovery_history_exposes_target_but_not_runtime_candidate() -> None:
    agent = object.__new__(ImgAgent)
    private_id = "runtime-private:place:right:000"
    public_target = "place:held:track_public_0001:free-support:01"
    call = RecoveryToolCall(
        tool_name="move_ee_to_grounded_instance",
        args={
            "arm": "right",
            "instance_id": "track_public_0001",
            "point_key": "place_world_m",
            "target_id": public_target,
            "_operation_action_mode": "place",
            "_operation_candidate_id": private_id,
        },
    )

    context = agent._recovery_tool_history_context(
        call=call,
        result_details={
            "arm": "right",
            "instance_id": "track_public_0001",
            "operation_action_mode": "place",
            "operation_candidate_id": private_id,
            "operation_target_id": public_target,
            "target_reached": True,
        },
        result_message="executed grounded move",
        result_success=True,
        last_grounded_point_by_arm={},
        last_grounded_reached_by_arm={},
    )

    rendered = ",".join(context)
    assert f"target={public_target}" in context
    assert "candidate=" not in rendered
    assert private_id not in rendered


def test_public_manipulation_state_recursively_removes_candidate_ids() -> None:
    private_ids = {
        "runtime-private:root",
        "runtime-private:nested",
        "runtime-private:list",
        "runtime-private:tuple",
        "runtime-private:mixed-case",
    }
    state = {
        "operation_candidate_id": "runtime-private:root",
        "target_id": "public-target-01",
        "nested": {
            "selected_operation_candidate_id": "runtime-private:nested",
            "safe": "preserved",
        },
        "items": [
            {"grasp_candidate_id": "runtime-private:list"},
            (
                {"place_candidate_id": "runtime-private:tuple"},
                {"Candidate_ID": "runtime-private:mixed-case"},
            ),
        ],
    }

    projected = public_manipulation_state(state)

    _assert_no_private_candidate_ids(projected)
    serialized = json.dumps(projected, sort_keys=True)
    assert not any(private_id in serialized for private_id in private_ids)
    assert projected["target_id"] == "public-target-01"
    assert projected["nested"]["safe"] == "preserved"
