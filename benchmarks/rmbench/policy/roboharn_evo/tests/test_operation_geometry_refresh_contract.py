from __future__ import annotations

import json

from policy.roboharn_evo.agent.operation_geometry_refresh_contract import (
    OperationGeometryRefreshContract,
)
from policy.roboharn_evo.agent.recovery.tool_specs import (
    RecoveryToolCall,
    RecoveryToolResult,
)


def _scene(*, position_state: str = "current_verified") -> dict:
    return {
        "instances": [
            {
                "instance_id": "track_0001",
                "track_id": "track_0001",
                "position_state": position_state,
                "action_geometry_state": "verified",
                "last_verified_world_m": [0.1, -0.1, 0.77],
                "position_tolerance_m": 0.03,
            }
        ]
    }


def _blocked(arm: str = "right") -> list[dict]:
    return [
        {
            "instance_id": "track_0001",
            "arm": arm,
            "action_mode": "grasp",
            "blocked_candidate_count": 1,
            "available_candidate_count": 0,
        }
    ]


def _retreat(*, arm: str = "right", displacement: float = 0.03):
    return (
        RecoveryToolCall(
            tool_name="retreat_arm",
            args={"arm": arm},
        ),
        RecoveryToolResult(
            tool_name="retreat_arm",
            success=True,
            details={
                "arm": arm,
                "observed_displacement_m": displacement,
            },
        ),
    )


def test_failed_or_unmeasured_motion_does_not_authorize_refresh() -> None:
    contract = OperationGeometryRefreshContract()
    call, result = _retreat(displacement=0.005)
    assert contract.authorize_from_results(
        calls=[call],
        results=[result],
        blocked_scopes=_blocked(),
        scene_memory=_scene(),
        env_step=4,
        capture_id=4,
    ) == []
    failed = RecoveryToolResult(
        tool_name="retreat_arm",
        success=False,
        details={"arm": "right", "observed_displacement_m": 0.03},
    )
    assert contract.authorize_from_results(
        calls=[call],
        results=[failed],
        blocked_scopes=_blocked(),
        scene_memory=_scene(),
        env_step=4,
        capture_id=4,
    ) == []


def test_authorization_is_arm_and_instance_scoped_without_candidate_id() -> None:
    contract = OperationGeometryRefreshContract()
    call, result = _retreat()
    created = contract.authorize_from_results(
        calls=[call],
        results=[result],
        blocked_scopes=[*_blocked("right"), *_blocked("left")],
        scene_memory=_scene(),
        env_step=4,
        capture_id=4,
    )
    assert len(created) == 1
    assert created[0]["instance_ref"] == "track_0001"
    assert created[0]["arm"] == "right"
    assert "candidate_id" not in json.dumps(created[0])


def test_simultaneous_two_arm_refresh_for_one_instance_fails_closed() -> None:
    contract = OperationGeometryRefreshContract()
    call = RecoveryToolCall(
        tool_name="safe_reset_posture",
        args={"arm": "both"},
    )
    result = RecoveryToolResult(
        tool_name="safe_reset_posture",
        success=True,
        details={
            "arm": "both",
            "selected_arms": ["left", "right"],
            "observed_displacements_m": {
                "left": 0.03,
                "right": 0.04,
            },
        },
    )

    created = contract.authorize_from_results(
        calls=[call],
        results=[result],
        blocked_scopes=[*_blocked("right"), *_blocked("left")],
        scene_memory=_scene(),
        env_step=4,
        capture_id=4,
    )

    assert created == []


def test_subset_blocked_scope_with_available_alternative_is_not_authorized() -> None:
    contract = OperationGeometryRefreshContract()
    call, result = _retreat()
    subset = _blocked()
    subset[0].update(
        {
            "available_candidate_count": 1,
            "all_candidates_blocked": False,
        }
    )
    assert contract.authorize_from_results(
        calls=[call],
        results=[result],
        blocked_scopes=subset,
        scene_memory=_scene(),
        env_step=4,
        capture_id=4,
    ) == []


def test_explicit_all_candidates_blocked_authorizes_without_count_field() -> None:
    contract = OperationGeometryRefreshContract()
    call, result = _retreat()
    scope = _blocked()
    scope[0].pop("available_candidate_count")
    scope[0]["all_candidates_blocked"] = True
    assert len(
        contract.authorize_from_results(
            calls=[call],
            results=[result],
            blocked_scopes=scope,
            scene_memory=_scene(),
            env_step=4,
            capture_id=4,
        )
    ) == 1


def test_motion_uncertain_position_cannot_receive_refresh_lease() -> None:
    contract = OperationGeometryRefreshContract()
    call, result = _retreat()
    assert contract.authorize_from_results(
        calls=[call],
        results=[result],
        blocked_scopes=_blocked(),
        scene_memory=_scene(position_state="motion_uncertain"),
        env_step=4,
        capture_id=4,
    ) == []


def test_external_authorization_is_consumed_by_one_distinct_capture() -> None:
    contract = OperationGeometryRefreshContract()
    call, result = _retreat()
    contract.authorize_from_results(
        calls=[call],
        results=[result],
        blocked_scopes=_blocked(),
        scene_memory=_scene(),
        env_step=4,
        capture_id=4,
    )
    first = contract.pending_relocation_leases(
        scene_memory=_scene(),
        env_step=5,
        capture_id=5,
    )
    repeated_same_capture = contract.pending_relocation_leases(
        scene_memory=_scene(),
        env_step=5,
        capture_id=5,
    )
    exhausted = contract.pending_relocation_leases(
        scene_memory=_scene(),
        env_step=6,
        capture_id=6,
    )
    assert len(first) == 1
    assert repeated_same_capture == first
    assert exhausted == []


def test_capture_must_be_strictly_newer_than_authorization_capture() -> None:
    contract = OperationGeometryRefreshContract()
    call, result = _retreat()
    contract.authorize_from_results(
        calls=[call],
        results=[result],
        blocked_scopes=_blocked(),
        scene_memory=_scene(),
        env_step=4,
        capture_id=4,
    )
    assert contract.pending_relocation_leases(
        scene_memory=_scene(),
        env_step=4,
        capture_id=4,
    ) == []
    assert contract.pending_relocation_leases(
        scene_memory=_scene(),
        env_step=4,
        capture_id=3,
    ) == []
    assert len(
        contract.pending_relocation_leases(
            scene_memory=_scene(),
            env_step=5,
            capture_id=5,
        )
    ) == 1


def test_clearance_must_be_last_same_arm_physical_action() -> None:
    contract = OperationGeometryRefreshContract()
    retreat_call, retreat_result = _retreat()
    later_call = RecoveryToolCall(
        tool_name="move_ee_to_pose",
        args={"arm": "right"},
    )
    later_result = RecoveryToolResult(
        tool_name="move_ee_to_pose",
        success=True,
        details={"arm": "right"},
    )
    assert contract.authorize_from_results(
        calls=[retreat_call, later_call],
        results=[retreat_result, later_result],
        blocked_scopes=_blocked(),
        scene_memory=_scene(),
        env_step=4,
        capture_id=4,
    ) == []


def test_observation_or_other_arm_motion_after_clearance_does_not_cancel() -> None:
    for later_call, later_result in (
        (
            RecoveryToolCall(tool_name="reobserve_scene", args={}),
            RecoveryToolResult(
                tool_name="reobserve_scene",
                success=True,
            ),
        ),
        (
            RecoveryToolCall(
                tool_name="move_ee_to_pose",
                args={"arm": "left"},
            ),
            RecoveryToolResult(
                tool_name="move_ee_to_pose",
                success=True,
                details={"arm": "left"},
            ),
        ),
    ):
        contract = OperationGeometryRefreshContract()
        retreat_call, retreat_result = _retreat()
        assert len(
            contract.authorize_from_results(
                calls=[retreat_call, later_call],
                results=[retreat_result, later_result],
                blocked_scopes=_blocked(),
                scene_memory=_scene(),
                env_step=4,
                capture_id=4,
            )
        ) == 1


def test_commanded_path_length_is_not_measured_clearance() -> None:
    contract = OperationGeometryRefreshContract()
    call = RecoveryToolCall(
        tool_name="move_to_home",
        args={"arm": "right"},
    )
    result = RecoveryToolResult(
        tool_name="move_to_home",
        success=True,
        details={
            "arm": "right",
            "applied_translation": {"right": 0.03},
        },
    )
    assert contract.authorize_from_results(
        calls=[call],
        results=[result],
        blocked_scopes=_blocked(),
        scene_memory=_scene(),
        env_step=4,
        capture_id=4,
    ) == []


def test_tracker_pending_state_takes_over_and_revokes_external_lease() -> None:
    contract = OperationGeometryRefreshContract()
    call, result = _retreat()
    contract.authorize_from_results(
        calls=[call],
        results=[result],
        blocked_scopes=_blocked(),
        scene_memory=_scene(),
        env_step=4,
        capture_id=4,
    )
    pending_scene = _scene()
    pending_scene["instances"][0][
        "action_geometry_state"
    ] = "relocation_pending"
    assert contract.pending_relocation_leases(
        scene_memory=pending_scene,
        env_step=5,
        capture_id=5,
    ) == []


def test_reset_revokes_all_refresh_authorizations() -> None:
    contract = OperationGeometryRefreshContract()
    call, result = _retreat()
    contract.authorize_from_results(
        calls=[call],
        results=[result],
        blocked_scopes=_blocked(),
        scene_memory=_scene(),
        env_step=4,
        capture_id=4,
    )
    contract.reset()
    assert contract.pending_relocation_leases(
        scene_memory=_scene(),
        env_step=5,
        capture_id=5,
    ) == []
