from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Sequence
from roboharn_evo.agent.gripper_state import gripper_command_state, snapshot_gripper_commands

from .tool_specs import RecoveryToolCall


_MARKER = "_runtime_ambiguous_grasp_return"
_RETURN_TO_GRASP = "return_to_grasp"
_OPEN_AT_GRASP = "open_at_grasp"
_RETREAT_TO_APPROACH = "retreat_to_approach"
_OBSERVE_AFTER_RETURN = "observe_after_return"
_PHASES = (
    _RETURN_TO_GRASP,
    _OPEN_AT_GRASP,
    _RETREAT_TO_APPROACH,
    _OBSERVE_AFTER_RETURN,
)
_TOOLS_BY_PHASE = {
    _RETURN_TO_GRASP: "move_ee_to_pose",
    _OPEN_AT_GRASP: "open_gripper",
    _RETREAT_TO_APPROACH: "move_ee_to_pose",
    _OBSERVE_AFTER_RETURN: "reobserve_scene",
}
_AMBIGUOUS_EXHAUSTED_REASON = (
    "same_state_ambiguous_grasp_reobserve_budget_exhausted"
)


@dataclass(frozen=True, slots=True)
class AmbiguousGraspReturnLimits:
    """Task-independent geometric limits for one ambiguous-grasp return."""

    corridor_radius_m: float = 0.012
    endpoint_tolerance_m: float = 0.008
    minimum_ingress_length_m: float = 0.01
    maximum_ingress_length_m: float = 0.12
    maximum_return_to_grasp_m: float = 0.04
    maximum_translation_per_step_m: float = 0.02
    maximum_return_steps: int = 3
    maximum_retreat_steps: int = 8
    quaternion_norm_tolerance: float = 0.05
    quaternion_alignment_min_abs_dot: float = 0.995
    diagnostic_lift_consistency_tolerance_m: float = 0.01


DEFAULT_AMBIGUOUS_GRASP_RETURN_LIMITS = AmbiguousGraspReturnLimits()


@dataclass(frozen=True, slots=True)
class AmbiguousGraspReturnPlan:
    applicable: bool
    authorized: bool
    reason: str
    calls: tuple[RecoveryToolCall, ...] = ()
    arm: str = ""
    held_instance_id: str = ""
    grasp_attempt_nonce: str = ""


@dataclass(frozen=True, slots=True)
class AmbiguousGraspReturnReduction:
    manipulation_state: dict[str, Any]
    physical_return_succeeded: bool
    fresh_snapshot_confirmed: bool
    attempt_completed: bool
    reason: str


def plan_ambiguous_grasp_return(
    *,
    manipulation_state: Any,
    robot_state: Any,
    evidence_decision: Any,
    include_reobserve: bool = True,
    limits: AmbiguousGraspReturnLimits = (
        DEFAULT_AMBIGUOUS_GRASP_RETURN_LIMITS
    ),
) -> AmbiguousGraspReturnPlan:
    """Build a bounded reverse-ingress release for an ambiguous grasp.

    This function is intentionally a pure planner.  It neither executes calls
    nor edits manipulation state.  An attempt is eligible only after one
    completed diagnostic lift and an exact, same-state evidence-budget
    exhaustion decision.  Every geometry or identity uncertainty returns an
    empty, unauthorized plan.
    """

    limit_error = _limits_error(limits)
    if limit_error:
        return _blocked(limit_error, applicable=True)

    states = manipulation_state if isinstance(manipulation_state, dict) else {}
    pending_returns = [
        (arm, value)
        for arm, value in states.items()
        if arm in {"left", "right"}
        and isinstance(value, dict)
        and str(value.get("phase", "") or "").strip().lower()
        == "ambiguous_grasp_return_pending"
    ]
    if pending_returns:
        if len(pending_returns) != 1:
            return _blocked(
                "more than one ambiguous grasp return is pending",
                applicable=True,
            )
        arm, state = pending_returns[0]
        return _plan_pending_return(
            arm=arm,
            state=state,
            robot_state=robot_state,
            include_reobserve=include_reobserve,
            limits=limits,
        )

    decision_error = _decision_error(evidence_decision)
    if decision_error:
        return _blocked(decision_error, applicable=False)

    pending = [
        arm
        for arm, value in states.items()
        if arm in {"left", "right"}
        and isinstance(value, dict)
        and str(value.get("phase", "") or "").strip().lower()
        == "grasp_candidate"
        and value.get("holding_confirmed") is not True
        and value.get("transport_authorized") is not True
    ]
    if len(pending) > 1:
        return _blocked(
            "more than one grasp_candidate attempt is active",
            applicable=True,
        )

    decision_key = dict(evidence_decision.get("state_key") or {})
    arm = _single_arm(decision_key.get("arm"))
    if arm is None:
        return _blocked(
            "evidence decision lacks one exact physical arm",
            applicable=True,
        )
    state = states.get(arm)
    if not isinstance(state, dict) or arm not in pending:
        return _blocked(
            "evidence decision does not identify an active grasp_candidate",
            applicable=False,
        )

    instance_id = str(state.get("held_instance_id", "") or "").strip()
    if not instance_id:
        return _blocked("missing exact held_instance_id", applicable=True)
    nonce = str(state.get("grasp_attempt_nonce", "") or "").strip()
    if not nonce:
        return _blocked("missing exact grasp_attempt_nonce", applicable=True)
    identity_error = _decision_identity_error(
        decision_key,
        arm=arm,
        held_instance_id=instance_id,
        grasp_attempt_nonce=nonce,
    )
    if identity_error:
        return _blocked(
            identity_error,
            applicable=True,
            arm=arm,
            held_instance_id=instance_id,
            grasp_attempt_nonce=nonce,
        )

    grasp = _vector(state.get("grasp_ee_target_world_m"), length=3)
    if grasp is None:
        return _blocked_bound(
            "missing or invalid exact grasp pose",
            arm,
            instance_id,
            nonce,
        )
    approach = _vector(state.get("grasp_approach_world_m"), length=3)
    if approach is None:
        return _blocked_bound(
            "missing or invalid exact approach pose",
            arm,
            instance_id,
            nonce,
        )

    robots = robot_state if isinstance(robot_state, dict) else {}
    arm_robot = robots.get(arm)
    if not isinstance(arm_robot, dict):
        return _blocked_bound(
            "current robot pose is missing for the exact arm",
            arm,
            instance_id,
            nonce,
        )
    current = _vector(arm_robot.get("xyz"), length=3)
    if current is None:
        return _blocked_bound(
            "current robot xyz is missing or invalid",
            arm,
            instance_id,
            nonce,
        )
    quaternion = _validated_quaternion(
        arm_robot.get("quat_wxyz"),
        norm_tolerance=limits.quaternion_norm_tolerance,
    )
    if quaternion is None:
        return _blocked_bound(
            "current robot quaternion is missing, invalid, or non-unit",
            arm,
            instance_id,
            nonce,
        )
    if gripper_command_state(arm_robot) != "closed":
        return _blocked_bound(
            "exact arm lacks a maintained closing command",
            arm,
            instance_id,
            nonce,
        )

    ingress = _subtract(approach, grasp)
    ingress_length = _norm(ingress)
    if (
        ingress_length < limits.minimum_ingress_length_m
        or ingress_length > limits.maximum_ingress_length_m
    ):
        return _blocked_bound(
            "recorded grasp-to-approach distance limit is violated",
            arm,
            instance_id,
            nonce,
        )
    corridor_error = _corridor_error(
        point=current,
        grasp=grasp,
        approach=approach,
        corridor_radius_m=limits.corridor_radius_m,
        endpoint_tolerance_m=limits.endpoint_tolerance_m,
    )
    if corridor_error:
        return _blocked_bound(
            corridor_error,
            arm,
            instance_id,
            nonce,
        )
    return_distance = _distance(current, grasp)
    if return_distance > limits.maximum_return_to_grasp_m:
        return _blocked_bound(
            "current-to-grasp distance limit is exceeded",
            arm,
            instance_id,
            nonce,
        )

    diagnostic = state.get("diagnostic_lift_evidence")
    diagnostic_status, diagnostic_reason = _diagnostic_status(
        diagnostic,
        arm=arm,
        held_instance_id=instance_id,
        grasp_attempt_nonce=nonce,
        grasp=grasp,
        current=current,
        approach=approach,
        limits=limits,
    )
    if diagnostic_status != "ambiguous":
        return _blocked(
            diagnostic_reason,
            applicable=diagnostic_status == "invalid",
            arm=arm,
            held_instance_id=instance_id,
            grasp_attempt_nonce=nonce,
        )

    return_steps = _step_count(
        return_distance,
        maximum_step=limits.maximum_translation_per_step_m,
    )
    retreat_steps = _step_count(
        ingress_length,
        maximum_step=limits.maximum_translation_per_step_m,
    )
    if return_steps > limits.maximum_return_steps:
        return _blocked_bound(
            "return-to-grasp step bound is exceeded",
            arm,
            instance_id,
            nonce,
        )
    if retreat_steps > limits.maximum_retreat_steps:
        return _blocked_bound(
            "grasp-to-approach retreat step bound is exceeded",
            arm,
            instance_id,
            nonce,
        )

    calls = [
        _move_call(
            phase=_RETURN_TO_GRASP,
            arm=arm,
            held_instance_id=instance_id,
            grasp_attempt_nonce=nonce,
            target=grasp,
            quaternion=quaternion,
            distance=return_distance,
            steps=return_steps,
            limits=limits,
        ),
        RecoveryToolCall(
            tool_name="open_gripper",
            args={
                **_transaction_args(
                    phase=_OPEN_AT_GRASP,
                    arm=arm,
                    held_instance_id=instance_id,
                    grasp_attempt_nonce=nonce,
                ),
                "release_held_instance_id": instance_id,
                "_guard_reason": (
                    "release the exact ambiguous grasp attempt only after "
                    "returning to its recorded grasp pose"
                ),
            },
        ),
        _move_call(
            phase=_RETREAT_TO_APPROACH,
            arm=arm,
            held_instance_id=instance_id,
            grasp_attempt_nonce=nonce,
            target=approach,
            quaternion=quaternion,
            distance=ingress_length,
            steps=retreat_steps,
            limits=limits,
        ),
    ]
    if include_reobserve:
        calls.append(
            RecoveryToolCall(
                tool_name="reobserve_scene",
                args={
                    **_transaction_args(
                        phase=_OBSERVE_AFTER_RETURN,
                        arm=arm,
                        held_instance_id=instance_id,
                        grasp_attempt_nonce=nonce,
                    ),
                    "_guard_reason": (
                        "capture a fresh snapshot after the exact ambiguous "
                        "grasp attempt was physically returned"
                    ),
                },
            )
        )
    return AmbiguousGraspReturnPlan(
        applicable=True,
        authorized=True,
        reason="ambiguous_grasp_return_ready",
        calls=tuple(calls),
        arm=arm,
        held_instance_id=instance_id,
        grasp_attempt_nonce=nonce,
    )


def _plan_pending_return(
    *,
    arm: str,
    state: dict[str, Any],
    robot_state: Any,
    include_reobserve: bool,
    limits: AmbiguousGraspReturnLimits,
) -> AmbiguousGraspReturnPlan:
    transaction = state.get("ambiguous_grasp_return")
    if not isinstance(transaction, dict):
        return _blocked(
            "pending ambiguous grasp return lost its exact transaction",
            applicable=True,
            arm=arm,
        )
    status = str(transaction.get("status", "") or "").strip().lower()
    allowed_statuses = {
        "return_to_grasp_pending",
        "at_grasp_closed",
        "gripper_opened_retreat_pending",
        "approach_reached_confirmation_pending",
    }
    if status not in allowed_statuses:
        return _blocked(
            "pending ambiguous grasp return has an invalid status",
            applicable=True,
            arm=arm,
        )
    instance_id = str(
        transaction.get("held_instance_id", "") or ""
    ).strip()
    nonce = str(transaction.get("grasp_attempt_nonce", "") or "").strip()
    if not (
        _single_arm(transaction.get("arm")) == arm
        and instance_id
        and nonce
        and str(state.get("held_instance_id", "") or "").strip()
        == instance_id
        and str(state.get("grasp_attempt_nonce", "") or "").strip()
        == nonce
    ):
        return _blocked(
            "pending ambiguous grasp return identity is inconsistent",
            applicable=True,
            arm=arm,
            held_instance_id=instance_id,
            grasp_attempt_nonce=nonce,
        )
    grasp = _vector(transaction.get("grasp_world_m"), length=3)
    approach = _vector(transaction.get("approach_world_m"), length=3)
    quaternion = _validated_quaternion(
        transaction.get("quat_wxyz"),
        norm_tolerance=limits.quaternion_norm_tolerance,
    )
    state_grasp = _vector(state.get("grasp_ee_target_world_m"), length=3)
    state_approach = _vector(state.get("grasp_approach_world_m"), length=3)
    if (
        grasp is None
        or approach is None
        or quaternion is None
        or state_grasp is None
        or state_approach is None
        or _distance(grasp, state_grasp) > 1e-8
        or _distance(approach, state_approach) > 1e-8
    ):
        return _blocked_bound(
            "pending ambiguous grasp return geometry is missing or altered",
            arm,
            instance_id,
            nonce,
        )
    ingress_length = _distance(grasp, approach)
    if not (
        limits.minimum_ingress_length_m
        <= ingress_length
        <= limits.maximum_ingress_length_m
    ):
        return _blocked_bound(
            "pending grasp-to-approach distance limit is violated",
            arm,
            instance_id,
            nonce,
        )

    robots = robot_state if isinstance(robot_state, dict) else {}
    arm_robot = robots.get(arm)
    current = (
        _vector(arm_robot.get("xyz"), length=3)
        if isinstance(arm_robot, dict)
        else None
    )
    current_quaternion = (
        _validated_quaternion(
            arm_robot.get("quat_wxyz"),
            norm_tolerance=limits.quaternion_norm_tolerance,
        )
        if isinstance(arm_robot, dict)
        else None
    )
    command = gripper_command_state(arm_robot)
    if current is None or current_quaternion is None or command == "unknown":
        return _blocked_bound(
            "pending return current robot pose, gripper, or quaternion is missing",
            arm,
            instance_id,
            nonce,
        )
    if (
        abs(
            _dot(
                _normalized(current_quaternion),
                _normalized(quaternion),
            )
        )
        < limits.quaternion_alignment_min_abs_dot
    ):
        return _blocked_bound(
            "pending return current quaternion differs from the saved transaction",
            arm,
            instance_id,
            nonce,
        )
    corridor_error = _corridor_error(
        point=current,
        grasp=grasp,
        approach=approach,
        corridor_radius_m=limits.corridor_radius_m,
        endpoint_tolerance_m=limits.endpoint_tolerance_m,
    )
    if corridor_error:
        return _blocked_bound(
            corridor_error,
            arm,
            instance_id,
            nonce,
        )

    gripper_is_open = status in {
        "gripper_opened_retreat_pending",
        "approach_reached_confirmation_pending",
    }
    if gripper_is_open:
        if command != "open":
            return _blocked_bound(
                "pending return requires the exact arm to remain open gripper",
                arm,
                instance_id,
                nonce,
            )
    elif command != "closed":
        return _blocked_bound(
            "pending return requires the exact arm to remain closed gripper",
            arm,
            instance_id,
            nonce,
        )

    calls: list[RecoveryToolCall] = []
    if status == "return_to_grasp_pending":
        distance = _distance(current, grasp)
        steps = _step_count(
            distance,
            maximum_step=limits.maximum_translation_per_step_m,
        )
        if (
            distance > limits.maximum_return_to_grasp_m
            or steps > limits.maximum_return_steps
        ):
            return _blocked_bound(
                "pending current-to-grasp distance or step limit is exceeded",
                arm,
                instance_id,
                nonce,
            )
        calls.append(
            _move_call(
                phase=_RETURN_TO_GRASP,
                arm=arm,
                held_instance_id=instance_id,
                grasp_attempt_nonce=nonce,
                target=grasp,
                quaternion=quaternion,
                distance=distance,
                steps=steps,
                limits=limits,
            )
        )
    if status in {"return_to_grasp_pending", "at_grasp_closed"}:
        if status == "at_grasp_closed" and (
            _distance(current, grasp) > limits.endpoint_tolerance_m
        ):
            return _blocked_bound(
                "closed pending return is not at the exact grasp pose",
                arm,
                instance_id,
                nonce,
            )
        calls.append(
            RecoveryToolCall(
                tool_name="open_gripper",
                args={
                    **_transaction_args(
                        phase=_OPEN_AT_GRASP,
                        arm=arm,
                        held_instance_id=instance_id,
                        grasp_attempt_nonce=nonce,
                    ),
                    "release_held_instance_id": instance_id,
                    "_guard_reason": (
                        "resume the exact saved return transaction at its "
                        "recorded grasp pose"
                    ),
                },
            )
        )
    if status != "approach_reached_confirmation_pending":
        remaining = _distance(current, approach)
        if status in {"return_to_grasp_pending", "at_grasp_closed"}:
            remaining = ingress_length
        steps = _step_count(
            remaining,
            maximum_step=limits.maximum_translation_per_step_m,
        )
        if (
            remaining > limits.maximum_ingress_length_m
            or steps > limits.maximum_retreat_steps
        ):
            return _blocked_bound(
                "pending retreat distance or step limit is exceeded",
                arm,
                instance_id,
                nonce,
            )
        calls.append(
            _move_call(
                phase=_RETREAT_TO_APPROACH,
                arm=arm,
                held_instance_id=instance_id,
                grasp_attempt_nonce=nonce,
                target=approach,
                quaternion=quaternion,
                distance=remaining,
                steps=steps,
                limits=limits,
            )
        )
    elif _distance(current, approach) > limits.endpoint_tolerance_m:
        return _blocked_bound(
            "confirmation-pending return is not at the saved approach pose",
            arm,
            instance_id,
            nonce,
        )

    if include_reobserve:
        calls.append(
            RecoveryToolCall(
                tool_name="reobserve_scene",
                args={
                    **_transaction_args(
                        phase=_OBSERVE_AFTER_RETURN,
                        arm=arm,
                        held_instance_id=instance_id,
                        grasp_attempt_nonce=nonce,
                    ),
                    "_guard_reason": (
                        "confirm the resumed ambiguous grasp return with one "
                        "bounded fresh snapshot"
                    ),
                },
            )
        )
    return AmbiguousGraspReturnPlan(
        applicable=True,
        authorized=True,
        reason="ambiguous_grasp_return_resume_ready",
        calls=tuple(calls),
        arm=arm,
        held_instance_id=instance_id,
        grasp_attempt_nonce=nonce,
    )


def is_runtime_ambiguous_grasp_return_call(
    call: RecoveryToolCall,
) -> bool:
    phase = str((call.args or {}).get(_MARKER, "") or "").strip()
    return bool(
        phase in _TOOLS_BY_PHASE
        and call.tool_name == _TOOLS_BY_PHASE[phase]
        and _single_arm((call.args or {}).get("arm")) is not None
        and str(
            (call.args or {}).get("held_instance_id", "") or ""
        ).strip()
        and str(
            (call.args or {}).get("grasp_attempt_nonce", "") or ""
        ).strip()
    )


def reduce_ambiguous_grasp_return_results(
    manipulation_state: Any,
    *,
    calls: Sequence[RecoveryToolCall],
    results: Sequence[Any],
    post_return_snapshot: Any | None,
    limits: AmbiguousGraspReturnLimits = (
        DEFAULT_AMBIGUOUS_GRASP_RETURN_LIMITS
    ),
) -> AmbiguousGraspReturnReduction:
    """Reduce exact phase progress without forgetting irreversible actions.

    Opening the gripper is an irreversible boundary.  Once observed successful,
    the returned state records that fact and subsequent plans can only retreat
    or confirm; they can never lower or open a second time.
    """

    states = deepcopy(
        manipulation_state if isinstance(manipulation_state, dict) else {}
    )
    if not calls and not results:
        return _reduce_pending_snapshot_only(
            states,
            post_return_snapshot=post_return_snapshot,
            limits=limits,
        )
    transaction = _return_transaction(calls)
    if transaction is None:
        return _reduction(states, reason="invalid runtime return markers")
    arm, instance_id, nonce, phases = transaction
    state = states.get(arm)
    if not _state_matches_transaction(
        state,
        arm=arm,
        held_instance_id=instance_id,
        grasp_attempt_nonce=nonce,
    ):
        return _reduction(states, reason="runtime return identity is stale")
    assert isinstance(state, dict)

    grasp = _vector(state.get("grasp_ee_target_world_m"), length=3)
    approach = _vector(state.get("grasp_approach_world_m"), length=3)
    if grasp is None or approach is None:
        return _reduction(states, reason="recorded return geometry is missing")
    expected_quaternion = _saved_or_called_quaternion(
        state=state,
        calls=calls,
        limits=limits,
    )
    if expected_quaternion is None:
        return _reduction(states, reason="recorded return quaternion is missing")
    if len(results) != len(calls):
        return _reduction(states, reason="return result cardinality mismatch")
    if not _calls_match_recorded_geometry(
        calls,
        grasp=grasp,
        approach=approach,
        expected_quaternion=expected_quaternion,
        limits=limits,
    ):
        return _reduction(states, reason="runtime return geometry was altered")

    status, last_physical_step = _existing_return_progress(state)
    if not status:
        status = "return_to_grasp_pending"
    for call, result, phase in zip(calls, results, phases):
        succeeded, result_step = _phase_result_succeeded(
            phase=phase,
            call=call,
            result=result,
            arm=arm,
            limits=limits,
        )
        if not succeeded:
            break
        if (
            phase != _OBSERVE_AFTER_RETURN
            and result_step is not None
            and result_step <= last_physical_step
        ):
            break
        if phase == _RETURN_TO_GRASP:
            status = "at_grasp_closed"
        elif phase == _OPEN_AT_GRASP:
            status = "gripper_opened_retreat_pending"
        elif phase == _RETREAT_TO_APPROACH:
            status = "approach_reached_confirmation_pending"
        if phase != _OBSERVE_AFTER_RETURN and result_step is not None:
            last_physical_step = result_step

    physical_ok = status == "approach_reached_confirmation_pending"
    if physical_ok and _post_snapshot_confirms_return(
        snapshot=post_return_snapshot,
        arm=arm,
        approach=approach,
        expected_quaternion=expected_quaternion,
        state=state,
        final_physical_step=last_physical_step,
        limits=limits,
    ):
        states.pop(arm, None)
        return AmbiguousGraspReturnReduction(
            manipulation_state=states,
            physical_return_succeeded=True,
            fresh_snapshot_confirmed=True,
            attempt_completed=True,
            reason="ambiguous_grasp_return_completed",
        )

    states[arm] = _pending_return_state(
        state,
        arm=arm,
        held_instance_id=instance_id,
        grasp_attempt_nonce=nonce,
        grasp=grasp,
        approach=approach,
        quaternion=expected_quaternion,
        status=status,
        last_physical_step=last_physical_step,
    )
    return AmbiguousGraspReturnReduction(
        manipulation_state=states,
        physical_return_succeeded=physical_ok,
        fresh_snapshot_confirmed=False,
        attempt_completed=False,
        reason=(
            "fresh post-return snapshot is not verified"
            if physical_ok
            else "ambiguous grasp physical return is resumable"
        ),
    )


def _reduce_pending_snapshot_only(
    states: dict[str, Any],
    *,
    post_return_snapshot: Any,
    limits: AmbiguousGraspReturnLimits,
) -> AmbiguousGraspReturnReduction:
    pending = [
        (arm, state)
        for arm, state in states.items()
        if arm in {"left", "right"}
        and isinstance(state, dict)
        and str(state.get("phase", "") or "").strip().lower()
        == "ambiguous_grasp_return_pending"
    ]
    if len(pending) != 1:
        return _reduction(
            states,
            reason="snapshot-only confirmation lacks one pending return",
        )
    arm, state = pending[0]
    transaction = state.get("ambiguous_grasp_return")
    if not isinstance(transaction, dict) or str(
        transaction.get("status", "") or ""
    ).strip().lower() != "approach_reached_confirmation_pending":
        return _reduction(
            states,
            reason="physical return has not reached confirmation state",
        )
    instance_id = str(
        transaction.get("held_instance_id", "") or ""
    ).strip()
    nonce = str(transaction.get("grasp_attempt_nonce", "") or "").strip()
    if not _state_matches_transaction(
        state,
        arm=arm,
        held_instance_id=instance_id,
        grasp_attempt_nonce=nonce,
    ):
        return _reduction(
            states,
            reason="snapshot-only return identity is stale",
        )
    approach = _vector(state.get("grasp_approach_world_m"), length=3)
    quaternion = _validated_quaternion(
        transaction.get("quat_wxyz"),
        norm_tolerance=limits.quaternion_norm_tolerance,
    )
    last_step = _nonnegative_integer(
        transaction.get("last_physical_step")
    )
    if approach is None or quaternion is None or last_step is None:
        return _reduction(
            states,
            reason="snapshot-only return geometry is incomplete",
        )
    if not _post_snapshot_confirms_return(
        snapshot=post_return_snapshot,
        arm=arm,
        approach=approach,
        expected_quaternion=quaternion,
        state=state,
        final_physical_step=last_step,
        limits=limits,
    ):
        return AmbiguousGraspReturnReduction(
            manipulation_state=states,
            physical_return_succeeded=True,
            fresh_snapshot_confirmed=False,
            attempt_completed=False,
            reason="fresh post-return snapshot is not verified",
        )
    states.pop(arm, None)
    return AmbiguousGraspReturnReduction(
        manipulation_state=states,
        physical_return_succeeded=True,
        fresh_snapshot_confirmed=True,
        attempt_completed=True,
        reason="ambiguous_grasp_return_completed",
    )


def _diagnostic_status(
    value: Any,
    *,
    arm: str,
    held_instance_id: str,
    grasp_attempt_nonce: str,
    grasp: list[float],
    current: list[float],
    approach: list[float],
    limits: AmbiguousGraspReturnLimits,
) -> tuple[str, str]:
    if not isinstance(value, dict):
        return "invalid", "diagnostic lift evidence is missing"
    validation = value.get("runtime_grasp_validation")
    if value.get("positive_attachment_evidence") is True or (
        isinstance(validation, dict) and validation.get("verified") is True
    ):
        return "resolved", "diagnostic lift already confirmed attachment"
    if value.get("negative_attachment_evidence") is True or (
        isinstance(validation, dict) and validation.get("verified") is False
    ):
        return "resolved", "diagnostic lift already disproved attachment"
    if value.get("transport_authorized") is not False:
        return "invalid", "diagnostic lift transport authority is not false"
    observed_lift = _finite_float(value.get("observed_lift_m"))
    prelift = _vector(value.get("prelift_ee_world_m"), length=3)
    postlift = _vector(value.get("postlift_ee_world_m"), length=3)
    lift_step = _nonnegative_integer(value.get("lift_env_step"))
    if (
        observed_lift is None
        or observed_lift <= 0.0
        or prelift is None
        or postlift is None
        or lift_step is None
    ):
        return "invalid", "diagnostic lift completion record is incomplete"
    if _distance(prelift, grasp) > limits.endpoint_tolerance_m:
        return "invalid", "diagnostic prelift pose does not match grasp pose"
    if _distance(postlift, current) > limits.endpoint_tolerance_m:
        return "invalid", "current pose does not match diagnostic postlift pose"
    measured_lift = _distance(prelift, postlift)
    if (
        abs(measured_lift - observed_lift)
        > limits.diagnostic_lift_consistency_tolerance_m
        or measured_lift > limits.maximum_return_to_grasp_m
    ):
        return "invalid", "diagnostic lift distance is inconsistent or unsafe"
    if _corridor_error(
        point=postlift,
        grasp=grasp,
        approach=approach,
        corridor_radius_m=limits.corridor_radius_m,
        endpoint_tolerance_m=limits.endpoint_tolerance_m,
    ):
        return "invalid", "diagnostic lift left the recorded ingress corridor"
    if isinstance(validation, dict):
        if not (
            validation.get("applicable") is True
            and validation.get("verified") is None
            and _single_arm(validation.get("arm")) == arm
            and str(
                validation.get("held_instance_id", "") or ""
            ).strip()
            == held_instance_id
            and str(
                validation.get("grasp_attempt_nonce", "") or ""
            ).strip()
            == grasp_attempt_nonce
        ):
            return "invalid", "diagnostic lift validation identity is invalid"
    return "ambiguous", ""


def _decision_error(value: Any) -> str:
    if not isinstance(value, dict):
        return "ambiguous-grasp evidence decision is missing"
    try:
        used = int(value.get("stationary_reobserves_used"))
        limit = int(value.get("stationary_reobserve_limit"))
    except (TypeError, ValueError):
        return "same-state reobserve budget accounting is missing"
    if not (
        value.get("allow_reobserve") is False
        and str(value.get("evidence_situation", "") or "").strip()
        == "ambiguous_grasp"
        and str(
            value.get("next_information_action", "") or ""
        ).strip()
        == "perform_controlled_return_and_regrasp"
        and str(value.get("exhausted_reason", "") or "").strip()
        == _AMBIGUOUS_EXHAUSTED_REASON
        and limit >= 0
        and used >= limit
        and isinstance(value.get("state_key"), dict)
    ):
        return "same-state ambiguous-grasp reobserve budget is not exhausted"
    return ""


def _decision_identity_error(
    key: dict[str, Any],
    *,
    arm: str,
    held_instance_id: str,
    grasp_attempt_nonce: str,
) -> str:
    if (
        str(key.get("track_id", "") or "").strip() != held_instance_id
        or _single_arm(key.get("arm")) != arm
        or str(key.get("manipulation_phase", "") or "").strip().lower()
        != "grasp_candidate"
        or str(key.get("grasp_attempt_nonce", "") or "").strip()
        != grasp_attempt_nonce
    ):
        return "evidence budget does not match the exact grasp attempt"
    return ""


def _corridor_error(
    *,
    point: list[float],
    grasp: list[float],
    approach: list[float],
    corridor_radius_m: float,
    endpoint_tolerance_m: float,
) -> str:
    ingress = _subtract(approach, grasp)
    length = _norm(ingress)
    if length <= 1e-9:
        return "recorded grasp and approach do not define a segment"
    unit = _scale(ingress, 1.0 / length)
    relative = _subtract(point, grasp)
    progress = _dot(relative, unit)
    if progress < -endpoint_tolerance_m or progress > length + endpoint_tolerance_m:
        return "current point lies outside the recorded grasp-approach segment"
    nearest = _add(grasp, _scale(unit, progress))
    if _distance(point, nearest) > corridor_radius_m:
        return "current point lies outside the recorded ingress corridor"
    return ""


def _move_call(
    *,
    phase: str,
    arm: str,
    held_instance_id: str,
    grasp_attempt_nonce: str,
    target: list[float],
    quaternion: list[float],
    distance: float,
    steps: int,
    limits: AmbiguousGraspReturnLimits,
) -> RecoveryToolCall:
    per_step = max(0.001, distance / max(1, steps))
    return RecoveryToolCall(
        tool_name="move_ee_to_pose",
        args={
            **_transaction_args(
                phase=phase,
                arm=arm,
                held_instance_id=held_instance_id,
                grasp_attempt_nonce=grasp_attempt_nonce,
            ),
            "target_pose": [*target, *quaternion],
            "max_translation": round(
                min(per_step, limits.maximum_translation_per_step_m),
                6,
            ),
            "steps": int(steps),
            "_guard_reason": (
                "follow the exact recorded grasp ingress in reverse for an "
                "identity-bound ambiguous grasp attempt"
            ),
        },
    )


def _transaction_args(
    *,
    phase: str,
    arm: str,
    held_instance_id: str,
    grasp_attempt_nonce: str,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "held_instance_id": held_instance_id,
        "grasp_attempt_nonce": grasp_attempt_nonce,
        _MARKER: phase,
    }


def _return_transaction(
    calls: Sequence[RecoveryToolCall],
) -> tuple[str, str, str, tuple[str, ...]] | None:
    if not calls:
        return None
    if not all(is_runtime_ambiguous_grasp_return_call(call) for call in calls):
        return None
    phases = tuple(
        str(call.args.get(_MARKER, "") or "").strip()
        for call in calls
    )
    allowed = {
        (_RETURN_TO_GRASP, _OPEN_AT_GRASP, _RETREAT_TO_APPROACH),
        (
            _RETURN_TO_GRASP,
            _OPEN_AT_GRASP,
            _RETREAT_TO_APPROACH,
            _OBSERVE_AFTER_RETURN,
        ),
        (_OPEN_AT_GRASP, _RETREAT_TO_APPROACH),
        (_OPEN_AT_GRASP, _RETREAT_TO_APPROACH, _OBSERVE_AFTER_RETURN),
        (_RETREAT_TO_APPROACH,),
        (_RETREAT_TO_APPROACH, _OBSERVE_AFTER_RETURN),
        (_OBSERVE_AFTER_RETURN,),
    }
    if phases not in allowed:
        return None
    arms = {_single_arm(call.args.get("arm")) for call in calls}
    instances = {
        str(call.args.get("held_instance_id", "") or "").strip()
        for call in calls
    }
    nonces = {
        str(call.args.get("grasp_attempt_nonce", "") or "").strip()
        for call in calls
    }
    if len(arms) != 1 or None in arms or len(instances) != 1 or len(nonces) != 1:
        return None
    arm = next(iter(arms))
    instance_id = next(iter(instances))
    nonce = next(iter(nonces))
    if not instance_id or not nonce:
        return None
    for call, phase in zip(calls, phases):
        if phase == _OPEN_AT_GRASP and str(
            call.args.get("release_held_instance_id", "") or ""
        ).strip() != instance_id:
            return None
    return arm, instance_id, nonce, phases


def _calls_match_recorded_geometry(
    calls: Sequence[RecoveryToolCall],
    *,
    grasp: list[float],
    approach: list[float],
    expected_quaternion: list[float],
    limits: AmbiguousGraspReturnLimits,
) -> bool:
    motion_quaternions: list[list[float]] = []
    for call in calls:
        phase = str(call.args.get(_MARKER, "") or "").strip()
        if phase not in {_RETURN_TO_GRASP, _RETREAT_TO_APPROACH}:
            continue
        expected = grasp if phase == _RETURN_TO_GRASP else approach
        pose = _vector(call.args.get("target_pose"), length=7)
        step = _finite_float(call.args.get("max_translation"))
        steps = _positive_integer(call.args.get("steps"))
        if (
            pose is None
            or _distance(pose[:3], expected) > 1e-8
            or _validated_quaternion(
                pose[3:],
                norm_tolerance=limits.quaternion_norm_tolerance,
            )
            is None
            or step is None
            or step <= 0.0
            or step > limits.maximum_translation_per_step_m + 1e-9
            or steps is None
        ):
            return False
        assert pose is not None
        motion_quaternions.append(pose[3:])
    return all(
        abs(
            _dot(
                _normalized(quaternion),
                _normalized(expected_quaternion),
            )
        )
        >= limits.quaternion_alignment_min_abs_dot
        for quaternion in motion_quaternions
    )


def _saved_or_called_quaternion(
    *,
    state: dict[str, Any],
    calls: Sequence[RecoveryToolCall],
    limits: AmbiguousGraspReturnLimits,
) -> list[float] | None:
    transaction = state.get("ambiguous_grasp_return")
    if isinstance(transaction, dict):
        saved = _validated_quaternion(
            transaction.get("quat_wxyz"),
            norm_tolerance=limits.quaternion_norm_tolerance,
        )
        if saved is not None:
            return saved
    for call in calls:
        phase = str(call.args.get(_MARKER, "") or "").strip()
        if phase not in {_RETURN_TO_GRASP, _RETREAT_TO_APPROACH}:
            continue
        pose = _vector(call.args.get("target_pose"), length=7)
        if pose is None:
            return None
        return _validated_quaternion(
            pose[3:],
            norm_tolerance=limits.quaternion_norm_tolerance,
        )
    return None


def _existing_return_progress(
    state: dict[str, Any],
) -> tuple[str, int]:
    transaction = state.get("ambiguous_grasp_return")
    if not isinstance(transaction, dict):
        return "", -1
    status = str(transaction.get("status", "") or "").strip().lower()
    if status not in {
        "return_to_grasp_pending",
        "at_grasp_closed",
        "gripper_opened_retreat_pending",
        "approach_reached_confirmation_pending",
    }:
        return "", -1
    step = _nonnegative_integer(transaction.get("last_physical_step"))
    return status, (-1 if step is None else step)


def _phase_result_succeeded(
    *,
    phase: str,
    call: RecoveryToolCall,
    result: Any,
    arm: str,
    limits: AmbiguousGraspReturnLimits,
) -> tuple[bool, int | None]:
    if not (
        getattr(result, "success", False)
        and getattr(result, "tool_name", "") == call.tool_name
    ):
        return False, None
    details = dict(getattr(result, "details", {}) or {})
    if phase == _OBSERVE_AFTER_RETURN:
        return True, _nonnegative_integer(details.get("step_count"))
    if _single_arm(details.get("arm")) != arm:
        return False, None
    step = _nonnegative_integer(details.get("step_count"))
    if step is None:
        return False, None
    if phase in {_RETURN_TO_GRASP, _RETREAT_TO_APPROACH}:
        expected = _vector(call.args.get("target_pose"), length=7)
        observed = _vector(details.get("observed_pose"), length=7)
        if (
            details.get("target_reached") is not True
            or expected is None
            or observed is None
            or _distance(expected[:3], observed[:3])
            > limits.endpoint_tolerance_m
        ):
            return False, step
        observed_quaternion = _validated_quaternion(
            observed[3:],
            norm_tolerance=limits.quaternion_norm_tolerance,
        )
        if observed_quaternion is None or (
            abs(
                _dot(
                    _normalized(observed_quaternion),
                    _normalized(expected[3:]),
                )
            )
            < limits.quaternion_alignment_min_abs_dot
        ):
            return False, step
        return True, step
    if phase == _OPEN_AT_GRASP:
        return gripper_command_state({"gripper_command": details.get("gripper_value")}) == "open", step
    return False, step


def _pending_return_state(
    state: dict[str, Any],
    *,
    arm: str,
    held_instance_id: str,
    grasp_attempt_nonce: str,
    grasp: list[float],
    approach: list[float],
    quaternion: list[float],
    status: str,
    last_physical_step: int,
) -> dict[str, Any]:
    return {
        **state,
        "phase": "ambiguous_grasp_return_pending",
        "held_instance_id": held_instance_id,
        "grasp_attempt_nonce": grasp_attempt_nonce,
        "holding_confirmed": False,
        "transport_authorized": False,
        "ambiguous_grasp_return": {
            "status": status,
            "arm": arm,
            "held_instance_id": held_instance_id,
            "grasp_attempt_nonce": grasp_attempt_nonce,
            "grasp_world_m": list(grasp),
            "approach_world_m": list(approach),
            "quat_wxyz": list(quaternion),
            "last_physical_step": int(last_physical_step),
        },
        "evidence": (
            "ambiguous_grasp_return_incomplete;exact_transaction_resume_required"
        ),
    }


def _post_snapshot_confirms_return(
    *,
    snapshot: Any,
    arm: str,
    approach: list[float],
    expected_quaternion: list[float],
    state: dict[str, Any],
    final_physical_step: int,
    limits: AmbiguousGraspReturnLimits,
) -> bool:
    if snapshot is None:
        return False
    snapshot_step = _nonnegative_integer(getattr(snapshot, "step_count", None))
    diagnostic = state.get("diagnostic_lift_evidence")
    lift_step = _nonnegative_integer(
        diagnostic.get("lift_env_step")
        if isinstance(diagnostic, dict)
        else None
    )
    pose = _vector(getattr(snapshot, f"{arm}_endpose", None), length=7)
    if (
        snapshot_step is None
        or lift_step is None
        or snapshot_step < final_physical_step
        or snapshot_step <= lift_step
        or pose is None
        or _distance(pose[:3], approach) > limits.endpoint_tolerance_m
    ):
        return False
    observed_quat = _validated_quaternion(
        pose[3:],
        norm_tolerance=limits.quaternion_norm_tolerance,
    )
    if observed_quat is None:
        return False
    if (
        abs(_dot(_normalized(observed_quat), _normalized(expected_quaternion)))
        < limits.quaternion_alignment_min_abs_dot
    ):
        return False
    return gripper_command_state({"gripper_command": snapshot_gripper_commands(snapshot).get(arm)}) == "open"


def _state_matches_transaction(
    value: Any,
    *,
    arm: str,
    held_instance_id: str,
    grasp_attempt_nonce: str,
) -> bool:
    if not isinstance(value, dict):
        return False
    phase = str(value.get("phase", "") or "").strip().lower()
    if phase not in {
        "grasp_candidate",
        "ambiguous_grasp_return_pending",
    }:
        return False
    if not (
        value.get("holding_confirmed") is not True
        and value.get("transport_authorized") is not True
        and str(value.get("held_instance_id", "") or "").strip()
        == held_instance_id
        and str(value.get("grasp_attempt_nonce", "") or "").strip()
        == grasp_attempt_nonce
    ):
        return False
    if phase == "grasp_candidate":
        return True
    transaction = value.get("ambiguous_grasp_return")
    if not (
        isinstance(transaction, dict)
        and _single_arm(transaction.get("arm")) == arm
        and str(transaction.get("held_instance_id", "") or "").strip()
        == held_instance_id
        and str(transaction.get("grasp_attempt_nonce", "") or "").strip()
        == grasp_attempt_nonce
    ):
        return False
    saved_grasp = _vector(transaction.get("grasp_world_m"), length=3)
    saved_approach = _vector(transaction.get("approach_world_m"), length=3)
    state_grasp = _vector(value.get("grasp_ee_target_world_m"), length=3)
    state_approach = _vector(value.get("grasp_approach_world_m"), length=3)
    return bool(
        saved_grasp is not None
        and saved_approach is not None
        and state_grasp is not None
        and state_approach is not None
        and _distance(saved_grasp, state_grasp) <= 1e-8
        and _distance(saved_approach, state_approach) <= 1e-8
    )


def _limits_error(limits: AmbiguousGraspReturnLimits) -> str:
    positive = (
        limits.corridor_radius_m,
        limits.endpoint_tolerance_m,
        limits.minimum_ingress_length_m,
        limits.maximum_ingress_length_m,
        limits.maximum_return_to_grasp_m,
        limits.maximum_translation_per_step_m,
        limits.quaternion_norm_tolerance,
        limits.quaternion_alignment_min_abs_dot,
        limits.diagnostic_lift_consistency_tolerance_m,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in positive):
        return "ambiguous-grasp return limits must be finite and positive"
    if (
        limits.minimum_ingress_length_m > limits.maximum_ingress_length_m
        or limits.maximum_return_steps <= 0
        or limits.maximum_retreat_steps <= 0
        or limits.quaternion_alignment_min_abs_dot > 1.0
    ):
        return "ambiguous-grasp return limits are internally inconsistent"
    return ""


def _blocked_bound(
    reason: str,
    arm: str,
    instance_id: str,
    nonce: str,
) -> AmbiguousGraspReturnPlan:
    return _blocked(
        reason,
        applicable=True,
        arm=arm,
        held_instance_id=instance_id,
        grasp_attempt_nonce=nonce,
    )


def _blocked(
    reason: str,
    *,
    applicable: bool,
    arm: str = "",
    held_instance_id: str = "",
    grasp_attempt_nonce: str = "",
) -> AmbiguousGraspReturnPlan:
    return AmbiguousGraspReturnPlan(
        applicable=applicable,
        authorized=False,
        reason=str(reason),
        calls=(),
        arm=arm,
        held_instance_id=held_instance_id,
        grasp_attempt_nonce=grasp_attempt_nonce,
    )


def _reduction(
    states: dict[str, Any],
    *,
    reason: str,
) -> AmbiguousGraspReturnReduction:
    return AmbiguousGraspReturnReduction(
        manipulation_state=states,
        physical_return_succeeded=False,
        fresh_snapshot_confirmed=False,
        attempt_completed=False,
        reason=reason,
    )


def _step_count(distance: float, *, maximum_step: float) -> int:
    return max(1, int(math.ceil(max(0.0, distance - 1e-12) / maximum_step)))


def _single_arm(value: Any) -> str | None:
    arm = str(value or "").strip().lower()
    return arm if arm in {"left", "right"} else None


def _vector(value: Any, *, length: int) -> list[float] | None:
    if isinstance(value, (str, bytes, bytearray, dict)):
        return None
    try:
        if len(value) != length:
            return None
        result = [float(value[index]) for index in range(length)]
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _validated_quaternion(
    value: Any,
    *,
    norm_tolerance: float,
) -> list[float] | None:
    quaternion = _vector(value, length=4)
    if quaternion is None:
        return None
    norm = _norm(quaternion)
    if norm <= 1e-9 or abs(norm - 1.0) > norm_tolerance:
        return None
    return quaternion


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nonnegative_integer(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _positive_integer(value: Any) -> int | None:
    result = _nonnegative_integer(value)
    return result if result is not None and result > 0 else None


def _normalized(value: list[float]) -> list[float]:
    norm = _norm(value)
    return [item / norm for item in value]


def _add(first: list[float], second: list[float]) -> list[float]:
    return [first[index] + second[index] for index in range(3)]


def _subtract(first: list[float], second: list[float]) -> list[float]:
    return [first[index] - second[index] for index in range(3)]


def _scale(value: list[float], factor: float) -> list[float]:
    return [item * factor for item in value]


def _dot(first: list[float], second: list[float]) -> float:
    return sum(first[index] * second[index] for index in range(len(first)))


def _norm(value: list[float]) -> float:
    return math.sqrt(_dot(value, value))


def _distance(first: list[float], second: list[float]) -> float:
    return _norm(_subtract(first, second))
