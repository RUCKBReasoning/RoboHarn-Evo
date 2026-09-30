from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from roboharn_evo.agent.gripper_state import gripper_command_state

from .tool_specs import RecoveryToolCall


@dataclass(frozen=True, slots=True)
class FailedGraspClearanceLimits:
    ingress_corridor_radius_m: float = 0.015
    endpoint_tolerance_m: float = 0.01
    minimum_remaining_clearance_m: float = 0.01
    maximum_clearance_m: float = 0.05
    maximum_step_m: float = 0.02
    maximum_steps: int = 3


DEFAULT_FAILED_GRASP_CLEARANCE_LIMITS = FailedGraspClearanceLimits()
_CLEARANCE_MARKER = "_runtime_failed_grasp_clearance"
_CLEARANCE_PHASE_RELEASE = "release"
_CLEARANCE_PHASE_MOVE = "move"
_CLEARANCE_PHASE_OBSERVE = "observe"
_CLEARANCE_PHASE_OBSERVE_REACHED = "observe_reached"
_CLEARANCE_PENDING_PHASE = "failed_grasp_clearance_pending"


def stage_failed_grasp_clearance(
    calls: list[RecoveryToolCall],
    *,
    manipulation_state: Any,
    robot_state: Any,
    grasp_transport_policy: str = "strict",
    release_guard_enabled: bool = True,
    limits: FailedGraspClearanceLimits = (
        DEFAULT_FAILED_GRASP_CLEARANCE_LIMITS
    ),
) -> list[RecoveryToolCall]:
    """Normalize release after a runtime-proven failed grasp.

    Any planner batch containing an ``open_gripper`` for a pending grasp is
    runtime-owned.  Only exact, two-view negative motion evidence may release
    the gripper, and the resulting batch contains no unrelated physical
    actions.  Once release starts, a persisted clearance obligation is retried
    until the recorded reverse-ingress move and a fresh observation succeed.
    """

    policy = str(grasp_transport_policy or "").strip().lower().replace(
        "-", "_"
    )
    if policy not in {"strict", "evidence_only"}:
        raise ValueError(
            "grasp_transport_policy must be 'strict' or 'evidence_only'"
        )
    original = [
        RecoveryToolCall(
            tool_name=call.tool_name,
            args={
                key: value
                for key, value in dict(call.args or {}).items()
                if key != _CLEARANCE_MARKER
            },
        )
        for call in calls
    ]
    if not release_guard_enabled:
        # An explicit release is model-authoritative in this mode.  Do not
        # replace it with, or inject, a runtime-owned release/clearance plan.
        return original
    if policy == "evidence_only":
        # Keep visual failure evidence available to the planner, but never
        # turn it into an automatic open/retreat/observe transaction.
        return original
    states = manipulation_state if isinstance(manipulation_state, dict) else {}
    clearance_pending = [
        (arm, state)
        for arm, state in states.items()
        if arm in {"left", "right"}
        and isinstance(state, dict)
        and str(state.get("phase", "") or "").strip().lower()
        == _CLEARANCE_PENDING_PHASE
    ]
    if clearance_pending:
        if len(clearance_pending) != 1:
            return _blocked_observation(
                "blocked recovery because more than one failed-grasp "
                "clearance obligation is active"
            )
        arm, state = clearance_pending[0]
        return _stage_pending_clearance(
            arm=arm,
            state=state,
            robot_state=robot_state,
            limits=limits,
        )

    open_calls = [call for call in original if call.tool_name == "open_gripper"]
    if not open_calls:
        return original
    pending_states = {
        arm: state
        for arm, state in states.items()
        if arm in {"left", "right"} and _pending_grasp_state(state)
    }
    if not pending_states:
        return original

    relevant: list[tuple[RecoveryToolCall, str, dict[str, Any]]] = []
    for call in open_calls:
        arm = _single_arm((call.args or {}).get("arm"))
        if arm is None:
            return _blocked_observation(
                "blocked open_gripper because a pending grasp requires one "
                "exact physical arm"
            )
        state = pending_states.get(arm)
        if isinstance(state, dict):
            relevant.append((call, arm, state))
    if not relevant:
        return original
    if len(relevant) != 1:
        return _blocked_observation(
            "blocked open_gripper because failed-grasp release must contain "
            "one exact pending transaction"
        )

    open_call, arm, state = relevant[0]
    validation = _matched_grasp_validation(state, arm=arm)
    if validation is None or validation.get("verified") is not False:
        return _blocked_observation(
            "blocked open_gripper because the exact grasp transaction lacks "
            "a runtime-proven two-view failure; keep the gripper closed and "
            "collect another fresh observation"
        )
    held_instance_id = str(state.get("held_instance_id", "") or "").strip()
    release_ref = str(
        (open_call.args or {}).get("release_held_instance_id", "") or ""
    ).strip()
    if not release_ref or release_ref != held_instance_id:
        return _blocked_observation(
            "blocked open_gripper because release_held_instance_id does not "
            "exactly match the runtime failed-grasp transaction"
        )

    clearance = _clearance_call(
        arm=arm,
        state=state,
        robot_state=robot_state,
        require_closed_gripper=True,
        limits=limits,
    )
    if clearance is None:
        return _blocked_observation(
            "blocked open_gripper because runtime could not prove a bounded "
            "reverse path inside the recorded grasp-ingress corridor"
        )
    release = _failed_release_call(
        arm=arm,
        held_instance_id=held_instance_id,
    )
    return [release, clearance, _clearance_observation(arm)]


def apply_failed_grasp_clearance_results(
    manipulation_state: Any,
    *,
    calls: list[RecoveryToolCall],
    results: list[Any],
    env_step: int,
    fresh_physical_snapshot_satisfies_observation: bool = False,
) -> dict[str, Any]:
    """Reduce runtime-only clearance calls into a persistent obligation.

    A successful gripper release never deletes the grasp transaction by
    itself.  State is removed only after the reverse-ingress target is observed
    reached and a subsequent observation succeeds.  A failed or partial move
    therefore remains retryable on the next control turn.
    """

    states = manipulation_state if isinstance(manipulation_state, dict) else {}
    updated = {
        str(arm): dict(state)
        for arm, state in states.items()
        if isinstance(state, dict)
    }
    for index, call in enumerate(calls):
        marker = str((call.args or {}).get(_CLEARANCE_MARKER, "") or "").strip()
        if marker not in {
            _CLEARANCE_PHASE_RELEASE,
            _CLEARANCE_PHASE_MOVE,
            _CLEARANCE_PHASE_OBSERVE,
            _CLEARANCE_PHASE_OBSERVE_REACHED,
        }:
            continue
        arm = _single_arm((call.args or {}).get("arm"))
        if arm is None:
            continue
        state = updated.get(arm)
        if not isinstance(state, dict):
            continue
        result = results[index] if index < len(results) else None
        success = bool(result is not None and getattr(result, "success", False))
        details = dict(getattr(result, "details", {}) or {})
        if marker == _CLEARANCE_PHASE_RELEASE:
            if not success:
                continue
            move_call = next(
                (
                    candidate
                    for candidate in calls[index + 1 :]
                    if str(
                        (candidate.args or {}).get(_CLEARANCE_MARKER, "")
                        or ""
                    ).strip()
                    == _CLEARANCE_PHASE_MOVE
                    and _single_arm((candidate.args or {}).get("arm")) == arm
                ),
                None,
            )
            updated[arm] = {
                **state,
                "phase": _CLEARANCE_PENDING_PHASE,
                "holding_confirmed": False,
                "transport_authorized": False,
                "failed_grasp_clearance_status": "gripper_opened",
                "failed_grasp_clearance_move_args": (
                    _public_clearance_args(move_call.args)
                    if move_call is not None
                    else {}
                ),
                "updated_step": int(env_step),
                "evidence": (
                    "runtime_proven_failed_grasp_released;"
                    "reverse_ingress_clearance_required"
                ),
            }
            continue
        if (
            str(state.get("phase", "") or "").strip().lower()
            != _CLEARANCE_PENDING_PHASE
        ):
            continue
        if marker == _CLEARANCE_PHASE_MOVE:
            reached = success and details.get("target_reached") is True
            if reached and fresh_physical_snapshot_satisfies_observation:
                # RMBench recovery motions return the post-motion snapshot.
                # During the explicit-reobserve ablation this is the fresh
                # clearance observation, so retaining an observation-only
                # obligation would create an artificial deadlock.
                updated.pop(arm, None)
                continue
            updated[arm] = {
                **state,
                "failed_grasp_clearance_status": (
                    "clearance_reached" if reached else "gripper_opened"
                ),
                "updated_step": int(env_step),
                "evidence": (
                    "reverse_ingress_clearance_reached;fresh_observation_required"
                    if reached
                    else "reverse_ingress_clearance_incomplete;retry_required"
                ),
            }
            continue
        if (
            marker
            in {
                _CLEARANCE_PHASE_OBSERVE,
                _CLEARANCE_PHASE_OBSERVE_REACHED,
            }
            and success
            and (
                marker == _CLEARANCE_PHASE_OBSERVE_REACHED
                or str(
                    state.get("failed_grasp_clearance_status", "") or ""
                ).strip().lower()
                == "clearance_reached"
            )
        ):
            updated.pop(arm, None)
    return updated


def is_runtime_failed_grasp_clearance_call(call: RecoveryToolCall) -> bool:
    return str(
        (call.args or {}).get(_CLEARANCE_MARKER, "") or ""
    ).strip() in {
        _CLEARANCE_PHASE_RELEASE,
        _CLEARANCE_PHASE_MOVE,
        _CLEARANCE_PHASE_OBSERVE,
        _CLEARANCE_PHASE_OBSERVE_REACHED,
    }


def _stage_pending_clearance(
    *,
    arm: str,
    state: dict[str, Any],
    robot_state: Any,
    limits: FailedGraspClearanceLimits,
) -> list[RecoveryToolCall]:
    if (
        str(state.get("failed_grasp_clearance_status", "") or "")
        .strip()
        .lower()
        == "clearance_reached"
    ):
        return [_clearance_observation(arm)]
    robots = robot_state if isinstance(robot_state, dict) else {}
    arm_robot = robots.get(arm)
    command = gripper_command_state(arm_robot)
    if command not in {"closed", "open"}:
        return _blocked_observation(
            "failed-grasp clearance is pending, but current gripper state is "
            "unknown; keep the arm fixed and refresh robot state"
        )
    if command == "closed":
        held_instance_id = str(
            state.get("held_instance_id", "") or ""
        ).strip()
        if not held_instance_id:
            return _blocked_observation(
                "failed-grasp clearance lost its exact held-instance identity"
            )
        return [
            _failed_release_call(
                arm=arm,
                held_instance_id=held_instance_id,
            ),
            _clearance_observation(arm),
        ]
    clearance = _clearance_call(
        arm=arm,
        state=state,
        robot_state=robot_state,
        require_closed_gripper=False,
        limits=limits,
    )
    if clearance is None:
        current = (
            _xyz(arm_robot.get("xyz"))
            if isinstance(arm_robot, dict)
            else None
        )
        approach = _xyz(state.get("grasp_approach_world_m"))
        if (
            current is not None
            and approach is not None
            and _distance(current, approach) <= limits.endpoint_tolerance_m
        ):
            return [_clearance_observation(arm, endpoint_reached=True)]
        return _blocked_observation(
            "failed-grasp release already occurred, but a safe continuation "
            "inside the recorded ingress corridor is not currently provable; "
            "keep the arm fixed and refresh robot state"
        )
    return [clearance, _clearance_observation(arm)]


def _failed_release_call(
    *,
    arm: str,
    held_instance_id: str,
) -> RecoveryToolCall:
    return RecoveryToolCall(
        tool_name="open_gripper",
        args={
            "arm": arm,
            "release_held_instance_id": held_instance_id,
            _CLEARANCE_MARKER: _CLEARANCE_PHASE_RELEASE,
            "_guard_reason": (
                "runtime multiview motion evidence proved this exact grasp "
                "failed; release before reversing the recorded ingress"
            ),
        },
    )


def _clearance_call(
    *,
    arm: str,
    state: dict[str, Any],
    robot_state: Any,
    require_closed_gripper: bool,
    limits: FailedGraspClearanceLimits,
) -> RecoveryToolCall | None:
    robots = robot_state if isinstance(robot_state, dict) else {}
    arm_robot = robots.get(arm)
    if not isinstance(arm_robot, dict):
        return None
    command = gripper_command_state(arm_robot)
    current = _xyz(arm_robot.get("xyz"))
    quaternion = _quat(arm_robot.get("quat_wxyz"))
    grasp = _xyz(state.get("grasp_ee_target_world_m"))
    approach = _xyz(state.get("grasp_approach_world_m"))
    if (
        command != ("closed" if require_closed_gripper else "open")
        or current is None
        or quaternion is None
        or grasp is None
        or approach is None
    ):
        return None
    retreat = _reverse_ingress_target(
        current=current,
        grasp=grasp,
        approach=approach,
        limits=limits,
    )
    if retreat is None:
        return None
    target, distance = retreat
    steps = min(
        limits.maximum_steps,
        max(1, int(math.ceil(distance / limits.maximum_step_m))),
    )
    return RecoveryToolCall(
        tool_name="move_ee_to_pose",
        args={
            "arm": arm,
            "target_pose": [*target, *quaternion],
            "max_translation": round(
                min(distance / steps, limits.maximum_step_m),
                6,
            ),
            "steps": steps,
            _CLEARANCE_MARKER: _CLEARANCE_PHASE_MOVE,
            "_guard_reason": (
                "reverse the recorded grasp-ingress segment after a runtime-"
                "proven failed grasp before any new physical action"
            ),
        },
    )


def _clearance_observation(
    arm: str,
    *,
    endpoint_reached: bool = False,
) -> RecoveryToolCall:
    return RecoveryToolCall(
        tool_name="reobserve_scene",
        args={
            "arm": arm,
            _CLEARANCE_MARKER: (
                _CLEARANCE_PHASE_OBSERVE_REACHED
                if endpoint_reached
                else _CLEARANCE_PHASE_OBSERVE
            ),
            "_guard_reason": (
                "refresh perception after completing failed-grasp clearance"
            ),
            "_post_action_feedback": True,
        },
    )


def _blocked_observation(reason: str) -> list[RecoveryToolCall]:
    return [
        RecoveryToolCall(
            tool_name="reobserve_scene",
            args={"_guard_reason": reason},
        )
    ]


def _public_clearance_args(value: Any) -> dict[str, Any]:
    args = dict(value or {}) if isinstance(value, dict) else {}
    return {
        key: args[key]
        for key in (
            "arm",
            "target_pose",
            "max_translation",
            "steps",
        )
        if key in args
    }


def _pending_grasp_state(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and str(value.get("phase", "") or "").strip().lower()
        == "grasp_candidate"
        and value.get("holding_confirmed") is not True
        and value.get("transport_authorized") is not True
        and str(value.get("held_instance_id", "") or "").strip()
        and str(value.get("grasp_candidate_id", "") or "").strip()
    )


def _matched_grasp_validation(
    value: Any,
    *,
    arm: str,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if (
        str(value.get("phase", "") or "").strip().lower()
        != "grasp_candidate"
        or value.get("holding_confirmed") is True
        or value.get("transport_authorized") is True
        or not str(value.get("held_instance_id", "") or "").strip()
        or not str(value.get("grasp_candidate_id", "") or "").strip()
    ):
        return None
    diagnostic = value.get("diagnostic_lift_evidence")
    validation = (
        diagnostic.get("runtime_grasp_validation")
        if isinstance(diagnostic, dict)
        else None
    )
    if not (
        isinstance(validation, dict)
        and validation.get("applicable") is True
        and str(validation.get("arm", "") or "").strip().lower() == arm
        and str(validation.get("held_instance_id", "") or "").strip()
        == str(value.get("held_instance_id", "") or "").strip()
        and str(validation.get("grasp_candidate_id", "") or "").strip()
        == str(value.get("grasp_candidate_id", "") or "").strip()
        and str(value.get("grasp_attempt_nonce", "") or "").strip()
        and str(validation.get("grasp_attempt_nonce", "") or "").strip()
        == str(value.get("grasp_attempt_nonce", "") or "").strip()
    ):
        return None
    if validation.get("verified") is False:
        try:
            negative_camera_count = int(
                validation.get("negative_camera_count", 0) or 0
            )
        except (TypeError, ValueError):
            negative_camera_count = 0
        if not (
            str(validation.get("failure_kind", "") or "")
            .strip()
            .lower()
            == "object_stationary_during_lift"
            and negative_camera_count >= 2
        ):
            return None
    return dict(validation)


def _reverse_ingress_target(
    *,
    current: list[float],
    grasp: list[float],
    approach: list[float],
    limits: FailedGraspClearanceLimits,
) -> tuple[list[float], float] | None:
    ingress = _subtract(approach, grasp)
    ingress_length = _norm(ingress)
    if ingress_length <= 1e-8:
        return None
    unit = [item / ingress_length for item in ingress]
    relative = _subtract(current, grasp)
    progress = _dot(relative, unit)
    if (
        progress < -limits.endpoint_tolerance_m
        or progress > ingress_length + limits.endpoint_tolerance_m
    ):
        return None
    nearest = _add(grasp, _scale(unit, progress))
    if _distance(current, nearest) > limits.ingress_corridor_radius_m:
        return None
    remaining = max(0.0, ingress_length - progress)
    if remaining <= limits.minimum_remaining_clearance_m:
        return None
    distance = min(limits.maximum_clearance_m, remaining)
    return _add(current, _scale(unit, distance)), distance


def _single_arm(value: Any) -> str | None:
    arm = str(value or "").strip().lower()
    return arm if arm in {"left", "right"} else None


def _xyz(value: Any) -> list[float] | None:
    return _finite_vector(value, length=3)


def _quat(value: Any) -> list[float] | None:
    quaternion = _finite_vector(value, length=4)
    if quaternion is None or _norm(quaternion) <= 1e-8:
        return None
    return quaternion


def _finite_vector(value: Any, *, length: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < length:
        return None
    try:
        result = [float(value[index]) for index in range(length)]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


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
