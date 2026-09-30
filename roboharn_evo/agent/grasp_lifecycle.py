from __future__ import annotations

import math
from typing import Any
from roboharn_evo.agent.gripper_state import gripper_command_state

from roboharn_evo.agent.operation_candidates import (
    valid_pending_held_object_to_tcp_attachment,
)
from roboharn_evo.agent.recovery.tool_specs import RecoveryToolCall


_GRASP_CLOSE_BOUNDARY_MARKER = "_runtime_grasp_close_boundary"
_GRASP_DIAGNOSTIC_LIFT_MARKER = "_runtime_grasp_diagnostic_lift"
_GRASP_DIAGNOSTIC_MOTION_MARKER = "_runtime_grasp_diagnostic_motion"
_MIN_DIAGNOSTIC_MOTION_M = 0.01
_MAX_DIAGNOSTIC_MOTION_M = 0.03
_MAX_DIAGNOSTIC_STEP_M = 0.01
_MAX_DIAGNOSTIC_STEPS = 3
_INGRESS_CORRIDOR_RADIUS_M = 0.015
_INGRESS_ENDPOINT_TOLERANCE_M = 0.01
_PHYSICAL_TOOLS = {
    "contact_displace",
    "move_ee_to_pose",
    "move_ee_to_grounded_instance",
    "move_to_home",
    "retreat_arm",
    "lift_ee",
    "open_gripper",
    "close_gripper",
    "safe_reset_posture",
}

_STRICT_GRASP_TRANSPORT_POLICY = "strict"
_EVIDENCE_ONLY_GRASP_TRANSPORT_POLICY = "evidence_only"


def grasp_attempt_metadata(
    *,
    arm: Any,
    held_instance_id: Any,
    grasp_candidate_id: Any,
    env_step: Any,
) -> dict[str, Any]:
    """Return immutable identity for one physical close transaction."""

    normalized_arm = str(arm or "").strip().lower()
    instance_id = str(held_instance_id or "").strip()
    candidate_id = str(grasp_candidate_id or "").strip()
    try:
        step = int(env_step)
    except (TypeError, ValueError):
        return {}
    if (
        normalized_arm not in {"left", "right"}
        or not instance_id
        or not candidate_id
        or step < 0
    ):
        return {}
    nonce = f"{normalized_arm}:{instance_id}:{candidate_id}:{step}"
    return {
        "grasp_attempt_step": step,
        "grasp_attempt_nonce": nonce,
    }


def bind_pending_grasp_attachment(
    attachment: Any,
    *,
    grasp_candidate_id: Any,
    grasp_attempt: Any,
) -> dict[str, Any] | None:
    """Bind close-time geometry to one attempt without granting authority."""

    if not isinstance(attachment, dict) or not isinstance(
        grasp_attempt,
        dict,
    ):
        return None
    candidate_id = str(grasp_candidate_id or "").strip()
    nonce = str(grasp_attempt.get("grasp_attempt_nonce", "") or "").strip()
    try:
        step = int(grasp_attempt.get("grasp_attempt_step"))
    except (TypeError, ValueError):
        return None
    if not candidate_id or not nonce:
        return None
    return {
        **attachment,
        "grasp_candidate_id": candidate_id,
        "grasp_attempt_nonce": nonce,
        "capture_step": step,
        "source": "pending_lift_verification",
    }


def promote_verified_grasp_attachment(
    arm_state: Any,
    *,
    validation: Any = None,
) -> dict[str, Any] | None:
    """Atomically promote exact pending geometry after runtime verification."""

    if not isinstance(arm_state, dict):
        return None
    attachment = arm_state.get("held_object_to_tcp_attachment")
    if not valid_pending_held_object_to_tcp_attachment(
        attachment,
        grasp_candidate_id=arm_state.get("grasp_candidate_id"),
        grasp_attempt_nonce=arm_state.get("grasp_attempt_nonce"),
        grasp_attempt_step=arm_state.get("grasp_attempt_step"),
    ):
        return None
    single_fixed_camera = bool(
        isinstance(validation, dict)
        and validation.get("single_view_fast_path") is True
    )
    return {
        **dict(attachment),
        "source": (
            "runtime_single_fixed_camera_grasp_motion_verified"
            if single_fixed_camera
            else "runtime_multiview_grasp_motion_verified"
        ),
        "authority": (
            "runtime_single_fixed_camera_grasp_motion"
            if single_fixed_camera
            else "runtime_multiview_grasp_motion"
        ),
    }


def apply_close_time_grasp_transport_policy(
    state: Any,
    *,
    grasp_transport_policy: str = _STRICT_GRASP_TRANSPORT_POLICY,
) -> dict[str, Any]:
    """Apply transport policy without presenting an assumption as proof.

    Strict mode retains the pending grasp transaction.  Evidence-only mode
    authorizes transport from the exact close-time object/TCP geometry while
    keeping ``holding_confirmed`` false and the evidence status explicit.
    """

    current = dict(state) if isinstance(state, dict) else {}
    policy = _normalized_grasp_transport_policy(grasp_transport_policy)
    if policy == _STRICT_GRASP_TRANSPORT_POLICY:
        return current
    attachment = current.get("held_object_to_tcp_attachment")
    if valid_pending_held_object_to_tcp_attachment(
        attachment,
        grasp_candidate_id=current.get("grasp_candidate_id"),
        grasp_attempt_nonce=current.get("grasp_attempt_nonce"),
        grasp_attempt_step=current.get("grasp_attempt_step"),
    ):
        provisional_attachment = {
            **dict(attachment),
            "source": "runtime_close_time_grasp_attachment_assumption",
            "authority": "runtime_evidence_only_grasp_transport_policy",
        }
    else:
        provisional_attachment = None
    return {
        **current,
        "phase": "holding_provisional",
        "holding_confirmed": False,
        "transport_authorized": provisional_attachment is not None,
        "grasp_transport_policy": _EVIDENCE_ONLY_GRASP_TRANSPORT_POLICY,
        "attachment_evidence_status": "unknown",
        **(
            {"held_object_to_tcp_attachment": provisional_attachment}
            if provisional_attachment is not None
            else {}
        ),
        "evidence": (
            "close_succeeded;attachment_assumed_not_confirmed;"
            + (
                "transport_authorized_by_evidence_only_policy"
                if provisional_attachment is not None
                else "transport_geometry_unavailable"
            )
        ),
    }


def stage_grasp_verification_boundary(
    calls: list[RecoveryToolCall],
    *,
    manipulation_state: Any,
    robot_state: Any = None,
    grasp_transport_policy: str = _STRICT_GRASP_TRANSPORT_POLICY,
    release_guard_enabled: bool = True,
) -> list[RecoveryToolCall]:
    """Separate close-time capture from runtime diagnostic motion.

    Runtime state and the object/TCP attachment are committed after the close
    batch.  A later, bounded reverse-ingress move can then compare two
    observations belonging to the exact same grasp attempt.  ``lift_ee``
    remains a planner-facing compatibility request, but the physical motion is
    generated exclusively from the recorded grasp/approach geometry and the
    current robot pose.
    """

    original: list[RecoveryToolCall] = []
    for call in calls:
        args = dict(call.args or {})
        args.pop(_GRASP_CLOSE_BOUNDARY_MARKER, None)
        args.pop(_GRASP_DIAGNOSTIC_LIFT_MARKER, None)
        args.pop(_GRASP_DIAGNOSTIC_MOTION_MARKER, None)
        original.append(RecoveryToolCall(tool_name=call.tool_name, args=args))
    if (
        _normalized_grasp_transport_policy(grasp_transport_policy)
        == _EVIDENCE_ONLY_GRASP_TRANSPORT_POLICY
    ):
        return original
    if any(
        bool((call.args or {}).get("_runtime_failed_grasp_clearance"))
        for call in original
    ):
        return original
    states = manipulation_state if isinstance(manipulation_state, dict) else {}
    pending_states = {
        arm: state
        for arm, state in states.items()
        if arm in {"left", "right"}
        and isinstance(state, dict)
        and str(state.get("phase", "") or "").strip().lower()
        == "grasp_candidate"
        and _portable_grasp_state(state)
        and state.get("holding_confirmed") is not True
    }
    pending_arms = set(pending_states)
    released_arms: set[str] = set()
    pending_physical: list[RecoveryToolCall] = []
    for call in original:
        call_arm = _single_arm((call.args or {}).get("arm"))
        if call_arm not in pending_arms:
            continue
        if not release_guard_enabled and call.tool_name == "open_gripper":
            # The explicit open ends the prior pending attachment for staging
            # purposes.  It and later same-arm motion are not grasp-evidence
            # gated, while unrelated earlier calls retain their own guards.
            released_arms.add(call_arm)
            continue
        if (
            call.tool_name in _PHYSICAL_TOOLS
            and call_arm not in released_arms
        ):
            pending_physical.append(call)
    pending_lifts = [
        call
        for call in pending_physical
        if call.tool_name == "lift_ee"
    ]
    if pending_lifts:
        if any(
            isinstance(pending_states[arm].get("diagnostic_lift_evidence"), dict)
            for arm in {
                _single_arm((call.args or {}).get("arm"))
                for call in pending_lifts
            }
            if arm in pending_states
        ):
            return [
                _observation(
                    "blocked repeated physical lift while the prior grasp "
                    "verification remains unresolved"
                )
            ]
        if len(pending_lifts) != 1:
            return [_observation("ambiguous pending-grasp diagnostic lift")]
        arm = _single_arm((pending_lifts[0].args or {}).get("arm"))
        assert arm is not None
        diagnostic_motion = _reverse_ingress_diagnostic_motion(
            pending_lifts[0],
            state=pending_states[arm],
            robot_state=robot_state,
        )
        if diagnostic_motion is None:
            return [
                _observation(
                    "blocked diagnostic motion because current robot pose or "
                    "recorded grasp-ingress geometry is unavailable or unsafe"
                )
            ]
        return [
            diagnostic_motion,
            _observation(
                "verify exact grasp attachment after the bounded reverse-"
                "ingress diagnostic motion",
                post_action_feedback=True,
            ),
        ]
    if pending_physical:
        runtime_failed_release = all(
            call.tool_name == "open_gripper"
            and _runtime_validation(
                pending_states[
                    _single_arm((call.args or {}).get("arm"))
                ]
            ).get("verified")
            is False
            for call in pending_physical
        )
        if not runtime_failed_release:
            return [
                _observation(
                    "blocked same-arm physical motion while grasp attachment "
                    "is pending; request one bounded diagnostic motion or "
                    "complete runtime failed-grasp clearance"
                )
            ]

    for close_index, close_call in enumerate(original):
        if close_call.tool_name != "close_gripper":
            continue
        arm = _single_arm((close_call.args or {}).get("arm"))
        if arm is None:
            continue
        preceding_observation = next(
            (
                index
                for index, call in enumerate(original[:close_index])
                if call.tool_name == "reobserve_scene"
            ),
            None,
        )
        if preceding_observation is not None:
            # Dispatch stops at an observation.  Never move a later close in
            # front of that boundary while staging the batch.
            return original[: preceding_observation + 1]
        setup_index = _preceding_grasp_setup_index(
            original,
            close_index=close_index,
            arm=arm,
        )
        if setup_index is None:
            if _has_prior_grasp_setup(
                original,
                close_index=close_index,
                arm=arm,
            ):
                # A same-arm physical action invalidated the earlier setup.
                # Execute only the safe prefix and require a fresh grounded
                # setup on the next observation before closing.
                return [
                    *original[:close_index],
                    _observation(
                        "blocked close_gripper because same-arm motion "
                        "invalidated the preceding grounded grasp setup"
                    ),
                ]
            continue
        prefix = list(original[: close_index + 1])
        marked_close_args = dict(prefix[-1].args or {})
        marked_close_args[_GRASP_CLOSE_BOUNDARY_MARKER] = True
        prefix[-1] = RecoveryToolCall(
            tool_name="close_gripper",
            args=marked_close_args,
        )
        return [
            *prefix,
            _observation(
                "commit close-time grasp identity and TCP geometry before "
                "diagnostic motion",
                post_action_feedback=True,
            ),
        ]
    return original


def apply_grasp_close_boundary_effect(
    effect: Any,
    *,
    calls: list[RecoveryToolCall],
    results: list[Any] | None = None,
    grasp_transport_policy: str = _STRICT_GRASP_TRANSPORT_POLICY,
) -> dict[str, Any]:
    result = dict(effect) if isinstance(effect, dict) else {}
    if (
        _normalized_grasp_transport_policy(grasp_transport_policy)
        == _EVIDENCE_ONLY_GRASP_TRANSPORT_POLICY
    ):
        physical_indexes = [
            index
            for index, call in enumerate(calls)
            if call.tool_name != "reobserve_scene"
        ]
        if not physical_indexes:
            return result
        close_index = physical_indexes[-1]
        close_call = calls[close_index]
        arm = _single_arm((close_call.args or {}).get("arm"))
        setup_index = (
            _preceding_grasp_setup_index(
                calls,
                close_index=close_index,
                arm=arm,
            )
            if arm is not None
            else None
        )
        if (
            close_call.tool_name != "close_gripper"
            or arm is None
            or setup_index is None
        ):
            return result
        if results is not None:
            setup_result = (
                results[setup_index]
                if setup_index < len(results)
                else None
            )
            close_result = (
                results[close_index]
                if close_index < len(results)
                else None
            )
            setup_details = (
                getattr(setup_result, "details", {}) or {}
                if setup_result is not None
                else {}
            )
            if (
                setup_result is None
                or not bool(getattr(setup_result, "success", False))
                or setup_details.get("target_reached") is False
                or close_result is None
                or not bool(getattr(close_result, "success", False))
            ):
                return result
        result["evidence_only_model_effect"] = {
            key: result.get(key)
            for key in (
                "effect_verified",
                "effect_type",
                "failure_reason",
                "evidence_summary",
            )
            if key in result
        }
        result.update(
            {
                "effect_verified": "unverified",
                "effect_type": "grasp",
                "subtask_status": "in_progress",
                "recommended_control": "continue",
                "failure_reason": "",
                "next_constraint": (
                    "Continue the task-directed action without mandatory "
                    "diagnostic motion; attachment remains unconfirmed."
                ),
                "memory_update": (
                    "Close succeeded at an exact grounded grasp; transport "
                    "is policy-authorized while attachment evidence remains "
                    "unknown."
                ),
                "authority": (
                    "runtime_evidence_only_grasp_transport_policy"
                ),
            }
        )
        return result
    if not any(
        call.tool_name == "close_gripper"
        and (call.args or {}).get(_GRASP_CLOSE_BOUNDARY_MARKER) is True
        for call in calls
    ):
        return result
    result.update(
        {
            "effect_verified": "unverified",
            "effect_type": "grasp",
            "subtask_status": "in_progress",
            "recommended_control": "retry",
            "next_constraint": (
                "Run one bounded reverse-ingress diagnostic motion with a "
                "fresh observation; do not transport or release before "
                "attachment verification."
            ),
            "memory_update": (
                "Close-time grasp transaction captured; attachment remains "
                "pending deterministic motion verification."
            ),
            "authority": "runtime_grasp_verification_boundary",
        }
    )
    return result


def _normalized_grasp_transport_policy(value: Any) -> str:
    policy = str(value or "").strip().lower().replace("-", "_")
    if policy not in {
        _STRICT_GRASP_TRANSPORT_POLICY,
        _EVIDENCE_ONLY_GRASP_TRANSPORT_POLICY,
    }:
        raise ValueError(
            "grasp_transport_policy must be 'strict' or 'evidence_only'"
        )
    return policy


def is_runtime_grasp_diagnostic_lift_call(
    call: RecoveryToolCall,
) -> bool:
    """Compatibility predicate for either legacy or current runtime probes."""

    return bool(
        (
            call.tool_name == "lift_ee"
            and (call.args or {}).get(_GRASP_DIAGNOSTIC_LIFT_MARKER) is True
        )
        or (
            call.tool_name == "move_ee_to_pose"
            and (call.args or {}).get(_GRASP_DIAGNOSTIC_MOTION_MARKER) is True
        )
    )


def merge_pending_grasp_validation(
    manipulation_state: Any,
    *,
    validation: Any,
    env_step: Any,
) -> dict[str, Any]:
    """Persist deferred negative/ambiguous evidence on the exact attempt."""

    states = manipulation_state if isinstance(manipulation_state, dict) else {}
    updated = {
        str(arm): dict(state)
        for arm, state in states.items()
        if isinstance(state, dict)
    }
    if not isinstance(validation, dict) or validation.get("applicable") is not True:
        return updated
    arm = _single_arm(validation.get("arm"))
    state = updated.get(arm) if arm is not None else None
    if not isinstance(state, dict):
        return updated
    if not (
        str(state.get("phase", "") or "").strip().lower()
        == "grasp_candidate"
        and str(validation.get("held_instance_id", "") or "").strip()
        == str(state.get("held_instance_id", "") or "").strip()
        and str(validation.get("grasp_candidate_id", "") or "").strip()
        == str(state.get("grasp_candidate_id", "") or "").strip()
        and str(validation.get("grasp_attempt_nonce", "") or "").strip()
        == str(state.get("grasp_attempt_nonce", "") or "").strip()
    ):
        return updated
    if validation.get("verified") is True:
        return updated
    diagnostic = state.get("diagnostic_lift_evidence")
    if not isinstance(diagnostic, dict):
        return updated
    try:
        step = int(env_step)
    except (TypeError, ValueError):
        step = -1
    updated[arm] = {
        **state,
        "diagnostic_lift_evidence": {
            **diagnostic,
            "runtime_grasp_validation": dict(validation),
            "track_status": (
                "observed_by_multiple_views"
                if validation.get("verified") is False
                else "multiview_motion_ambiguous"
            ),
            "track_stability": (
                "stationary_during_lift"
                if validation.get("verified") is False
                else "attachment_not_proven"
            ),
            "negative_attachment_evidence": (
                validation.get("verified") is False
            ),
        },
        "holding_confirmed": False,
        "transport_authorized": False,
        "updated_step": step,
    }
    return updated


def _preceding_grasp_setup_index(
    calls: list[RecoveryToolCall],
    *,
    close_index: int,
    arm: str,
) -> int | None:
    for index in range(close_index - 1, -1, -1):
        call = calls[index]
        if call.tool_name == "reobserve_scene":
            return None
        if not _physical_call_affects_arm(call, arm=arm):
            continue
        if _is_grasp_setup(call, arm=arm):
            return index
        return None
    return None


def _has_prior_grasp_setup(
    calls: list[RecoveryToolCall],
    *,
    close_index: int,
    arm: str,
) -> bool:
    return any(
        _is_grasp_setup(call, arm=arm)
        for call in calls[:close_index]
    )


def _is_grasp_setup(call: RecoveryToolCall, *, arm: str) -> bool:
    if (
        call.tool_name != "move_ee_to_grounded_instance"
        or _single_arm((call.args or {}).get("arm")) != arm
    ):
        return False
    args = dict(call.args or {})
    point_key = str(args.get("point_key", "") or "").strip().lower()
    action_mode = str(
        args.get("_operation_action_mode", args.get("action_mode", ""))
        or ""
    ).strip().lower()
    return action_mode == "grasp" and point_key in {
        "grasp",
        "grasp_world",
        "grasp_world_m",
        "contact",
        "contact_world",
        "contact_world_m",
    }


def _physical_call_affects_arm(
    call: RecoveryToolCall,
    *,
    arm: str,
) -> bool:
    if call.tool_name not in _PHYSICAL_TOOLS:
        return False
    raw_arm = str((call.args or {}).get("arm", "") or "").strip().lower()
    aliases = {
        "left_arm": "left",
        "right_arm": "right",
        "all": "both",
        "dual": "both",
    }
    normalized = aliases.get(raw_arm, raw_arm)
    if normalized in {"left", "right"}:
        return normalized == arm
    # A dual-arm or malformed physical call is not transparent to either
    # arm's close transaction; fail closed rather than reusing stale setup.
    return True


def _reverse_ingress_diagnostic_motion(
    request: RecoveryToolCall,
    *,
    state: dict[str, Any],
    robot_state: Any,
) -> RecoveryToolCall | None:
    """Build one safe diagnostic move without trusting planner coordinates."""

    held_instance_id = str(
        state.get("held_instance_id", "") or ""
    ).strip()
    if (
        not held_instance_id
        or not valid_pending_held_object_to_tcp_attachment(
            state.get("held_object_to_tcp_attachment"),
            grasp_candidate_id=state.get("grasp_candidate_id"),
            grasp_attempt_nonce=state.get("grasp_attempt_nonce"),
            grasp_attempt_step=state.get("grasp_attempt_step"),
        )
    ):
        return None
    arm = _single_arm((request.args or {}).get("arm"))
    robots = robot_state if isinstance(robot_state, dict) else {}
    arm_robot = robots.get(arm) if arm is not None else None
    if not isinstance(arm_robot, dict):
        return None
    current = _finite_vector(arm_robot.get("xyz"), length=3)
    quaternion = _finite_vector(arm_robot.get("quat_wxyz"), length=4)
    grasp = _finite_vector(state.get("grasp_ee_target_world_m"), length=3)
    approach = _finite_vector(state.get("grasp_approach_world_m"), length=3)
    if (
        current is None
        or quaternion is None
        or gripper_command_state(arm_robot) != "closed"
        or grasp is None
        or approach is None
    ):
        return None

    reverse_ingress = _subtract(approach, grasp)
    ingress_length = _norm(reverse_ingress)
    if ingress_length <= 1e-8:
        return None
    unit = [component / ingress_length for component in reverse_ingress]
    relative = _subtract(current, grasp)
    progress = _dot(relative, unit)
    if (
        progress < -_INGRESS_ENDPOINT_TOLERANCE_M
        or progress > ingress_length + _INGRESS_ENDPOINT_TOLERANCE_M
    ):
        return None
    nearest = _add(grasp, _scale(unit, progress))
    if _norm(_subtract(current, nearest)) > _INGRESS_CORRIDOR_RADIUS_M:
        return None
    remaining = max(0.0, ingress_length - progress)
    if remaining < _MIN_DIAGNOSTIC_MOTION_M:
        return None

    distance = min(
        remaining,
        _MAX_DIAGNOSTIC_MOTION_M,
    )
    required_steps = int(math.ceil(distance / _MAX_DIAGNOSTIC_STEP_M))
    steps = min(_MAX_DIAGNOSTIC_STEPS, max(1, required_steps))
    target = _add(current, _scale(unit, distance))
    return RecoveryToolCall(
        tool_name="move_ee_to_pose",
        args={
            "arm": arm,
            "target_pose": [*target, *quaternion],
            "max_translation": round(distance / steps, 6),
            "steps": steps,
            _GRASP_DIAGNOSTIC_MOTION_MARKER: True,
            "_guard_reason": (
                "runtime-generated reverse-ingress diagnostic motion for "
                "pending grasp attachment verification"
            ),
        },
    )


def _observation(reason: str, *, post_action_feedback: bool = False) -> RecoveryToolCall:
    return RecoveryToolCall(
        tool_name="reobserve_scene",
        args={"_guard_reason": reason, "_post_action_feedback": post_action_feedback},
    )


def _single_arm(value: Any) -> str | None:
    arm = str(value or "").strip().lower()
    return arm if arm in {"left", "right"} else None


def _runtime_validation(state: dict[str, Any]) -> dict[str, Any]:
    diagnostic = state.get("diagnostic_lift_evidence")
    validation = (
        diagnostic.get("runtime_grasp_validation")
        if isinstance(diagnostic, dict)
        else None
    )
    return dict(validation) if isinstance(validation, dict) else {}


def _portable_grasp_state(state: dict[str, Any]) -> bool:
    action_mode = str(state.get("operation_action_mode", "") or "").strip().lower()
    # Older persisted pending states predate ``operation_action_mode``.  They
    # are retained for compatibility, while an explicit contact/place mode is
    # never granted portable-grasp diagnostic authority.
    return not action_mode or action_mode == "grasp"


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_vector(value: Any, *, length: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    result = [_finite_float(component) for component in value]
    if any(component is None for component in result):
        return None
    return [float(component) for component in result if component is not None]


def _subtract(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _add(left: list[float], right: list[float]) -> list[float]:
    return [a + b for a, b in zip(left, right)]


def _scale(vector: list[float], scalar: float) -> list[float]:
    return [component * scalar for component in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vector: list[float]) -> float:
    return math.sqrt(_dot(vector, vector))
