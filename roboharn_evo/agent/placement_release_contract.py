from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from roboharn_evo.agent.arm_contract import normalize_physical_arm
from roboharn_evo.agent.operation_candidates import (
    manipulation_state_allows_transport,
    place_target_tolerance_m,
    propagate_held_object_world_m,
    validate_place_candidate,
)
from roboharn_evo.agent.recovery.tool_specs import RecoveryToolCall


SUPPORT_CONTACT_RELEASE_SOURCE = (
    "bounded_downward_support_contact_resistance"
)

_IDENTITY_RECOVERY_MIN_TOLERANCE_M = 0.03
_IDENTITY_RECOVERY_MAX_TOLERANCE_M = 0.08
_IDENTITY_RECOVERY_TOLERANCE_SCALE = 2.0
_IDENTITY_RECOVERY_ANCHOR_DEDUP_M = 0.005

_PLACE_STATE_KEYS = (
    "place_target_id",
    "place_candidate_id",
    "held_object_target_world_m",
    "held_extent_m",
    "release_ee_target_world_m",
    "pre_release_validated",
)


@dataclass(frozen=True)
class SupportContactLimits:
    """Existing controller/geometry limits reused by the release contract."""

    no_progress_m: float
    max_contact_command_m: float
    max_support_geometry_drift_m: float
    max_pose_drift_m: float


def release_identity_recovery_anchors(
    arm_state: Any,
) -> list[dict[str, Any]]:
    """Return bounded world anchors for post-release identity recovery.

    The committed release target remains the primary association anchor.  If
    the released object is not there, runtime may still distinguish a real
    placement failure from a missing observation by looking near positions
    that are already part of the same manipulation transaction: the target
    itself and the object's pre-grasp position.  These anchors contain no task
    name, class, color, or fixed workspace coordinate.

    The wider recovery tolerance is used only to identify a unique physical
    observation.  Final placement success continues to use the unchanged,
    tighter ``place_target_tolerance_m`` threshold.
    """

    state = dict(arm_state or {}) if isinstance(arm_state, dict) else {}
    base_tolerance = place_target_tolerance_m(
        state.get("held_extent_m")
    )
    recovery_tolerance = min(
        _IDENTITY_RECOVERY_MAX_TOLERANCE_M,
        max(
            _IDENTITY_RECOVERY_MIN_TOLERANCE_M,
            base_tolerance * _IDENTITY_RECOVERY_TOLERANCE_SCALE,
        ),
    )
    sources = (
        (
            "committed_release_target",
            state.get("held_object_target_world_m"),
        ),
        (
            "pregrasp_observed_position",
            state.get("pregrasp_object_world_m"),
        ),
        (
            "pregrasp_contact_position",
            state.get("pregrasp_object_contact_world_m"),
        ),
    )
    anchors: list[dict[str, Any]] = []
    for source, raw_world in sources:
        world = _xyz(raw_world)
        if world is None:
            continue
        if any(
            math.sqrt(
                sum(
                    (world[index] - existing["world_m"][index]) ** 2
                    for index in range(3)
                )
            )
            <= _IDENTITY_RECOVERY_ANCHOR_DEDUP_M
            for existing in anchors
        ):
            continue
        anchors.append(
            {
                "world_m": world,
                "tolerance_m": recovery_tolerance,
                "source": source,
            }
        )
    return anchors


def create_support_contact_release_evidence(
    *,
    scene_memory: Any,
    held_instance: Any,
    arm: str,
    arm_state: Any,
    robot_arm_state: Any,
    calibration: Any,
    gripper_closed: bool,
    call_args: Any,
    result_details: Any,
    limits: SupportContactLimits,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]] | None:
    held = dict(held_instance or {}) if isinstance(held_instance, dict) else {}
    state = dict(arm_state or {}) if isinstance(arm_state, dict) else {}
    if (
        not gripper_closed
        or not manipulation_state_allows_transport(state)
        or str(held.get("status", "") or "").strip().lower()
        not in {"visible", "tracked"}
        or str(held.get("position_state", "") or "").strip().lower()
        == "motion_uncertain"
    ):
        return None
    metrics = resisted_settle_metrics(
        args=call_args,
        details=result_details,
        limits=limits,
    )
    quality = held.get("quality")
    extent = (
        quality.get("world_extent_m")
        if isinstance(quality, dict)
        else None
    )
    current_object = current_held_object_world_m(
        held_instance=held,
        arm_state=state,
        robot_arm_state=robot_arm_state,
        calibration=calibration,
    )
    details = (
        dict(result_details or {})
        if isinstance(result_details, dict)
        else {}
    )
    release_pose = details.get(
        "observed_pose",
        details.get("executed_pose"),
    )
    if _pose7(release_pose) is None and isinstance(robot_arm_state, dict):
        xyz = _xyz(robot_arm_state.get("xyz"))
        quat = _quat(robot_arm_state.get("quat_wxyz"))
        release_pose = [*(xyz or []), *(quat or [])]
    candidate = (
        build_support_contact_candidate(
            scene_memory=scene_memory,
            held_instance=held,
            arm=arm,
            current_object_world_m=current_object,
            held_extent_m=extent,
            release_ee_pose=release_pose,
            limits=limits,
        )
        if metrics is not None
        else None
    )
    if candidate is None or metrics is None:
        return None
    selected, validation = candidate
    return selected, validation, metrics


def authorize_support_contact_release(
    *,
    scene_memory: Any,
    held_instance: Any,
    arm: str,
    arm_state: Any,
    robot_arm_state: Any,
    calibration: Any,
    gripper_closed: bool,
    limits: SupportContactLimits,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    state = dict(arm_state or {}) if isinstance(arm_state, dict) else {}
    held = dict(held_instance or {}) if isinstance(held_instance, dict) else {}
    lease = state.get("support_contact_release")
    candidate = lease.get("candidate") if isinstance(lease, dict) else None
    held_id = str(state.get("held_instance_id", "") or "").strip()
    target_id = str(
        candidate.get("target_id", "")
        if isinstance(candidate, dict)
        else ""
    ).strip()
    if (
        not gripper_closed
        or not manipulation_state_allows_transport(state)
        or state.get("release_ready") is not True
        or state.get("release_readiness_source")
        != SUPPORT_CONTACT_RELEASE_SOURCE
        or not isinstance(lease, dict)
        or lease.get("state") != "ready"
        or not isinstance(candidate, dict)
        or not held_id
        or str(candidate.get("held_instance_id", "") or "").strip()
        != held_id
        or str(held.get("instance_id", "") or "").strip() != held_id
        or str(lease.get("target_id", "") or "").strip() != target_id
        or str(state.get("place_target_id", "") or "").strip()
        != target_id
        or str(held.get("status", "") or "").strip().lower()
        not in {"visible", "tracked"}
        or str(held.get("position_state", "") or "").strip().lower()
        == "motion_uncertain"
    ):
        return None
    current_object = current_held_object_world_m(
        held_instance=held,
        arm_state=state,
        robot_arm_state=robot_arm_state,
        calibration=calibration,
    )
    current_ee = (
        robot_arm_state.get("xyz")
        if isinstance(robot_arm_state, dict)
        else None
    )
    return revalidate_support_contact_candidate(
        scene_memory=scene_memory,
        held_instance=held,
        candidate=candidate,
        current_object_world_m=current_object,
        current_ee_world_m=current_ee,
        limits=limits,
    )


def current_held_object_world_m(
    *,
    held_instance: Any,
    arm_state: Any,
    robot_arm_state: Any,
    calibration: Any,
) -> list[float] | None:
    held = dict(held_instance or {}) if isinstance(held_instance, dict) else {}
    state = dict(arm_state or {}) if isinstance(arm_state, dict) else {}
    propagated = propagate_held_object_world_m(
        robot_arm_state=robot_arm_state,
        attachment=state.get("held_object_to_tcp_attachment"),
        calibration=calibration,
    )
    return _xyz(propagated) or _xyz(
        held.get("latest_world_m", held.get("world_m"))
    )


def drop_redundant_support_contact_settles(
    calls: list[RecoveryToolCall],
    *,
    authorizations: dict[
        str,
        tuple[dict[str, Any], dict[str, Any]] | None,
    ],
    holding_states: dict[str, dict[str, Any] | None],
    limits: SupportContactLimits,
) -> tuple[list[RecoveryToolCall], list[dict[str, Any]]]:
    filtered: list[RecoveryToolCall] = []
    dropped: list[dict[str, Any]] = []
    index = 0
    while index < len(calls):
        settle = calls[index]
        release = calls[index + 1] if index + 1 < len(calls) else None
        arm = _call_arm(settle)
        holding_state = holding_states.get(arm)
        release_ref = str(
            (release.args or {}).get("release_held_instance_id", "")
            if release is not None
            else ""
        ).strip().lower()
        held_ref = str(
            (holding_state or {}).get("held_instance_id", "") or ""
        ).strip().lower()
        authorization = authorizations.get(arm)
        if not (
            authorization is not None
            and release is not None
            and settle.tool_name == "contact_displace"
            and release.tool_name == "open_gripper"
            and _call_arm(release) == arm
            and release_ref
            and release_ref == held_ref
            and is_bounded_downward_settle(settle.args, limits=limits)
        ):
            filtered.append(settle)
            index += 1
            continue
        args = dict(release.args or {})
        args["_support_contact_release_settle_dropped"] = True
        filtered.append(RecoveryToolCall(tool_name="open_gripper", args=args))
        candidate, _ = authorization
        dropped.append(
            {
                "arm": arm,
                "held_instance_id": held_ref,
                "target_id": candidate.get("target_id"),
            }
        )
        index += 2
    return filtered, dropped


def is_bounded_downward_settle(
    args: Any,
    *,
    limits: SupportContactLimits,
) -> bool:
    payload = dict(args or {}) if isinstance(args, dict) else {}
    axis = str(payload.get("axis", "z") or "z").strip().lower()
    direction = str(
        payload.get("direction", "negative") or "negative"
    ).strip().lower()
    try:
        distance = abs(float(payload.get("distance", 0.0)))
    except (TypeError, ValueError):
        return False
    return bool(
        axis == "z"
        and direction in {"negative", "down", "downward", "-"}
        and limits.no_progress_m < distance
        <= limits.max_contact_command_m
    )


def resisted_settle_metrics(
    *,
    args: Any,
    details: Any,
    limits: SupportContactLimits,
) -> dict[str, float] | None:
    """Return deterministic support-resistance metrics or reject the motion.

    A successful tool return alone is insufficient.  The command must be a
    bounded downward settle, the observed total/axial motion must remain below
    the controller's existing no-progress epsilon, and the adapter must not
    report collision, unreachable motion, or explicit invalid contact.
    """

    payload = dict(args or {}) if isinstance(args, dict) else {}
    result = dict(details or {}) if isinstance(details, dict) else {}
    if (
        not is_bounded_downward_settle(payload, limits=limits)
        or result.get("collision") is True
        or result.get("unreachable") is True
        or result.get("valid_contact") is False
    ):
        return None
    axis = str(result.get("axis", "z") or "z").strip().lower()
    try:
        commanded = float(
            result.get(
                "signed_distance",
                -abs(float(payload.get("distance", 0.0))),
            )
        )
        observed_axis = float(
            result.get("observed_axis_displacement_m")
        )
    except (TypeError, ValueError):
        return None
    observed_xyz = _xyz(result.get("observed_displacement_xyz"))
    if (
        axis != "z"
        or not math.isfinite(commanded)
        or commanded >= 0.0
        or not math.isfinite(observed_axis)
        or observed_xyz is None
    ):
        return None
    observed_total = math.sqrt(sum(value * value for value in observed_xyz))
    observed_lateral = math.hypot(observed_xyz[0], observed_xyz[1])
    if (
        abs(observed_axis) > limits.no_progress_m
        or observed_total > limits.no_progress_m
    ):
        return None
    return {
        "commanded_axis_displacement_m": commanded,
        "observed_axis_displacement_m": observed_axis,
        "observed_total_displacement_m": observed_total,
        "observed_lateral_drift_m": observed_lateral,
    }


def build_support_contact_candidate(
    *,
    scene_memory: Any,
    held_instance: Any,
    arm: str,
    current_object_world_m: Any,
    held_extent_m: Any,
    release_ee_pose: Any,
    limits: SupportContactLimits,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Build a generic runtime target at a physically supported held pose.

    The target is derived from current world geometry.  It is tried against
    the same minimal placement target kinds already used by the framework:
    free support and grounded object top.  No task name, coordinate, color, or
    language phrase participates in selection.
    """

    scene = dict(scene_memory or {}) if isinstance(scene_memory, dict) else {}
    held = dict(held_instance or {}) if isinstance(held_instance, dict) else {}
    held_id = str(held.get("instance_id", "") or "").strip()
    normalized_arm = str(arm or "").strip().lower()
    current_object = _xyz(current_object_world_m)
    extent = _xyz(held_extent_m)
    ee_pose = _pose7(release_ee_pose)
    if (
        not held_id
        or normalized_arm not in {"left", "right"}
        or current_object is None
        or extent is None
        or any(value <= 0.0 for value in extent)
        or ee_pose is None
    ):
        return None

    instances = [
        item
        for item in scene.get("instances", []) or []
        if isinstance(item, dict)
    ]
    support_specs: list[tuple[str, str, str]] = [
        (
            "free_support",
            "",
            "bounded_downward_contact_on_inferred_support_plane",
        )
    ]
    support_specs.extend(
        (
            "object_top",
            str(item.get("instance_id", "") or "").strip(),
            "bounded_downward_contact_on_grounded_object_top",
        )
        for item in instances
        if (
            str(item.get("instance_id", "") or "").strip()
            and str(item.get("instance_id", "") or "").strip() != held_id
        )
    )

    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for source_index, (
        target_kind,
        support_id,
        support_evidence,
    ) in enumerate(support_specs):
        target_id = _support_target_id(
            held_id=held_id,
            target_kind=target_kind,
            support_id=support_id,
            current_object=current_object,
        )
        candidate = {
            "candidate_id": f"{target_id}:arm:{normalized_arm}",
            "target_id": target_id,
            "target_kind": target_kind,
            "action_mode": "place",
            "arm": normalized_arm,
            "held_instance_id": held_id,
            "ee_target_pose": ee_pose,
            "approach_pose": ee_pose,
            "held_object_target_world_m": current_object,
            "held_extent_m": [abs(value) for value in extent],
            "support_instance_id": support_id or None,
            "support_evidence": support_evidence,
            "vacated_by_instance_id": None,
            "source_candidate_index": source_index,
            "geometry_source": "runtime_bounded_downward_support_contact",
            "post_release_verification_required": True,
        }
        validation = validate_place_candidate(
            scene,
            held_instance=held,
            candidate=candidate,
        )
        support_drift = _finite_float(
            validation.get("target_geometry_drift_m")
        )
        if (
            validation.get("valid") is not True
            or support_drift is None
            or support_drift > limits.max_support_geometry_drift_m
        ):
            continue
        candidate.update(validation)
        candidates.append((support_drift, candidate, dict(validation)))
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item[0],
            str(item[1].get("target_id", "")),
        )
    )
    _, candidate, validation = candidates[0]
    return candidate, validation


def revalidate_support_contact_candidate(
    *,
    scene_memory: Any,
    held_instance: Any,
    candidate: Any,
    current_object_world_m: Any,
    current_ee_world_m: Any,
    limits: SupportContactLimits,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    scene = dict(scene_memory or {}) if isinstance(scene_memory, dict) else {}
    held = dict(held_instance or {}) if isinstance(held_instance, dict) else {}
    selected = dict(candidate or {}) if isinstance(candidate, dict) else {}
    current_object = _xyz(current_object_world_m)
    expected_object = _xyz(selected.get("held_object_target_world_m"))
    current_ee = _xyz(current_ee_world_m)
    release_ee = _xyz(selected.get("ee_target_pose"))
    extent = _xyz(selected.get("held_extent_m"))
    if any(
        value is None
        for value in (
            current_object,
            expected_object,
            current_ee,
            release_ee,
            extent,
        )
    ):
        return None
    assert current_object is not None
    assert expected_object is not None
    assert current_ee is not None
    assert release_ee is not None
    assert extent is not None
    xy_error = math.hypot(
        current_object[0] - expected_object[0],
        current_object[1] - expected_object[1],
    )
    z_error = abs(current_object[2] - expected_object[2])
    ee_drift = _distance(current_ee, release_ee)
    if (
        xy_error > place_target_tolerance_m(extent)
        or z_error > limits.max_support_geometry_drift_m
        or ee_drift > limits.max_pose_drift_m
    ):
        return None
    validation = validate_place_candidate(
        scene,
        held_instance=held,
        candidate=selected,
    )
    support_drift = _finite_float(
        validation.get("target_geometry_drift_m")
    )
    if (
        validation.get("valid") is not True
        or support_drift is None
        or support_drift > limits.max_support_geometry_drift_m
    ):
        return None
    selected.update(validation)
    validation = {
        **validation,
        "current_object_xy_error_m": round(xy_error, 6),
        "current_object_z_error_m": round(z_error, 6),
        "current_ee_drift_m": round(ee_drift, 6),
    }
    return selected, validation


def install_support_contact_release_state(
    *,
    arm_state: Any,
    candidate: dict[str, Any],
    validation: dict[str, Any],
    metrics: dict[str, float],
    env_step: int,
) -> dict[str, Any]:
    base = clear_support_contact_release_state(arm_state)
    prior_place_contract = {
        key: base[key]
        for key in _PLACE_STATE_KEYS
        if key in base
    }
    lease = {
        "state": "ready",
        "target_id": candidate.get("target_id"),
        "candidate": dict(candidate),
        "validation": dict(validation),
        "created_step": int(env_step),
        "source": SUPPORT_CONTACT_RELEASE_SOURCE,
        "prior_place_contract": prior_place_contract,
        **{
            key: round(float(value), 6)
            for key, value in metrics.items()
        },
    }
    return {
        **base,
        "release_ready": True,
        "release_readiness_source": SUPPORT_CONTACT_RELEASE_SOURCE,
        "support_contact_release": lease,
        "place_target_id": str(candidate.get("target_id", "") or ""),
        "place_candidate_id": str(
            candidate.get("candidate_id", "") or ""
        ),
        "held_object_target_world_m": candidate.get(
            "held_object_target_world_m"
        ),
        "held_extent_m": candidate.get("held_extent_m"),
        "release_ee_target_world_m": _xyz(candidate.get("ee_target_pose")),
        "pre_release_validated": validation.get("valid") is True,
        "updated_step": int(env_step),
    }


def clear_support_contact_release_state(arm_state: Any) -> dict[str, Any]:
    state = dict(arm_state or {}) if isinstance(arm_state, dict) else {}
    lease = state.get("support_contact_release")
    if not isinstance(lease, dict):
        return state
    for key in (
        "release_ready",
        "release_readiness_source",
        "support_contact_release",
        *_PLACE_STATE_KEYS,
    ):
        state.pop(key, None)
    prior_place_contract = lease.get("prior_place_contract")
    if isinstance(prior_place_contract, dict):
        state.update(dict(prior_place_contract))
    return state


def _support_target_id(
    *,
    held_id: str,
    target_kind: str,
    support_id: str,
    current_object: list[float],
) -> str:
    payload = json.dumps(
        {
            "held": held_id,
            "kind": target_kind,
            "support": support_id,
            "anchor": current_object,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    anchor_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return (
        f"place:held:{held_id}:support-contact:{target_kind}:"
        f"{support_id or 'plane'}:{anchor_id}"
    )


def _pose7(value: Any) -> list[float] | None:
    try:
        pose = [float(item) for item in list(value)]
    except (TypeError, ValueError):
        return None
    if len(pose) != 7 or not all(math.isfinite(item) for item in pose):
        return None
    quat_norm = math.sqrt(sum(item * item for item in pose[3:7]))
    return pose if quat_norm > 0.0 else None


def _xyz(value: Any) -> list[float] | None:
    try:
        xyz = [float(item) for item in list(value)[:3]]
    except (TypeError, ValueError):
        return None
    if len(xyz) != 3 or not all(math.isfinite(item) for item in xyz):
        return None
    return xyz


def _quat(value: Any) -> list[float] | None:
    try:
        quat = [float(item) for item in list(value)[:4]]
    except (TypeError, ValueError):
        return None
    if len(quat) != 4 or not all(math.isfinite(item) for item in quat):
        return None
    norm = math.sqrt(sum(item * item for item in quat))
    return quat if norm > 0.0 else None


def _call_arm(call: RecoveryToolCall) -> str:
    arm = str((call.args or {}).get("arm", "") or "").strip().lower()
    aliases = {
        "all": "both",
        "dual": "both",
        "left_arm": "left",
        "right_arm": "right",
    }
    return normalize_physical_arm(aliases.get(arm, arm))


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _distance(first: list[float], second: list[float]) -> float:
    return math.sqrt(
        sum((first[index] - second[index]) ** 2 for index in range(3))
    )
