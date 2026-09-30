from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


STRICT_ACTION_GEOMETRY_REPAIR_PENDING_POLICY = "strict"
SAFE_MOTION_ACTION_GEOMETRY_REPAIR_PENDING_POLICY = "safe_motion"
DISABLED_ACTION_GEOMETRY_REPAIR_PENDING_POLICY = "disabled"

VALID_ACTION_GEOMETRY_REPAIR_PENDING_POLICIES = {
    STRICT_ACTION_GEOMETRY_REPAIR_PENDING_POLICY,
    SAFE_MOTION_ACTION_GEOMETRY_REPAIR_PENDING_POLICY,
    DISABLED_ACTION_GEOMETRY_REPAIR_PENDING_POLICY,
}

_EXECUTABLE_POSITION_STATES = {
    "current_verified",
    "memory_valid",
}


def normalize_action_geometry_repair_pending_policy(
    value: Any,
    *,
    field_name: str = "action_geometry_repair_pending_policy",
) -> str:
    """Return one validated repair-pending policy name."""

    policy = str(value or "").strip().lower().replace("-", "_")
    if policy not in VALID_ACTION_GEOMETRY_REPAIR_PENDING_POLICIES:
        choices = ", ".join(
            sorted(VALID_ACTION_GEOMETRY_REPAIR_PENDING_POLICIES)
        )
        raise ValueError(
            f"{field_name} must be one of {{{choices}}}; got {value!r}"
        )
    return policy


@dataclass(frozen=True, slots=True)
class RepairPendingSafeMotion:
    """One runtime-derived, non-contact move for repairing action geometry."""

    target_pose: tuple[float, ...]
    phase: str
    reference_world_m: tuple[float, float, float]
    reference_source: str
    object_top_z_m: float
    safe_ee_z_m: float
    translation_m: float
    max_translation_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_pose": list(self.target_pose),
            "phase": self.phase,
            "reference_world_m": list(self.reference_world_m),
            "reference_source": self.reference_source,
            "object_top_z_m": self.object_top_z_m,
            "safe_ee_z_m": self.safe_ee_z_m,
            "translation_m": self.translation_m,
            "max_translation_m": self.max_translation_m,
        }


def plan_repair_pending_safe_motion(
    *,
    instance: Mapping[str, Any],
    current_pose: Sequence[Any] | None,
    gripper_state: Any,
    ee_to_tcp_m: Any,
    approach_clearance_m: Any,
    max_translation_m: float = 0.08,
    position_tolerance_m: float = 0.005,
) -> RepairPendingSafeMotion | None:
    """Plan a lift-first move using position memory, never quarantined poses.

    The move is authorized only for an open gripper and an executable object
    position.  It first raises the EE to a clearance derived from the current
    object top, calibrated EE-to-TCP distance, and configured approach height.
    Only after that clearance is reached may it translate laterally above the
    object.  No approach/grasp/contact pose from ``instance`` is consulted.
    """

    if str(gripper_state or "").strip().lower() != "open":
        return None
    position_state = str(
        instance.get("position_state", "") or ""
    ).strip().lower()
    if position_state not in _EXECUTABLE_POSITION_STATES:
        return None

    pose = _pose7(current_pose)
    if pose is None:
        return None
    reference, reference_source = _position_reference(
        instance,
        position_state=position_state,
    )
    if reference is None:
        return None
    object_top_z = _object_top_z(instance, reference_world=reference)
    if object_top_z is None:
        return None

    tcp_offset = _nonnegative_finite(ee_to_tcp_m)
    clearance = _nonnegative_finite(approach_clearance_m)
    step_limit = _positive_finite(max_translation_m)
    tolerance = _positive_finite(position_tolerance_m)
    if (
        tcp_offset is None
        or clearance is None
        or step_limit is None
        or tolerance is None
    ):
        return None

    current_xyz = list(pose[:3])
    # If the EE origin is already below the object's top plus its calibrated
    # TCP reach, an unmodelled lift cannot be certified as collision-free.
    minimum_noncontact_z = object_top_z + tcp_offset
    if current_xyz[2] + tolerance < minimum_noncontact_z:
        return None
    safe_ee_z = minimum_noncontact_z + clearance

    target_xyz = list(current_xyz)
    if current_xyz[2] + tolerance < safe_ee_z:
        phase = "vertical_clearance"
        target_xyz[2] = min(safe_ee_z, current_xyz[2] + step_limit)
    else:
        phase = "lateral_standoff"
        dx = reference[0] - current_xyz[0]
        dy = reference[1] - current_xyz[1]
        lateral_distance = math.hypot(dx, dy)
        if lateral_distance <= tolerance:
            return None
        scale = min(1.0, step_limit / lateral_distance)
        target_xyz[0] += dx * scale
        target_xyz[1] += dy * scale
        # Never descend during repair-pending safe motion.
        target_xyz[2] = max(current_xyz[2], safe_ee_z)

    translation = math.sqrt(
        sum(
            (target_xyz[index] - current_xyz[index]) ** 2
            for index in range(3)
        )
    )
    if not math.isfinite(translation) or translation <= 1e-6:
        return None
    if translation > step_limit + 1e-8:
        return None

    target_pose = tuple(target_xyz + list(pose[3:7]))
    return RepairPendingSafeMotion(
        target_pose=target_pose,
        phase=phase,
        reference_world_m=reference,
        reference_source=reference_source,
        object_top_z_m=round(object_top_z, 6),
        safe_ee_z_m=round(safe_ee_z, 6),
        translation_m=round(translation, 6),
        max_translation_m=round(step_limit, 6),
    )


def repair_pending_retreat_is_safe(
    *,
    instance: Mapping[str, Any],
    current_pose: Sequence[Any] | None,
    gripper_state: Any,
    axis: Any,
    direction: Any,
    distance_m: Any,
    maximum_distance_m: float = 0.12,
    distance_tolerance_m: float = 0.001,
) -> bool:
    """Return whether a bounded retreat does not approach the pending object."""

    if str(gripper_state or "").strip().lower() != "open":
        return False
    pose = _pose7(current_pose)
    if pose is None:
        return False
    position_state = str(
        instance.get("position_state", "") or ""
    ).strip().lower()
    if position_state not in _EXECUTABLE_POSITION_STATES:
        return False
    reference, _ = _position_reference(
        instance,
        position_state=position_state,
    )
    if reference is None:
        return False

    normalized_axis = str(axis or "z").strip().lower()
    if normalized_axis not in {"x", "y", "z"}:
        return False
    normalized_direction = str(direction or "positive").strip().lower()
    if normalized_direction not in {"positive", "negative"}:
        return False
    # A repair-pending clearance action may move laterally away or upward,
    # but it must never disguise a downward approach as a "retreat".
    if normalized_axis == "z" and normalized_direction == "negative":
        return False
    distance = _positive_finite(distance_m)
    maximum = _positive_finite(maximum_distance_m)
    tolerance = _nonnegative_finite(distance_tolerance_m)
    if (
        distance is None
        or maximum is None
        or tolerance is None
        or distance > maximum + 1e-8
    ):
        return False

    current_xyz = list(pose[:3])
    target_xyz = list(current_xyz)
    signed_distance = (
        distance if normalized_direction == "positive" else -distance
    )
    target_xyz[{"x": 0, "y": 1, "z": 2}[normalized_axis]] += (
        signed_distance
    )
    current_distance = math.sqrt(
        sum(
            (current_xyz[index] - reference[index]) ** 2
            for index in range(3)
        )
    )
    target_distance = math.sqrt(
        sum(
            (target_xyz[index] - reference[index]) ** 2
            for index in range(3)
        )
    )
    return bool(
        math.isfinite(current_distance)
        and math.isfinite(target_distance)
        and target_distance + tolerance >= current_distance
    )


def _position_reference(
    instance: Mapping[str, Any],
    *,
    position_state: str,
) -> tuple[tuple[float, float, float] | None, str]:
    keys = (
        ("latest_world_m", "world_m", "last_verified_world_m")
        if position_state == "current_verified"
        else (
            "last_verified_world_m",
            "stable_world_m",
            "world_m",
        )
    )
    for key in keys:
        value = _xyz3(instance.get(key))
        if value is not None:
            return value, key
    return None, ""


def _object_top_z(
    instance: Mapping[str, Any],
    *,
    reference_world: tuple[float, float, float],
) -> float | None:
    quality = instance.get("quality")
    if isinstance(quality, Mapping):
        world_z_max = _finite_float(quality.get("world_z_max_m"))
        if world_z_max is not None:
            return world_z_max

    for key in ("top_surface_world_m", "stable_top_surface_world_m"):
        top_surface = _xyz3(instance.get(key))
        if top_surface is not None:
            return top_surface[2]

    bbox_max = _xyz3(instance.get("bbox_world_max"))
    if bbox_max is not None:
        return bbox_max[2]

    extent = None
    if isinstance(quality, Mapping):
        extent = _xyz3(quality.get("world_extent_m"))
    if extent is None:
        extent = _xyz3(instance.get("extent_m"))
    if extent is not None:
        return reference_world[2] + abs(extent[2]) / 2.0
    return reference_world[2]


def _pose7(value: Sequence[Any] | None) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return None
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(values) != 7 or not all(math.isfinite(item) for item in values):
        return None
    quat_norm = math.sqrt(sum(item * item for item in values[3:7]))
    if quat_norm <= 1e-8:
        return None
    return values[:3] + tuple(item / quat_norm for item in values[3:7])


def _xyz3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return None
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        return None
    return values


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_finite(value: Any) -> float | None:
    number = _finite_float(value)
    return number if number is not None and number >= 0.0 else None


def _positive_finite(value: Any) -> float | None:
    number = _finite_float(value)
    return number if number is not None and number > 0.0 else None
