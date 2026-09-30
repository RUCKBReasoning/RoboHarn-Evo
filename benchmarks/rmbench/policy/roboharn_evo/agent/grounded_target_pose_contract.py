from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .operation_candidates import (
    normalize_grounded_point_key,
    operation_pose_candidates,
    select_operation_pose_candidate,
)


_UNSET = object()

_GROUNDED_QUATERNION_KEYS = {
    "approach_world_m": "approach_quat_wxyz",
    "grasp_world_m": "grasp_quat_wxyz",
    "contact_world_m": "contact_quat_wxyz",
    "place_world_m": "place_quat_wxyz",
}

_GROUNDED_QUATERNION_MODES = {
    "grounded",
    "scene",
    "affordance",
    "auto",
}

_CURRENT_QUATERNION_MODES = {
    "",
    "preserve",
    "current",
}


@dataclass(frozen=True, slots=True)
class GroundedTargetPoseError:
    """Machine-readable failure returned by the grounded-pose contract."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GroundedTargetPoseResolution:
    """Result of resolving one materialized grounded-instance target pose."""

    target_pose: np.ndarray | None = None
    error: GroundedTargetPoseError | None = None

    @property
    def success(self) -> bool:
        return self.error is None


def authorize_memory_valid_final_grounded_action(
    *,
    enabled: bool,
    raw_instance: Mapping[str, Any] | None,
    materialized_instance: Mapping[str, Any],
    arm: Any,
    point_key: Any,
    action_mode: Any,
    candidate_id: Any = "",
    blocked_candidate_ids: Any = None,
    requested_target_id: Any = None,
    offset_xyz: Any = _UNSET,
    preserve_height: Any = False,
) -> dict[str, Any] | None:
    """Authorize visibility relaxation without relaxing pose identity.

    The caller may choose to execute an exact, verified contact/grasp pose
    from physically valid world memory even when the object is not visible in
    the current frame.  Placement, arbitrary offsets, motion-uncertain state,
    and mismatched private candidates remain outside this contract.
    """

    if not enabled or not isinstance(raw_instance, Mapping):
        return None
    normalized_arm = str(arm or "").strip().lower()
    normalized_mode = str(action_mode or "").strip().lower()
    normalized_point_key = normalize_grounded_point_key(point_key)
    expected_point_key = (
        f"{normalized_mode}_world_m"
        if normalized_mode in {"contact", "grasp"}
        else ""
    )
    position_state = str(
        materialized_instance.get("position_state", "") or ""
    ).strip().lower()
    geometry_state = str(
        materialized_instance.get("action_geometry_state", "verified")
        or "verified"
    ).strip().lower()
    raw_offset = [0.0, 0.0, 0.0] if offset_xyz is _UNSET else offset_xyz
    offset = _xyz_array(raw_offset)
    if (
        normalized_arm not in {"left", "right"}
        or normalized_point_key != expected_point_key
        or position_state != "memory_valid"
        or geometry_state != "verified"
        or _xyz_array(materialized_instance.get(normalized_point_key)) is None
        or offset is None
        or float(np.linalg.norm(offset)) > 1e-5
        or preserve_height is not False
    ):
        return None

    candidates = operation_pose_candidates(dict(raw_instance))
    normalized_candidate_id = str(candidate_id or "").strip()
    if candidates:
        if not normalized_candidate_id:
            return None
        selected = select_operation_pose_candidate(
            dict(raw_instance),
            arm=normalized_arm,
            action_mode=normalized_mode,
            blocked_candidate_ids=blocked_candidate_ids,
            requested_candidate_id=normalized_candidate_id,
            requested_target_id=requested_target_id,
        )
        if (
            selected is None
            or str(selected.get("candidate_id", "") or "").strip()
            != normalized_candidate_id
            or str(selected.get("arm", "") or "").strip().lower()
            != normalized_arm
            or str(
                selected.get("action_mode", "") or ""
            ).strip().lower()
            != normalized_mode
            or str(
                materialized_instance.get(
                    "selected_operation_candidate_id",
                    "",
                )
                or ""
            ).strip()
            != normalized_candidate_id
        ):
            return None

    return {
        "instance_id": str(
            materialized_instance.get("instance_id", "") or ""
        ).strip(),
        "track_id": str(
            materialized_instance.get("track_id", "") or ""
        ).strip(),
        "arm": normalized_arm,
        "point_key": normalized_point_key,
        "action_mode": normalized_mode,
        "candidate_id": normalized_candidate_id,
        "position_state": position_state,
        "position_source": materialized_instance.get("position_source"),
    }


def resolve_grounded_target_pose(
    *,
    instance: Mapping[str, Any],
    point_key: Any = "approach_world_m",
    offset_xyz: Any = _UNSET,
    preserve_height: Any = False,
    target_quat_wxyz: Any = _UNSET,
    quat_wxyz: Any = _UNSET,
    current_pose: Any = None,
) -> GroundedTargetPoseResolution:
    """Resolve the final 7D pose for a materialized grounded instance.

    This is the side-effect-free form of
    ``RMBenchRecoveryAdapter._target_pose_from_grounded_instance``.  The two
    quaternion arguments preserve the adapter's alias semantics: when
    ``target_quat_wxyz`` is supplied it takes precedence over ``quat_wxyz``,
    including when its explicit value is ``None``.

    ``offset_xyz`` also distinguishes omission (the adapter's zero-vector
    default) from an explicit ``None`` (an invalid finite-3D value).
    """

    if not isinstance(instance, Mapping):
        return _failure(
            "invalid_grounded_instance",
            "grounded instance must be a mapping",
            {"instance_type": type(instance).__name__},
        )

    normalized_point_key = normalize_grounded_point_key(point_key)
    point = _xyz_array(instance.get(normalized_point_key))
    if point is None:
        return _failure(
            "missing_grounded_point",
            f"scene instance has no finite {normalized_point_key}",
            {
                "instance_id": instance.get("instance_id"),
                "point_key": normalized_point_key,
                "available_keys": sorted(str(key) for key in instance),
            },
        )

    raw_offset = [0.0, 0.0, 0.0] if offset_xyz is _UNSET else offset_xyz
    offset = _xyz_array(raw_offset)
    if offset is None:
        return _failure(
            "invalid_offset_xyz",
            "offset_xyz must be a finite 3D vector",
            {"offset_xyz": None if offset_xyz is _UNSET else offset_xyz},
        )

    if not isinstance(preserve_height, bool):
        return _failure(
            "invalid_preserve_height",
            "preserve_height must be a boolean",
            {"preserve_height": preserve_height},
        )

    target_xyz = point + offset
    validated_current_pose: np.ndarray | None = None
    if preserve_height:
        validated_current_pose = _pose_array(current_pose)
        if validated_current_pose is None:
            return _failure(
                "invalid_current_pose",
                "current_pose must be a finite 7D xyz+quat_wxyz pose with a non-zero quaternion",
                {"current_pose": current_pose},
            )
        target_xyz[2] = validated_current_pose[2] + offset[2]

    raw_quaternion = _selected_quaternion_argument(
        target_quat_wxyz=target_quat_wxyz,
        quat_wxyz=quat_wxyz,
    )
    grounded_quaternion_key = _GROUNDED_QUATERNION_KEYS.get(
        normalized_point_key,
        "",
    )
    grounded_quaternion = (
        _quat_array(instance.get(grounded_quaternion_key))
        if grounded_quaternion_key
        else None
    )
    quaternion_mode = (
        str(raw_quaternion).strip().lower()
        if isinstance(raw_quaternion, str)
        else ""
    )

    if raw_quaternion is None and grounded_quaternion is not None:
        quaternion = grounded_quaternion
    elif quaternion_mode in _GROUNDED_QUATERNION_MODES:
        if grounded_quaternion is None:
            display_key = grounded_quaternion_key or "grounded quaternion"
            return _failure(
                "missing_grounded_quaternion",
                f"scene instance has no finite {display_key}",
                {
                    "instance_id": instance.get("instance_id"),
                    "point_key": normalized_point_key,
                    "grounded_quaternion_key": grounded_quaternion_key,
                },
            )
        quaternion = grounded_quaternion
    elif raw_quaternion is None or (
        isinstance(raw_quaternion, str)
        and quaternion_mode in _CURRENT_QUATERNION_MODES
    ):
        if validated_current_pose is None:
            validated_current_pose = _pose_array(current_pose)
        if validated_current_pose is None:
            return _failure(
                "invalid_current_pose",
                "current_pose must be a finite 7D xyz+quat_wxyz pose with a non-zero quaternion",
                {"current_pose": current_pose},
            )
        quaternion = validated_current_pose[3:7]
    else:
        quaternion = _quat_array(raw_quaternion)
        if quaternion is None:
            return _failure(
                "invalid_target_quaternion",
                "target_quat_wxyz must be a finite 4D quaternion",
                {"target_quat_wxyz": raw_quaternion},
            )

    target_pose = np.concatenate([target_xyz, quaternion]).astype(
        np.float32
    )
    return GroundedTargetPoseResolution(target_pose=target_pose)


def _selected_quaternion_argument(
    *,
    target_quat_wxyz: Any,
    quat_wxyz: Any,
) -> Any:
    if target_quat_wxyz is not _UNSET:
        return target_quat_wxyz
    if quat_wxyz is not _UNSET:
        return quat_wxyz
    return None


def _xyz_array(value: Any) -> np.ndarray | None:
    try:
        xyz = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        return None
    return xyz.copy()


def _quat_array(value: Any) -> np.ndarray | None:
    try:
        quaternion = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        return None
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-6:
        return None
    return quaternion / norm


def _pose_array(value: Any) -> np.ndarray | None:
    try:
        pose = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        return None
    quaternion_norm = float(np.linalg.norm(pose[3:7]))
    if quaternion_norm <= 1e-6:
        return None
    normalized = pose.copy()
    normalized[3:7] /= quaternion_norm
    return normalized


def _failure(
    code: str,
    message: str,
    details: dict[str, Any],
) -> GroundedTargetPoseResolution:
    return GroundedTargetPoseResolution(
        error=GroundedTargetPoseError(
            code=code,
            message=message,
            details=details,
        )
    )


__all__ = [
    "authorize_memory_valid_final_grounded_action",
    "GroundedTargetPoseError",
    "GroundedTargetPoseResolution",
    "resolve_grounded_target_pose",
]
