from __future__ import annotations

import math
from typing import Any


VERIFIED_OPERATION_TARGET_ANCHOR_KEY = (
    "verified_operation_target_anchor"
)
OPERATION_GEOMETRY_TRANSLATION_CHANGE_M = 0.01
OPERATION_GEOMETRY_ROTATION_CHANGE_RAD = math.radians(10.0)
POSITION_ONLY_ACTION_GEOMETRY_QUARANTINE_LEASE_KEY = (
    "allow_position_only_action_geometry_quarantine"
)
_EXECUTABLE_POSE_KEYS = (
    "object_contact_pose",
    "tcp_pose",
    "ee_target_pose",
    "approach_pose",
)


def operation_target_anchor_from_event(
    event: Any,
    *,
    env_step: int,
    max_tolerance_m: float,
) -> dict[str, Any] | None:
    """Build an identity anchor from a runtime-verified operation target.

    The target is supporting provenance, not a replacement for the observed
    object centroid.  Both coordinates remain available to association until
    a later physical event makes the object's position uncertain.
    """

    value = event if isinstance(event, dict) else {}
    world = _xyz(value.get("operation_target_world_m"))
    tolerance = _positive_float(
        value.get("operation_target_tolerance_m")
    )
    if world is None or tolerance is None:
        return None
    return {
        "state": "active",
        "kind": "verified_operation_target",
        "world_m": world,
        "tolerance_m": round(
            min(float(max_tolerance_m), tolerance),
            6,
        ),
        "target_id": str(
            value.get("operation_target_id", "") or ""
        ).strip(),
        "verified_step": int(env_step),
        "source": str(
            value.get("source", "verified_runtime_operation")
            or "verified_runtime_operation"
        ).strip(),
        "arm": str(value.get("arm", "") or "").strip().lower(),
    }


def verified_position_anchors(
    track: Any,
    *,
    default_tolerance_m: float,
    max_tolerance_m: float,
) -> list[dict[str, Any]]:
    """Return every active world anchor that can constrain identity.

    Visibility is deliberately not required.  A verified position remains
    valid through occlusion; only a physical motion-uncertainty event revokes
    it in the owning tracker.
    """

    value = track if isinstance(track, dict) else {}
    if str(value.get("position_state", "") or "").strip() not in {
        "current_verified",
        "memory_valid",
    }:
        return []
    anchors: list[dict[str, Any]] = []
    primary_world = _xyz(value.get("last_verified_world_m"))
    if primary_world is None:
        primary_world = _xyz(
            value.get("stable_world_m", value.get("world_m"))
        )
    primary_tolerance = _positive_float(
        value.get("position_tolerance_m")
    )
    if primary_tolerance is None:
        primary_tolerance = float(default_tolerance_m)
    if primary_world is not None:
        anchors.append(
            {
                "state": "active",
                "kind": "verified_observation",
                "world_m": primary_world,
                "tolerance_m": min(
                    float(max_tolerance_m),
                    primary_tolerance,
                ),
                "verified_step": value.get("last_verified_step"),
                "source": value.get("last_verified_source"),
            }
        )

    operation_anchor = value.get(
        VERIFIED_OPERATION_TARGET_ANCHOR_KEY
    )
    if (
        isinstance(operation_anchor, dict)
        and operation_anchor.get("state") == "active"
    ):
        operation_world = _xyz(operation_anchor.get("world_m"))
        operation_tolerance = _positive_float(
            operation_anchor.get("tolerance_m")
        )
        if operation_world is not None and operation_tolerance is not None:
            anchors.append(
                {
                    **dict(operation_anchor),
                    "world_m": operation_world,
                    "tolerance_m": min(
                        float(max_tolerance_m),
                        operation_tolerance,
                    ),
                }
            )
    return anchors


def eligible_anchor_matches(
    world_m: Any,
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    world = _xyz(world_m)
    if world is None:
        return []
    matches: list[dict[str, Any]] = []
    for anchor in anchors:
        anchor_world = _xyz(anchor.get("world_m"))
        tolerance = _positive_float(anchor.get("tolerance_m"))
        if anchor_world is None or tolerance is None:
            continue
        distance = math.sqrt(
            sum(
                (world[index] - anchor_world[index]) ** 2
                for index in range(3)
            )
        )
        if distance <= tolerance:
            matches.append(
                {
                    **dict(anchor),
                    "distance_m": distance,
                }
            )
    return sorted(
        matches,
        key=lambda item: (
            float(item["distance_m"])
            / max(float(item["tolerance_m"]), 1e-9),
            str(item.get("kind", "")),
        ),
    )


def executable_candidate_sets_consistent(
    first: Any,
    second: Any,
    *,
    max_translation_m: float = (
        OPERATION_GEOMETRY_TRANSLATION_CHANGE_M
    ),
    max_rotation_rad: float = (
        OPERATION_GEOMETRY_ROTATION_CHANGE_RAD
    ),
) -> bool | None:
    """Compare two observations of the arm-specific executable poses."""

    left = _candidate_groups(first)
    right = _candidate_groups(second)
    if not left and not right:
        return None
    if set(left) != set(right):
        return False
    compared = False
    for key in sorted(left):
        before_group = left[key]
        after_group = right[key]
        if len(before_group) != len(after_group):
            return False
        adjacency: list[list[int]] = []
        for before in before_group:
            compatible: list[int] = []
            for index, after in enumerate(after_group):
                pair = _executable_candidates_consistent(
                    before,
                    after,
                    max_translation_m=max_translation_m,
                    max_rotation_rad=max_rotation_rad,
                )
                if pair is True:
                    compatible.append(index)
                    compared = True
            adjacency.append(compatible)
        if _perfect_matching_count(adjacency, len(after_group)) != 1:
            return False
    return True if compared else None


def _candidate_groups(
    value: Any,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    if not isinstance(value, (list, tuple)):
        return {}
    result: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("arm", "") or "").strip().lower(),
            str(item.get("action_mode", "") or "").strip().lower(),
            str(item.get("target_id", "") or "").strip(),
        )
        if not key[0] or not key[1]:
            continue
        result.setdefault(key, []).append(dict(item))
    return result


def _executable_candidates_consistent(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    max_translation_m: float,
    max_rotation_rad: float,
) -> bool | None:
    compared = False
    for pose_key in _EXECUTABLE_POSE_KEYS:
        before_pose = _pose7(before.get(pose_key))
        after_pose = _pose7(after.get(pose_key))
        if (before_pose is None) != (after_pose is None):
            return False
        if before_pose is None or after_pose is None:
            continue
        compared = True
        translation = math.sqrt(
            sum(
                (before_pose[index] - after_pose[index]) ** 2
                for index in range(3)
            )
        )
        rotation = _quaternion_distance_rad(
            before_pose[3:7],
            after_pose[3:7],
        )
        if (
            translation > float(max_translation_m)
            or rotation is None
            or rotation > float(max_rotation_rad)
        ):
            return False
    return True if compared else None


def _perfect_matching_count(
    adjacency: list[list[int]],
    target_count: int,
) -> int:
    count = 0

    def search(index: int, used: set[int]) -> None:
        nonlocal count
        if count > 1:
            return
        if index >= len(adjacency):
            if len(used) == target_count:
                count += 1
            return
        for target in adjacency[index]:
            if target in used:
                continue
            search(index + 1, {*used, target})

    search(0, set())
    return count


def _pose7(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 7:
        return None
    parsed: list[float] = []
    for item in value[:7]:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        parsed.append(number)
    return parsed


def _quaternion_distance_rad(
    left: list[float],
    right: list[float],
) -> float | None:
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    dot = abs(
        sum(left[index] * right[index] for index in range(4))
        / (left_norm * right_norm)
    )
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _xyz(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    parsed: list[float] = []
    for item in value[:3]:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        parsed.append(number)
    return parsed


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None
