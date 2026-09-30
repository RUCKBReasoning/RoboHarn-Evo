from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterable

from roboharn_evo.agent.perception.position_identity_contract import (
    POSITION_ONLY_ACTION_GEOMETRY_QUARANTINE_LEASE_KEY,
)
from roboharn_evo.agent.recovery.tool_specs import REOBSERVE_SCENE_TOOL

_INFORMATION_GAIN_TOOLS = {
    "retreat_arm",
    "move_to_home",
    "safe_reset_posture",
}


@dataclass(frozen=True, slots=True)
class OperationGeometryRefreshLimits:
    """Bounds for a one-observation refresh authorization."""

    minimum_clearance_motion_m: float = 0.01
    max_step_age: int = 20
    minimum_position_tolerance_m: float = 0.015
    maximum_position_tolerance_m: float = 0.05


class OperationGeometryRefreshContract:
    """Bridge a measured clearance action to Scene Memory recalibration.

    A blocked runtime pose is not reopened by a planner retry or by a new
    candidate ID.  A successful, measured information-gain motion creates a
    one-capture lease scoped to the affected public instance and arm.  Scene
    Memory then remains responsible for two independent clean observations;
    this class never marks geometry verified itself.
    """

    def __init__(
        self,
        limits: OperationGeometryRefreshLimits = (
            OperationGeometryRefreshLimits()
        ),
    ) -> None:
        if limits.minimum_clearance_motion_m <= 0.0:
            raise ValueError(
                "minimum_clearance_motion_m must be positive"
            )
        if limits.max_step_age < 1:
            raise ValueError("max_step_age must be positive")
        self.limits = limits
        self._leases: dict[tuple[str, str], dict[str, Any]] = {}
        self._next_authorization = 1

    def reset(self) -> None:
        self._leases.clear()
        self._next_authorization = 1

    def export_replay_state(self) -> dict[str, Any]:
        return {
            "leases": [
                {
                    "instance_ref": key[0],
                    "arm": key[1],
                    "value": copy.deepcopy(value),
                }
                for key, value in sorted(self._leases.items())
            ],
            "next_authorization": self._next_authorization,
        }

    def restore_replay_state(self, payload: dict[str, Any]) -> None:
        leases = payload.get("leases", [])
        next_authorization = payload.get(
            "next_authorization",
            1,
        )
        if not isinstance(leases, list) or not isinstance(
            next_authorization,
            int,
        ):
            raise TypeError(
                "operation geometry replay state is invalid"
            )
        restored: dict[tuple[str, str], dict[str, Any]] = {}
        for item in leases:
            if not isinstance(item, dict) or not isinstance(
                item.get("value"),
                dict,
            ):
                raise TypeError(
                    "operation geometry replay lease is invalid"
                )
            key = (
                str(item.get("instance_ref", "")),
                str(item.get("arm", "")),
            )
            if not all(key):
                raise ValueError(
                    "operation geometry replay lease key is invalid"
                )
            restored[key] = copy.deepcopy(item["value"])
        self._leases = restored
        self._next_authorization = max(1, next_authorization)

    def authorize_from_results(
        self,
        *,
        calls: Iterable[Any],
        results: Iterable[Any],
        blocked_scopes: Iterable[Any],
        scene_memory: Any,
        env_step: int,
        capture_id: int,
    ) -> list[dict[str, Any]]:
        """Create leases for blocked scopes whose arm measurably cleared."""

        moved_arms = _measured_information_gain_arms(
            calls,
            results,
            minimum_motion_m=(
                self.limits.minimum_clearance_motion_m
            ),
        )
        if not moved_arms:
            return []
        instances = _instances_by_ref(scene_memory)
        grouped_modes: dict[tuple[str, str], set[str]] = {}
        for raw_scope in blocked_scopes:
            scope = raw_scope if isinstance(raw_scope, dict) else {}
            instance_ref = str(
                scope.get("instance_id", "") or ""
            ).strip()
            arm = str(scope.get("arm", "") or "").strip().lower()
            mode = str(
                scope.get("action_mode", "") or ""
            ).strip().lower()
            try:
                blocked_count = int(
                    scope.get("blocked_candidate_count", 0)
                )
            except (TypeError, ValueError):
                blocked_count = 0
            if (
                not instance_ref
                or arm not in moved_arms
                or not mode
                or blocked_count < 1
                or not _scope_has_no_available_candidate(scope)
            ):
                continue
            grouped_modes.setdefault((instance_ref, arm), set()).add(
                mode
            )

        # Scene Memory currently serializes one repair scope per object.  Do
        # not silently let two simultaneous arm leases overwrite each other;
        # require the caller to refresh the arms in separate bounded cycles.
        arms_by_instance: dict[str, set[str]] = {}
        for instance_ref, arm in grouped_modes:
            arms_by_instance.setdefault(instance_ref, set()).add(arm)
        competing_instances = {
            instance_ref
            for instance_ref, arms in arms_by_instance.items()
            if len(arms) > 1
        }

        created: list[dict[str, Any]] = []
        for (instance_ref, arm), modes in sorted(grouped_modes.items()):
            if instance_ref in competing_instances:
                continue
            instance = instances.get(instance_ref.lower())
            if not isinstance(instance, dict):
                continue
            if str(
                instance.get("position_state", "") or ""
            ).strip().lower() == "motion_uncertain":
                continue
            anchor = _xyz(
                instance.get(
                    "last_verified_world_m",
                    instance.get("world_m"),
                )
            )
            if anchor is None:
                continue
            tolerance = _finite_float(
                instance.get("position_tolerance_m")
            )
            if tolerance is None:
                tolerance = 0.035
            tolerance = min(
                self.limits.maximum_position_tolerance_m,
                max(
                    self.limits.minimum_position_tolerance_m,
                    tolerance,
                ),
            )
            authorization_id = (
                f"geometry-refresh-{self._next_authorization:04d}"
            )
            self._next_authorization += 1
            lease = {
                "state": "active",
                "authorization_id": authorization_id,
                "instance_ref": instance_ref,
                "target_world_m": anchor,
                "tolerance_m": tolerance,
                "arm": arm,
                "action_modes": sorted(modes),
                "started_step": int(env_step),
                "expires_step": int(env_step)
                + self.limits.max_step_age,
                "issued_capture_id": int(capture_id),
                "observation_attempts": 0,
                "last_observation_capture_id": None,
                "max_observation_attempts": 1,
                "source": (
                    "blocked_operation_geometry_refresh_after_clearance"
                ),
                "rebuild_action_geometry_from_observations": True,
                POSITION_ONLY_ACTION_GEOMETRY_QUARANTINE_LEASE_KEY: True,
            }
            self._leases[(instance_ref, arm)] = lease
            created.append(dict(lease))
        return created

    def pending_relocation_leases(
        self,
        *,
        scene_memory: Any,
        env_step: int,
        capture_id: int,
    ) -> list[dict[str, Any]]:
        """Consume each external authorization on at most one fresh capture."""

        instances = _instances_by_ref(scene_memory)
        active: list[dict[str, Any]] = []
        for key, raw_lease in list(self._leases.items()):
            lease = dict(raw_lease)
            instance_ref, _ = key
            instance = instances.get(instance_ref.lower())
            if (
                int(env_step) > int(lease["expires_step"])
                or not isinstance(instance, dict)
                or str(
                    instance.get("position_state", "") or ""
                ).strip().lower()
                == "motion_uncertain"
            ):
                self._leases.pop(key, None)
                continue
            if str(
                instance.get("action_geometry_state", "") or ""
            ).strip().lower() == "relocation_pending":
                # Scene Memory now owns the internal two-frame repair lease.
                self._leases.pop(key, None)
                continue
            last_capture = lease.get(
                "last_observation_capture_id"
            )
            if last_capture is None:
                try:
                    issued_capture = int(
                        lease["issued_capture_id"]
                    )
                except (KeyError, TypeError, ValueError):
                    self._leases.pop(key, None)
                    continue
                # The image that preceded the clearance action cannot prove
                # that the clearance exposed new geometry.  Older/replayed
                # captures are equally non-informative.  Keep the lease alive
                # for a future capture, but do not expose it to Scene Memory.
                if int(capture_id) <= issued_capture:
                    continue
                if int(lease.get("observation_attempts", 0)) >= 1:
                    self._leases.pop(key, None)
                    continue
                lease["observation_attempts"] = 1
                lease["last_observation_capture_id"] = int(capture_id)
                self._leases[key] = lease
            elif int(last_capture) != int(capture_id):
                # One authorization can feed exactly one observation update.
                self._leases.pop(key, None)
                continue
            active.append(dict(lease))
        return active


def _measured_information_gain_arms(
    calls: Iterable[Any],
    results: Iterable[Any],
    *,
    minimum_motion_m: float,
) -> set[str]:
    calls_list = list(calls)
    results_list = list(results)
    last_physical_index: dict[str, int] = {}
    for index, call in enumerate(calls_list):
        tool_name = str(
            getattr(call, "tool_name", "") or ""
        ).strip()
        if tool_name == REOBSERVE_SCENE_TOOL:
            continue
        result = (
            results_list[index]
            if index < len(results_list)
            else None
        )
        details = getattr(result, "details", {}) or {}
        if details.get("skipped") is True:
            continue
        args = getattr(call, "args", {}) or {}
        affected_arms = _result_arms(details, args)
        if not affected_arms:
            # An unscoped physical result is unsafe to treat as unrelated to
            # either arm.  This keeps a later, malformed action from silently
            # validating an earlier clearance.
            affected_arms = {"left", "right"}
        for arm in affected_arms:
            last_physical_index[arm] = index

    moved: set[str] = set()
    for index, call in enumerate(calls_list):
        tool_name = str(
            getattr(call, "tool_name", "") or ""
        ).strip()
        if tool_name not in _INFORMATION_GAIN_TOOLS:
            continue
        if index >= len(results_list):
            continue
        result = results_list[index]
        if getattr(result, "success", False) is not True:
            continue
        details = getattr(result, "details", {}) or {}
        args = getattr(call, "args", {}) or {}
        for arm in _result_arms(details, args):
            displacement = _arm_displacement_m(details, arm)
            if (
                displacement is not None
                and displacement >= minimum_motion_m
                and last_physical_index.get(arm) == index
            ):
                moved.add(arm)
    return moved


def _result_arms(details: dict[str, Any], args: dict[str, Any]) -> set[str]:
    values: list[Any] = []
    selected = details.get("selected_arms")
    if isinstance(selected, (list, tuple, set)):
        values.extend(selected)
    values.append(details.get("arm", args.get("arm")))
    normalized: set[str] = set()
    for value in values:
        arm = str(value or "").strip().lower()
        if arm == "both":
            normalized.update({"left", "right"})
        elif arm in {"left", "right"}:
            normalized.add(arm)
    return normalized


def _arm_displacement_m(
    details: dict[str, Any],
    arm: str,
) -> float | None:
    mapping = details.get("observed_displacements_m")
    if isinstance(mapping, dict):
        value = _finite_float(mapping.get(arm))
        if value is not None:
            return abs(value)
    if str(details.get("arm", "") or "").strip().lower() == arm:
        value = _finite_float(
            details.get("observed_displacement_m")
        )
        if value is not None:
            return abs(value)
    return None


def _scope_has_no_available_candidate(scope: dict[str, Any]) -> bool:
    """Return whether the public scope is exhausted, fail closed on conflict."""

    if "available_candidate_count" in scope:
        try:
            return int(scope["available_candidate_count"]) == 0
        except (TypeError, ValueError):
            return False
    return scope.get("all_candidates_blocked") is True


def _instances_by_ref(scene_memory: Any) -> dict[str, dict[str, Any]]:
    scene = scene_memory if isinstance(scene_memory, dict) else {}
    indexed: dict[str, dict[str, Any]] = {}
    for instance in scene.get("instances", []) or []:
        if not isinstance(instance, dict):
            continue
        for field in (
            "instance_id",
            "track_id",
            "oracle_id",
            "oracle_source_path",
        ):
            ref = str(instance.get(field, "") or "").strip().lower()
            if ref:
                indexed.setdefault(ref, instance)
    return indexed


def _xyz(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        parsed = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return parsed if all(math.isfinite(item) for item in parsed) else None


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


__all__ = [
    "OperationGeometryRefreshContract",
    "OperationGeometryRefreshLimits",
]
