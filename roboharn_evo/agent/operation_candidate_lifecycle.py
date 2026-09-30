from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, cast

from roboharn_evo.agent.grounded_target_pose_contract import (
    resolve_grounded_target_pose,
)
from roboharn_evo.agent.operation_candidates import (
    materialize_operation_candidate,
    normalize_grounded_point_key,
)
from roboharn_evo.agent.perception.position_identity_contract import (
    OPERATION_GEOMETRY_ROTATION_CHANGE_RAD,
    OPERATION_GEOMETRY_TRANSLATION_CHANGE_M,
)

OperationCandidateScope = tuple[str, str, str, str]
_POSE_KEYS = ("object_contact_pose", "tcp_pose", "ee_target_pose", "approach_pose")
_ARGUMENT_UNSET = object()


def _replay_scope(value: Any) -> OperationCandidateScope:
    if not isinstance(value, list) or len(value) != 4:
        raise TypeError(
            "operation-candidate replay scope is invalid"
        )
    scope = cast(
        OperationCandidateScope,
        tuple(str(item) for item in value),
    )
    if any(not item for item in scope[:3]):
        raise ValueError(
            "operation-candidate replay scope is invalid"
        )
    return scope


@dataclass(frozen=True, slots=True)
class OperationCandidateLifecycleLimits:
    """Bounds for failure memory attached to executable pose geometry."""

    failure_threshold: int = 2
    progress_epsilon_m: float = 0.002
    geometry_translation_change_m: float = (
        OPERATION_GEOMETRY_TRANSLATION_CHANGE_M
    )
    geometry_rotation_change_rad: float = (
        OPERATION_GEOMETRY_ROTATION_CHANGE_RAD
    )
    max_versions_per_scope: int = 8


@dataclass(slots=True)
class _FailureRecord:
    record_id: int
    scope: OperationCandidateScope
    candidate_aliases: set[str]
    failure_count: int
    last_target_error_m: float | None
    geometry: dict[str, Any]
    geometry_revision: str
    blocked: bool
    failure_reason: str
    env_step: int | None
    geometry_env_step: int | None
    geometry_capture_id: int | None
    point_key: str
    execution_spec: dict[str, Any]
    execution_spec_revision: str
    executed_target_pose: list[float] | None


@dataclass(slots=True)
class _CurrentCandidateResolution:
    candidate: dict[str, Any]
    candidate_id: str
    scope: OperationCandidateScope
    records: list[_FailureRecord]
    current_by_record: dict[int, dict[str, Any]]
    current_target_pose_by_record: dict[int, list[float] | None]
    equivalent: list[_FailureRecord]
    trusted: bool
    fresh: bool
    known_alternate: bool

    def blocked_record(self) -> _FailureRecord | None:
        if self.equivalent:
            return min(
                self.equivalent,
                key=lambda item: geometry_distance_score(
                    item.geometry,
                    self.current_by_record[item.record_id],
                ),
            )
        if self.records and not self.known_alternate and (
            not self.trusted or not self.fresh
        ):
            return self.records[0]
        return None


class OperationCandidateLifecycle:
    """Bound retries to physical geometry rather than a reusable slot ID.

    Historical blocked versions are retained so an old pose cannot regain a
    retry budget after a retreat.  Planner output is reconciled against the
    candidates that exist now, so superseded or renamed slots never leak as
    current state.  Revalidation requires both accepted, fresh, materially
    changed candidate geometry and a materially changed final executor target.
    """

    def __init__(
        self,
        limits: OperationCandidateLifecycleLimits = OperationCandidateLifecycleLimits(),
    ) -> None:
        if limits.failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        if limits.progress_epsilon_m < 0.0:
            raise ValueError("progress_epsilon_m must be nonnegative")
        if limits.geometry_translation_change_m <= 0.0:
            raise ValueError("geometry_translation_change_m must be positive")
        if limits.geometry_rotation_change_rad <= 0.0:
            raise ValueError("geometry_rotation_change_rad must be positive")
        if limits.max_versions_per_scope < 1:
            raise ValueError("max_versions_per_scope must be positive")
        self.limits = limits
        self._records: dict[OperationCandidateScope, list[_FailureRecord]] = {}
        self._reported_revalidations: set[tuple[int, str]] = set()
        self._known_candidates: dict[
            tuple[OperationCandidateScope, str], dict[str, Any]
        ] = {}
        self._next_record_id = 1

    def reset(self) -> None:
        self._records.clear()
        self._reported_revalidations.clear()
        self._known_candidates.clear()
        self._next_record_id = 1

    def export_replay_state(self) -> dict[str, Any]:
        records = []
        for _scope, values in sorted(self._records.items()):
            for record in values:
                records.append(
                    {
                        "record_id": record.record_id,
                        "scope": list(record.scope),
                        "candidate_aliases": sorted(
                            record.candidate_aliases
                        ),
                        "failure_count": record.failure_count,
                        "last_target_error_m": (
                            record.last_target_error_m
                        ),
                        "geometry": copy.deepcopy(record.geometry),
                        "geometry_revision": record.geometry_revision,
                        "blocked": record.blocked,
                        "failure_reason": record.failure_reason,
                        "env_step": record.env_step,
                        "geometry_env_step": record.geometry_env_step,
                        "geometry_capture_id": (
                            record.geometry_capture_id
                        ),
                        "point_key": record.point_key,
                        "execution_spec": copy.deepcopy(
                            record.execution_spec
                        ),
                        "execution_spec_revision": (
                            record.execution_spec_revision
                        ),
                        "executed_target_pose": copy.deepcopy(
                            record.executed_target_pose
                        ),
                    }
                )
        return {
            "records": records,
            "reported_revalidations": [
                [record_id, revision]
                for record_id, revision in sorted(
                    self._reported_revalidations
                )
            ],
            "known_candidates": [
                {
                    "scope": list(scope),
                    "candidate_id": candidate_id,
                    "candidate": copy.deepcopy(candidate),
                }
                for (scope, candidate_id), candidate in sorted(
                    self._known_candidates.items()
                )
            ],
            "next_record_id": self._next_record_id,
        }

    def restore_replay_state(self, payload: dict[str, Any]) -> None:
        records = payload.get("records", [])
        reported = payload.get("reported_revalidations", [])
        known = payload.get("known_candidates", [])
        next_record_id = payload.get("next_record_id", 1)
        if (
            not isinstance(records, list)
            or not isinstance(reported, list)
            or not isinstance(known, list)
            or not isinstance(next_record_id, int)
        ):
            raise TypeError(
                "operation-candidate replay state is invalid"
            )
        restored_records: dict[
            OperationCandidateScope,
            list[_FailureRecord],
        ] = {}
        for item in records:
            if not isinstance(item, dict):
                raise TypeError(
                    "operation-candidate replay record is invalid"
                )
            scope = _replay_scope(item.get("scope"))
            record = _FailureRecord(
                record_id=int(item["record_id"]),
                scope=scope,
                candidate_aliases={
                    str(value)
                    for value in item.get("candidate_aliases", [])
                },
                failure_count=int(item["failure_count"]),
                last_target_error_m=item.get("last_target_error_m"),
                geometry=copy.deepcopy(item.get("geometry", {})),
                geometry_revision=str(
                    item.get("geometry_revision", "")
                ),
                blocked=bool(item.get("blocked", False)),
                failure_reason=str(
                    item.get("failure_reason", "")
                ),
                env_step=item.get("env_step"),
                geometry_env_step=item.get("geometry_env_step"),
                geometry_capture_id=item.get("geometry_capture_id"),
                point_key=str(item.get("point_key", "")),
                execution_spec=copy.deepcopy(
                    item.get("execution_spec", {})
                ),
                execution_spec_revision=str(
                    item.get("execution_spec_revision", "")
                ),
                executed_target_pose=copy.deepcopy(
                    item.get("executed_target_pose")
                ),
            )
            restored_records.setdefault(scope, []).append(record)
        restored_reported: set[tuple[int, str]] = set()
        for item in reported:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(
                    "operation-candidate replay revalidation is invalid"
                )
            restored_reported.add((int(item[0]), str(item[1])))
        restored_known: dict[
            tuple[OperationCandidateScope, str],
            dict[str, Any],
        ] = {}
        for item in known:
            if not isinstance(item, dict) or not isinstance(
                item.get("candidate"),
                dict,
            ):
                raise TypeError(
                    "operation-candidate replay candidate is invalid"
                )
            scope = _replay_scope(item.get("scope"))
            candidate_id = str(item.get("candidate_id", ""))
            if not candidate_id:
                raise ValueError(
                    "operation-candidate replay candidate ID is invalid"
                )
            restored_known[(scope, candidate_id)] = copy.deepcopy(
                item["candidate"]
            )
        self._records = restored_records
        self._reported_revalidations = restored_reported
        self._known_candidates = restored_known
        self._next_record_id = max(1, next_record_id)

    def bind_attempt(
        self,
        *,
        instance_id: Any,
        candidate: Any,
        instance: Any = None,
        point_key: Any = None,
        offset_xyz: Any = _ARGUMENT_UNSET,
        preserve_height: Any = False,
        target_quat_wxyz: Any = None,
        current_pose: Any = None,
        dispatch_env_step: Any = None,
        observation_generation: Any = None,
        observation_capture_id: Any = None,
    ) -> dict[str, Any]:
        """Freeze the candidate stage, planner modifiers, and provenance."""

        value = _candidate_with_instance_provenance(candidate, instance)
        candidate_id = str(value.get("candidate_id", "") or "").strip()
        if not candidate_id:
            return {}
        normalized_point_key = _execution_point_key(
            point_key,
            action_mode=value.get("action_mode"),
        )
        execution_spec = candidate_execution_spec(
            value,
            point_key=normalized_point_key,
            offset_xyz=offset_xyz,
            preserve_height=preserve_height,
            target_quat_wxyz=target_quat_wxyz,
        )
        if not execution_spec:
            return {}
        geometry = candidate_execution_geometry(
            value,
            execution_spec=execution_spec,
        )
        if not geometry.get("poses"):
            return {}
        geometry_step, geometry_capture = candidate_geometry_provenance(value)
        resolved_target_pose = resolve_candidate_target_pose(
            candidate=value,
            instance=instance,
            execution_spec=execution_spec,
            current_pose=current_pose,
        )
        return {
            "version": 1,
            "scope": list(candidate_scope(instance_id, value)),
            "candidate_id": candidate_id,
            "point_key": normalized_point_key,
            "execution_spec": execution_spec,
            "execution_spec_revision": _mapping_revision(
                execution_spec
            ),
            "geometry": geometry,
            "geometry_revision": candidate_geometry_revision(geometry),
            "candidate_geometry_revision": candidate_geometry_revision(
                candidate_geometry(value)
            ),
            "dispatch_env_step": _integer(dispatch_env_step),
            "observation_generation": _integer(observation_generation),
            "observation_capture_id": _integer(observation_capture_id),
            "geometry_env_step": geometry_step,
            "geometry_capture_id": geometry_capture,
            "resolved_target_pose": resolved_target_pose,
        }

    def record_success(
        self,
        attempt: Any,
        *,
        executed_target_pose: Any = None,
    ) -> None:
        parsed = _attempt(attempt)
        if parsed is None:
            return
        scope, _, point_key, execution_spec, geometry = parsed
        success_target_pose = _pose7(executed_target_pose)
        if success_target_pose is None:
            success_target_pose = _pose7(
                attempt.get("resolved_target_pose")
            )
        equivalent = [
            record
            for record in self._equivalent_records(
                scope,
                point_key,
                geometry,
            )
            if success_target_pose is not None
            and record.executed_target_pose is not None
            and not target_pose_changed(
                record.executed_target_pose,
                success_target_pose,
                limits=self.limits,
            )
        ]
        if not equivalent:
            return
        removed_ids = {record.record_id for record in equivalent}
        self._records[scope] = [
            record for record in self._records.get(scope, []) if record.record_id not in removed_ids
        ]
        if not self._records[scope]:
            self._records.pop(scope, None)
        self._reported_revalidations = {
            item for item in self._reported_revalidations if item[0] not in removed_ids
        }

    def record_failure(
        self,
        attempt: Any,
        *,
        target_error_m: Any,
        target_id: Any = "",  # accepted for API compatibility; frozen scope is authoritative
        failure_reason: Any = "no_progress",
        env_step: Any = None,
        observation_generation: Any = None,
        observation_capture_id: Any = None,
        executed_target_pose: Any = None,
    ) -> dict[str, Any] | None:
        parsed = _attempt(attempt)
        if parsed is None:
            return None
        scope, candidate_id, point_key, execution_spec, geometry = parsed
        equivalent = self._equivalent_records(
            scope,
            point_key,
            geometry,
        )
        previous = min(equivalent, key=lambda item: geometry_distance_score(item.geometry, geometry), default=None)
        error = _finite_float(target_error_m)
        count = 1 if previous is None else previous.failure_count + 1
        was_blocked = any(record.blocked for record in equivalent)
        if (
            not was_blocked
            and previous is not None
            and error is not None
            and previous.last_target_error_m is not None
            and error < previous.last_target_error_m - self.limits.progress_epsilon_m
        ):
            count = 1
        aliases = {candidate_id}
        for record in equivalent:
            aliases.update(record.candidate_aliases)
        actual_target_pose = _pose7(executed_target_pose)
        if actual_target_pose is None:
            actual_target_pose = _pose7(
                attempt.get("resolved_target_pose")
            )
        record = _FailureRecord(
            record_id=self._next_record_id,
            scope=scope,
            candidate_aliases=aliases,
            failure_count=count,
            last_target_error_m=error,
            geometry=geometry,
            geometry_revision=candidate_geometry_revision(geometry),
            blocked=(
                was_blocked
                or count >= self.limits.failure_threshold
            ),
            failure_reason=str(failure_reason or "no_progress").strip() or "no_progress",
            env_step=_first_integer(env_step, attempt.get("dispatch_env_step")),
            geometry_env_step=_integer(attempt.get("geometry_env_step")),
            geometry_capture_id=_integer(attempt.get("geometry_capture_id")),
            point_key=point_key,
            execution_spec=execution_spec,
            execution_spec_revision=_mapping_revision(
                execution_spec
            ),
            executed_target_pose=actual_target_pose,
        )
        self._next_record_id += 1
        removed_ids = {item.record_id for item in equivalent}
        records = [item for item in self._records.get(scope, []) if item.record_id not in removed_ids]
        records.append(record)
        evicted = records[:-self.limits.max_versions_per_scope]
        self._records[scope] = records[-self.limits.max_versions_per_scope :]
        stale_ids = removed_ids | {item.record_id for item in evicted}
        self._reported_revalidations = {
            item for item in self._reported_revalidations if item[0] not in stale_ids
        }
        if not record.blocked or was_blocked:
            return None
        return self._trace_payload(record, candidate_id=candidate_id)

    def blocked_candidate_ids(
        self,
        *,
        instance_id: Any,
        arm: Any,
        action_mode: Any,
        candidates: Iterable[Any],
        instance: Any = None,
        requested_target_id: Any = None,
        point_key: Any = None,
        offset_xyz: Any = _ARGUMENT_UNSET,
        preserve_height: Any = False,
        target_quat_wxyz: Any = None,
        current_pose: Any = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Resolve blocks against the current accepted candidate versions."""

        arm_name = str(arm or "").strip().lower()
        mode = str(action_mode or "").strip().lower()
        requested_target = str(requested_target_id or "").strip()
        blocked: list[str] = []
        revalidated: list[dict[str, Any]] = []
        for raw_candidate in _matching_candidates(candidates, arm_name, mode, requested_target):
            resolved = self._resolve_current_candidate(
                instance_id=instance_id,
                raw_candidate=raw_candidate,
                instance=instance,
                point_key=point_key,
                offset_xyz=offset_xyz,
                preserve_height=preserve_height,
                target_quat_wxyz=target_quat_wxyz,
                current_pose=current_pose,
            )
            candidate = resolved.candidate
            candidate_id = resolved.candidate_id
            scope = resolved.scope
            records = resolved.records
            known_key = (scope, candidate_id)
            geometry = candidate_geometry(candidate)
            if not records:
                self._remember_candidate(known_key, geometry)
                continue
            if resolved.equivalent:
                for record in resolved.equivalent:
                    record.candidate_aliases.add(candidate_id)
                blocked.append(candidate_id)
                continue
            if resolved.known_alternate:
                continue
            if not resolved.trusted or not resolved.fresh:
                blocked.append(candidate_id)
                continue
            aliases = [record for record in records if candidate_id in record.candidate_aliases]
            if not aliases:
                self._remember_candidate(known_key, geometry)
                continue  # a trusted, genuinely different alternate candidate
            for record in aliases:
                current_geometry = resolved.current_by_record[
                    record.record_id
                ]
                transition = (record.record_id, candidate_id)
                if transition in self._reported_revalidations:
                    continue
                self._reported_revalidations.add(transition)
                revalidated.append(
                    {
                        **self._trace_payload(record, candidate_id=candidate_id),
                        "status": "revalidated_after_geometry_change",
                        "prior_geometry_revision": record.geometry_revision,
                        "geometry_revision": candidate_geometry_revision(
                            current_geometry
                        ),
                        **candidate_geometry_change_metrics(
                            record.geometry,
                            current_geometry,
                        ),
                        **target_pose_change_metrics(
                            record.executed_target_pose,
                            resolved.current_target_pose_by_record.get(
                                record.record_id
                            ),
                        ),
                    }
                )
            self._remember_candidate(known_key, geometry)
        return sorted(set(blocked)), revalidated

    def planner_payload(self, instances: Iterable[Any]) -> list[dict[str, Any]]:
        """Report public scope counts without exposing runtime candidate IDs."""

        grouped: dict[
            OperationCandidateScope,
            dict[str, Any],
        ] = {}
        for raw_instance in instances:
            if not isinstance(raw_instance, dict):
                continue
            instance_id = str(raw_instance.get("instance_id", "") or "").strip()
            for raw_candidate in raw_instance.get("operation_pose_candidates", []) or []:
                if not isinstance(raw_candidate, dict):
                    continue
                try:
                    resolved = self._resolve_current_candidate(
                        instance_id=instance_id,
                        raw_candidate=raw_candidate,
                        instance=raw_instance,
                    )
                except ValueError:
                    continue
                group = grouped.setdefault(
                    resolved.scope,
                    {
                        "total": 0,
                        "blocked_records": [],
                    },
                )
                group["total"] += 1
                blocked_record = resolved.blocked_record()
                if blocked_record is not None:
                    group["blocked_records"].append(blocked_record)

        payload: list[dict[str, Any]] = []
        for group in grouped.values():
            blocked_records = group["blocked_records"]
            if not blocked_records:
                continue
            record = max(
                blocked_records,
                key=lambda item: (
                    item.failure_count,
                    item.record_id,
                ),
            )
            blocked_count = len(blocked_records)
            available_count = max(0, int(group["total"]) - blocked_count)
            public = self._public_payload(record)
            public.update(
                {
                    "blocked_candidate_count": blocked_count,
                    "available_candidate_count": available_count,
                    "runtime_will_select_next": available_count > 0,
                    "all_candidates_blocked": available_count == 0,
                    "status": (
                        "operation_candidate_subset_blocked"
                        if available_count > 0
                        else "all_operation_candidates_blocked"
                    ),
                }
            )
            payload.append(public)
        return sorted(
            payload,
            key=lambda item: (
                str(item.get("instance_id", "")),
                str(item.get("arm", "")),
                str(item.get("action_mode", "")),
                str(item.get("target_id", "")),
            ),
        )

    def _resolve_current_candidate(
        self,
        *,
        instance_id: Any,
        raw_candidate: Any,
        instance: Any,
        point_key: Any = None,
        offset_xyz: Any = _ARGUMENT_UNSET,
        preserve_height: Any = False,
        target_quat_wxyz: Any = None,
        current_pose: Any = None,
    ) -> _CurrentCandidateResolution:
        candidate = _candidate_with_instance_provenance(
            raw_candidate,
            instance,
        )
        candidate_id = str(
            candidate.get("candidate_id", "") or ""
        ).strip()
        scope = candidate_scope(instance_id, candidate)
        records = [
            item
            for item in self._records.get(scope, [])
            if item.blocked
        ]
        current_by_record = {
            record.record_id: candidate_execution_geometry(
                candidate,
                execution_spec=record.execution_spec,
            )
            for record in records
        }
        prospective_spec = candidate_execution_spec(
            candidate,
            point_key=point_key,
            offset_xyz=offset_xyz,
            preserve_height=preserve_height,
            target_quat_wxyz=target_quat_wxyz,
        )
        prospective_target_pose = resolve_candidate_target_pose(
            candidate=candidate,
            instance=instance,
            execution_spec=prospective_spec,
            current_pose=current_pose,
        )
        current_target_pose_by_record = {
            record.record_id: prospective_target_pose
            for record in records
        }
        equivalent = [
            record
            for record in records
            if not current_by_record[record.record_id].get("poses")
            or not candidate_geometry_changed(
                record.geometry,
                current_by_record[record.record_id],
                limits=self.limits,
            )
            or not target_pose_changed(
                record.executed_target_pose,
                current_target_pose_by_record[record.record_id],
                limits=self.limits,
            )
        ]
        geometry = candidate_geometry(candidate)
        known_geometry = self._known_candidates.get(
            (scope, candidate_id)
        )
        known_alternate = bool(
            not equivalent
            and known_geometry is not None
            and not any(
                candidate_id in record.candidate_aliases
                for record in records
            )
            and not candidate_geometry_changed(
                known_geometry,
                geometry,
                limits=self.limits,
            )
        )
        return _CurrentCandidateResolution(
            candidate=candidate,
            candidate_id=candidate_id,
            scope=scope,
            records=records,
            current_by_record=current_by_record,
            current_target_pose_by_record=(
                current_target_pose_by_record
            ),
            equivalent=equivalent,
            trusted=operation_geometry_is_trusted(
                instance,
                candidate,
            ),
            fresh=all(
                candidate_geometry_is_fresh(
                    record,
                    candidate,
                    instance,
                )
                for record in records
            ),
            known_alternate=known_alternate,
        )

    def _remember_candidate(
        self,
        key: tuple[OperationCandidateScope, str],
        geometry: dict[str, Any],
    ) -> None:
        self._known_candidates[key] = geometry
        scope = key[0]
        scoped_keys = [item for item in self._known_candidates if item[0] == scope]
        for stale_key in scoped_keys[: -2 * self.limits.max_versions_per_scope]:
            self._known_candidates.pop(stale_key, None)

    def _equivalent_records(
        self,
        scope: OperationCandidateScope,
        point_key: str,
        geometry: dict[str, Any],
    ) -> list[_FailureRecord]:
        return [
            record
            for record in self._records.get(scope, [])
            if record.point_key == point_key
            and not candidate_geometry_changed(
                record.geometry,
                geometry,
                limits=self.limits,
            )
        ]

    @staticmethod
    def _public_payload(record: _FailureRecord) -> dict[str, Any]:
        instance_id, arm, mode, target_id = record.scope
        payload: dict[str, Any] = {
            "instance_id": instance_id,
            "arm": arm,
            "action_mode": mode,
            "failure_count": record.failure_count,
            "last_target_error_m": record.last_target_error_m,
            "status": "candidate_blocked_after_repeated_no_progress",
        }
        if mode == "place" and target_id:
            payload["target_id"] = target_id
            payload["status"] = "place_target_blocked_after_repeated_no_progress"
        return payload

    @staticmethod
    def _trace_payload(record: _FailureRecord, *, candidate_id: str) -> dict[str, Any]:
        payload = OperationCandidateLifecycle._public_payload(record)
        payload.update(
            {
                "candidate_id": candidate_id,
                "failure_reason": record.failure_reason,
                "point_key": record.point_key,
                "execution_spec_revision": (
                    record.execution_spec_revision
                ),
                "geometry_revision": record.geometry_revision,
            }
        )
        if record.env_step is not None:
            payload["blocked_env_step"] = record.env_step
        if record.geometry_env_step is not None:
            payload["geometry_env_step"] = record.geometry_env_step
        if record.geometry_capture_id is not None:
            payload["geometry_capture_id"] = record.geometry_capture_id
        if record.executed_target_pose is not None:
            payload["executed_target_pose"] = list(
                record.executed_target_pose
            )
        return payload


def candidate_scope(instance_id: Any, candidate: Any) -> OperationCandidateScope:
    value = candidate if isinstance(candidate, dict) else {}
    normalized_instance = str(instance_id or "").strip()
    arm = str(value.get("arm", "") or "").strip().lower()
    mode = str(value.get("action_mode", "") or "").strip().lower()
    target_id = str(value.get("target_id", "") or "").strip() if mode == "place" else ""
    if not normalized_instance or arm not in {"left", "right"} or not mode:
        raise ValueError("operation candidate scope is incomplete")
    return normalized_instance, arm, mode, target_id


def operation_geometry_is_trusted(instance: Any, candidate: Any = None) -> bool:
    value = instance if isinstance(instance, dict) else {}
    operation = candidate if isinstance(candidate, dict) else {}
    if str(operation.get("action_mode", "") or "").strip().lower() == "place":
        holding_status = str(
            operation.get("holding_status", "") or ""
        ).strip().lower()
        transport_policy = str(
            operation.get("grasp_transport_policy", "") or ""
        ).strip().lower()
        trusted_holding = bool(
            holding_status == "verified"
            or (
                holding_status == "provisional_evidence_only"
                and transport_policy == "evidence_only"
            )
        )
        return bool(
            str(operation.get("geometry_source", "") or "").strip()
            == "runtime_dynamic_place_geometry"
            and trusted_holding
            and operation.get("valid") is True
            and operation.get("support_valid") is True
            and operation.get("free") is True
            and operation.get("reachable_estimate", True) is not False
            and str(operation.get("target_id", "") or "").strip()
        )
    status = str(value.get("status", "visible") or "").strip().lower()
    position_state = str(
        value.get("position_state", "current_verified" if status == "visible" else "") or ""
    ).strip().lower()
    geometry_state = str(value.get("action_geometry_state", "verified") or "").strip().lower()
    return bool(
        status == "visible"
        and position_state == "current_verified"
        and geometry_state == "verified"
        and value.get("actionable", True) is not False
    )


def candidate_geometry_is_fresh(
    record: _FailureRecord, candidate: dict[str, Any], instance: Any
) -> bool:
    if str(candidate.get("action_mode", "") or "").strip().lower() == "place":
        return True  # dynamic place geometry is revalidated at generation and dispatch
    step, capture = candidate_geometry_provenance(candidate)
    if record.geometry_capture_id is not None and capture is not None:
        return capture > record.geometry_capture_id
    if record.geometry_env_step is not None and step is not None:
        return step > record.geometry_env_step
    value = instance if isinstance(instance, dict) else {}
    if isinstance(
        value.get("operation_pose_candidate_provenance"),
        dict,
    ):
        # Once Scene Memory publishes the authoritative per-candidate map,
        # omission is meaningful: this candidate has no accepted fresh
        # geometry.  Falling back to the instance timestamp would let a
        # metadata-only observation unlock an old failed pose.
        return False
    verified_step = _integer(value.get("last_verified_step"))
    return bool(
        verified_step is not None
        and record.geometry_env_step is not None
        and verified_step > record.geometry_env_step
    )


def candidate_geometry_provenance(candidate: Any) -> tuple[int | None, int | None]:
    value = candidate if isinstance(candidate, dict) else {}
    step = _integer(
        value.get("_geometry_observation_env_step", value.get("_geometry_generation_env_step"))
    )
    capture = _integer(value.get("_geometry_observation_capture_id"))
    return step, capture


def _candidate_with_instance_provenance(
    candidate: Any,
    instance: Any,
) -> dict[str, Any]:
    value = dict(candidate) if isinstance(candidate, dict) else {}
    scene_instance = instance if isinstance(instance, dict) else {}
    candidate_id = str(value.get("candidate_id", "") or "").strip()
    provenance_map = scene_instance.get(
        "operation_pose_candidate_provenance"
    )
    if isinstance(provenance_map, dict):
        for key in (
            "_geometry_observation_env_step",
            "_geometry_generation_env_step",
            "_geometry_observation_capture_id",
        ):
            value.pop(key, None)
    provenance = (
        provenance_map.get(candidate_id)
        if isinstance(provenance_map, dict)
        else None
    )
    if isinstance(provenance, dict):
        step = _integer(
            provenance.get("geometry_observation_env_step")
        )
        capture = _integer(
            provenance.get("geometry_observation_capture_id")
        )
        if step is not None:
            value["_geometry_observation_env_step"] = step
        if capture is not None:
            value["_geometry_observation_capture_id"] = capture
    return value


def _execution_point_key(value: Any, *, action_mode: Any) -> str:
    raw = str(value or "").strip()
    if raw:
        return normalize_grounded_point_key(raw)
    mode = str(action_mode or "").strip().lower()
    return {
        "contact": "contact_world_m",
        "place": "place_world_m",
    }.get(mode, "grasp_world_m")


def candidate_execution_spec(
    candidate: Any,
    *,
    point_key: Any,
    offset_xyz: Any = _ARGUMENT_UNSET,
    preserve_height: Any = False,
    target_quat_wxyz: Any = None,
) -> dict[str, Any]:
    """Freeze planner modifiers without duplicating adapter pose resolution."""

    value = candidate if isinstance(candidate, dict) else {}
    normalized_point = _execution_point_key(
        point_key,
        action_mode=value.get("action_mode"),
    )
    if _candidate_pose_for_point(value, normalized_point) is None:
        return {}
    offset = (
        [0.0, 0.0, 0.0]
        if offset_xyz is _ARGUMENT_UNSET
        else _xyz(offset_xyz)
    )
    if offset is None or not isinstance(preserve_height, bool):
        return {}
    spec: dict[str, Any] = {
        "point_key": normalized_point,
        "offset_xyz": offset,
        "preserve_height": bool(preserve_height),
    }
    raw_quaternion = target_quat_wxyz
    quaternion_mode = (
        str(raw_quaternion).strip().lower()
        if isinstance(raw_quaternion, str)
        else ""
    )
    if raw_quaternion is None:
        spec["quaternion_mode"] = "grounded_default"
    elif isinstance(raw_quaternion, str):
        if quaternion_mode not in {
            "",
            "preserve",
            "current",
            "grounded",
            "scene",
            "affordance",
            "auto",
        }:
            return {}
        spec["quaternion_mode"] = quaternion_mode or "current"
    else:
        explicit_quaternion = _quat4(raw_quaternion)
        if explicit_quaternion is None:
            return {}
        spec["quaternion_mode"] = "explicit"
        spec["explicit_quat_wxyz"] = explicit_quaternion
    return spec


def candidate_execution_geometry(
    candidate: Any,
    *,
    execution_spec: Any,
) -> dict[str, Any]:
    """Return only the candidate pose used by the attempted grounded stage.

    The shared grounded-target contract remains the sole authority for
    applying offsets, preserve-height behavior, and live robot orientation.
    This function retains the independently measured candidate-stage evidence.
    """

    value = candidate if isinstance(candidate, dict) else {}
    spec = execution_spec if isinstance(execution_spec, dict) else {}
    point_key = _execution_point_key(
        spec.get("point_key"),
        action_mode=value.get("action_mode"),
    )
    pose = _candidate_pose_for_point(value, point_key)
    if pose is None:
        return {"poses": {}, "approach_direction": None}
    return {
        "poses": {"candidate_stage_pose": pose},
        "approach_direction": None,
    }


def resolve_candidate_target_pose(
    *,
    candidate: Any,
    instance: Any,
    execution_spec: Any,
    current_pose: Any = None,
) -> list[float] | None:
    """Resolve the final runtime target through the shared pose contract.

    Candidate geometry and the command's modifiers are deliberately kept as
    separate evidence.  A blocked attempt is revalidated only when fresh
    Scene Memory geometry changes *and* the pose that would actually be sent
    to the executor changes.  This prevents an inverse ``offset_xyz`` (or a
    live-current quaternion) from making unchanged physical work look new.
    """

    value = candidate if isinstance(candidate, dict) else {}
    scene_instance = instance if isinstance(instance, dict) else {}
    spec = execution_spec if isinstance(execution_spec, dict) else {}
    if not value or not spec:
        return None
    materialized = materialize_operation_candidate(
        scene_instance,
        value,
    )
    quaternion_mode = str(
        spec.get("quaternion_mode", "") or ""
    ).strip().lower()
    if quaternion_mode == "grounded_default":
        quaternion: Any = None
    elif quaternion_mode == "explicit":
        quaternion = spec.get("explicit_quat_wxyz")
    else:
        quaternion = quaternion_mode
    resolution = resolve_grounded_target_pose(
        instance=materialized,
        point_key=spec.get("point_key"),
        offset_xyz=spec.get("offset_xyz"),
        preserve_height=spec.get("preserve_height", False),
        target_quat_wxyz=quaternion,
        current_pose=current_pose,
    )
    if not resolution.success or resolution.target_pose is None:
        return None
    return _pose7(resolution.target_pose.tolist())


def _candidate_pose_for_point(
    candidate: dict[str, Any],
    point_key: str,
) -> list[float] | None:
    if point_key == "approach_world_m":
        return _pose7(candidate.get("approach_pose"))
    if point_key in {
        "grasp_world_m",
        "contact_world_m",
        "place_world_m",
    }:
        return _pose7(candidate.get("ee_target_pose"))
    return None


def candidate_geometry(candidate: Any) -> dict[str, Any]:
    value = candidate if isinstance(candidate, dict) else {}
    return {
        "poses": {
            key: pose
            for key in _POSE_KEYS
            if (pose := _pose7(value.get(key))) is not None
        },
        "approach_direction": _xyz(value.get("approach_direction")),
    }


def candidate_geometry_revision(geometry: Any) -> str:
    encoded = json.dumps(
        _canonical_geometry(geometry), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=8).hexdigest()


def _mapping_revision(value: Any) -> str:
    encoded = json.dumps(
        value if isinstance(value, dict) else {},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=8).hexdigest()


def candidate_geometry_changed(
    previous: Any,
    current: Any,
    *,
    limits: OperationCandidateLifecycleLimits,
) -> bool:
    metrics = candidate_geometry_change_metrics(previous, current)
    return bool(
        (metrics["max_pose_translation_change_m"] or 0.0)
        >= limits.geometry_translation_change_m
        or (metrics["max_pose_rotation_change_rad"] or 0.0)
        >= limits.geometry_rotation_change_rad
        or (metrics["approach_direction_change_rad"] or 0.0)
        >= limits.geometry_rotation_change_rad
    )


def target_pose_changed(
    previous: Any,
    current: Any,
    *,
    limits: OperationCandidateLifecycleLimits,
) -> bool:
    metrics = target_pose_change_metrics(previous, current)
    return bool(
        (metrics["executed_target_translation_change_m"] or 0.0)
        >= limits.geometry_translation_change_m
        or (metrics["executed_target_rotation_change_rad"] or 0.0)
        >= limits.geometry_rotation_change_rad
    )


def target_pose_change_metrics(
    previous: Any,
    current: Any,
) -> dict[str, float | None]:
    left = _pose7(previous)
    right = _pose7(current)
    if left is None or right is None:
        return {
            "executed_target_translation_change_m": None,
            "executed_target_rotation_change_rad": None,
        }
    return {
        "executed_target_translation_change_m": _distance(
            left[:3], right[:3]
        ),
        "executed_target_rotation_change_rad": (
            _quaternion_distance_rad(left[3:7], right[3:7])
        ),
    }


def candidate_geometry_change_metrics(previous: Any, current: Any) -> dict[str, Any]:
    left = previous if isinstance(previous, dict) else {}
    right = current if isinstance(current, dict) else {}
    left_poses = left.get("poses") if isinstance(left.get("poses"), dict) else {}
    right_poses = right.get("poses") if isinstance(right.get("poses"), dict) else {}
    translations: list[float] = []
    rotations: list[float] = []
    for key in sorted(set(left_poses) & set(right_poses)):
        prior_pose = _pose7(left_poses.get(key))
        next_pose = _pose7(right_poses.get(key))
        if prior_pose is None or next_pose is None:
            continue
        translations.append(_distance(prior_pose[:3], next_pose[:3]))
        rotation = _quaternion_distance_rad(prior_pose[3:7], next_pose[3:7])
        if rotation is not None:
            rotations.append(rotation)
    return {
        "geometry_schema_changed": set(left_poses) != set(right_poses),
        "max_pose_translation_change_m": max(translations) if translations else None,
        "max_pose_rotation_change_rad": max(rotations) if rotations else None,
        "approach_direction_change_rad": _vector_angle_rad(
            _xyz(left.get("approach_direction")), _xyz(right.get("approach_direction"))
        ),
    }


def geometry_distance_score(previous: Any, current: Any) -> float:
    metrics = candidate_geometry_change_metrics(previous, current)
    return float(metrics["max_pose_translation_change_m"] or 0.0) + float(
        metrics["max_pose_rotation_change_rad"] or 0.0
    )


def _attempt(
    value: Any,
) -> tuple[
    OperationCandidateScope,
    str,
    str,
    dict[str, Any],
    dict[str, Any],
] | None:
    if not isinstance(value, dict):
        return None
    raw_scope = value.get("scope")
    geometry = value.get("geometry")
    candidate_id = str(value.get("candidate_id", "") or "").strip()
    point_key = normalize_grounded_point_key(
        value.get("point_key")
    )
    execution_spec = value.get("execution_spec")
    if not isinstance(raw_scope, (list, tuple)) or len(raw_scope) != 4:
        return None
    scope = tuple(str(item or "").strip() for item in raw_scope)
    if not scope[0] or scope[1] not in {"left", "right"} or not scope[2]:
        return None
    if (
        not candidate_id
        or not point_key
        or not isinstance(execution_spec, dict)
        or not isinstance(geometry, dict)
    ):
        return None
    return (
        scope,
        candidate_id,
        point_key,
        dict(execution_spec),
        dict(geometry),
    )


def _matching_candidates(
    candidates: Iterable[Any], arm: str, mode: str, target_id: str
) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and str(candidate.get("candidate_id", "") or "").strip()
        and str(candidate.get("arm", "") or "").strip().lower() == arm
        and str(candidate.get("action_mode", "") or "").strip().lower() == mode
        and (
            not target_id
            or str(candidate.get("target_id", "") or "").strip() == target_id
        )
    ]


def _canonical_geometry(geometry: Any) -> dict[str, Any]:
    value = geometry if isinstance(geometry, dict) else {}
    poses: dict[str, list[float]] = {}
    for key, raw_pose in (value.get("poses") or {}).items():
        pose = _pose7(raw_pose)
        if pose is None:
            continue
        quaternion = pose[3:7]
        if quaternion and quaternion[0] < 0.0:
            quaternion = [-item for item in quaternion]
        poses[str(key)] = [
            *(round(item, 3) for item in pose[:3]),
            *(round(item, 4) for item in quaternion),
        ]
    direction = _xyz(value.get("approach_direction"))
    return {
        "poses": poses,
        "approach_direction": None if direction is None else [round(item, 4) for item in direction],
    }


def _pose7(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 7:
        return None
    parsed = [_finite_float(item) for item in value[:7]]
    if any(item is None for item in parsed):
        return None
    quaternion = _quat4(parsed[3:7])
    if quaternion is None:
        return None
    return [
        *(float(item) for item in parsed[:3]),
        *quaternion,
    ]


def _quat4(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    parsed = [_finite_float(item) for item in value[:4]]
    if any(item is None for item in parsed):
        return None
    quaternion = [float(item) for item in parsed]
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm <= 1e-12:
        return None
    normalized = [item / norm for item in quaternion]
    return (
        [-item for item in normalized]
        if normalized[0] < 0.0
        else normalized
    )


def _xyz(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    parsed = [_finite_float(item) for item in value[:3]]
    return None if any(item is None for item in parsed) else [float(item) for item in parsed]


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_integer(*values: Any) -> int | None:
    for value in values:
        parsed = _integer(value)
        if parsed is not None:
            return parsed
    return None


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _quaternion_distance_rad(left: list[float], right: list[float]) -> float | None:
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    dot = abs(sum(left[index] * right[index] for index in range(4)) / (left_norm * right_norm))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _vector_angle_rad(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None and right is None:
        return None
    if left is None or right is None:
        return math.pi
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    dot = sum(left[index] * right[index] for index in range(3)) / (left_norm * right_norm)
    return math.acos(min(1.0, max(-1.0, dot)))
