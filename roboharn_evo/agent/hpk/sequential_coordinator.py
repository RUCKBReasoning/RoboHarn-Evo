from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from roboharn_evo.agent.hpk.episode_finalizer import (
    HPKEpisodeFinalizer,
    EpisodeFinalizationResult,
    SnapshotRef,
    TransitionStrategyContext,
)
from roboharn_evo.agent.hpk.schemas import canonical_json_bytes, validate_content_id


class SequentialCoordinatorError(RuntimeError):
    """A lease, episode order, or explicit snapshot CAS invariant failed."""


class EpisodeFinalizerProtocol(Protocol):
    def finalize(
        self,
        *,
        expected_episode_id: str | int,
        parent: SnapshotRef,
        manifest_path: str | Path,
        trace_path: str | Path,
        expected_manifest_sha256: str,
        expected_trace_sha256: str,
        transition_contexts: Mapping[str, TransitionStrategyContext],
        created_at: str,
        expected_runtime_binding: Mapping[str, object] | None = None,
        writeback_allowed: bool = True,
    ) -> EpisodeFinalizationResult: ...


def _expected_episode_id(value: str | int) -> str | int:
    if isinstance(value, bool) or not (
        isinstance(value, int)
        and value >= 0
        or isinstance(value, str)
        and bool(value.strip())
    ):
        raise SequentialCoordinatorError(
            "expected_episode_id must be a non-negative integer or non-empty string"
        )
    return value


def _episode_control_key(value: str | int) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "expected_episode_id_type": type(value).__name__,
                "expected_episode_id": value,
            }
        )
    ).hexdigest()


def _lease_token(*, expected_episode_id: str | int, snapshot: SnapshotRef) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "episode_control_key": _episode_control_key(expected_episode_id),
                "snapshot_id": snapshot.snapshot_id,
                "manifest_sha256": snapshot.manifest_sha256,
            }
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class EpisodeLease:
    expected_episode_id: str | int
    episode_control_key: str
    token: str
    snapshot_ref: SnapshotRef


@dataclass(frozen=True, slots=True)
class CoordinatedEpisodeResult:
    lease: EpisodeLease
    finalization: EpisodeFinalizationResult
    next_snapshot_ref: SnapshotRef

    @property
    def advanced(self) -> bool:
        return self.next_snapshot_ref.identity() != self.lease.snapshot_ref.identity()


class SequentialHPKCoordinator:
    """One-rollout-at-a-time coordinator with no filesystem discovery."""

    def __init__(
        self,
        *,
        initial_snapshot_ref: SnapshotRef,
        finalizer: EpisodeFinalizerProtocol | HPKEpisodeFinalizer,
    ) -> None:
        try:
            self._head = initial_snapshot_ref.reload()
        except Exception as exc:
            raise SequentialCoordinatorError(
                f"initial pinned snapshot is invalid: {type(exc).__name__}: {exc}"
            ) from exc
        self._finalizer = finalizer
        self._active: EpisodeLease | None = None
        self._completed: dict[str, CoordinatedEpisodeResult] = {}
        self._lock = threading.RLock()

    @property
    def head(self) -> SnapshotRef:
        with self._lock:
            return self._head

    @property
    def active_lease(self) -> EpisodeLease | None:
        with self._lock:
            return self._active

    def begin_episode(self, expected_episode_id: str | int) -> EpisodeLease:
        """Pin the current explicit head for one sequential episode."""

        expected_episode_id = _expected_episode_id(expected_episode_id)
        episode_control_key = _episode_control_key(expected_episode_id)
        with self._lock:
            if episode_control_key in self._completed:
                raise SequentialCoordinatorError(
                    "an already completed episode cannot be leased again"
                )
            if self._active is not None:
                if self._active.episode_control_key == episode_control_key:
                    return self._active
                raise SequentialCoordinatorError(
                    "sequential coordinator already has an active episode"
                )
            try:
                pinned = self._head.reload()
            except Exception as exc:
                raise SequentialCoordinatorError(
                    f"current pinned head failed CAS reload: {type(exc).__name__}: {exc}"
                ) from exc
            if pinned.identity() != self._head.identity():
                raise SequentialCoordinatorError("current snapshot head changed")
            lease = EpisodeLease(
                expected_episode_id=expected_episode_id,
                episode_control_key=episode_control_key,
                token=_lease_token(
                    expected_episode_id=expected_episode_id, snapshot=pinned
                ),
                snapshot_ref=pinned,
            )
            self._active = lease
            return lease

    def snapshot_for_episode(self, lease: EpisodeLease) -> SnapshotRef:
        """Return the same pinned snapshot for every access within a lease."""

        with self._lock:
            self._require_active(lease)
            return self._active.snapshot_ref  # type: ignore[union-attr]

    def finalize_episode(
        self,
        lease: EpisodeLease,
        *,
        manifest_path: str | Path,
        trace_path: str | Path,
        expected_manifest_sha256: str,
        expected_trace_sha256: str,
        transition_contexts: Mapping[str, TransitionStrategyContext],
        created_at: str,
        expected_runtime_binding: Mapping[str, object] | None = None,
        writeback_allowed: bool = True,
    ) -> CoordinatedEpisodeResult:
        """Finalize the lease and advance the head only to its published child."""

        with self._lock:
            completed = self._completed.get(lease.episode_control_key)
            if completed is not None:
                if completed.lease.token != lease.token:
                    raise SequentialCoordinatorError(
                        "completed episode was replayed with a different lease"
                    )
                return completed
            self._require_active(lease)
            if self._head.identity() != lease.snapshot_ref.identity():
                raise SequentialCoordinatorError(
                    "stale episode parent: coordinator head no longer matches the lease"
                )
            try:
                finalizer_kwargs = {
                    "expected_episode_id": lease.expected_episode_id,
                    "parent": lease.snapshot_ref,
                    "manifest_path": manifest_path,
                    "trace_path": trace_path,
                    "expected_manifest_sha256": expected_manifest_sha256,
                    "expected_trace_sha256": expected_trace_sha256,
                    "transition_contexts": transition_contexts,
                    "created_at": created_at,
                }
                if expected_runtime_binding is not None:
                    finalizer_kwargs["expected_runtime_binding"] = (
                        expected_runtime_binding
                    )
                if writeback_allowed is not True:
                    finalizer_kwargs["writeback_allowed"] = writeback_allowed
                result = self._finalizer.finalize(
                    **finalizer_kwargs,
                )
            except Exception as exc:
                # An unexpected orchestration error never advances the head.  Keep
                # the active lease so the caller can record/handle the failure.
                raise SequentialCoordinatorError(
                    f"episode finalization failed before a safe result: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            try:
                validate_content_id(
                    result.episode_id,
                    prefix="afkepisode",
                    path="finalization.episode_id",
                )
            except Exception as exc:
                raise SequentialCoordinatorError(
                    f"finalizer returned an invalid public episode group: {exc}"
                ) from exc
            if result.parent.identity() != lease.snapshot_ref.identity():
                raise SequentialCoordinatorError(
                    "finalizer returned a result for a different parent"
                )
            next_ref = lease.snapshot_ref
            if result.child is not None:
                if result.active.identity() != result.child.identity():
                    raise SequentialCoordinatorError(
                        "published finalization did not activate its child"
                    )
                try:
                    next_ref = result.child.reload()
                except Exception as exc:
                    raise SequentialCoordinatorError(
                        f"published child failed explicit reload: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                parent_in_child = next_ref.snapshot.manifest["parent"]
                if parent_in_child != lease.snapshot_ref.identity():
                    raise SequentialCoordinatorError(
                        "published child lineage does not match the episode lease"
                    )
                if (
                    result.episode_id
                    not in next_ref.snapshot.manifest["source_episode_ids"]
                ):
                    raise SequentialCoordinatorError(
                        "published child does not contain the finalized public episode group"
                    )
            elif result.active.identity() != lease.snapshot_ref.identity():
                raise SequentialCoordinatorError(
                    "abstained finalization must continue using its parent"
                )
            coordinated = CoordinatedEpisodeResult(
                lease=lease,
                finalization=result,
                next_snapshot_ref=next_ref,
            )
            self._head = next_ref
            self._active = None
            self._completed[lease.episode_control_key] = coordinated
            return coordinated

    def _require_active(self, lease: EpisodeLease) -> None:
        if not isinstance(lease, EpisodeLease):
            raise SequentialCoordinatorError("lease must be an EpisodeLease")
        if self._active is None:
            raise SequentialCoordinatorError("there is no active episode")
        if (
            self._active.token != lease.token
            or self._active.episode_control_key != lease.episode_control_key
        ):
            raise SequentialCoordinatorError(
                "lease does not match the coordinator's active episode"
            )
        if self._active.snapshot_ref.identity() != lease.snapshot_ref.identity():
            raise SequentialCoordinatorError("lease snapshot identity mismatch")


__all__ = [
    "CoordinatedEpisodeResult",
    "EpisodeFinalizerProtocol",
    "EpisodeLease",
    "SequentialHPKCoordinator",
    "SequentialCoordinatorError",
]
