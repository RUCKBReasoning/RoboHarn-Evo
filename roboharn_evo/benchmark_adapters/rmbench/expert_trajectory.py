"""Read-only RMBench scripted-expert import for Self-Evolution Phase A.

Only artifacts named by an explicit ``roboharn_evo/expert_bootstrap_manifest/v1`` are
opened.  This module never discovers episodes, decodes RGB, copies raw data,
starts a benchmark, or writes to the dataset root.  HDF5 support is imported
lazily so base RoboHarn-Evo imports remain lightweight.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import pickle
import pickletools
import re
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import numpy as np

from roboharn_evo.agent.reflector.schemas import (
    ExpertBootstrapManifestV1,
    SchemaValidationError,
    SubtaskSegmentV1,
    TrajectoryRecordV1,
)


_REQUIRED_ARTIFACTS = (
    "hdf5",
    "planner_path",
    "language_annotation",
    "instruction",
)
_ARTIFACT_KIND = {
    "hdf5": "hdf5",
    "planner_path": "pkl",
    "language_annotation": "json",
    "instruction": "json",
    "qwen_annotation": "jsonl",
}
_MAX_JSON_BYTES = 64 * 1024 * 1024
_HARD_MAX_PICKLE_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_PICKLE_BYTES = _HARD_MAX_PICKLE_BYTES
_MAX_PICKLE_OPCODES = 250_000
_MAX_PICKLE_MEMO_ENTRIES = 250_000
_MAX_PICKLE_NODES = 1_000_000
_MAX_PICKLE_ARRAY_ELEMENTS = 10_000_000
_MAX_NUMPY_ITEMSIZE = 16
_MAX_HDF5_FRAMES = 10_000
_MAX_HDF5_OBJECTS = 4_096
_MAX_HDF5_RANK = 8
_MAX_HDF5_DIMENSION = 1_000_000
_MAX_HDF5_EVIDENCE_KEY_REFERENCES = 250_000
_MAX_HDF5_ROOT_ATTRIBUTES = 256
_MAX_HDF5_METADATA_TEXT_BYTES = 1024 * 1024
_DEFAULT_MAX_ENTRIES = 100
_DEFAULT_MAX_TOTAL_ARTIFACTS = 500
_DEFAULT_MAX_TOTAL_DECLARED_BYTES = 64 * 1024**3
_DEFAULT_MAX_TOTAL_FRAMES = 100_000
_DEFAULT_MAX_TOTAL_EVIDENCE_ENTRIES = 500_000
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SAFE_EXPLICIT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_GENERATED_TRAJECTORY_ID_PREFIX = "rmbench_v1_"

_ALLOWED_PICKLE_OPCODES = frozenset(
    {
        "APPEND",
        "APPENDS",
        "BINFLOAT",
        "BINGET",
        "BININT",
        "BININT1",
        "BININT2",
        "BINBYTES",
        "BINBYTES8",
        "BINSTRING",
        "BINUNICODE",
        "BINUNICODE8",
        "BINPUT",
        "BUILD",
        "BYTEARRAY8",
        "DICT",
        "DUP",
        "EMPTY_DICT",
        "EMPTY_LIST",
        "EMPTY_TUPLE",
        "FLOAT",
        "FRAME",
        "GET",
        "GLOBAL",
        "INT",
        "LIST",
        "LONG",
        "LONG1",
        "LONG4",
        "LONG_BINGET",
        "LONG_BINPUT",
        "MARK",
        "MEMOIZE",
        "NEWFALSE",
        "NEWTRUE",
        "NONE",
        "POP",
        "POP_MARK",
        "PROTO",
        "PUT",
        "REDUCE",
        "SETITEM",
        "SETITEMS",
        "SHORT_BINBYTES",
        "SHORT_BINSTRING",
        "SHORT_BINUNICODE",
        "STACK_GLOBAL",
        "STOP",
        "STRING",
        "TUPLE",
        "TUPLE1",
        "TUPLE2",
        "TUPLE3",
        "UNICODE",
    }
)


class ManifestImportError(ValueError):
    """A fail-closed manifest, path, integrity, or dependency error."""


class OptionalExpertDependencyError(ManifestImportError):
    """An optional dependency required to inspect an authorized artifact."""


class ResourceBudgetExceeded(ManifestImportError):
    """A manifest or converted entry exceeded an explicit global budget."""


class ArtifactMutationError(ManifestImportError):
    """An authorized artifact changed or its path was substituted during parsing."""


class _EntryConversionError(ValueError):
    def __init__(
        self, reason: str, *, scope: str, artifact_role: str | None = None
    ) -> None:
        self.reason = reason
        self.scope = scope
        self.artifact_role = artifact_role
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ConverterAbstention:
    """Auditable per-entry or per-artifact abstention."""

    reason: str
    scope: str
    entry_index: int
    trajectory_id: str
    artifact_role: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": "roboharn_evo/bootstrap_abstention/v1",
            "schema_version": 1,
            "status": "abstained",
            "stage": "import",
            "reason": self.reason,
            "scope": self.scope,
            "entry_index": self.entry_index,
            "trajectory_id": self.trajectory_id,
        }
        if self.artifact_role is not None:
            result["artifact_role"] = self.artifact_role
        if self.details:
            result["details"] = copy.deepcopy(dict(self.details))
        return result


@dataclass(frozen=True, slots=True)
class RMBenchImportBatch:
    """JSON-ready result of one manifest-only import."""

    trajectories: tuple[dict[str, Any], ...]
    subtask_segments: tuple[dict[str, Any], ...]
    evidence_entries: tuple[dict[str, Any], ...]
    abstentions: tuple[dict[str, Any], ...]
    input_manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectories": copy.deepcopy(list(self.trajectories)),
            "subtask_segments": copy.deepcopy(list(self.subtask_segments)),
            "evidence_entries": copy.deepcopy(list(self.evidence_entries)),
            "abstentions": copy.deepcopy(list(self.abstentions)),
            "input_manifest": copy.deepcopy(self.input_manifest),
        }


@dataclass(frozen=True, slots=True)
class RMBenchImportedEntry:
    """One converted manifest entry yielded only after all source FDs are closed."""

    entry_index: int
    trajectory_id: str
    trajectory: dict[str, Any] | None
    subtask_segments: tuple[dict[str, Any], ...]
    evidence_entries: tuple[dict[str, Any], ...]
    abstentions: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class RMBenchImportResourceUsage:
    """Observed global resource use for one import stream."""

    entries: int
    artifacts: int
    declared_bytes: int
    frames: int
    evidence_entries: int

    def to_dict(self) -> dict[str, int]:
        return {
            "entries": self.entries,
            "artifacts": self.artifacts,
            "declared_bytes": self.declared_bytes,
            "frames": self.frames,
            "evidence_entries": self.evidence_entries,
        }


@dataclass(frozen=True, slots=True)
class _ResourceLimits:
    max_entries: int
    max_total_artifacts: int
    max_total_declared_bytes: int
    max_total_frames: int
    max_total_evidence_entries: int

    @classmethod
    def create(
        cls,
        *,
        max_entries: int,
        max_total_artifacts: int,
        max_total_declared_bytes: int,
        max_total_frames: int,
        max_total_evidence_entries: int,
    ) -> _ResourceLimits:
        values = {
            "max_entries": max_entries,
            "max_total_artifacts": max_total_artifacts,
            "max_total_declared_bytes": max_total_declared_bytes,
            "max_total_frames": max_total_frames,
            "max_total_evidence_entries": max_total_evidence_entries,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        return cls(**values)

    def to_dict(self) -> dict[str, int]:
        return {
            "max_entries": self.max_entries,
            "max_total_artifacts": self.max_total_artifacts,
            "max_total_declared_bytes": self.max_total_declared_bytes,
            "max_total_frames": self.max_total_frames,
            "max_total_evidence_entries": self.max_total_evidence_entries,
        }


@dataclass(slots=True)
class _ResourceUsageState:
    limits: _ResourceLimits
    entries: int
    artifacts: int
    declared_bytes: int
    frames: int = 0
    evidence_entries: int = 0

    def snapshot(self) -> RMBenchImportResourceUsage:
        return RMBenchImportResourceUsage(
            entries=self.entries,
            artifacts=self.artifacts,
            declared_bytes=self.declared_bytes,
            frames=self.frames,
            evidence_entries=self.evidence_entries,
        )

    def consume_frames(self, count: int) -> None:
        proposed = self.frames + count
        if proposed > self.limits.max_total_frames:
            raise ResourceBudgetExceeded(
                "max_total_frames exceeded: "
                f"limit={self.limits.max_total_frames}, attempted={proposed}"
            )
        self.frames = proposed

    def ensure_evidence_capacity(self, count: int) -> None:
        proposed = self.evidence_entries + count
        if proposed > self.limits.max_total_evidence_entries:
            raise ResourceBudgetExceeded(
                "max_total_evidence_entries exceeded: "
                f"limit={self.limits.max_total_evidence_entries}, attempted={proposed}"
            )

    def consume_evidence(self, count: int) -> None:
        self.ensure_evidence_capacity(count)
        self.evidence_entries += count


@dataclass(frozen=True, slots=True)
class _ArtifactSnapshot:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _AuthorizedArtifact:
    role: str
    path: Path
    relative_path: str
    size_bytes: int
    sha256: str
    source_ref_id: str
    fd: int
    authorized_snapshot: _ArtifactSnapshot

    def open_binary(self) -> BinaryIO:
        """Return a reader for the already-authorized inode, never its path."""

        duplicate = os.dup(self.fd)
        try:
            os.lseek(duplicate, 0, os.SEEK_SET)
            return os.fdopen(duplicate, "rb", closefd=True)
        except Exception:
            os.close(duplicate)
            raise


@dataclass(frozen=True, slots=True)
class _ArtifactPlan:
    role: str
    descriptor: Mapping[str, Any]
    default_source_ref_id: str
    label: str


@dataclass(frozen=True, slots=True)
class _EntryPlan:
    entry_index: int
    entry: Mapping[str, Any]
    trajectory_id: str
    artifacts: tuple[_ArtifactPlan, ...]


class RMBenchImportStream(Iterator[RMBenchImportedEntry]):
    """Closable one-entry-at-a-time stream with usage visible after exhaustion."""

    def __init__(
        self,
        iterator: Iterator[RMBenchImportedEntry],
        usage: _ResourceUsageState,
        input_manifest: Mapping[str, Any],
        owned_dataset_root_fd: int,
    ) -> None:
        self._iterator = iterator
        self._usage = usage
        self._input_manifest = copy.deepcopy(dict(input_manifest))
        self._owned_dataset_root_fd: int | None = owned_dataset_root_fd
        self._closed = False

    @property
    def usage(self) -> RMBenchImportResourceUsage:
        return self._usage.snapshot()

    @property
    def limits(self) -> Mapping[str, int]:
        return self._usage.limits.to_dict()

    @property
    def input_manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._input_manifest)

    def __iter__(self) -> RMBenchImportStream:
        return self

    def __next__(self) -> RMBenchImportedEntry:
        if self._closed:
            raise StopIteration
        try:
            return next(self._iterator)
        except StopIteration:
            self.close()
            raise
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self._iterator, "close", None)
            if callable(close):
                close()
        finally:
            descriptor = self._owned_dataset_root_fd
            self._owned_dataset_root_fd = None
            if descriptor is not None:
                os.close(descriptor)

    def __enter__(self) -> RMBenchImportStream:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    try:
        offset = 0
        while True:
            chunk = os.pread(fd, 1024 * 1024, offset)
            if not chunk:
                break
            digest.update(chunk)
            offset += len(chunk)
    except OSError as exc:
        raise ManifestImportError(
            f"cannot read authorized artifact descriptor: {exc}"
        ) from exc
    return digest.hexdigest()


def _snapshot_from_stat(info: os.stat_result, digest: str) -> _ArtifactSnapshot:
    return _ArtifactSnapshot(
        device=info.st_dev,
        inode=info.st_ino,
        size_bytes=info.st_size,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
        sha256=digest,
    )


def _capture_artifact_snapshot(fd: int, *, label: str) -> _ArtifactSnapshot:
    """Hash one stable descriptor and bind all observable mutation metadata."""

    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactMutationError(
                f"authorized artifact is no longer regular: {label}"
            )
        digest = _sha256_fd(fd)
        after = os.fstat(fd)
    except ArtifactMutationError:
        raise
    except OSError as exc:
        raise ArtifactMutationError(
            f"cannot verify authorized artifact {label}: {exc}"
        ) from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ArtifactMutationError(
            f"authorized artifact changed while hashing: {label}"
        )
    return _snapshot_from_stat(after, digest)


def _assert_same_artifact_snapshot(
    actual: _ArtifactSnapshot,
    expected: _ArtifactSnapshot,
    *,
    label: str,
    phase: str,
) -> None:
    if actual != expected:
        raise ArtifactMutationError(
            "authorized artifact changed or path substitution detected "
            f"{phase}: {label}"
        )


def _strict_json_bytes(raw: bytes, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ManifestImportError(f"{label} is not strict UTF-8 JSON: {exc}") from exc


def _secure_read_manifest_path(
    manifest_path: str | os.PathLike[str],
) -> tuple[Path, bytes]:
    """Read one manifest through descriptor-relative, no-follow path traversal."""

    path = Path(manifest_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if ".." in path.parts or len(path.parts) < 2:
        raise ManifestImportError("input manifest path is unsafe")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None or not os.supports_dir_fd or os.open not in os.supports_dir_fd:
        raise ManifestImportError("secure no-follow manifest open is unavailable")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | nofollow
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | nofollow
    directory_fd: int | None = None
    descriptor: int | None = None
    try:
        directory_fd = os.open(path.anchor, directory_flags)
        for part in path.parts[1:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(path.parts[-1], file_flags, dir_fd=directory_fd)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ManifestImportError("input manifest must be a regular file")
        if opened.st_size > _MAX_JSON_BYTES:
            raise ManifestImportError("input manifest exceeds safe JSON size")
        raw = os.pread(descriptor, opened.st_size + 1, 0)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if len(raw) != opened.st_size or any(
            getattr(opened, field) != getattr(after, field) for field in stable_fields
        ):
            raise ManifestImportError("input manifest changed while being read")
        current = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise ManifestImportError("input manifest path substitution detected")
        return path, raw
    except ManifestImportError:
        raise
    except OSError as exc:
        raise ManifestImportError(
            "cannot securely read input manifest (missing, symlink, or path "
            f"substitution): {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_fd is not None:
            os.close(directory_fd)


def _secure_open_dataset_root(path: Path) -> int:
    """Bind an absolute dataset root without following any path component."""

    if not path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
        raise ManifestImportError("dataset_root path is unsafe")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ManifestImportError("secure no-follow dataset root open is unavailable")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | nofollow
    current_fd: int | None = None
    try:
        current_fd = os.open(path.anchor, flags)
        for part in path.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except OSError as exc:
        if current_fd is not None:
            os.close(current_fd)
        raise ManifestImportError(
            "dataset_root cannot be securely opened (missing, symlink, or "
            f"non-directory component): {exc}"
        ) from exc


def _read_json(artifact: _AuthorizedArtifact, *, label: str) -> Any:
    try:
        if artifact.size_bytes > _MAX_JSON_BYTES:
            raise ManifestImportError(f"{label} exceeds {_MAX_JSON_BYTES} bytes")
        with artifact.open_binary() as handle:
            raw = handle.read(_MAX_JSON_BYTES + 1)
        if len(raw) != artifact.size_bytes:
            raise ManifestImportError(f"{label} changed size while being read")
    except OSError as exc:
        raise ManifestImportError(f"cannot read {label}: {exc}") from exc
    return _strict_json_bytes(raw, label=label)


def _safe_trajectory_id(entry: Mapping[str, Any]) -> str:
    explicit = entry.get("trajectory_id")
    if explicit is not None:
        if (
            not isinstance(explicit, str)
            or not explicit
            or _SAFE_EXPLICIT_ID_RE.fullmatch(explicit) is None
            or explicit.startswith(_GENERATED_TRAJECTORY_ID_PREFIX)
        ):
            raise ManifestImportError(
                "explicit trajectory_id must use only ASCII letters, digits, '.', "
                "'_', or '-', and must not use the reserved 'rmbench_v1_' prefix"
            )
        return explicit

    task = entry.get("task")
    task_config = entry.get("task_config")
    episode_id = entry.get("episode_id")
    if not isinstance(task, str) or not task:
        raise ManifestImportError("entry has invalid task for trajectory identity")
    if not isinstance(task_config, str) or not task_config:
        raise ManifestImportError(
            "entry has invalid task_config for trajectory identity"
        )
    if isinstance(episode_id, bool) or not isinstance(episode_id, (int, str)):
        raise ManifestImportError(
            "entry has invalid typed episode_id for trajectory identity"
        )

    episode_type = "int" if isinstance(episode_id, int) else "str"
    identity = {
        "episode_id": {"type": episode_type, "value": episode_id},
        "task": task,
        "task_config": task_config,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()

    def readable(value: str, *, fallback: str) -> str:
        normalized = _SAFE_ID_RE.sub("_", value).strip("_")
        return (normalized or fallback)[:32]

    task_part = readable(task, fallback="task")
    config_part = readable(task_config, fallback="config")
    episode_part = readable(str(episode_id), fallback="episode")
    return (
        f"{_GENERATED_TRAJECTORY_ID_PREFIX}{task_part}_{config_part}_"
        f"ep_{episode_type}_{episode_part}_{digest}"
    )


def _safe_source_ref_id(
    trajectory_id: str, role: str, descriptor: Mapping[str, Any]
) -> str:
    explicit = descriptor.get("source_ref_id")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ManifestImportError(f"artifact {role} has invalid source_ref_id")
        return explicit
    return f"{trajectory_id}_{role}"


def _relative_parts(value: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ManifestImportError(
            "artifact relative_path must be a non-empty POSIX path"
        )
    path = PurePosixPath(value)
    raw_parts = tuple(value.split("/"))
    if (
        path.is_absolute()
        or not raw_parts
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ManifestImportError(f"unsafe artifact relative_path {value!r}")
    return raw_parts


def _open_regular_artifact(
    dataset_root: Path,
    dataset_root_fd: int,
    relative_path: str,
) -> tuple[Path, int, os.stat_result]:
    """Open a manifest path beneath a stable root FD without following symlinks."""

    parts = _relative_parts(relative_path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None or not os.supports_dir_fd or os.open not in os.supports_dir_fd:
        raise ManifestImportError(
            "secure descriptor-relative artifact open is unavailable"
        )
    directory_fd = os.dup(dataset_root_fd)
    artifact_fd: int | None = None
    try:
        for part in parts[:-1]:
            try:
                component = os.stat(part, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise ManifestImportError(
                    f"authorized artifact path cannot be inspected: {relative_path}: {exc}"
                ) from exc
            if stat.S_ISLNK(component.st_mode):
                raise ManifestImportError(
                    f"authorized artifact path contains symlink: {relative_path}"
                )
            if not stat.S_ISDIR(component.st_mode):
                raise ManifestImportError(
                    f"authorized artifact parent is not a directory: {relative_path}"
                )
            try:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | nofollow,
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise ManifestImportError(
                    f"authorized artifact path changed or contains symlink: {relative_path}: {exc}"
                ) from exc
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            leaf = os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise ManifestImportError(
                f"authorized artifact path cannot be inspected: {relative_path}: {exc}"
            ) from exc
        if stat.S_ISLNK(leaf.st_mode):
            raise ManifestImportError(
                f"authorized artifact path contains symlink: {relative_path}"
            )
        if not stat.S_ISREG(leaf.st_mode):
            raise ManifestImportError(
                f"authorized artifact is not a regular file: {relative_path}"
            )
        try:
            artifact_fd = os.open(
                parts[-1],
                os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | nofollow,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise ManifestImportError(
                f"authorized artifact path changed or contains symlink: {relative_path}: {exc}"
            ) from exc
        opened = os.fstat(artifact_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ManifestImportError(
                f"authorized artifact is not a regular file: {relative_path}"
            )
        if (opened.st_dev, opened.st_ino) != (leaf.st_dev, leaf.st_ino):
            raise ManifestImportError(
                f"authorized artifact path substitution detected: {relative_path}"
            )
        return dataset_root.joinpath(*parts), artifact_fd, opened
    except Exception:
        if artifact_fd is not None:
            os.close(artifact_fd)
        raise
    finally:
        os.close(directory_fd)


class _NumpyNdarrayToken:
    """Non-callable marker used by the exact ndarray reconstruction grammar."""


_NUMPY_NDARRAY_TOKEN = _NumpyNdarrayToken()


class _SafeNumpyArray:
    """Metadata-only ndarray proxy; raw pickle bytes never become an ndarray."""

    __slots__ = ("dtype", "shape")

    def __init__(self) -> None:
        self.shape: tuple[int, ...] = ()
        self.dtype = np.dtype("u1")

    def __setstate__(self, state: Any) -> None:
        if not isinstance(state, tuple) or len(state) != 5:
            raise pickle.UnpicklingError("NumPy array state must be a five-item tuple")
        version, raw_shape, raw_dtype, fortran_order, raw_data = state
        if version not in {0, 1} or isinstance(version, bool):
            raise pickle.UnpicklingError("unsupported NumPy array pickle version")
        if (
            not isinstance(raw_shape, tuple)
            or not raw_shape
            or len(raw_shape) > _MAX_HDF5_RANK
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value > _MAX_HDF5_DIMENSION
                for value in raw_shape
            )
        ):
            raise pickle.UnpicklingError("unsafe NumPy array shape")
        if not isinstance(raw_dtype, np.dtype):
            raise pickle.UnpicklingError("NumPy array state has invalid dtype")
        if (
            raw_dtype.hasobject
            or raw_dtype.fields
            or raw_dtype.kind not in "biuf"
            or raw_dtype.itemsize <= 0
            or raw_dtype.itemsize > _MAX_NUMPY_ITEMSIZE
        ):
            raise pickle.UnpicklingError(f"unsafe NumPy dtype {raw_dtype}")
        if not isinstance(fortran_order, bool):
            raise pickle.UnpicklingError("NumPy array order flag must be boolean")
        if not isinstance(raw_data, (bytes, bytearray)):
            raise pickle.UnpicklingError("NumPy array payload must be inline bytes")
        elements = 1
        for dimension in raw_shape:
            elements *= dimension
            if elements > _MAX_PICKLE_ARRAY_ELEMENTS:
                raise pickle.UnpicklingError(
                    "planner pickle exceeds safe numeric element limit"
                )
        expected_bytes = elements * raw_dtype.itemsize
        if len(raw_data) != expected_bytes:
            raise pickle.UnpicklingError(
                "NumPy array byte length does not match shape and dtype"
            )
        self.shape = tuple(raw_shape)
        self.dtype = raw_dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


def _safe_numpy_reconstruct(
    subtype: Any, shape: Any, dtype_code: Any
) -> _SafeNumpyArray:
    if (
        subtype is not _NUMPY_NDARRAY_TOKEN
        or shape != (0,)
        or dtype_code not in {b"b", "b"}
    ):
        raise pickle.UnpicklingError("unsafe NumPy ndarray reconstruction arguments")
    return _SafeNumpyArray()


def _safe_numpy_dtype(*args: Any, **kwargs: Any) -> np.dtype[Any]:
    try:
        value = np.dtype(*args, **kwargs)
    except (TypeError, ValueError, OverflowError) as exc:
        raise pickle.UnpicklingError("invalid NumPy dtype") from exc
    if (
        value.hasobject
        or value.fields
        or value.kind not in "biuf"
        or value.itemsize <= 0
        or value.itemsize > _MAX_NUMPY_ITEMSIZE
    ):
        raise pickle.UnpicklingError(f"unsafe NumPy dtype {value}")
    return value


class _RestrictedPlannerUnpickler(pickle.Unpickler):
    """Parse only the exact, allocation-bounded NumPy ndarray pickle grammar."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy" and name == "ndarray":
            return _NUMPY_NDARRAY_TOKEN
        if module == "numpy" and name == "dtype":
            return _safe_numpy_dtype
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"}:
            if name == "_reconstruct":
                return _safe_numpy_reconstruct
        raise pickle.UnpicklingError(f"forbidden pickle global {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        del pid
        raise pickle.UnpicklingError("pickle persistent IDs are forbidden")


def _prevalidate_pickle_vm(payload: bytes) -> None:
    """Bound pickle VM allocations before invoking the stdlib unpickler."""

    if len(payload) > _HARD_MAX_PICKLE_BYTES:
        raise pickle.UnpicklingError("planner pickle exceeds the hard byte limit")
    opcode_count = 0
    memo_entries = 0
    stop_position: int | None = None
    try:
        for opcode, argument, position in pickletools.genops(payload):
            opcode_count += 1
            if opcode_count > _MAX_PICKLE_OPCODES:
                raise pickle.UnpicklingError("planner pickle exceeds safe opcode limit")
            name = opcode.name
            if name not in _ALLOWED_PICKLE_OPCODES:
                raise pickle.UnpicklingError(
                    f"planner pickle opcode {name} is forbidden"
                )
            if name == "PROTO" and (
                not isinstance(argument, int) or argument < 2 or argument > 5
            ):
                raise pickle.UnpicklingError(
                    "planner pickle protocol must be between 2 and 5"
                )
            if name == "FRAME" and (
                not isinstance(argument, int)
                or argument < 0
                or position + 9 + argument > len(payload)
            ):
                raise pickle.UnpicklingError(
                    "planner pickle declares an unsafe frame length"
                )
            if name == "MEMOIZE":
                memo_entries += 1
            elif name in {"PUT", "BINPUT", "LONG_BINPUT"}:
                if not isinstance(argument, int) or argument < 0:
                    raise pickle.UnpicklingError(
                        "planner pickle has an invalid memo index"
                    )
                memo_entries = max(memo_entries, argument + 1)
            elif name in {"GET", "BINGET", "LONG_BINGET"} and (
                not isinstance(argument, int)
                or argument < 0
                or argument >= _MAX_PICKLE_MEMO_ENTRIES
            ):
                raise pickle.UnpicklingError(
                    "planner pickle memo lookup exceeds the safe limit"
                )
            if memo_entries > _MAX_PICKLE_MEMO_ENTRIES:
                raise pickle.UnpicklingError("planner pickle exceeds safe memo limit")
            if name in {
                "INT",
                "BININT",
                "BININT1",
                "BININT2",
                "LONG",
                "LONG1",
                "LONG4",
            } and isinstance(argument, int):
                if argument.bit_length() > 64:
                    raise pickle.UnpicklingError(
                        "planner pickle integer exceeds the safe width"
                    )
            if (
                isinstance(argument, (bytes, bytearray, str))
                and len(argument) > _HARD_MAX_PICKLE_BYTES
            ):
                raise pickle.UnpicklingError(
                    "planner pickle inline payload exceeds the hard byte limit"
                )
            if name == "STOP":
                stop_position = position
    except pickle.UnpicklingError:
        raise
    except (OverflowError, UnicodeError, ValueError) as exc:
        raise pickle.UnpicklingError(
            f"planner pickle opcode stream is malformed: {exc}"
        ) from exc
    if stop_position is None or stop_position != len(payload) - 1:
        raise pickle.UnpicklingError(
            "planner pickle must contain exactly one complete opcode stream"
        )


def _restricted_pickle_load(payload: bytes) -> Any:
    _prevalidate_pickle_vm(payload)
    return _RestrictedPlannerUnpickler(io.BytesIO(payload)).load()


def _validate_pickle_value(
    value: Any,
    *,
    path: str = "planner",
    depth: int = 0,
    counters: dict[str, int] | None = None,
) -> None:
    if counters is None:
        counters = {"nodes": 0, "array_elements": 0}
    counters["nodes"] += 1
    if counters["nodes"] > _MAX_PICKLE_NODES:
        raise pickle.UnpicklingError("planner pickle exceeds safe node limit")
    if depth > 32:
        raise pickle.UnpicklingError("planner pickle exceeds safe nesting depth")
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, _SafeNumpyArray):
        if (
            value.dtype.hasobject
            or value.dtype.fields
            or value.dtype.kind not in "biuf"
        ):
            raise pickle.UnpicklingError(
                f"{path} contains unsafe NumPy dtype {value.dtype}"
            )
        counters["array_elements"] += int(value.size)
        if counters["array_elements"] > _MAX_PICKLE_ARRAY_ELEMENTS:
            raise pickle.UnpicklingError(
                "planner pickle exceeds safe numeric element limit"
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise pickle.UnpicklingError(
                    f"{path} contains a non-string mapping key"
                )
            _validate_pickle_value(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
                counters=counters,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_pickle_value(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
                counters=counters,
            )
        return
    raise pickle.UnpicklingError(
        f"{path} contains forbidden type {type(value).__name__}"
    )


def _planner_summary(
    artifact: _AuthorizedArtifact,
    *,
    max_bytes: int,
    max_evidence_entries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if artifact.size_bytes > max_bytes or artifact.size_bytes > _HARD_MAX_PICKLE_BYTES:
        raise pickle.UnpicklingError(
            f"planner pickle exceeds safe {max_bytes}-byte limit"
        )
    with artifact.open_binary() as handle:
        raw = handle.read(_HARD_MAX_PICKLE_BYTES + 1)
    if len(raw) != artifact.size_bytes:
        raise pickle.UnpicklingError("planner pickle changed size while being read")
    payload = _restricted_pickle_load(raw)
    _validate_pickle_value(payload)
    if not isinstance(payload, dict) or set(payload) != {
        "left_joint_path",
        "right_joint_path",
    }:
        raise pickle.UnpicklingError(
            "planner pickle must contain exactly left_joint_path and right_joint_path"
        )

    planner_segment_count = 0
    for arm in ("left", "right"):
        arm_segments = payload[f"{arm}_joint_path"]
        if not isinstance(arm_segments, (list, tuple)):
            raise pickle.UnpicklingError(f"{arm}_joint_path must be an array")
        if len(arm_segments) > 100_000:
            raise pickle.UnpicklingError(f"{arm}_joint_path has too many segments")
        planner_segment_count += len(arm_segments)
    if planner_segment_count > max_evidence_entries:
        raise ResourceBudgetExceeded(
            "max_total_evidence_entries exceeded while reading planner metadata: "
            f"remaining={max_evidence_entries}, attempted={planner_segment_count}"
        )

    summaries: list[dict[str, Any]] = []
    locators: list[dict[str, Any]] = []
    for arm in ("left", "right"):
        segments = payload[f"{arm}_joint_path"]
        arm_summaries: list[dict[str, Any]] = []
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                raise pickle.UnpicklingError(
                    f"{arm} planner segment {index} must be an object"
                )
            if not {"status", "position", "velocity"}.issubset(segment):
                raise pickle.UnpicklingError(
                    f"{arm} planner segment {index} lacks required fields"
                )
            status_value = segment["status"]
            if not isinstance(status_value, str) or not status_value.strip():
                raise pickle.UnpicklingError(
                    f"{arm} planner segment {index} has empty status"
                )
            status_text = status_value
            position = segment["position"]
            velocity = segment["velocity"]
            if not isinstance(position, _SafeNumpyArray) or not isinstance(
                velocity, _SafeNumpyArray
            ):
                raise pickle.UnpicklingError(
                    f"{arm} planner segment {index} paths must be arrays"
                )
            if (
                position.ndim != 2
                or velocity.ndim != 2
                or position.shape[1:] != (6,)
                or velocity.shape != position.shape
            ):
                raise pickle.UnpicklingError(
                    f"{arm} planner segment {index} must have matching [N,6] position/velocity"
                )
            summary = {
                "segment_index": index,
                "status": status_text,
                "position_shape": list(position.shape),
                "velocity_shape": list(velocity.shape),
                "dtype": str(position.dtype),
            }
            arm_summaries.append(summary)
            locators.append({"arm": arm, **summary})
        summaries.append({"arm": arm, "segments": arm_summaries})
    return summaries, locators


def _load_h5py() -> Any:
    try:
        import h5py  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise OptionalExpertDependencyError(
            "RMBench expert HDF5 inspection requires h5py; install roboharn-evo[expert]"
        ) from exc
    return h5py


def _hdf5_metadata(artifact: _AuthorizedArtifact) -> dict[str, Any]:
    h5py = _load_h5py()
    datasets: dict[str, dict[str, Any]] = {}
    object_count = 0
    metadata_text_bytes = 0
    try:
        with artifact.open_binary() as raw_handle:
            with h5py.File(raw_handle, "r") as handle:

                def walk(group: Any, prefix: str = "") -> None:
                    nonlocal metadata_text_bytes, object_count
                    for name in group.keys():
                        object_count += 1
                        if object_count > _MAX_HDF5_OBJECTS:
                            raise _EntryConversionError(
                                "hdf5_object_limit_exceeded",
                                scope="hdf5_metadata",
                                artifact_role="hdf5",
                            )
                        full_name = f"{prefix}/{name}" if prefix else str(name)
                        metadata_text_bytes += len(
                            full_name.encode("utf-8", "surrogatepass")
                        )
                        if metadata_text_bytes > _MAX_HDF5_METADATA_TEXT_BYTES:
                            raise _EntryConversionError(
                                "hdf5_metadata_budget_exceeded",
                                scope="hdf5_metadata",
                                artifact_role="hdf5",
                            )
                        link = group.get(name, getlink=True)
                        if not isinstance(link, h5py.HardLink):
                            raise _EntryConversionError(
                                "hdf5_nonlocal_link",
                                scope="hdf5_metadata",
                                artifact_role="hdf5",
                            )
                        obj = group.get(name)
                        if isinstance(obj, h5py.Group):
                            walk(obj, full_name)
                        elif isinstance(obj, h5py.Dataset):
                            if bool(getattr(obj, "is_virtual", False)):
                                raise _EntryConversionError(
                                    "hdf5_virtual_dataset",
                                    scope="hdf5_metadata",
                                    artifact_role="hdf5",
                                )
                            creation = obj.id.get_create_plist()
                            if creation.get_external_count() > 0:
                                raise _EntryConversionError(
                                    "hdf5_external_storage",
                                    scope="hdf5_metadata",
                                    artifact_role="hdf5",
                                )
                            shape = tuple(int(value) for value in obj.shape)
                            if len(shape) > _MAX_HDF5_RANK or any(
                                value < 0 or value > _MAX_HDF5_DIMENSION
                                for value in shape
                            ):
                                raise _EntryConversionError(
                                    "hdf5_dimension_limit_exceeded",
                                    scope="hdf5_metadata",
                                    artifact_role="hdf5",
                                )
                            datasets[full_name] = {
                                "path": full_name,
                                "shape": list(shape),
                                "dtype": str(obj.dtype),
                            }
                            metadata_text_bytes += len(
                                str(obj.dtype).encode(
                                    "utf-8",
                                    "surrogatepass",
                                )
                            )
                            if metadata_text_bytes > _MAX_HDF5_METADATA_TEXT_BYTES:
                                raise _EntryConversionError(
                                    "hdf5_metadata_budget_exceeded",
                                    scope="hdf5_metadata",
                                    artifact_role="hdf5",
                                )
                        else:
                            raise _EntryConversionError(
                                "hdf5_unknown_object",
                                scope="hdf5_metadata",
                                artifact_role="hdf5",
                            )

                walk(handle)
                vector = datasets.get("joint_action/vector")
                head_rgb = datasets.get("observation/head_camera/rgb")
                if vector is None or head_rgb is None:
                    raise _EntryConversionError(
                        "hdf5_missing_required_dataset",
                        scope="hdf5_metadata",
                        artifact_role="hdf5",
                    )
                vector_shape = vector["shape"]
                if (
                    len(vector_shape) != 2
                    or vector_shape[1] != 14
                    or vector_shape[0] <= 0
                ):
                    raise _EntryConversionError(
                        "hdf5_invalid_joint_vector_shape",
                        scope="hdf5_metadata",
                        artifact_role="hdf5",
                    )
                frame_count = int(vector_shape[0])
                if frame_count > _MAX_HDF5_FRAMES:
                    raise _EntryConversionError(
                        "hdf5_frame_limit_exceeded",
                        scope="hdf5_metadata",
                        artifact_role="hdf5",
                    )
                if not head_rgb["shape"] or head_rgb["shape"][0] != frame_count:
                    raise _EntryConversionError(
                        "hdf5_frame_count_mismatch",
                        scope="hdf5_metadata",
                        artifact_role="hdf5",
                    )
                time_series_prefixes = (
                    "joint_action/",
                    "endpose/",
                    "observation/",
                    "third_view_rgb",
                    "pointcloud",
                )
                for dataset_name, descriptor in datasets.items():
                    if dataset_name.startswith(time_series_prefixes):
                        shape = descriptor["shape"]
                        if not shape or shape[0] != frame_count:
                            raise _EntryConversionError(
                                "hdf5_frame_count_mismatch",
                                scope="hdf5_metadata",
                                artifact_role="hdf5",
                            )
                attribute_names: set[str] = set()
                attribute_text_bytes = 0
                for attribute_number, key in enumerate(handle.attrs.keys()):
                    if attribute_number >= _MAX_HDF5_ROOT_ATTRIBUTES:
                        raise _EntryConversionError(
                            "hdf5_attribute_limit_exceeded",
                            scope="hdf5_metadata",
                            artifact_role="hdf5",
                        )
                    text = str(key)
                    attribute_text_bytes += len(text.encode("utf-8", "surrogatepass"))
                    if (
                        metadata_text_bytes + attribute_text_bytes
                        > _MAX_HDF5_METADATA_TEXT_BYTES
                    ):
                        raise _EntryConversionError(
                            "hdf5_metadata_budget_exceeded",
                            scope="hdf5_metadata",
                            artifact_role="hdf5",
                        )
                    attribute_names.add(text)
    except _EntryConversionError:
        raise
    except (MemoryError, OSError, RuntimeError, ValueError) as exc:
        raise _EntryConversionError(
            "hdf5_unreadable",
            scope="hdf5_metadata",
            artifact_role="hdf5",
        ) from exc

    depth_paths = sorted(name for name in datasets if name.endswith("/depth"))
    object_pose_paths = sorted(
        name for name in datasets if "object" in name.lower() and "pose" in name.lower()
    )
    pointcloud = datasets.get("pointcloud")
    pointcloud_available = bool(
        pointcloud and len(pointcloud["shape"]) >= 2 and pointcloud["shape"][1] > 0
    )
    explicit_success_paths = sorted(
        name for name in datasets if name.lower() in {"success", "episode_success"}
    )
    explicit_success_attrs = sorted(
        name
        for name in attribute_names
        if name.lower() in {"success", "episode_success"}
    )
    frame_dataset_keys = sorted(
        name
        for name, descriptor in datasets.items()
        if descriptor["shape"] and descriptor["shape"][0] == frame_count
    )
    if frame_count * len(frame_dataset_keys) > _MAX_HDF5_EVIDENCE_KEY_REFERENCES:
        raise _EntryConversionError(
            "hdf5_evidence_budget_exceeded",
            scope="hdf5_metadata",
            artifact_role="hdf5",
        )
    return {
        "frame_count": frame_count,
        "datasets": [datasets[name] for name in sorted(datasets)],
        "frame_dataset_keys": frame_dataset_keys,
        "depth": {"available": bool(depth_paths), "dataset_paths": depth_paths},
        "pointcloud": {
            "available": pointcloud_available,
            "shape": pointcloud["shape"] if pointcloud else None,
            "reason": None
            if pointcloud_available
            else ("empty_width" if pointcloud else "absent"),
        },
        "object_pose": {
            "available": bool(object_pose_paths),
            "dataset_paths": object_pose_paths,
        },
        "explicit_success": {
            "available": bool(explicit_success_paths or explicit_success_attrs),
            "dataset_paths": explicit_success_paths,
            "attribute_names": explicit_success_attrs,
        },
        "joint_action": {
            "available": True,
            "semantics": "observed_robot_state_not_commanded_action",
            "commanded_action_available": False,
        },
    }


class RMBenchExpertTrajectoryImporter:
    """Convert only the RMBench episodes authorized by one frozen manifest."""

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        *,
        max_pickle_bytes: int = _DEFAULT_MAX_PICKLE_BYTES,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_total_artifacts: int = _DEFAULT_MAX_TOTAL_ARTIFACTS,
        max_total_declared_bytes: int = _DEFAULT_MAX_TOTAL_DECLARED_BYTES,
        max_total_frames: int = _DEFAULT_MAX_TOTAL_FRAMES,
        max_total_evidence_entries: int = _DEFAULT_MAX_TOTAL_EVIDENCE_ENTRIES,
    ) -> None:
        self._manifest_path = Path(manifest_path)
        if (
            not isinstance(max_pickle_bytes, int)
            or isinstance(max_pickle_bytes, bool)
            or max_pickle_bytes <= 0
            or max_pickle_bytes > _HARD_MAX_PICKLE_BYTES
        ):
            raise ValueError(
                "max_pickle_bytes must be a positive integer no greater "
                f"than {_HARD_MAX_PICKLE_BYTES}"
            )
        self._max_pickle_bytes = max_pickle_bytes
        self._resource_limits = _ResourceLimits.create(
            max_entries=max_entries,
            max_total_artifacts=max_total_artifacts,
            max_total_declared_bytes=max_total_declared_bytes,
            max_total_frames=max_total_frames,
            max_total_evidence_entries=max_total_evidence_entries,
        )
        self._bound_manifest: dict[str, Any] | None = None
        self._bound_dataset_root: Path | None = None
        self._bound_dataset_root_fd: int | None = None

    @classmethod
    def _from_validated_authority(
        cls,
        *,
        manifest: Mapping[str, Any],
        dataset_root: Path,
        dataset_root_fd: int,
        max_pickle_bytes: int = _DEFAULT_MAX_PICKLE_BYTES,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_total_artifacts: int = _DEFAULT_MAX_TOTAL_ARTIFACTS,
        max_total_declared_bytes: int = _DEFAULT_MAX_TOTAL_DECLARED_BYTES,
        max_total_frames: int = _DEFAULT_MAX_TOTAL_FRAMES,
        max_total_evidence_entries: int = _DEFAULT_MAX_TOTAL_EVIDENCE_ENTRIES,
    ) -> RMBenchExpertTrajectoryImporter:
        """Build an importer over a caller-owned, already bound root FD.

        This internal constructor is used by bootstrap so the default import
        never reopens a replaceable manifest pathname.  The descriptor is
        borrowed and duplicated by ``convert``; ownership stays with caller.
        """

        instance = cls.__new__(cls)
        if (
            not isinstance(max_pickle_bytes, int)
            or isinstance(max_pickle_bytes, bool)
            or max_pickle_bytes <= 0
            or max_pickle_bytes > _HARD_MAX_PICKLE_BYTES
        ):
            raise ValueError(
                "max_pickle_bytes must be a positive integer no greater "
                f"than {_HARD_MAX_PICKLE_BYTES}"
            )
        try:
            validated = ExpertBootstrapManifestV1.from_dict(
                copy.deepcopy(dict(manifest))
            )
        except SchemaValidationError as exc:
            raise ManifestImportError(f"input manifest schema invalid: {exc}") from exc
        root_info = os.fstat(dataset_root_fd)
        if not stat.S_ISDIR(root_info.st_mode):
            raise ManifestImportError("bound dataset_root must be a directory")
        instance._manifest_path = None
        instance._max_pickle_bytes = max_pickle_bytes
        instance._resource_limits = _ResourceLimits.create(
            max_entries=max_entries,
            max_total_artifacts=max_total_artifacts,
            max_total_declared_bytes=max_total_declared_bytes,
            max_total_frames=max_total_frames,
            max_total_evidence_entries=max_total_evidence_entries,
        )
        instance._bound_manifest = validated.to_dict()
        instance._bound_dataset_root = Path(dataset_root)
        instance._bound_dataset_root_fd = dataset_root_fd
        return instance

    def stream(self) -> RMBenchImportStream:
        """Return a closable one-entry stream over the frozen manifest authority."""

        if self._bound_manifest is None:
            manifest_path, manifest = self._load_manifest()
            dataset_root: Path | None = None
            bound_dataset_root_fd: int | None = None
        else:
            manifest = copy.deepcopy(self._bound_manifest)
            assert self._bound_dataset_root is not None
            assert self._bound_dataset_root_fd is not None
            dataset_root = self._bound_dataset_root
            bound_dataset_root_fd = self._bound_dataset_root_fd
        if manifest.get("benchmark", "rmbench") != "rmbench":
            raise ManifestImportError(
                "RMBench importer requires manifest benchmark='rmbench'"
            )
        if dataset_root is None:
            dataset_root = self._dataset_root(manifest, manifest_path)
            dataset_root_fd = _secure_open_dataset_root(dataset_root)
        else:
            assert bound_dataset_root_fd is not None
            try:
                dataset_root_fd = os.dup(bound_dataset_root_fd)
            except OSError as exc:
                raise ManifestImportError(
                    f"bound dataset_root descriptor is unavailable: {exc}"
                ) from exc
        try:
            root_stat = os.fstat(dataset_root_fd)
            plans, usage = self._preflight_manifest(manifest)
            iterator = self._iterate_entries(
                manifest=manifest,
                dataset_root=dataset_root,
                dataset_root_fd=dataset_root_fd,
                root_stat=root_stat,
                plans=plans,
                usage=usage,
            )
            return RMBenchImportStream(
                iterator,
                usage,
                manifest,
                dataset_root_fd,
            )
        except Exception:
            os.close(dataset_root_fd)
            raise

    def convert(self) -> RMBenchImportBatch:
        """Compatibility batch facade over the bounded one-entry stream."""

        trajectories: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        evidence_entries: list[dict[str, Any]] = []
        abstentions: list[dict[str, Any]] = []
        with self.stream() as stream:
            manifest = stream.input_manifest
            for converted in stream:
                if converted.trajectory is not None:
                    trajectories.append(converted.trajectory)
                    segments.extend(converted.subtask_segments)
                    evidence_entries.extend(converted.evidence_entries)
                abstentions.extend(converted.abstentions)
        return RMBenchImportBatch(
            trajectories=tuple(copy.deepcopy(trajectories)),
            subtask_segments=tuple(copy.deepcopy(segments)),
            evidence_entries=tuple(copy.deepcopy(evidence_entries)),
            abstentions=tuple(copy.deepcopy(abstentions)),
            input_manifest=copy.deepcopy(manifest),
        )

    def _iterate_entries(
        self,
        *,
        manifest: Mapping[str, Any],
        dataset_root: Path,
        dataset_root_fd: int,
        root_stat: os.stat_result,
        plans: Sequence[_EntryPlan],
        usage: _ResourceUsageState,
    ) -> Iterator[RMBenchImportedEntry]:
        for plan in plans:
            authorized = self._authorize_entry(
                plan=plan,
                dataset_root=dataset_root,
                dataset_root_fd=dataset_root_fd,
            )
            try:
                for artifact_plan in plan.artifacts:
                    artifact = authorized[artifact_plan.role]
                    before_parse = _capture_artifact_snapshot(
                        artifact.fd,
                        label=artifact_plan.label,
                    )
                    _assert_same_artifact_snapshot(
                        before_parse,
                        artifact.authorized_snapshot,
                        label=artifact_plan.label,
                        phase="between authorization and parsing",
                    )
                try:
                    try:
                        (
                            record,
                            entry_segments,
                            entry_evidence,
                            entry_abstentions,
                        ) = self._convert_entry(
                            entry_index=plan.entry_index,
                            entry=plan.entry,
                            trajectory_id=plan.trajectory_id,
                            artifacts=authorized,
                            manifest=manifest,
                            usage=usage,
                        )
                    except _EntryConversionError as exc:
                        converted = RMBenchImportedEntry(
                            entry_index=plan.entry_index,
                            trajectory_id=plan.trajectory_id,
                            trajectory=None,
                            subtask_segments=(),
                            evidence_entries=(),
                            abstentions=(
                                ConverterAbstention(
                                    reason=exc.reason,
                                    scope=exc.scope,
                                    entry_index=plan.entry_index,
                                    trajectory_id=plan.trajectory_id,
                                    artifact_role=exc.artifact_role,
                                ).to_dict(),
                            ),
                        )
                    else:
                        converted = RMBenchImportedEntry(
                            entry_index=plan.entry_index,
                            trajectory_id=plan.trajectory_id,
                            trajectory=record,
                            subtask_segments=tuple(entry_segments),
                            evidence_entries=tuple(entry_evidence),
                            abstentions=tuple(entry_abstentions),
                        )
                finally:
                    self._verify_authorized_entry(
                        plan=plan,
                        artifacts=authorized,
                        dataset_root=dataset_root,
                        dataset_root_fd=dataset_root_fd,
                    )
            finally:
                for artifact in authorized.values():
                    os.close(artifact.fd)
            yield converted

        try:
            current_root = os.stat(dataset_root, follow_symlinks=False)
        except OSError as exc:
            raise ManifestImportError(
                f"dataset_root changed during import: {exc}"
            ) from exc
        if not stat.S_ISDIR(current_root.st_mode) or (
            current_root.st_dev,
            current_root.st_ino,
        ) != (root_stat.st_dev, root_stat.st_ino):
            raise ManifestImportError(
                "dataset_root path substitution detected during import"
            )

    def _load_manifest(self) -> tuple[Path, dict[str, Any]]:
        assert self._manifest_path is not None
        path, raw = _secure_read_manifest_path(self._manifest_path)
        payload = _strict_json_bytes(raw, label="input manifest")
        if not isinstance(payload, dict):
            raise ManifestImportError("input manifest must be one JSON object")
        try:
            validated = ExpertBootstrapManifestV1.from_dict(payload)
        except SchemaValidationError as exc:
            raise ManifestImportError(f"input manifest schema invalid: {exc}") from exc
        return path, validated.to_dict()

    @staticmethod
    def _dataset_root(manifest: Mapping[str, Any], _manifest_path: Path) -> Path:
        raw = Path(str(manifest["dataset_root"]))
        if not raw.is_absolute():
            raise ManifestImportError(
                "relative dataset_root is not accepted because it can be "
                "rebound through the manifest parent; use an absolute path"
            )
        if ".." in raw.parts or len(raw.parts) < 2:
            raise ManifestImportError("dataset_root path is unsafe")
        return raw

    def _preflight_manifest(
        self,
        manifest: Mapping[str, Any],
    ) -> tuple[tuple[_EntryPlan, ...], _ResourceUsageState]:
        """Validate every declaration and global budget without opening artifacts."""

        entries = manifest["entries"]
        entry_count = len(entries)
        if entry_count > self._resource_limits.max_entries:
            raise ResourceBudgetExceeded(
                "max_entries exceeded: "
                f"limit={self._resource_limits.max_entries}, attempted={entry_count}"
            )
        shared_descriptors = dict(manifest.get("shared_artifacts", {}))
        artifact_count = len(shared_descriptors) + sum(
            len(dict(entry["artifacts"])) for entry in entries
        )
        if artifact_count > self._resource_limits.max_total_artifacts:
            raise ResourceBudgetExceeded(
                "max_total_artifacts exceeded: "
                f"limit={self._resource_limits.max_total_artifacts}, "
                f"attempted={artifact_count}"
            )
        declared_bytes = sum(
            int(descriptor["size_bytes"]) for descriptor in shared_descriptors.values()
        )
        declared_bytes += sum(
            int(descriptor["size_bytes"])
            for entry in entries
            for descriptor in dict(entry["artifacts"]).values()
            if "shared_artifact_ref" not in descriptor
        )
        if declared_bytes > self._resource_limits.max_total_declared_bytes:
            raise ResourceBudgetExceeded(
                "max_total_declared_bytes exceeded: "
                f"limit={self._resource_limits.max_total_declared_bytes}, "
                f"attempted={declared_bytes}"
            )

        seen_source_ids: set[str] = set()
        seen_paths: set[str] = set()
        seen_hashes: set[str] = set()
        shared_plans: dict[str, _ArtifactPlan] = {}
        used_shared: set[str] = set()

        def declare(
            *,
            role: str,
            descriptor: Mapping[str, Any],
            default_source_ref_id: str,
            label: str,
        ) -> _ArtifactPlan:
            relative_path = str(descriptor["relative_path"])
            expected_hash = str(descriptor["sha256"])
            if relative_path in seen_paths:
                raise ManifestImportError(
                    f"duplicate resolved artifact path {relative_path!r}"
                )
            if expected_hash in seen_hashes:
                raise ManifestImportError(
                    f"duplicate artifact content SHA-256 {expected_hash!r}"
                )
            explicit_source_id = descriptor.get("source_ref_id")
            source_ref_id = (
                str(explicit_source_id)
                if explicit_source_id is not None
                else default_source_ref_id
            )
            if source_ref_id in seen_source_ids:
                raise ManifestImportError(f"duplicate source_ref_id {source_ref_id!r}")
            seen_paths.add(relative_path)
            seen_hashes.add(expected_hash)
            seen_source_ids.add(source_ref_id)
            return _ArtifactPlan(
                role=role,
                descriptor=descriptor,
                default_source_ref_id=source_ref_id,
                label=label,
            )

        for shared_id, raw_descriptor in shared_descriptors.items():
            descriptor = dict(raw_descriptor)
            role = str(descriptor["artifact_role"])
            if role not in _ARTIFACT_KIND:
                raise ManifestImportError(
                    f"shared artifact {shared_id!r} has unknown role {role!r}"
                )
            safe_shared_id = _SAFE_ID_RE.sub("_", str(shared_id)).strip("_")
            if not safe_shared_id:
                raise ManifestImportError(
                    f"shared artifact {shared_id!r} has invalid identifier"
                )
            shared_plans[str(shared_id)] = declare(
                role=role,
                descriptor=descriptor,
                default_source_ref_id=f"shared_{safe_shared_id}",
                label=f"shared artifact {shared_id}",
            )

        plans: list[_EntryPlan] = []
        for entry_index, raw_entry in enumerate(entries):
            entry = dict(raw_entry)
            if entry.get("benchmark", "rmbench") != "rmbench":
                raise ManifestImportError(
                    f"entry {entry_index} requires benchmark='rmbench'"
                )
            trajectory_id = _safe_trajectory_id(entry)
            artifacts = entry["artifacts"]
            if not isinstance(artifacts, dict) or not set(_REQUIRED_ARTIFACTS).issubset(
                artifacts
            ):
                raise ManifestImportError(
                    f"entry {entry_index} lacks required RMBench artifacts"
                )
            entry_plans: list[_ArtifactPlan] = []
            for role, raw_descriptor in artifacts.items():
                if role not in _ARTIFACT_KIND:
                    raise ManifestImportError(
                        f"entry {entry_index} has unknown artifact role {role!r}"
                    )
                descriptor = dict(raw_descriptor)
                shared_id = descriptor.get("shared_artifact_ref")
                if shared_id is not None:
                    shared_plan = shared_plans.get(str(shared_id))
                    if shared_plan is None:
                        raise ManifestImportError(
                            f"entry {entry_index} references unknown shared artifact {shared_id!r}"
                        )
                    if shared_plan.role != role:
                        raise ManifestImportError(
                            f"entry {entry_index} shared artifact role mismatch for {role}"
                        )
                    used_shared.add(str(shared_id))
                    entry_plans.append(shared_plan)
                    continue
                entry_plans.append(
                    declare(
                        role=role,
                        descriptor=descriptor,
                        default_source_ref_id=_safe_source_ref_id(
                            trajectory_id, role, descriptor
                        ),
                        label=f"entry {entry_index} role {role}",
                    )
                )
            plans.append(
                _EntryPlan(
                    entry_index=entry_index,
                    entry=entry,
                    trajectory_id=trajectory_id,
                    artifacts=tuple(entry_plans),
                )
            )
        unused_shared = sorted(set(shared_plans) - used_shared)
        if unused_shared:
            raise ManifestImportError(
                "unreferenced shared artifact(s): " + ", ".join(unused_shared)
            )
        usage = _ResourceUsageState(
            limits=self._resource_limits,
            entries=entry_count,
            artifacts=artifact_count,
            declared_bytes=declared_bytes,
        )
        return tuple(plans), usage

    @staticmethod
    def _authorize_entry(
        *,
        plan: _EntryPlan,
        dataset_root: Path,
        dataset_root_fd: int,
    ) -> dict[str, _AuthorizedArtifact]:
        authorized: dict[str, _AuthorizedArtifact] = {}
        try:
            for artifact_plan in plan.artifacts:
                descriptor = artifact_plan.descriptor
                relative_path = str(descriptor["relative_path"])
                size_bytes = int(descriptor["size_bytes"])
                expected_hash = str(descriptor["sha256"])
                path, fd, _opened = _open_regular_artifact(
                    dataset_root,
                    dataset_root_fd,
                    relative_path,
                )
                try:
                    snapshot = _capture_artifact_snapshot(
                        fd,
                        label=artifact_plan.label,
                    )
                    if snapshot.size_bytes != size_bytes:
                        raise ManifestImportError(
                            f"artifact size mismatch for {artifact_plan.label}: "
                            f"expected {size_bytes}, got {snapshot.size_bytes}"
                        )
                    if snapshot.sha256 != expected_hash:
                        raise ManifestImportError(
                            f"artifact SHA-256 mismatch for {artifact_plan.label}"
                        )
                    authorized[artifact_plan.role] = _AuthorizedArtifact(
                        role=artifact_plan.role,
                        path=path,
                        relative_path=relative_path,
                        size_bytes=size_bytes,
                        sha256=expected_hash,
                        source_ref_id=artifact_plan.default_source_ref_id,
                        fd=fd,
                        authorized_snapshot=snapshot,
                    )
                except Exception:
                    os.close(fd)
                    raise
            return authorized
        except Exception:
            for artifact in authorized.values():
                os.close(artifact.fd)
            raise

    @staticmethod
    def _verify_authorized_entry(
        *,
        plan: _EntryPlan,
        artifacts: Mapping[str, _AuthorizedArtifact],
        dataset_root: Path,
        dataset_root_fd: int,
    ) -> None:
        for artifact_plan in plan.artifacts:
            artifact = artifacts[artifact_plan.role]
            after_parse = _capture_artifact_snapshot(
                artifact.fd,
                label=artifact_plan.label,
            )
            _assert_same_artifact_snapshot(
                after_parse,
                artifact.authorized_snapshot,
                label=artifact_plan.label,
                phase="while parsing",
            )
            _, reopened_fd, _reopened = _open_regular_artifact(
                dataset_root,
                dataset_root_fd,
                artifact.relative_path,
            )
            try:
                reopened_snapshot = _capture_artifact_snapshot(
                    reopened_fd,
                    label=artifact_plan.label,
                )
                _assert_same_artifact_snapshot(
                    reopened_snapshot,
                    artifact.authorized_snapshot,
                    label=artifact_plan.label,
                    phase="or its path was substituted while parsing",
                )
            finally:
                os.close(reopened_fd)

    def _convert_entry(
        self,
        *,
        entry_index: int,
        entry: Mapping[str, Any],
        trajectory_id: str,
        artifacts: Mapping[str, _AuthorizedArtifact],
        manifest: Mapping[str, Any],
        usage: _ResourceUsageState,
    ) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        created_at = manifest.get("created_at", _utc_now())
        if not isinstance(created_at, str):
            raise _EntryConversionError("invalid_created_at", scope="entry")
        instruction = self._instruction(entry, artifacts["instruction"])
        metadata = _hdf5_metadata(artifacts["hdf5"])
        frame_count = metadata["frame_count"]
        usage.consume_frames(frame_count)
        if metadata["explicit_success"]["available"]:
            raise _EntryConversionError(
                "hdf5_explicit_success_value_unsupported",
                scope="hdf5_metadata",
                artifact_role="hdf5",
            )
        language = self._language_spans(entry, artifacts["language_annotation"])
        if sum(item[1] for item in language) != frame_count:
            raise _EntryConversionError(
                "language_span_coverage_mismatch",
                scope="subtask_segmentation",
                artifact_role="language_annotation",
            )

        source_refs = [
            self._source_ref(
                artifact=artifacts[role],
                entry=entry,
                trajectory_id=trajectory_id,
            )
            for role in artifacts
        ]
        usage.consume_evidence(frame_count)
        frame_evidence = [
            {
                "schema": "roboharn_evo/evidence_index_entry/v1",
                "schema_version": 1,
                "evidence_ref": f"{trajectory_id}#frame/{index:06d}",
                "source_ref_id": artifacts["hdf5"].source_ref_id,
                "locator": {
                    "kind": "hdf5_frame",
                    "frame_index": index,
                    "dataset_keys": copy.deepcopy(metadata["frame_dataset_keys"]),
                },
            }
            for index in range(frame_count)
        ]
        evidence_entries = list(frame_evidence)
        entry_abstentions: list[dict[str, Any]] = []
        planner_derivation: dict[str, Any]
        try:
            planner_summaries, planner_locators = _planner_summary(
                artifacts["planner_path"],
                max_bytes=self._max_pickle_bytes,
                max_evidence_entries=(
                    usage.limits.max_total_evidence_entries - usage.evidence_entries
                ),
            )
        except ResourceBudgetExceeded:
            raise
        except (
            OSError,
            pickle.PickleError,
            ValueError,
            TypeError,
            AttributeError,
            EOFError,
            MemoryError,
            OverflowError,
            RecursionError,
        ) as exc:
            planner_derivation = {
                "kind": "rmbench_planner_metadata",
                "available": False,
                "reason": "unsafe_or_unsupported_planner_pickle",
                "source_ref_id": artifacts["planner_path"].source_ref_id,
            }
            entry_abstentions.append(
                ConverterAbstention(
                    reason="unsafe_or_unsupported_planner_pickle",
                    scope="planner_metadata",
                    entry_index=entry_index,
                    trajectory_id=trajectory_id,
                    artifact_role="planner_path",
                    details={"error_type": type(exc).__name__},
                ).to_dict()
            )
        else:
            usage.consume_evidence(len(planner_locators))
            planner_derivation = {
                "kind": "rmbench_planner_metadata",
                "available": True,
                "source_ref_id": artifacts["planner_path"].source_ref_id,
                "arms": planner_summaries,
            }
            for locator in planner_locators:
                evidence_entries.append(
                    {
                        "schema": "roboharn_evo/evidence_index_entry/v1",
                        "schema_version": 1,
                        "evidence_ref": (
                            f"{trajectory_id}#plan/{locator['arm']}/"
                            f"{locator['segment_index']:06d}"
                        ),
                        "source_ref_id": artifacts["planner_path"].source_ref_id,
                        "locator": {
                            "kind": "planner_segment",
                            **copy.deepcopy(locator),
                        },
                    }
                )

        segments = self._segments(
            trajectory_id=trajectory_id,
            task_family=str(entry.get("task_family", entry["task"])),
            language=language,
            created_at=created_at,
        )
        derivations: list[dict[str, Any]] = [
            {
                "kind": "rmbench_hdf5_metadata",
                "source_ref_id": artifacts["hdf5"].source_ref_id,
                **copy.deepcopy(metadata),
            },
            planner_derivation,
            {
                "kind": "benchmark_scripted_annotation",
                "source_ref_id": artifacts["language_annotation"].source_ref_id,
                "oracle_derived": True,
                "frame_span_semantics": "contiguous_frame_count",
            },
        ]
        if "qwen_annotation" in artifacts:
            try:
                derivations.extend(
                    self._qwen_derivations(
                        entry,
                        artifacts["qwen_annotation"],
                        instruction=instruction,
                        segments=segments,
                    )
                )
            except _EntryConversionError as exc:
                entry_abstentions.append(
                    ConverterAbstention(
                        reason=exc.reason,
                        scope=exc.scope,
                        entry_index=entry_index,
                        trajectory_id=trajectory_id,
                        artifact_role=exc.artifact_role,
                    ).to_dict()
                )

        trajectory = {
            "trajectory_id": trajectory_id,
            "instruction": instruction,
            # RMBench 专家 HDF5 包含有序帧，没有原生 RoboHarn-Evo trace。
            # Frame order lives in the separate evidence index; synthetic trace
            # events would incorrectly masquerade as raw events.
            "trace_events": [],
            "outcome": {
                "status": "success",
                "evidence_source": "benchmark_collection_contract",
                "explicit_in_artifact": False,
            },
            "schema_version": 1,
        }
        provenance = {
            "schema": "roboharn_evo/provenance/v1",
            "schema_version": 1,
            "source_kind": "benchmark_expert",
            "source_subtype": "benchmark_scripted_expert",
            "source_refs": source_refs,
            "producer": {
                "kind": "importer",
                "name": "rmbench_expert_importer",
                "version": "1",
            },
            "created_at": created_at,
            "information_access": {
                "rollout_oracle_used": None,
                "expert_prior_used": None,
                "cross_rollout_memory_enabled": None,
            },
            "oracle_derived": True,
            "expert_derived": True,
            "license_status": manifest.get("license_status", "unverified"),
        }
        record_payload = {
            "schema": "roboharn_evo/trajectory_record/v1",
            "schema_version": 1,
            "trajectory": trajectory,
            "provenance": provenance,
            "evidence_index": evidence_entries,
            "scene_memory": {
                "available": False,
                "snapshot": {},
                "source": "unavailable",
            },
            "subtask_segments": segments,
            "derivations": derivations,
        }
        try:
            record = TrajectoryRecordV1.from_dict(record_payload).to_dict()
        except SchemaValidationError as exc:
            raise _EntryConversionError(
                "normalized_trajectory_invalid", scope="entry"
            ) from exc
        return record, segments, evidence_entries, entry_abstentions

    @staticmethod
    def _instruction(
        entry: Mapping[str, Any],
        artifact: _AuthorizedArtifact,
    ) -> str:
        try:
            payload = _read_json(artifact, label="instruction artifact")
        except ManifestImportError as exc:
            raise _EntryConversionError(
                "instruction_unreadable",
                scope="instruction",
                artifact_role="instruction",
            ) from exc
        if not isinstance(payload, dict):
            raise _EntryConversionError(
                "instruction_malformed",
                scope="instruction",
                artifact_role="instruction",
            )
        variant = entry["instruction_variant"]
        values = payload.get(variant)
        if not isinstance(values, list) or not values:
            raise _EntryConversionError(
                "instruction_variant_unavailable",
                scope="instruction",
                artifact_role="instruction",
            )
        index = entry.get("instruction_index", 0)
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(values)
        ):
            raise _EntryConversionError(
                "instruction_index_invalid",
                scope="instruction",
                artifact_role="instruction",
            )
        instruction = values[index]
        if not isinstance(instruction, str):
            raise _EntryConversionError(
                "instruction_malformed",
                scope="instruction",
                artifact_role="instruction",
            )
        return instruction

    @staticmethod
    def _language_spans(
        entry: Mapping[str, Any],
        artifact: _AuthorizedArtifact,
    ) -> list[tuple[str, int]]:
        try:
            payload = _read_json(artifact, label="language annotation artifact")
        except ManifestImportError as exc:
            raise _EntryConversionError(
                "language_annotation_unreadable",
                scope="subtask_segmentation",
                artifact_role="language_annotation",
            ) from exc
        if not isinstance(payload, dict):
            raise _EntryConversionError(
                "language_annotation_malformed",
                scope="subtask_segmentation",
                artifact_role="language_annotation",
            )
        episode_key = f"episode_{entry['episode_id']}"
        raw_spans = payload.get(episode_key)
        if not isinstance(raw_spans, list) or not raw_spans:
            raise _EntryConversionError(
                "language_episode_unavailable",
                scope="subtask_segmentation",
                artifact_role="language_annotation",
            )
        spans: list[tuple[str, int]] = []
        for raw_span in raw_spans:
            if not isinstance(raw_span, list) or len(raw_span) != 2:
                raise _EntryConversionError(
                    "language_annotation_malformed",
                    scope="subtask_segmentation",
                    artifact_role="language_annotation",
                )
            text, frame_count = raw_span
            if (
                not isinstance(text, str)
                or not isinstance(frame_count, int)
                or isinstance(frame_count, bool)
                or frame_count <= 0
            ):
                raise _EntryConversionError(
                    "language_annotation_malformed",
                    scope="subtask_segmentation",
                    artifact_role="language_annotation",
                )
            spans.append((text, frame_count))
        return spans

    @staticmethod
    def _source_ref(
        *,
        artifact: _AuthorizedArtifact,
        entry: Mapping[str, Any],
        trajectory_id: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source_ref_id": artifact.source_ref_id,
            "content_sha256": artifact.sha256,
            "artifact_kind": _ARTIFACT_KIND[artifact.role],
            "relative_path": artifact.relative_path,
            "size_bytes": artifact.size_bytes,
            "trajectory_id": trajectory_id,
            "benchmark": "rmbench",
            "task": entry["task"],
            "seed": entry["seed"],
            "episode_id": entry["episode_id"],
        }
        split = entry.get("split")
        if isinstance(split, str) and split:
            result["split"] = split
        return result

    @staticmethod
    def _segments(
        *,
        trajectory_id: str,
        task_family: str,
        language: Sequence[tuple[str, int]],
        created_at: str,
    ) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        start = 0
        for segment_index, (instruction, frame_count) in enumerate(language):
            end = start + frame_count - 1
            before_ref = f"{trajectory_id}#frame/{start:06d}"
            after_ref = f"{trajectory_id}#frame/{end:06d}"
            evidence_refs = list(dict.fromkeys((before_ref, after_ref)))
            payload = {
                "schema": "roboharn_evo/subtask_segment/v1",
                "schema_version": 1,
                "segment_id": f"{trajectory_id}/subtask_{segment_index:02d}",
                "trajectory_id": trajectory_id,
                "segment_index": segment_index,
                "context": {
                    "subtask_instruction": instruction,
                    "participants": [],
                    "task_family": task_family,
                },
                "transition": {
                    "before_event_refs": [before_ref],
                    "action_event_refs": [],
                    "after_event_refs": [after_ref],
                },
                "outcome": {
                    "status": "unknown",
                    "observed_effects": [],
                },
                "derivation": {
                    "producer": {
                        "kind": "benchmark",
                        "name": "benchmark_scripted_annotation",
                        "version": "1",
                    },
                    "confidence": "medium",
                    "evidence_event_refs": evidence_refs,
                    "created_at": created_at,
                    "source": "benchmark_scripted_annotation",
                    "oracle_derived": True,
                    "frame_range": {
                        "start_inclusive": start,
                        "end_inclusive": end,
                    },
                },
            }
            try:
                normalized = SubtaskSegmentV1.from_dict(payload).to_dict()
            except SchemaValidationError as exc:
                raise _EntryConversionError(
                    "subtask_segment_invalid",
                    scope="subtask_segmentation",
                    artifact_role="language_annotation",
                ) from exc
            segments.append(normalized)
            start = end + 1
        return segments

    @staticmethod
    def _qwen_derivations(
        entry: Mapping[str, Any],
        artifact: _AuthorizedArtifact,
        *,
        instruction: str,
        segments: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if artifact.size_bytes > _MAX_JSON_BYTES:
            raise _EntryConversionError(
                "qwen_annotation_too_large",
                scope="derived_annotation",
                artifact_role="qwen_annotation",
            )
        derivations: list[dict[str, Any]] = []
        seen_segments: set[int] = set()
        expected_episode_id = f"{entry['task']}/episode{entry['episode_id']}"
        try:
            with artifact.open_binary() as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.rstrip(b"\r\n")
                    if not line:
                        continue
                    payload = _strict_json_bytes(
                        line,
                        label=f"qwen annotation line {line_number}",
                    )
                    if not isinstance(payload, dict):
                        raise ManifestImportError("Qwen JSONL line must be an object")
                    if payload.get("episode_id") != expected_episode_id:
                        continue
                    segment_id = payload.get("segment_id")
                    chunk_id = payload.get("chunk_id")
                    t_start = payload.get("t_start")
                    t_end = payload.get("t_end")
                    if any(
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value < 0
                        for value in (segment_id, chunk_id, t_start, t_end)
                    ):
                        raise ManifestImportError(
                            "Qwen segment/chunk/frame identifiers must be non-negative integers"
                        )
                    assert isinstance(segment_id, int)
                    assert isinstance(chunk_id, int)
                    assert isinstance(t_start, int)
                    assert isinstance(t_end, int)
                    if segment_id in seen_segments or segment_id != len(derivations):
                        raise ManifestImportError(
                            "Qwen segment rows must be unique and ordered from zero"
                        )
                    if chunk_id != segment_id or segment_id >= len(segments):
                        raise ManifestImportError(
                            "Qwen chunk/segment identity is inconsistent"
                        )
                    normalized_segment = segments[segment_id]
                    frame_range = normalized_segment["derivation"]["frame_range"]
                    if (
                        t_start != frame_range["start_inclusive"]
                        or t_end != frame_range["end_inclusive"]
                    ):
                        raise ManifestImportError(
                            "Qwen frame range does not match scripted annotation"
                        )
                    if (
                        payload.get("subtask")
                        != normalized_segment["context"]["subtask_instruction"]
                    ):
                        raise ManifestImportError(
                            "Qwen subtask does not match scripted annotation"
                        )
                    if payload.get("task") != instruction:
                        raise ManifestImportError(
                            "Qwen task text does not match selected instruction"
                        )
                    memory_text = payload.get("memory_text")
                    if not isinstance(memory_text, str):
                        raise ManifestImportError("Qwen memory_text must be a string")
                    model_id = payload.get("model_name")
                    label_source = payload.get("label_source")
                    if (
                        not isinstance(model_id, str)
                        or not model_id.strip()
                        or not isinstance(label_source, str)
                        or not label_source.strip()
                    ):
                        raise ManifestImportError(
                            "Qwen producer model_name/label_source must be non-empty strings"
                        )
                    derivation: dict[str, Any] = {
                        "kind": "model_derived_memory_annotation",
                        "source_ref_id": artifact.source_ref_id,
                        "line_number": line_number,
                        "content_sha256": hashlib.sha256(line).hexdigest(),
                        "memory_text": memory_text,
                        "raw_fact": False,
                        "producer": {
                            "kind": "model",
                            "model_id": model_id,
                            "label_source": label_source,
                        },
                        "confidence": "unknown",
                        "segment_id": segment_id,
                        "chunk_id": chunk_id,
                        "subtask": payload["subtask"],
                        "t_start": t_start,
                        "t_end": t_end,
                    }
                    previous = payload.get("previous_memory_text")
                    if isinstance(previous, str):
                        derivation["previous_memory_text"] = previous
                    derivations.append(derivation)
                    seen_segments.add(segment_id)
        except _EntryConversionError:
            raise
        except (OSError, ManifestImportError) as exc:
            raise _EntryConversionError(
                "qwen_annotation_unreadable",
                scope="derived_annotation",
                artifact_role="qwen_annotation",
            ) from exc
        if not derivations:
            raise _EntryConversionError(
                "qwen_episode_unavailable",
                scope="derived_annotation",
                artifact_role="qwen_annotation",
            )
        return derivations


def import_rmbench_expert_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    max_pickle_bytes: int = _DEFAULT_MAX_PICKLE_BYTES,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    max_total_artifacts: int = _DEFAULT_MAX_TOTAL_ARTIFACTS,
    max_total_declared_bytes: int = _DEFAULT_MAX_TOTAL_DECLARED_BYTES,
    max_total_frames: int = _DEFAULT_MAX_TOTAL_FRAMES,
    max_total_evidence_entries: int = _DEFAULT_MAX_TOTAL_EVIDENCE_ENTRIES,
) -> RMBenchImportBatch:
    """Import one explicit manifest without discovering any other episode."""

    return RMBenchExpertTrajectoryImporter(
        manifest_path,
        max_pickle_bytes=max_pickle_bytes,
        max_entries=max_entries,
        max_total_artifacts=max_total_artifacts,
        max_total_declared_bytes=max_total_declared_bytes,
        max_total_frames=max_total_frames,
        max_total_evidence_entries=max_total_evidence_entries,
    ).convert()


__all__ = [
    "ConverterAbstention",
    "ManifestImportError",
    "OptionalExpertDependencyError",
    "RMBenchExpertTrajectoryImporter",
    "RMBenchImportBatch",
    "import_rmbench_expert_manifest",
]
