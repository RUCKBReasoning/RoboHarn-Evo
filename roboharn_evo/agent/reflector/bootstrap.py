"""Atomic orchestration for the offline Self-Evolution Phase A bootstrap.

The default production path composes the built-in explicit-manifest RMBench
importer with the built-in candidate-only procedure Reflector.  Only that
default path carries the manifest-only and side-effect-free component
assurance documented by this module.  Optional Python component hooks are an
in-process compatibility/test seam: their side effects cannot be sandboxed or
verified, and every resulting audit records that limitation explicitly.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .candidate import CandidateProcedureReflector, ScriptedProcedureBackend
from .contracts import Reflector, ReflectorInput
from .evidence import EvidenceIndex, EvidenceRefV1
from .schemas import (
    ExpertBootstrapManifestV1,
    ProcedureExperienceV1,
    SchemaValidationError,
    SubtaskSegmentV1,
    TrajectoryRecordV1,
    stable_experience_id,
    validate_reflector_candidate,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PHASE_A_ROOT = _REPO_ROOT / "eval_result" / "self_evolution" / "phase_a"
_TMP_ROOT = Path("/tmp")

_MAX_INPUT_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_STAGED_JSONL_RECORD_BYTES = 256 * 1024 * 1024
_MAX_REPORT_FIELD_CHARS = 160
_MAX_REPORT_ABSTENTION_GROUPS = 100
_MARKDOWN_SPECIALS = frozenset("\\`*_{}[]<>()#+-.!|")
_RENAME_NOREPLACE = 1

_OUTPUT_FILES = (
    "input_manifest.json",
    "normalized_trajectories.jsonl",
    "evidence_index.jsonl",
    "subtask_segments.jsonl",
    "candidate_experiences.jsonl",
    "abstentions.jsonl",
    "run_manifest.json",
    "review_report.md",
)

_JSONL_SCHEMAS = {
    "normalized_trajectories.jsonl": "roboharn_evo/trajectory_record/v1",
    "evidence_index.jsonl": "roboharn_evo/evidence_index_entry/v1",
    "subtask_segments.jsonl": "roboharn_evo/subtask_segment/v1",
    "candidate_experiences.jsonl": "roboharn_evo/experience/procedure/v1",
    "abstentions.jsonl": "roboharn_evo/bootstrap_abstention/v1",
}


class BootstrapError(RuntimeError):
    """Base error for a failed Phase A bootstrap run."""


class OutputBoundaryError(BootstrapError):
    """Raised before writing when the requested output is not isolated."""


class _AtomicDirectoryNoReplaceUnavailable(BootstrapError):
    """Signal that this filesystem needs the safe symlink publication path."""


@dataclass(frozen=True, slots=True)
class BootstrapRunResult:
    """Summary of an atomically published Phase A output directory."""

    output_root: Path
    input_manifest_sha256: str
    counts: Mapping[str, int]
    file_sha256: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "input_manifest_sha256": self.input_manifest_sha256,
            "counts": dict(self.counts),
            "file_sha256": dict(self.file_sha256),
        }


@dataclass(frozen=True, slots=True)
class BootstrapResourceLimits:
    """Fail-closed budgets for the built-in episode-streaming importer.

    The conservative defaults intentionally do not admit RMBench's complete
    expert pool.  A full-pool operator must inspect the manifest and provide
    explicit limits suitable for that authorized run.
    """

    max_entries: int = 100
    max_total_artifacts: int = 500
    max_total_declared_bytes: int = 64 * 1024**3
    max_total_frames: int = 100_000
    max_total_evidence_entries: int = 500_000

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_entries": self.max_entries,
            "max_total_artifacts": self.max_total_artifacts,
            "max_total_declared_bytes": self.max_total_declared_bytes,
            "max_total_frames": self.max_total_frames,
            "max_total_evidence_entries": self.max_total_evidence_entries,
        }


@dataclass(frozen=True, slots=True)
class _PreparedImportChunk:
    trajectories: tuple[dict[str, Any], ...]
    evidence_entries: tuple[dict[str, Any], ...]
    segments: tuple[dict[str, Any], ...]
    abstentions: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class _ImportProcessingState:
    processing_mode: str
    resource_limits: Mapping[str, int] | None
    resource_usage: Mapping[str, int] | None = None


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
        )
    except (TypeError, ValueError) as exc:
        raise BootstrapError(f"output is not strict JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    """Hash canonical JSON without a presentation newline."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BootstrapError(f"configuration is not strict JSON: {exc}") from exc
    return _sha256_bytes(encoded)


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BootstrapError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise BootstrapError(f"{label} must contain one JSON object")
    return decoded


def _mapping_copy(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return copy.deepcopy(dict(result))
    raise BootstrapError(f"{label} must be a mapping or expose to_dict()")


def _sequence(value: Any, *, label: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BootstrapError(f"{label} must be an array")
    return list(value)


def _is_same_or_descendant(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _paths_overlap(left: Path, right: Path) -> bool:
    return _is_same_or_descendant(left, right) or _is_same_or_descendant(right, left)


def _assert_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        except OSError as exc:
            raise OutputBoundaryError(
                f"cannot inspect output path component {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise OutputBoundaryError(
                f"output path must not contain a symlink component: {current}"
            )
        if current != path and not stat.S_ISDIR(mode):
            raise OutputBoundaryError(
                f"output parent component is not a directory: {current}"
            )


def _resolve_output_root(output_root: str | os.PathLike[str]) -> Path:
    raw = Path(output_root)
    if not raw.is_absolute():
        raise OutputBoundaryError("--output-root must be an absolute path")
    if ".." in raw.parts:
        raise OutputBoundaryError("--output-root must not contain '..'")
    _assert_no_symlink_components(raw)
    resolved = raw.resolve(strict=False)
    if resolved.exists():
        raise OutputBoundaryError(f"output root already exists: {resolved}")

    phase_root = _PHASE_A_ROOT.resolve(strict=False)
    tmp_root = _TMP_ROOT.resolve(strict=True)
    under_phase_a = resolved != phase_root and resolved.is_relative_to(phase_root)
    under_tmp = resolved != tmp_root and resolved.is_relative_to(tmp_root)
    if not (under_phase_a or under_tmp):
        raise OutputBoundaryError(
            f"output root must be a fresh child of /tmp or {phase_root}"
        )
    return resolved


def _bind_dataset_root(manifest: Mapping[str, Any]) -> tuple[Path, int]:
    """Bind one absolute dataset root to a no-follow directory descriptor."""

    root = Path(str(manifest["dataset_root"]))
    if not root.is_absolute():
        raise BootstrapError(
            "relative dataset_root is not accepted because it can be rebound "
            "through the manifest parent; use an absolute path"
        )
    if ".." in root.parts or len(root.parts) < 2:
        raise BootstrapError("dataset_root path is unsafe")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise BootstrapError("secure no-follow dataset root open is unavailable")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | nofollow
    current_fd: int | None = None
    try:
        current_fd = os.open(root.anchor, directory_flags)
        for part in root.parts[1:]:
            next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        info = os.fstat(current_fd)
        if not stat.S_ISDIR(info.st_mode):
            raise BootstrapError("dataset_root must be a directory")
        return root, current_fd
    except BootstrapError:
        if current_fd is not None:
            os.close(current_fd)
        raise
    except OSError as exc:
        if current_fd is not None:
            os.close(current_fd)
        raise BootstrapError(
            "dataset_root cannot be securely bound (missing, symlink, or "
            f"non-directory component): {exc}"
        ) from exc


def _validate_output_isolation(output_root: Path, dataset_root: Path) -> None:
    protected = (
        dataset_root,
        (_REPO_ROOT / "benchmarks" / "rmbench").resolve(strict=True),
        (_REPO_ROOT / "tcm" / "resources").resolve(strict=True),
    )
    for protected_path in protected:
        if _paths_overlap(output_root, protected_path):
            raise OutputBoundaryError(
                "output root overlaps a source/protected tree: "
                f"output={output_root}, protected={protected_path}"
            )


def _same_inode(info: os.stat_result, identity: tuple[int, int]) -> bool:
    return (info.st_dev, info.st_ino) == identity


def _secure_output_parent_fd(output_root: Path) -> int:
    """Open/create the output parent beneath one trusted root without symlinks."""

    phase_root = _PHASE_A_ROOT.resolve(strict=False)
    tmp_root = _TMP_ROOT.resolve(strict=True)
    if output_root.is_relative_to(tmp_root):
        trusted_root = tmp_root
    elif output_root.is_relative_to(phase_root):
        # Start at the already imported repository root so missing
        # ``eval_result/self_evolution/phase_a`` components can be created via
        # mkdirat without ever following a substituted path component.
        trusted_root = _REPO_ROOT.resolve(strict=True)
    else:  # Defense in depth; _resolve_output_root enforces this first.
        raise OutputBoundaryError("output root is outside an authorized root")

    try:
        relative_parent = output_root.parent.relative_to(trusted_root)
    except ValueError as exc:
        raise OutputBoundaryError("output parent is outside its trusted root") from exc

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        current_fd = os.open(trusted_root, directory_flags)
    except OSError as exc:
        raise OutputBoundaryError(
            f"cannot open trusted output root {trusted_root}: {exc}"
        ) from exc

    try:
        for part in relative_parent.parts:
            created = False
            try:
                next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current_fd)
                    created = True
                except FileExistsError:
                    # A concurrent creator won.  The no-follow open below is
                    # still the authority check; a symlink or non-directory
                    # fails closed.
                    pass
                next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            if created:
                os.fsync(current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except OSError as exc:
        os.close(current_fd)
        raise OutputBoundaryError(
            "output parent is unsafe, unavailable, or contains a symlink/"
            f"non-directory component: {exc}"
        ) from exc


def _create_staging_directory(
    *,
    parent_fd: int,
    output_name: str,
) -> tuple[str, int, tuple[int, int]]:
    encoded_name = os.fsencode(output_name)
    label = (
        output_name
        if len(encoded_name) <= 96
        else hashlib.sha256(encoded_name).hexdigest()[:16]
    )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _ in range(128):
        staging_name = f".{label}.phase-a-object-{secrets.token_hex(12)}"
        try:
            os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            staging_fd = os.open(
                staging_name,
                directory_flags,
                dir_fd=parent_fd,
            )
        except OSError:
            try:
                os.rmdir(staging_name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
        info = os.fstat(staging_fd)
        return staging_name, staging_fd, (info.st_dev, info.st_ino)
    raise BootstrapError("could not allocate a unique staging directory")


def _rename_directory_noreplace(
    *,
    parent_fd: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Atomically publish a directory without replacing any existing inode."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise _AtomicDirectoryNoReplaceUnavailable(
            "atomic no-replace publication is unavailable on this platform"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return

    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise OutputBoundaryError(
            "output root appeared during run; existing path was preserved: "
            f"{destination_name}"
        )
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise _AtomicDirectoryNoReplaceUnavailable(
            "filesystem does not support atomic no-replace directory "
            "publication; refusing an unsafe fallback"
        )
    raise BootstrapError(
        "atomic output publication failed: "
        f"[{error_number}] {os.strerror(error_number)}"
    )


def _symlink_directory_noreplace(
    *,
    parent_fd: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Publish a complete same-parent backing directory without clobbering."""

    if not source_name or "/" in source_name or source_name in {".", ".."}:
        raise BootstrapError("invalid Phase A backing-directory name")
    try:
        os.symlink(
            source_name,
            destination_name,
            target_is_directory=True,
            dir_fd=parent_fd,
        )
    except FileExistsError as exc:
        raise OutputBoundaryError(
            "output root appeared during run; existing path was preserved: "
            f"{destination_name}"
        ) from exc
    except OSError as exc:
        raise BootstrapError(
            f"atomic same-parent symlink publication failed: {exc}"
        ) from exc


def _published_symlink_identity(
    *,
    parent_fd: int,
    destination_name: str,
    source_name: str,
) -> tuple[int, int]:
    info = os.stat(
        destination_name,
        dir_fd=parent_fd,
        follow_symlinks=False,
    )
    target = os.readlink(destination_name, dir_fd=parent_fd)
    if not stat.S_ISLNK(info.st_mode) or target != source_name:
        raise BootstrapError(
            "atomically published output link failed identity verification"
        )
    return (info.st_dev, info.st_ino)


def _verify_published_target(
    *,
    parent_fd: int,
    destination_name: str,
    expected_identity: tuple[int, int],
    follow_symlink: bool,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    if not follow_symlink:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    target_fd: int | None = None
    try:
        target_fd = os.open(destination_name, flags, dir_fd=parent_fd)
        info = os.fstat(target_fd)
    except OSError as exc:
        raise BootstrapError(
            f"cannot verify atomically published output target: {exc}"
        ) from exc
    finally:
        if target_fd is not None:
            os.close(target_fd)
    if not stat.S_ISDIR(info.st_mode) or not _same_inode(
        info,
        expected_identity,
    ):
        raise BootstrapError("atomically published output target was substituted")


def _unlink_owned_output_symlink(
    *,
    parent_fd: int,
    destination_name: str,
    source_name: str,
    link_identity: tuple[int, int] | None,
) -> bool:
    """Rollback only the exact symlink created by this invocation."""

    try:
        current = os.stat(
            destination_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        target = os.readlink(destination_name, dir_fd=parent_fd)
    except OSError:
        return False
    if (
        not stat.S_ISLNK(current.st_mode)
        or target != source_name
        or link_identity is None
        or not _same_inode(current, link_identity)
    ):
        return False
    try:
        os.unlink(destination_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError:
        return False
    return True


def _rollback_owned_directory_publication(
    *,
    parent_fd: int,
    destination_name: str,
    source_name: str,
    expected_identity: tuple[int, int],
) -> bool:
    try:
        current = os.stat(
            destination_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError:
        return False
    if not stat.S_ISDIR(current.st_mode) or not _same_inode(
        current,
        expected_identity,
    ):
        return False
    try:
        _rename_directory_noreplace(
            parent_fd=parent_fd,
            source_name=destination_name,
            destination_name=source_name,
        )
        os.fsync(parent_fd)
    except BootstrapError:
        return False
    return True


def _select_publication_strategy(parent_fd: int) -> str:
    """Probe the held filesystem and return the actual no-clobber strategy."""

    for _ in range(32):
        token = secrets.token_hex(12)
        source_name = f".phase-a-publish-probe-source-{token}"
        destination_name = f".phase-a-publish-probe-target-{token}"
        try:
            os.mkdir(source_name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        source = os.stat(
            source_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        identity = (source.st_dev, source.st_ino)
        moved = False
        try:
            try:
                _rename_directory_noreplace(
                    parent_fd=parent_fd,
                    source_name=source_name,
                    destination_name=destination_name,
                )
                moved = True
            except _AtomicDirectoryNoReplaceUnavailable:
                return "same_parent_relative_symlink"
            except OutputBoundaryError:
                continue
            target = os.stat(
                destination_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(target.st_mode) or not _same_inode(
                target,
                identity,
            ):
                raise BootstrapError(
                    "directory no-replace publication probe was substituted"
                )
            return "directory_rename_noreplace"
        finally:
            cleanup_name = destination_name if moved else source_name
            try:
                current = os.stat(
                    cleanup_name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if stat.S_ISDIR(current.st_mode) and _same_inode(
                    current,
                    identity,
                ):
                    os.rmdir(cleanup_name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except OSError:
                pass
    raise BootstrapError("could not safely probe output publication support")


def _cleanup_staging_directory(
    *,
    parent_fd: int,
    staging_fd: int,
    staging_name: str,
    staging_identity: tuple[int, int],
    created_files: Mapping[str, tuple[int, int]],
) -> None:
    """Delete only files and the staging inode created by this invocation."""

    for name, identity in created_files.items():
        try:
            info = os.stat(name, dir_fd=staging_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(info.st_mode) or not _same_inode(info, identity):
            raise BootstrapError(
                f"staging cleanup refused to delete a substituted artifact: {name}"
            )
        os.unlink(name, dir_fd=staging_fd)

    if os.listdir(staging_fd):
        raise BootstrapError(
            "staging cleanup found unowned entries and left them untouched"
        )
    try:
        current = os.stat(
            staging_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(current.st_mode) or not _same_inode(
        current,
        staging_identity,
    ):
        raise BootstrapError(
            "staging cleanup refused to remove a substituted directory"
        )
    os.rmdir(staging_name, dir_fd=parent_fd)


def _load_input_manifest(
    manifest_path: str | os.PathLike[str],
) -> tuple[Path, bytes, ExpertBootstrapManifestV1]:
    lexical_path = Path(manifest_path)
    if not lexical_path.is_absolute():
        lexical_path = Path.cwd() / lexical_path
    if ".." in lexical_path.parts:
        raise BootstrapError("input manifest path must not contain '..'")
    if len(lexical_path.parts) < 2:
        raise BootstrapError("--manifest must identify one regular file")

    # Resolve every component through directory descriptors.  Unlike an
    # lstat-then-read sequence, this prevents a checked parent or final file
    # from being substituted with a symlink between validation and opening.
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        directory_fd = os.open(lexical_path.anchor, directory_flags)
        for part in lexical_path.parts[1:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            lexical_path.parts[-1],
            file_flags,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise BootstrapError(
            "input manifest path is unsafe, missing, or contains a symlink/"
            f"non-directory component: {exc}"
        ) from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)

    assert file_fd is not None
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise BootstrapError("--manifest must identify one regular file")
        if before.st_size > _MAX_INPUT_MANIFEST_BYTES:
            raise BootstrapError("input manifest exceeds the 64 MiB size limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                file_fd,
                min(1024 * 1024, _MAX_INPUT_MANIFEST_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_INPUT_MANIFEST_BYTES:
                raise BootstrapError("input manifest exceeds the 64 MiB size limit")
        after = os.fstat(file_fd)
    except OSError as exc:
        raise BootstrapError(f"cannot read input manifest: {exc}") from exc
    finally:
        os.close(file_fd)

    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise BootstrapError("input manifest changed while it was being read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise BootstrapError("input manifest changed while it was being read")

    path = lexical_path
    payload = _strict_json_object(raw, label="input manifest")
    try:
        record = ExpertBootstrapManifestV1.from_dict(payload)
    except SchemaValidationError as exc:
        raise BootstrapError(f"input manifest validation failed: {exc}") from exc
    return path, raw, record


def _default_importer(
    manifest: ExpertBootstrapManifestV1,
    dataset_root: Path,
    dataset_root_fd: int,
    resource_limits: BootstrapResourceLimits,
) -> Any:
    # Kept lazy so importing ``tcm`` or this module never imports h5py.
    from roboharn_evo.benchmark_adapters.rmbench.expert_trajectory import (
        RMBenchExpertTrajectoryImporter,
    )

    return RMBenchExpertTrajectoryImporter._from_validated_authority(
        manifest=manifest.to_dict(),
        dataset_root=dataset_root,
        dataset_root_fd=dataset_root_fd,
        **resource_limits.to_dict(),
    )


def _normalize_import_abstention(value: Any, *, number: int) -> dict[str, Any]:
    if isinstance(value, str):
        payload: dict[str, Any] = {"reason": value}
    else:
        payload = _mapping_copy(value, label=f"import abstention {number}")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise BootstrapError(f"import abstention {number} lacks a non-empty reason")
    scope = payload.get("scope", "expert_import")
    if not isinstance(scope, str) or not scope.strip():
        raise BootstrapError(f"import abstention {number} has an invalid scope")
    status_value = payload.get("status", "abstained")
    if status_value != "abstained":
        raise BootstrapError(
            f"import abstention {number} has non-abstention status {status_value!r}"
        )
    schema = payload.get("schema", "roboharn_evo/bootstrap_abstention/v1")
    if schema not in {"roboharn_evo/bootstrap_abstention/v1", "tcm/bootstrap_abstention/v1"}:
        raise BootstrapError(
            f"import abstention {number} has unsupported schema {schema!r}"
        )
    schema_version = payload.get("schema_version", 1)
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise BootstrapError(
            f"import abstention {number} has unsupported schema_version "
            f"{schema_version!r}"
        )
    stage = payload.get("stage", "import")
    if stage != "import":
        raise BootstrapError(
            f"import abstention {number} has conflicting stage {stage!r}"
        )
    payload["schema"] = schema
    payload["schema_version"] = schema_version
    payload["status"] = "abstained"
    payload["stage"] = stage
    payload["scope"] = scope
    payload["reason"] = reason
    _canonical_json_bytes(payload)
    return payload


def _reflection_abstention(
    *, summary: str, trajectory_id: str, segment_id: str
) -> dict[str, Any]:
    reason = "reflector_abstained"
    scope = "procedure_candidate"
    try:
        decoded = json.loads(summary)
    except (json.JSONDecodeError, TypeError):
        decoded = None
    if isinstance(decoded, Mapping):
        raw_reason = decoded.get("reason")
        raw_scope = decoded.get("scope")
        if isinstance(raw_reason, str) and raw_reason.strip():
            reason = raw_reason
        if isinstance(raw_scope, str) and raw_scope.strip():
            scope = raw_scope
    return {
        "schema": "roboharn_evo/bootstrap_abstention/v1",
        "schema_version": 1,
        "status": "abstained",
        "stage": "reflection",
        "scope": scope,
        "reason": reason,
        "trajectory_id": trajectory_id,
        "segment_id": segment_id,
    }


def _reflection_error_abstention(
    *,
    reason: str,
    trajectory_id: str,
    segment_id: str,
    error: Exception | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "roboharn_evo/bootstrap_abstention/v1",
        "schema_version": 1,
        "status": "abstained",
        "stage": "reflection",
        "scope": "procedure_candidate",
        "reason": reason,
        "trajectory_id": trajectory_id,
        "segment_id": segment_id,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
    return payload


def _normalize_reflector_output(
    output: Any,
    *,
    segment_number: int,
) -> tuple[str, dict[str, Any], list[str]]:
    """Copy and shape-check one untrusted Reflector result.

    A custom implementation only promises the public attributes at runtime;
    it may return an unrelated object, raise from an attribute, or attach
    values that violate the protocol annotations.  None of those conditions
    may abort an otherwise valid offline bootstrap.
    """

    try:
        summary = output.summary
        candidate_value = output.candidate_experience
        evidence_value = output.evidence_event_ids
    except Exception as exc:
        raise TypeError(
            f"Reflector output for segment {segment_number} lacks readable attributes"
        ) from exc
    if not isinstance(summary, str):
        raise TypeError(
            f"Reflector output summary for segment {segment_number} must be a string"
        )
    candidate = _mapping_copy(
        candidate_value,
        label=f"candidate experience for segment {segment_number}",
    )
    if isinstance(evidence_value, (str, bytes, bytearray)) or not isinstance(
        evidence_value, Sequence
    ):
        raise TypeError(
            f"Reflector evidence_event_ids for segment {segment_number} must be an array"
        )
    refs = list(evidence_value)
    if any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        raise TypeError(
            f"Reflector evidence_event_ids for segment {segment_number} "
            "must contain non-empty strings"
        )
    if len(refs) != len(set(refs)):
        raise ValueError(
            f"Reflector evidence_event_ids for segment {segment_number} contain duplicates"
        )
    if not candidate and refs:
        raise ValueError(
            f"Reflector abstention for segment {segment_number} must not claim evidence"
        )
    return summary, candidate, refs


def _segment_evidence_refs(segment: Mapping[str, Any]) -> frozenset[str]:
    """Return evidence explicitly declared by one validated normalized segment."""

    transition = segment["transition"]
    derivation = segment["derivation"]
    refs = [
        *transition["before_event_refs"],
        *transition["action_event_refs"],
        *transition["after_event_refs"],
        *derivation["evidence_event_refs"],
    ]
    for effect in segment["outcome"].get("observed_effects", []):
        refs.extend(effect["evidence_event_refs"])
    return frozenset(refs)


def _evidence_index_snapshot(
    index: EvidenceIndex,
) -> tuple[list[dict[str, Any]], str | None, frozenset[str]]:
    """Capture every semantic component of the mutable in-memory index."""

    source_ref_ids = object.__getattribute__(index, "_source_ref_ids")
    return index.to_entries(), index.trajectory_id, frozenset(source_ref_ids)


def _validate_candidate_binding(
    candidate: ProcedureExperienceV1,
    *,
    source_record: Mapping[str, Any],
    segment: Mapping[str, Any],
    trajectory_segment_ids: frozenset[str],
) -> None:
    """Bind an otherwise valid custom candidate to the current source slice."""

    payload = candidate.to_dict()
    expected_experience_id = stable_experience_id(
        payload["kind"],
        payload["condition"],
        payload["guidance"],
        payload["predicted_effects"],
    )
    if payload["experience_id"] != expected_experience_id:
        raise SchemaValidationError(
            "candidate experience_id does not match its immutable semantics"
        )
    trajectory_id = str(segment["trajectory_id"])
    segment_id = str(segment["segment_id"])
    evidence = payload["evidence"]
    supporting = evidence["supporting_segment_refs"]
    opposing = evidence["opposing_segment_refs"]
    if segment_id not in supporting:
        raise SchemaValidationError(
            "candidate supporting_segment_refs must include the current segment"
        )
    unresolved_segments = sorted(set((*supporting, *opposing)) - trajectory_segment_ids)
    if unresolved_segments:
        raise SchemaValidationError(
            "candidate segment refs do not resolve in the current trajectory"
        )

    candidate_provenance = payload["provenance"]
    source_provenance = copy.deepcopy(dict(source_record["provenance"]))
    candidate_binding = copy.deepcopy(candidate_provenance)
    source_binding = copy.deepcopy(source_provenance)
    # The Reflector owns its producer identity and derivation timestamp.  All
    # source facts must remain exactly those of the trajectory being processed.
    for provenance in (candidate_binding, source_binding):
        provenance.pop("producer", None)
        provenance.pop("created_at", None)
    if candidate_binding != source_binding:
        raise SchemaValidationError(
            "candidate provenance does not match the current trajectory provenance"
        )
    if candidate_provenance["producer"].get("kind") != "reflector":
        raise SchemaValidationError(
            "candidate provenance producer must identify a reflector"
        )
    for source_ref in candidate_provenance["source_refs"]:
        declared_trajectory = source_ref.get("trajectory_id")
        if declared_trajectory is not None and declared_trajectory != trajectory_id:
            raise SchemaValidationError(
                "candidate provenance contains a cross-trajectory source ref"
            )


def _validate_candidate_evidence(
    refs: Sequence[str],
    *,
    candidate: ProcedureExperienceV1,
    segment: Mapping[str, Any],
    evidence_index: EvidenceIndex,
) -> None:
    """Resolve output evidence and constrain it to the current segment/source."""

    if not refs:
        raise SchemaValidationError(
            "procedure candidate must expose at least one evidence_event_id"
        )
    trajectory_id = str(segment["trajectory_id"])
    resolved = evidence_index.validate_refs(refs, trajectory_id=trajectory_id)
    declared_refs = _segment_evidence_refs(segment)
    if any(ref not in declared_refs for ref in refs):
        raise SchemaValidationError(
            "candidate evidence_event_ids are not declared by the current segment"
        )
    candidate_source_ids = {
        str(source_ref["source_ref_id"])
        for source_ref in candidate.to_dict()["provenance"]["source_refs"]
    }
    if any(entry["source_ref_id"] not in candidate_source_ids for entry in resolved):
        raise SchemaValidationError(
            "candidate evidence_event_ids do not match candidate provenance"
        )


def _hdf5_frame_count(record: Mapping[str, Any], *, trajectory_id: str) -> int:
    derivations = record.get("derivations", [])
    if isinstance(derivations, (str, bytes, bytearray)) or not isinstance(
        derivations, Sequence
    ):
        raise BootstrapError(f"trajectory {trajectory_id!r} has invalid derivations")
    hdf5_metadata = [
        item
        for item in derivations
        if isinstance(item, Mapping) and item.get("kind") == "rmbench_hdf5_metadata"
    ]
    if len(hdf5_metadata) != 1:
        raise BootstrapError(
            f"trajectory {trajectory_id!r} must have exactly one "
            "rmbench_hdf5_metadata derivation"
        )
    frame_count = hdf5_metadata[0].get("frame_count")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
    ):
        raise BootstrapError(
            f"trajectory {trajectory_id!r} has an invalid HDF5 frame_count"
        )
    return frame_count


def _validate_dense_segment_sets(
    *,
    trajectories: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    indexes_by_id: Mapping[str, EvidenceIndex],
) -> None:
    """Prove each imported segment set exactly partitions HDF5 frames 0..T-1."""

    segments_by_trajectory: dict[str, list[Mapping[str, Any]]] = {
        str(record["trajectory"]["trajectory_id"]): [] for record in trajectories
    }
    for segment in segments:
        segments_by_trajectory[str(segment["trajectory_id"])].append(segment)

    for record in trajectories:
        trajectory_id = str(record["trajectory"]["trajectory_id"])
        frame_count = _hdf5_frame_count(record, trajectory_id=trajectory_id)
        trajectory_segments = segments_by_trajectory[trajectory_id]
        if not trajectory_segments:
            raise BootstrapError(
                f"trajectory {trajectory_id!r} has no subtask segments"
            )
        evidence_index = indexes_by_id[trajectory_id]
        previous_end: int | None = None
        for expected_index, segment in enumerate(trajectory_segments):
            segment_id = str(segment["segment_id"])
            if segment["segment_index"] != expected_index:
                raise BootstrapError(
                    f"trajectory {trajectory_id!r} segment_index values must be "
                    f"sequential from zero; {segment_id!r} has "
                    f"{segment['segment_index']!r}, expected {expected_index}"
                )

            derivation = segment["derivation"]
            frame_range = derivation.get("frame_range")
            if not isinstance(frame_range, Mapping):
                raise BootstrapError(
                    f"subtask segment {segment_id!r} lacks a frame_range"
                )
            start = frame_range.get("start_inclusive")
            end = frame_range.get("end_inclusive")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start < 0
                or end < start
            ):
                raise BootstrapError(
                    f"subtask segment {segment_id!r} has an invalid frame_range"
                )
            expected_start = 0 if previous_end is None else previous_end + 1
            if start != expected_start:
                raise BootstrapError(
                    f"trajectory {trajectory_id!r} segment frame ranges must be "
                    f"contiguous from zero; {segment_id!r} starts at {start}, "
                    f"expected {expected_start}"
                )

            expected_before = f"{trajectory_id}#frame/{start:06d}"
            expected_after = f"{trajectory_id}#frame/{end:06d}"
            transition = segment["transition"]
            if transition["before_event_refs"] != [expected_before]:
                raise BootstrapError(
                    f"subtask segment {segment_id!r} before_event_refs must "
                    "identify its exact frame_range start"
                )
            if transition["after_event_refs"] != [expected_after]:
                raise BootstrapError(
                    f"subtask segment {segment_id!r} after_event_refs must "
                    "identify its exact frame_range end"
                )
            try:
                evidence_index.validate_refs(
                    derivation["evidence_event_refs"],
                    trajectory_id=trajectory_id,
                )
            except (SchemaValidationError, TypeError, ValueError) as exc:
                raise BootstrapError(
                    f"subtask segment {segment_id!r} has unresolved "
                    f"derivation evidence: {exc}"
                ) from exc
            previous_end = end

        assert previous_end is not None
        if previous_end != frame_count - 1:
            raise BootstrapError(
                f"trajectory {trajectory_id!r} segment coverage ends at "
                f"{previous_end}, expected HDF5 frame {frame_count - 1}"
            )


def _prepare_imported_records(
    batch: Any,
    expected_manifest: ExpertBootstrapManifestV1,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, EvidenceIndex],
]:
    try:
        batch_manifest = _mapping_copy(
            batch.input_manifest,
            label="import batch input_manifest",
        )
        raw_trajectories = _sequence(
            batch.trajectories,
            label="import batch trajectories",
        )
        raw_segments = _sequence(
            batch.subtask_segments,
            label="import batch subtask_segments",
        )
        raw_evidence = _sequence(
            batch.evidence_entries,
            label="import batch evidence_entries",
        )
        raw_abstentions = _sequence(
            batch.abstentions,
            label="import batch abstentions",
        )
    except AttributeError as exc:
        raise BootstrapError(f"importer returned an incomplete batch: {exc}") from exc

    validated_batch_manifest = ExpertBootstrapManifestV1.from_dict(batch_manifest)
    if validated_batch_manifest.to_dict() != expected_manifest.to_dict():
        raise BootstrapError("importer changed or substituted the input manifest")

    trajectories: list[dict[str, Any]] = []
    records_by_id: dict[str, TrajectoryRecordV1] = {}
    indexes_by_id: dict[str, EvidenceIndex] = {}
    embedded_evidence_by_ref: dict[str, dict[str, Any]] = {}
    embedded_segments_by_id: dict[str, dict[str, Any]] = {}
    embedded_segments_in_order: list[dict[str, Any]] = []
    for number, raw_record in enumerate(raw_trajectories):
        payload = _mapping_copy(raw_record, label=f"trajectory {number}")
        try:
            record = TrajectoryRecordV1.from_dict(payload)
        except SchemaValidationError as exc:
            raise BootstrapError(f"trajectory {number} is invalid: {exc}") from exc
        normalized = record.to_dict()
        trajectory_id = normalized["trajectory"]["trajectory_id"]
        if trajectory_id in records_by_id:
            raise BootstrapError(f"duplicate normalized trajectory {trajectory_id!r}")
        index = EvidenceIndex.from_trajectory_record(record)
        for entry in index.to_entries():
            ref = str(entry["evidence_ref"])
            if ref in embedded_evidence_by_ref:
                raise BootstrapError(f"duplicate evidence reference {ref!r}")
            embedded_evidence_by_ref[ref] = entry
        records_by_id[trajectory_id] = record
        indexes_by_id[trajectory_id] = index
        for embedded_segment in normalized.get("subtask_segments", []):
            segment_id = str(embedded_segment["segment_id"])
            if segment_id in embedded_segments_by_id:
                raise BootstrapError(
                    f"duplicate embedded subtask segment {segment_id!r}"
                )
            embedded_segments_by_id[segment_id] = copy.deepcopy(embedded_segment)
            embedded_segments_in_order.append(copy.deepcopy(embedded_segment))
        trajectories.append(normalized)

    evidence_entries: list[dict[str, Any]] = []
    batch_evidence_by_ref: dict[str, dict[str, Any]] = {}
    for number, raw_entry in enumerate(raw_evidence):
        entry = _mapping_copy(raw_entry, label=f"evidence entry {number}")
        try:
            ref = EvidenceRefV1.parse(entry.get("evidence_ref")).value
        except (SchemaValidationError, TypeError) as exc:
            raise BootstrapError(f"evidence entry {number} is invalid: {exc}") from exc
        if ref in batch_evidence_by_ref:
            raise BootstrapError(f"duplicate batch evidence reference {ref!r}")
        batch_evidence_by_ref[ref] = entry
        evidence_entries.append(entry)
    if set(batch_evidence_by_ref) != set(embedded_evidence_by_ref):
        raise BootstrapError(
            "batch evidence entries do not match normalized trajectory evidence indexes"
        )
    for ref, entry in batch_evidence_by_ref.items():
        if entry != embedded_evidence_by_ref[ref]:
            raise BootstrapError(
                f"batch evidence entry {ref!r} differs from the trajectory record"
            )

    segments: list[dict[str, Any]] = []
    segment_ids: set[str] = set()
    for number, raw_segment in enumerate(raw_segments):
        payload = _mapping_copy(raw_segment, label=f"subtask segment {number}")
        trajectory_id = payload.get("trajectory_id")
        if not isinstance(trajectory_id, str) or trajectory_id not in indexes_by_id:
            raise BootstrapError(
                f"subtask segment {number} refers to an unknown trajectory"
            )
        try:
            record = SubtaskSegmentV1.from_dict(
                payload,
                evidence_index=indexes_by_id[trajectory_id],
            )
        except SchemaValidationError as exc:
            raise BootstrapError(f"subtask segment {number} is invalid: {exc}") from exc
        normalized = record.to_dict()
        segment_id = normalized["segment_id"]
        if segment_id in segment_ids:
            raise BootstrapError(f"duplicate subtask segment {segment_id!r}")
        segment_ids.add(segment_id)
        segments.append(normalized)

    batch_segments_by_id = {str(segment["segment_id"]): segment for segment in segments}
    if set(batch_segments_by_id) != set(embedded_segments_by_id):
        raise BootstrapError(
            "batch subtask segments do not match trajectory embedded subtask_segments"
        )
    for segment_id, segment in batch_segments_by_id.items():
        if segment != embedded_segments_by_id[segment_id]:
            raise BootstrapError(
                f"batch subtask segment {segment_id!r} differs from the trajectory record"
            )
    if segments != embedded_segments_in_order:
        raise BootstrapError(
            "batch subtask segment order differs from trajectory embedded order"
        )

    _validate_dense_segment_sets(
        trajectories=trajectories,
        segments=segments,
        indexes_by_id=indexes_by_id,
    )

    abstentions = [
        _normalize_import_abstention(value, number=number)
        for number, value in enumerate(raw_abstentions)
    ]
    return trajectories, evidence_entries, segments, abstentions, indexes_by_id


def _prepare_import_chunk(
    batch: Any,
    expected_manifest: ExpertBootstrapManifestV1,
) -> _PreparedImportChunk:
    trajectories, evidence_entries, segments, abstentions, _ = (
        _prepare_imported_records(batch, expected_manifest)
    )
    return _PreparedImportChunk(
        trajectories=tuple(trajectories),
        evidence_entries=tuple(evidence_entries),
        segments=tuple(segments),
        abstentions=tuple(abstentions),
    )


def _prepare_streamed_import_entry(
    entry: Any,
) -> _PreparedImportChunk:
    try:
        entry_index = entry.entry_index
        trajectory = entry.trajectory
        trajectory_id = entry.trajectory_id
        raw_segments = tuple(entry.subtask_segments)
        raw_evidence = tuple(entry.evidence_entries)
        raw_abstentions = tuple(entry.abstentions)
    except (AttributeError, TypeError) as exc:
        raise BootstrapError(
            f"built-in importer yielded an invalid streamed entry: {exc}"
        ) from exc
    if (
        isinstance(entry_index, bool)
        or not isinstance(entry_index, int)
        or entry_index < 0
        or not isinstance(trajectory_id, str)
        or not trajectory_id
    ):
        raise BootstrapError("built-in importer yielded an invalid entry identity")
    abstentions = tuple(
        _normalize_import_abstention(value, number=number)
        for number, value in enumerate(raw_abstentions)
    )
    for abstention in abstentions:
        if (
            abstention.get("entry_index") != entry_index
            or abstention.get("trajectory_id") != trajectory_id
        ):
            raise BootstrapError(
                "streamed import abstention is not bound to its enclosing entry"
            )
    if trajectory is None:
        if raw_segments or raw_evidence or not abstentions:
            raise BootstrapError(
                "an abstained streamed entry must contain only explicit abstentions"
            )
        return _PreparedImportChunk(
            trajectories=(),
            evidence_entries=(),
            segments=(),
            abstentions=abstentions,
        )

    try:
        record = TrajectoryRecordV1.from_dict(trajectory)
    except SchemaValidationError as exc:
        raise BootstrapError(f"streamed trajectory is invalid: {exc}") from exc
    normalized = record.to_dict()
    normalized_id = str(normalized["trajectory"]["trajectory_id"])
    if trajectory_id != normalized_id:
        raise BootstrapError(
            "streamed entry trajectory_id differs from its normalized trajectory"
        )
    index = EvidenceIndex.from_trajectory_record(record)
    embedded_evidence = tuple(normalized["evidence_index"])
    if len(raw_evidence) != len(embedded_evidence) or any(
        raw != embedded
        for raw, embedded in zip(raw_evidence, embedded_evidence, strict=True)
    ):
        raise BootstrapError(
            "streamed evidence entries do not match the normalized trajectory"
        )
    embedded_segments = tuple(normalized.get("subtask_segments", []))
    if len(raw_segments) != len(embedded_segments) or any(
        raw != embedded
        for raw, embedded in zip(raw_segments, embedded_segments, strict=True)
    ):
        raise BootstrapError(
            "streamed subtask segments do not match the normalized trajectory"
        )
    for number, segment in enumerate(embedded_segments):
        try:
            SubtaskSegmentV1.from_dict(segment, evidence_index=index)
        except SchemaValidationError as exc:
            raise BootstrapError(
                f"streamed subtask segment {number} is invalid: {exc}"
            ) from exc
    _validate_dense_segment_sets(
        trajectories=[normalized],
        segments=embedded_segments,
        indexes_by_id={normalized_id: index},
    )
    return _PreparedImportChunk(
        trajectories=(normalized,),
        evidence_entries=embedded_evidence,
        segments=embedded_segments,
        abstentions=abstentions,
    )


def _reflect_candidates(
    *,
    trajectories: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    indexes_by_id: Mapping[str, EvidenceIndex],
    reflector: Reflector,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectories_by_id = {
        str(record["trajectory"]["trajectory_id"]): copy.deepcopy(dict(record))
        for record in trajectories
    }
    segment_ids_by_trajectory: dict[str, set[str]] = {
        trajectory_id: set() for trajectory_id in trajectories_by_id
    }
    for segment in segments:
        segment_ids_by_trajectory[str(segment["trajectory_id"])].add(
            str(segment["segment_id"])
        )
    candidates: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    for number, raw_segment in enumerate(segments):
        segment = copy.deepcopy(dict(raw_segment))
        trajectory_id = str(segment["trajectory_id"])
        source_record = trajectories_by_id[trajectory_id]
        record_snapshot = copy.deepcopy(source_record)
        segment_snapshot = copy.deepcopy(segment)
        trajectory_record = TrajectoryRecordV1.from_dict(source_record)
        request_index = EvidenceIndex.from_entries(
            indexes_by_id[trajectory_id].to_entries(),
            source_refs=source_record["provenance"]["source_refs"],
            trajectory_id=trajectory_id,
        )
        request = ReflectorInput(
            trajectory=trajectory_record.to_unified_trajectory(),
            scene_memory=copy.deepcopy(source_record["scene_memory"]),
            experience_context={
                "normalized_segment": copy.deepcopy(segment),
                "evidence_index": request_index,
                "provenance": copy.deepcopy(source_record["provenance"]),
            },
        )
        index_snapshot = _evidence_index_snapshot(request_index)

        def assert_unmodified() -> None:
            if (
                source_record != record_snapshot
                or segment != segment_snapshot
                or _evidence_index_snapshot(request_index) != index_snapshot
            ):
                raise BootstrapError(
                    f"reflector mutated caller-owned input for segment {number}"
                )

        try:
            output = reflector.reflect(request)
        except Exception as exc:
            # A custom Reflector is an untrusted optional component at this
            # boundary.  Its failure is an explicit abstention, not a guessed
            # candidate and not a partial output-directory failure.
            assert_unmodified()
            abstentions.append(
                _reflection_error_abstention(
                    reason="reflector_exception",
                    error=exc,
                    trajectory_id=trajectory_id,
                    segment_id=str(segment["segment_id"]),
                )
            )
            continue
        assert_unmodified()

        try:
            summary, candidate, refs = _normalize_reflector_output(
                output,
                segment_number=number,
            )
        except Exception as exc:
            assert_unmodified()
            abstentions.append(
                _reflection_error_abstention(
                    reason="invalid_reflector_output",
                    error=exc,
                    trajectory_id=trajectory_id,
                    segment_id=str(segment["segment_id"]),
                )
            )
            continue
        assert_unmodified()

        if candidate:
            try:
                validated = validate_reflector_candidate(candidate)
                if not isinstance(validated, ProcedureExperienceV1):
                    raise SchemaValidationError(
                        "successful benchmark expert produced a non-procedure candidate"
                    )
                _validate_candidate_binding(
                    validated,
                    source_record=source_record,
                    segment=segment,
                    trajectory_segment_ids=frozenset(
                        segment_ids_by_trajectory[trajectory_id]
                    ),
                )
            except Exception as exc:
                assert_unmodified()
                abstentions.append(
                    _reflection_error_abstention(
                        reason="invalid_reflector_candidate",
                        error=exc,
                        trajectory_id=trajectory_id,
                        segment_id=str(segment["segment_id"]),
                    )
                )
                continue
            assert_unmodified()
            try:
                _validate_candidate_evidence(
                    refs,
                    candidate=validated,
                    segment=segment,
                    evidence_index=indexes_by_id[trajectory_id],
                )
            except Exception as exc:
                assert_unmodified()
                abstentions.append(
                    _reflection_error_abstention(
                        reason="invalid_reflector_evidence",
                        error=exc,
                        trajectory_id=trajectory_id,
                        segment_id=str(segment["segment_id"]),
                    )
                )
                continue
            assert_unmodified()
            candidates.append(validated.to_dict())
            continue

        abstentions.append(
            _reflection_abstention(
                summary=summary,
                trajectory_id=trajectory_id,
                segment_id=str(segment["segment_id"]),
            )
        )
    return candidates, abstentions


def _reflect_staged_trajectory(
    payload: Mapping[str, Any],
    *,
    reflector: Reflector,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and reflect one normalized trajectory read from staging."""

    try:
        record = TrajectoryRecordV1.from_dict(dict(payload))
    except SchemaValidationError as exc:
        raise BootstrapError(f"staged normalized trajectory is invalid: {exc}") from exc
    normalized = record.to_dict()
    trajectory_id = str(normalized["trajectory"]["trajectory_id"])
    evidence_index = EvidenceIndex.from_trajectory_record(record)
    segments: list[dict[str, Any]] = []
    for number, raw_segment in enumerate(normalized.get("subtask_segments", [])):
        try:
            segment = SubtaskSegmentV1.from_dict(
                raw_segment,
                evidence_index=evidence_index,
            )
        except SchemaValidationError as exc:
            raise BootstrapError(
                f"staged subtask segment {number} is invalid: {exc}"
            ) from exc
        segments.append(segment.to_dict())
    indexes = {trajectory_id: evidence_index}
    _validate_dense_segment_sets(
        trajectories=[normalized],
        segments=segments,
        indexes_by_id=indexes,
    )
    return _reflect_candidates(
        trajectories=[normalized],
        segments=segments,
        indexes_by_id=indexes,
        reflector=reflector,
    )


def _write_bytes(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    created_files: dict[str, tuple[int, int]],
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_fd: int | None = None
    try:
        file_fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode):
            raise BootstrapError(f"staging artifact is not regular: {name}")
        created_files[name] = (info.st_dev, info.st_ino)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(file_fd, view[offset:])
            if written <= 0:
                raise OSError(errno.EIO, "short write without progress")
            offset += written
        os.fsync(file_fd)
    except OSError as exc:
        raise BootstrapError(f"failed to write {name}: {exc}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)


def _write_json(
    directory_fd: int,
    name: str,
    payload: Any,
    *,
    created_files: dict[str, tuple[int, int]],
) -> None:
    _write_bytes(
        directory_fd,
        name,
        _canonical_json_bytes(payload, indent=2),
        created_files=created_files,
    )


def _write_jsonl(
    directory_fd: int,
    name: str,
    records: Sequence[Mapping[str, Any]],
    *,
    created_files: dict[str, tuple[int, int]],
) -> None:
    payload = b"".join(_canonical_json_bytes(dict(record)) for record in records)
    _write_bytes(
        directory_fd,
        name,
        payload,
        created_files=created_files,
    )


class _JsonlSink:
    """Append-only staging writer with incremental hash, size, and count."""

    def __init__(
        self,
        directory_fd: int,
        name: str,
        *,
        schema: str,
        created_files: dict[str, tuple[int, int]],
    ) -> None:
        self.name = name
        self.schema = schema
        self.record_count = 0
        self.size_bytes = 0
        self._digest = hashlib.sha256()
        self._directory_fd = directory_fd
        self._created_files = created_files
        self._fd: int | None = None
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            self._fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
            info = os.fstat(self._fd)
            if not stat.S_ISREG(info.st_mode):
                raise BootstrapError(f"staging artifact is not regular: {name}")
            created_files[name] = (info.st_dev, info.st_ino)
        except OSError as exc:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            raise BootstrapError(f"failed to create {name}: {exc}") from exc

    @property
    def identity(self) -> tuple[int, int]:
        return self._created_files[self.name]

    def write(self, record: Mapping[str, Any]) -> None:
        if self._fd is None:
            raise BootstrapError(f"staging writer is closed: {self.name}")
        payload = _canonical_json_bytes(dict(record))
        view = memoryview(payload)
        offset = 0
        try:
            while offset < len(view):
                written = os.write(self._fd, view[offset:])
                if written <= 0:
                    raise OSError(errno.EIO, "short write without progress")
                offset += written
        except OSError as exc:
            raise BootstrapError(f"failed to append {self.name}: {exc}") from exc
        self._digest.update(payload)
        self.size_bytes += len(payload)
        self.record_count += 1

    def finalize_descriptor(self) -> dict[str, Any]:
        if self._fd is None:
            raise BootstrapError(f"staging writer is closed: {self.name}")
        try:
            os.fsync(self._fd)
            info = os.fstat(self._fd)
        except OSError as exc:
            raise BootstrapError(f"failed to finalize {self.name}: {exc}") from exc
        if (
            not stat.S_ISREG(info.st_mode)
            or not _same_inode(info, self.identity)
            or info.st_size != self.size_bytes
        ):
            raise BootstrapError(f"staging artifact changed while writing: {self.name}")
        descriptor = _file_descriptor(
            self._directory_fd,
            self.name,
            expected_identity=self.identity,
            record_count=self.record_count,
            schema=self.schema,
        )
        if (
            descriptor["sha256"] != self._digest.hexdigest()
            or descriptor["size_bytes"] != self.size_bytes
        ):
            raise BootstrapError(
                f"incremental integrity mismatch for staging artifact: {self.name}"
            )
        return descriptor

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def _iter_staged_jsonl(
    directory_fd: int,
    name: str,
    *,
    expected_identity: tuple[int, int],
) -> Iterator[dict[str, Any]]:
    """Read one record at a time from an owned, stable staging JSONL file."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor: int | None = None
    handle: Any = None
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not _same_inode(
            before, expected_identity
        ):
            raise BootstrapError(f"staging artifact was substituted: {name}")
        handle = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = None
        line_number = 0
        while True:
            raw = handle.readline(_MAX_STAGED_JSONL_RECORD_BYTES + 1)
            if not raw:
                break
            line_number += 1
            if len(raw) > _MAX_STAGED_JSONL_RECORD_BYTES or not raw.endswith(b"\n"):
                raise BootstrapError(
                    f"staged {name} record {line_number} exceeds its safe bound"
                )
            yield _strict_json_object(
                raw,
                label=f"staged {name} record {line_number}",
            )
        after = os.fstat(handle.fileno())
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field) for field in stable_fields
        ):
            raise BootstrapError(f"staging artifact changed while reading: {name}")
    except OSError as exc:
        raise BootstrapError(f"failed to read staged {name}: {exc}") from exc
    finally:
        if handle is not None:
            handle.close()
        elif descriptor is not None:
            os.close(descriptor)


def _file_descriptor(
    directory_fd: int,
    name: str,
    *,
    expected_identity: tuple[int, int],
    record_count: int,
    schema: str,
) -> dict[str, Any]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    file_fd: int | None = None
    try:
        file_fd = os.open(name, flags, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or not _same_inode(
            before,
            expected_identity,
        ):
            raise BootstrapError(
                f"staging artifact was substituted before hashing: {name}"
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(file_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field) for field in stable_fields
        ):
            raise BootstrapError(f"staging artifact changed while hashing: {name}")
    except OSError as exc:
        raise BootstrapError(f"failed to hash {name}: {exc}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)

    return {
        "sha256": digest.hexdigest(),
        "size_bytes": before.st_size,
        "record_count": record_count,
        "schema": schema,
    }


def _bounded_markdown_field(value: Any) -> str:
    """Render one untrusted value as bounded, single-line Markdown text."""

    text = str(value)
    if len(text) > _MAX_REPORT_FIELD_CHARS:
        text = text[:_MAX_REPORT_FIELD_CHARS] + "…"
    rendered: list[str] = []
    for character in text:
        if unicodedata.category(character).startswith("C"):
            rendered.append(f"\\u{ord(character):04x}")
        elif character in _MARKDOWN_SPECIALS:
            rendered.append("\\" + character)
        else:
            rendered.append(character)
    return "".join(rendered)


def _type_identity(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _reflector_configuration_identity(reflector: Reflector) -> dict[str, Any]:
    """Return a stable, secret-free identity for known Reflector settings."""

    identity = _type_identity(reflector)
    if type(reflector) is not CandidateProcedureReflector:
        return {
            "class": identity,
            "configuration_status": "unknown",
            "reason": "custom_reflector_configuration_not_exposed",
        }

    # These fields are owned by CandidateProcedureReflector and validated by
    # its constructor.  Do not serialize arbitrary __dict__ values: a custom
    # backend may carry credentials or other private runtime objects.
    backend = object.__getattribute__(reflector, "_backend")
    producer_name = object.__getattribute__(reflector, "_producer_name")
    producer_version = object.__getattribute__(reflector, "_producer_version")
    created_at = object.__getattribute__(reflector, "_created_at")
    backend_identity: dict[str, Any] = {
        "class": _type_identity(backend),
    }
    if type(backend) is ScriptedProcedureBackend:
        backend_identity.update(
            {
                "configuration_status": "known",
                "settings": {},
            }
        )
    else:
        backend_identity.update(
            {
                "configuration_status": "unknown",
                "reason": "custom_backend_configuration_not_exposed",
            }
        )
    return {
        "class": identity,
        "configuration_status": "known",
        "backend": backend_identity,
        "producer": {
            "name": producer_name,
            "version": producer_version,
            "created_at_override": created_at,
        },
    }


def _component_assurance(
    *,
    builtin_importer: bool,
    builtin_reflector: bool,
) -> dict[str, Any]:
    """Describe only guarantees established by the selected component path.

    Trust is based on whether the caller omitted both injection hooks, never
    on a caller-controlled class name or protocol implementation.
    """

    builtin_components = builtin_importer and builtin_reflector
    return {
        "level": (
            "builtin_components_verified"
            if builtin_components
            else "custom_components_untrusted"
        ),
        "guarantees_limited": not builtin_components,
        "manifest_only_verified": builtin_components,
        "importer_manifest_only_verified": builtin_importer,
        "side_effect_free_verified": builtin_components,
        "raw_artifact_copy_absence_verified": builtin_components,
        "importer_raw_artifact_copy_absence_verified": builtin_importer,
        "custom_importer": not builtin_importer,
        "custom_reflector": not builtin_reflector,
        "candidate_only_published_output_verified": True,
        "no_clobber_publication_verified": True,
    }


def _render_review_report(
    *,
    manifest_sha256: str,
    counts: Mapping[str, int],
    abstentions: Sequence[Mapping[str, Any]] = (),
    abstention_reason_counts: Mapping[tuple[str, str], int] | None = None,
    component_assurance: Mapping[str, Any],
) -> str:
    if abstention_reason_counts is None:
        reasons = Counter(
            (
                _bounded_markdown_field(item.get("scope", "unknown")),
                _bounded_markdown_field(item.get("reason", "unknown")),
            )
            for item in abstentions
        )
    else:
        reasons = Counter(
            {
                (
                    _bounded_markdown_field(scope),
                    _bounded_markdown_field(reason),
                ): count
                for (scope, reason), count in abstention_reason_counts.items()
            }
        )
    sorted_reasons = sorted(reasons.items())
    reason_lines = [
        f"- Scope: {scope}; reason: {reason}; count: {count}"
        for (scope, reason), count in sorted_reasons[:_MAX_REPORT_ABSTENTION_GROUPS]
    ]
    if len(sorted_reasons) > _MAX_REPORT_ABSTENTION_GROUPS:
        reason_lines.append(
            "- Additional distinct reason/scope groups omitted: "
            f"{len(sorted_reasons) - _MAX_REPORT_ABSTENTION_GROUPS}"
        )
    if not reason_lines:
        reason_lines = ["- None"]
    if component_assurance["guarantees_limited"]:
        assurance_lines = [
            "## Component assurance",
            "",
            "This run invoked custom Python components in the current process. "
            "They are recorded as `custom_components_untrusted`; this is a test/"
            "compatibility seam, not a sandbox.",
            "",
            "The orchestrator validated the returned schemas, candidate-only "
            "published output, and no-clobber publication. It did **not** verify "
            "the custom components' file access, source selection, raw-data "
            "handling, network access, process/service creation, or other side "
            "effects.",
            "",
        ]
        raw_artifact_lines = [
            "Because a custom in-process component was used, absence of raw "
            "artifacts inside opaque returned JSON fields or outside this output "
            "is not verified.",
            "",
        ]
        non_claim_lines = [
            "The RoboHarn-Evo orchestrator itself has no rollout, automatic-promotion, or "
            "runtime-retrieval integration. No claim is made about side effects "
            "performed by custom in-process components.",
            "",
        ]
        introduction = (
            "This is a candidate-only output audit with limited custom-component "
            "assurance."
        )
    else:
        assurance_lines = [
            "## Component assurance",
            "",
            "The built-in importer and Reflector were used. Manifest-only source "
            "selection and the documented offline component boundary were verified.",
            "",
        ]
        raw_artifact_lines = [
            "Outputs contain normalized metadata, derivations, and content-addressed "
            "evidence references only; raw HDF5, PKL, RGB, image, video, and "
            "unparsed model-response artifacts were not copied. Manifest-authorized "
            "Qwen memory fields, when present, remain explicitly labeled model "
            "derivations rather than raw facts.",
            "",
        ]
        non_claim_lines = [
            "No rollout, simulator, robot action, GPU job, model service, or "
            "network request was started or integrated by the built-in pipeline. "
            "No claim of rollout improvement, task success, efficiency, "
            "generalization, or model capability is made.",
            "",
        ]
        introduction = "This is a read-only, offline, candidate-only bootstrap audit."

    return "\n".join(
        [
            "# Self-Evolution Phase A review report",
            "",
            introduction,
            "",
            f"- Input manifest SHA-256: `{manifest_sha256}`",
            f"- Authorized manifest entries: {counts['input_entries']}",
            f"- Normalized trajectories: {counts['normalized_trajectories']}",
            f"- Evidence index entries: {counts['evidence_index_entries']}",
            f"- Subtask segments: {counts['subtask_segments']}",
            f"- Procedure candidates: {counts['candidate_experiences']}",
            f"- Explicit abstentions: {counts['abstentions']}",
            "",
            "## Abstention reasons",
            "",
            *reason_lines,
            "",
            *assurance_lines,
            "## Scope and non-claims",
            "",
            *non_claim_lines,
            "Candidates remain `candidate`; they were not accepted, promoted, "
            "written to packaged Experience/Skills, or exposed to runtime "
            "retrieval by the orchestrator.",
            "",
            *raw_artifact_lines,
            "",
        ]
    )


def _publish_outputs(
    *,
    output_root: Path,
    input_manifest: Mapping[str, Any],
    input_manifest_sha256: str,
    import_chunks: Iterable[_PreparedImportChunk],
    reflector: Reflector,
    importer_identity: str,
    component_assurance: Mapping[str, Any],
    processing_state: _ImportProcessingState,
) -> BootstrapRunResult:
    counts = {
        "input_entries": len(input_manifest["entries"]),
        "normalized_trajectories": 0,
        "evidence_index_entries": 0,
        "subtask_segments": 0,
        "candidate_experiences": 0,
        "abstentions": 0,
    }
    abstention_reasons: Counter[tuple[str, str]] = Counter()

    parent_fd: int | None = None
    staging_fd: int | None = None
    staging_name = ""
    staging_identity = (0, 0)
    created_files: dict[str, tuple[int, int]] = {}
    sinks: dict[str, _JsonlSink] = {}
    publication_strategy = ""
    published = False
    try:
        parent_fd = _secure_output_parent_fd(output_root)
        try:
            os.stat(
                output_root.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise OutputBoundaryError(f"output root already exists: {output_root}")
        publication_strategy = _select_publication_strategy(parent_fd)

        staging_name, staging_fd, staging_identity = _create_staging_directory(
            parent_fd=parent_fd,
            output_name=output_root.name,
        )
        _write_json(
            staging_fd,
            "input_manifest.json",
            input_manifest,
            created_files=created_files,
        )
        for name, schema in _JSONL_SCHEMAS.items():
            sinks[name] = _JsonlSink(
                staging_fd,
                name,
                schema=schema,
                created_files=created_files,
            )

        chunk_iterator = iter(import_chunks)
        try:
            for chunk in chunk_iterator:
                for record in chunk.trajectories:
                    sinks["normalized_trajectories.jsonl"].write(record)
                for record in chunk.evidence_entries:
                    sinks["evidence_index.jsonl"].write(record)
                for record in chunk.segments:
                    sinks["subtask_segments.jsonl"].write(record)
                for record in chunk.abstentions:
                    sinks["abstentions.jsonl"].write(record)
                    abstention_reasons[
                        (
                            str(record.get("scope", "unknown")),
                            str(record.get("reason", "unknown")),
                        )
                    ] += 1
        finally:
            close_iterator = getattr(chunk_iterator, "close", None)
            if callable(close_iterator):
                close_iterator()

        # Import is complete before reflection starts.  Reading one normalized
        # trajectory line at a time preserves the historical ordering (all
        # import abstentions precede reflection abstentions) without retaining
        # the complete expert pool in memory.
        normalized_descriptor = sinks[
            "normalized_trajectories.jsonl"
        ].finalize_descriptor()
        for trajectory in _iter_staged_jsonl(
            staging_fd,
            "normalized_trajectories.jsonl",
            expected_identity=sinks["normalized_trajectories.jsonl"].identity,
        ):
            candidates, reflection_abstentions = _reflect_staged_trajectory(
                trajectory,
                reflector=reflector,
            )
            for candidate in candidates:
                sinks["candidate_experiences.jsonl"].write(candidate)
            for record in reflection_abstentions:
                sinks["abstentions.jsonl"].write(record)
                abstention_reasons[
                    (
                        str(record.get("scope", "unknown")),
                        str(record.get("reason", "unknown")),
                    )
                ] += 1

        counts.update(
            {
                "normalized_trajectories": sinks[
                    "normalized_trajectories.jsonl"
                ].record_count,
                "evidence_index_entries": sinks["evidence_index.jsonl"].record_count,
                "subtask_segments": sinks["subtask_segments.jsonl"].record_count,
                "candidate_experiences": sinks[
                    "candidate_experiences.jsonl"
                ].record_count,
                "abstentions": sinks["abstentions.jsonl"].record_count,
            }
        )
        report = _render_review_report(
            manifest_sha256=input_manifest_sha256,
            counts=counts,
            abstention_reason_counts=abstention_reasons,
            component_assurance=component_assurance,
        )
        _write_bytes(
            staging_fd,
            "review_report.md",
            report.encode("utf-8"),
            created_files=created_files,
        )

        jsonl_integrity = {
            name: (
                normalized_descriptor
                if name == "normalized_trajectories.jsonl"
                else sink.finalize_descriptor()
            )
            for name, sink in sinks.items()
        }
        file_integrity: dict[str, Any] = {
            "input_manifest.json": _file_descriptor(
                staging_fd,
                "input_manifest.json",
                expected_identity=created_files["input_manifest.json"],
                record_count=1,
                schema=input_manifest["schema"],
            ),
            **jsonl_integrity,
            "review_report.md": _file_descriptor(
                staging_fd,
                "review_report.md",
                expected_identity=created_files["review_report.md"],
                record_count=1,
                schema="tcm/self_evolution_review_report/phase_a",
            ),
        }
        configuration = {
            "source_selection": (
                "explicit_manifest_only"
                if component_assurance["manifest_only_verified"]
                else "custom_components_unverified"
            ),
            "importer_source_selection": (
                "explicit_manifest_only"
                if component_assurance["importer_manifest_only_verified"]
                else "custom_importer_unverified"
            ),
            "importer": importer_identity,
            "reflector": _type_identity(reflector),
            "reflector_configuration": _reflector_configuration_identity(reflector),
            "component_assurance": copy.deepcopy(dict(component_assurance)),
            "assurance_scope": (
                "tcm_orchestrator_and_builtin_components"
                if not component_assurance["guarantees_limited"]
                else "orchestrator_output_validation_only"
            ),
            "candidate_only": True,
            "automatic_promotion": False,
            "runtime_retrieval": False,
            "rollout_integration": False,
            "raw_artifact_copy": (
                False
                if component_assurance["raw_artifact_copy_absence_verified"]
                else None
            ),
            "input_trust_model": (
                "operator_authorized_expected_read_only"
                if component_assurance["manifest_only_verified"]
                else "custom_components_untrusted"
            ),
            "hdf5_parser_isolation": (
                "in_process_not_sandboxed"
                if component_assurance["importer_manifest_only_verified"]
                else "custom_importer_unverified"
            ),
            "processing_mode": processing_state.processing_mode,
            "resource_limits_enforced": (
                processing_state.processing_mode == "bounded_episode_stream"
            ),
            "resource_limits": (
                copy.deepcopy(dict(processing_state.resource_limits))
                if processing_state.resource_limits is not None
                else None
            ),
            "resource_usage": (
                copy.deepcopy(dict(processing_state.resource_usage))
                if processing_state.resource_usage is not None
                else None
            ),
            "atomic_publication": True,
            "publication_strategy": publication_strategy,
        }
        run_manifest = {
            "schema": "roboharn_evo/expert_bootstrap_run/v1",
            "schema_version": 1,
            "run_id": output_root.name,
            "created_at": _utc_now(),
            "phase": "self_evolution_phase_a",
            "mode": "offline_candidate_only",
            "producer": {
                "kind": "importer",
                "name": "tcm_expert_trajectory_bootstrap",
                "version": "1",
            },
            "configuration": configuration,
            "configuration_sha256": _canonical_json_sha256(configuration),
            "input": {
                "manifest_sha256": input_manifest_sha256,
                "manifest_schema": input_manifest["schema"],
                "manifest_schema_version": input_manifest["schema_version"],
                "dataset_id": input_manifest["dataset_id"],
                "entry_count": len(input_manifest["entries"]),
            },
            "output_schemas": {
                "input_manifest": input_manifest["schema"],
                "normalized_trajectories": "roboharn_evo/trajectory_record/v1",
                "evidence_index": "roboharn_evo/evidence_index_entry/v1",
                "subtask_segments": "roboharn_evo/subtask_segment/v1",
                "candidate_experiences": "roboharn_evo/experience/procedure/v1",
                "abstentions": "roboharn_evo/bootstrap_abstention/v1",
                "run_manifest": "roboharn_evo/expert_bootstrap_run/v1",
            },
            "counts": counts,
            "file_integrity": {
                "algorithm": "sha256",
                "files": file_integrity,
                "run_manifest_self_hash": "excluded_recursive_digest",
            },
            "publication": {
                "strategy": publication_strategy,
                "archive_guidance": {
                    "dereference_output_symlink": True,
                    "archive_hidden_backing_with_link": True,
                    "public_file_count": len(_OUTPUT_FILES),
                },
            },
        }
        _write_json(
            staging_fd,
            "run_manifest.json",
            run_manifest,
            created_files=created_files,
        )

        for sink in sinks.values():
            sink.close()
        actual_files = set(os.listdir(staging_fd))
        if actual_files != set(_OUTPUT_FILES):
            raise BootstrapError(
                "staging output does not contain exactly the eight Phase A artifacts"
            )
        for name in _OUTPUT_FILES:
            info = os.stat(name, dir_fd=staging_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or not _same_inode(
                info,
                created_files[name],
            ):
                raise BootstrapError(
                    "staging output contains an unexpected/substituted "
                    f"artifact: {name}"
                )

        # Persist all staging entries before the selected single-operation,
        # no-clobber namespace publication point.
        os.fsync(staging_fd)
        if publication_strategy == "directory_rename_noreplace":
            _rename_directory_noreplace(
                parent_fd=parent_fd,
                source_name=staging_name,
                destination_name=output_root.name,
            )
            published = True
            try:
                _verify_published_target(
                    parent_fd=parent_fd,
                    destination_name=output_root.name,
                    expected_identity=staging_identity,
                    follow_symlink=False,
                )
            except Exception:
                if _rollback_owned_directory_publication(
                    parent_fd=parent_fd,
                    destination_name=output_root.name,
                    source_name=staging_name,
                    expected_identity=staging_identity,
                ):
                    published = False
                raise
        elif publication_strategy == "same_parent_relative_symlink":
            _symlink_directory_noreplace(
                parent_fd=parent_fd,
                source_name=staging_name,
                destination_name=output_root.name,
            )
            published = True
            link_identity: tuple[int, int] | None = None
            try:
                link_identity = _published_symlink_identity(
                    parent_fd=parent_fd,
                    destination_name=output_root.name,
                    source_name=staging_name,
                )
                _verify_published_target(
                    parent_fd=parent_fd,
                    destination_name=output_root.name,
                    expected_identity=staging_identity,
                    follow_symlink=True,
                )
            except Exception:
                if _unlink_owned_output_symlink(
                    parent_fd=parent_fd,
                    destination_name=output_root.name,
                    source_name=staging_name,
                    link_identity=link_identity,
                ):
                    published = False
                raise
        else:
            raise BootstrapError(
                f"unsupported publication strategy {publication_strategy!r}"
            )
        os.fsync(parent_fd)

        # The directory FD still names the exact published inode, so these
        # hashes cannot be redirected by replacing the lexical output path.
        all_hashes = {
            name: _file_descriptor(
                staging_fd,
                name,
                expected_identity=created_files[name],
                record_count=0,
                schema="hash_only",
            )["sha256"]
            for name in _OUTPUT_FILES
        }
    finally:
        try:
            for sink in sinks.values():
                sink.close()
            if not published and parent_fd is not None and staging_fd is not None:
                _cleanup_staging_directory(
                    parent_fd=parent_fd,
                    staging_fd=staging_fd,
                    staging_name=staging_name,
                    staging_identity=staging_identity,
                    created_files=created_files,
                )
        finally:
            if staging_fd is not None:
                os.close(staging_fd)
            if parent_fd is not None:
                os.close(parent_fd)

    return BootstrapRunResult(
        output_root=output_root,
        input_manifest_sha256=input_manifest_sha256,
        counts=copy.deepcopy(counts),
        file_sha256=all_hashes,
    )


def bootstrap_expert_trajectories(
    *,
    manifest_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    reflector: Reflector | None = None,
    importer: Callable[[Path], Any] | None = None,
    resource_limits: BootstrapResourceLimits | None = None,
) -> BootstrapRunResult:
    """Run and atomically publish one explicit-manifest Phase A bootstrap.

    ``output_root`` is the fresh run directory itself.  It must be an absolute
    child of ``/tmp`` or ``<RoboHarn-Evo>/eval_result/self_evolution/phase_a``.  The
    function never overwrites an existing path.
    """

    dataset_root_fd: int | None = None
    import_stream: Any | None = None
    try:
        component_assurance = _component_assurance(
            builtin_importer=importer is None,
            builtin_reflector=reflector is None,
        )
        manifest, manifest_bytes, manifest_record = _load_input_manifest(manifest_path)
        resolved_output = _resolve_output_root(output_root)
        dataset_root, dataset_root_fd = _bind_dataset_root(manifest_record)
        _validate_output_isolation(resolved_output, dataset_root)
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        if resource_limits is not None and not isinstance(
            resource_limits, BootstrapResourceLimits
        ):
            raise TypeError("resource_limits must be BootstrapResourceLimits")
        if importer is not None and resource_limits is not None:
            raise BootstrapError(
                "resource_limits cannot be guaranteed for a custom importer"
            )
        selected_limits = resource_limits or BootstrapResourceLimits()

        selected_importer = importer
        importer_identity = (
            "roboharn_evo.benchmark_adapters.rmbench.expert_trajectory."
            "RMBenchExpertTrajectoryImporter.stream"
            if importer is None
            else (
                f"{getattr(selected_importer, '__module__', '<unknown>')}."
                f"{getattr(selected_importer, '__qualname__', type(selected_importer).__qualname__)}"
            )
        )
        selected_reflector = (
            reflector if reflector is not None else CandidateProcedureReflector()
        )
        if selected_importer is None:
            processing_state = _ImportProcessingState(
                processing_mode="bounded_episode_stream",
                resource_limits=selected_limits.to_dict(),
            )
            try:
                importer_instance = _default_importer(
                    manifest_record,
                    dataset_root,
                    dataset_root_fd,
                    selected_limits,
                )
                import_stream = importer_instance.stream()
                stream_manifest = ExpertBootstrapManifestV1.from_dict(
                    import_stream.input_manifest
                )
                if stream_manifest.to_dict() != manifest_record.to_dict():
                    raise BootstrapError(
                        "built-in importer changed or substituted the input manifest"
                    )
                if dict(import_stream.limits) != selected_limits.to_dict():
                    raise BootstrapError(
                        "built-in importer did not bind the requested resource limits"
                    )
            except Exception as exc:
                raise BootstrapError(
                    "manifest-only expert import preflight failed "
                    f"({type(exc).__name__}): {exc}"
                ) from exc

            def streamed_chunks() -> Iterator[_PreparedImportChunk]:
                processed_entries = 0
                try:
                    with import_stream:
                        for expected_index, imported_entry in enumerate(import_stream):
                            actual_index = getattr(imported_entry, "entry_index", None)
                            if actual_index != expected_index:
                                raise BootstrapError(
                                    "built-in importer stream entry order mismatch: "
                                    f"expected {expected_index}, got {actual_index!r}"
                                )
                            yield _prepare_streamed_import_entry(
                                imported_entry,
                            )
                            processed_entries += 1
                        expected_count = len(manifest_record.to_dict()["entries"])
                        if processed_entries != expected_count:
                            raise BootstrapError(
                                "built-in importer stream ended before every manifest "
                                f"entry was processed: expected {expected_count}, "
                                f"got {processed_entries}"
                            )
                except BootstrapError:
                    raise
                except Exception as exc:
                    raise BootstrapError(
                        "manifest-only expert import failed "
                        f"({type(exc).__name__}): {exc}"
                    ) from exc
                finally:
                    processing_state.resource_usage = import_stream.usage.to_dict()

            import_chunks: Iterable[_PreparedImportChunk] = streamed_chunks()
        else:
            try:
                batch = selected_importer(manifest)
            except Exception as exc:
                raise BootstrapError(
                    f"custom expert importer failed ({type(exc).__name__}): {exc}"
                ) from exc
            import_chunks = (_prepare_import_chunk(batch, manifest_record),)
            processing_state = _ImportProcessingState(
                processing_mode="custom_materialized_batch",
                resource_limits=None,
                resource_usage=None,
            )
        return _publish_outputs(
            output_root=resolved_output,
            input_manifest=manifest_record.to_dict(),
            input_manifest_sha256=manifest_sha256,
            import_chunks=import_chunks,
            reflector=selected_reflector,
            importer_identity=importer_identity,
            component_assurance=component_assurance,
            processing_state=processing_state,
        )
    except (BootstrapError, OutputBoundaryError):
        raise
    except (OSError, SchemaValidationError, TypeError, ValueError) as exc:
        raise BootstrapError(f"Phase A bootstrap failed: {exc}") from exc
    finally:
        try:
            if import_stream is not None:
                import_stream.close()
        finally:
            if dataset_root_fd is not None:
                os.close(dataset_root_fd)


# Concise alias for programmatic callers.
run_bootstrap = bootstrap_expert_trajectories


__all__ = [
    "BootstrapError",
    "BootstrapResourceLimits",
    "BootstrapRunResult",
    "OutputBoundaryError",
    "bootstrap_expert_trajectories",
    "run_bootstrap",
]
