from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

from roboharn_evo.agent.hpk.evolving_store import (
    ADVANCE_DIRNAME,
    CHILD_CLAIM_DIRNAME,
    MEMBER_SPECS,
    PUBLICATION_PENDING_FILENAME,
    ManifestV2,
    LoadedEvolvingSnapshot,
    SNAPSHOT_V2_RUNTIME_STATUS,
    SNAPSHOT_V2_SCHEMA,
    canonical_entries_jsonl,
    canonical_jsonl_records,
    child_claim_payload,
    entry_lifecycle_sets,
    information_access_flags,
    load_evolving_snapshot,
    parent_advance_filename,
    publication_pending_payload,
    promotion_policy_from_ref,
    snapshot_advance_payload,
    snapshot_v2_id_for,
    validate_evolving_snapshot_lineage,
)
from roboharn_evo.agent.hpk.policy_config import LoadedHPKPolicy
from roboharn_evo.agent.hpk.schemas import EntryV1, canonical_json_bytes


class HPKSnapshotPublicationError(RuntimeError):
    """Publication failed; ``usable_parent`` remains the authoritative state."""

    def __init__(
        self, message: str, *, usable_parent: LoadedEvolvingSnapshot | None
    ) -> None:
        super().__init__(message)
        self.usable_parent = usable_parent


@dataclass(frozen=True, slots=True)
class PublishedChild:
    manifest_path: Path
    manifest_sha256: str
    snapshot_id: str
    snapshot: LoadedEvolvingSnapshot


COMMITTED_OBJECTS_DIRNAME = ".afk-committed-objects"


def _fail(message: str, *, parent: LoadedEvolvingSnapshot | None) -> None:
    raise HPKSnapshotPublicationError(message, usable_parent=parent)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{path} must be an object", parent=None)
    try:
        return json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise HPKSnapshotPublicationError(
            f"{path} is not canonical JSON: {exc}", usable_parent=None
        ) from exc


def _policy_identity(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, LoadedHPKPolicy):
        payload: dict[str, Any] = value.identity()
        if name == "promotion":
            payload["payload"] = value.payload
    elif (
        name == "promotion"
        and hasattr(value, "policy_id")
        and hasattr(value, "config_sha256")
        and callable(getattr(value, "to_dict", None))
    ):
        payload = {
            "policy_id": str(value.policy_id),
            "config_sha256": str(value.config_sha256),
            "payload": value.to_dict(),
        }
    elif hasattr(value, "policy_id") and hasattr(value, "config_sha256"):
        payload = {
            "policy_id": str(value.policy_id),
            "config_sha256": str(value.config_sha256),
        }
    else:
        payload = _canonical_mapping(value, path=f"policy_refs.{name}")
    expected_fields = (
        {"policy_id", "config_sha256", "payload"}
        if name == "promotion"
        else {"policy_id", "config_sha256"}
    )
    if set(payload) != expected_fields:
        _fail(f"policy_refs.{name} has unexpected fields", parent=None)
    policy_id = payload.get("policy_id")
    digest = payload.get("config_sha256")
    if not isinstance(policy_id, str) or not policy_id.strip():
        _fail(f"policy_refs.{name}.policy_id is invalid", parent=None)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        _fail(f"policy_refs.{name}.config_sha256 is invalid", parent=None)
    if name == "promotion":
        policy = promotion_policy_from_ref(payload)
        return {
            "policy_id": policy.policy_id,
            "config_sha256": policy.config_sha256,
            "payload": policy.to_dict(),
        }
    return {"policy_id": policy_id.strip(), "config_sha256": digest}


def _normalize_policy_refs(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if set(value) != {"updater", "promotion", "geometry"}:
        _fail("policy_refs must contain updater, promotion, and geometry", parent=None)
    return {
        name: _policy_identity(value[name], name=name)
        for name in ("updater", "promotion", "geometry")
    }


def _normalize_batch(value: Mapping[str, Any] | Any | None) -> dict[str, str | None]:
    if value is None:
        return {"batch_id": None, "sha256": None}
    if isinstance(value, Mapping):
        payload = _canonical_mapping(value, path="evidence_batch")
    elif hasattr(value, "batch_id") and hasattr(value, "evidence_set_sha256"):
        payload = {
            "batch_id": str(value.batch_id),
            "sha256": str(value.evidence_set_sha256),
        }
    else:
        _fail("evidence_batch must be a mapping or EvidenceBatch identity", parent=None)
    if set(payload) != {"batch_id", "sha256"}:
        _fail("evidence_batch must contain batch_id and sha256", parent=None)
    batch_id = payload["batch_id"]
    digest = payload["sha256"]
    if (
        not isinstance(batch_id, str)
        or not batch_id.startswith("afkbatch_")
        or len(batch_id) != len("afkbatch_") + 64
    ):
        _fail("evidence_batch.batch_id is invalid", parent=None)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        _fail("evidence_batch.sha256 is invalid", parent=None)
    return {"batch_id": batch_id, "sha256": digest}


def _write_fsynced(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Linux no-clobber atomic directory rename; fail rather than weaken it."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _control_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(value)) + b"\n"


def _read_exact_regular(path: Path, *, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HPKSnapshotPublicationError(
            f"{label} is unavailable: {exc}", usable_parent=None
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        _fail(f"{label} must be a non-symlink regular file", parent=None)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _fail(f"{label} changed while being read", parent=None)
        value = b"".join(chunks)
        if len(value) != after.st_size:
            _fail(f"{label} changed size while being read", parent=None)
        return value
    finally:
        os.close(descriptor)


def _write_temp_control_file(directory: Path, data: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=".hpk-control-", dir=directory)
    path = Path(raw_path)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def _atomic_hardlink_claim(
    *,
    directory: Path,
    filename: str,
    data: bytes,
    label: str,
    parent: LoadedEvolvingSnapshot | None,
) -> bool:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.is_symlink():
        _fail(f"{label} directory must not be a symlink", parent=parent)
    os.chmod(directory, 0o700)
    temporary = _write_temp_control_file(directory, data)
    destination = directory / filename
    try:
        try:
            os.link(temporary, destination, follow_symlinks=False)
            created = True
        except FileExistsError:
            created = False
            existing = _read_exact_regular(destination, label=label)
            if existing != data:
                _fail(f"{label} is already claimed by different content", parent=parent)
        except OSError as exc:
            raise HPKSnapshotPublicationError(
                f"{label} atomic hardlink failed: {exc}", usable_parent=parent
            ) from exc
        _fsync_directory(directory)
        return created
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(directory)


def _control_file_equals(path: Path, expected: bytes) -> bool:
    try:
        return _read_exact_regular(path, label="snapshot publication claim") == expected
    except HPKSnapshotPublicationError:
        return False


def _materialized_child_matches(
    destination: Path,
    *,
    manifest_bytes: bytes,
    member_bytes: Mapping[str, bytes],
    pending_bytes: bytes,
) -> bool:
    try:
        metadata = destination.lstat()
    except OSError:
        return False
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        return False
    expected = {
        "snapshot_manifest.json": manifest_bytes,
        PUBLICATION_PENDING_FILENAME: pending_bytes,
        **{MEMBER_SPECS[name][0]: value for name, value in member_bytes.items()},
    }
    actual_names = {path.name for path in destination.iterdir()}
    if actual_names != set(expected):
        return False
    try:
        return all(
            _read_exact_regular(destination / name, label=f"materialized {name}")
            == value
            for name, value in expected.items()
        )
    except HPKSnapshotPublicationError:
        return False


def _materialize_child(
    staging: Path,
    destination: Path,
    *,
    manifest_bytes: bytes,
    member_bytes: Mapping[str, bytes],
    pending_bytes: bytes,
    parent: LoadedEvolvingSnapshot | None,
) -> None:
    try:
        _rename_noreplace(staging, destination)
        return
    except OSError as exc:
        unsupported = exc.errno in {
            errno.EINVAL,
            errno.ENOSYS,
            getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
            errno.ENOTSUP,
        }
        if not unsupported:
            if destination.exists() and _materialized_child_matches(
                destination,
                manifest_bytes=manifest_bytes,
                member_bytes=member_bytes,
                pending_bytes=pending_bytes,
            ):
                return
            raise
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError:
        metadata = destination.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            _fail("claimed child destination is not a real directory", parent=parent)
    expected_order = [
        (PUBLICATION_PENDING_FILENAME, pending_bytes),
        ("snapshot_manifest.json", manifest_bytes),
        *[(MEMBER_SPECS[name][0], member_bytes[name]) for name in sorted(member_bytes)],
    ]
    for filename, expected_bytes in expected_order:
        source = staging / filename
        target = destination / filename
        try:
            os.link(source, target, follow_symlinks=False)
        except FileExistsError:
            if _read_exact_regular(target, label=f"materialized {filename}") != (
                expected_bytes
            ):
                _fail(f"materialized child file collision: {filename}", parent=parent)
        except OSError as exc:
            raise HPKSnapshotPublicationError(
                f"materialized child hardlink failed: {exc}", usable_parent=parent
            ) from exc
    _fsync_directory(destination)
    if not _materialized_child_matches(
        destination,
        manifest_bytes=manifest_bytes,
        member_bytes=member_bytes,
        pending_bytes=pending_bytes,
    ):
        _fail("materialized child bytes changed during fallback rename", parent=parent)
    shutil.rmtree(staging)
    _fsync_directory(destination.parent)


def _directory_inventory(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            _fail("committed directory trees must not contain symlinks", parent=None)
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            result[relative] = {"kind": "directory"}
        elif path.is_file():
            data = _read_exact_regular(path, label=f"artifact member {relative}")
            result[relative] = {
                "kind": "file",
                "sha256": _sha256(data),
                "size_bytes": len(data),
            }
        else:
            _fail("committed directory trees contain an unsupported node", parent=None)
    return result


def _materialize_directory_tree_hardlinks(
    source: Path,
    destination: Path,
    *,
    inventory: Mapping[str, Mapping[str, Any]],
) -> None:
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError:
        metadata = destination.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            _fail("content-addressed object collision", parent=None)
    directory_names = sorted(
        (
            name
            for name, descriptor in inventory.items()
            if descriptor["kind"] == "directory"
        ),
        key=lambda value: (len(Path(value).parts), value),
    )
    for name in directory_names:
        target_directory = destination / name
        try:
            target_directory.mkdir(mode=0o700)
        except FileExistsError:
            metadata = target_directory.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                _fail("content-addressed object directory collision", parent=None)
    for name, descriptor in sorted(inventory.items()):
        if descriptor["kind"] != "file":
            continue
        source_file = source / name
        target_file = destination / name
        try:
            os.link(source_file, target_file, follow_symlinks=False)
        except FileExistsError:
            data = _read_exact_regular(
                target_file, label=f"content-addressed object {name}"
            )
            if (
                _sha256(data) != descriptor["sha256"]
                or len(data) != descriptor["size_bytes"]
            ):
                _fail("content-addressed object file collision", parent=None)
        except OSError as exc:
            raise HPKSnapshotPublicationError(
                f"content-addressed object hardlink failed: {exc}",
                usable_parent=None,
            ) from exc
    directories = [destination / name for name in reversed(directory_names)]
    directories.append(destination)
    for directory in directories:
        _fsync_directory(directory)
    if _directory_inventory(destination) != dict(inventory):
        _fail("content-addressed object changed during materialization", parent=None)
    shutil.rmtree(source)
    _fsync_directory(destination.parent)


def publish_directory_noreplace(staging: Path, destination: Path) -> str:
    """Atomically publish a fully-fsynced auxiliary directory without clobber.

    ``renameat2(RENAME_NOREPLACE)`` remains the fast path.  Filesystems that
    reject it use a content-addressed hidden object plus one atomic symlink
    creation as the visible commit pointer.  A crash before the pointer leaves
    only a hidden orphan; ``destination`` is never partially visible.
    """

    source = Path(staging).absolute()
    target = Path(destination).absolute()
    if source.parent.resolve(strict=True) != target.parent.resolve(strict=True):
        _fail("staging and destination must share one explicit parent", parent=None)
    if target.exists() or target.is_symlink():
        _fail(f"refusing to clobber existing directory: {target}", parent=None)
    inventory = _directory_inventory(source)
    tree_sha = _sha256(canonical_json_bytes(inventory))
    _fsync_directory(source)
    _fsync_directory(source.parent)
    try:
        _rename_noreplace(source, target)
        _fsync_directory(target.parent)
        return tree_sha
    except OSError as exc:
        if exc.errno not in {
            errno.EINVAL,
            errno.ENOSYS,
            getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
            errno.ENOTSUP,
        }:
            raise HPKSnapshotPublicationError(
                f"atomic directory publication failed: {exc}", usable_parent=None
            ) from exc

    objects = target.parent / COMMITTED_OBJECTS_DIRNAME
    objects.mkdir(parents=True, exist_ok=True, mode=0o700)
    if objects.is_symlink():
        _fail("committed object directory must not be a symlink", parent=None)
    os.chmod(objects, 0o700)
    object_name = f"{target.name}.{tree_sha}"
    materialized = objects / object_name
    _materialize_directory_tree_hardlinks(
        source,
        materialized,
        inventory=inventory,
    )
    if _directory_inventory(materialized) != inventory:
        _fail("content-addressed artifact object changed", parent=None)
    _fsync_directory(materialized)
    _fsync_directory(objects)
    relative_target = os.path.relpath(materialized, target.parent)
    try:
        os.symlink(relative_target, target, target_is_directory=True)
    except FileExistsError as exc:
        raise HPKSnapshotPublicationError(
            f"refusing to clobber existing directory: {target}", usable_parent=None
        ) from exc
    except OSError as exc:
        raise HPKSnapshotPublicationError(
            f"atomic committed-object pointer failed: {exc}", usable_parent=None
        ) from exc
    _fsync_directory(target.parent)
    return tree_sha


def _cas_parent(
    parent: LoadedEvolvingSnapshot,
    *,
    expected_parent_snapshot_id: str,
    expected_parent_manifest_sha256: str,
) -> LoadedEvolvingSnapshot:
    if parent.snapshot_id != expected_parent_snapshot_id:
        _fail("stale parent snapshot ID", parent=parent)
    if parent.manifest_sha256 != expected_parent_manifest_sha256:
        _fail("stale parent manifest SHA-256", parent=parent)
    try:
        current = load_evolving_snapshot(
            parent.manifest_path,
            expected_parent_manifest_sha256,
        )
    except Exception as exc:
        raise HPKSnapshotPublicationError(
            f"parent CAS reload failed: {exc}", usable_parent=parent
        ) from exc
    if current.snapshot_id != expected_parent_snapshot_id:
        _fail("parent changed during CAS reload", parent=parent)
    return current


def _after_advance_commit() -> None:
    """Fault-injection seam after the durable parent advance is visible."""


def publish_child_snapshot(
    *,
    parent: LoadedEvolvingSnapshot | None,
    expected_parent_snapshot_id: str | None,
    expected_parent_manifest_sha256: str | None,
    entries: Sequence[EntryV1 | Mapping[str, Any]],
    evidence_records: Sequence[Mapping[str, Any] | Any] = (),
    private_provenance_records: Sequence[Mapping[str, Any] | Any] = (),
    update_decision_records: Sequence[Mapping[str, Any] | Any] = (),
    promotion_decision_records: Sequence[Mapping[str, Any] | Any] = (),
    rejected_evidence_records: Sequence[Mapping[str, Any] | Any] = (),
    revalidation_entry_ids: Sequence[str] = (),
    policy_refs: Mapping[str, Any],
    evidence_batch: Any | None,
    source_episode_ids: Sequence[str],
    created_at: str,
    destination_root: str | os.PathLike[str],
    runtime_source_identity: Mapping[str, Any],
    purpose: str = "evolving_update",
) -> PublishedChild:
    """Publish one deterministic root/child snapshot without modifying parent."""

    if parent is None:
        if (
            expected_parent_snapshot_id is not None
            or expected_parent_manifest_sha256 is not None
        ):
            _fail("root snapshot cannot declare an expected parent", parent=None)
        parent_ref = {"snapshot_id": None, "manifest_sha256": None}
    else:
        if not isinstance(expected_parent_snapshot_id, str) or not isinstance(
            expected_parent_manifest_sha256, str
        ):
            _fail(
                "child publication requires explicit expected parent ID/hash",
                parent=parent,
            )
        parent = _cas_parent(
            parent,
            expected_parent_snapshot_id=expected_parent_snapshot_id,
            expected_parent_manifest_sha256=expected_parent_manifest_sha256,
        )
        parent_ref = {
            "snapshot_id": parent.snapshot_id,
            "manifest_sha256": parent.manifest_sha256,
        }

    output_root = Path(destination_root)
    if not output_root.is_absolute():
        _fail("destination_root must be explicit and absolute", parent=parent)
    if output_root.exists() and output_root.is_symlink():
        _fail("destination_root must not be a symlink", parent=parent)
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_root = output_root.resolve(strict=True)
    os.chmod(output_root, 0o700)

    try:
        entry_bytes, typed_entries = canonical_entries_jsonl(entries)
        generic_inputs = {
            "evidence": evidence_records,
            "private_provenance": private_provenance_records,
            "update_decisions": update_decision_records,
            "promotion_decisions": promotion_decision_records,
            "rejected_evidence": rejected_evidence_records,
        }
        member_bytes: dict[str, bytes] = {"entries": entry_bytes}
        member_records: dict[str, tuple[dict[str, Any], ...]] = {}
        for name, values in generic_inputs.items():
            encoded, normalized = canonical_jsonl_records(name, values)
            member_bytes[name] = encoded
            member_records[name] = normalized
        lifecycle = entry_lifecycle_sets(typed_entries, revalidation_entry_ids)
        policies = _normalize_policy_refs(policy_refs)
        batch = _normalize_batch(evidence_batch)
        if member_records["evidence"] and batch["batch_id"] is None:
            _fail(
                "non-empty evidence requires an explicit evidence batch", parent=parent
            )
        episode_ids = sorted(set(str(value) for value in source_episode_ids))
        if len(episode_ids) != len(tuple(source_episode_ids)) or any(
            not value.startswith("afkepisode_") or len(value) != len("afkepisode_") + 64
            for value in episode_ids
        ):
            _fail(
                "source_episode_ids must be sorted-unique public episode IDs",
                parent=parent,
            )
        identities = _canonical_mapping(
            runtime_source_identity,
            path="runtime_source_identity",
        )
        if set(identities) != {"runtime_identity", "source_identity"}:
            _fail("runtime_source_identity fields are not exact", parent=parent)
        descriptors = {
            name: {
                "path": filename,
                "media_type": media_type,
                "sha256": _sha256(member_bytes[name]),
                "size_bytes": len(member_bytes[name]),
                "record_count": (
                    len(typed_entries)
                    if name == "entries"
                    else len(member_records[name])
                ),
            }
            for name, (filename, media_type) in MEMBER_SPECS.items()
        }
        counts = {
            "entry_count": len(typed_entries),
            "accepted_count": len(lifecycle["accepted"]),
            "candidate_count": len(lifecycle["candidate"]),
            "revalidation_count": len(lifecycle["revalidation"]),
            "deprecated_count": len(lifecycle["deprecated"]),
            "evidence_count": len(member_records["evidence"]),
            "private_provenance_count": len(member_records["private_provenance"]),
            "update_decision_count": len(member_records["update_decisions"]),
            "promotion_decision_count": len(member_records["promotion_decisions"]),
            "rejected_evidence_count": len(member_records["rejected_evidence"]),
        }
        manifest_payload: dict[str, Any] = {
            "schema": SNAPSHOT_V2_SCHEMA,
            "schema_version": 2,
            "snapshot_id": "",
            "parent": parent_ref,
            "created_at": created_at,
            "purpose": purpose,
            "members": descriptors,
            "entry_ids": lifecycle,
            "policy_refs": policies,
            "evidence_batch": batch,
            "source_episode_ids": episode_ids,
            "information_access_flags": information_access_flags(
                typed_entries,
                member_records["evidence"],
                promotion_policy_ref=policies["promotion"],
            ),
            "counts": counts,
            "runtime_source_identity": identities,
            "runtime_status": SNAPSHOT_V2_RUNTIME_STATUS,
            "immutable": True,
        }
        manifest_payload["snapshot_id"] = snapshot_v2_id_for(manifest_payload)
        manifest = ManifestV2.from_dict(manifest_payload)
        if parent is not None:
            validate_evolving_snapshot_lineage(
                parent=parent,
                manifest=manifest,
                entries=typed_entries,
                evidence_records=member_records["evidence"],
                private_provenance_records=member_records["private_provenance"],
                update_decision_records=member_records["update_decisions"],
                promotion_decision_records=member_records["promotion_decisions"],
                rejected_evidence_records=member_records["rejected_evidence"],
            )
        manifest_bytes = canonical_json_bytes(manifest.to_dict()) + b"\n"
        manifest_sha = _sha256(manifest_bytes)
        destination = output_root / manifest.snapshot_id
        claim = child_claim_payload(
            snapshot_id=manifest.snapshot_id,
            manifest_sha256=manifest_sha,
        )
        claim_bytes = _control_bytes(claim)
        claim_path = output_root / CHILD_CLAIM_DIRNAME / f"{manifest.snapshot_id}.json"
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                _fail(
                    f"refusing to clobber existing child: {destination}",
                    parent=parent,
                )
            try:
                existing = load_evolving_snapshot(
                    destination / "snapshot_manifest.json", manifest_sha
                )
            except Exception:
                # A prior crash may leave a byte-complete child without its
                # parent advance.  Exact bytes are checked before reuse below.
                if not _control_file_equals(claim_path, claim_bytes):
                    _fail(
                        f"refusing to reuse unclaimed child: {destination}",
                        parent=parent,
                    )
            else:
                return PublishedChild(
                    manifest_path=existing.manifest_path,
                    manifest_sha256=manifest_sha,
                    snapshot_id=existing.snapshot_id,
                    snapshot=existing,
                )

        staging = Path(tempfile.mkdtemp(prefix=".hpk-snapshot-v2-", dir=output_root))
        os.chmod(staging, 0o700)
        try:
            for name, (filename, _media_type) in MEMBER_SPECS.items():
                _write_fsynced(staging / filename, member_bytes[name])
            _write_fsynced(staging / "snapshot_manifest.json", manifest_bytes)
            _fsync_directory(staging)
            validated = load_evolving_snapshot(
                (staging / "snapshot_manifest.json").absolute(), manifest_sha
            )
            if validated.snapshot_id != manifest.snapshot_id:
                _fail(
                    "temporary child validation changed snapshot identity",
                    parent=parent,
                )
            pending = publication_pending_payload(
                snapshot_id=manifest.snapshot_id,
                manifest_sha256=manifest_sha,
                parent=parent_ref,
            )
            pending_bytes = _control_bytes(pending)
            _write_fsynced(staging / PUBLICATION_PENDING_FILENAME, pending_bytes)
            _fsync_directory(staging)

            _atomic_hardlink_claim(
                directory=output_root / CHILD_CLAIM_DIRNAME,
                filename=f"{manifest.snapshot_id}.json",
                data=claim_bytes,
                label="snapshot child claim",
                parent=parent,
            )
            if parent is not None:
                _cas_parent(
                    parent,
                    expected_parent_snapshot_id=parent.snapshot_id,
                    expected_parent_manifest_sha256=parent.manifest_sha256,
                )
            _fsync_directory(output_root)
            _materialize_child(
                staging,
                destination,
                manifest_bytes=manifest_bytes,
                member_bytes=member_bytes,
                pending_bytes=pending_bytes,
                parent=parent,
            )
            _fsync_directory(output_root)
            if parent is not None:
                _cas_parent(
                    parent,
                    expected_parent_snapshot_id=parent.snapshot_id,
                    expected_parent_manifest_sha256=parent.manifest_sha256,
                )
            advance = snapshot_advance_payload(
                parent=parent_ref,
                child_snapshot_id=manifest.snapshot_id,
                child_manifest_sha256=manifest_sha,
            )
            _atomic_hardlink_claim(
                directory=output_root / ADVANCE_DIRNAME,
                filename=parent_advance_filename(parent_ref),
                data=_control_bytes(advance),
                label="snapshot parent advance",
                parent=parent,
            )
            _fsync_directory(output_root)
            _after_advance_commit()
            published = load_evolving_snapshot(
                destination / "snapshot_manifest.json", manifest_sha
            )
            if parent is not None:
                _cas_parent(
                    parent,
                    expected_parent_snapshot_id=parent.snapshot_id,
                    expected_parent_manifest_sha256=parent.manifest_sha256,
                )
            return PublishedChild(
                manifest_path=published.manifest_path,
                manifest_sha256=manifest_sha,
                snapshot_id=published.snapshot_id,
                snapshot=published,
            )
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise
    except HPKSnapshotPublicationError:
        raise
    except BaseException as exc:
        raise HPKSnapshotPublicationError(
            f"child snapshot publication failed: {type(exc).__name__}: {exc}",
            usable_parent=parent,
        ) from exc


__all__ = [
    "HPKSnapshotPublicationError",
    "COMMITTED_OBJECTS_DIRNAME",
    "PublishedChild",
    "publish_child_snapshot",
    "publish_directory_noreplace",
]
