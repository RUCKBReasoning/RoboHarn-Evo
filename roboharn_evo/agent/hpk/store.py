from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.compatibility import normalize_knowledge_metadata
from roboharn_evo.agent.hpk.schemas import (
    HPKValidationError,
    EntryV1,
    SnapshotManifestV1,
    canonical_json_bytes,
)


MAX_MANIFEST_BYTES = 1 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_ENTRY_BYTES = 256 * 1024
MAX_ENTRY_COUNT = 4096

RUN_SCOPES = frozenset(
    {"formal_no_prior", "integration", "expert_prior", "oracle_diagnostic"}
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class HPKSnapshotError(HPKValidationError):
    """HPK 快照验证失败。"""


@dataclass(frozen=True, slots=True)
class LoadedHPKSnapshot:
    """Immutable result of one successful explicit snapshot load."""

    manifest_path: Path
    manifest_sha256: str
    member_path: Path
    member_sha256: str
    manifest: SnapshotManifestV1
    entries: tuple[EntryV1, ...]

    @property
    def snapshot_id(self) -> str:
        return str(self.manifest["snapshot_id"])

    @property
    def accepted_entry_ids(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.manifest["accepted_entry_ids"])

    def entry_by_id(self, entry_id: str) -> EntryV1 | None:
        requested = str(entry_id or "").strip()
        return next(
            (entry for entry in self.entries if entry["entry_id"] == requested),
            None,
        )


def _error(message: str) -> None:
    raise HPKSnapshotError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _error(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        _error(f"{label} must be a boolean")
    return value


def _resolve_regular_path(path: Path, *, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HPKSnapshotError(f"{label} does not exist: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        _error(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        _error(f"{label} must be a regular file: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise HPKSnapshotError(f"{label} cannot be resolved: {path}: {exc}") from exc
    try:
        resolved_metadata = resolved.lstat()
    except OSError as exc:
        raise HPKSnapshotError(
            f"resolved {label} does not exist: {resolved}: {exc}"
        ) from exc
    if stat.S_ISLNK(resolved_metadata.st_mode) or not stat.S_ISREG(
        resolved_metadata.st_mode
    ):
        _error(f"resolved {label} must be a non-symlink regular file: {resolved}")
    return resolved


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_regular_nofollow(
    path: Path, *, label: str, max_bytes: int
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HPKSnapshotError(f"cannot open {label}: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _error(f"{label} must remain a regular file while open")
        if before.st_size > max_bytes:
            _error(f"{label} exceeds {max_bytes} bytes")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65536, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                _error(f"{label} exceeds {max_bytes} bytes")
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            _error(f"{label} changed while being read")
        data = b"".join(chunks)
        if len(data) != after.st_size:
            _error(f"{label} size changed while being read")
        return data, _stat_identity(after)
    finally:
        os.close(descriptor)


def _reject_constant(value: str) -> None:
    raise HPKSnapshotError(f"non-finite JSON number is forbidden: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _error(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def _parse_json_object(value: bytes, *, label: str) -> dict[str, Any]:
    if value.startswith(b"\xef\xbb\xbf"):
        _error(f"{label} must not contain a UTF-8 BOM")
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HPKSnapshotError(f"{label} is not valid UTF-8: {exc}") from exc
    try:
        decoded = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise HPKSnapshotError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        _error(f"{label} must contain one JSON object")
    return decoded


def _parse_canonical_manifest(value: bytes) -> dict[str, Any]:
    if not value.endswith(b"\n") or value.endswith(b"\n\n") or b"\r" in value:
        _error("snapshot manifest must end in exactly one LF and contain no CR")
    payload = _parse_json_object(value[:-1], label="snapshot manifest")
    if canonical_json_bytes(payload) + b"\n" != value:
        _error("snapshot manifest bytes are not canonical JSON plus one LF")
    return payload


def _parse_canonical_entries(value: bytes) -> tuple[EntryV1, ...]:
    if not value or not value.endswith(b"\n") or b"\r" in value:
        _error("entries.jsonl must be non-empty, LF-terminated, and contain no CR")
    raw_lines = value.splitlines(keepends=True)
    if len(raw_lines) > MAX_ENTRY_COUNT:
        _error(f"entries.jsonl exceeds {MAX_ENTRY_COUNT} records")
    entries: list[EntryV1] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.endswith(b"\n") or raw_line == b"\n":
            _error(f"entries.jsonl line {line_number} is blank or not LF-terminated")
        if len(raw_line) > MAX_ENTRY_BYTES:
            _error(f"entries.jsonl line {line_number} exceeds {MAX_ENTRY_BYTES} bytes")
        payload = _parse_json_object(
            raw_line[:-1], label=f"entries.jsonl line {line_number}"
        )
        if canonical_json_bytes(payload) + b"\n" != raw_line:
            _error(f"entries.jsonl line {line_number} is not canonical JSON")
        try:
            entry = EntryV1.from_dict(payload)
        except HPKValidationError as exc:
            raise HPKSnapshotError(
                f"invalid EntryV1 at entries.jsonl line {line_number}: {exc}"
            ) from exc
        if not entry.accepted:
            _error(
                f"static snapshot entry {entry['entry_id']} must have status=accepted"
            )
        entries.append(entry)
    entry_ids = [str(entry["entry_id"]) for entry in entries]
    if entry_ids != sorted(entry_ids):
        _error("entries.jsonl records must be sorted by entry_id")
    if len(entry_ids) != len(set(entry_ids)):
        _error("entries.jsonl contains a duplicate entry_id")
    return tuple(entries)


def _entry_aggregates(
    entries: tuple[EntryV1, ...],
) -> tuple[dict[str, Any], dict[str, bool]]:
    capabilities: dict[str, Any] = {
        "task_families": sorted(
            {str(entry["condition"]["task_family"]) for entry in entries}
        ),
        "domain_ids": sorted(
            {
                str(domain_id)
                for entry in entries
                for domain_id in entry["provenance"]["domain_ids"]
            }
        ),
        "operations": sorted(
            {str(entry["task_strategy"]["operation"]) for entry in entries}
        ),
        "strategy_families": sorted(
            {str(entry["geometric_strategy"]["strategy_family"]) for entry in entries}
        ),
        "target_relations": sorted(
            {
                str(relation)
                for entry in entries
                if (
                    relation := entry["geometric_strategy"]["target_relation"][
                        "relation"
                    ]
                )
                is not None
            }
        ),
        "hard_constraints": sorted(
            {
                str(constraint)
                for entry in entries
                for constraint in entry["geometric_strategy"]["hard_constraints"]
            }
        ),
        "geometry_source_classes": sorted(
            {
                str(
                    entry["geometric_strategy"]["capability_evidence"][
                        "geometry_source_class"
                    ]
                )
                for entry in entries
            }
        ),
        "semantic_part_observed": any(
            entry["geometric_strategy"]["capability_evidence"]["semantic_part_observed"]
            is True
            for entry in entries
        ),
    }
    flags = {
        "expert_prior_present": any(
            entry["provenance"]["expert_derived"] is True for entry in entries
        ),
        "human_integration_prior_present": any(
            entry["provenance"]["human_prior_used"] is True for entry in entries
        ),
        "oracle_derived_present": any(
            entry["provenance"]["oracle_derived"] is True for entry in entries
        ),
        "learned_hpk_present": any(
            normalize_knowledge_metadata(entry["provenance"])["learned_hpk"] is True for entry in entries
        ),
        "all_entries_formal_evaluation_eligible": all(
            entry["provenance"]["formal_evaluation_eligible"] is True
            for entry in entries
        ),
    }
    return capabilities, flags


def _validate_manifest_entry_binding(
    manifest: SnapshotManifestV1,
    entries: tuple[EntryV1, ...],
    *,
    member_bytes: bytes,
) -> None:
    member = manifest["member"]
    entry_ids = [str(entry["entry_id"]) for entry in entries]
    if manifest["accepted_entry_ids"] != entry_ids:
        _error("manifest accepted_entry_ids do not match member record order")
    if member["record_count"] != len(entries):
        _error("manifest member.record_count does not match entries.jsonl")
    if member["size_bytes"] != len(member_bytes):
        _error("manifest member.size_bytes does not match entries.jsonl")
    if member["sha256"] != _sha256_bytes(member_bytes):
        _error("manifest member.sha256 does not match entries.jsonl")
    capabilities, flags = _entry_aggregates(entries)
    if manifest["capabilities"] != capabilities:
        _error("manifest capabilities are not the exact entry aggregate")
    if normalize_knowledge_metadata(manifest["information_access_flags"]) != flags:
        _error("manifest information_access_flags are not the exact entry aggregate")


def _validate_provenance_gate(
    manifest: SnapshotManifestV1,
    entries: tuple[EntryV1, ...],
    *,
    run_scope: str,
    allow_expert_prior: bool,
    allow_human_integration_prior: bool,
    allow_oracle_evidence: bool,
) -> None:
    if run_scope not in RUN_SCOPES:
        _error(f"run_scope must be one of {sorted(RUN_SCOPES)}")
    if run_scope == "integration" and manifest["purpose"] != "static_integration":
        _error("integration run_scope requires purpose=static_integration")
    if run_scope != "integration" and manifest["purpose"] != "static_evaluation":
        _error("non-integration run_scope requires purpose=static_evaluation")

    for entry in entries:
        provenance = entry["provenance"]
        source_kind = provenance["source_kind"]
        entry_id = entry["entry_id"]
        if run_scope == "formal_no_prior":
            if (
                source_kind != "agent_generated"
                or provenance["expert_derived"]
                or provenance["human_prior_used"]
                or provenance["oracle_derived"]
                or not provenance["formal_evaluation_eligible"]
                or entry["acceptance_scope"] != "formal_no_prior"
            ):
                _error(f"entry {entry_id} is ineligible for formal_no_prior runtime")
        elif run_scope == "integration":
            if source_kind != "human_authored" or not allow_human_integration_prior:
                _error(
                    f"entry {entry_id} requires explicit human integration prior permission"
                )
        elif run_scope == "expert_prior":
            if (
                source_kind != "benchmark_expert"
                or provenance["oracle_derived"]
                or not allow_expert_prior
            ):
                _error(
                    f"entry {entry_id} requires explicit benchmark expert prior permission"
                )
        else:
            if not provenance["oracle_derived"] or not allow_oracle_evidence:
                _error(
                    f"entry {entry_id} requires explicit oracle diagnostic permission"
                )


def load_hpk_snapshot(
    manifest_path: str | os.PathLike[str],
    expected_manifest_sha256: str,
    *,
    run_scope: str,
    allow_expert_prior: bool = False,
    allow_human_integration_prior: bool = False,
    allow_oracle_evidence: bool = False,
) -> LoadedHPKSnapshot:
    """Load one explicitly named immutable snapshot with no discovery or writes."""

    expected_digest = _require_sha256(
        expected_manifest_sha256, label="expected_manifest_sha256"
    )
    for label, value in (
        ("allow_expert_prior", allow_expert_prior),
        ("allow_human_integration_prior", allow_human_integration_prior),
        ("allow_oracle_evidence", allow_oracle_evidence),
    ):
        _require_bool(value, label=label)
    if run_scope not in RUN_SCOPES:
        _error(f"run_scope must be one of {sorted(RUN_SCOPES)}")
    raw_manifest_path = Path(manifest_path)
    if not raw_manifest_path.is_absolute():
        _error("snapshot manifest path must be explicit and absolute")
    resolved_manifest = _resolve_regular_path(
        raw_manifest_path, label="snapshot manifest"
    )
    manifest_bytes, manifest_stat_identity = _read_regular_nofollow(
        resolved_manifest,
        label="snapshot manifest",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    actual_manifest_sha256 = _sha256_bytes(manifest_bytes)
    if actual_manifest_sha256 != expected_digest:
        _error(
            "snapshot manifest SHA-256 mismatch: "
            f"expected {expected_digest}, got {actual_manifest_sha256}"
        )
    manifest_payload = _parse_canonical_manifest(manifest_bytes)
    try:
        manifest = SnapshotManifestV1.from_dict(manifest_payload)
    except HPKValidationError as exc:
        raise HPKSnapshotError(f"invalid SnapshotManifestV1: {exc}") from exc

    snapshot_root = resolved_manifest.parent.resolve(strict=True)
    member_path = snapshot_root / "entries.jsonl"
    resolved_member = _resolve_regular_path(member_path, label="snapshot member")
    if resolved_member.parent != snapshot_root:
        _error("snapshot member resolves outside the snapshot root")
    member_bytes, member_stat_identity = _read_regular_nofollow(
        resolved_member,
        label="snapshot member",
        max_bytes=MAX_MEMBER_BYTES,
    )
    member_digest = _sha256_bytes(member_bytes)
    if member_digest != manifest["member"]["sha256"]:
        _error("snapshot member SHA-256 does not match manifest")
    entries = _parse_canonical_entries(member_bytes)
    _validate_manifest_entry_binding(manifest, entries, member_bytes=member_bytes)
    _validate_provenance_gate(
        manifest,
        entries,
        run_scope=run_scope,
        allow_expert_prior=allow_expert_prior,
        allow_human_integration_prior=allow_human_integration_prior,
        allow_oracle_evidence=allow_oracle_evidence,
    )
    final_manifest_bytes, final_manifest_stat_identity = _read_regular_nofollow(
        resolved_manifest,
        label="snapshot manifest final immutability check",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    final_member_bytes, final_member_stat_identity = _read_regular_nofollow(
        resolved_member,
        label="snapshot member final immutability check",
        max_bytes=MAX_MEMBER_BYTES,
    )
    if (
        final_manifest_bytes != manifest_bytes
        or final_manifest_stat_identity != manifest_stat_identity
    ):
        _error("snapshot manifest changed during load")
    if (
        final_member_bytes != member_bytes
        or final_member_stat_identity != member_stat_identity
    ):
        _error("snapshot member changed during load")
    return LoadedHPKSnapshot(
        manifest_path=resolved_manifest,
        manifest_sha256=actual_manifest_sha256,
        member_path=resolved_member,
        member_sha256=member_digest,
        manifest=manifest,
        entries=entries,
    )


__all__ = [
    "HPKSnapshotError",
    "LoadedHPKSnapshot",
    "MAX_ENTRY_BYTES",
    "MAX_ENTRY_COUNT",
    "MAX_MANIFEST_BYTES",
    "MAX_MEMBER_BYTES",
    "RUN_SCOPES",
    "load_hpk_snapshot",
]
