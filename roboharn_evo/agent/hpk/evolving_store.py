from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from roboharn_evo.agent.hpk.action_transition import ActionEffectTransitionV1
from roboharn_evo.agent.hpk.evolving_schemas import (
    LIFECYCLE_STATES,
    UPDATE_REASON_ORDER,
    UPDATE_REASONS,
    EvidenceV2,
    UpdateDecisionV1,
    evidence_set_sha256,
)
from roboharn_evo.agent.hpk.compatibility import normalize_knowledge_metadata
from roboharn_evo.agent.hpk.promotion import PromotionPolicyV1
from roboharn_evo.agent.hpk.policy_config import (
    EVOLVING_GEOMETRY_POLICY_CONFIG_SHA256 as GEOMETRY_V2_CONFIG_SHA256,
    EVOLVING_GEOMETRY_POLICY_IDENTITIES,
    SAFE_EXPLORATION_GEOMETRY_POLICY_CONFIG_SHA256,
)
from roboharn_evo.agent.hpk.schemas import (
    EntryV1,
    canonical_json_bytes,
    stable_content_id,
    validate_content_id,
)


SNAPSHOT_V2_SCHEMA = "roboharn_evo/hpk/snapshot/v2"
SNAPSHOT_V2_VERSION = 2
SNAPSHOT_V2_PREFIX = "afksnap_"
SNAPSHOT_V2_PURPOSES = frozenset({"evolving_update", "evolving_integration"})
SNAPSHOT_V2_RUNTIME_STATUS = "evolving_ready"
SNAPSHOT_PUBLICATION_PENDING_SCHEMA = "roboharn_evo/hpk/snapshot_publication_pending/v1"
SNAPSHOT_ADVANCE_SCHEMA = "roboharn_evo/hpk/snapshot_advance/v1"
SNAPSHOT_CHILD_CLAIM_SCHEMA = "roboharn_evo/hpk/snapshot_child_claim/v1"
PUBLICATION_PENDING_FILENAME = ".afk_publication_pending.json"
ADVANCE_DIRNAME = ".afk-advances"
CHILD_CLAIM_DIRNAME = ".afk-child-claims"
FORMAL_PROMOTION_POLICY_ID = "hpk_promotion_formal/v1"
FORMAL_PROMOTION_POLICY_CONFIG_SHA256 = (
    "fe43906797e17a514948f5d9a52582ded06bc6090a99f27c404ed862a1456f94"
)
DEV_PROMOTION_POLICY_ID = "hpk_promotion_integration_dev/v1"
DEV_PROMOTION_POLICY_CONFIG_SHA256 = (
    "258aae7acac1532c8e453090d49d9654e89ed08523f12cd436e9d9c8a03c88a6"
)
EVOLVING_GEOMETRY_POLICY_ID = "hpk_geometry_policy/v2"
EVOLVING_GEOMETRY_POLICY_CONFIG_SHA256 = GEOMETRY_V2_CONFIG_SHA256
SAFE_EXPLORATION_GEOMETRY_POLICY_ID = "hpk_geometry_policy/v3"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 512 * 1024
MAX_RECORD_COUNT = 100_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_ID_RE = re.compile(r"^afksnap_[0-9a-f]{64}$")
_ENTRY_ID_RE = re.compile(r"^afkentry_[0-9a-f]{64}$")
_PUBLIC_EPISODE_ID_RE = re.compile(r"^afkepisode_[0-9a-f]{64}$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)

MEMBER_SPECS: dict[str, tuple[str, str]] = {
    "entries": ("entries.jsonl", "application/x-ndjson"),
    "evidence": ("evidence.jsonl", "application/x-ndjson"),
    "private_provenance": ("private_provenance.jsonl", "application/x-ndjson"),
    "update_decisions": ("update_decisions.jsonl", "application/x-ndjson"),
    "promotion_decisions": ("promotion_decisions.jsonl", "application/x-ndjson"),
    "rejected_evidence": ("rejected_evidence.jsonl", "application/x-ndjson"),
}

_MANIFEST_KEYS = {
    "schema",
    "schema_version",
    "snapshot_id",
    "parent",
    "created_at",
    "purpose",
    "members",
    "entry_ids",
    "policy_refs",
    "evidence_batch",
    "source_episode_ids",
    "information_access_flags",
    "counts",
    "runtime_source_identity",
    "runtime_status",
    "immutable",
}
_ENTRY_SET_KEYS = {
    "all",
    "accepted",
    "candidate",
    "revalidation",
    "deprecated",
}
_FLAG_KEYS = {
    "expert_prior_present",
    "human_integration_prior_present",
    "oracle_derived_present",
    "learned_hpk_present",
    "all_entries_formal_evaluation_eligible",
}
_COUNT_KEYS = {
    "entry_count",
    "accepted_count",
    "candidate_count",
    "revalidation_count",
    "deprecated_count",
    "evidence_count",
    "private_provenance_count",
    "update_decision_count",
    "promotion_decision_count",
    "rejected_evidence_count",
}
_PROMOTION_DECISION_V1_KEYS = {
    "schema",
    "promotion_decision_id",
    "update_decision_id",
    "entry_id",
    "from_lifecycle",
    "to_lifecycle",
    "promotion_policy_id",
    "promotion_policy_config_sha256",
    "reason_codes",
    "created_at",
}
_PROMOTION_DECISION_V2_KEYS = _PROMOTION_DECISION_V1_KEYS | {
    "development_only",
    "formal_evaluation_eligible",
    "eligibility_authority",
    "legacy_entry_v1_eligibility_overridden",
}
_REJECTED_EVIDENCE_KEYS = {
    "schema",
    "rejection_id",
    "entry_id",
    "evidence_id",
    "reason_codes",
    "created_at",
}
_REJECTION_REASONS = frozenset(
    {
        "expert_evidence_disallowed",
        "infrastructure_failure",
        "oracle_evidence_disallowed",
        "strategy_identity_mismatch",
    }
)

_ENTRY_IMMUTABLE_FIELDS = (
    "schema",
    "entry_id",
    "condition",
    "task_strategy",
    "geometric_strategy",
    "expected_effect",
    "provenance",
    "acceptance_scope",
)
_APPEND_ONLY_MEMBER_KINDS = (
    "evidence",
    "private_provenance",
    "update_decisions",
    "promotion_decisions",
    "rejected_evidence",
)


class HPKEvolvingSnapshotError(ValueError):
    """Raised on any SnapshotV2 integrity or lineage violation."""


def _fail(message: str) -> None:
    raise HPKEvolvingSnapshotError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact(value: Mapping[str, Any], keys: set[str], *, path: str) -> None:
    actual = set(value)
    if actual != keys:
        _fail(
            f"{path} fields mismatch: missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{path} must be an object")
    return dict(value)


def _sha(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{path} must be a lowercase SHA-256 digest")
    return value


def _integer(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{path} must be a non-negative integer")
    return value


def _timestamp(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        _fail(f"{path} must be a frozen UTC timestamp ending in Z")
    return value


def _timestamp_value(value: str) -> datetime:
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _sorted_unique_strings(
    value: Any,
    *,
    path: str,
    pattern: re.Pattern[str] | None = None,
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        _fail(f"{path} must be an array of strings")
    if value != sorted(value) or len(value) != len(set(value)):
        _fail(f"{path} must be sorted and unique")
    if pattern is not None and any(pattern.fullmatch(item) is None for item in value):
        _fail(f"{path} contains an invalid content ID")
    return tuple(value)


def _json_copy(value: Any, *, path: str) -> Any:
    try:
        encoded = canonical_json_bytes(value)
        return json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise HPKEvolvingSnapshotError(f"{path} is not canonical JSON: {exc}") from exc


def _validate_public_identity(value: Any, *, path: str) -> dict[str, Any]:
    payload = _mapping(_json_copy(value, path=path), path=path)
    if not payload:
        _fail(f"{path} must not be empty")
    forbidden = {
        "path",
        "seed",
        "episode_id",
        "candidate_id",
        "track_id",
        "pose",
        "coordinates",
    }

    def walk(item: Any, current: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                normalized = str(key).lower().replace("-", "_")
                if normalized in forbidden:
                    _fail(f"{current}.{key} contains a private identity field")
                walk(nested, f"{current}.{key}")
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                walk(nested, f"{current}[{index}]")

    walk(payload, path)
    return payload


def promotion_policy_from_ref(
    promotion_policy_ref: Mapping[str, Any],
) -> PromotionPolicyV1:
    """Resolve one self-contained, frozen promotion-policy reference."""

    reference = _mapping(promotion_policy_ref, path="promotion policy reference")
    _exact(
        reference,
        {"policy_id", "config_sha256", "payload"},
        path="promotion policy reference",
    )
    policy_id = reference["policy_id"]
    if not isinstance(policy_id, str) or not policy_id.strip():
        _fail("promotion policy reference.policy_id is invalid")
    config_sha = _sha(
        reference["config_sha256"],
        path="promotion policy reference.config_sha256",
    )
    try:
        policy = PromotionPolicyV1.from_mapping(
            _mapping(reference["payload"], path="promotion policy reference.payload")
        )
    except Exception as exc:
        raise HPKEvolvingSnapshotError(
            f"promotion policy reference.payload is invalid: {exc}"
        ) from exc
    if policy.policy_id != policy_id or policy.config_sha256 != config_sha:
        _fail("promotion policy payload identity/hash mismatch")
    frozen_identities = {
        (
            "afk_promotion_formal/v1",
            "99213322502158fe4f02884dc40743027a0522a17ee06e8fc6edae2f63845ee2",
        ),
        (
            "afk_promotion_integration_dev/v1",
            "c91111d2f4e15d18107e5fe363dd6e6057adbd099db198ff8b1987cfdfcd753c",
        ),
        (
            FORMAL_PROMOTION_POLICY_ID,
            FORMAL_PROMOTION_POLICY_CONFIG_SHA256,
        ),
        (
            DEV_PROMOTION_POLICY_ID,
            DEV_PROMOTION_POLICY_CONFIG_SHA256,
        ),
    }
    if (policy.policy_id, policy.config_sha256) not in frozen_identities:
        _fail("promotion policy reference is not a frozen P0-D/E policy")
    return policy


def snapshot_v2_id_for(payload: Mapping[str, Any]) -> str:
    identity = _json_copy(dict(payload), path="snapshot identity")
    identity.pop("snapshot_id", None)
    return SNAPSHOT_V2_PREFIX + _sha256(canonical_json_bytes(identity))


def _parent_ref(value: Mapping[str, Any], *, path: str) -> dict[str, str | None]:
    parent = _mapping(value, path=path)
    _exact(parent, {"snapshot_id", "manifest_sha256"}, path=path)
    snapshot_id = parent["snapshot_id"]
    manifest_sha = parent["manifest_sha256"]
    if snapshot_id is None or manifest_sha is None:
        if parent != {"snapshot_id": None, "manifest_sha256": None}:
            _fail(f"{path} fields must both be null for a root")
        return {"snapshot_id": None, "manifest_sha256": None}
    if (
        not isinstance(snapshot_id, str)
        or _SNAPSHOT_ID_RE.fullmatch(snapshot_id) is None
    ):
        _fail(f"{path}.snapshot_id is invalid")
    return {
        "snapshot_id": snapshot_id,
        "manifest_sha256": _sha(manifest_sha, path=f"{path}.manifest_sha256"),
    }


def parent_advance_filename(parent: Mapping[str, Any]) -> str:
    normalized = _parent_ref(parent, path="parent advance identity")
    return "afkadvance_" + _sha256(canonical_json_bytes(normalized)) + ".json"


def publication_pending_payload(
    *,
    snapshot_id: str,
    manifest_sha256: str,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    if _SNAPSHOT_ID_RE.fullmatch(str(snapshot_id)) is None:
        _fail("publication pending snapshot_id is invalid")
    return {
        "schema": SNAPSHOT_PUBLICATION_PENDING_SCHEMA,
        "snapshot_id": str(snapshot_id),
        "manifest_sha256": _sha(
            manifest_sha256, path="publication pending manifest_sha256"
        ),
        "parent": _parent_ref(parent, path="publication pending parent"),
    }


def child_claim_payload(*, snapshot_id: str, manifest_sha256: str) -> dict[str, Any]:
    if _SNAPSHOT_ID_RE.fullmatch(str(snapshot_id)) is None:
        _fail("child claim snapshot_id is invalid")
    return {
        "schema": SNAPSHOT_CHILD_CLAIM_SCHEMA,
        "snapshot_id": str(snapshot_id),
        "manifest_sha256": _sha(manifest_sha256, path="child claim manifest_sha256"),
    }


def snapshot_advance_payload(
    *,
    parent: Mapping[str, Any],
    child_snapshot_id: str,
    child_manifest_sha256: str,
) -> dict[str, Any]:
    normalized_parent = _parent_ref(parent, path="snapshot advance parent")
    if _SNAPSHOT_ID_RE.fullmatch(str(child_snapshot_id)) is None:
        _fail("snapshot advance child snapshot_id is invalid")
    child_sha = _sha(
        child_manifest_sha256, path="snapshot advance child manifest_sha256"
    )
    payload: dict[str, Any] = {
        "schema": SNAPSHOT_ADVANCE_SCHEMA,
        "advance_id": "pending",
        "parent": normalized_parent,
        "child": {
            "snapshot_id": str(child_snapshot_id),
            "manifest_sha256": child_sha,
            "manifest_path": (f"{child_snapshot_id}/snapshot_manifest.json"),
        },
    }
    identity = dict(payload)
    identity.pop("advance_id")
    payload["advance_id"] = stable_content_id("afkadvance", identity)
    return payload


class ManifestV2(Mapping[str, Any]):
    """不可变的 HPK v2 快照清单。"""

    __slots__ = ("_canonical",)

    def __init__(self, payload: Mapping[str, Any]) -> None:
        validated = self._validate(_mapping(payload, path="ManifestV2"))
        self._canonical = canonical_json_bytes(validated)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ManifestV2":
        return cls(payload)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical.decode("utf-8"))

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(_MANIFEST_KEYS)

    @property
    def snapshot_id(self) -> str:
        return str(self.to_dict()["snapshot_id"])

    @staticmethod
    def _validate(raw: dict[str, Any]) -> dict[str, Any]:
        payload = _json_copy(raw, path="ManifestV2")
        _exact(payload, _MANIFEST_KEYS, path="ManifestV2")
        if payload["schema"] not in {SNAPSHOT_V2_SCHEMA, "tcm/afk/snapshot/v2"}:
            _fail("ManifestV2.schema is unsupported")
        if payload["schema_version"] != SNAPSHOT_V2_VERSION:
            _fail("ManifestV2.schema_version must equal 2")
        if (
            not isinstance(payload["snapshot_id"], str)
            or _SNAPSHOT_ID_RE.fullmatch(payload["snapshot_id"]) is None
        ):
            _fail("ManifestV2.snapshot_id is invalid")
        parent = _mapping(payload["parent"], path="ManifestV2.parent")
        _exact(parent, {"snapshot_id", "manifest_sha256"}, path="ManifestV2.parent")
        if parent["snapshot_id"] is None or parent["manifest_sha256"] is None:
            if parent != {"snapshot_id": None, "manifest_sha256": None}:
                _fail("ManifestV2 root parent fields must both be null")
        else:
            if (
                not isinstance(parent["snapshot_id"], str)
                or _SNAPSHOT_ID_RE.fullmatch(parent["snapshot_id"]) is None
            ):
                _fail("ManifestV2.parent.snapshot_id is invalid")
            if parent["snapshot_id"] == payload["snapshot_id"]:
                _fail("ManifestV2 cannot identify itself as its parent")
            _sha(parent["manifest_sha256"], path="ManifestV2.parent.manifest_sha256")
        _timestamp(payload["created_at"], path="ManifestV2.created_at")
        if payload["purpose"] not in SNAPSHOT_V2_PURPOSES:
            _fail("ManifestV2.purpose is unsupported")

        members = _mapping(payload["members"], path="ManifestV2.members")
        _exact(members, set(MEMBER_SPECS), path="ManifestV2.members")
        for name, (expected_path, expected_media) in MEMBER_SPECS.items():
            descriptor = _mapping(members[name], path=f"ManifestV2.members.{name}")
            _exact(
                descriptor,
                {"path", "media_type", "sha256", "size_bytes", "record_count"},
                path=f"ManifestV2.members.{name}",
            )
            if (
                descriptor["path"] != expected_path
                or descriptor["media_type"] != expected_media
            ):
                _fail(f"ManifestV2.members.{name} path/media type is not frozen")
            _sha(descriptor["sha256"], path=f"ManifestV2.members.{name}.sha256")
            _integer(
                descriptor["size_bytes"], path=f"ManifestV2.members.{name}.size_bytes"
            )
            _integer(
                descriptor["record_count"],
                path=f"ManifestV2.members.{name}.record_count",
            )

        entry_ids = _mapping(payload["entry_ids"], path="ManifestV2.entry_ids")
        _exact(entry_ids, _ENTRY_SET_KEYS, path="ManifestV2.entry_ids")
        sets = {
            key: set(
                _sorted_unique_strings(
                    entry_ids[key],
                    path=f"ManifestV2.entry_ids.{key}",
                    pattern=_ENTRY_ID_RE,
                )
            )
            for key in _ENTRY_SET_KEYS
        }
        if (
            sets["accepted"] & sets["candidate"]
            or sets["accepted"] & sets["deprecated"]
            or sets["candidate"] & sets["deprecated"]
        ):
            _fail("ManifestV2 lifecycle sets must be disjoint")
        if sets["all"] != sets["accepted"] | sets["candidate"] | sets["deprecated"]:
            _fail("ManifestV2.entry_ids.all must equal the lifecycle union")
        if not sets["revalidation"] <= sets["candidate"]:
            _fail("ManifestV2 revalidation IDs must be candidate entries")

        policies = _mapping(payload["policy_refs"], path="ManifestV2.policy_refs")
        _exact(
            policies,
            {"updater", "promotion", "geometry"},
            path="ManifestV2.policy_refs",
        )
        for name in ("updater", "geometry"):
            reference = _mapping(policies[name], path=f"ManifestV2.policy_refs.{name}")
            _exact(
                reference,
                {"policy_id", "config_sha256"},
                path=f"ManifestV2.policy_refs.{name}",
            )
            if (
                not isinstance(reference["policy_id"], str)
                or not reference["policy_id"].strip()
            ):
                _fail(f"ManifestV2.policy_refs.{name}.policy_id is invalid")
            _sha(
                reference["config_sha256"],
                path=f"ManifestV2.policy_refs.{name}.config_sha256",
            )
        try:
            embedded_promotion = promotion_policy_from_ref(
                _mapping(
                    policies["promotion"],
                    path="ManifestV2.policy_refs.promotion",
                )
            )
        except HPKEvolvingSnapshotError as exc:
            raise HPKEvolvingSnapshotError(
                f"ManifestV2.policy_refs.promotion is invalid: {exc}"
            ) from exc
        from roboharn_evo.agent.hpk.updater import updater_policy_identity

        if policies["updater"] != updater_policy_identity(
            embedded_promotion, legacy=policies["updater"]["policy_id"] == "afk_updater/v1"
        ):
            _fail(
                "ManifestV2.policy_refs.updater disagrees with embedded promotion policy"
            )
        geometry_identity = (
            policies["geometry"]["policy_id"],
            policies["geometry"]["config_sha256"],
        )
        if geometry_identity not in EVOLVING_GEOMETRY_POLICY_IDENTITIES:
            _fail("ManifestV2 requires an exact frozen geometry v2 or v3 policy")

        batch = _mapping(payload["evidence_batch"], path="ManifestV2.evidence_batch")
        _exact(batch, {"batch_id", "sha256"}, path="ManifestV2.evidence_batch")
        if batch["batch_id"] is None or batch["sha256"] is None:
            if batch != {"batch_id": None, "sha256": None}:
                _fail("ManifestV2 empty evidence batch fields must both be null")
        else:
            if (
                not isinstance(batch["batch_id"], str)
                or re.fullmatch(r"afkbatch_[0-9a-f]{64}", batch["batch_id"]) is None
            ):
                _fail("ManifestV2.evidence_batch.batch_id is invalid")
            _sha(batch["sha256"], path="ManifestV2.evidence_batch.sha256")
        _sorted_unique_strings(
            payload["source_episode_ids"],
            path="ManifestV2.source_episode_ids",
            pattern=_PUBLIC_EPISODE_ID_RE,
        )

        flags = _mapping(
            normalize_knowledge_metadata(payload["information_access_flags"]),
            path="ManifestV2.information_access_flags",
        )
        _exact(flags, _FLAG_KEYS, path="ManifestV2.information_access_flags")
        if any(not isinstance(value, bool) for value in flags.values()):
            _fail("ManifestV2 information access flags must be booleans")
        counts = _mapping(payload["counts"], path="ManifestV2.counts")
        _exact(counts, _COUNT_KEYS, path="ManifestV2.counts")
        for key, value in counts.items():
            _integer(value, path=f"ManifestV2.counts.{key}")
        identities = _mapping(
            payload["runtime_source_identity"],
            path="ManifestV2.runtime_source_identity",
        )
        _exact(
            identities,
            {"runtime_identity", "source_identity"},
            path="ManifestV2.runtime_source_identity",
        )
        _validate_public_identity(
            identities["runtime_identity"], path="runtime_identity"
        )
        _validate_public_identity(identities["source_identity"], path="source_identity")
        if payload["runtime_status"] != SNAPSHOT_V2_RUNTIME_STATUS:
            _fail("ManifestV2.runtime_status must equal evolving_ready")
        if payload["immutable"] is not True:
            _fail("ManifestV2.immutable must be true")
        expected_id = snapshot_v2_id_for(payload)
        if payload["snapshot_id"] != expected_id:
            _fail(f"ManifestV2.snapshot_id mismatch: expected {expected_id}")
        return payload


EvolvingSnapshotManifestV2 = ManifestV2


@dataclass(frozen=True, slots=True)
class EvolvingSnapshotState:
    entries: tuple[EntryV1, ...]
    revalidation_entry_ids: tuple[str, ...] = ()
    evidence_records: tuple[dict[str, Any], ...] = ()
    private_provenance_records: tuple[dict[str, Any], ...] = ()
    update_decision_records: tuple[dict[str, Any], ...] = ()
    promotion_decision_records: tuple[dict[str, Any], ...] = ()
    rejected_evidence_records: tuple[dict[str, Any], ...] = ()

    @classmethod
    def empty(cls) -> "EvolvingSnapshotState":
        return cls(entries=())


@dataclass(frozen=True, slots=True)
class LoadedEvolvingSnapshot:
    manifest_path: Path
    manifest_sha256: str
    manifest: ManifestV2
    all_entries: tuple[EntryV1, ...]
    accepted_entries: tuple[EntryV1, ...]
    evidence_records: tuple[dict[str, Any], ...]
    private_provenance_records: tuple[dict[str, Any], ...]
    update_decision_records: tuple[dict[str, Any], ...]
    promotion_decision_records: tuple[dict[str, Any], ...]
    rejected_evidence_records: tuple[dict[str, Any], ...]

    @property
    def snapshot_id(self) -> str:
        return self.manifest.snapshot_id

    @property
    def entries(self) -> tuple[EntryV1, ...]:
        """Runtime-safe accepted projection."""

        return self.accepted_entries

    @property
    def state(self) -> EvolvingSnapshotState:
        return EvolvingSnapshotState(
            entries=self.all_entries,
            revalidation_entry_ids=tuple(self.manifest["entry_ids"]["revalidation"]),
            evidence_records=self.evidence_records,
            private_provenance_records=self.private_provenance_records,
            update_decision_records=self.update_decision_records,
            promotion_decision_records=self.promotion_decision_records,
            rejected_evidence_records=self.rejected_evidence_records,
        )


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_object(data: bytes, *, path: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=lambda value: _fail(f"non-finite JSON value: {value}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise HPKEvolvingSnapshotError(f"{path} is invalid JSON: {exc}") from exc
    return _mapping(value, path=path)


def _read_regular(path: Path, *, label: str, maximum: int) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HPKEvolvingSnapshotError(
            f"{label} does not exist: {path}: {exc}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a non-symlink regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if before.st_size > maximum:
            _fail(f"{label} exceeds its size limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65536, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                _fail(f"{label} exceeds its size limit")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _fail(f"{label} changed while being read")
        data = b"".join(chunks)
        if len(data) != after.st_size:
            _fail(f"{label} changed size while being read")
        return data
    finally:
        os.close(descriptor)


def _record_identity(kind: str, record: Mapping[str, Any]) -> str:
    keys = {
        "evidence": ("evidence_id",),
        "private_provenance": ("evidence_id", "provenance_id", "record_id"),
        "update_decisions": ("update_decision_id", "decision_id", "record_id"),
        "promotion_decisions": ("promotion_decision_id", "decision_id", "record_id"),
        "rejected_evidence": ("rejection_id", "evidence_id", "record_id"),
    }[kind]
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    _fail(f"{kind} record lacks a stable public identity")


def _content_id(value: Any, *, prefix: str, path: str) -> str:
    try:
        validate_content_id(value, prefix=prefix, path=path)
    except Exception as exc:
        raise HPKEvolvingSnapshotError(str(exc)) from exc
    return str(value)


def _reason_codes(
    value: Any,
    *,
    allowed: frozenset[str],
    path: str,
    frozen_order: Sequence[str] | None = None,
) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        _fail(f"{path} must be an array of strings")
    reasons = list(value)
    if len(reasons) != len(set(reasons)):
        _fail(f"{path} must not contain duplicates")
    unknown = sorted(set(reasons) - allowed)
    if unknown:
        _fail(f"{path} contains unsupported reason code(s): {unknown}")
    expected = (
        [reason for reason in frozen_order if reason in set(reasons)]
        if frozen_order is not None
        else sorted(reasons)
    )
    if reasons != expected:
        _fail(f"{path} does not use its frozen order")
    return reasons


def _validate_private_provenance_record(
    value: Mapping[str, Any], *, path: str
) -> dict[str, Any]:
    payload = _json_copy(dict(value), path=path)
    _exact(
        payload,
        {"evidence_id", "transition_digest", "private_transition"},
        path=path,
    )
    _content_id(payload["evidence_id"], prefix="afkev", path=f"{path}.evidence_id")
    digest = _sha(payload["transition_digest"], path=f"{path}.transition_digest")
    try:
        transition = ActionEffectTransitionV1.from_dict(
            _mapping(payload["private_transition"], path=f"{path}.private_transition")
        )
    except Exception as exc:
        raise HPKEvolvingSnapshotError(
            f"{path}.private_transition is invalid: {exc}"
        ) from exc
    actual = _sha256(canonical_json_bytes(transition.to_dict()))
    if digest != actual:
        _fail(f"{path}.transition_digest does not match private_transition")
    return payload


def _validate_promotion_decision_record(
    value: Mapping[str, Any], *, path: str
) -> dict[str, Any]:
    payload = _json_copy(dict(value), path=path)
    schema = payload.get("schema")
    if schema in {"roboharn_evo/hpk/promotion_decision/v1", "tcm/afk/promotion_decision/v1"}:
        _exact(payload, _PROMOTION_DECISION_V1_KEYS, path=path)
    elif schema in {"roboharn_evo/hpk/promotion_decision/v2", "tcm/afk/promotion_decision/v2"}:
        _exact(payload, _PROMOTION_DECISION_V2_KEYS, path=path)
        development_only = payload["development_only"]
        formal_eligible = payload["formal_evaluation_eligible"]
        legacy_overridden = payload["legacy_entry_v1_eligibility_overridden"]
        if any(
            not isinstance(item, bool)
            for item in (development_only, formal_eligible, legacy_overridden)
        ):
            _fail(f"{path} eligibility fields must be booleans")
        if development_only == formal_eligible:
            _fail(f"{path} eligibility flags must be exact opposites")
        if legacy_overridden is not development_only:
            _fail(f"{path} legacy override marker must equal development_only")
        if payload["eligibility_authority"] != "snapshot_v2_promotion_policy":
            _fail(f"{path}.eligibility_authority is unsupported")
    else:
        _fail(f"{path}.schema is unsupported")
    _content_id(
        payload["promotion_decision_id"],
        prefix="afkpromotion",
        path=f"{path}.promotion_decision_id",
    )
    _content_id(
        payload["update_decision_id"],
        prefix="afkupdate",
        path=f"{path}.update_decision_id",
    )
    _content_id(payload["entry_id"], prefix="afkentry", path=f"{path}.entry_id")
    for name in ("from_lifecycle", "to_lifecycle"):
        if payload[name] not in LIFECYCLE_STATES:
            _fail(f"{path}.{name} is not a frozen lifecycle state")
    if (
        not isinstance(payload["promotion_policy_id"], str)
        or not payload["promotion_policy_id"].strip()
    ):
        _fail(f"{path}.promotion_policy_id must be non-empty")
    _sha(
        payload["promotion_policy_config_sha256"],
        path=f"{path}.promotion_policy_config_sha256",
    )
    _reason_codes(
        payload["reason_codes"],
        allowed=UPDATE_REASONS,
        path=f"{path}.reason_codes",
        frozen_order=UPDATE_REASON_ORDER,
    )
    _timestamp(payload["created_at"], path=f"{path}.created_at")
    identity_payload = dict(payload)
    identity_payload.pop("promotion_decision_id")
    expected = stable_content_id("afkpromotion", identity_payload)
    if payload["promotion_decision_id"] != expected:
        _fail(f"{path}.promotion_decision_id content identity mismatch")
    return payload


def _validate_rejected_evidence_record(
    value: Mapping[str, Any], *, path: str
) -> dict[str, Any]:
    payload = _json_copy(dict(value), path=path)
    _exact(payload, _REJECTED_EVIDENCE_KEYS, path=path)
    if payload["schema"] not in {"roboharn_evo/hpk/rejected_evidence/v1", "tcm/afk/rejected_evidence/v1"}:
        _fail(f"{path}.schema is unsupported")
    _content_id(
        payload["rejection_id"],
        prefix="afkrejection",
        path=f"{path}.rejection_id",
    )
    _content_id(payload["entry_id"], prefix="afkentry", path=f"{path}.entry_id")
    _content_id(payload["evidence_id"], prefix="afkev", path=f"{path}.evidence_id")
    _reason_codes(
        payload["reason_codes"],
        allowed=_REJECTION_REASONS,
        path=f"{path}.reason_codes",
    )
    _timestamp(payload["created_at"], path=f"{path}.created_at")
    identity_payload = dict(payload)
    identity_payload.pop("rejection_id")
    expected = stable_content_id("afkrejection", identity_payload)
    if payload["rejection_id"] != expected:
        _fail(f"{path}.rejection_id content identity mismatch")
    return payload


def _validate_member_record(
    kind: str, value: Mapping[str, Any], *, path: str
) -> dict[str, Any]:
    try:
        if kind == "evidence":
            return EvidenceV2.from_dict(value).to_dict()
        if kind == "private_provenance":
            return _validate_private_provenance_record(value, path=path)
        if kind == "update_decisions":
            return UpdateDecisionV1.from_dict(value).to_dict()
        if kind == "promotion_decisions":
            return _validate_promotion_decision_record(value, path=path)
        if kind == "rejected_evidence":
            return _validate_rejected_evidence_record(value, path=path)
    except HPKEvolvingSnapshotError:
        raise
    except Exception as exc:
        raise HPKEvolvingSnapshotError(f"{path} is invalid: {exc}") from exc
    _fail(f"unsupported member kind: {kind}")


def canonical_jsonl_records(
    kind: str, records: Sequence[Mapping[str, Any] | Any]
) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    if kind not in MEMBER_SPECS or kind == "entries":
        _fail(f"unsupported generic member kind: {kind}")
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(records):
        if isinstance(value, Mapping):
            payload = dict(value)
        else:
            method = getattr(value, "to_dict", None)
            if not callable(method):
                _fail(f"{kind}[{index}] must be a mapping or expose to_dict()")
            payload = _mapping(method(), path=f"{kind}[{index}]")
        normalized.append(
            _validate_member_record(kind, payload, path=f"{kind}[{index}]")
        )
    normalized.sort(key=lambda item: _record_identity(kind, item))
    identities = [_record_identity(kind, item) for item in normalized]
    if len(identities) != len(set(identities)):
        _fail(f"{kind} contains duplicate stable identities")
    encoded = b"".join(canonical_json_bytes(item) + b"\n" for item in normalized)
    return encoded, tuple(normalized)


def canonical_entries_jsonl(
    entries: Sequence[EntryV1 | Mapping[str, Any]],
) -> tuple[bytes, tuple[EntryV1, ...]]:
    typed = tuple(
        entry if isinstance(entry, EntryV1) else EntryV1.from_dict(entry)
        for entry in entries
    )
    ordered = tuple(sorted(typed, key=lambda item: str(item["entry_id"])))
    ids = [str(item["entry_id"]) for item in ordered]
    if len(ids) != len(set(ids)):
        _fail("entries contain duplicate entry IDs")
    return b"".join(
        canonical_json_bytes(item.to_dict()) + b"\n" for item in ordered
    ), ordered


def _parse_jsonl(data: bytes, *, kind: str) -> tuple[dict[str, Any], ...]:
    if not data:
        return ()
    if not data.endswith(b"\n") or b"\r" in data:
        _fail(f"{kind} member must use exactly LF-terminated JSONL")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(data.splitlines(), 1):
        if not line or len(line) > MAX_RECORD_BYTES:
            _fail(f"{kind} line {index} is empty or too large")
        payload = _parse_object(line, path=f"{kind} line {index}")
        if canonical_json_bytes(payload) != line:
            _fail(f"{kind} line {index} is not canonical JSON")
        records.append(
            _validate_member_record(kind, payload, path=f"{kind} line {index}")
        )
    if len(records) > MAX_RECORD_COUNT:
        _fail(f"{kind} contains too many records")
    identities = [_record_identity(kind, item) for item in records]
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        _fail(f"{kind} records must be sorted and unique")
    return tuple(records)


def _parse_entries(data: bytes) -> tuple[EntryV1, ...]:
    if not data:
        return ()
    raw = _parse_jsonl_as_objects(data, kind="entries")
    entries = tuple(EntryV1.from_dict(item) for item in raw)
    ids = [str(item["entry_id"]) for item in entries]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        _fail("entries records must be sorted and unique")
    return entries


def _parse_jsonl_as_objects(data: bytes, *, kind: str) -> tuple[dict[str, Any], ...]:
    if not data:
        return ()
    if not data.endswith(b"\n") or b"\r" in data:
        _fail(f"{kind} member must use LF-terminated JSONL")
    values = []
    for index, line in enumerate(data.splitlines(), 1):
        if not line or len(line) > MAX_RECORD_BYTES:
            _fail(f"{kind} line {index} is empty or too large")
        payload = _parse_object(line, path=f"{kind} line {index}")
        if canonical_json_bytes(payload) != line:
            _fail(f"{kind} line {index} is not canonical JSON")
        values.append(payload)
    return tuple(values)


def entry_lifecycle_sets(
    entries: Sequence[EntryV1], revalidation_ids: Sequence[str]
) -> dict[str, list[str]]:
    accepted = sorted(
        str(item["entry_id"]) for item in entries if item["status"] == "accepted"
    )
    candidate = sorted(
        str(item["entry_id"]) for item in entries if item["status"] == "candidate"
    )
    deprecated = sorted(
        str(item["entry_id"]) for item in entries if item["status"] == "deprecated"
    )
    revalidation = sorted(set(str(item) for item in revalidation_ids))
    if not set(revalidation) <= set(candidate):
        _fail("revalidation IDs must refer to candidate entries")
    return {
        "all": sorted(accepted + candidate + deprecated),
        "accepted": accepted,
        "candidate": candidate,
        "revalidation": revalidation,
        "deprecated": deprecated,
    }


def promotion_policy_formal_eligibility(
    promotion_policy_ref: Mapping[str, Any],
) -> bool:
    policy = promotion_policy_from_ref(promotion_policy_ref)
    return bool(policy["formal_evaluation_eligible"])


def information_access_flags(
    entries: Sequence[EntryV1],
    evidence_records: Sequence[Mapping[str, Any]] = (),
    *,
    promotion_policy_ref: Mapping[str, Any],
) -> dict[str, bool]:
    policy_formal = promotion_policy_formal_eligibility(promotion_policy_ref)
    return {
        "expert_prior_present": (
            any(item["provenance"]["expert_derived"] is True for item in entries)
            or any(item.get("expert_derived") is True for item in evidence_records)
        ),
        "human_integration_prior_present": any(
            item["provenance"]["human_prior_used"] is True for item in entries
        ),
        "oracle_derived_present": (
            any(item["provenance"]["oracle_derived"] is True for item in entries)
            or any(item.get("oracle_derived") is True for item in evidence_records)
        ),
        "learned_hpk_present": any(
            normalize_knowledge_metadata(item["provenance"])["learned_hpk"] is True for item in entries
        ),
        "all_entries_formal_evaluation_eligible": all(
            item["provenance"]["formal_evaluation_eligible"] is True for item in entries
        )
        and policy_formal,
    }


def _validate_snapshot_member_bindings(
    parsed: Mapping[str, Any], manifest: ManifestV2
) -> None:
    """Validate cross-member identities, not only descriptor hashes."""

    from roboharn_evo.agent.hpk.evidence_store import EvidenceStore

    evidence = tuple(parsed["evidence"])
    private = tuple(parsed["private_provenance"])
    public_by_id = {str(value["evidence_id"]): value for value in evidence}
    private_by_id = {str(value["evidence_id"]): value for value in private}
    if len(public_by_id) != len(evidence) or len(private_by_id) != len(private):
        _fail("evidence/private provenance contains duplicate evidence IDs")
    if set(public_by_id) != set(private_by_id):
        _fail("evidence and private provenance IDs must match one-to-one")
    try:
        store = EvidenceStore(
            (
                EvidenceV2.from_dict(public_by_id[evidence_id]),
                ActionEffectTransitionV1.from_dict(
                    private_by_id[evidence_id]["private_transition"]
                ),
            )
            for evidence_id in sorted(public_by_id)
        )
        batch = store.to_batch()
    except Exception as exc:
        raise HPKEvolvingSnapshotError(
            f"public/private evidence binding is invalid: {exc}"
        ) from exc

    declared_batch = manifest["evidence_batch"]
    declared_sources = tuple(manifest["source_episode_ids"])
    if evidence:
        if declared_batch != {
            "batch_id": batch.batch_id,
            "sha256": batch.evidence_set_sha256,
        }:
            _fail("manifest evidence batch identity does not match member contents")
        if declared_sources != batch.source_episode_ids:
            _fail("manifest source episode IDs do not match evidence contents")
    elif declared_batch != {"batch_id": None, "sha256": None}:
        _fail("empty evidence members require a null evidence batch identity")
    elif declared_sources:
        _fail("empty evidence members cannot declare source episode IDs")

    entries_by_id = {str(value["entry_id"]): value for value in parsed["entries"]}
    entry_ids = set(entries_by_id)
    evidence_ids = set(public_by_id)
    from roboharn_evo.agent.hpk.updater import strategy_key_for_entry

    expected_strategy_by_entry = {
        entry_id: strategy_key_for_entry(entry).stable_id
        for entry_id, entry in entries_by_id.items()
    }
    verdict_for_ref_group = {
        "supporting": "support",
        "opposing": "oppose",
        "unverified": "unverified",
    }
    for entry_id, entry in entries_by_id.items():
        for ref_group, expected_verdict in verdict_for_ref_group.items():
            for evidence_id in entry["evidence_refs"][ref_group]:
                record = public_by_id.get(str(evidence_id))
                if record is None:
                    _fail("entry evidence ref is outside the evidence member")
                if record["verdict"] != expected_verdict:
                    _fail("entry evidence ref verdict disagrees with its ref group")
                if record["strategy_key_id"] != expected_strategy_by_entry[entry_id]:
                    _fail("entry evidence ref has a different strategy identity")

    updates: dict[str, UpdateDecisionV1] = {}
    for raw in parsed["update_decisions"]:
        decision = UpdateDecisionV1.from_dict(raw)
        if decision.stable_id in updates:
            _fail("update decision IDs must be unique")
        if decision["entry_id"] not in entry_ids:
            _fail("update decision refers to an entry outside this snapshot")
        if not set(decision["evidence_ids"]) <= evidence_ids:
            _fail("update decision refers to evidence outside this snapshot")
        if (
            decision["strategy_key_id"]
            != expected_strategy_by_entry[str(decision["entry_id"])]
        ):
            _fail("update decision strategy identity disagrees with its entry")
        decision_evidence = tuple(
            EvidenceV2.from_dict(public_by_id[evidence_id])
            for evidence_id in decision["evidence_ids"]
        )
        if decision["evidence_set_sha256"] != evidence_set_sha256(decision_evidence):
            _fail("update decision evidence-set hash disagrees with its evidence")
        expected_counts = {
            verdict: sum(record["verdict"] == verdict for record in decision_evidence)
            for verdict in ("support", "oppose", "unverified")
        }
        if decision["counts"] != expected_counts:
            _fail("update decision counts disagree with its evidence")
        expected_episode_counts = {
            verdict: len(
                {
                    str(record["episode_evidence_group_id"])
                    for record in decision_evidence
                    if record["verdict"] == verdict
                }
            )
            for verdict in ("support", "oppose", "unverified")
        }
        if decision["distinct_episode_counts"] != expected_episode_counts:
            _fail("update decision episode counts disagree with its evidence")
        expected_scene_counts = {
            verdict: len(
                {
                    str(record["source_identity"]["scene_signature_sha256"])
                    for record in decision_evidence
                    if record["verdict"] == verdict
                }
            )
            for verdict in ("support", "oppose", "unverified")
        }
        if decision["distinct_scene_signature_counts"] != expected_scene_counts:
            _fail("update decision scene counts disagree with its evidence")
        updates[decision.stable_id] = decision

    promotions = {
        str(value["update_decision_id"]): value
        for value in parsed["promotion_decisions"]
    }
    if len(promotions) != len(parsed["promotion_decisions"]):
        _fail("promotion decisions repeat an update decision identity")
    if set(promotions) != set(updates):
        _fail("promotion decisions must correspond one-to-one with updates")
    promotion_ref = manifest["policy_refs"]["promotion"]
    policy_formal = promotion_policy_formal_eligibility(promotion_ref)
    for decision_id, promotion in promotions.items():
        decision = updates[decision_id]
        expected_fields = {
            "entry_id": decision["entry_id"],
            "from_lifecycle": decision["from_lifecycle"],
            "to_lifecycle": decision["to_lifecycle"],
            "promotion_policy_id": decision["promotion_policy_id"],
            "promotion_policy_config_sha256": decision[
                "promotion_policy_config_sha256"
            ],
            "reason_codes": decision["reason_codes"],
            "created_at": decision["created_at"],
        }
        if any(promotion[name] != value for name, value in expected_fields.items()):
            _fail("promotion decision disagrees with its update decision")
        if (
            decision["promotion_policy_id"] != promotion_ref["policy_id"]
            or decision["promotion_policy_config_sha256"]
            != promotion_ref["config_sha256"]
        ):
            _fail("promotion decision disagrees with manifest policy reference")
        if promotion["schema"] in {"roboharn_evo/hpk/promotion_decision/v2", "tcm/afk/promotion_decision/v2"} and (
            promotion["formal_evaluation_eligible"] is not policy_formal
            or promotion["development_only"] is policy_formal
            or promotion["legacy_entry_v1_eligibility_overridden"]
            is not (not policy_formal)
        ):
            _fail("promotion receipt eligibility disagrees with frozen policy")

    for rejection in parsed["rejected_evidence"]:
        if rejection["entry_id"] not in entry_ids:
            _fail("rejected evidence record refers to an entry outside this snapshot")
        if rejection["evidence_id"] not in evidence_ids:
            _fail("rejected evidence record refers outside the evidence member")


def _records_by_identity(
    kind: str, records: Sequence[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    return {_record_identity(kind, record): record for record in records}


def _entry_ref_sets(entry: EntryV1) -> dict[str, set[str]]:
    refs = entry["evidence_refs"]
    return {
        "support": set(str(value) for value in refs["supporting"]),
        "oppose": set(str(value) for value in refs["opposing"]),
        "unverified": set(str(value) for value in refs["unverified"]),
    }


def _effective_lifecycle(entry: EntryV1, manifest: ManifestV2) -> str:
    entry_id = str(entry["entry_id"])
    if entry_id in set(manifest["entry_ids"]["revalidation"]):
        return "candidate_for_revalidation"
    return str(entry["status"])


def _validate_entry_update_derivation(
    *,
    parent_entry: EntryV1 | None,
    child_entry: EntryV1,
    parent_manifest: ManifestV2,
    child_manifest: ManifestV2,
    decision: UpdateDecisionV1,
    new_global_evidence_ids: set[str],
    evidence_by_id: Mapping[str, EvidenceV2],
    promotion_policy: PromotionPolicyV1,
    prior_revalidation_decision: UpdateDecisionV1 | None,
) -> None:
    entry_id = str(child_entry["entry_id"])
    if decision["entry_id"] != entry_id:
        _fail("new update decision refers to the wrong child entry")
    for field in _ENTRY_IMMUTABLE_FIELDS:
        if parent_entry is not None and (
            canonical_json_bytes(parent_entry[field])
            != canonical_json_bytes(child_entry[field])
        ):
            _fail(f"existing entry immutable field changed: {entry_id}.{field}")

    before_refs = (
        {"support": set(), "oppose": set(), "unverified": set()}
        if parent_entry is None
        else _entry_ref_sets(parent_entry)
    )
    after_refs = _entry_ref_sets(child_entry)
    for verdict in ("support", "oppose", "unverified"):
        if not before_refs[verdict] <= after_refs[verdict]:
            _fail(f"existing entry evidence refs regressed: {entry_id}.{verdict}")
    before_all = set().union(*before_refs.values())
    after_all = set().union(*after_refs.values())
    appended_refs = after_all - before_all
    if not appended_refs:
        _fail("a changed learned entry requires newly appended evidence")
    if appended_refs != set(decision["new_evidence_ids"]):
        _fail("update decision new evidence does not derive the entry delta")
    if not appended_refs <= new_global_evidence_ids:
        _fail("entry update reused evidence that was not new in this child")
    if after_all != set(decision["evidence_ids"]):
        _fail("update decision evidence set does not match the child entry")

    expected_from = (
        "candidate"
        if parent_entry is None
        else _effective_lifecycle(parent_entry, parent_manifest)
    )
    expected_to = _effective_lifecycle(child_entry, child_manifest)
    if decision["from_lifecycle"] != expected_from:
        _fail("update decision does not start at the parent lifecycle")
    if decision["to_lifecycle"] != expected_to:
        _fail("update decision does not end at the child lifecycle")
    if decision["persisted_entry_status"] != child_entry["status"]:
        _fail("update decision persisted status differs from the child entry")

    statistics = child_entry["effect_statistics"]
    expected_counts = {
        "support": statistics["support_count"],
        "oppose": statistics["oppose_count"],
        "unverified": statistics["unverified_count"],
    }
    if decision["counts"] != expected_counts:
        _fail("update decision counts do not derive child entry statistics")
    for field in (
        "posterior_alpha",
        "posterior_beta",
        "estimated_success_probability",
        "lower_confidence_bound",
    ):
        if decision[field] != statistics[field]:
            _fail(f"update decision {field} differs from child entry statistics")
    if decision["created_at"] != statistics["last_updated_at"]:
        _fail("update decision timestamp differs from child entry statistics")
    if parent_entry is not None and _timestamp_value(
        statistics["last_updated_at"]
    ) < _timestamp_value(parent_entry["effect_statistics"]["last_updated_at"]):
        _fail("child entry last_updated_at regressed")
    if decision["promotion_policy_id"] != child_entry["promotion_policy_id"]:
        _fail("update decision policy differs from child entry policy")
    if decision["evaluation_ref"] != child_entry["evaluation_ref"]:
        _fail("update decision evaluation ref differs from child entry")
    expected_evaluation_status = (
        "passed"
        if expected_to == "accepted"
        else "failed"
        if expected_to in {"candidate_for_revalidation", "deprecated"}
        else "not_evaluated"
    )
    if child_entry["evaluation_status"] != expected_evaluation_status:
        _fail("child entry evaluation status is not derived from its lifecycle")

    from roboharn_evo.agent.hpk.updater import HPKUpdater

    updater = HPKUpdater(promotion_policy)
    decision_evidence = tuple(
        evidence_by_id[evidence_id] for evidence_id in decision["evidence_ids"]
    )
    try:
        if parent_entry is None:
            replay = updater.create_candidate_entry(
                condition=child_entry["condition"],
                task_strategy=child_entry["task_strategy"],
                geometric_strategy=child_entry["geometric_strategy"],
                expected_effect=child_entry["expected_effect"],
                provenance=child_entry["provenance"],
                acceptance_scope=str(child_entry["acceptance_scope"]),
                created_at=str(decision["created_at"]),
                evidence=decision_evidence,
            )
        else:
            replay = updater.update_entry(
                parent_entry,
                decision_evidence,
                current_lifecycle=expected_from,
                prior_decision=prior_revalidation_decision,
            )
    except Exception as exc:
        raise HPKEvolvingSnapshotError(
            f"promotion-policy replay rejected the child update: {exc}"
        ) from exc
    if replay.rejected_evidence:
        _fail("promotion-policy replay rejected evidence attached by the child")
    if canonical_json_bytes(replay.entry.to_dict()) != canonical_json_bytes(
        child_entry.to_dict()
    ):
        _fail("child entry differs from authoritative promotion-policy replay")
    if canonical_json_bytes(replay.decision.to_dict()) != canonical_json_bytes(
        decision.to_dict()
    ):
        _fail("update decision differs from authoritative promotion-policy replay")


def _validate_parent_child_lineage(
    *,
    parent: LoadedEvolvingSnapshot,
    child_manifest: ManifestV2,
    child_parsed: Mapping[str, Any],
) -> None:
    """Fail closed unless a child is an append-only derivation of its parent."""

    expected_parent = {
        "snapshot_id": parent.snapshot_id,
        "manifest_sha256": parent.manifest_sha256,
    }
    if child_manifest["parent"] != expected_parent:
        _fail("child manifest does not identify the supplied parent")
    if child_manifest["policy_refs"] != parent.manifest["policy_refs"]:
        _fail("child policy references differ from its immutable lineage")
    if _timestamp_value(child_manifest["created_at"]) < _timestamp_value(
        parent.manifest["created_at"]
    ):
        _fail("child snapshot timestamp regressed")
    promotion_policy = promotion_policy_from_ref(
        child_manifest["policy_refs"]["promotion"]
    )

    parent_records = {
        "evidence": parent.evidence_records,
        "private_provenance": parent.private_provenance_records,
        "update_decisions": parent.update_decision_records,
        "promotion_decisions": parent.promotion_decision_records,
        "rejected_evidence": parent.rejected_evidence_records,
    }
    new_records: dict[str, dict[str, Mapping[str, Any]]] = {}
    for kind in _APPEND_ONLY_MEMBER_KINDS:
        before = _records_by_identity(kind, parent_records[kind])
        after = _records_by_identity(kind, child_parsed[kind])
        missing = sorted(set(before) - set(after))
        if missing:
            _fail(f"child removed append-only {kind} history: {missing}")
        for identity, record in before.items():
            if canonical_json_bytes(record) != canonical_json_bytes(after[identity]):
                _fail(f"child rewrote append-only {kind} history: {identity}")
        new_records[kind] = {
            identity: after[identity] for identity in sorted(set(after) - set(before))
        }

    parent_entries = {str(entry["entry_id"]): entry for entry in parent.all_entries}
    child_entries = {str(entry["entry_id"]): entry for entry in child_parsed["entries"]}
    missing_entries = sorted(set(parent_entries) - set(child_entries))
    if missing_entries:
        _fail(f"child removed existing entry identities: {missing_entries}")
    from roboharn_evo.agent.hpk.updater import strategy_key_for_entry

    entry_by_strategy: dict[str, EntryV1] = {}
    for entry in child_entries.values():
        strategy_id = strategy_key_for_entry(entry).stable_id
        if strategy_id in entry_by_strategy:
            _fail("child contains multiple entries for one strategy identity")
        entry_by_strategy[strategy_id] = entry

    new_decisions = {
        identity: UpdateDecisionV1.from_dict(record)
        for identity, record in new_records["update_decisions"].items()
    }
    decisions_by_entry: dict[str, list[UpdateDecisionV1]] = {}
    for decision in new_decisions.values():
        decisions_by_entry.setdefault(str(decision["entry_id"]), []).append(decision)

    parent_evidence_ids = {
        str(record["evidence_id"]) for record in parent.evidence_records
    }
    child_evidence_ids = {
        str(record["evidence_id"]) for record in child_parsed["evidence"]
    }
    new_evidence_ids = child_evidence_ids - parent_evidence_ids
    evidence_by_id = {
        str(record["evidence_id"]): EvidenceV2.from_dict(record)
        for record in child_parsed["evidence"]
    }
    attached_entry_by_evidence: dict[str, str] = {}
    for entry_id, child_entry in child_entries.items():
        parent_entry = parent_entries.get(entry_id)
        changed = parent_entry is None or (
            canonical_json_bytes(parent_entry.to_dict())
            != canonical_json_bytes(child_entry.to_dict())
        )
        decisions = decisions_by_entry.pop(entry_id, [])
        if not changed:
            if decisions:
                _fail("unchanged entry has an unexplained new update decision")
            continue
        if (
            parent_entry is None
            and child_entry["provenance"]["source_kind"] == "human_authored"
            and not set().union(*_entry_ref_sets(child_entry).values())
        ):
            if decisions:
                _fail("human integration entry cannot carry learned update history")
            if child_manifest["purpose"] != "evolving_integration":
                _fail("human-authored entries are limited to integration snapshots")
            continue
        if len(decisions) != 1:
            _fail("each changed learned entry requires exactly one new update decision")
        decision = decisions[0]
        prior_revalidation_decision = None
        if (
            parent_entry is not None
            and _effective_lifecycle(parent_entry, parent.manifest)
            == "candidate_for_revalidation"
        ):
            markers = [
                UpdateDecisionV1.from_dict(record)
                for record in parent.update_decision_records
                if record["entry_id"] == entry_id
                and record["to_lifecycle"] == "candidate_for_revalidation"
            ]
            if not markers:
                _fail("revalidation parent entry has no immutable decision marker")
            prior_revalidation_decision = sorted(
                markers,
                key=lambda value: (str(value["created_at"]), value.stable_id),
            )[-1]
        _validate_entry_update_derivation(
            parent_entry=parent_entry,
            child_entry=child_entry,
            parent_manifest=parent.manifest,
            child_manifest=child_manifest,
            decision=decision,
            new_global_evidence_ids=new_evidence_ids,
            evidence_by_id=evidence_by_id,
            promotion_policy=promotion_policy,
            prior_revalidation_decision=prior_revalidation_decision,
        )
        for evidence_id in decision["new_evidence_ids"]:
            if evidence_id in attached_entry_by_evidence:
                _fail("new evidence is attached by more than one update decision")
            attached_entry_by_evidence[str(evidence_id)] = entry_id
    if decisions_by_entry:
        _fail("new update decision has no corresponding child entry delta")

    new_rejections = tuple(new_records["rejected_evidence"].values())
    rejected_new_evidence = [str(record["evidence_id"]) for record in new_rejections]
    if len(rejected_new_evidence) != len(set(rejected_new_evidence)):
        _fail("new rejected-evidence records repeat one evidence identity")
    rejection_by_evidence = {
        str(record["evidence_id"]): record for record in new_rejections
    }
    rejected_set = set(rejection_by_evidence)
    if not rejected_set <= new_evidence_ids:
        _fail("new rejection record does not refer to newly appended evidence")
    attached_set = set(attached_entry_by_evidence)
    if attached_set & rejected_set:
        _fail("new evidence cannot be both attached and rejected")
    if new_evidence_ids != attached_set | rejected_set:
        _fail("new evidence is not accounted for by an update or rejection")

    for evidence_id in sorted(new_evidence_ids):
        evidence = evidence_by_id[evidence_id]
        strategy_id = evidence.strategy_key_id
        matched_entry = (
            None if strategy_id is None else entry_by_strategy.get(strategy_id)
        )
        if matched_entry is None:
            _fail("new evidence has no unique entry for its strategy identity")
        matched_entry_id = str(matched_entry["entry_id"])
        rejection_reasons: list[str] = []
        if evidence["oracle_derived"] and not promotion_policy["allow_oracle_evidence"]:
            rejection_reasons.append("oracle_evidence_disallowed")
        if evidence["expert_derived"] and not promotion_policy["allow_expert_prior"]:
            rejection_reasons.append("expert_evidence_disallowed")
        if not evidence["infrastructure_valid"]:
            rejection_reasons.append("infrastructure_failure")
        rejection_reasons.sort()
        rejection = rejection_by_evidence.get(evidence_id)
        attached_entry_id = attached_entry_by_evidence.get(evidence_id)
        if rejection_reasons:
            if attached_entry_id is not None or rejection is None:
                _fail("policy-ineligible new evidence was not rejected exactly")
            if (
                rejection["entry_id"] != matched_entry_id
                or rejection["reason_codes"] != rejection_reasons
                or rejection["created_at"] != evidence["created_at"]
            ):
                _fail("new evidence rejection differs from authoritative policy route")
        else:
            if rejection is not None:
                _fail("policy-eligible new evidence was falsely rejected")
            if attached_entry_id != matched_entry_id:
                _fail("policy-eligible new evidence was attached to the wrong entry")

    if not set(parent.manifest["source_episode_ids"]) <= set(
        child_manifest["source_episode_ids"]
    ):
        _fail("child source episode history regressed")


def validate_evolving_snapshot_lineage(
    *,
    parent: LoadedEvolvingSnapshot,
    manifest: ManifestV2 | Mapping[str, Any],
    entries: Sequence[EntryV1 | Mapping[str, Any]],
    evidence_records: Sequence[Mapping[str, Any]],
    private_provenance_records: Sequence[Mapping[str, Any]],
    update_decision_records: Sequence[Mapping[str, Any]],
    promotion_decision_records: Sequence[Mapping[str, Any]],
    rejected_evidence_records: Sequence[Mapping[str, Any]],
) -> None:
    """Validate one explicit parent/child pair without scanning mutable heads."""

    typed_manifest = (
        manifest if isinstance(manifest, ManifestV2) else ManifestV2.from_dict(manifest)
    )
    typed_entries = tuple(
        entry if isinstance(entry, EntryV1) else EntryV1.from_dict(entry)
        for entry in entries
    )
    parsed: dict[str, Any] = {
        "entries": typed_entries,
        "evidence": tuple(dict(value) for value in evidence_records),
        "private_provenance": tuple(
            dict(value) for value in private_provenance_records
        ),
        "update_decisions": tuple(dict(value) for value in update_decision_records),
        "promotion_decisions": tuple(
            dict(value) for value in promotion_decision_records
        ),
        "rejected_evidence": tuple(dict(value) for value in rejected_evidence_records),
    }
    _validate_snapshot_member_bindings(parsed, typed_manifest)
    _validate_parent_child_lineage(
        parent=parent,
        child_manifest=typed_manifest,
        child_parsed=parsed,
    )


def _read_canonical_control_file(path: Path, *, label: str) -> dict[str, Any]:
    raw = _read_regular(path, label=label, maximum=64 * 1024)
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw:
        _fail(f"{label} must end in exactly one LF")
    payload = _parse_object(raw[:-1], path=label)
    if canonical_json_bytes(payload) + b"\n" != raw:
        _fail(f"{label} is not canonical JSON plus LF")
    return payload


def _require_committed_publication(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    manifest: ManifestV2,
) -> None:
    root = manifest_path.resolve(strict=True).parent
    pending_path = root / PUBLICATION_PENDING_FILENAME
    try:
        pending_path.lstat()
    except FileNotFoundError:
        # SnapshotV2 artifacts created before the committed-ref protocol remain
        # byte-compatible and explicitly loadable by their pinned manifest.
        return
    except OSError as exc:
        raise HPKEvolvingSnapshotError(
            f"snapshot publication marker is unavailable: {exc}"
        ) from exc
    pending = _read_canonical_control_file(
        pending_path, label="snapshot publication marker"
    )
    expected_pending = publication_pending_payload(
        snapshot_id=manifest.snapshot_id,
        manifest_sha256=manifest_sha256,
        parent=manifest["parent"],
    )
    if pending.get("schema") == "tcm/afk/snapshot_publication_pending/v1":
        expected_pending["schema"] = pending["schema"]
    if pending != expected_pending:
        _fail("snapshot publication marker does not match the manifest")

    output_root = root.parent
    advance_path = (
        output_root / ADVANCE_DIRNAME / parent_advance_filename(manifest["parent"])
    )
    try:
        advance = _read_canonical_control_file(
            advance_path, label="snapshot parent advance"
        )
    except HPKEvolvingSnapshotError as exc:
        raise HPKEvolvingSnapshotError(
            "snapshot child is an uncommitted orphan: " + str(exc)
        ) from exc
    expected_advance = snapshot_advance_payload(
        parent=manifest["parent"],
        child_snapshot_id=manifest.snapshot_id,
        child_manifest_sha256=manifest_sha256,
    )
    if advance.get("schema") == "tcm/afk/snapshot_advance/v1":
        expected_advance["schema"] = advance["schema"]
        expected_advance["advance_id"] = stable_content_id(
            "afkadvance", {key: value for key, value in expected_advance.items() if key != "advance_id"}
        )
    if advance != expected_advance:
        _fail("snapshot child is not the committed advance for its parent")


def load_evolving_snapshot(
    manifest_path: str | os.PathLike[str], expected_manifest_sha256: str
) -> LoadedEvolvingSnapshot:
    raw_path = Path(manifest_path)
    if not raw_path.is_absolute():
        _fail("evolving snapshot manifest path must be explicit and absolute")
    expected = _sha(expected_manifest_sha256, path="expected_manifest_sha256")
    manifest_bytes = _read_regular(
        raw_path, label="evolving manifest", maximum=MAX_MANIFEST_BYTES
    )
    if _sha256(manifest_bytes) != expected:
        _fail("evolving snapshot manifest SHA-256 mismatch")
    if (
        not manifest_bytes.endswith(b"\n")
        or manifest_bytes.endswith(b"\n\n")
        or b"\r" in manifest_bytes
    ):
        _fail("evolving manifest must end in exactly one LF")
    manifest_payload = _parse_object(manifest_bytes[:-1], path="evolving manifest")
    if canonical_json_bytes(manifest_payload) + b"\n" != manifest_bytes:
        _fail("evolving manifest is not canonical JSON plus LF")
    manifest = ManifestV2.from_dict(manifest_payload)
    _require_committed_publication(
        manifest_path=raw_path,
        manifest_sha256=expected,
        manifest=manifest,
    )
    root = raw_path.resolve(strict=True).parent
    parsed: dict[str, Any] = {}
    member_bytes_by_name: dict[str, bytes] = {}
    for name, (filename, _media) in MEMBER_SPECS.items():
        member_path = root / filename
        if member_path.resolve(strict=True).parent != root:
            _fail(f"{name} member escapes the snapshot directory")
        member_bytes = _read_regular(member_path, label=name, maximum=MAX_MEMBER_BYTES)
        member_bytes_by_name[name] = member_bytes
        descriptor = manifest["members"][name]
        if descriptor["sha256"] != _sha256(member_bytes):
            _fail(f"{name} member SHA-256 mismatch")
        if descriptor["size_bytes"] != len(member_bytes):
            _fail(f"{name} member size mismatch")
        records = (
            _parse_entries(member_bytes)
            if name == "entries"
            else _parse_jsonl(member_bytes, kind=name)
        )
        if descriptor["record_count"] != len(records):
            _fail(f"{name} member record count mismatch")
        parsed[name] = records

    entries: tuple[EntryV1, ...] = parsed["entries"]
    _validate_snapshot_member_bindings(parsed, manifest)
    if manifest["parent"]["snapshot_id"] is None and any(
        parsed[name] for name in MEMBER_SPECS
    ):
        _fail("SnapshotV2 root K0 must have empty entries and history members")
    lifecycle = entry_lifecycle_sets(entries, manifest["entry_ids"]["revalidation"])
    if manifest["entry_ids"] != lifecycle:
        _fail("manifest lifecycle entry sets do not match entries member")
    flags = information_access_flags(
        entries,
        parsed["evidence"],
        promotion_policy_ref=manifest["policy_refs"]["promotion"],
    )
    if normalize_knowledge_metadata(manifest["information_access_flags"]) != flags:
        _fail("manifest information access flags do not match entries")
    counts = {
        "entry_count": len(entries),
        "accepted_count": len(lifecycle["accepted"]),
        "candidate_count": len(lifecycle["candidate"]),
        "revalidation_count": len(lifecycle["revalidation"]),
        "deprecated_count": len(lifecycle["deprecated"]),
        "evidence_count": len(parsed["evidence"]),
        "private_provenance_count": len(parsed["private_provenance"]),
        "update_decision_count": len(parsed["update_decisions"]),
        "promotion_decision_count": len(parsed["promotion_decisions"]),
        "rejected_evidence_count": len(parsed["rejected_evidence"]),
    }
    if manifest["counts"] != counts:
        _fail("manifest counts do not match snapshot members")
    parent_ref = manifest["parent"]
    if parent_ref["snapshot_id"] is not None:
        parent_directory = root.parent / str(parent_ref["snapshot_id"])
        try:
            parent_metadata = parent_directory.lstat()
        except OSError as exc:
            raise HPKEvolvingSnapshotError(
                f"declared parent snapshot is unavailable: {exc}"
            ) from exc
        if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
            parent_metadata.st_mode
        ):
            _fail("declared parent snapshot must be a non-symlink directory")
        parent_manifest_path = parent_directory / "snapshot_manifest.json"
        loaded_parent = load_evolving_snapshot(
            parent_manifest_path.absolute(),
            str(parent_ref["manifest_sha256"]),
        )
        if loaded_parent.snapshot_id != parent_ref["snapshot_id"]:
            _fail("declared parent snapshot identity mismatch")
        _validate_parent_child_lineage(
            parent=loaded_parent,
            child_manifest=manifest,
            child_parsed=parsed,
        )
    accepted_ids = set(lifecycle["accepted"])
    accepted = tuple(item for item in entries if item["entry_id"] in accepted_ids)
    if (
        _read_regular(raw_path, label="evolving manifest", maximum=MAX_MANIFEST_BYTES)
        != manifest_bytes
    ):
        _fail("evolving manifest changed during snapshot validation")
    for name, (filename, _media) in MEMBER_SPECS.items():
        if (
            _read_regular(root / filename, label=name, maximum=MAX_MEMBER_BYTES)
            != member_bytes_by_name[name]
        ):
            _fail(f"{name} member changed during snapshot validation")
    return LoadedEvolvingSnapshot(
        manifest_path=raw_path.resolve(strict=True),
        manifest_sha256=expected,
        manifest=manifest,
        all_entries=entries,
        accepted_entries=accepted,
        evidence_records=parsed["evidence"],
        private_provenance_records=parsed["private_provenance"],
        update_decision_records=parsed["update_decisions"],
        promotion_decision_records=parsed["promotion_decisions"],
        rejected_evidence_records=parsed["rejected_evidence"],
    )


__all__ = [
    "ADVANCE_DIRNAME",
    "HPKEvolvingSnapshotError",
    "CHILD_CLAIM_DIRNAME",
    "DEV_PROMOTION_POLICY_CONFIG_SHA256",
    "DEV_PROMOTION_POLICY_ID",
    "EvolvingSnapshotManifestV2",
    "EvolvingSnapshotState",
    "EVOLVING_GEOMETRY_POLICY_CONFIG_SHA256",
    "EVOLVING_GEOMETRY_POLICY_ID",
    "SAFE_EXPLORATION_GEOMETRY_POLICY_CONFIG_SHA256",
    "SAFE_EXPLORATION_GEOMETRY_POLICY_ID",
    "FORMAL_PROMOTION_POLICY_CONFIG_SHA256",
    "FORMAL_PROMOTION_POLICY_ID",
    "LoadedEvolvingSnapshot",
    "MEMBER_SPECS",
    "ManifestV2",
    "PUBLICATION_PENDING_FILENAME",
    "SNAPSHOT_V2_SCHEMA",
    "canonical_entries_jsonl",
    "canonical_jsonl_records",
    "child_claim_payload",
    "entry_lifecycle_sets",
    "information_access_flags",
    "load_evolving_snapshot",
    "parent_advance_filename",
    "publication_pending_payload",
    "promotion_policy_formal_eligibility",
    "promotion_policy_from_ref",
    "snapshot_advance_payload",
    "snapshot_v2_id_for",
    "validate_evolving_snapshot_lineage",
]
