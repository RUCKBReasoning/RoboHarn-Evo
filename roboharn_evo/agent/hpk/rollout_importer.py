"""Explicit, fail-closed importer for Agent rollout action evidence.

The importer never searches a result directory and never selects a "latest"
trace.  A caller always supplies the manifest and its public trace explicitly.
Manifest v2 additionally pins one owner-only HPK sidecar by path and hash.  The
sidecar accepts only a frozen event whitelist, and only its native transition
events are consumed.  Legacy v1 RMBench traces remain supported conservatively,
but their missing action nonce, typed effects, and private before/after state
force an unverified verdict.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.action_transition import (
    ATTRIBUTION_REASON_ORDER,
    PUBLIC_ACTION_EFFECT_TRANSITION_SCHEMA,
    ActionEffectTransitionV1,
    build_action_effect_transition,
    public_transition_projection,
)
from roboharn_evo.agent.hpk.audit import PrivateRankingAuditV1
from roboharn_evo.agent.hpk.candidate_features import candidate_semantic_features
from roboharn_evo.agent.hpk.compatibility import normalize_hpk_event_name
from roboharn_evo.agent.hpk.policy_config import (
    EVOLVING_GEOMETRY_POLICY_IDENTITIES,
    load_evolving_geometry_policy,
    load_safe_exploration_geometry_policy,
)
from roboharn_evo.agent.hpk.runtime_binding import validate_runtime_binding
from roboharn_evo.agent.hpk.safe_exploration import (
    SAFE_EXPLORATION_PRIVATE_EVENT,
    SAFE_EXPLORATION_PUBLIC_EVENT,
    validate_safe_exploration_private_audit,
    validate_safe_exploration_public_audit,
)
from roboharn_evo.agent.hpk.schemas import (
    AbstractEffectV1,
    HPKValidationError,
    AuditV1,
    CandidateGeometryFeaturesV1,
    canonical_json_bytes,
    reject_private_transferable,
    stable_content_id,
    validate_content_id,
)
from roboharn_evo.agent.hpk.semantic_knowledge import SEMANTIC_USAGE_SCHEMA
from roboharn_evo.agent.recovery.tool_specs import RecoveryToolCall

ROLLOUT_IMPORT_MANIFEST_V1_SCHEMA = "roboharn_evo/hpk/rollout_import_manifest/v1"
ROLLOUT_IMPORT_MANIFEST_V2_SCHEMA = "roboharn_evo/hpk/rollout_import_manifest/v2"
# Compatibility name retained for existing v1 offline-replay producers.
ROLLOUT_IMPORT_MANIFEST_SCHEMA = ROLLOUT_IMPORT_MANIFEST_V1_SCHEMA
ACTION_EFFECT_TRANSITION_EVENT = "action_effect_transition"
PUBLIC_TRACE_EVENT_WHITELIST = frozenset(
    {
        "hpk_action_effect_transition_public",
        "hpk_geometry_usage_binding",
        "hpk_motion_realization",
        "hpk_public_usage_audit",
        SAFE_EXPLORATION_PUBLIC_EVENT,
        "episode_end",
        "episode_start",
    }
)
PRIVATE_SIDECAR_EVENT_WHITELIST = frozenset(
    {
        "hpk_action_attempt_prepared_private",
        "hpk_action_tool_boundary_private",
        "hpk_action_tool_result_bundle_private",
        "hpk_private_ranking_audit",
        SAFE_EXPLORATION_PRIVATE_EVENT,
        ACTION_EFFECT_TRANSITION_EVENT,
    }
)
_PRIVATE_SIDECAR_ENVELOPE_KEYS = frozenset(
    {"event", "timestamp", "episode_id", "seed", "env_step"}
)
_PUBLIC_TRACE_ENVELOPE_KEYS = frozenset(
    {"event", "timestamp", "episode_id", "seed", "env_step"}
)

_PUBLIC_TRANSITION_KEYS = frozenset(
    {
        "schema",
        "public_transition_id",
        "transition_id",
        "episode_id",
        "action_attempt_nonce_sha256",
        "env_step_before",
        "env_step_after",
        "condition_id",
        "task_strategy_id",
        "retrieved_hpk_entry_ids",
        "selected_hpk_entry_id",
        "geometric_strategy_id",
        "operation",
        "arm",
        "target_role",
        "target_relation",
        "candidate_geometry_features",
        "geometric_compliance",
        "tool_call_ref_sha256",
        "tool_result_ref_sha256",
        "physical_action_executed",
        "motion_status",
        "realization_status",
        "pre_effect_state_sha256",
        "post_effect_state_sha256",
        "effect_observation_scope",
        "target_identity_status",
        "expected_effect",
        "observed_effect",
        "verifier_sources",
        "verifier_conflicts",
        "infrastructure_valid",
        "oracle_derived",
        "expert_derived",
        "evidence_verdict",
        "attribution_reasons",
    }
)
_GEOMETRY_BINDING_KEYS = frozenset(
    {
        "snapshot_id",
        "snapshot_manifest_sha256",
        "selected_entry_id",
        "condition_id",
        "task_strategy_id",
        "selected_geometric_strategy_id",
        "public_usage_audit_id",
        "private_ranking_audit_id",
        "geometry_policy_id",
        "geometry_policy_sha256",
        "strategy_realization_status",
        "motion_status",
        "source_disclosure",
    }
)
_MOTION_REALIZATION_KEYS = frozenset(
    {
        "schema",
        "snapshot_id",
        "snapshot_manifest_sha256",
        "selected_hpk_entry_id",
        "condition_id",
        "task_strategy_id",
        "geometric_strategy_id",
        "public_usage_audit_id",
        "private_ranking_audit_id",
        "strategy_realization_status",
        "motion_status",
        "effect_verified",
        "effect_type",
        "subtask_status",
    }
)

_PHYSICAL_RECOVERY_TOOLS = frozenset(
    {
        "close_gripper",
        "contact_displace",
        "lift_ee",
        "move_ee_to_grounded_instance",
        "move_ee_to_pose",
        "move_to_home",
        "open_gripper",
        "retreat_arm",
        "safe_reset_posture",
    }
)
_OPERATION_EFFECT_TERMINALS = {
    "grasp": "close_gripper",
    "place": "open_gripper",
    "contact": "contact_displace",
}
_VALID_EPISODE_LABELS = frozenset({"valid_success", "valid_task_failure"})
_INVALID_EPISODE_LABELS = frozenset(
    {"infrastructure_invalid", "incomplete_or_corrupt", "user_interrupted"}
)


class RolloutImportError(ValueError):
    """Raised when an explicitly supplied rollout source is invalid."""


@dataclass(frozen=True, slots=True)
class EpisodeClassification:
    episode_id: str | int
    label: str
    infrastructure_valid: bool
    natural_episode_end: bool
    terminal_task_outcome: str
    reason: str


@dataclass(frozen=True, slots=True)
class ImportAbstention:
    code: str
    event_ref: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "event_ref": self.event_ref,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RolloutImportResult:
    rollout_id: str
    manifest_path: Path
    trace_path: Path
    trace_sha256: str
    episode: EpisodeClassification
    transitions: tuple[ActionEffectTransitionV1, ...]
    abstentions: tuple[ImportAbstention, ...]
    manifest_schema: str = ROLLOUT_IMPORT_MANIFEST_V1_SCHEMA
    private_transition_trace_path: Path | None = None
    private_transition_trace_sha256: str | None = None
    transition_source_sha256: str = ""
    runtime_binding: dict[str, Any] | None = None
    ranked_usage_transition_links: tuple[tuple[str, str], ...] = ()
    rollout_manifest_sha256: str = ""

    def __post_init__(self) -> None:
        # Preserve the old positional construction API while giving legacy v1
        # results the same source identity they had before split traces existed.
        if not self.transition_source_sha256:
            object.__setattr__(self, "transition_source_sha256", self.trace_sha256)

    @property
    def public_trace_path(self) -> Path:
        """The explicitly supplied public trace (the v1 trace for legacy input)."""

        return self.trace_path

    @property
    def public_trace_sha256(self) -> str:
        """The public trace hash; ``trace_sha256`` remains its compatibility alias."""

        return self.trace_sha256

    @property
    def snapshot_update_allowed(self) -> bool:
        return bool(
            self.episode.infrastructure_valid
            and any(item.update_eligible for item in self.transitions)
        )

    @property
    def evidence_verdicts(self) -> tuple[str, ...]:
        return tuple(item.evidence_verdict for item in self.transitions)

    def public_summary(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "manifest_schema": self.manifest_schema,
            "trace_sha256": self.trace_sha256,
            "rollout_manifest_sha256": self.rollout_manifest_sha256,
            "private_transition_trace_sha256": (self.private_transition_trace_sha256),
            "transition_source_sha256": self.transition_source_sha256,
            "runtime_binding_id": (
                None
                if self.runtime_binding is None
                else self.runtime_binding["binding_id"]
            ),
            "ranked_usage_transition_links": [
                {"usage_audit_id": audit_id, "transition_id": transition_id}
                for audit_id, transition_id in self.ranked_usage_transition_links
            ],
            "episode_id": self.episode.episode_id,
            "episode_validity": self.episode.label,
            "infrastructure_valid": self.episode.infrastructure_valid,
            "transition_ids": [item.stable_id for item in self.transitions],
            "evidence_verdicts": list(self.evidence_verdicts),
            "snapshot_update_allowed": self.snapshot_update_allowed,
            "abstentions": [item.to_dict() for item in self.abstentions],
        }


@dataclass(frozen=True, slots=True)
class _TraceEvent:
    ordinal: int
    payload: dict[str, Any]
    event_ref: str


def _fail(message: str) -> None:
    raise RolloutImportError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _expected_source_sha256(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_explicit_regular_file(path: Path, *, label: str) -> bytes:
    absolute = path.expanduser().absolute()
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise RolloutImportError(f"{label} is unavailable: {absolute}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        _fail(f"{label} must not be a symlink: {absolute}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a regular file: {absolute}")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise RolloutImportError(f"could not open {label}: {absolute}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _decode_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RolloutImportError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} must contain one JSON object")
    try:
        canonical_json_bytes(value)
    except HPKValidationError as exc:
        raise RolloutImportError(f"invalid {label}: {exc}") from exc
    return value


def _validate_source_descriptor(value: Any, *, path: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        _fail(f"{path} must contain exactly path and sha256")
    declared_path = value["path"]
    if not isinstance(declared_path, str) or not declared_path.strip():
        _fail(f"{path}.path must be a non-empty string")
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(item not in "0123456789abcdef" for item in digest)
    ):
        _fail(f"{path}.sha256 must be a lowercase SHA-256 digest")
    return {"path": declared_path.strip(), "sha256": digest}


def _validate_information_access(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict) or set(value) != {
        "oracle_derived",
        "expert_derived",
    }:
        _fail(
            "manifest.information_access must contain exactly "
            "oracle_derived and expert_derived"
        )
    if not all(isinstance(value[key], bool) for key in value):
        _fail("manifest information-access flags must be boolean")
    return {
        "oracle_derived": value["oracle_derived"],
        "expert_derived": value["expert_derived"],
    }


def _validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = copy.deepcopy(dict(value))
    legacy_schemas = {
        "tcm/afk/rollout_import_manifest/v1": ROLLOUT_IMPORT_MANIFEST_V1_SCHEMA,
        "tcm/afk/rollout_import_manifest/v2": ROLLOUT_IMPORT_MANIFEST_V2_SCHEMA,
    }
    if manifest.get("schema") in legacy_schemas:
        manifest["schema"] = legacy_schemas[manifest["schema"]]
    schema = manifest.get("schema")
    if schema == ROLLOUT_IMPORT_MANIFEST_V1_SCHEMA:
        required = {
            "schema",
            "rollout_id",
            "episode_id",
            "trace",
            "information_access",
        }
    elif schema == ROLLOUT_IMPORT_MANIFEST_V2_SCHEMA:
        required = {
            "schema",
            "rollout_id",
            "episode_id",
            "public_trace",
            "private_transition_trace",
            "runtime_binding",
            "information_access",
        }
    else:
        _fail(f"unsupported rollout import manifest schema: {manifest.get('schema')!r}")
    if set(manifest) != required:
        _fail(
            "rollout import manifest field mismatch; "
            f"missing={sorted(required - set(manifest))}, "
            f"unknown={sorted(set(manifest) - required)}"
        )
    if (
        not isinstance(manifest["rollout_id"], str)
        or not manifest["rollout_id"].strip()
    ):
        _fail("manifest.rollout_id must be a non-empty string")
    episode_id = manifest["episode_id"]
    if not (
        isinstance(episode_id, str)
        and episode_id.strip()
        or isinstance(episode_id, int)
        and not isinstance(episode_id, bool)
        and episode_id >= 0
    ):
        _fail("manifest.episode_id must be a non-empty string or non-negative integer")
    if schema == ROLLOUT_IMPORT_MANIFEST_V1_SCHEMA:
        manifest["trace"] = _validate_source_descriptor(
            manifest["trace"], path="manifest.trace"
        )
    else:
        manifest["public_trace"] = _validate_source_descriptor(
            manifest["public_trace"], path="manifest.public_trace"
        )
        manifest["private_transition_trace"] = _validate_source_descriptor(
            manifest["private_transition_trace"],
            path="manifest.private_transition_trace",
        )
        try:
            manifest["runtime_binding"] = validate_runtime_binding(
                manifest["runtime_binding"]
            )
        except Exception as exc:
            raise RolloutImportError(
                f"manifest.runtime_binding is invalid: {exc}"
            ) from exc
    manifest["information_access"] = _validate_information_access(
        manifest["information_access"]
    )
    return manifest


def _resolve_declared_trace(
    manifest_path: Path,
    descriptor: Mapping[str, Any],
) -> Path:
    declared = Path(str(descriptor["path"]))
    if not declared.is_absolute():
        declared = manifest_path.parent / declared
    return declared.absolute()


def _parse_jsonl(
    raw: bytes,
    *,
    trace_sha256: str,
    label: str = "trace",
    allow_empty: bool = False,
) -> tuple[_TraceEvent, ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RolloutImportError(f"{label} is not UTF-8: {exc}") from exc
    events: list[_TraceEvent] = []
    for ordinal, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            _fail(f"{label} line {ordinal} is blank")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RolloutImportError(
                f"invalid {label} JSON at line {ordinal}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            _fail(f"{label} line {ordinal} must contain a JSON object")
        event = payload.get("event")
        if not isinstance(event, str) or not event.strip():
            _fail(f"{label} line {ordinal} has no event name")
        try:
            canonical_json_bytes(payload)
        except HPKValidationError as exc:
            raise RolloutImportError(
                f"invalid {label} value at line {ordinal}: {exc}"
            ) from exc
        event_ref = stable_content_id(
            "afkevent",
            {
                "trace_sha256": trace_sha256,
                "ordinal": ordinal,
                "event": payload,
            },
        )
        payload["event"] = normalize_hpk_event_name(event)
        if payload.get("schema") == "tcm/afk/semantic_usage/v1":
            payload["schema"] = SEMANTIC_USAGE_SCHEMA
        events.append(_TraceEvent(ordinal, payload, event_ref))
    if not events and not allow_empty:
        _fail(f"{label} contains no events")
    return tuple(events)


def _same_typed_episode_id(value: Any, expected: str | int) -> bool:
    return type(value) is type(expected) and value == expected


def _exact_public_fields(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    *,
    line: int,
    event: str,
) -> None:
    actual = set(payload)
    if actual != set(expected):
        _fail(
            f"manifest v2 public_trace {event} fields mismatch at line {line}: "
            f"missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )


def _public_sha(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{path} must be a lowercase SHA-256 digest or null")
    return value


def _public_content_id(value: Any, *, prefix: str, path: str) -> str:
    try:
        validate_content_id(value, prefix=prefix, path=path)
    except Exception as exc:
        raise RolloutImportError(str(exc)) from exc
    return str(value)


def _validate_public_source_disclosure(value: Any, *, path: str) -> None:
    if not isinstance(value, Mapping):
        _fail(f"{path} must be an object")
    expected = {
        "source_kind",
        "acceptance_scope",
        "expert_prior_used",
        "human_prior_used",
        "oracle_evidence_used",
        "learned_afk",
        "learned_hpk",
        "formal_evaluation_eligible",
    }
    if set(value) != expected:
        _fail(f"{path} fields mismatch")
    if value["source_kind"] not in {
        "agent_generated",
        "benchmark_expert",
        "human_authored",
    }:
        _fail(f"{path}.source_kind is unsupported")
    if value["acceptance_scope"] not in {
        "formal_no_prior",
        "expert_prior",
        "integration_only",
        "oracle_diagnostic",
    }:
        _fail(f"{path}.acceptance_scope is unsupported")
    if any(
        not isinstance(value[key], bool)
        for key in expected - {"source_kind", "acceptance_scope"}
    ):
        _fail(f"{path} disclosure flags must be booleans")


def _reject_public_leakage(payload: Mapping[str, Any], *, line: int) -> None:
    def privacy_projection(value: Any) -> Any:
        if isinstance(value, Mapping):
            projected: dict[str, Any] = {}
            for raw_key, child in value.items():
                key = str(raw_key)
                normalized = key.lower()
                # Event-specific parsers above validate every structural ID,
                # digest, and schema.  Remove those known-safe wire values
                # before the free-text leakage scan so versioned IDs such as
                # ``afk_geometry_policy/v1`` are not mistaken for local paths.
                if (
                    normalized == "schema"
                    or normalized.endswith("_id")
                    or normalized.endswith("_ids")
                    or normalized.endswith("_sha256")
                ):
                    continue
                projected[key] = privacy_projection(child)
            return projected
        if isinstance(value, list):
            return [privacy_projection(item) for item in value]
        return copy.deepcopy(value)

    try:
        reject_private_transferable(
            privacy_projection(payload), path="manifest_v2.public_event"
        )
    except Exception as exc:
        raise RolloutImportError(
            f"manifest v2 public_trace contains private data at line {line}: {exc}"
        ) from exc


def _validate_public_transition(
    payload: Mapping[str, Any], *, expected_episode_id: str | int, line: int
) -> None:
    _exact_public_fields(
        payload,
        _PUBLIC_TRANSITION_KEYS,
        line=line,
        event="hpk_action_effect_transition_public",
    )
    if payload["schema"] != PUBLIC_ACTION_EFFECT_TRANSITION_SCHEMA:
        _fail(f"public transition schema is unsupported at line {line}")
    if not _same_typed_episode_id(payload["episode_id"], expected_episode_id):
        _fail(f"public transition episode_id mismatch at line {line}")
    _public_content_id(
        payload["public_transition_id"],
        prefix="afkpubtrans",
        path="public_transition_id",
    )
    _public_content_id(
        payload["transition_id"], prefix="afktransition", path="transition_id"
    )
    for key, prefix in (
        ("condition_id", "afkc"),
        ("task_strategy_id", "afku"),
        ("selected_hpk_entry_id", "afkentry"),
        ("geometric_strategy_id", "afkz"),
    ):
        if payload[key] is not None:
            _public_content_id(payload[key], prefix=prefix, path=key)
    retrieved = payload["retrieved_hpk_entry_ids"]
    if not isinstance(retrieved, list) or len(retrieved) > 1:
        _fail(f"retrieved_hpk_entry_ids is invalid at line {line}")
    for entry_id in retrieved:
        _public_content_id(entry_id, prefix="afkentry", path="retrieved entry")
    selected = payload["selected_hpk_entry_id"]
    if (selected is None and retrieved) or (
        selected is not None and retrieved != [selected]
    ):
        _fail(f"public transition selected entry binding is invalid at line {line}")
    for key in (
        "action_attempt_nonce_sha256",
        "tool_call_ref_sha256",
        "tool_result_ref_sha256",
        "pre_effect_state_sha256",
        "post_effect_state_sha256",
    ):
        _public_sha(payload[key], path=key)
    for key in ("env_step_before", "env_step_after"):
        step = payload[key]
        if step is not None and (
            isinstance(step, bool) or not isinstance(step, int) or step < 0
        ):
            _fail(f"{key} is invalid at line {line}")
    if (
        payload["env_step_before"] is not None
        and payload["env_step_after"] is not None
        and payload["env_step_after"] < payload["env_step_before"]
    ):
        _fail(f"public transition step order is invalid at line {line}")
    if payload["candidate_geometry_features"] is not None:
        try:
            CandidateGeometryFeaturesV1.from_dict(
                payload["candidate_geometry_features"]
            )
        except Exception as exc:
            raise RolloutImportError(
                f"public transition candidate features are invalid at line {line}: {exc}"
            ) from exc
    for key in ("expected_effect", "observed_effect"):
        if payload[key] is not None:
            try:
                AbstractEffectV1.from_dict(payload[key])
            except Exception as exc:
                raise RolloutImportError(
                    f"public transition {key} is invalid at line {line}: {exc}"
                ) from exc
    if payload["operation"] not in {None, "contact", "grasp", "place"}:
        _fail(f"public transition operation is invalid at line {line}")
    if payload["arm"] not in {None, "left", "right"}:
        _fail(f"public transition arm is invalid at line {line}")
    if payload["geometric_compliance"] not in {True, False, "unverified"}:
        _fail(f"public transition compliance is invalid at line {line}")
    for key in (
        "physical_action_executed",
        "infrastructure_valid",
        "oracle_derived",
        "expert_derived",
    ):
        if not isinstance(payload[key], bool):
            _fail(f"public transition {key} must be boolean at line {line}")
    reasons = payload["attribution_reasons"]
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) for reason in reasons)
        or len(reasons) != len(set(reasons))
    ):
        _fail(f"public transition attribution_reasons is invalid at line {line}")
    frozen_reasons = [
        reason for reason in ATTRIBUTION_REASON_ORDER if reason in set(reasons)
    ]
    if reasons != frozen_reasons:
        _fail(
            f"public transition attribution_reasons does not use the frozen "
            f"order at line {line}"
        )
    identity = dict(payload)
    public_id = identity.pop("public_transition_id")
    if public_id != stable_content_id("afkpubtrans", identity):
        _fail(f"public_transition_id content mismatch at line {line}")
    privacy_projection = dict(payload)
    privacy_projection.pop("episode_id")
    # The exact frozen enum was validated above.  Its task-neutral symbolic
    # names include words such as ``candidate`` and ``geometry`` that the
    # free-text leakage detector intentionally rejects, so do not rescan them
    # as untrusted prose.
    privacy_projection.pop("attribution_reasons")
    _reject_public_leakage(privacy_projection, line=line)


def _validate_public_hpk_event(
    item: _TraceEvent,
    *,
    expected_episode_id: str | int,
    expected_runtime_binding: Mapping[str, Any],
) -> None:
    event = str(item.payload["event"])
    envelope = set(_PUBLIC_TRACE_ENVELOPE_KEYS)
    payload = {
        key: copy.deepcopy(value)
        for key, value in item.payload.items()
        if key not in envelope
    }
    if event == "hpk_public_usage_audit":
        if payload.get("schema") == SEMANTIC_USAGE_SCHEMA:
            expected_payload = {
                "schema",
                "rendered",
                "knowledge_count",
                "usage",
            }
            if (
                set(payload) != expected_payload
                or payload.get("rendered") is not True
                or not isinstance(payload.get("knowledge_count"), int)
                or payload.get("knowledge_count", 0) < 1
                or payload.get("usage")
                != "injected; behavioral effect unverified"
            ):
                _fail(
                    f"manifest v2 semantic usage audit is invalid at line "
                    f"{item.ordinal}"
                )
            _exact_public_fields(
                item.payload,
                _PUBLIC_TRACE_ENVELOPE_KEYS | expected_payload,
                line=item.ordinal,
                event=event,
            )
            return
        try:
            typed_audit = AuditV1.from_dict(payload)
        except Exception as exc:
            raise RolloutImportError(
                f"manifest v2 public usage audit is invalid at line "
                f"{item.ordinal}: {exc}"
            ) from exc
        expected = envelope | set(typed_audit.to_dict())
        _exact_public_fields(
            item.payload, frozenset(expected), line=item.ordinal, event=event
        )
        _reject_public_leakage(payload, line=item.ordinal)
        return
    if event == SAFE_EXPLORATION_PUBLIC_EVENT:
        try:
            validate_safe_exploration_public_audit(
                payload,
                expected_runtime_binding=expected_runtime_binding,
            )
        except Exception as exc:
            raise RolloutImportError(
                f"manifest v2 safe exploration public audit is invalid at "
                f"line {item.ordinal}: {exc}"
            ) from exc
        _exact_public_fields(
            item.payload,
            _PUBLIC_TRACE_ENVELOPE_KEYS | set(payload),
            line=item.ordinal,
            event=event,
        )
        _reject_public_leakage(payload, line=item.ordinal)
        return
    if event == "hpk_geometry_usage_binding":
        _exact_public_fields(
            payload,
            _GEOMETRY_BINDING_KEYS,
            line=item.ordinal,
            event=event,
        )
        for key, prefix in (
            ("snapshot_id", "afksnap"),
            ("selected_entry_id", "afkentry"),
            ("condition_id", "afkc"),
            ("task_strategy_id", "afku"),
            ("selected_geometric_strategy_id", "afkz"),
            ("public_usage_audit_id", "afkaudit"),
            ("private_ranking_audit_id", "afkprivrank"),
        ):
            _public_content_id(payload[key], prefix=prefix, path=key)
        _public_sha(
            payload["snapshot_manifest_sha256"], path="snapshot_manifest_sha256"
        )
        _public_sha(payload["geometry_policy_sha256"], path="geometry_policy_sha256")
        expected_geometry = expected_runtime_binding["policy_refs"]["geometry"]
        if (
            payload["geometry_policy_id"] != expected_geometry["policy_id"]
            or payload["geometry_policy_sha256"] != expected_geometry["config_sha256"]
        ):
            _fail(
                f"geometry policy identity differs from runtime binding at line "
                f"{item.ordinal}"
            )
        if (
            payload["strategy_realization_status"]
            not in {
                "satisfied",
                "violated",
                "unknown",
            }
            or payload["motion_status"] != "pending"
        ):
            _fail(f"geometry binding status is invalid at line {item.ordinal}")
        _validate_public_source_disclosure(
            payload["source_disclosure"], path="geometry source_disclosure"
        )
        privacy_payload = dict(payload)
        privacy_payload.pop("geometry_policy_id")
        _reject_public_leakage(privacy_payload, line=item.ordinal)
        return
    if event == "hpk_action_effect_transition_public":
        transition_payload = dict(payload)
        transition_payload["episode_id"] = expected_episode_id
        _validate_public_transition(
            transition_payload,
            expected_episode_id=expected_episode_id,
            line=item.ordinal,
        )
        if item.payload["env_step"] != transition_payload["env_step_after"]:
            _fail(
                f"public transition envelope env_step mismatch at line {item.ordinal}"
            )
        return
    if event == "hpk_motion_realization":
        _exact_public_fields(
            payload,
            _MOTION_REALIZATION_KEYS,
            line=item.ordinal,
            event=event,
        )
        for key, prefix in (
            ("snapshot_id", "afksnap"),
            ("selected_hpk_entry_id", "afkentry"),
            ("condition_id", "afkc"),
            ("task_strategy_id", "afku"),
            ("geometric_strategy_id", "afkz"),
            ("public_usage_audit_id", "afkaudit"),
            ("private_ranking_audit_id", "afkprivrank"),
        ):
            _public_content_id(payload[key], prefix=prefix, path=key)
        _public_sha(
            payload["snapshot_manifest_sha256"], path="snapshot_manifest_sha256"
        )
        if payload["motion_status"] not in {
            "completed",
            "failed_before_effect",
            "unknown",
        } or payload["strategy_realization_status"] not in {
            "satisfied",
            "violated",
            "unknown",
        }:
            _fail(f"motion realization status is invalid at line {item.ordinal}")
        if payload["effect_verified"] not in {"true", "false", "unverified"}:
            _fail(f"effect verification is invalid at line {item.ordinal}")
        _reject_public_leakage(payload, line=item.ordinal)


def _validate_v2_public_events(
    events: Sequence[_TraceEvent],
    *,
    expected_episode_id: str | int,
    expected_runtime_binding: Mapping[str, Any],
) -> None:
    for item in events:
        event = item.payload.get("event")
        if event not in PUBLIC_TRACE_EVENT_WHITELIST:
            _fail(
                "manifest v2 public_trace contains an event outside the frozen "
                f"candidate-free whitelist at line {item.ordinal}: {event!r}"
            )
        missing = _PUBLIC_TRACE_ENVELOPE_KEYS - set(item.payload)
        if missing:
            _fail(
                "manifest v2 public_trace envelope is missing "
                f"{sorted(missing)} at line {item.ordinal}"
            )
        if not _same_typed_episode_id(
            item.payload.get("episode_id"), expected_episode_id
        ):
            _fail(
                "manifest v2 public_trace episode_id must exactly match the "
                f"manifest at line {item.ordinal}"
            )
        timestamp = item.payload.get("timestamp")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
        ):
            _fail(
                "manifest v2 public_trace timestamp must be a finite number "
                f"at line {item.ordinal}"
            )
        seed = item.payload.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < -1:
            _fail(
                "manifest v2 public_trace seed must be an integer >= -1 "
                f"at line {item.ordinal}"
            )
        env_step = item.payload.get("env_step")
        if isinstance(env_step, bool) or not isinstance(env_step, int) or env_step < -1:
            _fail(
                "manifest v2 public_trace env_step must be an integer >= -1 "
                f"at line {item.ordinal}"
            )
        if event == "episode_start":
            _exact_public_fields(
                item.payload,
                _PUBLIC_TRACE_ENVELOPE_KEYS | {"runtime_binding"},
                line=item.ordinal,
                event=event,
            )
            try:
                actual_binding = validate_runtime_binding(
                    item.payload["runtime_binding"]
                )
            except Exception as exc:
                raise RolloutImportError(
                    f"manifest v2 episode_start runtime binding is invalid at "
                    f"line {item.ordinal}: {exc}"
                ) from exc
            if canonical_json_bytes(actual_binding) != canonical_json_bytes(
                expected_runtime_binding
            ):
                _fail("manifest v2 episode_start runtime binding differs from manifest")
        elif event == "episode_end":
            expected = _PUBLIC_TRACE_ENVELOPE_KEYS | {
                "result",
                "natural_episode_end",
                "episode_validity",
            }
            _exact_public_fields(
                item.payload,
                frozenset(expected),
                line=item.ordinal,
                event=event,
            )
            if (
                not isinstance(item.payload["result"], str)
                or not item.payload["result"].strip()
                or not isinstance(item.payload["natural_episode_end"], bool)
            ):
                _fail(
                    f"manifest v2 episode_end values are invalid at line {item.ordinal}"
                )
            validity = item.payload["episode_validity"]
            if not isinstance(validity, Mapping) or set(validity) != {
                "label",
                "reason",
            }:
                _fail(f"manifest v2 episode_validity is invalid at line {item.ordinal}")
            if validity["label"] not in _VALID_EPISODE_LABELS | _INVALID_EPISODE_LABELS:
                _fail(
                    f"manifest v2 episode validity label is invalid at line {item.ordinal}"
                )
            if not isinstance(validity["reason"], str):
                _fail(
                    f"manifest v2 episode validity reason is invalid at line {item.ordinal}"
                )
            _reject_public_leakage(
                {
                    "result": item.payload["result"],
                    "natural_episode_end": item.payload["natural_episode_end"],
                    "episode_validity": dict(validity),
                },
                line=item.ordinal,
            )
        else:
            _validate_public_hpk_event(
                item,
                expected_episode_id=expected_episode_id,
                expected_runtime_binding=expected_runtime_binding,
            )
    names = [str(item.payload["event"]) for item in events]
    if (
        not names
        or names[0] != "episode_start"
        or names[-1] != "episode_end"
        or names.count("episode_start") != 1
        or names.count("episode_end") != 1
    ):
        _fail(
            "manifest v2 public_trace requires one first episode_start and one "
            "final episode_end"
        )
    timestamps = [float(item.payload["timestamp"]) for item in events]
    if timestamps != sorted(timestamps):
        _fail("manifest v2 public_trace timestamps must be nondecreasing")
    steps = [int(item.payload["env_step"]) for item in events]
    if steps[0] != -1:
        _fail("manifest v2 public_trace episode_start env_step must equal -1")
    if any(step < 0 for step in steps[1:]):
        _fail("manifest v2 public_trace non-start env_step must be non-negative")
    if steps[1:] != sorted(steps[1:]):
        _fail("manifest v2 public_trace env_step values must be nondecreasing")


def _validate_v2_private_sidecar_events(
    events: Sequence[_TraceEvent], *, expected_episode_id: str | int
) -> None:
    for item in events:
        event = item.payload.get("event")
        if event not in PRIVATE_SIDECAR_EVENT_WHITELIST:
            _fail(
                "manifest v2 private_transition_trace contains an event outside "
                f"the frozen private whitelist at line {item.ordinal}: {event!r}"
            )
        missing = _PRIVATE_SIDECAR_ENVELOPE_KEYS - set(item.payload)
        if missing:
            _fail(
                "manifest v2 private_transition_trace envelope is missing "
                f"{sorted(missing)} at line {item.ordinal}"
            )
        if not _same_typed_episode_id(
            item.payload.get("episode_id"), expected_episode_id
        ):
            _fail(
                "manifest v2 private_transition_trace episode_id must exactly "
                f"match the manifest at line {item.ordinal}"
            )
        timestamp = item.payload.get("timestamp")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
        ):
            _fail(
                "manifest v2 private_transition_trace timestamp must be a finite "
                f"number at line {item.ordinal}"
            )
        seed = item.payload.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < -1:
            _fail(
                "manifest v2 private_transition_trace seed must be an integer "
                f">= -1 at line {item.ordinal}"
            )
        env_step = item.payload.get("env_step")
        if isinstance(env_step, bool) or not isinstance(env_step, int) or env_step < -1:
            _fail(
                "manifest v2 private_transition_trace env_step must be an integer "
                f">= -1 at line {item.ordinal}"
            )
        if event == SAFE_EXPLORATION_PRIVATE_EVENT:
            payload = {
                key: copy.deepcopy(value)
                for key, value in item.payload.items()
                if key not in _PRIVATE_SIDECAR_ENVELOPE_KEYS
            }
            try:
                validate_safe_exploration_private_audit(payload)
            except Exception as exc:
                raise RolloutImportError(
                    f"manifest v2 safe exploration private audit is invalid at "
                    f"line {item.ordinal}: {exc}"
                ) from exc
        elif event == "hpk_private_ranking_audit":
            payload = {
                key: copy.deepcopy(value)
                for key, value in item.payload.items()
                if key not in _PRIVATE_SIDECAR_ENVELOPE_KEYS
            }
            try:
                PrivateRankingAuditV1.from_dict(payload)
            except Exception as exc:
                raise RolloutImportError(
                    f"manifest v2 private ranking audit is invalid at line "
                    f"{item.ordinal}: {exc}"
                ) from exc


def _validate_v2_safe_exploration_audits(
    public_events: Sequence[_TraceEvent],
    private_events: Sequence[_TraceEvent],
    *,
    expected_runtime_binding: Mapping[str, Any],
) -> None:
    """Pair candidate-free public receipts with owner-only selected ranks."""

    policy = load_safe_exploration_geometry_policy()
    profile = policy.payload["safe_exploration"]
    public_by_id: dict[str, dict[str, Any]] = {}
    selection_counts: dict[str, int] = {}
    accepted_conditions: set[str] = set()
    for item in public_events:
        if item.payload.get("event") == "hpk_geometry_usage_binding":
            condition_id = item.payload.get("condition_id")
            if isinstance(condition_id, str):
                accepted_conditions.add(condition_id)
            continue
        if item.payload.get("event") != SAFE_EXPLORATION_PUBLIC_EVENT:
            continue
        payload = {
            key: copy.deepcopy(value)
            for key, value in item.payload.items()
            if key not in _PUBLIC_TRACE_ENVELOPE_KEYS
        }
        validated = validate_safe_exploration_public_audit(
            payload,
            expected_runtime_binding=expected_runtime_binding,
        )
        audit_id = str(validated["audit_id"])
        if audit_id in public_by_id:
            _fail("manifest v2 duplicates a safe exploration public audit")
        condition_id = str(validated["condition_id"])
        if (
            profile["disable_after_accepted_exact_match"] is True
            and condition_id in accepted_conditions
        ):
            _fail("safe exploration occurred after an accepted exact match")
        selection_counts[condition_id] = selection_counts.get(condition_id, 0) + 1
        if (
            selection_counts[condition_id]
            > profile["max_selections_per_condition_per_episode"]
        ):
            _fail("safe exploration exceeded the per-condition episode limit")
        public_by_id[audit_id] = validated

    private_by_public: dict[str, dict[str, Any]] = {}
    for item in private_events:
        if item.payload.get("event") != SAFE_EXPLORATION_PRIVATE_EVENT:
            continue
        payload = {
            key: copy.deepcopy(value)
            for key, value in item.payload.items()
            if key not in _PRIVATE_SIDECAR_ENVELOPE_KEYS
        }
        public_id = str(payload.get("public_audit_id", "") or "")
        public = public_by_id.get(public_id)
        if public is None:
            _fail("safe exploration private audit lacks its public projection")
        validated = validate_safe_exploration_private_audit(
            payload,
            expected_public_audit=public,
        )
        if public_id in private_by_public:
            _fail("manifest v2 duplicates a safe exploration private audit")
        private_by_public[public_id] = validated
    if set(public_by_id) != set(private_by_public):
        _fail("safe exploration public/private audit sets differ")


def _validate_v2_safe_exploration_attempt_links(
    private_events: Sequence[_TraceEvent],
    transitions: Sequence[ActionEffectTransitionV1],
) -> None:
    """Prove each explored private rank is the candidate in one native attempt."""

    private_by_id: dict[str, dict[str, Any]] = {}
    for item in private_events:
        if item.payload.get("event") != SAFE_EXPLORATION_PRIVATE_EVENT:
            continue
        payload = {
            key: copy.deepcopy(value)
            for key, value in item.payload.items()
            if key not in _PRIVATE_SIDECAR_ENVELOPE_KEYS
        }
        validated = validate_safe_exploration_private_audit(payload)
        private_by_id[str(validated["private_audit_id"])] = validated
    if not private_by_id:
        return

    links_by_private: dict[str, set[str]] = {}
    tokens_by_private: dict[str, set[str]] = {}
    guarded_calls_by_nonce: dict[str, list[dict[str, Any]]] = {}
    for item in private_events:
        if item.payload.get("event") != "hpk_action_attempt_prepared_private":
            continue
        binding = item.payload.get("binding")
        if not isinstance(binding, Mapping):
            continue
        public_id = binding.get("safe_exploration_public_audit_id")
        private_id = binding.get("safe_exploration_private_audit_id")
        if public_id is None and private_id is None:
            continue
        if not isinstance(private_id, str) or private_id not in private_by_id:
            _fail("prepared safe exploration binding has an unknown private audit")
        private = private_by_id[private_id]
        if public_id != private["public_audit_id"]:
            _fail("prepared safe exploration public/private audit IDs differ")
        nonce = str(item.payload.get("action_attempt_nonce", "") or "")
        if not nonce:
            _fail("prepared safe exploration binding lacks an attempt nonce")
        expected_effect = binding.get("expected_effect")
        try:
            effect_id = AbstractEffectV1.from_dict(expected_effect).stable_id
        except Exception as exc:
            raise RolloutImportError(
                f"prepared safe exploration expected effect is invalid: {exc}"
            ) from exc
        if (
            binding.get("condition_id") != private["condition_id"]
            or binding.get("task_strategy_id") != private["task_strategy_id"]
            or effect_id != private["expected_effect_id"]
            or binding.get("selected_candidate_private_ref")
            != private["selected_candidate_private_ref"]
        ):
            _fail("prepared safe exploration binding differs from its private audit")
        links_by_private.setdefault(private_id, set()).add(nonce)
        selection_token = str(binding.get("_hpk_selection_token", "") or "")
        try:
            validate_content_id(
                selection_token,
                prefix="afkselection",
                path="safe exploration selection token",
            )
        except Exception as exc:
            raise RolloutImportError(str(exc)) from exc
        tokens_by_private.setdefault(private_id, set()).add(selection_token)
        raw_calls = item.payload.get("guarded_calls")
        if not isinstance(raw_calls, list):
            _fail("safe exploration prepared attempt lacks guarded calls")
        guarded_calls_by_nonce.setdefault(nonce, []).extend(
            copy.deepcopy(dict(call)) for call in raw_calls if isinstance(call, Mapping)
        )
    if set(links_by_private) != set(private_by_id):
        _fail("safe exploration audit lacks its prepared native attempt")
    if any(len(tokens) != 1 for tokens in tokens_by_private.values()):
        _fail("safe exploration audit is reused by multiple selection tokens")

    transition_by_nonce = {
        str(item["action_attempt_nonce"] or ""): item for item in transitions
    }
    profile = load_safe_exploration_geometry_policy().payload
    max_step_gap = int(profile["semantic_attempt"]["max_pending_env_step_gap"])

    def has_marker(nonce: str, marker: str) -> bool:
        return any(
            isinstance(call.get("args"), Mapping) and call["args"].get(marker) is True
            for call in guarded_calls_by_nonce.get(nonce, [])
        )

    for private_id, nonces in links_by_private.items():
        private = private_by_id[private_id]
        linked_transitions: list[ActionEffectTransitionV1] = []
        for nonce in nonces:
            transition = transition_by_nonce.get(nonce)
            if transition is None:
                _fail("safe exploration prepared attempt lacks its native transition")
            linked_transitions.append(transition)
            if (
                transition["condition_id"] != private["condition_id"]
                or transition["task_strategy_id"] != private["task_strategy_id"]
                or transition["selected_candidate_private_ref"]
                != private["selected_candidate_private_ref"]
                or transition["expected_effect"] is None
                or AbstractEffectV1.from_dict(transition["expected_effect"]).stable_id
                != private["expected_effect_id"]
            ):
                _fail("safe exploration transition differs from its private audit")
        if (
            sum(
                item.evidence_verdict in {"support", "oppose"}
                for item in linked_transitions
            )
            > 1
        ):
            _fail("one safe exploration selection has multiple verified effects")
        ordered = sorted(
            linked_transitions,
            key=lambda item: (
                -1 if item["env_step_before"] is None else item["env_step_before"],
                item.stable_id,
            ),
        )
        if len(ordered) > 1:
            if ordered[0].evidence_verdict != "unverified":
                _fail("safe exploration cannot continue after a verified effect")
            for previous, current in zip(ordered, ordered[1:]):
                previous_after = previous["env_step_after"]
                current_before = current["env_step_before"]
                if (
                    previous_after is None
                    or current_before is None
                    or current_before < previous_after
                    or current_before - previous_after > max_step_gap
                ):
                    _fail("safe exploration continuation step gap is invalid")
            for continuation in ordered[1:-1]:
                nonce = str(continuation["action_attempt_nonce"])
                if (
                    continuation.evidence_verdict != "unverified"
                    or not has_marker(nonce, "_runtime_partial_approach_continuation")
                    or has_marker(nonce, "_runtime_occlusion_geometry_lease")
                ):
                    _fail("safe exploration pre-close continuation FSM is invalid")
            final = ordered[-1]
            final_nonce = str(final["action_attempt_nonce"])
            if final.evidence_verdict in {"support", "oppose"}:
                if not has_marker(final_nonce, "_runtime_occlusion_geometry_lease"):
                    _fail("safe exploration verified effect lacks its close lease")
            elif not (
                has_marker(final_nonce, "_runtime_partial_approach_continuation")
                or has_marker(final_nonce, "_runtime_occlusion_geometry_lease")
            ):
                _fail("safe exploration later unverified attempt is unmarked")


def _validate_v2_ranked_geometry_chains(
    public_events: Sequence[_TraceEvent],
    private_events: Sequence[_TraceEvent],
    transitions: Sequence[ActionEffectTransitionV1],
    *,
    expected_runtime_binding: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Bind a public behavior-change receipt to the action that used it."""

    expected_snapshot = expected_runtime_binding["snapshot_ref"]
    public_by_id: dict[str, AuditV1] = {}
    geometry_by_public: dict[str, dict[str, Any]] = {}
    motions_by_public: dict[str, list[dict[str, Any]]] = {}
    for item in public_events:
        event = item.payload.get("event")
        payload = {
            key: copy.deepcopy(value)
            for key, value in item.payload.items()
            if key not in _PUBLIC_TRACE_ENVELOPE_KEYS
        }
        if event == "hpk_public_usage_audit":
            if payload.get("schema") == SEMANTIC_USAGE_SCHEMA:
                continue
            audit = AuditV1.from_dict(payload)
            if (
                audit["retrieval_stage"] != "post_binding_geometry"
                or audit["selected_entry_id"] is None
            ):
                continue
            if (
                audit["snapshot_id"] != expected_snapshot["snapshot_id"]
                or audit["snapshot_manifest_sha256"]
                != expected_snapshot["manifest_sha256"]
            ):
                _fail("ranked HPK public audit differs from the runtime snapshot")
            if audit.stable_id in public_by_id:
                _fail("manifest v2 duplicates a ranked HPK public usage audit")
            public_by_id[audit.stable_id] = audit
        elif event == "hpk_geometry_usage_binding":
            public_id = str(payload.get("public_usage_audit_id", "") or "")
            if public_id in geometry_by_public:
                _fail("manifest v2 duplicates an HPK geometry usage binding")
            geometry_by_public[public_id] = payload
        elif event == "hpk_motion_realization":
            public_id = str(payload.get("public_usage_audit_id", "") or "")
            motions_by_public.setdefault(public_id, []).append(payload)

    private_by_public: dict[str, PrivateRankingAuditV1] = {}
    private_by_id: dict[str, PrivateRankingAuditV1] = {}
    for item in private_events:
        if item.payload.get("event") != "hpk_private_ranking_audit":
            continue
        payload = {
            key: copy.deepcopy(value)
            for key, value in item.payload.items()
            if key not in _PRIVATE_SIDECAR_ENVELOPE_KEYS
        }
        ranking = PrivateRankingAuditV1.from_dict(payload)
        public_id = str(ranking["public_usage_audit_id"])
        if public_id in private_by_public or ranking.stable_id in private_by_id:
            _fail("manifest v2 duplicates a private ranking audit")
        private_by_public[public_id] = ranking
        private_by_id[ranking.stable_id] = ranking

    if set(public_by_id) != set(private_by_public):
        _fail("ranked HPK public/private audit sets differ")
    if set(public_by_id) != set(geometry_by_public):
        _fail("ranked HPK usage audits and geometry bindings differ")
    if any(public_id not in public_by_id for public_id in motions_by_public):
        _fail("HPK motion realization references an unknown usage audit")

    for public_id, audit in public_by_id.items():
        ranking = private_by_public[public_id]
        geometry = geometry_by_public[public_id]
        expected_changed = (
            ranking["baseline_selected_candidate_private_ref"]
            != ranking["selected_candidate_private_ref"]
        )
        expected_realization = (
            "satisfied"
            if ranking["geometric_compliance"] is True
            else "violated"
            if ranking["geometric_compliance"] is False
            else "unknown"
        )
        if (
            ranking["snapshot_id"] != audit["snapshot_id"]
            or ranking["snapshot_manifest_sha256"] != audit["snapshot_manifest_sha256"]
            or ranking["selected_entry_id"] != audit["selected_entry_id"]
            or ranking["selected_geometric_strategy_id"]
            != audit["selected_geometric_strategy_id"]
            or ranking["geometric_compliance"] != audit["geometric_compliance"]
            or audit["behavior_changed"] is not expected_changed
        ):
            _fail("ranked HPK public/private usage audit mismatch")
        if (
            geometry["public_usage_audit_id"] != public_id
            or geometry["private_ranking_audit_id"] != ranking.stable_id
            or geometry["snapshot_id"] != audit["snapshot_id"]
            or geometry["snapshot_manifest_sha256"] != audit["snapshot_manifest_sha256"]
            or geometry["selected_entry_id"] != audit["selected_entry_id"]
            or geometry["condition_id"] != audit["condition_id"]
            or geometry["task_strategy_id"] != audit["task_strategy_id"]
            or geometry["selected_geometric_strategy_id"]
            != audit["selected_geometric_strategy_id"]
            or geometry["strategy_realization_status"] != expected_realization
        ):
            _fail("ranked HPK geometry binding differs from its audit pair")
        for motion in motions_by_public.get(public_id, []):
            if (
                motion["private_ranking_audit_id"] != ranking.stable_id
                or motion["snapshot_id"] != audit["snapshot_id"]
                or motion["snapshot_manifest_sha256"]
                != audit["snapshot_manifest_sha256"]
                or motion["selected_hpk_entry_id"] != audit["selected_entry_id"]
                or motion["condition_id"] != audit["condition_id"]
                or motion["task_strategy_id"] != audit["task_strategy_id"]
                or motion["geometric_strategy_id"]
                != audit["selected_geometric_strategy_id"]
            ):
                _fail("HPK motion realization differs from its ranked audit pair")

    pair_by_nonce: dict[str, str] = {}
    for item in private_events:
        if item.payload.get("event") != "hpk_action_attempt_prepared_private":
            continue
        payload = _private_payload(item)
        binding = payload.get("binding")
        if not isinstance(binding, Mapping):
            continue
        public_id = binding.get("public_usage_audit_id")
        private_id = binding.get("private_ranking_audit_id")
        if public_id is None and private_id is None:
            continue
        if not isinstance(public_id, str) or public_id not in public_by_id:
            _fail("prepared HPK ranking binding has an unknown public audit")
        ranking = private_by_public[public_id]
        audit = public_by_id[public_id]
        if private_id != ranking.stable_id:
            _fail("prepared HPK ranking public/private audit IDs differ")
        if (
            binding.get("_hpk_runtime_binding_id")
            != expected_runtime_binding["binding_id"]
            or binding.get("selected_hpk_entry_id") != audit["selected_entry_id"]
            or binding.get("retrieved_hpk_entry_ids") != audit["retrieved_entry_ids"]
            or binding.get("condition_id") != audit["condition_id"]
            or binding.get("task_strategy_id") != audit["task_strategy_id"]
            or binding.get("geometric_strategy_id")
            != audit["selected_geometric_strategy_id"]
            or binding.get("selected_candidate_private_ref")
            != ranking["selected_candidate_private_ref"]
            or binding.get("geometric_compliance") != ranking["geometric_compliance"]
        ):
            _fail("prepared action differs from its ranked HPK audit chain")
        nonce = str(payload.get("action_attempt_nonce", "") or "")
        if not nonce:
            _fail("prepared ranked HPK action lacks its nonce")
        previous = pair_by_nonce.get(nonce)
        if previous is not None and previous != public_id:
            _fail("one action nonce references different HPK ranking audits")
        pair_by_nonce[nonce] = public_id

    links: set[tuple[str, str]] = set()
    transitions_by_public: dict[str, list[ActionEffectTransitionV1]] = {}
    for transition in transitions:
        nonce = str(transition["action_attempt_nonce"] or "")
        public_id = pair_by_nonce.get(nonce)
        if public_id is None:
            continue
        audit = public_by_id[public_id]
        ranking = private_by_public[public_id]
        if (
            transition["selected_hpk_entry_id"] != audit["selected_entry_id"]
            or transition["retrieved_hpk_entry_ids"] != audit["retrieved_entry_ids"]
            or transition["condition_id"] != audit["condition_id"]
            or transition["task_strategy_id"] != audit["task_strategy_id"]
            or transition["geometric_strategy_id"]
            != audit["selected_geometric_strategy_id"]
            or transition["selected_candidate_private_ref"]
            != ranking["selected_candidate_private_ref"]
            or transition["geometric_compliance"] != ranking["geometric_compliance"]
        ):
            _fail("native transition differs from its ranked HPK audit chain")
        links.add((public_id, transition.stable_id))
        transitions_by_public.setdefault(public_id, []).append(transition)

    for public_id, motions in motions_by_public.items():
        linked = transitions_by_public.get(public_id, [])
        if not linked:
            _fail("HPK motion realization lacks its native ranked transition")
        for motion in motions:
            if not any(
                transition["motion_status"] == motion["motion_status"]
                and transition["realization_status"]
                == motion["strategy_realization_status"]
                for transition in linked
            ):
                _fail("HPK motion realization status differs from native transition")
    return tuple(sorted(links))


def _combined_transition_source_sha256(
    *, public_trace_sha256: str, private_transition_trace_sha256: str
) -> str:
    return _sha256_bytes(
        canonical_json_bytes(
            {
                "private_transition_trace_sha256": (private_transition_trace_sha256),
                "public_trace_sha256": public_trace_sha256,
            }
        )
    )


def classify_episode_events(
    events: Sequence[Mapping[str, Any] | _TraceEvent],
    *,
    expected_episode_id: str | int,
) -> EpisodeClassification:
    payloads = [
        item.payload if isinstance(item, _TraceEvent) else dict(item) for item in events
    ]
    starts = [item for item in payloads if item.get("event") == "episode_start"]
    ends = [item for item in payloads if item.get("event") == "episode_end"]
    if len(starts) != 1:
        return EpisodeClassification(
            expected_episode_id,
            "incomplete_or_corrupt",
            False,
            False,
            "unknown",
            f"expected exactly one episode_start, found {len(starts)}",
        )
    if starts[0].get("episode_id") != expected_episode_id:
        return EpisodeClassification(
            expected_episode_id,
            "incomplete_or_corrupt",
            False,
            False,
            "unknown",
            "episode_start ID does not match the pinned manifest",
        )
    envelope_ids = {
        item.get("episode_id")
        for item in payloads
        if item.get("episode_id") is not None
    }
    if envelope_ids != {expected_episode_id}:
        return EpisodeClassification(
            expected_episode_id,
            "incomplete_or_corrupt",
            False,
            False,
            "unknown",
            "trace mixes episode IDs or disagrees with the manifest",
        )
    interrupted = any(
        item.get("event") in {"episode_interrupt", "episode_exception"}
        for item in payloads
    )
    if len(ends) != 1:
        return EpisodeClassification(
            expected_episode_id,
            "incomplete_or_corrupt",
            False,
            False,
            "unknown",
            f"expected exactly one episode_end, found {len(ends)}",
        )
    end = ends[0]
    validity = end.get("episode_validity")
    label = str(validity.get("label", "") if isinstance(validity, dict) else "")
    natural = bool(end.get("natural_episode_end", False))
    if interrupted:
        return EpisodeClassification(
            expected_episode_id,
            "incomplete_or_corrupt",
            False,
            False,
            str(end.get("result", "unknown") or "unknown"),
            "trace contains an interrupt or unhandled exception",
        )
    if label in _VALID_EPISODE_LABELS and natural:
        return EpisodeClassification(
            expected_episode_id,
            label,
            True,
            True,
            str(end.get("result", "unknown") or "unknown"),
            "authoritative episode_end marks a valid benchmark attempt",
        )
    if label in _INVALID_EPISODE_LABELS:
        reason = (
            str(validity.get("reason", "") or "") if isinstance(validity, dict) else ""
        )
        return EpisodeClassification(
            expected_episode_id,
            label,
            False,
            natural,
            str(end.get("result", "unknown") or "unknown"),
            reason or "authoritative episode_end marks an invalid attempt",
        )
    return EpisodeClassification(
        expected_episode_id,
        "incomplete_or_corrupt",
        False,
        natural,
        str(end.get("result", "unknown") or "unknown"),
        f"unsupported or inconsistent episode validity label: {label!r}",
    )


def _event_step(value: Mapping[str, Any]) -> int | None:
    for key in ("env_step", "step"):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            return item
    return None


def _result_step(value: Mapping[str, Any]) -> int | None:
    details = value.get("details")
    if isinstance(details, dict):
        item = details.get("step_count")
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            return item
    return None


def _tool_names_from_router(value: Mapping[str, Any]) -> list[str]:
    raw = value.get("tools")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item)]
    calls = value.get("tool_calls")
    if not isinstance(calls, list):
        return []
    return [
        str(item.get("tool_name"))
        for item in calls
        if isinstance(item, dict) and str(item.get("tool_name", ""))
    ]


def _physical_result_items(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = value.get("results")
    if not isinstance(raw, list):
        return []
    return [
        dict(item)
        for item in raw
        if isinstance(item, Mapping)
        and str(item.get("tool_name", "")) in _PHYSICAL_RECOVERY_TOOLS
    ]


def _execution_status(
    results: Sequence[Mapping[str, Any]],
) -> tuple[bool, str]:
    if not results:
        return False, "unknown"
    executed = False
    failed = False
    for item in results:
        details = item.get("details")
        details = dict(details) if isinstance(details, Mapping) else {}
        skipped = details.get("skipped") is True
        partial_steps = details.get("executed_steps")
        item_executed = bool(
            not skipped
            and (
                item.get("success") is True
                or isinstance(partial_steps, int)
                and not isinstance(partial_steps, bool)
                and partial_steps > 0
            )
        )
        executed = executed or item_executed
        if skipped or item.get("success") is not True:
            failed = True
        if details.get("target_reached") is False:
            failed = True
    if failed:
        return executed, "failed_before_effect"
    return executed, "completed" if executed else "unknown"


def _selected_candidate(
    batch_events: Sequence[_TraceEvent],
) -> tuple[_TraceEvent | None, dict[str, Any] | None]:
    candidates = [
        item
        for item in batch_events
        if item.payload.get("event") == "operation_candidate_selected"
    ]
    identities = {
        str(item.payload.get("candidate_id", "") or "") for item in candidates
    }
    identities.discard("")
    if len(identities) != 1:
        return None, None
    selected = next(
        item
        for item in candidates
        if str(item.payload.get("candidate_id", "") or "") in identities
    )
    return selected, selected.payload


def _candidate_features(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    projected = candidate_semantic_features({}, value)
    if getattr(projected, "resolved", True) is False:
        return None
    return projected.to_dict()


def _latest_event(events: Sequence[_TraceEvent], event_name: str) -> _TraceEvent | None:
    return next(
        (item for item in reversed(events) if item.payload.get("event") == event_name),
        None,
    )


def _binding_from_batch(
    events: Sequence[_TraceEvent],
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if candidate is not None and isinstance(candidate.get("hpk_usage_binding"), dict):
        return dict(candidate["hpk_usage_binding"])
    binding = _latest_event(events, "hpk_geometry_usage_binding")
    return {} if binding is None else dict(binding.payload)


def _retrieved_entries(
    events: Sequence[_TraceEvent], binding: Mapping[str, Any]
) -> list[str]:
    selected = str(binding.get("selected_entry_id", "") or "").strip()
    for item in reversed(events):
        payload = item.payload
        if payload.get("event") != "hpk_public_usage_audit":
            continue
        if selected and payload.get("selected_entry_id") != selected:
            continue
        raw = payload.get("retrieved_entry_ids")
        if isinstance(raw, list):
            return [str(entry) for entry in raw if str(entry)]
    return [selected] if selected else []


def _geometric_compliance(
    events: Sequence[_TraceEvent], binding: Mapping[str, Any]
) -> bool | str:
    private = _latest_event(events, "hpk_private_ranking_audit")
    if private is not None:
        value = private.payload.get("geometric_compliance")
        if isinstance(value, bool) or value == "unverified":
            return value
    selected = str(binding.get("selected_entry_id", "") or "")
    for item in reversed(events):
        payload = item.payload
        if (
            payload.get("event") == "hpk_public_usage_audit"
            and selected
            and payload.get("selected_entry_id") == selected
        ):
            value = payload.get("geometric_compliance")
            if isinstance(value, bool) or value == "unverified":
                return value
    return "unverified"


def _motion_binding(
    events: Sequence[_TraceEvent], binding: Mapping[str, Any]
) -> tuple[str, str]:
    selected = str(binding.get("selected_entry_id", "") or "")
    for item in reversed(events):
        payload = item.payload
        if payload.get("event") != "hpk_motion_realization":
            continue
        if selected and payload.get("selected_entry_id") != selected:
            continue
        return (
            str(payload.get("motion_status", "unknown") or "unknown"),
            str(payload.get("strategy_realization_status", "unknown") or "unknown"),
        )
    return (
        "unknown",
        str(binding.get("strategy_realization_status", "unknown") or "unknown"),
    )


def _operation_and_arm(
    candidate: Mapping[str, Any] | None,
    router: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str | None]:
    operation = ""
    arm = ""
    if candidate is not None:
        operation = str(candidate.get("action_mode", "") or "").strip().lower()
        arm = str(candidate.get("arm", "") or "").strip().lower()
    calls = router.get("tool_calls")
    if isinstance(calls, list):
        operations = {
            str(item.get("args", {}).get("action_mode", "") or "").strip().lower()
            for item in calls
            if isinstance(item, dict) and isinstance(item.get("args"), dict)
        }
        operations.discard("")
        arms = {
            str(item.get("args", {}).get("arm", "") or "").strip().lower()
            for item in calls
            if isinstance(item, dict) and isinstance(item.get("args"), dict)
        }
        arms.discard("")
        if not operation and len(operations) == 1:
            operation = next(iter(operations))
        if not arm and len(arms) == 1:
            arm = next(iter(arms))
    if not arm:
        arms = {
            str(item.get("details", {}).get("arm", "") or "").strip().lower()
            for item in results
            if isinstance(item.get("details"), dict)
        }
        arms.discard("")
        if len(arms) == 1:
            arm = next(iter(arms))
    return (
        operation if operation in {"contact", "grasp", "place"} else None,
        arm if arm in {"left", "right"} else None,
    )


def _verifier_sources(events: Sequence[_TraceEvent]) -> list[str]:
    verifier = _latest_event(events, "action_effect_verification")
    if verifier is None:
        verifier = _latest_event(events, "environment_success_effect_commit")
    if verifier is None:
        return []
    sources = {"vlm_action_effect_verifier"}
    result = verifier.payload.get("result")
    if isinstance(result, dict):
        if isinstance(result.get("runtime_grasp_validation"), dict):
            sources.add("runtime_grasp_validation")
        if isinstance(result.get("runtime_place_validation"), dict):
            sources.add("runtime_place_validation")
    order = (
        "scene_memory_delta",
        "robot_state",
        "attachment_state",
        "runtime_grasp_validation",
        "runtime_place_validation",
        "vlm_action_effect_verifier",
    )
    return [item for item in order if item in sources]


def _legacy_batch_transition(
    *,
    episode: EpisodeClassification,
    episode_id: str | int,
    router: _TraceEvent,
    result_event: _TraceEvent,
    batch_events: Sequence[_TraceEvent],
    oracle_derived: bool,
    expert_derived: bool,
) -> ActionEffectTransitionV1 | None:
    results = _physical_result_items(result_event.payload)
    if not results:
        return None
    _selected_event, candidate = _selected_candidate(batch_events)
    binding = _binding_from_batch(batch_events, candidate)
    executed, derived_motion = _execution_status(results)
    bound_motion, realization = _motion_binding(batch_events, binding)
    motion = bound_motion if bound_motion != "unknown" else derived_motion
    before = _event_step(router.payload)
    after_candidates = [
        step for step in (_result_step(item) for item in results) if step is not None
    ]
    after = (
        max(after_candidates) if after_candidates else _event_step(result_event.payload)
    )
    operation, arm = _operation_and_arm(candidate, router.payload, results)
    selected_entry = str(binding.get("selected_entry_id", "") or "").strip() or None
    selected_candidate = (
        str(candidate.get("candidate_id", "") or "").strip()
        if candidate is not None
        else ""
    ) or None
    physical_count = len(results)
    actual_call_event = _latest_event(batch_events, "recovery_guard") or router
    nonce = (
        str(
            binding.get("action_attempt_nonce", "")
            or router.payload.get("action_attempt_nonce", "")
            or ""
        ).strip()
        or None
    )
    return build_action_effect_transition(
        episode_id=episode_id,
        action_attempt_nonce=nonce,
        env_step_before=before,
        env_step_after=after,
        condition_id=(str(binding.get("condition_id", "") or "").strip() or None),
        task_strategy_id=(
            str(binding.get("task_strategy_id", "") or "").strip() or None
        ),
        retrieved_hpk_entry_ids=_retrieved_entries(batch_events, binding),
        selected_hpk_entry_id=selected_entry,
        geometric_strategy_id=(
            str(binding.get("selected_geometric_strategy_id", "") or "").strip() or None
        ),
        operation=operation,
        arm=arm,
        target_role=None,
        target_relation=None,
        selected_candidate_private_ref=selected_candidate,
        candidate_geometry_features=_candidate_features(candidate),
        geometric_compliance=_geometric_compliance(batch_events, binding),
        tool_call_ref=actual_call_event.event_ref,
        tool_result_ref=result_event.event_ref,
        physical_action_executed=executed,
        motion_status=motion,
        realization_status=realization,
        pre_effect_state=None,
        post_effect_state=None,
        effect_observation_scope=(
            "batch_unseparated" if physical_count > 1 else "missing"
        ),
        target_identity_status="bound" if selected_candidate else "unknown",
        expected_effect=None,
        observed_effect=None,
        verifier_sources=_verifier_sources(batch_events),
        verifier_conflicts=[],
        infrastructure_valid=episode.infrastructure_valid,
        oracle_derived=oracle_derived,
        expert_derived=expert_derived,
    )


def _import_legacy_batches(
    events: Sequence[_TraceEvent],
    *,
    episode: EpisodeClassification,
    oracle_derived: bool,
    expert_derived: bool,
) -> tuple[tuple[ActionEffectTransitionV1, ...], tuple[ImportAbstention, ...]]:
    transitions: list[ActionEffectTransitionV1] = []
    abstentions: list[ImportAbstention] = []
    pending_router: _TraceEvent | None = None
    pending_events: list[_TraceEvent] = []
    for item in events:
        name = item.payload.get("event")
        if name == "recovery_router":
            if pending_router is not None:
                abstentions.append(
                    ImportAbstention(
                        "unclosed_recovery_batch",
                        pending_router.event_ref,
                        "a new router event appeared before a recovery_result",
                    )
                )
            tools = _tool_names_from_router(item.payload)
            pending_router = (
                item
                if any(tool in _PHYSICAL_RECOVERY_TOOLS for tool in tools)
                else None
            )
            pending_events = [item] if pending_router is not None else []
            continue
        if pending_router is None:
            continue
        pending_events.append(item)
        if name != "recovery_result":
            continue
        transition = _legacy_batch_transition(
            episode=episode,
            episode_id=episode.episode_id,
            router=pending_router,
            result_event=item,
            batch_events=pending_events,
            oracle_derived=oracle_derived,
            expert_derived=expert_derived,
        )
        if transition is not None:
            transitions.append(transition)
        pending_router = None
        pending_events = []
    if pending_router is not None:
        abstentions.append(
            ImportAbstention(
                "unclosed_recovery_batch",
                pending_router.event_ref,
                "the physical recovery batch has no recovery_result",
            )
        )
    return tuple(transitions), tuple(abstentions)


def _private_payload(item: _TraceEvent) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in item.payload.items()
        if key not in _PRIVATE_SIDECAR_ENVELOPE_KEYS
    }


def _snapshot_step(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    step = value.get("step_count")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return None
    return step


def _validate_v2_native_segment(
    *,
    prepared_event: _TraceEvent,
    bundle_event: _TraceEvent,
    boundary_events: Sequence[_TraceEvent],
    transition: ActionEffectTransitionV1,
    nonce: str,
    runtime_binding_id: str,
) -> dict[str, Any]:
    prepared = _private_payload(prepared_event)
    _exact_public_fields(
        prepared,
        frozenset(
            {
                "schema",
                "event_id",
                "action_attempt_nonce",
                "runtime_binding_id",
                "guarded_calls",
                "physical_dispatch_indices",
                "binding",
                "snapshot_before",
            }
        ),
        line=prepared_event.ordinal,
        event="private prepared attempt",
    )
    if prepared["schema"] != "trace/hpk_action_attempt_prepared/v2":
        _fail("manifest v2 prepared attempt schema is invalid")
    if prepared["action_attempt_nonce"] != nonce:
        _fail("manifest v2 prepared attempt nonce is invalid")
    if prepared["runtime_binding_id"] != runtime_binding_id:
        _fail("manifest v2 prepared attempt runtime_binding_id mismatch")
    guarded_calls = prepared["guarded_calls"]
    if not isinstance(guarded_calls, list) or not guarded_calls:
        _fail("manifest v2 prepared guarded_calls must be non-empty")
    for index, call in enumerate(guarded_calls):
        if not isinstance(call, Mapping) or set(call) != {"tool_name", "args"}:
            _fail(f"manifest v2 guarded_calls[{index}] has an invalid shape")
        if not isinstance(call["tool_name"], str) or not isinstance(
            call["args"], Mapping
        ):
            _fail(f"manifest v2 guarded_calls[{index}] is not typed")
    expected_physical = [
        index
        for index, call in enumerate(guarded_calls)
        if call["tool_name"] in _PHYSICAL_RECOVERY_TOOLS
    ]
    if prepared["physical_dispatch_indices"] != expected_physical:
        _fail("manifest v2 prepared physical dispatch indices are invalid")
    expected_tool_call_ref = stable_content_id(
        "afktcall",
        {
            "episode_id": transition["episode_id"],
            "action_attempt_nonce": nonce,
            "runtime_binding_id": runtime_binding_id,
            "guarded_calls": guarded_calls,
        },
    )
    if prepared["event_id"] != expected_tool_call_ref:
        _fail("manifest v2 prepared tool_call_ref content mismatch")
    prepared_binding = prepared["binding"]
    if not isinstance(prepared_binding, Mapping):
        _fail("manifest v2 prepared binding must be an object")
    if prepared_binding.get("_hpk_runtime_binding_id") != runtime_binding_id:
        _fail("manifest v2 prepared binding has the wrong runtime binding")
    binding_pairs = (
        ("condition_id", "condition_id"),
        ("task_strategy_id", "task_strategy_id"),
        ("retrieved_hpk_entry_ids", "retrieved_hpk_entry_ids"),
        ("selected_hpk_entry_id", "selected_hpk_entry_id"),
        ("geometric_strategy_id", "geometric_strategy_id"),
        ("operation", "operation"),
        ("arm", "arm"),
        ("target_role", "target_role"),
        ("target_relation", "target_relation"),
        ("selected_candidate_private_ref", "selected_candidate_private_ref"),
        ("candidate_geometry_features", "candidate_geometry_features"),
        ("geometric_compliance", "geometric_compliance"),
        ("expected_effect", "expected_effect"),
        ("oracle_derived", "oracle_derived"),
        ("expert_derived", "expert_derived"),
    )
    if any(
        prepared_binding.get(binding_key) != transition[transition_key]
        for binding_key, transition_key in binding_pairs
    ):
        _fail("manifest v2 prepared strategy binding differs from transition")
    matching_boundaries = [
        item
        for item in boundary_events
        if _private_payload(item).get("action_attempt_nonce") == nonce
        and _private_payload(item).get("tool_call_bundle_ref") == expected_tool_call_ref
    ]
    grouped_boundaries: dict[int, list[_TraceEvent]] = {}
    for item in matching_boundaries:
        payload = _private_payload(item)
        _exact_public_fields(
            payload,
            frozenset(
                {
                    "schema",
                    "event_id",
                    "tool_call_bundle_ref",
                    "action_attempt_nonce",
                    "runtime_binding_id",
                    "phase",
                    "dispatch_index",
                    "call",
                    "result",
                    "snapshot_before",
                    "snapshot_after",
                    "environment_success",
                    "halt_reason",
                }
            ),
            line=item.ordinal,
            event="private tool boundary",
        )
        if (
            payload["schema"] != "trace/hpk_action_tool_boundary/v2"
            or payload["runtime_binding_id"] != runtime_binding_id
        ):
            _fail("manifest v2 tool boundary schema is invalid")
        index = payload["dispatch_index"]
        if isinstance(index, bool) or not isinstance(index, int):
            _fail("manifest v2 tool boundary dispatch_index is invalid")
        grouped_boundaries.setdefault(index, []).append(item)
    if set(grouped_boundaries) != set(range(len(guarded_calls))):
        _fail("manifest v2 tool boundaries do not cover every guarded call")
    dispatch_records: list[dict[str, Any]] = []
    prior_ordinal = prepared_event.ordinal
    for index, guarded_call in enumerate(guarded_calls):
        ordered = sorted(grouped_boundaries[index], key=lambda item: item.ordinal)
        phases = [_private_payload(item)["phase"] for item in ordered]
        if phases not in (["before", "after"], ["skipped"]):
            _fail("manifest v2 tool boundary phase sequence is invalid")
        if ordered[0].ordinal <= prior_ordinal:
            _fail("manifest v2 tool boundaries are not globally ordered")
        prior_ordinal = ordered[-1].ordinal
        closed = _private_payload(ordered[-1])
        if any(
            _private_payload(item)["call"] != dict(guarded_call)
            or _private_payload(item)["tool_call_bundle_ref"] != expected_tool_call_ref
            for item in ordered
        ):
            _fail("manifest v2 tool boundary call binding mismatch")
        call_ref = stable_content_id(
            "afktcall",
            {
                "tool_call_bundle_ref": expected_tool_call_ref,
                "dispatch_index": index,
                "call": dict(guarded_call),
            },
        )
        if phases[0] == "before":
            before_payload = _private_payload(ordered[0])
            if (
                before_payload["event_id"] != call_ref
                or before_payload["result"] is not None
            ):
                _fail("manifest v2 before boundary identity is invalid")
        result_ref = None
        if closed["result"] is not None:
            result_ref = stable_content_id(
                "afktresult",
                {
                    "tool_call_ref": call_ref,
                    "phase": closed["phase"],
                    "result": closed["result"],
                    "snapshot_after": closed["snapshot_after"],
                },
            )
        if closed["event_id"] != (result_ref or call_ref):
            _fail("manifest v2 closed boundary identity is invalid")
        dispatch_records.append(
            {
                "phase": closed["phase"],
                "dispatch_index": index,
                "tool_name": guarded_call["tool_name"],
                "physical": index in expected_physical,
                "tool_call_ref": call_ref,
                "tool_result_ref": result_ref,
                "snapshot_before": closed["snapshot_before"],
                "snapshot_after": closed["snapshot_after"],
                "result": closed["result"],
                "environment_success": closed["environment_success"],
                "halt_reason": closed["halt_reason"],
            }
        )
    if prior_ordinal >= bundle_event.ordinal:
        _fail("manifest v2 result bundle precedes its closed boundaries")
    bundle = _private_payload(bundle_event)
    _exact_public_fields(
        bundle,
        frozenset(
            {
                "schema",
                "event_id",
                "tool_call_ref",
                "action_attempt_nonce",
                "runtime_binding_id",
                "dispatch_results",
            }
        ),
        line=bundle_event.ordinal,
        event="private result bundle",
    )
    if (
        bundle["schema"] != "trace/hpk_action_tool_result_bundle/v2"
        or bundle["action_attempt_nonce"] != nonce
        or bundle["runtime_binding_id"] != runtime_binding_id
        or bundle["tool_call_ref"] != expected_tool_call_ref
        or bundle["dispatch_results"] != dispatch_records
    ):
        _fail("manifest v2 result bundle content mismatch")
    expected_result_ref = stable_content_id(
        "afktresult",
        {
            "tool_call_ref": expected_tool_call_ref,
            "action_attempt_nonce": nonce,
            "runtime_binding_id": runtime_binding_id,
            "dispatch_results": dispatch_records,
        },
    )
    if bundle["event_id"] != expected_result_ref:
        _fail("manifest v2 result bundle identity mismatch")
    return {
        "prepared_ordinal": prepared_event.ordinal,
        "bundle_ordinal": bundle_event.ordinal,
        "boundary_ordinals": [item.ordinal for item in matching_boundaries],
        "tool_call_ref": expected_tool_call_ref,
        "tool_result_ref": expected_result_ref,
        "guarded_calls": [dict(item) for item in guarded_calls],
        "physical_indices": expected_physical,
        "dispatch_records": dispatch_records,
        "binding": dict(prepared_binding),
    }


def _validate_v2_native_causal_chains_single_segment_legacy(
    events: Sequence[_TraceEvent],
    transitions: Sequence[ActionEffectTransitionV1],
) -> None:
    prepared_events = [
        item
        for item in events
        if item.payload["event"] == "hpk_action_attempt_prepared_private"
    ]
    boundary_events = [
        item
        for item in events
        if item.payload["event"] == "hpk_action_tool_boundary_private"
    ]
    bundle_events = [
        item
        for item in events
        if item.payload["event"] == "hpk_action_tool_result_bundle_private"
    ]
    transition_events = [
        item
        for item in events
        if item.payload["event"] == ACTION_EFFECT_TRANSITION_EVENT
    ]
    transition_by_id = {item.stable_id: item for item in transitions}
    if len(transition_events) != len(transitions):
        _fail("manifest v2 private sidecar transition count is inconsistent")
    used_prepared: set[int] = set()
    used_boundaries: set[int] = set()
    used_bundles: set[int] = set()
    for transition_event in transition_events:
        transition_envelope = _private_payload(transition_event)
        _exact_public_fields(
            transition_envelope,
            frozenset({"schema", "event_id", "transition"}),
            line=transition_event.ordinal,
            event="private native transition",
        )
        if transition_envelope["schema"] != "trace/action_effect_transition/v1":
            _fail("manifest v2 native transition event schema is invalid")
        transition = ActionEffectTransitionV1.from_dict(
            _native_transition_payload(transition_event.payload)
        )
        if transition_envelope["event_id"] != transition.stable_id:
            _fail("manifest v2 native transition event identity mismatch")
        if transition.stable_id not in transition_by_id:
            _fail("manifest v2 native transition identity is inconsistent")
        nonce = str(transition["action_attempt_nonce"] or "").strip()
        if not nonce:
            _fail("manifest v2 native transition requires an action nonce")
        matching_prepared = [
            item
            for item in prepared_events
            if _private_payload(item).get("action_attempt_nonce") == nonce
        ]
        matching_bundles = [
            item
            for item in bundle_events
            if _private_payload(item).get("action_attempt_nonce") == nonce
        ]
        matching_boundaries = [
            item
            for item in boundary_events
            if _private_payload(item).get("action_attempt_nonce") == nonce
        ]
        if len(matching_prepared) != 1 or len(matching_bundles) != 1:
            _fail(
                "manifest v2 native transition requires exactly one prepared "
                "attempt and result bundle"
            )
        prepared_event = matching_prepared[0]
        bundle_event = matching_bundles[0]
        if not (
            prepared_event.ordinal < bundle_event.ordinal < transition_event.ordinal
        ):
            _fail("manifest v2 native causal events are out of order")
        prepared = _private_payload(prepared_event)
        _exact_public_fields(
            prepared,
            frozenset(
                {
                    "schema",
                    "event_id",
                    "action_attempt_nonce",
                    "guarded_calls",
                    "physical_dispatch_indices",
                    "binding",
                    "snapshot_before",
                }
            ),
            line=prepared_event.ordinal,
            event="private prepared attempt",
        )
        if prepared["schema"] != "trace/hpk_action_attempt_prepared/v1":
            _fail("manifest v2 prepared attempt schema is invalid")
        guarded_calls = prepared["guarded_calls"]
        if not isinstance(guarded_calls, list) or not guarded_calls:
            _fail("manifest v2 prepared guarded_calls must be non-empty")
        for index, call in enumerate(guarded_calls):
            if not isinstance(call, Mapping) or set(call) != {"tool_name", "args"}:
                _fail(f"manifest v2 guarded_calls[{index}] has an invalid shape")
            if not isinstance(call["tool_name"], str) or not isinstance(
                call["args"], Mapping
            ):
                _fail(f"manifest v2 guarded_calls[{index}] is not typed")
        expected_physical = [
            index
            for index, call in enumerate(guarded_calls)
            if call["tool_name"] in _PHYSICAL_RECOVERY_TOOLS
        ]
        if prepared["physical_dispatch_indices"] != expected_physical:
            _fail("manifest v2 prepared physical dispatch indices are invalid")
        expected_tool_call_ref = stable_content_id(
            "afktcall",
            {
                "episode_id": transition["episode_id"],
                "action_attempt_nonce": nonce,
                "guarded_calls": guarded_calls,
            },
        )
        if (
            prepared["event_id"] != expected_tool_call_ref
            or transition["tool_call_ref"] != expected_tool_call_ref
        ):
            _fail("manifest v2 prepared tool_call_ref content mismatch")
        prepared_binding = prepared["binding"]
        if not isinstance(prepared_binding, Mapping):
            _fail("manifest v2 prepared binding must be an object")
        binding_pairs = (
            ("condition_id", "condition_id"),
            ("task_strategy_id", "task_strategy_id"),
            ("retrieved_hpk_entry_ids", "retrieved_hpk_entry_ids"),
            ("selected_hpk_entry_id", "selected_hpk_entry_id"),
            ("geometric_strategy_id", "geometric_strategy_id"),
            ("operation", "operation"),
            ("arm", "arm"),
            ("target_role", "target_role"),
            ("target_relation", "target_relation"),
            ("selected_candidate_private_ref", "selected_candidate_private_ref"),
            ("candidate_geometry_features", "candidate_geometry_features"),
            ("geometric_compliance", "geometric_compliance"),
            ("realization_status", "realization_status"),
            ("target_identity_status", "target_identity_status"),
            ("expected_effect", "expected_effect"),
            ("oracle_derived", "oracle_derived"),
            ("expert_derived", "expert_derived"),
        )
        if any(
            prepared_binding.get(binding_key) != transition[transition_key]
            for binding_key, transition_key in binding_pairs
        ):
            _fail("manifest v2 prepared strategy binding differs from transition")

        grouped_boundaries: dict[int, list[_TraceEvent]] = {}
        for item in matching_boundaries:
            payload = _private_payload(item)
            _exact_public_fields(
                payload,
                frozenset(
                    {
                        "schema",
                        "event_id",
                        "tool_call_bundle_ref",
                        "action_attempt_nonce",
                        "phase",
                        "dispatch_index",
                        "call",
                        "result",
                        "snapshot_before",
                        "snapshot_after",
                        "environment_success",
                        "halt_reason",
                    }
                ),
                line=item.ordinal,
                event="private tool boundary",
            )
            if payload["schema"] != "trace/hpk_action_tool_boundary/v1":
                _fail("manifest v2 tool boundary schema is invalid")
            index = payload["dispatch_index"]
            if isinstance(index, bool) or not isinstance(index, int):
                _fail("manifest v2 tool boundary dispatch_index is invalid")
            grouped_boundaries.setdefault(index, []).append(item)
        if set(grouped_boundaries) != set(range(len(guarded_calls))):
            _fail("manifest v2 tool boundaries do not cover every guarded call")
        dispatch_records: list[dict[str, Any]] = []
        prior_ordinal = prepared_event.ordinal
        for index, guarded_call in enumerate(guarded_calls):
            ordered = sorted(grouped_boundaries[index], key=lambda item: item.ordinal)
            phases = [_private_payload(item)["phase"] for item in ordered]
            if phases not in (["before", "after"], ["skipped"]):
                _fail("manifest v2 tool boundary phase sequence is invalid")
            if ordered[0].ordinal <= prior_ordinal:
                _fail("manifest v2 tool boundaries are not globally ordered")
            prior_ordinal = ordered[-1].ordinal
            closed_event = ordered[-1]
            closed = _private_payload(closed_event)
            if any(
                _private_payload(item)["call"] != dict(guarded_call)
                or _private_payload(item)["tool_call_bundle_ref"]
                != expected_tool_call_ref
                for item in ordered
            ):
                _fail("manifest v2 tool boundary call binding mismatch")
            call_ref = stable_content_id(
                "afktcall",
                {
                    "tool_call_bundle_ref": expected_tool_call_ref,
                    "dispatch_index": index,
                    "call": dict(guarded_call),
                },
            )
            if phases[0] == "before":
                before_payload = _private_payload(ordered[0])
                if (
                    before_payload["event_id"] != call_ref
                    or before_payload["result"] is not None
                ):
                    _fail("manifest v2 before boundary identity is invalid")
            result_ref = None
            if closed["result"] is not None:
                result_ref = stable_content_id(
                    "afktresult",
                    {
                        "tool_call_ref": call_ref,
                        "phase": closed["phase"],
                        "result": closed["result"],
                        "snapshot_after": closed["snapshot_after"],
                    },
                )
            if closed["event_id"] != (result_ref or call_ref):
                _fail("manifest v2 closed boundary identity is invalid")
            dispatch_records.append(
                {
                    "phase": closed["phase"],
                    "dispatch_index": index,
                    "tool_name": guarded_call["tool_name"],
                    "physical": index in expected_physical,
                    "tool_call_ref": call_ref,
                    "tool_result_ref": result_ref,
                    "snapshot_before": closed["snapshot_before"],
                    "snapshot_after": closed["snapshot_after"],
                    "result": closed["result"],
                    "environment_success": closed["environment_success"],
                    "halt_reason": closed["halt_reason"],
                }
            )
        if prior_ordinal >= bundle_event.ordinal:
            _fail("manifest v2 result bundle precedes its closed boundaries")
        bundle = _private_payload(bundle_event)
        _exact_public_fields(
            bundle,
            frozenset(
                {
                    "schema",
                    "event_id",
                    "tool_call_ref",
                    "action_attempt_nonce",
                    "dispatch_results",
                }
            ),
            line=bundle_event.ordinal,
            event="private result bundle",
        )
        if (
            bundle["schema"] != "trace/hpk_action_tool_result_bundle/v1"
            or bundle["tool_call_ref"] != expected_tool_call_ref
            or bundle["dispatch_results"] != dispatch_records
        ):
            _fail("manifest v2 result bundle content mismatch")
        expected_result_ref = stable_content_id(
            "afktresult",
            {
                "tool_call_ref": expected_tool_call_ref,
                "action_attempt_nonce": nonce,
                "dispatch_results": dispatch_records,
            },
        )
        if (
            bundle["event_id"] != expected_result_ref
            or transition["tool_result_ref"] != expected_result_ref
        ):
            _fail("manifest v2 result bundle identity mismatch")
        if expected_physical:
            first_physical = dispatch_records[expected_physical[0]]
            if transition["env_step_before"] != _snapshot_step(
                first_physical["snapshot_before"]
            ):
                _fail("manifest v2 transition physical start step mismatch")
        if transition["env_step_after"] != _snapshot_step(
            dispatch_records[-1]["snapshot_after"]
        ):
            _fail("manifest v2 transition effect observation step mismatch")
        used_prepared.add(prepared_event.ordinal)
        used_bundles.add(bundle_event.ordinal)
        used_boundaries.update(item.ordinal for item in matching_boundaries)
    if len(used_prepared) != len(prepared_events):
        _fail("manifest v2 private sidecar contains an unused prepared attempt")
    if len(used_bundles) != len(bundle_events):
        _fail("manifest v2 private sidecar contains an unused result bundle")
    if len(used_boundaries) != len(boundary_events):
        _fail("manifest v2 private sidecar contains unused tool boundaries")


def _validate_v2_native_causal_chains(
    events: Sequence[_TraceEvent],
    transitions: Sequence[ActionEffectTransitionV1],
    *,
    runtime_binding_id: str,
    max_semantic_segments: int,
) -> None:
    """Recompute every segment and the final semantic-attempt aggregate."""

    prepared_events = [
        item
        for item in events
        if item.payload["event"] == "hpk_action_attempt_prepared_private"
    ]
    boundary_events = [
        item
        for item in events
        if item.payload["event"] == "hpk_action_tool_boundary_private"
    ]
    bundle_events = [
        item
        for item in events
        if item.payload["event"] == "hpk_action_tool_result_bundle_private"
    ]
    transition_events = [
        item
        for item in events
        if item.payload["event"] == ACTION_EFFECT_TRANSITION_EVENT
    ]
    transition_by_id = {item.stable_id: item for item in transitions}
    if len(transition_events) != len(transitions):
        _fail("manifest v2 private sidecar transition count is inconsistent")
    used_prepared: set[int] = set()
    used_boundaries: set[int] = set()
    used_bundles: set[int] = set()
    for transition_event in transition_events:
        transition_envelope = _private_payload(transition_event)
        _exact_public_fields(
            transition_envelope,
            frozenset({"schema", "event_id", "runtime_binding_id", "transition"}),
            line=transition_event.ordinal,
            event="private native transition",
        )
        if (
            transition_envelope["schema"] != "trace/action_effect_transition/v2"
            or transition_envelope["runtime_binding_id"] != runtime_binding_id
        ):
            _fail("manifest v2 native transition event schema is invalid")
        transition = ActionEffectTransitionV1.from_dict(
            _native_transition_payload(transition_event.payload)
        )
        if (
            transition_envelope["event_id"] != transition.stable_id
            or transition.stable_id not in transition_by_id
        ):
            _fail("manifest v2 native transition identity is inconsistent")
        nonce = str(transition["action_attempt_nonce"] or "").strip()
        if not nonce:
            _fail("manifest v2 native transition requires an action nonce")
        matching_prepared = sorted(
            (
                item
                for item in prepared_events
                if _private_payload(item).get("action_attempt_nonce") == nonce
            ),
            key=lambda item: item.ordinal,
        )
        matching_bundles = [
            item
            for item in bundle_events
            if _private_payload(item).get("action_attempt_nonce") == nonce
        ]
        if not matching_prepared or len(matching_bundles) != len(matching_prepared):
            _fail(
                "manifest v2 native transition requires a prepared attempt and "
                "one result bundle per segment"
            )
        if len(matching_prepared) > max_semantic_segments:
            _fail("manifest v2 semantic attempt exceeds the frozen segment limit")
        segment_results: list[dict[str, Any]] = []
        prior_end = -1
        for prepared_event in matching_prepared:
            prepared_ref = _private_payload(prepared_event).get("event_id")
            candidates = [
                item
                for item in matching_bundles
                if _private_payload(item).get("tool_call_ref") == prepared_ref
            ]
            if len(candidates) != 1:
                _fail(
                    "manifest v2 prepared segment lacks one uniquely bound result bundle"
                )
            bundle_event = candidates[0]
            if not (
                prior_end
                < prepared_event.ordinal
                < bundle_event.ordinal
                < transition_event.ordinal
            ):
                _fail("manifest v2 native segment events are out of order")
            segment = _validate_v2_native_segment(
                prepared_event=prepared_event,
                bundle_event=bundle_event,
                boundary_events=boundary_events,
                transition=transition,
                nonce=nonce,
                runtime_binding_id=runtime_binding_id,
            )
            prior_end = int(segment["bundle_ordinal"])
            segment_results.append(segment)
            used_prepared.add(int(segment["prepared_ordinal"]))
            used_bundles.add(int(segment["bundle_ordinal"]))
            used_boundaries.update(segment["boundary_ordinals"])
        call_refs = [str(item["tool_call_ref"]) for item in segment_results]
        result_refs = [str(item["tool_result_ref"]) for item in segment_results]
        if len(segment_results) == 1:
            expected_call_ref = call_refs[0]
            expected_result_ref = result_refs[0]
        else:
            expected_call_ref = stable_content_id(
                "afktcall",
                {
                    "episode_id": transition["episode_id"],
                    "action_attempt_nonce": nonce,
                    "runtime_binding_id": runtime_binding_id,
                    "segment_tool_call_refs": call_refs,
                },
            )
            expected_result_ref = stable_content_id(
                "afktresult",
                {
                    "tool_call_ref": expected_call_ref,
                    "action_attempt_nonce": nonce,
                    "runtime_binding_id": runtime_binding_id,
                    "segment_tool_result_refs": result_refs,
                },
            )
        if (
            transition["tool_call_ref"] != expected_call_ref
            or transition["tool_result_ref"] != expected_result_ref
        ):
            _fail("manifest v2 aggregate tool reference content mismatch")
        all_calls = [
            RecoveryToolCall(
                tool_name=str(call["tool_name"]),
                args=copy.deepcopy(dict(call["args"])),
            )
            for segment in segment_results
            for call in segment["guarded_calls"]
        ]
        offsets: list[int] = []
        running = 0
        for segment in segment_results:
            offsets.append(running)
            running += len(segment["guarded_calls"])
        all_records: list[dict[str, Any]] = []
        all_physical: list[int] = []
        for offset, segment in zip(offsets, segment_results, strict=True):
            all_physical.extend(offset + index for index in segment["physical_indices"])
            for record in segment["dispatch_records"]:
                copied = copy.deepcopy(record)
                copied["dispatch_index"] = offset + copied["dispatch_index"]
                all_records.append(copied)
        if not all_physical:
            _fail("manifest v2 semantic transition has no physical action")
        by_index = {item["dispatch_index"]: item for item in all_records}
        physical_records = [by_index[index] for index in all_physical]
        env_before = _snapshot_step(physical_records[0]["snapshot_before"])
        env_after = _snapshot_step(all_records[-1]["snapshot_after"])
        if (
            transition["env_step_before"] != env_before
            or transition["env_step_after"] != env_after
        ):
            _fail("manifest v2 transition step boundary mismatch")
        physical_executed = False
        physical_failed = False
        for record in physical_records:
            result = record.get("result")
            result = result if isinstance(result, Mapping) else {}
            details = result.get("details")
            details = details if isinstance(details, Mapping) else {}
            skipped = record["phase"] == "skipped" or details.get("skipped") is True
            success = result.get("success") is True
            partial = details.get("executed_steps")
            physical_executed = bool(
                physical_executed
                or not skipped
                and (
                    success
                    or isinstance(partial, int)
                    and not isinstance(partial, bool)
                    and partial > 0
                )
            )
            if skipped or not success or details.get("target_reached") is False:
                physical_failed = True
        motion_status = (
            "failed_before_effect"
            if physical_failed
            else "completed"
            if physical_executed
            else "unknown"
        )
        if (
            transition["physical_action_executed"] is not physical_executed
            or transition["motion_status"] != motion_status
        ):
            _fail("manifest v2 transition execution status differs from boundaries")
        explicit_observations = [
            record for record in all_records if record["tool_name"] == "reobserve_scene"
        ]
        evidence_record = (
            explicit_observations[-1] if explicit_observations else all_records[-1]
        )
        evidence_result = evidence_record.get("result")
        evidence_result = (
            evidence_result if isinstance(evidence_result, Mapping) else {}
        )
        evidence_step = _snapshot_step(evidence_record.get("snapshot_after"))
        fresh = bool(
            env_before is not None
            and evidence_record["phase"] == "after"
            and evidence_result.get("success") is True
            and evidence_step is not None
            and evidence_step > env_before
        )
        from roboharn_evo.agent.hpk.transition_runtime import prove_semantic_attempt_group

        semantic_binding = dict(segment_results[0]["binding"])
        grouped = prove_semantic_attempt_group(
            calls=all_calls,
            binding=semantic_binding,
        )
        terminal = _OPERATION_EFFECT_TERMINALS.get(
            str(semantic_binding.get("operation", "") or "").strip().lower()
        )
        terminal_executed = bool(
            terminal and any(call.tool_name == terminal for call in all_calls)
        )
        if (
            terminal_executed
            and (grouped or len(all_physical) == 1)
            and fresh
            and transition["post_effect_state"] is not None
            and transition["expected_effect"] is not None
        ):
            expected_scope = "independent"
        elif len(all_physical) > 1 and not grouped:
            expected_scope = "batch_unseparated"
        else:
            expected_scope = "missing"
        if transition["effect_observation_scope"] != expected_scope:
            _fail("manifest v2 transition observation scope differs from boundaries")
    if len(used_prepared) != len(prepared_events):
        _fail("manifest v2 private sidecar contains an unused prepared segment")
    if len(used_bundles) != len(bundle_events):
        _fail("manifest v2 private sidecar contains an unused result bundle")
    if len(used_boundaries) != len(boundary_events):
        _fail("manifest v2 private sidecar contains unused tool boundaries")


def _validate_v2_public_transition_projections(
    public_events: Sequence[_TraceEvent],
    transitions: Sequence[ActionEffectTransitionV1],
) -> None:
    records = [
        item
        for item in public_events
        if item.payload["event"] == "hpk_action_effect_transition_public"
    ]
    if len(records) != len(transitions):
        _fail("manifest v2 public/private transition counts differ")
    by_transition = {str(item.payload.get("transition_id")): item for item in records}
    if len(by_transition) != len(records):
        _fail("manifest v2 public transition identities are duplicated")
    for transition in transitions:
        item = by_transition.get(transition.stable_id)
        if item is None:
            _fail("manifest v2 private transition lacks its public projection")
        expected = public_transition_projection(transition)
        actual = {
            key: copy.deepcopy(value)
            for key, value in item.payload.items()
            if key
            not in {
                "event",
                "timestamp",
                "seed",
                "env_step",
            }
        }
        if canonical_json_bytes(actual) != canonical_json_bytes(expected):
            _fail("manifest v2 public transition projection content mismatch")


def _native_transition_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    nested = value.get("transition")
    if isinstance(nested, Mapping):
        return copy.deepcopy(dict(nested))
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key not in {"event", "timestamp", "seed", "env_step"}
    }


def _import_native_transitions(
    events: Sequence[_TraceEvent],
    *,
    episode: EpisodeClassification,
    oracle_derived: bool,
    expert_derived: bool,
    require_resolved_tool_references: bool = True,
) -> tuple[ActionEffectTransitionV1, ...]:
    result: list[ActionEffectTransitionV1] = []
    seen: set[str] = set()
    seen_nonces: set[str] = set()
    previous_after: int | None = None
    reference_positions: dict[str, int] = {}
    for source_event in events:
        reference_positions[source_event.event_ref] = source_event.ordinal
        explicit_id = source_event.payload.get("event_id")
        if explicit_id is None:
            continue
        if not isinstance(explicit_id, str) or not explicit_id.strip():
            _fail(f"trace line {source_event.ordinal} has an invalid explicit event_id")
        explicit_id = explicit_id.strip()
        if explicit_id in reference_positions:
            _fail(f"duplicate event reference in trace: {explicit_id}")
        reference_positions[explicit_id] = source_event.ordinal
    for item in events:
        if item.payload.get("event") != ACTION_EFFECT_TRANSITION_EVENT:
            continue
        try:
            transition = ActionEffectTransitionV1.from_dict(
                _native_transition_payload(item.payload)
            )
        except HPKValidationError as exc:
            raise RolloutImportError(
                f"invalid native action transition at trace line {item.ordinal}: {exc}"
            ) from exc
        if not _same_typed_episode_id(transition["episode_id"], episode.episode_id):
            _fail(f"native transition at line {item.ordinal} has the wrong episode_id")
        if (
            transition["oracle_derived"] != oracle_derived
            or transition["expert_derived"] != expert_derived
        ):
            _fail(
                f"native transition at line {item.ordinal} contradicts manifest "
                "information-access flags"
            )
        if transition.stable_id in seen:
            _fail(
                f"duplicate native action transition at trace line {item.ordinal}: "
                f"{transition.stable_id}"
            )
        seen.add(transition.stable_id)
        nonce = str(transition["action_attempt_nonce"] or "").strip()
        if nonce:
            if nonce in seen_nonces:
                _fail(
                    f"duplicate action_attempt_nonce at trace line {item.ordinal}: "
                    f"{nonce}"
                )
            seen_nonces.add(nonce)
        before = transition["env_step_before"]
        after = transition["env_step_after"]
        if (
            previous_after is not None
            and before is not None
            and before < previous_after
        ):
            _fail(
                f"native transition at line {item.ordinal} overlaps or reverses "
                "the preceding action transition"
            )
        if after is not None:
            previous_after = after
        envelope_step = _event_step(item.payload)
        if after is not None and envelope_step is not None and envelope_step != after:
            _fail(
                f"native transition at line {item.ordinal} disagrees with its "
                "event-envelope env_step"
            )
        if require_resolved_tool_references:
            call_ref = transition["tool_call_ref"]
            result_ref = transition["tool_result_ref"]
            for label, reference in (
                ("tool_call_ref", call_ref),
                ("tool_result_ref", result_ref),
            ):
                if reference is not None and reference not in reference_positions:
                    _fail(
                        f"native transition at line {item.ordinal} has an unresolved "
                        f"{label}: {reference}"
                    )
            if call_ref is not None and result_ref is not None:
                call_position = reference_positions[call_ref]
                result_position = reference_positions[result_ref]
                if not call_position < result_position < item.ordinal:
                    _fail(
                        f"native transition at line {item.ordinal} has invalid tool "
                        "call/result ordering"
                    )
        result.append(transition)
    return tuple(result)


class AgentRolloutImporter:
    """Importer with no implicit path discovery or mutable source selection."""

    def import_files(
        self,
        *,
        manifest_path: str | Path,
        trace_path: str | Path,
        expected_manifest_sha256: str | None = None,
        expected_trace_sha256: str | None = None,
    ) -> RolloutImportResult:
        expected_manifest_sha256 = _expected_source_sha256(
            expected_manifest_sha256,
            label="expected rollout manifest SHA-256",
        )
        expected_trace_sha256 = _expected_source_sha256(
            expected_trace_sha256,
            label="expected rollout trace SHA-256",
        )
        manifest_file = Path(manifest_path).expanduser().absolute()
        trace_file = Path(trace_path).expanduser().absolute()
        manifest_raw = _read_explicit_regular_file(
            manifest_file, label="rollout import manifest"
        )
        manifest_sha256 = _sha256_bytes(manifest_raw)
        if (
            expected_manifest_sha256 is not None
            and manifest_sha256 != expected_manifest_sha256
        ):
            _fail(
                "rollout import manifest external SHA-256 mismatch: "
                f"expected={expected_manifest_sha256}, actual={manifest_sha256}"
            )
        manifest = _validate_manifest(
            _decode_json_object(manifest_raw, label="rollout import manifest")
        )
        schema = str(manifest["schema"])
        public_descriptor = (
            manifest["trace"]
            if schema == ROLLOUT_IMPORT_MANIFEST_V1_SCHEMA
            else manifest["public_trace"]
        )
        declared_trace = _resolve_declared_trace(manifest_file, public_descriptor)
        if declared_trace != trace_file:
            _fail(
                "explicit trace path does not match the pinned manifest: "
                f"declared={declared_trace}, supplied={trace_file}"
            )
        public_label = (
            "rollout trace"
            if schema == ROLLOUT_IMPORT_MANIFEST_V1_SCHEMA
            else "public rollout trace"
        )
        trace_raw = _read_explicit_regular_file(trace_file, label=public_label)
        trace_sha256 = _sha256_bytes(trace_raw)
        if expected_trace_sha256 is not None and trace_sha256 != expected_trace_sha256:
            _fail(
                "rollout trace external SHA-256 mismatch: "
                f"expected={expected_trace_sha256}, actual={trace_sha256}"
            )
        if trace_sha256 != public_descriptor["sha256"]:
            _fail(
                f"{public_label} SHA-256 mismatch: "
                f"expected={public_descriptor['sha256']}, actual={trace_sha256}"
            )
        access = manifest["information_access"]
        private_transition_trace_path: Path | None = None
        private_transition_trace_sha256: str | None = None
        if schema == ROLLOUT_IMPORT_MANIFEST_V2_SCHEMA:
            public_events = _parse_jsonl(
                trace_raw,
                trace_sha256=trace_sha256,
                label="public rollout trace",
            )
            _validate_v2_public_events(
                public_events,
                expected_episode_id=manifest["episode_id"],
                expected_runtime_binding=manifest["runtime_binding"],
            )
            episode = classify_episode_events(
                public_events,
                expected_episode_id=manifest["episode_id"],
            )
            private_descriptor = manifest["private_transition_trace"]
            private_transition_trace_path = _resolve_declared_trace(
                manifest_file, private_descriptor
            )
            if private_transition_trace_path == trace_file:
                _fail(
                    "manifest v2 public_trace and private_transition_trace "
                    "must resolve to different files"
                )
            private_raw = _read_explicit_regular_file(
                private_transition_trace_path,
                label="private transition trace",
            )
            private_transition_trace_sha256 = _sha256_bytes(private_raw)
            if private_transition_trace_sha256 != private_descriptor["sha256"]:
                _fail(
                    "private transition trace SHA-256 mismatch: "
                    f"expected={private_descriptor['sha256']}, "
                    f"actual={private_transition_trace_sha256}"
                )
            private_events = _parse_jsonl(
                private_raw,
                trace_sha256=private_transition_trace_sha256,
                label="private transition trace",
                allow_empty=True,
            )
            _validate_v2_private_sidecar_events(
                private_events, expected_episode_id=manifest["episode_id"]
            )
            _validate_v2_safe_exploration_audits(
                public_events,
                private_events,
                expected_runtime_binding=manifest["runtime_binding"],
            )
            transitions = _import_native_transitions(
                private_events,
                episode=episode,
                oracle_derived=bool(access["oracle_derived"]),
                expert_derived=bool(access["expert_derived"]),
                require_resolved_tool_references=False,
            )
            has_private_causal_events = any(
                item.payload.get("event")
                in {
                    "hpk_action_attempt_prepared_private",
                    "hpk_action_tool_boundary_private",
                    "hpk_action_tool_result_bundle_private",
                    ACTION_EFFECT_TRANSITION_EVENT,
                }
                for item in private_events
            )
            if transitions or has_private_causal_events:
                geometry_ref = manifest["runtime_binding"]["policy_refs"]["geometry"]
                geometry_policy = (
                    load_safe_exploration_geometry_policy()
                    if geometry_ref["policy_id"] in {"hpk_geometry_policy/v3", "afk_geometry_policy/v3"}
                    else load_evolving_geometry_policy()
                )
                if (geometry_ref["policy_id"], geometry_ref["config_sha256"]) not in EVOLVING_GEOMETRY_POLICY_IDENTITIES:
                    _fail("runtime binding geometry policy is not repository-frozen")
                _validate_v2_native_causal_chains(
                    private_events,
                    transitions,
                    runtime_binding_id=manifest["runtime_binding"]["binding_id"],
                    max_semantic_segments=int(
                        geometry_policy.payload["semantic_attempt"]["max_segments"]
                    ),
                )
            _validate_v2_public_transition_projections(
                public_events,
                transitions,
            )
            _validate_v2_safe_exploration_attempt_links(
                private_events,
                transitions,
            )
            ranked_usage_transition_links = _validate_v2_ranked_geometry_chains(
                public_events,
                private_events,
                transitions,
                expected_runtime_binding=manifest["runtime_binding"],
            )
            abstentions: tuple[ImportAbstention, ...] = ()
            if not transitions:
                abstentions = (
                    ImportAbstention(
                        "no_action_transition",
                        None,
                        "the explicitly pinned private transition trace contains "
                        "no native action transition",
                    ),
                )
            transition_source_sha256 = _combined_transition_source_sha256(
                public_trace_sha256=trace_sha256,
                private_transition_trace_sha256=(private_transition_trace_sha256),
            )
        else:
            ranked_usage_transition_links = ()
            events = _parse_jsonl(trace_raw, trace_sha256=trace_sha256)
            episode = classify_episode_events(
                events,
                expected_episode_id=manifest["episode_id"],
            )
            native = any(
                item.payload.get("event") == ACTION_EFFECT_TRANSITION_EVENT
                for item in events
            )
            if native:
                transitions = _import_native_transitions(
                    events,
                    episode=episode,
                    oracle_derived=bool(access["oracle_derived"]),
                    expert_derived=bool(access["expert_derived"]),
                )
                abstentions = ()
            else:
                transitions, abstentions = _import_legacy_batches(
                    events,
                    episode=episode,
                    oracle_derived=bool(access["oracle_derived"]),
                    expert_derived=bool(access["expert_derived"]),
                )
                if not transitions:
                    abstentions = (
                        *abstentions,
                        ImportAbstention(
                            "no_action_transition",
                            None,
                            "the explicitly supplied trace contains no importable "
                            "physical action",
                        ),
                    )
            transition_source_sha256 = trace_sha256
        if not episode.infrastructure_valid:
            abstentions = (
                *abstentions,
                ImportAbstention(
                    "episode_infrastructure_invalid",
                    None,
                    episode.reason,
                ),
            )
        return RolloutImportResult(
            rollout_id=str(manifest["rollout_id"]),
            manifest_path=manifest_file,
            trace_path=trace_file,
            trace_sha256=trace_sha256,
            manifest_schema=schema,
            private_transition_trace_path=private_transition_trace_path,
            private_transition_trace_sha256=(private_transition_trace_sha256),
            transition_source_sha256=transition_source_sha256,
            runtime_binding=(
                copy.deepcopy(manifest["runtime_binding"])
                if schema == ROLLOUT_IMPORT_MANIFEST_V2_SCHEMA
                else None
            ),
            ranked_usage_transition_links=ranked_usage_transition_links,
            rollout_manifest_sha256=manifest_sha256,
            episode=episode,
            transitions=transitions,
            abstentions=abstentions,
        )


def import_rollout(
    *,
    manifest_path: str | Path,
    trace_path: str | Path,
    expected_manifest_sha256: str | None = None,
    expected_trace_sha256: str | None = None,
) -> RolloutImportResult:
    return AgentRolloutImporter().import_files(
        manifest_path=manifest_path,
        trace_path=trace_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_trace_sha256=expected_trace_sha256,
    )


__all__ = [
    "ACTION_EFFECT_TRANSITION_EVENT",
    "PRIVATE_SIDECAR_EVENT_WHITELIST",
    "PUBLIC_TRACE_EVENT_WHITELIST",
    "ROLLOUT_IMPORT_MANIFEST_SCHEMA",
    "ROLLOUT_IMPORT_MANIFEST_V1_SCHEMA",
    "ROLLOUT_IMPORT_MANIFEST_V2_SCHEMA",
    "AgentRolloutImporter",
    "EpisodeClassification",
    "ImportAbstention",
    "RolloutImportError",
    "RolloutImportResult",
    "classify_episode_events",
    "import_rollout",
]
