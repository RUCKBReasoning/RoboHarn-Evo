from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from roboharn_evo.agent.hpk.action_transition import ActionEffectTransitionV1
from roboharn_evo.agent.hpk.evidence_store import (
    EvidenceStore,
    private_provenance_sha256,
)
from roboharn_evo.agent.hpk.evolving_schemas import (
    EvidenceV2,
    StrategyKeyV1,
    UpdateDecisionV1,
    build_evidence_v2,
    build_strategy_key,
)
from roboharn_evo.agent.hpk.evolving_store import (
    LoadedEvolvingSnapshot,
    load_evolving_snapshot,
)
from roboharn_evo.agent.hpk.promotion import PromotionPolicyV1
from roboharn_evo.agent.hpk.rollout_importer import (
    ROLLOUT_IMPORT_MANIFEST_V2_SCHEMA,
    AgentRolloutImporter,
    RolloutImportError,
    RolloutImportResult,
)
from roboharn_evo.agent.hpk.runtime_binding import (
    build_runtime_binding,
    validate_runtime_binding,
)
from roboharn_evo.agent.hpk.schemas import (
    AbstractEffectV1,
    HPKUnresolved,
    ConditionV1,
    EntryV1,
    GeometricStrategyV1,
    TaskStrategyV1,
    canonical_json_bytes,
    stable_content_id,
    validate_content_id,
)
from roboharn_evo.agent.hpk.semantic_knowledge import (
    semantic_object_from_mapping,
    validate_semantic_object,
    validate_semantic_reasoning,
)
from roboharn_evo.agent.hpk.snapshot_publisher import (
    HPKSnapshotPublicationError,
    PublishedChild,
    publish_child_snapshot,
)
from roboharn_evo.agent.hpk.strategy_extractor import (
    ExtractedHPKStrategy,
    extract_baseline_strategy,
    strategy_key_id_for,
)
from roboharn_evo.agent.hpk.updater import (
    HPKUpdater,
    EntryUpdateResult,
    strategy_key_for_entry,
    updater_policy_identity,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?Z$"
)


class EpisodeFinalizationError(RuntimeError):
    """The caller supplied an internally inconsistent finalization request."""


class RolloutImporterProtocol(Protocol):
    def import_files(
        self,
        *,
        manifest_path: str | Path,
        trace_path: str | Path,
        expected_manifest_sha256: str,
        expected_trace_sha256: str,
    ) -> RolloutImportResult: ...


SnapshotPublisher = Callable[..., PublishedChild]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise EpisodeFinalizationError(f"{label} is unavailable: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise EpisodeFinalizationError(f"{label} must be a non-symlink regular file")
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
            raise EpisodeFinalizationError(f"{label} changed while being read")
        data = b"".join(chunks)
        if len(data) != after.st_size:
            raise EpisodeFinalizationError(f"{label} changed size while being read")
        return data
    finally:
        os.close(descriptor)


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise EpisodeFinalizationError(f"{label} is not canonical JSON: {exc}") from exc


def _optional_sha256(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EpisodeFinalizationError(
            f"{label} must be a lowercase SHA-256 digest or null"
        )
    return value


def _timestamp(value: str) -> str:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise EpisodeFinalizationError("created_at must be a frozen UTC timestamp")
    return value


def _expected_episode_id(value: str | int) -> str | int:
    if isinstance(value, bool) or not (
        isinstance(value, int)
        and value >= 0
        or isinstance(value, str)
        and bool(value.strip())
    ):
        raise EpisodeFinalizationError(
            "expected_episode_id must be a non-negative integer or non-empty string"
        )
    return value


def _episode_control_key(value: str | int) -> str:
    return _sha256_bytes(
        canonical_json_bytes(
            {"expected_episode_id_type": type(value).__name__, "value": value}
        )
    )


def _public_episode_group_id(
    value: str | int,
    *,
    rollout_manifest_sha256: str | None,
    trace_sha256: str | None,
) -> str:
    del value
    return stable_content_id(
        "afkepisode",
        {
            "rollout_manifest_sha256": rollout_manifest_sha256,
            "trace_sha256": trace_sha256,
        },
    )


@dataclass(frozen=True, slots=True)
class SnapshotRef:
    """An explicit, hash-pinned loaded snapshot; never a mutable latest pointer."""

    manifest_path: Path
    manifest_sha256: str
    snapshot_id: str
    snapshot: LoadedEvolvingSnapshot

    @classmethod
    def from_loaded(cls, value: LoadedEvolvingSnapshot) -> "SnapshotRef":
        return cls(
            manifest_path=value.manifest_path,
            manifest_sha256=value.manifest_sha256,
            snapshot_id=value.snapshot_id,
            snapshot=value,
        )

    @classmethod
    def from_published(cls, value: PublishedChild) -> "SnapshotRef":
        return cls.from_loaded(value.snapshot)

    def reload(self) -> "SnapshotRef":
        loaded = load_evolving_snapshot(self.manifest_path, self.manifest_sha256)
        if loaded.snapshot_id != self.snapshot_id:
            raise EpisodeFinalizationError("pinned snapshot identity changed")
        return SnapshotRef.from_loaded(loaded)

    def identity(self) -> dict[str, str]:
        return {
            "snapshot_id": self.snapshot_id,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class TransitionStrategyContext:
    """Typed runtime context omitted from the private transition envelope.

    Candidate identity and coordinates deliberately do not occur here.  The
    context contributes only transferable condition/task semantics and one
    content-derived scene signature used to count independent support.
    """

    condition: ConditionV1
    task_strategy: TaskStrategyV1
    semantic_object: dict[str, str]
    reasoning: dict[str, Any]
    scene_signature: str
    confidence: float | None = None
    proposal_id: str | None = None
    proposed_geometric_strategy: GeometricStrategyV1 | None = None
    proposed_expected_effect: AbstractEffectV1 | None = None

    @classmethod
    def from_values(
        cls,
        *,
        condition: ConditionV1 | Mapping[str, Any],
        task_strategy: TaskStrategyV1 | Mapping[str, Any],
        semantic_object: Mapping[str, Any] | None = None,
        reasoning: Mapping[str, Any] | None = None,
        scene_signature: str,
        confidence: float | None = None,
        proposal_id: str | None = None,
        proposed_geometric_strategy: GeometricStrategyV1
        | Mapping[str, Any]
        | None = None,
        proposed_expected_effect: AbstractEffectV1 | Mapping[str, Any] | None = None,
    ) -> "TransitionStrategyContext":
        typed_condition = (
            condition
            if isinstance(condition, ConditionV1)
            else ConditionV1.from_dict(condition)
        )
        typed_task = (
            task_strategy
            if isinstance(task_strategy, TaskStrategyV1)
            else TaskStrategyV1.from_dict(task_strategy)
        )
        object_source = (
            semantic_object
            if semantic_object is not None
            else typed_condition["manipulated_object"]
        )
        typed_object = validate_semantic_object(
            semantic_object_from_mapping(object_source)
        )
        typed_reasoning = validate_semantic_reasoning(
            reasoning
            or {
                "observed_problem": "not available",
                "failure_analysis": "not available",
                "strategy_rationale": "not available",
                "causal_hypothesis": "not available",
                "expected_observation": "not available",
                "failure_condition": "not available",
                "source": "not available",
                "confidence": 0.0,
            }
        )
        if _SHA256_RE.fullmatch(str(scene_signature)) is None:
            raise EpisodeFinalizationError(
                "scene_signature must be a lowercase SHA-256 digest"
            )
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise EpisodeFinalizationError("confidence must be within [0, 1] or null")
        proposal_values = (
            proposal_id,
            proposed_geometric_strategy,
            proposed_expected_effect,
        )
        if any(item is not None for item in proposal_values) and not all(
            item is not None for item in proposal_values
        ):
            raise EpisodeFinalizationError(
                "proposal context requires ID, geometry, and expected effect together"
            )
        typed_geometry = None
        typed_effect = None
        if proposal_id is not None:
            validate_content_id(
                proposal_id,
                prefix="afkproposal",
                path="proposal_id",
            )
            typed_geometry = (
                proposed_geometric_strategy
                if isinstance(proposed_geometric_strategy, GeometricStrategyV1)
                else GeometricStrategyV1.from_dict(proposed_geometric_strategy)
            )
            typed_effect = (
                proposed_expected_effect
                if isinstance(proposed_expected_effect, AbstractEffectV1)
                else AbstractEffectV1.from_dict(proposed_expected_effect)
            )
        return cls(
            condition=typed_condition,
            task_strategy=typed_task,
            semantic_object=typed_object,
            reasoning=typed_reasoning,
            scene_signature=str(scene_signature),
            confidence=None if confidence is None else float(confidence),
            proposal_id=proposal_id,
            proposed_geometric_strategy=typed_geometry,
            proposed_expected_effect=typed_effect,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "condition": self.condition.to_dict(),
            "task_strategy": self.task_strategy.to_dict(),
            "semantic_object": copy.deepcopy(self.semantic_object),
            "reasoning": copy.deepcopy(self.reasoning),
            "scene_signature": self.scene_signature,
            "confidence": self.confidence,
        }
        if self.proposal_id is not None:
            assert self.proposed_geometric_strategy is not None
            assert self.proposed_expected_effect is not None
            payload["proposal_id"] = self.proposal_id
            payload["proposed_geometric_strategy"] = (
                self.proposed_geometric_strategy.to_dict()
            )
            payload["proposed_expected_effect"] = (
                self.proposed_expected_effect.expected_projection()
            )
        return payload


@dataclass(frozen=True, slots=True)
class FinalizationAbstention:
    code: str
    detail: str
    transition_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "transition_id": self.transition_id,
        }


@dataclass(frozen=True, slots=True)
class EpisodeFinalizationResult:
    episode_id: str
    status: str
    parent: SnapshotRef
    active: SnapshotRef
    child: SnapshotRef | None
    evidence_ids: tuple[str, ...]
    updated_entry_ids: tuple[str, ...]
    decision_ids: tuple[str, ...]
    abstentions: tuple[FinalizationAbstention, ...]

    @property
    def published(self) -> bool:
        return self.child is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "status": self.status,
            "parent": self.parent.identity(),
            "active": self.active.identity(),
            "child": None if self.child is None else self.child.identity(),
            "evidence_ids": list(self.evidence_ids),
            "updated_entry_ids": list(self.updated_entry_ids),
            "decision_ids": list(self.decision_ids),
            "abstentions": [value.to_dict() for value in self.abstentions],
        }


@dataclass(frozen=True, slots=True)
class _ResolvedAttempt:
    transition: ActionEffectTransitionV1
    context: TransitionStrategyContext
    strategy_key: StrategyKeyV1
    extracted: ExtractedHPKStrategy | None
    existing_entry_id: str | None


def _abstained(
    *,
    episode_id: str,
    parent: SnapshotRef,
    code: str,
    detail: str,
    inherited: Sequence[FinalizationAbstention] = (),
) -> EpisodeFinalizationResult:
    return EpisodeFinalizationResult(
        episode_id=episode_id,
        status="abstained",
        parent=parent,
        active=parent,
        child=None,
        evidence_ids=(),
        updated_entry_ids=(),
        decision_ids=(),
        abstentions=(*inherited, FinalizationAbstention(code, detail)),
    )


def _parent_evidence_store(parent: LoadedEvolvingSnapshot) -> EvidenceStore:
    public = {
        str(value.get("evidence_id")): EvidenceV2.from_dict(value)
        for value in parent.evidence_records
    }
    private: dict[str, ActionEffectTransitionV1] = {}
    for index, value in enumerate(parent.private_provenance_records):
        evidence_id = value.get("evidence_id")
        transition = value.get("private_transition")
        declared_sha = value.get("transition_digest")
        if not isinstance(evidence_id, str) or not isinstance(transition, Mapping):
            raise EpisodeFinalizationError(
                f"parent private provenance record {index} is not an evidence envelope"
            )
        typed = ActionEffectTransitionV1.from_dict(transition)
        if declared_sha != private_provenance_sha256(typed):
            raise EpisodeFinalizationError(
                f"parent private provenance hash mismatch for {evidence_id}"
            )
        private[evidence_id] = typed
    if set(public) != set(private):
        raise EpisodeFinalizationError(
            "parent public/private evidence members do not have identical IDs"
        )
    return EvidenceStore((public[key], private[key]) for key in sorted(public))


def _records_from_batch(
    batch: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def decode(raw: bytes, *, label: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for ordinal, line in enumerate(raw.splitlines(), 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise EpisodeFinalizationError(
                    f"{label} batch line {ordinal} is not an object"
                )
            result.append(value)
        return result

    return (
        decode(batch.public_jsonl, label="public evidence"),
        decode(batch.private_jsonl, label="private evidence"),
    )


def _merge_records(
    kind: str,
    old: Sequence[Mapping[str, Any]],
    new: Sequence[Mapping[str, Any]],
    identity_fields: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    merged: dict[str, dict[str, Any]] = {}
    for raw in (*old, *new):
        value = _json_copy(dict(raw), label=kind)
        identity = next(
            (
                str(value[field])
                for field in identity_fields
                if isinstance(value.get(field), str) and value[field]
            ),
            None,
        )
        if identity is None:
            raise EpisodeFinalizationError(f"{kind} record has no stable identity")
        previous = merged.get(identity)
        if previous is not None and previous != value:
            raise EpisodeFinalizationError(f"{kind} identity collision: {identity}")
        merged[identity] = value
    return tuple(merged[key] for key in sorted(merged))


def _promotion_receipt(
    decision: UpdateDecisionV1,
    *,
    policy: PromotionPolicyV1,
) -> dict[str, Any]:
    payload = {
        "schema": "roboharn_evo/hpk/promotion_decision/v2",
        "update_decision_id": decision.stable_id,
        "entry_id": decision["entry_id"],
        "from_lifecycle": decision["from_lifecycle"],
        "to_lifecycle": decision["to_lifecycle"],
        "promotion_policy_id": decision["promotion_policy_id"],
        "promotion_policy_config_sha256": decision["promotion_policy_config_sha256"],
        "reason_codes": decision["reason_codes"],
        "created_at": decision["created_at"],
        # EntryV1 predates development-only evolving snapshots and forces a
        # non-oracle agent_generated entry to claim formal eligibility.  The
        # SnapshotV2 policy receipt is the authoritative eligibility envelope
        # until EntryV2 can represent this distinction directly.
        "development_only": policy["development_only"],
        "formal_evaluation_eligible": policy["formal_evaluation_eligible"],
        "eligibility_authority": "snapshot_v2_promotion_policy",
        "legacy_entry_v1_eligibility_overridden": policy["development_only"],
    }
    payload["promotion_decision_id"] = stable_content_id("afkpromotion", payload)
    return payload


def _rejection_receipt(
    *, entry_id: str, evidence_id: str, reason_codes: Sequence[str], created_at: str
) -> dict[str, Any]:
    payload = {
        "schema": "roboharn_evo/hpk/rejected_evidence/v1",
        "entry_id": entry_id,
        "evidence_id": evidence_id,
        "reason_codes": sorted(set(str(value) for value in reason_codes)),
        "created_at": created_at,
    }
    payload["rejection_id"] = stable_content_id("afkrejection", payload)
    return payload


class HPKEpisodeFinalizer:
    """Sequential, idempotent finalizer for one explicit evolving parent."""

    def __init__(
        self,
        *,
        promotion_policy: PromotionPolicyV1 | Mapping[str, Any],
        policy_refs: Mapping[str, Any],
        destination_root: str | Path,
        runtime_source_identity: Mapping[str, Any],
        domain_ids: Sequence[str],
        config_sha256: str | None,
        model_identity_sha256: str | None,
        runtime_source_sha256: str | None,
        runtime_schema: str,
        importer: RolloutImporterProtocol | None = None,
        publisher: SnapshotPublisher = publish_child_snapshot,
    ) -> None:
        self._policy = (
            promotion_policy
            if isinstance(promotion_policy, PromotionPolicyV1)
            else PromotionPolicyV1.from_mapping(promotion_policy)
        )
        self._policy_refs = _json_copy(dict(policy_refs), label="policy_refs")
        if self._policy_refs.get("updater") != updater_policy_identity(
            self._policy,
            legacy=self._policy_refs.get("updater", {}).get("policy_id") == "afk_updater/v1",
        ):
            raise EpisodeFinalizationError(
                "updater policy reference does not match frozen updater semantics"
            )
        promotion_ref = self._policy_refs.get("promotion")
        if not isinstance(promotion_ref, Mapping) or dict(promotion_ref) != {
            "policy_id": self._policy.policy_id,
            "config_sha256": self._policy.config_sha256,
            "payload": self._policy.to_dict(),
        }:
            raise EpisodeFinalizationError(
                "promotion policy reference does not match the updater policy"
            )
        root = Path(destination_root).expanduser().absolute()
        if not root.is_absolute():
            raise EpisodeFinalizationError("destination_root must be absolute")
        self._destination_root = root
        self._runtime_source_identity = _json_copy(
            dict(runtime_source_identity), label="runtime_source_identity"
        )
        normalized_domains = tuple(sorted(set(str(value) for value in domain_ids)))
        if not normalized_domains or len(normalized_domains) != len(tuple(domain_ids)):
            raise EpisodeFinalizationError("domain_ids must be non-empty and unique")
        self._domain_ids = normalized_domains
        self._config_sha256 = _optional_sha256(config_sha256, label="config_sha256")
        self._model_identity_sha256 = _optional_sha256(
            model_identity_sha256, label="model_identity_sha256"
        )
        self._runtime_source_sha256 = _optional_sha256(
            runtime_source_sha256, label="runtime_source_sha256"
        )
        if not isinstance(runtime_schema, str) or not runtime_schema.strip():
            raise EpisodeFinalizationError("runtime_schema must be non-empty")
        self._runtime_schema = runtime_schema.strip()
        runtime_identity = self._runtime_source_identity.get("runtime_identity")
        source_identity = self._runtime_source_identity.get("source_identity")
        if not isinstance(runtime_identity, Mapping) or not isinstance(
            source_identity, Mapping
        ):
            raise EpisodeFinalizationError(
                "runtime_source_identity must contain runtime_identity and source_identity"
            )
        self._provenance_schema = str(runtime_identity.get("schema", "") or "").strip()
        self._provenance_manifest_sha256 = _optional_sha256(
            runtime_identity.get("manifest_sha256"),
            label="runtime_source_identity.runtime_identity.manifest_sha256",
        )
        if (
            not self._provenance_schema
            or self._provenance_manifest_sha256 is None
            or runtime_identity.get("effective_config_sha256") != self._config_sha256
            or runtime_identity.get("agent_service_identity_sha256")
            != self._model_identity_sha256
            or source_identity.get("runtime_tree_sha256") != self._runtime_source_sha256
        ):
            raise EpisodeFinalizationError(
                "runtime_source_identity disagrees with finalizer provenance hashes"
            )
        self._importer = importer or AgentRolloutImporter()
        self._publisher = publisher
        self._lock = threading.RLock()
        self._fingerprints: dict[str, str] = {}
        self._results: dict[str, EpisodeFinalizationResult] = {}

    def _request_fingerprint(
        self,
        *,
        parent: SnapshotRef,
        expected_episode_id: str | int,
        manifest_sha256: str,
        trace_sha256: str,
        expected_manifest_sha256: str,
        expected_trace_sha256: str,
        contexts: Mapping[str, TransitionStrategyContext],
        created_at: str,
        expected_runtime_binding: Mapping[str, Any] | None,
        writeback_allowed: bool,
    ) -> str:
        payload = {
            "parent": parent.identity(),
            "expected_episode_id_type": type(expected_episode_id).__name__,
            "expected_episode_id": expected_episode_id,
            "manifest_sha256": manifest_sha256,
            "trace_sha256": trace_sha256,
            "expected_manifest_sha256": expected_manifest_sha256,
            "expected_trace_sha256": expected_trace_sha256,
            "contexts": {key: contexts[key].to_dict() for key in sorted(contexts)},
            "created_at": created_at,
            "expected_runtime_binding": (
                None
                if expected_runtime_binding is None
                else dict(expected_runtime_binding)
            ),
            "writeback_allowed": writeback_allowed,
        }
        return _sha256_bytes(canonical_json_bytes(payload))

    def _resolve_attempt(
        self,
        *,
        transition: ActionEffectTransitionV1,
        context: TransitionStrategyContext,
        entries_by_id: Mapping[str, EntryV1],
    ) -> _ResolvedAttempt | FinalizationAbstention:
        if transition["condition_id"] != context.condition.stable_id:
            return FinalizationAbstention(
                "condition_identity_mismatch",
                "the pinned transition context does not match condition_id",
                transition.stable_id,
            )
        if transition["task_strategy_id"] != context.task_strategy.stable_id:
            return FinalizationAbstention(
                "task_strategy_identity_mismatch",
                "the pinned transition context does not match task_strategy_id",
                transition.stable_id,
            )
        selected = transition["selected_hpk_entry_id"]
        if selected is not None:
            entry = entries_by_id.get(str(selected))
            if entry is None:
                return FinalizationAbstention(
                    "selected_entry_not_in_parent",
                    "the selected HPK entry is absent from the pinned parent",
                    transition.stable_id,
                )
            key = strategy_key_for_entry(entry)
            return _ResolvedAttempt(transition, context, key, None, str(selected))

        if context.proposal_id is not None:
            geometry = context.proposed_geometric_strategy
            expected_effect = context.proposed_expected_effect
            if geometry is None or expected_effect is None:
                return FinalizationAbstention(
                    "proposal_strategy_context_incomplete",
                    "the VLM proposal context is incomplete",
                    transition.stable_id,
                )
            if transition["geometric_compliance"] is not True:
                return FinalizationAbstention(
                    "proposal_candidate_noncompliant",
                    "the executed candidate was not proven compliant with the proposal",
                    transition.stable_id,
                )
            if transition["geometric_strategy_id"] != geometry.stable_id:
                return FinalizationAbstention(
                    "proposal_geometry_identity_mismatch",
                    "the transition geometry differs from the scheduled proposal",
                    transition.stable_id,
                )
            transition_expected = transition["expected_effect"]
            if not isinstance(transition_expected, Mapping) or canonical_json_bytes(
                AbstractEffectV1.from_dict(transition_expected).expected_projection()
            ) != canonical_json_bytes(expected_effect.expected_projection()):
                return FinalizationAbstention(
                    "proposal_effect_identity_mismatch",
                    "the transition expected effect differs from the proposal",
                    transition.stable_id,
                )
            extracted = ExtractedHPKStrategy(
                condition=context.condition,
                task_strategy=context.task_strategy,
                geometric_strategy=geometry,
                expected_effect=expected_effect,
                strategy_key_id=strategy_key_id_for(
                    condition=context.condition,
                    task_strategy=context.task_strategy,
                    geometric_strategy=geometry,
                    expected_effect=expected_effect,
                ),
            )
            key = build_strategy_key(
                condition=extracted.condition,
                task_strategy=extracted.task_strategy,
                geometric_strategy=extracted.geometric_strategy,
                expected_effect=extracted.expected_effect,
            )
            return _ResolvedAttempt(transition, context, key, extracted, None)

        features = transition["candidate_geometry_features"]
        expected = transition["expected_effect"]
        if not isinstance(features, Mapping) or not isinstance(expected, Mapping):
            return FinalizationAbstention(
                "baseline_strategy_input_unresolved",
                "candidate features or expected effect are missing",
                transition.stable_id,
            )
        try:
            extracted = extract_baseline_strategy(
                condition=context.condition,
                task_strategy=context.task_strategy,
                candidate_features=features,
                expected_effect=expected,
                selected_hpk_entry_id=None,
                allow_oracle_geometry=False,
            )
        except Exception as exc:
            return FinalizationAbstention(
                "baseline_strategy_extraction_failed",
                f"typed strategy extraction failed: {type(exc).__name__}: {exc}",
                transition.stable_id,
            )
        if isinstance(extracted, HPKUnresolved):
            return FinalizationAbstention(
                "baseline_strategy_unresolved",
                extracted.reason,
                transition.stable_id,
            )
        bound_geometry_id = transition["geometric_strategy_id"]
        if (
            bound_geometry_id is not None
            and bound_geometry_id != extracted.geometric_strategy.stable_id
        ):
            return FinalizationAbstention(
                "geometric_strategy_identity_mismatch",
                "extracted baseline geometry does not match the transition binding",
                transition.stable_id,
            )
        key = build_strategy_key(
            condition=extracted.condition,
            task_strategy=extracted.task_strategy,
            geometric_strategy=extracted.geometric_strategy,
            expected_effect=extracted.expected_effect,
        )
        return _ResolvedAttempt(transition, context, key, extracted, None)

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
        expected_runtime_binding: Mapping[str, Any] | None = None,
        writeback_allowed: bool = True,
    ) -> EpisodeFinalizationResult:
        """Finalize one private episode binding and expose only its public group."""

        expected_episode_id = _expected_episode_id(expected_episode_id)
        if not isinstance(writeback_allowed, bool):
            raise EpisodeFinalizationError("writeback_allowed must be a boolean")
        expected_manifest_sha256 = _optional_sha256(
            expected_manifest_sha256,
            label="expected_manifest_sha256",
        )
        expected_trace_sha256 = _optional_sha256(
            expected_trace_sha256,
            label="expected_trace_sha256",
        )
        if expected_manifest_sha256 is None or expected_trace_sha256 is None:
            raise EpisodeFinalizationError(
                "episode finalization requires external manifest and trace hashes"
            )
        control_key = _episode_control_key(expected_episode_id)
        public_episode_id = _public_episode_group_id(
            expected_episode_id,
            rollout_manifest_sha256=None,
            trace_sha256=None,
        )
        created_at = _timestamp(created_at)
        caller_runtime_binding = None
        if expected_runtime_binding is not None:
            try:
                caller_runtime_binding = validate_runtime_binding(
                    expected_runtime_binding
                )
            except Exception as exc:
                raise EpisodeFinalizationError(
                    f"expected_runtime_binding is invalid: {type(exc).__name__}: {exc}"
                ) from exc
        contexts = dict(transition_contexts)
        if any(
            not isinstance(key, str) or not isinstance(value, TransitionStrategyContext)
            for key, value in contexts.items()
        ):
            raise EpisodeFinalizationError(
                "transition_contexts must map transition IDs to typed contexts"
            )
        with self._lock:
            try:
                pinned_parent = parent.reload()
            except Exception as exc:
                return _abstained(
                    episode_id=public_episode_id,
                    parent=parent,
                    code="parent_snapshot_invalid",
                    detail=f"pinned parent reload failed: {type(exc).__name__}: {exc}",
                )

            manifest_file = Path(manifest_path).expanduser().absolute()
            trace_file = Path(trace_path).expanduser().absolute()
            try:
                before_manifest = _read_regular_bytes(
                    manifest_file, label="rollout import manifest"
                )
                before_trace = _read_regular_bytes(trace_file, label="rollout trace")
            except Exception as exc:
                return _abstained(
                    episode_id=public_episode_id,
                    parent=pinned_parent,
                    code="rollout_source_unavailable",
                    detail=f"explicit rollout source read failed: {type(exc).__name__}: {exc}",
                )
            manifest_sha = _sha256_bytes(before_manifest)
            trace_sha = _sha256_bytes(before_trace)
            public_episode_id = _public_episode_group_id(
                expected_episode_id,
                rollout_manifest_sha256=manifest_sha,
                trace_sha256=trace_sha,
            )
            fingerprint = self._request_fingerprint(
                parent=pinned_parent,
                expected_episode_id=expected_episode_id,
                manifest_sha256=manifest_sha,
                trace_sha256=trace_sha,
                expected_manifest_sha256=expected_manifest_sha256,
                expected_trace_sha256=expected_trace_sha256,
                contexts=contexts,
                created_at=created_at,
                expected_runtime_binding=caller_runtime_binding,
                writeback_allowed=writeback_allowed,
            )
            previous_fingerprint = self._fingerprints.get(control_key)
            if previous_fingerprint is not None:
                if previous_fingerprint == fingerprint:
                    return self._results[control_key]
                return _abstained(
                    episode_id=public_episode_id,
                    parent=pinned_parent,
                    code="episode_replay_conflict",
                    detail="the episode was already finalized with different pinned input",
                )

            if (
                manifest_sha != expected_manifest_sha256
                or trace_sha != expected_trace_sha256
            ):
                result = _abstained(
                    episode_id=public_episode_id,
                    parent=pinned_parent,
                    code="rollout_source_identity_mismatch",
                    detail="explicit rollout source differs from its external hash pin",
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result

            try:
                imported = self._importer.import_files(
                    manifest_path=manifest_file,
                    trace_path=trace_file,
                    expected_manifest_sha256=expected_manifest_sha256,
                    expected_trace_sha256=expected_trace_sha256,
                )
                after_manifest = _read_regular_bytes(
                    manifest_file, label="rollout import manifest"
                )
                after_trace = _read_regular_bytes(trace_file, label="rollout trace")
                if after_manifest != before_manifest or after_trace != before_trace:
                    raise RolloutImportError(
                        "rollout import source changed during finalization"
                    )
            except Exception as exc:
                result = _abstained(
                    episode_id=public_episode_id,
                    parent=pinned_parent,
                    code="rollout_import_failed",
                    detail=f"explicit rollout import failed: {type(exc).__name__}: {exc}",
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result
            if (
                type(imported.episode.episode_id) is not type(expected_episode_id)
                or imported.episode.episode_id != expected_episode_id
            ):
                result = _abstained(
                    episode_id=public_episode_id,
                    parent=pinned_parent,
                    code="episode_identity_mismatch",
                    detail="the imported private episode does not match the expected ID",
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result
            if imported.trace_sha256 != trace_sha:
                result = _abstained(
                    episode_id=public_episode_id,
                    parent=pinned_parent,
                    code="trace_identity_mismatch",
                    detail="importer trace identity differs from the pinned source bytes",
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result
            if imported.rollout_manifest_sha256 != manifest_sha:
                result = _abstained(
                    episode_id=public_episode_id,
                    parent=pinned_parent,
                    code="manifest_identity_mismatch",
                    detail="importer manifest identity differs from pinned source bytes",
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result
            if imported.manifest_schema != ROLLOUT_IMPORT_MANIFEST_V2_SCHEMA:
                result = _abstained(
                    episode_id=public_episode_id,
                    parent=pinned_parent,
                    code="legacy_manifest_not_writeback_eligible",
                    detail=(
                        "evolving snapshot writeback requires a provenance-bound "
                        "rollout import manifest v2"
                    ),
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result
            if caller_runtime_binding is None:
                expected_binding = build_runtime_binding(
                    snapshot_id=pinned_parent.snapshot_id,
                    snapshot_manifest_sha256=pinned_parent.manifest_sha256,
                    policy_refs=self._policy_refs,
                    provenance_schema=self._provenance_schema,
                    provenance_manifest_sha256=self._provenance_manifest_sha256,
                    config_sha256=self._config_sha256,
                    model_identity_sha256=self._model_identity_sha256,
                    runtime_source_sha256=self._runtime_source_sha256,
                    evidence_runtime_schema=self._runtime_schema,
                )
            else:
                expected_binding = caller_runtime_binding
            expected_provenance = {
                "schema": self._provenance_schema,
                "manifest_sha256": self._provenance_manifest_sha256,
                "config_sha256": self._config_sha256,
                "model_identity_sha256": self._model_identity_sha256,
                "runtime_source_sha256": self._runtime_source_sha256,
                "evidence_runtime_schema": self._runtime_schema,
            }
            if (
                expected_binding["snapshot_ref"] != pinned_parent.identity()
                or expected_binding["policy_refs"] != self._policy_refs
                or expected_binding["provenance"] != expected_provenance
            ):
                result = _abstained(
                    episode_id=public_episode_id,
                    parent=pinned_parent,
                    code="finalizer_runtime_binding_mismatch",
                    detail=(
                        "caller runtime binding disagrees with the finalizer's "
                        "pinned parent, policies, model, configuration, or runtime source"
                    ),
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result
            if imported.runtime_binding is None or canonical_json_bytes(
                imported.runtime_binding
            ) != canonical_json_bytes(expected_binding):
                result = _abstained(
                    episode_id=public_episode_id,
                    parent=pinned_parent,
                    code="runtime_provenance_binding_mismatch",
                    detail=(
                        "rollout runtime binding does not match the pinned parent, "
                        "policies, model, configuration, and runtime source"
                    ),
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result

            episode_id = public_episode_id

            imported_abstentions = tuple(
                FinalizationAbstention(value.code, value.detail, value.event_ref)
                for value in imported.abstentions
            )
            if not (
                imported.episode.natural_episode_end
                and imported.episode.infrastructure_valid
                and imported.episode.label in {"valid_success", "valid_task_failure"}
            ):
                result = _abstained(
                    episode_id=episode_id,
                    parent=pinned_parent,
                    code="episode_not_valid_natural_end",
                    detail=(
                        f"classification={imported.episode.label}; "
                        f"reason={imported.episode.reason}"
                    ),
                    inherited=imported_abstentions,
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result
            if not writeback_allowed:
                result = _abstained(
                    episode_id=episode_id,
                    parent=pinned_parent,
                    code="sequential_writeback_disabled",
                    detail=(
                        "the sequential acceptance controller kept this episode "
                        "read-only and no descendant snapshot may be published"
                    ),
                    inherited=imported_abstentions,
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result
            if not any(
                transition["infrastructure_valid"]
                and transition["condition_id"] is not None
                and transition["task_strategy_id"] is not None
                for transition in imported.transitions
            ):
                result = _abstained(
                    episode_id=episode_id,
                    parent=pinned_parent,
                    code="no_update_eligible_action_transition",
                    detail="the valid episode has no update-eligible action transition",
                    inherited=imported_abstentions,
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result

            try:
                store = _parent_evidence_store(pinned_parent.snapshot)
                entries_by_id = {
                    str(entry["entry_id"]): entry
                    for entry in pinned_parent.snapshot.all_entries
                }
                entries_by_strategy = {
                    strategy_key_for_entry(entry).stable_id: entry
                    for entry in pinned_parent.snapshot.all_entries
                }
            except Exception as exc:
                result = _abstained(
                    episode_id=episode_id,
                    parent=pinned_parent,
                    code="parent_evidence_invalid",
                    detail=f"could not restore parent evidence: {type(exc).__name__}: {exc}",
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result

            abstentions = list(imported_abstentions)
            attempts: dict[str, list[_ResolvedAttempt]] = {}
            inserted_ids: list[str] = []
            for transition in imported.transitions:
                if not (
                    transition["infrastructure_valid"]
                    and transition["condition_id"] is not None
                    and transition["task_strategy_id"] is not None
                ):
                    abstentions.append(
                        FinalizationAbstention(
                            "transition_not_update_eligible",
                            "the transition lacks a valid typed strategy binding",
                            transition.stable_id,
                        )
                    )
                    continue
                context = contexts.get(transition.stable_id)
                if context is None:
                    abstentions.append(
                        FinalizationAbstention(
                            "transition_context_missing",
                            "no explicit typed context was supplied for this transition",
                            transition.stable_id,
                        )
                    )
                    continue
                resolved = self._resolve_attempt(
                    transition=transition,
                    context=context,
                    entries_by_id=entries_by_id,
                )
                if isinstance(resolved, FinalizationAbstention):
                    abstentions.append(resolved)
                    continue
                try:
                    evidence = build_evidence_v2(
                        transition,
                        strategy_key=resolved.strategy_key,
                        scene_signature=context.scene_signature,
                        rollout_manifest_sha256=manifest_sha,
                        trace_sha256=imported.trace_sha256,
                        config_sha256=self._config_sha256,
                        model_identity_sha256=self._model_identity_sha256,
                        runtime_source_sha256=self._runtime_source_sha256,
                        runtime_schema=self._runtime_schema,
                        created_at=created_at,
                        confidence=context.confidence,
                    )
                    insertion = store.with_evidence(evidence, transition)
                    store = insertion.store
                except Exception as exc:
                    abstentions.append(
                        FinalizationAbstention(
                            "evidence_construction_failed",
                            f"typed evidence rejected: {type(exc).__name__}: {exc}",
                            transition.stable_id,
                        )
                    )
                    continue
                if not insertion.inserted:
                    abstentions.append(
                        FinalizationAbstention(
                            "duplicate_evidence",
                            "the immutable transition is already present in the parent",
                            transition.stable_id,
                        )
                    )
                    continue
                inserted_ids.append(insertion.evidence_id)
                attempts.setdefault(resolved.strategy_key.stable_id, []).append(
                    resolved
                )

            if not inserted_ids:
                result = _abstained(
                    episode_id=episode_id,
                    parent=pinned_parent,
                    code="no_new_typed_evidence",
                    detail="no new evidence survived typed construction and deduplication",
                    inherited=abstentions,
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result

            updater = HPKUpdater(self._policy)
            updated_entries = dict(entries_by_id)
            decisions: list[UpdateDecisionV1] = []
            rejection_records: list[dict[str, Any]] = []
            revalidation_ids = set(
                pinned_parent.snapshot.manifest["entry_ids"]["revalidation"]
            )
            for strategy_key_id in sorted(attempts):
                group = attempts[strategy_key_id]
                same_key_evidence = store.records_for_strategy(strategy_key_id)
                existing = entries_by_strategy.get(strategy_key_id)
                try:
                    if existing is None:
                        extracted = next(
                            (
                                item.extracted
                                for item in group
                                if item.extracted is not None
                            ),
                            None,
                        )
                        if extracted is None:
                            raise EpisodeFinalizationError(
                                "selected entry strategy is absent from the parent"
                            )
                        if any(
                            item.transition["oracle_derived"]
                            or item.transition["expert_derived"]
                            for item in group
                        ):
                            raise EpisodeFinalizationError(
                                "baseline learned entries require agent-rollout evidence"
                            )
                        update = updater.create_candidate_entry(
                            condition=extracted.condition,
                            task_strategy=extracted.task_strategy,
                            geometric_strategy=extracted.geometric_strategy,
                            expected_effect=extracted.expected_effect,
                            provenance={
                                "source_kind": "agent_generated",
                                "expert_derived": False,
                                "oracle_derived": False,
                                "human_prior_used": False,
                                "learned_hpk": True,
                                "formal_evaluation_eligible": True,
                                "domain_ids": list(self._domain_ids),
                            },
                            acceptance_scope="formal_no_prior",
                            created_at=created_at,
                            evidence=same_key_evidence,
                        )
                    else:
                        current_lifecycle = (
                            "candidate_for_revalidation"
                            if str(existing["entry_id"]) in revalidation_ids
                            else None
                        )
                        marker = (
                            self._revalidation_marker(
                                pinned_parent.snapshot,
                                entry_id=str(existing["entry_id"]),
                            )
                            if current_lifecycle is not None
                            else None
                        )
                        update = updater.update_entry(
                            existing,
                            same_key_evidence,
                            current_lifecycle=current_lifecycle,
                            prior_decision=marker,
                        )
                except Exception as exc:
                    for item in group:
                        abstentions.append(
                            FinalizationAbstention(
                                "entry_update_failed",
                                f"typed updater rejected the strategy: {type(exc).__name__}: {exc}",
                                item.transition.stable_id,
                            )
                        )
                    continue

                self._collect_rejections(
                    update=update,
                    created_at=created_at,
                    target=rejection_records,
                )
                current_group_ids = {
                    store.evidence_for_transition(item.transition.stable_id).stable_id
                    for item in group
                    if store.evidence_for_transition(item.transition.stable_id)
                    is not None
                }
                if not current_group_ids & set(update.new_evidence_ids):
                    continue
                updated_entries[str(update.entry["entry_id"])] = update.entry
                entries_by_strategy[strategy_key_id] = update.entry
                decisions.append(update.decision)
                if update.decision["to_lifecycle"] == "candidate_for_revalidation":
                    revalidation_ids.add(str(update.entry["entry_id"]))
                else:
                    revalidation_ids.discard(str(update.entry["entry_id"]))

            if not decisions and not rejection_records:
                result = _abstained(
                    episode_id=episode_id,
                    parent=pinned_parent,
                    code="no_applicable_entry_update",
                    detail="new evidence did not produce an applicable entry update",
                    inherited=abstentions,
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result

            batch = store.to_batch()
            evidence_records, private_records = _records_from_batch(batch)
            update_records = _merge_records(
                "update_decisions",
                pinned_parent.snapshot.update_decision_records,
                [value.to_dict() for value in decisions],
                ("decision_id", "update_decision_id", "record_id"),
            )
            promotion_records = _merge_records(
                "promotion_decisions",
                pinned_parent.snapshot.promotion_decision_records,
                [_promotion_receipt(value, policy=self._policy) for value in decisions],
                ("promotion_decision_id", "decision_id", "record_id"),
            )
            rejected_records = _merge_records(
                "rejected_evidence",
                pinned_parent.snapshot.rejected_evidence_records,
                rejection_records,
                ("rejection_id", "evidence_id", "record_id"),
            )
            try:
                published = self._publisher(
                    parent=pinned_parent.snapshot,
                    expected_parent_snapshot_id=pinned_parent.snapshot_id,
                    expected_parent_manifest_sha256=pinned_parent.manifest_sha256,
                    entries=tuple(
                        updated_entries[key] for key in sorted(updated_entries)
                    ),
                    evidence_records=evidence_records,
                    private_provenance_records=private_records,
                    update_decision_records=update_records,
                    promotion_decision_records=promotion_records,
                    rejected_evidence_records=rejected_records,
                    revalidation_entry_ids=tuple(sorted(revalidation_ids)),
                    policy_refs=self._policy_refs,
                    evidence_batch=batch,
                    source_episode_ids=batch.source_episode_ids,
                    created_at=created_at,
                    destination_root=self._destination_root,
                    runtime_source_identity=self._runtime_source_identity,
                    purpose="evolving_update",
                )
            except HPKSnapshotPublicationError as exc:
                result = _abstained(
                    episode_id=episode_id,
                    parent=pinned_parent,
                    code="snapshot_publication_failed",
                    detail=f"parent retained after publication failure: {exc}",
                    inherited=abstentions,
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result
            except Exception as exc:
                result = _abstained(
                    episode_id=episode_id,
                    parent=pinned_parent,
                    code="snapshot_publication_failed",
                    detail=(
                        "parent retained after publisher exception: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    inherited=abstentions,
                )
                self._fingerprints[control_key] = fingerprint
                self._results[control_key] = result
                return result

            child = SnapshotRef.from_published(published)
            result = EpisodeFinalizationResult(
                episode_id=episode_id,
                status="published",
                parent=pinned_parent,
                active=child,
                child=child,
                evidence_ids=tuple(sorted(inserted_ids)),
                updated_entry_ids=tuple(
                    sorted(str(value["entry_id"]) for value in decisions)
                ),
                decision_ids=tuple(sorted(value.stable_id for value in decisions)),
                abstentions=tuple(abstentions),
            )
            self._fingerprints[control_key] = fingerprint
            self._results[control_key] = result
            return result

    @staticmethod
    def _revalidation_marker(
        snapshot: LoadedEvolvingSnapshot, *, entry_id: str
    ) -> UpdateDecisionV1:
        candidates = []
        for raw in snapshot.update_decision_records:
            try:
                decision = UpdateDecisionV1.from_dict(raw)
            except Exception:
                continue
            if (
                decision["entry_id"] == entry_id
                and decision["to_lifecycle"] == "candidate_for_revalidation"
            ):
                candidates.append(decision)
        if not candidates:
            raise EpisodeFinalizationError(
                "revalidation entry has no immutable UpdateDecision marker"
            )
        return sorted(
            candidates,
            key=lambda value: (str(value["created_at"]), value.stable_id),
        )[-1]

    @staticmethod
    def _collect_rejections(
        *,
        update: EntryUpdateResult,
        created_at: str,
        target: list[dict[str, Any]],
    ) -> None:
        for rejected in update.rejected_evidence:
            target.append(
                _rejection_receipt(
                    entry_id=str(update.entry["entry_id"]),
                    evidence_id=rejected.evidence_id,
                    reason_codes=rejected.reason_codes,
                    created_at=created_at,
                )
            )


__all__ = [
    "HPKEpisodeFinalizer",
    "EpisodeFinalizationError",
    "EpisodeFinalizationResult",
    "FinalizationAbstention",
    "RolloutImporterProtocol",
    "SnapshotRef",
    "TransitionStrategyContext",
]
