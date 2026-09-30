from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from roboharn_evo.agent.hpk.action_transition import (
    ActionEffectTransitionV1,
)
from roboharn_evo.agent.hpk.evolving_schemas import EvidenceV2, evidence_set_sha256
from roboharn_evo.agent.hpk.schemas import (
    HPKValidationError,
    AbstractEffectV1,
    canonical_json_bytes,
    stable_content_id,
)


class EvidenceStoreError(HPKValidationError):
    """Fail-closed evidence-store error with a stable machine code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


@dataclass(frozen=True, slots=True)
class EvidenceInsertResult:
    store: "EvidenceStore"
    inserted: bool
    evidence_id: str


@dataclass(frozen=True, slots=True)
class EvidenceBatch:
    """Deterministic public/private members ready for snapshot publication."""

    batch_id: str
    evidence_set_sha256: str
    public_member_sha256: str
    private_member_sha256: str
    record_count: int
    public_jsonl: bytes
    private_jsonl: bytes
    source_episode_ids: tuple[str, ...]


def _typed_evidence(value: EvidenceV2 | Mapping[str, Any]) -> EvidenceV2:
    return value if isinstance(value, EvidenceV2) else EvidenceV2.from_dict(value)


def _typed_private_transition(
    value: ActionEffectTransitionV1 | Mapping[str, Any],
) -> ActionEffectTransitionV1:
    return (
        value
        if isinstance(value, ActionEffectTransitionV1)
        else ActionEffectTransitionV1.from_dict(value)
    )


def private_provenance_sha256(
    value: ActionEffectTransitionV1 | Mapping[str, Any],
) -> str:
    transition = _typed_private_transition(value)
    return hashlib.sha256(canonical_json_bytes(transition.to_dict())).hexdigest()


def _strategy_binding_errors(
    evidence: EvidenceV2,
    transition: ActionEffectTransitionV1,
) -> list[str]:
    if evidence.strategy_key_id is None:
        return []
    expected_effect = transition["expected_effect"]
    expected_effect_id = (
        AbstractEffectV1.from_dict(expected_effect).stable_id
        if isinstance(expected_effect, Mapping)
        else None
    )
    comparisons = (
        ("condition_id", transition["condition_id"]),
        ("task_strategy_id", transition["task_strategy_id"]),
        ("geometric_strategy_id", transition["geometric_strategy_id"]),
        ("expected_effect_id", expected_effect_id),
    )
    return [
        name
        for name, transition_value in comparisons
        if transition_value is not None and evidence[name] != transition_value
    ]


def _validate_public_private_binding(
    evidence: EvidenceV2,
    transition: ActionEffectTransitionV1,
) -> None:
    mismatches: list[str] = []
    for name, public_value, private_value in (
        (
            "infrastructure_valid",
            evidence["infrastructure_valid"],
            transition["infrastructure_valid"],
        ),
        ("oracle_derived", evidence["oracle_derived"], transition["oracle_derived"]),
        ("expert_derived", evidence["expert_derived"], transition["expert_derived"]),
    ):
        if public_value != private_value:
            mismatches.append(name)
    mismatches.extend(_strategy_binding_errors(evidence, transition))
    private_verdict = str(transition["evidence_verdict"])
    private_reasons = tuple(transition["attribution_reasons"])
    public_verdict = str(evidence["verdict"])
    public_reasons = tuple(evidence["attribution_reasons"])
    if (public_verdict, public_reasons) != (private_verdict, private_reasons):
        baseline_resolution_allowed = bool(
            private_verdict == "unverified"
            and private_reasons == ("strategy_identity_unresolved",)
            and transition["geometric_strategy_id"] is None
            and evidence.strategy_key_id is not None
            and evidence["condition_id"] == transition["condition_id"]
            and evidence["task_strategy_id"] == transition["task_strategy_id"]
            and evidence["geometric_strategy_id"] is not None
            and public_reasons == ()
            and public_verdict in {"support", "oppose"}
        )
        expected = transition["expected_effect"]
        if baseline_resolution_allowed and isinstance(expected, Mapping):
            expected_id = AbstractEffectV1.from_dict(expected).stable_id
            baseline_resolution_allowed = bool(
                evidence["expected_effect_id"] == expected_id
                and evidence["observed_effect"] == transition["observed_effect"]
                and (
                    evidence["observed_effect"]["verifiability"] == "verified"
                    if public_verdict == "support"
                    else evidence["observed_effect"]["verifiability"] == "contradicted"
                )
            )
        if not baseline_resolution_allowed:
            mismatches.append("verdict_or_attribution_reasons")
    actual_private_sha = private_provenance_sha256(transition)
    if evidence["transition_digest"] != actual_private_sha:
        mismatches.append("transition_digest")
    expected_episode_group = stable_content_id(
        "afkepisode",
        {
            "rollout_manifest_sha256": evidence["source_identity"][
                "rollout_manifest_sha256"
            ],
            "trace_sha256": evidence["source_identity"]["trace_sha256"],
        },
    )
    if evidence["episode_evidence_group_id"] != expected_episode_group:
        mismatches.append("episode_evidence_group_id")
    if mismatches:
        raise EvidenceStoreError(
            "public_private_binding_mismatch",
            "public evidence disagrees with private transition: "
            + ", ".join(sorted(set(mismatches))),
        )


class EvidenceStore:
    """Persistent-value store with evidence and transition-level deduplication."""

    __slots__ = (
        "_public_by_id",
        "_private_by_id",
        "_evidence_by_transition",
        "_evidence_by_attempt",
    )

    def __init__(
        self,
        records: Iterable[
            tuple[
                EvidenceV2 | Mapping[str, Any],
                ActionEffectTransitionV1 | Mapping[str, Any],
            ]
        ] = (),
    ) -> None:
        public: dict[str, EvidenceV2] = {}
        private: dict[str, ActionEffectTransitionV1] = {}
        by_transition: dict[str, str] = {}
        by_attempt: dict[tuple[str, str], str] = {}
        for raw_evidence, raw_transition in records:
            evidence = _typed_evidence(raw_evidence)
            transition = _typed_private_transition(raw_transition)
            _validate_public_private_binding(evidence, transition)
            evidence_id = evidence.stable_id
            transition_id = str(transition["transition_id"])
            nonce = transition["action_attempt_nonce"]
            attempt_key = (
                (
                    str(transition["episode_id"]),
                    str(nonce),
                )
                if nonce is not None
                else None
            )
            existing = public.get(evidence_id)
            if existing is not None:
                if (
                    existing.to_dict() != evidence.to_dict()
                    or private[evidence_id].to_dict() != transition.to_dict()
                ):
                    raise EvidenceStoreError(
                        "evidence_id_collision",
                        f"{evidence_id} resolves to different immutable bytes",
                    )
                continue
            if transition_id in by_transition:
                raise EvidenceStoreError(
                    "duplicate_transition_conflict",
                    f"transition {transition_id} already produced evidence",
                )
            if attempt_key is not None and attempt_key in by_attempt:
                raise EvidenceStoreError(
                    "duplicate_action_attempt_conflict",
                    "one episode/action_attempt_nonce produced multiple evidence records",
                )
            public[evidence_id] = evidence
            private[evidence_id] = transition
            by_transition[transition_id] = evidence_id
            if attempt_key is not None:
                by_attempt[attempt_key] = evidence_id
        self._public_by_id = MappingProxyType(public)
        self._private_by_id = MappingProxyType(private)
        self._evidence_by_transition = MappingProxyType(by_transition)
        self._evidence_by_attempt = MappingProxyType(by_attempt)

    @classmethod
    def empty(cls) -> "EvidenceStore":
        return cls()

    def __len__(self) -> int:
        return len(self._public_by_id)

    def __iter__(self) -> Iterator[EvidenceV2]:
        for evidence_id in sorted(self._public_by_id):
            yield EvidenceV2.from_dict(self._public_by_id[evidence_id].to_dict())

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._public_by_id))

    @property
    def transition_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._evidence_by_transition))

    def get(self, evidence_id: str) -> EvidenceV2 | None:
        value = self._public_by_id.get(str(evidence_id))
        return None if value is None else EvidenceV2.from_dict(value.to_dict())

    def get_private_transition(
        self, evidence_id: str
    ) -> ActionEffectTransitionV1 | None:
        value = self._private_by_id.get(str(evidence_id))
        return (
            None
            if value is None
            else ActionEffectTransitionV1.from_dict(value.to_dict())
        )

    def evidence_for_transition(self, transition_id: str) -> EvidenceV2 | None:
        evidence_id = self._evidence_by_transition.get(str(transition_id))
        return None if evidence_id is None else self.get(evidence_id)

    def records_for_strategy(self, strategy_key_id: str) -> tuple[EvidenceV2, ...]:
        requested = str(strategy_key_id)
        return tuple(
            evidence for evidence in self if evidence.strategy_key_id == requested
        )

    def select(self, evidence_ids: Sequence[str]) -> tuple[EvidenceV2, ...]:
        requested = [str(value) for value in evidence_ids]
        if len(requested) != len(set(requested)):
            raise EvidenceStoreError(
                "duplicate_evidence_request",
                "requested evidence IDs must be unique",
            )
        missing = sorted(set(requested) - set(self._public_by_id))
        if missing:
            raise EvidenceStoreError(
                "evidence_ref_out_of_bounds",
                "missing evidence ID(s): " + ", ".join(missing),
            )
        return tuple(self.get(value) for value in sorted(requested))  # type: ignore[arg-type]

    def with_evidence(
        self,
        evidence: EvidenceV2 | Mapping[str, Any],
        private_transition: ActionEffectTransitionV1 | Mapping[str, Any],
    ) -> EvidenceInsertResult:
        typed_evidence = _typed_evidence(evidence)
        typed_transition = _typed_private_transition(private_transition)
        _validate_public_private_binding(typed_evidence, typed_transition)
        evidence_id = typed_evidence.stable_id
        existing = self._public_by_id.get(evidence_id)
        if existing is not None:
            if (
                existing.to_dict() != typed_evidence.to_dict()
                or self._private_by_id[evidence_id].to_dict()
                != typed_transition.to_dict()
            ):
                raise EvidenceStoreError(
                    "evidence_id_collision",
                    f"{evidence_id} resolves to different immutable bytes",
                )
            return EvidenceInsertResult(self, False, evidence_id)
        transition_id = str(typed_transition["transition_id"])
        if transition_id in self._evidence_by_transition:
            raise EvidenceStoreError(
                "duplicate_transition_conflict",
                f"transition {transition_id} already produced evidence",
            )
        nonce = typed_transition["action_attempt_nonce"]
        attempt_key = (
            (
                str(typed_transition["episode_id"]),
                str(nonce),
            )
            if nonce is not None
            else None
        )
        if attempt_key is not None and attempt_key in self._evidence_by_attempt:
            raise EvidenceStoreError(
                "duplicate_action_attempt_conflict",
                "one episode/action_attempt_nonce produced multiple evidence records",
            )
        records = [
            (self._public_by_id[key], self._private_by_id[key])
            for key in sorted(self._public_by_id)
        ]
        records.append((typed_evidence, typed_transition))
        return EvidenceInsertResult(EvidenceStore(records), True, evidence_id)

    def to_batch(self) -> EvidenceBatch:
        records = list(self)
        public_jsonl = b"".join(
            canonical_json_bytes(record.to_dict()) + b"\n" for record in records
        )
        private_lines: list[bytes] = []
        for record in records:
            transition = self._private_by_id[record.stable_id]
            envelope = {
                "evidence_id": record.stable_id,
                "transition_digest": private_provenance_sha256(transition),
                "private_transition": transition.to_dict(),
            }
            private_lines.append(canonical_json_bytes(envelope) + b"\n")
        private_jsonl = b"".join(private_lines)
        public_sha = hashlib.sha256(public_jsonl).hexdigest()
        private_sha = hashlib.sha256(private_jsonl).hexdigest()
        set_sha = evidence_set_sha256(records)
        batch_payload = {
            "evidence_set_sha256": set_sha,
            "public_member_sha256": public_sha,
            "private_member_sha256": private_sha,
            "record_count": len(records),
        }
        return EvidenceBatch(
            batch_id=stable_content_id("afkbatch", batch_payload),
            evidence_set_sha256=set_sha,
            public_member_sha256=public_sha,
            private_member_sha256=private_sha,
            record_count=len(records),
            public_jsonl=public_jsonl,
            private_jsonl=private_jsonl,
            source_episode_ids=tuple(
                sorted({str(record["episode_evidence_group_id"]) for record in records})
            ),
        )

    def to_records(
        self,
    ) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
        return tuple(
            (
                copy.deepcopy(self._public_by_id[key].to_dict()),
                copy.deepcopy(self._private_by_id[key].to_dict()),
            )
            for key in sorted(self._public_by_id)
        )


ImmutableEvidenceStore = EvidenceStore


__all__ = [
    "HPKValidationError",
    "EvidenceBatch",
    "EvidenceInsertResult",
    "EvidenceStore",
    "EvidenceStoreError",
    "ImmutableEvidenceStore",
    "private_provenance_sha256",
]
