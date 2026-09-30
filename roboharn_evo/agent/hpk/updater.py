from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from roboharn_evo.agent.hpk.evidence_store import EvidenceStore
from roboharn_evo.agent.hpk.evolving_schemas import (
    UPDATE_DECISION_SCHEMA,
    UPDATE_REASON_ORDER,
    EvidenceV2,
    StrategyKeyV1,
    UpdateDecisionV1,
    build_strategy_key,
    evidence_set_sha256,
    update_decision_id_for,
)
from roboharn_evo.agent.hpk.leakage_gate import validate_transferable_entry
from roboharn_evo.agent.hpk.promotion import (
    LCB_METHOD,
    LIFECYCLE_TRANSITION_PROFILE,
    PromotionPolicyV1,
    evaluate_lifecycle,
)
from roboharn_evo.agent.hpk.schemas import (
    ENTRY_SCHEMA,
    AbstractEffectV1,
    HPKValidationError,
    ConditionV1,
    EntryV1,
    GeometricStrategyV1,
    TaskStrategyV1,
    canonical_json_bytes,
    entry_id_for,
    stable_content_id,
)
from roboharn_evo.agent.hpk.semantic_knowledge import (
    validate_semantic_evidence,
    validate_semantic_knowledge,
)

UPDATER_POLICY_ID = "hpk_updater/v1"
UPDATER_POLICY_SCHEMA = "roboharn_evo/hpk/updater_policy/v1"


def updater_policy_payload(
    policy: PromotionPolicyV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the pure updater semantics derived from the pinned policy."""

    typed = (
        policy if isinstance(policy, PromotionPolicyV1) else PromotionPolicyV1(policy)
    )
    return {
        "schema": UPDATER_POLICY_SCHEMA,
        "policy_id": UPDATER_POLICY_ID,
        "posterior_method": "beta_bernoulli_verified_only/v1",
        "prior_alpha": typed["prior_alpha"],
        "prior_beta": typed["prior_beta"],
        "lcb_method": typed["lcb_method"],
        "lcb_delta": typed["lcb_delta"],
        "unverified_affects_posterior": False,
        "deduplication_identity": "immutable_evidence_id_and_transition/v1",
        "timestamp_rule": "max_parent_and_attached_evidence_created_at/v1",
        "lifecycle_transition_profile": LIFECYCLE_TRANSITION_PROFILE,
    }


def updater_policy_identity(
    policy: PromotionPolicyV1 | Mapping[str, Any],
    *,
    legacy: bool = False,
) -> dict[str, str]:
    payload = updater_policy_payload(policy)
    if legacy:
        payload["schema"] = "tcm/afk/updater_policy/v1"
        payload["policy_id"] = "afk_updater/v1"
    return {
        "policy_id": payload["policy_id"],
        "config_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class RejectedEvidence:
    evidence_id: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EntryUpdateResult:
    entry: EntryV1
    decision: UpdateDecisionV1
    attached_evidence_ids: tuple[str, ...]
    new_evidence_ids: tuple[str, ...]
    rejected_evidence: tuple[RejectedEvidence, ...]
    changed: bool


def strategy_key_for_entry(entry: EntryV1 | Mapping[str, Any]) -> StrategyKeyV1:
    typed = entry if isinstance(entry, EntryV1) else EntryV1.from_dict(entry)
    return build_strategy_key(
        condition=ConditionV1.from_dict(typed["condition"]),
        task_strategy=TaskStrategyV1.from_dict(typed["task_strategy"]),
        geometric_strategy=GeometricStrategyV1.from_dict(typed["geometric_strategy"]),
        expected_effect=AbstractEffectV1.from_dict(typed["expected_effect"]),
    )


def _typed_evidence_sequence(
    evidence: EvidenceStore | Sequence[EvidenceV2 | Mapping[str, Any]],
) -> tuple[EvidenceV2, ...]:
    if isinstance(evidence, EvidenceStore):
        return tuple(evidence)
    records = tuple(
        item if isinstance(item, EvidenceV2) else EvidenceV2.from_dict(item)
        for item in evidence
    )
    if len({record.stable_id for record in records}) != len(records):
        raise HPKValidationError("updater evidence contains duplicate evidence IDs")
    transitions = [str(record["transition_digest"]) for record in records]
    if len(transitions) != len(set(transitions)):
        raise HPKValidationError("updater evidence contains duplicate transitions")
    return records


def _existing_refs(entry: EntryV1) -> dict[str, list[str]]:
    refs = entry["evidence_refs"]
    return {
        "support": list(refs["supporting"]),
        "oppose": list(refs["opposing"]),
        "unverified": list(refs["unverified"]),
    }


def _record_policy_rejections(
    record: EvidenceV2,
    policy: PromotionPolicyV1,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if record["oracle_derived"] and not policy["allow_oracle_evidence"]:
        reasons.append("oracle_evidence_disallowed")
    if record["expert_derived"] and not policy["allow_expert_prior"]:
        reasons.append("expert_evidence_disallowed")
    if not record["infrastructure_valid"]:
        reasons.append("infrastructure_failure")
    return tuple(reasons)


def _timestamp_max(values: Sequence[str]) -> str:
    if not values:
        raise HPKValidationError("deterministic timestamp source must not be empty")
    parsed = [
        (datetime.fromisoformat(value[:-1] + "+00:00"), value) for value in values
    ]
    return max(parsed, key=lambda item: item[0])[1]


def _statistics(
    *,
    support_count: int,
    oppose_count: int,
    unverified_count: int,
    last_updated_at: str,
    policy: PromotionPolicyV1,
) -> dict[str, Any]:
    alpha = float(policy["prior_alpha"]) + support_count
    beta = float(policy["prior_beta"]) + oppose_count
    estimated = alpha / (alpha + beta)
    verified_count = support_count + oppose_count
    if policy["lcb_method"] != LCB_METHOD:
        raise HPKValidationError("unsupported LCB method")
    if verified_count == 0:
        lcb = 0.0
    else:
        empirical = support_count / verified_count
        radius = math.sqrt(
            math.log(1.0 / float(policy["lcb_delta"])) / (2.0 * verified_count)
        )
        lcb = max(0.0, empirical - radius)
    return {
        "support_count": support_count,
        "oppose_count": oppose_count,
        "unverified_count": unverified_count,
        "posterior_alpha": round(alpha, 12),
        "posterior_beta": round(beta, 12),
        "estimated_success_probability": round(estimated, 12),
        "lower_confidence_bound": round(lcb, 12),
        "last_updated_at": last_updated_at,
    }


def _evaluation_ref_payload(
    *,
    entry_id: str,
    from_lifecycle: str,
    to_lifecycle: str,
    policy: PromotionPolicyV1,
    evidence_sha256: str,
    statistics: Mapping[str, Any],
    reason_codes: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": "roboharn_evo/hpk/promotion_evaluation/v1",
        "entry_id": entry_id,
        "from_lifecycle": from_lifecycle,
        "to_lifecycle": to_lifecycle,
        "promotion_policy_id": policy.policy_id,
        "promotion_policy_config_sha256": policy.config_sha256,
        "evidence_set_sha256": evidence_sha256,
        "effect_statistics": copy.deepcopy(dict(statistics)),
        "reason_codes": list(reason_codes),
    }


def _decision_payload(
    *,
    entry_id: str,
    strategy_key_id: str,
    from_lifecycle: str,
    to_lifecycle: str,
    policy: PromotionPolicyV1,
    evidence: Sequence[EvidenceV2],
    new_evidence_ids: Sequence[str],
    statistics: Mapping[str, Any],
    counts: Mapping[str, int],
    distinct_episode_counts: Mapping[str, int],
    distinct_scene_signature_counts: Mapping[str, int],
    reason_codes: Sequence[str],
    evaluation_ref: str | None,
    created_at: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": UPDATE_DECISION_SCHEMA,
        "decision_id": "pending",
        "entry_id": entry_id,
        "strategy_key_id": strategy_key_id,
        "from_lifecycle": from_lifecycle,
        "to_lifecycle": to_lifecycle,
        "persisted_entry_status": (
            "candidate"
            if to_lifecycle == "candidate_for_revalidation"
            else to_lifecycle
        ),
        "promotion_policy_id": policy.policy_id,
        "promotion_policy_config_sha256": policy.config_sha256,
        "evidence_set_sha256": evidence_set_sha256(evidence),
        "evidence_ids": sorted(record.stable_id for record in evidence),
        "new_evidence_ids": sorted(str(value) for value in new_evidence_ids),
        "counts": dict(counts),
        "distinct_episode_counts": dict(distinct_episode_counts),
        "distinct_scene_signature_counts": dict(distinct_scene_signature_counts),
        "posterior_alpha": statistics["posterior_alpha"],
        "posterior_beta": statistics["posterior_beta"],
        "estimated_success_probability": statistics["estimated_success_probability"],
        "lower_confidence_bound": statistics["lower_confidence_bound"],
        "reason_codes": list(reason_codes),
        "evaluation_ref": evaluation_ref,
        "created_at": created_at,
    }
    payload["decision_id"] = update_decision_id_for(payload)
    return payload


class HPKUpdater:
    """Apply a complete immutable EvidenceV2 set to one stable EntryV1."""

    def __init__(self, policy: PromotionPolicyV1 | Mapping[str, Any]) -> None:
        self.policy = (
            policy
            if isinstance(policy, PromotionPolicyV1)
            else PromotionPolicyV1(policy)
        )
        self.updater_policy_identity = updater_policy_identity(self.policy)

    def update_entry(
        self,
        entry: EntryV1 | Mapping[str, Any],
        evidence: EvidenceStore | Sequence[EvidenceV2 | Mapping[str, Any]],
        *,
        current_lifecycle: str | None = None,
        prior_decision: UpdateDecisionV1 | Mapping[str, Any] | None = None,
    ) -> EntryUpdateResult:
        typed_entry = entry if isinstance(entry, EntryV1) else EntryV1.from_dict(entry)
        entry_gate = validate_transferable_entry(
            typed_entry,
            learned_entry=typed_entry["provenance"]["source_kind"] == "agent_generated",
        )
        entry_gate.require()
        expected_key = strategy_key_for_entry(typed_entry)
        supplied = _typed_evidence_sequence(evidence)
        supplied_by_id = {record.stable_id: record for record in supplied}
        refs_before = _existing_refs(typed_entry)
        existing_ids = {
            evidence_id for values in refs_before.values() for evidence_id in values
        }
        missing_existing = sorted(existing_ids - set(supplied_by_id))
        if missing_existing:
            raise HPKValidationError(
                "updater requires immutable records for all existing evidence refs: "
                + ", ".join(missing_existing)
            )

        accepted_records: list[EvidenceV2] = []
        rejected: list[RejectedEvidence] = []
        for record in sorted(supplied, key=lambda item: item.stable_id):
            if record.strategy_key_id != expected_key.stable_id:
                rejected.append(
                    RejectedEvidence(record.stable_id, ("strategy_identity_mismatch",))
                )
                continue
            policy_rejections = _record_policy_rejections(record, self.policy)
            if policy_rejections:
                rejected.append(RejectedEvidence(record.stable_id, policy_rejections))
                continue
            accepted_records.append(record)

        if existing_ids - {record.stable_id for record in accepted_records}:
            raise HPKValidationError(
                "an existing evidence ref became ineligible under the frozen policy"
            )
        accepted_by_id = {record.stable_id: record for record in accepted_records}
        refs_after = {name: set(values) for name, values in refs_before.items()}
        for record in accepted_records:
            refs_after[str(record["verdict"])].add(record.stable_id)
        all_ids = set().union(*refs_after.values())
        if all_ids != set(accepted_by_id):
            extras = sorted(set(accepted_by_id) - all_ids)
            # All eligible records passed explicitly to this updater are part
            # of this entry's evidence set; no silent evidence selection.
            for evidence_id in extras:
                record = accepted_by_id[evidence_id]
                refs_after[str(record["verdict"])].add(evidence_id)
            all_ids = set().union(*refs_after.values())
        attached_records = [
            accepted_by_id[evidence_id] for evidence_id in sorted(all_ids)
        ]
        new_ids = tuple(sorted(all_ids - existing_ids))
        timestamps = [str(typed_entry["effect_statistics"]["last_updated_at"])]
        timestamps.extend(
            str(accepted_by_id[evidence_id]["created_at"]) for evidence_id in new_ids
        )
        last_updated_at = _timestamp_max(timestamps)
        statistics = _statistics(
            support_count=len(refs_after["support"]),
            oppose_count=len(refs_after["oppose"]),
            unverified_count=len(refs_after["unverified"]),
            last_updated_at=last_updated_at,
            policy=self.policy,
        )
        lifecycle = evaluate_lifecycle(
            typed_entry,
            attached_records,
            policy=self.policy,
            lower_confidence_bound=float(statistics["lower_confidence_bound"]),
            current_lifecycle=current_lifecycle,
            prior_decision=prior_decision,
        )
        reasons = list(lifecycle.reason_codes)
        if rejected and "ineligible_evidence" not in reasons:
            reasons.append("ineligible_evidence")
        if not new_ids and "no_new_evidence" not in reasons:
            reasons.append("no_new_evidence")
        reasons = [value for value in UPDATE_REASON_ORDER if value in set(reasons)]
        evidence_sha = evidence_set_sha256(attached_records)
        evaluation_ref = None
        if lifecycle.to_lifecycle == "accepted":
            if (
                not new_ids
                and typed_entry["status"] == "accepted"
                and typed_entry["evaluation_status"] == "passed"
                and typed_entry["promotion_policy_id"] == self.policy.policy_id
            ):
                # Exact replay under the same frozen policy is a no-op.  Do
                # not mint a second evaluation identity for the same evidence.
                evaluation_ref = typed_entry["evaluation_ref"]
            else:
                evaluation_ref = stable_content_id(
                    "afkeval",
                    _evaluation_ref_payload(
                        entry_id=str(typed_entry["entry_id"]),
                        from_lifecycle=lifecycle.from_lifecycle,
                        to_lifecycle=lifecycle.to_lifecycle,
                        policy=self.policy,
                        evidence_sha256=evidence_sha,
                        statistics=statistics,
                        reason_codes=reasons,
                    ),
                )
        persisted_status = (
            "candidate"
            if lifecycle.to_lifecycle == "candidate_for_revalidation"
            else lifecycle.to_lifecycle
        )
        updated_payload = typed_entry.to_dict()
        updated_payload.update(
            {
                "status": persisted_status,
                "effect_statistics": statistics,
                "evidence_refs": {
                    "supporting": sorted(refs_after["support"]),
                    "opposing": sorted(refs_after["oppose"]),
                    "unverified": sorted(refs_after["unverified"]),
                },
                "promotion_policy_id": self.policy.policy_id,
                "evaluation_status": (
                    "passed"
                    if lifecycle.to_lifecycle == "accepted"
                    else "failed"
                    if lifecycle.to_lifecycle
                    in {"candidate_for_revalidation", "deprecated"}
                    else "not_evaluated"
                ),
                "evaluation_ref": evaluation_ref,
            }
        )
        # The stable entry identity excludes lifecycle/statistics/evidence.
        updated_payload["entry_id"] = entry_id_for(updated_payload)
        updated = EntryV1.from_dict(updated_payload)
        decision_payload = _decision_payload(
            entry_id=str(updated["entry_id"]),
            strategy_key_id=expected_key.stable_id,
            from_lifecycle=lifecycle.from_lifecycle,
            to_lifecycle=lifecycle.to_lifecycle,
            policy=self.policy,
            evidence=attached_records,
            new_evidence_ids=new_ids,
            statistics=statistics,
            counts=lifecycle.counts,
            distinct_episode_counts=lifecycle.distinct_episode_counts,
            distinct_scene_signature_counts=(lifecycle.distinct_scene_signature_counts),
            reason_codes=reasons,
            evaluation_ref=evaluation_ref,
            created_at=last_updated_at,
        )
        decision = UpdateDecisionV1.from_dict(decision_payload)
        changed = updated.to_dict() != typed_entry.to_dict()
        return EntryUpdateResult(
            entry=updated,
            decision=decision,
            attached_evidence_ids=tuple(sorted(all_ids)),
            new_evidence_ids=new_ids,
            rejected_evidence=tuple(rejected),
            changed=changed,
        )

    def create_candidate_entry(
        self,
        *,
        condition: ConditionV1 | Mapping[str, Any],
        task_strategy: TaskStrategyV1 | Mapping[str, Any],
        geometric_strategy: GeometricStrategyV1 | Mapping[str, Any],
        expected_effect: AbstractEffectV1 | Mapping[str, Any],
        provenance: Mapping[str, Any],
        acceptance_scope: str,
        created_at: str,
        evidence: EvidenceStore | Sequence[EvidenceV2 | Mapping[str, Any]] = (),
    ) -> EntryUpdateResult:
        typed_condition = (
            condition if isinstance(condition, ConditionV1) else ConditionV1(condition)
        )
        typed_task = (
            task_strategy
            if isinstance(task_strategy, TaskStrategyV1)
            else TaskStrategyV1(task_strategy)
        )
        typed_geometry = (
            geometric_strategy
            if isinstance(geometric_strategy, GeometricStrategyV1)
            else GeometricStrategyV1(geometric_strategy)
        )
        typed_effect = (
            expected_effect
            if isinstance(expected_effect, AbstractEffectV1)
            else AbstractEffectV1(expected_effect)
        )
        initial: dict[str, Any] = {
            "schema": ENTRY_SCHEMA,
            "entry_id": "pending",
            "status": "candidate",
            "condition": typed_condition.to_dict(),
            "task_strategy": typed_task.to_dict(),
            "geometric_strategy": typed_geometry.to_dict(),
            "expected_effect": typed_effect.expected_projection(),
            "effect_statistics": {
                "support_count": 0,
                "oppose_count": 0,
                "unverified_count": 0,
                "posterior_alpha": float(self.policy["prior_alpha"]),
                "posterior_beta": float(self.policy["prior_beta"]),
                "estimated_success_probability": float(self.policy["prior_alpha"])
                / (
                    float(self.policy["prior_alpha"]) + float(self.policy["prior_beta"])
                ),
                "lower_confidence_bound": 0.0,
                "last_updated_at": created_at,
            },
            "evidence_refs": {
                "supporting": [],
                "opposing": [],
                "unverified": [],
            },
            "promotion_policy_id": self.policy.policy_id,
            "evaluation_status": "not_evaluated",
            "evaluation_ref": None,
            "provenance": copy.deepcopy(dict(provenance)),
            "acceptance_scope": acceptance_scope,
        }
        initial["entry_id"] = entry_id_for(initial)
        candidate = EntryV1.from_dict(initial)
        return self.update_entry(candidate, evidence)


def update_entry(
    entry: EntryV1 | Mapping[str, Any],
    evidence: EvidenceStore | Sequence[EvidenceV2 | Mapping[str, Any]],
    *,
    policy: PromotionPolicyV1 | Mapping[str, Any],
    current_lifecycle: str | None = None,
    prior_decision: UpdateDecisionV1 | Mapping[str, Any] | None = None,
) -> EntryUpdateResult:
    return HPKUpdater(policy).update_entry(
        entry,
        evidence,
        current_lifecycle=current_lifecycle,
        prior_decision=prior_decision,
    )


def same_semantic_strategy(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    """Compare the knowledge-bearing fields directly, without an identity key."""

    left_value = validate_semantic_knowledge(left)
    right_value = validate_semantic_knowledge(right)
    fields = (
        "object",
        "condition",
        "task_strategy",
        "geometric_strategy",
        "expected_effect",
    )
    return all(left_value[field] == right_value[field] for field in fields)


def append_semantic_evidence(
    knowledge: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach direct semantic evidence and recompute counts without opaque refs."""

    current = validate_semantic_knowledge(knowledge)
    record = validate_semantic_evidence(evidence)
    for field in ("category", "color", "shape", "role"):
        expected = current["object"][field]
        observed = record["object"][field]
        if expected != observed and "unknown" not in {expected, observed}:
            raise HPKValidationError(
                f"evidence.object.{field}: does not describe the knowledge object"
            )
    if record not in current["evidence"]:
        current["evidence"].append(record)
    current["statistics"] = {
        verdict: sum(item["verdict"] == verdict for item in current["evidence"])
        for verdict in ("support", "oppose", "unverified")
    }
    return validate_semantic_knowledge(current)


def apply_semantic_promotion(
    knowledge: Mapping[str, Any],
    policy: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Apply only promotion conditions provable from the semantic record itself.

    The compact record intentionally has no episode or scene identifiers.  A
    policy requiring multiple distinct episodes or scenes therefore remains a
    candidate instead of inventing provenance.  The existing integration
    policy, which requires one deterministic support, remains usable.
    """

    current = validate_semantic_knowledge(knowledge)
    if current["status"] != "candidate":
        return current
    to_dict = getattr(policy, "to_dict", None)
    raw_policy = to_dict() if callable(to_dict) else policy
    if not isinstance(raw_policy, Mapping):
        raise HPKValidationError("semantic promotion policy must be an object")
    required = {
        "min_support_count",
        "max_oppose_count",
        "min_distinct_support_episodes",
        "min_distinct_scene_signatures",
    }
    missing = required - set(raw_policy)
    if missing:
        raise HPKValidationError(
            "semantic promotion policy is missing: " + ", ".join(sorted(missing))
        )
    if (
        int(raw_policy["min_distinct_support_episodes"]) > 1
        or int(raw_policy["min_distinct_scene_signatures"]) > 1
    ):
        return current
    if (
        current["statistics"]["support"]
        >= int(raw_policy["min_support_count"])
        and current["statistics"]["oppose"]
        <= int(raw_policy["max_oppose_count"])
    ):
        current["status"] = "accepted"
    return validate_semantic_knowledge(current)


__all__ = [
    "UPDATER_POLICY_ID",
    "UPDATER_POLICY_SCHEMA",
    "HPKUpdater",
    "EntryUpdateResult",
    "RejectedEvidence",
    "append_semantic_evidence",
    "apply_semantic_promotion",
    "same_semantic_strategy",
    "strategy_key_for_entry",
    "update_entry",
    "updater_policy_identity",
    "updater_policy_payload",
]
