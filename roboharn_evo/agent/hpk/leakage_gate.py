from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from roboharn_evo.agent.hpk.evolving_schemas import EvidenceV2
from roboharn_evo.agent.hpk.promotion import PromotionPolicyV1
from roboharn_evo.agent.hpk.schemas import (
    HPKValidationError,
    EntryV1,
    reject_private_transferable,
)


GATE_REASON_ORDER = (
    "private_runtime_data_leakage",
    "seed_specific_answer",
    "benchmark_hidden_state",
    "oracle_evidence_disallowed",
    "expert_evidence_disallowed",
    "task_name_hardcoded_guidance",
    "fixed_arm_transfer",
    "unsupported_semantic_part",
    "unsupported_geometry_relation",
    "missing_effect_evidence",
    "motion_not_completed",
    "realization_not_satisfied",
    "geometric_noncompliance",
    "verifier_conflict",
    "no_fresh_post_observation",
    "infrastructure_failure",
    "evidence_ref_out_of_bounds",
    "duplicate_transition",
    "strategy_identity_mismatch",
    "schema_capability_incompatible",
    "single_episode_overfit",
    "scene_condition_contradicted",
    "model_free_text_mutation",
    "illegal_lifecycle_transition",
    "unverified_not_promotable",
)
GATE_REASONS = frozenset(GATE_REASON_ORDER)

_PRIVATE_KEY_RE = re.compile(
    r"(?:^|_)(?:candidate|track|instance)(?:_|$).*(?:id|ref)(?:_|$)|"
    r"(?:^|_)selected_candidate(?:_|$)|"
    r"(?:^|_)(?:pose|quaternion|quat|se3|world_m|absolute_xyz|coordinate)(?:_|$)",
    re.IGNORECASE,
)
_SEED_KEY_RE = re.compile(r"(?:^|_)seed(?:_|$)", re.IGNORECASE)
_HIDDEN_KEY_RE = re.compile(
    r"(?:hidden_state|correct_combination|ground_truth|oracle_answer|episode_answer)",
    re.IGNORECASE,
)
_MODEL_MUTATION_KEY_RE = re.compile(
    r"(?:model_patch|free_text_update|llm_rewrite|reflector_output|feedback_policy)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class GateResult:
    accepted: bool
    reason_codes: tuple[str, ...]

    def require(self) -> None:
        if not self.accepted:
            raise HPKValidationError(
                "HPK gate rejected input: " + ", ".join(self.reason_codes)
            )


def _ordered(values: Sequence[str]) -> tuple[str, ...]:
    supplied = set(values)
    unknown = supplied - GATE_REASONS
    if unknown:
        raise ValueError("unknown HPK gate reason(s): " + ", ".join(sorted(unknown)))
    return tuple(reason for reason in GATE_REASON_ORDER if reason in supplied)


def _result(reasons: Sequence[str]) -> GateResult:
    ordered = _ordered(reasons)
    return GateResult(accepted=not ordered, reason_codes=ordered)


def _preflight_mapping_reasons(value: Any) -> list[str]:
    reasons: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                key = str(raw_key)
                if _PRIVATE_KEY_RE.search(key):
                    reasons.append("private_runtime_data_leakage")
                if _SEED_KEY_RE.search(key):
                    reasons.append("seed_specific_answer")
                if _HIDDEN_KEY_RE.search(key):
                    reasons.append("benchmark_hidden_state")
                if _MODEL_MUTATION_KEY_RE.search(key):
                    reasons.append("model_free_text_mutation")
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        # String privacy is enforced by each strict transferable schema.  A
        # generic scan here would misclassify structural content IDs such as
        # afkc_/afkz_/afkev_ as free-text leakage.

    visit(value)
    return reasons


def validate_public_evidence(
    value: EvidenceV2 | Mapping[str, Any],
    *,
    allow_oracle_evidence: bool,
    allow_expert_prior: bool,
    for_promotion: bool = True,
) -> GateResult:
    """Validate one public evidence record without inspecting private payload."""

    raw = value.to_dict() if isinstance(value, EvidenceV2) else value
    reasons = _preflight_mapping_reasons(raw)
    try:
        evidence = (
            value if isinstance(value, EvidenceV2) else EvidenceV2.from_dict(value)
        )
    except (TypeError, ValueError):
        reasons.append("schema_capability_incompatible")
        return _result(reasons)
    if evidence["oracle_derived"] and not allow_oracle_evidence:
        reasons.append("oracle_evidence_disallowed")
    if evidence["expert_derived"] and not allow_expert_prior:
        reasons.append("expert_evidence_disallowed")
    if not evidence["infrastructure_valid"]:
        reasons.append("infrastructure_failure")
    if evidence["motion_status"] != "completed":
        reasons.append("motion_not_completed")
    if evidence["realization_status"] != "satisfied":
        reasons.append("realization_not_satisfied")
    if evidence["geometric_compliance"] is not True:
        reasons.append("geometric_noncompliance")
    if evidence["verifier_conflict"]:
        reasons.append("verifier_conflict")
    if not evidence["fresh_post_observation"]:
        reasons.append("no_fresh_post_observation")
    if evidence.strategy_key_id is None:
        reasons.append("strategy_identity_mismatch")
    if evidence["observed_effect"]["verifiability"] == "unverified":
        reasons.append("missing_effect_evidence")
    if for_promotion and evidence["verdict"] == "unverified":
        reasons.append("unverified_not_promotable")
    return _result(reasons)


def validate_transferable_entry(
    value: EntryV1 | Mapping[str, Any],
    *,
    learned_entry: bool,
    current_condition_id: str | None = None,
) -> GateResult:
    """Validate one typed entry without task-specific repair or inference."""

    raw = value.to_dict() if isinstance(value, EntryV1) else value
    reasons = _preflight_mapping_reasons(raw)
    try:
        entry = value if isinstance(value, EntryV1) else EntryV1.from_dict(value)
        reject_private_transferable(entry.to_dict(), path="EntryV1")
    except (TypeError, ValueError):
        reasons.append("schema_capability_incompatible")
        return _result(reasons)
    task_family = str(entry["condition"]["task_family"]).strip().casefold()
    guidance = " ".join(
        str(entry["task_strategy"].get(key, "") or "") for key in ("subgoal_purpose",)
    ).casefold()
    source_text = str(
        entry["task_strategy"]["source"].get("planner_subtask_text", "") or ""
    ).casefold()
    if task_family and (task_family in guidance or task_family in source_text):
        reasons.append("task_name_hardcoded_guidance")
    if learned_entry and entry["task_strategy"]["preferred_arm"] != "either":
        reasons.append("fixed_arm_transfer")
    grasp = entry["geometric_strategy"]["grasp"]
    capability = entry["geometric_strategy"]["capability_evidence"]
    if grasp["region"] == "semantic_part" and (
        not grasp["semantic_part"] or not capability["semantic_part_observed"]
    ):
        reasons.append("unsupported_semantic_part")
    relation = entry["geometric_strategy"]["target_relation"]["relation"]
    if relation not in {None, "center_of"}:
        reasons.append("unsupported_geometry_relation")
    if learned_entry and not entry["expected_effect"]["expected_predicates"]:
        reasons.append("missing_effect_evidence")
    if current_condition_id is not None:
        condition_id = (
            getattr(
                __import__("roboharn_evo.agent.hpk.schemas", fromlist=["ConditionV1"]),
                "ConditionV1",
            )
            .from_dict(entry["condition"])
            .stable_id
        )
        if condition_id != current_condition_id:
            reasons.append("scene_condition_contradicted")
    return _result(reasons)


def validate_entry_evidence_binding(
    entry: EntryV1 | Mapping[str, Any],
    evidence: Sequence[EvidenceV2 | Mapping[str, Any]],
) -> GateResult:
    typed_entry = entry if isinstance(entry, EntryV1) else EntryV1.from_dict(entry)
    records: list[EvidenceV2] = []
    reasons: list[str] = []
    try:
        records = [
            item if isinstance(item, EvidenceV2) else EvidenceV2.from_dict(item)
            for item in evidence
        ]
    except (TypeError, ValueError):
        return _result(["schema_capability_incompatible"])
    ids = [record.stable_id for record in records]
    if len(ids) != len(set(ids)):
        reasons.append("duplicate_transition")
    transitions = [str(record["transition_digest"]) for record in records]
    if len(transitions) != len(set(transitions)):
        reasons.append("duplicate_transition")
    from roboharn_evo.agent.hpk.evolving_schemas import build_strategy_key
    from roboharn_evo.agent.hpk.schemas import (
        AbstractEffectV1,
        ConditionV1,
        GeometricStrategyV1,
        TaskStrategyV1,
    )

    expected_key = build_strategy_key(
        condition=ConditionV1.from_dict(typed_entry["condition"]),
        task_strategy=TaskStrategyV1.from_dict(typed_entry["task_strategy"]),
        geometric_strategy=GeometricStrategyV1.from_dict(
            typed_entry["geometric_strategy"]
        ),
        expected_effect=AbstractEffectV1.from_dict(typed_entry["expected_effect"]),
    )
    for record in records:
        if record.strategy_key_id != expected_key.stable_id:
            reasons.append("strategy_identity_mismatch")
    entry_refs = {
        str(ref) for values in typed_entry["evidence_refs"].values() for ref in values
    }
    supplied = set(ids)
    if not entry_refs <= supplied:
        reasons.append("evidence_ref_out_of_bounds")
    return _result(reasons)


def validate_promotion_inputs(
    entry: EntryV1 | Mapping[str, Any],
    evidence: Sequence[EvidenceV2 | Mapping[str, Any]],
    *,
    policy: PromotionPolicyV1 | Mapping[str, Any],
) -> GateResult:
    typed_policy = (
        policy if isinstance(policy, PromotionPolicyV1) else PromotionPolicyV1(policy)
    )
    reasons = list(validate_transferable_entry(entry, learned_entry=True).reason_codes)
    binding = validate_entry_evidence_binding(entry, evidence)
    reasons.extend(binding.reason_codes)
    records: list[EvidenceV2] = []
    for item in evidence:
        try:
            record = (
                item if isinstance(item, EvidenceV2) else EvidenceV2.from_dict(item)
            )
        except (TypeError, ValueError):
            reasons.append("schema_capability_incompatible")
            continue
        records.append(record)
        evidence_gate = validate_public_evidence(
            record,
            allow_oracle_evidence=bool(typed_policy["allow_oracle_evidence"]),
            allow_expert_prior=bool(typed_policy["allow_expert_prior"]),
            for_promotion=True,
        )
        reasons.extend(evidence_gate.reason_codes)
    support_episodes = {
        str(record["episode_evidence_group_id"])
        for record in records
        if record["verdict"] == "support"
    }
    support_scenes = {
        str(record["source_identity"]["scene_signature_sha256"])
        for record in records
        if record["verdict"] == "support"
    }
    if len(support_episodes) < int(typed_policy["min_distinct_support_episodes"]):
        reasons.append("single_episode_overfit")
    if len(support_scenes) < int(typed_policy["min_distinct_scene_signatures"]):
        reasons.append("single_episode_overfit")
    return _result(reasons)


class HPKLeakageGate:
    validate_public_evidence = staticmethod(validate_public_evidence)
    validate_transferable_entry = staticmethod(validate_transferable_entry)
    validate_entry_evidence_binding = staticmethod(validate_entry_evidence_binding)
    validate_promotion_inputs = staticmethod(validate_promotion_inputs)


__all__ = [
    "HPKLeakageGate",
    "GATE_REASON_ORDER",
    "GATE_REASONS",
    "GateResult",
    "validate_entry_evidence_binding",
    "validate_promotion_inputs",
    "validate_public_evidence",
    "validate_transferable_entry",
]
