from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from roboharn_evo.agent.hpk.evolving_schemas import (
    LIFECYCLE_STATES,
    UPDATE_REASON_ORDER,
    EvidenceV2,
    UpdateDecisionV1,
)
from roboharn_evo.agent.hpk.schemas import HPKValidationError, EntryV1, canonical_json_bytes


PROMOTION_POLICY_SCHEMA = "roboharn_evo/hpk/promotion_policy/v1"
LCB_METHOD = "hoeffding_v1"
LIFECYCLE_TRANSITION_PROFILE = "monotonic_revalidation_to_deprecated/v1"

_POLICY_FIELDS = frozenset(
    {
        "schema",
        "policy_id",
        "development_only",
        "formal_evaluation_eligible",
        "prior_alpha",
        "prior_beta",
        "lcb_method",
        "lcb_delta",
        "min_distinct_support_episodes",
        "min_support_count",
        "max_oppose_count",
        "accept_lcb_threshold",
        "demote_lcb_threshold",
        "min_distinct_scene_signatures",
        "min_oppose_count_for_revalidation",
        "min_distinct_oppose_episodes_for_revalidation",
        "min_oppose_count_for_deprecation",
        "min_distinct_oppose_episodes_for_deprecation",
        "allow_expert_prior",
        "allow_oracle_evidence",
    }
)
_POLICY_ID_RE = re.compile(r"^(?:hpk|afk)_[a-z0-9_]+/v[1-9][0-9]*$")


def _fail(path: str, message: str) -> None:
    raise HPKValidationError(f"{path}: {message}")


def _strict_bool(value: Any, *, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")
    return value


def _integer(value: Any, *, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    *,
    path: str,
    minimum: float,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a number")
    result = float(value)
    if not (float("-inf") < result < float("inf")):
        _fail(path, "must be finite")
    if result < minimum or (strict_minimum and result == minimum):
        operator = ">" if strict_minimum else ">="
        _fail(path, f"must be {operator} {minimum}")
    if maximum is not None and result > maximum:
        _fail(path, f"must be <= {maximum}")
    return result


@dataclass(frozen=True, slots=True)
class PromotionPolicyV1:
    """Strict semantic policy constructed from one already loaded Mapping."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            _fail("PromotionPolicyV1", "must be constructed from a Mapping")
        data = copy.deepcopy(dict(self.payload))
        canonical_json_bytes(data)
        missing = sorted(_POLICY_FIELDS - set(data))
        unknown = sorted(set(data) - _POLICY_FIELDS)
        if missing:
            _fail("PromotionPolicyV1", "missing field(s): " + ", ".join(missing))
        if unknown:
            _fail("PromotionPolicyV1", "unknown field(s): " + ", ".join(unknown))
        if data["schema"] not in {PROMOTION_POLICY_SCHEMA, "tcm/afk/promotion_policy/v1"}:
            _fail(
                "PromotionPolicyV1.schema",
                f"must equal {PROMOTION_POLICY_SCHEMA!r}",
            )
        policy_id = data["policy_id"]
        if not isinstance(policy_id, str) or _POLICY_ID_RE.fullmatch(policy_id) is None:
            _fail("PromotionPolicyV1.policy_id", "must be versioned hpk_<name>/v<N>")
        development = _strict_bool(
            data["development_only"], path="PromotionPolicyV1.development_only"
        )
        formal = _strict_bool(
            data["formal_evaluation_eligible"],
            path="PromotionPolicyV1.formal_evaluation_eligible",
        )
        if development == formal:
            _fail(
                "PromotionPolicyV1",
                "exactly one of development_only and formal_evaluation_eligible must be true",
            )
        _number(
            data["prior_alpha"],
            path="PromotionPolicyV1.prior_alpha",
            minimum=0.0,
            strict_minimum=True,
        )
        _number(
            data["prior_beta"],
            path="PromotionPolicyV1.prior_beta",
            minimum=0.0,
            strict_minimum=True,
        )
        if data["lcb_method"] != LCB_METHOD:
            _fail(
                "PromotionPolicyV1.lcb_method",
                f"must equal {LCB_METHOD!r}",
            )
        delta = _number(
            data["lcb_delta"],
            path="PromotionPolicyV1.lcb_delta",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
        )
        if delta >= 1.0:
            _fail("PromotionPolicyV1.lcb_delta", "must be < 1")
        integer_fields = (
            "min_distinct_support_episodes",
            "min_support_count",
            "max_oppose_count",
            "min_distinct_scene_signatures",
            "min_oppose_count_for_revalidation",
            "min_distinct_oppose_episodes_for_revalidation",
            "min_oppose_count_for_deprecation",
            "min_distinct_oppose_episodes_for_deprecation",
        )
        for name in integer_fields:
            _integer(data[name], path=f"PromotionPolicyV1.{name}")
        for name in (
            "min_distinct_support_episodes",
            "min_support_count",
            "min_distinct_scene_signatures",
        ):
            if int(data[name]) < 1:
                _fail(f"PromotionPolicyV1.{name}", "must be >= 1")
        for name in (
            "min_oppose_count_for_revalidation",
            "min_distinct_oppose_episodes_for_revalidation",
            "min_oppose_count_for_deprecation",
            "min_distinct_oppose_episodes_for_deprecation",
        ):
            if int(data[name]) < 1:
                _fail(f"PromotionPolicyV1.{name}", "must be >= 1")
        accept = _number(
            data["accept_lcb_threshold"],
            path="PromotionPolicyV1.accept_lcb_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        demote = _number(
            data["demote_lcb_threshold"],
            path="PromotionPolicyV1.demote_lcb_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        if demote > accept:
            _fail(
                "PromotionPolicyV1.demote_lcb_threshold",
                "must be <= accept_lcb_threshold",
            )
        for name in ("allow_expert_prior", "allow_oracle_evidence"):
            _strict_bool(data[name], path=f"PromotionPolicyV1.{name}")
        if formal:
            if data["allow_expert_prior"] or data["allow_oracle_evidence"]:
                _fail(
                    "PromotionPolicyV1",
                    "formal no-prior policy cannot allow expert or oracle evidence",
                )
            if int(data["min_distinct_support_episodes"]) < 2:
                _fail(
                    "PromotionPolicyV1.min_distinct_support_episodes",
                    "formal policy requires multiple independent episodes",
                )
            if int(data["min_distinct_scene_signatures"]) < 2:
                _fail(
                    "PromotionPolicyV1.min_distinct_scene_signatures",
                    "formal policy requires multiple scene signatures",
                )
            if int(data["min_oppose_count_for_revalidation"]) < 2:
                _fail(
                    "PromotionPolicyV1.min_oppose_count_for_revalidation",
                    "formal policy cannot demote on one opposition",
                )
            if int(data["min_distinct_oppose_episodes_for_revalidation"]) < 2:
                _fail(
                    "PromotionPolicyV1.min_distinct_oppose_episodes_for_revalidation",
                    "formal policy cannot demote on one episode",
                )
        object.__setattr__(self, "payload", MappingProxy(data))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PromotionPolicyV1":
        return cls(value)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.payload))

    def __getitem__(self, key: str) -> Any:
        return copy.deepcopy(self.payload[key])

    @property
    def policy_id(self) -> str:
        return str(self.payload["policy_id"])

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


class MappingProxy(Mapping[str, Any]):
    """Deep-copying immutable mapping used inside the frozen dataclass."""

    __slots__ = ("_data",)

    def __init__(self, value: Mapping[str, Any]) -> None:
        self._data = copy.deepcopy(dict(value))

    def __getitem__(self, key: str) -> Any:
        return copy.deepcopy(self._data[key])

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


@dataclass(frozen=True, slots=True)
class LifecycleEvaluation:
    from_lifecycle: str
    to_lifecycle: str
    reason_codes: tuple[str, ...]
    counts: Mapping[str, int]
    distinct_episode_counts: Mapping[str, int]
    distinct_scene_signature_counts: Mapping[str, int]


def _ordered_reasons(values: Sequence[str]) -> tuple[str, ...]:
    supplied = set(values)
    unknown = supplied - set(UPDATE_REASON_ORDER)
    if unknown:
        raise ValueError("unknown update reason(s): " + ", ".join(sorted(unknown)))
    return tuple(value for value in UPDATE_REASON_ORDER if value in supplied)


def _typed_evidence(
    values: Sequence[EvidenceV2 | Mapping[str, Any]],
) -> tuple[EvidenceV2, ...]:
    records = tuple(
        value if isinstance(value, EvidenceV2) else EvidenceV2.from_dict(value)
        for value in values
    )
    if len({record.stable_id for record in records}) != len(records):
        raise HPKValidationError("promotion evidence contains duplicate evidence IDs")
    return records


def _counts(records: Sequence[EvidenceV2]) -> dict[str, int]:
    return {
        verdict: sum(record["verdict"] == verdict for record in records)
        for verdict in ("support", "oppose", "unverified")
    }


def _distinct_counts(
    records: Sequence[EvidenceV2], *, provenance_field: str
) -> dict[str, int]:
    return {
        verdict: len(
            {
                str(
                    record["episode_evidence_group_id"]
                    if provenance_field == "episode_evidence_group_id"
                    else record["source_identity"][provenance_field]
                )
                for record in records
                if record["verdict"] == verdict
            }
        )
        for verdict in ("support", "oppose", "unverified")
    }


def _promotion_reasons(
    *,
    counts: Mapping[str, int],
    distinct_episodes: Mapping[str, int],
    distinct_scenes: Mapping[str, int],
    lower_confidence_bound: float,
    policy: PromotionPolicyV1,
) -> list[str]:
    reasons: list[str] = []
    if counts["support"] < int(policy["min_support_count"]):
        reasons.append("insufficient_support")
    if distinct_episodes["support"] < int(policy["min_distinct_support_episodes"]):
        reasons.append("insufficient_distinct_support_episodes")
    if distinct_scenes["support"] < int(policy["min_distinct_scene_signatures"]):
        reasons.append("insufficient_distinct_scene_signatures")
    if counts["oppose"] > int(policy["max_oppose_count"]):
        reasons.append("too_many_oppositions")
    if lower_confidence_bound < float(policy["accept_lcb_threshold"]):
        reasons.append("lcb_below_accept_threshold")
    return reasons


def evaluate_lifecycle(
    entry: EntryV1 | Mapping[str, Any],
    evidence: Sequence[EvidenceV2 | Mapping[str, Any]],
    *,
    policy: PromotionPolicyV1 | Mapping[str, Any],
    lower_confidence_bound: float,
    current_lifecycle: str | None = None,
    prior_decision: UpdateDecisionV1 | Mapping[str, Any] | None = None,
) -> LifecycleEvaluation:
    """Apply the frozen lifecycle state machine without modifying an entry."""

    typed_entry = entry if isinstance(entry, EntryV1) else EntryV1.from_dict(entry)
    typed_policy = (
        policy if isinstance(policy, PromotionPolicyV1) else PromotionPolicyV1(policy)
    )
    records = _typed_evidence(evidence)
    state = str(current_lifecycle or typed_entry["status"])
    if state not in LIFECYCLE_STATES:
        raise HPKValidationError(f"unknown lifecycle state: {state!r}")
    if state == "candidate_for_revalidation" and prior_decision is None:
        raise HPKValidationError(
            "candidate_for_revalidation requires its prior UpdateDecisionV1 marker"
        )
    marker = None
    if prior_decision is not None:
        marker = (
            prior_decision
            if isinstance(prior_decision, UpdateDecisionV1)
            else UpdateDecisionV1.from_dict(prior_decision)
        )
        if marker["entry_id"] != typed_entry["entry_id"]:
            raise HPKValidationError("revalidation marker entry_id mismatch")
        if marker["to_lifecycle"] != state:
            raise HPKValidationError("revalidation marker lifecycle mismatch")

    counts = _counts(records)
    distinct_episodes = _distinct_counts(
        records, provenance_field="episode_evidence_group_id"
    )
    distinct_scenes = _distinct_counts(
        records, provenance_field="scene_signature_sha256"
    )
    promotion_reasons = _promotion_reasons(
        counts=counts,
        distinct_episodes=distinct_episodes,
        distinct_scenes=distinct_scenes,
        lower_confidence_bound=lower_confidence_bound,
        policy=typed_policy,
    )
    promotable = not promotion_reasons
    adverse = bool(
        lower_confidence_bound < float(typed_policy["demote_lcb_threshold"])
        or counts["oppose"] > int(typed_policy["max_oppose_count"])
    )
    can_revalidate = bool(
        counts["oppose"] >= int(typed_policy["min_oppose_count_for_revalidation"])
        and distinct_episodes["oppose"]
        >= int(typed_policy["min_distinct_oppose_episodes_for_revalidation"])
    )
    can_deprecate = bool(
        counts["oppose"] >= int(typed_policy["min_oppose_count_for_deprecation"])
        and distinct_episodes["oppose"]
        >= int(typed_policy["min_distinct_oppose_episodes_for_deprecation"])
    )
    marker_ids = set(marker["evidence_ids"]) if marker is not None else set()
    new_opposition = any(
        record["verdict"] == "oppose" and record.stable_id not in marker_ids
        for record in records
    )

    if state == "deprecated":
        to_state = "deprecated"
        reasons = ["deprecated_terminal"]
    elif state == "candidate":
        if promotable:
            to_state = "accepted"
            reasons = ["promotion_thresholds_met"]
        else:
            to_state = "candidate"
            reasons = promotion_reasons
    elif state == "accepted":
        if adverse and can_revalidate:
            to_state = "candidate_for_revalidation"
            reasons = ["revalidation_threshold_crossed"]
        else:
            to_state = "accepted"
            reasons = ["accepted_retained"]
            if adverse and not can_revalidate:
                reasons.append("revalidation_threshold_not_met")
    else:
        if adverse and can_deprecate and new_opposition:
            to_state = "deprecated"
            reasons = ["deprecation_threshold_crossed"]
        else:
            to_state = "candidate_for_revalidation"
            reasons = ["revalidation_pending"]
            if adverse and not can_deprecate:
                reasons.append("deprecation_threshold_not_met")

    return LifecycleEvaluation(
        from_lifecycle=state,
        to_lifecycle=to_state,
        reason_codes=_ordered_reasons(reasons),
        counts=MappingProxy(counts),
        distinct_episode_counts=MappingProxy(distinct_episodes),
        distinct_scene_signature_counts=MappingProxy(distinct_scenes),
    )


__all__ = [
    "LCB_METHOD",
    "LIFECYCLE_TRANSITION_PROFILE",
    "LifecycleEvaluation",
    "PROMOTION_POLICY_SCHEMA",
    "PromotionPolicyV1",
    "evaluate_lifecycle",
]
