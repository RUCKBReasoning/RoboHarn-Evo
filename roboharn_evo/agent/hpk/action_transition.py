from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Iterator, Mapping
from typing import Any, Literal

from roboharn_evo.agent.hpk.schemas import (
    HPKValidationError,
    AbstractEffectV1,
    CandidateGeometryFeaturesV1,
    canonical_json,
    canonical_json_bytes,
    contains_private_transfer_text,
    stable_content_id,
    validate_content_id,
)


ACTION_EFFECT_TRANSITION_SCHEMA = "roboharn_evo/hpk/action_effect_transition/v1"
PUBLIC_ACTION_EFFECT_TRANSITION_SCHEMA = "roboharn_evo/hpk/action_effect_transition_public/v1"

EVIDENCE_VERDICTS = frozenset({"support", "oppose", "unverified"})
MOTION_STATUSES = frozenset(
    {"completed", "failed_before_effect", "interrupted", "unknown"}
)
REALIZATION_STATUSES = frozenset({"satisfied", "violated", "unknown"})
OPERATIONS = frozenset({"contact", "grasp", "place"})
ARMS = frozenset({"left", "right"})
TARGET_RELATIONS = frozenset({"center_of"})
EFFECT_OBSERVATION_SCOPES = frozenset({"independent", "batch_unseparated", "missing"})
TARGET_IDENTITY_STATUSES = frozenset({"bound", "lost", "unknown"})
GEOMETRIC_COMPLIANCE_VALUES = frozenset({True, False, "unverified"})

DETERMINISTIC_VERIFIER_SOURCES = frozenset(
    {
        "scene_memory_delta",
        "robot_state",
        "attachment_state",
        "runtime_grasp_validation",
        "runtime_place_validation",
    }
)
KNOWN_VERIFIER_SOURCES = DETERMINISTIC_VERIFIER_SOURCES | {"vlm_action_effect_verifier"}

ATTRIBUTION_REASON_ORDER = (
    "infrastructure_invalid",
    "action_not_executed",
    "missing_action_attempt_nonce",
    "missing_step_boundary",
    "non_monotonic_step_boundary",
    "strategy_identity_unresolved",
    "selected_candidate_unbound",
    "candidate_geometry_unresolved",
    "candidate_geometric_noncompliance",
    "motion_not_completed",
    "realization_not_satisfied",
    "target_identity_unresolved",
    "missing_independent_post_action_observation",
    "batch_effect_unseparated",
    "missing_expected_effect",
    "missing_observed_effect",
    "verifier_conflict",
    "model_claim_only",
    "expected_effect_unresolved",
    "observed_effect_unverified",
    "causal_attribution_unresolved",
)

_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "transition_id",
        "episode_id",
        "action_attempt_nonce",
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
        "selected_candidate_private_ref",
        "candidate_geometry_features",
        "geometric_compliance",
        "tool_call_ref",
        "tool_result_ref",
        "physical_action_executed",
        "motion_status",
        "realization_status",
        "pre_effect_state",
        "post_effect_state",
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


def _fail(path: str, message: str) -> None:
    raise HPKValidationError(f"{path}: {message}")


def _mapping_or_none(value: Any, *, path: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        _fail(path, "must be an object or null")
    return copy.deepcopy(dict(value))


def _nullable_text(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string or null")
    return value.strip()


def _nullable_step(value: Any, *, path: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(path, "must be a non-negative integer or null")
    return value


def _string_list(
    value: Any,
    *,
    path: str,
    allowed: frozenset[str] | None = None,
) -> list[str]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            _fail(f"{path}[{index}]", "must be a non-empty string")
        token = item.strip()
        if allowed is not None and token not in allowed:
            _fail(f"{path}[{index}]", f"must be one of {sorted(allowed)}")
        result.append(token)
    if len(result) != len(set(result)):
        _fail(path, "must not contain duplicates")
    return result


def _effect(
    value: Any,
    *,
    path: str,
) -> AbstractEffectV1 | None:
    if value is None:
        return None
    if isinstance(value, AbstractEffectV1):
        return value
    if not isinstance(value, Mapping):
        _fail(path, "must be an AbstractEffectV1 object or null")
    return AbstractEffectV1.from_dict(value)


def _feature_record(value: Any) -> CandidateGeometryFeaturesV1 | None:
    if value is None:
        return None
    if isinstance(value, CandidateGeometryFeaturesV1):
        return value
    if not isinstance(value, Mapping):
        _fail(
            "ActionEffectTransitionV1.candidate_geometry_features",
            "must be CandidateGeometryFeaturesV1 or null",
        )
    return CandidateGeometryFeaturesV1.from_dict(value)


def _effect_payload(value: Any) -> dict[str, Any] | None:
    typed = _effect(value, path="effect")
    return None if typed is None else typed.to_dict()


def _has_deterministic_effect_source(value: Mapping[str, Any]) -> bool:
    sources = value.get("verifier_sources", [])
    return isinstance(sources, list) and bool(
        set(sources) & DETERMINISTIC_VERIFIER_SOURCES
    )


def determine_action_effect_attribution(
    value: Mapping[str, Any],
) -> tuple[Literal["support", "oppose", "unverified"], tuple[str, ...]]:
    """Apply the action-local support/oppose/unverified rules.

    Episode success/failure is intentionally absent from this function.  A
    terminal task outcome therefore cannot retroactively reward or punish an
    earlier action.
    """

    reasons: set[str] = set()
    before = value.get("env_step_before")
    after = value.get("env_step_after")
    expected = _effect_payload(value.get("expected_effect"))
    observed = _effect_payload(value.get("observed_effect"))
    verifier_sources = set(value.get("verifier_sources", []) or [])

    if value.get("infrastructure_valid") is not True:
        reasons.add("infrastructure_invalid")
    if value.get("physical_action_executed") is not True:
        reasons.add("action_not_executed")
    if not str(value.get("action_attempt_nonce") or "").strip():
        reasons.add("missing_action_attempt_nonce")
    if before is None or after is None:
        reasons.add("missing_step_boundary")
    elif not isinstance(before, int) or not isinstance(after, int) or after <= before:
        reasons.add("non_monotonic_step_boundary")
        reasons.add("missing_independent_post_action_observation")
    if not all(
        str(value.get(key) or "").strip()
        for key in ("condition_id", "task_strategy_id", "geometric_strategy_id")
    ):
        reasons.add("strategy_identity_unresolved")
    if not str(value.get("selected_candidate_private_ref") or "").strip():
        reasons.add("selected_candidate_unbound")
    if value.get("candidate_geometry_features") is None:
        reasons.add("candidate_geometry_unresolved")
    if value.get("geometric_compliance") is not True:
        reasons.add("candidate_geometric_noncompliance")
    if (
        not str(value.get("tool_call_ref") or "").strip()
        or not str(value.get("tool_result_ref") or "").strip()
    ):
        reasons.add("causal_attribution_unresolved")
    if value.get("motion_status") != "completed":
        reasons.add("motion_not_completed")
    if value.get("realization_status") != "satisfied":
        reasons.add("realization_not_satisfied")
    if value.get("target_identity_status") != "bound":
        reasons.add("target_identity_unresolved")
    observation_scope = value.get("effect_observation_scope")
    if observation_scope != "independent":
        reasons.add("missing_independent_post_action_observation")
    if observation_scope == "batch_unseparated":
        reasons.add("batch_effect_unseparated")
    if value.get("pre_effect_state") is None or value.get("post_effect_state") is None:
        reasons.add("missing_independent_post_action_observation")
    if expected is None:
        reasons.add("missing_expected_effect")
    if observed is None:
        reasons.add("missing_observed_effect")
    if value.get("verifier_conflicts"):
        reasons.add("verifier_conflict")
    if verifier_sources and not (verifier_sources & DETERMINISTIC_VERIFIER_SOURCES):
        reasons.add("model_claim_only")
    if not verifier_sources and observed is not None:
        reasons.add("model_claim_only")

    if expected is not None:
        expected_predicates = set(expected.get("expected_predicates", []))
        if not expected_predicates:
            reasons.add("expected_effect_unresolved")
    else:
        expected_predicates = set()

    if reasons:
        verdict: Literal["support", "oppose", "unverified"] = "unverified"
    elif observed is None or expected is None:
        verdict = "unverified"
    elif observed.get("effect_type") != expected.get("effect_type"):
        reasons.add("observed_effect_unverified")
        verdict = "unverified"
    elif observed.get("verifiability") == "contradicted":
        verdict = "oppose"
    elif observed.get("verifiability") == "verified" and expected_predicates <= set(
        observed.get("observed_predicates", [])
    ):
        verdict = "support"
    else:
        reasons.add("observed_effect_unverified")
        verdict = "unverified"

    ordered = tuple(reason for reason in ATTRIBUTION_REASON_ORDER if reason in reasons)
    return verdict, ordered


def transition_identity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop("transition_id", None)
    payload.pop("evidence_verdict", None)
    payload.pop("attribution_reasons", None)
    return payload


def transition_id_for(value: Mapping[str, Any]) -> str:
    return stable_content_id("afktransition", transition_identity_payload(value))


class ActionEffectTransitionV1(Mapping[str, Any]):
    """Deeply immutable, strict private action-effect transition."""

    __slots__ = ("_data",)

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            _fail("ActionEffectTransitionV1", "must be an object")
        data = copy.deepcopy(dict(payload))
        # Reuse canonical validation to reject non-JSON and non-finite values.
        canonical_json_bytes(data)
        fields = set(data)
        if fields != _REQUIRED_FIELDS:
            missing = sorted(_REQUIRED_FIELDS - fields)
            unknown = sorted(fields - _REQUIRED_FIELDS)
            _fail(
                "ActionEffectTransitionV1",
                f"field mismatch; missing={missing}, unknown={unknown}",
            )
        if data["schema"] != ACTION_EFFECT_TRANSITION_SCHEMA:
            _fail(
                "ActionEffectTransitionV1.schema",
                f"expected {ACTION_EFFECT_TRANSITION_SCHEMA!r}",
            )
        validate_content_id(
            data["transition_id"],
            prefix="afktransition",
            path="ActionEffectTransitionV1.transition_id",
        )
        episode_id = data["episode_id"]
        if not (
            isinstance(episode_id, str)
            and episode_id.strip()
            or isinstance(episode_id, int)
            and not isinstance(episode_id, bool)
            and episode_id >= 0
        ):
            _fail(
                "ActionEffectTransitionV1.episode_id",
                "must be a non-empty string or non-negative integer",
            )
        nonce = data["action_attempt_nonce"]
        if nonce is not None and (not isinstance(nonce, str) or not nonce.strip()):
            _fail(
                "ActionEffectTransitionV1.action_attempt_nonce",
                "must be a non-empty string or null",
            )
        before = _nullable_step(
            data["env_step_before"],
            path="ActionEffectTransitionV1.env_step_before",
        )
        after = _nullable_step(
            data["env_step_after"],
            path="ActionEffectTransitionV1.env_step_after",
        )
        if before is not None and after is not None and after < before:
            _fail(
                "ActionEffectTransitionV1.env_step_after",
                "must be >= env_step_before",
            )
        for key, prefix in (
            ("condition_id", "afkc"),
            ("task_strategy_id", "afku"),
            ("selected_hpk_entry_id", "afkentry"),
            ("geometric_strategy_id", "afkz"),
        ):
            if data[key] is not None:
                validate_content_id(
                    data[key],
                    prefix=prefix,
                    path=f"ActionEffectTransitionV1.{key}",
                )
        retrieved = _string_list(
            data["retrieved_hpk_entry_ids"],
            path="ActionEffectTransitionV1.retrieved_hpk_entry_ids",
        )
        for index, entry_id in enumerate(retrieved):
            validate_content_id(
                entry_id,
                prefix="afkentry",
                path=(f"ActionEffectTransitionV1.retrieved_hpk_entry_ids[{index}]"),
            )
        selected_entry = data["selected_hpk_entry_id"]
        if selected_entry is not None and selected_entry not in retrieved:
            _fail(
                "ActionEffectTransitionV1.selected_hpk_entry_id",
                "must occur in retrieved_hpk_entry_ids",
            )
        operation = data["operation"]
        if operation is not None and operation not in OPERATIONS:
            _fail(
                "ActionEffectTransitionV1.operation",
                f"must be one of {sorted(OPERATIONS)} or null",
            )
        arm = data["arm"]
        if arm is not None and arm not in ARMS:
            _fail(
                "ActionEffectTransitionV1.arm",
                f"must be one of {sorted(ARMS)} or null",
            )
        target_role = _nullable_text(
            data["target_role"], path="ActionEffectTransitionV1.target_role"
        )
        if target_role is not None and contains_private_transfer_text(target_role):
            _fail(
                "ActionEffectTransitionV1.target_role",
                "must not contain runtime-private identity or geometry",
            )
        target_relation = data["target_relation"]
        if target_relation is not None and target_relation not in TARGET_RELATIONS:
            _fail(
                "ActionEffectTransitionV1.target_relation",
                f"must be one of {sorted(TARGET_RELATIONS)} or null",
            )
        _nullable_text(
            data["selected_candidate_private_ref"],
            path="ActionEffectTransitionV1.selected_candidate_private_ref",
        )
        _feature_record(data["candidate_geometry_features"])
        if data["geometric_compliance"] not in GEOMETRIC_COMPLIANCE_VALUES:
            _fail(
                "ActionEffectTransitionV1.geometric_compliance",
                "must be true, false, or 'unverified'",
            )
        for key in ("tool_call_ref", "tool_result_ref"):
            _nullable_text(data[key], path=f"ActionEffectTransitionV1.{key}")
        if not isinstance(data["physical_action_executed"], bool):
            _fail(
                "ActionEffectTransitionV1.physical_action_executed",
                "must be boolean",
            )
        if data["motion_status"] not in MOTION_STATUSES:
            _fail(
                "ActionEffectTransitionV1.motion_status",
                f"must be one of {sorted(MOTION_STATUSES)}",
            )
        if data["realization_status"] not in REALIZATION_STATUSES:
            _fail(
                "ActionEffectTransitionV1.realization_status",
                f"must be one of {sorted(REALIZATION_STATUSES)}",
            )
        _mapping_or_none(
            data["pre_effect_state"],
            path="ActionEffectTransitionV1.pre_effect_state",
        )
        _mapping_or_none(
            data["post_effect_state"],
            path="ActionEffectTransitionV1.post_effect_state",
        )
        if data["effect_observation_scope"] not in EFFECT_OBSERVATION_SCOPES:
            _fail(
                "ActionEffectTransitionV1.effect_observation_scope",
                f"must be one of {sorted(EFFECT_OBSERVATION_SCOPES)}",
            )
        if data["target_identity_status"] not in TARGET_IDENTITY_STATUSES:
            _fail(
                "ActionEffectTransitionV1.target_identity_status",
                f"must be one of {sorted(TARGET_IDENTITY_STATUSES)}",
            )
        expected = _effect(
            data["expected_effect"],
            path="ActionEffectTransitionV1.expected_effect",
        )
        observed = _effect(
            data["observed_effect"],
            path="ActionEffectTransitionV1.observed_effect",
        )
        if (
            expected is not None
            and observed is not None
            and expected["effect_type"] != observed["effect_type"]
            and data["evidence_verdict"] != "unverified"
        ):
            _fail(
                "ActionEffectTransitionV1.evidence_verdict",
                "different expected/observed effect types require unverified",
            )
        sources = _string_list(
            data["verifier_sources"],
            path="ActionEffectTransitionV1.verifier_sources",
            allowed=KNOWN_VERIFIER_SOURCES,
        )
        if observed is not None:
            observed_sources = set(observed.to_dict().get("verifier_sources", []))
            if not observed_sources <= set(sources):
                _fail(
                    "ActionEffectTransitionV1.verifier_sources",
                    "must include every observed_effect verifier source",
                )
        conflicts = _string_list(
            data["verifier_conflicts"],
            path="ActionEffectTransitionV1.verifier_conflicts",
        )
        for index, conflict in enumerate(conflicts):
            if not re.fullmatch(
                r"[a-z][a-z0-9_.:-]{0,127}", conflict
            ) or contains_private_transfer_text(conflict):
                _fail(
                    f"ActionEffectTransitionV1.verifier_conflicts[{index}]",
                    "must be a bounded public conflict code, not free text",
                )
        for key in ("infrastructure_valid", "oracle_derived", "expert_derived"):
            if not isinstance(data[key], bool):
                _fail(f"ActionEffectTransitionV1.{key}", "must be boolean")
        if data["evidence_verdict"] not in EVIDENCE_VERDICTS:
            _fail(
                "ActionEffectTransitionV1.evidence_verdict",
                f"must be one of {sorted(EVIDENCE_VERDICTS)}",
            )
        reasons = _string_list(
            data["attribution_reasons"],
            path="ActionEffectTransitionV1.attribution_reasons",
            allowed=frozenset(ATTRIBUTION_REASON_ORDER),
        )
        expected_verdict, expected_reasons = determine_action_effect_attribution(data)
        if data["evidence_verdict"] != expected_verdict:
            _fail(
                "ActionEffectTransitionV1.evidence_verdict",
                f"content requires {expected_verdict!r}",
            )
        if reasons != list(expected_reasons):
            _fail(
                "ActionEffectTransitionV1.attribution_reasons",
                f"content requires {list(expected_reasons)!r}",
            )
        expected_id = transition_id_for(data)
        if data["transition_id"] != expected_id:
            _fail(
                "ActionEffectTransitionV1.transition_id",
                f"content mismatch; expected {expected_id}",
            )
        self._data = data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionEffectTransitionV1":
        return cls(payload)

    @classmethod
    def from_json(cls, payload: str) -> "ActionEffectTransitionV1":
        import json

        try:
            decoded = json.loads(payload)
        except (json.JSONDecodeError, TypeError) as exc:
            raise HPKValidationError(f"invalid transition JSON: {exc}") from exc
        return cls(decoded)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def to_json(self) -> str:
        return canonical_json(self._data)

    @property
    def stable_id(self) -> str:
        return str(self._data["transition_id"])

    @property
    def evidence_verdict(self) -> str:
        return str(self._data["evidence_verdict"])

    @property
    def update_eligible(self) -> bool:
        return bool(
            self._data["infrastructure_valid"]
            and self._data["condition_id"]
            and self._data["task_strategy_id"]
            and self._data["geometric_strategy_id"]
        )

    def __getitem__(self, key: str) -> Any:
        return copy.deepcopy(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def migrate_action_effect_transition(payload: Mapping[str, Any]) -> ActionEffectTransitionV1:
    data = copy.deepcopy(dict(payload))
    if data.get("schema") == ACTION_EFFECT_TRANSITION_SCHEMA:
        return ActionEffectTransitionV1(data)
    if data.get("schema") != "tcm/afk/action_effect_transition/v1":
        _fail("ActionEffectTransitionV1.schema", "unsupported legacy schema")
    if data.get("transition_id") != transition_id_for(data):
        _fail("ActionEffectTransitionV1.transition_id", "legacy content mismatch")
    for previous, current in (
        ("retrieved_afk_entry_ids", "retrieved_hpk_entry_ids"),
        ("selected_afk_entry_id", "selected_hpk_entry_id"),
    ):
        if previous in data:
            if current in data:
                _fail("ActionEffectTransitionV1", f"duplicate fields: {previous}, {current}")
            data[current] = data.pop(previous)
    data["schema"] = ACTION_EFFECT_TRANSITION_SCHEMA
    data["transition_id"] = transition_id_for(data)
    return ActionEffectTransitionV1(data)


def build_action_effect_transition(
    **facts: Any,
) -> ActionEffectTransitionV1:
    """Build a transition while deriving its verdict, reasons, and content ID."""

    payload = copy.deepcopy(facts)
    payload["schema"] = ACTION_EFFECT_TRANSITION_SCHEMA
    for key in ("transition_id", "evidence_verdict", "attribution_reasons"):
        payload.pop(key, None)
    for key in ("expected_effect", "observed_effect"):
        value = payload.get(key)
        if isinstance(value, AbstractEffectV1):
            payload[key] = value.to_dict()
    features = payload.get("candidate_geometry_features")
    if isinstance(features, CandidateGeometryFeaturesV1):
        payload["candidate_geometry_features"] = features.to_dict()
    verdict, reasons = determine_action_effect_attribution(payload)
    payload["evidence_verdict"] = verdict
    payload["attribution_reasons"] = list(reasons)
    payload["transition_id"] = transition_id_for(payload)
    return ActionEffectTransitionV1.from_dict(payload)


def _state_sha256(value: Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _private_ref_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def public_transition_projection(
    value: ActionEffectTransitionV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Return an ID-only public audit with no candidate, track, pose, or state."""

    transition = (
        value
        if isinstance(value, ActionEffectTransitionV1)
        else ActionEffectTransitionV1.from_dict(value)
    )
    private = transition.to_dict()
    public = {
        "schema": PUBLIC_ACTION_EFFECT_TRANSITION_SCHEMA,
        "transition_id": private["transition_id"],
        "episode_id": private["episode_id"],
        "action_attempt_nonce_sha256": _private_ref_sha256(
            private["action_attempt_nonce"]
        ),
        "env_step_before": private["env_step_before"],
        "env_step_after": private["env_step_after"],
        "condition_id": private["condition_id"],
        "task_strategy_id": private["task_strategy_id"],
        "retrieved_hpk_entry_ids": private["retrieved_hpk_entry_ids"],
        "selected_hpk_entry_id": private["selected_hpk_entry_id"],
        "geometric_strategy_id": private["geometric_strategy_id"],
        "operation": private["operation"],
        "arm": private["arm"],
        "target_role": private["target_role"],
        "target_relation": private["target_relation"],
        "candidate_geometry_features": private["candidate_geometry_features"],
        "geometric_compliance": private["geometric_compliance"],
        "tool_call_ref_sha256": _private_ref_sha256(private["tool_call_ref"]),
        "tool_result_ref_sha256": _private_ref_sha256(private["tool_result_ref"]),
        "physical_action_executed": private["physical_action_executed"],
        "motion_status": private["motion_status"],
        "realization_status": private["realization_status"],
        "pre_effect_state_sha256": _state_sha256(private["pre_effect_state"]),
        "post_effect_state_sha256": _state_sha256(private["post_effect_state"]),
        "effect_observation_scope": private["effect_observation_scope"],
        "target_identity_status": private["target_identity_status"],
        "expected_effect": private["expected_effect"],
        "observed_effect": private["observed_effect"],
        "verifier_sources": private["verifier_sources"],
        "verifier_conflicts": private["verifier_conflicts"],
        "infrastructure_valid": private["infrastructure_valid"],
        "oracle_derived": private["oracle_derived"],
        "expert_derived": private["expert_derived"],
        "evidence_verdict": private["evidence_verdict"],
        "attribution_reasons": private["attribution_reasons"],
    }
    public["public_transition_id"] = stable_content_id("afkpubtrans", public)
    return public


redact_action_effect_transition = public_transition_projection


__all__ = [
    "ACTION_EFFECT_TRANSITION_SCHEMA",
    "ATTRIBUTION_REASON_ORDER",
    "ActionEffectTransitionV1",
    "DETERMINISTIC_VERIFIER_SOURCES",
    "PUBLIC_ACTION_EFFECT_TRANSITION_SCHEMA",
    "build_action_effect_transition",
    "determine_action_effect_attribution",
    "public_transition_projection",
    "redact_action_effect_transition",
    "transition_id_for",
    "transition_identity_payload",
    "migrate_action_effect_transition",
]
