from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from typing import Any, ClassVar

from roboharn_evo.agent.hpk.schemas import (
    ABSTRACT_EFFECT_SCHEMA,
    HPKValidationError,
    AbstractEffectV1,
    ConditionV1,
    EVIDENCE_VERDICTS,
    GeometricStrategyV1,
    MOTION_STATUSES,
    REALIZATION_STATUSES,
    TASK_STRATEGY_NORMALIZATION_VERSION,
    TaskStrategyV1,
    canonical_json_bytes,
    stable_content_id,
    validate_content_id,
)


STRATEGY_KEY_SCHEMA = "roboharn_evo/hpk/strategy_key/v1"
EVIDENCE_V2_SCHEMA = "roboharn_evo/hpk/evidence/v2"
UPDATE_DECISION_SCHEMA = "roboharn_evo/hpk/update_decision/v1"

LIFECYCLE_STATES = frozenset(
    {"candidate", "accepted", "candidate_for_revalidation", "deprecated"}
)
PERSISTED_ENTRY_STATUSES = frozenset({"candidate", "accepted", "deprecated"})
EFFECT_VERIFIABILITY_VALUES = frozenset({"verified", "contradicted", "unverified"})
GEOMETRIC_COMPLIANCE_VALUES = frozenset({"true", "false", "unverified"})
EVIDENCE_SOURCE_KINDS = frozenset(
    {"agent_rollout", "benchmark_expert", "oracle_diagnostic"}
)

ABSTENTION_REASON_ORDER = (
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
    "terminal_outcome_only",
    "tool_success_only",
)
ABSTENTION_REASONS = frozenset(ABSTENTION_REASON_ORDER)

UPDATE_REASON_ORDER = (
    "no_new_evidence",
    "ineligible_evidence",
    "insufficient_support",
    "insufficient_distinct_support_episodes",
    "insufficient_distinct_scene_signatures",
    "too_many_oppositions",
    "lcb_below_accept_threshold",
    "promotion_thresholds_met",
    "accepted_retained",
    "revalidation_threshold_not_met",
    "revalidation_threshold_crossed",
    "revalidation_pending",
    "deprecation_threshold_not_met",
    "deprecation_threshold_crossed",
    "deprecated_terminal",
)
UPDATE_REASONS = frozenset(UPDATE_REASON_ORDER)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_RUNTIME_SCHEMA_RE = re.compile(
    r"^[a-z][a-z0-9_.-]*(?:/[a-z][a-z0-9_.-]*)*/v[1-9][0-9]*$"
)


def _fail(path: str, message: str) -> None:
    raise HPKValidationError(f"{path}: {message}" if path else message)


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    result = copy.deepcopy(dict(value))
    canonical_json_bytes(result)
    return result


def _exact_fields(
    value: Mapping[str, Any],
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    path: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        _fail(path, "missing required field(s): " + ", ".join(missing))
    if unknown:
        _fail(path, "unknown field(s): " + ", ".join(unknown))


def _string(value: Any, *, path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        _fail(path, "must be a string")
    if not allow_empty and not value.strip():
        _fail(path, "must be a non-empty string")
    return value


def _enum(value: Any, *, allowed: frozenset[str], path: str) -> str:
    result = _string(value, path=path)
    if result not in allowed:
        _fail(path, f"must be one of {sorted(allowed)}")
    return result


def _boolean(value: Any, *, path: str) -> bool:
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
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a number")
    result = float(value)
    if not (float("-inf") < result < float("inf")):
        _fail(path, "must be finite")
    if minimum is not None and result < minimum:
        _fail(path, f"must be >= {minimum}")
    if maximum is not None and result > maximum:
        _fail(path, f"must be <= {maximum}")
    return result


def _sha256(value: Any, *, path: str, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    result = _string(value, path=path)
    if _SHA256_RE.fullmatch(result) is None:
        _fail(path, "must be a lowercase SHA-256 digest")
    return result


def _utc(value: Any, *, path: str) -> str:
    result = _string(value, path=path)
    if _UTC_RE.fullmatch(result) is None:
        _fail(path, "must be RFC3339 UTC with a trailing Z")
    try:
        datetime.fromisoformat(result[:-1] + "+00:00")
    except ValueError as exc:
        _fail(path, f"invalid UTC timestamp ({exc})")
    return result


def _string_list(
    value: Any,
    *,
    path: str,
    allowed: frozenset[str] | None = None,
    sorted_values: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        text = _string(item, path=f"{path}[{index}]")
        if allowed is not None and text not in allowed:
            _fail(f"{path}[{index}]", f"must be one of {sorted(allowed)}")
        result.append(text)
    if len(result) != len(set(result)):
        _fail(path, "must not contain duplicate values")
    if sorted_values and result != sorted(result):
        _fail(path, "must be sorted lexicographically")
    return result


class _FrozenRecord(Mapping[str, Any]):
    """Small immutable mapping facade local to the evolving schema family."""

    SCHEMA: ClassVar[str]
    LEGACY_SCHEMA: ClassVar[str]
    __slots__ = ("_data",)

    def __init__(self, payload: Mapping[str, Any]) -> None:
        data = _mapping(payload, path=self.__class__.__name__)
        if data.get("schema") not in {self.SCHEMA, self.LEGACY_SCHEMA}:
            _fail(
                f"{self.__class__.__name__}.schema",
                f"unsupported schema; expected {self.SCHEMA!r}",
            )
        self._validate(data)
        self._data = data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "_FrozenRecord":
        return cls(payload)

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        del payload

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self._data)

    def __getitem__(self, key: str) -> Any:
        return copy.deepcopy(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _transferable_task_strategy(value: Any, *, path: str) -> dict[str, Any]:
    if isinstance(value, TaskStrategyV1):
        return value.transferable_dict()
    payload = _mapping(value, path=path)
    if "source" in payload:
        return TaskStrategyV1.from_dict(payload).transferable_dict()
    synthetic = {
        **payload,
        "source": {
            "planner_subtask_text": "execute current typed strategy",
            "selected_skill": "typed-strategy-projection",
            "action_mode": payload.get("operation"),
            "normalization_version": TASK_STRATEGY_NORMALIZATION_VERSION,
        },
    }
    typed = TaskStrategyV1.from_dict(synthetic)
    if typed.transferable_dict() != payload:
        _fail(path, "is not the exact TaskStrategyV1 transferable projection")
    return payload


def _typed_task_from_transferable(value: Any, *, path: str) -> TaskStrategyV1:
    task = _transferable_task_strategy(value, path=path)
    return TaskStrategyV1.from_dict(
        {
            **task,
            "source": {
                "planner_subtask_text": "execute current typed strategy",
                "selected_skill": "typed-strategy-projection",
                "action_mode": task["operation"],
                "normalization_version": TASK_STRATEGY_NORMALIZATION_VERSION,
            },
        }
    )


def strategy_key_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Match strategy_extractor's exact full-transferable key projection."""

    value = dict(payload)
    from roboharn_evo.agent.hpk.strategy_extractor import strategy_key_payload

    return strategy_key_payload(
        condition=ConditionV1.from_dict(value.get("condition")),
        task_strategy=_typed_task_from_transferable(
            value.get("task_strategy"), path="StrategyKeyV1.task_strategy"
        ),
        geometric_strategy=GeometricStrategyV1.from_dict(
            value.get("geometric_strategy")
        ),
        expected_effect=AbstractEffectV1.from_dict(value.get("expected_effect")),
    )


def strategy_key_id_for(payload: Mapping[str, Any]) -> str:
    value = strategy_key_identity_payload(payload)
    from roboharn_evo.agent.hpk.strategy_extractor import strategy_key_id_for as authoritative_id

    return authoritative_id(
        condition=ConditionV1.from_dict(value["condition"]),
        task_strategy=_typed_task_from_transferable(
            value["task_strategy"], path="StrategyKeyV1.task_strategy"
        ),
        geometric_strategy=GeometricStrategyV1.from_dict(value["geometric_strategy"]),
        expected_effect=AbstractEffectV1.from_dict(value["expected_effect"]),
    )


class StrategyKeyV1(_FrozenRecord):
    """Stable typed identity for one ``(c, u, z, expected effect)`` tuple."""

    SCHEMA = STRATEGY_KEY_SCHEMA
    LEGACY_SCHEMA = "tcm/afk/strategy_key/v1"

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        path = cls.__name__
        _exact_fields(
            payload,
            required=(
                "schema",
                "strategy_key_id",
                "condition",
                "task_strategy",
                "geometric_strategy",
                "expected_effect",
            ),
            path=path,
        )
        validate_content_id(
            payload["strategy_key_id"],
            prefix="afkstrategy",
            path=f"{path}.strategy_key_id",
        )
        # Parsing the exact identity projection validates all nested schemas,
        # the source-free task projection, and expected-effect projection.
        strategy_key_identity_payload(payload)
        expected = strategy_key_id_for(payload)
        if payload["strategy_key_id"] != expected:
            _fail(
                f"{path}.strategy_key_id",
                f"content identity mismatch; expected {expected}",
            )

    @property
    def stable_id(self) -> str:
        return str(self._data["strategy_key_id"])

    @property
    def condition_id(self) -> str:
        return ConditionV1.from_dict(self._data["condition"]).stable_id

    @property
    def task_strategy_id(self) -> str:
        return _typed_task_from_transferable(
            self._data["task_strategy"], path="StrategyKeyV1.task_strategy"
        ).stable_id

    @property
    def geometric_strategy_id(self) -> str:
        return GeometricStrategyV1.from_dict(self._data["geometric_strategy"]).stable_id

    @property
    def expected_effect_id(self) -> str:
        return AbstractEffectV1.from_dict(self._data["expected_effect"]).stable_id


def build_strategy_key(
    *,
    condition: ConditionV1 | Mapping[str, Any],
    task_strategy: TaskStrategyV1 | Mapping[str, Any],
    geometric_strategy: GeometricStrategyV1 | Mapping[str, Any],
    expected_effect: AbstractEffectV1 | Mapping[str, Any],
) -> StrategyKeyV1:
    typed_condition = (
        condition
        if isinstance(condition, ConditionV1)
        else ConditionV1.from_dict(condition)
    )
    transferable_task = _transferable_task_strategy(
        task_strategy, path="StrategyKeyV1.task_strategy"
    )
    typed_geometry = (
        geometric_strategy
        if isinstance(geometric_strategy, GeometricStrategyV1)
        else GeometricStrategyV1.from_dict(geometric_strategy)
    )
    typed_effect = (
        expected_effect
        if isinstance(expected_effect, AbstractEffectV1)
        else AbstractEffectV1.from_dict(expected_effect)
    )
    payload = {
        "schema": STRATEGY_KEY_SCHEMA,
        "strategy_key_id": "pending",
        "condition": typed_condition.to_dict(),
        "task_strategy": transferable_task,
        "geometric_strategy": typed_geometry.to_dict(),
        "expected_effect": typed_effect.expected_projection(),
    }
    payload["strategy_key_id"] = strategy_key_id_for(payload)
    return StrategyKeyV1.from_dict(payload)


def evidence_v2_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result.pop("evidence_id", None)
    # Event time is provenance.  Re-importing one immutable transition must
    # deduplicate even if a caller reconstructs the same public record later.
    result.pop("created_at", None)
    return result


def evidence_v2_id_for(payload: Mapping[str, Any]) -> str:
    # EvidenceV2 intentionally retains EntryV1's frozen ``afkev_`` reference
    # namespace.  The record's schema, not its ID prefix, carries the version.
    return stable_content_id("afkev", evidence_v2_identity_payload(payload))


class EvidenceV2(_FrozenRecord):
    """Public, immutable action-level evidence linked to private provenance."""

    SCHEMA = EVIDENCE_V2_SCHEMA
    LEGACY_SCHEMA = "tcm/afk/evidence/v2"

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        path = cls.__name__
        _exact_fields(
            payload,
            required=(
                "schema",
                "evidence_id",
                "transition_digest",
                "episode_evidence_group_id",
                "strategy_key_id",
                "condition_id",
                "task_strategy_id",
                "geometric_strategy_id",
                "expected_effect_id",
                "verdict",
                "confidence",
                "motion_status",
                "realization_status",
                "geometric_compliance",
                "observed_effect",
                "fresh_post_observation",
                "verifier_conflict",
                "attribution_reasons",
                "infrastructure_valid",
                "oracle_derived",
                "expert_derived",
                "model_identity",
                "config_identity",
                "schema_identity",
                "source_identity",
                "created_at",
            ),
            path=path,
        )
        validate_content_id(
            payload["evidence_id"], prefix="afkev", path=f"{path}.evidence_id"
        )
        _sha256(payload["transition_digest"], path=f"{path}.transition_digest")
        validate_content_id(
            payload["episode_evidence_group_id"],
            prefix="afkepisode",
            path=f"{path}.episode_evidence_group_id",
        )
        strategy_key_id = payload["strategy_key_id"]
        if strategy_key_id is not None:
            validate_content_id(
                strategy_key_id,
                prefix="afkstrategy",
                path=f"{path}.strategy_key_id",
            )
        identity_values: dict[str, str | None] = {}
        for key, prefix in (
            ("condition_id", "afkc"),
            ("task_strategy_id", "afku"),
            ("geometric_strategy_id", "afkz"),
            ("expected_effect_id", "afkfx"),
        ):
            value = payload[key]
            if value is not None:
                validate_content_id(value, prefix=prefix, path=f"{path}.{key}")
            identity_values[key] = value
        verdict = _enum(
            payload["verdict"], allowed=EVIDENCE_VERDICTS, path=f"{path}.verdict"
        )
        _number(
            payload["confidence"],
            path=f"{path}.confidence",
            minimum=0.0,
            maximum=1.0,
        )
        motion = _enum(
            payload["motion_status"],
            allowed=MOTION_STATUSES,
            path=f"{path}.motion_status",
        )
        realization = _enum(
            payload["realization_status"],
            allowed=REALIZATION_STATUSES,
            path=f"{path}.realization_status",
        )
        compliance = payload["geometric_compliance"]
        if isinstance(compliance, bool):
            compliance_token = "true" if compliance else "false"
        else:
            compliance_token = _enum(
                compliance,
                allowed=GEOMETRIC_COMPLIANCE_VALUES,
                path=f"{path}.geometric_compliance",
            )
        observed = AbstractEffectV1.from_dict(payload["observed_effect"])
        effect_status = str(observed["verifiability"])
        post_fresh = _boolean(
            payload["fresh_post_observation"],
            path=f"{path}.fresh_post_observation",
        )
        verifier_conflict = _boolean(
            payload["verifier_conflict"], path=f"{path}.verifier_conflict"
        )
        infrastructure_valid = _boolean(
            payload["infrastructure_valid"],
            path=f"{path}.infrastructure_valid",
        )
        _boolean(payload["oracle_derived"], path=f"{path}.oracle_derived")
        _boolean(payload["expert_derived"], path=f"{path}.expert_derived")
        abstentions = _string_list(
            payload["attribution_reasons"],
            path=f"{path}.attribution_reasons",
            allowed=ABSTENTION_REASONS,
        )
        expected_order = [
            reason for reason in ABSTENTION_REASON_ORDER if reason in abstentions
        ]
        if abstentions != expected_order:
            _fail(f"{path}.attribution_reasons", "must use frozen reason order")

        model_identity = payload["model_identity"]
        _sha256(model_identity, path=f"{path}.model_identity", nullable=True)
        config_identity = payload["config_identity"]
        _sha256(config_identity, path=f"{path}.config_identity", nullable=True)
        schema_identity = _mapping(
            payload["schema_identity"], path=f"{path}.schema_identity"
        )
        _exact_fields(
            schema_identity,
            required=("action_transition_schema", "runtime_schema"),
            path=f"{path}.schema_identity",
        )
        if schema_identity["action_transition_schema"] not in {
            "roboharn_evo/hpk/action_effect_transition/v1",
            "tcm/afk/action_effect_transition/v1",
        }:
            _fail(
                f"{path}.schema_identity.action_transition_schema",
                "must identify the frozen ActionEffectTransitionV1",
            )
        runtime_schema = _string(
            schema_identity["runtime_schema"],
            path=f"{path}.schema_identity.runtime_schema",
        )
        if _RUNTIME_SCHEMA_RE.fullmatch(runtime_schema) is None:
            _fail(
                f"{path}.schema_identity.runtime_schema",
                "must be a versioned public schema identity, not a path",
            )
        source_identity = _mapping(
            payload["source_identity"], path=f"{path}.source_identity"
        )
        _exact_fields(
            source_identity,
            required=(
                "rollout_manifest_sha256",
                "trace_sha256",
                "runtime_source_sha256",
                "scene_signature_sha256",
            ),
            path=f"{path}.source_identity",
        )
        for name in (
            "rollout_manifest_sha256",
            "trace_sha256",
            "runtime_source_sha256",
        ):
            _sha256(
                source_identity[name],
                path=f"{path}.source_identity.{name}",
                nullable=True,
            )
        _sha256(
            source_identity["scene_signature_sha256"],
            path=f"{path}.source_identity.scene_signature_sha256",
        )
        _utc(payload["created_at"], path=f"{path}.created_at")

        attributable = bool(
            motion == "completed"
            and realization == "satisfied"
            and compliance_token == "true"
            and post_fresh
            and infrastructure_valid
            and not verifier_conflict
            and not abstentions
        )
        if verdict in {"support", "oppose"} and not attributable:
            _fail(
                f"{path}.verdict",
                "support/oppose require completed, compliant, freshly observed, "
                "infrastructure-valid, conflict-free realization",
            )
        identities_resolved = bool(
            strategy_key_id is not None
            and all(value is not None for value in identity_values.values())
        )
        if verdict in {"support", "oppose"} and not identities_resolved:
            _fail(
                f"{path}.strategy_key_id",
                "support/oppose require a resolved strategy identity",
            )
        if (
            not identities_resolved
            and "strategy_identity_unresolved" not in abstentions
        ):
            _fail(
                f"{path}.attribution_reasons",
                "missing strategy identity requires strategy_identity_unresolved",
            )
        if verdict == "support" and effect_status != "verified":
            _fail(f"{path}.verdict", "support requires a verified effect")
        if verdict == "oppose" and effect_status != "contradicted":
            _fail(f"{path}.verdict", "oppose requires a contradicted effect")
        if verdict == "unverified" and not abstentions:
            _fail(
                f"{path}.attribution_reasons",
                "unverified evidence requires at least one stable abstention reason",
            )
        if verdict != "unverified" and abstentions:
            _fail(
                f"{path}.attribution_reasons",
                "support/oppose cannot carry abstention reasons",
            )
        expected_id = evidence_v2_id_for(payload)
        if payload["evidence_id"] != expected_id:
            _fail(
                f"{path}.evidence_id",
                f"content identity mismatch; expected {expected_id}",
            )

    @property
    def stable_id(self) -> str:
        return str(self._data["evidence_id"])

    @property
    def strategy_key_id(self) -> str | None:
        value = self._data["strategy_key_id"]
        return None if value is None else str(value)


def resolve_transition_evidence_verdict(
    transition: Any,
    strategy_key: StrategyKeyV1 | Mapping[str, Any] | None,
) -> tuple[str, tuple[str, ...]]:
    """Resolve only a baseline transition's previously absent typed ``z``.

    The private transition remains immutable.  Supplying an extracted strategy
    key may remove ``strategy_identity_unresolved`` only when condition/task,
    any pre-existing z, and expected effect all agree.  No motion, observation,
    verifier, infrastructure, or target failure can be repaired here.
    """

    from roboharn_evo.agent.hpk.action_transition import ActionEffectTransitionV1

    typed_transition = (
        transition
        if isinstance(transition, ActionEffectTransitionV1)
        else ActionEffectTransitionV1.from_dict(transition)
    )
    typed_key = (
        strategy_key
        if isinstance(strategy_key, StrategyKeyV1)
        else None
        if strategy_key is None
        else StrategyKeyV1.from_dict(strategy_key)
    )
    reasons = set(str(value) for value in typed_transition["attribution_reasons"])
    if typed_key is not None and "strategy_identity_unresolved" in reasons:
        expected_effect = typed_transition["expected_effect"]
        bindings = (
            (typed_transition["condition_id"], typed_key.condition_id),
            (typed_transition["task_strategy_id"], typed_key.task_strategy_id),
            (
                typed_transition["geometric_strategy_id"],
                typed_key.geometric_strategy_id,
            ),
            (
                None
                if not isinstance(expected_effect, Mapping)
                else AbstractEffectV1.from_dict(expected_effect).stable_id,
                typed_key.expected_effect_id,
            ),
        )
        if all(
            current is not None and current == expected
            for current, expected in bindings
            if current is not None
        ) and all(
            current is not None
            for current, _ in (bindings[0], bindings[1], bindings[3])
        ):
            reasons.remove("strategy_identity_unresolved")
    if reasons:
        return (
            "unverified",
            tuple(reason for reason in ABSTENTION_REASON_ORDER if reason in reasons),
        )
    expected = typed_transition["expected_effect"]
    observed = typed_transition["observed_effect"]
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        return "unverified", ("observed_effect_unverified",)
    expected_typed = AbstractEffectV1.from_dict(expected)
    observed_typed = AbstractEffectV1.from_dict(observed)
    if observed_typed["effect_type"] != expected_typed["effect_type"]:
        return "unverified", ("observed_effect_unverified",)
    if observed_typed["verifiability"] == "contradicted":
        return "oppose", ()
    if observed_typed["verifiability"] == "verified" and set(
        expected_typed["expected_predicates"]
    ) <= set(observed_typed.to_dict().get("observed_predicates", [])):
        return "support", ()
    return "unverified", ("observed_effect_unverified",)


def build_evidence_v2(
    transition: Any,
    *,
    strategy_key: StrategyKeyV1 | Mapping[str, Any] | None,
    scene_signature: str,
    rollout_manifest_sha256: str | None,
    trace_sha256: str | None,
    config_sha256: str | None,
    model_identity_sha256: str | None,
    runtime_source_sha256: str | None,
    runtime_schema: str,
    created_at: str,
    confidence: float | None = None,
) -> EvidenceV2:
    """Project one private ActionEffectTransition into immutable public evidence."""

    from roboharn_evo.agent.hpk.action_transition import ActionEffectTransitionV1

    typed_transition = (
        transition
        if isinstance(transition, ActionEffectTransitionV1)
        else ActionEffectTransitionV1.from_dict(transition)
    )
    typed_key = (
        strategy_key
        if isinstance(strategy_key, StrategyKeyV1)
        else None
        if strategy_key is None
        else StrategyKeyV1.from_dict(strategy_key)
    )
    transition_digest = hashlib.sha256(
        canonical_json_bytes(typed_transition.to_dict())
    ).hexdigest()
    observed = typed_transition["observed_effect"]
    if isinstance(observed, Mapping):
        observed_effect = AbstractEffectV1.from_dict(observed).to_dict()
    else:
        observed_effect = {
            "schema": ABSTRACT_EFFECT_SCHEMA,
            "effect_type": "unknown",
            "expected_predicates": [],
            "observed_predicates": [],
            "verifiability": "unverified",
            "verifier_sources": [],
        }
    oracle = bool(typed_transition["oracle_derived"])
    expert = bool(typed_transition["expert_derived"])
    episode_group_id = stable_content_id(
        "afkepisode",
        {
            "rollout_manifest_sha256": rollout_manifest_sha256,
            "trace_sha256": trace_sha256,
        },
    )
    verdict, attribution_reasons = resolve_transition_evidence_verdict(
        typed_transition, typed_key
    )
    if confidence is None:
        confidence = 1.0 if verdict in {"support", "oppose"} else 0.0
    payload: dict[str, Any] = {
        "schema": EVIDENCE_V2_SCHEMA,
        "evidence_id": "pending",
        "transition_digest": transition_digest,
        "episode_evidence_group_id": episode_group_id,
        "strategy_key_id": None if typed_key is None else typed_key.stable_id,
        "condition_id": None if typed_key is None else typed_key.condition_id,
        "task_strategy_id": None if typed_key is None else typed_key.task_strategy_id,
        "geometric_strategy_id": (
            None if typed_key is None else typed_key.geometric_strategy_id
        ),
        "expected_effect_id": None
        if typed_key is None
        else typed_key.expected_effect_id,
        "verdict": verdict,
        "confidence": confidence,
        "motion_status": typed_transition["motion_status"],
        "realization_status": typed_transition["realization_status"],
        "geometric_compliance": typed_transition["geometric_compliance"],
        "observed_effect": observed_effect,
        "fresh_post_observation": bool(
            typed_transition["effect_observation_scope"] == "independent"
            and typed_transition["post_effect_state"] is not None
            and isinstance(typed_transition["env_step_before"], int)
            and isinstance(typed_transition["env_step_after"], int)
            and typed_transition["env_step_after"] > typed_transition["env_step_before"]
        ),
        "verifier_conflict": bool(typed_transition["verifier_conflicts"]),
        "attribution_reasons": list(attribution_reasons),
        "infrastructure_valid": typed_transition["infrastructure_valid"],
        "oracle_derived": oracle,
        "expert_derived": expert,
        "model_identity": model_identity_sha256,
        "config_identity": config_sha256,
        "schema_identity": {
            "action_transition_schema": typed_transition["schema"],
            "runtime_schema": runtime_schema,
        },
        "source_identity": {
            "rollout_manifest_sha256": rollout_manifest_sha256,
            "trace_sha256": trace_sha256,
            "runtime_source_sha256": runtime_source_sha256,
            "scene_signature_sha256": scene_signature,
        },
        "created_at": created_at,
    }
    payload["evidence_id"] = evidence_v2_id_for(payload)
    return EvidenceV2.from_dict(payload)


def update_decision_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result.pop("decision_id", None)
    return result


def update_decision_id_for(payload: Mapping[str, Any]) -> str:
    return stable_content_id("afkupdate", update_decision_identity_payload(payload))


class UpdateDecisionV1(_FrozenRecord):
    """记录 HPK 条目的状态与更新结果。"""

    SCHEMA = UPDATE_DECISION_SCHEMA
    LEGACY_SCHEMA = "tcm/afk/update_decision/v1"

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        path = cls.__name__
        _exact_fields(
            payload,
            required=(
                "schema",
                "decision_id",
                "entry_id",
                "strategy_key_id",
                "from_lifecycle",
                "to_lifecycle",
                "persisted_entry_status",
                "promotion_policy_id",
                "promotion_policy_config_sha256",
                "evidence_set_sha256",
                "evidence_ids",
                "new_evidence_ids",
                "counts",
                "distinct_episode_counts",
                "distinct_scene_signature_counts",
                "posterior_alpha",
                "posterior_beta",
                "estimated_success_probability",
                "lower_confidence_bound",
                "reason_codes",
                "evaluation_ref",
                "created_at",
            ),
            path=path,
        )
        validate_content_id(
            payload["decision_id"],
            prefix="afkupdate",
            path=f"{path}.decision_id",
        )
        validate_content_id(
            payload["entry_id"], prefix="afkentry", path=f"{path}.entry_id"
        )
        validate_content_id(
            payload["strategy_key_id"],
            prefix="afkstrategy",
            path=f"{path}.strategy_key_id",
        )
        from_state = _enum(
            payload["from_lifecycle"],
            allowed=LIFECYCLE_STATES,
            path=f"{path}.from_lifecycle",
        )
        to_state = _enum(
            payload["to_lifecycle"],
            allowed=LIFECYCLE_STATES,
            path=f"{path}.to_lifecycle",
        )
        persisted = _enum(
            payload["persisted_entry_status"],
            allowed=PERSISTED_ENTRY_STATUSES,
            path=f"{path}.persisted_entry_status",
        )
        expected_persisted = (
            "candidate" if to_state == "candidate_for_revalidation" else to_state
        )
        if persisted != expected_persisted:
            _fail(
                f"{path}.persisted_entry_status",
                f"must equal {expected_persisted!r} for {to_state!r}",
            )
        _string(payload["promotion_policy_id"], path=f"{path}.promotion_policy_id")
        _sha256(
            payload["promotion_policy_config_sha256"],
            path=f"{path}.promotion_policy_config_sha256",
        )
        _sha256(payload["evidence_set_sha256"], path=f"{path}.evidence_set_sha256")
        evidence_ids = _string_list(
            payload["evidence_ids"],
            path=f"{path}.evidence_ids",
            sorted_values=True,
        )
        new_evidence_ids = _string_list(
            payload["new_evidence_ids"],
            path=f"{path}.new_evidence_ids",
            sorted_values=True,
        )
        for name, values in (
            ("evidence_ids", evidence_ids),
            ("new_evidence_ids", new_evidence_ids),
        ):
            for index, value in enumerate(values):
                validate_content_id(
                    value, prefix="afkev", path=f"{path}.{name}[{index}]"
                )
        if not set(new_evidence_ids) <= set(evidence_ids):
            _fail(f"{path}.new_evidence_ids", "must be a subset of evidence_ids")
        counts = _mapping(payload["counts"], path=f"{path}.counts")
        _exact_fields(
            counts,
            required=("support", "oppose", "unverified"),
            path=f"{path}.counts",
        )
        for key in counts:
            _integer(counts[key], path=f"{path}.counts.{key}")
        if sum(int(value) for value in counts.values()) != len(evidence_ids):
            _fail(f"{path}.counts", "must sum to evidence_ids length")
        for field in (
            "distinct_episode_counts",
            "distinct_scene_signature_counts",
        ):
            values = _mapping(payload[field], path=f"{path}.{field}")
            _exact_fields(
                values,
                required=("support", "oppose", "unverified"),
                path=f"{path}.{field}",
            )
            for key in values:
                _integer(values[key], path=f"{path}.{field}.{key}")
        _number(payload["posterior_alpha"], path=f"{path}.posterior_alpha", minimum=0.0)
        _number(payload["posterior_beta"], path=f"{path}.posterior_beta", minimum=0.0)
        _number(
            payload["estimated_success_probability"],
            path=f"{path}.estimated_success_probability",
            minimum=0.0,
            maximum=1.0,
        )
        _number(
            payload["lower_confidence_bound"],
            path=f"{path}.lower_confidence_bound",
            minimum=0.0,
            maximum=1.0,
        )
        reasons = _string_list(
            payload["reason_codes"],
            path=f"{path}.reason_codes",
            allowed=UPDATE_REASONS,
        )
        expected_reasons = [
            reason for reason in UPDATE_REASON_ORDER if reason in reasons
        ]
        if reasons != expected_reasons:
            _fail(f"{path}.reason_codes", "must use frozen reason order")
        evaluation_ref = payload["evaluation_ref"]
        if evaluation_ref is not None:
            validate_content_id(
                evaluation_ref, prefix="afkeval", path=f"{path}.evaluation_ref"
            )
        if to_state == "accepted" and evaluation_ref is None:
            _fail(
                f"{path}.evaluation_ref", "accepted lifecycle requires evaluation_ref"
            )
        if to_state != "accepted" and evaluation_ref is not None:
            _fail(
                f"{path}.evaluation_ref",
                "only accepted lifecycle may carry evaluation_ref",
            )
        _utc(payload["created_at"], path=f"{path}.created_at")
        legal = {
            "candidate": {"candidate", "accepted"},
            "accepted": {"accepted", "candidate_for_revalidation"},
            "candidate_for_revalidation": {
                "candidate_for_revalidation",
                "deprecated",
            },
            "deprecated": {"deprecated"},
        }
        if to_state not in legal[from_state]:
            _fail(
                f"{path}.to_lifecycle",
                f"illegal lifecycle transition {from_state!r} -> {to_state!r}",
            )
        expected_id = update_decision_id_for(payload)
        if payload["decision_id"] != expected_id:
            _fail(
                f"{path}.decision_id",
                f"content identity mismatch; expected {expected_id}",
            )

    @property
    def stable_id(self) -> str:
        return str(self._data["decision_id"])


def evidence_set_sha256(evidence: Sequence[EvidenceV2 | Mapping[str, Any]]) -> str:
    records = [
        value if isinstance(value, EvidenceV2) else EvidenceV2.from_dict(value)
        for value in evidence
    ]
    # This is the semantic set identity used by updater decisions. Exact public
    # bytes (including created_at) remain independently hash-pinned by the
    # EvidenceStore member descriptor and collision checks.
    payload = sorted(record.stable_id for record in records)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


__all__ = [
    "ABSTENTION_REASON_ORDER",
    "ABSTENTION_REASONS",
    "EFFECT_VERIFIABILITY_VALUES",
    "EVIDENCE_SOURCE_KINDS",
    "EVIDENCE_V2_SCHEMA",
    "EvidenceV2",
    "LIFECYCLE_STATES",
    "PERSISTED_ENTRY_STATUSES",
    "STRATEGY_KEY_SCHEMA",
    "StrategyKeyV1",
    "UPDATE_DECISION_SCHEMA",
    "UPDATE_REASON_ORDER",
    "UPDATE_REASONS",
    "UpdateDecisionV1",
    "build_evidence_v2",
    "build_strategy_key",
    "evidence_set_sha256",
    "evidence_v2_id_for",
    "resolve_transition_evidence_verdict",
    "strategy_key_id_for",
    "update_decision_id_for",
]
