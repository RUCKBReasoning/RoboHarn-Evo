from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from roboharn_evo.agent.hpk.schemas import (
    CONTEXT_SCHEMA,
    GEOMETRY_SOURCE_CLASSES,
    HARD_CONSTRAINTS,
    HELD_STATES,
    OPERATIONS,
    SCENE_PREDICATES,
    STRATEGY_FAMILIES,
    TARGET_RELATIONS,
    HPKUnresolved,
    HPKValidationError,
    ConditionV1,
    ContextV1,
    EntryV1,
    GeometricStrategyV1,
    TaskStrategyV1,
    canonical_json,
    context_sha256,
    reject_private_transferable,
)
from roboharn_evo.agent.hpk.semantic_knowledge import (
    validate_semantic_knowledge,
    validate_semantic_query,
)

RETRIEVER_PROFILE = "hpk_retriever/v1"
TASK_USAGE_TRANSPORT_AUDIT_PROFILE = "hpk_task_usage_transport_audit/v1"
HIERARCHICAL_PHYSICAL_KNOWLEDGE_CONTEXT_OPEN = "<hierarchical_physical_knowledge_context>"
HIERARCHICAL_PHYSICAL_KNOWLEDGE_CONTEXT_CLOSE = "</hierarchical_physical_knowledge_context>"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENTRY_ID_RE = re.compile(r"^afkentry_[0-9a-f]{64}$")
_BARE_FILE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])[^\s/\\]+\."
    r"(?:avi|bin|csv|h5|hdf5|jpeg|jpg|json|jsonl|log|mp4|npy|npz|pkl|png|"
    r"pt|pth|text|toml|txt|yaml|yml)(?::\d+)?\b",
    re.IGNORECASE,
)
_NUMBER_TOKEN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_EXPLICIT_XYZ_TEXT_RE = re.compile(
    rf"\bx\s*[:=]\s*{_NUMBER_TOKEN}.{{0,48}}"
    rf"\by\s*[:=]\s*{_NUMBER_TOKEN}.{{0,48}}"
    rf"\bz\s*[:=]\s*{_NUMBER_TOKEN}",
    re.IGNORECASE,
)
_AXIS_ASSIGNMENT_RES = tuple(
    re.compile(rf"\b{axis}\s*[:=]\s*{_NUMBER_TOKEN}\b", re.IGNORECASE)
    for axis in ("x", "y", "z")
)
_WORLD_COORDINATE_TEXT_RE = re.compile(
    rf"\bworld\s+(?:coordinates?|positions?)\b.{{0,64}}?{_NUMBER_TOKEN}"
    rf"(?:\s*,?\s+){_NUMBER_TOKEN}(?:\s*,?\s+){_NUMBER_TOKEN}",
    re.IGNORECASE | re.DOTALL,
)
_SPACED_PRIVATE_ID_RE = re.compile(
    r"\b(?:operation\s+candidate|candidate|track|instance)"
    r"(?:\s+(?:id|no\.?|number))?\s+(?:#\s*)?"
    r"[A-Za-z0-9_.:-]+\b",
    re.IGNORECASE,
)
_PRIVATE_HPK_REFERENCE_RE = re.compile(
    r"\b(?:afkentry|afkc|afku|afkz|afkfx|afkev|afkeval|afksnap|"
    r"afkaudit|afkprivrank)_"
    r"[0-9a-f]{64}\b",
    re.IGNORECASE,
)
_PRIVATE_GEOMETRY_DIAGNOSTIC_RE = re.compile(
    rf"(?:\b(?:reach_distance_m|candidate[_\s]+(?:rank|score)|"
    rf"private\s+(?:rank|score)|rank)\s*[:=]\s*{_NUMBER_TOKEN}\b"
    r"|\b(?:geometry_source_class|reach_distance_bucket|geometric_compliance|"
    r"afk_scores?|afk_score_components|baseline_rank|candidate_rank|"
    r"selected_candidate_private_ref|candidate_rejection_details)\b)",
    re.IGNORECASE,
)
_POSE_ORIENTATION_TEXT_RE = re.compile(
    rf"\b(?:absolute\s+(?:pose|position)|world\s+(?:pose|position)|pose|"
    rf"quaternion|quat_wxyz|se\s*\(?3\)?)\b.{{0,64}}?{_NUMBER_TOKEN}"
    rf"(?:\s*,?\s+){_NUMBER_TOKEN}(?:\s*,?\s+){_NUMBER_TOKEN}",
    re.IGNORECASE | re.DOTALL,
)
_PRIVATE_RUN_METADATA_RE = re.compile(
    r"\b(?:seed|episode[_\s]+id|benchmark[_\s]+hidden[_\s]+state|"
    r"oracle[_\s]+contact[_\s]+matrix|correct[_\s]+combination|"
    r"ground[_\s]+truth|episode[_\s]+answer|oracle[_\s]+answer|"
    r"episode[_\s]+specific[_\s]+answer)\b",
    re.IGNORECASE,
)
_RAW_PLANNER_SOURCE_RE = re.compile(
    r"\b(?:planner[_\s]+subtask[_\s]+text|raw[_\s]+planner[_\s]+source|"
    r"planner[_\s]+source[_\s]+text)\b",
    re.IGNORECASE,
)
_CAPABILITY_KEYS = frozenset(
    {
        "task_families",
        "domain_ids",
        "operations",
        "strategy_families",
        "target_relations",
        "hard_constraints",
        "geometry_source_classes",
        "semantic_part_observed",
    }
)
_PRE_PLANNER_QUERY_KEYS = frozenset(
    {
        "task_family",
        "manipulation_phase",
        "operation",
        "manipulated_role",
        "held_state",
        "target_role",
        "target_relation",
        "positive_scene_predicates",
        "satisfied_preconditions",
    }
)
_TRANSPORT_AUDIT_KEYS = frozenset(
    {
        "profile",
        "entry_id",
        "context_sha256",
        "render_count",
        "usage_status",
    }
)


def _fail(path: str, message: str) -> None:
    raise HPKValidationError(f"{path}: {message}")


def _mapping_copy(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return copy.deepcopy(dict(value))


def _exact_fields(
    value: Mapping[str, Any], *, expected: frozenset[str], path: str
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        _fail(path, "fields must match the frozen profile (" + "; ".join(details) + ")")


def _string(value: Any, *, path: str, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string" + (" or null" if nullable else ""))
    return value


def _string_set(
    value: Any,
    *,
    path: str,
    allowed: frozenset[str] | None = None,
    nonempty: bool = False,
) -> frozenset[str]:
    if not isinstance(value, (list, tuple, set, frozenset)) or isinstance(
        value, (str, bytes, bytearray)
    ):
        _fail(path, "must be an array of unique strings")
    result: list[str] = []
    for index, item in enumerate(value):
        text = _string(item, path=f"{path}[{index}]")
        assert text is not None
        if allowed is not None and text not in allowed:
            _fail(f"{path}[{index}]", f"must be one of {sorted(allowed)}")
        result.append(text)
    if len(result) != len(set(result)):
        _fail(path, "must not contain duplicate values")
    if nonempty and not result:
        _fail(path, "must not be empty")
    return frozenset(result)


@dataclass(frozen=True, slots=True)
class HPKDomainCapabilities:
    """Exact capability sets used by both retrieval stages.

    This is an internal typed runtime value, not a new persistent HPK schema.
    """

    task_families: frozenset[str]
    domain_ids: frozenset[str]
    operations: frozenset[str]
    strategy_families: frozenset[str]
    target_relations: frozenset[str]
    hard_constraints: frozenset[str]
    geometry_source_classes: frozenset[str]
    semantic_part_observed: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HPKDomainCapabilities":
        payload = _mapping_copy(value, path=cls.__name__)
        _exact_fields(payload, expected=_CAPABILITY_KEYS, path=cls.__name__)
        semantic_part_observed = payload["semantic_part_observed"]
        if not isinstance(semantic_part_observed, bool):
            _fail(
                f"{cls.__name__}.semantic_part_observed",
                "must be a boolean",
            )
        result = cls(
            task_families=_string_set(
                payload["task_families"],
                path=f"{cls.__name__}.task_families",
                nonempty=True,
            ),
            domain_ids=_string_set(
                payload["domain_ids"],
                path=f"{cls.__name__}.domain_ids",
                nonempty=True,
            ),
            operations=_string_set(
                payload["operations"],
                path=f"{cls.__name__}.operations",
                allowed=OPERATIONS,
                nonempty=True,
            ),
            strategy_families=_string_set(
                payload["strategy_families"],
                path=f"{cls.__name__}.strategy_families",
                allowed=STRATEGY_FAMILIES,
                nonempty=True,
            ),
            target_relations=_string_set(
                payload["target_relations"],
                path=f"{cls.__name__}.target_relations",
                allowed=TARGET_RELATIONS,
            ),
            hard_constraints=_string_set(
                payload["hard_constraints"],
                path=f"{cls.__name__}.hard_constraints",
                allowed=HARD_CONSTRAINTS,
            ),
            geometry_source_classes=_string_set(
                payload["geometry_source_classes"],
                path=f"{cls.__name__}.geometry_source_classes",
                allowed=GEOMETRY_SOURCE_CLASSES,
                nonempty=True,
            ),
            semantic_part_observed=semantic_part_observed,
        )
        reject_private_transferable(result.to_dict(), path=cls.__name__)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_families": sorted(self.task_families),
            "domain_ids": sorted(self.domain_ids),
            "operations": sorted(self.operations),
            "strategy_families": sorted(self.strategy_families),
            "target_relations": sorted(self.target_relations),
            "hard_constraints": sorted(self.hard_constraints),
            "geometry_source_classes": sorted(self.geometry_source_classes),
            "semantic_part_observed": self.semantic_part_observed,
        }


@dataclass(frozen=True, slots=True)
class HPKPrePlannerQuery:
    """Facts reliably known before one planner request."""

    task_family: str | None
    manipulation_phase: str | None
    operation: str | None
    manipulated_role: str | None
    held_state: str | None
    target_role: str | None
    target_relation: str | None
    positive_scene_predicates: frozenset[str]
    satisfied_preconditions: frozenset[str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HPKPrePlannerQuery":
        payload = _mapping_copy(value, path=cls.__name__)
        _exact_fields(payload, expected=_PRE_PLANNER_QUERY_KEYS, path=cls.__name__)

        def optional_text(key: str) -> str | None:
            raw = payload[key]
            if raw is None or raw == "":
                return None
            return _string(raw, path=f"{cls.__name__}.{key}", nullable=True)

        operation = optional_text("operation")
        if operation is not None and operation not in OPERATIONS:
            _fail(f"{cls.__name__}.operation", f"must be one of {sorted(OPERATIONS)}")
        held_state = optional_text("held_state")
        if held_state is not None and held_state not in HELD_STATES:
            _fail(
                f"{cls.__name__}.held_state",
                f"must be one of {sorted(HELD_STATES)}",
            )
        relation = optional_text("target_relation")
        if relation is not None and relation not in TARGET_RELATIONS:
            _fail(
                f"{cls.__name__}.target_relation",
                f"must be one of {sorted(TARGET_RELATIONS)}",
            )
        result = cls(
            task_family=optional_text("task_family"),
            manipulation_phase=optional_text("manipulation_phase"),
            operation=operation,
            manipulated_role=optional_text("manipulated_role"),
            held_state=held_state,
            target_role=optional_text("target_role"),
            target_relation=relation,
            positive_scene_predicates=_string_set(
                payload["positive_scene_predicates"],
                path=f"{cls.__name__}.positive_scene_predicates",
                allowed=SCENE_PREDICATES,
            ),
            satisfied_preconditions=_string_set(
                payload["satisfied_preconditions"],
                path=f"{cls.__name__}.satisfied_preconditions",
            ),
        )
        reject_private_transferable(result.to_dict(), path=cls.__name__)
        return result

    @property
    def has_required_information(self) -> bool:
        required = (
            self.task_family,
            self.manipulation_phase,
            self.operation,
            self.manipulated_role,
            self.held_state,
        )
        if any(item is None for item in required) or self.held_state == "unknown":
            return False
        if self.operation == "place":
            return self.target_role is not None and self.target_relation is not None
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_family": self.task_family,
            "manipulation_phase": self.manipulation_phase,
            "operation": self.operation,
            "manipulated_role": self.manipulated_role,
            "held_state": self.held_state,
            "target_role": self.target_role,
            "target_relation": self.target_relation,
            "positive_scene_predicates": sorted(self.positive_scene_predicates),
            "satisfied_preconditions": sorted(self.satisfied_preconditions),
        }


@dataclass(frozen=True, slots=True)
class HPKRetrievalResult:
    """One deterministic zero-or-one retrieval decision."""

    stage: str
    selected_entry: EntryV1 | None
    context: ContextV1 | None
    geometric_strategy: GeometricStrategyV1 | None
    match_reason: str | None
    rejection_reasons: tuple[str, ...]

    @property
    def selected_entry_id(self) -> str | None:
        if self.selected_entry is None:
            return None
        return str(self.selected_entry["entry_id"])

    @property
    def retrieved_entry_ids(self) -> tuple[str, ...]:
        selected = self.selected_entry_id
        return () if selected is None else (selected,)


def _no_match(stage: str, *reasons: str) -> HPKRetrievalResult:
    order = {
        value: index
        for index, value in enumerate(
            (
                "insufficient_information",
                "condition_unresolved",
                "task_strategy_unresolved",
                "target_unbound",
                "no_exact_match",
                "domain_capability_mismatch",
                "context_budget_exceeded",
                "zero_compliant_candidates",
            )
        )
    }
    unique = sorted(set(reasons), key=order.__getitem__)
    return HPKRetrievalResult(
        stage=stage,
        selected_entry=None,
        context=None,
        geometric_strategy=None,
        match_reason=None,
        rejection_reasons=tuple(unique),
    )


def _selected_without_context(
    *,
    stage: str,
    entry: EntryV1,
    reason: str,
) -> HPKRetrievalResult:
    return HPKRetrievalResult(
        stage=stage,
        selected_entry=entry,
        context=None,
        geometric_strategy=None,
        match_reason="exact_match",
        rejection_reasons=(reason,),
    )


def _entry_sort_key(entry: EntryV1) -> tuple[float, float, int, int, str]:
    statistics = entry["effect_statistics"]
    if entry["provenance"]["source_kind"] == "human_authored":
        # Integration fixtures carry no evidence and expose confidence 0.0.
        # Ignore caller-authored pseudo-statistics so they cannot create a
        # hidden priority among otherwise exact matches.
        return (0.0, 0.0, 0, 0, str(entry["entry_id"]))
    return (
        -float(statistics["lower_confidence_bound"]),
        -float(statistics["estimated_success_probability"]),
        -int(statistics["support_count"]),
        int(statistics["oppose_count"]),
        str(entry["entry_id"]),
    )


def _capability_compatible(
    entry: EntryV1,
    *,
    snapshot: HPKDomainCapabilities,
    current: HPKDomainCapabilities,
) -> bool:
    condition = entry["condition"]
    strategy = entry["geometric_strategy"]
    provenance = entry["provenance"]
    task_family = str(condition["task_family"])
    operation = str(condition["operation"])
    strategy_family = str(strategy["strategy_family"])
    target_relation = strategy["target_relation"]["relation"]
    hard_constraints = set(strategy["hard_constraints"])
    source_class = str(strategy["capability_evidence"]["geometry_source_class"])
    semantic_part_required = strategy["grasp"]["region"] == "semantic_part"
    entry_domains = set(provenance["domain_ids"])

    for capabilities in (snapshot, current):
        if task_family not in capabilities.task_families:
            return False
        if operation not in capabilities.operations:
            return False
        if strategy_family not in capabilities.strategy_families:
            return False
        if (
            target_relation is not None
            and target_relation not in capabilities.target_relations
        ):
            return False
        if not hard_constraints <= capabilities.hard_constraints:
            return False
        if source_class not in capabilities.geometry_source_classes:
            return False
        if semantic_part_required and not capabilities.semantic_part_observed:
            return False
    if not entry_domains <= snapshot.domain_ids:
        return False
    return bool(entry_domains & current.domain_ids)


def _pre_planner_exact_match(entry: EntryV1, query: HPKPrePlannerQuery) -> bool:
    condition = entry["condition"]
    strategy = entry["task_strategy"]
    manipulated = condition["manipulated_object"]
    target = condition["target"]
    exact_pairs = (
        (condition["task_family"], query.task_family),
        (condition["manipulation_phase"], query.manipulation_phase),
        (condition["operation"], query.operation),
        (manipulated["role"], query.manipulated_role),
        (manipulated["held_state"], query.held_state),
        (target["role"], query.target_role),
        (target["relation"], query.target_relation),
        (strategy["operation"], query.operation),
        (strategy["manipulated_role"], query.manipulated_role),
        (strategy["target_role"], query.target_role),
        (strategy["target_relation"], query.target_relation),
        (strategy["manipulation_phase"], query.manipulation_phase),
    )
    if any(left != right for left, right in exact_pairs):
        return False
    if not set(condition["scene_predicates"]) <= set(query.positive_scene_predicates):
        return False
    return set(condition["preconditions"]) <= set(query.satisfied_preconditions)


def _typed_condition(value: Any) -> ConditionV1 | None:
    if value is None or isinstance(value, HPKUnresolved):
        return None
    if isinstance(value, ConditionV1):
        return value
    return ConditionV1.from_dict(_mapping_copy(value, path="current_condition"))


def _typed_task_strategy(value: Any) -> TaskStrategyV1 | None:
    if value is None or isinstance(value, HPKUnresolved):
        return None
    if isinstance(value, TaskStrategyV1):
        return value
    return TaskStrategyV1.from_dict(_mapping_copy(value, path="current_task_strategy"))


def _reject_private_context_text(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_private_context_text(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_private_context_text(item, path=f"{path}[{index}]")
    elif path.endswith((".schema", ".entry_id")):
        # Frozen schema identifiers intentionally contain slash-delimited
        # version components; Context's sole entry ID is explicitly allowed.
        return
    elif isinstance(value, str):
        if (
            HIERARCHICAL_PHYSICAL_KNOWLEDGE_CONTEXT_OPEN in value
            or HIERARCHICAL_PHYSICAL_KNOWLEDGE_CONTEXT_CLOSE in value
        ):
            _fail(path, "model-visible text must not contain an HPK context delimiter")
        if "/" in value or "\\" in value or _BARE_FILE_TOKEN_RE.search(value):
            _fail(path, "local path text is forbidden in planner context")
        axes_assigned = all(pattern.search(value) for pattern in _AXIS_ASSIGNMENT_RES)
        if (
            _EXPLICIT_XYZ_TEXT_RE.search(value)
            or axes_assigned
            or _WORLD_COORDINATE_TEXT_RE.search(value)
            or _POSE_ORIENTATION_TEXT_RE.search(value)
        ):
            _fail(path, "world coordinate text is forbidden in planner context")
        if _SPACED_PRIVATE_ID_RE.search(value):
            _fail(
                path, "runtime-private identifier text is forbidden in planner context"
            )
        if _PRIVATE_HPK_REFERENCE_RE.search(value):
            _fail(path, "private HPK reference text is forbidden in planner context")
        if _PRIVATE_GEOMETRY_DIAGNOSTIC_RE.search(value):
            _fail(path, "private geometry diagnostic is forbidden in planner context")
        if _PRIVATE_RUN_METADATA_RE.search(value):
            _fail(path, "private run metadata is forbidden in planner context")
        if _RAW_PLANNER_SOURCE_RE.search(value):
            _fail(path, "raw planner source text is forbidden in planner context")


def _validated_context(value: ContextV1 | Mapping[str, Any]) -> ContextV1:
    typed = (
        value
        if isinstance(value, ContextV1)
        else ContextV1.from_dict(_mapping_copy(value, path="ContextV1"))
    )
    _reject_private_context_text(typed.to_dict(), path="ContextV1")
    summary = typed["condition_summary"]
    guidance = typed["task_guidance"]
    effect = typed["expected_effect"]
    disclosure = typed["source_disclosure"]
    for key in (
        "operation",
        "manipulated_role",
        "target_role",
        "target_relation",
        "manipulation_phase",
    ):
        if summary[key] != guidance[key]:
            _fail(
                f"ContextV1.task_guidance.{key}",
                "must match condition_summary",
            )
    if effect["effect_type"] != guidance["operation"]:
        _fail(
            "ContextV1.expected_effect.effect_type",
            "must match task guidance operation",
        )
    if guidance["operation"] == "place" and (
        guidance["target_role"] is None or guidance["target_relation"] is None
    ):
        _fail("ContextV1.task_guidance", "place requires a target role and relation")

    source_kind = disclosure["source_kind"]
    scope = disclosure["acceptance_scope"]
    expert = disclosure["expert_prior_used"]
    human = disclosure["human_prior_used"]
    oracle = disclosure["oracle_evidence_used"]
    learned = disclosure.get("learned_hpk", disclosure.get("learned_afk"))
    formal = disclosure["formal_evaluation_eligible"]
    if oracle and (scope != "oracle_diagnostic" or formal):
        _fail(
            "ContextV1.source_disclosure",
            "oracle evidence requires oracle_diagnostic and formal ineligibility",
        )
    if source_kind == "agent_generated":
        expected_scope = "oracle_diagnostic" if oracle else "formal_no_prior"
        if expert or human or not learned or scope != expected_scope:
            _fail(
                "ContextV1.source_disclosure",
                "agent-generated disclosure is inconsistent",
            )
        if not oracle and not formal:
            _fail(
                "ContextV1.source_disclosure",
                "non-oracle agent-generated context must be formal eligible",
            )
    elif source_kind == "benchmark_expert":
        if not expert or human or not learned:
            _fail(
                "ContextV1.source_disclosure",
                "benchmark-expert disclosure is inconsistent",
            )
        if not oracle and scope != "expert_prior":
            _fail(
                "ContextV1.source_disclosure",
                "benchmark expert requires expert_prior scope",
            )
    elif (
        expert
        or oracle
        or not human
        or learned
        or formal
        or scope != "integration_only"
        or typed["confidence"] != 0.0
    ):
        _fail(
            "ContextV1.source_disclosure",
            "human-authored disclosure is inconsistent",
        )
    return typed


def build_hierarchical_physical_knowledge_context(
    entry: EntryV1 | Mapping[str, Any],
) -> ContextV1:
    """Return the exact transferable planner projection for one accepted entry."""

    typed = (
        entry
        if isinstance(entry, EntryV1)
        else EntryV1.from_dict(_mapping_copy(entry, path="EntryV1"))
    )
    if not typed.accepted:
        _fail("EntryV1.status", "planner context requires an accepted entry")
    condition = typed["condition"]
    task_strategy = typed["task_strategy"]
    expected_effect = typed["expected_effect"]
    statistics = typed["effect_statistics"]
    provenance = typed["provenance"]
    manipulated = condition["manipulated_object"]
    target = condition["target"]
    is_human_fixture = provenance["source_kind"] == "human_authored"
    confidence = (
        0.0 if is_human_fixture else float(statistics["lower_confidence_bound"])
    )
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        _fail("ContextV1.confidence", "must be finite and in [0, 1]")
    payload = {
        "schema": CONTEXT_SCHEMA,
        "entry_id": typed["entry_id"],
        "condition_summary": {
            "task_family": condition["task_family"],
            "manipulation_phase": condition["manipulation_phase"],
            "operation": condition["operation"],
            "manipulated_role": manipulated["role"],
            "held_state": manipulated["held_state"],
            "target_role": target["role"],
            "target_relation": target["relation"],
            "scene_predicates": condition["scene_predicates"],
            "preconditions": condition["preconditions"],
        },
        "task_guidance": {
            "operation": task_strategy["operation"],
            "manipulated_role": task_strategy["manipulated_role"],
            "target_role": task_strategy["target_role"],
            "target_relation": task_strategy["target_relation"],
            "manipulation_phase": task_strategy["manipulation_phase"],
            "subgoal_purpose": task_strategy["subgoal_purpose"],
            "preferred_arm": task_strategy["preferred_arm"],
        },
        "expected_effect": {
            "effect_type": expected_effect["effect_type"],
            "predicates": expected_effect["expected_predicates"],
        },
        "confidence": confidence,
        "source_disclosure": {
            "source_kind": provenance["source_kind"],
            "acceptance_scope": typed["acceptance_scope"],
            "expert_prior_used": bool(provenance["expert_derived"]),
            "human_prior_used": bool(provenance["human_prior_used"]),
            "oracle_evidence_used": bool(provenance["oracle_derived"]),
            "learned_hpk": bool(provenance.get("learned_hpk", provenance.get("learned_afk"))),
            "formal_evaluation_eligible": bool(
                provenance["formal_evaluation_eligible"]
            ),
        },
    }
    reject_private_transferable(payload, path="ContextV1")
    return _validated_context(ContextV1.from_dict(payload))


def hierarchical_physical_knowledge_context_sha256(
    context: ContextV1 | Mapping[str, Any],
) -> str:
    typed = _validated_context(context)
    return context_sha256(typed)


def _canonical_context_json(context: ContextV1 | Mapping[str, Any]) -> str:
    typed = _validated_context(context)
    rendered = canonical_json(typed.to_dict())
    return rendered


def hierarchical_physical_knowledge_context_char_count(
    context: ContextV1 | Mapping[str, Any],
) -> int:
    """Return exact final-block characters without constructing that block."""

    context_json = _canonical_context_json(context)
    return (
        len(HIERARCHICAL_PHYSICAL_KNOWLEDGE_CONTEXT_OPEN)
        + 1
        + len(context_json)
        + 1
        + len(HIERARCHICAL_PHYSICAL_KNOWLEDGE_CONTEXT_CLOSE)
    )


def render_hierarchical_physical_knowledge_context(
    context: ContextV1 | Mapping[str, Any],
) -> str:
    """验证并渲染模型使用的 HPK 上下文。"""

    rendered = _canonical_context_json(context)
    return (
        HIERARCHICAL_PHYSICAL_KNOWLEDGE_CONTEXT_OPEN
        + "\n"
        + rendered
        + "\n"
        + HIERARCHICAL_PHYSICAL_KNOWLEDGE_CONTEXT_CLOSE
    )


def _transport_receipt(context: ContextV1) -> dict[str, Any]:
    return {
        "profile": TASK_USAGE_TRANSPORT_AUDIT_PROFILE,
        "entry_id": context["entry_id"],
        "context_sha256": hierarchical_physical_knowledge_context_sha256(context),
        "render_count": 1,
        "usage_status": "injected_use_unverified",
    }


def validate_hpk_task_usage_transport_audit(
    value: Mapping[str, Any],
    *,
    expected_context: ContextV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the non-model server/client rendering receipt."""

    payload = _mapping_copy(value, path="hpk_task_usage_audit")
    _exact_fields(
        payload,
        expected=_TRANSPORT_AUDIT_KEYS,
        path="hpk_task_usage_audit",
    )
    if payload["profile"] != TASK_USAGE_TRANSPORT_AUDIT_PROFILE:
        _fail("hpk_task_usage_audit.profile", "unsupported profile")
    entry_id = payload["entry_id"]
    if not isinstance(entry_id, str) or _ENTRY_ID_RE.fullmatch(entry_id) is None:
        _fail("hpk_task_usage_audit.entry_id", "invalid knowledge entry identifier")
    context_hash = payload["context_sha256"]
    if not isinstance(context_hash, str) or _SHA256_RE.fullmatch(context_hash) is None:
        _fail("hpk_task_usage_audit.context_sha256", "must be lowercase SHA-256")
    if payload["render_count"] != 1 or isinstance(payload["render_count"], bool):
        _fail("hpk_task_usage_audit.render_count", "must equal integer 1")
    if payload["usage_status"] != "injected_use_unverified":
        _fail(
            "hpk_task_usage_audit.usage_status",
            "must be 'injected_use_unverified'",
        )
    typed = _validated_context(expected_context)
    if entry_id != typed["entry_id"]:
        _fail("hpk_task_usage_audit.entry_id", "does not match sent context")
    expected_hash = hierarchical_physical_knowledge_context_sha256(typed)
    if context_hash != expected_hash:
        _fail(
            "hpk_task_usage_audit.context_sha256",
            "does not match sent context",
        )
    reject_private_transferable(payload, path="hpk_task_usage_audit")
    return payload


def validate_and_render_hierarchical_physical_knowledge_context(
    context: ContextV1 | Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Shared server entry point: validate, render once, and issue a receipt."""

    typed = _validated_context(context)
    rendered = render_hierarchical_physical_knowledge_context(typed)
    receipt = validate_hpk_task_usage_transport_audit(
        _transport_receipt(typed),
        expected_context=typed,
    )
    return rendered, receipt


class HPKRetriever:
    """执行内存中的精确条件检索。"""

    profile = RETRIEVER_PROFILE

    def __init__(
        self,
        entries: Sequence[EntryV1 | Mapping[str, Any]],
        *,
        snapshot_capabilities: HPKDomainCapabilities | Mapping[str, Any],
    ) -> None:
        if isinstance(entries, (str, bytes, bytearray)) or not isinstance(
            entries, Sequence
        ):
            _fail("HPKRetriever.entries", "must be an array")
        typed_entries: list[EntryV1] = []
        seen_ids: set[str] = set()
        for index, entry in enumerate(entries):
            typed = (
                entry
                if isinstance(entry, EntryV1)
                else EntryV1.from_dict(
                    _mapping_copy(entry, path=f"HPKRetriever.entries[{index}]")
                )
            )
            if typed["status"] != "accepted" or not typed.accepted:
                _fail(
                    f"HPKRetriever.entries[{index}].status",
                    "static retrieval requires accepted entries only",
                )
            entry_id = str(typed["entry_id"])
            if entry_id in seen_ids:
                _fail("HPKRetriever.entries", f"duplicate entry_id {entry_id!r}")
            seen_ids.add(entry_id)
            typed_entries.append(typed)
        if not typed_entries:
            _fail("HPKRetriever.entries", "must not be empty")
        self._entries = tuple(typed_entries)
        self._snapshot_capabilities = HPKDomainCapabilities.from_mapping(
            snapshot_capabilities.to_dict()
            if isinstance(snapshot_capabilities, HPKDomainCapabilities)
            else snapshot_capabilities
        )

    @property
    def entries(self) -> tuple[EntryV1, ...]:
        return self._entries

    @property
    def snapshot_capabilities(self) -> HPKDomainCapabilities:
        return self._snapshot_capabilities

    def retrieve_pre_planner(
        self,
        query: HPKPrePlannerQuery | Mapping[str, Any],
        *,
        current_capabilities: HPKDomainCapabilities | Mapping[str, Any],
        max_prompt_chars: int,
    ) -> HPKRetrievalResult:
        typed_query = HPKPrePlannerQuery.from_mapping(
            query.to_dict() if isinstance(query, HPKPrePlannerQuery) else query
        )
        capabilities = HPKDomainCapabilities.from_mapping(
            current_capabilities.to_dict()
            if isinstance(current_capabilities, HPKDomainCapabilities)
            else current_capabilities
        )
        if isinstance(max_prompt_chars, bool) or not isinstance(max_prompt_chars, int):
            _fail("max_prompt_chars", "must be an integer")
        if max_prompt_chars < 1:
            _fail("max_prompt_chars", "must be >= 1")
        if not typed_query.has_required_information:
            return _no_match("pre_planner_task", "insufficient_information")

        exact = [
            entry
            for entry in self._entries
            if _pre_planner_exact_match(entry, typed_query)
        ]
        if not exact:
            return _no_match("pre_planner_task", "no_exact_match")
        compatible = [
            entry
            for entry in exact
            if _capability_compatible(
                entry,
                snapshot=self._snapshot_capabilities,
                current=capabilities,
            )
        ]
        if not compatible:
            return _no_match("pre_planner_task", "domain_capability_mismatch")
        selected = sorted(compatible, key=_entry_sort_key)[0]
        context = build_hierarchical_physical_knowledge_context(selected)
        if hierarchical_physical_knowledge_context_char_count(context) > max_prompt_chars:
            return _selected_without_context(
                stage="pre_planner_task",
                entry=selected,
                reason="context_budget_exceeded",
            )
        return HPKRetrievalResult(
            stage="pre_planner_task",
            selected_entry=selected,
            context=context,
            geometric_strategy=None,
            match_reason="exact_match",
            rejection_reasons=(),
        )

    def retrieve_geometry(
        self,
        condition: ConditionV1 | Mapping[str, Any] | HPKUnresolved | None,
        task_strategy: TaskStrategyV1 | Mapping[str, Any] | HPKUnresolved | None,
        *,
        current_capabilities: HPKDomainCapabilities | Mapping[str, Any],
        target_bound: bool,
        operation_hint: str | None = None,
    ) -> HPKRetrievalResult:
        if not isinstance(target_bound, bool):
            _fail("target_bound", "must be a boolean")
        if operation_hint is not None and operation_hint not in OPERATIONS:
            _fail("operation_hint", f"must be one of {sorted(OPERATIONS)} or null")
        if operation_hint == "place" and not target_bound:
            return _no_match("post_binding_geometry", "target_unbound")
        typed_condition = _typed_condition(condition)
        typed_strategy = _typed_task_strategy(task_strategy)
        unresolved: list[str] = []
        if typed_condition is None:
            unresolved.append("condition_unresolved")
        if typed_strategy is None:
            unresolved.append("task_strategy_unresolved")
        if unresolved:
            return _no_match("post_binding_geometry", *unresolved)
        assert typed_condition is not None
        assert typed_strategy is not None
        if operation_hint is not None and (
            typed_condition["operation"] != operation_hint
            or typed_strategy["operation"] != operation_hint
        ):
            return _no_match("post_binding_geometry", "task_strategy_unresolved")
        if typed_strategy["operation"] == "place" and not target_bound:
            return _no_match("post_binding_geometry", "target_unbound")
        capabilities = HPKDomainCapabilities.from_mapping(
            current_capabilities.to_dict()
            if isinstance(current_capabilities, HPKDomainCapabilities)
            else current_capabilities
        )

        exact = []
        for entry in self._entries:
            entry_condition = ConditionV1.from_dict(entry["condition"])
            entry_strategy = TaskStrategyV1.from_dict(entry["task_strategy"])
            if (
                entry_condition.stable_id == typed_condition.stable_id
                and entry_strategy.stable_id == typed_strategy.stable_id
            ):
                exact.append(entry)
        if not exact:
            return _no_match("post_binding_geometry", "no_exact_match")
        compatible = [
            entry
            for entry in exact
            if _capability_compatible(
                entry,
                snapshot=self._snapshot_capabilities,
                current=capabilities,
            )
        ]
        if not compatible:
            return _no_match("post_binding_geometry", "domain_capability_mismatch")
        selected = sorted(compatible, key=_entry_sort_key)[0]
        geometric_strategy = GeometricStrategyV1.from_dict(
            selected["geometric_strategy"]
        )
        return HPKRetrievalResult(
            stage="post_binding_geometry",
            selected_entry=selected,
            context=None,
            geometric_strategy=geometric_strategy,
            match_reason="exact_match",
            rejection_reasons=(),
        )


def _semantic_attribute_score(knowledge_value: str, query_value: str) -> int | None:
    if knowledge_value == query_value:
        return 2
    if "unknown" in {knowledge_value, query_value}:
        return 0
    return None


def retrieve_semantic_knowledge(
    entries: Sequence[Mapping[str, Any]],
    query: Mapping[str, Any],
    *,
    accepted_only: bool = True,
) -> tuple[dict[str, Any], ...]:
    """根据语义条件对 HPK 记录排序。"""

    typed_query = validate_semantic_query(query)
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for source_index, source in enumerate(entries):
        knowledge = validate_semantic_knowledge(source)
        if accepted_only and knowledge["status"] != "accepted":
            continue
        if knowledge["condition"] != typed_query["condition"]:
            continue
        if knowledge["task_strategy"]["operation"] != typed_query["task_strategy"][
            "operation"
        ]:
            continue
        if knowledge["expected_effect"] != typed_query["expected_effect"]:
            continue
        knowledge_arm = knowledge["task_strategy"]["preferred_arm"]
        query_arm = typed_query["task_strategy"]["preferred_arm"]
        if knowledge_arm != query_arm and "either arm" not in {
            knowledge_arm,
            query_arm,
        }:
            continue

        score = 0
        for field in ("category", "color", "shape", "role", "held_state"):
            field_score = _semantic_attribute_score(
                knowledge["object"][field], typed_query["object"][field]
            )
            if field_score is None:
                break
            score += field_score
        else:
            ranked.append((score, source_index, knowledge))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(item[2] for item in ranked)


__all__ = [
    "HIERARCHICAL_PHYSICAL_KNOWLEDGE_CONTEXT_CLOSE",
    "HIERARCHICAL_PHYSICAL_KNOWLEDGE_CONTEXT_OPEN",
    "RETRIEVER_PROFILE",
    "TASK_USAGE_TRANSPORT_AUDIT_PROFILE",
    "HPKDomainCapabilities",
    "HPKPrePlannerQuery",
    "HPKRetrievalResult",
    "HPKRetriever",
    "hierarchical_physical_knowledge_context_char_count",
    "hierarchical_physical_knowledge_context_sha256",
    "build_hierarchical_physical_knowledge_context",
    "render_hierarchical_physical_knowledge_context",
    "retrieve_semantic_knowledge",
    "validate_hpk_task_usage_transport_audit",
    "validate_and_render_hierarchical_physical_knowledge_context",
]
