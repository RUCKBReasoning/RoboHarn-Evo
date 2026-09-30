from __future__ import annotations

import copy
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.family_router import KnowledgeFamilyRouter
from roboharn_evo.agent.hpk.compatibility import normalize_hpk_v3_config
from roboharn_evo.agent.hpk.goal_consistency import (
    GoalConsistentCandidateSetV31,
    RuntimeGoalBindingV31,
    build_realization_hypothesis_v31,
    build_subtask_goal_contract_v31,
    filter_goal_consistent_candidates_v31,
    merge_runtime_goal_bindings_v31,
    runtime_candidate_missing_prerequisite_v31,
    runtime_goal_semantics_v31,
    unresolved_feasibility_report_v31,
)
from roboharn_evo.agent.hpk.hierarchical_knowledge import (
    ActionEvidenceV31,
    ActionEvidenceV3,
    ActionKnowledgeV3,
    HPKV3ValidationError,
    RealizationHypothesisV31,
    RuntimeFeasibilityReportV31,
    SubtaskGoalContractV31,
    SubtaskKnowledgeV3,
    subtask_goal_contract_json_schema,
)
from roboharn_evo.agent.hpk.incremental_maintainer import (
    IncrementalKnowledgeMaintainer,
    load_atomic_knowledge,
)
from roboharn_evo.agent.hpk.knowledge_family import (
    KnowledgeFamilyCatalogV1,
    render_action_knowledge_card,
    render_task_knowledge_card,
)
from roboharn_evo.agent.hpk.rgb_evidence import RGBEvidenceIndex
from roboharn_evo.agent.hpk.rgb_retrieval import RGBRetrievalContext, RuntimeMultimodalBackend
from roboharn_evo.agent.hpk.rgb_maintenance import RGBKnowledgeReviewer
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (
    HierarchicalReflectionBackend,
    _strict_response,
)

_MODES = frozenset({"off", "rule", "full"})
_ACTIONS = frozenset({"grasp", "place", "contact"})
_RUNTIME_ONLY_KEY_TOKENS = frozenset(
    {
        "bbox",
        "camera",
        "candidate",
        "confidence",
        "frame",
        "hash",
        "id",
        "ids",
        "joint",
        "mask",
        "matrix",
        "path",
        "pixel",
        "pose",
        "quaternion",
        "rank",
        "revision",
        "score",
        "step",
        "timestamp",
        "world",
    }
)
_RUNTIME_REFERENCE_RE = re.compile(
    r"\b(?:track|instance|candidate)[_-][a-z0-9_-]+\b|"
    r"\b(?:track|instance|candidate)\s+(?:id\s*)?\d+\b",
    flags=re.IGNORECASE,
)
_ARM_BINDING_RE = re.compile(
    r"\b(left|right)\s+(?:arms?|grippers?|hands?|end[\s-]?effectors?)\b",
    flags=re.IGNORECASE,
)

_SUBTASK_RETRIEVAL_PROMPT = """Select and ground reusable task knowledge by meaning.

Compare the current overall goal, task state, current planner subtask, and relevant relations with the supplied supported SubtaskKnowledge candidates. Select a candidate only when its functional subtask, purpose, ordering role, completion condition, and next-step relation apply now. When knowledge applies, write grounded_subtask as the current executable subtask: apply the reusable strategy while preserving every current object, target, relation, location, and arm binding that remains relevant. Never replace a specific current object with an unbound generic object and never invent an ID. When no knowledge applies, select null and return grounded_subtask null. Treat paraphrases as equivalent and ignore only wording and source index. Return strict JSON only. The source index is temporary and must not be copied into knowledge."""

_ACTION_RETRIEVAL_PROMPT = """Ground reusable action knowledge into current candidate geometry.

First select supported ActionKnowledge whose action, object state, relative geometry, and expected physical effect apply to the current query. Then select the currently available candidate geometry that best realizes that knowledge. When a Goal Contract is present, every candidate shown already preserves it; change only the realization ranking and never reinterpret or replace its object, target, relation, or expected effect. Treat paraphrases as equivalent, but preserve every distinction that can change applicability, grounded realization, or physical effect. Decide from the current condition whether object attributes, relations, locations, or arm constraints matter; do not assume they are incidental. Ignore only wording, source index, and candidate list order. Return null selections when no knowledge or current candidate geometry is semantically compatible. Return strict JSON only; indices are temporary and never become knowledge fields."""


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    return normalized


def _mode(value: Any) -> str:
    normalized = _text(value, label="mode").casefold()
    if normalized not in _MODES:
        raise ValueError("mode must be off, rule, or full")
    return normalized


def _finite_json_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    try:
        parsed = json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"{label} must be an object")
    forbidden = {
        "candidate_id",
        "track_id",
        "episode_id",
        "knowledge_id",
        "proposal_id",
    }

    def contains_runtime_id(item: Any) -> bool:
        if isinstance(item, dict):
            return bool(forbidden.intersection(item)) or any(
                contains_runtime_id(child) for child in item.values()
            )
        if isinstance(item, list):
            return any(contains_runtime_id(child) for child in item)
        return False

    if contains_runtime_id(parsed):
        raise ValueError(f"{label} must contain semantic geometry, not runtime IDs")
    return parsed


def _natural_language_projection(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _natural_language_projection(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_natural_language_projection(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.replace("_", " ").split())
    return value


def _semantic_projection(value: Any) -> Any:
    """Remove executor identity and numeric geometry from a VLM retrieval query."""

    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key).strip()
            tokens = {token for token in key.casefold().split("_") if token}
            if tokens.intersection(_RUNTIME_ONLY_KEY_TOKENS) or (
                key != "instances" and key.casefold().endswith("_instances")
            ):
                continue
            child = _semantic_projection(item)
            if child not in (None, "", [], {}):
                projected[key.replace("_", " ")] = child
        return projected
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        projected_items = [
            child
            for item in value
            if (child := _semantic_projection(item)) not in (None, "", [], {})
        ]
        return projected_items[:16]
    if isinstance(value, str):
        text = " ".join(value.replace("_", " ").split())
        return _RUNTIME_REFERENCE_RE.sub("the current object", text)
    if isinstance(value, bool):
        return value
    # Counts, coordinates, scores, steps, and other numeric runtime details do
    # not decide whether natural-language knowledge is semantically applicable.
    if isinstance(value, (int, float)) or value is None:
        return None
    return " ".join(str(value).split())


def _semantic_text(value: Any, *, fallback: str) -> str:
    projected = _semantic_projection(value)
    if projected in (None, "", [], {}):
        return fallback
    text = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
    return text[:12000]


def build_subtask_knowledge_query(
    *,
    overall_goal: str,
    scene_memory: Mapping[str, Any] | None,
    active_skill: Any = None,
) -> SubtaskKnowledgeQuery:
    """Build one task-neutral semantic query from the current planner view."""

    scene = dict(scene_memory or {})
    skill_state: dict[str, Any] = {}
    if active_skill is not None:
        for field in ("skill_name", "status", "preferred_arm", "semantic_tags"):
            value = getattr(active_skill, field, None)
            if value not in (None, "", [], {}):
                skill_state[field] = value
    task_state = _semantic_text(
        {"scene": scene, "active skill": skill_state},
        fallback="the current task state has not yet been observed",
    )
    relation_values: list[str] = []
    for field in (
        "task_focus",
        "operation_targets",
        "reference_regions",
        "spatial_state",
        "manipulation_state",
        "uncertainty",
    ):
        value = scene.get(field)
        if value not in (None, "", [], {}):
            relation_values.append(
                _semantic_text(value, fallback="current relation unavailable")
            )
    return SubtaskKnowledgeQuery(
        overall_goal=overall_goal,
        task_state=task_state,
        relevant_relations=tuple(relation_values),
    )


@dataclass(frozen=True, slots=True)
class SubtaskKnowledgeQuery:
    overall_goal: str
    task_state: str
    relevant_relations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "overall_goal", _text(self.overall_goal, label="overall_goal")
        )
        object.__setattr__(
            self, "task_state", _text(self.task_state, label="task_state")
        )
        if isinstance(self.relevant_relations, (str, bytes)):
            raise TypeError("relevant_relations must be a sequence of strings")
        object.__setattr__(
            self,
            "relevant_relations",
            tuple(
                _text(value, label="relevant_relation")
                for value in self.relevant_relations
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_goal": self.overall_goal,
            "task_state": self.task_state,
            "relevant_relations": list(self.relevant_relations),
        }


@dataclass(frozen=True, slots=True)
class ActionKnowledgeQuery:
    action: str
    object_description: str
    held_state: str | None = None
    support_relation: str | None = None
    target_relation: str | None = None
    intended_effect: str | None = None
    candidate_geometry: tuple[Mapping[str, Any], ...] = ()
    goal_contract: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        action = _text(self.action, label="action").casefold()
        if action not in _ACTIONS:
            raise ValueError("action must be grasp, place, or contact")
        object.__setattr__(self, "action", action)
        object.__setattr__(
            self,
            "object_description",
            _text(self.object_description, label="object_description"),
        )
        for field in (
            "held_state",
            "support_relation",
            "target_relation",
            "intended_effect",
        ):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _text(value, label=field))
        if isinstance(self.candidate_geometry, (str, bytes)):
            raise TypeError("candidate_geometry must be a sequence of objects")
        object.__setattr__(
            self,
            "candidate_geometry",
            tuple(
                _finite_json_mapping(value, label=f"candidate_geometry[{index}]")
                for index, value in enumerate(self.candidate_geometry)
            ),
        )
        if self.goal_contract is not None:
            object.__setattr__(
                self,
                "goal_contract",
                SubtaskGoalContractV31(self.goal_contract).to_dict(),
            )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "action": self.action,
            "object_description": self.object_description,
            "held_state": self.held_state,
            "support_relation": self.support_relation,
            "target_relation": self.target_relation,
            "intended_effect": self.intended_effect,
            "candidate_geometry": _natural_language_projection(
                copy.deepcopy(list(self.candidate_geometry))
            ),
        }
        if self.goal_contract is not None:
            result["goal_contract"] = copy.deepcopy(dict(self.goal_contract))
        return result


def build_action_knowledge_query(
    *,
    scene_memory: Mapping[str, Any] | None,
    instance: Mapping[str, Any],
    arm: str,
    action_mode: str,
    eligible_candidates: Sequence[Mapping[str, Any]],
    geometry_policy: Any = None,
    goal_contract: SubtaskGoalContractV31 | Mapping[str, Any] | None = None,
) -> ActionKnowledgeQuery:
    """Project current eligible geometry into one task-neutral action query."""

    from roboharn_evo.agent.hpk.candidate_features import candidate_semantic_features
    from roboharn_evo.agent.hpk.schemas import HPKUnresolved

    scene = dict(scene_memory or {})
    candidate_geometry: list[dict[str, Any]] = []
    for candidate in eligible_candidates:
        features = candidate_semantic_features(
            scene,
            candidate,
            geometry_policy=geometry_policy,
        )
        if isinstance(features, HPKUnresolved):
            candidate_geometry.append(
                {
                    "action_mode": str(candidate.get("action_mode", action_mode)),
                    "geometry_source_class": "unknown",
                }
            )
        else:
            candidate_geometry.append(features.to_dict())

    held_state: str | None = None
    for source in (instance,):
        raw = source.get("held_state")
        if isinstance(raw, str) and raw.strip():
            held_state = " ".join(raw.replace("_", " ").split())
            break
    manipulation = scene.get("manipulation_state")
    if isinstance(manipulation, Mapping):
        arm_state = manipulation.get(str(arm).strip().casefold())
        if isinstance(arm_state, Mapping) and "held_instance_id" in arm_state:
            held_state = (
                "selected arm is holding an object"
                if str(arm_state.get("held_instance_id", "") or "").strip()
                else "selected arm is not holding an object"
            )

    def semantic_field(*names: str) -> str | None:
        for source in (instance, scene):
            for name in names:
                value = source.get(name)
                if isinstance(value, str) and value.strip():
                    return " ".join(value.replace("_", " ").split())
        return None

    relations = {
        str(value.get("target_relation", "") or "").strip()
        for value in candidate_geometry
        if str(value.get("target_relation", "") or "").strip()
    }
    target_relation = next(iter(relations)) if len(relations) == 1 else None
    return ActionKnowledgeQuery(
        action=action_mode,
        object_description=_semantic_text(
            {"object": dict(instance)},
            fallback="the current grounded object",
        ),
        held_state=held_state,
        support_relation=semantic_field("support_relation", "support_state"),
        target_relation=target_relation
        or semantic_field("target_relation", "placement_relation"),
        intended_effect=_EXPECTED_ACTION_EFFECTS[str(action_mode).strip().casefold()][
            "physical_effect"
        ],
        candidate_geometry=tuple(candidate_geometry),
        goal_contract=(
            None
            if goal_contract is None
            else (
                goal_contract.to_dict()
                if isinstance(goal_contract, SubtaskGoalContractV31)
                else dict(goal_contract)
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class SubtaskKnowledgeMatch:
    knowledge: SubtaskKnowledgeV3
    grounded_subtask: str
    reason: str
    subtask_goal: SubtaskGoalContractV31 | None = None

    def compact_context(self) -> dict[str, Any]:
        return self.knowledge.to_dict()


@dataclass(frozen=True, slots=True)
class ActionKnowledgeMatch:
    knowledge: ActionKnowledgeV3
    selected_candidate_geometry: dict[str, Any]
    reason: str

    def compact_context(self) -> dict[str, Any]:
        value = self.knowledge.to_dict()
        return {
            "condition": value["condition"],
            "geometric_strategy": value["geometric_strategy"],
            "expected_effect": value["expected_effect"],
            "evidence_summary": value["evidence_summary"],
            "status": value["status"],
        }


def _selection_schema(
    *,
    with_candidate: bool,
    with_grounded_subtask: bool = False,
    with_goal_contract: bool = False,
) -> dict[str, Any]:
    nullable_index = {
        "anyOf": [
            {"type": "integer", "minimum": 0},
            {"type": "null"},
        ]
    }
    properties: dict[str, Any] = {
        "selected_knowledge_index": nullable_index,
        "reason": {"type": "string", "pattern": r"^[^_]+$"},
    }
    if with_candidate:
        properties["selected_candidate_geometry_index"] = copy.deepcopy(nullable_index)
    if with_grounded_subtask:
        properties["grounded_subtask"] = {
            "anyOf": [
                {"type": "string", "minLength": 1},
                {"type": "null"},
            ]
        }
    if with_goal_contract:
        properties["subtask_goal"] = {
            "anyOf": [subtask_goal_contract_json_schema(), {"type": "null"}]
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def hierarchical_retrieval_output_json_schema(schema_name: str) -> dict[str, Any]:
    """Return the canonical provider schema for one hierarchical retrieval call."""

    if schema_name in {"hpk_v3_subtask_retrieval", "hpk_v31_subtask_retrieval"}:
        return _selection_schema(
            with_candidate=False,
            with_grounded_subtask=True,
            with_goal_contract=schema_name == "hpk_v31_subtask_retrieval",
        )
    if schema_name == "hpk_v3_action_retrieval":
        return _selection_schema(with_candidate=True)
    if schema_name == "hpk_v3_action_prompt_grounding":
        natural_text = {
            "type": "string",
            "minLength": 1,
            "pattern": r"^[^_]+$",
        }
        return {
            "type": "object",
            "properties": {
                "selected_knowledge_index": {
                    "anyOf": [
                        {"type": "integer", "minimum": 0},
                        {"type": "null"},
                    ]
                },
                "grounded_action_guidance": {"anyOf": [natural_text, {"type": "null"}]},
                "reason": natural_text,
            },
            "required": [
                "selected_knowledge_index",
                "grounded_action_guidance",
                "reason",
            ],
            "additionalProperties": False,
        }
    raise ValueError("unsupported hierarchical retrieval schema name")


def _supported(values: Sequence[Any], record_type: type) -> tuple[Any, ...]:
    typed = tuple(
        value if isinstance(value, record_type) else record_type(value)
        for value in values
    )
    return tuple(value for value in typed if value["status"] == "supported")


def _retrieval_input_tokens(completion: Any) -> int | None:
    audit = getattr(completion, "audit", None)
    if not isinstance(audit, Mapping):
        return None
    sources = [audit]
    for key in ("usage", "provider_usage"):
        nested = audit.get(key)
        if isinstance(nested, Mapping):
            sources.append(nested)
    for source in sources:
        for key in ("input_tokens", "prompt_tokens"):
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return None


def _retrieval_call_audit(
    completion: Any,
    *,
    input_text: str,
    elapsed_ms: float,
) -> dict[str, Any]:
    backend_audit = getattr(completion, "audit", None) or {}
    return {
        "retrieval_calls": backend_audit.get("retrieval_calls", 1),
        "retrieval_input_characters": backend_audit.get("retrieval_input_characters", len(input_text)),
        "retrieval_input_tokens": _retrieval_input_tokens(completion),
        "retrieval_latency_ms": round(elapsed_ms, 3),
        **({"rgb_requests": backend_audit["rgb_requests"]} if "rgb_requests" in backend_audit else {}),
    }


def _merge_retrieval_audits(
    route_audit: Mapping[str, Any],
    final_call_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(route_audit))
    if final_call_audit is None:
        return result
    for key in (
        "retrieval_calls",
        "retrieval_input_characters",
        "retrieval_latency_ms",
    ):
        result[key] = result.get(key, 0) + final_call_audit.get(key, 0)
    route_tokens = result.get("retrieval_input_tokens")
    final_tokens = final_call_audit.get("retrieval_input_tokens")
    if route_tokens is None:
        result["retrieval_input_tokens"] = final_tokens
    elif final_tokens is not None:
        result["retrieval_input_tokens"] = int(route_tokens) + int(final_tokens)
    result["retrieval_latency_ms"] = round(float(result["retrieval_latency_ms"]), 3)
    if "rgb_requests" in final_call_audit:
        result["rgb_requests"] = copy.deepcopy(final_call_audit["rgb_requests"])
    return result


def _selected_atomic_source_index(
    selected_payload: Any,
    knowledge: Sequence[Any],
    source_indices: Sequence[int],
) -> int | None:
    if not isinstance(selected_payload, Mapping):
        return None
    selected = copy.deepcopy(dict(selected_payload))
    for source_index, value in zip(source_indices, knowledge, strict=True):
        if value.to_dict() == selected:
            return int(source_index)
    return None


class VLMSubtaskKnowledgeRetriever:
    def __init__(self, backend: HierarchicalReflectionBackend, *, rgb_context: RGBRetrievalContext | None = None) -> None:
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must expose complete")
        self._backend = backend
        self._rgb_context = rgb_context
        self.last_call_audit: dict[str, Any] | None = None

    def retrieve(
        self,
        knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
        query: SubtaskKnowledgeQuery,
        *,
        baseline_subtask: str,
        goal_contract: SubtaskGoalContractV31 | None = None,
        require_goal_contract: bool = False,
    ) -> SubtaskKnowledgeMatch | None:
        self.last_call_audit = None
        candidates = _supported(knowledge, SubtaskKnowledgeV3)
        if self._rgb_context is not None:
            candidates = tuple(value for value in candidates if self._rgb_context.eligible("task", value))
        if not candidates:
            return None
        payload = {
            "query": query.to_dict(),
            "current_planner_subtask": _text(
                baseline_subtask,
                label="baseline_subtask",
            ),
            "supported_task_knowledge": [
                {"source_index": index, "knowledge": render_task_knowledge_card(value)}
                for index, value in enumerate(candidates)
            ],
        }
        instructions = _SUBTASK_RETRIEVAL_PROMPT
        if require_goal_contract:
            payload["current_planner_goal"] = (
                None if goal_contract is None else goal_contract.to_dict()
            )
            instructions += (
                "\nAlso return subtask_goal. When rewriting the current subtask, "
                "return the complete semantic Goal Contract for grounded_subtask "
                "in this same response. Keep the goal aligned with the current "
                "object, target, relation, expected effect, and completion condition. "
                "The contract contains natural-language values without IDs, "
                "coordinates, or poses. For a place goal, supply its required "
                "target role and relation. For null knowledge selection or a "
                "non-manipulation subtask return subtask_goal: null."
                " Ground one currently executable primitive. The rewritten "
                "subtask and its contract must refer to the same immediate "
                "operation and local completion condition. Reusable Task "
                "Knowledge may describe a broader sequence: use its purpose "
                "and ordering information to guide this step without expanding "
                "the current primitive into that entire sequence."
                " Include only target roles and relations required for this "
                "primitive; later destinations belong in its broader purpose."
            )
        input_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        started = time.perf_counter()
        try:
            completion_arguments = dict(
                instructions=instructions,
                output_schema=_selection_schema(
                    with_candidate=False,
                    with_grounded_subtask=True,
                    with_goal_contract=require_goal_contract,
                ),
                schema_name=(
                    "hpk_v31_subtask_retrieval"
                    if require_goal_contract
                    else "hpk_v3_subtask_retrieval"
                ),
            )
            completion = (
                self._backend.complete(input_text=input_text, images=(), **completion_arguments)
                if self._rgb_context is None
                else self._rgb_context.complete(self._backend, kind="task", candidates=candidates, payload=payload, **completion_arguments)
            )
        except Exception:
            self.last_call_audit = _retrieval_call_audit(
                None,
                input_text=input_text,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
            raise
        self.last_call_audit = _retrieval_call_audit(
            completion,
            input_text=input_text,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        result = _strict_response(completion.output)
        expected_fields = {
            "selected_knowledge_index",
            "grounded_subtask",
            "reason",
        }
        if require_goal_contract:
            expected_fields.add("subtask_goal")
        if set(result) != expected_fields:
            raise HPKV3ValidationError("subtask retrieval response fields mismatch")
        selected = result["selected_knowledge_index"]
        grounded_subtask = result["grounded_subtask"]
        reason = _text(result["reason"], label="retrieval reason")
        if selected is None:
            if grounded_subtask is not None or result.get("subtask_goal") is not None:
                raise HPKV3ValidationError(
                    "subtask retrieval returned grounding without knowledge"
                )
            return None
        if (
            isinstance(selected, bool)
            or not isinstance(selected, int)
            or not 0 <= selected < len(candidates)
        ):
            raise HPKV3ValidationError("subtask retrieval selected index is invalid")
        return SubtaskKnowledgeMatch(
            candidates[selected],
            _text(grounded_subtask, label="grounded_subtask"),
            reason,
            (
                SubtaskGoalContractV31(result["subtask_goal"])
                if result.get("subtask_goal") is not None
                else None
            ),
        )


class VLMActionKnowledgeRetriever:
    def __init__(self, backend: HierarchicalReflectionBackend, *, rgb_context: RGBRetrievalContext | None = None) -> None:
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must expose complete")
        self._backend = backend
        self._rgb_context = rgb_context
        self.last_call_audit: dict[str, Any] | None = None

    def retrieve(
        self,
        knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
        query: ActionKnowledgeQuery,
    ) -> ActionKnowledgeMatch | None:
        self.last_call_audit = None
        candidates = tuple(
            value
            for value in _supported(knowledge, ActionKnowledgeV3)
            if value["condition"]["action"] == query.action
            and (self._rgb_context is None or self._rgb_context.eligible("action", value))
        )
        if not candidates or not query.candidate_geometry:
            return None
        payload = {
            "query": query.to_dict(),
            "supported_action_knowledge": [
                {
                    "source_index": index,
                    "knowledge": render_action_knowledge_card(value),
                }
                for index, value in enumerate(candidates)
            ],
        }
        input_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        started = time.perf_counter()
        try:
            completion_arguments = dict(
                instructions=_ACTION_RETRIEVAL_PROMPT,
                output_schema=_selection_schema(with_candidate=True),
                schema_name="hpk_v3_action_retrieval",
            )
            completion = (
                self._backend.complete(input_text=input_text, images=(), **completion_arguments)
                if self._rgb_context is None
                else self._rgb_context.complete(self._backend, kind="action", candidates=candidates, payload=payload, **completion_arguments)
            )
        except Exception:
            self.last_call_audit = _retrieval_call_audit(
                None,
                input_text=input_text,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
            raise
        self.last_call_audit = _retrieval_call_audit(
            completion,
            input_text=input_text,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        result = _strict_response(completion.output)
        if set(result) != {
            "selected_knowledge_index",
            "selected_candidate_geometry_index",
            "reason",
        }:
            raise HPKV3ValidationError("action retrieval response fields mismatch")
        knowledge_index = result["selected_knowledge_index"]
        geometry_index = result["selected_candidate_geometry_index"]
        reason = _text(result["reason"], label="retrieval reason")
        if knowledge_index is None or geometry_index is None:
            if knowledge_index is not None or geometry_index is not None:
                raise HPKV3ValidationError(
                    "action retrieval must select both knowledge and candidate geometry"
                )
            return None
        for label, selected, length in (
            ("knowledge", knowledge_index, len(candidates)),
            ("candidate geometry", geometry_index, len(query.candidate_geometry)),
        ):
            if (
                isinstance(selected, bool)
                or not isinstance(selected, int)
                or not 0 <= selected < length
            ):
                raise HPKV3ValidationError(
                    f"action retrieval selected {label} index is invalid"
                )
        return ActionKnowledgeMatch(
            candidates[knowledge_index],
            copy.deepcopy(dict(query.candidate_geometry[geometry_index])),
            reason,
        )


class AgentApiHierarchicalRetrievalBackend:
    """Use the rollout's existing structured text endpoint for v3 retrieval."""

    def __init__(
        self,
        planner_url: str,
        *,
        timeout_sec: int = 600,
        headers: Mapping[str, str] | None = None,
        multimodal_backend: HierarchicalReflectionBackend | None = None,
        execution_evidence_enabled: bool = True,
    ) -> None:
        from roboharn_evo.agent.hpk.vlm_strategy_proposer import (
            AgentApiStrategyProposalBackend,
        )

        self._backend = AgentApiStrategyProposalBackend(
            planner_url,
            timeout_sec=timeout_sec,
            headers=headers,
        )
        self._multimodal_backend = multimodal_backend
        self.execution_evidence_enabled = execution_evidence_enabled

    def complete(
        self,
        *,
        instructions: str,
        input_text: str,
        images: Sequence[Any],
        output_schema: Mapping[str, Any],
        schema_name: str,
    ) -> Any:
        if not self.execution_evidence_enabled:
            from roboharn_evo.agent.execution_feedback import DIRECT_FEEDBACK_INSTRUCTIONS

            instructions += "\n" + DIRECT_FEEDBACK_INSTRUCTIONS
        if images or (schema_name.startswith("hpk_rgb_") and self._multimodal_backend is not None):
            if self._multimodal_backend is None:
                raise ValueError("RGB retrieval requires a configured multimodal backend")
            return self._multimodal_backend.complete(
                instructions=instructions, input_text=input_text, images=images,
                output_schema=output_schema, schema_name=schema_name,
            )
        completion = self._backend.complete(
            instructions=instructions,
            input_text=input_text,
            output_schema=output_schema,
            schema_name=schema_name,
        )
        from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (
            HierarchicalBackendCompletion,
        )

        return HierarchicalBackendCompletion(
            output=completion.output,
            audit=completion.audit,
        )


def build_hierarchical_hpk_runtime(
    config: Mapping[str, Any],
    *,
    planner_agent_api: Mapping[str, Any],
) -> HierarchicalHPKRetrievalRuntime:
    """构建 HPK v3 的 rule 或 full 运行模式。"""

    config = normalize_hpk_v3_config(config)
    mode = _mode(config.get("mode", "off"))
    if mode == "off":
        raise ValueError("off mode does not construct a hierarchical HPK runtime")
    goal_consistency_enabled = config.get("hpk_goal_consistency_enabled", False)
    if not isinstance(goal_consistency_enabled, bool):
        raise TypeError("hpk_v3.hpk_goal_consistency_enabled must be a boolean")
    if mode == "rule":
        return HierarchicalHPKRetrievalRuntime(
            mode="rule",
            task_knowledge=(),
            action_knowledge=(),
            hpk_goal_consistency_enabled=goal_consistency_enabled,
        )

    store_root = _text(config.get("store_root"), label="hpk_v3.store_root")
    from roboharn_evo.agent.hpk.hierarchical_store import load_hierarchical_store

    task_knowledge, action_knowledge = load_hierarchical_store(store_root)
    evidence_index = RGBEvidenceIndex.load(store_root) if (Path(store_root) / "evidence_index.jsonl").is_file() else None
    if evidence_index is not None:
        evidence_index.validate_knowledge(task_knowledge, action_knowledge)
    headers = {
        str(key): str(value)
        for key, value in dict(planner_agent_api.get("extra_headers", {}) or {}).items()
    }
    auth_token = str(planner_agent_api.get("auth_token", "") or "")
    if auth_token:
        headers[str(planner_agent_api.get("auth_header", "Authorization"))] = auth_token
    timeout_sec = int(
        config.get(
            "timeout_sec",
            planner_agent_api.get("timeout_sec", 600),
        )
    )
    retrieval_rgb_enabled = config.get("retrieval_rgb_enabled", False)
    maintenance_rgb_enabled = config.get("maintenance_rgb_enabled", False)
    if not isinstance(retrieval_rgb_enabled, bool) or not isinstance(maintenance_rgb_enabled, bool):
        raise TypeError("RGB configuration fields must be boolean")
    multimodal_backend = None
    rgb_context = None
    if (retrieval_rgb_enabled or maintenance_rgb_enabled) and evidence_index is None:
        raise ValueError("RGB-enabled knowledge requires its evidence index")
    if evidence_index is not None:
        multimodal_backend = RuntimeMultimodalBackend(
            str(planner_agent_api.get("server_url", "")),
            model=str(config.get("retrieval_model", "gpt-5.5")),
            reasoning_effort=str(config.get("retrieval_reasoning_effort", "xhigh")),
            max_images=int(config.get("max_rgb_images", 32)), timeout_sec=timeout_sec,
        )
    if retrieval_rgb_enabled:
        rgb_context = RGBRetrievalContext(
            evidence_index, task_knowledge=task_knowledge,
            action_knowledge=action_knowledge, max_images=int(config.get("max_rgb_images", 32)),
            camera_names=config.get("historical_rgb_cameras"),
        )
        rgb_context.audit_root = Path(_text(config.get("rgb_audit_root"), label="hpk_v3.rgb_audit_root"))
    backend = AgentApiHierarchicalRetrievalBackend(
        str(planner_agent_api.get("server_url", "")),
        timeout_sec=timeout_sec,
        headers=headers,
        multimodal_backend=multimodal_backend,
        execution_evidence_enabled=config.get("execution_evidence_enabled", True),
    )
    from roboharn_evo.agent.hpk.family_router import FamilyRoutingConfig
    from roboharn_evo.agent.hpk.family_store import (
        KnowledgeFamilyStoreError,
        load_knowledge_family_catalog,
    )

    family_catalog = None
    family_catalog_error = None
    try:
        family_catalog = load_knowledge_family_catalog(
            store_root,
            task_knowledge=task_knowledge,
            action_knowledge=action_knowledge,
        )
    except KnowledgeFamilyStoreError as exc:
        family_catalog_error = type(exc).__name__
    family_router = KnowledgeFamilyRouter(
        backend,
        catalog=family_catalog,
        config=FamilyRoutingConfig.from_mapping(config.get("family_routing")),
        catalog_error=family_catalog_error,
    )
    from roboharn_evo.agent.hpk.semantic_consolidator import VLMKnowledgeConsolidator

    knowledge_consolidator = VLMKnowledgeConsolidator(backend)
    incremental_maintainer = (
        None
        if family_catalog is None
        else IncrementalKnowledgeMaintainer(
            backend,
            consolidator=knowledge_consolidator,
            rgb_reviewer=None if evidence_index is None else RGBKnowledgeReviewer(backend, max_images=int(config.get("max_rgb_images", 32)), camera_names=config.get("historical_rgb_cameras")),
        )
    )

    trajectory_reflector = None
    reflection_transport = None
    reflection_authorization = None
    reflection_enabled = config.get("hierarchical_reflection", False)
    if not isinstance(reflection_enabled, bool):
        raise TypeError("hpk_v3.hierarchical_reflection must be a boolean")
    if evidence_index is not None and config.get("knowledge_updates_enabled", True):
        if not reflection_enabled or incremental_maintainer is None:
            raise ValueError("updating RGB-indexed knowledge requires trajectory reflection and a Skill catalog")
    if reflection_enabled:
        from urllib.parse import urlsplit, urlunsplit

        from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (
            ExistingGPTMultimodalReflectorBackend,
            VLMHierarchicalReflector,
        )
        from roboharn_evo.agent.reflector.multimodal_transport import (
            ImageEgressAuthorization,
            OpenAICompatibleMultimodalTransport,
        )

        planner_url = urlsplit(str(planner_agent_api.get("server_url", "")))
        service_url = urlunsplit((planner_url.scheme, planner_url.netloc, "", "", ""))
        reflection_authorization = ImageEgressAuthorization.operator_granted(
            assertion="HPK v3 full-mode real rollout reflection",
            scope="hpk_v3_real_rollout_action_observations",
        )
        reflection_transport = OpenAICompatibleMultimodalTransport(
            service_url=service_url,
            model=str(config.get("reflection_model", "gpt-5.5")),
            reasoning_effort=str(config.get("reflection_reasoning_effort", "xhigh")),
            timeout_sec=timeout_sec,
            max_images=int(config.get("max_reflection_images", 16)),
        )
        trajectory_reflector = VLMHierarchicalReflector(
            ExistingGPTMultimodalReflectorBackend(
                reflection_transport,
                authorization=reflection_authorization,
            )
        )

    return HierarchicalHPKRetrievalRuntime(
        mode="full",
        task_knowledge=task_knowledge,
        action_knowledge=action_knowledge,
        subtask_retriever=VLMSubtaskKnowledgeRetriever(backend, rgb_context=rgb_context),
        action_retriever=VLMActionKnowledgeRetriever(backend, rgb_context=rgb_context),
        store_root=store_root,
        knowledge_consolidator=knowledge_consolidator,
        trajectory_reflector=trajectory_reflector,
        reflection_transport=reflection_transport,
        reflection_authorization=reflection_authorization,
        max_reflection_images=int(config.get("max_reflection_images", 16)),
        family_router=family_router,
        family_catalog=family_catalog,
        incremental_maintainer=incremental_maintainer,
        hpk_goal_consistency_enabled=goal_consistency_enabled,
        knowledge_updates_enabled=config.get("knowledge_updates_enabled", True),
        rgb_context=rgb_context,
        evidence_index=evidence_index,
        maintenance_rgb_enabled=maintenance_rgb_enabled,
    )


def apply_subtask_knowledge_decision(
    *,
    mode: str,
    baseline_subtask: str,
    query: SubtaskKnowledgeQuery,
    knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
    retriever: VLMSubtaskKnowledgeRetriever | None,
    preferred_arm: str | None = None,
    goal_contract: SubtaskGoalContractV31 | None = None,
    require_goal_contract: bool = False,
) -> tuple[str, dict[str, Any]]:
    normalized_mode = _mode(mode)
    baseline = _text(baseline_subtask, label="baseline_subtask")
    if normalized_mode != "full":
        return baseline, {
            "mode": normalized_mode,
            "subtask_before": baseline,
            "subtask_after": baseline,
            "knowledge_adopted": False,
        }
    if retriever is None:
        raise TypeError("full mode requires a subtask retriever")
    selected = retriever.retrieve(
        knowledge,
        query,
        baseline_subtask=baseline,
        **(
            {"goal_contract": goal_contract, "require_goal_contract": True}
            if require_goal_contract
            else {}
        ),
    )
    application_mode = "baseline"
    after = baseline
    opaque_references_preserved = True
    arm_binding_preserved = True
    baseline_arms = {
        match.group(1).casefold() for match in _ARM_BINDING_RE.finditer(baseline)
    }
    required_arms = set(baseline_arms)
    if isinstance(preferred_arm, str):
        normalized_arm = " ".join(preferred_arm.strip().casefold().split())
        if normalized_arm in {"left", "left arm"}:
            required_arms.add("left")
        elif normalized_arm in {"right", "right arm"}:
            required_arms.add("right")
    proposed_arms: set[str] = set()
    if selected is not None:
        proposed = selected.grounded_subtask
        binding_terms = {
            match.group(0).casefold()
            for match in _RUNTIME_REFERENCE_RE.finditer(baseline)
        }
        proposed_terms = {
            match.group(0).casefold()
            for match in _RUNTIME_REFERENCE_RE.finditer(proposed)
        }
        proposed_arms = {
            match.group(1).casefold() for match in _ARM_BINDING_RE.finditer(proposed)
        }
        opaque_references_preserved = binding_terms.issubset(proposed_terms)
        arm_binding_preserved = not (
            (
                proposed_arms
                and required_arms
                and not proposed_arms.issubset(required_arms)
            )
            or not baseline_arms.issubset(proposed_arms)
        )
        if opaque_references_preserved and arm_binding_preserved:
            after = proposed
            application_mode = "model_grounded_rewrite"
        else:
            strategy = selected.knowledge["subtask_strategy"]
            guidance_parts = [
                f"reusable strategy: {strategy['subtask']}",
                f"purpose: {strategy['purpose']}",
                f"broader strategy completion condition: {strategy['completion_condition']}",
            ]
            if strategy["planned_next_subtask"] is not None:
                guidance_parts.append(f"then: {strategy['planned_next_subtask']}")
            after = (
                baseline
                + "\nHPK guidance; preserve the current object, target, relation, "
                "and arm bindings: " + "; ".join(guidance_parts)
            )
            application_mode = "baseline_bindings_preserved_with_guidance"
    audit = {
        "mode": normalized_mode,
        "subtask_before": baseline,
        "subtask_after": after,
        "knowledge_adopted": selected is not None,
        "application_mode": application_mode,
        "retrieval_reason": None if selected is None else selected.reason,
        "retrieved_knowledge": (
            None if selected is None else selected.compact_context()
        ),
        "binding_preservation": {
            "opaque_references_preserved": opaque_references_preserved,
            "arm_binding_preserved": arm_binding_preserved,
            "required_arms": sorted(required_arms),
            "proposed_arms": sorted(proposed_arms),
        },
    }
    if require_goal_contract:
        selected_goal = (
            selected.subtask_goal
            if selected is not None and application_mode == "model_grounded_rewrite"
            else goal_contract
        )
        audit["subtask_goal"] = (
            None if selected_goal is None else selected_goal.to_dict()
        )
    return after, audit


def _natural_runtime_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().lower().replace("_", " ").split())
    if not text or text == "unknown":
        return None
    text = text.replace("align principal axis 0", "align with the first principal axis")
    text = text.replace(
        "align principal axis 1", "align with the second principal axis"
    )
    return text


def _runtime_geometry_to_v3(
    action: str,
    features: Mapping[str, Any],
) -> dict[str, Any]:
    geometry: dict[str, Any] = {"avoid": []}
    direction = _natural_runtime_value(features.get("approach_direction_bucket"))
    family = _natural_runtime_value(features.get("approach_family"))
    orientation = _natural_runtime_value(features.get("orientation_relation"))
    if direction is not None:
        geometry["approach_direction"] = (
            "from above" if direction == "above" else direction
        )
    if family is not None:
        geometry["approach_reference"] = (
            "object frame" if family == "principal axis relative" else family
        )
    if orientation is not None:
        geometry["orientation_relation"] = orientation
    if action == "grasp":
        region = _natural_runtime_value(features.get("grasp_region"))
        if region not in {None, "observed surface"}:
            geometry["interaction_region"] = region
    elif action == "place":
        relation = _natural_runtime_value(features.get("target_relation"))
        if relation is not None:
            geometry["placement_relation"] = relation
        if (
            features.get("support_valid") is True
            and features.get("target_region_free") is True
        ):
            geometry["clearance_or_support_constraint"] = (
                "use a valid free support region"
            )
    elif action == "contact":
        part = _natural_runtime_value(features.get("semantic_part"))
        if part is not None:
            geometry["interaction_region"] = part
        relation = _natural_runtime_value(features.get("target_relation"))
        if relation is not None:
            geometry["contact_relation"] = relation
    return geometry


_EXPECTED_ACTION_EFFECTS = {
    "grasp": {
        "physical_effect": "the object becomes attached to the gripper",
        "verification_observation": "the object moves with the gripper after lift",
    },
    "place": {
        "physical_effect": "the object is released at the intended support",
        "verification_observation": "the object remains at the intended support after release",
    },
    "contact": {
        "physical_effect": "the intended contact changes the target state",
        "verification_observation": "the expected target state change is observed after contact",
    },
}


@dataclass(slots=True)
class _PendingOnlineAction:
    local_ref: str
    action: str
    arm: str
    candidate_ref: str
    condition: dict[str, Any]
    geometric_strategy: dict[str, Any]
    expected_effect: dict[str, Any]
    goal_contract: SubtaskGoalContractV31 | None = None
    runtime_goal_binding: RuntimeGoalBindingV31 | None = None
    realization_status: str = "unresolved"
    evidence: ActionEvidenceV3 | ActionEvidenceV31 | None = None


@dataclass(frozen=True, slots=True)
class _PreparedGoalContext:
    contract: SubtaskGoalContractV31
    runtime_binding: RuntimeGoalBindingV31
    selected_candidate_ref: str


def _runtime_validation(
    action_effect: Mapping[str, Any],
) -> tuple[str | None, Mapping[str, Any] | None]:
    for action, field in (
        ("grasp", "runtime_grasp_validation"),
        ("place", "runtime_place_validation"),
        ("contact", "runtime_contact_validation"),
    ):
        value = action_effect.get(field)
        if isinstance(value, Mapping):
            return action, value
    return None, None


def _runtime_evidence_verdict(
    action_effect: Mapping[str, Any],
    validation: Mapping[str, Any] | None,
) -> str:
    if validation is None:
        return "unverified"
    if validation.get("applicable") is not True:
        return "unverified"
    verified = validation.get("verified")
    effect_verified = (
        str(action_effect.get("effect_verified", "unverified") or "unverified")
        .strip()
        .lower()
    )
    if verified is True and effect_verified == "true":
        return "support"
    if verified is False and effect_verified == "false":
        return "oppose"
    return "unverified"


def _action_verdict_counts(
    values: Sequence[ActionKnowledgeV3],
) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for value in values:
        action = value["condition"]["action"]
        for verdict in ("support", "oppose", "unverified"):
            result[(action, verdict)] = result.get((action, verdict), 0) + int(
                value["evidence_summary"][verdict]
            )
    return {key: count for key, count in result.items() if count}


class HierarchicalHPKRetrievalRuntime:
    """Small full-mode adapter for the existing planner and candidate seams."""

    def __init__(
        self,
        *,
        mode: str,
        task_knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
        action_knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
        subtask_retriever: VLMSubtaskKnowledgeRetriever | None = None,
        action_retriever: VLMActionKnowledgeRetriever | None = None,
        store_root: str | Path | None = None,
        knowledge_consolidator: Any = None,
        trajectory_reflector: Any = None,
        reflection_transport: Any = None,
        reflection_authorization: Any = None,
        max_reflection_images: int = 16,
        family_router: KnowledgeFamilyRouter | None = None,
        family_catalog: KnowledgeFamilyCatalogV1 | Mapping[str, Any] | None = None,
        incremental_maintainer: IncrementalKnowledgeMaintainer | None = None,
        hpk_goal_consistency_enabled: bool = False,
        knowledge_updates_enabled: bool = True,
        rgb_context: RGBRetrievalContext | None = None,
        evidence_index: RGBEvidenceIndex | None = None,
        maintenance_rgb_enabled: bool = False,
    ) -> None:
        self.mode = _mode(mode)
        if not isinstance(knowledge_updates_enabled, bool):
            raise TypeError("knowledge_updates_enabled must be a boolean")
        self.knowledge_updates_enabled = knowledge_updates_enabled
        self._rgb_context = rgb_context
        self.retrieval_rgb_enabled = rgb_context is not None
        self._evidence_index = evidence_index
        self.maintenance_rgb_enabled = maintenance_rgb_enabled
        self.task_knowledge = tuple(
            value
            if isinstance(value, SubtaskKnowledgeV3)
            else SubtaskKnowledgeV3(value)
            for value in task_knowledge
        )
        self.action_knowledge = tuple(
            value if isinstance(value, ActionKnowledgeV3) else ActionKnowledgeV3(value)
            for value in action_knowledge
        )
        if self.mode == "full" and (
            subtask_retriever is None or action_retriever is None
        ):
            raise TypeError("full mode requires independent task and action retrievers")
        self._subtask_retriever = subtask_retriever
        self._action_retriever = action_retriever
        self._preplanner_query: SubtaskKnowledgeQuery | None = None
        self._last_action_usage_audit: dict[str, Any] | None = None
        self._store_root = None if store_root is None else Path(store_root)
        if knowledge_consolidator is not None and not callable(
            getattr(knowledge_consolidator, "consolidate", None)
        ):
            raise TypeError("knowledge_consolidator must expose consolidate")
        self._knowledge_consolidator = knowledge_consolidator
        if trajectory_reflector is not None and not callable(
            getattr(trajectory_reflector, "reflect", None)
        ):
            raise TypeError("trajectory_reflector must expose reflect")
        if reflection_transport is not None and not callable(
            getattr(reflection_transport, "capability_preflight", None)
        ):
            raise TypeError("reflection_transport must expose capability_preflight")
        if (
            isinstance(max_reflection_images, bool)
            or not isinstance(max_reflection_images, int)
            or max_reflection_images <= 0
        ):
            raise ValueError("max_reflection_images must be a positive integer")
        self._trajectory_reflector = trajectory_reflector
        self._reflection_transport = reflection_transport
        self._reflection_authorization = reflection_authorization
        self._max_reflection_images = max_reflection_images
        if family_router is not None and not isinstance(
            family_router, KnowledgeFamilyRouter
        ):
            raise TypeError("family_router must be KnowledgeFamilyRouter or None")
        self._family_router = family_router
        self._family_catalog = (
            None
            if family_catalog is None
            else family_catalog
            if isinstance(family_catalog, KnowledgeFamilyCatalogV1)
            else KnowledgeFamilyCatalogV1(family_catalog)
        )
        if incremental_maintainer is not None and not isinstance(
            incremental_maintainer, IncrementalKnowledgeMaintainer
        ):
            raise TypeError(
                "incremental_maintainer must be IncrementalKnowledgeMaintainer or None"
            )
        if (self._family_catalog is None) != (incremental_maintainer is None):
            raise ValueError(
                "family_catalog and incremental_maintainer must be configured together"
            )
        self._incremental_maintainer = incremental_maintainer
        if not isinstance(hpk_goal_consistency_enabled, bool):
            raise TypeError("hpk_goal_consistency_enabled must be a boolean")
        self.hpk_goal_consistency_enabled = hpk_goal_consistency_enabled
        self._current_task_goal_source: dict[str, Any] | None = None
        self._current_task_goal_binding = RuntimeGoalBindingV31()
        self._current_goal_contract: SubtaskGoalContractV31 | None = None
        self._last_runtime_feasibility_report: RuntimeFeasibilityReportV31 | None = None
        self._prepared_goal_contexts: dict[str, _PreparedGoalContext] = {}
        self._next_online_action_ref = 1
        self._pending_online_actions: dict[str, _PendingOnlineAction] = {}

    @property
    def hierarchical_full_enabled(self) -> bool:
        return self.mode == "full"

    @property
    def hierarchical_v3_runtime(self) -> bool:
        return True

    def set_preplanner_query(self, query: SubtaskKnowledgeQuery | None) -> None:
        if query is not None and not isinstance(query, SubtaskKnowledgeQuery):
            raise TypeError("query must be SubtaskKnowledgeQuery or None")
        self._preplanner_query = query

    def set_current_rgb(self, images, *, audit_root: str | Path | None = None) -> None:
        if self._rgb_context is not None:
            self._rgb_context.set_current(images, audit_root=audit_root)

    def reset_episode(self) -> None:
        self._preplanner_query = None
        self._last_action_usage_audit = None
        self._current_task_goal_source = None
        self._current_task_goal_binding = RuntimeGoalBindingV31()
        self._current_goal_contract = None
        self._last_runtime_feasibility_report = None
        self._prepared_goal_contexts.clear()
        self._pending_online_actions.clear()
        self._next_online_action_ref = 1

    def set_current_task_strategy(
        self,
        task_strategy: Mapping[str, Any] | SubtaskGoalContractV31 | None,
        *,
        runtime_binding: RuntimeGoalBindingV31 | Mapping[str, Any] | None = None,
    ) -> None:
        """Install one transient structured Task strategy for the next actions.

        This is the gradual adapter seam for v3.1.  It never writes a Store and
        deliberately does not infer roles or relations from free-form task text.
        """

        if task_strategy is None:
            self._current_task_goal_source = None
            self._current_task_goal_binding = RuntimeGoalBindingV31()
        elif isinstance(task_strategy, SubtaskGoalContractV31):
            self._current_task_goal_source = task_strategy.to_dict()
            self._current_task_goal_binding = RuntimeGoalBindingV31.from_value(
                runtime_binding
            )
        else:
            self._current_task_goal_source = _finite_json_mapping(
                task_strategy,
                label="task_strategy",
            )
            self._current_task_goal_binding = RuntimeGoalBindingV31.from_value(
                runtime_binding
            )
        self._current_goal_contract = None
        self._last_runtime_feasibility_report = None

    @property
    def current_task_goal(self) -> SubtaskGoalContractV31 | None:
        source = self._current_task_goal_source
        if source is None or "operation" not in source:
            return None
        return SubtaskGoalContractV31(source)

    @property
    def current_goal_contract(self) -> SubtaskGoalContractV31 | None:
        return (
            None
            if self._current_goal_contract is None
            else SubtaskGoalContractV31(self._current_goal_contract.to_dict())
        )

    @property
    def last_runtime_feasibility_report(
        self,
    ) -> RuntimeFeasibilityReportV31 | None:
        return (
            None
            if self._last_runtime_feasibility_report is None
            else RuntimeFeasibilityReportV31(
                self._last_runtime_feasibility_report.to_dict()
            )
        )

    @staticmethod
    def _instance_private_ref(instance: Mapping[str, Any]) -> str | None:
        for key in ("instance_id", "track_id"):
            value = str(instance.get(key, "") or "").strip()
            if value:
                return value
        return None

    def resolve_goal_consistent_candidates(
        self,
        *,
        scene_memory: Mapping[str, Any] | None,
        instance: Mapping[str, Any],
        action_mode: str,
        eligible_candidates: Sequence[Mapping[str, Any]],
        requested_target_id: Any = None,
        runtime_missing_prerequisite: str | None = None,
        available_realization_families: Sequence[str] = (),
        repairable_variables: Sequence[str] = (),
        all_runtime_candidates: Sequence[Mapping[str, Any]] = (),
        arm: str | None = None,
        blocked_candidate_refs: Sequence[str] = (),
    ) -> GoalConsistentCandidateSetV31:
        """Freeze the current Goal Contract and filter an already legal set."""

        if not self.hpk_goal_consistency_enabled:
            raise RuntimeError("HPK v3.1 goal consistency is not enabled")
        action = _text(action_mode, label="action_mode").casefold()
        target_ref = str(requested_target_id or "").strip() or None
        try:
            binding = merge_runtime_goal_bindings_v31(
                self._current_task_goal_binding,
                RuntimeGoalBindingV31(
                    manipulated_object_ref=self._instance_private_ref(instance),
                    required_target_ref=target_ref,
                ),
            )
        except (TypeError, ValueError, HPKV3ValidationError):
            report = unresolved_feasibility_report_v31(
                failed_stage="goal contract binding",
                missing_prerequisite=None,
                repairable_variables=(),
                non_repairable_reason=(
                    "the current Runtime object or target binding conflicts with "
                    "the frozen Task binding"
                ),
            )
            self._current_goal_contract = None
            self._last_runtime_feasibility_report = report
            return GoalConsistentCandidateSetV31(
                (), None, RuntimeGoalBindingV31(), report
            )

        if self._current_task_goal_source is None:
            report = unresolved_feasibility_report_v31(
                failed_stage="goal contract construction",
                missing_prerequisite=(
                    "the planner has not provided a structured Task strategy with "
                    "purpose and completion condition"
                ),
                repairable_variables=("structured Task strategy",),
            )
            self._current_goal_contract = None
            self._last_runtime_feasibility_report = report
            return GoalConsistentCandidateSetV31((), None, binding, report)

        scene = dict(scene_memory or {})
        raw_targets = scene.get("operation_targets")
        target_records = tuple(
            value
            for value in (raw_targets if isinstance(raw_targets, Sequence) else ())
            if isinstance(value, Mapping)
        )
        try:
            runtime_semantics = runtime_goal_semantics_v31(
                instance=instance,
                operation=action,
                required_target_ref=binding.required_target_ref,
                target_records=target_records,
                candidates=eligible_candidates,
                expected_effect=_EXPECTED_ACTION_EFFECTS[action],
            )
            contract = build_subtask_goal_contract_v31(
                self._current_task_goal_source,
                runtime_semantics=runtime_semantics,
                operation=action,
                expected_effect=_EXPECTED_ACTION_EFFECTS[action],
            )
        except (KeyError, TypeError, ValueError, HPKV3ValidationError) as exc:
            report = unresolved_feasibility_report_v31(
                failed_stage="goal contract construction",
                missing_prerequisite=str(exc),
                repairable_variables=("structured Task strategy and Runtime binding",),
            )
            self._current_goal_contract = None
            self._last_runtime_feasibility_report = report
            return GoalConsistentCandidateSetV31((), None, binding, report)

        if runtime_missing_prerequisite is None and arm is not None:
            runtime_missing_prerequisite = runtime_candidate_missing_prerequisite_v31(
                all_runtime_candidates,
                operation=action,
                arm=arm,
                required_target_ref=binding.required_target_ref,
                blocked_candidate_refs=blocked_candidate_refs,
            )
        result = filter_goal_consistent_candidates_v31(
            eligible_candidates,
            contract=contract,
            runtime_binding=binding,
            target_records=target_records,
            additional_realization_families=available_realization_families,
            repairable_variables=repairable_variables,
            runtime_missing_prerequisite=runtime_missing_prerequisite,
        )
        self._current_task_goal_binding = binding
        self._current_goal_contract = contract
        self._last_runtime_feasibility_report = result.feasibility_report
        return result

    def build_realization_hypothesis(
        self,
        *,
        observed_problem: str,
        change: Mapping[str, Any],
        runtime_realization: str,
        support_condition: str,
        oppose_condition: str,
        rationale: str,
        preserve: Mapping[str, Any] | None = None,
        status: str = "pending hypothesis",
    ) -> RealizationHypothesisV31:
        if self._current_goal_contract is None:
            raise RuntimeError("a resolved current Goal Contract is required")
        if self._last_runtime_feasibility_report is None:
            raise RuntimeError("a current Runtime feasibility report is required")
        return build_realization_hypothesis_v31(
            contract=self._current_goal_contract,
            feasibility_report=self._last_runtime_feasibility_report,
            observed_problem=observed_problem,
            change=change,
            runtime_realization=runtime_realization,
            support_condition=support_condition,
            oppose_condition=oppose_condition,
            rationale=rationale,
            preserve=preserve,
            status=status,
        )

    def build_preplanner_query(
        self,
        *,
        overall_goal: str,
        scene_memory: Mapping[str, Any] | None,
        active_skill: Any = None,
    ) -> SubtaskKnowledgeQuery | None:
        if self.mode != "full":
            return None
        return build_subtask_knowledge_query(
            overall_goal=overall_goal,
            scene_memory=scene_memory,
            active_skill=active_skill,
        )

    def build_action_query(
        self,
        *,
        scene_memory: Mapping[str, Any] | None,
        instance: Mapping[str, Any],
        arm: str,
        action_mode: str,
        eligible_candidates: Sequence[Mapping[str, Any]],
        geometry_policy: Any = None,
    ) -> ActionKnowledgeQuery:
        return build_action_knowledge_query(
            scene_memory=scene_memory,
            instance=instance,
            arm=arm,
            action_mode=action_mode,
            eligible_candidates=eligible_candidates,
            geometry_policy=geometry_policy,
            goal_contract=(
                self._current_goal_contract
                if self.hpk_goal_consistency_enabled
                else None
            ),
        )

    def apply_v3_planner_prediction(
        self,
        *,
        task: str,
        prediction: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        result = copy.deepcopy(dict(prediction))
        planner_goal = None
        structured_goal_output = (
            self.hpk_goal_consistency_enabled and "subtask_goal" in result
        )
        if self.hpk_goal_consistency_enabled:
            self.set_current_task_strategy(None)
        if structured_goal_output and result["subtask_goal"] is not None:
            planner_goal = SubtaskGoalContractV31(result["subtask_goal"])
        baseline = _text(result.get("subtask_text"), label="prediction.subtask_text")
        query = self._preplanner_query or SubtaskKnowledgeQuery(
            overall_goal=task,
            task_state=baseline,
        )
        route_audit: dict[str, Any] | None = None
        task_knowledge: Sequence[SubtaskKnowledgeV3] = self.task_knowledge
        source_indices = tuple(range(len(self.task_knowledge)))
        try:
            if self._family_router is not None:
                route = self._family_router.route_task(
                    self.task_knowledge,
                    query,
                    baseline_subtask=baseline,
                )
                task_knowledge = route.knowledge
                source_indices = route.source_indices
                route_audit = route.audit
            selected, audit = apply_subtask_knowledge_decision(
                mode=self.mode,
                baseline_subtask=baseline,
                query=query,
                knowledge=task_knowledge,
                retriever=self._subtask_retriever,
                preferred_arm=result.get("preferred_arm"),
                goal_contract=planner_goal,
                require_goal_contract=structured_goal_output,
            )
            if route_audit is not None:
                final_call = (
                    None
                    if self._subtask_retriever is None
                    else self._subtask_retriever.last_call_audit
                )
                route_audit = _merge_retrieval_audits(route_audit, final_call)
                selected_payload = audit.get("retrieved_knowledge")
                selected_source = _selected_atomic_source_index(
                    selected_payload,
                    task_knowledge,
                    source_indices,
                )
                route_audit.update(
                    {
                        "selected_atomic_knowledge": selected_source,
                        "knowledge_selected": selected_source is not None,
                        "final_behavior_changed": selected != baseline,
                        "task_subtask_before": baseline,
                        "task_subtask_after": selected,
                    }
                )
                audit = {**route_audit, **audit}
                audit["behavior_changed"] = selected != baseline
        except Exception as exc:  # noqa: BLE001 - preserve the baseline planner result
            if self.retrieval_rgb_enabled:
                raise
            selected = baseline
            if route_audit is not None and self._subtask_retriever is not None:
                route_audit = _merge_retrieval_audits(
                    route_audit,
                    self._subtask_retriever.last_call_audit,
                )
            audit = {
                "mode": self.mode,
                "subtask_before": baseline,
                "subtask_after": baseline,
                "knowledge_adopted": False,
                "retrieval_reason": "semantic retrieval failed; planner result retained",
                "retrieval_error": type(exc).__name__,
                "retrieved_knowledge": None,
            }
            if route_audit is not None:
                audit = {**route_audit, **audit}
        result["subtask_text"] = selected
        if structured_goal_output:
            goal = audit.get(
                "subtask_goal", None if planner_goal is None else planner_goal.to_dict()
            )
            self.set_current_task_strategy(
                None if goal is None else SubtaskGoalContractV31(goal)
            )
            result["subtask_goal"] = goal
            audit["subtask_goal"] = goal
        elif self.hpk_goal_consistency_enabled:
            retrieved = audit.get("retrieved_knowledge")
            if (
                audit.get("knowledge_adopted") is True
                and audit.get("application_mode") == "model_grounded_rewrite"
                and isinstance(retrieved, Mapping)
                and isinstance(retrieved.get("subtask_strategy"), Mapping)
            ):
                # Use the reusable semantic Task strategy.  The grounded text
                # may contain current private references and therefore is not
                # itself a VLM-facing Goal Contract.
                self.set_current_task_strategy(retrieved)
            else:
                self.set_current_task_strategy(None)
        return result, audit

    def retrieve_action(
        self, query: ActionKnowledgeQuery
    ) -> ActionKnowledgeMatch | None:
        if self.mode != "full":
            return None
        assert self._action_retriever is not None
        action_knowledge: Sequence[ActionKnowledgeV3] = self.action_knowledge
        source_indices = tuple(range(len(self.action_knowledge)))
        route_audit: dict[str, Any] | None = None
        if self._family_router is not None:
            route = self._family_router.route_action(self.action_knowledge, query)
            action_knowledge = route.knowledge
            source_indices = route.source_indices
            route_audit = route.audit
            self._last_action_usage_audit = copy.deepcopy(route_audit)
        try:
            match = self._action_retriever.retrieve(action_knowledge, query)
        except Exception:
            if route_audit is not None:
                self._last_action_usage_audit = _merge_retrieval_audits(
                    route_audit,
                    self._action_retriever.last_call_audit,
                )
            raise
        if route_audit is not None:
            route_audit = _merge_retrieval_audits(
                route_audit,
                self._action_retriever.last_call_audit,
            )
            selected_source = _selected_atomic_source_index(
                None if match is None else match.knowledge.to_dict(),
                action_knowledge,
                source_indices,
            )
            route_audit.update(
                {
                    "selected_atomic_knowledge": selected_source,
                    "knowledge_selected": selected_source is not None,
                }
            )
            self._last_action_usage_audit = route_audit
        else:
            self._last_action_usage_audit = None
        return match

    def consume_action_retrieval_audit(self) -> dict[str, Any]:
        audit = copy.deepcopy(self._last_action_usage_audit or {})
        self._last_action_usage_audit = None
        return audit

    @property
    def online_store_enabled(self) -> bool:
        return (
            self.knowledge_updates_enabled
            and self.mode == "full"
            and self._store_root is not None
            and self._knowledge_consolidator is not None
        )

    @property
    def online_incremental_maintenance_enabled(self) -> bool:
        return self.online_store_enabled and self._incremental_maintainer is not None

    @property
    def trajectory_reflection_enabled(self) -> bool:
        return self.knowledge_updates_enabled and self.mode == "full" and self._trajectory_reflector is not None

    def prepare_action_usage(
        self,
        *,
        query: ActionKnowledgeQuery,
        selected_candidate_features: Mapping[str, Any],
        audit: Mapping[str, Any],
        goal_contract: SubtaskGoalContractV31 | Mapping[str, Any] | None = None,
        runtime_goal_binding: RuntimeGoalBindingV31 | Mapping[str, Any] | None = None,
        selected_candidate: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Attach semantic learning inputs and one trace-only local reference."""

        if not isinstance(query, ActionKnowledgeQuery):
            raise TypeError("query must be ActionKnowledgeQuery")
        features = _finite_json_mapping(
            selected_candidate_features,
            label="selected_candidate_features",
        )
        condition = {
            "action": query.action,
            "object_description": query.object_description,
        }
        for field in ("held_state", "support_relation", "target_relation"):
            value = getattr(query, field)
            if value is not None:
                condition[field] = value
        local_ref = f"action {self._next_online_action_ref}"
        self._next_online_action_ref += 1
        typed_contract: SubtaskGoalContractV31 | None = None
        if self.hpk_goal_consistency_enabled:
            if goal_contract is None or runtime_goal_binding is None:
                raise RuntimeError(
                    "goal-consistent action usage requires its Goal Contract and Runtime binding"
                )
            if not isinstance(selected_candidate, Mapping):
                raise RuntimeError(
                    "goal-consistent action usage requires the selected current candidate"
                )
            typed_contract = (
                goal_contract
                if isinstance(goal_contract, SubtaskGoalContractV31)
                else SubtaskGoalContractV31(goal_contract)
            )
            typed_binding = RuntimeGoalBindingV31.from_value(runtime_goal_binding)
            selected_ref = str(selected_candidate.get("candidate_id", "") or "").strip()
            if not selected_ref:
                raise RuntimeError(
                    "selected current candidate lacks its private reference"
                )
            self._prepared_goal_contexts[local_ref] = _PreparedGoalContext(
                typed_contract,
                typed_binding,
                selected_ref,
            )
        result = copy.deepcopy(dict(audit))
        result.update(
            {
                "schema": "roboharn_evo/hpk/v3/action_usage",
                "mode": "full",
                "learning_condition": condition,
                "selected_candidate_features": features,
                "_runtime_metadata": {"local_ref": local_ref},
            }
        )
        if typed_contract is not None:
            result["goal_contract"] = typed_contract.to_dict()
            result["realization_status"] = "pending"
        return result

    @staticmethod
    def _call_args(call: Any) -> dict[str, Any]:
        value = getattr(call, "args", None)
        if value is None and isinstance(call, Mapping):
            value = call.get("args")
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _candidate_ref(args: Mapping[str, Any]) -> str:
        direct = str(args.get("_operation_candidate_id", "") or "").strip()
        if direct:
            return direct
        attempt = args.get("_operation_candidate_attempt")
        if isinstance(attempt, Mapping):
            return str(attempt.get("candidate_id", "") or "").strip()
        return ""

    def _pendings_from_calls(
        self,
        calls: Sequence[Any],
    ) -> tuple[_PendingOnlineAction, ...]:
        found: list[_PendingOnlineAction] = []
        seen: set[str] = set()
        for call in calls:
            args = self._call_args(call)
            audit = args.get("_hpk_usage_binding")
            if not isinstance(audit, Mapping) or audit.get("schema") != (
                "roboharn_evo/hpk/v3/action_usage"
            ):
                continue
            condition = audit.get("learning_condition")
            features = audit.get("selected_candidate_features")
            if not isinstance(condition, Mapping) or not isinstance(features, Mapping):
                continue
            metadata = audit.get("_runtime_metadata")
            local_ref = (
                str(metadata.get("local_ref", "") or "").strip()
                if isinstance(metadata, Mapping)
                else ""
            )
            action = str(condition.get("action", "") or "").strip().lower()
            arm = str(args.get("arm", "") or "").strip().lower()
            candidate_ref = self._candidate_ref(args)
            if action not in _ACTIONS or arm not in {"left", "right"}:
                continue
            if not local_ref or not candidate_ref:
                continue
            pending = self._pending_online_actions.get(local_ref)
            if pending is None:
                goal_contract: SubtaskGoalContractV31 | None = None
                runtime_goal_binding: RuntimeGoalBindingV31 | None = None
                realization_status = "unresolved"
                prepared = self._prepared_goal_contexts.get(local_ref)
                if prepared is not None:
                    raw_contract = audit.get("goal_contract")
                    try:
                        goal_contract = SubtaskGoalContractV31(raw_contract)
                    except (TypeError, ValueError, HPKV3ValidationError):
                        goal_contract = prepared.contract
                        realization_status = "goal inconsistent"
                    else:
                        realization_status = (
                            "goal consistent"
                            if goal_contract.to_dict() == prepared.contract.to_dict()
                            and candidate_ref == prepared.selected_candidate_ref
                            and action == prepared.contract["operation"]
                            and (
                                prepared.runtime_binding.required_target_ref is None
                                or str(args.get("target_id", "") or "").strip()
                                == prepared.runtime_binding.required_target_ref
                            )
                            else "goal inconsistent"
                        )
                    runtime_goal_binding = prepared.runtime_binding
                condition_fields = {
                    key: (
                        _natural_runtime_value(condition[key])
                        if isinstance(condition[key], str)
                        else copy.deepcopy(condition[key])
                    )
                    for key in (
                        "action",
                        "object_description",
                        "held_state",
                        "support_relation",
                        "target_relation",
                    )
                    if condition.get(key) not in (None, "")
                }
                pending = _PendingOnlineAction(
                    local_ref=local_ref,
                    action=action,
                    arm=arm,
                    candidate_ref=candidate_ref,
                    condition=condition_fields,
                    geometric_strategy=_runtime_geometry_to_v3(
                        action,
                        features,
                    ),
                    expected_effect=copy.deepcopy(_EXPECTED_ACTION_EFFECTS[action]),
                    goal_contract=goal_contract,
                    runtime_goal_binding=runtime_goal_binding,
                    realization_status=realization_status,
                )
                self._pending_online_actions[local_ref] = pending
            elif (
                pending.action != action
                or pending.arm != arm
                or pending.candidate_ref != candidate_ref
            ):
                continue
            if local_ref not in seen:
                found.append(pending)
                seen.add(local_ref)
        return tuple(found)

    def _pending_from_validation(
        self,
        action: str | None,
        validation: Mapping[str, Any] | None,
    ) -> _PendingOnlineAction | None:
        if action is None or validation is None:
            return None
        arm = str(validation.get("arm", "") or "").strip().lower()
        compatible = [
            pending
            for pending in self._pending_online_actions.values()
            if pending.action == action
            and (arm not in {"left", "right"} or pending.arm == arm)
            and (
                pending.evidence is None or pending.evidence["verdict"] == "unverified"
            )
        ]
        return compatible[0] if len(compatible) == 1 else None

    @staticmethod
    def _execution_completed(results: Sequence[Any]) -> bool:
        observed = []
        for result in results:
            details = getattr(result, "details", {})
            if isinstance(details, Mapping) and details.get("skipped") is True:
                continue
            success = getattr(result, "success", None)
            if isinstance(success, bool):
                observed.append(success)
        return bool(observed and all(observed))

    @staticmethod
    def _execution_preserves_goal_binding(
        pending: _PendingOnlineAction,
        *,
        results: Sequence[Any],
        validation: Mapping[str, Any] | None,
    ) -> bool:
        binding = pending.runtime_goal_binding
        if binding is None:
            return pending.goal_contract is None
        for result in results:
            details = getattr(result, "details", {})
            if not isinstance(details, Mapping):
                continue
            actual_candidate = str(
                details.get("operation_candidate_id", "") or ""
            ).strip()
            if actual_candidate and actual_candidate != pending.candidate_ref:
                return False
        if validation is None:
            return True
        actual_target = str(
            validation.get("target_id", validation.get("operation_target_id", "")) or ""
        ).strip()
        if (
            binding.required_target_ref is not None
            and actual_target
            and actual_target != binding.required_target_ref
        ):
            return False
        actual_object = str(
            validation.get("held_instance_id", validation.get("instance_id", "")) or ""
        ).strip()
        return not (
            binding.manipulated_object_ref is not None
            and actual_object
            and actual_object != binding.manipulated_object_ref
        )

    @staticmethod
    def _evidence_for_pending(
        pending: _PendingOnlineAction,
        *,
        verdict: str,
        delayed: bool,
        execution_completed: bool,
    ) -> ActionEvidenceV3 | ActionEvidenceV31:
        before_state = {
            key: copy.deepcopy(pending.condition[key])
            for key in ("held_state", "support_relation", "target_relation")
            if key in pending.condition
        }
        before_state["relevant_relations"] = []
        after_state: dict[str, Any] = {"relevant_relations": []}
        if verdict == "support" and pending.action == "grasp":
            after_state["held_state"] = "held"
        elif verdict == "support" and pending.action == "place":
            after_state["held_state"] = "not held"
            if "target_relation" in pending.condition:
                after_state["target_relation"] = pending.condition["target_relation"]
        observed_strategy = (
            "; ".join(
                str(value)
                for key, value in pending.geometric_strategy.items()
                if key != "avoid" and value not in (None, "")
            )
            or f"the observed {pending.action} geometry"
        )
        if pending.goal_contract is not None and pending.realization_status != (
            "goal consistent"
        ):
            observed_result = (
                "the executed realization was not proven to preserve the current goal"
            )
            missing_evidence = "a goal-consistent execution of the frozen contract"
        elif verdict == "support":
            observed_result = pending.expected_effect["verification_observation"]
            missing_evidence = None
        elif verdict == "oppose":
            observed_result = f"the expected {pending.action} effect did not occur"
            missing_evidence = None
        else:
            observed_result = "the physical effect was not confirmed"
            missing_evidence = "a confirming independent post action observation"
        payload = {
            "before_state": before_state,
            "executed_action": {
                "action": pending.action,
                "executed_arm": f"{pending.arm} arm",
                "observed_strategy": observed_strategy,
            },
            "execution_status": (
                "completed" if execution_completed else "not completed"
            ),
            "after_state": after_state,
            "observed_result": observed_result,
            "verdict": verdict,
            "evidence_timing": "delayed" if delayed else "immediate",
            "verifier_source": f"deterministic runtime {pending.action} verifier",
        }
        if missing_evidence is not None:
            payload["missing_evidence"] = missing_evidence
        if pending.goal_contract is not None:
            payload["goal_contract"] = pending.goal_contract.to_dict()
            payload["realization_status"] = pending.realization_status
            return ActionEvidenceV31(payload)
        return ActionEvidenceV3(payload)

    def record_action_effect(
        self,
        *,
        calls: Sequence[Any],
        results: Sequence[Any],
        action_effect: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        """Reduce all views/verifiers for one execution to one pending evidence."""

        if self.mode != "full":
            return ()
        pendings = self._pendings_from_calls(calls)
        action, validation = _runtime_validation(action_effect)
        if not pendings:
            delayed_pending = self._pending_from_validation(action, validation)
            pendings = () if delayed_pending is None else (delayed_pending,)
        if not pendings:
            return ()
        execution_completed = self._execution_completed(results)
        updates: list[dict[str, Any]] = []
        for pending in pendings:
            if pending.goal_contract is not None and not (
                self._execution_preserves_goal_binding(
                    pending,
                    results=results,
                    validation=validation,
                )
            ):
                pending.realization_status = "goal inconsistent"
            verdict = _runtime_evidence_verdict(action_effect, validation)
            if (
                len(pendings) != 1
                or action != pending.action
                or not execution_completed
                or (
                    pending.goal_contract is not None
                    and pending.realization_status != "goal consistent"
                )
            ):
                verdict = "unverified"
            delayed = pending.evidence is not None
            evidence = self._evidence_for_pending(
                pending,
                verdict=verdict,
                delayed=delayed,
                execution_completed=execution_completed,
            )
            previous = pending.evidence
            if previous is not None:
                previous_verdict = previous["verdict"]
                if previous_verdict in {"support", "oppose"}:
                    if verdict == "unverified" or verdict == previous_verdict:
                        evidence = previous
                    else:
                        evidence = self._evidence_for_pending(
                            pending,
                            verdict="unverified",
                            delayed=True,
                            execution_completed=execution_completed,
                        )
            pending.evidence = evidence
            updates.append(
                {
                    "_runtime_metadata": {"local_ref": pending.local_ref},
                    "evidence": evidence.to_dict(),
                }
            )
        return tuple(updates)

    def finalize_online_episode(
        self,
        *,
        rollout_dir: str | Path | None = None,
        result: str = "",
    ) -> dict[str, Any]:
        """Commit finalized action evidence once, for use by the next episode."""

        if not self.knowledge_updates_enabled:
            evidence = [pending.evidence.to_dict() for pending in self._pending_online_actions.values() if pending.evidence is not None]
            return {
                "status": "knowledge updates disabled",
                "store_read_only": True,
                "evidence_count": len(evidence),
                "evidence": evidence,
                "task_knowledge_count": len(self.task_knowledge),
                "action_knowledge_count": len(self.action_knowledge),
            }
        if not self.online_store_enabled:
            return {
                "status": "online Store disabled",
                "evidence_count": 0,
                "task_knowledge_count": len(self.task_knowledge),
                "action_knowledge_count": len(self.action_knowledge),
            }
        finalized = [
            (pending, pending.evidence)
            for pending in self._pending_online_actions.values()
            if pending.evidence is not None
        ]
        additions = []
        evidence_records = []
        for pending, evidence in finalized:
            assert evidence is not None
            verdict = evidence["verdict"]
            summary = {
                "support": int(verdict == "support"),
                "oppose": int(verdict == "oppose"),
                "unverified": int(verdict == "unverified"),
                "independent_verified_trials": int(verdict in {"support", "oppose"}),
            }
            additions.append(
                ActionKnowledgeV3(
                    {
                        "condition": pending.condition,
                        "geometric_strategy": pending.geometric_strategy,
                        "expected_effect": pending.expected_effect,
                        "evidence_summary": summary,
                        "status": (
                            "supported"
                            if verdict == "support"
                            else "contested"
                            if verdict == "oppose"
                            else "candidate"
                        ),
                    }
                )
            )
            evidence_records.append(evidence.to_dict())

        reflection_summary: dict[str, Any] = {
            "status": (
                "trajectory reflection disabled"
                if additions
                else "no completed action evidence; trajectory package not generated"
            )
        }
        reflected_tasks: tuple[SubtaskKnowledgeV3, ...] = ()
        reflected_actions: tuple[ActionKnowledgeV3, ...] = ()
        reflected_evidence = None
        if additions and self.trajectory_reflection_enabled:
            if rollout_dir is None:
                raise ValueError(
                    "real rollout reflection requires an explicit rollout directory"
                )
            if (
                self._reflection_transport is not None
                and getattr(self._reflection_transport, "capability_report", None)
                is None
            ):
                self._reflection_transport.capability_preflight(
                    authorization=self._reflection_authorization,
                )
            from roboharn_evo.agent.hpk.real_rollout_reflection import (
                build_real_rollout_reflection_input,
                reflect_real_rollout,
            )

            rollout_input = build_real_rollout_reflection_input(
                rollout_dir,
                result=result,
                max_action_executions=self._max_reflection_images // 2 if self._evidence_index is not None else self._max_reflection_images,
                bind_rgb_evidence=self._evidence_index is not None,
            )
            reflection_summary = reflect_real_rollout(
                self._trajectory_reflector,
                rollout_input,
                output_dir=Path(rollout_dir) / "hpk_v3_reflection",
            )
            atomic_path = Path(rollout_dir) / "hpk_v3_reflection/atomic_knowledge.json"
            if atomic_path.is_file():
                reflected_tasks, reflected_actions = load_atomic_knowledge(atomic_path)
                if self._evidence_index is not None:
                    reflected_evidence = RGBEvidenceIndex.load(rollout_dir, filename="hpk_v3_reflection/evidence_index.jsonl")
                if reflected_actions and _action_verdict_counts(
                    reflected_actions
                ) != _action_verdict_counts(tuple(additions)):
                    raise HPKV3ValidationError(
                        "reflected Action evidence must exactly preserve runtime verdicts"
                    )
        from roboharn_evo.agent.hpk.hierarchical_store import save_hierarchical_store

        task_additions = reflected_tasks
        action_additions = reflected_actions or tuple(additions)
        incremental_audit = None
        if not task_additions and not action_additions:
            return {"status": "no completed action evidence; Store unchanged", "evidence_count": 0, "task_knowledge_count": len(self.task_knowledge), "action_knowledge_count": len(self.action_knowledge)}
        if self._incremental_maintainer is not None:
            assert self._family_catalog is not None
            maintained = self._incremental_maintainer.maintain(
                task_knowledge=self.task_knowledge,
                action_knowledge=self.action_knowledge,
                catalog=self._family_catalog,
                new_task_knowledge=task_additions,
                new_action_knowledge=action_additions,
                output_root=self._store_root,
                evidence_index=self._evidence_index,
                new_evidence_index=reflected_evidence,
                maintenance_rgb_enabled=self.maintenance_rgb_enabled,
                review_output_root=None if rollout_dir is None else Path(rollout_dir) / "hpk_rgb_maintenance",
            )
            self.task_knowledge = maintained.task_knowledge
            self.action_knowledge = maintained.action_knowledge
            self._family_catalog = maintained.catalog
            self._evidence_index = maintained.evidence_index
            if self._rgb_context is not None:
                self._rgb_context.evidence = maintained.evidence_index
                self._rgb_context.knowledge = {"task": self.task_knowledge, "action": self.action_knowledge}
            incremental_audit = maintained.audit
            if self._family_router is not None:
                self._family_router.replace_catalog(maintained.catalog)
        else:
            if task_additions or action_additions:
                consolidated = self._knowledge_consolidator.consolidate(
                    task_knowledge=(*self.task_knowledge, *task_additions),
                    action_knowledge=(*self.action_knowledge, *action_additions),
                )
                self.task_knowledge = consolidated.task_knowledge
                self.action_knowledge = consolidated.action_knowledge
            save_hierarchical_store(
                self._store_root,
                task_knowledge=self.task_knowledge,
                action_knowledge=self.action_knowledge,
            )
        self._pending_online_actions.clear()
        self._prepared_goal_contexts.clear()
        changed = bool(task_additions or action_additions)
        return {
            "status": (
                "task and action knowledge updated"
                if task_additions
                else "action knowledge updated"
                if action_additions
                else "no completed action evidence; Store unchanged"
            ),
            "evidence_count": len(evidence_records),
            "task_knowledge_count": len(self.task_knowledge),
            "action_knowledge_count": len(self.action_knowledge),
            "semantic_consolidation_performed": changed,
            "incremental_maintenance_performed": (
                self._incremental_maintainer is not None
            ),
            "incremental_maintenance": incremental_audit,
            "reflected_task_knowledge_ingested": len(reflected_tasks),
            "reflected_action_knowledge_ingested": len(reflected_actions),
            "action_writeback_source": (
                "reflected semantics with runtime-authoritative verdicts"
                if reflected_actions
                else "runtime action evidence"
            ),
            "duplicate_action_executions_avoided": (
                len(additions) if reflected_actions else 0
            ),
            "trajectory_reflection": reflection_summary,
            "evidence": evidence_records,
        }


__all__ = [
    "ActionKnowledgeMatch",
    "ActionKnowledgeQuery",
    "AgentApiHierarchicalRetrievalBackend",
    "HierarchicalHPKRetrievalRuntime",
    "SubtaskKnowledgeMatch",
    "SubtaskKnowledgeQuery",
    "VLMActionKnowledgeRetriever",
    "VLMSubtaskKnowledgeRetriever",
    "apply_subtask_knowledge_decision",
    "build_action_knowledge_query",
    "build_hierarchical_hpk_runtime",
    "build_subtask_knowledge_query",
    "hierarchical_retrieval_output_json_schema",
]
