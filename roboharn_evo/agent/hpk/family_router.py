from __future__ import annotations

import copy
import json
import math
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from roboharn_evo.agent.hpk.hierarchical_knowledge import (
    ActionKnowledgeV3,
    SubtaskKnowledgeV3,
)
from roboharn_evo.agent.hpk.knowledge_family import (
    KnowledgeFamilyCatalogV1,
    render_action_family_routing_card,
    render_action_knowledge_card,
    render_task_family_routing_card,
    render_task_knowledge_card,
    validate_catalog_against_knowledge,
)
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (
    HierarchicalReflectionBackend,
    _strict_response,
)

_FAMILY_ROUTING_PROMPT = """Route the current robot decision to at most the allowed number of Knowledge Families.

Families are recall-only descriptions. Select a Family when it may contain atomic knowledge applicable to the current decision. Prefer recall when two Families remain plausible. Return an empty selection when none applies. Do not produce a subtask, geometry, pose, candidate, or strategy. Do not treat a Family summary as evidence or guidance. Ignore wording and temporary indices. Return strict JSON only."""

_MEMBER_SHORTLIST_PROMPT = """Shortlist atomic HPK knowledge that may apply to the current robot decision.

Select at most the requested number of supplied member indices. Preserve semantic distinctions that can change applicability, ordering, grounding, geometry, or expected physical effect. Prefer recall when uncertain. Return an empty selection when no member applies. Do not write a new strategy, pose, candidate, or knowledge record. Indices are temporary. Return strict JSON only."""


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class FamilyRoutingConfig:
    exhaustive_threshold: int = 24
    max_selected_families: int = 2
    max_atomic_shortlist: int = 8
    max_atomic_shortlist_input: int = 24
    local_idf_recall: bool = True

    def __post_init__(self) -> None:
        for field in (
            "exhaustive_threshold",
            "max_selected_families",
            "max_atomic_shortlist",
            "max_atomic_shortlist_input",
        ):
            object.__setattr__(
                self,
                field,
                _positive_int(getattr(self, field), label=field),
            )
        if self.max_selected_families > 2:
            raise ValueError("max_selected_families must not exceed 2")
        if self.max_atomic_shortlist > 8:
            raise ValueError("max_atomic_shortlist must not exceed 8")
        if not isinstance(self.local_idf_recall, bool):
            raise TypeError("local_idf_recall must be a boolean")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> FamilyRoutingConfig:
        if value is not None and not isinstance(value, Mapping):
            raise TypeError("Family routing config must be an object")
        payload = dict(value or {})
        allowed = {
            "exhaustive_threshold",
            "max_selected_families",
            "max_atomic_shortlist",
            "max_atomic_shortlist_input",
            "local_idf_recall",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown Family routing config fields: {sorted(unknown)}")
        return cls(**payload)

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "exhaustive_threshold": self.exhaustive_threshold,
            "max_selected_families": self.max_selected_families,
            "max_atomic_shortlist": self.max_atomic_shortlist,
            "max_atomic_shortlist_input": self.max_atomic_shortlist_input,
            "local_idf_recall": self.local_idf_recall,
        }


@dataclass(frozen=True, slots=True)
class FamilyRouteResult:
    knowledge: tuple[Any, ...]
    source_indices: tuple[int, ...]
    audit: dict[str, Any]


def _route_schema(*, max_items: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selected_family_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "maxItems": max_items,
            },
            "reason": {"type": "string", "pattern": r"^[^_]+$"},
        },
        "required": ["selected_family_indices", "reason"],
        "additionalProperties": False,
    }


def _shortlist_schema(*, max_items: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selected_member_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "maxItems": max_items,
            },
            "reason": {"type": "string", "pattern": r"^[^_]+$"},
        },
        "required": ["selected_member_indices", "reason"],
        "additionalProperties": False,
    }


def family_routing_output_json_schema(schema_name: str) -> dict[str, Any]:
    if schema_name in {
        "hpk_v3_task_family_routing",
        "hpk_v3_action_family_routing",
    }:
        return _route_schema(max_items=2)
    if schema_name in {
        "hpk_v3_task_family_member_shortlist",
        "hpk_v3_action_family_member_shortlist",
    }:
        return _shortlist_schema(max_items=8)
    raise ValueError("unsupported Knowledge Family routing schema name")


def _input_tokens(completion: Any) -> int | None:
    audit = getattr(completion, "audit", None)
    if not isinstance(audit, Mapping):
        return None
    candidates = [audit]
    for key in ("usage", "provider_usage"):
        nested = audit.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested)
    for source in candidates:
        for key in ("input_tokens", "prompt_tokens"):
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return None


def _strict_indices(
    value: Any,
    *,
    allowed: set[int],
    max_items: int,
    label: str,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    if len(value) > max_items:
        raise ValueError(f"{label} exceeds its configured maximum")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item not in allowed:
            raise ValueError(f"{label} contains an index that was not presented")
        result.append(item)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate indices")
    return tuple(result)


_TERM_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _terms(value: str) -> frozenset[str]:
    """Return language-agnostic lexical terms without a task word list."""

    return frozenset(_TERM_PATTERN.findall(value.casefold()))


def _recall_text(value: Any) -> str:
    """Flatten semantic values while ignoring schema field names and indices."""

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(_recall_text(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(_recall_text(item) for item in value)
    return ""


def _idf_rank(
    query_text: str, documents: Sequence[str]
) -> tuple[tuple[int, float], ...]:
    """Rank recall documents by corpus-derived lexical overlap only.

    This is deliberately a recall heuristic.  It neither adopts a Family nor
    selects executable knowledge; the existing semantic retriever remains the
    final authority over the returned atomic shortlist.
    """

    if not documents:
        return ()
    query_terms = _terms(query_text)
    document_terms = tuple(_terms(value) for value in documents)
    if len(document_terms) == 1:
        overlap = query_terms.intersection(document_terms[0])
        return ((0, float(len(overlap))),)
    frequency = Counter(term for terms in document_terms for term in terms)
    weights = {
        term: math.log((len(document_terms) + 1) / (count + 1))
        for term, count in frequency.items()
    }
    ranked: list[tuple[int, float]] = []
    for index, terms in enumerate(document_terms):
        numerator = sum(weights[term] for term in query_terms.intersection(terms))
        denominator = math.sqrt(
            sum(weight * weight for term, weight in weights.items() if term in terms)
        )
        ranked.append((index, 0.0 if denominator == 0.0 else numerator / denominator))
    return tuple(sorted(ranked, key=lambda item: (-item[1], item[0])))


class KnowledgeFamilyRouter:
    """Recall atomic members; never return executable Family guidance."""

    def __init__(
        self,
        backend: HierarchicalReflectionBackend,
        *,
        catalog: KnowledgeFamilyCatalogV1 | Mapping[str, Any] | None,
        config: FamilyRoutingConfig | Mapping[str, Any] | None = None,
        catalog_error: str | None = None,
    ) -> None:
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must expose complete")
        self._backend = backend
        self._catalog = (
            None
            if catalog is None
            else catalog
            if isinstance(catalog, KnowledgeFamilyCatalogV1)
            else KnowledgeFamilyCatalogV1(catalog)
        )
        self.config = (
            config
            if isinstance(config, FamilyRoutingConfig)
            else FamilyRoutingConfig.from_mapping(config)
        )
        self._catalog_error = None if catalog_error is None else str(catalog_error)

    def _base_audit(self, eligible_count: int) -> dict[str, Any]:
        return {
            "eligible_atomic_count": eligible_count,
            "exhaustive_or_family_path": "exhaustive",
            "family_count_presented": 0,
            "selected_family_indices": [],
            "selected_family_summaries": [],
            "member_count_before_shortlist": eligible_count,
            "atomic_count_after_shortlist": eligible_count,
            "member_shortlist_performed": False,
            "selected_atomic_knowledge": None,
            "knowledge_selected": False,
            "retrieval_calls": 0,
            "retrieval_input_characters": 0,
            "retrieval_input_tokens": None,
            "retrieval_latency_ms": 0.0,
        }

    def replace_catalog(
        self,
        catalog: KnowledgeFamilyCatalogV1 | Mapping[str, Any],
    ) -> None:
        self._catalog = (
            catalog
            if isinstance(catalog, KnowledgeFamilyCatalogV1)
            else KnowledgeFamilyCatalogV1(catalog)
        )
        self._catalog_error = None

    def _complete(
        self,
        *,
        instructions: str,
        payload: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        schema_name: str,
        audit: dict[str, Any],
    ) -> dict[str, Any]:
        input_text = json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        started = time.perf_counter()
        completion = None
        try:
            completion = self._backend.complete(
                instructions=instructions,
                input_text=input_text,
                images=(),
                output_schema=output_schema,
                schema_name=schema_name,
            )
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            audit["retrieval_calls"] += 1
            audit["retrieval_input_characters"] += len(input_text)
            audit["retrieval_latency_ms"] = round(
                float(audit["retrieval_latency_ms"]) + elapsed_ms,
                3,
            )
        assert completion is not None
        tokens = _input_tokens(completion)
        if tokens is not None:
            current = audit["retrieval_input_tokens"]
            audit["retrieval_input_tokens"] = (
                tokens if current is None else int(current) + tokens
            )
        return _strict_response(completion.output)

    def _local_recall(
        self,
        *,
        values: tuple[Any, ...],
        eligible_indices: tuple[int, ...],
        families: tuple[tuple[int, Any], ...],
        query: Mapping[str, Any],
        render_family: Any,
        render_member: Any,
        audit: dict[str, Any],
    ) -> FamilyRouteResult | None:
        """Use a generic lexical index for recall, never for final adoption."""

        if not families:
            return None
        eligible_set = set(eligible_indices)
        family_members = {
            family_index: tuple(
                index for index in family["member_indices"] if index in eligible_set
            )
            for family_index, family in families
        }
        family_documents = [
            "\n".join(
                (
                    render_family(family),
                    *(
                        render_member(values[index])
                        for index in family_members[family_index]
                    ),
                )
            )
            for family_index, family in families
        ]
        query_text = _recall_text(query)
        ranking = _idf_rank(query_text, family_documents)
        selected_local = tuple(
            local_index for local_index, score in ranking if score > 0.0
        )[: self.config.max_selected_families]
        if not selected_local:
            return None
        selected = tuple(families[index] for index in selected_local)
        selected_family_indices = tuple(index for index, _family in selected)
        selected_by_index = dict(families)
        member_indices = tuple(
            index
            for index in eligible_indices
            if any(
                index in family_members[family_index]
                for family_index in selected_family_indices
            )
        )
        audit.update(
            {
                "exhaustive_or_family_path": "family",
                "family_recall_method": "local idf token overlap",
                "family_count_presented": len(families),
                "selected_family_indices": list(selected_family_indices),
                "selected_family_summaries": [
                    {
                        "family_name": selected_by_index[index]["family_name"],
                        "routing_summary": selected_by_index[index]["routing_summary"],
                    }
                    for index in selected_family_indices
                ],
                "family_routing_reason": (
                    "local recall shortlisted semantically overlapping Family text; "
                    "the final model still decides atomic applicability"
                ),
                "local_family_scores": [
                    {
                        "family_index": families[local_index][0],
                        "score": round(score, 6),
                    }
                    for local_index, score in ranking
                    if local_index in selected_local
                ],
                "member_count_before_shortlist": len(member_indices),
            }
        )
        if len(member_indices) > self.config.max_atomic_shortlist:
            audit["member_shortlist_performed"] = True
            ranked_by_family: list[list[int]] = []
            for family_index in selected_family_indices:
                members = list(family_members[family_index])
                member_ranking = _idf_rank(
                    query_text,
                    [render_member(values[index]) for index in members],
                )
                ranked_by_family.append(
                    [members[local_index] for local_index, _score in member_ranking]
                )
            shortlist: list[int] = []
            depth = 0
            while len(shortlist) < self.config.max_atomic_shortlist:
                added = False
                for ranked_members in ranked_by_family:
                    if depth < len(ranked_members):
                        shortlist.append(ranked_members[depth])
                        added = True
                        if len(shortlist) == self.config.max_atomic_shortlist:
                            break
                if not added:
                    break
                depth += 1
            member_indices = tuple(shortlist)
            audit["member_shortlist_reason"] = (
                "local recall retained the strongest members from every selected Family"
            )
        audit["atomic_count_after_shortlist"] = len(member_indices)
        return FamilyRouteResult(
            tuple(values[index] for index in member_indices),
            member_indices,
            audit,
        )

    def route_task(
        self,
        knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
        query: Any,
        *,
        baseline_subtask: str,
    ) -> FamilyRouteResult:
        tasks = tuple(
            value
            if isinstance(value, SubtaskKnowledgeV3)
            else SubtaskKnowledgeV3(value)
            for value in knowledge
        )
        eligible_indices = tuple(
            index for index, value in enumerate(tasks) if value["status"] == "supported"
        )
        return self._route(
            kind="task",
            tasks=tasks,
            actions=(),
            eligible_indices=eligible_indices,
            query={
                "query": query.to_dict(),
                "current_planner_subtask": str(baseline_subtask),
            },
        )

    def route_action(
        self,
        knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
        query: Any,
    ) -> FamilyRouteResult:
        actions = tuple(
            value if isinstance(value, ActionKnowledgeV3) else ActionKnowledgeV3(value)
            for value in knowledge
        )
        eligible_indices = tuple(
            index
            for index, value in enumerate(actions)
            if value["status"] == "supported"
            and value["condition"]["action"] == query.action
        )
        query_payload = query.to_dict()
        query_payload.pop("candidate_geometry", None)
        return self._route(
            kind="action",
            tasks=(),
            actions=actions,
            eligible_indices=eligible_indices,
            query={"query": query_payload},
            action=query.action,
        )

    def _route(
        self,
        *,
        kind: str,
        tasks: tuple[SubtaskKnowledgeV3, ...],
        actions: tuple[ActionKnowledgeV3, ...],
        eligible_indices: tuple[int, ...],
        query: Mapping[str, Any],
        action: str | None = None,
    ) -> FamilyRouteResult:
        values: tuple[Any, ...] = tasks if kind == "task" else actions
        eligible = tuple(values[index] for index in eligible_indices)
        audit = self._base_audit(len(eligible))
        if len(eligible) <= self.config.exhaustive_threshold:
            if self._catalog is None:
                audit["exhaustive_or_family_path"] = "exhaustive_catalog_fallback"
                audit["catalog_fallback_reason"] = (
                    self._catalog_error or "catalog missing"
                )
            return FamilyRouteResult(eligible, eligible_indices, audit)
        if self._catalog is None:
            audit["exhaustive_or_family_path"] = "exhaustive_catalog_fallback"
            audit["catalog_fallback_reason"] = self._catalog_error or "catalog missing"
            return FamilyRouteResult(eligible, eligible_indices, audit)
        try:
            if kind == "task":
                catalog = validate_catalog_against_knowledge(
                    {
                        "task_families": self._catalog["task_families"],
                        "action_families": [],
                    },
                    task_knowledge=tasks,
                    action_knowledge=(),
                )
                families = tuple(enumerate(catalog.task_families))
                render_family = render_task_family_routing_card
                render_member = render_task_knowledge_card
            else:
                catalog = validate_catalog_against_knowledge(
                    {
                        "task_families": [],
                        "action_families": self._catalog["action_families"],
                    },
                    task_knowledge=(),
                    action_knowledge=actions,
                )
                families = tuple(
                    (index, family)
                    for index, family in enumerate(catalog.action_families)
                    if family["action"] == action
                )
                render_family = render_action_family_routing_card
                render_member = render_action_knowledge_card
            eligible_set = set(eligible_indices)
            families = tuple(
                (index, family)
                for index, family in families
                if eligible_set.intersection(family["member_indices"])
            )
            if self.config.local_idf_recall:
                local_result = self._local_recall(
                    values=values,
                    eligible_indices=eligible_indices,
                    families=families,
                    query=query,
                    render_family=render_family,
                    render_member=render_member,
                    audit=audit,
                )
                if local_result is not None:
                    return local_result
                audit["local_recall_fallback_reason"] = "no informative lexical overlap"
            family_payload = {
                **copy.deepcopy(dict(query)),
                "max_selected_families": self.config.max_selected_families,
                "family_routing_cards": [
                    {"family_index": index, "card": render_family(family)}
                    for index, family in families
                ],
            }
            audit["exhaustive_or_family_path"] = "family"
            audit["family_count_presented"] = len(families)
            routed = self._complete(
                instructions=_FAMILY_ROUTING_PROMPT,
                payload=family_payload,
                output_schema=_route_schema(
                    max_items=self.config.max_selected_families
                ),
                schema_name=f"hpk_v3_{kind}_family_routing",
                audit=audit,
            )
            if set(routed) != {"selected_family_indices", "reason"}:
                raise ValueError("Family routing response fields mismatch")
            presented = {index for index, _family in families}
            selected_family_indices = _strict_indices(
                routed["selected_family_indices"],
                allowed=presented,
                max_items=self.config.max_selected_families,
                label="selected_family_indices",
            )
            selected_by_index = dict(families)
            audit["selected_family_indices"] = list(selected_family_indices)
            audit["selected_family_summaries"] = [
                {
                    "family_name": selected_by_index[index]["family_name"],
                    "routing_summary": selected_by_index[index]["routing_summary"],
                }
                for index in selected_family_indices
            ]
            audit["family_routing_reason"] = str(routed["reason"])
            member_indices = tuple(
                index
                for index in eligible_indices
                if any(
                    index in selected_by_index[family_index]["member_indices"]
                    for family_index in selected_family_indices
                )
            )
            audit["member_count_before_shortlist"] = len(member_indices)
            if len(member_indices) > self.config.max_atomic_shortlist_input:
                audit["member_shortlist_performed"] = True
                member_payload = {
                    **copy.deepcopy(dict(query)),
                    "max_selected_members": self.config.max_atomic_shortlist,
                    "atomic_member_cards": [
                        {
                            "member_index": index,
                            "card": render_member(values[index]),
                        }
                        for index in member_indices
                    ],
                }
                shortlisted = self._complete(
                    instructions=_MEMBER_SHORTLIST_PROMPT,
                    payload=member_payload,
                    output_schema=_shortlist_schema(
                        max_items=self.config.max_atomic_shortlist
                    ),
                    schema_name=f"hpk_v3_{kind}_family_member_shortlist",
                    audit=audit,
                )
                if set(shortlisted) != {"selected_member_indices", "reason"}:
                    raise ValueError("member shortlist response fields mismatch")
                member_indices = _strict_indices(
                    shortlisted["selected_member_indices"],
                    allowed=set(member_indices),
                    max_items=self.config.max_atomic_shortlist,
                    label="selected_member_indices",
                )
                audit["member_shortlist_reason"] = str(shortlisted["reason"])
            audit["atomic_count_after_shortlist"] = len(member_indices)
            return FamilyRouteResult(
                tuple(values[index] for index in member_indices),
                member_indices,
                audit,
            )
        except Exception as exc:  # noqa: BLE001 - routing must fall back to exhaustive
            fallback = copy.deepcopy(audit)
            fallback.update(
                {
                    "exhaustive_or_family_path": "exhaustive_router_fallback",
                    "catalog_fallback_reason": type(exc).__name__,
                    "member_count_before_shortlist": len(eligible),
                    "atomic_count_after_shortlist": len(eligible),
                    "selected_atomic_knowledge": None,
                    "knowledge_selected": False,
                }
            )
            return FamilyRouteResult(eligible, eligible_indices, fallback)


__all__ = [
    "FamilyRouteResult",
    "FamilyRoutingConfig",
    "KnowledgeFamilyRouter",
    "family_routing_output_json_schema",
]
