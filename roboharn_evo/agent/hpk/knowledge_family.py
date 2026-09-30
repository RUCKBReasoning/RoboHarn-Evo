from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from roboharn_evo.agent.hpk.hierarchical_knowledge import (
    ActionKnowledgeV3,
    HPKV3ValidationError,
    SubtaskKnowledgeV3,
)
from roboharn_evo.agent.hpk.hierarchical_knowledge import _text as _atomic_text
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (
    HierarchicalReflectionBackend,
    _strict_response,
)

_ACTIONS = frozenset({"grasp", "place", "contact"})


def _fail(path: str, message: str) -> None:
    raise HPKV3ValidationError(f"{path}: {message}")


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    try:
        payload = json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise HPKV3ValidationError(f"{path}: must be finite JSON") from exc
    if not isinstance(payload, dict):
        _fail(path, "must be an object")
    return payload


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            path,
            f"fields mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}",
        )


def _text(value: Any, *, path: str, max_chars: int = 1200) -> str:
    return _atomic_text(value, path=path, max_chars=max_chars)


def _texts(value: Any, *, path: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(path, "must be an array")
    if len(value) > 32:
        _fail(path, "must contain at most 32 values")
    result = [
        _text(item, path=f"{path}[{index}]", max_chars=600)
        for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        _fail(path, "must not contain duplicate values")
    return result


def _member_indices(value: Any, *, path: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(path, "must be an array")
    if not value:
        _fail(path, "must contain at least one index")
    result: list[int] = []
    for ordinal, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            _fail(f"{path}[{ordinal}]", "must be a non-negative integer")
        result.append(item)
    if len(result) != len(set(result)):
        _fail(path, "must not contain duplicate indices")
    return result


class _Record(Mapping[str, Any]):
    _validator: ClassVar[Any]

    def __init__(self, value: Mapping[str, Any]) -> None:
        self._data = self._validator(value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> _Record:
        return cls(value)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"


_COMMON_FAMILY_FIELDS = {
    "family_name",
    "routing_summary",
    "applicability_summary",
    "strategy_summary",
    "exclusions",
    "member_indices",
}


def _common_family(value: Any, *, path: str, expected: set[str]) -> dict[str, Any]:
    payload = _mapping(value, path=path)
    _exact_fields(payload, expected, path=path)
    return {
        "family_name": _text(
            payload["family_name"], path=f"{path}.family_name", max_chars=200
        ),
        "routing_summary": _text(
            payload["routing_summary"], path=f"{path}.routing_summary"
        ),
        "applicability_summary": _text(
            payload["applicability_summary"],
            path=f"{path}.applicability_summary",
        ),
        "strategy_summary": _text(
            payload["strategy_summary"], path=f"{path}.strategy_summary"
        ),
        "exclusions": _texts(payload["exclusions"], path=f"{path}.exclusions"),
        "member_indices": _member_indices(
            payload["member_indices"], path=f"{path}.member_indices"
        ),
    }


def _validate_task_family(value: Any) -> dict[str, Any]:
    path = "task_knowledge_family"
    result = _common_family(
        value,
        path=path,
        expected=_COMMON_FAMILY_FIELDS | {"completion_summary"},
    )
    payload = _mapping(value, path=path)
    result["completion_summary"] = _text(
        payload["completion_summary"], path=f"{path}.completion_summary"
    )
    return result


class TaskKnowledgeFamilyV1(_Record):
    _validator = staticmethod(_validate_task_family)


def _validate_action_family(value: Any) -> dict[str, Any]:
    path = "action_knowledge_family"
    result = _common_family(
        value,
        path=path,
        expected=_COMMON_FAMILY_FIELDS | {"action", "expected_effect_summary"},
    )
    payload = _mapping(value, path=path)
    action = _text(payload["action"], path=f"{path}.action", max_chars=20)
    if action not in _ACTIONS:
        _fail(f"{path}.action", f"unsupported action {action!r}")
    return {
        "action": action,
        **result,
        "expected_effect_summary": _text(
            payload["expected_effect_summary"],
            path=f"{path}.expected_effect_summary",
        ),
    }


class ActionKnowledgeFamilyV1(_Record):
    _validator = staticmethod(_validate_action_family)


def _validate_catalog(value: Any) -> dict[str, Any]:
    path = "knowledge_family_catalog"
    payload = _mapping(value, path=path)
    _exact_fields(payload, {"task_families", "action_families"}, path=path)
    task_raw = payload["task_families"]
    action_raw = payload["action_families"]
    for label, values in (("task_families", task_raw), ("action_families", action_raw)):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            _fail(f"{path}.{label}", "must be an array")
    return {
        "task_families": [TaskKnowledgeFamilyV1(item).to_dict() for item in task_raw],
        "action_families": [
            ActionKnowledgeFamilyV1(item).to_dict() for item in action_raw
        ],
    }


class KnowledgeFamilyCatalogV1(_Record):
    _validator = staticmethod(_validate_catalog)

    @property
    def task_families(self) -> tuple[TaskKnowledgeFamilyV1, ...]:
        return tuple(TaskKnowledgeFamilyV1(value) for value in self["task_families"])

    @property
    def action_families(self) -> tuple[ActionKnowledgeFamilyV1, ...]:
        return tuple(
            ActionKnowledgeFamilyV1(value) for value in self["action_families"]
        )


def _typed_tasks(
    values: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
) -> tuple[SubtaskKnowledgeV3, ...]:
    return tuple(
        value if isinstance(value, SubtaskKnowledgeV3) else SubtaskKnowledgeV3(value)
        for value in values
    )


def _typed_actions(
    values: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
) -> tuple[ActionKnowledgeV3, ...]:
    return tuple(
        value if isinstance(value, ActionKnowledgeV3) else ActionKnowledgeV3(value)
        for value in values
    )


def validate_catalog_against_knowledge(
    catalog: KnowledgeFamilyCatalogV1 | Mapping[str, Any],
    *,
    task_knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
    action_knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
) -> KnowledgeFamilyCatalogV1:
    """Validate routing membership against the current atomic Store."""

    typed_catalog = (
        catalog
        if isinstance(catalog, KnowledgeFamilyCatalogV1)
        else KnowledgeFamilyCatalogV1(catalog)
    )
    tasks = _typed_tasks(task_knowledge)
    actions = _typed_actions(action_knowledge)
    task_memberships = [0] * len(tasks)
    action_memberships = [0] * len(actions)

    for family_index, family in enumerate(typed_catalog.task_families):
        for member_index in family["member_indices"]:
            if member_index >= len(tasks):
                _fail(
                    f"knowledge_family_catalog.task_families[{family_index}]"
                    ".member_indices",
                    f"index {member_index} is outside the Task Store",
                )
            task_memberships[member_index] += 1

    for family_index, family in enumerate(typed_catalog.action_families):
        for member_index in family["member_indices"]:
            if member_index >= len(actions):
                _fail(
                    f"knowledge_family_catalog.action_families[{family_index}]"
                    ".member_indices",
                    f"index {member_index} is outside the Action Store",
                )
            if actions[member_index]["condition"]["action"] != family["action"]:
                _fail(
                    f"knowledge_family_catalog.action_families[{family_index}]",
                    "contains an atomic member with a different action",
                )
            action_memberships[member_index] += 1

    for label, values, memberships in (
        ("Task", tasks, task_memberships),
        ("Action", actions, action_memberships),
    ):
        for index, (value, count) in enumerate(zip(values, memberships, strict=True)):
            if value["status"] == "supported" and count != 1:
                _fail(
                    "knowledge_family_catalog",
                    f"supported {label} member {index} must belong to exactly one "
                    f"family, found {count}",
                )
    return typed_catalog


def render_task_family_routing_card(
    family: TaskKnowledgeFamilyV1 | Mapping[str, Any],
) -> str:
    typed = (
        family
        if isinstance(family, TaskKnowledgeFamilyV1)
        else TaskKnowledgeFamilyV1(family)
    )
    exclusions = "; ".join(typed["exclusions"]) or "none"
    return (
        f"Family: {typed['family_name']}\n"
        f"Route when: {typed['routing_summary']}\n"
        f"Applicable when: {typed['applicability_summary']}\n"
        f"Exclude when: {exclusions}"
    )


def render_action_family_routing_card(
    family: ActionKnowledgeFamilyV1 | Mapping[str, Any],
) -> str:
    typed = (
        family
        if isinstance(family, ActionKnowledgeFamilyV1)
        else ActionKnowledgeFamilyV1(family)
    )
    exclusions = "; ".join(typed["exclusions"]) or "none"
    return (
        f"Action: {typed['action']}\n"
        f"Family: {typed['family_name']}\n"
        f"Route when: {typed['routing_summary']}\n"
        f"Applicable when: {typed['applicability_summary']}\n"
        f"Expected effect: {typed['expected_effect_summary']}\n"
        f"Exclude when: {exclusions}"
    )


def render_task_knowledge_card(
    knowledge: SubtaskKnowledgeV3 | Mapping[str, Any],
) -> str:
    typed = (
        knowledge
        if isinstance(knowledge, SubtaskKnowledgeV3)
        else SubtaskKnowledgeV3(knowledge)
    )
    condition = typed["condition"]
    strategy = typed["subtask_strategy"]
    evidence = typed["evidence_summary"]
    relations = "; ".join(condition["relevant_relations"]) or "none"
    next_subtask = strategy["planned_next_subtask"] or "none"
    return (
        f"Goal: {condition['overall_goal']}\n"
        f"Condition: {condition['task_state']}; {relations}\n"
        f"Subtask: {strategy['subtask']}\n"
        f"Purpose: {strategy['purpose']}\n"
        f"Selection basis: {strategy['selection_basis_summary']}\n"
        f"Complete when: {strategy['completion_condition']}\n"
        f"Next: {next_subtask}\n"
        f"Evidence: {evidence['support']} support, {evidence['oppose']} oppose, "
        f"{evidence['unverified']} unverified"
    )


def knowledge_family_catalog_json_schema() -> dict[str, Any]:
    """Strict provider schema for semantic Family grouping."""

    natural_text = {"type": "string", "minLength": 1, "pattern": r"^[^_]+$"}
    texts = {"type": "array", "items": natural_text, "maxItems": 32}
    members = {
        "type": "array",
        "items": {"type": "integer", "minimum": 0},
        "minItems": 1,
    }

    def object_schema(properties: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": copy.deepcopy(dict(properties)),
            "required": list(properties),
            "additionalProperties": False,
        }

    common = {
        "family_name": natural_text,
        "routing_summary": natural_text,
        "applicability_summary": natural_text,
        "strategy_summary": natural_text,
        "exclusions": texts,
        "member_indices": members,
    }
    task_family = object_schema({**common, "completion_summary": natural_text})
    action_family = object_schema(
        {
            "action": {"type": "string", "enum": sorted(_ACTIONS)},
            **common,
            "expected_effect_summary": natural_text,
        }
    )
    return object_schema(
        {
            "task_families": {"type": "array", "items": task_family},
            "action_families": {"type": "array", "items": action_family},
        }
    )


_CATALOG_BUILDER_PROMPT = """Build a compact routing catalog for reusable robot manipulation knowledge.

Group supported Task Knowledge by the shared planning decision problem. Group supported Action Knowledge first by exact action and then by the shared object or interaction profile, local condition, intended effect, and comparable geometry problem. Do not group distinctions that can change applicability, grounding, physical realization, or expected effect.

Every supplied source index must appear exactly once in the corresponding families. An Action Family may contain only one exact action. Family summaries are recall descriptions, not executable guidance and not evidence.

Write the invariant local decision problem, not the demonstrated task outcome. Before returning each Family, apply this counterfactual test: if all object labels, overall goals, task instructions, and benchmark scenarios were replaced while the same local relations, decision role, geometry problem, and expected effect remained, every summary field must still be true. If not, rewrite it using those invariant relations and effects. Do not copy a member's named task sequence or end goal into a Family name or summary. Exclude incidental color, arm, benchmark name, trajectory wording, coordinates, poses, candidate names, IDs, hashes, and file paths. Do not use task-specific rules or invent knowledge. Return strict JSON only."""


def build_knowledge_family_catalog_input(
    *,
    task_knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
    action_knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
) -> str:
    tasks = _typed_tasks(task_knowledge)
    actions = _typed_actions(action_knowledge)
    payload = {
        "supported_task_knowledge": [
            {"source_index": index, "compact_card": render_task_knowledge_card(value)}
            for index, value in enumerate(tasks)
            if value["status"] == "supported"
        ],
        "supported_action_knowledge": [
            {
                "source_index": index,
                "action": value["condition"]["action"],
                "compact_card": render_action_knowledge_card(value),
            }
            for index, value in enumerate(actions)
            if value["status"] == "supported"
        ],
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class KnowledgeFamilyBuildResult:
    catalog: KnowledgeFamilyCatalogV1
    raw_response: dict[str, Any] | str


class VLMKnowledgeFamilyCatalogBuilder:
    """Use one semantic model call to group supported atomic knowledge."""

    def __init__(self, backend: HierarchicalReflectionBackend) -> None:
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must expose complete")
        self._backend = backend

    def build(
        self,
        *,
        task_knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
        action_knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
    ) -> KnowledgeFamilyBuildResult:
        tasks = _typed_tasks(task_knowledge)
        actions = _typed_actions(action_knowledge)
        if not any(value["status"] == "supported" for value in (*tasks, *actions)):
            catalog = KnowledgeFamilyCatalogV1(
                {"task_families": [], "action_families": []}
            )
            return KnowledgeFamilyBuildResult(catalog=catalog, raw_response={})
        completion = self._backend.complete(
            instructions=_CATALOG_BUILDER_PROMPT,
            input_text=build_knowledge_family_catalog_input(
                task_knowledge=tasks,
                action_knowledge=actions,
            ),
            images=(),
            output_schema=knowledge_family_catalog_json_schema(),
            schema_name="hpk_v3_knowledge_family_catalog",
        )
        parsed = _strict_response(completion.output)
        if set(parsed) != {"task_families", "action_families"}:
            _fail("knowledge_family_catalog", "response fields mismatch")
        catalog = validate_catalog_against_knowledge(
            parsed,
            task_knowledge=tasks,
            action_knowledge=actions,
        )
        raw_response: dict[str, Any] | str
        if isinstance(completion.output, Mapping):
            raw_response = copy.deepcopy(dict(completion.output))
        else:
            raw_response = completion.output
        return KnowledgeFamilyBuildResult(
            catalog=catalog,
            raw_response=raw_response,
        )


def render_action_knowledge_card(
    knowledge: ActionKnowledgeV3 | Mapping[str, Any],
) -> str:
    typed = (
        knowledge
        if isinstance(knowledge, ActionKnowledgeV3)
        else ActionKnowledgeV3(knowledge)
    )
    condition = typed["condition"]
    geometry = typed["geometric_strategy"]
    effect = typed["expected_effect"]
    evidence = typed["evidence_summary"]
    condition_parts = [condition["object_description"]]
    condition_parts.extend(
        condition[key]
        for key in ("held_state", "support_relation", "target_relation")
        if key in condition
    )
    geometry_parts = [
        geometry[key]
        for key in (
            "approach_direction",
            "approach_reference",
            "interaction_region",
            "orientation_relation",
            "placement_relation",
            "contact_relation",
            "clearance_or_support_constraint",
        )
        if key in geometry
    ]
    geometry_parts.extend(f"avoid {value}" for value in geometry["avoid"])
    return (
        f"Action: {condition['action']}\n"
        f"Condition: {'; '.join(condition_parts)}\n"
        f"Geometry: {'; '.join(geometry_parts)}\n"
        f"Expected effect: {effect['physical_effect']}; verify by "
        f"{effect['verification_observation']}\n"
        f"Evidence: {evidence['support']} support, {evidence['oppose']} oppose, "
        f"{evidence['unverified']} unverified, "
        f"{evidence['independent_verified_trials']} independent verified trials"
    )


__all__ = [
    "ActionKnowledgeFamilyV1",
    "KnowledgeFamilyBuildResult",
    "KnowledgeFamilyCatalogV1",
    "TaskKnowledgeFamilyV1",
    "VLMKnowledgeFamilyCatalogBuilder",
    "build_knowledge_family_catalog_input",
    "knowledge_family_catalog_json_schema",
    "render_action_family_routing_card",
    "render_action_knowledge_card",
    "render_task_family_routing_card",
    "render_task_knowledge_card",
    "validate_catalog_against_knowledge",
]
