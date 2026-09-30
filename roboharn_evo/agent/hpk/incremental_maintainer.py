from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.family_store import save_hierarchical_store_with_catalog
from roboharn_evo.agent.hpk.hierarchical_knowledge import (
    ActionKnowledgeV3,
    HPKV3ValidationError,
    SubtaskKnowledgeV3,
)
from roboharn_evo.agent.hpk.knowledge_family import (
    ActionKnowledgeFamilyV1,
    KnowledgeFamilyCatalogV1,
    TaskKnowledgeFamilyV1,
    render_action_family_routing_card,
    render_action_knowledge_card,
    render_task_family_routing_card,
    render_task_knowledge_card,
    validate_catalog_against_knowledge,
)
from roboharn_evo.agent.hpk.semantic_consolidator import VLMKnowledgeConsolidator
from roboharn_evo.agent.hpk.rgb_evidence import RGBEvidenceIndex, knowledge_with_evidence
from roboharn_evo.agent.hpk.rgb_maintenance import RGBKnowledgeReviewer
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (
    HierarchicalReflectionBackend,
    _strict_response,
)

_ASSIGNMENT_PROMPT = """Assign new atomic HPK knowledge to retrieval Families by meaning.

Compare applicability, decision role, geometry or effect semantics, and exclusions. Use an existing Family only when its shared decision problem preserves every distinction that can change applicability, grounding, or expected effect. Otherwise create one new Family. Group multiple new units together only when they share that decision problem. Action units may be grouped only with the same exact grasp, place, or contact action.

Do not decide from a benchmark name, task name, object color, arm, source index, keyword overlap, or wording alone. Do not invent knowledge, evidence, poses, candidates, IDs, hashes, or paths. Every new unit index must appear exactly once. Return strict JSON only."""

_SUMMARY_PROMPT = """Update only the supplied affected Knowledge Family summaries.

For each Family, summarize the common decision problem represented by its consolidated atomic member cards. Preserve distinctions that affect applicability, grounding, geometry, completion, or expected physical effect. State only invariants that remain true after replacing every object label, overall goal, task instruction, and benchmark scenario while keeping the same local relations, decision role, geometry problem, and expected effect. Never copy a member's named task sequence or end goal into the Family name or summaries. Exclude member-specific color, arm, benchmark or task name, coordinates, poses, candidates, IDs, hashes, paths, and evidence claims. A Family summary is recall metadata, not executable guidance and not evidence. Return one summary for every supplied Family index and no others. Return strict JSON only."""


def _fail(path: str, message: str) -> None:
    raise HPKV3ValidationError(f"{path}: {message}")


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


def load_atomic_knowledge(
    path: str | Path,
) -> tuple[tuple[SubtaskKnowledgeV3, ...], tuple[ActionKnowledgeV3, ...]]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HPKV3ValidationError(
            f"invalid atomic knowledge file {source}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "task_knowledge",
        "action_knowledge",
    }:
        raise HPKV3ValidationError(
            "atomic knowledge must contain task_knowledge and action_knowledge"
        )
    return (
        _typed_tasks(payload["task_knowledge"]),
        _typed_actions(payload["action_knowledge"]),
    )


def _natural_text() -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "pattern": r"^[^_]+$"}


def _nullable_index() -> dict[str, Any]:
    return {
        "anyOf": [
            {"type": "integer", "minimum": 0},
            {"type": "null"},
        ]
    }


def _nullable_object(value: Mapping[str, Any]) -> dict[str, Any]:
    return {"anyOf": [copy.deepcopy(dict(value)), {"type": "null"}]}


def _object_schema(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": copy.deepcopy(dict(properties)),
        "required": list(properties),
        "additionalProperties": False,
    }


def _family_summary_schema(*, action: bool) -> dict[str, Any]:
    text = _natural_text()
    properties: dict[str, Any] = {
        "family_name": text,
        "routing_summary": text,
        "applicability_summary": text,
        "strategy_summary": text,
    }
    if action:
        properties = {
            "action": {"type": "string", "enum": ["contact", "grasp", "place"]},
            **properties,
            "expected_effect_summary": text,
        }
    else:
        properties["completion_summary"] = text
    properties["exclusions"] = {
        "type": "array",
        "items": text,
        "maxItems": 32,
    }
    return _object_schema(properties)


def family_assignment_json_schema() -> dict[str, Any]:
    indices = {
        "type": "array",
        "items": {"type": "integer", "minimum": 0},
        "minItems": 1,
    }
    task_group = _object_schema(
        {
            "new_unit_indices": indices,
            "existing_family_index": _nullable_index(),
            "new_family": _nullable_object(_family_summary_schema(action=False)),
        }
    )
    action_group = _object_schema(
        {
            "new_unit_indices": indices,
            "existing_family_index": _nullable_index(),
            "new_family": _nullable_object(_family_summary_schema(action=True)),
        }
    )
    return _object_schema(
        {
            "task_assignments": {"type": "array", "items": task_group},
            "action_assignments": {"type": "array", "items": action_group},
        }
    )


def family_summary_update_json_schema() -> dict[str, Any]:
    task = _object_schema(
        {
            "family_index": {"type": "integer", "minimum": 0},
            **_family_summary_schema(action=False)["properties"],
        }
    )
    action = _object_schema(
        {
            "family_index": {"type": "integer", "minimum": 0},
            **_family_summary_schema(action=True)["properties"],
        }
    )
    return _object_schema(
        {
            "task_family_summaries": {"type": "array", "items": task},
            "action_family_summaries": {"type": "array", "items": action},
        }
    )


def _without_members(family: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(family))
    result.pop("member_indices", None)
    return result


def _with_members(summary: Mapping[str, Any], members: Sequence[int]) -> dict[str, Any]:
    return {**copy.deepcopy(dict(summary)), "member_indices": list(members)}


def build_family_assignment_input(
    *,
    catalog: KnowledgeFamilyCatalogV1,
    new_task_knowledge: Sequence[SubtaskKnowledgeV3],
    new_action_knowledge: Sequence[ActionKnowledgeV3],
) -> str:
    payload = {
        "task_family_cards": [
            {"family_index": index, "card": render_task_family_routing_card(family)}
            for index, family in enumerate(catalog.task_families)
        ],
        "action_family_cards": [
            {
                "family_index": index,
                "action": family["action"],
                "card": render_action_family_routing_card(family),
            }
            for index, family in enumerate(catalog.action_families)
        ],
        "new_task_knowledge": [
            {"new_unit_index": index, "compact_card": render_task_knowledge_card(value)}
            for index, value in enumerate(new_task_knowledge)
        ],
        "new_action_knowledge": [
            {
                "new_unit_index": index,
                "action": value["condition"]["action"],
                "compact_card": render_action_knowledge_card(value),
            }
            for index, value in enumerate(new_action_knowledge)
        ],
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _family_memberships(
    families: Sequence[Mapping[str, Any]],
) -> dict[int, int]:
    result: dict[int, int] = {}
    for family_index, family in enumerate(families):
        for member_index in family["member_indices"]:
            if member_index in result:
                _fail("knowledge_family_catalog", "atomic member has two families")
            result[member_index] = family_index
    return result


def _validate_assignment_groups(
    raw: Any,
    *,
    kind: str,
    new_values: Sequence[Any],
    families: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        _fail(f"family_assignment.{kind}_assignments", "must be an array")
    assignments: list[dict[str, Any]] = []
    new_families: list[dict[str, Any]] = []
    seen: set[int] = set()
    for group_index, item in enumerate(raw):
        path = f"family_assignment.{kind}_assignments[{group_index}]"
        if not isinstance(item, Mapping) or set(item) != {
            "new_unit_indices",
            "existing_family_index",
            "new_family",
        }:
            _fail(path, "fields mismatch")
        indices = item["new_unit_indices"]
        if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
            _fail(f"{path}.new_unit_indices", "must be an array")
        if not indices:
            _fail(f"{path}.new_unit_indices", "must not be empty")
        typed_indices: list[int] = []
        for value in indices:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < len(new_values)
                or value in seen
            ):
                _fail(
                    f"{path}.new_unit_indices", "contains an invalid or duplicate index"
                )
            seen.add(value)
            typed_indices.append(value)
        existing = item["existing_family_index"]
        proposed = item["new_family"]
        if existing is None:
            if not isinstance(proposed, Mapping):
                _fail(path, "new Family summary is required")
            if kind == "task":
                typed_family = TaskKnowledgeFamilyV1(
                    _with_members(proposed, [0])
                ).to_dict()
            else:
                typed_family = ActionKnowledgeFamilyV1(
                    _with_members(proposed, [0])
                ).to_dict()
                actions = {
                    new_values[index]["condition"]["action"] for index in typed_indices
                }
                if actions != {typed_family["action"]}:
                    _fail(path, "new Action Family mixes action types")
            new_family_index = len(families) + len(new_families)
            new_families.append(_without_members(typed_family))
            family_index = new_family_index
        else:
            if proposed is not None:
                _fail(path, "existing assignment must not also create a Family")
            if (
                isinstance(existing, bool)
                or not isinstance(existing, int)
                or not 0 <= existing < len(families)
            ):
                _fail(path, "existing Family index was not presented")
            if kind == "action":
                actions = {
                    new_values[index]["condition"]["action"] for index in typed_indices
                }
                if actions != {families[existing]["action"]}:
                    _fail(path, "Action assignment crosses action types")
            family_index = existing
        assignments.extend(
            {"new_unit_index": index, "family_index": family_index}
            for index in typed_indices
        )
    if seen != set(range(len(new_values))):
        _fail(
            f"family_assignment.{kind}_assignments",
            "must cover every new unit exactly once",
        )
    return assignments, new_families


def _evidence_totals(
    tasks: Sequence[SubtaskKnowledgeV3],
    actions: Sequence[ActionKnowledgeV3],
) -> dict[str, dict[str, int]]:
    task = {key: 0 for key in ("support", "oppose", "unverified")}
    action = {
        key: 0
        for key in (
            "support",
            "oppose",
            "unverified",
            "independent_verified_trials",
        )
    }
    for value in tasks:
        for key in task:
            task[key] += value["evidence_summary"][key]
    for value in actions:
        for key in action:
            action[key] += value["evidence_summary"][key]
    return {"task": task, "action": action}


def _sum_totals(
    first: Mapping[str, Mapping[str, int]],
    second: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        kind: {
            key: int(first[kind][key]) + int(second[kind][key]) for key in first[kind]
        }
        for kind in first
    }


def _evidence_report(
    *,
    existing_tasks: Sequence[SubtaskKnowledgeV3],
    existing_actions: Sequence[ActionKnowledgeV3],
    new_tasks: Sequence[SubtaskKnowledgeV3],
    new_actions: Sequence[ActionKnowledgeV3],
    result_tasks: Sequence[SubtaskKnowledgeV3],
    result_actions: Sequence[ActionKnowledgeV3],
) -> dict[str, Any]:
    before = _evidence_totals(existing_tasks, existing_actions)
    additions = _evidence_totals(new_tasks, new_actions)
    expected = _sum_totals(before, additions)
    actual = _evidence_totals(result_tasks, result_actions)
    error = sum(
        abs(expected[kind][key] - actual[kind][key])
        for kind in expected
        for key in expected[kind]
    )
    return {
        "existing": before,
        "new": additions,
        "expected_after": expected,
        "actual_after": actual,
        "evidence_conservation_error": error,
    }


def _exact_semantic_duplicate_rate(
    tasks: Sequence[SubtaskKnowledgeV3],
    actions: Sequence[ActionKnowledgeV3],
) -> float:
    semantic_records: list[str] = []
    for kind, values in (("task", tasks), ("action", actions)):
        for value in values:
            payload = value.to_dict()
            payload.pop("evidence_summary")
            payload.pop("status")
            semantic_records.append(
                kind
                + ":"
                + json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    if not semantic_records:
        return 0.0
    duplicate_count = len(semantic_records) - len(set(semantic_records))
    return duplicate_count / len(semantic_records)


@dataclass(frozen=True, slots=True)
class IncrementalMaintenanceResult:
    task_knowledge: tuple[SubtaskKnowledgeV3, ...]
    action_knowledge: tuple[ActionKnowledgeV3, ...]
    catalog: KnowledgeFamilyCatalogV1
    audit: dict[str, Any]
    evidence_index: RGBEvidenceIndex | None = None


class IncrementalKnowledgeMaintainer:
    """Assign once, consolidate affected Families, and rewrite one coherent Store."""

    def __init__(
        self,
        assignment_backend: HierarchicalReflectionBackend,
        *,
        consolidator: VLMKnowledgeConsolidator,
        summary_backend: HierarchicalReflectionBackend | None = None,
        rgb_reviewer: RGBKnowledgeReviewer | None = None,
    ) -> None:
        for label, backend in (
            ("assignment_backend", assignment_backend),
            ("summary_backend", summary_backend or assignment_backend),
        ):
            if not callable(getattr(backend, "complete", None)):
                raise TypeError(f"{label} must expose complete")
        if not callable(getattr(consolidator, "consolidate", None)):
            raise TypeError("consolidator must expose consolidate")
        self._assignment_backend = assignment_backend
        self._summary_backend = summary_backend or assignment_backend
        self._consolidator = consolidator
        self._rgb_reviewer = rgb_reviewer or RGBKnowledgeReviewer(assignment_backend)

    def maintain(
        self,
        *,
        task_knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
        action_knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
        catalog: KnowledgeFamilyCatalogV1 | Mapping[str, Any],
        new_task_knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
        new_action_knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
        output_root: str | Path | None = None,
        evidence_index: RGBEvidenceIndex | None = None,
        new_evidence_index: RGBEvidenceIndex | None = None,
        maintenance_rgb_enabled: bool = False,
        review_output_root: str | Path | None = None,
    ) -> IncrementalMaintenanceResult:
        tasks = _typed_tasks(task_knowledge)
        actions = _typed_actions(action_knowledge)
        additions_task = _typed_tasks(new_task_knowledge)
        additions_action = _typed_actions(new_action_knowledge)
        combined_evidence = None
        if (evidence_index is None) != (new_evidence_index is None):
            raise ValueError("RGB maintenance requires both stored and incoming evidence indices")
        if evidence_index is not None:
            evidence_index.validate_knowledge(tasks, actions)
            new_evidence_index.validate_knowledge(additions_task, additions_action)
            combined_evidence = evidence_index.append(new_evidence_index, task_offset=len(tasks), action_offset=len(actions))
            if review_output_root is None:
                raise ValueError("RGB maintenance requires a review output directory")
        typed_catalog = validate_catalog_against_knowledge(
            catalog,
            task_knowledge=tasks,
            action_knowledge=actions,
        )
        if not additions_task and not additions_action:
            if output_root is not None:
                save_hierarchical_store_with_catalog(
                    output_root,
                    task_knowledge=tasks,
                    action_knowledge=actions,
                    catalog=typed_catalog,
                    evidence_index=evidence_index,
                )
            report = _evidence_report(
                existing_tasks=tasks,
                existing_actions=actions,
                new_tasks=(),
                new_actions=(),
                result_tasks=tasks,
                result_actions=actions,
            )
            return IncrementalMaintenanceResult(
                task_knowledge=tasks,
                action_knowledge=actions,
                catalog=typed_catalog,
                evidence_index=evidence_index,
                audit={
                    "status": "no new atomic knowledge",
                    "affected_task_families": [],
                    "affected_action_families": [],
                    "model_calls": 0,
                    **report,
                },
            )

        assignment_input = build_family_assignment_input(
            catalog=typed_catalog,
            new_task_knowledge=additions_task,
            new_action_knowledge=additions_action,
        )
        assignment_completion = self._assignment_backend.complete(
            instructions=_ASSIGNMENT_PROMPT,
            input_text=assignment_input,
            images=(),
            output_schema=family_assignment_json_schema(),
            schema_name="hpk_v3_incremental_family_assignment",
        )
        assignment = _strict_response(assignment_completion.output)
        if set(assignment) != {"task_assignments", "action_assignments"}:
            _fail("family_assignment", "response fields mismatch")

        task_families = [family.to_dict() for family in typed_catalog.task_families]
        action_families = [family.to_dict() for family in typed_catalog.action_families]
        task_assignments, new_task_families = _validate_assignment_groups(
            assignment["task_assignments"],
            kind="task",
            new_values=additions_task,
            families=task_families,
        )
        action_assignments, new_action_families = _validate_assignment_groups(
            assignment["action_assignments"],
            kind="action",
            new_values=additions_action,
            families=action_families,
        )
        task_families.extend(
            _with_members(summary, [0]) for summary in new_task_families
        )
        action_families.extend(
            _with_members(summary, [0]) for summary in new_action_families
        )

        task_owner = _family_memberships(typed_catalog["task_families"])
        action_owner = _family_memberships(typed_catalog["action_families"])
        task_origins = {family: [index for index in range(len(tasks)) if task_owner.get(index) == family] for family in range(len(task_families))}
        action_origins = {family: [index for index in range(len(actions)) if action_owner.get(index) == family] for family in range(len(action_families))}
        for item in task_assignments:
            task_origins[item["family_index"]].append(len(tasks) + item["new_unit_index"])
        for item in action_assignments:
            action_origins[item["family_index"]].append(len(actions) + item["new_unit_index"])
        task_new_by_family: dict[int, list[SubtaskKnowledgeV3]] = {}
        action_new_by_family: dict[int, list[ActionKnowledgeV3]] = {}
        for item in task_assignments:
            task_new_by_family.setdefault(item["family_index"], []).append(
                additions_task[item["new_unit_index"]]
            )
        for item in action_assignments:
            action_new_by_family.setdefault(item["family_index"], []).append(
                additions_action[item["new_unit_index"]]
            )
        affected_task = tuple(sorted(task_new_by_family))
        affected_action = tuple(sorted(action_new_by_family))

        task_outputs: list[tuple[SubtaskKnowledgeV3, ...]] = []
        task_consolidation_inputs: dict[int, tuple[SubtaskKnowledgeV3, ...]] = {}
        consolidation_atomic_input_count = 0
        for family_index in range(len(task_families)):
            existing = tuple(
                value
                for index, value in enumerate(tasks)
                if task_owner.get(index) == family_index
            )
            new_values = tuple(task_new_by_family.get(family_index, ()))
            if new_values:
                consolidation_atomic_input_count += len(existing) + len(new_values)
                task_consolidation_inputs[family_index] = (*existing, *new_values)
                task_outputs.append(())
            else:
                task_outputs.append(existing)

        action_outputs: list[tuple[ActionKnowledgeV3, ...]] = []
        action_consolidation_inputs: dict[int, tuple[ActionKnowledgeV3, ...]] = {}
        for family_index in range(len(action_families)):
            existing = tuple(
                value
                for index, value in enumerate(actions)
                if action_owner.get(index) == family_index
            )
            new_values = tuple(action_new_by_family.get(family_index, ()))
            if new_values:
                consolidation_atomic_input_count += len(existing) + len(new_values)
                action_consolidation_inputs[family_index] = (*existing, *new_values)
                action_outputs.append(())
            else:
                action_outputs.append(existing)

        consolidate_partitions = getattr(
            self._consolidator,
            "consolidate_partitions",
            None,
        )
        partitioned_consolidation = callable(consolidate_partitions)
        consolidation_model_calls = 0
        rgb_group_sources = {"task": {}, "action": {}}
        if partitioned_consolidation:
            consolidated = consolidate_partitions(
                task_partitions=task_consolidation_inputs,
                action_partitions=action_consolidation_inputs,
            )
            consolidation_model_calls = int(
                bool(task_consolidation_inputs or action_consolidation_inputs)
            )
            for family_index, values in consolidated.task_partitions:
                task_outputs[family_index] = values
            for family_index, values in consolidated.action_partitions:
                action_outputs[family_index] = values
            if combined_evidence is not None:
                response = _strict_response(consolidated.raw_response)
                for kind, origins in (("task", task_origins), ("action", action_origins)):
                    for partition in response[f"{kind}_partitions"]:
                        family = partition["family_index"]
                        rgb_group_sources[kind][family] = [[origins[family][index] for index in group["source_indices"]] for group in partition["groups"]]
        else:
            if combined_evidence is not None:
                raise TypeError("RGB incremental maintenance requires partitioned semantic consolidation")
            for family_index, values in task_consolidation_inputs.items():
                consolidated = self._consolidator.consolidate(
                    task_knowledge=values,
                    action_knowledge=(),
                )
                consolidation_model_calls += 1
                if consolidated.action_knowledge:
                    _fail(
                        "incremental_maintenance",
                        "Task consolidation returned Action units",
                    )
                task_outputs[family_index] = consolidated.task_knowledge
            for family_index, values in action_consolidation_inputs.items():
                consolidated = self._consolidator.consolidate(
                    task_knowledge=(),
                    action_knowledge=values,
                )
                consolidation_model_calls += 1
                if consolidated.task_knowledge:
                    _fail(
                        "incremental_maintenance",
                        "Action consolidation returned Task units",
                    )
                action_outputs[family_index] = consolidated.action_knowledge

        for family_index, values in enumerate(action_outputs):
            expected_action = action_families[family_index]["action"]
            if any(value["condition"]["action"] != expected_action for value in values):
                _fail(
                    "incremental_maintenance",
                    "Action consolidation changed action type",
                )

        unassigned_tasks = tuple(
            value for index, value in enumerate(tasks) if index not in task_owner
        )
        unassigned_actions = tuple(
            value for index, value in enumerate(actions) if index not in action_owner
        )
        result_tasks: list[SubtaskKnowledgeV3] = []
        for family_index, values in enumerate(task_outputs):
            start = len(result_tasks)
            result_tasks.extend(values)
            task_families[family_index]["member_indices"] = list(
                range(start, len(result_tasks))
            )
        result_tasks.extend(unassigned_tasks)
        result_actions: list[ActionKnowledgeV3] = []
        for family_index, values in enumerate(action_outputs):
            start = len(result_actions)
            result_actions.extend(values)
            action_families[family_index]["member_indices"] = list(
                range(start, len(result_actions))
            )
        result_actions.extend(unassigned_actions)

        maintained_evidence = None
        review_reports = []
        if combined_evidence is not None:
            task_groups = [group for family in range(len(task_families)) for group in rgb_group_sources["task"].get(family, [[index] for index in task_origins[family]])]
            action_groups = [group for family in range(len(action_families)) for group in rgb_group_sources["action"].get(family, [[index] for index in action_origins[family]])]
            task_groups.extend([index] for index in range(len(tasks)) if index not in task_owner)
            action_groups.extend([index] for index in range(len(actions)) if index not in action_owner)
            maintained_evidence = combined_evidence.regroup(task_groups=task_groups, action_groups=action_groups)
            for kind, values, families, affected in (("task", result_tasks, task_families, affected_task), ("action", result_actions, action_families, affected_action)):
                for index, value in enumerate(values):
                    values[index] = knowledge_with_evidence(value.to_dict(), maintained_evidence.for_knowledge(kind, index), kind=kind)
                for family in affected:
                    for index in families[family]["member_indices"]:
                        values[index], maintained_evidence, report = self._rgb_reviewer.review(
                            kind=kind, knowledge_index=index, knowledge=values[index], evidence=maintained_evidence,
                            maintenance_rgb_enabled=maintenance_rgb_enabled,
                            output_root=Path(review_output_root) / f"{kind}_{index}",
                        )
                        review_reports.append(report)
                outputs = task_outputs if kind == "task" else action_outputs
                for family in range(len(families)):
                    outputs[family] = tuple(values[index] for index in families[family]["member_indices"])
            maintained_evidence.validate_knowledge(result_tasks, result_actions)

        summary_payload = {
            "affected_task_families": [
                {
                    "family_index": index,
                    "previous_summary": _without_members(task_families[index]),
                    "consolidated_member_cards": [
                        render_task_knowledge_card(value)
                        for value in task_outputs[index]
                    ],
                }
                for index in affected_task
            ],
            "affected_action_families": [
                {
                    "family_index": index,
                    "previous_summary": _without_members(action_families[index]),
                    "consolidated_member_cards": [
                        render_action_knowledge_card(value)
                        for value in action_outputs[index]
                    ],
                }
                for index in affected_action
            ],
        }
        summary_completion = self._summary_backend.complete(
            instructions=_SUMMARY_PROMPT,
            input_text=json.dumps(
                summary_payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ),
            images=(),
            output_schema=family_summary_update_json_schema(),
            schema_name="hpk_v3_affected_family_summary_update",
        )
        summaries = _strict_response(summary_completion.output)
        if set(summaries) != {
            "task_family_summaries",
            "action_family_summaries",
        }:
            _fail("family_summary_update", "response fields mismatch")
        self._apply_summary_updates(
            summaries["task_family_summaries"],
            kind="task",
            expected=set(affected_task),
            families=task_families,
        )
        self._apply_summary_updates(
            summaries["action_family_summaries"],
            kind="action",
            expected=set(affected_action),
            families=action_families,
        )

        result_catalog = validate_catalog_against_knowledge(
            {
                "task_families": task_families,
                "action_families": action_families,
            },
            task_knowledge=result_tasks,
            action_knowledge=result_actions,
        )
        if combined_evidence is None:
            evidence_report = _evidence_report(
                existing_tasks=tasks, existing_actions=actions,
                new_tasks=additions_task, new_actions=additions_action,
                result_tasks=result_tasks, result_actions=result_actions,
            )
        else:
            before_events = {(record["knowledge_type"], record["execution_event"]) for record in combined_evidence.records}
            after_events = {(record["knowledge_type"], record["execution_event"]) for record in maintained_evidence.records}
            evidence_report = {
                "evidence_conservation_error": len(before_events.symmetric_difference(after_events)),
                "unique_execution_events_before": len(before_events),
                "unique_execution_events_after": len(after_events),
                "evidence_counting_unit": "recorded execution event",
                "maintenance_reviews": review_reports,
            }
        if evidence_report["evidence_conservation_error"] != 0:
            _fail("incremental_maintenance", "evidence conservation failed")
        if output_root is not None:
            save_hierarchical_store_with_catalog(
                output_root,
                task_knowledge=result_tasks,
                action_knowledge=result_actions,
                catalog=result_catalog,
                evidence_index=maintained_evidence,
            )
        return IncrementalMaintenanceResult(
            task_knowledge=tuple(result_tasks),
            action_knowledge=tuple(result_actions),
            catalog=result_catalog,
            evidence_index=maintained_evidence,
            audit={
                "status": "incremental maintenance completed",
                "new_task_knowledge_count": len(additions_task),
                "new_action_knowledge_count": len(additions_action),
                "affected_task_families": list(affected_task),
                "affected_action_families": list(affected_action),
                "created_task_families": len(new_task_families),
                "created_action_families": len(new_action_families),
                "model_calls": 2 + consolidation_model_calls + sum(len(report["calls"]) for report in review_reports),
                "evidence_review_calls": sum(len(report["calls"]) for report in review_reports),
                "partitioned_consolidation": partitioned_consolidation,
                "affected_family_consolidation_calls": consolidation_model_calls,
                "family_assignment_count": len(additions_task) + len(additions_action),
                "family_assignment_coverage": 1.0,
                "action_partition_purity": 1.0,
                "affected_family_summary_coverage": 1.0,
                "exact_semantic_duplicate_rate": _exact_semantic_duplicate_rate(
                    result_tasks,
                    result_actions,
                ),
                "total_atomic_count_before": len(tasks) + len(actions),
                "consolidation_atomic_input_count": (consolidation_atomic_input_count),
                **evidence_report,
            },
        )

    @staticmethod
    def _apply_summary_updates(
        raw: Any,
        *,
        kind: str,
        expected: set[int],
        families: list[dict[str, Any]],
    ) -> None:
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            _fail(f"family_summary_update.{kind}", "must be an array")
        seen: set[int] = set()
        for item in raw:
            if not isinstance(item, Mapping) or "family_index" not in item:
                _fail(f"family_summary_update.{kind}", "summary fields mismatch")
            family_index = item["family_index"]
            if (
                isinstance(family_index, bool)
                or not isinstance(family_index, int)
                or family_index not in expected
                or family_index in seen
            ):
                _fail(f"family_summary_update.{kind}", "unexpected Family index")
            seen.add(family_index)
            summary = dict(item)
            summary.pop("family_index")
            members = families[family_index]["member_indices"]
            if kind == "task":
                typed = TaskKnowledgeFamilyV1(_with_members(summary, members)).to_dict()
            else:
                typed = ActionKnowledgeFamilyV1(
                    _with_members(summary, members)
                ).to_dict()
                if typed["action"] != families[family_index]["action"]:
                    _fail(
                        "family_summary_update.action",
                        "summary changed the Family action",
                    )
            families[family_index] = typed
        if seen != expected:
            _fail(
                f"family_summary_update.{kind}",
                "must update every affected Family exactly once",
            )


__all__ = [
    "IncrementalKnowledgeMaintainer",
    "IncrementalMaintenanceResult",
    "build_family_assignment_input",
    "family_assignment_json_schema",
    "family_summary_update_json_schema",
    "load_atomic_knowledge",
]
