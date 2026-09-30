from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from roboharn_evo.agent.hpk.hierarchical_knowledge import (
    ActionKnowledgeV3,
    HPKV3ValidationError,
    SubtaskKnowledgeV3,
)
from roboharn_evo.agent.hpk.hierarchical_store import (
    merge_action_units,
    merge_subtask_units,
)
from roboharn_evo.agent.hpk.rgb_evidence import RGBEvidenceIndex, knowledge_with_evidence
from roboharn_evo.agent.hpk.vlm_hierarchical_reflector import (
    HierarchicalReflectionBackend,
    _strict_response,
)

_SYSTEM_PROMPT = """You consolidate reusable robot manipulation knowledge by meaning.

The input contains atomic task knowledge and action knowledge from different trajectories. Group paraphrases when their reusable causal meaning is the same. Decide from each condition whether an object attribute, scene relation, arm choice, or location is task-relevant or merely incidental; do not assume either in advance. Ignore only wording and source index. Do not group any distinction that can change applicability, task ordering, grounded action realization, or expected physical effect.

For each group, write one short natural-language canonical knowledge content. Canonical content must be object- and trajectory-independent while preserving every distinction that can change planning or grounded geometry. Source indices are temporary grouping references only; never copy them into canonical knowledge. Do not output IDs, hashes, file paths, poses, candidate names, or hidden metadata. Return strict JSON only. Every source index must appear exactly once in its corresponding task or action groups."""

_PARTITIONED_SYSTEM_PROMPT = """Consolidate reusable robot manipulation knowledge independently inside each supplied Knowledge Family partition.

Within one partition, group paraphrases only when their reusable causal meaning is the same. Preserve every distinction that can change applicability, task ordering, grounded realization, geometry, or expected physical effect. Never compare, merge, or move units across two Family partitions. Return every supplied family index exactly once and every local source index exactly once inside that partition.

Write short object- and trajectory-independent canonical content. Source and Family indices are temporary request references only. Do not output IDs, hashes, paths, poses, candidates, or hidden metadata. Return strict JSON only."""


def _object_schema(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": copy.deepcopy(dict(properties)),
        "required": list(properties),
        "additionalProperties": False,
    }


def _nullable_string() -> dict[str, Any]:
    return {"anyOf": [{"type": "string", "pattern": r"^[^_]+$"}, {"type": "null"}]}


def _string() -> dict[str, Any]:
    return {"type": "string", "pattern": r"^[^_]+$"}


def _string_array() -> dict[str, Any]:
    return {"type": "array", "items": _string(), "maxItems": 32}


def semantic_consolidation_json_schema() -> dict[str, Any]:
    """Strict provider schema for one image-free consolidation call."""

    task_canonical = _object_schema(
        {
            "condition": _object_schema(
                {
                    "overall_goal": _string(),
                    "task_state": _string(),
                    "relevant_relations": _string_array(),
                }
            ),
            "subtask_strategy": _object_schema(
                {
                    "subtask": _string(),
                    "purpose": _string(),
                    "selection_basis_summary": _string(),
                    "completion_condition": _string(),
                    "planned_next_subtask": _nullable_string(),
                }
            ),
        }
    )
    action_canonical = _object_schema(
        {
            "condition": _object_schema(
                {
                    "action": {
                        "type": "string",
                        "enum": ["contact", "grasp", "place"],
                    },
                    "object_description": _string(),
                    "held_state": _nullable_string(),
                    "support_relation": _nullable_string(),
                    "target_relation": _nullable_string(),
                }
            ),
            "geometric_strategy": _object_schema(
                {
                    "approach_direction": _nullable_string(),
                    "approach_reference": _nullable_string(),
                    "interaction_region": _nullable_string(),
                    "orientation_relation": _nullable_string(),
                    "placement_relation": _nullable_string(),
                    "contact_relation": _nullable_string(),
                    "clearance_or_support_constraint": _nullable_string(),
                    "avoid": _string_array(),
                }
            ),
            "expected_effect": _object_schema(
                {
                    "physical_effect": _string(),
                    "verification_observation": _string(),
                }
            ),
        }
    )

    def group(canonical: Mapping[str, Any]) -> dict[str, Any]:
        return _object_schema(
            {
                "source_indices": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "minItems": 1,
                },
                "canonical": canonical,
            }
        )

    return _object_schema(
        {
            "task_groups": {"type": "array", "items": group(task_canonical)},
            "action_groups": {
                "type": "array",
                "items": group(action_canonical),
            },
        }
    )


def partitioned_semantic_consolidation_json_schema() -> dict[str, Any]:
    """Strict schema for batching isolated affected-Family consolidations."""

    base = semantic_consolidation_json_schema()
    task_group = base["properties"]["task_groups"]["items"]
    action_group = base["properties"]["action_groups"]["items"]

    def partition(group: Mapping[str, Any]) -> dict[str, Any]:
        return _object_schema(
            {
                "family_index": {"type": "integer", "minimum": 0},
                "groups": {"type": "array", "items": copy.deepcopy(dict(group))},
            }
        )

    return _object_schema(
        {
            "task_partitions": {
                "type": "array",
                "items": partition(task_group),
            },
            "action_partitions": {
                "type": "array",
                "items": partition(action_group),
            },
        }
    )


def _semantic_content(value: Any) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("evidence_summary")
    payload.pop("status")
    return payload


def build_semantic_consolidation_input(
    *,
    task_knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
    action_knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
) -> str:
    tasks = tuple(
        value if isinstance(value, SubtaskKnowledgeV3) else SubtaskKnowledgeV3(value)
        for value in task_knowledge
    )
    actions = tuple(
        value if isinstance(value, ActionKnowledgeV3) else ActionKnowledgeV3(value)
        for value in action_knowledge
    )

    payload = {
        "task_units": [
            {"source_index": index, "semantic_content": _semantic_content(value)}
            for index, value in enumerate(tasks)
        ],
        "action_units": [
            {"source_index": index, "semantic_content": _semantic_content(value)}
            for index, value in enumerate(actions)
        ],
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def build_partitioned_semantic_consolidation_input(
    *,
    task_partitions: Mapping[int, Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]]],
    action_partitions: Mapping[int, Sequence[ActionKnowledgeV3 | Mapping[str, Any]]],
) -> str:
    """Render isolated Family partitions with partition-local source indices."""

    def index(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HPKV3ValidationError("Family partition index must be non-negative")
        return value

    task_payload = []
    for family_index, values in sorted(task_partitions.items()):
        typed = tuple(
            value
            if isinstance(value, SubtaskKnowledgeV3)
            else SubtaskKnowledgeV3(value)
            for value in values
        )
        if not typed:
            raise HPKV3ValidationError("Task Family partition must not be empty")
        task_payload.append(
            {
                "family_index": index(family_index),
                "units": [
                    {
                        "source_index": source_index,
                        "semantic_content": _semantic_content(value),
                    }
                    for source_index, value in enumerate(typed)
                ],
            }
        )
    action_payload = []
    for family_index, values in sorted(action_partitions.items()):
        typed = tuple(
            value if isinstance(value, ActionKnowledgeV3) else ActionKnowledgeV3(value)
            for value in values
        )
        if not typed:
            raise HPKV3ValidationError("Action Family partition must not be empty")
        action_payload.append(
            {
                "family_index": index(family_index),
                "units": [
                    {
                        "source_index": source_index,
                        "semantic_content": _semantic_content(value),
                    }
                    for source_index, value in enumerate(typed)
                ],
            }
        )
    return json.dumps(
        {
            "task_partitions": task_payload,
            "action_partitions": action_payload,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class SemanticConsolidationResult:
    task_knowledge: tuple[SubtaskKnowledgeV3, ...]
    action_knowledge: tuple[ActionKnowledgeV3, ...]
    raw_response: dict[str, Any] | str
    evidence_index: RGBEvidenceIndex | None = None


@dataclass(frozen=True, slots=True)
class PartitionedSemanticConsolidationResult:
    task_partitions: tuple[tuple[int, tuple[SubtaskKnowledgeV3, ...]], ...]
    action_partitions: tuple[tuple[int, tuple[ActionKnowledgeV3, ...]], ...]
    raw_response: dict[str, Any] | str


class VLMKnowledgeConsolidator:
    """Make one semantic grouping call; deterministic code only aggregates evidence."""

    def __init__(self, backend: HierarchicalReflectionBackend) -> None:
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must expose complete")
        self._backend = backend

    def consolidate(
        self,
        *,
        task_knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
        action_knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
        evidence_index: RGBEvidenceIndex | None = None,
    ) -> SemanticConsolidationResult:
        tasks = tuple(
            value
            if isinstance(value, SubtaskKnowledgeV3)
            else SubtaskKnowledgeV3(value)
            for value in task_knowledge
        )
        actions = tuple(
            value if isinstance(value, ActionKnowledgeV3) else ActionKnowledgeV3(value)
            for value in action_knowledge
        )
        if evidence_index is not None:
            evidence_index.validate_knowledge(tasks, actions)
        completion = self._backend.complete(
            instructions=_SYSTEM_PROMPT,
            input_text=build_semantic_consolidation_input(
                task_knowledge=tasks,
                action_knowledge=actions,
            ),
            images=(),
            output_schema=semantic_consolidation_json_schema(),
            schema_name="hpk_v3_semantic_consolidation",
        )
        return self.materialize_response(
            task_knowledge=tasks, action_knowledge=actions,
            response=completion.output, evidence_index=evidence_index,
        )

    @staticmethod
    def materialize_response(*, task_knowledge, action_knowledge, response, evidence_index: RGBEvidenceIndex | None = None) -> SemanticConsolidationResult:
        tasks = tuple(value if isinstance(value, SubtaskKnowledgeV3) else SubtaskKnowledgeV3(value) for value in task_knowledge)
        actions = tuple(value if isinstance(value, ActionKnowledgeV3) else ActionKnowledgeV3(value) for value in action_knowledge)
        if evidence_index is not None:
            evidence_index.validate_knowledge(tasks, actions)
        parsed = _strict_response(response)
        if set(parsed) != {"task_groups", "action_groups"}:
            raise HPKV3ValidationError(
                "semantic consolidation must contain task_groups and action_groups"
            )
        merged_tasks = merge_subtask_units(tasks, groups=parsed["task_groups"])
        merged_actions = merge_action_units(actions, groups=parsed["action_groups"])
        merged_evidence = None
        if evidence_index is not None:
            merged_evidence = evidence_index.regroup(
                task_groups=[group["source_indices"] for group in parsed["task_groups"]],
                action_groups=[group["source_indices"] for group in parsed["action_groups"]],
            )
            merged_tasks = tuple(knowledge_with_evidence(value.to_dict(), merged_evidence.for_knowledge("task", index), kind="task") for index, value in enumerate(merged_tasks))
            merged_actions = tuple(knowledge_with_evidence(value.to_dict(), merged_evidence.for_knowledge("action", index), kind="action") for index, value in enumerate(merged_actions))
            merged_evidence.validate_knowledge(merged_tasks, merged_actions)
        raw_response: dict[str, Any] | str
        if isinstance(response, Mapping):
            raw_response = copy.deepcopy(dict(response))
        else:
            raw_response = response
        return SemanticConsolidationResult(
            task_knowledge=merged_tasks,
            action_knowledge=merged_actions,
            raw_response=raw_response,
            evidence_index=merged_evidence,
        )

    def consolidate_partitions(
        self,
        *,
        task_partitions: Mapping[
            int,
            Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
        ],
        action_partitions: Mapping[
            int,
            Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
        ],
    ) -> PartitionedSemanticConsolidationResult:
        """Consolidate all affected Families in one strictly isolated call."""

        typed_tasks = {
            family_index: tuple(
                value
                if isinstance(value, SubtaskKnowledgeV3)
                else SubtaskKnowledgeV3(value)
                for value in values
            )
            for family_index, values in task_partitions.items()
        }
        typed_actions = {
            family_index: tuple(
                value
                if isinstance(value, ActionKnowledgeV3)
                else ActionKnowledgeV3(value)
                for value in values
            )
            for family_index, values in action_partitions.items()
        }
        if not typed_tasks and not typed_actions:
            return PartitionedSemanticConsolidationResult((), (), {})
        completion = self._backend.complete(
            instructions=_PARTITIONED_SYSTEM_PROMPT,
            input_text=build_partitioned_semantic_consolidation_input(
                task_partitions=typed_tasks,
                action_partitions=typed_actions,
            ),
            images=(),
            output_schema=partitioned_semantic_consolidation_json_schema(),
            schema_name="hpk_v3_partitioned_semantic_consolidation",
        )
        parsed = _strict_response(completion.output)
        if set(parsed) != {"task_partitions", "action_partitions"}:
            raise HPKV3ValidationError(
                "partitioned consolidation response fields mismatch"
            )

        def parse_partitions(
            raw: Any,
            *,
            expected: Mapping[int, tuple[Any, ...]],
            merge: Any,
            label: str,
        ) -> tuple[tuple[int, tuple[Any, ...]], ...]:
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise HPKV3ValidationError(f"{label} partitions must be an array")
            outputs: dict[int, tuple[Any, ...]] = {}
            for item in raw:
                if not isinstance(item, Mapping) or set(item) != {
                    "family_index",
                    "groups",
                }:
                    raise HPKV3ValidationError(f"{label} partition fields mismatch")
                family_index = item["family_index"]
                if (
                    isinstance(family_index, bool)
                    or not isinstance(family_index, int)
                    or family_index not in expected
                    or family_index in outputs
                ):
                    raise HPKV3ValidationError(
                        f"{label} partition Family index is invalid"
                    )
                outputs[family_index] = merge(
                    expected[family_index],
                    groups=item["groups"],
                )
            if set(outputs) != set(expected):
                raise HPKV3ValidationError(
                    f"{label} partitions must cover every requested Family"
                )
            return tuple(sorted(outputs.items()))

        task_outputs = parse_partitions(
            parsed["task_partitions"],
            expected=typed_tasks,
            merge=merge_subtask_units,
            label="Task",
        )
        action_outputs = parse_partitions(
            parsed["action_partitions"],
            expected=typed_actions,
            merge=merge_action_units,
            label="Action",
        )
        raw_response: dict[str, Any] | str
        if isinstance(completion.output, Mapping):
            raw_response = copy.deepcopy(dict(completion.output))
        else:
            raw_response = completion.output
        return PartitionedSemanticConsolidationResult(
            task_partitions=task_outputs,
            action_partitions=action_outputs,
            raw_response=raw_response,
        )


__all__ = [
    "PartitionedSemanticConsolidationResult",
    "SemanticConsolidationResult",
    "VLMKnowledgeConsolidator",
    "build_partitioned_semantic_consolidation_input",
    "build_semantic_consolidation_input",
    "partitioned_semantic_consolidation_json_schema",
    "semantic_consolidation_json_schema",
]
