"""Derive matched Q1 method views from one consolidated source pool."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.hierarchical_knowledge import (
    ActionKnowledgeV3,
    SubtaskKnowledgeV3,
)
from roboharn_evo.agent.hpk.hierarchical_store import (
    load_hierarchical_store,
    save_hierarchical_store,
)


METHOD_VIEW_SCHEMA = "roboharn_evo/rmbench/hpk_formal_q1_method_views/v1"


class FormalSourceError(ValueError):
    """The formal source pool cannot produce matched method inputs."""


def select_reflection_evidence(
    descriptions: Sequence[str],
    images: Sequence[Any],
    *,
    max_images: int,
) -> tuple[list[str], list[Any], list[int]]:
    """Retain ordered, uniformly spaced visual boundaries within one call cap."""

    if len(descriptions) != len(images) or not images:
        raise FormalSourceError("reflection images and descriptions must align")
    if (
        isinstance(max_images, bool)
        or not isinstance(max_images, int)
        or max_images < 2
    ):
        raise FormalSourceError("max_images must be an integer of at least two")
    if len(images) <= max_images:
        indices = list(range(len(images)))
    else:
        indices = [
            (index * (len(images) - 1)) // (max_images - 1)
            for index in range(max_images)
        ]
        if len(set(indices)) != max_images:
            raise FormalSourceError("uniform image selection produced duplicates")
    selected_descriptions = []
    for source_index in indices:
        if source_index == 0:
            binding = "initial trajectory observation"
        else:
            binding = f"observation after ordered action chunk {source_index}"
        selected_descriptions.append(
            f"{binding}: {' '.join(str(descriptions[source_index]).split())}"
        )
    return (
        selected_descriptions,
        [images[index] for index in indices],
        indices,
    )


def _join_strings(values: Any) -> str:
    if isinstance(values, str):
        return " ".join(values.split())
    if isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        return "; ".join(text for item in values if (text := _join_strings(item)))
    if isinstance(values, Mapping):
        return "; ".join(
            text for item in values.values() if (text := _join_strings(item))
        )
    return ""


def _task_paragraph(value: SubtaskKnowledgeV3) -> str:
    condition = value["condition"]
    strategy = value["subtask_strategy"]
    parts = [
        "Past executions encountered",
        _join_strings(condition),
        f"A useful next step was {strategy['subtask']}.",
        f"The purpose was {strategy['purpose']}.",
        f"Completion was checked as {strategy['completion_condition']}.",
    ]
    if strategy["planned_next_subtask"] is not None:
        parts.append(f"The following step was {strategy['planned_next_subtask']}.")
    evidence = value["evidence_summary"]
    parts.append(
        "Observed evidence was "
        f"{evidence['support']} supporting, {evidence['oppose']} opposing, and "
        f"{evidence['unverified']} unverified executions."
    )
    return " ".join(part for part in parts if part)


def _action_paragraph(value: ActionKnowledgeV3) -> str:
    condition = value["condition"]
    strategy = value["geometric_strategy"]
    effect = value["expected_effect"]
    evidence = value["evidence_summary"]
    return " ".join(
        (
            f"Past executions of {condition['action']} encountered {_join_strings(condition)}.",
            f"The relative interaction used {_join_strings(strategy)}.",
            f"The expected physical result was {_join_strings(effect)}.",
            "Observed evidence was "
            f"{evidence['support']} supporting, {evidence['oppose']} opposing, and "
            f"{evidence['unverified']} unverified executions.",
        )
    )


def render_flat_reflection(
    task_knowledge: Sequence[SubtaskKnowledgeV3],
    action_knowledge: Sequence[ActionKnowledgeV3],
    *,
    max_chars: int = 4000,
) -> tuple[str, dict[str, int]]:
    """Flatten the same consolidated information without typed retrieval."""

    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 256:
        raise FormalSourceError("flat reflection budget must be at least 256 chars")
    task_paragraphs = [("task", _task_paragraph(value)) for value in task_knowledge]
    action_paragraphs = [
        ("action", _action_paragraph(value)) for value in action_knowledge
    ]
    paragraphs: list[tuple[str, str]] = []
    for index in range(max(len(task_paragraphs), len(action_paragraphs))):
        if index < len(task_paragraphs):
            paragraphs.append(task_paragraphs[index])
        if index < len(action_paragraphs):
            paragraphs.append(action_paragraphs[index])
    included: list[str] = []
    included_task = 0
    included_action = 0
    for kind, paragraph in paragraphs:
        candidate = "\n\n".join((*included, paragraph))
        if len(candidate) > max_chars:
            continue
        included.append(paragraph)
        included_task += kind == "task"
        included_action += kind == "action"
    if not included:
        first_kind, first = (
            paragraphs[0]
            if paragraphs
            else ("none", "No reusable lesson was observed.")
        )
        included = [first[:max_chars].rstrip()]
        included_task = int(first_kind == "task")
        included_action = int(first_kind == "action")
    lesson = "\n\n".join(included)
    return lesson, {
        "max_chars": max_chars,
        "lesson_chars": len(lesson),
        "task_records_available": len(task_knowledge),
        "task_records_rendered": included_task,
        "action_records_available": len(action_knowledge),
        "action_records_rendered": included_action,
    }


def materialize_method_views(
    *,
    consolidated_store: str | Path,
    output_root: str | Path,
    source_collections: Sequence[str],
    flat_max_chars: int = 4000,
) -> dict[str, Any]:
    """Create Off/Flat/Task/Action/Full views from one immutable source pool."""

    full_tasks, full_actions = load_hierarchical_store(consolidated_store)
    if not full_tasks or not full_actions:
        raise FormalSourceError(
            "the consolidated source must contain both Task and Action Knowledge"
        )
    sources = [str(value).strip() for value in source_collections]
    if len(sources) != 10 or any(not value for value in sources):
        raise FormalSourceError(
            "formal method views require exactly 10 source trajectories"
        )
    if len(set(sources)) != 10:
        raise FormalSourceError("source trajectory labels must be unique")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    for method, tasks, actions in (
        ("task", full_tasks, ()),
        ("action", (), full_actions),
        ("full", full_tasks, full_actions),
    ):
        save_hierarchical_store(
            root / method / "store",
            task_knowledge=tasks,
            action_knowledge=actions,
        )
    lesson, flat_report = render_flat_reflection(
        full_tasks,
        full_actions,
        max_chars=flat_max_chars,
    )
    flat_root = root / "flat"
    flat_root.mkdir(parents=True)
    (flat_root / "flat_reflection.txt").write_text(lesson + "\n", encoding="utf-8")
    (root / "off").mkdir()
    report = {
        "schema": METHOD_VIEW_SCHEMA,
        "source_collections": sources,
        "same_source_pool_for_all_methods": True,
        "consolidated_counts": {
            "task": len(full_tasks),
            "action": len(full_actions),
        },
        "methods": {
            "off": {"persistent_experience": False},
            "flat": {
                "persistent_experience": True,
                "representation": "one bounded natural-language reflection",
                **flat_report,
            },
            "task": {
                "task_records": len(full_tasks),
                "action_records": 0,
            },
            "action": {
                "task_records": 0,
                "action_records": len(full_actions),
            },
            "full": {
                "task_records": len(full_tasks),
                "action_records": len(full_actions),
            },
        },
        "online_reflection": False,
        "online_update": False,
        "evaluation_store_read_only": True,
    }
    (root / "method_views.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "METHOD_VIEW_SCHEMA",
    "FormalSourceError",
    "materialize_method_views",
    "render_flat_reflection",
    "select_reflection_evidence",
]
