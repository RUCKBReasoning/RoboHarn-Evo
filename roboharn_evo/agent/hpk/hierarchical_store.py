from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.hierarchical_knowledge import (
    ActionKnowledgeV3,
    HPKV3ValidationError,
    SubtaskKnowledgeV3,
)
from roboharn_evo.agent.hpk.rgb_evidence import EVIDENCE_INDEX_FILENAME, RGBEvidenceIndex

TASK_KNOWLEDGE_FILENAME = "task_knowledge.jsonl"
ACTION_KNOWLEDGE_FILENAME = "action_knowledge.jsonl"


class HierarchicalHPKStoreError(HPKV3ValidationError):
    """A Store record or semantic grouping is invalid."""


def _merged_status(summary: Mapping[str, int]) -> str:
    if summary["oppose"]:
        return "contested"
    if summary["support"]:
        return "supported"
    return "candidate"


def _group_indices(
    groups: Sequence[Mapping[str, Any]],
    *,
    source_count: int,
    label: str,
) -> tuple[tuple[tuple[int, ...], dict[str, Any]], ...]:
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise HierarchicalHPKStoreError(f"{label} groups must be an array")
    parsed: list[tuple[tuple[int, ...], dict[str, Any]]] = []
    seen: set[int] = set()
    for group_index, source in enumerate(groups):
        if not isinstance(source, Mapping) or set(source) != {
            "source_indices",
            "canonical",
        }:
            raise HierarchicalHPKStoreError(
                f"{label} group {group_index} must contain source_indices and canonical"
            )
        raw_indices = source["source_indices"]
        if isinstance(raw_indices, (str, bytes)) or not isinstance(
            raw_indices, Sequence
        ):
            raise HierarchicalHPKStoreError(
                f"{label} group {group_index} source_indices must be an array"
            )
        indices = tuple(raw_indices)
        if not indices:
            raise HierarchicalHPKStoreError(
                f"{label} group {group_index} must contain at least one source"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= source_count
            for value in indices
        ):
            raise HierarchicalHPKStoreError(
                f"{label} group {group_index} contains an invalid source index"
            )
        if len(indices) != len(set(indices)) or seen.intersection(indices):
            raise HierarchicalHPKStoreError(
                f"{label} source indices must occur exactly once"
            )
        canonical = source["canonical"]
        if not isinstance(canonical, Mapping):
            raise HierarchicalHPKStoreError(
                f"{label} group {group_index} canonical value must be an object"
            )
        seen.update(indices)
        parsed.append((indices, copy.deepcopy(dict(canonical))))
    if seen != set(range(source_count)):
        raise HierarchicalHPKStoreError(
            f"{label} groups must cover every source exactly once"
        )
    return tuple(parsed)


def merge_subtask_units(
    values: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
    *,
    groups: Sequence[Mapping[str, Any]],
) -> tuple[SubtaskKnowledgeV3, ...]:
    """Merge model-grouped task units while summing evidence deterministically."""

    typed = tuple(
        value if isinstance(value, SubtaskKnowledgeV3) else SubtaskKnowledgeV3(value)
        for value in values
    )
    result: list[SubtaskKnowledgeV3] = []
    for indices, canonical in _group_indices(
        groups,
        source_count=len(typed),
        label="task",
    ):
        summary = {"support": 0, "oppose": 0, "unverified": 0}
        for index in indices:
            source_summary = typed[index]["evidence_summary"]
            for key in summary:
                summary[key] += source_summary[key]
        payload = {
            **canonical,
            "evidence_summary": summary,
            "status": _merged_status(summary),
        }
        result.append(SubtaskKnowledgeV3(payload))
    return tuple(result)


def merge_action_units(
    values: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
    *,
    groups: Sequence[Mapping[str, Any]],
) -> tuple[ActionKnowledgeV3, ...]:
    """Merge model-grouped action units without turning unverified into support."""

    typed = tuple(
        value if isinstance(value, ActionKnowledgeV3) else ActionKnowledgeV3(value)
        for value in values
    )
    result: list[ActionKnowledgeV3] = []
    for group_index, (indices, canonical) in enumerate(
        _group_indices(groups, source_count=len(typed), label="action")
    ):
        source_actions = {typed[index]["condition"]["action"] for index in indices}
        canonical_condition = canonical.get("condition")
        canonical_action = (
            canonical_condition.get("action")
            if isinstance(canonical_condition, Mapping)
            else None
        )
        if len(source_actions) != 1 or canonical_action not in source_actions:
            raise HierarchicalHPKStoreError(
                f"action group {group_index} must not mix action types"
            )
        summary = {
            "support": 0,
            "oppose": 0,
            "unverified": 0,
            "independent_verified_trials": 0,
        }
        for index in indices:
            source_summary = typed[index]["evidence_summary"]
            for key in summary:
                summary[key] += source_summary[key]
        payload = {
            **canonical,
            "evidence_summary": summary,
            "status": _merged_status(summary),
        }
        result.append(ActionKnowledgeV3(payload))
    return tuple(result)


def _load_jsonl(path: Path, record_type: type) -> tuple[Any, ...]:
    if not path.exists():
        return ()
    records = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            records.append(record_type(json.loads(line)))
        except (json.JSONDecodeError, HPKV3ValidationError) as exc:
            raise HierarchicalHPKStoreError(
                f"{path.name} line {index + 1} is invalid: {exc}"
            ) from exc
    return tuple(records)


def _save_jsonl(path: Path, values: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value.to_dict(), ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def load_hierarchical_store(
    root: str | Path,
) -> tuple[tuple[SubtaskKnowledgeV3, ...], tuple[ActionKnowledgeV3, ...]]:
    directory = Path(root)
    return (
        _load_jsonl(directory / TASK_KNOWLEDGE_FILENAME, SubtaskKnowledgeV3),
        _load_jsonl(directory / ACTION_KNOWLEDGE_FILENAME, ActionKnowledgeV3),
    )


def save_hierarchical_store(
    root: str | Path,
    *,
    task_knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
    action_knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
    evidence_index: RGBEvidenceIndex | None = None,
) -> Path:
    """Write already-consolidated atomic knowledge; no trajectory package is stored."""

    directory = Path(root)
    tasks = tuple(
        value if isinstance(value, SubtaskKnowledgeV3) else SubtaskKnowledgeV3(value)
        for value in task_knowledge
    )
    actions = tuple(
        value if isinstance(value, ActionKnowledgeV3) else ActionKnowledgeV3(value)
        for value in action_knowledge
    )
    if evidence_index is None and (directory / EVIDENCE_INDEX_FILENAME).exists():
        raise ValueError("updating RGB knowledge requires its updated evidence index")
    if evidence_index is not None:
        evidence_index.validate_knowledge(tasks, actions)
    _save_jsonl(directory / TASK_KNOWLEDGE_FILENAME, tasks)
    _save_jsonl(directory / ACTION_KNOWLEDGE_FILENAME, actions)
    if evidence_index is not None:
        evidence_index.save(directory)
    return directory


__all__ = [
    "ACTION_KNOWLEDGE_FILENAME",
    "TASK_KNOWLEDGE_FILENAME",
    "HierarchicalHPKStoreError",
    "load_hierarchical_store",
    "merge_action_units",
    "merge_subtask_units",
    "save_hierarchical_store",
]
