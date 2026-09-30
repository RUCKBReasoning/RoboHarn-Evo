from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.hierarchical_knowledge import ActionKnowledgeV3, SubtaskKnowledgeV3
from roboharn_evo.agent.hpk.hierarchical_store import save_hierarchical_store
from roboharn_evo.agent.hpk.rgb_evidence import RGBEvidenceIndex
from roboharn_evo.agent.hpk.knowledge_family import (
    KnowledgeFamilyCatalogV1,
    validate_catalog_against_knowledge,
)

KNOWLEDGE_FAMILY_FILENAME = "knowledge_families.json"


class KnowledgeFamilyStoreError(ValueError):
    """The family sidecar is missing, malformed, or inconsistent with its Store."""


def load_knowledge_family_catalog(
    root: str | Path,
    *,
    task_knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
    action_knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
) -> KnowledgeFamilyCatalogV1:
    path = Path(root) / KNOWLEDGE_FAMILY_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return validate_catalog_against_knowledge(
            payload,
            task_knowledge=task_knowledge,
            action_knowledge=action_knowledge,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise KnowledgeFamilyStoreError(f"invalid {path}: {exc}") from exc


def save_knowledge_family_catalog(
    root: str | Path,
    catalog: KnowledgeFamilyCatalogV1 | Mapping[str, Any],
    *,
    task_knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
    action_knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
) -> Path:
    typed = validate_catalog_against_knowledge(
        catalog,
        task_knowledge=task_knowledge,
        action_knowledge=action_knowledge,
    )
    path = Path(root) / KNOWLEDGE_FAMILY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(typed.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def save_hierarchical_store_with_catalog(
    root: str | Path,
    *,
    task_knowledge: Sequence[SubtaskKnowledgeV3 | Mapping[str, Any]],
    action_knowledge: Sequence[ActionKnowledgeV3 | Mapping[str, Any]],
    catalog: KnowledgeFamilyCatalogV1 | Mapping[str, Any],
    evidence_index: RGBEvidenceIndex | None = None,
) -> Path:
    """Validate one coherent Store view, then rewrite facts and routing sidecar."""

    typed_catalog = validate_catalog_against_knowledge(
        catalog,
        task_knowledge=task_knowledge,
        action_knowledge=action_knowledge,
    )
    directory = save_hierarchical_store(
        root,
        task_knowledge=task_knowledge,
        action_knowledge=action_knowledge,
        evidence_index=evidence_index,
    )
    save_knowledge_family_catalog(
        directory,
        typed_catalog,
        task_knowledge=task_knowledge,
        action_knowledge=action_knowledge,
    )
    return directory


__all__ = [
    "KNOWLEDGE_FAMILY_FILENAME",
    "KnowledgeFamilyStoreError",
    "load_knowledge_family_catalog",
    "save_hierarchical_store_with_catalog",
    "save_knowledge_family_catalog",
]
