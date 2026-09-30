from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from roboharn_evo.agent.hpk.schemas import HPKValidationError
from roboharn_evo.agent.hpk.semantic_knowledge import validate_semantic_knowledge


class SemanticHPKStoreError(HPKValidationError):
    pass


def load_semantic_knowledge(path: str | Path) -> tuple[dict[str, Any], ...]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SemanticHPKStoreError(f"cannot read semantic HPK: {exc}") from exc
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SemanticHPKStoreError(
                f"knowledge.jsonl line {index + 1} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise SemanticHPKStoreError(
                f"knowledge.jsonl line {index + 1} must be an object"
            )
        records.append(validate_semantic_knowledge(value))
    return tuple(records)


def save_semantic_knowledge(
    path: str | Path, records: Sequence[Mapping[str, Any]]
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    validated = [validate_semantic_knowledge(value) for value in records]
    raw = "".join(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        for value in validated
    )
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(raw, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def update_semantic_knowledge(
    path: str | Path,
    candidate: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    promotion_policy: Mapping[str, Any] | Any,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Update one semantic strategy in the single knowledge file."""

    from roboharn_evo.agent.hpk.updater import (
        append_semantic_evidence,
        apply_semantic_promotion,
        same_semantic_strategy,
    )

    destination = Path(path)
    records = list(load_semantic_knowledge(destination)) if destination.exists() else []
    typed_candidate = validate_semantic_knowledge(candidate)
    match_index = next(
        (
            index
            for index, current in enumerate(records)
            if same_semantic_strategy(current, typed_candidate)
        ),
        None,
    )
    current = typed_candidate if match_index is None else records[match_index]
    updated = append_semantic_evidence(current, evidence)
    updated = apply_semantic_promotion(updated, promotion_policy)
    if match_index is None:
        records.append(updated)
    else:
        records[match_index] = updated
    save_semantic_knowledge(destination, records)
    return updated, tuple(records)


__all__ = [
    "SemanticHPKStoreError",
    "load_semantic_knowledge",
    "save_semantic_knowledge",
    "update_semantic_knowledge",
]
