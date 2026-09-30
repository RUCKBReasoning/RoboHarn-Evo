#!/usr/bin/env python3
"""Materialize one selected task's public official-prompt contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path


def _read_object(path: Path) -> dict:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("selection must be one object")
    return dict(value)


def _selected_task(selection: Mapping, *, role: str = "primary") -> str:
    if role not in {"primary", "reserve"}:
        raise ValueError("selection role must be primary or reserve")
    values = selection.get("selections")
    selected = [
        item
        for item in values or []
        if isinstance(item, Mapping) and item.get("role") == role
    ]
    if len(selected) != 1:
        raise ValueError(f"selection must contain exactly one {role}")
    return " ".join(str(selected[0].get("small_task") or "").strip().split())


def build_official_prompt(
    inventory_path: Path, selection_path: Path, *, role: str = "primary"
) -> tuple[str, str]:
    task = _selected_task(_read_object(selection_path), role=role)
    records = []
    for line in (
        inventory_path.expanduser().resolve().read_text(encoding="utf-8").splitlines()
    ):
        if line.strip():
            value = json.loads(line)
            if isinstance(value, Mapping) and value.get("small_task") == task:
                records.append(value)
    if len(records) != 1:
        raise ValueError("inventory must contain exactly one selected task")
    fragments = records[0].get("prompt_text_fragments")
    if not isinstance(fragments, list):
        raise ValueError("selected task has no public prompt fragments")
    lines = [" ".join(str(item).strip().split()) for item in fragments]
    lines = [item for item in lines if item]
    if not lines:
        raise ValueError("selected task public prompt is empty")
    return task, "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a selected ESI task's audited public prompt fragments."
    )
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--selection-role",
        choices=("primary", "reserve"),
        default="primary",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"official prompt output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    task, prompt = build_official_prompt(
        args.candidate_inventory,
        args.selection,
        role=args.selection_role,
    )
    (output / "official_prompt.txt").write_text(prompt, encoding="utf-8")
    (output / "prompt_audit.json").write_text(
        json.dumps(
            {
                "schema": "roboharn_evo/esi_bench/public_prompt_materialization/v1",
                "small_task": task,
                "source": "static AST audit of official prompt builder",
                "answer_or_outcome_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
