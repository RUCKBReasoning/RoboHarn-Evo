#!/usr/bin/env python3
"""Build a public, task-agnostic packet for choosing ESI source tasks."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.benchmark_adapters.esi_bench import (  # noqa: E402
    ESIKnowledgeError,
    load_split_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an offline ESI task-candidate audit packet."
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks" / "esi_bench" / "upstream",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _task_key(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().replace("&", "and").split())


def _load_registry(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("esi_candidate_registry", path)
    if spec is None or spec.loader is None:
        raise ESIKnowledgeError("cannot load the copied ESI task registry")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _literal_strings(node: ast.AST) -> tuple[str, ...]:
    values: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            text = " ".join(child.value.strip().split())
            if text and text not in values:
                values.append(text)
    return tuple(values)


def _declared_actions(tree: ast.Module) -> tuple[str, ...]:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "VALID_ACTIONS"
            for target in targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            return ()
        if isinstance(value, (set, list, tuple)):
            return tuple(sorted({str(item) for item in value if str(item).strip()}))
    return ()


def _prompt_components(path: Path) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    prompt_functions = tuple(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("build_system_prompt")
    )
    if not any(node.name == "build_system_prompt" for node in prompt_functions):
        raise ESIKnowledgeError(f"task module has no build_system_prompt: {path}")
    function_source = "\n\n".join(
        ast.get_source_segment(source, node) or "" for node in prompt_functions
    )
    fragments: list[str] = []
    for node in prompt_functions:
        for text in _literal_strings(node):
            if text not in fragments:
                fragments.append(text)
    return _declared_actions(tree), function_source, tuple(fragments)


def audit_prompt_module(path: Path) -> dict[str, Any]:
    actions, function_source, fragments = _prompt_components(path)
    dependency = path.with_name("common.py")
    dependency_used = False
    if len(fragments) < 2 and dependency.is_file() and dependency != path:
        common_actions, common_source, common_fragments = _prompt_components(dependency)
        actions = tuple(sorted(set(actions).union(common_actions)))
        function_source = (
            function_source + "\n\n# shared prompt builder\n" + common_source
        )
        fragments = tuple(dict.fromkeys((*fragments, *common_fragments)))
        dependency_used = True
    return {
        "declared_action_vocabulary": list(actions),
        "prompt_text_fragments": list(fragments),
        "prompt_builder_source": function_source,
        "shared_prompt_builder_included": dependency_used,
    }


def build_candidate_inventory(
    *, split_manifest: Path, upstream_root: Path
) -> tuple[dict[str, Any], ...]:
    manifest = load_split_manifest(split_manifest)
    registry = _load_registry(
        upstream_root / "src" / "active_explore" / "tasks" / "_registry.py"
    )
    module_map = {
        (_task_key(big), _task_key(small)): module
        for (big, small), module in registry.SMALL_TASK_MODULES.items()
    }
    split_counts = {
        part: Counter(item.small_task for item in getattr(manifest, part))
        for part in ("source", "development", "heldout")
    }
    split_scene_groups = {
        part: {
            small_task: len(
                {
                    item.scene_group
                    for item in getattr(manifest, part)
                    if item.small_task == small_task
                }
            )
            for small_task in split_counts[part]
        }
        for part in ("source", "development", "heldout")
    }
    source_by_task: dict[str, list[Any]] = {}
    for item in manifest.source:
        source_by_task.setdefault(item.small_task, []).append(item)
    records = []
    for small_task, source_items in sorted(source_by_task.items()):
        exemplar = sorted(source_items, key=lambda item: item.instance_ref)[0]
        module_name = module_map.get(
            (_task_key(exemplar.big_task), _task_key(small_task))
        )
        if module_name is None:
            raise ESIKnowledgeError(
                f"split task is absent from copied registry: {small_task}"
            )
        module_path = (
            upstream_root / "src" / "active_explore" / Path(*module_name.split("."))
        ).with_suffix(".py")
        prompt_audit = audit_prompt_module(module_path)
        records.append(
            {
                "big_task": exemplar.big_task,
                "small_task": small_task,
                "source_instance_count": split_counts["source"][small_task],
                "development_instance_count": split_counts["development"][small_task],
                "heldout_instance_count": split_counts["heldout"][small_task],
                "source_scene_group_count": split_scene_groups["source"][small_task],
                "development_scene_group_count": split_scene_groups["development"].get(
                    small_task, 0
                ),
                "heldout_scene_group_count": split_scene_groups["heldout"].get(
                    small_task, 0
                ),
                "representative_public_question": exemplar.question_text,
                **prompt_audit,
            }
        )
    return tuple(records)


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(value), ensure_ascii=False, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"candidate audit output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = build_candidate_inventory(
        split_manifest=args.split_manifest.expanduser().resolve(),
        upstream_root=args.upstream_root.expanduser().resolve(),
    )
    _write_jsonl(output / "candidate_inventory.jsonl", records)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "scientific_scope": "offline ESI source-task candidate audit",
                "task_count": len(records),
                "selection_rule": "frozen semantic judge over public prompts and split counts",
                "task_specific_rules_used": False,
                "external_model_calls": 0,
                "heldout_outcomes_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"task_count": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
