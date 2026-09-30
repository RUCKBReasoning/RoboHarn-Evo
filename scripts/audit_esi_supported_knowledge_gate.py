#!/usr/bin/env python3
"""Evaluate the preregistered ESI supported-knowledge gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = (REPOSITORY_ROOT / "eval_result" / "esi_bench" / "runs").resolve()
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.benchmark_adapters.esi_bench import (  # noqa: E402
    ESIKnowledgeError,
    TaskKnowledgeNoveltyAudit,
    load_frozen_task_store,
    load_shuffle_manifest,
)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    values = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ESIKnowledgeError(
                f"{path.name} line {line_number} must be one object"
            )
        values.append(dict(value))
    return tuple(values)


def evaluate_supported_knowledge_gate(
    store_root: str | Path,
    novelty_audit_path: str | Path,
) -> dict[str, Any]:
    root = Path(store_root).expanduser().resolve()
    novelty_path = Path(novelty_audit_path).expanduser().resolve()
    reasons: list[str] = []
    try:
        store = load_frozen_task_store(root)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        build_audit: dict[str, Any] = {}
        try:
            loaded_audit = json.loads(
                (root / "build_audit.json").read_text(encoding="utf-8")
            )
            if isinstance(loaded_audit, Mapping):
                build_audit = dict(loaded_audit)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        support_no_go = bool(
            build_audit.get("development_supported_knowledge_count") == 0
            and build_audit.get("frozen_task_knowledge_count") == 0
            and build_audit.get("evaluation_store_written") is False
        )
        task_only = build_audit.get("action_knowledge_count") == 0
        return {
            "schema": "roboharn_evo/esi_bench/supported_knowledge_gate/v1",
            "result_status": "NEGATIVE_RESULT" if support_no_go else "BLOCKED",
            "go_for_frozen_probe": False,
            "knowledge_count": 0,
            "qualifying_knowledge_count": 0,
            "checks": {
                "store_loadable": False,
                "at_least_two_distinct_units": False,
                "support_gate": False,
                "novelty_gate": False,
                "nontrivial_shuffle": False,
                "task_only_store": task_only,
            },
            "reasons": [
                (
                    "knowledge build produced zero development-supported units; "
                    "no frozen Store was written"
                    if support_no_go
                    else f"frozen Store is unavailable or invalid: {exc}"
                )
            ],
        }

    knowledge_count = len(store.knowledge)
    semantic_units = [
        json.dumps(
            {
                "condition": item["condition"],
                "subtask_strategy": item["subtask_strategy"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in store.knowledge
    ]
    distinct = knowledge_count >= 2 and len(set(semantic_units)) == knowledge_count
    if not distinct:
        reasons.append("fewer than two distinct supported Task Knowledge units")

    support_outcomes = []
    for index, (item, metadata) in enumerate(
        zip(store.knowledge, store.metadata, strict=True)
    ):
        evidence = item["evidence_summary"]
        passed = bool(
            int(evidence["support"]) >= 3
            and int(evidence["oppose"]) == 0
            and len(metadata.source_refs) >= int(evidence["support"])
        )
        support_outcomes.append(passed)
        if not passed:
            reasons.append(
                f"knowledge unit {index} lacks three independent supports or has opposition"
            )
    support_gate = bool(support_outcomes) and all(support_outcomes)

    novelty_gate = False
    novelty_labels: list[str] = []
    try:
        novelty_records = _read_jsonl(novelty_path)
        by_index = {}
        for record in novelty_records:
            if set(record) != {"source_index", "semantic_content", "judgment"}:
                raise ESIKnowledgeError("novelty audit fields mismatch")
            source_index = record["source_index"]
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or source_index in by_index
            ):
                raise ESIKnowledgeError("novelty source indices are invalid")
            by_index[source_index] = record
        if set(by_index) != set(range(knowledge_count)):
            raise ESIKnowledgeError("novelty audit does not cover the frozen Store")
        for index, item in enumerate(store.knowledge):
            expected_semantic = {
                "condition": item["condition"],
                "subtask_strategy": item["subtask_strategy"],
            }
            record = by_index[index]
            if record["semantic_content"] != expected_semantic:
                raise ESIKnowledgeError("novelty audit semantic content mismatch")
            novelty = TaskKnowledgeNoveltyAudit.from_mapping(record["judgment"])
            novelty_labels.append(novelty.classification)
        novelty_gate = bool(novelty_labels) and all(
            label in {"partially novel", "substantially novel"}
            for label in novelty_labels
        )
        if not novelty_gate:
            reasons.append("at least one Task Knowledge unit is fully covered")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        reasons.append(f"novelty audit is unavailable or invalid: {exc}")

    shuffle_gate = False
    try:
        shuffle = load_shuffle_manifest(root / "shuffle_manifest.json")
        shuffle_gate = bool(
            shuffle.source_unit_count == knowledge_count
            and knowledge_count >= 2
            and not shuffle.fixed_points
        )
        if not shuffle_gate:
            reasons.append("shuffle manifest does not match the supported Store")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        reasons.append(f"nontrivial shuffle is unavailable or invalid: {exc}")

    task_only = (
        not (root / "action_knowledge.jsonl").exists()
        or not (root / "action_knowledge.jsonl").read_text(encoding="utf-8").strip()
    )
    if not task_only:
        reasons.append("ESI evaluation Store contains Action Knowledge")

    checks = {
        "store_loadable": True,
        "at_least_two_distinct_units": distinct,
        "support_gate": support_gate,
        "novelty_gate": novelty_gate,
        "nontrivial_shuffle": shuffle_gate,
        "task_only_store": task_only,
    }
    go = all(checks.values())
    return {
        "schema": "roboharn_evo/esi_bench/supported_knowledge_gate/v1",
        "result_status": "MECHANISM" if go else "BLOCKED",
        "go_for_frozen_probe": go,
        "knowledge_count": knowledge_count,
        "qualifying_knowledge_count": sum(
            support and novelty != "fully covered"
            for support, novelty in zip(
                support_outcomes,
                novelty_labels,
                strict=False,
            )
        ),
        "novelty_classifications": novelty_labels,
        "checks": checks,
        "reasons": reasons,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether an ESI Store may enter the frozen probe."
    )
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--novelty-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output_root.expanduser().resolve()
    if not output.is_relative_to(RUNS_ROOT):
        raise ESIKnowledgeError("gate output must stay under ESI runs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"gate output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result = evaluate_supported_knowledge_gate(
        args.store_root,
        args.novelty_audit,
    )
    (output / "support_novelty_gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["go_for_frozen_probe"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
