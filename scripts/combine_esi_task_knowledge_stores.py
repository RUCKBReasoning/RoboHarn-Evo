#!/usr/bin/env python3
"""Combine independently validated ESI Task Knowledge stores."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.benchmark_adapters.esi_bench import (  # noqa: E402
    ESIKnowledgeError,
    FrozenTaskKnowledgeStore,
    build_shuffle_manifest,
    load_frozen_task_store,
    load_split_manifest,
    save_frozen_task_store,
    save_shuffle_manifest,
    validate_store_split_provenance,
)


def combine_stores(
    stores: tuple[FrozenTaskKnowledgeStore, ...],
) -> FrozenTaskKnowledgeStore:
    if len(stores) < 2:
        raise ESIKnowledgeError("at least two frozen stores are required")
    knowledge = tuple(item for store in stores for item in store.knowledge)
    metadata = tuple(item for store in stores for item in store.metadata)
    flat_lessons = tuple(item for store in stores for item in store.flat_lessons)
    source_refs = [ref for item in metadata for ref in item.source_refs]
    if len(source_refs) != len(set(source_refs)):
        raise ESIKnowledgeError("input stores contain duplicate source refs")
    return FrozenTaskKnowledgeStore(
        knowledge=knowledge,
        metadata=tuple(
            replace(item, line_index=index) for index, item in enumerate(metadata)
        ),
        flat_lessons=flat_lessons,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine validated ESI stores and freeze a shuffled control."
    )
    parser.add_argument("--store-root", type=Path, action="append", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shuffle-seed", type=int, default=20260826)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"combined store output is not empty: {output}")
    roots = tuple(path.expanduser().resolve() for path in args.store_root)
    combined = combine_stores(tuple(load_frozen_task_store(path) for path in roots))
    manifest = load_split_manifest(args.split_manifest)
    validate_store_split_provenance(combined, manifest)
    save_frozen_task_store(output, combined)
    shuffle = build_shuffle_manifest(len(combined.knowledge), seed=args.shuffle_seed)
    save_shuffle_manifest(output / "shuffle_manifest.json", shuffle)
    audit = {
        "schema": "roboharn_evo/esi_bench/combined_task_store_audit/v1",
        "input_store_count": len(roots),
        "input_knowledge_counts": [
            len(load_frozen_task_store(path).knowledge) for path in roots
        ],
        "combined_knowledge_count": len(combined.knowledge),
        "combined_source_trajectory_count": len(combined.flat_lessons),
        "shuffle_permutation": list(shuffle.permutation),
        "shuffle_fixed_points": list(shuffle.fixed_points),
    }
    (output / "combination_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
