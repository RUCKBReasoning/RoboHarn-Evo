#!/usr/bin/env python3
"""Export answer-free ESI instance metadata for deterministic split building."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.benchmark_adapters.esi_bench import (  # noqa: E402
    ESISplitError,
    ESISplitInstance,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize public ESI question metadata without answers or simulator state."
    )
    parser.add_argument("--json-clean-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def normalize_public_instances(json_clean_root: Path) -> tuple[ESISplitInstance, ...]:
    root = json_clean_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ESI json_clean root is missing: {root}")

    instances = []
    seen_refs: set[str] = set()
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ESISplitError(f"cannot read ESI metadata: {path}") from exc
        if not isinstance(payload, Mapping):
            raise ESISplitError(f"ESI metadata must be an object: {path}")
        if "json_paths" in payload:
            continue

        required = {"id", "small_task", "big_task", "scene", "room", "question"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ESISplitError(f"ESI question metadata is missing {missing}: {path}")
        instance_ref = str(payload["id"]).strip()
        if instance_ref in seen_refs:
            raise ESISplitError(f"duplicate ESI instance_ref: {instance_ref}")
        seen_refs.add(instance_ref)
        instances.append(
            ESISplitInstance(
                instance_ref=instance_ref,
                small_task=str(payload["small_task"]),
                big_task=str(payload["big_task"]),
                scene_group=str(payload["scene"]),
                room_group=str(payload["room"]),
                question_text=str(payload["question"]),
                metadata_path=str(path),
                question_index=0,
            )
        )
    if not instances:
        raise ESISplitError("ESI json_clean root contains no canonical questions")
    return tuple(instances)


def write_public_instances(path: Path, instances: tuple[ESISplitInstance, ...]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        for instance in instances:
            stream.write(
                json.dumps(
                    instance.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def main() -> int:
    args = _parse_args()
    instances = normalize_public_instances(args.json_clean_root)
    write_public_instances(args.output_jsonl, instances)
    print(
        json.dumps(
            {
                "instance_count": len(instances),
                "small_task_count": len({item.small_task for item in instances}),
                "big_task_count": len({item.big_task for item in instances}),
                "output_jsonl": str(args.output_jsonl.expanduser().resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
