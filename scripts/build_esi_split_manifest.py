#!/usr/bin/env python3
"""Build a deterministic ESI source/development/heldout split manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from roboharn_evo.benchmark_adapters.esi_bench import (  # noqa: E402
    ESISplitError,
    ESISplitInstance,
    audit_split_counts,
    build_split_manifest,
    save_split_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split normalized public ESI instance metadata without importing "
            "OmniGibson."
        )
    )
    parser.add_argument("--instances-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--exclude-small-task",
        action="append",
        default=[],
        help=(
            "Explicitly exclude a public small-task label that cannot satisfy "
            "the three-way disjoint-group requirement. May be repeated."
        ),
    )
    parser.add_argument(
        "--exclude-publicly-unsplittable",
        action="store_true",
        help=(
            "Exclude and audit tasks whose public scene+room/question "
            "components cannot form three disjoint splits."
        ),
    )
    parser.add_argument(
        "--exclude-instance-ref",
        action="append",
        default=[],
        help=(
            "Reserve an instance already used for setup, development, or a "
            "runtime gate so it cannot enter any frozen split. May be repeated."
        ),
    )
    return parser.parse_args()


def _read_instances(path: Path) -> tuple[ESISplitInstance, ...]:
    values = []
    for line_number, line in enumerate(
        path.expanduser().resolve().read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ESISplitError(
                f"instances JSONL line {line_number} is invalid"
            ) from exc
        if not isinstance(value, Mapping):
            raise ESISplitError(f"instances JSONL line {line_number} must be an object")
        values.append(ESISplitInstance.from_mapping(value))
    if not values:
        raise ESISplitError("instances JSONL is empty")
    return tuple(values)


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(value), ensure_ascii=False, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _exclude_tasks(
    instances: Sequence[ESISplitInstance], names: Sequence[str]
) -> tuple[tuple[ESISplitInstance, ...], tuple[str, ...], int]:
    excluded = tuple(
        sorted({" ".join(name.strip().casefold().split()) for name in names})
    )
    if any(not name for name in excluded):
        raise ESISplitError("excluded small-task labels must be non-empty")
    available = {item.small_task.casefold() for item in instances}
    unknown = sorted(set(excluded).difference(available))
    if unknown:
        raise ESISplitError(f"excluded small-task labels are absent: {unknown}")
    selected = tuple(
        item for item in instances if item.small_task.casefold() not in excluded
    )
    if not selected:
        raise ESISplitError("small-task exclusions removed every ESI instance")
    return selected, excluded, len(instances) - len(selected)


def _exclude_instance_refs(
    instances: Sequence[ESISplitInstance], refs: Sequence[str]
) -> tuple[tuple[ESISplitInstance, ...], tuple[str, ...]]:
    excluded = tuple(sorted({" ".join(ref.strip().split()) for ref in refs}))
    if any(not ref for ref in excluded):
        raise ESISplitError("excluded instance refs must be non-empty")
    available = {item.instance_ref for item in instances}
    unknown = sorted(set(excluded).difference(available))
    if unknown:
        raise ESISplitError(f"excluded instance refs are absent: {unknown}")
    selected = tuple(item for item in instances if item.instance_ref not in excluded)
    if not selected:
        raise ESISplitError("instance exclusions removed every ESI instance")
    return selected, excluded


def main() -> int:
    args = _parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"split output root is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    all_instances, excluded_instance_refs = _exclude_instance_refs(
        _read_instances(args.instances_jsonl), args.exclude_instance_ref
    )
    count_audit = audit_split_counts(all_instances)
    publicly_unsplittable = tuple(
        task
        for task, counts in count_audit.items()
        if counts["scene_room_question_component_count"] < 3
    )
    requested_exclusions = list(args.exclude_small_task)
    if args.exclude_publicly_unsplittable:
        requested_exclusions.extend(publicly_unsplittable)
    instances, excluded_tasks, excluded_count = _exclude_tasks(
        all_instances, requested_exclusions
    )
    manifest = build_split_manifest(
        instances,
        seed=args.seed,
    )
    save_split_manifest(output / "split_manifest.json", manifest)
    (output / "esi_split_audit.json").write_text(
        json.dumps(
            {
                "seed": manifest.seed,
                "excluded_instance_refs": list(excluded_instance_refs),
                "excluded_instance_ref_count": len(excluded_instance_refs),
                "excluded_small_tasks": list(excluded_tasks),
                "excluded_instance_count": excluded_count,
                "publicly_unsplittable_small_tasks": list(publicly_unsplittable),
                "count_audit_before_grouping": dict(manifest.count_audit),
                "grouping_by_task": dict(manifest.grouping_by_task),
                "final_counts": {
                    "source": len(manifest.source),
                    "development": len(manifest.development),
                    "heldout": len(manifest.heldout),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for name in ("source", "development", "heldout"):
        _write_jsonl(
            output / f"{name}_instances.jsonl",
            [item.to_dict() for item in getattr(manifest, name)],
        )
    print(
        json.dumps(
            {
                "seed": manifest.seed,
                "excluded_instance_refs": list(excluded_instance_refs),
                "excluded_instance_ref_count": len(excluded_instance_refs),
                "excluded_small_tasks": list(excluded_tasks),
                "excluded_instance_count": excluded_count,
                "publicly_unsplittable_small_tasks": list(publicly_unsplittable),
                "source_count": len(manifest.source),
                "development_count": len(manifest.development),
                "heldout_count": len(manifest.heldout),
                "grouping_by_task": dict(manifest.grouping_by_task),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
