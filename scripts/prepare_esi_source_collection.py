#!/usr/bin/env python3
"""Preregister a result-blind ESI source-trajectory collection queue."""

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
    ESIKnowledgeError,
    ESISplitInstance,
    load_split_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze an ordered ESI source collection queue and call budget."
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--selection-role",
        choices=("primary", "reserve"),
        default="primary",
        help="Use the preregistered primary or reserve candidate.",
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt-cap", type=int, default=6)
    parser.add_argument("--queue-offset", type=int, default=0)
    parser.add_argument("--required-correct", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.99)
    return parser.parse_args()


def _load_selected_task(path: Path, *, role: str = "primary") -> str:
    if role not in {"primary", "reserve"}:
        raise ESIKnowledgeError("selection role must be primary or reserve")
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ESIKnowledgeError("cannot read source-task selection") from exc
    if not isinstance(value, Mapping):
        raise ESIKnowledgeError("source-task selection must be one object")
    selections = value.get("selections")
    if not isinstance(selections, list):
        raise ESIKnowledgeError("source-task selection has no selections")
    selected = [
        item
        for item in selections
        if isinstance(item, Mapping) and item.get("role") == role
    ]
    if len(selected) != 1:
        raise ESIKnowledgeError(f"source-task selection must have exactly one {role}")
    task = " ".join(str(selected[0].get("small_task") or "").strip().split())
    if not task:
        raise ESIKnowledgeError(f"{role} source task is empty")
    return task


def _runner_task(metadata_path: str) -> str:
    try:
        value = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ESIKnowledgeError("cannot read source instance metadata") from exc
    if not isinstance(value, Mapping):
        raise ESIKnowledgeError("source instance metadata must be one object")
    task = " ".join(
        str(value.get("runner_task") or value.get("task") or "").strip().split()
    )
    if not task:
        raise ESIKnowledgeError("source instance has no runner task")
    return task


def select_independent_source_instances(
    source: Sequence[ESISplitInstance],
    *,
    small_task: str,
    attempt_cap: int,
    queue_offset: int = 0,
) -> tuple[ESISplitInstance, ...]:
    if isinstance(attempt_cap, bool) or not isinstance(attempt_cap, int):
        raise ESIKnowledgeError("attempt_cap must be an integer")
    if attempt_cap < 1:
        raise ESIKnowledgeError("attempt_cap must be positive")
    if (
        isinstance(queue_offset, bool)
        or not isinstance(queue_offset, int)
        or queue_offset < 0
    ):
        raise ESIKnowledgeError("queue_offset must be a non-negative integer")
    normalized_task = " ".join(small_task.casefold().replace("_", " ").split())
    candidates = sorted(
        (
            item
            for item in source
            if " ".join(item.small_task.casefold().replace("_", " ").split())
            == normalized_task
        ),
        key=lambda item: (
            item.scene_group.casefold(),
            item.room_group.casefold(),
            item.instance_ref,
        ),
    )
    chosen = []
    seen_scenes = set()
    for item in candidates:
        scene = item.scene_group.casefold()
        if scene in seen_scenes:
            continue
        chosen.append(item)
        seen_scenes.add(scene)
        if len(chosen) == attempt_cap + queue_offset:
            break
    if len(chosen) != attempt_cap + queue_offset:
        raise ESIKnowledgeError(
            "primary source task lacks enough independent scene groups"
        )
    return tuple(chosen[queue_offset:])


def source_collection_budget(
    *,
    attempt_cap: int,
    required_correct: int,
    max_steps: int,
) -> dict[str, int]:
    for label, value in (
        ("attempt_cap", attempt_cap),
        ("required_correct", required_correct),
        ("max_steps", max_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ESIKnowledgeError(f"{label} must be a positive integer")
    if required_correct > attempt_cap:
        raise ESIKnowledgeError("required_correct cannot exceed attempt_cap")
    base_image_payloads = sum(min(step, 6) for step in range(1, max_steps + 1))
    forced_choice_image_payloads = min(max_steps, 5) + 1
    evaluated_calls = attempt_cap * (max_steps + 1)
    evaluated_images = attempt_cap * (
        base_image_payloads + forced_choice_image_payloads
    )
    summary_calls = required_correct
    summary_images = required_correct * max_steps
    return {
        "source_episodes": attempt_cap,
        "evaluated_model_calls": evaluated_calls,
        "source_trace_summary_calls": summary_calls,
        "total_model_calls": evaluated_calls + summary_calls,
        "automatic_retries": 0,
        "unique_simulator_frames": attempt_cap * max_steps,
        "evaluated_model_image_payloads": evaluated_images,
        "source_trace_summary_image_payloads": summary_images,
        "total_model_image_payloads": evaluated_images + summary_images,
        "simulator_camera_actions": attempt_cap * max(0, max_steps - 1),
        "retrieval_model_calls": 0,
        "sam3_calls": 0,
    }


def build_collection_plan(
    *,
    primary_task: str,
    selected: Sequence[ESISplitInstance],
    required_correct: int,
    max_steps: int,
    min_steps: int,
    threshold: float,
    queue_offset: int = 0,
) -> dict[str, Any]:
    if (
        isinstance(required_correct, bool)
        or not isinstance(required_correct, int)
        or required_correct < 1
        or required_correct > len(selected)
    ):
        raise ESIKnowledgeError("required_correct must fit inside the attempt queue")
    if (
        isinstance(min_steps, bool)
        or not isinstance(min_steps, int)
        or min_steps < 1
        or min_steps > max_steps
    ):
        raise ESIKnowledgeError("min_steps must be between 1 and max_steps")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ESIKnowledgeError("threshold must be numeric")
    threshold = float(threshold)
    if not 0.0 < threshold <= 1.0:
        raise ESIKnowledgeError("threshold must be in (0, 1]")
    return {
        "schema": "roboharn_evo/esi_bench/source_collection_plan/v1",
        "primary_small_task": primary_task,
        "selection_policy": {
            "uses_source_split_only": True,
            "outcome_blind_order": "scene_group, room_group, instance_ref",
            "independence_key": "scene_group",
            "required_correct_trajectories": required_correct,
            "attempt_cap": len(selected),
            "consumed_queue_prefix": queue_offset,
            "stop_after_required_correct": True,
        },
        "episode_config": {
            "hpk_mode": "off",
            "model": "gpt-5.5",
            "reasoning_effort": "xhigh",
            "max_steps": max_steps,
            "min_steps": min_steps,
            "confidence_threshold": threshold,
            "automatic_retries": 0,
        },
        "maximum_budget": source_collection_budget(
            attempt_cap=len(selected),
            required_correct=required_correct,
            max_steps=max_steps,
        ),
        "ordered_instances": [
            {
                "queue_position": index,
                **item.to_dict(),
                "runner_task": _runner_task(item.metadata_path),
            }
            for index, item in enumerate(selected, 1)
        ],
        "transfer_boundary": (
            "scene, room, instance, metadata path, source answer, and simulator state "
            "are orchestration-only and cannot enter Task Knowledge"
        ),
    }


def main() -> int:
    args = _parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"source collection output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selected_task = _load_selected_task(args.selection, role=args.selection_role)
    manifest = load_split_manifest(args.split_manifest)
    selected = select_independent_source_instances(
        manifest.source,
        small_task=selected_task,
        attempt_cap=args.attempt_cap,
        queue_offset=args.queue_offset,
    )
    plan = build_collection_plan(
        primary_task=selected_task,
        selected=selected,
        required_correct=args.required_correct,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        threshold=args.threshold,
        queue_offset=args.queue_offset,
    )
    (output / "source_collection_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan["maximum_budget"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
