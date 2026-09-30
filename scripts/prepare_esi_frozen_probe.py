#!/usr/bin/env python3
"""Freeze an outcome-blind ESI development-state mechanism probe."""

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

PROBE_PROFILES = {
    "five_condition": (
        "off",
        "flat",
        "task_core",
        "task_full",
        "shuffled_full",
    ),
    "four_condition_exploratory": (
        "off",
        "flat",
        "task_core",
        "task_full",
    ),
}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ESIKnowledgeError("selection must be one object")
    return dict(value)


def selected_tasks(selection: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    values = selection.get("selections")
    if not isinstance(values, list):
        raise ESIKnowledgeError("selection has no candidate list")
    result = []
    for role in ("primary", "reserve"):
        matching = [
            item
            for item in values
            if isinstance(item, Mapping) and item.get("role") == role
        ]
        if len(matching) != 1:
            raise ESIKnowledgeError(f"selection must contain exactly one {role}")
        task = " ".join(str(matching[0].get("small_task") or "").strip().split())
        if not task:
            raise ESIKnowledgeError(f"selected {role} task is empty")
        result.append((role, task))
    return tuple(result)


def select_development_states(
    development: Sequence[ESISplitInstance],
    *,
    selected: Sequence[tuple[str, str]],
    per_task_count: int,
    excluded_instance_refs: Sequence[str] = (),
) -> tuple[tuple[str, ESISplitInstance], ...]:
    if (
        isinstance(per_task_count, bool)
        or not isinstance(per_task_count, int)
        or per_task_count < 1
    ):
        raise ESIKnowledgeError("per_task_count must be positive")
    excluded = {
        " ".join(str(value or "").strip().split())
        for value in excluded_instance_refs
        if str(value or "").strip()
    }
    chosen = []
    seen_scenes = set()
    for role, task in selected:
        normalized = " ".join(task.casefold().replace("_", " ").split())
        candidates = sorted(
            (
                item
                for item in development
                if " ".join(item.small_task.casefold().replace("_", " ").split())
                == normalized
                and item.instance_ref not in excluded
            ),
            key=lambda item: (
                item.scene_group.casefold(),
                item.room_group.casefold(),
                item.instance_ref,
            ),
        )
        task_states = []
        for item in candidates:
            scene = item.scene_group.casefold()
            if scene in seen_scenes:
                continue
            seen_scenes.add(scene)
            task_states.append((role, item))
            if len(task_states) == per_task_count:
                break
        if len(task_states) != per_task_count:
            raise ESIKnowledgeError(
                f"selected {role} task lacks independent development scenes"
            )
        chosen.extend(task_states)
    return tuple(chosen)


def excluded_refs_from_plans(paths: Sequence[Path]) -> tuple[str, ...]:
    refs = []
    for path in paths:
        value = _read_object(path)
        states = value.get("ordered_states")
        if value.get("split_part") != "development" or not isinstance(states, list):
            raise ESIKnowledgeError("excluded probe plan is not development-only")
        for state in states:
            if not isinstance(state, Mapping):
                raise ESIKnowledgeError("excluded probe plan contains an invalid state")
            ref = " ".join(str(state.get("instance_ref") or "").strip().split())
            if not ref:
                raise ESIKnowledgeError("excluded probe plan state has no instance_ref")
            refs.append(ref)
    if len(refs) != len(set(refs)):
        raise ESIKnowledgeError("excluded probe plans contain duplicate instance refs")
    return tuple(refs)


def _runner_task(metadata_path: str) -> str:
    value = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ESIKnowledgeError("development metadata must be one object")
    task = " ".join(
        str(value.get("runner_task") or value.get("task") or "").strip().split()
    )
    if not task:
        raise ESIKnowledgeError("development metadata has no runner task")
    return task


def frozen_probe_budget(
    state_count: int,
    *,
    probe_profile: str = "five_condition",
) -> dict[str, Any]:
    if (
        isinstance(state_count, bool)
        or not isinstance(state_count, int)
        or state_count < 1
    ):
        raise ESIKnowledgeError("state_count must be positive")
    try:
        conditions = PROBE_PROFILES[probe_profile]
    except KeyError as exc:
        raise ESIKnowledgeError("unknown frozen probe profile") from exc
    retrievals_per_state = 3 if "shuffled_full" in conditions else 2
    return {
        "capture_phase": {
            "simulator_episodes": state_count,
            "unique_images": state_count,
            "evaluated_model_calls": 0,
            "retrieval_model_calls": 0,
            "executed_actions": 0,
            "automatic_retries": 0,
        },
        "decision_phase": {
            "conditions": list(conditions),
            "annotation_calls": 1,
            "evaluated_model_calls": state_count * len(conditions),
            "retrieval_model_calls": state_count * retrievals_per_state,
            "total_model_calls": 1
            + state_count * (len(conditions) + retrievals_per_state),
            "model_image_payloads": state_count
            * (1 + len(conditions) + retrievals_per_state),
            "simulator_episodes": 0,
            "executed_actions": 0,
            "automatic_retries": 0,
        },
    }


def build_plan(
    states: Sequence[tuple[str, ESISplitInstance]],
    *,
    excluded_instance_refs: Sequence[str] = (),
    probe_profile: str = "five_condition",
) -> dict[str, Any]:
    return {
        "schema": "roboharn_evo/esi_bench/frozen_probe_plan/v2",
        "probe_profile": probe_profile,
        "split_part": "development",
        "selection_policy": {
            "outcome_blind": True,
            "ordered_by": "role, scene_group, room_group, instance_ref",
            "independence_key": "scene_group across all selected states",
            "answer_or_score_used": False,
            "excluded_prior_state_count": len(tuple(excluded_instance_refs)),
        },
        "excluded_instance_refs": list(excluded_instance_refs),
        "budgets": frozen_probe_budget(
            len(states),
            probe_profile=probe_profile,
        ),
        "ordered_states": [
            {
                "state_index": index,
                "selection_role": role,
                **item.to_dict(),
                "runner_task": _runner_task(item.metadata_path),
            }
            for index, (role, item) in enumerate(states, 1)
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze ESI development states before simulator capture."
    )
    task_source = parser.add_mutually_exclusive_group(required=True)
    task_source.add_argument("--selection", type=Path)
    task_source.add_argument(
        "--small-task",
        action="append",
        help="Repeat for an explicit, public-taxonomy balanced task set.",
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-task-count", type=int, default=6)
    parser.add_argument(
        "--probe-profile",
        choices=tuple(PROBE_PROFILES),
        default="five_condition",
    )
    parser.add_argument(
        "--exclude-plan",
        type=Path,
        action="append",
        default=[],
        help="Outcome-blind prior frozen plan whose instance refs cannot be reused.",
    )
    parser.add_argument(
        "--exclude-instance-ref",
        action="append",
        default=[],
        help="Repeat for a public development identity used outside a frozen plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"frozen probe plan output is not empty: {output}")
    manifest = load_split_manifest(args.split_manifest)
    excluded_refs = tuple(
        dict.fromkeys(
            (
                *excluded_refs_from_plans(args.exclude_plan),
                *(
                    " ".join(str(value or "").strip().split())
                    for value in args.exclude_instance_ref
                    if str(value or "").strip()
                ),
            )
        )
    )
    selected = (
        selected_tasks(_read_object(args.selection))
        if args.selection is not None
        else tuple(
            (f"task_{index}", task) for index, task in enumerate(args.small_task, 1)
        )
    )
    if len(selected) < 2:
        raise ESIKnowledgeError("frozen probe requires at least two balanced tasks")
    states = select_development_states(
        manifest.development,
        selected=selected,
        per_task_count=args.per_task_count,
        excluded_instance_refs=excluded_refs,
    )
    plan = build_plan(
        states,
        excluded_instance_refs=excluded_refs,
        probe_profile=args.probe_profile,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_probe_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan["budgets"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
