from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from roboharn_evo.agent.hpk.semantic_knowledge import validate_semantic_knowledge


def _annotation_chunks(
    path: Path,
    episode_key: str,
) -> list[tuple[str, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = payload.get(episode_key)
    if not isinstance(chunks, list):
        raise TypeError(f"missing language annotations for {episode_key}")
    start = 0
    result: list[tuple[str, int, int]] = []
    for item in chunks:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("language annotation contains an invalid chunk")
        instruction, length = item
        if not isinstance(instruction, str):
            raise TypeError("language annotation instruction must be text")
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError("language annotation length must be an integer")
        if not instruction.strip() or length <= 0:
            raise ValueError(
                "language annotation fields must be non-empty and positive"
            )
        end = start + length
        result.append((instruction.strip(), start, end))
        start = end
    if not result:
        raise ValueError(f"language annotations are empty for {episode_key}")
    return result


def _instruction_for_frame(
    chunks: list[tuple[str, int, int]],
    frame: int,
) -> tuple[str, int]:
    for instruction, start, end in chunks:
        if start <= frame < end:
            return instruction, start
    raise ValueError(f"frame {frame} is outside the language annotation ranges")


def _crossing(values: np.ndarray, *, start: int, closing: bool) -> int:
    for index in range(max(1, start), len(values)):
        before = float(values[index - 1])
        after = float(values[index])
        if closing and before >= 0.95 and after < 0.95:
            return index
        if not closing and before <= 0.05 and after > 0.05:
            return index
    raise ValueError("expected gripper transition is absent")


def _first_at(values: np.ndarray, *, start: int, predicate) -> int:
    for index in range(start, len(values)):
        if predicate(float(values[index])):
            return index
    raise ValueError("gripper transition did not complete")


def _quaternion_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-8 or second_norm <= 1e-8:
        raise ValueError("end-effector quaternion has near-zero norm")
    first = first / first_norm
    second = second / second_norm
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(cosine)))


def _first_gripper_close(handle: h5py.File) -> tuple[str, int]:
    candidates: list[tuple[int, str]] = []
    for arm in ("left", "right"):
        key = f"endpose/{arm}_gripper"
        if key not in handle:
            continue
        values = np.asarray(handle[key][:], dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"{key} must be a finite one-dimensional trajectory")
        try:
            candidates.append((_crossing(values, start=0, closing=True), arm))
        except ValueError:
            continue
    if not candidates:
        raise ValueError("trajectory contains no gripper closing transition")
    close_start, arm = min(candidates)
    return arm, close_start


def extract_candidate(
    trajectory_path: Path,
    annotation_path: Path,
    *,
    episode_key: str,
    task: str,
    object_category: str = "object",
    object_color: str = "unknown",
    object_shape: str = "unknown",
    object_role: str = "target",
) -> tuple[dict[str, Any], dict[str, Any]]:
    chunks = _annotation_chunks(annotation_path, episode_key)

    with h5py.File(trajectory_path, "r") as handle:
        arm, close_start = _first_gripper_close(handle)
        gripper = np.asarray(handle[f"endpose/{arm}_gripper"][:], dtype=np.float64)
        endpose = np.asarray(handle[f"endpose/{arm}_endpose"][:], dtype=np.float64)
    if endpose.ndim != 2 or endpose.shape[1] != 7:
        raise ValueError(f"endpose/{arm}_endpose must have shape [frames, 7]")
    if len(gripper) != len(endpose) or not np.all(np.isfinite(endpose)):
        raise ValueError(
            "gripper and end-effector trajectories must align and be finite"
        )

    instruction, annotated_start = _instruction_for_frame(chunks, close_start)
    close_end = _first_at(
        gripper, start=close_start, predicate=lambda value: value <= 0.05
    )
    release_start = _crossing(gripper, start=close_end + 1, closing=False)
    lift_frame = int(close_end + np.argmax(endpose[close_end:release_start, 2]))

    approach_start = max(annotated_start, close_start - 40)
    approach_peak = int(
        approach_start + np.argmax(endpose[approach_start : close_start + 1, 2])
    )
    approach_drop_m = float(endpose[approach_peak, 2] - endpose[close_start, 2])
    lift_m = float(endpose[lift_frame, 2] - endpose[close_end, 2])
    orientation_change_deg = _quaternion_angle_deg(
        endpose[close_end, 3:7], endpose[lift_frame, 3:7]
    )

    approach_direction = "from above" if approach_drop_m >= 0.03 else "unknown"
    orientation_relation = (
        "maintain" if orientation_change_deg <= 10.0 else "unconstrained"
    )
    object_semantics = {
        "category": object_category,
        "color": object_color,
        "shape": object_shape,
        "role": object_role,
        "held_state": "not held",
    }
    object_label = (
        f"{object_color} {object_category}"
        if object_color.strip().lower() != "unknown"
        else object_category
    )
    evidence = {
        "object": object_semantics,
        "action": f"grasp the {object_label} with the {arm} arm",
        "observed_result": (
            "the gripper closed and the end effector lifted while remaining closed; "
            "object motion was not recorded"
        ),
        "verdict": "unverified",
        "outcome_reasoning": {
            "verdict": "unverified",
            "reason": (
                "the expert motion is consistent with a grasp, but the trajectory "
                "does not record object attachment"
            ),
            "missing_evidence": "object pose or attachment state after gripper closure",
            "source": "RMBench GT end effector and gripper trajectory",
        },
    }
    knowledge = validate_semantic_knowledge(
        {
            "object": object_semantics,
            "condition": {"task": task, "phase": "grasp candidate"},
            "task_strategy": {
                "operation": "grasp",
                "preferred_arm": f"{arm} arm",
            },
            "geometric_strategy": {
                "approach": {
                    "direction": approach_direction,
                    "reference": "world frame",
                },
                "orientation": {
                    "relation": orientation_relation,
                    "reference": "world frame",
                    "order": "not applicable",
                },
                "grasp_region": "unknown",
            },
            "reasoning": {
                "observed_problem": "not available",
                "failure_analysis": "not available",
                "strategy_rationale": (
                    "the expert demonstration approached from above and kept its "
                    "wrist orientation during lift"
                ),
                "causal_hypothesis": (
                    "closing around the target object before lifting may keep it attached"
                ),
                "expected_observation": (
                    "the target object moves with the lifted gripper"
                ),
                "failure_condition": (
                    "the target object remains at its original support after closure and lift"
                ),
                "source": "RMBench GT trajectory replay",
                "confidence": 0.25,
            },
            "expected_effect": {"object_attached": True},
            "evidence": [evidence],
            "statistics": {"support": 0, "oppose": 0, "unverified": 1},
            "status": "candidate",
        }
    )
    summary = {
        "source_trajectory": str(trajectory_path),
        "source_annotation": str(annotation_path),
        "episode": episode_key,
        "task": task,
        "object": object_semantics,
        "selected_instruction": instruction,
        "arm": arm,
        "close_start_frame": close_start,
        "close_end_frame": close_end,
        "lift_frame": lift_frame,
        "release_start_frame": release_start,
        "approach_drop_m": round(approach_drop_m, 6),
        "lift_m": round(lift_m, 6),
        "orientation_change_deg": round(orientation_change_deg, 4),
        "verdict": "unverified",
        "why_not_support": (
            "the HDF5 records robot motion but no object pose or attachment state"
        ),
    }
    return knowledge, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--language-annotation", type=Path, required=True)
    parser.add_argument("--episode-key", default="episode_0")
    parser.add_argument("--task", required=True)
    parser.add_argument("--object-category", default="object")
    parser.add_argument("--object-color", default="unknown")
    parser.add_argument("--object-shape", default="unknown")
    parser.add_argument("--object-role", default="target")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    knowledge, summary = extract_candidate(
        args.trajectory.expanduser().resolve(),
        args.language_annotation.expanduser().resolve(),
        episode_key=args.episode_key,
        task=args.task,
        object_category=args.object_category,
        object_color=args.object_color,
        object_shape=args.object_shape,
        object_role=args.object_role,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hpk.json").write_text(
        json.dumps(knowledge, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "hpk": str(output_dir / "hpk.json"),
                "summary": str(output_dir / "replay_summary.json"),
                "verdict": knowledge["evidence"][0]["verdict"],
                "status": knowledge["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
