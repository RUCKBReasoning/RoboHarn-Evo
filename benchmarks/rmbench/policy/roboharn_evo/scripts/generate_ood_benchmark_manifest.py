#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FILENAME_RE = re.compile(
    r"(?P<prefix>.+?)_failure_ep(?P<episode_idx>\d+)_step(?P<annotated_step>\d+)_(?P<timestamp>\d{8}_\d{6})$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an overview OOD benchmark manifest from existing benchmark data.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--subset", type=str, default="all", choices=["all", "dev", "test"])
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def parse_sample_id(sample_id: str) -> dict[str, Any]:
    match = FILENAME_RE.match(sample_id)
    if not match:
        return {
            "parsed_prefix": sample_id,
            "episode_idx": None,
            "annotated_step": None,
            "timestamp": None,
        }
    return {
        "parsed_prefix": match.group("prefix"),
        "episode_idx": int(match.group("episode_idx")),
        "annotated_step": int(match.group("annotated_step")),
        "timestamp": match.group("timestamp"),
    }


def resolve_group_name(ood_json_path: Path) -> str:
    return ood_json_path.parent.name


def resolve_related_paths(data_root: Path, group_name: str, sample_id: str) -> dict[str, Path]:
    return {
        "action_json_path": data_root / "orig_action_data" / group_name / f"{sample_id}.json",
        "video_path_agent": data_root / "orig_video_data" / group_name / f"{sample_id}.mp4",
        "video_path_wrist": data_root / "orig_video_data" / group_name / f"{sample_id}_wrist_view.mp4",
        "image_dir_agent": data_root / "orig_image_data" / group_name / sample_id / "agent",
        "image_dir_wrist": data_root / "orig_image_data" / group_name / sample_id / "wrist",
    }


def maybe_path(path: Path) -> str | None:
    return str(path) if path.exists() else None


def build_key_frame_path(image_dir_agent: Path, annotated_step: int | None) -> str | None:
    if annotated_step is None:
        return None
    frame_path = image_dir_agent / f"frame_{annotated_step:06d}.jpg"
    return str(frame_path) if frame_path.exists() else None


def build_record(*, data_root: Path, ood_json_path: Path, strict: bool) -> tuple[dict[str, Any], dict[str, int]]:
    stats = {
        "missing_paths": 0,
    }
    ood_payload = load_json(ood_json_path)
    sample_id = ood_json_path.stem
    parsed = parse_sample_id(sample_id)
    group_name = resolve_group_name(ood_json_path)
    related_paths = resolve_related_paths(data_root, group_name, sample_id)

    action_json_path = related_paths["action_json_path"]
    if not action_json_path.exists() and strict:
        raise FileNotFoundError(f"Missing action JSON for sample {sample_id}: {action_json_path}")
    action_payload = load_json(action_json_path) if action_json_path.exists() else {}

    for key, path in related_paths.items():
        if not path.exists() and key in {"action_json_path", "video_path_agent", "image_dir_agent"}:
            stats["missing_paths"] += 1
            if strict:
                raise FileNotFoundError(f"Missing required path for sample {sample_id}: {path}")

    annotated_step = ood_payload.get("annotated_step", parsed["annotated_step"])
    try:
        annotated_step = int(annotated_step) if annotated_step is not None else None
    except Exception:
        annotated_step = parsed["annotated_step"]

    image_dir_agent = related_paths["image_dir_agent"]
    record = {
        "sample_id": sample_id,
        "model_name": ood_payload.get("model_name") or group_name.split("_libero_")[0],
        "task_suite_name": ood_payload.get("task_suite_name") or "_".join(group_name.split("_")[-2:]),
        "task_id": action_payload.get("task_id", ood_payload.get("task_id")),
        "task_description": action_payload.get("task_description", ood_payload.get("task_description", "")),
        "episode_idx": action_payload.get("episode_idx", ood_payload.get("episode_idx", parsed["episode_idx"])),
        "annotated_step": annotated_step,
        "timestamp": parsed["timestamp"],
        "error_category": ood_payload.get("error_category"),
        "target_error_category": ood_payload.get("target_error_category"),
        "s_stage": ood_payload.get("s_stage"),
        "d_stage": ood_payload.get("d_stage"),
        "video_path_agent": maybe_path(related_paths["video_path_agent"]),
        "video_path_wrist": maybe_path(related_paths["video_path_wrist"]),
        "image_dir_agent": maybe_path(related_paths["image_dir_agent"]),
        "image_dir_wrist": maybe_path(related_paths["image_dir_wrist"]),
        "key_frame_path": build_key_frame_path(image_dir_agent, annotated_step),
        "action_json_path": maybe_path(related_paths["action_json_path"]),
        "ood_json_path": str(ood_json_path),
        "success": action_payload.get("success"),
        "num_steps": action_payload.get("num_steps"),
        "initial_state": action_payload.get("initial_state", ood_payload.get("initial_state")),
        "has_wrist_view": related_paths["video_path_wrist"].exists() or related_paths["image_dir_wrist"].exists(),
        "preferred_input_mode": "video_snippet",
        "annotation_source": "manual_from_OOD_data",
        "notes": "",
    }
    return record, stats


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_path = args.output.resolve()

    ood_root = data_root / "OOD_data"
    ood_files = sorted(ood_root.rglob("*.json"))
    if args.limit > 0:
        ood_files = ood_files[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    missing_paths = 0
    with output_path.open("w", encoding="utf-8") as f:
        for ood_json_path in ood_files:
            record, stats = build_record(
                data_root=data_root,
                ood_json_path=ood_json_path,
                strict=args.strict,
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
            missing_paths += stats["missing_paths"]

    print(json.dumps(
        {
            "status": "ok",
            "subset": args.subset,
            "total_samples": total,
            "missing_paths": missing_paths,
            "output": str(output_path),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
