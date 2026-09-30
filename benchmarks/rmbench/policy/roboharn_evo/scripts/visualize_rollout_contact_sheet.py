#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


CAMERAS = ("head", "left", "right")
IMPORTANT_EVENTS = {
    "episode_start",
    "episode_progress",
    "episode_interrupt",
    "episode_agent_terminal_failure",
    "episode_exception",
    "episode_end",
    "instruction_set",
    "observation_preprocess",
    "observation_preprocess_finalized",
    "scene_memory_update",
    "control_turn_start",
    "control_turn_result",
    "control_decision",
    "pure_tool_control_control_turn_budget_exhausted",
    "decision_check",
    "vla_request",
    "vla_response",
    "monitor_signal",
    "recovery_policy",
    "recovery_router",
    "recovery_dispatch",
    "operation_candidate_selected",
    "recovery_result",
    "rollout_video_finalize",
    "rollout_report_finalize",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PPT-friendly PNG contact sheet for one RMBench/RoboHarn-Evo rollout.")
    parser.add_argument("--input", type=Path, required=True, help="Eval run dir, rollout dir, events JSONL, trace JSONL, or mp4.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path. Defaults to <run>/rollout_contact_sheet.png.")
    parser.add_argument("--max-rows", type=int, default=24, help="Maximum rows in the contact sheet.")
    parser.add_argument("--step-stride", type=int, default=0, help="Use every Nth saved rollout step. 0 means event-aware sampling.")
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--thumb-height", type=int, default=240)
    parser.add_argument("--text-width", type=int, default=560)
    parser.add_argument("--title", default="RoboHarn-Evo Rollout Contact Sheet")
    parser.add_argument("--no-overlay-segmentation", action="store_true", help="Do not draw SAM bbox overlays on camera frames.")
    parser.add_argument("--max-video-frames", type=int, default=24, help="Rows to sample when input only has an mp4.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def find_run_dir(input_path: Path) -> Path:
    path = input_path.expanduser().resolve()
    if path.is_file():
        if path.name in {"events.jsonl", "meta.json"} and path.parent.name.startswith("episode_") and path.parent.name.endswith("_rollout"):
            return path.parent.parent
        return path.parent
    if path.name.startswith("episode_") and path.name.endswith("_rollout"):
        return path.parent
    return path


def find_rollout_dirs(run_dir: Path, input_path: Path) -> list[Path]:
    path = input_path.expanduser().resolve()
    if path.is_dir() and path.name.startswith("episode_") and path.name.endswith("_rollout"):
        return [path]
    return sorted(run_dir.glob("episode_*_rollout"))


def find_trace_files(run_dir: Path, input_path: Path) -> list[Path]:
    path = input_path.expanduser().resolve()
    if path.is_file() and path.suffix == ".jsonl":
        return [path]
    candidates: list[Path] = []
    candidates.extend(sorted(run_dir.glob("episode_*_agent_trace.jsonl")))
    candidates.extend(sorted(run_dir.glob("*agent_trace.jsonl")))
    candidates.extend(sorted(run_dir.glob("episode_*_rollout/events.jsonl")))
    candidates.extend(sorted(run_dir.glob("events.jsonl")))
    return dedupe_paths(candidates)


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return result


def get_font(size: int = 16, *, mono: bool = False) -> ImageFont.ImageFont:
    names = ("DejaVuSansMono.ttf", "DejaVuSans.ttf") if mono else ("DejaVuSans.ttf", "DejaVuSansMono.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def normalize_video_frame(frame: Any) -> Image.Image:
    try:
        import numpy as np

        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[-1] >= 4:
            array = array[..., :3]
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.dtype != np.uint8:
            max_value = float(np.nanmax(array)) if array.size else 1.0
            if max_value <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"cannot convert video frame: {exc}") from exc


def thumbnail_with_geometry(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, float, int, int]:
    thumb = ImageOps.contain(image, size, method=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (245, 245, 245))
    offset_x = (size[0] - thumb.width) // 2
    offset_y = (size[1] - thumb.height) // 2
    canvas.paste(thumb, (offset_x, offset_y))
    scale = thumb.width / image.width if image.width else 1.0
    return canvas, scale, offset_x, offset_y


def parse_step_from_name(path: Path) -> int | None:
    match = re.search(r"step_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    match = re.match(r"(\d+)_", path.stem)
    if match:
        return int(match.group(1))
    return None


def frame_rows_from_rollout(rollout_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rollout in rollout_dirs:
        head_paths = sorted((rollout / "head").glob("step_*.png"))
        for head_path in head_paths:
            step = parse_step_from_name(head_path)
            if step is None:
                continue
            frames = {"head": head_path}
            for camera in ("left", "right"):
                candidate = rollout / camera / f"step_{step:06d}.png"
                if candidate.exists():
                    frames[camera] = candidate
            rows.append({"kind": "rollout_frames", "step": step, "frames": frames, "source": rollout})
    return sorted(rows, key=lambda item: int(item["step"]))


def frame_rows_from_event_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        frames = record.get("frames")
        if not isinstance(frames, dict):
            continue
        normalized: dict[str, Path] = {}
        for camera in CAMERAS:
            raw_path = frames.get(camera)
            if not raw_path:
                continue
            path = Path(str(raw_path)).expanduser()
            if path.exists():
                normalized[camera] = path.resolve()
        if not normalized:
            continue
        summary = record.get("snapshot_summary") if isinstance(record.get("snapshot_summary"), dict) else {}
        step = record.get("env_step", record.get("step", summary.get("step_count", record.get("index", index))))
        try:
            step_int = int(step)
        except Exception:
            step_int = index
        rows.append({"kind": "event_frames", "step": step_int, "frames": normalized, "record": record, "source": "events"})
    return rows


def sample_video_rows(video_path: Path, max_rows: int) -> list[dict[str, Any]]:
    try:
        import imageio.v3 as iio

        frames = []
        for index, frame in enumerate(iio.imiter(video_path)):
            frames.append((index, frame))
    except Exception as exc:
        raise RuntimeError(f"cannot read video {video_path}: {exc}") from exc
    if not frames:
        return []
    selected_indexes = sample_indexes(len(frames), max_rows)
    rows: list[dict[str, Any]] = []
    for output_index, frame_index in enumerate(selected_indexes):
        source_index, frame = frames[frame_index]
        rows.append(
            {
                "kind": "video_frame",
                "step": source_index,
                "video_frame": normalize_video_frame(frame),
                "record": {"event": "video_frame", "index": output_index, "frame": source_index},
                "source": video_path,
            }
        )
    return rows


def event_only_rows(records: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = [record for record in records if record.get("event") in IMPORTANT_EVENTS] or records
    for index in sample_indexes(len(selected), max_rows):
        record = selected[index]
        step = event_step(record)
        rows.append(
            {
                "kind": "event_only",
                "step": int(step) if step is not None else index,
                "record": record,
                "source": "trace",
            }
        )
    return rows


def sample_indexes(length: int, max_items: int) -> list[int]:
    if length <= max_items:
        return list(range(length))
    if max_items <= 1:
        return [0]
    return sorted({round(i * (length - 1) / (max_items - 1)) for i in range(max_items)})


def event_step(record: dict[str, Any]) -> int | None:
    raw = record.get("env_step", record.get("step"))
    if raw is None and isinstance(record.get("snapshot_summary"), dict):
        raw = record["snapshot_summary"].get("step_count")
    try:
        return int(raw)
    except Exception:
        return None


def choose_rows(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    max_rows: int,
    step_stride: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    if rows[0].get("kind") in {"event_frames", "video_frame"}:
        indexes = sample_indexes(len(rows), max_rows)
        return [rows[index] for index in indexes]
    if step_stride > 0:
        selected = [row for idx, row in enumerate(rows) if idx % step_stride == 0]
        if rows[-1] not in selected:
            selected.append(rows[-1])
        return selected[:max_rows]

    step_to_row = {int(row["step"]): row for row in rows}
    available_steps = sorted(step_to_row)
    wanted_steps: list[int] = []
    if available_steps:
        wanted_steps.extend([available_steps[0], available_steps[-1]])
    for record in records:
        if record.get("event") not in IMPORTANT_EVENTS:
            continue
        step = event_step(record)
        if step is None:
            continue
        wanted_steps.append(nearest_step(available_steps, step))

    selected_steps: list[int] = []
    for step in sorted(set(wanted_steps)):
        if step not in selected_steps:
            selected_steps.append(step)

    target_uniform = max_rows - len(selected_steps)
    if target_uniform > 0:
        for index in sample_indexes(len(available_steps), target_uniform):
            selected_steps.append(available_steps[index])
    selected_steps = sorted(set(selected_steps))

    if len(selected_steps) > max_rows:
        selected_steps = reduce_preserving_edges(selected_steps, max_rows)
    return [step_to_row[step] for step in selected_steps]


def nearest_step(steps: list[int], target: int) -> int:
    if not steps:
        return target
    best = min(steps, key=lambda step: abs(step - target))
    return int(best)


def reduce_preserving_edges(values: list[int], max_items: int) -> list[int]:
    if len(values) <= max_items:
        return values
    if max_items <= 2:
        return values[:max_items]
    middle = values[1:-1]
    indexes = sample_indexes(len(middle), max_items - 2)
    return [values[0], *[middle[index] for index in indexes], values[-1]]


def compact_value(value: Any, max_chars: int = 180) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = " ".join(text.replace("\n", " ").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def summarize_segmentation(item: dict[str, Any]) -> str:
    object_id = str(item.get("object_id", "object"))
    if not item.get("success"):
        error = item.get("error")
        detections = item.get("num_detections")
        suffix = f", detections={detections}" if detections is not None else ""
        return f"SAM {object_id}: not found{suffix}" + (f", error={compact_value(error, 80)}" if error else "")
    bbox = item.get("bbox_xyxy")
    centroid = item.get("centroid_px")
    score = item.get("score")
    parts = [f"SAM {object_id}: ok"]
    if bbox is not None:
        parts.append(f"bbox={compact_value(bbox, 64)}")
    if centroid is not None:
        parts.append(f"center={compact_value(centroid, 48)}")
    if score is not None:
        try:
            parts.append(f"score={float(score):.3f}")
        except Exception:
            parts.append(f"score={score}")
    grounding = item.get("grounding_3d")
    if isinstance(grounding, dict):
        if grounding.get("success"):
            world = grounding.get("centroid_world")
            approach = grounding.get("approach_point_world")
            parts.append(f"world={compact_value(world, 64)}")
            if approach is not None:
                parts.append(f"approach={compact_value(approach, 64)}")
        elif grounding.get("error"):
            parts.append(f"3d={compact_value(grounding.get('error'), 80)}")
    return ", ".join(parts)


def summarize_event(record: dict[str, Any]) -> list[str]:
    event = str(record.get("event", "unknown"))
    if event == "episode_start":
        return [f"event: episode_start", f"task: {record.get('task_name', '')}", f"instruction: {compact_value(record.get('instruction', ''), 170)}"]
    if event == "episode_progress":
        return [
            f"event: episode_progress",
            f"step: {record.get('step', record.get('env_step', ''))} / {record.get('step_limit', '')}",
            f"reward: {record.get('max_reward')}  success: {record.get('eval_success')}",
            f"agent: {compact_value({k: record.get(k) for k in ('active_skill', 'monitor_phase', 'monitor_status', 'recovery_pending')}, 180)}",
        ]
    if event == "episode_end":
        return [
            f"event: episode_end  result: {record.get('result')}",
            f"success: {record.get('success')}  steps: {record.get('total_steps')}  reward: {record.get('max_reward')}",
            f"failure: {compact_value(record.get('failure_reason', ''), 160)}",
        ]
    if event == "instruction_set":
        return [f"event: instruction_set", f"instruction: {compact_value(record.get('instruction', ''), 220)}"]
    if event in {"observation_preprocess", "observation_preprocess_finalized"}:
        lines = [f"event: observation_preprocess", f"latency: {record.get('latency_sec', '')}s"]
        segments = record.get("segmentation")
        if isinstance(segments, list) and segments:
            for item in segments[:4]:
                if isinstance(item, dict):
                    lines.append(summarize_segmentation(item))
        else:
            lines.append("segmentation: empty")
        scene_memory = record.get("scene_memory")
        if isinstance(scene_memory, dict) and scene_memory.get("summary"):
            lines.append(f"scene: {compact_value(scene_memory.get('summary'), 220)}")
        return lines
    if event == "scene_memory_update":
        scene_memory = record.get("scene_memory") if isinstance(record.get("scene_memory"), dict) else {}
        lines = ["event: scene_memory_update"]
        if scene_memory.get("summary"):
            lines.append(f"summary: {compact_value(scene_memory.get('summary'), 260)}")
        task_focus = scene_memory.get("task_focus")
        if isinstance(task_focus, dict):
            lines.append(f"focus: {compact_value(task_focus, 220)}")
        instances = scene_memory.get("instances")
        if isinstance(instances, list):
            names = [item.get("instance_id") for item in instances[:6] if isinstance(item, dict)]
            lines.append(f"instances: {compact_value(names, 180)}")
        return lines
    if event == "operation_candidate_selected":
        return [
            "event: operation_candidate_selected",
            (
                f"candidate: {record.get('candidate_id', '')}  "
                f"revision: {record.get('candidate_geometry_revision', '')}"
            ),
            (
                f"instance: {record.get('instance_id', '')}  "
                f"target: {record.get('target_id', '')}"
            ),
            (
                f"arm/mode/camera: {record.get('arm', '')} / "
                f"{record.get('action_mode', '')} / "
                f"{record.get('source_camera', '')}"
            ),
            f"poses: {compact_value({key: record.get(key) for key in ('object_contact_pose', 'tcp_pose', 'ee_target_pose', 'approach_pose')}, 260)}",
        ]
    if event in {"control_turn_start", "decision_check"}:
        return [f"event: {event}", f"trigger: {record.get('trigger', record.get('need_decision', ''))}", f"status: {compact_value(record, 180)}"]
    if event in {"control_turn_result", "control_decision"}:
        return [
            f"event: {event}",
            f"mode: {record.get('action_mode', '')}  skill: {record.get('selected_skill', '')}",
            f"subtask: {compact_value(record.get('subtask_text', record.get('rendered_instruction', '')), 210)}",
            f"trigger: {record.get('trigger', '')}",
        ]
    if event == "monitor_signal":
        return [
            "event: monitor_signal",
            f"signal: {record.get('signal', '')}  score: {record.get('progress_score', '')}",
            f"reason: {compact_value(record.get('reason', ''), 220)}",
        ]
    if event in {"recovery_policy", "recovery_router", "recovery_dispatch", "recovery_result"}:
        fields = {
            key: record.get(key)
            for key in ("action", "signal_name", "reason", "workflow", "plan", "success", "message", "tool_name")
            if key in record
        }
        return [f"event: {event}", compact_value(fields, 240)]
    if event == "vla_request":
        return ["event: vla_request", f"subtask: {compact_value(record.get('subtask', ''), 220)}"]
    if event == "vla_response":
        return ["event: vla_response", f"latency: {record.get('latency_sec', '')}s  shape: {record.get('action_chunk_shape', '')}"]
    if event == "recovery_tool":
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        summary = record.get("snapshot_summary") if isinstance(record.get("snapshot_summary"), dict) else {}
        lines = [
            f"event: recovery_tool  tool: {record.get('tool_name', '-')}",
            f"args: {compact_value(record.get('args', {}), 180)}",
            f"success: {result.get('success')}  message: {compact_value(result.get('message', ''), 160)}",
            f"step: {summary.get('step_count')} / {summary.get('step_limit')}  reward: {summary.get('max_reward')}",
            f"check_success: {summary.get('check_success')}",
        ]
        progress = record.get("progress")
        if progress:
            lines.append(f"progress: {compact_value(progress, 210)}")
        return lines
    if event == "initial":
        summary = record.get("snapshot_summary") if isinstance(record.get("snapshot_summary"), dict) else {}
        return [
            "event: initial",
            f"step: {summary.get('step_count')} / {summary.get('step_limit')}  reward: {summary.get('max_reward')}",
            f"check_success: {summary.get('check_success')}",
        ]
    return [f"event: {event}", compact_value({k: v for k, v in record.items() if k != "timestamp"}, 260)]


def records_in_window(records: list[dict[str, Any]], previous_step: int | None, current_step: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        step = event_step(record)
        if step is None:
            continue
        if previous_step is None:
            if step <= current_step:
                selected.append(record)
        elif previous_step < step <= current_step:
            selected.append(record)
    return selected


def row_records(row: dict[str, Any], records: list[dict[str, Any]], previous_step: int | None) -> list[dict[str, Any]]:
    if isinstance(row.get("record"), dict):
        return [row["record"]]
    current_step = int(row.get("step", 0))
    selected = records_in_window(records, previous_step, current_step)
    important = [record for record in selected if record.get("event") in IMPORTANT_EVENTS]
    return important or selected[-3:]


def parse_bbox(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    except Exception:
        return None


def segmentation_boxes(records: list[dict[str, Any]], camera: str) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[float, float, float, float]]] = set()
    for record in records:
        if record.get("event") not in {"observation_preprocess", "observation_preprocess_finalized"}:
            continue
        segments = record.get("segmentation")
        if not isinstance(segments, list):
            continue
        for item in segments:
            if not isinstance(item, dict) or not item.get("success"):
                continue
            item_camera = str(item.get("camera", "head") or "head")
            if item_camera != camera:
                continue
            candidates = []
            detections = item.get("detections")
            if isinstance(detections, list):
                candidates.extend(d for d in detections if isinstance(d, dict))
            if not candidates:
                candidates.append(item)
            for candidate in candidates:
                bbox = parse_bbox(candidate.get("bbox_xyxy"))
                if bbox is None:
                    continue
                label = "SAM " + str(item.get("object_id", "object"))
                key = (label, tuple(round(value, 2) for value in bbox))
                if key in seen:
                    continue
                seen.add(key)
                boxes.append(
                    {
                        "bbox": bbox,
                        "label": label,
                        "score": candidate.get("score", item.get("score")),
                        "style": "candidate",
                    }
                )
                if len(boxes) >= 12:
                    return boxes
    return boxes


def scene_memory_focus_boxes(records: list[dict[str, Any]], camera: str) -> list[dict[str, Any]]:
    scene_memory = latest_scene_memory(records)
    if not scene_memory:
        return []
    instances = scene_memory.get("instances")
    if not isinstance(instances, list):
        return []
    instance_by_id = {
        str(item.get("instance_id")): item
        for item in instances
        if isinstance(item, dict) and item.get("instance_id")
    }
    focus = scene_memory.get("task_focus")
    if not isinstance(focus, dict):
        focus = {}
    boxes: list[dict[str, Any]] = []
    for role_key, label_role in (("target_instances", "target"), ("tool_instances", "tool")):
        raw_ids = focus.get(role_key)
        if not isinstance(raw_ids, list):
            continue
        for instance_id in raw_ids:
            instance = instance_by_id.get(str(instance_id))
            if not isinstance(instance, dict):
                continue
            item_camera = str(instance.get("camera", "head") or "head")
            if item_camera != camera:
                continue
            bbox = parse_bbox(instance.get("bbox_xyxy"))
            if bbox is None:
                continue
            boxes.append(
                {
                    "bbox": bbox,
                    "label": f"SELECTED {label_role}:{instance.get('instance_id')}",
                    "score": instance.get("score"),
                    "style": "selected_target" if label_role == "target" else "selected_tool",
                }
            )
    return boxes


def latest_scene_memory(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in reversed(records):
        scene_memory = record.get("scene_memory")
        if isinstance(scene_memory, dict) and scene_memory:
            return scene_memory
        if record.get("event") == "scene_memory_update":
            scene_memory = record.get("scene_memory")
            if isinstance(scene_memory, dict):
                return scene_memory
    return {}


def draw_bbox_overlay(
    image: Image.Image,
    *,
    boxes: list[dict[str, Any]],
    scale: float,
    offset_x: int,
    offset_y: int,
    font: ImageFont.ImageFont,
) -> None:
    if not boxes:
        return
    draw = ImageDraw.Draw(image)
    colors = [(0, 220, 110), (255, 60, 60), (50, 120, 255), (255, 180, 30)]
    for index, box in enumerate(boxes):
        bbox = box["bbox"]
        style = str(box.get("style", "candidate"))
        if style == "selected_target":
            color = (0, 245, 255)
        elif style == "selected_tool":
            color = (255, 70, 220)
        else:
            color = colors[index % len(colors)]
        x1 = int(round(bbox[0] * scale + offset_x))
        y1 = int(round(bbox[1] * scale + offset_y))
        x2 = int(round(bbox[2] * scale + offset_x))
        y2 = int(round(bbox[3] * scale + offset_y))
        stroke = 4 if style.startswith("selected") else 2
        for inset in range(stroke):
            draw.rectangle((x1 - inset, y1 - inset, x2 + inset, y2 + inset), outline=color)
        label = str(box.get("label", "object"))
        score = box.get("score")
        if score is not None:
            try:
                label = f"{label} {float(score):.2f}"
            except Exception:
                label = f"{label} {score}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle((x1, max(0, y1 - 20), x1 + text_bbox[2] - text_bbox[0] + 8, y1), fill=(0, 0, 0))
        draw.text((x1 + 4, max(0, y1 - 19)), label, fill=color, font=font)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if current and bbox[2] - bbox[0] > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
            while current and draw.textbbox((0, 0), current, font=font)[2] > max_width:
                split_at = max(1, math.floor(len(current) * max_width / max(1, draw.textbbox((0, 0), current, font=font)[2])))
                lines.append(current[:split_at])
                current = current[split_at:]
        lines.append(current)
    return lines or [""]


def draw_camera_label(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, label: str, font: ImageFont.ImageFont) -> None:
    draw.rectangle((x, y, x + width, y + 24), fill=(0, 0, 0))
    draw.text((x + 8, y + 4), label, fill=(230, 230, 230), font=font)


def draw_text_panel(
    sheet: Image.Image,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    row_index: int,
    step: int,
    records: list[dict[str, Any]],
    font: ImageFont.ImageFont,
    header_font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((x, y, x + width, y + height), fill=(8, 10, 14))
    text_x = x + 12
    text_y = y + 8
    draw.text((text_x, text_y), f"#{row_index} step/frame: {step}", fill=(255, 255, 255), font=header_font)
    text_y += 26
    lines: list[str] = []
    for record in records[:5]:
        lines.extend(summarize_event(record))
    if not lines:
        lines = ["event: -"]
    max_y = y + height - 8
    for line in lines:
        for wrapped in wrap_text(draw, line, font, width - 24):
            if text_y + 18 > max_y:
                draw.text((text_x, text_y), "...", fill=(210, 210, 210), font=font)
                return
            draw.text((text_x, text_y), wrapped, fill=(235, 238, 242), font=font)
            text_y += 18


def draw_title(sheet: Image.Image, title: str, subtitle: str, height: int) -> None:
    draw = ImageDraw.Draw(sheet)
    title_font = get_font(24)
    sub_font = get_font(15)
    draw.rectangle((0, 0, sheet.width, height), fill=(23, 31, 42))
    draw.text((14, 10), title, fill=(255, 255, 255), font=title_font)
    draw.text((14, 42), subtitle, fill=(210, 218, 228), font=sub_font)


def build_contact_sheet(
    *,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    output_path: Path,
    title: str,
    source: Path,
    thumb_size: tuple[int, int],
    text_width: int,
    overlay_segmentation: bool,
) -> None:
    if not rows:
        raise RuntimeError("no frames found for contact sheet")
    row_h = int(thumb_size[1])
    title_h = 70
    width = thumb_size[0] * 3 + text_width
    height = title_h + row_h * len(rows)
    sheet = Image.new("RGB", (width, height), (0, 0, 0))
    draw_title(sheet, title, f"source: {source}", title_h)
    label_font = get_font(14)
    text_font = get_font(15, mono=True)
    header_font = get_font(17, mono=True)

    previous_step: int | None = None
    for row_index, row in enumerate(rows):
        y = title_h + row_index * row_h
        step = int(row.get("step", row_index))
        records_for_row = row_records(row, records, previous_step)
        if row.get("kind") == "video_frame":
            frame = row["video_frame"]
            thumb, _, _, _ = thumbnail_with_geometry(frame, (thumb_size[0] * 3, thumb_size[1]))
            sheet.paste(thumb, (0, y))
            draw = ImageDraw.Draw(sheet)
            draw_camera_label(draw, 0, y, thumb_size[0] * 3, "episode video", label_font)
        else:
            frames = row.get("frames", {})
            for col, camera in enumerate(CAMERAS):
                x = col * thumb_size[0]
                path = frames.get(camera)
                if path:
                    image = open_rgb(Path(path))
                    thumb, scale, offset_x, offset_y = thumbnail_with_geometry(image, thumb_size)
                    if overlay_segmentation:
                        boxes = segmentation_boxes(records_for_row, camera)
                        boxes.extend(scene_memory_focus_boxes(records_for_row, camera))
                        draw_bbox_overlay(
                            thumb,
                            boxes=boxes,
                            scale=scale,
                            offset_x=offset_x,
                            offset_y=offset_y,
                            font=label_font,
                        )
                else:
                    thumb = Image.new("RGB", thumb_size, (238, 238, 238))
                    ImageDraw.Draw(thumb).text((12, 12), "missing", fill=(80, 80, 80), font=label_font)
                sheet.paste(thumb, (x, y))
                draw = ImageDraw.Draw(sheet)
                draw_camera_label(draw, x, y, thumb_size[0], camera, label_font)
        draw_text_panel(
            sheet,
            x=thumb_size[0] * 3,
            y=y,
            width=text_width,
            height=row_h,
            row_index=row_index,
            step=step,
            records=records_for_row,
            font=text_font,
            header_font=header_font,
        )
        previous_step = step

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def find_video(run_dir: Path, input_path: Path) -> Path | None:
    path = input_path.expanduser().resolve()
    if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        return path
    candidates = sorted(run_dir.glob("episode*.mp4"))
    if candidates:
        return candidates[0]
    candidates = sorted(run_dir.glob("episode_*_rollout/video/head.mp4"))
    return candidates[0] if candidates else None


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    run_dir = find_run_dir(input_path)
    rollout_dirs = find_rollout_dirs(run_dir, input_path)
    trace_files = find_trace_files(run_dir, input_path)
    records: list[dict[str, Any]] = []
    for path in trace_files:
        records.extend(load_jsonl(path))

    rows = frame_rows_from_event_records(records)
    if not rows:
        rows = frame_rows_from_rollout(rollout_dirs)
        rows = choose_rows(rows, records, max_rows=max(1, int(args.max_rows)), step_stride=max(0, int(args.step_stride)))
    else:
        rows = choose_rows(rows, records, max_rows=max(1, int(args.max_rows)), step_stride=0)
    if not rows:
        video = find_video(run_dir, input_path)
        if video is not None:
            try:
                rows = sample_video_rows(video, max(1, int(args.max_video_frames)))
            except Exception as exc:
                print(f"[warn] skipping unreadable video {video}: {exc}", flush=True)
    if not rows and records:
        rows = event_only_rows(records, max(1, int(args.max_rows)))

    output_path = args.output.expanduser().resolve() if args.output else run_dir / "rollout_contact_sheet.png"
    build_contact_sheet(
        rows=rows,
        records=records,
        output_path=output_path,
        title=str(args.title),
        source=input_path,
        thumb_size=(max(120, int(args.thumb_width)), max(90, int(args.thumb_height))),
        text_width=max(260, int(args.text_width)),
        overlay_segmentation=not bool(args.no_overlay_segmentation),
    )
    print(
        json.dumps(
            {
                "contact_sheet": str(output_path),
                "rows": len(rows),
                "trace_files": [str(path) for path in trace_files],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
