#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PLACEHOLDER_SIZE = (640, 480)
PANEL_PADDING = 16
LINE_SPACING = 8
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (0, 0, 0)
HEADER_COLOR = (220, 220, 220)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize OOD benchmark results with black text panels under images.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--only-failed", action="store_true")
    parser.add_argument("--font-size", type=int, default=22)
    parser.add_argument("--position", type=str, default="bottom", choices=["bottom", "right"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
        return records
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [item for item in payload["results"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported input format for {path}")


def should_include(record: dict[str, Any], only_failed: bool) -> bool:
    if not only_failed:
        return True
    if record.get("status") in {"error", "failed"}:
        return True
    if record.get("error"):
        return True
    if record.get("success") is False:
        return True
    return False


def get_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for ch in paragraph:
            candidate = current + ch
            width, _ = measure_text(draw, candidate, font)
            if current and width > max_width:
                lines.append(current)
                current = ch
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines or [""]


def build_text_lines(record: dict[str, Any]) -> list[tuple[str, tuple[int, int, int]]]:
    response = record.get("response") or {}
    scenario = response.get("OOD_scenario") or "N/A"
    reason = response.get("reason") or "N/A"
    lines: list[tuple[str, tuple[int, int, int]]] = []
    sample_id = record.get("sample_id")
    if sample_id:
        lines.append((f"sample: {sample_id}", HEADER_COLOR))
    task_description = record.get("task_description")
    if task_description:
        lines.append((f"task: {task_description}", HEADER_COLOR))
    lines.append((f"OOD scenario: {scenario}", TEXT_COLOR))
    lines.append((f"Reason: {reason}", TEXT_COLOR))
    status = record.get("status")
    error = record.get("error")
    if status and status != "ok":
        lines.append((f"status: {status}", TEXT_COLOR))
    if error:
        lines.append((f"error: {error}", TEXT_COLOR))
    return lines


def load_image(record: dict[str, Any]) -> tuple[Image.Image, bool, str | None]:
    key_frame_path = record.get("key_frame_path")
    if key_frame_path:
        path = Path(str(key_frame_path))
        if path.exists() and path.is_file():
            return Image.open(path).convert("RGB"), False, None
    image = Image.new("RGB", PLACEHOLDER_SIZE, PANEL_COLOR)
    return image, True, key_frame_path


def render_bottom_panel(image: Image.Image, record: dict[str, Any], font: ImageFont.ImageFont) -> Image.Image:
    temp = Image.new("RGB", (image.width, 10), PANEL_COLOR)
    draw = ImageDraw.Draw(temp)
    max_text_width = image.width - PANEL_PADDING * 2
    wrapped: list[tuple[str, tuple[int, int, int]]] = []
    for text, color in build_text_lines(record):
        for line in wrap_text(draw, text, font, max_text_width):
            wrapped.append((line, color))
    line_height = measure_text(draw, "Ag", font)[1]
    panel_height = PANEL_PADDING * 2 + len(wrapped) * line_height + max(0, len(wrapped) - 1) * LINE_SPACING
    output = Image.new("RGB", (image.width, image.height + panel_height), PANEL_COLOR)
    output.paste(image, (0, 0))
    panel = Image.new("RGB", (image.width, panel_height), PANEL_COLOR)
    output.paste(panel, (0, image.height))
    draw = ImageDraw.Draw(output)
    y = image.height + PANEL_PADDING
    for line, color in wrapped:
        draw.text((PANEL_PADDING, y), line, fill=color, font=font)
        y += line_height + LINE_SPACING
    return output


def add_missing_image_notice(record: dict[str, Any]) -> dict[str, Any]:
    clone = dict(record)
    response = dict(clone.get("response") or {})
    error = clone.get("error")
    key_frame_path = clone.get("key_frame_path")
    clone["error"] = error or f"Image not found: {key_frame_path}"
    clone["status"] = clone.get("status") or "error"
    clone["response"] = response
    return clone


def output_name(index: int, record: dict[str, Any]) -> str:
    sample_id = str(record.get("sample_id") or f"sample_{index}")
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in sample_id)
    return f"{index:05d}_{safe}.jpg"


def main() -> None:
    args = parse_args()
    records = load_records(args.input.resolve())
    if args.max_samples > 0:
        records = records[: args.max_samples]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    font = get_font(args.font_size)

    written = 0
    placeholders = 0
    for index, record in enumerate(records):
        if not should_include(record, args.only_failed):
            continue
        image, is_placeholder, _ = load_image(record)
        if is_placeholder:
            record = add_missing_image_notice(record)
            placeholders += 1
        if args.position != "bottom":
            raise NotImplementedError("Only bottom panel visualization is currently implemented.")
        output = render_bottom_panel(image, record, font)
        out_path = output_dir / output_name(index, record)
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_path}. Use --overwrite to replace it.")
        output.save(out_path)
        written += 1

    print(json.dumps(
        {
            "status": "ok",
            "input": str(args.input.resolve()),
            "output_dir": str(output_dir),
            "written": written,
            "placeholders": placeholders,
            "only_failed": bool(args.only_failed),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
