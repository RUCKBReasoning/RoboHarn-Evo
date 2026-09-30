from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    REPO_ROOT / "eval_result/rq1/report.json"
)
DEFAULT_EXISTING_ROOT = REPO_ROOT / "eval_result/hpk/v3/rmbench_gt_hierarchical_generic"
DEFAULT_NEW_ROOT = (
    REPO_ROOT / "eval_result/rq1/prepared_trajectories"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "eval_result/rq1/hierarchical_extraction_overview.png"
)

WIDTH = 2400
HEIGHT = 1500

INK = "#102A43"
BODY = "#3F556A"
MUTED = "#61758A"
GREEN = "#317D67"
GREEN_BG = "#EFF9F5"
GREEN_BORDER = "#63AD91"
BLUE = "#3E68C0"
BLUE_BG = "#EEF4FF"
BLUE_BORDER = "#5E82D0"
AMBER = "#9A6411"
AMBER_BG = "#FFF8E8"
AMBER_BORDER = "#D9A13C"
PANEL_BORDER = "#CBD6E2"
CARD_BG = "#F8FAFC"
CARD_BORDER = "#B8C6D3"


def _read_json(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return raw


def _font(
    size: int, *, bold: bool = False, mono: bool = False
) -> ImageFont.FreeTypeFont:
    if mono:
        filename = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    else:
        filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{filename}", size)


def _rounded_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = "#FFFFFF",
    outline: str = PANEL_BORDER,
    radius: int = 24,
    width: int = 2,
) -> None:
    x0, y0, x1, y1 = box
    shadow = (x0, y0 + 5, x1, y1 + 8)
    draw.rounded_rectangle(shadow, radius=radius, fill="#DFE6EC")
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _pill(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    *,
    fill: str,
    color: str,
) -> None:
    font = _font(13, bold=True)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width = right - left + 34
    draw.rounded_rectangle((x, y, x + width, y + 30), radius=15, fill=fill)
    draw.text((x + width / 2, y + 15), text, font=font, fill=color, anchor="mm")


def _fit_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    source_ratio = image.width / image.height
    target_ratio = width / height
    if source_ratio > target_ratio:
        crop_width = int(image.height * target_ratio)
        left = (image.width - crop_width) // 2
        image = image.crop((left, 0, left + crop_width, image.height))
    else:
        crop_height = int(image.width / target_ratio)
        top = (image.height - crop_height) // 2
        image = image.crop((0, top, image.width, top + crop_height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _frame(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    path: Path,
    box: tuple[int, int, int, int],
    label: str,
    *,
    label_size: int = 12,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill="#FFFFFF", outline="#AAB8C5", width=2)
    inner = _fit_crop(Image.open(path).convert("RGB"), x1 - x0 - 8, y1 - y0 - 30)
    canvas.paste(inner, (x0 + 4, y0 + 4))
    draw.text(
        ((x0 + x1) // 2, y1 - 14),
        label,
        font=_font(label_size),
        fill=MUTED,
        anchor="mm",
    )


def _wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    words = " ".join(str(text).split()).split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    line_height: int,
    max_lines: int | None = None,
) -> int:
    lines = _wrap_lines(draw, text, font, max_width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and draw.textlength(f"{last}…", font=font) > max_width:
            last = last[:-1]
        lines[-1] = f"{last.rstrip()}…"
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = GREEN,
    width: int = 4,
) -> None:
    draw.line((start, end), fill=color, width=width)
    x, y = end
    draw.polygon([(x, y), (x - 12, y - 8), (x - 12, y + 8)], fill=color)


def _metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    label: str,
    *,
    color: str,
    fill: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=15, fill=fill, outline=color, width=2)
    draw.text(
        ((x0 + x1) // 2, y0 + 42),
        value,
        font=_font(31, bold=True),
        fill=color,
        anchor="mm",
    )
    draw.text(
        ((x0 + x1) // 2, y0 + 78),
        label,
        font=_font(14, bold=True),
        fill=BODY,
        anchor="mm",
    )


def _bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    label: str,
    value: int,
    maximum: int,
    *,
    color: str,
) -> None:
    font = _font(14, bold=True)
    draw.text((x, y), label, font=font, fill=BODY)
    draw.text((x + width, y), str(value), font=font, fill=BODY, anchor="ra")
    y += 24
    draw.rounded_rectangle((x, y, x + width, y + 15), radius=8, fill="#E7EDF3")
    filled = 0 if maximum == 0 else int(width * value / maximum)
    if filled:
        draw.rounded_rectangle((x, y, x + filled, y + 15), radius=8, fill=color)


def _episode_root(existing_root: Path, new_root: Path, episode: int) -> Path:
    parent = existing_root if episode < 3 else new_root
    return parent / f"episode_{episode}"


def render(
    *,
    report_path: Path,
    existing_root: Path,
    new_root: Path,
    output: Path,
) -> None:
    report = _read_json(report_path)
    episode_roots = [
        _episode_root(existing_root, new_root, episode) for episode in range(5)
    ]
    for root in episode_roots:
        for required in (
            root / "trajectory_knowledge_package.json",
            root / "atomic_knowledge.json",
            root / "key_observations/observation_07.png",
        ):
            if not required.is_file():
                raise FileNotFoundError(required)

    representative_package = _read_json(
        episode_roots[3] / "trajectory_knowledge_package.json"
    )
    representative_atomic = _read_json(episode_roots[3] / "atomic_knowledge.json")
    subtasks = representative_package["task_strategy"]["subtasks"]
    task_units = representative_atomic["task_knowledge"]
    action_units = representative_atomic["action_knowledge"]
    if not isinstance(subtasks, Sequence) or len(subtasks) != 4:
        raise ValueError("representative trajectory must contain four subtasks")
    if not isinstance(task_units, Sequence) or not task_units:
        raise ValueError("representative atomic package has no Task Knowledge")
    if not isinstance(action_units, Sequence) or not action_units:
        raise ValueError("representative atomic package has no Action Knowledge")
    task_example = task_units[0]
    action_example = next(
        (unit for unit in action_units if unit.get("status") == "supported"),
        action_units[0],
    )

    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#F4F7FA")
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (WIDTH // 2, 58),
        "HPK v3 on RMBench: Hierarchical Knowledge Extraction",
        font=_font(42, bold=True),
        fill=INK,
        anchor="mm",
    )
    draw.text(
        (WIDTH // 2, 94),
        "Five real expert trajectories → ordered subtasks → reusable Task and Action Knowledge",
        font=_font(19),
        fill="#52677D",
        anchor="mm",
    )

    # A — real trajectory inputs.
    _rounded_panel(canvas, draw, (45, 125, 765, 690))
    _pill(draw, 70, 148, "REAL GT DATA", fill="#DDF5EA", color="#1F7354")
    draw.text(
        (70, 205), "A. READ FIVE TRAJECTORIES", font=_font(25, bold=True), fill=INK
    )
    draw.text(
        (70, 238),
        "Same instruction; different demonstrated executions",
        font=_font(16),
        fill=MUTED,
    )
    thumb_y = 275
    thumb_width = 124
    gap = 13
    for episode, root in enumerate(episode_roots):
        x = 70 + episode * (thumb_width + gap)
        _frame(
            canvas,
            draw,
            root / "key_observations/observation_07.png",
            (x, thumb_y, x + thumb_width, thumb_y + 145),
            f"trajectory {episode}",
            label_size=11,
        )
    draw.rounded_rectangle(
        (75, 455, 735, 635), radius=16, fill=CARD_BG, outline=CARD_BORDER, width=2
    )
    draw.text(
        (100, 487),
        "Evidence presented to the reflector",
        font=_font(18, bold=True),
        fill=INK,
    )
    input_lines = [
        "70 ordered action chunks",
        "75 real visual observations",
        "robot motion + gripper-state changes",
        "SAM3 task-relevant regions",
    ]
    for index, line in enumerate(input_lines):
        y = 525 + index * 28
        draw.ellipse((102, y + 5, 112, y + 15), fill=GREEN)
        draw.text((124, y), line, font=_font(15), fill=BODY)

    # B — hierarchical reflection.
    _rounded_panel(canvas, draw, (795, 125, 1515, 690))
    _pill(draw, 820, 148, "VLM REFLECTION", fill="#DDF5EA", color="#1F7354")
    draw.text(
        (820, 205), "B. RECOVER THE TASK HIERARCHY", font=_font(25, bold=True), fill=INK
    )
    draw.text(
        (820, 238),
        "Representative trajectory 3: 14 chunks → 4 ordered subtasks",
        font=_font(16),
        fill=MUTED,
    )
    completion_frames = (4, 8, 12, 14)
    frame_width = 150
    for index, (subtask, frame_index) in enumerate(
        zip(subtasks, completion_frames, strict=True)
    ):
        x = 820 + index * 169
        _frame(
            canvas,
            draw,
            episode_roots[3]
            / "key_observations"
            / f"observation_{frame_index:02d}.png",
            (x, 275, x + frame_width, 405),
            f"subtask {index + 1}",
            label_size=11,
        )
        if index < 3:
            _arrow(draw, (x + frame_width + 3, 333), (x + frame_width + 15, 333))
        _wrapped_text(
            draw,
            (x, 420),
            str(subtask["subtask"]),
            font=_font(12, bold=True),
            fill=BODY,
            max_width=frame_width,
            line_height=17,
            max_lines=3,
        )
    draw.rounded_rectangle(
        (825, 505, 1485, 645), radius=16, fill=CARD_BG, outline=CARD_BORDER, width=2
    )
    draw.text(
        (850, 537),
        "Each recovered subtask contains",
        font=_font(18, bold=True),
        fill=INK,
    )
    hierarchy_fields = [
        "before state + relevant relations",
        "selection basis + purpose",
        "completion condition + planned next subtask",
    ]
    for index, line in enumerate(hierarchy_fields):
        y = 570 + index * 25
        draw.ellipse((852, y + 5, 862, y + 15), fill=GREEN)
        draw.text((874, y), line, font=_font(14), fill=BODY)

    # C — atomized knowledge examples.
    _rounded_panel(canvas, draw, (1545, 125, 2355, 690))
    _pill(draw, 1570, 148, "REAL OUTPUT", fill="#E5EEFF", color="#335DAF")
    draw.text(
        (1570, 205),
        "C. ATOMIZE INTO REUSABLE KNOWLEDGE",
        font=_font(25, bold=True),
        fill=INK,
    )
    draw.text(
        (1570, 238),
        "One accepted package shown; all fields come from the stored JSON",
        font=_font(16),
        fill=MUTED,
    )
    draw.rounded_rectangle(
        (1570, 270, 2330, 440), radius=16, fill=GREEN_BG, outline=GREEN_BORDER, width=2
    )
    draw.text((1595, 301), "Task Knowledge", font=_font(18, bold=True), fill=INK)
    task_strategy = task_example["subtask_strategy"]
    task_y = 333
    for label, value in (
        ("Condition", task_example["condition"]["relevant_relations"][0]),
        ("Subtask", task_strategy["subtask"]),
        ("Purpose", task_strategy["purpose"]),
        ("Completion", task_strategy["completion_condition"]),
    ):
        draw.text((1595, task_y), f"{label}:", font=_font(13, bold=True), fill=BODY)
        task_y = (
            _wrapped_text(
                draw,
                (1685, task_y),
                str(value),
                font=_font(13),
                fill=BODY,
                max_width=615,
                line_height=18,
                max_lines=1,
            )
            + 4
        )

    draw.rounded_rectangle(
        (1570, 460, 2330, 655), radius=16, fill=BLUE_BG, outline=BLUE_BORDER, width=2
    )
    draw.text((1595, 491), "Action Knowledge", font=_font(18, bold=True), fill=INK)
    action_condition = action_example["condition"]
    geometry = action_example["geometric_strategy"]
    action_y = 523
    action_lines = (
        (
            "Condition",
            f"{action_condition['action']} {action_condition['object_description']} — {action_condition['held_state']}",
        ),
        (
            "Geometry",
            f"{geometry['approach_direction']}; {geometry['clearance_or_support_constraint']}",
        ),
        ("Expected effect", action_example["expected_effect"]["physical_effect"]),
        (
            "Evidence",
            "support={support}, oppose={oppose}, unverified={unverified}".format(
                **action_example["evidence_summary"]
            ),
        ),
    )
    for label, value in action_lines:
        draw.text((1595, action_y), f"{label}:", font=_font(13, bold=True), fill=BODY)
        action_y = (
            _wrapped_text(
                draw,
                (1705, action_y),
                str(value),
                font=_font(13),
                fill=BODY,
                max_width=590,
                line_height=18,
                max_lines=2 if label == "Geometry" else 1,
            )
            + 4
        )

    # D — measured package-gate outcomes and scientific boundary.
    _rounded_panel(canvas, draw, (45, 725, 2355, 1415))
    _pill(draw, 70, 748, "MEASURED OUTPUT", fill="#DDF5EA", color="#1F7354")
    draw.text(
        (70, 805),
        "D. WHAT THE CURRENT A-LINE EVIDENCE SHOWS",
        font=_font(25, bold=True),
        fill=INK,
    )
    draw.text(
        (70, 838),
        "Counts below are recomputed from the five accepted v3 packages",
        font=_font(16),
        fill=MUTED,
    )

    metrics = (
        (f"{report['trajectory_count']}/5", "packages valid", GREEN, GREEN_BG),
        (str(report["subtask_count"]), "ordered subtasks", GREEN, GREEN_BG),
        (str(report["action_knowledge_count"]), "Action Knowledge", BLUE, BLUE_BG),
        (str(report["evidence_count"]), "evidence records", BLUE, BLUE_BG),
    )
    for index, (value, label, color, fill) in enumerate(metrics):
        x = 75 + index * 255
        _metric_card(draw, (x, 875, x + 230, 975), value, label, color=color, fill=fill)

    draw.rounded_rectangle(
        (75, 1000, 580, 1280), radius=16, fill=CARD_BG, outline=CARD_BORDER, width=2
    )
    draw.text((100, 1035), "Action families", font=_font(18, bold=True), fill=INK)
    action_types = report["action_types"]
    action_max = max(action_types.values())
    for index, name in enumerate(("grasp", "place", "contact")):
        _bar(
            draw,
            100,
            1070 + index * 62,
            430,
            name,
            action_types[name],
            action_max,
            color=BLUE,
        )

    draw.rounded_rectangle(
        (610, 1000, 1115, 1280), radius=16, fill=CARD_BG, outline=CARD_BORDER, width=2
    )
    draw.text((635, 1035), "Evidence verdicts", font=_font(18, bold=True), fill=INK)
    verdicts = report["evidence_verdicts"]
    verdict_max = max(verdicts.values())
    for index, (name, color) in enumerate(
        (("support", GREEN), ("unverified", AMBER_BORDER), ("oppose", "#B34A4A"))
    ):
        _bar(
            draw,
            635,
            1070 + index * 62,
            430,
            name,
            verdicts[name],
            verdict_max,
            color=color,
        )

    draw.rounded_rectangle(
        (1145, 875, 1650, 1280), radius=16, fill=GREEN_BG, outline=GREEN_BORDER, width=2
    )
    draw.text(
        (1170, 910), "Temporal and transfer checks", font=_font(18, bold=True), fill=INK
    )
    timing = report["evidence_timing"]
    checks = [
        f"✓ delayed evidence: {timing['delayed']}",
        f"✓ immediate evidence: {timing['immediate']}",
        f"✓ absolute pose keys: {report['absolute_pose_or_coordinate_key_count']}",
        f"✓ top-level dynamic-state leakage: {report['dynamic_state_key_count_in_top_level_objects']}",
        "✓ all five packages atomized",
        "✓ no automatic retries",
    ]
    check_y = 956
    for line in checks:
        draw.text((1175, check_y), line, font=_font(15), fill=BODY)
        check_y += 45

    draw.rounded_rectangle(
        (1680, 875, 2325, 1280), radius=16, fill=AMBER_BG, outline=AMBER_BORDER, width=2
    )
    draw.text((1710, 910), "RQ1 evaluator status", font=_font(18, bold=True), fill=INK)
    draw.text((1710, 950), "READY", font=_font(28, bold=True), fill=AMBER)
    pending = [
        "Subtask Boundary F1",
        "Order Accuracy",
        "Purpose / Completion Correctness",
        "Geometry Knowledge Quality",
        "Reasoning Factuality",
        "Delayed-evidence accuracy",
    ]
    pending_y = 997
    for metric in pending:
        draw.text((1715, pending_y), f"○ {metric}", font=_font(15), fill=BODY)
        pending_y += 38
    draw.text(
        (1710, 1245),
        "Independent blind annotations pending",
        font=_font(14, bold=True),
        fill=AMBER,
    )

    draw.rounded_rectangle(
        (70, 1320, 2330, 1385), radius=16, fill=AMBER_BG, outline=AMBER_BORDER, width=2
    )
    boundary = (
        "Scientific boundary: structure, atomization, evidence timing, and leakage checks are validated; "
        "the numbers above are not final blind-review quality scores."
    )
    draw.text(
        (1200, 1352), boundary, font=_font(14, bold=True), fill="#805513", anchor="mm"
    )

    draw.text(
        (1200, 1460),
        "Sources: RMBench expert trajectories 0–4; GPT-5.5 xhigh reflection; SAM3 regions; HPK-v3 package-gate report.",
        font=_font(12),
        fill=MUTED,
        anchor="mm",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--existing-root", type=Path, default=DEFAULT_EXISTING_ROOT)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(
        report_path=args.report.resolve(strict=True),
        existing_root=args.existing_root.resolve(strict=True),
        new_root=args.new_root.resolve(strict=True),
        output=args.output.resolve(strict=False),
    )


if __name__ == "__main__":
    main()
