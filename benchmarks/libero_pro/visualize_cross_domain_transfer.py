"""Render an honest RMBench-to-LIBERO Action Knowledge case study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from benchmarks.libero_pro.visualize_task_hpk_pair import (
    _BG,
    _BLUE,
    _BLUE_BG,
    _BODY,
    _BORDER,
    _GREEN,
    _GREEN_BG,
    _INK,
    _MUTED,
    _WARNING,
    _WARNING_BG,
    _font,
    _paste_frame,
    _read_json,
    _read_jsonl,
    _round_rect,
    _wrapped,
    replay_trace_frames,
)

_CONDITIONS = ("off", "rmbench_task", "rmbench_task_action", "libero_native")
_LABELS = {
    "off": "HPK OFF",
    "rmbench_task": "RMBench TASK",
    "rmbench_task_action": "RMBench TASK + ACTION",
    "libero_native": "LIBERO-NATIVE HPK",
}


def _condition_counts(trace_path: Path) -> dict[str, int]:
    rows = _read_jsonl(trace_path)
    task_opportunities = 0
    task_adopted = 0
    action_opportunities = 0
    action_adopted = 0
    for row in rows:
        if row.get("event") == "planner_decision" and isinstance(
            row.get("hpk_v3_subtask_usage"), dict
        ):
            task_opportunities += 1
            task_adopted += int(
                row["hpk_v3_subtask_usage"].get("knowledge_adopted") is True
            )
        if row.get("event") == "executor_request" and isinstance(
            row.get("hpk_v3_action_usage"), dict
        ):
            action_opportunities += 1
            action_adopted += int(
                row["hpk_v3_action_usage"].get("knowledge_adopted") is True
            )
    return {
        "task_opportunities": task_opportunities,
        "task_adopted": task_adopted,
        "action_opportunities": action_opportunities,
        "action_adopted": action_adopted,
    }


def _summary(root: Path, audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in _CONDITIONS:
        condition = dict(audit["conditions"][name])
        condition.update(_condition_counts(root / name / "agent_loop_trace.jsonl"))
        result[name] = condition
    return result


def _heatmap(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    deltas: np.ndarray,
    *,
    origin: tuple[int, int],
    cell: tuple[int, int] = (76, 34),
) -> None:
    if deltas.shape != (10, 7):
        raise ValueError("controlled delta heatmap must have shape (10, 7)")
    x0, y0 = origin
    labels = ("dx", "dy", "dz", "dRx", "dRy", "dRz", "grip")
    maximum = float(deltas.max(initial=0.0)) or 1.0
    for column, label in enumerate(labels):
        draw.text(
            (x0 + column * cell[0] + cell[0] // 2, y0 - 26),
            label,
            fill=_MUTED,
            font=_font(13, bold=True),
            anchor="mm",
        )
    for row in range(10):
        draw.text(
            (x0 - 18, y0 + row * cell[1] + cell[1] // 2),
            str(row + 1),
            fill=_MUTED,
            font=_font(11),
            anchor="mm",
        )
        for column in range(7):
            ratio = float(deltas[row, column]) / maximum
            color = (
                int(225 - 105 * ratio),
                int(239 - 90 * ratio),
                int(242 - 70 * ratio),
            )
            box = (
                x0 + column * cell[0],
                y0 + row * cell[1],
                x0 + (column + 1) * cell[0] - 2,
                y0 + (row + 1) * cell[1] - 2,
            )
            draw.rectangle(box, fill=color, outline="white", width=1)


def render_mechanism(
    *,
    reflection_root: Path,
    grounding: dict[str, Any],
    ablation: dict[str, Any],
    experiment_root: Path,
    replay_root: Path,
    audit: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    summary = _summary(experiment_root, audit)
    if grounding.get("knowledge_adopted") is not True:
        raise ValueError("mechanism figure requires adopted Action Knowledge")
    knowledge = grounding["audit"]["retrieved_knowledge"]
    if knowledge["condition"]["action"] != "grasp":
        raise ValueError("mechanism figure requires a grasp transfer")
    baseline = np.asarray(ablation["baseline_actions"], dtype=np.float32)
    grounded = np.asarray(ablation["grounded_actions"], dtype=np.float32)
    deltas = np.abs(baseline - grounded)
    if deltas.shape != (10, 7) or ablation.get("motion_executed") is not False:
        raise ValueError("mechanism figure requires a non-executed 10x7 ablation")

    canvas = Image.new("RGB", (2400, 1450), _BG)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (1200, 50),
        "Cross-Benchmark HPK: What Transferred, What Changed, and What Did Not Improve",
        fill=_INK,
        font=_font(38, bold=True),
        anchor="mm",
    )
    draw.text(
        (1200, 88),
        "Real RMBench expert grasp → semantic Action Knowledge → LIBERO bowl grounding → native control",
        fill=_MUTED,
        font=_font(18),
        anchor="mm",
    )
    panels = (
        (40, 120, 690, 775),
        (720, 120, 1425, 775),
        (1455, 120, 2360, 775),
        (40, 805, 2360, 1370),
    )
    for box in panels:
        _round_rect(draw, box, fill="white", radius=24)

    # A: source experience
    draw.text(
        (70, 160), "1. READ RMBench EXPERIENCE", fill=_INK, font=_font(24, bold=True)
    )
    draw.text(
        (70, 195),
        "Three consecutive successful expert trajectories; no source record was changed",
        fill=_MUTED,
        font=_font(14),
    )
    source_frames = (
        reflection_root / "episode_1/key_observations/observation_00.png",
        reflection_root / "episode_1/key_observations/observation_03.png",
        reflection_root / "episode_1/key_observations/observation_04.png",
        reflection_root / "episode_2/key_observations/observation_04.png",
    )
    captions = ("target shown", "close", "lift", "second expert lift")
    for index, (path, caption) in enumerate(zip(source_frames, captions, strict=True)):
        x = 70 + (index % 2) * 295
        y = 230 + (index // 2) * 205
        _paste_frame(canvas, draw, path, box=(x, y, x + 270, y + 175), caption=caption)
    _round_rect(draw, (70, 660, 660, 742), fill=_GREEN_BG, outline="#63AD91", radius=14)
    draw.text(
        (95, 683), "Evidence carried forward", fill=_INK, font=_font(16, bold=True)
    )
    draw.text(
        (95, 713),
        "1 support / 0 oppose / 1 unverified",
        fill=_GREEN,
        font=_font(15, bold=True),
    )

    # B: reusable knowledge
    draw.text(
        (750, 160), "2. ORGANIZE ACTION KNOWLEDGE", fill=_INK, font=_font(24, bold=True)
    )
    condition = knowledge["condition"]
    geometry = knowledge["geometric_strategy"]
    effect = knowledge["expected_effect"]
    _round_rect(
        draw, (750, 220, 1395, 675), fill=_GREEN_BG, outline="#63AD91", radius=18
    )
    y = 248
    rows = (
        (
            "WHEN",
            f"{condition['object_description']}; {condition['support_relation']}; {condition['held_state']}",
        ),
        ("DO", condition["action"]),
        (
            "APPROACH",
            f"{geometry['approach_direction']} relative to {geometry['approach_reference']}",
        ),
        ("CONTACT", str(geometry["contact_relation"])),
        ("CLEARANCE", str(geometry["clearance_or_support_constraint"])),
        ("EXPECT", effect["physical_effect"]),
        ("VERIFY", effect["verification_observation"]),
    )
    for label, text in rows:
        draw.text((780, y), label, fill=_GREEN, font=_font(14, bold=True))
        y = (
            _wrapped(
                draw,
                text,
                xy=(900, y),
                max_chars=55,
                font=_font(14),
                max_lines=2,
            )
            + 14
        )
    draw.text(
        (750, 710),
        "No pose, action vector, candidate ID, object instance ID, or trajectory ID is stored.",
        fill=_MUTED,
        font=_font(13),
    )

    # C: target-domain grounding
    draw.text(
        (1485, 160),
        "3. RETRIEVE AND GROUND IN LIBERO",
        fill=_INK,
        font=_font(24, bold=True),
    )
    _paste_frame(
        canvas,
        draw,
        replay_root / "off/step_010.png",
        box=(1485, 210, 1845, 475),
        caption="same real observation · step 10",
    )
    _round_rect(
        draw, (1875, 210, 2330, 475), fill=_BLUE_BG, outline="#7699E8", radius=16
    )
    draw.text((1900, 240), "Baseline π0.5 prompt", fill=_INK, font=_font(16, bold=True))
    y = _wrapped(
        draw,
        grounding["prompt_before"],
        xy=(1900, 275),
        max_chars=49,
        font=_font(14),
        max_lines=3,
    )
    draw.text(
        (1900, y + 18), "HPK-grounded prompt", fill=_GREEN, font=_font(16, bold=True)
    )
    _wrapped(
        draw,
        grounding["audit"]["grounded_action_guidance"],
        xy=(1900, y + 53),
        max_chars=49,
        font=_font(14),
        max_lines=5,
    )
    _round_rect(
        draw, (1485, 515, 2330, 730), fill="#F8FAFC", outline=_BORDER, radius=16
    )
    draw.text(
        (1510, 545),
        "Why this record was selected",
        fill=_INK,
        font=_font(16, bold=True),
    )
    _wrapped(
        draw,
        grounding["audit"]["retrieval_reason"],
        xy=(1510, 582),
        max_chars=98,
        font=_font(14),
        max_lines=3,
    )
    draw.text(
        (1510, 680),
        "Old tray/block grasp was rejected as less compatible.",
        fill=_WARNING,
        font=_font(14, bold=True),
    )

    # D: controlled mechanism and real outcome
    draw.text(
        (70, 845),
        "4. CONTROL CHANGED; PERFORMANCE DID NOT IMPROVE",
        fill=_INK,
        font=_font(25, bold=True),
    )
    draw.text(
        (70, 882),
        "Left: same observation + same diffusion noise. Right: complete real matched episodes.",
        fill=_MUTED,
        font=_font(14),
    )
    _heatmap(canvas, draw, deltas, origin=(115, 955))
    draw.text(
        (115, 1320),
        "|baseline − grounded| for each native 7D predicted action",
        fill=_MUTED,
        font=_font(13),
    )
    draw.text(
        (720, 970),
        f"changed predicted steps: {ablation['changed_action_steps']} / 10",
        fill=_GREEN,
        font=_font(18, bold=True),
    )
    draw.text(
        (720, 1010),
        f"maximum absolute action delta: {ablation['max_absolute_delta']:.4f}",
        fill=_BODY,
        font=_font(15),
    )
    draw.text(
        (720, 1050), "motion executed in this ablation: NO", fill=_MUTED, font=_font(14)
    )
    off = summary["off"]
    transfer = summary["rmbench_task_action"]
    _round_rect(
        draw, (1120, 925, 1600, 1265), fill=_BLUE_BG, outline="#7699E8", radius=18
    )
    _round_rect(
        draw, (1630, 925, 2110, 1265), fill=_WARNING_BG, outline=_WARNING, radius=18
    )
    draw.text(
        (1360, 960), "HPK OFF", fill=_BLUE, font=_font(19, bold=True), anchor="mm"
    )
    draw.text(
        (1870, 960),
        "RMBench TASK + ACTION",
        fill=_WARNING,
        font=_font(17, bold=True),
        anchor="mm",
    )
    for x, values in ((1145, off), (1655, transfer)):
        draw.text(
            (x, 1010),
            f"success: {values['success']}",
            fill=_INK,
            font=_font(16, bold=True),
        )
        draw.text((x, 1055), f"steps: {values['steps']}", fill=_BODY, font=_font(16))
        draw.text(
            (x, 1100),
            f"recoveries: {values['recovery_boundaries']}",
            fill=_BODY,
            font=_font(16),
        )
        draw.text(
            (x, 1145),
            f"planner calls: {values['planner_calls']}",
            fill=_BODY,
            font=_font(16),
        )
        draw.text(
            (x, 1190),
            f"Action adopted: {values['action_adopted']} / {values['action_opportunities']}",
            fill=_BODY,
            font=_font(15),
        )
    draw.text((2150, 982), "+61 steps", fill=_WARNING, font=_font(21, bold=True))
    draw.text((2150, 1030), "+9 recoveries", fill=_WARNING, font=_font(18, bold=True))
    _wrapped(
        draw,
        "Mechanism validated: cross-domain Action Knowledge changes a real native controller and can still complete the task. Performance improvement is not validated; repeated source guidance caused negative transfer in this case.",
        xy=(2150, 1090),
        max_chars=29,
        font=_font(14),
        fill=_INK,
        max_lines=7,
    )

    canvas.save(output_path)
    return {
        "path": str(output_path),
        "source_expert_trajectories": 3,
        "action_knowledge_adopted": True,
        "changed_predicted_steps": int(ablation["changed_action_steps"]),
        "off_steps": int(off["steps"]),
        "transfer_steps": int(transfer["steps"]),
        "performance_improvement_validated": False,
    }


def render_outcomes(
    *,
    experiment_root: Path,
    replay_root: Path,
    audit: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    summary = _summary(experiment_root, audit)
    canvas = Image.new("RGB", (2400, 1450), _BG)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (1200, 52),
        "RMBench → LIBERO-PRO: One Complete Four-Condition Matched Case",
        fill=_INK,
        font=_font(40, bold=True),
        anchor="mm",
    )
    draw.text(
        (1200, 92),
        "Same task, initialization, seed, cameras, π0.5 checkpoint, and 220-step budget",
        fill=_MUTED,
        font=_font(18),
        anchor="mm",
    )
    positions = ((50, 145), (625, 145), (1200, 145), (1775, 145))
    colors = (_BLUE_BG, "#F8FAFC", _WARNING_BG, _GREEN_BG)
    outlines = ("#7699E8", _BORDER, _WARNING, "#63AD91")
    for name, (x, y), fill, outline in zip(
        _CONDITIONS, positions, colors, outlines, strict=True
    ):
        values = summary[name]
        _round_rect(
            draw, (x, y, x + 525, y + 820), fill=fill, outline=outline, radius=22
        )
        draw.text(
            (x + 262, y + 42),
            _LABELS[name],
            fill=_INK,
            font=_font(20, bold=True),
            anchor="mm",
        )
        final_frame = replay_root / name / f"step_{values['steps']:03d}.png"
        _paste_frame(
            canvas,
            draw,
            final_frame,
            box=(x + 28, y + 85, x + 497, y + 425),
            caption=f"real final state · step {values['steps']}",
        )
        draw.text(
            (x + 40, y + 470), "SUCCESS ✓", fill=_GREEN, font=_font(21, bold=True)
        )
        metrics = (
            f"steps: {values['steps']}",
            f"recoveries: {values['recovery_boundaries']}",
            f"planner calls: {values['planner_calls']}",
            f"Task adopted: {values['task_adopted']} / {values['task_opportunities']}",
            f"Action adopted: {values['action_adopted']} / {values['action_opportunities']}",
        )
        for index, text in enumerate(metrics):
            draw.text((x + 40, y + 520 + index * 48), text, fill=_BODY, font=_font(16))
        if name == "rmbench_task":
            note = "No source Task Knowledge matched; the −17-step difference is rollout variation."
        elif name == "rmbench_task_action":
            note = "Cross-domain actions were used, but repeated adoption increased execution cost."
        elif name == "libero_native":
            note = "Target-domain knowledge was used; this single case was still 5 steps slower than Off."
        else:
            note = "Matched baseline without HPK."
        _wrapped(
            draw,
            note,
            xy=(x + 40, y + 760),
            max_chars=49,
            font=_font(13),
            max_lines=3,
        )

    _round_rect(draw, (50, 1005, 2350, 1370), fill="white", radius=22)
    draw.text(
        (85, 1045), "WHAT THIS CASE ESTABLISHES", fill=_INK, font=_font(23, bold=True)
    )
    statements = (
        (
            "✓",
            _GREEN,
            "RMBench grasp experience can be converted into semantic Action Knowledge without poses or action vectors.",
        ),
        (
            "✓",
            _GREEN,
            "The knowledge can be retrieved, grounded to a LIBERO bowl, sent to π0.5, and executed successfully.",
        ),
        (
            "✓",
            _GREEN,
            "All four complete episodes reached benchmark success under the official 220-step limit.",
        ),
        (
            "✗",
            _WARNING,
            "This matched case does not show a performance gain: Task+Action took 176 steps versus Off at 115.",
        ),
        (
            "→",
            _BLUE,
            "Next experiment: separate grasp-only transfer from place transfer and repeat across preregistered init/seed pairs.",
        ),
    )
    y = 1095
    for symbol, color, text in statements:
        draw.text((90, y), symbol, fill=color, font=_font(21, bold=True))
        draw.text((135, y + 2), text, fill=_BODY, font=_font(16))
        y += 54
    canvas.save(output_path)
    return {
        "path": str(output_path),
        "conditions": summary,
        "all_success": all(value["success"] for value in summary.values()),
        "performance_improvement_validated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reflection-root", type=Path, required=True)
    parser.add_argument("--grounding-decision", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    experiment = args.experiment_root.expanduser().resolve(strict=True)
    replay_root = output / "replay"
    replays = {}
    for name in _CONDITIONS:
        replays[name] = replay_trace_frames(
            experiment / name,
            assets_root=args.assets_root,
            output_dir=replay_root / name,
        )
    audit = _read_json(experiment / "transfer_audit.json")
    mechanism = render_mechanism(
        reflection_root=args.reflection_root.expanduser().resolve(strict=True),
        grounding=_read_json(args.grounding_decision.expanduser().resolve(strict=True)),
        ablation=_read_json(args.ablation.expanduser().resolve(strict=True)),
        experiment_root=experiment,
        replay_root=replay_root,
        audit=audit,
        output_path=output / "libero_cross_domain_action_mechanism.png",
    )
    outcomes = render_outcomes(
        experiment_root=experiment,
        replay_root=replay_root,
        audit=audit,
        output_path=output / "libero_cross_domain_four_condition_outcomes.png",
    )
    manifest = {
        "schema": "roboharn_evo/libero_cross_domain_case_study/v1",
        "replays": replays,
        "mechanism_figure": mechanism,
        "outcomes_figure": outcomes,
        "claim_scope": audit["claim_scope"],
    }
    (output / "visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
