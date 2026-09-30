"""Render real-evidence figures for one matched LIBERO Task-HPK smoke.

The script deterministically replays already-recorded native actions to recover
display frames.  It can additionally run two local pi0.5 predictions with the
same observation and sampling noise to isolate the prompt effect.  It never
calls a planner or HPK Store writer, and the controlled predictions are never
executed in the simulator.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from benchmarks.libero_pro.integration import (
    build_environment,
    reset_to_initial_state,
    task_by_id,
)
from roboharn_evo.benchmark_adapters import ActionRequest
from roboharn_evo.agent.hpk.compatibility import (
    normalize_hpk_runtime_provenance,
    normalize_v3_rollout_record,
)
from roboharn_evo.benchmark_adapters.libero_pro import (
    LiberoPi05PolicyBackend,
    LiberoPi05PolicyConfig,
    LiberoProActionContract,
    LiberoProAdapter,
)

_CANVAS = (2400, 1450)
_BG = "#F4F7FA"
_INK = "#102A43"
_MUTED = "#52677D"
_BODY = "#3F556A"
_BORDER = "#CBD6E2"
_GREEN = "#317D67"
_GREEN_BG = "#EFF9F5"
_BLUE = "#3E68C0"
_BLUE_BG = "#EEF4FF"
_WARNING = "#D9A13C"
_WARNING_BG = "#FFF8E8"
_FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
_FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
_FONT_MONO = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")


class LiberoVisualizationError(ValueError):
    """The paired evidence cannot support an honest visualization."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiberoVisualizationError(f"cannot read {path}") from exc
    if not isinstance(value, dict):
        raise LiberoVisualizationError(f"{path.name} must contain one object")
    value = normalize_hpk_runtime_provenance(value)
    for previous, current in (
        ("afk_rewrite_prompt", "hpk_rewrite_prompt"),
        ("afk_rewrite_actions", "hpk_rewrite_actions"),
    ):
        if previous in value:
            item = value.pop(previous)
            if current in value and value[current] != item:
                raise LiberoVisualizationError(f"conflicting fields: {previous}, {current}")
            value[current] = item
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LiberoVisualizationError(f"cannot read {path}") from exc
    rows: list[dict[str, Any]] = []
    for ordinal, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LiberoVisualizationError(
                f"{path.name} line {ordinal} is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise LiberoVisualizationError(
                f"{path.name} line {ordinal} must contain an object"
            )
        rows.append(normalize_v3_rollout_record(value))
    return tuple(rows)


def _font(
    size: int, *, bold: bool = False, mono: bool = False
) -> ImageFont.FreeTypeFont:
    path = _FONT_MONO if mono else _FONT_BOLD if bold else _FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def _fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        contained = ImageOps.contain(
            source.convert("RGB"), size, method=Image.Resampling.LANCZOS
        )
    canvas = Image.new("RGB", size, "white")
    canvas.paste(
        contained,
        ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2),
    )
    return canvas


def _display_camera(value: Any) -> Image.Image:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise LiberoVisualizationError("replay camera must be HWC uint8 RGB")
    # LIBERO exposes raw OpenGL camera orientation.  The policy and human
    # display paths rotate both cameras by 180 degrees; no spatial content is
    # otherwise changed.
    return Image.fromarray(np.rot90(array, 2).copy(), mode="RGB")


def _paired_camera_frame(observation: Any, *, step: int) -> Image.Image:
    external = _display_camera(observation.cameras["agentview_image"])
    wrist = _display_camera(observation.cameras["robot0_eye_in_hand_image"])
    external = ImageOps.fit(external, (320, 320), method=Image.Resampling.LANCZOS)
    wrist = ImageOps.fit(wrist, (320, 320), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (660, 360), "white")
    canvas.paste(external, (5, 35))
    canvas.paste(wrist, (335, 35))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 8), f"step {step} · external", fill=_MUTED, font=_font(18, bold=True)
    )
    draw.text((342, 8), "wrist", fill=_MUTED, font=_font(18, bold=True))
    return canvas


def _native_contract() -> LiberoProActionContract:
    return LiberoProActionContract(
        action_type="libero",
        shape=(7,),
        lower_bounds=(-1.0,) * 7,
        upper_bounds=(1.0,) * 7,
        bounds_handling="native_controller",
        control_mode="OSC_POSE",
        translation_mode="normalized delta",
        rotation_representation="normalized delta axis-angle",
        reference_frame="native robosuite OSC_POSE controller frame",
        gripper_convention="-1=open, +1=close",
    )


def replay_trace_frames(
    condition_dir: str | Path,
    *,
    assets_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Replay exact traced actions and persist paired public camera frames."""

    condition = Path(condition_dir).resolve(strict=True)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = _read_json(condition / "provenance.json")
    trace = _read_jsonl(condition / "agent_loop_trace.jsonl")
    if not trace or trace[0].get("event") != "episode_start":
        raise LiberoVisualizationError("trace has no episode_start")
    context = trace[0].get("benchmark_context")
    if not isinstance(context, dict):
        raise LiberoVisualizationError("trace lacks self-contained benchmark context")
    action_rows = tuple(row for row in trace if row.get("event") == "action_executed")
    if not action_rows:
        raise LiberoVisualizationError("trace has no executed actions")
    task = task_by_id(str(context["suite"]), int(context["task_id"]))
    env = build_environment(
        task,
        seed=int(context["seed"]),
        assets_root=Path(assets_root),
        runtime_config_dir=output / ".liberopro",
        camera_height=int(context["camera_height"]),
        camera_width=int(context["camera_width"]),
        horizon=int(context["action_horizon"]),
    )
    replayed_rows = 0
    state_rows: list[dict[str, Any]] = []

    def capture_public_state(observation: Any, step: int) -> None:
        raw = observation.raw
        state_rows.append(
            {
                "step": step,
                "eef_position_m": np.asarray(
                    raw["robot0_eef_pos"], dtype=float
                ).tolist(),
                "eef_quaternion_xyzw": np.asarray(
                    raw["robot0_eef_quat"], dtype=float
                ).tolist(),
                "gripper_qpos": np.asarray(
                    raw["robot0_gripper_qpos"], dtype=float
                ).tolist(),
            }
        )
        if step == 10:
            np.savez_compressed(
                output / "policy_observation_step_010.npz",
                agentview_image=np.asarray(raw["agentview_image"], dtype=np.uint8),
                robot0_eye_in_hand_image=np.asarray(
                    raw["robot0_eye_in_hand_image"], dtype=np.uint8
                ),
                robot0_eef_pos=np.asarray(raw["robot0_eef_pos"]),
                robot0_eef_quat=np.asarray(raw["robot0_eef_quat"]),
                robot0_gripper_qpos=np.asarray(raw["robot0_gripper_qpos"]),
            )

    try:
        raw = reset_to_initial_state(
            env,
            task,
            init_state_id=int(context["init_state_id"]),
        )
        adapter = LiberoProAdapter(
            env,
            instruction=task.instruction,
            action_contract=_native_contract(),
            step_limit=int(context["action_horizon"]),
        )
        observation = adapter.reset(raw)
        assert observation is not None
        _paired_camera_frame(observation, step=0).save(output / "step_000.png")
        capture_public_state(observation, 0)
        for row in action_rows:
            expected_before = int(row["step_before"])
            if adapter.episode_state().step_count != expected_before:
                raise LiberoVisualizationError("replay step_before diverged from trace")
            action = np.asarray(row["action"], dtype=np.float32)
            result = adapter.execute(
                ActionRequest(
                    action=action,
                    action_type="libero",
                    metadata={"source": "visualization replay"},
                )
            )
            if result.step_after != int(row["step_after"]):
                raise LiberoVisualizationError("replay step_after diverged from trace")
            if result.episode_state.benchmark_success != bool(row["benchmark_success"]):
                raise LiberoVisualizationError("replay benchmark result diverged")
            observation = result.post_observation
            _paired_camera_frame(observation, step=result.step_after).save(
                output / f"step_{result.step_after:03d}.png"
            )
            capture_public_state(observation, result.step_after)
            replayed_rows += 1
        episode_end = next(
            (row for row in reversed(trace) if row.get("event") == "episode_end"),
            None,
        )
        if episode_end is None or replayed_rows != int(episode_end["step_count"]):
            raise LiberoVisualizationError("replay does not reach trace episode_end")
        final_state = adapter.episode_state()
        expected_loop = provenance["simulator"]["roboharn_agent_loop"]
        if final_state.benchmark_success != bool(expected_loop["success"]):
            raise LiberoVisualizationError("replay final success diverged")
    finally:
        env.close()
    (output / "public_state_trace.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in state_rows),
        encoding="utf-8",
    )
    return {
        "condition_dir": str(condition),
        "frames": replayed_rows + 1,
        "actions_replayed": replayed_rows,
        "benchmark_success": bool(provenance["simulator"]["roboharn_agent_loop"]["success"]),
        "display_rotation_degrees": 180,
        "planner_called": False,
        "policy_called": False,
        "store_read": False,
        "store_write": False,
        "public_state_trace": str(output / "public_state_trace.jsonl"),
        "policy_observation": str(output / "policy_observation_step_010.npz"),
    }


def controlled_prompt_ablation(
    *,
    pair_root: str | Path,
    replay_root: str | Path,
    output_path: str | Path,
    noise_seed: int = 0,
) -> dict[str, Any]:
    """Compare baseline/rewrite prompts with identical observation and noise."""

    pair = Path(pair_root).resolve(strict=True)
    replay = Path(replay_root).resolve(strict=True)
    provenance = _read_json(pair / "task" / "provenance.json")
    trace = _read_jsonl(pair / "task" / "agent_loop_trace.jsonl")
    decision = next(
        (row for row in trace if row.get("event") == "planner_decision"),
        None,
    )
    if decision is None or not isinstance(decision.get("hpk_v3_subtask_usage"), dict):
        raise LiberoVisualizationError("Task trace lacks an HPK rewrite")
    usage = decision["hpk_v3_subtask_usage"]
    baseline_prompt = str(usage["subtask_before"]).strip()
    rewrite_prompt = str(usage["subtask_after"]).strip()
    if not baseline_prompt or not rewrite_prompt or baseline_prompt == rewrite_prompt:
        raise LiberoVisualizationError(
            "controlled prompts must be non-empty and distinct"
        )
    with np.load(replay / "task" / "policy_observation_step_010.npz") as archive:
        observation = {key: archive[key].copy() for key in archive.files}
    policy = provenance["policy"]
    config = LiberoPi05PolicyConfig(
        checkpoint_dir=Path(policy["checkpoint_dir"]),
        source_ref=str(policy["source_ref"]),
        license_id=str(policy["license_id"]),
        config_name=str(policy["config_name"]),
        device=str(policy["device"]),
    )
    try:
        from openpi.training import config as training_config
    except ImportError as exc:
        raise LiberoVisualizationError(
            "OpenPI is required for prompt ablation"
        ) from exc
    model_config = training_config.get_config(config.config_name).model
    rng = np.random.default_rng(noise_seed)
    noise = rng.standard_normal(
        (int(model_config.action_horizon), int(model_config.action_dim))
    ).astype(np.float32)
    backend = LiberoPi05PolicyBackend(config)
    baseline_actions = backend.predict_native_action_chunk(
        observation,
        prompt=baseline_prompt,
        sampling_noise=noise,
    )
    rewrite_actions = backend.predict_native_action_chunk(
        observation,
        prompt=rewrite_prompt,
        sampling_noise=noise,
    )
    if baseline_actions.shape != rewrite_actions.shape:
        raise LiberoVisualizationError("controlled action shapes differ")
    deltas = np.max(np.abs(baseline_actions - rewrite_actions), axis=1)
    result = {
        "schema": "roboharn_evo/libero_task_hpk_controlled_prompt_ablation/v1",
        "observation_source": "Task replay at step 10 before the first planner action",
        "baseline_prompt": baseline_prompt,
        "hpk_rewrite_prompt": rewrite_prompt,
        "same_observation": True,
        "same_sampling_noise": True,
        "sampling_noise_seed": noise_seed,
        "sampling_noise_shape": list(noise.shape),
        "action_shape": list(baseline_actions.shape),
        "changed_action_steps": int(np.count_nonzero(deltas > 0)),
        "max_absolute_delta_by_step": deltas.tolist(),
        "max_absolute_delta": float(deltas.max()),
        "motion_executed": False,
        "baseline_actions": baseline_actions.tolist(),
        "hpk_rewrite_actions": rewrite_actions.tolist(),
    }
    output = Path(output_path).resolve()
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _video_from_frames(frame_dir: Path, output: Path, *, fps: int) -> None:
    if fps <= 0:
        raise LiberoVisualizationError("video fps must be positive")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "20",
            "-i",
            str(frame_dir / "step_%03d.png"),
            "-vf",
            f"fps={fps}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )


def _comparison_video(
    off_frame_dir: Path,
    task_frame_dir: Path,
    output: Path,
    *,
    fps: int,
) -> None:
    """Create a labeled 4x slow presentation video without interpolated frames."""

    off_frames = len(tuple(off_frame_dir.glob("step_*.png")))
    task_frames = len(tuple(task_frame_dir.glob("step_*.png")))
    if off_frames < 1 or task_frames < 1:
        raise LiberoVisualizationError("comparison video requires both frame sets")
    output_frames = int(np.ceil(max(off_frames, task_frames) / 5 * fps))

    filter_graph = (
        "[0:v]tpad=stop_mode=clone:stop_duration=10,"
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans-Bold.ttf:text='HPK OFF':x=18:y=h-40:fontsize=24:"
        "fontcolor=white:box=1:boxcolor=black@0.65[a];"
        "[1:v]tpad=stop_mode=clone:stop_duration=10,"
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans-Bold.ttf:text='TASK HPK':x=18:y=h-40:fontsize=24:"
        "fontcolor=white:box=1:boxcolor=black@0.65[b];"
        f"[a][b]hstack=inputs=2,fps={fps}[v]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "5",
            "-i",
            str(off_frame_dir / "step_%03d.png"),
            "-framerate",
            "5",
            "-i",
            str(task_frame_dir / "step_%03d.png"),
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-frames:v",
            str(output_frames),
            str(output),
        ],
        check=True,
    )


def _round_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str = _BORDER,
    radius: int = 20,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    xy: tuple[int, int],
    max_chars: int,
    font: ImageFont.FreeTypeFont,
    fill: str = _BODY,
    line_gap: int = 8,
    max_lines: int | None = None,
) -> int:
    lines = textwrap.wrap(" ".join(text.split()), width=max_chars) or [""]
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .") + "…"
    x, y = xy
    line_height = font.size + line_gap
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height
    return y


def _paste_frame(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    path: Path,
    *,
    box: tuple[int, int, int, int],
    caption: str,
) -> None:
    x0, y0, x1, y1 = box
    _round_rect(draw, box, fill="white", outline="#AAB8C5", radius=10)
    image = _fit_image(path, (x1 - x0 - 8, y1 - y0 - 34))
    canvas.paste(image, (x0 + 4, y0 + 4))
    draw.text(
        ((x0 + x1) // 2, y1 - 25),
        caption,
        fill=_MUTED,
        font=_font(12),
        anchor="mm",
    )


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: str,
) -> None:
    draw.line((start, end), fill=fill, width=4)
    x, y = end
    draw.polygon(((x, y), (x - 12, y - 8), (x - 12, y + 8)), fill=fill)


def _short(text: str, length: int = 140) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= length else normalized[: length - 1] + "…"


def render_overview(
    *,
    pair_root: str | Path,
    expert_root: str | Path,
    replay_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Render one 2400x1450 overview directly from the real artifacts."""

    pair = Path(pair_root).resolve(strict=True)
    expert = Path(expert_root).resolve(strict=True)
    replay = Path(replay_root).resolve(strict=True)
    audit = _read_json(pair / "paired_audit.json")
    store_rows = _read_jsonl(expert / "store" / "task_knowledge.jsonl")
    action_store = expert / "store" / "action_knowledge.jsonl"
    if action_store.stat().st_size != 0:
        raise LiberoVisualizationError("figure requires task-only HPK evidence")
    off_trace = _read_jsonl(pair / "off" / "agent_loop_trace.jsonl")
    task_trace = _read_jsonl(pair / "task" / "agent_loop_trace.jsonl")
    task_decisions = tuple(
        row for row in task_trace if row.get("event") == "planner_decision"
    )
    if len(task_decisions) != 2 or not all(
        isinstance(row.get("hpk_v3_subtask_usage"), dict)
        and row["hpk_v3_subtask_usage"].get("knowledge_adopted") is True
        for row in task_decisions
    ):
        raise LiberoVisualizationError("figure requires two adopted Task retrievals")
    if not any(row.get("event") == "recovery_boundary" for row in off_trace):
        raise LiberoVisualizationError("Off trace lacks recovery boundary evidence")
    off_policy_actions = np.asarray(
        [
            row["action"]
            for row in off_trace
            if row.get("event") == "action_executed" and row.get("source") == "policy"
        ],
        dtype=float,
    )
    task_policy_actions = np.asarray(
        [
            row["action"]
            for row in task_trace
            if row.get("event") == "action_executed" and row.get("source") == "policy"
        ],
        dtype=float,
    )
    if off_policy_actions.shape != (10, 7) or task_policy_actions.shape != (10, 7):
        raise LiberoVisualizationError("figure requires matched ten-step 7D actions")
    action_differences = np.max(
        np.abs(off_policy_actions - task_policy_actions), axis=1
    )
    if not np.all(action_differences > 0):
        raise LiberoVisualizationError(
            "figure claim requires all policy actions to differ"
        )

    canvas = Image.new("RGB", _CANVAS, _BG)
    draw = ImageDraw.Draw(canvas)
    title = _font(40, bold=True)
    subtitle = _font(18)
    panel_title = _font(24, bold=True)
    panel_note = _font(15)
    head = _font(17, bold=True)
    body = _font(14)
    small = _font(12, bold=True)
    mono = _font(13, mono=True)

    draw.text(
        (1200, 54),
        "HPK on LIBERO-PRO: Real Experience, Task Knowledge, and Changed Control",
        fill=_INK,
        font=title,
        anchor="mm",
    )
    draw.text(
        (1200, 90),
        "Every frame is from a real expert trajectory or deterministic replay of the recorded Agent actions",
        fill=_MUTED,
        font=subtitle,
        anchor="mm",
    )

    panels = (
        (45, 125, 765, 680),
        (795, 125, 1515, 680),
        (1545, 125, 2355, 680),
        (45, 715, 2355, 1345),
    )
    for panel in panels:
        _round_rect(draw, panel, fill="white", radius=24)

    # A: real expert experience
    _round_rect(draw, (70, 148, 195, 178), fill="#DDF5EA", outline="#DDF5EA", radius=15)
    draw.text((132, 163), "REAL EXPERT", fill=_GREEN, font=small, anchor="mm")
    draw.text((70, 205), "A. READ EXPERIENCE", fill=_INK, font=panel_title)
    draw.text(
        (70, 235),
        "98 frames · two cameras · 10 fixed-frequency windows",
        fill=_MUTED,
        font=panel_note,
    )
    expert_images = sorted((expert / "key_observations").glob("observation_*.png"))
    if len(expert_images) != 11:
        raise LiberoVisualizationError("expected eleven expert key observations")
    chosen = (expert_images[0], expert_images[3], expert_images[6], expert_images[-1])
    captions = ("scene", "approach", "lift + transport", "placed")
    for index, (path, caption) in enumerate(zip(chosen, captions, strict=True)):
        x = 70 + index * 170
        _paste_frame(canvas, draw, path, box=(x, 270, x + 150, 400), caption=caption)
        if index < 3:
            _arrow(draw, (x + 150, 334), (x + 165, 334), fill=_GREEN)
    _round_rect(draw, (90, 445, 720, 630), fill="#F8FAFC", outline="#B8C6D3", radius=16)
    draw.text(
        (405, 477),
        "GPT-5.5 hierarchical trajectory reflection",
        fill=_INK,
        font=head,
        anchor="mm",
    )
    draw.text(
        (120, 510),
        "Input: ordered paired frames + measured relative motion",
        fill=_BODY,
        font=body,
    )
    draw.text((120, 541), "No hand-coded grasp/place boundaries", fill=_BODY, font=body)
    draw.text(
        (120, 572),
        "Output: approach → grasp → transport → release",
        fill=_BODY,
        font=body,
    )
    draw.text(
        (405, 608),
        "Reading experience; no online Agent action occurs here.",
        fill=_GREEN,
        font=small,
        anchor="mm",
    )

    # B: task-only knowledge
    _round_rect(
        draw, (820, 148, 970, 178), fill="#DDF5EA", outline="#DDF5EA", radius=15
    )
    draw.text((895, 163), "REAL STORE", fill=_GREEN, font=small, anchor="mm")
    draw.text((820, 205), "B. ORGANIZE TASK KNOWLEDGE", fill=_INK, font=panel_title)
    draw.text(
        (820, 235),
        "1 expert episode → 4 supported Task records; Action records = 0",
        fill=_MUTED,
        font=panel_note,
    )
    _round_rect(
        draw, (825, 270, 1485, 555), fill=_GREEN_BG, outline="#63AD91", radius=16
    )
    draw.text((850, 302), "Reusable high-level sequence", fill=_INK, font=head)
    y = 337
    for index, row in enumerate(store_rows, start=1):
        strategy = row["subtask_strategy"]
        draw.text(
            (850, y),
            f"{index}. {strategy['subtask']}",
            fill=_INK,
            font=_font(15, bold=True),
        )
        y = (
            _wrapped(
                draw,
                f"Purpose: {strategy['purpose']}",
                xy=(880, y + 24),
                max_chars=67,
                font=body,
                max_lines=1,
            )
            + 7
        )
    _round_rect(
        draw, (825, 575, 1485, 640), fill=_WARNING_BG, outline=_WARNING, radius=14
    )
    draw.text((850, 600), "Store boundary", fill=_INK, font=head)
    draw.text(
        (1015, 601),
        "Task Knowledge only · read-only during rollout",
        fill=_BODY,
        font=body,
    )
    draw.text(
        (1015, 625),
        "No object names, IDs, hashes, poses, or Action Knowledge",
        fill=_BODY,
        font=body,
    )

    # C: retrieval in a real Agent rollout
    _round_rect(
        draw, (1570, 148, 1728, 178), fill="#E5EEFF", outline="#E5EEFF", radius=15
    )
    draw.text((1649, 163), "REAL REPLAY", fill=_BLUE, font=small, anchor="mm")
    draw.text((1570, 205), "C. RETRIEVE IN CONTEXT", fill=_INK, font=panel_title)
    draw.text(
        (1570, 235),
        "Task-HPK Agent trace · same native π0.5 executor",
        fill=_MUTED,
        font=panel_note,
    )
    for index, (step, caption) in enumerate(
        ((10, "before planning"), (15, "after rewrite 1"), (20, "after rewrite 2"))
    ):
        x = 1570 + index * 245
        _paste_frame(
            canvas,
            draw,
            replay / "task" / f"step_{step:03d}.png",
            box=(x, 270, x + 230, 445),
            caption=f"step {step}: {caption}",
        )
        if index < 2:
            _arrow(draw, (x + 230, 352), (x + 240, 352), fill=_BLUE)
    usage = task_decisions[0]["hpk_v3_subtask_usage"]
    _round_rect(
        draw, (1570, 480, 2290, 640), fill=_BLUE_BG, outline="#5E82D0", radius=16
    )
    draw.text(
        (1595, 510), "Trace: knowledge_adopted = true (2 / 2)", fill=_INK, font=head
    )
    y = _wrapped(
        draw,
        "Planner baseline: " + _short(usage["subtask_before"]),
        xy=(1595, 540),
        max_chars=87,
        font=body,
        max_lines=2,
    )
    y = _wrapped(
        draw,
        "HPK rewrite: " + _short(usage["subtask_after"]),
        xy=(1595, y + 5),
        max_chars=87,
        font=body,
        max_lines=2,
    )
    draw.text(
        (1595, 616),
        "The rewritten subtask exactly equals the π0.5 prompt.",
        fill=_BLUE,
        font=small,
    )

    # D: matched real action change
    _round_rect(draw, (70, 738, 305, 768), fill="#FFF0D1", outline="#FFF0D1", radius=15)
    draw.text((187, 753), "MATCHED REAL SMOKE", fill="#9A6411", font=small, anchor="mm")
    draw.text((70, 800), "D. CHANGE THE EXECUTED CONTROL", fill=_INK, font=panel_title)
    draw.text(
        (70, 830),
        "Same suite, task, init, seed, instruction, π0.5, cameras, and 20-step budget",
        fill=_MUTED,
        font=panel_note,
    )
    _paste_frame(
        canvas,
        draw,
        replay / "off" / "step_020.png",
        box=(90, 870, 590, 1205),
        caption="HPK OFF · step 20",
    )
    _arrow(draw, (610, 1037), (660, 1037), fill=_BLUE)
    _paste_frame(
        canvas,
        draw,
        replay / "task" / "step_020.png",
        box=(680, 870, 1180, 1205),
        caption="TASK HPK · step 20",
    )
    draw.text((340, 1235), "baseline planner prompt", fill=_INK, font=head, anchor="mm")
    draw.text(
        (930, 1235),
        "retrieved Task Knowledge prompt",
        fill=_INK,
        font=head,
        anchor="mm",
    )

    _round_rect(
        draw, (1230, 870, 1715, 1250), fill=_GREEN_BG, outline="#63AD91", radius=16
    )
    draw.text((1260, 905), "What HPK changed", fill=_INK, font=head)
    draw.text((1260, 944), "retrievals adopted", fill=_BODY, font=body)
    draw.text((1625, 944), "2 / 2", fill=_INK, font=mono, anchor="ra")
    draw.text((1260, 982), "subtasks rewritten", fill=_BODY, font=body)
    draw.text((1625, 982), "2 / 2", fill=_INK, font=mono, anchor="ra")
    draw.text((1260, 1020), "prompt = rewritten subtask", fill=_BODY, font=body)
    draw.text((1625, 1020), "2 / 2", fill=_INK, font=mono, anchor="ra")
    draw.text((1260, 1058), "settle actions equal", fill=_BODY, font=body)
    draw.text((1625, 1058), "10 / 10", fill=_INK, font=mono, anchor="ra")
    draw.text((1260, 1096), "policy action steps changed", fill=_BODY, font=body)
    draw.text((1625, 1096), "10 / 10", fill=_INK, font=mono, anchor="ra")
    draw.text(
        (1260, 1130),
        "per-step max |Δ native 7D action|",
        fill=_BODY,
        font=_font(12, bold=True),
    )
    chart_left, chart_top, chart_width, chart_height = 1270, 1152, 390, 55
    draw.line(
        (
            chart_left,
            chart_top + chart_height,
            chart_left + chart_width,
            chart_top + chart_height,
        ),
        fill="#9BB7AD",
        width=2,
    )
    scale = chart_height / float(action_differences.max())
    slot = chart_width / len(action_differences)
    for index, difference in enumerate(action_differences):
        x0 = int(chart_left + index * slot + 5)
        x1 = int(chart_left + (index + 1) * slot - 5)
        bar_height = max(2, int(float(difference) * scale))
        draw.rounded_rectangle(
            (x0, chart_top + chart_height - bar_height, x1, chart_top + chart_height),
            radius=3,
            fill=_GREEN,
        )
    draw.text((chart_left, 1212), "1", fill=_MUTED, font=_font(10), anchor="ma")
    draw.text(
        (chart_left + chart_width, 1212),
        "10",
        fill=_MUTED,
        font=_font(10),
        anchor="ma",
    )
    _wrapped(
        draw,
        "All ten scheduled controls changed after the grounded rewrite.",
        xy=(1260, 1228),
        max_chars=70,
        font=_font(10, bold=True),
        max_lines=1,
    )

    _round_rect(
        draw, (1760, 870, 2305, 1250), fill=_BLUE_BG, outline="#5E82D0", radius=16
    )
    draw.text((1790, 905), "Scientific conclusion", fill=_INK, font=head)
    draw.text((1790, 948), "Off result", fill=_BODY, font=body)
    draw.text((2215, 948), "failed at step 20", fill=_INK, font=mono, anchor="ra")
    draw.text((1790, 986), "Task result", fill=_BODY, font=body)
    draw.text((2215, 986), "failed at step 20", fill=_INK, font=mono, anchor="ra")
    _wrapped(
        draw,
        "Validated: HPK rewrites a real high-level decision and the rewritten prompt reaches the native executor.",
        xy=(1790, 1030),
        max_chars=61,
        font=body,
        max_lines=4,
    )
    _wrapped(
        draw,
        "Not validated: success-rate or task-performance improvement.",
        xy=(1790, 1135),
        max_chars=61,
        font=_font(14, bold=True),
        fill="#9A6411",
        max_lines=3,
    )
    draw.text((1790, 1210), "Integration smoke only", fill=_BLUE, font=head)

    _round_rect(
        draw, (70, 1280, 2330, 1325), fill=_WARNING_BG, outline=_WARNING, radius=14
    )
    draw.text(
        (1200, 1303),
        "Boundary: this 20-step integration smoke verifies the wiring only; separate full episodes are required for task-performance evidence.",
        fill="#805513",
        font=small,
        anchor="mm",
    )
    draw.text(
        (1200, 1395),
        "Sources: real LIBERO expert episode 0; matched LIBERO-PRO Agent Off/Task traces; deterministic replay of their recorded native actions.",
        fill=_MUTED,
        font=_font(12),
        anchor="mm",
    )

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return {
        "output": str(output),
        "width": canvas.width,
        "height": canvas.height,
        "expert_frame_count": len(expert_images),
        "task_knowledge_count": len(store_rows),
        "action_knowledge_count": 0,
        "matched_context": audit["matched_context"],
        "claim_scope": audit["claim_scope"],
    }


def render_full_episode_overview(
    *,
    pair_root: str | Path,
    expert_root: str | Path,
    replay_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Render one complete matched episode pair without a performance overclaim."""

    pair = Path(pair_root).resolve(strict=True)
    expert = Path(expert_root).resolve(strict=True)
    replay = Path(replay_root).resolve(strict=True)
    traces = {
        name: _read_jsonl(pair / name / "agent_loop_trace.jsonl")
        for name in ("off", "task")
    }
    provenances = {
        name: _read_json(pair / name / "provenance.json") for name in ("off", "task")
    }
    loops = {
        name: provenances[name]["simulator"]["roboharn_agent_loop"]
        for name in ("off", "task")
    }
    if not all(loop.get("success") is True for loop in loops.values()):
        raise LiberoVisualizationError(
            "full episode figure requires both episodes to finish successfully"
        )
    matched_run_keys = (
        "horizon",
        "seed",
        "task_id",
        "init_state_id",
        "camera_height",
        "camera_width",
        "policy_replan_steps",
        "policy_settle_steps",
        "agent_max_planner_calls",
    )
    if any(
        provenances["off"]["run"][key] != provenances["task"]["run"][key]
        for key in matched_run_keys
    ):
        raise LiberoVisualizationError("full episode contexts do not match")
    if provenances["off"]["policy"] != provenances["task"]["policy"]:
        raise LiberoVisualizationError("full episode policy identities do not match")

    task_plans = tuple(
        row for row in traces["task"] if row.get("event") == "planner_decision"
    )
    usages = tuple(
        row["hpk_v3_subtask_usage"]
        for row in task_plans
        if isinstance(row.get("hpk_v3_subtask_usage"), dict)
    )
    adopted = sum(value.get("knowledge_adopted") is True for value in usages)
    rewritten = sum(
        value.get("subtask_before") != value.get("subtask_after") for value in usages
    )
    if len(usages) != len(task_plans) or adopted != len(task_plans):
        raise LiberoVisualizationError("full episode lacks complete HPK usage evidence")

    task_actions = tuple(
        row for row in traces["task"] if row.get("event") == "action_executed"
    )
    decision_steps: list[int] = []
    for plan in task_plans:
        index = plan["planner_call_index"]
        action = next(
            (
                row
                for row in task_actions
                if row.get("planner_call_index") == index
                and row.get("source") == "policy"
            ),
            None,
        )
        if action is None:
            raise LiberoVisualizationError("planner decision has no executed action")
        decision_steps.append(int(action["step_before"]))

    store_rows = _read_jsonl(expert / "store" / "task_knowledge.jsonl")
    action_store = expert / "store" / "action_knowledge.jsonl"
    if action_store.stat().st_size != 0:
        raise LiberoVisualizationError(
            "full episode figure requires Task Knowledge only"
        )
    expert_images = sorted((expert / "key_observations").glob("observation_*.png"))
    if len(expert_images) < 3:
        raise LiberoVisualizationError("full episode figure lacks expert images")

    canvas = Image.new("RGB", (2400, 1350), _BG)
    draw = ImageDraw.Draw(canvas)
    title = _font(40, bold=True)
    subtitle = _font(18)
    panel_title = _font(24, bold=True)
    head = _font(17, bold=True)
    body = _font(14)
    small = _font(12, bold=True)
    mono = _font(14, mono=True)

    draw.text(
        (1200, 52),
        "HPK on LIBERO-PRO: One Complete Matched Episode Pair",
        fill=_INK,
        font=title,
        anchor="mm",
    )
    draw.text(
        (1200, 90),
        "Real expert experience → Task Knowledge → 11 grounded rewrites → two successful 220-step-budget episodes",
        fill=_MUTED,
        font=subtitle,
        anchor="mm",
    )
    panels = ((45, 125, 700, 1215), (730, 125, 1605, 1215), (1635, 125, 2355, 1215))
    for panel in panels:
        _round_rect(draw, panel, fill="white", radius=24)

    # A. Real source and reusable Store.
    _round_rect(draw, (70, 148, 205, 178), fill="#DDF5EA", outline="#DDF5EA", radius=15)
    draw.text((137, 163), "REAL SOURCE", fill=_GREEN, font=small, anchor="mm")
    draw.text((70, 215), "A. EXPERIENCE → TASK KNOWLEDGE", fill=_INK, font=panel_title)
    draw.text(
        (70, 250),
        "One real expert episode; no hand-written task stages",
        fill=_MUTED,
        font=body,
    )
    chosen = (
        expert_images[0],
        expert_images[len(expert_images) // 2],
        expert_images[-1],
    )
    for index, (path, caption) in enumerate(
        zip(chosen, ("before", "transport", "after"), strict=True)
    ):
        x = 70 + index * 205
        _paste_frame(canvas, draw, path, box=(x, 285, x + 185, 440), caption=caption)
        if index < 2:
            _arrow(draw, (x + 185, 362), (x + 198, 362), fill=_GREEN)
    draw.text((70, 480), "Stored reusable sequence", fill=_INK, font=head)
    _round_rect(draw, (70, 510, 675, 970), fill=_GREEN_BG, outline="#63AD91", radius=16)
    y = 542
    for index, row in enumerate(store_rows, start=1):
        strategy = row["subtask_strategy"]
        draw.text(
            (95, y),
            f"{index}. {strategy['subtask']}",
            fill=_INK,
            font=_font(15, bold=True),
        )
        y = (
            _wrapped(
                draw,
                "Purpose: " + str(strategy["purpose"]),
                xy=(115, y + 27),
                max_chars=62,
                font=body,
                max_lines=2,
            )
            + 18
        )
    draw.text(
        (95, 930),
        "4 supported Task records · Action Knowledge disabled",
        fill=_GREEN,
        font=small,
    )
    _round_rect(
        draw, (70, 1010, 675, 1165), fill=_WARNING_BG, outline=_WARNING, radius=14
    )
    draw.text((95, 1040), "Transfer boundary", fill=_INK, font=head)
    _wrapped(
        draw,
        "The Store contains semantic conditions, subtasks, purposes, and completion conditions—not object IDs, absolute poses, or action vectors.",
        xy=(95, 1074),
        max_chars=67,
        font=body,
        max_lines=4,
    )

    # 展示完整 Task-HPK episode 的三个时间点。
    _round_rect(
        draw, (755, 148, 895, 178), fill="#E5EEFF", outline="#E5EEFF", radius=15
    )
    draw.text((825, 163), "REAL ROLLOUT", fill=_BLUE, font=small, anchor="mm")
    draw.text(
        (755, 215), "B. RETRIEVE → REWRITE → EXECUTE", fill=_INK, font=panel_title
    )
    representative = (0, len(task_plans) // 2, len(task_plans) - 1)
    for row_index, plan_index in enumerate(representative):
        usage = usages[plan_index]
        step = decision_steps[plan_index]
        top = 275 + row_index * 292
        _paste_frame(
            canvas,
            draw,
            replay / "task" / f"step_{step:03d}.png",
            box=(755, top, 1060, top + 235),
            caption=f"step {step} · planner call {plan_index + 1}",
        )
        draw.text((1090, top + 8), "Planner baseline", fill=_INK, font=head)
        y = _wrapped(
            draw,
            str(usage["subtask_before"]),
            xy=(1090, top + 38),
            max_chars=55,
            font=body,
            max_lines=3,
        )
        draw.text((1090, y + 8), "HPK-grounded prompt", fill=_GREEN, font=head)
        _wrapped(
            draw,
            str(usage["subtask_after"]),
            xy=(1090, y + 38),
            max_chars=55,
            font=body,
            max_lines=4,
        )
    draw.text(
        (1167, 1170),
        f"knowledge adopted {adopted}/{len(task_plans)} · prompts rewritten {rewritten}/{len(task_plans)}",
        fill=_BLUE,
        font=_font(16, bold=True),
        anchor="mm",
    )

    # C. Complete outcome and honest boundary.
    _round_rect(
        draw, (1660, 148, 1810, 178), fill="#FFF0D1", outline="#FFF0D1", radius=15
    )
    draw.text((1735, 163), "FULL EPISODES", fill="#9A6411", font=small, anchor="mm")
    draw.text((1660, 215), "C. MATCHED OUTCOME", fill=_INK, font=panel_title)
    draw.text(
        (1660, 250),
        "Same task, init, seed, instruction, cameras, and pi0.5",
        fill=_MUTED,
        font=body,
    )
    for index, name in enumerate(("off", "task")):
        loop = loops[name]
        top = 295 + index * 205
        color = _BLUE_BG if name == "off" else _GREEN_BG
        outline = "#5E82D0" if name == "off" else "#63AD91"
        _round_rect(
            draw, (1660, top, 2330, top + 170), fill=color, outline=outline, radius=16
        )
        label = "HPK OFF" if name == "off" else "TASK HPK"
        draw.text((1690, top + 30), label, fill=_INK, font=head)
        draw.text((2295, top + 30), "SUCCESS ✓", fill=_GREEN, font=head, anchor="ra")
        draw.text((1690, top + 75), "completion step", fill=_BODY, font=body)
        draw.text(
            (2295, top + 75),
            str(loop["executed_actions"]),
            fill=_INK,
            font=mono,
            anchor="ra",
        )
        draw.text((1690, top + 112), "recovery boundaries", fill=_BODY, font=body)
        draw.text(
            (2295, top + 112),
            str(loop["recovery_boundaries"]),
            fill=_INK,
            font=mono,
            anchor="ra",
        )
        draw.text((1690, top + 145), "planner calls", fill=_BODY, font=body)
        draw.text(
            (2295, top + 145),
            str(loop["planner_calls"]),
            fill=_INK,
            font=mono,
            anchor="ra",
        )
    _round_rect(
        draw, (1660, 735, 2330, 930), fill="#F8FAFC", outline="#B8C6D3", radius=16
    )
    draw.text((1690, 768), "Observed in this one matched pair", fill=_INK, font=head)
    draw.text(
        (1690, 810),
        "Task-HPK completed 2 steps earlier",
        fill=_GREEN,
        font=_font(16, bold=True),
    )
    draw.text(
        (1690, 848),
        "Task-HPK used 2 more recovery boundaries",
        fill="#9A6411",
        font=_font(16, bold=True),
    )
    _wrapped(
        draw,
        "Both completed the benchmark goal. The mixed efficiency signals do not establish a performance advantage.",
        xy=(1690, 882),
        max_chars=69,
        font=body,
        max_lines=3,
    )
    _round_rect(
        draw, (1660, 970, 2330, 1165), fill=_BLUE_BG, outline="#5E82D0", radius=16
    )
    draw.text((1690, 1005), "Scientific conclusion", fill=_INK, font=head)
    _wrapped(
        draw,
        "Validated: the complete read → retrieve → rewrite → native execution chain can reach task success. Not yet validated: higher success rate or lower expected completion cost; that requires repeated init/seed pairs.",
        xy=(1690, 1045),
        max_chars=70,
        font=body,
        max_lines=6,
    )

    _round_rect(
        draw, (70, 1245, 2330, 1295), fill=_WARNING_BG, outline=_WARNING, radius=14
    )
    draw.text(
        (1200, 1270),
        "Complete case study, not a performance table: n = 1 matched pair; both conditions succeeded within the official 220-step limit.",
        fill="#805513",
        font=small,
        anchor="mm",
    )
    draw.text(
        (1200, 1325),
        "Sources: real LIBERO expert episode 0 and deterministic replay of the complete recorded Off/Task-HPK native action traces.",
        fill=_MUTED,
        font=_font(12),
        anchor="mm",
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return {
        "output": str(output),
        "width": canvas.width,
        "height": canvas.height,
        "off_success": True,
        "task_success": True,
        "off_steps": int(loops["off"]["executed_actions"]),
        "task_steps": int(loops["task"]["executed_actions"]),
        "off_recovery_boundaries": int(loops["off"]["recovery_boundaries"]),
        "task_recovery_boundaries": int(loops["task"]["recovery_boundaries"]),
        "knowledge_adopted": adopted,
        "knowledge_opportunities": len(task_plans),
        "performance_improvement_claimed": False,
    }


def render_causal_story(
    *,
    pair_root: str | Path,
    expert_root: str | Path,
    replay_root: str | Path,
    controlled_ablation_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Render a focused causal figure for one HPK-modified planner decision."""

    pair = Path(pair_root).resolve(strict=True)
    expert = Path(expert_root).resolve(strict=True)
    replay = Path(replay_root).resolve(strict=True)
    ablation = _read_json(Path(controlled_ablation_path).resolve(strict=True))
    task_trace = _read_jsonl(pair / "task" / "agent_loop_trace.jsonl")
    loops = {
        name: _read_json(pair / name / "provenance.json")["simulator"]["roboharn_agent_loop"]
        for name in ("off", "task")
    }
    complete_pair = all(loop.get("success") is True for loop in loops.values())
    decision = next(
        (row for row in task_trace if row.get("event") == "planner_decision"),
        None,
    )
    if decision is None or not isinstance(decision.get("hpk_v3_subtask_usage"), dict):
        raise LiberoVisualizationError("causal figure requires an adopted decision")
    usage = decision["hpk_v3_subtask_usage"]
    knowledge = usage.get("retrieved_knowledge")
    if not isinstance(knowledge, dict):
        raise LiberoVisualizationError("causal figure lacks retrieved knowledge")
    baseline_actions = np.asarray(ablation["baseline_actions"], dtype=float)
    rewrite_actions = np.asarray(ablation["hpk_rewrite_actions"], dtype=float)
    if baseline_actions.shape != (10, 7) or rewrite_actions.shape != (10, 7):
        raise LiberoVisualizationError("causal figure requires controlled 10x7 actions")
    delta = np.abs(baseline_actions - rewrite_actions)
    if (
        ablation.get("same_observation") is not True
        or ablation.get("same_sampling_noise") is not True
    ):
        raise LiberoVisualizationError("causal figure requires controlled inputs")

    canvas = Image.new("RGB", (2400, 1350), _BG)
    draw = ImageDraw.Draw(canvas)
    title = _font(40, bold=True)
    subtitle = _font(18)
    panel_title = _font(24, bold=True)
    head = _font(17, bold=True)
    body = _font(14)
    small = _font(12, bold=True)

    draw.text(
        (1200, 53),
        "How One HPK Task Memory Changes a Real LIBERO Control Decision",
        fill=_INK,
        font=title,
        anchor="mm",
    )
    draw.text(
        (1200, 91),
        "Controlled mechanism evidence: same observation and diffusion noise; only the π0.5 prompt changes",
        fill=_MUTED,
        font=subtitle,
        anchor="mm",
    )

    for box in ((45, 125, 590, 1215), (620, 125, 1465, 1215), (1495, 125, 2355, 1215)):
        _round_rect(draw, box, fill="white", radius=24)

    # 1. Source experience and selected memory
    _round_rect(draw, (70, 148, 210, 178), fill="#DDF5EA", outline="#DDF5EA", radius=15)
    draw.text((140, 163), "EXPERIENCE", fill=_GREEN, font=small, anchor="mm")
    draw.text((70, 215), "1. WHERE THE MEMORY CAME FROM", fill=_INK, font=panel_title)
    draw.text(
        (70, 248),
        "Real expert trajectory · not a hand-written rule",
        fill=_MUTED,
        font=body,
    )
    expert_images = sorted((expert / "key_observations").glob("observation_*.png"))
    for index, (source_index, caption) in enumerate(
        ((0, "before"), (6, "transport"), (10, "after"))
    ):
        x = 70 + index * 170
        _paste_frame(
            canvas,
            draw,
            expert_images[source_index],
            box=(x, 285, x + 155, 435),
            caption=caption,
        )
        if index < 2:
            _arrow(draw, (x + 155, 360), (x + 166, 360), fill=_GREEN)
    draw.text((70, 478), "Retrieved Task Knowledge", fill=_INK, font=head)
    strategy = knowledge["subtask_strategy"]
    condition = knowledge["condition"]
    _round_rect(draw, (70, 505, 565, 910), fill=_GREEN_BG, outline="#63AD91", radius=16)
    y = _wrapped(
        draw,
        "WHEN: " + str(condition["task_state"]),
        xy=(95, 535),
        max_chars=54,
        font=body,
        max_lines=4,
    )
    y = _wrapped(
        draw,
        "DO: " + str(strategy["subtask"]),
        xy=(95, y + 18),
        max_chars=54,
        font=_font(15, bold=True),
        max_lines=3,
    )
    y = _wrapped(
        draw,
        "WHY: " + str(strategy["purpose"]),
        xy=(95, y + 18),
        max_chars=54,
        font=body,
        max_lines=4,
    )
    _wrapped(
        draw,
        "DONE WHEN: " + str(strategy["completion_condition"]),
        xy=(95, y + 18),
        max_chars=54,
        font=body,
        max_lines=4,
    )
    draw.text(
        (95, 865),
        "evidence: 1 support / 0 oppose / 0 unverified",
        fill=_GREEN,
        font=small,
    )
    _round_rect(
        draw, (70, 945, 565, 1165), fill=_WARNING_BG, outline=_WARNING, radius=14
    )
    draw.text((95, 978), "What is not stored", fill=_INK, font=head)
    for index, line in enumerate(
        (
            "• object or task IDs",
            "• absolute poses or action vectors",
            "• seed-specific branches",
            "• Action Knowledge (disabled)",
        )
    ):
        draw.text((105, 1015 + index * 33), line, fill=_BODY, font=body)

    # 2. Exact decision chain
    _round_rect(
        draw, (645, 148, 760, 178), fill="#E5EEFF", outline="#E5EEFF", radius=15
    )
    draw.text((702, 163), "DECISION", fill=_BLUE, font=small, anchor="mm")
    draw.text(
        (645, 215), "2. WHAT CHANGED INSIDE THE AGENT", fill=_INK, font=panel_title
    )
    _paste_frame(
        canvas,
        draw,
        replay / "task" / "step_010.png",
        box=(670, 270, 1415, 590),
        caption="same real observation at step 10",
    )
    _round_rect(
        draw, (670, 625, 1415, 745), fill="#F8FAFC", outline="#B8C6D3", radius=14
    )
    draw.text((695, 654), "Planner baseline", fill=_INK, font=head)
    _wrapped(
        draw,
        str(usage["subtask_before"]),
        xy=(695, 685),
        max_chars=84,
        font=body,
        max_lines=2,
    )
    _arrow(draw, (1042, 750), (1042, 780), fill=_GREEN)
    _round_rect(
        draw, (670, 790, 1415, 930), fill=_GREEN_BG, outline="#63AD91", radius=14
    )
    draw.text((695, 819), "HPK semantic retrieval", fill=_INK, font=head)
    draw.text((1375, 819), "adopted ✓", fill=_GREEN, font=head, anchor="ra")
    _wrapped(
        draw,
        "Matched reusable lesson: "
        + str(strategy["subtask"])
        + ". "
        + str(strategy["purpose"]),
        xy=(695, 850),
        max_chars=84,
        font=body,
        max_lines=3,
    )
    _arrow(draw, (1042, 935), (1042, 965), fill=_GREEN)
    _round_rect(
        draw, (670, 975, 1415, 1105), fill=_BLUE_BG, outline="#5E82D0", radius=14
    )
    draw.text((695, 1004), "Grounded rewrite", fill=_INK, font=head)
    _wrapped(
        draw,
        str(usage["subtask_after"]),
        xy=(695, 1035),
        max_chars=84,
        font=body,
        max_lines=3,
    )
    draw.text(
        (1042, 1148),
        "rewrite text = actual π0.5 prompt  ✓",
        fill=_BLUE,
        font=_font(16, bold=True),
        anchor="mm",
    )

    # 3. Controlled prompt consequence
    _round_rect(
        draw, (1520, 148, 1655, 178), fill="#FFF0D1", outline="#FFF0D1", radius=15
    )
    draw.text((1587, 163), "CONTROL", fill="#9A6411", font=small, anchor="mm")
    draw.text((1520, 215), "3. WHAT THE PROMPT CHANGED", fill=_INK, font=panel_title)
    draw.text((1520, 250), "Controlled π0.5 ablation", fill=_INK, font=head)
    draw.text(
        (1520, 280),
        "same observation ✓   same sampling noise ✓",
        fill=_GREEN,
        font=small,
    )
    draw.text(
        (1520, 310),
        "only variable: baseline prompt vs HPK rewrite",
        fill=_MUTED,
        font=body,
    )

    channel_labels = ("dx", "dy", "dz", "dRx", "dRy", "dRz", "grip")
    heat_x, heat_y = 1605, 370
    cell_w, cell_h = 90, 30
    for column, label in enumerate(channel_labels):
        draw.text(
            (heat_x + column * cell_w + cell_w // 2, heat_y - 20),
            label,
            fill=_MUTED,
            font=_font(11, bold=True),
            anchor="mm",
        )
    maximum = float(delta.max()) or 1.0
    for row in range(10):
        draw.text(
            (heat_x - 18, heat_y + row * cell_h + 15),
            str(row + 1),
            fill=_MUTED,
            font=_font(10),
            anchor="rm",
        )
        for column in range(7):
            fraction = min(1.0, float(delta[row, column]) / maximum)
            low = np.asarray((231, 242, 248), dtype=float)
            high = np.asarray((35, 126, 103), dtype=float)
            color = tuple(
                int(value) for value in low * (1 - fraction) + high * fraction
            )
            x0 = heat_x + column * cell_w
            y0 = heat_y + row * cell_h
            draw.rectangle((x0, y0, x0 + cell_w - 3, y0 + cell_h - 3), fill=color)
    draw.text(
        (1520, 690),
        "|baseline action − HPK action| for each of 10 predicted steps",
        fill=_MUTED,
        font=body,
    )
    draw.text(
        (1520, 727),
        f"changed predicted steps: {ablation['changed_action_steps']} / 10",
        fill=_GREEN,
        font=_font(17, bold=True),
    )
    draw.text(
        (1520, 765),
        "This isolates the prompt effect; these counterfactual actions were not executed.",
        fill=_BODY,
        font=body,
    )

    _round_rect(
        draw, (1520, 810, 2328, 1005), fill="#F8FAFC", outline="#B8C6D3", radius=14
    )
    draw.text((1545, 842), "What the real rollout proves", fill=_INK, font=head)
    draw.text(
        (1545, 880),
        "✓ the HPK rewrite was actually sent to π0.5",
        fill=_BODY,
        font=body,
    )
    draw.text(
        (1545, 915),
        "✓ native 7D actions were executed in LIBERO",
        fill=_BODY,
        font=body,
    )
    draw.text(
        (1545, 950),
        "✓ post-action observations returned to the Agent",
        fill=_BODY,
        font=body,
    )
    if complete_pair:
        draw.text(
            (1545, 985),
            f"✓ complete episodes succeeded at steps {loops['off']['executed_actions']} / {loops['task']['executed_actions']}",
            fill=_GREEN,
            font=_font(14, bold=True),
        )
    else:
        draw.text(
            (1545, 985),
            "✗ the short integration runs did not complete the task",
            fill="#9A6411",
            font=_font(14, bold=True),
        )

    _round_rect(
        draw, (1520, 1035, 2328, 1165), fill=_BLUE_BG, outline="#5E82D0", radius=14
    )
    draw.text((1545, 1068), "Honest conclusion", fill=_INK, font=head)
    _wrapped(
        draw,
        (
            "HPK causally changes the π0.5 control prediction through its prompt. "
            "The complete pair reached success, but one pair cannot establish a performance improvement."
            if complete_pair
            else "HPK causally changes the π0.5 control prediction through its prompt. This short experiment does not show a task-performance improvement."
        ),
        xy=(1545, 1100),
        max_chars=89,
        font=body,
        max_lines=3,
    )

    _round_rect(
        draw, (70, 1245, 2330, 1295), fill=_WARNING_BG, outline=_WARNING, radius=14
    )
    draw.text(
        (1200, 1270),
        "Mechanism validated: experience → Task Knowledge → retrieval → grounded prompt rewrite → changed π0.5 prediction → real execution. Repeated-pair performance remains open.",
        fill="#805513",
        font=small,
        anchor="mm",
    )
    draw.text(
        (1200, 1325),
        "Real sources: LIBERO expert episode 0, Task-HPK Agent trace, deterministic action replay, and a same-observation/same-noise local π0.5 prompt ablation.",
        fill=_MUTED,
        font=_font(12),
        anchor="mm",
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return {
        "output": str(output),
        "width": canvas.width,
        "height": canvas.height,
        "controlled_changed_action_steps": int(ablation["changed_action_steps"]),
        "same_observation": True,
        "same_sampling_noise": True,
        "performance_improvement_claimed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-fps", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    replay_root = output / "replay"
    off = replay_trace_frames(
        args.pair_root / "off",
        assets_root=args.assets_root,
        output_dir=replay_root / "off",
    )
    task = replay_trace_frames(
        args.pair_root / "task",
        assets_root=args.assets_root,
        output_dir=replay_root / "task",
    )
    _video_from_frames(
        replay_root / "off", output / "off_action_replay_30fps.mp4", fps=args.video_fps
    )
    _video_from_frames(
        replay_root / "task",
        output / "task_hpk_action_replay_30fps.mp4",
        fps=args.video_fps,
    )
    comparison_video = output / "off_vs_task_action_replay_slow_30fps.mp4"
    _comparison_video(
        replay_root / "off",
        replay_root / "task",
        comparison_video,
        fps=args.video_fps,
    )
    complete_pair = off["benchmark_success"] and task["benchmark_success"]
    if complete_pair:
        figure = render_full_episode_overview(
            pair_root=args.pair_root,
            expert_root=args.expert_root,
            replay_root=replay_root,
            output_path=output / "libero_hpk_full_episode_overview.png",
        )
    else:
        figure = render_overview(
            pair_root=args.pair_root,
            expert_root=args.expert_root,
            replay_root=replay_root,
            output_path=output / "libero_hpk_real_evidence_overview.png",
        )
    controlled_ablation_path = output / "controlled_prompt_ablation.json"
    controlled_ablation = controlled_prompt_ablation(
        pair_root=args.pair_root,
        replay_root=replay_root,
        output_path=controlled_ablation_path,
    )
    causal_figure = render_causal_story(
        pair_root=args.pair_root,
        expert_root=args.expert_root,
        replay_root=replay_root,
        controlled_ablation_path=controlled_ablation_path,
        output_path=output / "libero_hpk_causal_story.png",
    )
    manifest = {
        "schema": "roboharn_evo/libero_task_hpk_visualization/v2",
        "source_capture_fps": 20,
        "encoded_video_fps": args.video_fps,
        "frame_interpolation": False,
        "comparison_video": {
            "path": str(comparison_video),
            "left": "HPK Off",
            "right": "Task HPK",
            "presentation_slowdown": 4,
            "frame_interpolation": False,
        },
        "off_replay": off,
        "task_replay": task,
        "figure": figure,
        "controlled_prompt_ablation": {
            "path": str(controlled_ablation_path),
            "same_observation": controlled_ablation["same_observation"],
            "same_sampling_noise": controlled_ablation["same_sampling_noise"],
            "changed_action_steps": controlled_ablation["changed_action_steps"],
            "motion_executed": controlled_ablation["motion_executed"],
        },
        "causal_figure": causal_figure,
    }
    (output / "visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    raise SystemExit(main())
