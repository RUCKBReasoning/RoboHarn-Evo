#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (
    REPO_ROOT,
    REPO_ROOT / "policy",
    REPO_ROOT / "description" / "utils",
    REPO_ROOT / "script",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions
from policy.roboharn_evo.agent.environment import EnvSnapshot, RMBenchEnvAdapter
from policy.roboharn_evo.agent.recovery import RecoveryToolCall, RecoveryToolDispatcher


PANEL_PADDING = 12
TEXT_COLOR = (255, 255, 255)
HEADER_COLOR = (220, 220, 220)
PANEL_COLOR = (0, 0, 0)
THUMB_SIZE = (320, 240)
TEXT_WIDTH = 460


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Directly test RoboHarn-Evo recovery tools in a real RMBench environment.")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-seed", type=int, default=None)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--skip-expert-check", action="store_true")
    parser.add_argument("--max-seed-search", type=int, default=20)
    parser.add_argument(
        "--expert-tcp-replay",
        action="store_true",
        help=(
            "Diagnostic-only mode: capture the first expert-selected pre-grasp/grasp "
            "TCP targets on a solvable reset, then replay those targets through the "
            "RoboHarn-Evo recovery executor on a fresh reset of the same seed."
        ),
    )
    parser.add_argument(
        "--expert-grasp-index",
        type=int,
        default=0,
        help="Zero-based expert grasp call to replay when --expert-tcp-replay is enabled.",
    )
    parser.add_argument("--tool-call", action="append", default=[])
    parser.add_argument("--tools", nargs="*", default=["reobserve_scene", "retreat_arm", "reobserve_scene"])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "eval_result" / "tcm_recovery_tool_tests")
    parser.add_argument("--save-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--strict-tools", action="store_true")
    parser.add_argument("--render-freq", type=int, default=0)
    parser.add_argument("--no-sapien-test", action="store_true")
    return parser.parse_args()


def class_decorator(task_name: str) -> Any:
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        return env_class()
    except Exception as exc:
        raise SystemExit(f"No Task: {task_name}") from exc


def get_embodiment_config(robot_file: str) -> dict[str, Any]:
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_task_args(task_name: str, task_config: str, render_freq: int) -> dict[str, Any]:
    task_config_path = REPO_ROOT / "task_config" / f"{task_config}.yml"
    with task_config_path.open("r", encoding="utf-8") as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)

    task_args["task_name"] = task_name
    task_args["task_config"] = task_config
    task_args["policy_name"] = "roboharn_evo_recovery_tool_test"
    task_args["ckpt_setting"] = "recovery_tool"
    task_args["eval_mode"] = True
    task_args["render_freq"] = int(render_freq)
    task_args["eval_video_log"] = False
    task_args["eval_video_save_dir"] = None

    embodiment_type = task_args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_item: str) -> str:
        robot_file = embodiment_types[embodiment_item]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"No embodiment file for {embodiment_item}")
        return robot_file

    camera_config_path = CONFIGS_PATH + "_camera_config.yml"
    with open(camera_config_path, "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = camera_config[head_camera_type]["h"]
    task_args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])
    return task_args


def base_seed(args: argparse.Namespace) -> int:
    if args.raw_seed is not None:
        return int(args.raw_seed)
    return 100000 * (1 + int(args.seed))


def find_stable_seed(task_name: str, task_args: dict[str, Any], episode_id: int, start_seed: int, max_seed_search: int) -> tuple[int, dict[str, Any] | None]:
    for offset in range(max_seed_search):
        seed = start_seed + offset
        task_env = class_decorator(task_name)
        try:
            task_env.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **task_args)
            episode_info = task_env.play_once()
            if bool(getattr(task_env, "plan_success", False)) and bool(task_env.check_success()):
                return seed, episode_info
        except UnStableError:
            pass
        finally:
            try:
                task_env.close_env()
            except Exception:
                pass
    raise RuntimeError(f"No stable expert seed found in [{start_seed}, {start_seed + max_seed_search})")


def _pose7(value: Any) -> list[float] | None:
    try:
        pose = np.asarray(value, dtype=np.float64).reshape(7)
    except Exception:
        return None
    if not np.all(np.isfinite(pose)):
        return None
    quat_norm = float(np.linalg.norm(pose[3:]))
    if quat_norm <= 1e-8:
        return None
    pose = pose.copy()
    pose[3:] /= quat_norm
    return [round(float(item), 8) for item in pose]


def _expert_actor_label(actor: Any) -> str:
    for holder in (actor, getattr(actor, "actor", None)):
        if holder is None:
            continue
        get_name = getattr(holder, "get_name", None)
        if not callable(get_name):
            continue
        try:
            name = str(get_name() or "").strip()
        except Exception:
            name = ""
        if name:
            return name
    return type(actor).__name__


def capture_expert_grasps(
    task_name: str,
    task_args: dict[str, Any],
    episode_id: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Capture expert targets for diagnostics without exposing them to formal runtime."""
    task_env = class_decorator(task_name)
    captures: list[dict[str, Any]] = []
    episode_info: dict[str, Any] | None = None
    try:
        task_env.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **task_args)
        original_choose_grasp_pose = task_env.choose_grasp_pose

        def capture_choose_grasp_pose(
            actor: Any,
            arm_tag: str,
            pre_dis: float = 0.1,
            target_dis: float = 0.0,
            contact_point_id: Any = None,
        ) -> Any:
            result = original_choose_grasp_pose(
                actor,
                arm_tag=arm_tag,
                pre_dis=pre_dis,
                target_dis=target_dis,
                contact_point_id=contact_point_id,
            )
            pre_pose = None
            grasp_pose = None
            if isinstance(result, (list, tuple)) and len(result) == 2:
                pre_pose = _pose7(result[0])
                grasp_pose = _pose7(result[1])
            captures.append(
                {
                    "capture_index": len(captures),
                    "actor": _expert_actor_label(actor),
                    "arm": str(arm_tag),
                    "pre_grasp_distance_m": float(pre_dis),
                    "target_distance_m": float(target_dis),
                    "requested_contact_point_id": contact_point_id,
                    "pre_grasp_pose": pre_pose,
                    "grasp_pose": grasp_pose,
                }
            )
            return result

        # Instance assignment intentionally affects only this diagnostic expert run.
        task_env.choose_grasp_pose = capture_choose_grasp_pose
        episode_info = task_env.play_once()
        if not bool(getattr(task_env, "plan_success", False)) or not bool(task_env.check_success()):
            raise RuntimeError(
                "expert replay source episode did not finish successfully: "
                f"plan_success={getattr(task_env, 'plan_success', None)}, "
                f"check_success={task_env.check_success()}"
            )
        return captures, episode_info
    finally:
        try:
            task_env.close_env()
        except Exception:
            pass


def expert_tcp_replay_calls(capture: dict[str, Any]) -> list[RecoveryToolCall]:
    arm = str(capture.get("arm", "")).strip().lower()
    pre_pose = _pose7(capture.get("pre_grasp_pose"))
    grasp_pose = _pose7(capture.get("grasp_pose"))
    if arm not in {"left", "right"}:
        raise ValueError(f"expert grasp has unsupported arm: {arm!r}")
    if pre_pose is None or grasp_pose is None:
        raise ValueError("expert grasp capture does not contain finite pre-grasp and grasp poses")
    return [
        RecoveryToolCall(tool_name="open_gripper", args={"arm": arm}),
        RecoveryToolCall(
            tool_name="move_ee_to_pose",
            args={
                "arm": arm,
                "target_pose": pre_pose,
                "max_translation": 0.12,
                "steps": 8,
            },
        ),
        RecoveryToolCall(
            tool_name="move_ee_to_pose",
            args={
                "arm": arm,
                "target_pose": grasp_pose,
                "max_translation": 0.04,
                "steps": 4,
            },
        ),
    ]


def choose_instruction(task_env: Any, args: argparse.Namespace, episode_info: dict[str, Any] | None) -> str:
    if args.instruction:
        instruction = args.instruction
    elif episode_info and isinstance(episode_info.get("info"), dict):
        descriptions = generate_episode_descriptions(args.task_name, [episode_info["info"]], 1)
        choices = descriptions[0].get(args.instruction_type) or descriptions[0].get("unseen") or []
        instruction = str(np.random.choice(choices)) if choices else ""
    else:
        instruction = ""
    try:
        task_env.set_instruction(instruction=instruction)
    except Exception:
        pass
    return instruction


def build_recovery_calls(args: argparse.Namespace) -> list[RecoveryToolCall]:
    calls: list[RecoveryToolCall] = []
    for raw_call in args.tool_call:
        payload = json.loads(raw_call)
        if not isinstance(payload, dict):
            raise ValueError(f"--tool-call must decode to an object: {raw_call}")
        tool_name = str(payload.get("tool_name") or "").strip()
        call_args = payload.get("args") or {}
        if not tool_name:
            raise ValueError(f"--tool-call missing tool_name: {raw_call}")
        if not isinstance(call_args, dict):
            raise ValueError(f"--tool-call args must be object: {raw_call}")
        calls.append(RecoveryToolCall(tool_name=tool_name, args=call_args))
    if calls:
        return calls
    return [RecoveryToolCall(tool_name=str(tool), args={}) for tool in args.tools]


def snapshot_from_env(task_env: Any) -> EnvSnapshot:
    return RMBenchEnvAdapter.from_env(task_env, task_env.get_obs())


def snapshot_summary(snapshot: EnvSnapshot) -> dict[str, Any]:
    return {
        "step_count": snapshot.step_count,
        "step_limit": snapshot.step_limit,
        "eval_success": snapshot.eval_success,
        "check_success": snapshot.check_success,
        "max_reward": snapshot.max_reward,
        "instruction": snapshot.instruction,
        "joint_dim": int(snapshot.joint_vector.size),
        "left_endpose": snapshot.left_endpose.astype(float).tolist(),
        "right_endpose": snapshot.right_endpose.astype(float).tolist(),
    }


def measure_progress(before: EnvSnapshot, after: EnvSnapshot) -> dict[str, Any]:
    return {
        "joint_delta": float(np.linalg.norm(after.joint_vector - before.joint_vector)) if after.joint_vector.shape == before.joint_vector.shape else None,
        "left_ee_delta": float(np.linalg.norm(after.left_endpose[:3] - before.left_endpose[:3])),
        "right_ee_delta": float(np.linalg.norm(after.right_endpose[:3] - before.right_endpose[:3])),
        "head_image_delta_mean": float(np.mean(np.abs(after.head_rgb.astype(np.float32) - before.head_rgb.astype(np.float32)))) if after.head_rgb.shape == before.head_rgb.shape else None,
    }


def normalize_rgb(image: np.ndarray) -> Image.Image:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] >= 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        max_value = float(np.nanmax(array)) if array.size else 1.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)[:80]


def save_snapshot_images(snapshot: EnvSnapshot, frames_dir: Path, index: int, label: str) -> dict[str, str]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for camera_name, image in (
        ("head", snapshot.head_rgb),
        ("left", snapshot.left_rgb),
        ("right", snapshot.right_rgb),
    ):
        out_path = frames_dir / f"{index:03d}_{safe_label(label)}_{camera_name}.png"
        normalize_rgb(image).save(out_path)
        outputs[camera_name] = str(out_path)
    return outputs


def get_font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        current = ""
        for word in paragraph.split(" "):
            candidate = word if not current else f"{current} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if current and bbox[2] - bbox[0] > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines or [""]


def state_text(state: dict[str, Any]) -> list[str]:
    event = state.get("event", "")
    result = state.get("result") or {}
    summary = state.get("snapshot_summary") or {}
    lines = [
        f"#{state['index']} {event}",
        f"tool: {state.get('tool_name', '-')}",
        f"args: {json.dumps(state.get('args', {}), ensure_ascii=False)}",
    ]
    if result:
        lines.extend([
            f"success: {result.get('success')}",
            f"message: {result.get('message', '')}",
        ])
    lines.extend([
        f"step: {summary.get('step_count')} / {summary.get('step_limit')}",
        f"reward: {summary.get('max_reward')}",
        f"check_success: {summary.get('check_success')}",
    ])
    progress = state.get("progress") or {}
    if progress:
        lines.append(f"progress: {json.dumps(progress, ensure_ascii=False)}")
    return lines


def make_contact_sheet(states: list[dict[str, Any]], output_path: Path) -> None:
    font = get_font(16)
    header_font = get_font(18)
    row_h = THUMB_SIZE[1]
    col_w = THUMB_SIZE[0]
    width = col_w * 3 + TEXT_WIDTH
    height = row_h * len(states)
    sheet = Image.new("RGB", (width, height), PANEL_COLOR)
    for row, state in enumerate(states):
        y = row * row_h
        snapshot: EnvSnapshot = state["snapshot"]
        for col, (camera_name, image) in enumerate((
            ("head", snapshot.head_rgb),
            ("left", snapshot.left_rgb),
            ("right", snapshot.right_rgb),
        )):
            thumb = normalize_rgb(image).resize(THUMB_SIZE)
            sheet.paste(thumb, (col * col_w, y))
            draw = ImageDraw.Draw(sheet)
            draw.rectangle((col * col_w, y, col * col_w + col_w, y + 24), fill=(0, 0, 0))
            draw.text((col * col_w + 8, y + 3), camera_name, fill=HEADER_COLOR, font=header_font)
        draw = ImageDraw.Draw(sheet)
        x = col_w * 3 + PANEL_PADDING
        text_y = y + PANEL_PADDING
        for line in state_text(state):
            for wrapped in wrap_text(draw, line, font, TEXT_WIDTH - PANEL_PADDING * 2):
                draw.text((x, text_y), wrapped, fill=TEXT_COLOR, font=font)
                text_y += 20
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def maybe_save_gif(states: list[dict[str, Any]], output_path: Path) -> str | None:
    try:
        import imageio.v2 as imageio
    except Exception:
        print("[warn] imageio unavailable; skipping GIF", flush=True)
        return None
    frames = []
    for state in states:
        snapshot: EnvSnapshot = state["snapshot"]
        frames.append(np.asarray(normalize_rgb(snapshot.head_rgb).resize(THUMB_SIZE)))
    imageio.mimsave(output_path, frames, duration=0.8)
    return str(output_path)


def jsonable_state(state: dict[str, Any]) -> dict[str, Any]:
    clone = {k: v for k, v in state.items() if k != "snapshot"}
    return clone


def write_events(path: Path, states: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for state in states:
            f.write(json.dumps(jsonable_state(state), ensure_ascii=False) + "\n")


def result_to_dict(result: Any) -> dict[str, Any]:
    return {
        "tool_name": result.tool_name,
        "success": bool(result.success),
        "message": str(result.message),
        "details": dict(result.details or {}),
    }


def run_recovery_sequence(task_env: Any, calls: list[RecoveryToolCall], output_dir: Path, save_frames: bool, save_gif: bool, strict_tools: bool) -> dict[str, Any]:
    dispatcher = RecoveryToolDispatcher()
    frames_dir = output_dir / "frames"
    states: list[dict[str, Any]] = []

    initial_snapshot = snapshot_from_env(task_env)
    initial_state = {
        "index": 0,
        "event": "initial",
        "tool_name": None,
        "args": {},
        "snapshot": initial_snapshot,
        "snapshot_summary": snapshot_summary(initial_snapshot),
        "result": None,
        "progress": None,
    }
    if save_frames:
        initial_state["frames"] = save_snapshot_images(initial_snapshot, frames_dir, 0, "initial")
    states.append(initial_state)
    latest_snapshot = initial_snapshot

    for index, call in enumerate(calls, start=1):
        before_snapshot = latest_snapshot
        result = dispatcher.dispatch(call, task_env=task_env, latest_snapshot=latest_snapshot)
        latest_snapshot = dispatcher.latest_snapshot or snapshot_from_env(task_env)
        state = {
            "index": index,
            "event": "recovery_tool",
            "tool_name": call.tool_name,
            "args": dict(call.args),
            "snapshot": latest_snapshot,
            "snapshot_summary": snapshot_summary(latest_snapshot),
            "result": result_to_dict(result),
            "progress": measure_progress(before_snapshot, latest_snapshot),
        }
        if save_frames:
            state["frames"] = save_snapshot_images(latest_snapshot, frames_dir, index, call.tool_name)
        states.append(state)
        if strict_tools and not result.success:
            break

    contact_sheet = output_dir / "contact_sheet.png"
    make_contact_sheet(states, contact_sheet)
    gif_path = maybe_save_gif(states, output_dir / "recovery_sequence.gif") if save_gif else None
    events_path = output_dir / "events.jsonl"
    write_events(events_path, states)
    return {
        "states": states,
        "contact_sheet": str(contact_sheet),
        "events_jsonl": str(events_path),
        "frames_dir": str(frames_dir),
        "gif": gif_path,
    }


def main() -> int:
    args = parse_args()
    if not args.no_sapien_test:
        from test_render import Sapien_TEST

        Sapien_TEST()

    selected_seed = base_seed(args)
    episode_info: dict[str, Any] | None = None
    task_args = load_task_args(args.task_name, args.task_config, args.render_freq)

    if not args.skip_expert_check:
        selected_seed, episode_info = find_stable_seed(
            args.task_name,
            task_args,
            args.episode_id,
            selected_seed,
            args.max_seed_search,
        )

    expert_replay_capture: dict[str, Any] | None = None
    if args.expert_tcp_replay:
        captures, captured_episode_info = capture_expert_grasps(
            args.task_name,
            task_args,
            args.episode_id,
            selected_seed,
        )
        if not captures:
            raise RuntimeError("expert episode produced no grasp targets to replay")
        if args.expert_grasp_index < 0 or args.expert_grasp_index >= len(captures):
            raise IndexError(
                f"--expert-grasp-index {args.expert_grasp_index} is outside "
                f"[0, {len(captures)})"
            )
        expert_replay_capture = captures[args.expert_grasp_index]
        if captured_episode_info is not None:
            episode_info = captured_episode_info

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / args.task_name / args.task_config / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    task_env = class_decorator(args.task_name)
    try:
        task_env.setup_demo(now_ep_num=args.episode_id, seed=selected_seed, is_test=True, **task_args)
        instruction = choose_instruction(task_env, args, episode_info)
        calls = (
            expert_tcp_replay_calls(expert_replay_capture)
            if expert_replay_capture is not None
            else build_recovery_calls(args)
        )
        dispatcher = RecoveryToolDispatcher()
        available_tools = dispatcher.available_tools(task_env)
        results = run_recovery_sequence(
            task_env,
            calls,
            output_dir,
            save_frames=args.save_frames,
            save_gif=args.save_gif,
            strict_tools=args.strict_tools,
        )
        final_snapshot = results["states"][-1]["snapshot"]
        expert_replay_result: dict[str, Any] | None = None
        if expert_replay_capture is not None:
            movement_events = [
                event
                for event in results["states"]
                if event.get("tool_name") == "move_ee_to_pose"
            ]
            movement_results = [
                {
                    "success": bool((event.get("result") or {}).get("success", False)),
                    "target_reached": ((event.get("result") or {}).get("details") or {}).get("target_reached"),
                    "target_observation_error_m": (
                        ((event.get("result") or {}).get("details") or {}).get("target_observation_error_m")
                    ),
                    "target_error_history_m": (
                        ((event.get("result") or {}).get("details") or {}).get("target_error_history_m")
                    ),
                }
                for event in movement_events
            ]
            replay_pass = len(movement_results) == 2 and all(
                item["success"] and item["target_reached"] is True
                for item in movement_results
            )
            expert_replay_result = {
                "diagnostic_only": True,
                "exclude_from_capability_denominator": True,
                "capture": expert_replay_capture,
                "movement_results": movement_results,
                "pass": replay_pass,
            }
        summary = {
            "status": "ok",
            "task_name": args.task_name,
            "task_config": args.task_config,
            "seed": selected_seed,
            "episode_id": args.episode_id,
            "instruction": instruction,
            "available_tools": available_tools,
            "tool_calls": [{"tool_name": call.tool_name, "args": dict(call.args)} for call in calls],
            "expert_tcp_replay": expert_replay_result,
            "final_snapshot": snapshot_summary(final_snapshot),
            "outputs": {
                "output_dir": str(output_dir),
                "contact_sheet": results["contact_sheet"],
                "events_jsonl": results["events_jsonl"],
                "frames_dir": results["frames_dir"],
                "gif": results["gif"],
            },
            "events": [jsonable_state(state) for state in results["states"]],
        }
        summary_path = output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps({"status": "ok", "summary": str(summary_path), "contact_sheet": results["contact_sheet"]}, ensure_ascii=False, indent=2), flush=True)
        if expert_replay_result is not None and not expert_replay_result["pass"]:
            return 3
        return 0
    finally:
        try:
            task_env.close_env()
        except Exception:
            pass
        viewer = getattr(task_env, "viewer", None)
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
