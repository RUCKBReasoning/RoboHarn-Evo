#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
import yaml


RMBENCH_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
for import_root in (PROJECT_ROOT, RMBENCH_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from policy.roboharn_evo.deploy_policy import get_model
from policy.roboharn_evo.utils.io import decode_rgb_frame
from roboharn_evo.agent import RoboHarnAgentRuntime


CAMERA_NAMES = ("head", "left", "right")


@dataclass
class OfflineTrajectory:
    observations: list[dict[str, Any]]
    instruction: str
    task_name: str
    episode_id: int
    source_path: Path


class OfflineReplayEnv:
    def __init__(self, trajectory: OfflineTrajectory, *, success_step: int = -1) -> None:
        self.trajectory = trajectory
        self.take_action_cnt = 0
        self.step_lim = max(0, len(trajectory.observations) - 1)
        self.eval_success = False
        self.max_reward = 0.0
        self.instruction = trajectory.instruction
        self.suc = 0
        self.test_num = 0
        self.eval_video_path = None
        self.render_freq = 0
        self.viewer = None
        self._success_step = int(success_step)

    def get_instruction(self) -> str:
        return self.instruction

    def set_instruction(self, instruction: str) -> None:
        self.instruction = str(instruction)

    def check_success(self) -> bool:
        return bool(self.eval_success)

    def get_obs(self) -> dict[str, Any]:
        index = min(max(0, int(self.take_action_cnt)), len(self.trajectory.observations) - 1)
        observation = self.trajectory.observations[index]
        observation["instruction"] = self.instruction
        return observation

    def take_action(self, action: Any, action_type: str = "qpos") -> None:
        del action, action_type
        self.take_action_cnt = min(self.take_action_cnt + 1, self.step_lim)
        if self._success_step >= 0 and self.take_action_cnt >= self._success_step:
            self.eval_success = True
            self.max_reward = 1.0

    def close_env(self, clear_cache: bool = False) -> None:
        del clear_cache
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay existing HDF5 trajectories through the RoboHarn-Evo agent loop without calling a VLA policy.")
    parser.add_argument("--config", type=Path, required=True, help="RoboHarn-Evo deploy_policy.yml.")
    parser.add_argument("--traj", type=Path, required=True, help="One RMBench/RobotWin HDF5 trajectory.")
    parser.add_argument("--instruction", type=str, default="", help="Instruction override. If empty, read from --instruction-file or HDF5 attrs.")
    parser.add_argument("--instruction-file", type=Path, default=None, help="Optional episode instruction JSON file.")
    parser.add_argument("--instruction-type", type=str, default="seen", choices=["seen", "unseen"])
    parser.add_argument("--task-name", type=str, default="", help="Task name for metadata. Defaults to parent-derived name.")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_result/roboharn_evo_offline_replay"))
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum replay steps. 0 means use the full HDF5 trajectory.")
    parser.add_argument("--success-step", type=int, default=-1, help="Optional offline success step for smoke tests.")
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=0,
        help=(
            "Recorded frames consumed per offline VLA action chunk. "
            "0 infers from the configured executor max_chunk_steps/pi0_step, falling back to 50."
        ),
    )
    parser.add_argument("--offline-planner", action="store_true", help="Use a deterministic local planner stub instead of the configured VLM planner.")
    parser.add_argument("--offline-subtask", type=str, default="", help="Subtask returned by --offline-planner. Defaults to the instruction.")
    parser.add_argument(
        "--use-vlm-ood",
        action="store_true",
        help="Keep the configured VLM OOD API during offline replay. Default uses local no-OOD fallback to avoid per-frame OOD API calls.",
    )
    parser.add_argument(
        "--preprocess-every-chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When observation preprocessing is enabled, align its default interval with --chunk-steps. "
            "Explicit --overrides for agent.observation_preprocess.every_n_steps still win."
        ),
    )
    parser.add_argument("--no-report", action="store_true", help="Skip HTML report generation.")
    parser.add_argument("--no-contact-sheet", action="store_true", help="Skip rollout_contact_sheet.png generation.")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def load_config(config_path: Path, overrides: list[str] | None) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML dict: {config_path}")
    if overrides:
        _deep_update(config, _parse_override_pairs(overrides))
    return config


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _parse_override_pairs(pairs: list[str]) -> dict[str, Any]:
    if len(pairs) % 2 != 0:
        raise ValueError("--overrides expects key value pairs")
    override_dict: dict[str, Any] = {}
    for i in range(0, len(pairs), 2):
        key = pairs[i].lstrip("-")
        raw_value = pairs[i + 1]
        try:
            value = yaml.safe_load(raw_value)
        except Exception:
            value = raw_value
        current = override_dict
        parts = key.split(".")
        for part in parts[:-1]:
            nested = current.get(part)
            if not isinstance(nested, dict):
                nested = {}
                current[part] = nested
            current = nested
        current[parts[-1]] = value
    return override_dict


def load_instruction(*, h5: h5py.File, args: argparse.Namespace) -> str:
    if args.instruction.strip():
        return args.instruction.strip()
    if args.instruction_file is not None:
        with args.instruction_file.expanduser().open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        value = payload.get(args.instruction_type, payload.get("instruction", ""))
        if isinstance(value, list):
            return str(value[0]).strip()
        return str(value).strip()
    for key in ("instruction", "task", "language", "language_instruction"):
        if key in h5.attrs:
            return str(h5.attrs[key]).strip()
    return "replay the recorded manipulation trajectory"


def load_hdf5_trajectory(path: Path, *, instruction: str, task_name: str, episode_id: int, max_steps: int) -> OfflineTrajectory:
    observations: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        states = _read_dataset(handle, "joint_action/vector")
        if states is None:
            states = _read_dataset(handle, "observation/state")
        if states is None:
            raise KeyError("Missing joint state dataset. Expected 'joint_action/vector' or 'observation/state'.")
        states = np.asarray(states, dtype=np.float32)
        total_frames = int(states.shape[0])
        if max_steps > 0:
            total_frames = min(total_frames, max_steps + 1)
        for frame_index in range(total_frames):
            observations.append(_build_observation(handle, states, frame_index, instruction))
    return OfflineTrajectory(
        observations=observations,
        instruction=instruction,
        task_name=task_name,
        episode_id=episode_id,
        source_path=path,
    )


def _read_dataset(handle: h5py.File, path: str) -> Any | None:
    if path not in handle:
        return None
    return handle[path][()]


def _read_frame(handle: h5py.File, path: str, frame_index: int) -> Any | None:
    if path not in handle:
        return None
    dataset = handle[path]
    if getattr(dataset, "shape", ()) and len(dataset.shape) > 0:
        return dataset[frame_index]
    return dataset[()]


def _read_camera_aux_frame(handle: h5py.File, path: str, frame_index: int, *, key: str) -> Any | None:
    if path not in handle:
        return None
    dataset = handle[path]
    shape = tuple(getattr(dataset, "shape", ()))
    if key in {"intrinsic_cv", "cam2world_gl"} and shape in {(3, 3), (4, 4)}:
        return dataset[()]
    if key == "depth" and len(shape) == 2:
        return dataset[()]
    if shape and shape[0] > frame_index and len(shape) >= 1:
        return dataset[frame_index]
    return dataset[()]


def _decode_image(value: Any) -> np.ndarray:
    if isinstance(value, (bytes, np.bytes_)):
        return decode_rgb_frame(value)
    array = np.asarray(value)
    if array.ndim == 1 and array.dtype.kind in {"S", "O", "U"}:
        return decode_rgb_frame(array.item())
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    return np.asarray(array, dtype=np.uint8)


def _camera_payload(handle: h5py.File, camera: str, frame_index: int) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    rgb = _read_frame(handle, f"observation/{camera}_camera/rgb", frame_index)
    if rgb is None:
        raise KeyError(f"Missing RGB dataset: observation/{camera}_camera/rgb")
    payload["rgb"] = _decode_image(rgb)
    for key in ("depth", "intrinsic_cv", "cam2world_gl"):
        value = _read_camera_aux_frame(handle, f"observation/{camera}_camera/{key}", frame_index, key=key)
        if value is not None:
            payload[key] = np.asarray(value)
    return payload


def _read_endpose(handle: h5py.File, arm: str, frame_index: int) -> np.ndarray:
    value = _read_frame(handle, f"endpose/{arm}_endpose", frame_index)
    if value is None:
        value = _read_frame(handle, f"observation/{arm}_endpose", frame_index)
    if value is None:
        fallback = np.zeros(7, dtype=np.float32)
        fallback[3] = 1.0
        return fallback
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size >= 7:
        return array[:7]
    fallback = np.zeros(7, dtype=np.float32)
    fallback[3] = 1.0
    fallback[: min(array.size, 7)] = array[:7]
    return fallback


def _build_observation(handle: h5py.File, states: np.ndarray, frame_index: int, instruction: str) -> dict[str, Any]:
    state = np.asarray(states[frame_index], dtype=np.float32).reshape(-1)
    return {
        "observation": {
            f"{camera}_camera": _camera_payload(handle, camera, frame_index)
            for camera in CAMERA_NAMES
        },
        "joint_action": {
            "vector": state,
        },
        "endpose": {
            "left_endpose": _read_endpose(handle, "left", frame_index),
            "right_endpose": _read_endpose(handle, "right", frame_index),
        },
        "instruction": instruction,
        "_roboharn_evo_offline_replay": {
            "mode": "offline_replay",
            "frame_index": int(frame_index),
        },
    }


def infer_task_name(path: Path, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    parts = list(path.expanduser().resolve().parts)
    if "data" in parts:
        index = len(parts) - 1 - parts[::-1].index("data")
        if index >= 2:
            return parts[index - 2]
    return path.parent.parent.name or "offline_traj"


def infer_offline_chunk_steps(config: dict[str, Any], explicit_chunk_steps: int = 0) -> int:
    if explicit_chunk_steps > 0:
        return int(explicit_chunk_steps)
    executor_cfg = dict(config.get("executor", {}) or {})
    backend = str(executor_cfg.get("backend", "")).strip()
    candidate = 0
    if backend == "pi05_api":
        candidate = int(dict(executor_cfg.get("pi05_api", {}) or {}).get("max_chunk_steps", 0) or 0)
    elif backend in {"world_action_model", "wam"}:
        candidate = int(dict(executor_cfg.get("world_action_model", {}) or {}).get("max_chunk_steps", 0) or 0)
    elif backend == "pi05_adapter":
        candidate = int(dict(executor_cfg.get("pi05", {}) or {}).get("pi0_step", 0) or 0)
    elif backend in {"offline_replay", "replay"}:
        candidate = int(dict(executor_cfg.get("offline_replay", {}) or {}).get("chunk_steps", 0) or 0)
    return max(1, candidate or 50)


def override_contains(overrides: list[str] | None, dotted_key: str) -> bool:
    if not overrides:
        return False
    normalized = dotted_key.lstrip("-")
    return any(str(item).lstrip("-") == normalized for item in overrides)


def configure_offline_agent(
    config: dict[str, Any],
    *,
    offline_planner: bool,
    offline_subtask: str = "",
    chunk_steps: int = 0,
    use_vlm_ood: bool = False,
    preprocess_every_chunk: bool = True,
    preprocess_interval_overridden: bool = False,
    decision_interval_overridden: bool = False,
) -> dict[str, Any]:
    configured = dict(config)
    configured["agent"] = dict(configured.get("agent", {}) or {})
    configured["agent"]["enabled"] = True
    resolved_chunk_steps = infer_offline_chunk_steps(config, chunk_steps)
    if not decision_interval_overridden:
        configured["agent"]["decision_interval"] = resolved_chunk_steps
    if preprocess_every_chunk and not preprocess_interval_overridden:
        configured["agent"]["observation_preprocess"] = dict(configured["agent"].get("observation_preprocess", {}) or {})
        configured["agent"]["observation_preprocess"]["every_n_steps"] = resolved_chunk_steps
    if offline_planner:
        configured["planner"] = dict(configured.get("planner", {}) or {})
        configured["planner"]["backend"] = "offline_replay"
        configured["planner"].setdefault("offline_replay", {})
        if offline_subtask.strip():
            configured["planner"]["offline_replay"]["subtask_text"] = offline_subtask.strip()
    configured["executor"] = dict(configured.get("executor", {}) or {})
    configured["executor"]["backend"] = "offline_replay"
    configured["executor"].setdefault("offline_replay", {})
    configured["executor"]["offline_replay"].setdefault("action_dim", 14)
    configured["executor"]["offline_replay"]["chunk_steps"] = resolved_chunk_steps
    if not use_vlm_ood:
        configured["ood"] = {
            "backend": "offline_replay",
            "offline_replay": {
                "OOD_scenario": "none",
                "reason": "offline replay uses local OOD fallback; planner is called once per action chunk",
                "confidence": 0.0,
            },
        }
    return configured


def write_event(trace_file: Path, event: str, **payload: Any) -> None:
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, **payload}
    with trace_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    print("[offline_replay] " + json.dumps(record, ensure_ascii=False, default=_json_default), flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _run_visualizer(script_name: str, *, input_path: Path, output_path: Path) -> None:
    script_path = RMBENCH_ROOT / "policy" / "roboharn_evo" / "scripts" / script_name
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip(), flush=True)
    if completed.returncode != 0:
        print(
            f"[offline_replay] visualization_warning script={script_name} returncode={completed.returncode} stderr={completed.stderr.strip()}",
            flush=True,
        )


def finalize_visualization(run_dir: Path, rollout_dir: Path, agent: Any, *, skip_report: bool, skip_contact_sheet: bool) -> None:
    try:
        agent.finalize_rollout_videos(fps=10)
    except Exception as exc:
        print(f"[offline_replay] video_finalize_warning={exc!r}", flush=True)
    if not skip_report:
        _run_visualizer("visualize_rollout_report.py", input_path=run_dir, output_path=run_dir / "rollout_report.html")
    if not skip_contact_sheet:
        _run_visualizer("visualize_rollout_contact_sheet.py", input_path=run_dir, output_path=run_dir / "rollout_contact_sheet.png")
        _run_visualizer("visualize_rollout_contact_sheet.py", input_path=rollout_dir, output_path=rollout_dir / "contact_sheet.png")


def main() -> None:
    args = parse_args()
    traj_path = args.traj.expanduser().resolve()
    with h5py.File(traj_path, "r") as handle:
        instruction = load_instruction(h5=handle, args=args)
    task_name = infer_task_name(traj_path, args.task_name)
    trajectory = load_hdf5_trajectory(
        traj_path,
        instruction=instruction,
        task_name=task_name,
        episode_id=args.episode_id,
        max_steps=max(0, int(args.max_steps)),
    )
    if not trajectory.observations:
        raise ValueError(f"No frames loaded from {traj_path}")
    requested_max_steps = max(0, int(args.max_steps))
    resolved_max_steps = max(0, len(trajectory.observations) - 1)

    config = configure_offline_agent(
        load_config(args.config.expanduser().resolve(), args.overrides),
        offline_planner=bool(args.offline_planner),
        offline_subtask=args.offline_subtask or instruction,
        chunk_steps=max(0, int(args.chunk_steps)),
        use_vlm_ood=bool(args.use_vlm_ood),
        preprocess_every_chunk=bool(args.preprocess_every_chunk),
        preprocess_interval_overridden=override_contains(args.overrides, "agent.observation_preprocess.every_n_steps"),
        decision_interval_overridden=override_contains(args.overrides, "agent.decision_interval"),
    )
    offline_chunk_steps = int(config.get("executor", {}).get("offline_replay", {}).get("chunk_steps", 1))
    model = get_model(config)
    if not isinstance(model, RoboHarnAgentRuntime):
        raise TypeError("Offline replay requires agent.enabled=True and RoboHarnAgentRuntime.")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_dir = args.output_dir.expanduser().resolve() / task_name / timestamp
    rollout_dir = run_dir / f"episode_{args.episode_id:04d}_rollout"
    trace_file = run_dir / f"episode_{args.episode_id:04d}_agent_trace.jsonl"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    env = OfflineReplayEnv(trajectory, success_step=args.success_step)
    model.reset()
    model.set_trace_file(str(trace_file), episode_id=args.episode_id, seed=0)
    model.set_rollout_dump_dir(str(rollout_dir))
    model.write_rollout_meta(
        {
            "mode": "offline_replay",
            "episode_id": args.episode_id,
            "task_name": task_name,
            "instruction": instruction,
            "source_path": str(traj_path),
            "num_frames": len(trajectory.observations),
            "requested_max_steps": requested_max_steps,
            "resolved_max_steps": resolved_max_steps,
            "offline_chunk_steps": offline_chunk_steps,
            "offline_ood_mode": "configured_vlm" if args.use_vlm_ood else "local_no_ood",
        }
    )

    write_event(
        trace_file,
        "episode_start",
        mode="offline_replay",
        episode_id=args.episode_id,
        task_name=task_name,
        instruction=instruction,
        source_path=str(traj_path),
        total_frames=len(trajectory.observations),
        requested_max_steps=requested_max_steps,
        resolved_max_steps=resolved_max_steps,
        chunk_steps=offline_chunk_steps,
        ood_mode="configured_vlm" if args.use_vlm_ood else "local_no_ood",
    )
    try:
        while env.take_action_cnt < env.step_lim:
            model.run_eval_step(env, env.get_obs())
            if env.eval_success:
                break
    finally:
        write_event(
            trace_file,
            "episode_end",
            mode="offline_replay",
            episode_id=args.episode_id,
            success=bool(env.eval_success),
            total_steps=env.take_action_cnt,
            max_reward=float(env.max_reward),
        )
        finalize_visualization(
            run_dir,
            rollout_dir,
            model.session.agent,
            skip_report=bool(args.no_report),
            skip_contact_sheet=bool(args.no_contact_sheet),
        )

    print(json.dumps({
        "success": True,
        "run_dir": str(run_dir),
        "trace_file": str(trace_file),
        "rollout_dir": str(rollout_dir),
        "contact_sheet": str(run_dir / "rollout_contact_sheet.png"),
        "episode_contact_sheet": str(rollout_dir / "contact_sheet.png"),
        "report": str(run_dir / "rollout_report.html"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
