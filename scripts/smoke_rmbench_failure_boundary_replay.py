#!/usr/bin/env python3
"""Verify action-prefix failure-boundary replay in a real RMBench task.

This smoke does not call a model, segmentation service, or robot policy.  It
creates the same task/seed twice, records one current-joint simulator action,
replays it in the fresh scene, and writes a small readable report.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.rmbench.paths import (
    resolve_asset_reference,
    task_config_root,
)
from roboharn_evo.agent.components.memory_manager import MemoryManager
from roboharn_evo.agent.failure_boundary_replay import (
    ActionPrefixRecorder,
    capture_failure_boundary,
    compare_environment_to_boundary,
    restore_failure_boundary,
    save_failure_boundary,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round-trip one real RMBench simulator boundary."
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="simulator boundary round-trip smoke")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.load(handle.read(), Loader=yaml.FullLoader)
    if not isinstance(value, dict):
        raise TypeError(f"expected one YAML mapping: {path}")
    return value


def _load_task_args(task_name: str, task_config: str) -> dict[str, Any]:
    config_root = task_config_root()
    task_args = _load_yaml(config_root / f"{task_config}.yml")
    task_args.update(
        {
            "task_name": task_name,
            "task_config": task_config,
            "policy_name": "failure_boundary_round_trip_smoke",
            "ckpt_setting": "failure_boundary_round_trip_smoke",
            "eval_mode": True,
            "render_freq": 0,
            "eval_video_log": False,
            "eval_video_save_dir": None,
        }
    )

    embodiments = task_args.get("embodiment")
    if not isinstance(embodiments, list) or len(embodiments) not in {1, 3}:
        raise RuntimeError("task embodiment configuration is invalid")
    embodiment_table = _load_yaml(config_root / "_embodiment_config.yml")

    def embodiment_path(name: str) -> str:
        record = embodiment_table.get(name)
        if not isinstance(record, dict) or not record.get("file_path"):
            raise RuntimeError(f"embodiment is unavailable: {name}")
        return str(resolve_asset_reference(record["file_path"]))

    if len(embodiments) == 1:
        left_path = right_path = embodiment_path(embodiments[0])
        task_args["dual_arm_embodied"] = True
    else:
        left_path = embodiment_path(embodiments[0])
        right_path = embodiment_path(embodiments[1])
        task_args["embodiment_dis"] = embodiments[2]
        task_args["dual_arm_embodied"] = False
    task_args["left_robot_file"] = left_path
    task_args["right_robot_file"] = right_path
    task_args["left_embodiment_config"] = _load_yaml(Path(left_path) / "config.yml")
    task_args["right_embodiment_config"] = _load_yaml(Path(right_path) / "config.yml")

    camera_table = _load_yaml(config_root / "_camera_config.yml")
    camera_name = task_args["camera"]["head_camera_type"]
    camera = camera_table[camera_name]
    task_args["head_camera_h"] = camera["h"]
    task_args["head_camera_w"] = camera["w"]
    return task_args


def _make_environment(
    task_name: str,
    task_args: dict[str, Any],
    *,
    episode_id: int,
    seed: int,
    instruction: str,
) -> Any:
    import importlib

    module = importlib.import_module(f"benchmarks.rmbench.envs.{task_name}")
    task_type = getattr(module, task_name)
    environment = task_type()
    environment.setup_demo(
        now_ep_num=episode_id,
        seed=seed,
        is_test=True,
        **task_args,
    )
    environment.set_instruction(instruction=instruction)
    return environment


def _make_model(instruction: str) -> Any:
    store = MemoryManager(initial_memory_text="")
    store.reset(
        task=instruction,
        control_model_name="failure-boundary-smoke",
        available_policies=[],
        available_tools=[],
        mode="deployment",
    )
    agent = SimpleNamespace(
        memory_store=store,
        current_instruction=instruction,
        _pending_action_effect_verification=None,
        _pending_recovery_observation=None,
        _blocked_grounded_setups=set(),
        _grounded_setup_failures={},
        _grounded_geometry_leases={},
        _partial_grounded_approach_leases={},
        _recent_release_resolutions={},
    )
    return SimpleNamespace(session=SimpleNamespace(agent=agent))


def _current_joint_action(environment: Any) -> tuple[np.ndarray, str]:
    robot = getattr(environment, "robot", None)
    left_getter = getattr(robot, "get_left_arm_jointState", None)
    if not callable(left_getter):
        raise RuntimeError("RMBench robot exposes no left joint state")
    values = list(left_getter())
    active_arm = "left"
    if bool(getattr(environment, "is_dual_arm", False)):
        right_getter = getattr(robot, "get_right_arm_jointState", None)
        if not callable(right_getter):
            raise RuntimeError("dual-arm RMBench robot exposes no right joint state")
        values.extend(right_getter())
        active_arm = "both"
    action = np.asarray(values, dtype=np.float64).reshape(-1)
    if action.size == 0 or not np.all(np.isfinite(action)):
        raise RuntimeError("RMBench robot joint state is invalid")
    return action, active_arm


def _close_environment(environment: Any | None) -> None:
    if environment is None:
        return
    try:
        environment.close_env(clear_cache=True)
    except TypeError:
        environment.close_env()


def main() -> int:
    args = _parse_args()
    if args.seed < 0 or args.episode_id < 0:
        raise ValueError("seed and episode-id must be non-negative")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["RMBENCH_ASSETS_ROOT"] = str(args.assets_root.expanduser().resolve())

    source_environment = None
    restored_environment = None
    try:
        task_args = _load_task_args(args.task_name, args.task_config)
        source_environment = _make_environment(
            args.task_name,
            task_args,
            episode_id=args.episode_id,
            seed=args.seed,
            instruction=args.instruction,
        )
        source_model = _make_model(args.instruction)
        initial_boundary = capture_failure_boundary(
            source_environment,
            source_model,
            task=args.task_name,
            seed=args.seed,
            instruction=args.instruction,
            label="same-seed initial scene",
        )
        action, active_arm = _current_joint_action(source_environment)
        recorder = ActionPrefixRecorder(source_environment)
        recorder.start()
        try:
            source_environment.get_obs()
            source_environment.take_action(
                action,
                action_type="qpos",
                active_arm=active_arm,
            )
            source_environment.get_obs()
        finally:
            recorder.stop()
        if len(recorder.records) != 1:
            raise RuntimeError("the simulator action did not advance exactly once")
        boundary = capture_failure_boundary(
            source_environment,
            source_model,
            task=args.task_name,
            seed=args.seed,
            instruction=args.instruction,
            label=args.label,
            reason="state after one recorded simulator action",
            action_prefix=recorder.records,
            observation_calls_after_last_action=(
                recorder.observation_calls_after_last_action
            ),
        )
        boundary_path = save_failure_boundary(output_dir / "boundary.json", boundary)
        _close_environment(source_environment)
        source_environment = None

        restored_environment = _make_environment(
            args.task_name,
            task_args,
            episode_id=args.episode_id,
            seed=args.seed,
            instruction=args.instruction,
        )
        same_seed_initial_state = compare_environment_to_boundary(
            restored_environment,
            initial_boundary,
        )
        before_replay = compare_environment_to_boundary(
            restored_environment,
            boundary,
        )
        restored = restore_failure_boundary(
            restored_environment,
            _make_model(args.instruction),
            boundary,
            expected_task=args.task_name,
            expected_seed=args.seed,
            expected_instruction=args.instruction,
        )
        report = {
            "schema": "roboharn_evo/rmbench_failure_boundary_smoke/v1",
            "task": args.task_name,
            "task_config": args.task_config,
            "seed": args.seed,
            "episode_id": args.episode_id,
            "instruction": args.instruction,
            "model_calls": 0,
            "segmentation_calls": 0,
            "policy_generated_actions": 0,
            "source_simulator_actions_recorded": len(recorder.records),
            "replayed_simulator_actions": restored["action_prefix_replay"][
                "actions_replayed"
            ],
            "boundary": str(boundary_path),
            "same_seed_initial_state_matches": same_seed_initial_state,
            "fresh_scene_is_before_boundary": before_replay["matches"] is False,
            "restored_state": restored,
            "passed": bool(
                same_seed_initial_state["matches"]
                and before_replay["matches"] is False
                and restored["matches"]
            ),
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        report = {
            "schema": "roboharn_evo/rmbench_failure_boundary_smoke/v1",
            "task": args.task_name,
            "task_config": args.task_config,
            "seed": args.seed,
            "episode_id": args.episode_id,
            "instruction": args.instruction,
            "model_calls": 0,
            "segmentation_calls": 0,
            "policy_generated_actions": 0,
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    finally:
        _close_environment(source_environment)
        _close_environment(restored_environment)


if __name__ == "__main__":
    raise SystemExit(main())
