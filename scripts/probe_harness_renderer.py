#!/usr/bin/env python3
"""RMBench reset/30-fps readback diagnostic; no planner, SAM3, or VLA calls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.rmbench.integration import _runtime_process_boundary
from scripts.smoke_rmbench_failure_boundary_replay import _load_task_args, _make_environment
from roboharn_evo.agent.rollout_video import ContinuousRolloutVideoRecorder
from roboharn_evo.benchmark_adapters.rmbench.harness_vla.primitives import RMBenchPrimitives


def capture_cameras_serial(camera_rig):
    cameras = []
    if camera_rig.collect_wrist_camera:
        cameras.extend([camera_rig.left_camera, camera_rig.right_camera])
    cameras.extend(camera_rig.static_camera_list)
    for camera in cameras:
        camera.take_picture()
        # Readback is the existing public completion barrier. Do not overlap
        # multiple camera render submissions in this diagnostic variant.
        camera.get_picture("Color")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoiser", choices=["none", "oidn", "optix"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--hold-actions", type=int, default=50,
                        help="Native joint-target hold actions per cycle; includes normal servo/gravity updates.")
    parser.add_argument("--clear-cache-on-reset", action="store_true")
    parser.add_argument("--serial-cameras", action="store_true")
    parser.add_argument("--commands", type=Path, help="Optional recorded analytic commands to exercise moving wrist views.")
    args = parser.parse_args()
    if args.cycles < 1 or args.hold_actions < 1:
        parser.error("positive cycles and hold actions required")
    commands = [json.loads(line) for line in args.commands.read_text().splitlines() if line.strip()] if args.commands else []
    for command in commands:
        if command.get("action") not in {"move_to", "move_pose", "set_gripper", "release", "rotate_wrist", "rotate_pitch"}:
            parser.error("diagnostic only accepts recorded analytic commands; no model/service calls")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["RMBENCH_RAY_TRACING_DENOISER"] = args.denoiser
    report = {"diagnostic_only": True, "task": args.task, "seed": args.seed,
              "denoiser": args.denoiser, "cycles": args.cycles,
              "native_hold_actions_per_cycle": args.hold_actions,
              "stepping": "native take_action with current joint targets",
              "clear_cache_on_reset": args.clear_cache_on_reset,
              "serial_cameras": args.serial_cameras,
              "camera_capture_implementation": "diagnostic serial override" if args.serial_cameras else "native Camera.update_picture",
              "model_calls": 0, "sam_calls": 0, "vla_calls": 0, "completed": False}
    (output / "renderer_probe.json").write_text(json.dumps(report, indent=2))
    env = None
    recorder = None
    with _runtime_process_boundary(asset_dir=args.assets_root.resolve(strict=True),
                                   output_dir=output, runtime_configs=output / "cfg",
                                   workspace_dir=output / "workspace", cache_dir=output / "cache"):
        try:
            from benchmarks.rmbench.script.test_render import Sapien_TEST
            Sapien_TEST()
            task_args = _load_task_args(args.task, "demo_clean")
            task_args["save_path"] = str(output / "scene_data")
            task_args["data_type"].update(rgb=True, depth=True, endpose=True, qpos=True,
                                           third_view=True, pointcloud=False)
            env = _make_environment(args.task, task_args, episode_id=0, seed=args.seed,
                                    instruction="Renderer diagnostic only; not a task evaluation.")
            if args.serial_cameras:
                type(env.cameras).update_picture = capture_cameras_serial
            recorder = ContinuousRolloutVideoRecorder(output)
            for cycle in range(args.cycles):
                if cycle:
                    env._set_eval_video_frame_callback(None)
                    env.close_env(clear_cache=args.clear_cache_on_reset)
                    env.setup_demo(now_ep_num=0, seed=args.seed, is_test=True, **task_args)
                    env.set_instruction("Renderer diagnostic only; not a task evaluation.")
                recorder.attach(env)
                core = RMBenchPrimitives(env, output_dir=output / f"cycle_{cycle}", vla=None)
                core.save_observation()
                for command in commands:
                    result = core.execute(command["action"], {k:v for k,v in command.items() if k != "action"})
                    if result["result"].get("error"):
                        raise RuntimeError(result["result"]["error"])
                for hold in range(args.hold_actions):
                    # Match the actual control path, including passive-force
                    # compensation. Bare scene.step is not an equivalent hold.
                    core._action(core.snapshot.joint_vector.copy(), "qpos", "both")
                    if (hold + 1) % 10 == 0:
                        print(json.dumps({"cycle": cycle, "hold_actions": hold + 1,
                                          "frames": recorder.frame_counts}), flush=True)
                core.refresh()
                core.tool_index += 1
                core.save_observation()
            recording = recorder.close()
            report.update(completed=recording["complete"], recording=recording)
            (output / "renderer_probe.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
            return int(not report["completed"])
        finally:
            if recorder is not None:
                recorder.close()
            if env is not None:
                env.close_env()
                for side in ("left", "right"):
                    conn = getattr(env.robot, side + "_conn", None)
                    proc = getattr(env.robot, side + "_proc", None)
                    if conn is not None and proc is not None and proc.is_alive():
                        conn.send({"cmd": "exit"})
                        proc.join(timeout=5)
                        if proc.is_alive():
                            proc.terminate()
                            proc.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
