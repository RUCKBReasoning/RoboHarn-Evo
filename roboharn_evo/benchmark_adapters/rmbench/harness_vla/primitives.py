"""RMBench bindings for the fixed Harness VLA primitive vocabulary.

The planner and memory remain RPent's. Only robot sensors, kinematics and the
frozen policy interface are translated here; no task/object-specific strategy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from roboharn_evo.agent.environment import RMBenchEnvAdapter
from roboharn_evo.models.pi05_api_executor_adapter import _post_json


CAMERAS = ("head", "left", "right")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def project_pixels(depth_mm, intrinsic_cv, cam2world_gl, pixels):
    """Pixel centers + metric depth, never simulator object coordinates."""
    depth = np.asarray(depth_mm, dtype=float) / 1000.0
    pixels = np.asarray(pixels, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise ValueError("pixels must be finite [u, v] pairs")
    uv = np.rint(pixels).astype(int)
    if np.any(uv < 0) or np.any(uv[:, 0] >= depth.shape[1]) or np.any(uv[:, 1] >= depth.shape[0]):
        raise ValueError("pixel outside the selected camera image")
    z = depth[uv[:, 1], uv[:, 0]]
    valid = np.isfinite(z) & (z > 0)
    if not valid.any():
        raise ValueError("selected pixels have no observed depth")
    rays = np.column_stack([uv[valid], np.ones(valid.sum())]) @ np.linalg.inv(intrinsic_cv).T
    camera_cv = rays * z[valid, None]
    camera_gl = camera_cv * [1, -1, -1]
    transform = np.asarray(cam2world_gl, dtype=float)
    return camera_gl @ transform[:3, :3].T + transform[:3, 3]


class RMBenchPrimitives:
    def __init__(self, env, *, output_dir: Path, vla, sam_url: str = "", reset: Callable | None = None,
                 deadline: float = float("inf")):
        self.env, self.vla, self.sam_url = env, vla, sam_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reset_environment = reset
        self.deadline = deadline
        self.tool_index = 0
        self.reset_count = 0
        self.vla_calls = 0
        self.sam_calls = 0
        self.total_environment_actions = 0
        self.refresh()

    def refresh(self):
        self.observation = self.env.get_obs()
        # RMBench predicates may update button drives/counters. Sensor reads
        # must not advance that state machine outside native physics stepping.
        self.snapshot = RMBenchEnvAdapter.from_env(self.env, self.observation, oracle_objects_enabled=False,
                                                   evaluate_success_predicate=False)
        return self.snapshot

    @property
    def done(self):
        return (self.snapshot.eval_success or self.snapshot.step_count >= self.snapshot.step_limit
                or time.monotonic() >= self.deadline)

    def state(self):
        s = self.snapshot
        chunk_steps = getattr(getattr(self.vla, "config", None), "max_chunk_steps", None)
        measured_arm_qpos = {}
        robot = getattr(self.env, "robot", None)
        for side in ("left", "right"):
            read_joints = getattr(robot, f"get_{side}_arm_real_jointState", None)
            if callable(read_joints):
                # The native real-joint helper still appends a commanded
                # gripper value. Only its arm joints are measured positions.
                measured_arm_qpos[side] = np.asarray(read_joints(), dtype=float)[:-1]
        return jsonable({
            "task_language": s.instruction, "eval_success": s.eval_success,
            "environment_step": s.step_count, "step_limit": s.step_limit,
            "reset_count": self.reset_count, "qpos": s.joint_vector,
            "qpos_source": "native joint-action drive targets, not measured joint positions",
            "measured_arm_qpos": measured_arm_qpos,
            "wall_time_exhausted": time.monotonic() >= self.deadline,
            "wall_seconds_remaining": max(0., self.deadline - time.monotonic()) if np.isfinite(self.deadline) else None,
            "left_ee_pose_xyz_wxyz": s.left_endpose, "right_ee_pose_xyz_wxyz": s.right_endpose,
            "left_gripper": float(s.joint_vector[6]), "right_gripper": float(s.joint_vector[13]),
            "gripper_convention": "0 closed, 1 open",
            "gripper_state_source": "commanded normalized opening, not measured finger aperture",
            "action_frame_to_tcp": s.tcp_calibration_by_arm,
            "vla_calls": self.vla_calls, "sam_calls": self.sam_calls,
            "total_environment_actions_including_resets": self.total_environment_actions,
            "vla_max_actions_per_chunk": chunk_steps if isinstance(chunk_steps, int) else None,
        })

    def save_observation(self):
        result = self.state()
        views = {}
        for camera in CAMERAS:
            directory = self.output_dir / camera
            directory.mkdir(exist_ok=True)
            image_path = directory / f"step_{self.tool_index:06d}.png"
            Image.fromarray(getattr(self.snapshot, f"{camera}_rgb")).save(image_path)
            depth = getattr(self.snapshot, f"{camera}_depth")
            intrinsic = getattr(self.snapshot, f"{camera}_intrinsic_cv")
            transform = getattr(self.snapshot, f"{camera}_cam2world_gl")
            depth_path = directory / f"step_{self.tool_index:06d}_rgbd.npz"
            np.savez_compressed(depth_path, depth_mm=depth, intrinsic_cv=intrinsic, cam2world_gl=transform)
            views[camera] = {"rgb_path": str(image_path), "rgbd_path": str(depth_path),
                             "image_shape": list(getattr(self.snapshot, f"{camera}_rgb").shape)}
        result["views"] = views
        (self.output_dir / "state.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def execute(self, name: str, args: dict):
        allowed = {"view_driver_state", "back_project", "segment", "move_to", "move_pose",
                   "rotate_wrist", "rotate_pitch", "set_gripper", "release", "vla_act", "reset"}
        if name not in allowed:
            raise ValueError(f"unknown primitive: {name}")
        before = self.state()
        start = time.monotonic()
        # Persist the issued command even if a native call never returns.
        with (self.output_dir / "primitive_commands.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"action": name, **jsonable(args)}, ensure_ascii=False) + "\n")
        try:
            result = getattr(self, name)(**args)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
            self.refresh()
        self.tool_index += 1
        after = self.save_observation()
        event = {"event": "harness_primitive", "tool_index": self.tool_index,
                 "tool_name": name, "args": args, "result": jsonable(result),
                 "before": before, "after": after, "elapsed_sec": time.monotonic() - start}
        with (self.output_dir / "episode_0000_agent_trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return {"result": result, "state": after}

    def view_driver_state(self):
        # Observation does not execute a robot action.
        self.refresh()
        return self.state()

    def back_project(self, camera: str, pixels: list):
        if camera not in CAMERAS:
            raise ValueError("camera must be head, left or right")
        s = self.snapshot
        points = project_pixels(getattr(s, f"{camera}_depth"), getattr(s, f"{camera}_intrinsic_cv"),
                                getattr(s, f"{camera}_cam2world_gl"), pixels)
        return {"points_world_m": points.tolist(), "median_world_m": np.median(points, axis=0).tolist(),
                "camera": camera, "source": "observed RGB-D"}

    def segment(self, camera: str, prompt: str, min_score: float = 0.2):
        if camera not in CAMERAS or not self.sam_url:
            raise ValueError("camera or SAM3 endpoint unavailable")
        self.save_observation()
        image_path = self.output_dir / camera / f"step_{self.tool_index:06d}.png"
        self.sam_calls += 1
        result = _post_json(url=self.sam_url, timeout_sec=120, payload={
            "image_path": str(image_path), "text_prompt": prompt, "top_k": 3,
        })
        return result

    def _arm(self, arm: str):
        if arm not in {"left", "right", "both"}:
            raise ValueError("arm must be left, right or both")
        return ("left", "right") if arm == "both" else (arm,)

    def _action(self, action, kind, arm):
        if self.done:
            return
        before = self.snapshot.step_count
        with (self.output_dir / "robot_actions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"reset_count": self.reset_count, "step_before": before,
                                     "action_type": kind, "active_arm": arm,
                                     "action": jsonable(np.asarray(action, dtype=float))}) + "\n")
        self.env.take_action(np.asarray(action, dtype=float), action_type=kind, active_arm=arm)
        self.refresh()
        self.total_environment_actions += max(0, self.snapshot.step_count - before)

    def set_gripper(self, arm: str, gripper: str, steps: int = 1):
        arms = self._arm(arm)
        if gripper not in {"open", "closed"} or not 1 <= int(steps) <= 50:
            raise ValueError("gripper must be open/closed and steps in [1, 50]")
        for _ in range(int(steps)):
            action = self.snapshot.joint_vector.copy()
            for side in arms:
                action[6 if side == "left" else 13] = float(gripper == "open")
            self._action(action, "qpos", arm)
            if self.done:
                break
        return {"gripper": gripper, "arm": arm, "eval_success": self.snapshot.eval_success}

    def release(self, arm: str, steps: int = 3):
        return self.set_gripper(arm, "open", steps)

    def move_to(self, arm: str, xyz: list, quat_wxyz: list | None = None,
                tol: float = 0.012, max_steps: int = 20, step_clip: float = 0.05):
        if arm not in {"left", "right"}:
            raise ValueError("move one explicitly selected arm at a time")
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("xyz must be three finite world-frame meters")
        if not 0 < tol <= 0.05 or not 0 < step_clip <= 0.15 or not 1 <= max_steps <= 100:
            raise ValueError("invalid motion tolerance or step budget")
        quat = getattr(self.snapshot, f"{arm}_endpose")[3:].copy() if quat_wxyz is None else np.asarray(quat_wxyz, dtype=float)
        if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-6:
            raise ValueError("quat_wxyz must be a nonzero finite quaternion")
        quat /= np.linalg.norm(quat)
        for _ in range(max_steps):
            if self.done:
                break
            current = getattr(self.snapshot, f"{arm}_endpose")
            delta = xyz - current[:3]
            angle = 2 * np.arccos(np.clip(abs(float(np.dot(quat, current[3:]))), 0, 1))
            if np.linalg.norm(delta) <= tol and angle <= 0.1:
                break
            target = current.copy()
            target[:3] += delta * min(1., step_clip / max(np.linalg.norm(delta), 1e-9))
            target[3:] = quat
            left = target if arm == "left" else self.snapshot.left_endpose
            right = target if arm == "right" else self.snapshot.right_endpose
            action = np.r_[left, self.snapshot.joint_vector[6], right, self.snapshot.joint_vector[13]]
            self._action(action, "ee", arm)
        final = getattr(self.snapshot, f"{arm}_endpose")
        distance = float(np.linalg.norm(final[:3] - xyz))
        angle = float(2 * np.arccos(np.clip(abs(float(np.dot(quat, final[3:]))), 0, 1)))
        return {"target_reached": distance <= tol and angle <= 0.1,
                "position_error_m": distance, "orientation_error_rad": angle, "observed_pose": final.tolist()}

    def move_pose(self, arm: str, xyz: list, quat_wxyz: list, **kwargs):
        return self.move_to(arm, xyz, quat_wxyz, **kwargs)

    def _rotate(self, arm, axis, delta_rad):
        if arm not in {"left", "right"} or not np.isfinite(delta_rad):
            raise ValueError("explicit arm and finite rotation required")
        pose = getattr(self.snapshot, f"{arm}_endpose")
        rotation = Rotation.from_quat(np.roll(pose[3:], -1)) * Rotation.from_rotvec(np.asarray(axis) * delta_rad)
        return self.move_to(arm, pose[:3], np.roll(rotation.as_quat(), 1))

    def rotate_wrist(self, arm: str, delta_yaw: float):
        return self._rotate(arm, [0, 0, 1], delta_yaw)

    def rotate_pitch(self, arm: str, delta_pitch: float):
        return self._rotate(arm, [0, 1, 0], delta_pitch)

    def vla_act(self, prompt: str, arm: str = "both", max_chunks: int = 24,
                stop: str = "chunk_budget", lift_thresh: float = 0.05,
                gripper_closed_thresh: float = 0.15):
        arms = self._arm(arm)
        if stop not in {"chunk_budget", "lift_and_grip", "benchmark_success"} or not 1 <= max_chunks <= 100:
            raise ValueError("invalid VLA stop predicate or chunk budget")
        min_z = {side: float(getattr(self.snapshot, f"{side}_endpose")[2]) for side in arms}
        chunks = 0
        postcondition = False
        for _ in range(max_chunks):
            if self.done:
                break
            self.vla_calls += 1
            actions = self.vla.predict_action_chunk(observation=self.observation, task="", subtask=prompt, memory="")
            with (self.output_dir / "vla_predictions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"vla_call": self.vla_calls, "tool_index": self.tool_index + 1,
                                         "reset_count": self.reset_count, "step_before": self.snapshot.step_count,
                                         "prompt": prompt, "active_arm": arm,
                                         "policy_input_joint_targets": jsonable(self.snapshot.joint_vector),
                                         "predicted_actions": jsonable(actions)}, ensure_ascii=False) + "\n")
            chunks += 1
            for predicted in actions:
                action = np.asarray(predicted, dtype=float).copy()
                if action.shape != (14,) or not np.isfinite(action).all():
                    raise ValueError("VLA must return finite 14D joint targets")
                if arm == "left":
                    action[7:] = self.snapshot.joint_vector[7:]
                elif arm == "right":
                    action[:7] = self.snapshot.joint_vector[:7]
                self._action(action, "qpos", arm)
                lifted = []
                for side in arms:
                    z = float(getattr(self.snapshot, f"{side}_endpose")[2])
                    min_z[side] = min(min_z[side], z)
                    grip = self.snapshot.joint_vector[6 if side == "left" else 13]
                    lifted.append(z - min_z[side] >= lift_thresh and grip < gripper_closed_thresh)
                postcondition = (stop == "lift_and_grip" and all(lifted)) or (stop == "benchmark_success" and self.snapshot.eval_success)
                if postcondition or self.done:
                    break
            if postcondition or self.done:
                break
        return {"chunks_used": chunks, "stop_predicate": stop, "postcondition_met": postcondition,
                "eval_success": self.snapshot.eval_success,
                "note": "lift-and-grip is a primitive return condition, not proof of object attachment or task success"}

    def reset(self):
        if self.reset_environment is None:
            raise ValueError("reset is disabled during evaluation")
        if time.monotonic() >= self.deadline:
            raise ValueError("reference exploration wall-clock budget exhausted")
        self.reset_environment()
        self.reset_count += 1
        self.refresh()
        return {"reset_count": self.reset_count}
