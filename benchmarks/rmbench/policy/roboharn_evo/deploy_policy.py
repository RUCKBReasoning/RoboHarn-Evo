from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


from benchmarks.rmbench.paths import project_root
from roboharn_evo.agent import RoboHarnAgentRuntime, build_agent_runtime
from roboharn_evo.agent.paths import resolve_writable_path
from roboharn_evo.models.backend_factory import build_executor_backend, build_planner_backend
from roboharn_evo.models.backend_interfaces import ExecutorBackend, PlannerBackend
from roboharn_evo.models.robobrain_adapter import (
    DEFAULT_ROBOBRAIN_PLANNER_PROMPT,
)


REPO_ROOT = project_root()


DEFAULT_EXECUTOR_PROMPT = "{subtask}"

DEFAULT_ROLLOUT_DUMP_ROOT = "eval_result/roboharn_rollouts"


@dataclass
class EncodedObservation:
    raw: dict[str, Any]
    head_rgb: np.ndarray
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    joint_vector: np.ndarray
    left_endpose: np.ndarray
    right_endpose: np.ndarray


class RoboHarnDeployRuntime:
    def __init__(
        self,
        *,
        planner_runtime: PlannerBackend,
        executor_runtime: ExecutorBackend,
        planning_interval: int,
        initial_memory_text: str,
    ) -> None:
        self.planner_runtime = planner_runtime
        self.executor_runtime = executor_runtime
        self.planning_interval = max(1, int(planning_interval))
        self.initial_memory_text = str(initial_memory_text)
        self.obs_cache: list[EncodedObservation] = []
        self.current_memory_text = self.initial_memory_text
        self.current_subtask = ""
        self.current_instruction = ""
        self.segment_start_obs: EncodedObservation | None = None
        self.plan_count = 0

    def set_instruction(self, instruction: str) -> None:
        self.current_instruction = str(instruction)

    def reset(self) -> None:
        self.obs_cache.clear()
        self.current_memory_text = self.initial_memory_text
        self.current_subtask = ""
        self.current_instruction = ""
        self.segment_start_obs = None
        self.plan_count = 0
        self.planner_runtime.reset()
        self.executor_runtime.reset()

    def update_obs(self, obs: EncodedObservation) -> None:
        self.obs_cache.append(obs)
        if len(self.obs_cache) > 32:
            self.obs_cache = self.obs_cache[-32:]
        if self.segment_start_obs is None:
            self.segment_start_obs = obs

    def _build_planner_input(self, start_obs: EncodedObservation, end_obs: EncodedObservation) -> dict[str, Any]:
        planner_state = np.concatenate(
            [
                start_obs.joint_vector,
                end_obs.joint_vector,
                end_obs.left_endpose,
                end_obs.right_endpose,
            ],
            axis=0,
        ).astype(np.float32)
        return {
            "planner_start_image": start_obs.head_rgb,
            "planner_end_image": end_obs.head_rgb,
            "planner_state": planner_state,
            "prev_memory_text": self.current_memory_text,
        }

    def _should_replan(self) -> bool:
        if not self.current_subtask:
            return True
        return (self.plan_count % self.planning_interval) == 0

    def get_action(self) -> np.ndarray:
        assert self.obs_cache, "Observation cache is empty. Call update_obs first."
        assert self.current_instruction, "Instruction is empty. Call set_instruction first."
        latest_obs = self.obs_cache[-1]
        start_obs = self.segment_start_obs or latest_obs

        if self._should_replan():
            planner_obs = self._build_planner_input(start_obs=start_obs, end_obs=latest_obs)
            planner_prediction = self.planner_runtime.predict_planner_step(
                task=self.current_instruction,
                previous_memory_text=self.current_memory_text,
                planner_start_image=planner_obs["planner_start_image"],
                planner_end_image=planner_obs["planner_end_image"],
                planner_state=planner_obs["planner_state"],
            )
            self.current_memory_text = str(planner_prediction["memory_text"])
            self.current_subtask = str(planner_prediction["subtask_text"])
            self.segment_start_obs = latest_obs

        self.plan_count += 1
        return self.executor_runtime.predict_action_chunk(
            observation=latest_obs.raw,
            task="",
            subtask=self.current_subtask,
            memory="",
        )


def encode_obs(observation: dict[str, Any]) -> EncodedObservation:
    return EncodedObservation(
        raw=observation,
        head_rgb=np.asarray(observation["observation"]["head_camera"]["rgb"], dtype=np.uint8),
        left_rgb=np.asarray(observation["observation"]["left_camera"]["rgb"], dtype=np.uint8),
        right_rgb=np.asarray(observation["observation"]["right_camera"]["rgb"], dtype=np.uint8),
        joint_vector=np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
        left_endpose=np.asarray(observation["endpose"]["left_endpose"], dtype=np.float32),
        right_endpose=np.asarray(observation["endpose"]["right_endpose"], dtype=np.float32),
    )


def get_model(usr_args: dict[str, Any]) -> RoboHarnDeployRuntime | RoboHarnAgentRuntime:
    formal_binding = None
    formal_request = usr_args.get("rmbench_formal")
    if isinstance(formal_request, dict) and set(formal_request) == {"config_path"}:
        from roboharn_evo.benchmark_adapters.rmbench.formal_binding import (
            resolve_formal_q1_binding,
        )

        formal_binding = resolve_formal_q1_binding(usr_args, formal_request)
        usr_args = formal_binding.arguments
    planner_cfg = usr_args.get("planner", {})
    if not planner_cfg:
        planner_cfg = {
            "backend": usr_args.get("planner_backend", "robobrain_adapter"),
            "robobrain": {
                "mode": usr_args.get("planner_mode", "remote"),
                "model_id": usr_args.get("planner_model_id", "BAAI/RoboBrain2.5-8B-NV"),
                "device_map": usr_args.get("planner_device_map", "auto"),
                "server_url": usr_args.get("planner_server_url", "http://127.0.0.1:9001/plan"),
                "timeout_sec": usr_args.get("planner_timeout_sec", 120),
                "do_sample": usr_args.get("planner_do_sample", False),
                "temperature": usr_args.get("planner_temperature", 0.0),
                "prompt_template": usr_args.get("planner_prompt_template", DEFAULT_ROBOBRAIN_PLANNER_PROMPT),
            },
        }
    executor_cfg = usr_args.get("executor", {})
    if not executor_cfg:
        executor_cfg = {
            "backend": usr_args.get("executor_backend", "pi05_adapter"),
            "pi05": {
                "train_config_name": usr_args.get("executor_train_config_name", "pi05_aloha_robotwin_full"),
                "model_name": usr_args["executor_model_name"],
                "checkpoint_id": usr_args["executor_checkpoint_id"],
                "pi0_step": usr_args.get("executor_pi0_step", 50),
                "prompt_template": usr_args.get("executor_prompt_template", DEFAULT_EXECUTOR_PROMPT),
            },
        }

    agent_cfg = usr_args.get("agent", {})
    if bool(agent_cfg.get("enabled", False)):
        merged_args = dict(usr_args)
        merged_args["planner"] = planner_cfg
        merged_args["executor"] = executor_cfg
        model = build_agent_runtime(merged_args)
        dump_cfg = dict(agent_cfg.get("rollout_dump", {}) or {})
        env_dump_root = str(os.getenv("ROBOHARN_EVO_ROLLOUT_DUMP_ROOT") or "").strip()
        dump_root = str(env_dump_root or dump_cfg.get("root", "") or usr_args.get("rollout_dump_root", "")).strip()
        dump_enabled = bool(env_dump_root) or bool(dump_cfg.get("enabled", False))
        model.rollout_dump_root = str(
            resolve_writable_path(dump_root or DEFAULT_ROLLOUT_DUMP_ROOT)
        )
        model.rollout_dump_enabled = dump_enabled
        model.rollout_video_fps = int(dump_cfg.get("video_fps", usr_args.get("rollout_video_fps", 10)))
        model._active_rollout_key = ""
        hpk_v3_cfg = agent_cfg.get("hpk_v3", {})
        frozen_goal_value = (
            hpk_v3_cfg.get("rmbench_frozen_goal_context")
            if isinstance(hpk_v3_cfg, dict)
            else None
        )
        if frozen_goal_value not in (None, "", {}):
            from roboharn_evo.benchmark_adapters.rmbench.v31_goal_bridge import (
                RMBenchFrozenGoalContextV31,
                install_rmbench_v31_agent,
            )

            frozen_goal = (
                RMBenchFrozenGoalContextV31.from_mapping(frozen_goal_value)
                if isinstance(frozen_goal_value, dict)
                else RMBenchFrozenGoalContextV31.load(str(frozen_goal_value))
            )
            install_rmbench_v31_agent(model, frozen_goal)
        formal_method = usr_args.get("rmbench_formal")
        if formal_method not in (None, {}):
            from roboharn_evo.benchmark_adapters.rmbench.formal_methods import (
                install_flat_reflection_method,
            )

            install_flat_reflection_method(model, formal_method)
        if formal_binding is not None:
            model.rmbench_formal_method = formal_binding.method
            model.rmbench_formal_cell_id = formal_binding.cell_id
            model.rmbench_formal_source_pool = formal_binding.source_pool
        return model

    planner_runtime = build_planner_backend(planner_cfg)
    executor_runtime = build_executor_backend(executor_cfg)
    return RoboHarnDeployRuntime(
        planner_runtime=planner_runtime,
        executor_runtime=executor_runtime,
        planning_interval=int(usr_args.get("planning_interval", 1)),
        initial_memory_text=str(usr_args.get("initial_memory_text", "The task has started.")),
    )


def _safe_name(value: Any, *, fallback: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        text = fallback
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or fallback


def _task_env_attr(task_env: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(task_env, name):
            value = getattr(task_env, name)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            if value is not None:
                return value
    return default


def _maybe_configure_agent_dump(model: RoboHarnAgentRuntime, task_env: Any, observation: dict[str, Any]) -> None:
    if not bool(getattr(model, "rollout_dump_enabled", False)):
        return
    root_value = str(getattr(model, "rollout_dump_root", "") or "").strip() or DEFAULT_ROLLOUT_DUMP_ROOT
    root = resolve_writable_path(root_value)
    episode_id = _task_env_attr(task_env, ("episode_id", "episode_idx", "episode_index", "task_id"), default=-1)
    seed = _task_env_attr(task_env, ("seed", "env_seed", "episode_seed"), default=-1)
    instruction = ""
    try:
        instruction = str(task_env.get_instruction())
    except Exception:
        instruction = str(observation.get("instruction", ""))
    key = f"{episode_id}:{seed}:{instruction}"
    if getattr(model, "_active_rollout_key", "") == key:
        return
    if getattr(model, "_active_rollout_key", ""):
        model.finalize_rollout_videos(fps=int(getattr(model, "rollout_video_fps", 10)))
    episode_name = _safe_name(f"episode_{episode_id}", fallback="episode_unknown")
    seed_name = _safe_name(f"seed_{seed}", fallback="seed_unknown")
    task_name = _safe_name(instruction[:80], fallback="task")
    rollout_dir = resolve_writable_path(
        root / f"{episode_name}_{seed_name}_{task_name}"
    )
    model.set_trace_file(str(rollout_dir / "agent_trace.jsonl"), episode_id=int(episode_id) if str(episode_id).lstrip("-").isdigit() else -1, seed=int(seed) if str(seed).lstrip("-").isdigit() else -1)
    model.set_rollout_dump_dir(str(rollout_dir))
    model.write_rollout_meta(
        {
            "episode_id": episode_id,
            "seed": seed,
            "instruction": instruction,
            "rollout_dir": str(rollout_dir),
        }
    )
    model._active_rollout_key = key


def eval(TASK_ENV, model: RoboHarnDeployRuntime | RoboHarnAgentRuntime, observation: dict[str, Any]) -> None:
    if isinstance(model, RoboHarnAgentRuntime):
        _maybe_configure_agent_dump(model, TASK_ENV, observation)
        model.run_eval_step(TASK_ENV, observation)
        return

    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()
    if len(model.obs_cache) == 0:
        model.set_instruction(instruction)
        model.update_obs(obs)

    actions = model.get_action()
    for action in actions:
        TASK_ENV.take_action(action, action_type="qpos")
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model: RoboHarnDeployRuntime | RoboHarnAgentRuntime) -> None:
    if isinstance(model, RoboHarnAgentRuntime):
        if getattr(model, "_active_rollout_key", ""):
            model.finalize_rollout_videos(fps=int(getattr(model, "rollout_video_fps", 10)))
            model._active_rollout_key = ""
    model.reset()
