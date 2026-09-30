from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from roboharn_evo.benchmark_adapters.base import (
    ActionRequest,
    BenchmarkAdapter,
    EpisodeState,
    NeutralObservation,
)
from roboharn_evo.models.backend_interfaces import ExecutorBackend, PlannerBackend


class BenchmarkAgentLoopError(RuntimeError):
    """Raised when a benchmark-neutral loop contract is violated."""


class BenchmarkAgentBridge(Protocol):
    """Benchmark-owned projections without planning or task policy."""

    action_type: str
    replan_steps: int

    def initial_actions(self) -> Sequence[np.ndarray]: ...

    def planner_image(self, observation: NeutralObservation) -> np.ndarray: ...

    def planner_state(self, observation: NeutralObservation) -> np.ndarray: ...

    def executor_observation(
        self,
        observation: NeutralObservation,
    ) -> Mapping[str, Any]: ...

    def semantic_task_state(
        self,
        observation: NeutralObservation,
        episode_state: EpisodeState,
    ) -> Mapping[str, Any]: ...

    def validate_action_chunk(self, value: Any) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ActionPromptDecision:
    """One semantic Action Knowledge decision at the executor boundary."""

    prompt_before: str
    prompt_after: str
    knowledge_adopted: bool
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        before = str(self.prompt_before).strip()
        after = str(self.prompt_after).strip()
        if not before or not after:
            raise ValueError("Action Knowledge prompts must be non-empty")
        if not isinstance(self.knowledge_adopted, bool):
            raise TypeError("knowledge_adopted must be boolean")
        if self.knowledge_adopted and before == after:
            raise ValueError("adopted Action Knowledge must change the executor prompt")
        if not isinstance(self.audit, Mapping):
            raise TypeError("Action Knowledge audit must be an object")
        object.__setattr__(self, "prompt_before", before)
        object.__setattr__(self, "prompt_after", after)
        object.__setattr__(self, "audit", dict(self.audit))

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_before": self.prompt_before,
            "prompt_after": self.prompt_after,
            "knowledge_adopted": self.knowledge_adopted,
            **dict(self.audit),
        }


class ActionKnowledgePromptRuntime(Protocol):
    def reset_episode(self) -> None: ...

    def ground_executor_prompt(
        self,
        *,
        task: str,
        subtask: str,
        memory: str,
        semantic_tags: Mapping[str, Any],
        semantic_state: Mapping[str, Any],
    ) -> ActionPromptDecision | Mapping[str, Any]: ...


class PerceptionMemoryRuntime(Protocol):
    profile: str

    def reset_episode(
        self,
        *,
        task: str,
        initial_observation: NeutralObservation,
        initial_state: EpisodeState,
    ) -> None: ...

    def planner_context(self) -> Mapping[str, Any]: ...

    def planner_memory_text(self, previous_memory_text: str) -> str: ...

    def observe_before_action(
        self,
        *,
        task: str,
        subtask: str,
        memory_text: str,
        semantic_tags: Mapping[str, Any],
        observation: NeutralObservation,
        state: EpisodeState,
    ) -> Mapping[str, Any]: ...

    def observe_after_action(
        self,
        *,
        observation: NeutralObservation,
        state: EpisodeState,
    ) -> Mapping[str, Any]: ...

    def stats(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class BenchmarkAgentLoopConfig:
    max_actions: int
    max_planner_calls: int

    def __post_init__(self) -> None:
        for field_name in ("max_actions", "max_planner_calls"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class BenchmarkAgentLoopResult:
    task: str
    planner_calls: int
    executor_calls: int
    recovery_boundaries: int
    settle_actions: int
    policy_actions: int
    final_state: EpisodeState
    termination_reason: str
    final_subtask: str | None
    final_memory_text: str
    perception_memory_stats: Mapping[str, Any] | None = None

    @property
    def success(self) -> bool:
        return self.final_state.benchmark_success

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "planner_calls": self.planner_calls,
            "executor_calls": self.executor_calls,
            "recovery_boundaries": self.recovery_boundaries,
            "settle_actions": self.settle_actions,
            "policy_actions": self.policy_actions,
            "executed_actions": self.settle_actions + self.policy_actions,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "final_subtask": self.final_subtask,
            "final_memory_text": self.final_memory_text,
            "perception_memory_stats": (
                None
                if self.perception_memory_stats is None
                else dict(self.perception_memory_stats)
            ),
            "final_state": {
                "step_count": self.final_state.step_count,
                "step_limit": self.final_state.step_limit,
                "benchmark_success": self.final_state.benchmark_success,
                "check_success": self.final_state.check_success,
                "reward": self.final_state.reward,
                "terminated": self.final_state.terminated,
                "truncated": self.final_state.truncated,
            },
        }


@dataclass(frozen=True, slots=True)
class _ActiveSubtask:
    instruction: str
    preferred_arm: str
    semantic_tags: Mapping[str, Any]
    skill_name: str = "monitored-subtask-execution"
    status: str = "active"


EventSink = Callable[[Mapping[str, Any]], None]


class RoboHarnBenchmarkAgentLoop:
    """通过统一适配接口执行规划与动作。"""

    def __init__(
        self,
        *,
        planner: PlannerBackend,
        executor: ExecutorBackend,
        adapter: BenchmarkAdapter[Any],
        bridge: BenchmarkAgentBridge,
        config: BenchmarkAgentLoopConfig,
        task_hpk_runtime: Any = None,
        action_hpk_runtime: ActionKnowledgePromptRuntime | None = None,
        perception_memory_runtime: PerceptionMemoryRuntime | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        if not isinstance(planner, PlannerBackend):
            raise TypeError("planner must satisfy PlannerBackend")
        if not isinstance(executor, ExecutorBackend):
            raise TypeError("executor must satisfy ExecutorBackend")
        for method in (
            "initial_actions",
            "planner_image",
            "planner_state",
            "executor_observation",
            "semantic_task_state",
            "validate_action_chunk",
        ):
            if not callable(getattr(bridge, method, None)):
                raise TypeError(f"bridge must expose {method}")
        if not str(getattr(bridge, "action_type", "")).strip():
            raise TypeError("bridge.action_type must be non-empty")
        replan_steps = getattr(bridge, "replan_steps", None)
        if (
            isinstance(replan_steps, bool)
            or not isinstance(replan_steps, int)
            or replan_steps <= 0
        ):
            raise TypeError("bridge.replan_steps must be a positive integer")
        self.planner = planner
        self.executor = executor
        self.adapter = adapter
        self.bridge = bridge
        self.config = config
        self.task_hpk_runtime = task_hpk_runtime
        if action_hpk_runtime is not None:
            for method in ("reset_episode", "ground_executor_prompt"):
                if not callable(getattr(action_hpk_runtime, method, None)):
                    raise TypeError(f"action_hpk_runtime must expose {method}")
        self.action_hpk_runtime = action_hpk_runtime
        if perception_memory_runtime is not None:
            for method in (
                "reset_episode",
                "planner_context",
                "planner_memory_text",
                "observe_before_action",
                "observe_after_action",
                "stats",
            ):
                if not callable(getattr(perception_memory_runtime, method, None)):
                    raise TypeError(f"perception_memory_runtime must expose {method}")
            if str(getattr(perception_memory_runtime, "profile", "")) != (
                "perception_memory"
            ):
                raise TypeError(
                    "perception_memory_runtime profile must be perception_memory"
                )
        self.perception_memory_runtime = perception_memory_runtime
        self._event_sink = event_sink

    def run_episode(
        self,
        *,
        task: str,
        initial_observation: NeutralObservation,
    ) -> BenchmarkAgentLoopResult:
        task = str(task).strip()
        if not task:
            raise BenchmarkAgentLoopError("task must be non-empty")
        if not isinstance(initial_observation, NeutralObservation):
            raise BenchmarkAgentLoopError(
                "initial_observation must be a NeutralObservation"
            )

        self.planner.reset()
        self.executor.reset()
        reset_hpk = getattr(self.task_hpk_runtime, "reset_episode", None)
        if callable(reset_hpk):
            reset_hpk()
        if self.action_hpk_runtime is not None:
            self.action_hpk_runtime.reset_episode()

        observation = initial_observation
        segment_start = initial_observation
        state = self.adapter.episode_state()
        perception_runtime = self.perception_memory_runtime
        if perception_runtime is not None:
            perception_runtime.reset_episode(
                task=task,
                initial_observation=initial_observation,
                initial_state=state,
            )
        settle_actions = 0
        policy_actions = 0
        planner_calls = 0
        executor_calls = 0
        recovery_boundaries = 0
        memory_text = ""
        current_subtask: str | None = None
        active_subtask: _ActiveSubtask | None = None

        self._emit(
            "episode_start",
            task=task,
            action_type=self.bridge.action_type,
            task_hpk_enabled=self.task_hpk_runtime is not None,
            action_knowledge_enabled=self.action_hpk_runtime is not None,
            loop_profile=(
                "lite" if perception_runtime is None else perception_runtime.profile
            ),
        )

        for action_index, action in enumerate(self.bridge.initial_actions()):
            if self._finished(state) or state.step_count >= self.config.max_actions:
                break
            observation, state = self._execute_action(
                action=np.asarray(action),
                source="settle",
                planner_call_index=None,
                action_index=action_index,
            )
            settle_actions += 1
        segment_start = observation

        while not self._finished(state) and state.step_count < self.config.max_actions:
            if planner_calls >= self.config.max_planner_calls:
                break
            self._set_task_hpk_query(
                task=task,
                observation=observation,
                state=state,
                active_subtask=active_subtask,
                perception_memory_runtime=perception_runtime,
            )
            planner_memory_text = (
                memory_text
                if perception_runtime is None
                else perception_runtime.planner_memory_text(memory_text)
            )
            prediction = self.planner.predict_planner_step(
                task=task,
                previous_memory_text=planner_memory_text,
                planner_start_image=self.bridge.planner_image(segment_start),
                planner_end_image=self.bridge.planner_image(observation),
                planner_state=self.bridge.planner_state(observation),
            )
            planner_calls += 1
            decision = self._planner_decision(prediction)
            prior_subtask = current_subtask
            current_subtask = decision["subtask_text"]
            memory_text = decision["memory_text"]
            active_subtask = _ActiveSubtask(
                instruction=current_subtask,
                preferred_arm=decision["preferred_arm"],
                semantic_tags=decision["semantic_tags"],
                skill_name=decision["selected_skill"],
            )
            if prior_subtask is not None and decision["commit_label"] == "no_update":
                recovery_boundaries += 1
                self._emit(
                    "recovery_boundary",
                    planner_call_index=planner_calls - 1,
                    completed_executor_call_index=executor_calls - 1,
                    trigger="planner reported no committed state change after execution",
                    prior_subtask=prior_subtask,
                    recovery_subtask=current_subtask,
                )
            hpk_usage = prediction.get("hpk_v3_subtask_usage")
            self._emit(
                "planner_decision",
                planner_call_index=planner_calls - 1,
                commit_label=decision["commit_label"],
                selected_skill=decision["selected_skill"],
                preferred_arm=decision["preferred_arm"],
                subtask_before=prior_subtask,
                subtask_after=current_subtask,
                memory_text=memory_text,
                semantic_tags=decision["semantic_tags"],
                hpk_v3_subtask_usage=hpk_usage,
            )

            pre_execution_observation = observation
            pre_execution_state = state
            executor_prompt = current_subtask
            action_usage = None
            perception_before = None
            semantic_state = dict(self.bridge.semantic_task_state(observation, state))
            if perception_runtime is not None:
                perception_before = dict(
                    perception_runtime.observe_before_action(
                        task=task,
                        subtask=current_subtask,
                        memory_text=memory_text,
                        semantic_tags=decision["semantic_tags"],
                        observation=observation,
                        state=state,
                    )
                )
                semantic_state["scene memory"] = dict(
                    perception_runtime.planner_context()
                )
                self._emit(
                    "perception_memory_before_action",
                    planner_call_index=planner_calls - 1,
                    executor_call_index=executor_calls,
                    **perception_before,
                )
            if self.action_hpk_runtime is not None:
                raw_action_decision = self.action_hpk_runtime.ground_executor_prompt(
                    task=task,
                    subtask=current_subtask,
                    memory=memory_text,
                    semantic_tags=decision["semantic_tags"],
                    semantic_state=semantic_state,
                )
                if isinstance(raw_action_decision, ActionPromptDecision):
                    action_decision = raw_action_decision
                elif isinstance(raw_action_decision, Mapping):
                    try:
                        action_decision = ActionPromptDecision(
                            prompt_before=raw_action_decision["prompt_before"],
                            prompt_after=raw_action_decision["prompt_after"],
                            knowledge_adopted=raw_action_decision["knowledge_adopted"],
                            audit=raw_action_decision["audit"],
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        raise BenchmarkAgentLoopError(
                            "action HPK runtime returned an invalid prompt decision"
                        ) from exc
                else:
                    raise BenchmarkAgentLoopError(
                        "action HPK runtime must return a prompt decision object"
                    )
                if action_decision.prompt_before != current_subtask:
                    raise BenchmarkAgentLoopError(
                        "action HPK runtime changed the authoritative baseline prompt"
                    )
                executor_prompt = action_decision.prompt_after
                action_usage = action_decision.to_dict()
            chunk = self.bridge.validate_action_chunk(
                self.executor.predict_action_chunk(
                    observation=dict(self.bridge.executor_observation(observation)),
                    task=task,
                    subtask=executor_prompt,
                    memory=memory_text,
                )
            )
            executor_calls += 1
            if chunk.shape[0] < self.bridge.replan_steps:
                raise BenchmarkAgentLoopError(
                    "executor returned fewer actions than bridge.replan_steps"
                )
            self._emit(
                "executor_request",
                planner_call_index=planner_calls - 1,
                executor_call_index=executor_calls - 1,
                prompt=executor_prompt,
                prompt_before_action_hpk=current_subtask,
                hpk_v3_action_usage=action_usage,
                action_chunk_shape=list(chunk.shape),
                scheduled_actions=self.bridge.replan_steps,
            )
            for action_index, action in enumerate(chunk[: self.bridge.replan_steps]):
                if self._finished(state) or state.step_count >= self.config.max_actions:
                    break
                observation, state = self._execute_action(
                    action=action,
                    source="policy",
                    planner_call_index=planner_calls - 1,
                    action_index=action_index,
                )
                policy_actions += 1
            if perception_runtime is not None and state.step_count > (
                pre_execution_state.step_count
            ):
                perception_after = dict(
                    perception_runtime.observe_after_action(
                        observation=observation,
                        state=state,
                    )
                )
                self._emit(
                    "perception_memory_after_action",
                    planner_call_index=planner_calls - 1,
                    executor_call_index=executor_calls - 1,
                    **perception_after,
                )
            segment_start = pre_execution_observation

        reason = self._termination_reason(state, planner_calls=planner_calls)
        perception_stats = (
            None if perception_runtime is None else dict(perception_runtime.stats())
        )
        self._emit(
            "episode_end",
            termination_reason=reason,
            benchmark_success=state.benchmark_success,
            step_count=state.step_count,
            planner_calls=planner_calls,
            executor_calls=executor_calls,
            recovery_boundaries=recovery_boundaries,
            loop_profile=(
                "lite" if perception_runtime is None else perception_runtime.profile
            ),
            perception_memory_stats=perception_stats,
        )
        return BenchmarkAgentLoopResult(
            task=task,
            planner_calls=planner_calls,
            executor_calls=executor_calls,
            recovery_boundaries=recovery_boundaries,
            settle_actions=settle_actions,
            policy_actions=policy_actions,
            final_state=state,
            termination_reason=reason,
            final_subtask=current_subtask,
            final_memory_text=memory_text,
            perception_memory_stats=perception_stats,
        )

    def _set_task_hpk_query(
        self,
        *,
        task: str,
        observation: NeutralObservation,
        state: EpisodeState,
        active_subtask: _ActiveSubtask | None,
        perception_memory_runtime: PerceptionMemoryRuntime | None,
    ) -> None:
        runtime = self.task_hpk_runtime
        if runtime is None:
            return
        if getattr(runtime, "hierarchical_full_enabled", False) is not True:
            raise BenchmarkAgentLoopError(
                "task HPK loop requires hierarchical full mode"
            )
        builder = getattr(runtime, "build_preplanner_query", None)
        setter = getattr(self.planner, "set_hpk_preplanner_query", None)
        if not callable(builder) or not callable(setter):
            raise BenchmarkAgentLoopError(
                "task HPK requires query builder and planner query boundary"
            )
        scene_memory = dict(self.bridge.semantic_task_state(observation, state))
        if perception_memory_runtime is not None:
            scene_memory["structured scene memory"] = dict(
                perception_memory_runtime.planner_context()
            )
        setter(
            builder(
                overall_goal=task,
                scene_memory=scene_memory,
                active_skill=active_subtask,
            )
        )

    def _execute_action(
        self,
        *,
        action: np.ndarray,
        source: str,
        planner_call_index: int | None,
        action_index: int,
    ) -> tuple[NeutralObservation, EpisodeState]:
        result = self.adapter.execute(
            ActionRequest(
                action=action,
                action_type=self.bridge.action_type,
                metadata={
                    "source": source,
                    "planner_call_index": planner_call_index,
                    "action_index": action_index,
                },
            )
        )
        if not isinstance(result.post_observation, NeutralObservation):
            raise BenchmarkAgentLoopError(
                "neutral loop adapter returned a non-neutral observation"
            )
        self._emit(
            "action_executed",
            source=source,
            planner_call_index=planner_call_index,
            action_index=action_index,
            step_before=result.step_before,
            step_after=result.step_after,
            action=np.asarray(action, dtype=np.float32).reshape(-1).tolist(),
            reward=result.episode_state.reward,
            benchmark_success=result.episode_state.benchmark_success,
            terminated=result.episode_state.terminated,
            truncated=result.episode_state.truncated,
        )
        return result.post_observation, result.episode_state

    @staticmethod
    def _planner_decision(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise BenchmarkAgentLoopError("planner response must be an object")
        required = {
            "commit_label",
            "memory_text",
            "subtask_text",
            "preferred_arm",
        }
        missing = sorted(required - set(value))
        if missing:
            raise BenchmarkAgentLoopError(
                "planner response is missing: " + ", ".join(missing)
            )
        commit_label = str(value["commit_label"]).strip()
        if commit_label not in {"no_update", "subtask_complete", "state_change"}:
            raise BenchmarkAgentLoopError("planner commit_label is invalid")
        memory_text = str(value["memory_text"]).strip()
        subtask_text = str(value["subtask_text"]).strip()
        if not subtask_text:
            raise BenchmarkAgentLoopError("planner subtask_text must be non-empty")
        preferred_arm = str(value["preferred_arm"]).strip().lower()
        if preferred_arm not in {"left", "right", "either"}:
            raise BenchmarkAgentLoopError("planner preferred_arm is invalid")
        selected_skill = str(
            value.get("selected_skill", "monitored-subtask-execution")
        ).strip()
        if not selected_skill:
            raise BenchmarkAgentLoopError("planner selected_skill must be non-empty")
        semantic_tags = value.get("semantic_tags", {})
        if not isinstance(semantic_tags, Mapping):
            raise BenchmarkAgentLoopError("planner semantic_tags must be an object")
        return {
            "commit_label": commit_label,
            "memory_text": memory_text,
            "subtask_text": subtask_text,
            "preferred_arm": preferred_arm,
            "selected_skill": selected_skill,
            "semantic_tags": dict(semantic_tags),
        }

    @staticmethod
    def _finished(state: EpisodeState) -> bool:
        return state.benchmark_success or state.terminated or state.truncated

    def _termination_reason(
        self,
        state: EpisodeState,
        *,
        planner_calls: int,
    ) -> str:
        if state.benchmark_success:
            return "benchmark_success"
        if state.truncated:
            return "benchmark_truncated"
        if state.terminated:
            return "benchmark_terminated"
        if state.step_count >= self.config.max_actions:
            return "agent_action_limit"
        if planner_calls >= self.config.max_planner_calls:
            return "agent_planner_call_limit"
        return "agent_stopped"

    def _emit(self, event: str, **payload: Any) -> None:
        if self._event_sink is None:
            return
        self._event_sink(
            {
                "schema": "roboharn_evo/benchmark_agent_loop_event/v1",
                "event": event,
                **payload,
            }
        )


__all__ = [
    "ActionKnowledgePromptRuntime",
    "ActionPromptDecision",
    "BenchmarkAgentBridge",
    "BenchmarkAgentLoopConfig",
    "BenchmarkAgentLoopError",
    "BenchmarkAgentLoopResult",
    "PerceptionMemoryRuntime",
    "RoboHarnBenchmarkAgentLoop",
]
