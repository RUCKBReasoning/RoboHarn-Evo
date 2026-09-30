"""RoboHarn-Evo-side RMBench adapter.

This module imports no RMBench implementation. It works through the small
duck-typed environment surface that RMBench already exposes, allowing RoboHarn-Evo to
remain installable independently while RMBench stays unchanged in this phase.
"""

from __future__ import annotations

from typing import Any, Callable

from roboharn_evo.agent.environment import RMBenchEnvAdapter
from roboharn_evo.benchmark_adapters.base import (
    ActionRequest,
    ActionResult,
    BenchmarkAdapter,
    EpisodeState,
    UnifiedObservation,
)


class RMBenchAdapter(BenchmarkAdapter[dict[str, Any]]):
    """Translate the existing RMBench environment contract without policy."""

    def __init__(
        self,
        task_env: Any,
        *,
        observation_provider: Callable[[], dict[str, Any]] | None = None,
        oracle_objects_enabled: bool = False,
    ) -> None:
        self.task_env = task_env
        self._observation_provider = observation_provider or task_env.get_obs
        self.oracle_objects_enabled = bool(oracle_objects_enabled)

    def to_observation(self, raw_observation: dict[str, Any]) -> UnifiedObservation:
        return RMBenchEnvAdapter.from_env(
            self.task_env,
            raw_observation,
            oracle_objects_enabled=self.oracle_objects_enabled,
        )

    def execute(self, request: ActionRequest) -> ActionResult:
        step_before = int(getattr(self.task_env, "take_action_cnt", 0) or 0)
        # Exceptions deliberately propagate: hiding environment errors would
        # change the established runtime failure semantics.
        raw_result = self.task_env.take_action(
            request.action,
            action_type=request.action_type,
        )
        raw_observation = self._observation_provider()
        post_observation = self.to_observation(raw_observation)
        state = self._state_from_observation(post_observation)
        return ActionResult(
            request=request,
            raw_result=raw_result,
            post_observation=post_observation,
            episode_state=state,
            step_before=step_before,
            step_after=state.step_count,
        )

    def episode_state(self) -> EpisodeState:
        step_count = int(getattr(self.task_env, "take_action_cnt", 0) or 0)
        step_limit = int(getattr(self.task_env, "step_lim", 0) or 0)
        benchmark_success = bool(getattr(self.task_env, "eval_success", False))
        check_success = False
        checker = getattr(self.task_env, "check_success", None)
        if callable(checker):
            try:
                check_success = bool(checker())
            except Exception:
                check_success = False
        truncated = step_limit > 0 and step_count >= step_limit and not benchmark_success
        return EpisodeState(
            step_count=step_count,
            step_limit=step_limit,
            benchmark_success=benchmark_success,
            check_success=check_success,
            reward=float(getattr(self.task_env, "max_reward", 0.0) or 0.0),
            terminated=benchmark_success,
            truncated=truncated,
        )

    @staticmethod
    def _state_from_observation(observation: UnifiedObservation) -> EpisodeState:
        truncated = (
            observation.step_limit > 0
            and observation.step_count >= observation.step_limit
            and not observation.eval_success
        )
        return EpisodeState(
            step_count=observation.step_count,
            step_limit=observation.step_limit,
            benchmark_success=observation.eval_success,
            check_success=observation.check_success,
            reward=observation.max_reward,
            terminated=observation.eval_success,
            truncated=truncated,
        )


__all__ = ["RMBenchAdapter"]
