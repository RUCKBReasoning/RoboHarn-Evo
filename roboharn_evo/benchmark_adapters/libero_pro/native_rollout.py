"""Minimal LIBERO-native Off-policy rollout loop.

The loop owns only action-chunk iteration.  It does not plan subtasks, alter
policy actions, retry failures, inspect benchmark ground truth, or add
task-specific recovery behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from roboharn_evo.benchmark_adapters.base import (
    ActionRequest,
    BenchmarkAdapter,
    EpisodeState,
    NeutralObservation,
)


class LiberoNativeRolloutError(RuntimeError):
    """Raised when a native policy violates the rollout contract."""


class LiberoNativeChunkPolicy(Protocol):
    """Small policy surface needed by the native Off runner."""

    def reset(self) -> None: ...

    def predict_native_action_chunk(
        self,
        observation: NeutralObservation | Mapping[str, Any],
        *,
        prompt: str,
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class NativeActionRecord:
    """One policy-proposed action and its benchmark-owned result."""

    source: str
    policy_call_index: int | None
    action_index_in_chunk: int
    step_before: int
    step_after: int
    action: tuple[float, ...]
    reward: float
    benchmark_success: bool
    terminated: bool
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "policy_call_index": self.policy_call_index,
            "action_index_in_chunk": self.action_index_in_chunk,
            "step_before": self.step_before,
            "step_after": self.step_after,
            "action": list(self.action),
            "reward": self.reward,
            "benchmark_success": self.benchmark_success,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class LiberoNativeOffResult:
    """Portable result of one native-policy episode."""

    prompt: str
    policy_calls: int
    proposed_actions: int
    executed_actions: int
    settle_actions: int
    policy_actions: int
    final_state: EpisodeState
    termination_reason: str
    action_records: tuple[NativeActionRecord, ...]

    @property
    def success(self) -> bool:
        return self.final_state.benchmark_success

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "off",
            "policy_interface": "libero_native_action_chunk",
            "prompt": self.prompt,
            "policy_calls": self.policy_calls,
            "proposed_actions": self.proposed_actions,
            "executed_actions": self.executed_actions,
            "settle_actions": self.settle_actions,
            "policy_actions": self.policy_actions,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "final_state": {
                "step_count": self.final_state.step_count,
                "step_limit": self.final_state.step_limit,
                "benchmark_success": self.final_state.benchmark_success,
                "check_success": self.final_state.check_success,
                "reward": self.final_state.reward,
                "terminated": self.final_state.terminated,
                "truncated": self.final_state.truncated,
            },
            "actions": [record.to_dict() for record in self.action_records],
        }


ActionObserver = Callable[[NativeActionRecord], None]


def run_libero_native_off_episode(
    *,
    adapter: BenchmarkAdapter[Any],
    policy: LiberoNativeChunkPolicy,
    initial_observation: NeutralObservation,
    prompt: str,
    max_steps: int,
    replan_steps: int,
    settle_steps: int = 0,
    action_observer: ActionObserver | None = None,
) -> LiberoNativeOffResult:
    """Execute one policy episode without modifying or retrying its actions."""

    prompt = str(prompt).strip()
    if not prompt:
        raise LiberoNativeRolloutError("native Off prompt must be non-empty")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise LiberoNativeRolloutError("max_steps must be a positive integer")
    if (
        isinstance(replan_steps, bool)
        or not isinstance(replan_steps, int)
        or replan_steps <= 0
    ):
        raise LiberoNativeRolloutError("replan_steps must be a positive integer")
    if (
        isinstance(settle_steps, bool)
        or not isinstance(settle_steps, int)
        or settle_steps < 0
    ):
        raise LiberoNativeRolloutError("settle_steps must be a non-negative integer")
    if not isinstance(initial_observation, NeutralObservation):
        raise LiberoNativeRolloutError(
            "initial_observation must be a NeutralObservation"
        )

    policy.reset()
    observation = initial_observation
    state = adapter.episode_state()
    records: list[NativeActionRecord] = []
    policy_calls = 0
    proposed_actions = 0

    settle_action = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
    for settle_index in range(settle_steps):
        if _episode_finished(state) or state.step_count >= max_steps:
            break
        observation, state, record = _execute_native_action(
            adapter=adapter,
            action=settle_action,
            source="settle",
            policy_call_index=None,
            action_index_in_chunk=settle_index,
        )
        records.append(record)
        if action_observer is not None:
            action_observer(record)

    while not _episode_finished(state) and state.step_count < max_steps:
        chunk = validate_libero_native_action_chunk(
            policy.predict_native_action_chunk(observation, prompt=prompt)
        )
        policy_call_index = policy_calls
        policy_calls += 1
        proposed_actions += int(chunk.shape[0])
        if chunk.shape[0] < replan_steps:
            raise LiberoNativeRolloutError(
                "native policy predicted fewer actions than replan_steps: "
                f"{chunk.shape[0]} < {replan_steps}"
            )

        for action_index, action in enumerate(chunk[:replan_steps]):
            if _episode_finished(state) or state.step_count >= max_steps:
                break
            observation, state, record = _execute_native_action(
                adapter=adapter,
                action=action,
                source="policy",
                policy_call_index=policy_call_index,
                action_index_in_chunk=action_index,
            )
            records.append(record)
            if action_observer is not None:
                action_observer(record)

    return LiberoNativeOffResult(
        prompt=prompt,
        policy_calls=policy_calls,
        proposed_actions=proposed_actions,
        executed_actions=len(records),
        settle_actions=sum(record.source == "settle" for record in records),
        policy_actions=sum(record.source == "policy" for record in records),
        final_state=state,
        termination_reason=_termination_reason(state, max_steps=max_steps),
        action_records=tuple(records),
    )


def _execute_native_action(
    *,
    adapter: BenchmarkAdapter[Any],
    action: np.ndarray,
    source: str,
    policy_call_index: int | None,
    action_index_in_chunk: int,
) -> tuple[NeutralObservation, EpisodeState, NativeActionRecord]:
    result = adapter.execute(
        ActionRequest(
            action=action,
            action_type="libero",
            metadata={
                "source": source,
                "policy_call_index": policy_call_index,
                "action_index_in_chunk": action_index_in_chunk,
            },
        )
    )
    if not isinstance(result.post_observation, NeutralObservation):
        raise LiberoNativeRolloutError(
            "LIBERO native adapter returned a non-neutral observation"
        )
    state = result.episode_state
    record = NativeActionRecord(
        source=source,
        policy_call_index=policy_call_index,
        action_index_in_chunk=action_index_in_chunk,
        step_before=result.step_before,
        step_after=result.step_after,
        action=tuple(float(value) for value in action),
        reward=state.reward,
        benchmark_success=state.benchmark_success,
        terminated=state.terminated,
        truncated=state.truncated,
    )
    return result.post_observation, state, record


def validate_libero_native_action_chunk(value: Any) -> np.ndarray:
    try:
        chunk = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise LiberoNativeRolloutError(
            "native policy action chunk must be numeric"
        ) from exc
    if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != 7:
        raise LiberoNativeRolloutError(
            f"native policy action chunk must have shape (T, 7), got {chunk.shape}"
        )
    if not np.isfinite(chunk).all():
        raise LiberoNativeRolloutError(
            "native policy action chunk contains non-finite values"
        )
    return chunk


def _episode_finished(state: EpisodeState) -> bool:
    return state.benchmark_success or state.terminated or state.truncated


def _termination_reason(state: EpisodeState, *, max_steps: int) -> str:
    if state.benchmark_success:
        return "benchmark_success"
    if state.truncated:
        return "benchmark_truncated"
    if state.terminated:
        return "benchmark_terminated"
    if state.step_count >= max_steps:
        return "runner_step_limit"
    return "not_started"


__all__ = [
    "LiberoNativeChunkPolicy",
    "LiberoNativeOffResult",
    "LiberoNativeRolloutError",
    "NativeActionRecord",
    "run_libero_native_off_episode",
    "validate_libero_native_action_chunk",
]
