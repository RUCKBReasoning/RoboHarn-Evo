"""Minimal benchmark integration boundary used by RoboHarn-Evo.

Adapters translate data and perform a caller-requested environment action. They
do not select actions, validate plans, implement Guards, or solve tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar

from roboharn_evo.agent.environment import EnvSnapshot


UnifiedObservation = EnvSnapshot


@dataclass(frozen=True, slots=True)
class NeutralObservation:
    """Benchmark-neutral sensor envelope.

    The envelope deliberately does not prescribe camera names, robot layout,
    episode signals, coordinate frames, or action semantics.  Benchmark
    adapters copy only fields that the benchmark actually supplied.  The
    existing RMBench path continues to return :class:`EnvSnapshot` through the
    ``UnifiedObservation`` compatibility alias.
    """

    raw: Mapping[str, Any]
    instruction: str
    cameras: Mapping[str, Any] = field(default_factory=dict)
    proprioception: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Mapping[str, Any] = field(default_factory=dict)


BenchmarkObservation = UnifiedObservation | NeutralObservation


@dataclass(frozen=True, slots=True)
class ActionRequest:
    """RoboHarn-Evo 动作请求；动作块的迭代由 Agent runtime 负责。"""

    action: Any
    action_type: str = "qpos"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EpisodeState:
    """Benchmark-owned episode signals without Agent-side interpretation."""

    step_count: int
    step_limit: int
    benchmark_success: bool
    check_success: bool
    reward: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class ActionResult:
    """Observed result of exactly one action request.

    There is deliberately no action ``success`` field: some benchmarks, such
    as RMBench, return no action status. ``step_advanced`` is only an observed
    counter fact and must not be treated as task or motion success.
    """

    request: ActionRequest
    raw_result: Any
    post_observation: BenchmarkObservation
    episode_state: EpisodeState
    step_before: int
    step_after: int

    @property
    def step_advanced(self) -> bool:
        return self.step_after > self.step_before


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Portable record of benchmark-owned episode outcome data."""

    success: bool
    terminated: bool
    truncated: bool
    reward: float
    steps: int
    reason: str = ""


RawObservationT = TypeVar("RawObservationT")


class BenchmarkAdapter(ABC, Generic[RawObservationT]):
    """根据 RoboHarn-Evo/RMBench 调用流程定义的接口。"""

    @abstractmethod
    def to_observation(self, raw_observation: RawObservationT) -> BenchmarkObservation:
        """将调用者提供的 benchmark 观测转换为 RoboHarn-Evo 格式。"""

    @abstractmethod
    def execute(self, request: ActionRequest) -> ActionResult:
        """Execute one requested action and return the observed post-state."""

    @abstractmethod
    def episode_state(self) -> EpisodeState:
        """Read benchmark episode counters and authoritative signals."""

    def episode_result(self, state: EpisodeState | None = None) -> EpisodeResult:
        """Translate episode state without adding Agent-specific judgement."""
        current = self.episode_state() if state is None else state
        reason = "benchmark_success" if current.benchmark_success else ""
        if not reason and current.truncated:
            reason = "step_limit"
        if not reason and current.terminated:
            reason = "benchmark_terminated"
        return EpisodeResult(
            success=current.benchmark_success,
            terminated=current.terminated,
            truncated=current.truncated,
            reward=current.reward,
            steps=current.step_count,
            reason=reason,
        )


__all__ = [
    "ActionRequest",
    "ActionResult",
    "BenchmarkObservation",
    "BenchmarkAdapter",
    "EpisodeResult",
    "EpisodeState",
    "NeutralObservation",
    "UnifiedObservation",
]
