"""Data boundary for a future Reflector implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class UnifiedTrajectory:
    """Versioned, benchmark-neutral trajectory input for reflection."""

    trajectory_id: str
    instruction: str
    trace_events: Sequence[Mapping[str, Any]]
    outcome: Mapping[str, Any]
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class ReflectorInput:
    """Read-only snapshots supplied to a Reflector."""

    trajectory: UnifiedTrajectory
    scene_memory: Mapping[str, Any]
    experience_context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReflectorOutput:
    """Candidate experience output; never an automatic runtime mutation."""

    summary: str
    candidate_experience: Mapping[str, Any] = field(default_factory=dict)
    evidence_event_ids: Sequence[str] = field(default_factory=tuple)


@runtime_checkable
class Reflector(Protocol):
    def reflect(self, request: ReflectorInput) -> ReflectorOutput: ...


__all__ = [
    "Reflector",
    "ReflectorInput",
    "ReflectorOutput",
    "UnifiedTrajectory",
]
