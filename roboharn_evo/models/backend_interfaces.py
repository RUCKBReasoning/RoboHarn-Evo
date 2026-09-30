from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PlannerBackend(Protocol):
    def reset(self) -> None: ...

    def predict_planner_step(
        self,
        *,
        task: str,
        previous_memory_text: str,
        planner_start_image: np.ndarray,
        planner_end_image: np.ndarray,
        planner_state: np.ndarray,
    ) -> dict[str, Any]: ...


@runtime_checkable
class ExecutorBackend(Protocol):
    def reset(self) -> None: ...

    def predict_action_chunk(
        self,
        *,
        observation: dict[str, Any],
        task: str,
        subtask: str,
        memory: str,
    ) -> np.ndarray: ...


@runtime_checkable
class OODBackend(Protocol):
    def evaluate_ood(self, *, skill_payload: str) -> dict[str, Any]: ...


@runtime_checkable
class RecoveryBackend(Protocol):
    def plan_recovery(self, *, recovery_payload: dict[str, Any], media: list[dict[str, Any]] | None = None) -> dict[str, Any]: ...

