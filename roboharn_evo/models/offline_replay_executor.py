from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OfflineReplayExecutorConfig:
    action_dim: int = 14
    chunk_steps: int = 1


class OfflineReplayExecutor:
    """Executor stub for replaying existing trajectories through the agent loop.

    The recorded trajectory, not the action output, advances the offline replay
    environment. Returning a fixed dummy action lets the existing monitored
    rollout loop run without calling or retraining a VLA policy.
    """

    def __init__(self, config: OfflineReplayExecutorConfig) -> None:
        self.config = config

    def reset(self) -> None:
        return

    def predict_action_chunk(
        self,
        *,
        observation: dict,
        task: str,
        subtask: str,
        memory: str,
    ) -> np.ndarray:
        del observation, task, subtask, memory
        steps = max(1, int(self.config.chunk_steps))
        action_dim = max(1, int(self.config.action_dim))
        return np.zeros((steps, action_dim), dtype=np.float32)
