from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OfflineReplayPlannerConfig:
    subtask_text: str = ""
    memory_text: str = "Offline replay is observing a recorded trajectory."
    commit_label: str = "state_change"


class OfflineReplayPlanner:
    """Deterministic planner stub for offline replay smoke tests."""

    def __init__(self, config: OfflineReplayPlannerConfig) -> None:
        self.config = config

    def reset(self) -> None:
        return

    def predict_planner_step(
        self,
        *,
        task: str,
        previous_memory_text: str,
        planner_start_image: np.ndarray,
        planner_end_image: np.ndarray,
        planner_state: np.ndarray,
    ) -> dict[str, Any]:
        del planner_start_image, planner_end_image, planner_state
        subtask = self.config.subtask_text.strip() or self._fallback_subtask(task)
        memory = self.config.memory_text.strip() or previous_memory_text.strip() or "Offline replay is running."
        return {
            "commit_label": self.config.commit_label,
            "memory_text": memory,
            "subtask_text": subtask,
            "semantic_tags": {
                "task_family": "other",
                "subtask_type": "other",
                "state_tags": {},
                "tag_source": "offline_replay_planner",
            },
            "planner_backend": "offline_replay",
        }

    def _fallback_subtask(self, task: str) -> str:
        text = str(task).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            global_task = str(payload.get("global_task", "")).strip()
            if global_task:
                return global_task
        return text or "continue the recorded trajectory"
