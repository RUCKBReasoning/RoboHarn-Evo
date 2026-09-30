
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from roboharn_evo.benchmark_adapters.base import EpisodeState, NeutralObservation
from roboharn_evo.benchmark_adapters.libero_pro.native_policy import (
    LiberoNativePolicyError,
    build_pi05_libero_state,
)
from roboharn_evo.benchmark_adapters.libero_pro.native_rollout import (
    validate_libero_native_action_chunk,
)

LIBERO_AGENT_PLANNER_PROMPT = """You are the high-level planner in a benchmark-neutral robot manipulation loop.
The benchmark instruction is preserved exactly. The two supplied images are
the external camera at the start and end of the latest execution segment. The
state vector is public LIBERO proprioception: end-effector position,
axis-angle orientation, and gripper position. LIBERO has one manipulator, so
set preferred_arm to either; never invent a left or right arm.

Return JSON only with:
{
  "commit_label": "no_update | subtask_complete | state_change",
  "memory_text": "one concise sentence about committed observed task state",
  "selected_skill": "monitored-subtask-execution",
  "subtask_text": "one concrete visual-motor objective for the native policy",
  "preferred_arm": "either",
  "semantic_tags": {
    "task_family": "pick_and_place | open_drawer | close_drawer | articulated_object | tool_use | other",
    "subtask_type": "grasp | place | open | close | move | align | reobserve | recover | other",
    "state_tags": {}
  }
}

Do not output coordinates, action vectors, object IDs, benchmark answers, or
low-level motion sequences. Do not claim success unless it is visible. Return
no prose outside JSON.

Task instruction: {task}
Previous committed memory: {previous_memory_text}
State summary vector: {state_summary}
"""


@dataclass(frozen=True, slots=True)
class LiberoProAgentBridge:
    """Project only measured LIBERO capabilities; never invent RMBench fields."""

    settle_steps: int = 10
    replan_steps: int = 5
    action_type: str = "libero"

    def __post_init__(self) -> None:
        if (
            isinstance(self.settle_steps, bool)
            or not isinstance(self.settle_steps, int)
            or self.settle_steps < 0
        ):
            raise ValueError("settle_steps must be a non-negative integer")
        if (
            isinstance(self.replan_steps, bool)
            or not isinstance(self.replan_steps, int)
            or self.replan_steps <= 0
        ):
            raise ValueError("replan_steps must be a positive integer")
        if self.action_type != "libero":
            raise ValueError("LIBERO Agent bridge action_type must remain 'libero'")

    def initial_actions(self) -> tuple[np.ndarray, ...]:
        action = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
        return tuple(action.copy() for _ in range(self.settle_steps))

    def planner_image(self, observation: NeutralObservation) -> np.ndarray:
        value = observation.cameras.get("agentview_image")
        image = np.asarray(value)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise LiberoNativePolicyError(
                "Agent planner requires public agentview_image as HWC uint8 RGB"
            )
        return image.copy()

    def planner_state(self, observation: NeutralObservation) -> np.ndarray:
        return build_pi05_libero_state(observation).copy()

    def executor_observation(
        self,
        observation: NeutralObservation,
    ) -> dict[str, Any]:
        return dict(observation.raw)

    def semantic_task_state(
        self,
        observation: NeutralObservation,
        episode_state: EpisodeState,
    ) -> dict[str, Any]:
        return {
            "benchmark": "LIBERO-PRO",
            "instruction": observation.instruction,
            "episode status": {
                "benchmark success": episode_state.benchmark_success,
                "terminated": episode_state.terminated,
                "truncated": episode_state.truncated,
            },
            "public observation": {
                "camera names": sorted(observation.cameras),
                "proprioception names": sorted(observation.proprioception),
            },
        }

    def validate_action_chunk(self, value: Any) -> np.ndarray:
        return validate_libero_native_action_chunk(value)


__all__ = ["LIBERO_AGENT_PLANNER_PROMPT", "LiberoProAgentBridge"]
