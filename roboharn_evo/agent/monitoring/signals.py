from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


SignalLevel = Literal["info", "warning", "error"]

RUNNING = "running"
SUBTASK_SUCCESS = "subtask_success"
TASK_SUCCESS = "task_success"
STALL_DETECTED = "stall_detected"
STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
GRASP_LOST = "grasp_lost"
OBJECT_NOT_VISIBLE = "object_not_visible"
SCENE_DRIFT_DETECTED = "scene_drift_detected"
MOTION_BLOCKED = "motion_blocked"
INVALID_ACTION_PATTERN = "invalid_action_pattern"
NEEDS_RECOVERY_TOOLS = "needs_recovery_tools"
READY_TO_RETURN_TO_VLA = "ready_to_return_to_vla"
REQUIRES_REPLAN = "requires_replan"
REQUIRES_HUMAN = "requires_human"
TASK_LEVEL_RECOVERY_CONTROL = "task_level_recovery_control"


@dataclass(slots=True)
class MonitorSignal:
    name: str
    level: SignalLevel
    reason: str = ""
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def make_signal(
    name: str,
    *,
    level: SignalLevel = "info",
    reason: str = "",
    score: float = 0.0,
    details: dict[str, Any] | None = None,
) -> MonitorSignal:
    return MonitorSignal(
        name=name,
        level=level,
        reason=reason,
        score=score,
        details=dict(details or {}),
    )
