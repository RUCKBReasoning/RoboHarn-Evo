from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .recovery_primitives import RecoveryPrimitivePlan
from ..monitoring.signals import MonitorSignal

if TYPE_CHECKING:
    from .skill_workflow_loader import SkillRecoveryWorkflowLoader


@dataclass(slots=True)
class RecoveryRoute:
    signal_name: str
    workflow_name: str
    plan: RecoveryPrimitivePlan
    post_recovery_intent: str
    selected_arm: str = "none"
    reason: str = ""


class RecoveryPolicyResolver:
    def __init__(self, workflow_loader: SkillRecoveryWorkflowLoader) -> None:
        self._workflow_loader = workflow_loader

    @property
    def last_error(self) -> str:
        return str(getattr(self._workflow_loader, "last_error", "") or "")

    @property
    def last_error_kind(self) -> str:
        return str(getattr(self._workflow_loader, "last_error_kind", "") or "")

    def resolve(
        self,
        *,
        signal: MonitorSignal,
        global_task: str = "",
        current_subtask: str = "",
        recovery_state: dict | None = None,
        observation_summary: str = "",
        recovery_history: list[str] | None = None,
        available_tools: list[str] | set[str] | None = None,
        semantic_tags: dict | None = None,
        robot_state: dict | None = None,
        scene_memory: dict | None = None,
        observation_preprocess: dict | None = None,
        preferred_arm: str = "either",
        blocked_grounded_setups: list[dict] | None = None,
    ) -> RecoveryRoute | None:
        return self._workflow_loader.resolve(
            signal_name=signal.name,
            reason=signal.reason,
            ood_scenario=str(signal.details.get("ood_scenario", signal.name)),
            global_task=global_task,
            current_subtask=current_subtask,
            observation_summary=observation_summary,
            robot_state=robot_state,
            recovery_state=recovery_state,
            recovery_history=recovery_history,
            available_tools=available_tools,
            semantic_tags=semantic_tags,
            scene_memory=scene_memory,
            observation_preprocess=observation_preprocess,
            preferred_arm=preferred_arm,
            blocked_grounded_setups=blocked_grounded_setups,
        )
