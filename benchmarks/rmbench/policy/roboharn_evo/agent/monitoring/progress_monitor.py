from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .signals import MonitorSignal, RUNNING, STALL_DETECTED, STEP_BUDGET_EXHAUSTED, SUBTASK_SUCCESS, TASK_SUCCESS, make_signal


@dataclass(slots=True)
class ProgressAssessment:
    made_progress: bool
    progress_score: float
    signal: MonitorSignal


class ProgressMonitor:
    def assess_step(
        self,
        *,
        previous_snapshot: Any,
        current_snapshot: Any,
        steps_used: int,
        max_steps: int,
        stall_count: int,
        stall_patience: int,
    ) -> ProgressAssessment:
        progress = current_snapshot if isinstance(current_snapshot, dict) else None
        if progress is None:
            raise NotImplementedError("ProgressMonitor currently expects precomputed progress metrics as current_snapshot.")

        image_delta = float(progress.get("image_delta", 0.0))
        joint_delta = float(progress.get("joint_delta", 0.0))
        made_progress = bool(progress.get("made_progress", False))
        progress_score = 1.0 if made_progress else 0.0
        if image_delta >= 5.0 or joint_delta >= 0.05:
            progress_score = 1.0
        elif image_delta >= 2.0 or joint_delta >= 0.01:
            progress_score = 0.5
        else:
            progress_score = 0.0

        if bool(progress.get("success", False)):
            signal_name = TASK_SUCCESS if bool(progress.get("task_success", False)) else SUBTASK_SUCCESS
            return ProgressAssessment(
                made_progress=True,
                progress_score=1.0,
                signal=make_signal(signal_name, reason="environment success signal", score=1.0),
            )
        if steps_used >= max_steps:
            return ProgressAssessment(
                made_progress=made_progress,
                progress_score=progress_score,
                signal=make_signal(STEP_BUDGET_EXHAUSTED, level="warning", reason="skill step budget exhausted", score=progress_score),
            )
        if stall_count >= stall_patience:
            return ProgressAssessment(
                made_progress=made_progress,
                progress_score=progress_score,
                signal=make_signal(STALL_DETECTED, level="warning", reason="stall patience exhausted", score=progress_score),
            )
        return ProgressAssessment(
            made_progress=made_progress,
            progress_score=progress_score,
            signal=make_signal(RUNNING, reason="rollout healthy" if made_progress else "rollout active but no clear progress", score=progress_score),
        )

    def detect_success(self, snapshot: Any) -> MonitorSignal | None:
        if getattr(snapshot, "eval_success", False):
            return make_signal(TASK_SUCCESS, reason="environment success signal", score=1.0)
        if getattr(snapshot, "check_success", False):
            return make_signal(SUBTASK_SUCCESS, reason="environment success signal", score=1.0)
        return None
