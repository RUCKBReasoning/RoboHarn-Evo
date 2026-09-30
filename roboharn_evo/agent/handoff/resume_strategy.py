from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..recovery.tool_specs import RecoveryToolResult


ResumeMode = Literal["resume_same_subtask", "rewrite_instruction", "replan"]


@dataclass(slots=True)
class ResumeDecision:
    mode: ResumeMode
    reason: str


class ResumeStrategy:
    def decide(
        self,
        *,
        recovery_results: list[RecoveryToolResult],
        current_subtask: str,
        current_memory: str,
    ) -> ResumeDecision:
        raise NotImplementedError("Resume strategy is not implemented yet.")
