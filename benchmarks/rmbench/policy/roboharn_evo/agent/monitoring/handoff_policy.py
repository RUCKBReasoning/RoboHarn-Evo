from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .signals import MonitorSignal, NEEDS_RECOVERY_TOOLS, REQUIRES_HUMAN, REQUIRES_REPLAN, RUNNING, STALL_DETECTED, STEP_BUDGET_EXHAUSTED


HandoffTarget = Literal["vla", "recovery_tools", "replan", "human"]


@dataclass(slots=True)
class HandoffDecision:
    target: HandoffTarget
    reason: str
    signal_name: str


class HandoffPolicy:
    def decide(
        self,
        *,
        signals: list[MonitorSignal],
        recovery_pending: bool,
        active_subtask: str,
    ) -> HandoffDecision:
        if recovery_pending:
            return HandoffDecision(target="recovery_tools", reason="pending recovery action exists", signal_name=NEEDS_RECOVERY_TOOLS)
        if not active_subtask:
            return HandoffDecision(target="replan", reason="no active subtask to execute", signal_name=REQUIRES_REPLAN)
        for signal in signals:
            if signal.name == REQUIRES_HUMAN:
                return HandoffDecision(target="human", reason=signal.reason, signal_name=signal.name)
            if signal.name == REQUIRES_REPLAN:
                return HandoffDecision(target="replan", reason=signal.reason, signal_name=signal.name)
            if signal.name == NEEDS_RECOVERY_TOOLS:
                return HandoffDecision(target="recovery_tools", reason=signal.reason, signal_name=signal.name)
            if signal.name == STEP_BUDGET_EXHAUSTED:
                return HandoffDecision(target="replan", reason=signal.reason or "step budget exhausted", signal_name=signal.name)
            if signal.name == STALL_DETECTED:
                return HandoffDecision(target="recovery_tools", reason=signal.reason or "stall detected", signal_name=signal.name)
        return HandoffDecision(target="vla", reason="continue default VLA execution", signal_name=RUNNING)
