from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ControlSignal(str, Enum):
    STARTUP = "startup"
    INTERVAL = "interval"
    MONITOR_ALERT = "monitor_alert"
    NO_ACTIVE_SKILL = "no_active_skill"
    RECOVERY_PENDING = "recovery_pending"
    SUCCESS = "success"


class SkillTransition(str, Enum):
    CONTINUE = "continue"
    START = "start"
    REPLACE = "replace"
    RETRY = "retry"
    FINISH = "finish"


@dataclass(frozen=True)
class RuntimeDecision:
    trigger: ControlSignal
    commit_label: str
    memory_text: str
    subtask_text: str
    transition: SkillTransition
    note: str = ""
