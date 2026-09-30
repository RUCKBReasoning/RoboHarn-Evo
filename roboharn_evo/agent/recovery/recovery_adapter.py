from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .tool_specs import RecoveryToolCall, RecoveryToolResult


@dataclass(slots=True)
class RecoveryExecutionResult:
    result: RecoveryToolResult
    latest_snapshot: Any | None = None


@dataclass(slots=True)
class RecoveryCapabilities:
    supported_tools: set[str] = field(default_factory=set)
    motion_modes: set[str] = field(default_factory=set)
    gripper_api: str = "none"
    notes: dict[str, Any] = field(default_factory=dict)

    def available_tools(self) -> list[str]:
        return sorted(self.supported_tools)


class RecoveryAdapter(Protocol):
    def capabilities(self) -> RecoveryCapabilities:
        ...

    def execute(self, call: RecoveryToolCall, latest_snapshot: Any | None) -> RecoveryExecutionResult:
        ...
