from __future__ import annotations

from dataclasses import dataclass, field

from .tool_specs import RecoveryToolCall


@dataclass(slots=True)
class RecoveryPrimitivePlan:
    name: str
    tool_calls: list[RecoveryToolCall] = field(default_factory=list)
    expected_outcome: str = ""
