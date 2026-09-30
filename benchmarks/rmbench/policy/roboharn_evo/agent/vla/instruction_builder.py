from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VLAExecutionContext:
    retry_count: int = 0
    recovered: bool = False
    recovery_reason: str = ""
    stage: str = "execution"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VLAInstructionPayload:
    global_task: str
    current_subtask: str
    committed_memory: str
    execution_context: VLAExecutionContext


class VLAInstructionBuilder:
    def build(
        self,
        *,
        global_task: str,
        current_subtask: str,
        committed_memory: str,
        retry_count: int = 0,
        recovered: bool = False,
        recovery_reason: str = "",
        stage: str = "execution",
        extra: dict[str, Any] | None = None,
    ) -> VLAInstructionPayload:
        return VLAInstructionPayload(
            global_task=global_task,
            current_subtask=current_subtask,
            committed_memory=committed_memory,
            execution_context=VLAExecutionContext(
                retry_count=retry_count,
                recovered=recovered,
                recovery_reason=recovery_reason,
                stage=stage,
                extra=dict(extra or {}),
            ),
        )

    def render_text(self, payload: VLAInstructionPayload) -> str:
        return payload.current_subtask.strip()
