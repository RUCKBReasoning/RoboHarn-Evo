from __future__ import annotations

from typing import Any

from ..recovery.tool_specs import RecoveryToolResult


class SubtaskReentry:
    def can_resume(
        self,
        *,
        current_subtask: str,
        snapshot: Any,
        recovery_results: list[RecoveryToolResult],
    ) -> bool:
        raise NotImplementedError("Subtask reentry checking is not implemented yet.")

    def rewrite_subtask_instruction(
        self,
        *,
        global_task: str,
        current_subtask: str,
        committed_memory: str,
        recovery_reason: str,
    ) -> str:
        raise NotImplementedError("Subtask reentry instruction rewriting is not implemented yet.")
