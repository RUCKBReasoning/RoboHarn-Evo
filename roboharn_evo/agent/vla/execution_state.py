from __future__ import annotations

from dataclasses import dataclass, field

from .execution_request import VLAExecutionRequest


@dataclass(slots=True)
class VLAExecutionState:
    active: bool = False
    global_task: str = ""
    current_subtask: str = ""
    last_instruction_text: str = ""
    last_instruction_payload: dict[str, object] = field(default_factory=dict)
    last_action_chunk_size: int = 0
    recovered_since_last_instruction: bool = False
    last_recovery_reason: str = ""


class VLAExecutionStateStore:
    def __init__(self) -> None:
        self.state = VLAExecutionState()

    def activate(self, request: VLAExecutionRequest) -> None:
        self.state.active = True
        self.state.global_task = request.global_task
        self.state.current_subtask = request.current_subtask
        self.state.last_instruction_text = request.instruction_text
        self.state.last_instruction_payload = dict(request.instruction_payload)
        self.state.recovered_since_last_instruction = False
        self.state.last_recovery_reason = ""

    def record_action_chunk(self, chunk_size: int) -> None:
        self.state.last_action_chunk_size = max(0, int(chunk_size))

    def record_recovery(self, reason: str) -> None:
        self.state.recovered_since_last_instruction = True
        self.state.last_recovery_reason = reason

    def clear(self) -> None:
        self.state = VLAExecutionState()
