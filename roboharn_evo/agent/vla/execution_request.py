from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class VLAExecutionRequest:
    instruction_text: str
    instruction_payload: dict[str, Any]
    observation: Any
    current_subtask: str
    global_task: str
    committed_memory: str


def build_vla_execution_request(
    *,
    instruction_text: str,
    instruction_payload: dict[str, Any],
    observation: Any,
    current_subtask: str,
    global_task: str,
    committed_memory: str,
) -> VLAExecutionRequest:
    return VLAExecutionRequest(
        instruction_text=instruction_text,
        instruction_payload=instruction_payload,
        observation=observation,
        current_subtask=current_subtask,
        global_task=global_task,
        committed_memory=committed_memory,
    )
