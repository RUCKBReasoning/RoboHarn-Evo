from __future__ import annotations

import json
from typing import Any

from ...models.prompt_skills import load_prompt_skill


def _runtime_state_reasoning_harness() -> dict[str, Any]:
    try:
        return {
            "name": "runtime-state-reasoning",
            "instructions": load_prompt_skill("runtime-state-reasoning"),
        }
    except Exception:
        return {}


def build_reasoner_turn_payload(
    *,
    global_task: str,
    agent_state: dict[str, Any],
    available_skills: list[str],
    trigger: str,
    memory_harness: dict[str, Any] | None = None,
    runtime_evaluation: dict[str, Any] | None = None,
) -> str:
    payload = {
        "global_task": global_task.strip(),
        "trigger": trigger,
        "agent_state": agent_state,
        "available_skills": available_skills,
        "instruction": "Use runtime tools and current session state directly. Avoid reconstructing your own state summary if tool state is already available.",
    }
    if memory_harness:
        payload["memory_harness"] = memory_harness
    if runtime_evaluation is not None:
        payload["runtime_evaluation"] = dict(runtime_evaluation)
    runtime_state_reasoning = _runtime_state_reasoning_harness()
    if runtime_state_reasoning:
        payload["runtime_state_reasoning"] = runtime_state_reasoning
    return json.dumps(payload, ensure_ascii=False)
