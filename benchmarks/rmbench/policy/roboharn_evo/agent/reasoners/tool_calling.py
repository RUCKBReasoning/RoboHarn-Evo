from __future__ import annotations

import json
from typing import Any, Callable

from ..components.memory_manager import MemoryManager
from ..skills.registry import SkillRegistry


RUNTIME_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_agent_state",
            "description": "Read the full runtime state, including monitor and recovery policy state.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "env_summary",
            "description": "Read the latest environment summary string derived from observation and env status.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "List available skills the runtime can schedule.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_monitor_status",
            "description": "Read the protocolized monitor status for the active rollout.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_recovery_policy",
            "description": "Read current workflow-defined recovery budgets and pending action.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_reasoning_note",
            "description": "Append a short note to runtime tool-call history for tracing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "decide_next_action",
            "description": "Finalize the decision turn with a unified chat/tool-loop action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_mode": {
                        "type": "string",
                        "enum": ["continue", "start", "switch", "retry", "reset", "replan", "recover", "finish"],
                    },
                    "commit_label": {
                        "type": "string",
                        "enum": ["no_update", "subtask_complete", "state_change"],
                    },
                    "subtask_text": {"type": "string"},
                    "preferred_arm": {
                        "type": "string",
                        "enum": ["left", "right", "either"],
                    },
                    "memory_text": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["action_mode", "commit_label", "subtask_text", "memory_text", "preferred_arm"],
            },
        },
    },
]


class RuntimeToolExecutor:
    def __init__(
        self,
        *,
        memory_store: MemoryManager,
        skill_registry: SkillRegistry,
        get_env_summary: Callable[[], str],
    ) -> None:
        self.memory_store = memory_store
        self.skill_registry = skill_registry
        self.get_env_summary = get_env_summary
        self.last_decision: dict[str, Any] | None = None
        self._handlers: dict[str, Callable[..., str]] = {
            "read_agent_state": self._handle_read_agent_state,
            "env_summary": self._handle_env_summary,
            "list_skills": self._handle_list_skills,
            "read_monitor_status": self._handle_read_monitor_status,
            "read_recovery_policy": self._handle_read_recovery_policy,
            "record_reasoning_note": self._handle_record_reasoning_note,
            "decide_next_action": self._handle_decide_next_action,
        }

    def reset(self) -> None:
        self.last_decision = None

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        handler = self._handlers.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)
        try:
            return handler(**arguments)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    def _handle_read_agent_state(self) -> str:
        self.memory_store.set_monitor_status(phase="tool_call", status="blocked_on_tools", note="reasoner reading agent state")
        return json.dumps(self.memory_store.to_reasoner_context(), ensure_ascii=False)

    def _handle_env_summary(self) -> str:
        return json.dumps({"env_summary": self.get_env_summary()}, ensure_ascii=False)

    def _handle_list_skills(self) -> str:
        return json.dumps({"skills": self.skill_registry.list_skill_summaries()}, ensure_ascii=False)

    def _handle_read_monitor_status(self) -> str:
        return json.dumps(self.memory_store.state.monitor.to_dict(), ensure_ascii=False)

    def _handle_read_recovery_policy(self) -> str:
        return json.dumps(self.memory_store.state.recovery.to_dict(), ensure_ascii=False)

    def _handle_record_reasoning_note(self, note: str) -> str:
        self.memory_store.record_tool_call(f"reasoning_note: {note}")
        return json.dumps({"status": "ok"}, ensure_ascii=False)

    def _handle_decide_next_action(
        self,
        action_mode: str,
        commit_label: str,
        subtask_text: str,
        memory_text: str,
        preferred_arm: str,
        note: str = "",
    ) -> str:
        self.last_decision = {
            "action_mode": action_mode,
            "commit_label": commit_label,
            "subtask_text": subtask_text,
            "memory_text": memory_text,
            "preferred_arm": preferred_arm,
            "note": note,
        }
        self.memory_store.set_monitor_status(phase="reasoning", status="needs_reasoning", note=f"decision ready: {action_mode}")
        return json.dumps({"status": "decided", **self.last_decision}, ensure_ascii=False)
