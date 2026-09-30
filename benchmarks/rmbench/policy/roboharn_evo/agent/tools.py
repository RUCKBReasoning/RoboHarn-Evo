"""Tool definitions and executor for the agentic planner.

Each tool is an OpenAI function-calling schema + a handler method.
The ToolExecutor dispatches tool_calls from the LLM to the correct handler.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from .memory import MemoryManager


# ---------------------------------------------------------------------------
#  OpenAI function-calling tool schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "Read the full structured memory of the current task state.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": (
                "Update one or more fields of the structured memory. "
                "Allowed fields: phase (str), current_state (str), memory_text (str), "
                "observations (list[str] to append), plan (list[str] to replace)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phase": {"type": "string", "description": "High-level phase description."},
                    "current_state": {"type": "string", "description": "Description of the current scene state."},
                    "memory_text": {
                        "type": "string",
                        "description": "Committed executor-facing memory summary for this turn. Must not contain future plan.",
                    },
                    "observations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New observations to append (e.g. object colors, positions).",
                    },
                    "plan": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered list of upcoming subtask names (replaces existing plan).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_subtask_done",
            "description": "Mark a subtask as completed and move it from plan to completed list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the subtask to mark as done."},
                    "result": {"type": "string", "description": "Brief description of the outcome."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit",
            "description": (
                "Finalize this planning step. Returns the commit decision, updated memory text, "
                "and the next subtask for the executor VLA. This MUST be called exactly once to end the planning turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "commit_label": {
                        "type": "string",
                        "enum": ["no_update", "subtask_complete", "state_change"],
                        "description": "Whether and why the memory was updated.",
                    },
                    "subtask_text": {
                        "type": "string",
                        "description": "The next subtask instruction for the executor VLA.",
                    },
                },
                "required": ["commit_label", "subtask_text"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
#  ToolExecutor
# ---------------------------------------------------------------------------

def _split_service_tool_name(tool_name: str) -> tuple[str, str] | None:
    if "___" not in tool_name:
        return None
    service_name, local_tool_name = tool_name.split("___", 1)
    if not service_name or not local_tool_name:
        return None
    return service_name, local_tool_name


def _run_coroutine_sync(coro) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return str(asyncio.run(coro))
    raise RuntimeError("Cannot execute async service tools while an event loop is already running")


class ToolExecutor:
    """Dispatches OpenAI tool_call objects to handler methods."""

    def __init__(self, memory_manager: MemoryManager, service_tool_router: Callable[[str, str, dict[str, Any]], Any] | None = None) -> None:
        self.mm = memory_manager
        self.service_tool_router = service_tool_router
        self._handlers: dict[str, Callable[..., str]] = {
            "read_memory": self._handle_read_memory,
            "update_memory": self._handle_update_memory,
            "mark_subtask_done": self._handle_mark_subtask_done,
            "commit": self._handle_commit,
        }
        self.last_commit: dict[str, Any] | None = None

    def reset(self) -> None:
        self.last_commit = None

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        handler = self._handlers.get(tool_name)
        if handler is None and self.service_tool_router is not None:
            split_name = _split_service_tool_name(tool_name)
            if split_name is not None:
                service_name, local_tool_name = split_name
                try:
                    result = self.service_tool_router(service_name, local_tool_name, arguments)
                    if asyncio.iscoroutine(result):
                        result = _run_coroutine_sync(result)
                    return str(result)
                except Exception as exc:
                    return json.dumps({"error": str(exc)}, ensure_ascii=False)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)
        try:
            return handler(**arguments)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    # ---- handlers -----------------------------------------------------------

    def _handle_read_memory(self) -> str:
        return json.dumps(self.mm.read_full(), ensure_ascii=False)

    def _handle_update_memory(
        self,
        phase: str | None = None,
        current_state: str | None = None,
        memory_text: str | None = None,
        observations: list[str] | None = None,
        plan: list[str] | None = None,
    ) -> str:
        results: list[str] = []
        if phase is not None:
            results.append(self.mm.update_phase(phase))
        if current_state is not None:
            results.append(self.mm.update_current_state(current_state))
        if memory_text is not None:
            results.append(self.mm.update_memory_text(memory_text))
        if observations:
            for obs in observations:
                results.append(self.mm.add_observation(obs))
        if plan is not None:
            results.append(self.mm.set_plan(plan))
        return " ".join(results) if results else "No fields updated."

    def _handle_mark_subtask_done(self, name: str, result: str = "") -> str:
        return self.mm.mark_subtask_done(name, result)

    def _handle_commit(self, commit_label: str, subtask_text: str) -> str:
        allowed = {"no_update", "subtask_complete", "state_change"}
        if commit_label not in allowed:
            return json.dumps({"error": f"Invalid commit_label '{commit_label}'. Must be one of {allowed}."})
        if self.last_commit is not None:
            return json.dumps({"error": "commit has already been called in this planning turn."})
        self.last_commit = {
            "commit_label": commit_label,
            "memory_text": self.mm.read_text(),
            "subtask_text": subtask_text,
        }
        return json.dumps({"status": "committed", **self.last_commit}, ensure_ascii=False)
