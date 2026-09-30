"""Structured memory manager for the agentic planner.

The authoritative planner input is ``previous_memory_text`` from the caller.
Therefore the structured ledger must be re-seeded from that text every turn and
must not leak future plan items back into the executor-facing ``memory_text``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubtaskRecord:
    """One completed or in-progress subtask."""
    name: str
    status: str  # "pending" | "in_progress" | "done" | "failed"
    result: str = ""  # brief outcome description

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "result": self.result}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> SubtaskRecord:
        return cls(name=d["name"], status=d["status"], result=d.get("result", ""))


@dataclass
class StructuredMemory:
    """The full task-state ledger maintained by the agent.

    Fields:
        phase             – high-level phase description (e.g. "observing", "executing step 2")
        completed         – ordered list of completed subtask records
        current_state     – free-form description of what the scene looks like now
        memory_text       – executor-facing committed state summary for this turn
        observations      – noteworthy facts the agent has logged (colors, positions, etc.)
        plan              – ordered list of upcoming subtask names (agent's working plan)
    """
    phase: str = ""
    completed: list[SubtaskRecord] = field(default_factory=list)
    current_state: str = ""
    memory_text: str = ""
    observations: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)

    # ---- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "completed": [s.to_dict() for s in self.completed],
            "current_state": self.current_state,
            "memory_text": self.memory_text,
            "observations": list(self.observations),
            "plan": list(self.plan),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StructuredMemory:
        return cls(
            phase=d.get("phase", ""),
            completed=[SubtaskRecord.from_dict(s) for s in d.get("completed", [])],
            current_state=d.get("current_state", ""),
            memory_text=d.get("memory_text", ""),
            observations=list(d.get("observations", [])),
            plan=list(d.get("plan", [])),
        )

    def to_text(self, initial_text: str) -> str:
        """Return the executor-facing committed summary only.

        This deliberately excludes future plan items and other scratch fields.
        """
        if self.memory_text:
            return self.memory_text
        if self.current_state:
            return self.current_state
        return initial_text

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class MemoryManager:
    """Provides tool-callable methods for the agentic planner to manipulate memory."""

    def __init__(self, initial_text: str = "The task has started.") -> None:
        self.initial_text = initial_text
        self.memory = StructuredMemory(memory_text=initial_text)

    def reset(self, seed_text: str | None = None) -> None:
        self.memory = StructuredMemory(memory_text=seed_text or self.initial_text)

    def seed_from_previous_memory(self, previous_memory_text: str) -> None:
        self.reset(seed_text=previous_memory_text)

    # ---- read ----------------------------------------------------------------

    def read_full(self) -> dict[str, Any]:
        return self.memory.to_dict()

    def read_text(self) -> str:
        return self.memory.to_text(self.initial_text)

    # ---- write helpers -------------------------------------------------------

    def update_phase(self, phase: str) -> str:
        self.memory.phase = phase
        return f"Phase updated to: {phase}"

    def update_current_state(self, state: str) -> str:
        self.memory.current_state = state
        # Default the committed summary to the latest state description unless the agent overwrites it explicitly.
        self.memory.memory_text = state
        return f"Current state updated."

    def update_memory_text(self, memory_text: str) -> str:
        self.memory.memory_text = memory_text
        return "Committed memory text updated."

    def add_observation(self, observation: str) -> str:
        self.memory.observations.append(observation)
        return f"Observation logged ({len(self.memory.observations)} total)."

    def set_plan(self, plan: list[str]) -> str:
        self.memory.plan = list(plan)
        return f"Plan set with {len(plan)} steps."

    def peek_next_subtask(self) -> str:
        if self.memory.plan:
            return self.memory.plan[0]
        return ""

    def mark_subtask_done(self, name: str, result: str = "") -> str:
        # Move from plan to completed if present
        if name in self.memory.plan:
            self.memory.plan.remove(name)
        self.memory.completed.append(SubtaskRecord(name=name, status="done", result=result))
        return f"Subtask '{name}' marked done."

    def mark_subtask_failed(self, name: str, result: str = "") -> str:
        if name in self.memory.plan:
            self.memory.plan.remove(name)
        self.memory.completed.append(SubtaskRecord(name=name, status="failed", result=result))
        return f"Subtask '{name}' marked failed."
