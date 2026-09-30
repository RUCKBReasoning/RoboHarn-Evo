from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SkillStatus = Literal["pending", "running", "succeeded", "failed", "aborted"]
PlanStatus = Literal["pending", "running", "succeeded", "failed"]
ExecutionPhase = Literal["idle", "reasoning", "tool_call", "rollout", "monitoring", "recovery", "finished"]
ExecutionStatus = Literal[
    "idle",
    "running",
    "needs_reasoning",
    "blocked_on_tools",
    "rollout_active",
    "rollout_succeeded",
    "rollout_failed",
    "rollout_stalled",
    "waiting_retry",
    "waiting_reset",
    "waiting_replan",
    "finished",
]
RecoveryAction = Literal["retry", "reset", "replan", "abort"]


@dataclass
class TaskPlanItem:
    plan_id: str
    skill_name: str
    instruction: str
    status: PlanStatus = "pending"
    note: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "plan_id": self.plan_id,
            "skill_name": self.skill_name,
            "instruction": self.instruction,
            "status": self.status,
            "note": self.note,
        }


@dataclass
class RoleMemory:
    mode: str = "deployment"
    available_tools: list[str] = field(default_factory=list)
    available_policies: list[str] = field(default_factory=list)
    reasoner_backend: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "available_tools": list(self.available_tools),
            "available_policies": list(self.available_policies),
            "reasoner_backend": self.reasoner_backend,
        }


@dataclass
class TaskMemory:
    global_task: str = ""
    plan: list[TaskPlanItem] = field(default_factory=list)
    committed_facts: list[str] = field(default_factory=list)
    committed_memory_text: str = ""
    completed_skills: list[str] = field(default_factory=list)
    failed_skills: list[str] = field(default_factory=list)
    task_finished: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "global_task": self.global_task,
            "plan": [item.to_dict() for item in self.plan],
            "committed_facts": list(self.committed_facts),
            "committed_memory_text": self.committed_memory_text,
            "completed_skills": list(self.completed_skills),
            "failed_skills": list(self.failed_skills),
            "task_finished": self.task_finished,
        }


@dataclass
class WorkingMemory:
    active_skill_id: str | None = None
    active_policy_id: str | None = None
    active_instruction: str = ""
    recent_observation_summary: str = ""
    observation_preprocess: dict[str, object] = field(default_factory=dict)
    scene_memory: dict[str, object] = field(default_factory=dict)
    recent_tool_calls: list[str] = field(default_factory=list)
    retry_count_by_skill: dict[str, int] = field(default_factory=dict)
    stall_count: int = 0
    steps_since_last_decision: int = 0
    last_error: str = ""
    last_decision: str = ""
    last_trigger: str = ""
    last_commit_label: str = ""
    recovery_history: list[str] = field(default_factory=list)
    manipulation_state: dict[str, object] = field(default_factory=dict)
    semantic_tags: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "active_skill_id": self.active_skill_id,
            "active_policy_id": self.active_policy_id,
            "active_instruction": self.active_instruction,
            "recent_observation_summary": self.recent_observation_summary,
            "observation_preprocess": dict(self.observation_preprocess),
            "scene_memory": dict(self.scene_memory),
            "recent_tool_calls": list(self.recent_tool_calls),
            "retry_count_by_skill": dict(self.retry_count_by_skill),
            "stall_count": self.stall_count,
            "steps_since_last_decision": self.steps_since_last_decision,
            "last_error": self.last_error,
            "last_decision": self.last_decision,
            "last_trigger": self.last_trigger,
            "last_commit_label": self.last_commit_label,
            "recovery_history": list(self.recovery_history),
            "manipulation_state": dict(self.manipulation_state),
            "semantic_tags": dict(self.semantic_tags),
        }


@dataclass
class MonitorSnapshot:
    phase: ExecutionPhase = "idle"
    status: ExecutionStatus = "idle"
    rollout_id: str = ""
    active_skill_name: str = ""
    steps_used: int = 0
    stall_count: int = 0
    progress_score: float = 0.0
    env_signal: str = ""
    failure_reason: str = ""
    last_status_note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "status": self.status,
            "rollout_id": self.rollout_id,
            "active_skill_name": self.active_skill_name,
            "steps_used": self.steps_used,
            "stall_count": self.stall_count,
            "progress_score": self.progress_score,
            "env_signal": self.env_signal,
            "failure_reason": self.failure_reason,
            "last_status_note": self.last_status_note,
        }


@dataclass
class RecoveryPolicyState:
    retry_budget: int = 0
    reset_budget: int = 0
    replan_budget: int = 0
    retry_used: int = 0
    reset_used: int = 0
    replan_used: int = 0
    pending_action: RecoveryAction | str = ""
    pending_reason: str = ""
    last_action: RecoveryAction | str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "retry_budget": self.retry_budget,
            "reset_budget": self.reset_budget,
            "replan_budget": self.replan_budget,
            "retry_used": self.retry_used,
            "reset_used": self.reset_used,
            "replan_used": self.replan_used,
            "pending_action": self.pending_action,
            "pending_reason": self.pending_reason,
            "last_action": self.last_action,
        }


@dataclass
class SkillRunState:
    skill_id: str
    skill_name: str
    instruction: str
    policy_binding: str
    status: SkillStatus = "running"
    steps_used: int = 0
    retries_used: int = 0
    max_steps: int = 80
    max_retries: int = 2
    preferred_arm: str = "either"
    semantic_tags: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "instruction": self.instruction,
            "policy_binding": self.policy_binding,
            "status": self.status,
            "steps_used": self.steps_used,
            "retries_used": self.retries_used,
            "max_steps": self.max_steps,
            "max_retries": self.max_retries,
            "preferred_arm": self.preferred_arm,
            "semantic_tags": dict(self.semantic_tags),
        }


@dataclass
class AgentState:
    role: RoleMemory = field(default_factory=RoleMemory)
    task: TaskMemory = field(default_factory=TaskMemory)
    working: WorkingMemory = field(default_factory=WorkingMemory)
    active_skill: SkillRunState | None = None
    monitor: MonitorSnapshot = field(default_factory=MonitorSnapshot)
    recovery: RecoveryPolicyState = field(default_factory=RecoveryPolicyState)
    decision_count: int = 0
    recovery_attempts: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role.to_dict(),
            "task": self.task.to_dict(),
            "working": self.working.to_dict(),
            "active_skill": None if self.active_skill is None else self.active_skill.to_dict(),
            "monitor": self.monitor.to_dict(),
            "recovery": self.recovery.to_dict(),
            "decision_count": self.decision_count,
            "recovery_attempts": self.recovery_attempts,
        }
