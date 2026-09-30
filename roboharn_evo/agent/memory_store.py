from __future__ import annotations

from typing import Any

from roboharn_evo.agent.state import AgentState, RoleMemory, SkillRunState, TaskMemory, TaskPlanItem, WorkingMemory
from roboharn_evo.agent.skills.base import SkillSpec


class AgentMemoryStore:
    def __init__(self, *, initial_memory_text: str) -> None:
        self.initial_memory_text = initial_memory_text
        self.state = AgentState()
        self._plan_counter = 0

    def reset(self, *, task: str, reasoner_backend: str, available_policies: list[str], available_tools: list[str], mode: str) -> None:
        self._plan_counter = 0
        self.state = AgentState(
            role=RoleMemory(
                mode=mode,
                available_tools=list(available_tools),
                available_policies=list(available_policies),
                reasoner_backend=reasoner_backend,
            ),
            task=TaskMemory(
                global_task=task,
                committed_memory_text=self.initial_memory_text,
            ),
            working=WorkingMemory(),
        )

    def export_executor_memory(self) -> str:
        text = self.state.task.committed_memory_text.strip()
        return text if text else self.initial_memory_text

    def export_reasoner_memory(self) -> str:
        return self.export_executor_memory()

    def set_committed_memory(self, memory_text: str) -> None:
        text = memory_text.strip()
        if text:
            self.state.task.committed_memory_text = text
            if not self.state.task.committed_facts or self.state.task.committed_facts[-1] != text:
                self.state.task.committed_facts.append(text)

    def record_observation_summary(self, summary: str) -> None:
        self.state.working.recent_observation_summary = summary

    def record_observation_preprocess(self, payload: dict[str, Any]) -> None:
        self.state.working.observation_preprocess = dict(payload)

    def record_scene_memory(self, payload: dict[str, Any]) -> None:
        self.state.working.scene_memory = dict(payload)

    def record_tool_call(self, message: str) -> None:
        self.state.working.recent_tool_calls.append(message)
        if len(self.state.working.recent_tool_calls) > 20:
            self.state.working.recent_tool_calls = self.state.working.recent_tool_calls[-20:]

    def record_decision(self, *, trigger: str, note: str, commit_label: str = "") -> None:
        self.state.working.last_trigger = trigger
        self.state.working.last_decision = note
        self.state.working.last_commit_label = commit_label
        self.state.working.steps_since_last_decision = 0
        self.state.decision_count += 1

    def increment_step_counters(self) -> None:
        self.state.working.steps_since_last_decision += 1
        if self.state.active_skill is not None:
            self.state.active_skill.steps_used += 1

    def reset_stall_count(self) -> None:
        self.state.working.stall_count = 0

    def increment_stall_count(self) -> None:
        self.state.working.stall_count += 1

    def set_last_error(self, error: str) -> None:
        self.state.working.last_error = error

    def record_recovery(self, message: str) -> None:
        self.state.working.recovery_history.append(message)
        if len(self.state.working.recovery_history) > 20:
            self.state.working.recovery_history = self.state.working.recovery_history[-20:]

    def set_manipulation_arm_state(self, arm: str, payload: dict[str, Any]) -> None:
        normalized_arm = str(arm or "").strip().lower()
        if normalized_arm not in {"left", "right"}:
            raise ValueError(f"manipulation arm must be left or right, got {arm!r}")
        state = dict(self.state.working.manipulation_state)
        state[normalized_arm] = dict(payload)
        self.state.working.manipulation_state = state

    def clear_manipulation_arm_state(self, arm: str) -> None:
        normalized_arm = str(arm or "").strip().lower()
        if normalized_arm not in {"left", "right"}:
            return
        state = dict(self.state.working.manipulation_state)
        state.pop(normalized_arm, None)
        self.state.working.manipulation_state = state

    def _next_plan_id(self) -> str:
        self._plan_counter += 1
        return f"skill_{self._plan_counter}"

    def start_or_replace_skill(
        self,
        *,
        subtask_text: str,
        skill_spec: SkillSpec,
        force_new_attempt: bool = False,
    ) -> SkillRunState:
        rendered_instruction = skill_spec.render_instruction(subtask_text)
        current = self.state.active_skill
        same_skill_binding = bool(
            current is not None
            and current.instruction == rendered_instruction
            and current.skill_name == skill_spec.name
            and current.policy_binding == skill_spec.policy_binding
        )
        if same_skill_binding and not force_new_attempt:
            current.status = "running"
            self.state.working.active_skill_id = current.skill_id
            self.state.working.active_policy_id = current.policy_binding
            self.state.working.active_instruction = current.instruction
            return current

        if current is not None:
            for item in self.state.task.plan:
                if item.plan_id == current.skill_id and item.status == "running":
                    item.status = "failed"
                    item.note = "superseded by new skill"
                    break

        # Keep recovery/verifier evidence scoped to the skill attempt that produced it.  A retry
        # of the same active instruction returns above and keeps the evidence; a newly allocated
        # skill starts with a clean recovery scope.
        self.state.working.recovery_history = []

        plan_id = self._next_plan_id()
        run_state = SkillRunState(
            skill_id=plan_id,
            skill_name=skill_spec.name,
            instruction=rendered_instruction,
            policy_binding=skill_spec.policy_binding,
            max_steps=skill_spec.max_steps,
            max_retries=skill_spec.max_retries,
        )
        self.state.active_skill = run_state
        self.state.task.plan.append(
            TaskPlanItem(
                plan_id=plan_id,
                skill_name=skill_spec.name,
                instruction=run_state.instruction,
                status="running",
            )
        )
        self.state.working.active_skill_id = run_state.skill_id
        self.state.working.active_policy_id = run_state.policy_binding
        self.state.working.active_instruction = run_state.instruction
        return run_state

    def mark_active_skill_succeeded(self, note: str = "") -> None:
        active = self.state.active_skill
        if active is None:
            return
        active.status = "succeeded"
        for item in self.state.task.plan:
            if item.plan_id == active.skill_id:
                item.status = "succeeded"
                item.note = note
                break
        self.state.task.completed_skills.append(active.instruction)
        self.clear_active_skill()

    def mark_active_skill_failed(self, note: str = "") -> None:
        active = self.state.active_skill
        if active is None:
            return
        active.status = "failed"
        for item in self.state.task.plan:
            if item.plan_id == active.skill_id:
                item.status = "failed"
                item.note = note
                break
        self.state.task.failed_skills.append(active.instruction)
        self.clear_active_skill()

    def retry_active_skill(self) -> None:
        active = self.state.active_skill
        if active is None:
            return
        active.retries_used += 1
        self.state.working.retry_count_by_skill[active.skill_id] = active.retries_used
        active.steps_used = 0
        self.state.working.stall_count = 0

    def clear_active_skill(self) -> None:
        self.state.active_skill = None
        self.state.working.active_skill_id = None
        self.state.working.active_policy_id = None
        self.state.working.active_instruction = ""
        self.state.working.stall_count = 0

    def mark_task_finished(self, note: str = "") -> None:
        self.state.task.task_finished = True
        if note:
            self.state.task.committed_facts.append(note)

    def to_reasoner_context(self) -> dict[str, Any]:
        return self.state.to_dict()
