from __future__ import annotations

from typing import Any

from roboharn_evo.agent.arm_contract import normalize_preferred_arm
from roboharn_evo.agent.components.memory_manager.runtime_memory_tree.runtime_memory import ChatContext, RuntimeMemoryTree
from roboharn_evo.agent.experience import empty_semantic_tags, merge_semantic_tags, normalize_semantic_tags
from roboharn_evo.agent.state import AgentState, RoleMemory, SkillRunState, TaskMemory, TaskPlanItem, WorkingMemory


DEFAULT_MEMORY_PROMPTS: dict[str, str] = {
    "SELF_KNOWLEDGE_TEMPLATE": "You are the top-level Agentic Vision-Language-Action controller.\nMemory tree:\n{memory_tree}\n{self_knowledge_extension}",
    "SELF_KNOWLEDGE_EXTENSION": "",
    "KNOWLEDGE_GRAPH_CACHING_TEMPLATE": "{prefix} {name_cn} Updated at [{updated_at}]\nThis block is currently unused.",
    "SERVER_REGISTRY_TEMPLATE": "{prefix} {name_cn} Updated at [{updated_at}]\nAvailable internal services/tools:\n{services_list}",
    "TASK_TEMPLATE_TEMPLATE": "{prefix} {name_cn} Updated at [{updated_at}]\nTask templates are currently lightweight.",
    "TASK_SESSION_TEMPLATE": "{prefix} {name_cn} Updated at [{updated_at}]",
    "TASK_NODE_START_TEMPLATE": "{prefix} {name_cn} Updated at [{updated_at}]\nTask ID: {task_id}\nTask Brief:\n{task_brief}\nAction Guidance:\n{assistant_guidance}",
    "DEFAULT_TASK_BRIEF_TEMPLATE": "Help execute or supervise the user's current robot task.",
    "DEFAULT_ASSISTANT_GUIDANCE_TEMPLATE": "Prefer skill-based execution and monitored control boundaries.",
    "DEFAULT_COMPRESS_REQUEST_TEMPLATE": "Please summarize and compress the current task contexts.",
    "IMG_STR": "Robot visual context frame {frame_id}.",
    "USER_STR": "User input: {user_msg}",
}


class MemoryManager:
    def __init__(self, *, initial_memory_text: str, memory_prompts: dict[str, str] | None = None, img_threshold: int = 6) -> None:
        self.initial_memory_text = initial_memory_text
        self.memory_prompts = dict(DEFAULT_MEMORY_PROMPTS)
        if memory_prompts:
            self.memory_prompts.update(memory_prompts)
        self.runtime_tree = RuntimeMemoryTree(
            initial_memory_text=initial_memory_text,
            memory_prompts=self.memory_prompts,
            img_threshold=img_threshold,
        )
        self.state = AgentState()
        self._plan_counter = 0
        self._rollout_counter = 0

    def reset(self, *, task: str, control_model_name: str, available_policies: list[str], available_tools: list[str], mode: str, retry_budget: int = 2, reset_budget: int = 1, replan_budget: int = 2) -> None:
        self._plan_counter = 0
        self._rollout_counter = 0
        self.state = AgentState(
            role=RoleMemory(
                mode=mode,
                available_tools=list(available_tools),
                available_policies=list(available_policies),
                reasoner_backend=control_model_name,
            ),
            task=TaskMemory(
                global_task=task,
                committed_memory_text=self.initial_memory_text,
            ),
            working=WorkingMemory(),
        )
        self.state.recovery.retry_budget = max(0, retry_budget)
        self.state.recovery.reset_budget = max(0, reset_budget)
        self.state.recovery.replan_budget = max(0, replan_budget)
        self.runtime_tree.init_memory_tree(
            control_model_name=control_model_name,
            available_tools=available_tools,
            mode=mode,
            task_brief=task,
            assistant_guidance=self.memory_prompts.get("DEFAULT_ASSISTANT_GUIDANCE_TEMPLATE", ""),
        )
        self.set_monitor_status(phase="idle", status="needs_reasoning", note="runtime reset")

    @property
    def current_contexts(self) -> list[dict[str, Any]]:
        return self.runtime_tree.current_contexts()

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

    def add_user_message(self, message: str) -> None:
        if self.runtime_tree.current_task_node is None:
            return
        formatted = self.memory_prompts.get("USER_STR", "{user_msg}").format(user_msg=message)
        self.runtime_tree.current_task_node.contexts.append(ChatContext(role="user", content=formatted))

    def add_agent_message(self, message: str, tool_call_ids: list[str] | None = None) -> None:
        if self.runtime_tree.current_task_node is None:
            return
        self.runtime_tree.current_task_node.contexts.append(
            ChatContext(role="assistant", content=message, meta={"tool_call_ids": list(tool_call_ids or [])})
        )

    def add_tool_message(self, message: str, tool_call_id: str) -> None:
        if self.runtime_tree.current_task_node is None:
            return
        self.runtime_tree.current_task_node.contexts.append(
            ChatContext(role="tool", content=message, meta={"tool_call_id": tool_call_id})
        )

    def add_robot_image(self, *, image_data_url: str, frame_id: int) -> None:
        if self.runtime_tree.current_task_node is None:
            return
        caption = self.memory_prompts.get("IMG_STR", "Robot visual context frame {frame_id}.").format(frame_id=frame_id)
        self.runtime_tree.current_task_node.contexts.append(
            ChatContext(role="user_image", content=image_data_url, meta={"frame_id": frame_id, "caption": caption})
        )

    def compress_current_memory(self, drop_n: int = 2) -> None:
        if self.runtime_tree.current_task_node is None:
            return
        self.runtime_tree.current_task_node.compress_policy_discard_oldest(drop_n=drop_n)

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
            self.state.monitor.steps_used = self.state.active_skill.steps_used

    def reset_stall_count(self) -> None:
        self.state.working.stall_count = 0
        self.state.monitor.stall_count = 0

    def increment_stall_count(self) -> None:
        self.state.working.stall_count += 1
        self.state.monitor.stall_count = self.state.working.stall_count

    def set_last_error(self, error: str) -> None:
        self.state.working.last_error = error
        self.state.monitor.failure_reason = error

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

    def update_runtime_semantic_tags(self, tags: dict[str, Any] | None, *, overwrite: bool = True) -> dict[str, Any]:
        normalized = normalize_semantic_tags(tags)
        self.state.working.semantic_tags = merge_semantic_tags(
            self.state.working.semantic_tags,
            normalized,
            overwrite=overwrite,
        )
        return dict(self.state.working.semantic_tags)

    def update_active_skill_semantic_tags(self, tags: dict[str, Any] | None, *, overwrite: bool = True) -> dict[str, Any]:
        normalized = normalize_semantic_tags(tags)
        if self.state.active_skill is None:
            return self.update_runtime_semantic_tags(normalized, overwrite=overwrite)
        self.state.active_skill.semantic_tags = merge_semantic_tags(
            self.state.active_skill.semantic_tags,
            normalized,
            overwrite=overwrite,
        )
        self.state.working.semantic_tags = merge_semantic_tags(
            self.state.working.semantic_tags,
            self.state.active_skill.semantic_tags,
            overwrite=overwrite,
        )
        return dict(self.state.active_skill.semantic_tags)

    def _next_plan_id(self) -> str:
        self._plan_counter += 1
        return f"skill_{self._plan_counter}"

    def _next_rollout_id(self) -> str:
        self._rollout_counter += 1
        return f"rollout_{self._rollout_counter}"

    def set_monitor_status(self, *, phase: str, status: str, note: str = "", env_signal: str | None = None, failure_reason: str | None = None, progress_score: float | None = None) -> None:
        self.state.monitor.phase = phase
        self.state.monitor.status = status
        self.state.monitor.last_status_note = note
        if env_signal is not None:
            self.state.monitor.env_signal = env_signal
        if failure_reason is not None:
            self.state.monitor.failure_reason = failure_reason
        if progress_score is not None:
            self.state.monitor.progress_score = progress_score
        if self.state.active_skill is not None:
            self.state.monitor.active_skill_name = self.state.active_skill.skill_name
            self.state.monitor.steps_used = self.state.active_skill.steps_used
        else:
            self.state.monitor.active_skill_name = ""
            self.state.monitor.steps_used = 0

    def start_rollout(self) -> str:
        rollout_id = self._next_rollout_id()
        self.state.monitor.rollout_id = rollout_id
        self.set_monitor_status(phase="rollout", status="rollout_active", note="rollout started")
        return rollout_id

    def complete_rollout(self, *, env_signal: str, note: str, progress_score: float = 0.0) -> None:
        self.set_monitor_status(phase="monitoring", status="rollout_succeeded", note=note, env_signal=env_signal, progress_score=progress_score, failure_reason="")

    def fail_rollout(self, *, status: str, reason: str, env_signal: str = "", progress_score: float = 0.0) -> None:
        self.set_monitor_status(phase="monitoring", status=status, note=reason, env_signal=env_signal, failure_reason=reason, progress_score=progress_score)

    def set_recovery_policy(self, *, action: str, reason: str) -> None:
        self.state.recovery.pending_action = action
        self.state.recovery.pending_reason = reason
        self.set_monitor_status(phase="recovery", status=f"waiting_{action}" if action in {"retry", "reset", "replan"} else "rollout_failed", note=reason, failure_reason=reason)

    def consume_recovery_policy(self) -> tuple[str, str]:
        action = str(self.state.recovery.pending_action)
        reason = str(self.state.recovery.pending_reason)
        self.state.recovery.pending_action = ""
        self.state.recovery.pending_reason = ""
        self.state.recovery.last_action = action
        if action == "retry":
            self.state.recovery.retry_used += 1
        elif action == "reset":
            self.state.recovery.reset_used += 1
        elif action == "replan":
            self.state.recovery.replan_used += 1
        return action, reason

    def apply_skill_recovery_policy(self, skill_spec: Any) -> None:
        policy = getattr(skill_spec, "recovery_policy", None)
        if policy is None:
            self.state.recovery.retry_budget = 0
            self.state.recovery.reset_budget = 0
            self.state.recovery.replan_budget = 1
            return
        self.state.recovery.retry_budget = max(0, int(getattr(policy, "retry_budget", 0)))
        self.state.recovery.reset_budget = max(0, int(getattr(policy, "reset_budget", 0)))
        self.state.recovery.replan_budget = max(0, int(getattr(policy, "replan_budget", 1)))
        self.state.recovery.retry_used = 0
        self.state.recovery.reset_used = 0
        self.state.recovery.replan_used = 0
        self.state.recovery.pending_action = ""
        self.state.recovery.pending_reason = ""

    def start_or_replace_skill(
        self,
        *,
        subtask_text: str,
        skill_spec: Any,
        semantic_tags: dict[str, Any] | None = None,
        preferred_arm: str = "either",
        force_new_attempt: bool = False,
    ) -> SkillRunState:
        rendered_instruction = skill_spec.render_instruction(subtask_text)
        normalized_tags = normalize_semantic_tags(semantic_tags) if semantic_tags else empty_semantic_tags()
        normalized_preferred_arm = normalize_preferred_arm(preferred_arm)
        current = self.state.active_skill
        same_skill_binding = bool(
            current is not None
            and current.instruction == rendered_instruction
            and current.skill_name == skill_spec.name
            and current.policy_binding == skill_spec.policy_binding
        )
        if same_skill_binding and not force_new_attempt:
            current.status = "running"
            current.preferred_arm = normalized_preferred_arm
            if semantic_tags:
                current.semantic_tags = merge_semantic_tags(current.semantic_tags, normalized_tags)
            self.state.working.active_skill_id = current.skill_id
            self.state.working.active_policy_id = current.policy_binding
            self.state.working.active_instruction = current.instruction
            self.state.working.semantic_tags = merge_semantic_tags(self.state.working.semantic_tags, current.semantic_tags)
            self.set_monitor_status(phase="rollout", status="running", note="continuing active skill")
            return current

        if current is not None:
            for item in self.state.task.plan:
                if item.plan_id == current.skill_id and item.status == "running":
                    item.status = "failed"
                    item.note = "superseded by new skill"
                    break

        # Recovery evidence is attempt-local.  In particular, an after-action verifier result
        # for a completed skill must never be offered to the recovery router as evidence about
        # the next skill merely because both instructions describe a similar physical action.
        # Retries of the same active skill return above and intentionally retain their history.
        self.state.working.recovery_history = []

        plan_id = self._next_plan_id()
        run_state = SkillRunState(
            skill_id=plan_id,
            skill_name=skill_spec.name,
            instruction=rendered_instruction,
            policy_binding=skill_spec.policy_binding,
            max_steps=skill_spec.max_steps,
            max_retries=skill_spec.max_retries,
            preferred_arm=normalized_preferred_arm,
            semantic_tags=normalized_tags,
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
        self.apply_skill_recovery_policy(skill_spec)
        self.state.working.active_skill_id = run_state.skill_id
        self.state.working.active_policy_id = run_state.policy_binding
        self.state.working.active_instruction = run_state.instruction
        self.state.working.semantic_tags = dict(run_state.semantic_tags)
        self.set_monitor_status(phase="rollout", status="running", note="skill activated")
        return run_state

    def update_active_skill_preferred_arm(self, preferred_arm: str) -> str:
        normalized = normalize_preferred_arm(preferred_arm)
        if self.state.active_skill is not None:
            self.state.active_skill.preferred_arm = normalized
        return normalized

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
        self.set_monitor_status(phase="monitoring", status="rollout_succeeded", note=note)

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
        self.set_monitor_status(phase="monitoring", status="rollout_failed", note=note, failure_reason=note)

    def retry_active_skill(self) -> None:
        active = self.state.active_skill
        if active is None:
            return
        active.retries_used += 1
        self.state.working.retry_count_by_skill[active.skill_id] = active.retries_used
        active.steps_used = 0
        self.state.working.stall_count = 0
        self.state.monitor.steps_used = 0
        self.state.monitor.stall_count = 0
        self.set_monitor_status(phase="rollout", status="running", note="retry active skill", failure_reason="")

    def clear_active_skill(self) -> None:
        self.state.active_skill = None
        self.state.working.active_skill_id = None
        self.state.working.active_policy_id = None
        self.state.working.active_instruction = ""
        self.state.working.semantic_tags = {}
        self.state.working.stall_count = 0
        self.state.monitor.active_skill_name = ""
        self.state.monitor.steps_used = 0
        self.state.monitor.stall_count = 0

    def mark_task_finished(self, note: str = "") -> None:
        self.state.task.task_finished = True
        if note:
            self.state.task.committed_facts.append(note)
        self.set_monitor_status(phase="finished", status="finished", note=note or "task finished", env_signal="task_success")

    def reopen_task(self, note: str = "") -> None:
        self.state.task.task_finished = False
        self.set_monitor_status(
            phase="reasoning",
            status="needs_reasoning",
            note=note or "task finish was not validated",
            env_signal="running",
            failure_reason=note,
            progress_score=0.0,
        )

    def to_reasoner_context(self) -> dict[str, Any]:
        return self.state.to_dict()
