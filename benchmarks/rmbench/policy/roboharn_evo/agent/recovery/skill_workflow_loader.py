from __future__ import annotations

from typing import Any, Callable

from ..arm_contract import normalize_physical_arm, normalize_preferred_arm, normalize_selected_arm
from ..components.agent_tools.local_skill_registry import LocalSkill, LocalSkillRegistry
from ..experience import ExperienceQuery, ExperienceRetriever, normalize_semantic_tags
from ...models.backend_factory import build_recovery_backend
from ...models.backend_interfaces import RecoveryBackend

from .recovery_policies import RecoveryRoute
from .recovery_primitives import RecoveryPrimitivePlan
from .tool_specs import RECOVERY_TOOLS, RecoveryToolCall

_ALLOWED_RECOVERY_INTENTS = {"retry", "replan", "abort"}
_ARM_BOUND_PHYSICAL_TOOLS = RECOVERY_TOOLS - {"reobserve_scene"}
_ARM_BOUND_TASK_PROGRESS_TOOLS = {
    "close_gripper",
    "contact_displace",
    "lift_ee",
    "move_ee_to_grounded_instance",
    "move_ee_to_pose",
    "open_gripper",
}


class SkillRecoveryWorkflowLoader:
    """Build recovery routes from dynamic recovery-planner output.

    Recovery SKILL.md files are planning harnesses. This class loads router,
    workflow, and primitive SKILL bodies, sends them to a recovery backend, then
    validates the returned structured tool calls before dispatch.
    """

    def __init__(
        self,
        skill_registry: LocalSkillRegistry,
        recovery_backend: RecoveryBackend | None = None,
        backend_config: dict[str, Any] | None = None,
        experience_retriever: ExperienceRetriever | None = None,
        request_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._skill_registry = skill_registry
        self._recovery_backend = recovery_backend or build_recovery_backend(backend_config or {})
        self._experience_retriever = experience_retriever or ExperienceRetriever()
        self._request_observer = request_observer
        self.last_error = ""
        self.last_error_kind = ""

    def resolve(
        self,
        *,
        signal_name: str,
        reason: str,
        ood_scenario: str = "",
        global_task: str = "",
        current_subtask: str = "",
        observation_summary: str = "",
        robot_state: dict[str, Any] | None = None,
        recovery_state: dict[str, Any] | None = None,
        recovery_history: list[str] | None = None,
        available_tools: list[str] | set[str] | None = None,
        semantic_tags: dict[str, Any] | None = None,
        scene_memory: dict[str, Any] | None = None,
        observation_preprocess: dict[str, Any] | None = None,
        preferred_arm: str = "either",
        blocked_grounded_setups: list[dict[str, Any]] | None = None,
    ) -> RecoveryRoute | None:
        self.last_error = ""
        self.last_error_kind = ""
        router = self._skill_registry.get_skill("recovery-router", refresh=True)
        if router is None:
            self._record_error("missing recovery-router skill")
            return None
        workflow_skills = self._collect_skills("/recovery-workflows/")
        primitive_skills = self._collect_skills("/recovery-primitives/")
        workflow_payloads, primitive_payloads, signal_workflow_candidates = self._planning_skill_payloads(
            workflow_skills=workflow_skills,
            primitive_skills=primitive_skills,
            signal_name=str(signal_name or ""),
            ood_scenario=str(ood_scenario or signal_name or ""),
        )
        effective_tools = self._effective_available_tools(available_tools)
        validated_semantic_tags = normalize_semantic_tags(semantic_tags)
        retrieved_experience = self._retrieve_experience(
            ood_scenario=str(ood_scenario or signal_name),
            current_subtask=current_subtask,
            available_tools=sorted(effective_tools),
            semantic_tags=validated_semantic_tags,
        )
        payload = {
            "signal_name": str(signal_name).strip(),
            "OOD_scenario": str(ood_scenario or signal_name).strip(),
            "reason": reason,
            "global_task": str(global_task or "").strip(),
            "current_subtask": current_subtask,
            "observation_summary": observation_summary,
            "robot_state": dict(robot_state or {}),
            "scene_memory": dict(scene_memory or {}),
            "observation_preprocess": dict(observation_preprocess or {}),
            "preferred_arm": normalize_preferred_arm(preferred_arm),
            "blocked_grounded_setups": [
                dict(item) for item in (blocked_grounded_setups or []) if isinstance(item, dict)
            ],
            "recovery_state": dict(recovery_state or {}),
            "recovery_history": list(recovery_history or []),
            "available_tools": sorted(effective_tools),
            "semantic_tags": validated_semantic_tags,
            "retrieved_experience": retrieved_experience,
            "router_skill": self._skill_payload(router),
            "workflow_skills": workflow_payloads,
            "primitive_skills": primitive_payloads,
            "signal_workflow_candidates": signal_workflow_candidates,
            "skill_payload_mode": "signal_scoped" if signal_workflow_candidates else "full",
        }
        self._notify_request_observer(payload)
        try:
            raw_plan = self._recovery_backend.plan_recovery(recovery_payload=payload)
        except Exception as exc:
            self._record_error(
                f"recovery planner failed: {type(exc).__name__}: {exc}",
                kind="backend",
            )
            return None
        return self._build_route_from_plan(
            signal_name=str(signal_name).strip(),
            reason=reason,
            raw_plan=raw_plan,
            workflow_names={skill.name for skill in workflow_skills},
            available_tools=effective_tools,
            preferred_arm=preferred_arm,
        )

    def _notify_request_observer(self, payload: dict[str, Any]) -> None:
        observer = self._request_observer
        if observer is None:
            return
        try:
            observer(payload)
        except Exception:
            # Auditing must never affect recovery planning or execution.
            pass

    def _effective_available_tools(self, available_tools: list[str] | set[str] | None) -> set[str]:
        if available_tools is None:
            return set(RECOVERY_TOOLS)
        return {str(tool).strip() for tool in available_tools if str(tool).strip() in RECOVERY_TOOLS}

    def _retrieve_experience(
        self,
        *,
        ood_scenario: str,
        current_subtask: str,
        available_tools: list[str],
        semantic_tags: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            normalized_tags = normalize_semantic_tags(semantic_tags)
            state_tags = dict(normalized_tags.get("state_tags", {}))
            query = ExperienceQuery(
                OOD_scenario=ood_scenario,
                task_family=str(normalized_tags.get("task_family", "")),
                subtask_type=str(normalized_tags.get("subtask_type", "")),
                object_state=str(state_tags.get("object_state", "")),
                visibility_state=str(state_tags.get("visibility_state", "")),
                gripper_state=str(state_tags.get("gripper_state", "")),
                motion_state=str(state_tags.get("motion_state", "")),
                current_subtask=current_subtask,
                available_tools=available_tools,
                tag_source=str(normalized_tags.get("tag_source", "unknown")),
            )
            return self._experience_retriever.retrieve(query)
        except Exception as exc:
            print(f"[recovery_skill_loader] experience retrieval failed: {exc}", flush=True)
            return {"lessons": [], "similar_cases": [], "avoid_patterns": []}

    def _collect_skills(self, path_marker: str) -> list[LocalSkill]:
        skills = self._skill_registry.refresh()
        selected = [skill for skill in skills.values() if path_marker in skill.skill_md_path.as_posix()]
        return sorted(selected, key=lambda skill: skill.name)

    def _planning_skill_payloads(
        self,
        *,
        workflow_skills: list[LocalSkill],
        primitive_skills: list[LocalSkill],
        signal_name: str,
        ood_scenario: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        active_signals = {
            self._normalize_signal_name(value)
            for value in (signal_name, ood_scenario)
            if self._normalize_signal_name(value)
        }
        candidates = [
            skill
            for skill in workflow_skills
            if active_signals & set(self._skill_routing_signals(skill))
        ]
        candidate_names = [skill.name for skill in candidates]
        if not candidate_names:
            return (
                [self._skill_payload(skill) for skill in workflow_skills],
                [self._skill_payload(skill) for skill in primitive_skills],
                [],
            )
        candidate_set = set(candidate_names)
        return (
            [self._skill_payload(skill, include_body=skill.name in candidate_set) for skill in workflow_skills],
            [self._skill_payload(skill, include_body=False) for skill in primitive_skills],
            candidate_names,
        )

    def _skill_routing_signals(self, skill: LocalSkill) -> list[str]:
        metadata = getattr(skill, "metadata", {})
        raw_signals = metadata.get("signals", []) if isinstance(metadata, dict) else []
        if isinstance(raw_signals, str):
            raw_signals = [raw_signals]
        if not isinstance(raw_signals, list):
            return []
        return [
            normalized
            for normalized in (self._normalize_signal_name(item) for item in raw_signals)
            if normalized
        ]

    @staticmethod
    def _normalize_signal_name(value: Any) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    def _skill_payload(self, skill: LocalSkill, *, include_body: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": skill.name,
            "description": skill.description,
            "path": str(skill.skill_md_path),
            "signals": self._skill_routing_signals(skill),
        }
        if include_body:
            payload["body"] = skill.body
        return payload

    def _record_error(self, message: str, *, kind: str = "contract") -> None:
        self.last_error = str(message)
        self.last_error_kind = str(kind)
        print(f"[recovery_skill_loader] {self.last_error}", flush=True)

    def _build_route_from_plan(
        self,
        *,
        signal_name: str,
        reason: str,
        raw_plan: dict[str, Any],
        workflow_names: set[str],
        available_tools: set[str],
        preferred_arm: str = "either",
    ) -> RecoveryRoute | None:
        workflow_name = str(raw_plan.get("recovery_workflow", "")).strip()
        if workflow_name not in workflow_names:
            self._record_error(f"recovery planner returned unknown workflow {workflow_name!r}")
            return None
        intent = str(raw_plan.get("post_recovery_intent", "")).strip().lower()
        if intent not in _ALLOWED_RECOVERY_INTENTS:
            self._record_error(f"recovery planner returned invalid intent {intent!r}")
            return None
        raw_tool_calls = raw_plan.get("tool_calls", [])
        if raw_tool_calls is None:
            raw_tool_calls = []
        if not isinstance(raw_tool_calls, list):
            self._record_error("recovery planner returned non-list tool_calls")
            return None
        tool_calls: list[RecoveryToolCall] = []
        for item in raw_tool_calls:
            if not isinstance(item, dict):
                self._record_error(f"invalid recovery tool call {item!r}")
                return None
            tool_name = str(item.get("tool_name", "")).strip()
            if tool_name not in available_tools:
                self._record_error(f"recovery planner returned unavailable tool {tool_name!r}")
                return None
            args = item.get("args", {})
            if args is None:
                args = {}
            if not isinstance(args, dict):
                self._record_error(f"recovery tool args must be dict for {tool_name!r}")
                return None
            tool_calls.append(RecoveryToolCall(tool_name=tool_name, args=args))
        selected_arm = self._validate_selected_arm(
            raw_plan=raw_plan,
            tool_calls=tool_calls,
            preferred_arm=preferred_arm,
        )
        if selected_arm is None:
            return None
        plan = RecoveryPrimitivePlan(
            name=workflow_name,
            tool_calls=tool_calls,
            expected_outcome=str(raw_plan.get("stop_condition", "")).strip(),
        )
        return RecoveryRoute(
            signal_name=signal_name,
            workflow_name=workflow_name,
            plan=plan,
            post_recovery_intent=intent,
            selected_arm=selected_arm,
            reason=str(raw_plan.get("reason", reason)).strip() or reason,
        )

    def _validate_selected_arm(
        self,
        *,
        raw_plan: dict[str, Any],
        tool_calls: list[RecoveryToolCall],
        preferred_arm: str,
    ) -> str | None:
        for call in tool_calls:
            if call.tool_name not in _ARM_BOUND_PHYSICAL_TOOLS:
                continue
            arm = normalize_physical_arm((call.args or {}).get("arm"))
            if arm not in {"left", "right", "both"}:
                self._record_error(
                    f"physical recovery tool {call.tool_name!r} requires an explicit left/right/both arm"
                )
                return None

        progress_arms: set[str] = set()
        for call in tool_calls:
            if call.tool_name not in _ARM_BOUND_TASK_PROGRESS_TOOLS:
                continue
            arm = normalize_physical_arm((call.args or {}).get("arm"))
            if arm not in {"left", "right"}:
                self._record_error(
                    f"recovery task-progress tool {call.tool_name!r} requires one structured left/right arm"
                )
                return None
            progress_arms.add(arm)

        if not progress_arms:
            return "none"
        if len(progress_arms) != 1:
            self._record_error(
                f"recovery task-progress batch mixes selected arms: {sorted(progress_arms)!r}"
            )
            return None

        progress_arm = next(iter(progress_arms))
        if "selected_arm" in raw_plan:
            selected_arm = normalize_selected_arm(raw_plan.get("selected_arm"))
        else:
            selected_arm = progress_arm
        if selected_arm != progress_arm:
            self._record_error(
                "recovery selected_arm does not match task-progress tool calls: "
                f"selected_arm={selected_arm!r} tool_arm={progress_arm!r}"
            )
            return None

        normalized_preference = normalize_preferred_arm(preferred_arm)
        if normalized_preference in {"left", "right"} and selected_arm != normalized_preference:
            self._record_error(
                "recovery selected_arm violates the active planner preference: "
                f"selected_arm={selected_arm!r} preferred_arm={normalized_preference!r}"
            )
            return None
        return selected_arm
