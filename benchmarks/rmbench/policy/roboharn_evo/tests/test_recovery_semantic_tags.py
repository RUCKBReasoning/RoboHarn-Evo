from __future__ import annotations

from pathlib import Path
from typing import Any

from policy.roboharn_evo.agent.recovery.skill_workflow_loader import SkillRecoveryWorkflowLoader
from policy.roboharn_evo.models.agent_api_recovery_adapter import DEFAULT_RECOVERY_PROMPT


class FakeSkill:
    def __init__(self, name: str, path: str, *, metadata: dict[str, Any] | None = None) -> None:
        self.name = name
        self.description = name
        self.body = f"# {name}"
        self.metadata = dict(metadata or {})
        self.skill_md_path = type("SkillPath", (), {"as_posix": lambda self: path})()


class FakeRegistry:
    def __init__(self) -> None:
        self.router = FakeSkill("recovery-router", "/skills/recovery/recovery-router/SKILL.md")
        self.workflow = FakeSkill("recover-motion-blocked", "/skills/recovery/recovery-workflows/recover-motion-blocked/SKILL.md")
        self.primitive = FakeSkill("reobserve-scene", "/skills/recovery/recovery-primitives/reobserve-scene/SKILL.md")

    def get_skill(self, name: str, refresh: bool = False) -> FakeSkill | None:
        if name == "recovery-router":
            return self.router
        return None

    def refresh(self) -> dict[str, FakeSkill]:
        return {
            self.router.name: self.router,
            self.workflow.name: self.workflow,
            self.primitive.name: self.primitive,
        }


class FakeBackend:
    def __init__(self, workflow: str = "recover-motion-blocked") -> None:
        self.payload: dict[str, Any] | None = None
        self.workflow = workflow

    def plan_recovery(self, *, recovery_payload: dict[str, Any], media: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self.payload = recovery_payload
        return {
            "recovery_workflow": self.workflow,
            "tool_calls": [],
            "post_recovery_intent": "retry",
            "reason": "test",
            "stop_condition": "ready",
        }


class FakeRetriever:
    def __init__(self) -> None:
        self.query = None

    def retrieve(self, query, *, top_k_lessons: int = 3, top_k_cases: int = 2):
        self.query = query
        return {
            "query": {
                "OOD_scenario": query.OOD_scenario,
                "task_family": query.task_family,
                "subtask_type": query.subtask_type,
                "object_state": query.object_state,
                "visibility_state": query.visibility_state,
                "gripper_state": query.gripper_state,
                "motion_state": query.motion_state,
                "tag_source": query.tag_source,
            },
            "lessons": [],
            "similar_cases": [],
            "avoid_patterns": [],
        }


def test_recovery_payload_uses_validated_semantic_tags() -> None:
    backend = FakeBackend()
    retriever = FakeRetriever()
    loader = SkillRecoveryWorkflowLoader(
        FakeRegistry(),
        recovery_backend=backend,
        experience_retriever=retriever,
    )

    route = loader.resolve(
        signal_name="motion_blocked",
        reason="blocked",
        ood_scenario="motion_blocked",
        current_subtask="grasp the mug",
        available_tools=["reobserve_scene"],
        robot_state={"left": {"xyz": [0.1, 0.2, 0.3], "gripper": 0.4}},
        scene_memory={
            "instances": [
                {
                    "instance_id": "widget_left",
                    "world_m": [0.1, 0.2, 0.3],
                    "approach_world_m": [0.1, 0.2, 0.35],
                }
            ],
            "task_focus": {"target_instances": ["widget_left"]},
        },
        observation_preprocess={"stage": "observation_preprocess", "segmentation": [{"object_id": "widget"}]},
        semantic_tags={
            "task_family": "pick_and_place",
            "subtask_type": "grasp",
            "state_tags": {
                "motion_state": "blocked",
                "visibility_state": "object_not_visible",
                "object_state": "invalid",
            },
            "tag_source": "ood_vlm",
        },
        preferred_arm="right",
        blocked_grounded_setups=[
            {
                "instance_id": "widget_left",
                "arm": "left",
                "point_key": "contact_world_m",
                "status": "blocked_after_repeated_no_progress",
            }
        ],
    )

    assert route is not None
    assert backend.payload is not None
    assert backend.payload["semantic_tags"]["task_family"] == "pick_and_place"
    assert backend.payload["semantic_tags"]["subtask_type"] == "grasp"
    assert backend.payload["semantic_tags"]["state_tags"]["motion_state"] == "blocked"
    assert backend.payload["semantic_tags"]["state_tags"]["object_state"] == ""
    assert backend.payload["robot_state"]["left"]["xyz"] == [0.1, 0.2, 0.3]
    assert backend.payload["scene_memory"]["instances"][0]["instance_id"] == "widget_left"
    assert backend.payload["observation_preprocess"]["stage"] == "observation_preprocess"
    assert backend.payload["preferred_arm"] == "right"
    assert backend.payload["blocked_grounded_setups"][0]["instance_id"] == "widget_left"
    assert backend.payload["retrieved_experience"]["query"]["tag_source"] == "ood_vlm"
    assert retriever.query.task_family == "pick_and_place"
    assert retriever.query.subtask_type == "grasp"
    assert retriever.query.motion_state == "blocked"
    assert retriever.query.visibility_state == "object_not_visible"


def test_recovery_payload_expands_only_signal_matched_workflow_body() -> None:
    class SignalRegistry(FakeRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.workflow = FakeSkill(
                "task-level-recovery-control",
                "/skills/recovery/recovery-workflows/task-level-recovery-control/SKILL.md",
                metadata={"signals": ["task_level_recovery_control"]},
            )
            self.other_workflow = FakeSkill(
                "recover-motion-blocked",
                "/skills/recovery/recovery-workflows/recover-motion-blocked/SKILL.md",
            )

        def refresh(self) -> dict[str, FakeSkill]:
            return {
                self.router.name: self.router,
                self.workflow.name: self.workflow,
                self.other_workflow.name: self.other_workflow,
                self.primitive.name: self.primitive,
            }

    backend = FakeBackend(workflow="task-level-recovery-control")
    loader = SkillRecoveryWorkflowLoader(
        SignalRegistry(),
        recovery_backend=backend,
        experience_retriever=FakeRetriever(),
    )

    route = loader.resolve(
        signal_name="task_level_recovery_control",
        reason="pure agent tool control",
        available_tools=["reobserve_scene"],
    )

    assert route is not None
    assert backend.payload is not None
    assert backend.payload["skill_payload_mode"] == "signal_scoped"
    assert backend.payload["signal_workflow_candidates"] == ["task-level-recovery-control"]
    workflows = {item["name"]: item for item in backend.payload["workflow_skills"]}
    assert "body" in workflows["task-level-recovery-control"]
    assert "body" not in workflows["recover-motion-blocked"]
    assert all("body" not in item for item in backend.payload["primitive_skills"])


def test_recovery_prompts_require_observed_arm_selection() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / "skills/recovery/recovery-workflows/task-level-recovery-control/SKILL.md"
    )
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "Treat arm values in SKILL examples as illustrative" in DEFAULT_RECOVERY_PROMPT
    assert "selected_arm" in DEFAULT_RECOVERY_PROMPT
    assert "blocked_grounded_setups" in DEFAULT_RECOVERY_PROMPT
    assert "two observed effects show that the same arm did not move" in workflow
    assert "use it for one bounded approach probe" in workflow
    assert "require measurable horizontal source-object separation" in workflow
    assert "lower z bound to be above the source object's upper z bound" in workflow
    assert "horizontal AABBs to be non-overlapping" in workflow
    assert "first_observed_world_m" in workflow
    assert "omitted values default to a downward z displacement" in workflow
    assert '"gripper_precondition": "closed"' in workflow
    assert '"complete_transient_cycle": true' in workflow
    assert "same instance's grounded `approach_world_m`" in workflow
    assert "post_contact_clearance=camera_tangent" in workflow
    assert "not a fixed world axis or a task-text rule" in workflow
    assert 'gripper_precondition: "open" | "closed"' in DEFAULT_RECOVERY_PROMPT
    assert "Every physical tool call must include an explicit `args.arm`" in DEFAULT_RECOVERY_PROMPT
    assert "complete_transient_cycle: true" in DEFAULT_RECOVERY_PROMPT
    assert "live camera/contact geometry" in DEFAULT_RECOVERY_PROMPT


def test_recovery_route_enforces_structured_arm_preference_for_task_progress() -> None:
    loader = SkillRecoveryWorkflowLoader(
        FakeRegistry(),
        recovery_backend=FakeBackend(),
        experience_retriever=FakeRetriever(),
    )
    raw_plan = {
        "recovery_workflow": "recover-motion-blocked",
        "selected_arm": "right",
        "tool_calls": [
            {
                "tool_name": "move_ee_to_grounded_instance",
                "args": {"arm": "right", "instance_id": "widget_01"},
            }
        ],
        "post_recovery_intent": "retry",
    }

    route = loader._build_route_from_plan(
        signal_name="motion_blocked",
        reason="test",
        raw_plan=raw_plan,
        workflow_names={"recover-motion-blocked"},
        available_tools={"move_ee_to_grounded_instance"},
        preferred_arm="right",
    )
    assert route is not None
    assert route.selected_arm == "right"

    rejected = loader._build_route_from_plan(
        signal_name="motion_blocked",
        reason="test",
        raw_plan=raw_plan,
        workflow_names={"recover-motion-blocked"},
        available_tools={"move_ee_to_grounded_instance"},
        preferred_arm="left",
    )
    assert rejected is None
    assert "violates the active planner preference" in loader.last_error


def test_recovery_route_allows_non_preferred_arm_for_clearance_only_plan() -> None:
    loader = SkillRecoveryWorkflowLoader(
        FakeRegistry(),
        recovery_backend=FakeBackend(),
        experience_retriever=FakeRetriever(),
    )
    route = loader._build_route_from_plan(
        signal_name="motion_blocked",
        reason="clear occupied target envelope",
        raw_plan={
            "recovery_workflow": "recover-motion-blocked",
            "selected_arm": "left",
            "tool_calls": [{"tool_name": "retreat_arm", "args": {"arm": "left"}}],
            "post_recovery_intent": "retry",
        },
        workflow_names={"recover-motion-blocked"},
        available_tools={"retreat_arm"},
        preferred_arm="right",
    )

    assert route is not None
    assert route.selected_arm == "none"
    assert route.plan.tool_calls[0].args["arm"] == "left"


def test_recovery_route_rejects_missing_arm_for_physical_clearance_tool() -> None:
    loader = SkillRecoveryWorkflowLoader(
        FakeRegistry(),
        recovery_backend=FakeBackend(),
        experience_retriever=FakeRetriever(),
    )
    route = loader._build_route_from_plan(
        signal_name="motion_blocked",
        reason="clear occupied target envelope",
        raw_plan={
            "recovery_workflow": "recover-motion-blocked",
            "tool_calls": [{"tool_name": "retreat_arm", "args": {}}],
            "post_recovery_intent": "retry",
        },
        workflow_names={"recover-motion-blocked"},
        available_tools={"retreat_arm"},
        preferred_arm="either",
    )

    assert route is None
    assert "requires an explicit left/right/both arm" in loader.last_error
