from __future__ import annotations

from policy.roboharn_evo.agent.components.memory_manager import MemoryManager
from policy.roboharn_evo.agent.skills.base import SkillSpec


def test_start_or_replace_skill_stores_planner_semantic_tags() -> None:
    memory = MemoryManager(initial_memory_text="started")
    memory.reset(
        task="put the mug in the tray",
        control_model_name="planner",
        available_policies=["policy"],
        available_tools=[],
        mode="deployment",
    )
    skill = SkillSpec(
        name="monitored-subtask-execution",
        description="execute",
        policy_binding="policy",
        instruction_template="{subtask}",
    )

    run_state = memory.start_or_replace_skill(
        subtask_text="grasp the mug",
        skill_spec=skill,
        semantic_tags={
            "task_family": "pick_and_place",
            "subtask_type": "grasp",
            "state_tags": {"visibility_state": "visible"},
            "tag_source": "planner_vlm",
        },
    )

    assert run_state.semantic_tags["task_family"] == "pick_and_place"
    assert run_state.semantic_tags["subtask_type"] == "grasp"
    assert run_state.semantic_tags["state_tags"]["visibility_state"] == "visible"
    assert memory.state.working.semantic_tags == run_state.semantic_tags


def test_recovery_evidence_is_scoped_to_new_skill_but_retained_for_retry() -> None:
    memory = MemoryManager(initial_memory_text="started")
    memory.reset(
        task="perform repeated grounded actions",
        control_model_name="planner",
        available_policies=["policy"],
        available_tools=[],
        mode="deployment",
    )
    skill = SkillSpec(
        name="monitored-subtask-execution",
        description="execute",
        policy_binding="policy",
        instruction_template="{subtask}",
    )

    first = memory.start_or_replace_skill(subtask_text="first action", skill_spec=skill)
    memory.record_recovery("action_effect:effect=true,subtask=completed")
    memory.set_manipulation_arm_state(
        "right",
        {
            "phase": "holding",
            "held_instance_id": "track_0001",
            "holding_confirmed": True,
        },
    )
    retried = memory.start_or_replace_skill(subtask_text="first action", skill_spec=skill)

    assert retried.skill_id == first.skill_id
    assert memory.state.working.recovery_history == [
        "action_effect:effect=true,subtask=completed"
    ]

    memory.mark_active_skill_succeeded(note="verified")
    second = memory.start_or_replace_skill(subtask_text="second action", skill_spec=skill)

    assert second.skill_id != first.skill_id
    assert memory.state.working.recovery_history == []
    assert memory.state.working.manipulation_state == {
        "right": {
            "phase": "holding",
            "held_instance_id": "track_0001",
            "holding_confirmed": True,
        }
    }


def test_explicit_new_attempt_does_not_reuse_same_rendered_instruction() -> None:
    memory = MemoryManager(initial_memory_text="started")
    memory.reset(
        task="press the same control more than once",
        control_model_name="planner",
        available_policies=["policy"],
        available_tools=[],
        mode="deployment",
    )
    skill = SkillSpec(
        name="monitored-subtask-execution",
        description="execute",
        policy_binding="policy",
        instruction_template="{subtask}",
    )
    first = memory.start_or_replace_skill(
        subtask_text="press the selected control once",
        skill_spec=skill,
    )
    memory.record_recovery("action_effect:effect=true,subtask=completed")

    second = memory.start_or_replace_skill(
        subtask_text="press the selected control once",
        skill_spec=skill,
        force_new_attempt=True,
    )

    assert second.skill_id != first.skill_id
    assert memory.state.working.recovery_history == []
