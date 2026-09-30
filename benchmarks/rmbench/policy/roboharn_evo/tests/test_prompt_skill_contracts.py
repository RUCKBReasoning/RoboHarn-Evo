from __future__ import annotations

import json

from policy.roboharn_evo.agent.reasoners.prompts import build_reasoner_turn_payload
from policy.roboharn_evo.models.prompt_rendering import render_known_prompt_fields
from policy.roboharn_evo.models.prompt_skills import load_prompt_skill, resolve_prompt_template
from policy.roboharn_evo.scripts.serve_qwen_planner import (
    PERCEPTION_QUERY_SYSTEM_PROMPT,
    load_perception_query_normalization_prompt,
    normalize_prediction,
    normalize_recovery_prediction,
)


def test_control_turn_prompt_skill_loads_prompt_body() -> None:
    prompt = load_prompt_skill("control-turn-planner")
    assert "Task instruction: {task}" in prompt
    assert "Previous committed memory: {previous_memory_text}" in prompt
    assert "State summary vector: {state_summary}" in prompt
    assert "after-action verifier" in prompt
    assert "does not create a completion transition" in prompt
    assert "runtime_evaluation.global_task_success" in prompt
    assert "never disguise a completion/no-action claim" in prompt
    assert "one atomic event per subtask" in prompt
    assert "manipulation_state" in prompt
    assert "do not restart by opening and picking the same object" in prompt
    assert "request `reobserve_scene` only" in prompt
    assert "release_recovery_required" in prompt
    assert "bounded reacquire-and-replace" in prompt


def test_planning_skills_preserve_evidence_only_transport_semantics() -> None:
    runtime_reasoning = load_prompt_skill("runtime-state-reasoning")
    control_planner = load_prompt_skill("control-turn-planner")

    for prompt in (runtime_reasoning, control_planner):
        assert "grasp_transport_policy=strict" in prompt
        assert "grasp_transport_policy=evidence_only" in prompt
        assert "phase=holding_provisional" in prompt
        assert "transport_authorized=true" in prompt
        assert "holding_confirmed=true" in prompt
        assert "provisional" in prompt
        assert "negative visual attachment evidence" in prompt

    assert "never as `holding_confirmed=true`" in runtime_reasoning
    assert "automatic gripper opening" in runtime_reasoning
    assert "release_guard_enabled=false" in runtime_reasoning
    assert "same-batch final-place call" in runtime_reasoning
    assert "call this state provisional, not confirmed" in control_planner
    assert "does not relax target occupancy" in control_planner
    assert "Opening to abandon provisional state must be an explicit" in (
        control_planner
    )


def test_planning_skills_explain_action_geometry_repair_modes() -> None:
    runtime_reasoning = load_prompt_skill("runtime-state-reasoning")
    control_planner = load_prompt_skill("control-turn-planner")
    workflow = load_prompt_skill("task-level-recovery-control")

    for prompt in (runtime_reasoning, control_planner, workflow):
        assert "action_geometry_repair_pending_policy" in prompt
        assert "strict" in prompt
        assert "safe_motion" in prompt
        assert "disabled" in prompt
        assert "relocation_pending" in prompt

    assert "poses remain quarantined" in runtime_reasoning
    assert "Never treat a planner-authored pose" in control_planner
    assert "unrelated" in workflow
    assert "guard remains active" in workflow


def test_action_effect_prompt_requires_grounded_transient_contact_chain() -> None:
    prompt = load_prompt_skill("action-effect-verification")

    assert "transient contact event" in prompt
    assert "target_reached=true" in prompt
    assert "exactly one event" in prompt
    assert "complete_transient_cycle=true" in prompt
    assert "grounded return to `approach_world_m`" in prompt
    assert "post_contact_clearance=camera_tangent" in prompt
    assert "recover visibility separately" in prompt
    assert "Tool success without the contact-and-release chain remains insufficient" in prompt
    assert "global terminal authority" in prompt
    assert "check_success=true" in prompt
    assert "runtime_place_validation" in prompt
    assert "verified=true" in prompt
    assert "placement_recovery_required=true" in prompt
    assert "actual_spatial_state_signature" in prompt


def test_perception_prompt_assigns_instance_semantics_to_agent_api() -> None:
    normalization_prompt = load_perception_query_normalization_prompt()

    assert "instance_ref" in PERCEPTION_QUERY_SYSTEM_PROMPT
    assert "oracle_id" in PERCEPTION_QUERY_SYSTEM_PROMPT
    assert "Do not assume list order" in PERCEPTION_QUERY_SYSTEM_PROMPT
    assert "binding_requirement.required_any_roles" in PERCEPTION_QUERY_SYSTEM_PROMPT
    assert "context-only response" in PERCEPTION_QUERY_SYSTEM_PROMPT
    assert "candidate_discovery" in PERCEPTION_QUERY_SYSTEM_PROMPT
    assert "partial catalog" in PERCEPTION_QUERY_SYSTEM_PROMPT
    assert "post_detection_selection" in PERCEPTION_QUERY_SYSTEM_PROMPT
    assert "entity_scope=reference_set" in PERCEPTION_QUERY_SYSTEM_PROMPT
    assert "placement_relation=center_of" in PERCEPTION_QUERY_SYSTEM_PROMPT
    assert "member-level segmentation prompt" in PERCEPTION_QUERY_SYSTEM_PROMPT
    assert "exact identifier membership" in normalization_prompt
    assert "Do not choose by candidate-list order" in normalization_prompt
    assert "binding_requirement.required_any_roles" in normalization_prompt
    assert "bounded retry" in normalization_prompt
    assert "candidate_discovery" in normalization_prompt
    assert "partial catalog" in normalization_prompt
    assert "post_detection_selection" in normalization_prompt
    assert "entity_scope=reference_set" in normalization_prompt
    assert "placement_relation=center_of" in normalization_prompt
    assert "member-level segmentation prompt" in normalization_prompt


def test_prompt_template_resolves_from_skill() -> None:
    prompt = resolve_prompt_template({"prompt_skill": "control-turn-planner"}, "fallback")
    assert prompt != "fallback"
    assert "Return JSON only with this schema" in prompt


def test_prompt_skill_json_schema_braces_do_not_break_rendering() -> None:
    prompt = load_prompt_skill("control-turn-planner")
    rendered = render_known_prompt_fields(
        prompt,
        task="cover the block",
        previous_memory_text="The task has started.",
        state_summary="[0.0]",
    )
    assert '"commit_label"' in rendered
    assert "Task instruction: cover the block" in rendered
    assert "State summary vector: [0.0]" in rendered


def test_reasoner_payload_includes_runtime_state_reasoning_skill() -> None:
    payload = build_reasoner_turn_payload(
        global_task="cover the block",
        agent_state={"task": {}, "working": {}, "monitor": {}, "recovery": {}},
        available_skills=["monitored-subtask-execution"],
        trigger="interval",
        memory_harness={"name": "observation-memory-summarization", "instructions": "test"},
        runtime_evaluation={
            "environment_success": False,
            "check_success": True,
            "global_task_success": False,
            "global_success_authority": "environment_eval_success",
        },
    )
    decoded = json.loads(payload)
    assert decoded["runtime_state_reasoning"]["name"] == "runtime-state-reasoning"
    assert "Working memory" in decoded["runtime_state_reasoning"]["instructions"]
    assert decoded["runtime_evaluation"]["environment_success"] is False
    assert decoded["runtime_evaluation"]["check_success"] is True
    assert decoded["runtime_evaluation"]["global_task_success"] is False
    assert "manipulation_state" in decoded["runtime_state_reasoning"]["instructions"]


def test_recovery_skills_require_explicit_identity_bound_carry_contract() -> None:
    workflow = load_prompt_skill("task-level-recovery-control")
    move_primitive = load_prompt_skill("move-ee-to-grounded-instance")
    open_primitive = load_prompt_skill("open-gripper")

    assert "identity_binding_required=true" in workflow
    assert "exact destination `instance_id`" in workflow
    assert "exact `held_instance_id`" in workflow
    assert "release_held_instance_id" in workflow
    assert "scene_memory.operation_targets" in workflow
    assert 'action_mode = "place"' in workflow
    assert "release_pending_verification" in workflow
    assert "release_recovery_required" in workflow
    assert "use `reobserve_scene` only" in workflow
    assert "scene_memory.spatial_state" in workflow
    assert "identity_binding_required=true" in move_primitive
    assert '"held_instance_id": "object_left"' in move_primitive
    assert "target_id" in move_primitive
    assert "held-object-to-TCP" in move_primitive
    assert "release_held_instance_id" in open_primitive
    assert "release_target_id" in open_primitive
    assert "verification becomes observation-only" in open_primitive


def test_recovery_skills_distinguish_strict_and_evidence_only_grasps() -> None:
    workflow = load_prompt_skill("task-level-recovery-control")
    move_primitive = load_prompt_skill("move-ee-to-grounded-instance")
    open_primitive = load_prompt_skill("open-gripper")

    assert "grasp_transport_policy" in workflow
    assert "Under `strict`" in workflow
    assert "Under `evidence_only`" in workflow
    assert "phase=holding_provisional" in workflow
    assert "transport_authorized=true" in workflow
    assert "does not mean `holding_confirmed=true`" in workflow
    assert "automatic gripper opening" in workflow
    assert "explicit planner `open_gripper`" in workflow
    assert "release_guard_enabled" in workflow
    assert "exact held-instance/public-target/final-place contract" in workflow
    assert "do not copy the final `reobserve_scene`" in workflow

    assert "grasp_transport_policy" in move_primitive
    assert "phase=holding_provisional" in move_primitive
    assert "transport_authorized=true" in move_primitive
    assert "must not be reported as `holding_confirmed=true`" in move_primitive
    assert "must not force this primitive to insert diagnostic motion" in move_primitive
    assert "provisional authority does not relax target occupancy" in move_primitive

    assert "grasp_transport_policy=evidence_only" in open_primitive
    assert "release_guard_enabled" in open_primitive
    assert "This explicit open is allowed" in open_primitive
    assert "must not automatically insert `open_gripper`" in open_primitive
    assert "same-batch final-place call" in open_primitive
    assert "they are not authorization" in open_primitive
    assert "without claiming `holding_confirmed=true`" in open_primitive


def test_recovery_skills_require_candidate_approach_before_new_grasp() -> None:
    workflow = load_prompt_skill("task-level-recovery-control")
    move_primitive = load_prompt_skill("move-ee-to-grounded-instance")

    for prompt in (workflow, move_primitive):
        assert "every new portable-object" in prompt.lower()
        assert "arbitrary or unrelated end-effector pose" in prompt
        assert "approach_world_m" in prompt
        assert "grasp_world_m" in prompt
        assert "close_gripper" in prompt
        assert "static batch" in prompt
        assert "fixed world-axis offset" in prompt

    assert "same arm has already reached" in workflow
    assert "exact same instance and grasp attempt" in workflow
    assert "A separate re-observation between these calls is optional, not mandatory" in workflow
    assert "mandatory safe staging pose" in move_primitive
    assert "no intervening re-observation is required" in move_primitive


def test_planner_normalizer_preserves_valid_action_mode() -> None:
    prediction = normalize_prediction(
        {
            "commit_label": "state_change",
            "memory_text": "Current state is not globally successful.",
            "subtask_text": "Perform a bounded corrective manipulation.",
            "action_mode": "FINISH",
            "preferred_arm": "RIGHT_ARM",
        },
        previous_memory_text="Previous state.",
    )
    invalid = normalize_prediction(
        {
            "commit_label": "state_change",
            "memory_text": "Continue.",
            "subtask_text": "Perform the next bounded step.",
            "action_mode": "wait_forever",
        },
        previous_memory_text="Previous state.",
    )

    assert prediction["action_mode"] == "finish"
    assert prediction["preferred_arm"] == "right"
    assert "action_mode" not in invalid
    assert invalid["preferred_arm"] == "either"


def test_recovery_normalizer_preserves_structured_selected_arm() -> None:
    prediction = normalize_recovery_prediction(
        {
            "recovery_workflow": "task-level-recovery-control",
            "selected_arm": "RIGHT_ARM",
            "tool_calls": [
                {
                    "tool_name": "move_ee_to_grounded_instance",
                    "args": {"arm": "right", "instance_id": "target_01"},
                }
            ],
            "post_recovery_intent": "retry",
        }
    )
    inferred = normalize_recovery_prediction(
        {
            "recovery_workflow": "task-level-recovery-control",
            "tool_calls": [{"tool_name": "contact_displace", "args": {"arm": "left"}}],
            "post_recovery_intent": "retry",
        }
    )

    assert prediction["selected_arm"] == "right"
    assert inferred["selected_arm"] == "left"


def test_recovery_normalizer_preserves_transient_cycle_marker() -> None:
    prediction = normalize_recovery_prediction(
        {
            "recovery_workflow": "task-level-recovery-control",
            "selected_arm": "left",
            "tool_calls": [
                {
                    "tool_name": "contact_displace",
                    "args": {
                        "arm": "left",
                        "axis": "z",
                        "direction": "negative",
                        "distance": 0.012,
                        "complete_transient_cycle": True,
                    },
                }
            ],
            "post_recovery_intent": "retry",
        }
    )

    assert prediction["tool_calls"][0]["args"]["complete_transient_cycle"] is True


def test_observation_memory_harness_uses_existing_memory_surfaces() -> None:
    prompt = load_prompt_skill("observation-memory-summarization")
    assert "position_state=current_verified" in prompt
    assert "position_state=memory_valid" in prompt
    assert "position_state=motion_uncertain" in prompt
    assert "recent_observation_summary" in prompt
    assert "scene_memory" in prompt
    assert "observation_preprocess" in prompt
    assert "recovery_history" in prompt
    assert "Do not invent fields beyond" in prompt
    assert "world_m = null" in prompt
    assert "approach_world_m = null" in prompt
    assert "closed after approach point" in prompt
