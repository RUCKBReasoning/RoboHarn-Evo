from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from roboharn_evo.models.agent_api_planner_adapter import _post_json
from roboharn_evo.models.prompt_rendering import render_known_prompt_fields
from roboharn_evo.agent.execution_feedback import DIRECT_FEEDBACK_RECOVERY_PROMPT


DEFAULT_RECOVERY_PROMPT = """You are the recovery planner in a robot manipulation system.
You receive a structured payload containing recovery-router, workflow, and primitive SKILL harnesses.

Return JSON only with this schema:
{
  "recovery_workflow": "one workflow name from available workflow SKILLs",
  "selected_arm": "left | right | none",
  "tool_calls": [
    {"tool_name": "runtime tool name", "args": {}, "reason": "short evidence-based reason"}
  ],
  "post_recovery_intent": "retry | replan | abort",
  "reason": "short explanation for selected workflow and plan",
  "stop_condition": "condition for stopping recovery"
}

Rules:
- When `recovery_state.planner_context_mode=compact_v1`, `scene_memory` is the single `planner_scene_view/compact_v1` projection. It already merges stable identity, world position, public action availability, placement targets, reference regions, spatial relations, manipulation state, and key uncertainty. `observation_summary` and `observation_preprocess` may be empty by design; do not request or reconstruct duplicate scene copies.
- Use only tool names supplied in `available_tools`.
- Generate a situation-specific plan; reference plans in SKILL text are examples, not mandatory fixed sequences.
- Use grounded EE tools only when `scene_memory.instances` contains the requested finite `world_m` or `approach_world_m` evidence; do not invent coordinates.
- Prefer `move_ee_to_grounded_instance` with an existing exact `instance_id` over copying coordinates into the plan. `role` or `focus_key` alone is allowed only when the referenced focus contains exactly one instance and `scene_memory.task_focus.identity_binding_required` is false.
- When `scene_memory.task_focus.identity_binding_required=true`, every `move_ee_to_grounded_instance` call must include an exact `args.instance_id` or `args.instance_ref` from `scene_memory.instances`.
- When `scene_memory.operation_targets` is non-empty for a carried object in manipulation state, select one public `target_id`; never output a private `candidate_id`. Use `action_mode: "place"` and let runtime bind the held instance and choose the arm-specific executable candidate.
- If an operation target has `target_kind=reference_region` and its `placement_relation` matches the requested relational destination, use that target instead of a generic `free_support` fallback. Its metric coordinate has already been computed and revalidated by runtime.
- A place sequence uses the same public target twice: first `point_key: "approach_world_m"`, then `point_key: "place_world_m"`. These poses already preserve the measured held-object-to-TCP transform; do not substitute a destination object's grasp/contact pose or add `preserve_height`.
- In carry mode (`preserve_height=true`), `args.instance_id` is the destination and `args.held_instance_id` is the verifier-confirmed held object. They must be different exact scene-instance IDs.
- Treat `scene_memory.manipulation_state` in compact_v1, or `recovery_payload.recovery_state.manipulation_state` in legacy mode, as authoritative physical state across replans. If an arm has `holding_confirmed=true`, continue carry/place instead of opening and re-grasping that object.
- Read `recovery_payload.recovery_state.release_guard_enabled`. When it is false, an explicit `open_gripper` is dispatched after ordinary argument validation; `release_held_instance_id` and `release_target_id` are optional attribution metadata, and a same-batch `place_world_m` call is not required. Decide from the task and runtime state when opening is appropriate. Runtime records detachment, placement error, and stability afterward. When the flag is true, follow the legacy matching held-instance/target/final-place release contract.
- When execution_evidence_enabled=true and `manipulation_state` has `phase=release_pending_verification`, wait for runtime placement verification before another grasp.
- Select `selected_arm` from current geometry and robot state. Every task-progress tool call must use the same arm in `args.arm` and must respect `recovery_payload.preferred_arm`. Use `none` for re-observation or clearance-only plans; a clearance action may retreat a previously used non-preferred arm.
- Every physical tool call must include an explicit `args.arm` chosen from current evidence; only `reobserve_scene` has no arm. Clearance-only plans may still use `selected_arm: none` even though each physical clearance call names its arm.
- Read each `recovery_payload.blocked_grounded_setups` status. For a legacy blocked instance/arm/point setup, or when `all_candidates_blocked=true`, do not retry the same public action. When `runtime_will_select_next=true`, the listed block applies only to a private pose candidate: retrying the same public instance/arm/action_mode is allowed and runtime will select an unblocked candidate. Never request or invent a private candidate ID.
- Treat arm values in SKILL examples as illustrative. Select the arm from current robot state, grounded geometry, gripper state, and observed action effects rather than defaulting to `left`.
- In dual-arm environments, after repeated verified no-motion results for one arm, consider one bounded probe with the other free arm before repeating or abandoning the same grounded subtask.
- For approach or grasp motions, use the instance's grounded quaternion when available; request `target_quat_wxyz: "grounded"` instead of preserving an unrelated current orientation.
- When a planned motion requires a particular gripper state, encode it as `gripper_precondition: "open" | "closed"` in that motion's args. Do not rely on prose or implicit state changes.
- `robot_state[arm].gripper_command` is the executed controller target (0 closes, 1 opens); `gripper` is the benchmark-reported gripper reading. A nonzero reading during a maintained closing command can occur during closure or contact with an object. Use the manipulation state and observed object motion to assess attachment. Motion preserves the controller target.
- For a transient press/tap, set `complete_transient_cycle: true` on the inward grounded `contact_displace`. Runtime first releases the same arm to that instance's grounded approach, then uses valid live camera/contact geometry to add a separate same-batch visibility-clearance move before re-observation. This cleanup is part of the same event; do not use the marker for sustained contact or placement lowering.
- Before any test/confirmation press after rearrangement, read `scene_memory.spatial_state`, whose order, overlap, support relations, and signature are computed from current coordinates. Never trust a prose claim that the arrangement is correct when those coordinates disagree.
- EE recovery tools may use `args.steps` for bounded multi-step control. Keep steps small and purposeful; runtime caps unsafe values.
- It is allowed to return an empty tool_calls list when replanning is the right recovery action.
- When execution_evidence_enabled=true, the after-action verifier decides the subtask transition. When false, ordinary planning continues from the next observation and execution feedback.
- Do not output markdown, prose outside JSON, or extra keys.
"""


RECOVERY_PAYLOAD_REFERENCE = "The structured recovery payload is provided separately in the user message."

@dataclass(frozen=True)
class AgentApiRecoveryConfig:
    server_url: str
    timeout_sec: int
    prompt_template: str
    auth_token: str = ""
    auth_header: str = "Authorization"
    extra_headers: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None


class AgentApiRecoveryAdapter:
    def __init__(self, config: AgentApiRecoveryConfig) -> None:
        self.config = config

    def plan_recovery(self, *, recovery_payload: dict[str, Any], media: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        headers = dict(self.config.extra_headers or {})
        if self.config.auth_token:
            headers[self.config.auth_header] = self.config.auth_token
        prompt = render_known_prompt_fields(
            self.config.prompt_template,
            recovery_payload=RECOVERY_PAYLOAD_REFERENCE,
        )
        observation_state = (
            recovery_payload.get("evidence_payload", {})
            if recovery_payload.get("mode") == "action_effect_verification"
            else recovery_payload.get("recovery_state", {})
        )
        if isinstance(observation_state, dict) and observation_state.get("execution_evidence_enabled") is False:
            prompt = DIRECT_FEEDBACK_RECOVERY_PROMPT
        elif isinstance(observation_state, dict) and observation_state.get("reobserve_scene_enabled") is False:
            prompt += (
                "\nRuntime tool override: reobserve_scene is disabled. "
                "This overrides observation-tool examples in the workflow guides. "
                "Do not request it or create observation-only subtasks. "
                "Normal post-action observations and state updates still occur; "
                "use that evidence directly without requiring an explicit observation tool call. "
                "Read last_action_rejection and runtime_action_rejection when present: "
                "a retained physical prefix is not execution of the rejected action. "
                "Correct its specific precondition rather than repeating the same blocked plan. "
                "Missing evidence must remain unverified. Use only available tools "
                "for justified task progress or visibility changes; do not invent a "
                "replacement observation tool or move merely to bypass the disabled tool."
            )
        payload = {
            "recovery_payload": recovery_payload,
            "prompt": prompt,
            "media": list(media or []),
            **(self.config.extra_body or {}),
        }
        return _post_json(
            url=self.config.server_url,
            payload=payload,
            timeout_sec=self.config.timeout_sec,
            headers=headers,
        )
