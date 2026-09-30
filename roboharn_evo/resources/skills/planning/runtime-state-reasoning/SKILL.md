---
name: runtime-state-reasoning
description: "Reasoner payload contract for reading RoboHarn-Evo structured runtime memory and deciding how the planner should use task, working, perception, monitor, and recovery state."
runtime_role: reasoner_payload
---

# Runtime State Reasoning

Use this contract when interpreting `agent_state` inside a control turn.

## State Areas

### Task memory
- `global_task`: the full user task.
- `committed_memory_text`: the latest committed task-state summary.
- `committed_facts`: previous committed summaries.
- `plan`: skills that have been started and their statuses.
- `completed_skills` and `failed_skills`: execution history.
- `task_finished`: whether runtime considers the whole task finished.

### Working memory
- `active_skill_id`, `active_policy_id`, and `active_instruction`: what the executor is currently doing.
- `recent_observation_summary`: compact current observation summary.
- `observation_preprocess`: latest segmentation and grounding evidence.
- `scene_memory`: stable cross-frame object instances and task focus. During manipulation it may also expose public `operation_targets` and a coordinate-derived `spatial_state`; private executable candidates are intentionally absent from planner context.
- `recent_tool_calls`: recent tool-level actions.
- `stall_count` and `steps_since_last_decision`: local progress and decision cadence.
- `last_error`, `last_decision`, `last_trigger`, and `last_commit_label`: latest control state.
- `recovery_history`: recent recovery attempts.
- `manipulation_state`: runtime-owned per-arm grasp/attachment state. Unlike attempt-local recovery history, this physical state persists across replans. It distinguishes `holding_confirmed` from evidence-only `holding_provisional`, and separately reports whether transport is authorized. A placement release becomes `release_pending_verification` while target position, detachment, and stability are being verified. A detached, observed target miss becomes `release_recovery_required`.
- `grasp_transport_policy`: `strict` or `evidence_only`. Read it together with `manipulation_state`; do not infer confirmation from transport authority alone.
- `release_guard_enabled`: when `false`, an explicit `open_gripper` is model-authoritative after ordinary tool-argument validation. Release IDs are optional attribution metadata and placement observations are verified after execution. When `true`, the legacy held-instance/target/final-place release gate applies.
- `action_geometry_repair_pending_policy`: `strict`, `safe_motion`, or
  `disabled`. It controls only the guard for an instance whose
  `action_geometry_state` is `relocation_pending`.
- `semantic_tags`: compact tags for retrieval and recovery matching.

When `planner_context_mode=compact_v1`, `scene_memory.schema` is
`planner_scene_view/compact_v1`. It is the one planner-visible scene source:
raw `observation_preprocess`, repeated `recent_observation_summary`, and the
separate public `manipulation_state` copy are intentionally omitted. Read
`scene_memory.instances[*].operations` for public grasp/contact availability
and `scene_memory.manipulation_state` for the current attachment/release phase.
Complete masks, candidate poses, and attachment transforms remain private to
runtime. In `legacy` mode the older fields remain available.

### Monitor and recovery state
- `monitor`: rollout phase, status, active skill, progress score, stall count, and failure reason.
- `recovery`: retry, reset, and replan budgets and any pending recovery action.

## Usage Rules

- Treat structured runtime fields as the authoritative state source.
- Use raw images and observation summaries as evidence, but avoid reconstructing long histories from them.
- Use `scene_memory` for stable object identity and spatial grounding.
- Read each instance's `position_state` independently of `status`:
  `current_verified` permits the normal grounded contract, `memory_valid`
  permits only a safe approach before fresh verification, and
  `motion_uncertain` forbids use of historical coordinates.
- Treat `verified_roles` as interaction evidence. A later single semantic mask
  must not silently relabel an object whose physical role was already
  verified.
- Under repair policy `safe_motion`, only runtime-validated open-gripper
  retreat, a runtime-derived safe-height move, and fresh camera observation
  are allowed while action geometry is pending. Stored approach/grasp/contact
  poses remain quarantined, and descent/contact/close remain forbidden.
- Under repair policy `disabled`, the repair-pending guard alone is bypassed;
  all other enabled guards remain unchanged. Read `release_guard_enabled`
  separately instead of assuming the release gate is active.
- Use `monitor` and `recovery` to decide whether to continue, retry, recover, replan, or finish.
- Under `grasp_transport_policy=strict`, preserve the existing attachment-verification rule. Continue carry/place only after `manipulation_state` reports `holding_confirmed=true` and `transport_authorized=true`; otherwise preserve the pending grasp state without claiming attachment.
- Under `grasp_transport_policy=evidence_only`, `phase=holding_provisional` with `transport_authorized=true` permits carry/place using the exact close-time arm/object attachment. Preserve that capability across replans, but describe it as provisional and never as `holding_confirmed=true`.
- Under `evidence_only`, missing, ambiguous, or negative visual attachment evidence remains evidence for later reasoning; it does not by itself require a diagnostic motion, re-observation, automatic gripper opening, return to the grasp pose, or retreat. Do not replace an authorized carry/place subtask with one of those actions solely to obtain grasp confirmation.
- A planner may explicitly abandon an evidence-only provisional hypothesis by opening that gripper. Do not infer or insert that action automatically. With `release_guard_enabled=false`, runtime does not require matching release IDs or a same-batch final-place call before executing the open; use the prior runtime placement state to reason about whether opening is appropriate. With `release_guard_enabled=true`, preserve the legacy matching held-instance, public-target, final-place contract.
- If `manipulation_state` reports `holding_confirmed=true`, preserve that physical fact when selecting the next subtask. Continue carry/place with the same arm and object instead of restarting from open/approach/grasp.
- If it reports `release_pending_verification`, preserve the released object and do not start the next manipulation. Use reobservation only when the object is at target and just stability remains; request at most one bounded clearance when clearance is still false.
- If it reports `release_recovery_required`, the named instance has been released, is not held, and is observably outside the selected target tolerance. Reacquire that instance and replace it at a currently valid target instead of requesting more observation-only turns.
- Use `scene_memory.operation_targets[*].target_id` as the public placement choice. Never ask a planner to select a private `candidate_id`.
- A public target with `target_kind=reference_region` is runtime-computed from the grounded reference set. Select it when its `placement_relation` matches the requested destination; do not recompute its coordinate or replace it with a generic free-support target.
- For ordering or arrangement decisions, reason from `scene_memory.spatial_state` coordinates and relations rather than a previous planner's prose description.
- Keep committed memory short and only about facts that are already true.
- Do not convert all working-memory details into `memory_text`; only preserve what helps the next turn or executor.
- If perception evidence conflicts with previous memory, carry uncertainty instead of silently overwriting.

## Payload Contract

The control-turn payload should contain:

```json
{
  "global_task": "...",
  "trigger": "...",
  "agent_state": {},
  "available_skills": [],
  "memory_harness": {},
  "runtime_state_reasoning": {}
}
```

The planner should inspect `agent_state` first, then use the memory harness and available skills to produce the next control decision.
