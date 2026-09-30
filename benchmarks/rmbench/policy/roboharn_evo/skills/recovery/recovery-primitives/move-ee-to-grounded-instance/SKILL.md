---
name: move-ee-to-grounded-instance
description: "Recovery primitive tool guide for bounded end-effector motion to a scene-memory instance with 3D grounding."
---

# Move End Effector To Grounded Instance

## Overview

Use this primitive when recovery should move an end effector toward grounded manipulation geometry. For grasp/contact, the planner chooses an instance. For placement, the planner chooses a public `scene_memory.operation_targets[*].target_id`; runtime binds the transport-authorized held object and resolves the arm-specific executable pose. Private candidate identity never enters planner tool args.

## Inputs

- `scene_memory`: structured scene instances, task focus, uncertainty, and summary. Instances may carry `operation_pose_candidates`; the planner must not invent or copy a private `candidate_id` into tool args.
- `scene_memory.operation_targets`: compact public placement choices. Each entry has a stable `target_id`, target kind, support/occupancy state, object target coordinate, and executable arm options. It intentionally omits private arm-specific poses.
- `observation_preprocess`: latest perception and segmentation evidence.
- `robot_state`: current end-effector pose and gripper state.
- `current_subtask`: interrupted subtask.
- `recovery_history`: prior recovery tool calls and outcomes.
- `available_tools`: runtime tool allowlist.
- `grasp_transport_policy`: `strict` requires a confirmed attachment for carry/place; `evidence_only` may expose an exact close-time provisional attachment as transport-authorized without claiming confirmation.

## Runtime Tool

- `tool_name`: `move_ee_to_grounded_instance`
- `args_schema`:
  - `arm` (required string): `left` or `right`.
  - `instance_id` (optional string): exact `scene_memory.instances[*].instance_id`.
  - `instance_ref` (optional string): alias for `instance_id`.
  - `action_mode` (optional string): `grasp`, `contact`, or `place`. Set it explicitly to `place` for placement targets.
  - `target_id` (required for place string): exact public `scene_memory.operation_targets[*].target_id`. This is a target group, not a low-level pose candidate.
  - `role` (optional string): `target` or `tool`; runtime may use `scene_memory.task_focus.target_instances` or `tool_instances` only when that focus is unique and identity binding is not required.
  - `focus_key` (optional string): explicit scene-memory focus list such as `target_instances` or `tool_instances`; it cannot replace an exact instance ID when `identity_binding_required=true`.
  - `point_key` (optional string): `approach_world_m`, `grasp_world_m`, `contact_world_m`, `place_world_m`, `top_surface_world_m`, or `world_m`; aliases such as `approach`, `grasp`, `contact`, `place`, `top_surface`, and `centroid` are accepted.
  - `offset_xyz` (optional list): small 3D world-frame offset in meters.
  - `preserve_height` (optional boolean): when `true`, enter carry mode. Runtime resolves the held tool instance, raises it first when its current bottom surface does not clear the target top surface, then uses the grounded target for horizontal x/y alignment while preserving the resulting end-effector z plus `offset_xyz[2]`.
  - `held_instance_id` (optional string): exact held `scene_memory` instance. If omitted in carry mode, runtime uses `scene_memory.task_focus.tool_instances`.
  - `held_role` / `held_focus_key` (optional string): alternate held-instance focus selection. Defaults to `tool` / `tool_instances`.
  - `clearance_margin` (optional number): required vertical margin between the held object's bottom and destination's top before lateral carry. Runtime uses a conservative default and cap.
  - `max_clearance_lift` (optional number): maximum pre-carry lift that runtime may apply. The call fails instead of carrying through a collision when required clearance exceeds this bound.
  - `clearance_steps` (optional integer): bounded control steps for the pre-carry vertical lift.
  - `gripper_precondition` (optional string): explicit requested gripper state such as `open`; runtime may satisfy this precondition before executing the move.
  - `target_quat_wxyz` (optional list or string): 4D quaternion, `grounded`, or `preserve`. If omitted, runtime uses the matching grounded approach/grasp/contact quaternion when one exists and otherwise preserves the current orientation.
  - `max_translation` (optional number): maximum translation in meters for each internal recovery step. Runtime caps this conservatively.
  - `steps` (optional integer): number of bounded internal control steps. Default is 1; runtime caps this conservatively.
- `result_schema`: `RecoveryToolResult`

## Use When

- `scene_memory.instances` contains a visible or recently tracked instance with finite `world_m` or `approach_world_m`.
- The selected instance is named in `instance_id`, or the intended target/tool focus is already present in `scene_memory.task_focus`.
- If `scene_memory.task_focus.identity_binding_required=true`, the selected instance is named by exact `instance_id` or `instance_ref`; do not rely on `role` or `focus_key`.
- A short bounded move toward the grounded point can support retry, release, reobserve, grasp setup, or local contact recovery.
- The manipulation mode is clear enough to choose whether any gripper precondition is needed. Do not assume all grounded approaches are grasps.
- Use `grasp` only for a portable object that should become a rigid held attachment for later carry/place. Use `contact` for a press, push, brace, or anchored/articulated mechanism—even when the gripper closes around a handle before `contact_displace`. Object names do not determine this mode.
- A runtime transport-authorized held object needs horizontal alignment over a grounded destination while the end effector maintains its current clearance height.
- A runtime transport-authorized held object has a public free-support, vacated-pose, or object-top target and needs a transform-correct approach/final placement motion.
- Under `strict`, transport authority implies verified attachment. Under `evidence_only`, `phase=holding_provisional` and `transport_authorized=true` may provide the same bounded carry/place capability from the exact close-time attachment, but this must not be reported as `holding_confirmed=true`.
- An occluded instance has `position_state=memory_valid` and verified action
  geometry. Its safe `approach_world_m` remains executable. An exact zero-offset
  final `grasp_world_m`/`contact_world_m` may also be requested; runtime executes
  it only when the memory-valid-final-action switch is enabled and the same
  arm/private candidate remains bound. This never authorizes gripper closure,
  attachment, transport, or placement by itself.

## Do Not Use When

- `scene_memory` is empty or lacks finite 3D points for the desired instance.
- You would need to guess which instance is the target from task semantics not present in `scene_memory`.
- You would need to invent coordinates.
- The selected instance has `position_state=motion_uncertain`.
- The selected instance is `memory_valid` but action geometry is unavailable,
  pending repair, candidate/arm identity is mismatched, or the final request
  adds an offset/preserve-height override.
- A relative clearance primitive is safer.

## Tool Call Contract

Use an explicit instance when available:

```json
{
  "tool_name": "move_ee_to_grounded_instance",
  "args": {
    "arm": "left",
    "instance_id": "object_left",
    "point_key": "approach_world_m",
    "offset_xyz": [0.0, 0.0, 0.02],
    "target_quat_wxyz": "grounded",
    "max_translation": 0.025,
    "steps": 2
  },
  "reason": "move toward the stored approach point for the VLM-selected grounded instance"
}
```

Use a grasp/contact point, not an approach point, before closing the gripper:

```json
{
  "tool_name": "move_ee_to_grounded_instance",
  "args": {
    "arm": "left",
    "role": "tool",
    "point_key": "grasp_world_m",
    "target_quat_wxyz": "grounded",
    "max_translation": 0.015,
    "steps": 2
  },
  "reason": "move from the safe approach height to a plausible grasp point before closing the gripper"
}
```

For an anchored or articulated interaction, keep the operation in `contact`
mode even if the next step closes the gripper around the mechanism:

```json
{
  "tool_name": "move_ee_to_grounded_instance",
  "args": {
    "arm": "left",
    "instance_id": "mechanism_01",
    "action_mode": "contact",
    "point_key": "contact_world_m",
    "target_quat_wxyz": "grounded",
    "max_translation": 0.015,
    "steps": 2
  },
  "reason": "establish bounded articulated contact without creating portable-object transport state"
}
```

Use task focus only when `scene_memory.task_focus` already selected it:

```json
{
  "tool_name": "move_ee_to_grounded_instance",
  "args": {
    "arm": "left",
    "role": "target",
    "point_key": "approach",
    "max_translation": 0.025
  },
  "reason": "use the current scene-memory target focus instead of inventing coordinates"
}
```

Do not use this focus-only form when `identity_binding_required=true` or when the
selected focus contains more than one instance. Use the exact advertised
`instance_id` in those cases.

When carrying a runtime transport-authorized held object, align laterally without using the destination object's grasp height or orientation as a placement pose:

```json
{
  "tool_name": "move_ee_to_grounded_instance",
  "args": {
    "arm": "left",
    "instance_id": "destination_01",
    "held_instance_id": "object_left",
    "point_key": "world_m",
    "preserve_height": true,
    "target_quat_wxyz": "preserve",
    "max_translation": 0.04,
    "steps": 3
  },
  "reason": "align the held object over the selected destination while maintaining vertical clearance"
}
```

Prefer the dynamic placement contract when `scene_memory.operation_targets` is
available. First move to the target group's clearance pose:

```json
{
  "tool_name": "move_ee_to_grounded_instance",
  "args": {
    "arm": "left",
    "action_mode": "place",
    "target_id": "place:held:track_0001:vacated:track_0001",
    "point_key": "approach_world_m",
    "max_translation": 0.12,
    "steps": 3
  },
  "reason": "carry the held object to the runtime-computed clearance pose for the selected public target"
}
```

After runtime still exposes the same target and transport-authorized attachment,
command its final place pose. Under `evidence_only`, attachment confirmation is
not an additional prerequisite for this call:

```json
{
  "tool_name": "move_ee_to_grounded_instance",
  "args": {
    "arm": "left",
    "action_mode": "place",
    "target_id": "place:held:track_0001:vacated:track_0001",
    "point_key": "place_world_m",
    "max_translation": 0.06,
    "steps": 2
  },
  "reason": "lower the held object using the selected target's transform-correct final pose"
}
```

Use `object_top` targets only when the instruction actually requires support by
another object. For rearrangement or temporary buffering, prefer an unoccupied
`vacated_pose` or `free_support` target.

When the planner explicitly intends a grasp and the current gripper is closed, request the precondition instead of relying on runtime inference:

```json
{
  "tool_name": "move_ee_to_grounded_instance",
  "args": {
    "arm": "left",
    "role": "tool",
    "point_key": "approach_world_m",
    "gripper_precondition": "open",
    "max_translation": 0.025
  },
  "reason": "approach a grasp target with the gripper opened first"
}
```

## Expected Result

Runtime resolves the selected instance from internal scene-memory context, validates the 3D point and grounded orientation, clamps each internal step, preserves gripper state, and executes one or more bounded end-effector commands.

For `approach_world_m`, `grasp_world_m`, `contact_world_m`, and `place_world_m`, runtime prefers a structured candidate matching the requested arm/action mode. The result reports `operation_candidate_id`, `operation_target_id`, `operation_action_mode`, and geometry source for audit. Legacy single-point fields remain a compatibility fallback only for non-place modes. The planner never selects candidate IDs.

For place mode, runtime computes the current held-object-to-TCP relation from
fresh object and robot geometry, preserves that rigid attachment, and converts
the desired object target into an EE command. It does not move the EE to the
target center. Runtime recomputes target occupancy and support validity
immediately before execution; stale or occupied targets fail closed.

`strict` and `evidence_only` differ only in how the attachment gains transport
authority. Under `strict`, runtime verifies attachment before carry/place.
Under `evidence_only`, an exact close-time attachment may remain
`holding_provisional` with `transport_authorized=true`; the primitive may use it
for carry/place without calling it confirmed. Missing or ambiguous visual
evidence must not force this primitive to insert diagnostic motion,
re-observation, gripper opening, return-to-grasp, or retreat. Those actions may
still be requested explicitly for a separate task or safety reason.

With `preserve_height=true`, runtime validates a closed gripper, resolves the held and destination instances, computes clearance from their current geometry, and performs any required vertical lift before lateral motion. It then ignores the grounded destination's z coordinate, keeps the clearance-safe end-effector z (plus an optional z offset), and grounds x/y in the destination. Missing geometry, an open gripper, or excessive required lift fails safely. This is a carry-alignment mode, not a release or placement-success signal.

For every new portable-object `grasp` attempt, `approach_world_m` is the mandatory safe staging pose. Do not request `grasp_world_m` directly from an arbitrary or unrelated end-effector pose. Unless current recovery history shows that the same arm already reached the current approach for the exact same instance and grasp attempt, request the ordered sequence `approach_world_m` -> `grasp_world_m` -> `close_gripper`. These calls may be returned in one static batch; runtime stops the later calls if an earlier bounded target is not reached, and no intervening re-observation is required by this primitive. The candidate-defined approach follows its ingress geometry and may be above or beside the object; it is not a fixed world-axis offset. For deliberate `contact` interactions, use the separately grounded contact workflow rather than treating an articulated or anchored mechanism as a portable grasp.

Visibility and position validity are separate. A missing SAM mask does not
invalidate a prior coordinate. Runtime allows `memory_valid` only for the safe
approach stage; current visual/multiview evidence remains required for the
final operation unless the approach itself creates a geometrically proven,
one-shot robot self-occlusion lease.

## Failure Handling

- If the instance cannot be resolved, prefer `reobserve_scene` or `replan`.
- If `position_state=motion_uncertain`, reacquire/replan; do not use the
  retained historical coordinate.
- If identity binding is required, provide the exact public scene-instance ID. In carry mode, provide both the destination as `instance_id` and the held object as `held_instance_id`.
- In place mode, provide the exact public `target_id`, not a destination `instance_id`; runtime binds the exact held instance from transport-authorized manipulation state. Under `evidence_only`, provisional authority does not relax target occupancy, support, identity, or release validation.
- If 3D grounding is missing, do not fall back to guessed coordinates.
- Do not request a private candidate ID. If one candidate is blocked after repeated no-progress, a retry of the same public instance/arm/mode call selects the next unblocked candidate. If no candidate remains, change arm, refresh geometry, or replan.
