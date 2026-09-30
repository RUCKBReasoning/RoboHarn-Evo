---
name: task-level-recovery-control
description: "Recovery workflow harness for pure recovery-control debugging that advances the global instruction with grounded recovery tools."
signals:
  - task_level_recovery_control
---

# Task Level Recovery Control

## Overview

Use this workflow when `OOD_scenario` is `task_level_recovery_control`. This is not a new physical failure. It is a debug mode where recovery tools temporarily act as the low-level controller, so the plan should advance the global instruction using `scene_memory`, `robot_state`, and `recovery_history`.

This skill is a recovery planning harness. It should generate a short bounded tool sequence that makes observable task progress. It must not repeat the initial OOD clearance pattern unless the current observation contains new evidence of blockage.

## Inputs

当 `recovery_state.execution_evidence_enabled=false` 时，本文的验证要求、证据收集动作、待验证释放状态和证据阻止规则全部停用。根据最新图像、实际动作反馈和已有 HPK 继续规划。`holding_assumed` 记录已经执行的抓取命令及其几何关系；规划模型结合图像决定后续操作。每批动作结束后继续普通规划，打开夹爪立即清除该关联。

- `global_task`: original user instruction for the episode.
- `current_subtask`: active or debug subtask.
- `observation_summary`: current scene and robot state summary.
- `robot_state`: current end-effector poses and gripper states.
- `scene_memory`: structured instances, task focus, uncertainty, and summary.
- `observation_preprocess`: latest perception and grounding evidence.
- `recovery_history`: prior recovery tool calls and outcomes in this episode.
- `recovery_state.manipulation_state`: runtime-owned manipulation state, including confirmed or provisional transport authority, that persists across subtask replans even though attempt-local recovery history is reset.
- `grasp_transport_policy`: runtime grasp policy. `strict` requires attachment verification before transport; `evidence_only` permits an exact close-time provisional attachment to transport while retaining later visual results as evidence only.
- `release_guard_enabled`: `false` makes an explicit `open_gripper` model-authoritative after ordinary argument validation; `true` restores the legacy placement/holding release gate.
- `action_geometry_repair_pending_policy`: runtime action-geometry policy.
  `strict` blocks pending grounded geometry, `safe_motion` permits only
  runtime-validated non-contact clearance actions, and `disabled` bypasses
  only this one pending-geometry guard.
- `available_tools`: runtime recovery tools exposed by the executor.
- `tool_budget`: maximum tool calls allowed for this recovery attempt.
- `post_recovery_options`: usually `retry`, `replan`, or `abort`.
- `preferred_arm`: the top-level Agent's structured arm preference for task progress.
- `blocked_grounded_setups`: public instance/arm/point blocks plus aggregated private operation-candidate availability. Candidate IDs remain runtime-private; use `available_candidate_count`, `all_candidates_blocked`, and `runtime_will_select_next` to distinguish one failed pose from an exhausted public action.

When `recovery_state.planner_context_mode=compact_v1`, `scene_memory.schema` is
`planner_scene_view/compact_v1` and it is the only scene representation.
`observation_summary` and `observation_preprocess` are intentionally empty.
Each instance exposes public position/identity plus aggregated `operations`
availability; runtime keeps and selects the private executable pose candidate.
Read manipulation state from `scene_memory.manipulation_state` in this mode.

## Available Recovery Tools

Commonly useful tools:

- `move_ee_to_grounded_instance`: move toward a target or tool instance already selected in `scene_memory.task_focus`.
- `move_ee_to_pose`: move to an explicit bounded pose only when the pose is already available from validated runtime state.
- `open_gripper`: prepare to grasp or release.
- `close_gripper`: close around a portable grasp target, or establish/maintain a contact grip when the following action remains in `contact` mode.
- `contact_displace`: apply a small bounded displacement when the intended contact direction is known.
- `reobserve_scene`: refresh observation after a motion or gripper change.
- `retreat_arm` and `lift_ee`: use only when there is new evidence that local clearance is needed.

Use only runtime tools present in `available_tools`.

## Workflow Harness

### Step 0: Interpret Control Mode

- Treat `task_level_recovery_control` as task execution through tools, not as an OOD classifier.
- Do not plan cleanup or another task action after RMBench `eval_success=true`. Runtime treats that signal as terminal, stops all remaining physical calls (including release, clearance, and re-observation), and commits global completion; `check_success` alone is not sufficient in pure-control mode.
- Read `global_task`, `current_subtask`, and `scene_memory.task_focus` before selecting tools.
- Prefer scene-memory-selected instances over reinterpreting object identity from text.
- Read `robot_state.<arm>.gripper` before any approach, grasp, or placement action. A gripper value near `0.0` means closed; a value near `1.0` means open.
- Treat arm names in reference plans as examples, not defaults. Select an arm from current grounded geometry, observed end-effector poses, gripper state, and recent action effects.
- Return that task-progress choice in `selected_arm`, and use the same arm in every task-progress tool call. Respect `preferred_arm` when it is `left` or `right`; do not recover arm choice from natural-language subtask wording.
- Set `selected_arm` to `none` for re-observation or clearance-only plans. A clearance-only retreat may move a non-preferred arm that occupies the target region.
- Read `grasp_transport_policy` together with `recovery_state.manipulation_state`:
  - Under `strict`, preserve the existing attachment-verification contract. A portable grasp remains pending until runtime evidence confirms it, and runtime may request a bounded verification motion or observation before granting transport.
  - Under `evidence_only`, `phase=holding_provisional` together with `transport_authorized=true` is sufficient to continue carry/place using the exact close-time held-object-to-TCP attachment. This is provisional authority; it does not mean `holding_confirmed=true` and must never be described as a confirmed grasp.
  - Under `evidence_only`, missing, ambiguous, or negative visual attachment evidence is recorded for reasoning but does not by itself require a diagnostic motion, stationary re-observation, automatic gripper opening, return to the grasp pose, or retreat. Continue the requested task action unless a separate executor or physical-safety result blocks that action.
  - An explicit planner `open_gripper` may abandon a provisional grasp hypothesis. This is a deliberate action, not an automatic response to uncertain visual evidence. When `release_guard_enabled=false`, release IDs and a same-batch final-place call are not execution preconditions; when it is `true`, follow the legacy exact held-instance/public-target/final-place contract.
  - If the released instance has `action_geometry_state=relocation_pending`,
    obey `action_geometry_repair_pending_policy`. In `safe_motion`, request
    only `reobserve_scene` or a bounded open-gripper retreat; runtime may
    replace a grounded request with its own lift-first safe-height move.
    Never request direct descent, contact, close, or reuse of the quarantined
    approach/grasp/contact pose. In `disabled`, remember that every unrelated
    guard remains active.

### Step 1: Check Prior Recovery History

- If recent history repeatedly used `retreat_arm`, `lift_ee`, and `reobserve_scene` without task progress, do not repeat that pattern.
- If a previous grounded move failed, choose `reobserve_scene`, a different selected instance, or `replan`.
- For a legacy blocked instance/arm/point entry, or an operation scope with `all_candidates_blocked=true`, do not retry the identical public action. When `runtime_will_select_next=true`, retrying that public instance/arm/action_mode is allowed because runtime will select another unblocked private pose candidate. Never request or copy a private candidate ID.
- In a dual-arm scene, if two observed effects show that the same arm did not move toward a still-grounded instance, do not keep replanning the identical same-arm attempt. When the other arm is free, open, and not holding an object, use it for one bounded approach probe and reobserve before any grasp. Do not switch arms when the other arm is holding an object or current evidence makes the alternate approach unsafe.
- Before switching arms toward the same grounded target, require the previous arm to retreat outside the target envelope and reobserve in a separate round. Never converge both end effectors on one target while the previous arm remains nearby; runtime may block this as unsafe target occupancy.
- Check grounding for the tool you intend to use, not as a prerequisite for every recovery plan. Robot-state-only clearance (such as a bounded `lift_ee` or `retreat_arm`) does not require an object binding. An object-grounded action still requires a resolvable instance and valid geometry; if those are missing, use an available observation tool or return `replan`. Do not invent an object query solely to permit robot-only motion.
- Read recent `action_effect:` entries before planning.
- If `action_effect` says `effect=false`, do not repeat the same tool/instance/point_key without changing perception, pose, or strategy.
- If `action_effect` says `effect=unverified` after `close_gripper`, inspect `grasp_transport_policy` and structured manipulation state. Under `strict`, do not carry/place until attachment verification grants authority. Under `evidence_only`, continue when the exact arm state reports `phase=holding_provisional` and `transport_authorized=true`; lack of confirmation alone is not a failure.
- If `action_effect` says `effect=true`, continue to the next task-relevant action instead of repeating the verified action.
- If `recovery_state.manipulation_state` reports `holding_confirmed=true`, treat the named arm/object attachment as authoritative across replans. Continue the carry/place phase; do not open and re-grasp the same object.
- If `grasp_transport_policy=evidence_only` and manipulation state reports `phase=holding_provisional` with `transport_authorized=true`, continue the named arm/object carry or placement without calling the grasp confirmed. Later visual evidence may update confidence, but uncertainty alone must not pause same-arm motion.
- If manipulation state reports `phase=release_pending_verification`, the gripper has already opened but placement evidence is incomplete. Do not touch or re-grasp the object while it is at target. When `runtime_place_validation.object_at_target=true`, detachment and EE clearance are true, and only stability is missing, use `reobserve_scene` only; any additional physical action can disturb an otherwise valid placement. If clearance is false, perform at most one bounded clearance before reobserving.
- If manipulation state reports `phase=release_recovery_required`, the named released instance is detached and observably outside its selected target tolerance. It is not held. Reacquire that exact instance with a normal bounded approach/grasp verification sequence, then choose and revalidate a current placement target; do not emit another observation-only release-verification plan.
- For count-sensitive transient contact, execute at most one complete actuation cycle for the active subtask, then return control to the after-action verifier. A complete press/tap cycle is one inward `contact_displace` followed by release to the same instance's grounded `approach_world_m` before `reobserve_scene`. Runtime may then add a separate `post_contact_clearance=camera_tangent` grounded move, computed from live camera/contact geometry, before that re-observation. Release and clearance are cleanup for the same event, never additional repetitions. Internal displacement `steps` are controller interpolation, not additional task-level repetitions.
- If recent history already closed the same gripper and there is no runtime manipulation state for that grasp, do not close it again blindly. Both `holding_confirmed=true` and evidence-only `holding_provisional` with `transport_authorized=true` are explicit runtime states; do not discard either merely because a current camera cannot see the object.
- If the end effector is still above or offset from the selected instance, continue bounded approach and reobserve. Do not close the gripper merely because the correct instance has been selected.
- Treat `approach_world_m` as a safe pre-contact point, not as a grasp point.

### Step 2: Choose Bounded Task-Progress Tools

- Use `move_ee_to_grounded_instance` with `role = "tool"` when the next action should approach a manipulable tool selected by `scene_memory.task_focus`.
- Use `move_ee_to_grounded_instance` with `role = "target"` when the next action should approach the task target selected by `scene_memory.task_focus`.
- When `scene_memory.task_focus.identity_binding_required=true`, or the selected focus contains multiple instances, every grounded move must name the exact destination/manipuland in `instance_id`; `role` or `focus_key` alone is invalid.
- Use `point_key = "approach_world_m"` for approach motions before contact or grasp.
- Use `point_key = "grasp_world_m"` before a grasp/close action, or `point_key = "contact_world_m"` before a deliberate contact/push action.
- Choose the operation mode from the physical intent, not the object name. Use `grasp` only when the object is expected to become a portable rigid attachment and later be carried or placed. Use `contact` for pressing, pushing, bracing, or actuating an anchored/articulated mechanism, including a handle that is held closed while it is pulled. A `contact`-mode close never creates held-object or transport authority.
- For approach and grasp points with a finite grounded quaternion, use `target_quat_wxyz = "grounded"`; do not preserve an unrelated home or prior-action orientation.
- Use small `max_translation` and bounded `steps` so each recovery call remains inspectable.
- When `recovery_state.local_grounded_goal_control=true`, a grounded move follows its fixed target using measured robot feedback for up to 20 internal steps and the remaining episode budget. These steps do not call the VLM or SAM3. Read `target_reached` and `local_goal_stop_reason` before requesting another move; relative clearance distances are not extended. Without that capability, the requested `steps` remains the bound.
- Treat `executed_pose` as the commanded pose, not proof that the robot reached it. Read `observed_pose`, `target_observation_error_m`, and fresh visual evidence before claiming alignment or contact.
- A successful bounded pose command may still return `target_reached=false`. Runtime halts the remaining static batch in that case, skips dependent contact/gripper actions, and preserves the new observation for the next reasoning round. Continue the approach from the new pose; do not treat a skipped action as attempted contact.
- Scene Memory may expose multiple private operation-pose candidates for the same public instance or place target. Never put a `candidate_id` in public tool args. For grasp/contact select the public instance; for placement select one `scene_memory.operation_targets[*].target_id`. Runtime reports the selected private ID and, after repeated no-progress, blocks only that candidate.
- Use `open_gripper` or `close_gripper` only when the task state and robot state justify a gripper change.
- With `release_guard_enabled=false`, an explicit open of a transport-authorized arm executes without matching `release_held_instance_id`, `release_target_id`, or a same-batch `place_world_m`; those IDs are optional attribution metadata. Use runtime state and task intent to decide when to issue it, and do not present the command itself as successful placement. With `release_guard_enabled=true`, matching held-instance/target/final-place evidence is required by the legacy gate. Never insert an open automatically from visual uncertainty.
- Use `reobserve_scene` after motion or gripper actions only when a fresh observation is useful for the next decision. Under `evidence_only`, do not add it solely because a portable grasp lacks attachment confirmation.
- If the intended manipulation mode is grasping, explicitly call `open_gripper` before approach when the selected arm's gripper is closed and there is no evidence that it is already holding the intended object.
- For every new portable-object grasp attempt, do not request `grasp_world_m` directly from an arbitrary or unrelated end-effector pose. Unless current recovery history shows that the same arm has already reached the current `approach_world_m` for the exact same instance and grasp attempt, first call `move_ee_to_grounded_instance` with that arm, exact instance, `action_mode = "grasp"`, and `point_key = "approach_world_m"`.
- Prefer one ordered static batch for a new grasp: move the same arm and exact instance to `approach_world_m`, then to `grasp_world_m`, then call `close_gripper`. Runtime halts the remaining batch when a bounded pose target is not reached, so do not omit the approach merely to save a control turn. A separate re-observation between these calls is optional, not mandatory.
- The selected candidate's approach is its geometry-derived safe pre-contact pose. It may be above, beside, or otherwise offset from the object along the candidate's ingress direction; never replace it with a fixed world-axis offset. The planner names the same public instance and arm for approach and grasp and must not invent a private candidate ID.
- Only call `close_gripper` after a bounded move to `grasp_world_m` or another explicit grasp/contact pose and the latest observation supports the grasp stage. Do not close immediately after an `approach_world_m` move.
- If the empty gripper itself is the contact tool for a deliberate push, press, or brace, choose a mechanically stable contact shape from current geometry. When separated fingers would create split or unstable contact, require a compact closed gripper by adding `"gripper_precondition": "closed"` to the grounded contact move or `contact_displace`; do not rely on the current gripper state being preserved. Keep an open gripper only when current geometry provides evidence that it is the safer contact shape.
- If a tool call requires a specific gripper state before execution, include `gripper_precondition` in that tool call's args using `"open"` or `"closed"`. Runtime may satisfy explicit preconditions but should not infer manipulation intent from object names.
- A move to `contact_world_m` already commands the grounded full-contact EE pose. Do not follow it with a large displacement farther along the approach-to-contact direction; that is overtravel and can produce collision-driven lateral slip. Runtime requires the observed EE position to be within contact tolerance and caps same-direction overtravel. Use a pre-contact/standoff pose for a longer approach, or request only a small compliant displacement after the full-contact pose and then reobserve.
- Runtime may rewrite a full-contact-plus-inward-displacement sequence to stage from a displacement-matched pre-contact offset plus a small generic compliance margin. The bounded relative displacement then enters the grounded contact region without first commanding geometric overpenetration.
- For a press/tap, set `"complete_transient_cycle": true` on the inward `contact_displace`. Runtime requires the same-batch grounded contact setup, inserts a same-arm return to that instance's `approach_world_m`, and—only when valid current geometry permits it—adds a separate camera-tangent grounded clearance before re-observation. The direction is derived from the camera ray, contact normal, target 3D extent, and current robot pose; it is not a fixed world axis or a task-text rule. Never finish a transient press batch while the end effector remains at the depressed/contact pose or directly over the target view.
- Before a test/confirmation press after rearranging objects, inspect `scene_memory.spatial_state`. Its current coordinates, x/y orders, overlap/support relations, and signature are runtime-derived evidence. Do not press because planner prose or prior memory claims an arrangement is correct when the actual coordinate-derived state disagrees.
- Treat `reobserve_scene` as the end of an inspectable plan segment. Runtime truncates the batch there. Do not place actions that depend on its new observation later in the same static tool batch; return `retry` and reason again from the refreshed state.
- After an unverified contact attempt, do not repeat the same grounded pose, direction, and distance. Use observed pose error and fresh geometry to change alignment or return `replan`.
- Under `strict`, after a portable `grasp`-mode close, runtime may make one bounded verification motion back along the candidate's validated ingress path before authorizing carry or placement. This direction comes from `approach_pose - grasp_pose`; it is not a fixed upward lift. When strict manipulation state explicitly says this verification is pending, request `lift_ee` only as the existing public compatibility trigger; do not choose a world axis or target pose.
- Under `evidence_only`, do not request or insert that verification motion merely because the grasp is unconfirmed. If the exact provisional attachment is transport-authorized, proceed with the task's ordinary carry/place motion. For `contact` mode under either policy, keep the interaction in the bounded contact sequence and do not invoke portable-grasp verification.
- While a grasped object is being carried, do not use another object's `approach_world_m`, `grasp_world_m`, or `contact_world_m` as a placement pose. Those poses describe manipulation of that other object and do not encode the held-object-to-TCP transform.
- When `scene_memory.operation_targets` is available, choose the smallest task-sufficient public target: a `reference_region` whose `placement_relation` exactly matches an explicitly requested relational destination, an unoccupied `vacated_pose` for a known prior support pose, a `free_support` target for temporary buffering/rearrangement, or an `object_top` target only when stacking/support-on-object is intended. Do not substitute `free_support` when a verified matching `reference_region` exists, and do not choose `object_top` merely because it exists.
- If dynamic operation targets are unavailable and the compatibility carry-alignment mode is used, it still requires the exact destination `instance_id` and exact `held_instance_id`; this mode aligns at clearance height and is not itself a placement or release signal.
- For a selected public target, use `action_mode = "place"` and the same `target_id`. Move first to `point_key = "approach_world_m"` and reobserve if alignment/attachment is uncertain; then use `point_key = "place_world_m"` for the final target. Runtime privately chooses the held arm's candidate and maps the desired object pose through the current held-object-to-TCP transform. Do not add `preserve_height` or invent a downward displacement.
- Prefer to reach the intended final place before release. With `release_guard_enabled=false`, the final place and open may occur in separate control turns and release IDs are optional; runtime observes the actual outcome after opening. With `release_guard_enabled=true`, use matching release IDs under the legacy gate. Placement remains pending until the object is detached and stable across fresh observations.
- For uncover, remove, or move-aside subtasks, invert the placement logic: after grasp/lift is verified, command a bounded horizontal displacement away from the source instance using current geometry. Do not lower or release while the held object is still laterally over the source. Reobserve and require measurable horizontal source-object separation before opening the gripper; choose the horizontal axis and sign from the live scene rather than a fixed task-specific direction.
- A verified grasp or small coupled lift does not by itself prove vertical clearance for horizontal carry. When finite 3D extents are available, require the held object's lower z bound to be above the source object's upper z bound with a small positive margin before moving laterally. If this clearance is absent or uncertain, continue a bounded lift and reobserve while checking that the source instance stays fixed; do not drag both instances together.
- Measurable displacement alone is not enough to prove move-aside clearance. When both instances have finite horizontal extents, require their horizontal AABBs to be non-overlapping on at least one axis, with a small positive margin, before release. If extents are unavailable or identity is uncertain, continue bounded carry/reobserve instead of claiming uncover completion.
- When choosing a clear placement for a removed or temporarily displaced object, prefer an observed, unoccupied, support-valid prior pose over an invented coordinate. `first_observed_world_m` is immutable evidence of where that tracked instance was first seen, not an automatic command: use it only when the task/history and current geometry make returning there appropriate and safe. Move toward it with bounded relative displacements computed from live object geometry, preserve vertical clearance during lateral motion, then lower and verify support before release.
- `contact_displace` executes only its structured args; prose such as "lateral" does not change runtime defaults. Always provide explicit `arm`, `axis`, `direction`, `distance`, and `steps`. For move-aside, `axis` must be `x` or `y` and `direction` must be `positive` or `negative`; never omit them, because omitted values default to a downward z displacement.

### Step 3: Decide Next Intent

Prefer `retry` when another pure recovery-control round should continue task execution. Prefer `replan` when scene memory is missing, contradictory, or the next step needs semantic replanning. Use `abort` only if the state is unsafe or tools fail repeatedly.

## Output Contract

Return JSON only:

```json
{
  "selected_workflow": "task-level-recovery-control",
  "selected_arm": "left",
  "tool_calls": [
    {
      "tool_name": "move_ee_to_grounded_instance",
      "args": {
        "arm": "left",
        "role": "tool",
        "point_key": "approach_world_m",
        "target_quat_wxyz": "grounded",
        "max_translation": 0.025,
        "steps": 2
      },
      "reason": "perform a bounded approach toward the scene-memory-selected instance; do not close the gripper during early approach"
    },
    {
      "tool_name": "reobserve_scene",
      "args": {},
      "reason": "refresh scene memory after the bounded move"
    }
  ],
  "post_recovery_intent": "retry",
  "stop_condition": "task progress is observed, scene memory becomes invalid, a tool fails, or tool budget is exhausted"
}
```

When the observation shows the end effector is already at the approach point and grasping is intended, use a separate grasp-stage move before closing. The following strict-mode example requests attachment observation:

```json
{
  "selected_workflow": "task-level-recovery-control",
  "tool_calls": [
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
      "reason": "move from safe approach height to a plausible grasp point"
    },
    {
      "tool_name": "close_gripper",
      "args": {"arm": "left"},
      "reason": "close only after reaching the explicit grasp point"
    },
    {
      "tool_name": "reobserve_scene",
      "args": {},
      "reason": "verify whether the tool moved with the gripper"
    }
  ],
  "post_recovery_intent": "retry",
  "stop_condition": "grasp is confirmed, scene memory becomes invalid, or tool budget is exhausted"
}
```

Under `evidence_only`, do not copy the final `reobserve_scene` merely to verify
attachment. End the batch after the close, or continue with an ordinary
task-required same-arm action when its runtime grounding is already available.
On the next control turn, a state with `phase=holding_provisional` and
`transport_authorized=true` permits carry/place without upgrading it to
`holding_confirmed=true`.

Allowed `post_recovery_intent` values: `retry`, `replan`, `abort`.

## Failure Handling

- If the intended object-grounded action has no resolvable instance, do not invent coordinates; use an available observation tool or return `replan`. Missing object binding does not prohibit robot-state-only clearance.
- If repeated bounded moves do not reduce distance to the selected instance, return `replan`.
- If repeated no-motion evidence is specific to one arm and a free alternate arm exists, change the arm as the strategy before returning the same subtask to an identical failed plan.
- If a move-aside action changes only height and does not increase horizontal separation from the source instance, keep the grasp closed and change to a bounded horizontal strategy before release.
- If collision or blockage evidence appears, route future calls through a true OOD signal such as `motion_blocked`.
