---
name: recover-object-not-visible
description: "Recovery workflow harness for target or relevant object not visible during monitored rollout."
---

# Recover Object Not Visible

## Overview

Use this workflow when `OOD_scenario` is `object_not_visible`, or the target/relevant object cannot be reliably observed during the current subtask.

This skill is a recovery planning harness. It should generate a situation-specific plan that improves observability or returns control to planning. It does not define one mandatory primitive sequence.

## Inputs (provide per recovery planning call)

- `monitor_signal`: usually `object_not_visible`.
- `current_subtask`: the subtask that needs the missing object.
- `observation_summary`: visible objects, camera state, gripper/arm occlusion, target uncertainty.
- `recovery_history`: prior recovery tool calls and outcomes.
- `available_tools`: runtime recovery tools exposed by the executor.
- `tool_budget`: maximum recovery tool calls allowed.
- `post_recovery_options`: usually `retry`, `replan`, or `abort`.

## Available Recovery Tools

Commonly useful tools:

- `reobserve_scene`: refresh visual observation.
- `retreat_arm`: move the arm away if it may occlude the object.
- `lift_ee`: lift the end effector if it blocks the camera view or object.
- `move_ee_to_grounded_instance`: move toward an already grounded scene-memory instance only when it can improve observability safely.
- `safe_reset_posture`: move toward a conservative posture if local observability cannot be restored.

Use only runtime tools present in `available_tools`.

## Workflow Harness

### Step 0: Safety + Preconditions

- Do not execute grasp/place actions when the relevant object is not visible.
- Prefer observation refresh before physical motion when the robot is already safe.
- Avoid repeated identical reobserve calls unless an intermediate action changed the view or posture.

### Step 1: Assess current state

Determine whether the object is hidden by the arm/gripper, out of camera view, moved to an unexpected location, or the scene is too uncertain for retry.

### Step 2: Choose situation-specific tool calls

- Use `reobserve_scene` when the current image may be stale.
- Use `retreat_arm` or `lift_ee` when the robot may occlude the object.
- Use `move_ee_to_grounded_instance` only if the relevant instance is already selected in `scene_memory` and has finite 3D grounding.
- Use `safe_reset_posture` if a conservative posture is needed before replanning.
- For pure recovery-control debugging, express a short action sequence with multiple tool calls or bounded `steps` when a single reobserve is not enough.

### Step 3: Decide next intent

Prefer `retry` only if the object becomes visible and the current subtask remains valid. Prefer `replan` when the object remains missing or the scene has changed. Use `abort` only when observation/recovery fails or safety is uncertain.

## Output Contract

Return JSON only:

```json
{
  "selected_workflow": "recover-object-not-visible",
  "tool_calls": [
    {
      "tool_name": "reobserve_scene",
      "args": {},
      "reason": "refresh visual evidence before deciding recovery motion"
    },
    {
      "tool_name": "retreat_arm",
      "args": {"arm": "right"},
      "reason": "reduce arm occlusion of the target object"
    },
    {
      "tool_name": "reobserve_scene",
      "args": {},
      "reason": "check whether the target is visible after retreat"
    }
  ],
  "post_recovery_intent": "replan",
  "stop_condition": "target visible and reachable, target still not visible, tool failure, or tool budget exhausted"
}
```

Allowed `post_recovery_intent` values: `retry`, `replan`, `abort`.

## Reference Plans

Reference plans are examples only. Generate a situation-specific plan from the actual observation and recovery history.

### Refresh only

1. `reobserve_scene`
2. `post_recovery_intent = retry` if target becomes visible; otherwise `replan`.

### Occlusion by robot

1. `retreat_arm`
2. `reobserve_scene`
3. `post_recovery_intent = retry` if target is visible and reachable; otherwise `replan`.

## Failure Handling

- If observation remains unavailable after motion, prefer `replan`.
- If recovery motion fails, return `post_recovery_intent = abort` or `replan` with the failure reason.
- Do not continue the original subtask without visual confirmation of the relevant object.

## Notes

This workflow improves observability; it should not infer hidden benchmark labels or classify OOD itself.
