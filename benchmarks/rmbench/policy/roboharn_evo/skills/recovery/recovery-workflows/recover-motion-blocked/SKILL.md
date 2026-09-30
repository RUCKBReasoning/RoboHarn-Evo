---
name: recover-motion-blocked
description: "Recovery workflow harness for blocked motion or local contact obstruction during VLA rollout."
---

# Recover Motion Blocked

## Overview

Use this workflow when `OOD_scenario` is `motion_blocked`, local contact blocks progress, or the robot appears unable to continue the current motion safely.

This skill is a recovery planning harness. It should generate a situation-specific plan using available runtime recovery tools and current observations, not enforce a fixed primitive sequence.

## Inputs (provide per recovery planning call)

- `monitor_signal`: usually `motion_blocked`.
- `current_subtask`: the subtask that encountered blocked motion.
- `observation_summary`: contact, obstruction, end-effector posture, target visibility.
- `recovery_history`: prior tool calls and outcomes in this episode.
- `available_tools`: runtime recovery tools exposed by the executor.
- `tool_budget`: maximum tool calls allowed for this recovery attempt.
- `post_recovery_options`: usually `retry`, `replan`, or `abort`.

## Available Recovery Tools

Commonly useful tools:

- `retreat_arm`: move away from the blocked or contact region.
- `lift_ee`: create vertical clearance.
- `contact_displace`: apply a very small bounded displacement when contact direction is known.
- `move_ee_to_grounded_instance`: move toward a VLM-selected grounded scene-memory instance when valid 3D evidence is available.
- `reobserve_scene`: refresh observation after motion.
- `safe_reset_posture`: return toward a conservative posture if local recovery is insufficient.

Use only runtime tools present in `available_tools`.

## Workflow Harness

### Step 0: Safety + Preconditions

- Prefer small clearance motions before larger resets.
- Do not continue the original VLA motion while blockage is unresolved.
- Avoid repeated motions that push into the same obstruction.

### Step 1: Assess current state

Determine whether the blockage is local and recoverable, whether vertical clearance is likely useful, and whether the current subtask remains valid.

### Step 2: Choose situation-specific tool calls

- Use `retreat_arm` when the end effector should back away from contact.
- Use `lift_ee` when vertical clearance may resolve the obstruction.
- Use `contact_displace` only for small contact release motions with a known safe direction.
- Use `move_ee_to_grounded_instance` only when `scene_memory` already contains the selected finite 3D instance; do not invent coordinates.
- Use `reobserve_scene` after any motion that changes the robot or scene state.
- Use `safe_reset_posture` if local retreat/lift is insufficient or unsafe.
- For pure recovery-control debugging, express a short action sequence with multiple tool calls or bounded `steps` rather than a single tiny command.

### Step 3: Decide next intent

Prefer `retry` if blockage is cleared and the subtask remains valid. Prefer `replan` if the scene or target relation changed. Use `abort` if recovery tools fail or safety is uncertain.

## Output Contract

Return JSON only:

```json
{
  "selected_workflow": "recover-motion-blocked",
  "tool_calls": [
    {
      "tool_name": "retreat_arm",
      "args": {"arm": "right"},
      "reason": "move away from the blocked contact region"
    },
    {
      "tool_name": "lift_ee",
      "args": {"arm": "right"},
      "reason": "create vertical clearance"
    },
    {
      "tool_name": "reobserve_scene",
      "args": {},
      "reason": "verify whether blockage is cleared"
    }
  ],
  "post_recovery_intent": "retry",
  "stop_condition": "blockage cleared, tool fails, safety becomes uncertain, or tool budget is exhausted"
}
```

Allowed `post_recovery_intent` values: `retry`, `replan`, `abort`.

## Reference Plans

Reference plans are examples only. Generate a situation-specific plan from the actual observation and recovery history.

### Local blockage

1. `retreat_arm`
2. `reobserve_scene`
3. `post_recovery_intent = retry`

### Contact with low clearance

1. `retreat_arm`
2. `lift_ee`
3. `reobserve_scene`
4. `post_recovery_intent = retry` if the path is clear; otherwise `replan`.

## Failure Handling

- If `retreat_arm` fails, do not continue pushing; consider `lift_ee` or `safe_reset_posture`.
- If all local clearance tools fail, return `post_recovery_intent = abort` or `replan` with the failure reason.
- If reobserve indicates the target moved or scene drifted, prefer `replan`.

## Notes

This workflow assumes monitoring has already identified a blockage. It should not classify OOD itself.
