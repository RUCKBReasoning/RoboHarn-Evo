---
name: retreat-and-reobserve
description: "Recovery workflow harness for local stall that retreats the arm and refreshes scene observation."
---

# Retreat And Reobserve

## Overview

Use this workflow when monitored rollout is locally stalled but the current subtask may still be retryable after a safe retreat and observation refresh.

This skill is a recovery planning harness. It should choose situation-specific tool calls based on the current robot posture, scene state, and recovery history.

## Inputs (provide per recovery planning call)

- `monitor_signal`: usually `stall_detected`.
- `current_subtask`: the stalled subtask.
- `observation_summary`: robot posture, local contact, target visibility, and scene state.
- `recovery_history`: prior recovery tool calls and outcomes.
- `available_tools`: runtime recovery tools exposed by the executor.
- `tool_budget`: maximum recovery tool calls allowed.
- `post_recovery_options`: usually `retry`, `replan`, or `abort`.

## Available Recovery Tools

Commonly useful tools:

- `retreat_arm`: create clearance from local contact or a stuck posture.
- `lift_ee`: create vertical clearance if needed.
- `contact_displace`: apply a very small bounded contact release when direction is known.
- `move_ee_to_grounded_instance`: move toward a VLM-selected grounded scene-memory instance when the current subtask needs a short bounded reposition.
- `reobserve_scene`: refresh observation after movement.
- `safe_reset_posture`: use only when local retreat is insufficient or unsafe.

Use only runtime tools present in `available_tools`.

## Workflow Harness

### Step 0: Safety + Preconditions

- Do not continue the same blocked motion without changing posture or refreshing observation.
- Prefer small local retreat before broader reset posture.
- Keep the recovery short; stall recovery should not become a long independent task.

### Step 1: Assess current state

Determine whether the stall appears caused by local contact, camera/observation uncertainty, invalid subtask, or repeated VLA no-progress behavior.

### Step 2: Choose situation-specific tool calls

- Use `retreat_arm` when local clearance is needed.
- Use `lift_ee` if the end effector appears low or obstructed.
- Use `contact_displace` only when the needed release direction is evident.
- Use `move_ee_to_grounded_instance` only when scene memory provides the selected instance and finite 3D grounding.
- Use `reobserve_scene` after recovery motion.
- Use `safe_reset_posture` if local recovery is not enough.

### Step 3: Decide next intent

Prefer `retry` when local stall is cleared and the subtask remains valid. Prefer `replan` when repeated stall suggests the subtask is invalid. Use `abort` for unsafe state or repeated tool failure.

## Output Contract

Return JSON only:

```json
{
  "selected_workflow": "retreat-and-reobserve",
  "tool_calls": [
    {
      "tool_name": "retreat_arm",
      "args": {"arm": "left"},
      "reason": "create clearance from the stalled posture"
    },
    {
      "tool_name": "reobserve_scene",
      "args": {},
      "reason": "refresh observation before retry"
    }
  ],
  "post_recovery_intent": "retry",
  "stop_condition": "stall cleared, tool fails, repeated stall detected, or tool budget exhausted"
}
```

Allowed `post_recovery_intent` values: `retry`, `replan`, `abort`.

## Reference Plans

Reference plans are examples only. Generate a situation-specific plan from the actual observation and recovery history.

### Simple local stall

1. `retreat_arm`
2. `reobserve_scene`
3. `post_recovery_intent = retry`

### Low-clearance stall

1. `lift_ee`
2. `retreat_arm`
3. `reobserve_scene`
4. `post_recovery_intent = retry` or `replan`

## Failure Handling

- If `retreat_arm` fails, do not repeat it blindly; consider `lift_ee`, `safe_reset_posture`, or `abort`.
- If a stall repeats after this workflow, prefer `replan`.
- If observation remains uncertain, return `post_recovery_intent = replan`.

## Notes

This workflow is for local stall recovery, not for OOD classification.
