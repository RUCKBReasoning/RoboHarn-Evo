---
name: recover-scene-drift
description: "Recovery workflow harness for scene drift or changed task context detected during rollout."
---

# Recover Scene Drift

## Overview

Use this workflow when `OOD_scenario` is `scene_drift_detected`, or the scene appears changed enough that the current rollout/subtask context may no longer be valid.

This skill is a recovery planning harness. It should generate a situation-specific plan that refreshes observation and decides whether retry is still safe or replanning is required.

## Inputs (provide per recovery planning call)

- `monitor_signal`: usually `scene_drift_detected`.
- `current_subtask`: the subtask whose assumptions may be invalid.
- `observation_summary`: changed object positions, new obstacles, robot posture, target relation.
- `recovery_history`: prior recovery tool calls and outcomes.
- `available_tools`: runtime recovery tools exposed by the executor.
- `tool_budget`: maximum recovery tool calls allowed.
- `post_recovery_options`: usually `retry`, `replan`, or `abort`.

## Available Recovery Tools

Commonly useful tools:

- `reobserve_scene`: refresh scene state.
- `retreat_arm`: create clearance if robot motion may have caused scene change.
- `safe_reset_posture`: move toward a conservative posture if the scene is uncertain.

Use only runtime tools present in `available_tools`.

## Workflow Harness

### Step 0: Safety + Preconditions

- Do not continue a rollout whose scene assumptions are invalid.
- Prefer observation refresh before deciding retry or replan.
- Avoid unnecessary physical recovery if the right answer is simply to replan.

### Step 1: Assess current state

Identify what changed: object pose, target visibility, obstacle layout, robot pose, or task-relevant spatial relation.

### Step 2: Choose situation-specific tool calls

- Use `reobserve_scene` when updated visual state is needed.
- Use `retreat_arm` if the robot may be obstructing or disturbing the scene.
- Use `safe_reset_posture` only when uncertainty or safety requires a conservative posture.

### Step 3: Decide next intent

Prefer `replan` when scene drift invalidates the current subtask. Use `retry` only if the scene change is minor and the current subtask remains valid. Use `abort` when safety or state cannot be established.

## Output Contract

Return JSON only:

```json
{
  "selected_workflow": "recover-scene-drift",
  "tool_calls": [
    {
      "tool_name": "reobserve_scene",
      "args": {},
      "reason": "refresh scene state before replanning"
    }
  ],
  "post_recovery_intent": "replan",
  "stop_condition": "fresh observation obtained, safety uncertainty remains, or tool budget is exhausted"
}
```

Allowed `post_recovery_intent` values: `retry`, `replan`, `abort`.

## Reference Plans

Reference plans are examples only. Generate a situation-specific plan from the actual observation and recovery history.

### Observation-only scene drift

1. `reobserve_scene`
2. `post_recovery_intent = replan`

### Drift with robot occlusion or contact

1. `retreat_arm`
2. `reobserve_scene`
3. `post_recovery_intent = replan`

## Failure Handling

- If `reobserve_scene` fails, avoid retrying the current subtask blindly.
- If scene state remains uncertain, return `post_recovery_intent = replan` or `abort` depending on safety.
- If a physical recovery tool fails, record the failure reason for planner context.

## Notes

This workflow is for restoring reliable context, not for deciding original task success.
