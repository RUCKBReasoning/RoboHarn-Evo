---
name: go-home-and-retry
description: "Recovery workflow harness for step budget exhaustion that returns toward a reusable posture before replanning."
---

# Go Home And Retry

## Overview

Use this workflow when monitored rollout exhausts its step budget or the current rollout has become unproductive enough that the robot should return toward a reusable state before handing control back to planning.

This skill is a recovery planning harness. It does not define one mandatory tool sequence. Generate a situation-specific recovery plan from the current observation, current subtask, monitor signal, recovery history, and available tool budget.

## Inputs (provide per recovery planning call)

- `monitor_signal`: usually `step_budget_exhausted`.
- `current_subtask`: the subtask that exhausted its step budget.
- `observation_summary`: current scene and robot posture summary.
- `recovery_history`: prior tool calls and outcomes in this episode.
- `available_tools`: runtime recovery tools exposed by the executor.
- `tool_budget`: maximum number of recovery tool calls allowed for this recovery attempt.
- `post_recovery_options`: usually `retry`, `replan`, or `abort`.

## Available Recovery Tools

Commonly useful tools:

- `move_to_home`: move the robot toward the configured home pose.
- `safe_reset_posture`: move toward a conservative safe posture.
- `reobserve_scene`: refresh the observation after motion.

Use only runtime tools that are present in the supplied `available_tools` list.

## Workflow Harness

### Step 0: Safety + Preconditions

- Prefer a conservative posture if the robot may be close to collision, joint limits, or an unstable object.
- Do not reset the full episode unless the runtime tool explicitly represents that behavior.
- Keep the plan short; step-budget exhaustion usually means control should return to planner rather than continue long recovery.

### Step 1: Assess current state

Decide whether the robot is already in a reusable posture. If it is already safe and observable, `reobserve_scene` alone may be sufficient before replanning.

### Step 2: Choose situation-specific tool calls

Choose zero or more tool calls. Repeated calls are allowed only when justified by observations or previous tool results. Examples:

- Use `safe_reset_posture` if local posture appears unsafe.
- Use `move_to_home` if a reusable start posture is needed.
- Use `reobserve_scene` after motion to verify the post-recovery state.

### Step 3: Decide next intent

Prefer `replan` after returning to a reusable posture. Use `abort` only when tool calls fail or the scene cannot be made safe. Use `retry` only if the subtask remains clearly valid and retrying is safer than replanning.

## Output Contract

Return JSON only:

```json
{
  "selected_workflow": "go-home-and-retry",
  "tool_calls": [
    {
      "tool_name": "move_to_home",
      "args": {"arm": "both"},
      "reason": "return to a reusable posture before replanning"
    },
    {
      "tool_name": "reobserve_scene",
      "args": {},
      "reason": "verify the scene after moving home"
    }
  ],
  "post_recovery_intent": "replan",
  "stop_condition": "robot is in a reusable posture with refreshed observation, or tool budget is exhausted"
}
```

Allowed `post_recovery_intent` values: `retry`, `replan`, `abort`.

## Reference Plans

Reference plans are examples only. Generate a situation-specific plan from the actual observation and recovery history.

### Minimal observation refresh

1. `reobserve_scene`
2. `post_recovery_intent = replan`

### Conservative reset posture

1. `safe_reset_posture`
2. `move_to_home`
3. `reobserve_scene`
4. `post_recovery_intent = replan`

## Failure Handling

- If `move_to_home` fails, try `safe_reset_posture` only if tool budget remains and it is safe.
- If posture tools fail repeatedly, return `post_recovery_intent = abort` with the failure reason.
- If observation refresh fails, do not retry the same subtask blindly; prefer `replan` or `abort` depending on safety.

## Notes

This workflow is a planning harness, not a fixed primitive list. Runtime must validate all generated tool calls against the allowed recovery tool schema before execution.
