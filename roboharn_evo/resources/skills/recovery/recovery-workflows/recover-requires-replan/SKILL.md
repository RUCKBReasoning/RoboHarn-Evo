---
name: recover-requires-replan
description: "Recovery workflow harness for OOD states that should return control to planner instead of running unnecessary physical recovery tools."
---

# Recover Requires Replan

## Overview

Use this workflow when `OOD_scenario` is `requires_replan`, or when the current subtask is no longer valid and recovery should primarily hand control back to the planner.

This skill is a recovery planning harness. It may generate zero tool calls when physical recovery is unnecessary. It may also request a minimal observation or safe posture tool when the planner needs a reliable state before replanning.

## Inputs (provide per recovery planning call)

- `monitor_signal`: usually `requires_replan`.
- `current_subtask`: the subtask that is no longer valid.
- `observation_summary`: current scene and robot state.
- `recovery_history`: prior recovery calls and outcomes.
- `available_tools`: runtime recovery tools exposed by the executor.
- `tool_budget`: maximum recovery tool calls allowed.
- `post_recovery_options`: usually `replan` or `abort`; `retry` should be rare.

## Available Recovery Tools

Commonly useful tools:

- `reobserve_scene`: refresh observation before planner handoff.
- `safe_reset_posture`: move toward a conservative posture if needed before replanning.
- `move_to_home`: return to a reusable posture when available and appropriate.

Use only runtime tools present in `available_tools`.

## Workflow Harness

### Step 0: Safety + Preconditions

- Do not force a retry when the task context is invalid.
- Prefer zero or minimal tool calls unless the robot posture or observation is unsafe/uncertain.
- Preserve enough context for the planner to understand why replanning is required.

### Step 1: Assess current state

Determine whether replanning can happen immediately, or whether a fresh observation or safe posture is needed first.

### Step 2: Choose situation-specific tool calls

- Use no tool calls if current state is already safe and observable.
- Use `reobserve_scene` if planner needs updated scene evidence.
- Use `safe_reset_posture` or `move_to_home` if posture is unsafe or unsuitable for replanning.

### Step 3: Decide next intent

Prefer `replan`. Use `abort` only if state is unsafe or recovery tools fail. Use `retry` only if replanning is no longer necessary after updated observation.

## Output Contract

Return JSON only:

```json
{
  "selected_workflow": "recover-requires-replan",
  "tool_calls": [],
  "post_recovery_intent": "replan",
  "stop_condition": "planner can safely receive control or recovery budget is exhausted"
}
```

Allowed `post_recovery_intent` values: `retry`, `replan`, `abort`.

## Reference Plans

Reference plans are examples only. Generate a situation-specific plan from the actual observation and recovery history.

### Direct replan

1. No physical tool call.
2. `post_recovery_intent = replan`

### Reobserve before replan

1. `reobserve_scene`
2. `post_recovery_intent = replan`

### Safe posture before replan

1. `safe_reset_posture`
2. `reobserve_scene`
3. `post_recovery_intent = replan`

## Failure Handling

- If a requested observation or posture tool fails, return `post_recovery_intent = abort` or `replan` with the failure reason.
- Do not keep attempting recovery when the correct high-level action is planner handoff.

## Notes

This workflow is the minimal bridge from monitoring to planner when no physical recovery is necessary.
