---
name: reobserve-scene
description: "Recovery primitive tool guide for refreshing scene observation without changing task intent."
---

# Reobserve Scene

## Overview

Use this primitive to refresh visual/context observations after an abnormal rollout state or after another recovery tool changes robot posture or scene visibility.

## Inputs (provide per tool-call planning step)

- `observation_summary`: current scene summary and uncertainty.
- `current_subtask`: subtask that needs updated observation.
- `recovery_history`: previous recovery tool calls and outcomes.
- `available_tools`: runtime tool allowlist.

## Runtime Tool

- `tool_name`: `reobserve_scene`
- `args_schema`: `{}`
- `result_schema`: `RecoveryToolResult`

## Use When

- A physical recovery action just changed robot posture or scene visibility.
- The current observation may be stale or insufficient.
- The planner needs fresh evidence before retrying or replanning.

## Do Not Use When

- No new information is expected and no intermediate action changed the state.
- The tool budget is exhausted.
- Safety requires immediate abort rather than more observation.

## Tool Call Contract

Return or include this tool call object when appropriate:

```json
{
  "tool_name": "reobserve_scene",
  "args": {},
  "reason": "refresh scene observation after recovery motion"
}
```

## Expected Result

Runtime refreshes the latest observation snapshot and returns a structured success/failure result.

## Failure Handling

- If reobserve fails, avoid retrying the same subtask blindly.
- Prefer `replan` or `abort` depending on safety and remaining context.

## Notes

This primitive does not classify OOD and does not move objects intentionally.
