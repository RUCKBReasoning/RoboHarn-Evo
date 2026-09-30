---
name: safe-reset-posture
description: "Recovery primitive tool guide for moving the end effector toward a conservative safe original/home posture."
---

# Safe Reset Posture

## Overview

Use this primitive when local recovery is insufficient and the robot should move a small bounded EE step toward a conservative original/home posture before replanning, retrying, or aborting.

## Inputs (provide per tool-call planning step)

- `observation_summary`: current robot posture and safety uncertainty.
- `current_subtask`: subtask interrupted by unsafe or unrecoverable local state.
- `recovery_history`: previous recovery tool calls and outcomes.
- `available_tools`: runtime tool allowlist.

## Runtime Tool

- `tool_name`: `safe_reset_posture`
- `args_schema`:
  - `arm` (required string): explicitly choose `left`, `right`, or `both` from current evidence.
  - `max_translation` (optional number): maximum EE translation toward original pose in meters. Runtime uses a smaller cap than `move_to_home`.
- `result_schema`: `RecoveryToolResult`

## Use When

- The robot posture is uncertain or locally unsafe.
- Smaller local tools failed or are inappropriate.
- Replanning should start from a conservative posture.
- Use this instead of `move_to_home` when a smaller, safer EE step is preferred.

## Do Not Use When

- A minimal local recovery tool is clearly sufficient and safer.
- The environment state would be harmed by reset-like motion.
- The runtime adapter does not support a safe original/home EE pose lookup.

## Tool Call Contract

Return or include this tool call object when appropriate:

```json
{
  "tool_name": "safe_reset_posture",
  "args": {"arm": "both", "max_translation": 0.025},
  "reason": "move a small bounded step toward conservative original posture before replanning"
}
```

## Expected Result

The selected end effector moves a small bounded step toward RMBench original/home pose while preserving current orientation and gripper state, and runtime returns a structured result.

## Failure Handling

- If safe reset fails, return `post_recovery_intent = abort` or `replan` with a clear reason.
- Do not repeatedly call reset-like tools without new evidence or operator intervention.

## Notes

This primitive does not reset the full episode unless the runtime adapter explicitly defines that behavior.
