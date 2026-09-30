---
name: move-to-home
description: "Recovery primitive tool guide for moving the end effector toward its configured original/home pose."
---

# Move To Home

## Overview

Use this primitive when recovery requires moving one or both end effectors toward their RMBench original/home EE pose before replanning, retrying, or ending an unsafe local rollout.

## Inputs (provide per tool-call planning step)

- `observation_summary`: current robot posture and scene state.
- `current_subtask`: subtask interrupted by step budget exhaustion or unsafe posture.
- `recovery_history`: previous recovery tool calls and outcomes.
- `available_tools`: runtime tool allowlist.

## Runtime Tool

- `tool_name`: `move_to_home`
- `args_schema`:
  - `arm` (required string): explicitly choose `left`, `right`, or `both` from current evidence.
  - `max_translation` (optional number): maximum EE translation toward original pose in meters. Runtime caps this conservatively.
- `result_schema`: `RecoveryToolResult`

## Use When

- A reusable home/original EE pose is needed before replanning.
- Step budget exhaustion leaves the robot in an unsuitable posture.
- The next planner decision should start from a more predictable robot state.
- Choose `left` or `right` if only one arm is involved; use `both` when both arms should return toward home.

## Do Not Use When

- Moving home may collide with the current scene.
- A smaller local recovery tool is safer and sufficient.
- The runtime adapter does not support home/original EE pose lookup.

## Tool Call Contract

Return or include this tool call object when appropriate:

```json
{
  "tool_name": "move_to_home",
  "args": {"arm": "both", "max_translation": 0.05},
  "reason": "move end effectors toward original poses before replanning"
}
```

## Expected Result

The selected end effector moves a bounded step toward RMBench original/home pose while preserving current orientation and gripper state, and runtime returns a structured result.

## Failure Handling

- If home motion fails, consider `safe_reset_posture`, `replan`, or `abort` depending on safety.
- Do not repeat home motion blindly after failure.

## Notes

This primitive does not decide whether to retry or replan. The recovery workflow planner decides post-recovery intent.
