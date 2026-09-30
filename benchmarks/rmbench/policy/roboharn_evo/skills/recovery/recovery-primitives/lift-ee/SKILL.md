---
name: lift-ee
description: "Recovery primitive tool guide for lifting the end effector to escape local contact or obstruction."
---

# Lift End Effector

## Overview

Use this primitive when the arm or gripper appears locally blocked and needs bounded vertical EE clearance before retreat, reobserve, retry, or replan.

## Inputs (provide per tool-call planning step)

- `observation_summary`: evidence of low clearance, contact, or obstruction.
- `current_subtask`: subtask interrupted by the local issue.
- `recovery_history`: previous recovery tool calls and outcomes.
- `available_tools`: runtime tool allowlist.

## Runtime Tool

- `tool_name`: `lift_ee`
- `args_schema`:
  - `arm` (required string): explicitly choose `left`, `right`, or `both` from current evidence.
  - `distance` (optional number): total positive world-frame z lift distance in meters. Runtime caps this conservatively.
  - `steps` (optional integer): number of bounded internal control steps across the total distance. Default is 1; runtime caps this conservatively.
- `result_schema`: `RecoveryToolResult`

## Use When

- Vertical clearance may resolve a local obstruction.
- A retreat motion alone may scrape or collide.
- The end effector is too low or close to the object/table region.
- Choose `arm` from evidence; use `both` only when both arms are implicated or the responsible arm is unclear.

## Do Not Use When

- Lifting may collide with an overhead object or worsen the scene.
- The task requires holding a delicate object and lifting would be unsafe.
- The same lift command already failed without new evidence.

## Tool Call Contract

Return or include this tool call object when appropriate:

```json
{
  "tool_name": "lift_ee",
  "args": {"arm": "right", "distance": 0.03, "steps": 3},
  "reason": "create bounded vertical clearance from the obstruction"
}
```

## Expected Result

The selected end effector moves upward by one or more bounded safe deltas while preserving orientation and gripper state, and runtime returns a structured result.

## Failure Handling

- If lift fails, prefer `retreat_arm`, `safe_reset_posture`, `replan`, or `abort` depending on safety.
- Do not repeat lift blindly after failure.

## Notes

This primitive does not solve object placement or decide task completion.
