---
name: retreat-arm
description: "Recovery primitive tool guide for moving the end effector away from contact, blockage, or uncertain local state."
---

# Retreat Arm

## Overview

Use this primitive to create local end-effector clearance after blocked motion, failed grasp, uncertain contact, or a stalled posture.

## Inputs (provide per tool-call planning step)

- `observation_summary`: contact/obstruction evidence and current arm posture.
- `current_subtask`: subtask interrupted by the local issue.
- `recovery_history`: previous recovery tool calls and outcomes.
- `available_tools`: runtime tool allowlist.

## Runtime Tool

- `tool_name`: `retreat_arm`
- `args_schema`:
  - `arm` (required string): explicitly choose `left`, `right`, or `both` from current evidence.
  - `axis` (optional string): world-frame displacement axis, one of `x`, `y`, `z`. Default: `x`.
  - `direction` (optional string): `positive` or `negative`. Default: `negative`.
  - `distance` (optional number): total bounded EE displacement in meters. Runtime caps this conservatively.
  - `steps` (optional integer): number of bounded internal control steps across the total distance. Default is 1; runtime caps this conservatively.
- `result_schema`: `RecoveryToolResult`

## Use When

- The arm appears too close to an object, obstacle, or contact region.
- A local EE retreat can make retry or reobserve safer.
- A failed grasp should be cleared before another attempt.
- Choose `arm` from the current evidence; use `both` only when both arms are implicated or the responsible arm is unclear.

## Do Not Use When

- Retreat motion may collide with the environment.
- The correct action is immediate planner handoff without physical recovery.
- The same retreat command already failed without any new information.

## Tool Call Contract

Return or include this tool call object when appropriate:

```json
{
  "tool_name": "retreat_arm",
  "args": {"arm": "left", "axis": "x", "direction": "negative", "distance": 0.03, "steps": 3},
  "reason": "create bounded end-effector clearance from the local contact region"
}
```

If the responsible arm is uncertain, use:

```json
{
  "tool_name": "retreat_arm",
  "args": {"arm": "both", "axis": "x", "direction": "negative", "distance": 0.03},
  "reason": "create conservative dual-arm clearance before reobserving"
}
```

## Expected Result

The selected end effector moves by one or more bounded world-frame displacements while preserving orientation and gripper state, and runtime returns a structured result.

## Failure Handling

- If retreat fails, consider `lift_ee`, `safe_reset_posture`, `replan`, or `abort`.
- Do not repeat retreat blindly if it already failed.

## Notes

This primitive does not choose a new task target or replan the global task.
