---
name: move-ee-to-pose
description: "Recovery primitive tool guide for bounded end-effector motion to a validated absolute pose."
---

# Move End Effector To Pose

## Overview

Use this primitive only when recovery planning already has a validated world-frame end-effector target pose or target position. This is a low-level recovery tool, not a semantic grounding tool.

## Inputs

- `observation_summary`: current robot posture and relevant safety evidence.
- `robot_state`: current end-effector pose and gripper state.
- `recovery_history`: prior recovery tool calls and outcomes.
- `available_tools`: runtime tool allowlist.

## Runtime Tool

- `tool_name`: `move_ee_to_pose`
- `args_schema`:
  - `arm` (required string): `left` or `right`.
  - `target_pose` (optional list): 7D world pose `[x, y, z, qw, qx, qy, qz]`.
  - `target_xyz` (optional list): 3D world position `[x, y, z]`; runtime preserves the current quaternion unless `target_quat_wxyz` is provided.
  - `target_quat_wxyz` (optional list or string): 4D quaternion, or `preserve`.
  - `max_translation` (optional number): maximum translation in meters for each internal recovery step. Runtime caps this conservatively.
  - `steps` (optional integer): number of bounded internal control steps. Default is 1; runtime caps this conservatively.
- `result_schema`: `RecoveryToolResult`

## Use When

- The pose comes from validated geometry, a trusted controller, or a previous grounded computation.
- A short bounded pose correction can clear the recovery condition.
- The responsible arm is known.

## Do Not Use When

- The target pose is guessed from language only.
- The needed object grounding is available only as a scene instance; use `move_ee_to_grounded_instance` instead.
- The responsible arm is unclear.
- A smaller relative clearance primitive is sufficient.

## Tool Call Contract

```json
{
  "tool_name": "move_ee_to_pose",
  "args": {
    "arm": "left",
    "target_xyz": [0.12, -0.04, 0.82],
    "target_quat_wxyz": "preserve",
    "max_translation": 0.04,
    "steps": 3
  },
  "reason": "execute a bounded correction to a validated world-frame target position"
}
```

## Expected Result

Runtime clamps each internal translation step, preserves gripper state, executes one or more bounded end-effector commands, and returns a structured result.

## Failure Handling

- If validation fails, return `replan` or choose a safer primitive.
- Do not invent coordinates after a validation failure.
