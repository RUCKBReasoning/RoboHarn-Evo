---
name: close-gripper
description: "Recovery primitive tool guide for closing a selected gripper to re-establish or stabilize grasp state."
---

# Close Gripper

## Overview

Use this primitive when recovery requires re-closing a gripper after a failed or uncertain grasp, stabilizing a slipping object, or preparing for a retry after re-alignment.

## Inputs (provide per tool-call planning step)

- `observation_summary`: current gripper/object relation and which arm appears involved.
- `current_subtask`: the manipulation step that requires a stable grasp.
- `recovery_history`: previous recovery tool calls and outcomes.
- `available_tools`: runtime tool allowlist.

## Runtime Tool

- `tool_name`: `close_gripper`
- `args_schema`:
  - `arm` (required string): explicitly choose which gripper to close, one of `left`, `right`, or `both`.
  - Aliases accepted by runtime: `all`/`dual` -> `both`, `left_arm` -> `left`, `right_arm` -> `right`.
- `result_schema`: `RecoveryToolResult`

## Use When

- The object is visible and appears reachable by a specific gripper.
- The gripper should be closed to stabilize a slipping or weak grasp.
- A retry requires re-establishing grasp after reobserve or minor repositioning.
- In dual-arm tasks, choose `arm` from the current evidence:
  - `left` if the left gripper should grasp or stabilize the object.
  - `right` if the right gripper should grasp or stabilize the object.
  - `both` only when both grippers should close or the responsible gripper cannot be determined.

## Do Not Use When

- The target object is not visible or reachability is uncertain.
- Closing the gripper could crush, push, or destabilize the object.
- The recovery plan first needs `open_gripper`, `retreat_arm`, `lift_ee`, or `reobserve_scene`.

## Tool Call Contract

Return or include this tool call object when appropriate:

```json
{
  "tool_name": "close_gripper",
  "args": {"arm": "right"},
  "reason": "stabilize the object with the right gripper before retrying the subtask"
}
```

If the affected gripper is unclear but grasp stabilization is still necessary, use:

```json
{
  "tool_name": "close_gripper",
  "args": {"arm": "both"},
  "reason": "stabilize uncertain dual-gripper grasp state before retrying"
}
```

## Expected Result

The selected gripper closes and the runtime returns a structured success/failure result.

## Failure Handling

- If the tool fails, do not keep issuing repeated close commands without new evidence.
- Prefer `reobserve_scene`, `replan`, or `abort` if the object pose or gripper state remains uncertain.

## Notes

This primitive does not decide whether the grasp is valid by itself. Use observations, recovery history, and follow-up monitoring to decide retry, replan, or abort.
