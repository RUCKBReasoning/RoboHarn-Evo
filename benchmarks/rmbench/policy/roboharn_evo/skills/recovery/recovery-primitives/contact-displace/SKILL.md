---
name: contact-displace
description: "Recovery primitive tool guide for a very small bounded end-effector displacement during contact recovery."
---

# Contact Displace

## Overview

Use this primitive for a short, bounded end-effector displacement when local contact needs to be released or probed without launching a new task-level action.

## Inputs

- `observation_summary`: contact, blockage, or stall evidence.
- `robot_state`: current end-effector pose and gripper state.
- `current_subtask`: interrupted subtask.
- `recovery_history`: previous recovery tool calls and outcomes.
- `available_tools`: runtime tool allowlist.

## Runtime Tool

- `tool_name`: `contact_displace`
- `args_schema`:
  - `arm` (required string): explicitly choose `left` or `right`; `both` is not valid for task-progress contact.
  - `axis` (optional string): world-frame displacement axis, one of `x`, `y`, `z`. Default: `z`.
  - `direction` (optional string): `positive` or `negative`. Default: `negative`.
  - `distance` (optional number): total positive displacement in meters. Runtime caps this more tightly than ordinary retreat tools.
  - `steps` (optional integer): number of bounded internal control steps across the total distance. Default is 1; runtime caps this conservatively.
  - `gripper_precondition` (optional string): `open` or `closed`. Use it when the displacement requires a specific contact shape; runtime may satisfy the explicit precondition before moving.
  - `complete_transient_cycle` (optional boolean): set to `true` for a press/tap actuation that must release before re-observation. It requires a same-batch grounded `contact_world_m` setup. Runtime first returns the same arm to that instance's grounded `approach_world_m`; when valid live camera, contact-normal, and target-extent geometry are available, it then adds a separate camera-tangent clearance move before re-observation. Do not set it for placement lowering, sustained pushing, bracing, or hold-down actions.
- `result_schema`: `RecoveryToolResult`

## Use When

- The end effector is already near contact and needs a small release displacement.
- A full retreat or reset would be unnecessarily large.
- The contact direction is known from current posture or recent tool outcome.
- The action is a transient press/tap and `complete_transient_cycle = true` can close the actuation with a grounded release and geometry-derived visual clearance rather than leaving the end effector over the target.

## Do Not Use When

- The contact direction is unknown and a safe reobserve or retreat is more appropriate.
- The same displacement already failed without new evidence.
- The command would push deeper into an obstacle or object.
- The end effector was already commanded to a grounded full-contact pose and the requested displacement would add large same-direction overtravel. Runtime may cap such a request to a small compliance allowance.

## Tool Call Contract

```json
{
  "tool_name": "contact_displace",
  "args": {
    "arm": "left",
    "axis": "z",
    "direction": "positive",
    "distance": 0.015,
    "steps": 3,
    "gripper_precondition": "closed",
    "complete_transient_cycle": true
  },
  "reason": "apply a very small bounded displacement to release local contact"
}
```

## Expected Result

Runtime satisfies an explicit gripper precondition when needed, then performs one or more bounded relative end-effector displacements while preserving orientation and the resulting gripper state. For a grounded transient cycle, runtime also returns to the same instance's grounded approach before re-observation; that release is part of the same task-level event, not another repetition.

## Failure Handling

- If contact remains unresolved, prefer `retreat_arm`, `lift_ee`, `reobserve_scene`, or `replan`.
- Do not keep applying contact displacement in the same direction after failure.
