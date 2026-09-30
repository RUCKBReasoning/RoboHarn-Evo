---
name: open-gripper
description: "Recovery primitive tool guide for opening the gripper to release unstable or failed grasp state."
---

# Open Gripper

## Overview

Use this primitive when recovery requires clearing a failed grasp, releasing an unstable object, or preparing the arm for safe retreat.

## Inputs (provide per tool-call planning step)

- `observation_summary`: current gripper/object relation.
- `current_subtask`: the manipulation step that failed or became uncertain.
- `recovery_history`: previous recovery tool calls and outcomes.
- `recovery_state.manipulation_state`: exact held-instance, confirmation, provisional-authority, and placement state for each arm.
- `grasp_transport_policy`: `strict` or `evidence_only`; this controls whether uncertainty remains a verification obligation or a non-blocking evidence record.
- `release_guard_enabled`: `false` means an explicit open is dispatched after ordinary argument validation; `true` restores the legacy placement/holding release gate.
- `available_tools`: runtime tool allowlist.

## Runtime Tool

- `tool_name`: `open_gripper`
- `args_schema`:
  - `arm` (required string): explicitly choose which gripper to open, one of `left`, `right`, or `both`.
  - `release_held_instance_id` (optional string): attribution metadata naming the object believed to be released.
  - `release_target_id` (optional string): attribution metadata naming the intended public placement target.
  - Aliases accepted by runtime: `all`/`dual` -> `both`, `left_arm` -> `left`, `right_arm` -> `right`.
- `result_schema`: `RecoveryToolResult`

## Use When

- The object is no longer securely grasped.
- The gripper may be holding an object in an unstable or unsafe state.
- A retreat motion should happen only after releasing the gripper.
- Under `grasp_transport_policy=evidence_only`, the planner deliberately chooses to abandon a `holding_provisional` hypothesis. This explicit open is allowed, but it is not proof that the grasp failed and is not a successful placement.
- In dual-arm tasks, choose `arm` from the current evidence:
  - `left` if the left gripper is holding or interfering with the object.
  - `right` if the right gripper is holding or interfering with the object.
  - `both` only when both grippers are implicated or the failing gripper cannot be determined.

## Do Not Use When

- The task requires maintaining a stable grasp for the next immediate motion.
- The gripper state is unknown and opening it would create an unsafe drop.
- With `release_guard_enabled=true`, runtime reports a transport-authorized held instance and the legacy held-instance/target/final-place release contract is not satisfied.
- With `release_guard_enabled=false`, the absence of release IDs, same-batch final-place setup, or pre-release validation is not a runtime veto. The planner must decide from task intent and physical state whether opening would drop or misplace the object.
- The signal is purely visual uncertainty with no grasp-related evidence.

## Tool Call Contract

Return or include this tool call object when appropriate:

```json
{
  "tool_name": "open_gripper",
  "args": {
    "arm": "left",
    "release_held_instance_id": "object_left",
    "release_target_id": "place:held:object_left:vacated:object_left"
  },
  "reason": "clear failed grasp state on the left gripper before retreating"
}
```

If the affected gripper is unclear, use the default both-arm form:

```json
{
  "tool_name": "open_gripper",
  "args": {"arm": "both"},
  "reason": "clear uncertain failed grasp state before retreating"
}
```

## Expected Result

The gripper opens and the runtime returns a structured success/failure result.
Under `strict`, uncertain visual attachment evidence may remain subject to the
existing verification workflow. Under `evidence_only`, runtime must not automatically insert `open_gripper`, return-to-grasp, retreat, or re-observation because visual attachment evidence is missing, ambiguous, or negative. An open must be an explicit planner action.

With `release_guard_enabled=false`, runtime does not reject or delay an explicit
open because release IDs, a same-batch final-place call, holding confirmation,
or placement validation are absent or inconsistent. Those fields remain useful
for explanation, but they are not authorization. The planner is responsible for
deciding whether opening is physically appropriate. With
`release_guard_enabled=true`, the legacy held-instance/target/final-place gate
is restored. Opening to abandon a provisional hypothesis clears that state
without claiming `holding_confirmed=true`, grasp failure, or placement success.
Under either setting, gripper command success does not by itself
complete placement: when prior runtime state contains a placement context, the
runtime retains a
`release_pending_verification` state until fresh geometry confirms that the
object remains at the target, the EE has cleared it, and the object is stable
across observations. Once runtime reports that the EE has cleared the release
pose, placement verification becomes observation-only: do not issue another
retreat or any other physical action merely to obtain the next stability
sample. If deterministic runtime evidence instead reports
`placement_recovery_required=true`, manipulation state becomes
`release_recovery_required`: the released object is outside its selected target
and may be reacquired through a new bounded grasp-and-place sequence.

## Failure Handling

- If the tool fails, do not keep issuing repeated open commands without new evidence.
- Prefer `replan` or `abort` if the gripper state cannot be made safe.

## Notes

This primitive does not decide whether grasp was lost, and it does not decide retry/replan by itself.
