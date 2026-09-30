---
name: recover-grasp-lost
description: "Recovery workflow harness for lost or unstable grasp state during manipulation."
---

# Recover Grasp Lost

## Overview

Use this workflow when `OOD_scenario` is `grasp_lost` or the manipulated object appears no longer attached to or controlled by the gripper.

This skill is a recovery planning harness. It should generate a situation-specific recovery plan rather than enforce one fixed sequence. The plan may repeat observation or retreat tools when needed, but should stay within the recovery tool budget.

## Inputs (provide per recovery planning call)

- `monitor_signal`: usually `grasp_lost`.
- `current_subtask`: the manipulation subtask that lost grasp.
- `observation_summary`: object visibility, gripper state, and robot posture.
- `recovery_history`: prior recovery tool calls and outcomes.
- `available_tools`: runtime recovery tools exposed by the executor.
- `tool_budget`: maximum number of recovery tool calls allowed.
- `post_recovery_options`: usually `retry`, `replan`, or `abort`.

## Available Recovery Tools

Commonly useful tools:

- `open_gripper`: clear failed or unstable grasp state.
- `retreat_arm`: create clearance from the object or contact region.
- `lift_ee`: lift the end effector if local obstruction is likely.
- `move_ee_to_grounded_instance`: move toward a grounded target or approach point when scene memory already selected the instance.
- `reobserve_scene`: refresh visual state after release or retreat.

Use only runtime tools present in `available_tools`.

## Workflow Harness

### Step 0: Safety + Preconditions

- Do not keep applying grasp-related recovery if the object or arm state is uncertain.
- Release unstable grasp state before large retreat motions when this reduces risk.
- Do not decide task success here; this workflow only prepares for retry, replan, or abort.

### Step 1: Assess current state

Check whether the object remains visible and reachable, whether the gripper is still closed on anything, and whether the arm is near collision or obstruction.

### Step 2: Choose situation-specific tool calls

Choose tool calls based on the current state:

- Use `open_gripper` if the grasp state should be cleared.
- Use `retreat_arm` or `lift_ee` if the end effector needs clearance.
- Use `move_ee_to_grounded_instance` only when `scene_memory` contains the selected finite 3D instance needed for a bounded retry setup.
- Use `reobserve_scene` after physical recovery actions to reassess target visibility.
- Repeat `reobserve_scene` only if an intermediate tool changed the scene or posture.

### Step 3: Decide next intent

Prefer `retry` when the target remains visible and reachable after recovery. Prefer `replan` if the target moved, the scene changed, or the original subtask is no longer valid. Use `abort` if recovery tools fail or the scene is unsafe.

## Output Contract

Return JSON only:

```json
{
  "selected_workflow": "recover-grasp-lost",
  "tool_calls": [
    {
      "tool_name": "open_gripper",
      "args": {"arm": "right"},
      "reason": "clear the failed grasp state before retreating"
    },
    {
      "tool_name": "retreat_arm",
      "args": {"arm": "right"},
      "reason": "create clearance after releasing the object"
    },
    {
      "tool_name": "reobserve_scene",
      "args": {},
      "reason": "verify whether the target remains visible and reachable"
    }
  ],
  "post_recovery_intent": "retry",
  "stop_condition": "target is visible and reachable, recovery tool fails, or tool budget is exhausted"
}
```

Allowed `post_recovery_intent` values: `retry`, `replan`, `abort`.

## Reference Plans

Reference plans are examples only. Generate a situation-specific plan from the actual observation and recovery history.

### Standard failed grasp clearing

1. `open_gripper`
2. `retreat_arm`
3. `reobserve_scene`
4. `post_recovery_intent = retry` if target is visible and reachable; otherwise `replan`.

### Obstructed failed grasp

1. `open_gripper`
2. `lift_ee`
3. `retreat_arm`
4. `reobserve_scene`
5. `post_recovery_intent = replan` if the object shifted significantly.

## Failure Handling

- If `open_gripper` fails, avoid repeated forceful motion; prefer `abort` or `replan` depending on safety.
- If `retreat_arm` fails, try `lift_ee` only if it is available and safe.
- If the target is no longer visible after reobserve, return `post_recovery_intent = replan`.

## Notes

This workflow should not classify grasp loss. It assumes monitoring has already produced the relevant signal.
