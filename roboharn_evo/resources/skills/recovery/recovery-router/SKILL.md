---
name: recovery-router
description: "Recovery routing harness that selects a recovery workflow and post-recovery intent from an explicit OOD_scenario or monitor signal."
---

# Recovery Router

## Overview

Use this skill after monitoring has produced an explicit `OOD_scenario` or monitor signal. The router chooses a recovery workflow harness and a post-recovery intent for the current situation.

This skill does not execute tools. It returns a structured routing decision that a recovery workflow planner can use to generate concrete tool calls.

## Inputs (provide per routing call)

- `signal_name`: monitor signal or OOD scenario, such as `grasp_lost`, `motion_blocked`, or `stall_detected`.
- `OOD_scenario`: explicit OOD scenario when available.
- `reason`: monitor/OOD evidence.
- `current_subtask`: the active subtask when the signal occurred.
- `observation_summary`: current scene and robot state summary.
- `recovery_state`: retry/replan/reset budgets and previous recovery state.
- `recovery_history`: prior recovery tool calls and outcomes.
- `available_workflows`: recovery workflow SKILL names and descriptions.

## Available Recovery Workflows

Common workflow choices:

- `recover-object-not-visible`: target or relevant object is not visible.
- `recover-motion-blocked`: local motion is blocked or contact obstruction is likely.
- `recover-grasp-lost`: manipulated object is no longer controlled by the gripper.
- `recover-scene-drift`: scene changed enough to invalidate current rollout assumptions.
- `recover-requires-replan`: no physical recovery is needed; return control to planner.
- `task-level-recovery-control`: pure recovery-control debugging should continue the global instruction using grounded scene memory and recovery tools instead of repeating the same OOD clearance.
- `retreat-and-reobserve`: local stall may be cleared by retreating and refreshing observation.
- `go-home-and-retry`: step budget exhaustion or unsafe local state requires returning toward a reusable posture.

Use only workflows present in the supplied `available_workflows` list.

## Routing Harness

### Step 0: Preconditions

- Assume OOD/monitoring has already produced the signal; do not re-classify raw images here.
- Do not output low-level tool calls here.
- Prefer the least disruptive workflow that can safely restore control.

### Step 1: Read monitor/OOD signal

Map the signal and evidence to the most relevant recovery workflow. If the signal is ambiguous, prefer a workflow that refreshes observation or returns control to planning.

### Step 2: Select situation-specific workflow

Choose exactly one workflow. Use the current subtask and recovery history to avoid repeating a workflow that already failed without new evidence.

### Step 3: Select post-recovery intent

Choose one of:

- `retry`: recovery should prepare the same subtask for another attempt.
- `replan`: recovery should return control to the planner.
- `abort`: recovery should stop because state is unsafe or invalid.

## Output Contract

Return JSON only:

```json
{
  "selected_skill": "recovery-router",
  "OOD_scenario": "grasp_lost",
  "recovery_workflow": "recover-grasp-lost",
  "post_recovery_intent": "retry",
  "reason": "the object is no longer controlled by the gripper, so the failed grasp state should be cleared before retry"
}
```

Allowed `post_recovery_intent` values: `retry`, `replan`, `abort`.

## Reference Routing Policy

Reference mappings are examples only. Generate a situation-specific routing decision from the actual signal, task context, and recovery history.

- `object_not_visible` often routes to `recover-object-not-visible` and usually returns `replan` if the object remains unavailable.
- `motion_blocked` often routes to `recover-motion-blocked` and may return `retry` after local clearance.
- `grasp_lost` often routes to `recover-grasp-lost` and may return `retry` if the target remains visible and reachable.
- `scene_drift_detected` often routes to `recover-scene-drift` and usually returns `replan`.
- `requires_replan` often routes to `recover-requires-replan` and returns `replan`.
- `task_level_recovery_control` routes to `task-level-recovery-control` and usually returns `retry` while tool control should continue.
- `stall_detected` often routes to `retreat-and-reobserve`.
- `step_budget_exhausted` often routes to `go-home-and-retry` or `recover-requires-replan` depending on posture safety.

## Failure Handling

- If no workflow fits, return `recovery_workflow = "recover-requires-replan"` and `post_recovery_intent = "replan"`.
- If recovery history shows repeated failed physical recovery, return `post_recovery_intent = "abort"` or `replan` with a clear reason.
- Do not invent workflow names that are not present in `available_workflows`.

## Notes

This skill is a routing harness. Runtime validates the selected workflow before asking the workflow planner to produce concrete tool calls.
