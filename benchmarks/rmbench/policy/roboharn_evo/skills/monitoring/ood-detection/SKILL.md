---
name: ood-detection
description: "Skill contract for OOD and failure-semantic detection during VLA rollout monitoring. Use this skill to judge whether the current subtask has entered an out-of-domain, failed, blocked, or visually invalid state and return structured monitor signals instead of relying on hardcoded heuristics."
---

# OOD Detection

## Overview
Use this skill to judge whether the current rollout has entered a semantic OOD, blocked, failed, or visually invalid state that should surface as monitor signals.

This skill is part of the monitoring path, not the execution path:

- `policy/roboharn_evo/agent/monitoring/ood_detector.py` builds the payload and calls this skill
- `policy/roboharn_evo/agent/monitoring/ood_skill_evaluator.py` consumes a structured OOD result
- `policy/roboharn_evo/agent/core/img_agent.py` consumes the returned signals and decides whether to keep control on VLA or hand off to recovery tools

In other words:

- this skill judges semantic execution health
- it does **not** execute recovery actions
- it does **not** replace progress or success detection
- it does **not** directly decide low-level robot control
- it **must** make the OOD classification itself and return it explicitly

## Inputs

Provide one JSON-like payload per evaluation.

### Required fields
- `selected_skill`: should be `ood-detection`
- `current_subtask`: the active subtask being executed by VLA
- `observation_summary`: the latest summarized world state or scene summary
- `monitor_status`: current rollout monitor status
- `recovery_state`: current recovery counters, pending action, and recent recovery context
- `execution_context`: structured execution context from runtime
- `step_count`: current environment step count
- `step_limit`: current environment step limit

### Expected `execution_context` fields
The exact context may evolve, but current evaluations should expect fields such as:

- `progress_score`
- `step_count`
- `step_limit`
- `task_success`
- `action_chunk_empty`

### Evidence source rules
- The closed VLM must directly decide the OOD type from the full payload.
- Do not rely on the runtime to infer `grasp_lost`, `scene_drift_detected`, or other OOD classes from a free-text summary.
- If the same conclusion is supported by multiple fields, mention the strongest evidence in `reason` and place the rest in `details`.

## Defaulting Rules
If some fields are missing or weakly populated, do not fail noisily. Fall back conservatively.

- If the payload is insufficient for a confident OOD classification, set `OOD_scenario` to `none`.
- If `execution_context` is missing optional keys, evaluate only from the fields that are present.
- If multiple possible failure interpretations exist, choose the most conservative `OOD_scenario` or `none`.
- If a condition is already better handled by progress monitoring or task success logic, do not force an OOD label.

## Hard Rules
- Be conservative. False positives are worse than returning `OOD_scenario = none`.
- Do not fabricate scene facts, object identities, grasp states, or failure causes.
- Do not emit recovery tool plans or tool-call arguments.
- Prefer runtime monitor signal vocabulary when possible.
- Use this skill only for semantic OOD / blocked / invalid-state judgment.
- Keep `reason` short, evidence-based, and directly tied to the provided payload.
- **Do not output only a summary and expect the runtime to classify the OOD type afterward.**
- The closed VLM must explicitly output `OOD_scenario` itself.

## Workflow

### Step 0: Confirm the evaluation target
- Read `current_subtask` first.
- Identify what the VLA is currently supposed to achieve.
- Judge failure relative to the active subtask, not only relative to the global task.

### Step 1: Read the latest evidence
- Inspect `observation_summary`, `monitor_status`, `step_count`, `step_limit`, `recovery_state`, and `execution_context` together.
- Treat explicit statements such as camera failure, target disappearance, grasp loss, or contradictory world state as stronger evidence than generic wording.
- If the payload does not support a concrete OOD claim, do not infer one.

### Step 2: Choose exactly one primary OOD scenario
The primary classification must be returned in `OOD_scenario`.

Use one of:
- `none`
- `object_not_visible`
- `motion_blocked`
- `grasp_lost`
- `scene_drift_detected`
- `requires_replan`

Choose the narrowest justified class.

### Step 3: Return structured output
- Always return one top-level JSON object.
- `OOD_scenario` is the primary field.
- Optional supporting fields such as `reason`, `confidence`, `signals`, and `analysis_note` may be included.
- If additional runtime signals are returned, they must be consistent with `OOD_scenario`.

## Response Contract
Return one top-level JSON object.

```json
{
  "status": "ok",
  "selected_skill": "ood-detection",
  "OOD_scenario": "grasp_lost",
  "reason": "the manipulated object is no longer in gripper control after lifting",
  "confidence": 0.84,
  "signals": [
    {
      "name": "grasp_lost",
      "level": "warning",
      "reason": "the manipulated object is no longer in gripper control after lifting",
      "score": 0.84,
      "details": {
        "source": "ood-detection",
        "mode": "direct-vlm-judgement"
      }
    }
  ],
  "analysis_note": "optional concise explanation"
}
```

### Top-level fields
- `status`: usually `ok`
- `selected_skill`: always `ood-detection`
- `OOD_scenario`: required primary OOD classification
- `reason`: short evidence-based explanation
- `confidence`: optional confidence-like score in `[0.0, 1.0]`
- `signals`: optional structured monitor signals consistent with `OOD_scenario`
- `analysis_note`: optional concise explanation

### `OOD_scenario` allowed values
- `none`
- `object_not_visible`
- `motion_blocked`
- `grasp_lost`
- `scene_drift_detected`
- `requires_replan`

### Signal fields
If `signals` is present, each signal object should contain:

- `name`: runtime-aligned signal name, preferably lowercase snake case
- `level`: `info`, `warning`, or `error`
- `reason`: short evidence-based explanation
- `score`: confidence-like severity score in `[0.0, 1.0]`
- `details`: optional structured metadata such as `source`, `mode`, or supporting evidence keys

## Recommended Signal Vocabulary
Prefer these runtime-aligned names:

- `object_not_visible`
- `motion_blocked`
- `grasp_lost`
- `scene_drift_detected`
- `requires_replan`

Notes:
- `OOD_scenario` is the primary contract. `signals` is supporting structure.
- Do not rely on the runtime to derive `OOD_scenario` from `signals` or from a text summary.
- Prefer narrower classes like `object_not_visible` or `motion_blocked` over generic escalation labels.

## Failure Handling
- If the payload is malformed, incomplete, or semantically weak, return `OOD_scenario: "none"` rather than hallucinating a diagnosis.
- If evidence conflicts, prefer the class backed by the strongest explicit evidence.
- If the summary indicates sensor/view failure, classify that directly instead of over-interpreting downstream task semantics.
- If repeated failures are present but the root cause is still unclear, prefer `requires_replan` only when the payload supports that escalation.

## Scope Boundary
- This skill judges semantic OOD / failure states during rollout monitoring.
- It does not execute recovery actions.
- It does not replace success detection or generic progress monitoring.
- It does not replace planner reasoning for full task decomposition.
- It must output the OOD classification directly instead of delegating classification to runtime post-processing.

## References
- `policy/roboharn_evo/agent/monitoring/ood_detector.py`: payload construction and result normalization
- `policy/roboharn_evo/agent/monitoring/ood_skill_evaluator.py`: structured OOD result consumer / adapter
- `policy/roboharn_evo/agent/core/img_agent.py`: OOD signals are consumed inside monitored rollout and may trigger handoff/recovery
- `policy/roboharn_evo/agent/monitoring/signals.py`: runtime monitor signal vocabulary
