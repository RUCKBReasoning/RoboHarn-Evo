# Case: <short descriptive title>

## Metadata

- Case ID: `<case_id>`
- Trace ID: `<trace_id>`
- Task: `<global task>`
- Subtask: `<active subtask>`
- OOD_scenario: `<none | object_not_visible | motion_blocked | grasp_lost | scene_drift_detected | requires_replan>`
- Recovery workflow: `<workflow name>`
- Post recovery intent: `<retry | replan | abort>`
- Outcome: `<detect_only | recovery_success_retry | recovery_success_replan | recovery_failed | replan_without_tools | aborted | false_positive | false_negative | incomplete_trace>`
- Status: `candidate`

## Context

Describe the task state before the OOD event. Keep this factual and tied to runtime observations.

## Evidence

List the strongest evidence supporting the OOD scenario.

- Evidence 1:
- Evidence 2:

## Intervention

1. `<primitive or workflow step>`
2. `<primitive or workflow step>`
3. `<primitive or workflow step>`

## Outcome

Describe what happened after recovery: retry succeeded, replan was needed, or the episode failed.

## Notes

This is factual compression of one runtime trace. Do not turn it into a hard rule here. Do not include benchmark labels, annotation fields, or gold OOD labels that would leak into VLM inputs.
