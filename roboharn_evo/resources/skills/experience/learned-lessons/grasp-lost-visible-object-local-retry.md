# Lesson: visible grasp_lost should prefer local retry

## Metadata

- Lesson ID: `grasp_lost_visible_object_local_retry`
- Status: `accepted`
- Confidence: `medium`
- Support Count: `1`
- Opposing Count: `0`
- Last Validated At: ``

## Applies When

- OOD_scenario: `grasp_lost`
- Task family: `pick_and_place`
- Preconditions:
  - Target object is still likely visible or reachable.
  - The arm is not known to be mechanically blocked.
  - Recovery budget still permits retry.

## Recommendation

Prefer a local recovery workflow such as `recover-grasp-lost`, then retry the same subtask after reobserving the scene.

## Avoid

Avoid immediate full replan when the target remains visible and the failure is limited to the grasp state. Avoid `move_to_home` unless there is a collision or safety reason.

## Supporting Cases

- `template_case_grasp_lost_visible_object`

## Opposing Cases

- None recorded.

## Evaluation Result

- Status: `seed`
- Pass Rate: ``
- Payloads Matched: ``

## Promotion Criteria

This seed lesson is accepted as a starting heuristic for retrieval wiring. It should be revalidated against real RMBench recovery traces and downgraded if it causes repeated bad recovery choices.

## Notes

This is advisory retrieval context. Runtime tool validation and current environment capabilities remain authoritative.
