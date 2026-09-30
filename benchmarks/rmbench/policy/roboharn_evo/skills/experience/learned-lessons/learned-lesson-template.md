# Lesson: <short rule-like title>

## Metadata

- Lesson ID: `<lesson_id>`
- Status: `candidate | accepted | deprecated`
- Confidence: `low | medium | high`
- Support Count: `<number>`
- Opposing Count: `<number>`
- Last Validated At: `<ISO timestamp or empty>`

## Applies When

- OOD_scenario: `<scenario>`
- Task family: `<pick_and_place | open_drawer | close_drawer | other>`
- Preconditions:
  - `<condition 1>`
  - `<condition 2>`

## Recommendation

Describe the preferred recovery workflow or post-recovery intent.

Example:

> When `OOD_scenario = grasp_lost` and the target object remains visible and reachable, prefer `recover-grasp-lost` followed by `retry_same_subtask`.

## Avoid

Describe actions that were ineffective or risky in similar cases.

## Supporting Cases

- `<case_id_1>`
- `<case_id_2>`

## Opposing Cases

- `<case_id_that_failed_or_contradicted_the_rule>`

## Evaluation Result

- Status: `not_evaluated | passed | failed`
- Pass Rate: `<0.0-1.0>`
- Payloads Matched: `<number>`

## Promotion Criteria

This lesson should become `accepted` only after it is supported by multiple cases or a reviewed high-value case plus a regression check. A single successful recovery should normally stay as a case summary, not a learned lesson.

## Notes

This is an experience-derived heuristic, not a hard runtime rule. Deprecated lessons must remain in history but should not be retrieved into runtime prompts.
