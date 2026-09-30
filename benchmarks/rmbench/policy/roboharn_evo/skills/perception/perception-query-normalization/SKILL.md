---
name: perception-query-normalization
description: "Normalize VLM perception queries into stable schema fields before segmentation and scene-memory binding."
---

# Perception Query Normalization

This skill converts raw perception-query objects into stable segmentation and scene-memory fields.

It is a semantic normalization step. The runtime must not use Python keyword lists to infer object classes, descriptors, or task roles.

## Inputs

- `global_task`: original task instruction.
- `current_subtask`: current active subtask, if any.
- `committed_memory`: compact task memory.
- `observation_summary`: current observation and scene-memory summary.
- `raw_queries`: list of raw query objects from the perception-query planner or user config.
- `scene_instances`: currently visible grounded candidates plus bounded retained tracks with exact `instance_id` and `track_id` identifiers. Retained tracks are marked `recovery_binding_only=true` when their current-frame geometry is missing or inconsistent.
- `oracle_objects`: optional simulator candidate catalog with exact `oracle_id` identifiers.
- `instance_binding_phase`: `candidate_discovery`, `catalog_selection`, or `post_detection_selection`.
- `require_instance_binding`: whether target/tool queries must bind an advertised candidate identity.
- `binding_requirement`: optional structured postcondition. `required_any_roles` lists roles of which at least one must be returned; `failure_reason`, `previous_queries`, `retry_attempt`, and `force_refresh` describe a rejected earlier result.
- `max_queries`: maximum number of normalized queries to return.

Each raw query may contain:

- `object_id`: free-form object or part phrase.
- `text_prompt`: free-form segmentation prompt.
- `role`: intended role, one of `target`, `tool`, or `context`.
- `entity_scope`: `single_instance` for an ordinary object or `reference_set` for a visual set that jointly defines a spatial relation.
- `placement_relation`: optional `center_of` relation requested from a `reference_set`.
- `expected_count`: optional positive member count when the instruction states it explicitly.
- `instance_hint`: optional instance-level disambiguation.
- `instance_ref`: optional exact `track_id` or `instance_id` selected from `scene_instances`.
- `oracle_id`: optional exact identity selected from `oracle_objects`.
- `reason`: short rationale.

## Output Schema

Return JSON only:

```json
{
  "queries": [
    {
      "object_id": "stable object or part category",
      "text_prompt": "segmentation text prompt",
      "role": "target | tool | context",
      "entity_scope": "single_instance | reference_set",
      "placement_relation": "center_of or omitted",
      "expected_count": "positive integer or omitted",
      "instance_hint": "instance-level descriptors or empty string",
      "instance_ref": "exact advertised track_id/instance_id or empty string",
      "oracle_id": "exact advertised oracle_id or empty string",
      "reason": "short rationale"
    }
  ]
}
```

## Rules

- Return at most `max_queries` queries.
- Keep `object_id` stable across frames for the same object or part category.
- Put instance-level disambiguation in `instance_hint`, not in `object_id`.
- Preserve descriptors in `text_prompt` when they help segmentation.
- `role` must be inferred from the task, current subtask, memory, and raw query rationale.
- If the current subtask explicitly requires placing an object at the center of, or between members of, a visual reference set, preserve one `role=context`, `entity_scope=reference_set`, `placement_relation=center_of` query. Do not calculate a coordinate in the query.
- A `reference_set` is not a request to choose one member identity. Keep `instance_ref` empty and use `expected_count` only when the instruction states the count. Make `text_prompt` a short member-level segmentation prompt, normally the singular category of one member; do not describe the whole plural scene in the prompt.
- For ordinary objects, use `entity_scope=single_instance` and omit `placement_relation` and `expected_count`.
- When `binding_requirement.required_any_roles` is non-empty, the result must contain at least one semantically supported query with a listed role. Returning only `context` does not satisfy a `target`/`tool` requirement.
- When `binding_requirement.force_refresh=true`, treat `previous_queries` as rejected output and use `failure_reason` to correct the role or identity omission. Do not simply repeat the rejected query set.
- Never relabel a reference object merely to pass the role postcondition. If the task context and supplied candidates do not support a confident required-role query, return an empty query list; the runtime owns bounded retry and terminal handling.
- When visible candidates are supplied, use the task, images, memory, and candidate geometry to select the intended instance semantically.
- During `candidate_discovery`, treat `scene_instances` as a partial catalog. If a task-supported target/tool object has no matching candidate, preserve its query with an empty `instance_ref` so segmentation can create candidates. Do not substitute an unrelated advertised instance just because the catalog is non-empty.
- During `catalog_selection` or `post_detection_selection`, when `require_instance_binding=true`, every returned target/tool query must select an exact advertised identity. This is the strict execution-facing phase.
- Prefer a stable `track_id` for `instance_ref`; otherwise use an exact advertised `instance_id`.
- A candidate marked `recovery_binding_only=true` may be selected by exact reference to preserve the intended identity for retreat/re-observation. Its stored geometry is not current task-action evidence and must not authorize grounded task progress until a clean observation makes it visible and stable again.
- Preserve or select an exact `oracle_id` when an oracle catalog is supplied. Never invent, shorten, or rewrite identifiers.
- When `require_instance_binding=true`, omit a target/tool query if no candidate can be selected confidently. Do not choose by candidate-list order. This restriction does not suppress a task-supported unbound query during `candidate_discovery`.
- Do not invent objects not supported by the raw queries or current task context.
- Do not merge queries with different task roles unless they clearly refer to the same category and the higher-priority role should be preserved.
- Prefer fewer stable queries over many near-duplicates.
- Do not output markdown, explanations, or keys outside the schema.

## Runtime Contract

The runtime validates schema, required-role postconditions, exact identifier membership, finite grounding, current visibility or bounded retained-track status, supported relation operators, and maximum count. A result that violates `required_any_roles` is rejected rather than silently accepted. It will not interpret spatial language or repair semantic mistakes with keyword rules. A valid `reference_set` relation is converted into a separately verified public placement target; the language model does not provide its metric pose. Grounded moves to recovery-only tracks are replaced by re-observation and the remaining static contact batch is stopped.
