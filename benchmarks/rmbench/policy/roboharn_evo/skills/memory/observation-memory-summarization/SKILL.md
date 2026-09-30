---
name: observation-memory-summarization
description: "Memory harness for narrowing preprocessed robot observations into committed task state, scene focus, uncertainty, and next-step context."
runtime_role: memory_harness
---

# Observation Memory Summarization

Use this harness when converting raw observation, segmentation, grounding, scene memory, monitor signals, recovery history, and previous committed memory into compact memory for the next control turn.

Use the existing runtime memory surfaces:

- `memory_text` for committed task-state facts.
- `recent_observation_summary` for compact current-observation and execution-state text.
- `scene_memory` for stable instances, task focus, temporal tracks, and grounding.
- `observation_preprocess` for raw SAM/RGB-D/query evidence.
- `recovery_history` for prior recovery tool calls and outcomes.
- `semantic_tags` for retrieval/recovery matching when already provided.

Within `scene_memory.instances[*]`, preserve the runtime-provided position
contract rather than re-deriving it from visibility:

- `position_state=current_verified`: a current calibrated observation or
  verified object-to-TCP propagation supports the coordinate.
- `position_state=memory_valid`: the object is currently occluded, but no
  physical event has invalidated its last verified coordinate.
- `position_state=motion_uncertain`: contact, failed grasp, collision, or
  unverified release may have moved it; the historical coordinate is not
  executable.
- `last_verified_world_m`, `last_verified_score`, `last_verified_step`, and
  `position_source` describe the retained evidence.

## Responsibilities

- Preserve only task-relevant facts supported by the current observation, scene memory, monitor result, recovery history, or previous committed memory.
- Use observation preprocessing as evidence, not as the final decision. Segmentation and RGB-D grounding identify candidates, positions, confidence, and uncertainty.
- Treat `scene_memory.task_focus` as VLM-proposed focus from perception-query roles, not as a hard rule. Revise it when later visual evidence contradicts it.
- Keep object-instance bindings stable across frames when geometry/history is stable, even if segmentation rank, score, or wording changes.
- Summarize execution consequences from `recovery_history`, `robot_state`, and latest tool results so the next recovery/planning turn does not restart from raw JSON.
- Explicitly carry uncertainty when perception is weak, missing, contradictory, or only retained from history.
- Do not claim task progress unless the current observation, monitor signal, or verified action outcome supports it.
- Do not store future plans as completed state.

## Existing-Field Execution Digest

When recovery or tool control is active, build a compact execution digest using existing fields. The digest should be text, not a new schema.

Include the following when supported:

- Focus binding: current subtask, selected target/tool instance IDs, and `track_id` if present.
- Grounding validity: whether selected instances have finite `world_m`, `approach_world_m`, `grasp_world_m`, and `contact_world_m`.
- Instance quality: `visible` versus `tracked`, the explicit
  `position_state`, `stable` versus `missing_current_frame`, low score, null
  bbox/mask, or camera fallback.
- Robot state: active arm pose at a high level and gripper state from `robot_state`; do not infer holding from a closed gripper alone.
- Last recovery consequence: the last tool call sequence, success/failure, target instance, point key, and concrete failure reason.
- Action constraint: what must not be assumed or repeated next, based on evidence. Example: "closed after approach point; grasp not confirmed" or "reobserve repeated with no finite grounding."

Do not add durable fields such as `manipulation_phase` or `execution_state`. Put this information into `recent_observation_summary`, `memory_text` when it is a committed fact, or the planner/recovery reasoning for the current turn.

## Output Style

### `memory_text`

- One concise committed-state sentence for the executor and next planner turn.
- Preserve facts that remain true across steps, such as object order, completed subtasks, stable target/tool binding, or verified failures.
- Include active target/tool instance only when relevant and supported, for example `target=green_block_01` or `tool=lid_left`.
- Include metric grounding only when useful for action or disambiguation, using compact units such as centimeters.
- Include uncertainty in short form, for example `uncertain: low segmentation score but stable 3D position`.
- Do not turn intended next actions into completed memory.

Good examples:

```text
Green, red, and blue blocks are arranged left to right; current focus is target=green_block_01 and tool=lid_left with finite 3D grounding.
```

```text
Current focus target=green_block_01/tool=lid_01 is visible but ungrounded; recent reobserve did not recover finite approach_world_m.
```

### `recent_observation_summary`

When producing or revising an observation summary, prefer a compact, scanable structure:

```text
focus=target:... tool:...; grounding=...; robot=...; recovery=...; uncertainty=...
```

Keep it short enough for repeated control turns. Include only evidence that changes the next decision.

Examples:

```text
focus=target:green_block_01 tool:lid_left; grounding=tool has approach/grasp points, target has approach point; robot=left gripper open; recovery=last move approached tool, no grasp yet; uncertainty=lid score low but track stable.
```

```text
focus=target:green_block_01 tool:lid_01; grounding=target/tool visible but world_m and approach_world_m are null; recovery=3 repeated reobserve calls produced no new grounding; constraint=do not call grounded move until finite point exists.
```

## Evidence Priority
1. Environment success/check signals.
2. Verified physical events and manipulation state, including attachment and
   release verification.
3. Current visual observation and monitor evidence after world-coordinate
   association rejects identity-incompatible candidates.
4. Current observation preprocessing: segmentation, RGB-D grounding, and scene memory.
5. Recovery tool results and `recovery_history`.
6. Previous committed memory.
7. Learned experience.

When evidence conflicts, prefer current grounded geometry over old text memory, but preserve uncertainty instead of silently overwriting.

## Recovery-Aware Rules

- If `move_ee_to_grounded_instance(point_key=approach_world_m)` was followed by `close_gripper`, do not summarize this as a successful grasp. Write that grasp is unconfirmed unless there is motion or monitor evidence.
- If selected target/tool has `world_m = null` or `approach_world_m = null`, summarize it as ungrounded even if the instance is labelled `visible`.
- If a selected instance is `tracked` or `missing_current_frame`, use
  `position_state` rather than visibility alone. `memory_valid` retains a
  physically valid coordinate for a safe approach; `motion_uncertain` forbids
  grounded motion until reacquisition.
- If the same failure repeats, make the repetition explicit: `repeated no finite approach_world_m`, `repeated reobserve no new grounding`, or `same tool call failed`.
- If gripper is closed, only state `holding_confirmed` when there is direct evidence that the object moved with the end effector or a verification tool/monitor confirmed it.
- If no progress is supported, say so. Do not convert action attempts into task progress.

## Guardrails
- Keep semantic interpretation in the VLM decision. Python code should only transport fields and maintain deterministic geometry/history bookkeeping.
- If the current observation is insufficient, write what is known and what is uncertain instead of inventing object roles.
- If multiple instances exist, refer to stable instance IDs or VLM-provided instance hints rather than a raw SAM rank.
- Do not invent fields beyond the runtime-provided position and manipulation
  contracts. Improve the compact text written into existing memory surfaces.
- Do not hide uncertainty to make the state look cleaner. Recovery depends on knowing what is ungrounded, unverified, or stale.
