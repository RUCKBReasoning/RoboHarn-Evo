# RoboHarn-Evo Experience Library

`experience/` is the Heuristic Learning layer for RoboHarn-Evo. It stores what happened in real rollouts, compresses useful cases, promotes reviewed lessons, and provides compact retrieval context for recovery planning.

It is not an executable skill library. It must not directly control the robot.

## Data Flow

```text
ImgAgent trace / rollout events
  -> raw-traces/*.jsonl
  -> case-summaries/*.md
  -> learned-lessons/*.md candidate lessons
  -> regression evaluation
  -> learned-lessons/*.md accepted/deprecated lessons
  -> retrieval-index/index.jsonl
  -> retrieved_experience in /recover payload
```

## Layers

- `raw-traces/`: append-only factual ledgers. These are the source of truth and should keep event pointers back to the original trace.
- `case-summaries/`: factual compression of one case. Do not write generalized policy rules here.
- `learned-lessons/`: cross-case heuristics. Only lessons with `Status: accepted` can be retrieved into runtime prompts.
- `retrieval-index/`: metadata used by deterministic v1 retrieval.

## Grounding State

Recovery-related ledgers may include a compact `robot_state` snapshot at the
OOD/recovery decision point:

```text
left/right xyz, quat_wxyz, rpy, gripper, joint_norm, step
```

This is grounding evidence for comparing manipulation states. It should remain
factual and low dimensional; raw images and full action streams stay in rollout
dumps, not learned lessons.

## Continuous Learning Lifecycle

The v1 experience workflow uses online factual logging and offline experience internalization:

```text
online runtime:
  append traces and rollout events only

offline learner:
  extract trials -> summarize cases -> mine candidate lessons / avoid patterns
  -> evaluate candidates on recovery regression payloads
  -> promote accepted lessons or deprecate harmful lessons

next runtime:
  retrieve accepted lessons and verified avoid patterns as advisory context
```

Do not promote a lesson inside the same episode that produced it. New evidence
should first become factual ledgers and candidate lessons, then pass regression
checks before it can affect runtime recovery planning.

## Outcome Taxonomy

Use only these outcome values in recovery ledgers:

```text
detect_only
recovery_success_retry
recovery_success_replan
recovery_failed
replan_without_tools
aborted
false_positive
false_negative
incomplete_trace
```

## Runtime Guardrails

- Raw traces do not enter `/recover` prompts directly.
- Candidate and deprecated lessons do not enter runtime prompts.
- Benchmark labels, annotation fields, and gold OOD labels are forbidden in retrieval payloads.
- Retrieved experience is advisory context only; recovery tool names and args are still validated by runtime allowlists and current adapter capabilities.

## Offline Commands

Extract ledgers from a trace:

```bash
python policy/roboharn_evo/scripts/extract_experience_from_trace.py \
  --trace path/to/agent_trace.jsonl
```

Generate factual case summaries:

```bash
python policy/roboharn_evo/scripts/summarize_experience_cases.py --overwrite
```

Build retrieval index:

```bash
python policy/roboharn_evo/scripts/build_experience_index.py --overwrite
```

Validate records and leakage rules:

```bash
python policy/roboharn_evo/scripts/validate_experience_records.py --strict
```

Build regression payloads from extracted recovery trials:

```bash
python policy/roboharn_evo/scripts/build_recovery_regression_payloads.py --overwrite
```

Mine candidate lessons and avoid patterns:

```bash
python policy/roboharn_evo/scripts/mine_experience_lessons.py --overwrite
```

Evaluate candidate lessons:

```bash
python policy/roboharn_evo/scripts/eval_experience_lessons.py --overwrite
```

Promote candidates that pass evaluation:

```bash
python policy/roboharn_evo/scripts/promote_experience_lesson.py
```

Run the full offline cycle:

```bash
python policy/roboharn_evo/scripts/run_experience_learning_cycle.py \
  --trace path/to/agent_trace.jsonl \
  --promote \
  --strict
```
