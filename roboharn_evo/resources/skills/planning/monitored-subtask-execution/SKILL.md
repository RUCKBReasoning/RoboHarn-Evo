---
name: monitored-subtask-execution
description: "Execution-unit skill for one monitored subtask rollout. Use this skill when one executor-facing instruction should be run as a monitored action unit."
---

# Monitored Subtask Execution

Use this skill to execute exactly one narrow subtask under a monitored boundary.

## Responsibilities
- run one executor-facing instruction
- monitor for success/failure/stall/timeout
- stop or reset deterministically if needed
- return a structured execution result to the calling workflow skill
