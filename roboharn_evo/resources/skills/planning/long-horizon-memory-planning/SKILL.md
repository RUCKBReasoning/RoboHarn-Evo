---
name: long-horizon-execution
description: "Workflow skill for multi-step long-horizon task execution. Use this skill when the task must be decomposed into ordered subtasks with monitoring and recovery."
---

# Long Horizon Execution

Use this workflow when a global task must be decomposed into ordered subtasks.

## Responsibilities
- refine the ordered subtask plan
- select the next subtask
- check whether the previous subtask is complete
- decide retry / recover / continue / finish
- delegate each concrete execution step to a monitored execution boundary instead of directly improvising low-level control
