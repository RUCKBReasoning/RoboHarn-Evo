# RMBench adapter

`RMBenchAdapter` wraps the existing duck-typed RMBench task environment. It
converts a caller-supplied raw observation with the preserved
`RMBenchEnvAdapter.from_env`, executes exactly one explicitly requested action,
then reports benchmark-owned episode state. It contains no Guard, verifier,
recovery routing, task policy, expert trajectory, fixed coordinate, or action
sequence.

The application under `benchmarks/rmbench/` imports the independent
`roboharn_evo` runtime. Its evaluation entrypoint calls
`RoboHarnAgentRuntime.run_eval_step(task_env, observation)`, which delegates to
`AgentSession.run_eval_once`. The Agent's observation handling uses
`RMBenchEnvAdapter`. `RMBenchAdapter` provides the separate adapter API described
above.
