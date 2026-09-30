# RMBench agent policy

`deploy_policy.py` connects RMBench observations and actions to the shared
RoboHarn-Evo runtime under `roboharn_evo/`. Use `configs/rmbench_deploy_policy.yaml` from
the project root to configure the planner, executor, perception, and knowledge
components.

Service entry points are `scripts/serve_rmbench_agent_api.py`,
`scripts/serve_rmbench_pi05.py`, and `scripts/serve_rmbench_sam3.py` at the
project root. Each accepts `--help` without starting a model. Provide model
checkpoints, assets, and credentials in your own environment.

Run evaluations through `scripts/run_rmbench.py` or
`scripts/run_rmbench_parallel.py`. Task success remains defined by RMBench.
See the project README and `benchmarks/rmbench/README.md` for resource and
evaluation requirements.
