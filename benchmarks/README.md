# Benchmark integrations

`rmbench/` and `libero_pro/` contain benchmark applications and
task definitions. `roboharn_evo/benchmark_adapters/` translates observations, actions,
and feedback into the shared interfaces. Root `scripts/run_*.py` entry points
compose the benchmark, model services, and agent configuration.

Use benchmark applications from the source checkout. The `roboharn-evo` wheel
contains the shared Python implementation and its runtime configuration and
prompt resources. Simulation assets, checkpoints, and external datasets are
configured separately.

Each benchmark defines its own success and termination conditions. Consult its
README for runtime requirements and its NOTICE or SOURCE document for public
attribution. All supplied third-party license files remain applicable.
