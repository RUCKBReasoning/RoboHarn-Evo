# RMBench evaluation

`envs/` contains task definitions and benchmark success checks.
`script/eval_policy.py` runs episodes; `policy/roboharn_evo/deploy_policy.py` connects
observations and actions to the shared `roboharn_evo` runtime.

## Resources

Install SAPIEN and the dependencies required by the selected motion planner,
model executor, and segmentation service. Supply a licensed asset directory
containing `embodiments/` and `objects/`:

```bash
export RMBENCH_ASSETS_ROOT=/path/to/RMBench/assets
```

Model checkpoints and demonstration data are external. API credentials are
configured in the selected service environment. Runtime outputs must remain
under this project's `eval_result/` directory.

## Commands

Run from the project root:

```bash
python scripts/run_rmbench.py --list-tasks
python scripts/run_rmbench.py --dry-run --task cover_blocks --seed 0
python scripts/serve_rmbench_agent_api.py --help
python scripts/serve_rmbench_pi05.py --help
python scripts/serve_rmbench_sam3.py --help
```

After configuring the required services and assets, start an episode with
`scripts/run_rmbench.py --run`, supplying the task, seed, deployment
configuration, asset root, and output directory. `scripts/run_rmbench_parallel.py`
accepts a worker/job configuration such as `configs/rmbench_parallel.example.yaml`.

## Knowledge and experiment definitions

`experiments/` contains the frozen source/evaluation splits and method settings.
The original and clarified-instruction conditions have separate YAML files and
instruction directories. `scripts/build_cover_blocks_hpk.py` prepares RGB,
actions, and measured robot states; `scripts/run_hpk_rgb_source_reflection.py`
and `scripts/build_hpk_rgb_source_store.py` perform extraction and organization.
`scripts/export_cover_blocks_hpk.py` verifies and exports a knowledge package.

## Scoring

The task's `check_success()` and sticky `eval_success` determine success.
`max_reward` may describe partial progress. Action completion and model-reported
completion remain separate from the task's terminal score. Task limits and
configured `step_limit_mode` determine the effective execution budget.

See `NOTICE` and `LICENSE` for attribution and redistribution terms.
