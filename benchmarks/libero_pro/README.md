# LIBERO-PRO evaluation

This integration contains the `rpent-liberopro` 0.1.1 application, task
definitions, and packaged initial states. Asset resolution and imports use
explicit resource locations. Third-party attribution is in `NOTICE`; the code
license is in `rpent_liberopro-0.1.1.dist-info/licenses/LICENSE`.

## Environment and interfaces

Install the dependencies in `requirements.txt` and provide licensed external
assets containing `scenes/`, `stable_hope_objects/`, and
`stable_scanned_objects/`. Set `LIBERO_PRO_ASSETS_ROOT` or use the runner's
`--assets-root` argument. Custom tasks may require additional object assets.

The native interface uses one Panda robot and a seven-dimensional action:
translation, rotation, and gripper. The default cameras are `agentview` and
`robot0_eye_in_hand`. Benchmark goal predicates determine success. The adapter
preserves the native action contract and filters observations through its
declared public input fields.

`roboharn_evo/benchmark_adapters/libero_pro/native_policy.py` defines the OpenPI policy
projection. `native_rollout.py` implements native policy execution. The
benchmark-neutral agent loop passes planner subtasks to the native executor;
knowledge conditioning is implemented in the corresponding adapter modules.

## Task discovery

Run from the project root:

```bash
python scripts/run_libero_pro.py --list-suites
python scripts/run_libero_pro.py --list-tasks --suite libero_spatial_swap
python scripts/run_libero_pro.py --dry-run --suite libero_spatial_swap --task-id 0 --seed 0
python scripts/run_libero_pro.py --help
```

The integration exposes 16 PRO suites with ten task definitions each. Four
upstream initial-state sets are empty and are rejected by environment setup.
The remaining task definitions require a working simulator and external assets
before execution can be validated.

## Execution

Use `--smoke-reset` for an explicit simulator reset, `--action-json` for a
caller-supplied native action, and the documented native-policy or agent-loop
options for episodes. Supply the checkpoint, model configuration, policy source,
and its applicable license information. Generated trajectories and reports go
under `eval_result/`.

Use an external process timeout for simulator runs. The upstream reset loop can
retry without returning when its dependencies or assets fail. Missing terminal
results must not be interpreted as successful episodes.
