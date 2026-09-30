# RoboHarn-Evo

RoboHarn-Evo develops Hierarchical Physical Knowledge (HPK) from robot
interaction while keeping the vision-language model and low-level executor fixed.
HPK connects two levels of reusable knowledge:

- **Task Knowledge** records when a subtask applies, its objective, the reason
  for selecting it, and the observations needed to establish completion.
- **Action Knowledge** records interaction conditions, object-relative geometry,
  and the expected and observed effects of an operation.

The inner execution loop retrieves knowledge for subtask planning and action
grounding, computes motions from the current scene, and records physical outcomes.
The outer knowledge-update loop reflects on completed trajectories, consolidates
related entries, and revises their content and applicability using execution
evidence. Skill summaries index related entries for retrieval and incremental
maintenance. Action effects and subtask completion are assessed separately.

Paper: [RoboHarn-Evo: Evolving Hierarchical Physical Knowledge for Self-Improving
Robotic Manipulation](https://arxiv.org/pdf/2609.37583).

## Installation

Run from this directory in the Python environment for your selected benchmark:

```bash
python -m pip install -e '.[api,expert]'
```

The distribution name is `roboharn-evo`; the Python import namespace is
`roboharn_evo`. Simulator dependencies, model checkpoints,
SAM3 source and weights, and licensed simulation assets are installed separately.
The benchmark applications under `benchmarks/` are used from the source checkout.

## Implementation

HPK implementation modules are under `roboharn_evo/agent/hpk/`; configuration uses the
`agent.hpk_v3` namespace.

- `roboharn_evo/agent/runtime.py` constructs the agent; `roboharn_evo/agent/core/img_agent.py`
  implements the interaction and control loops.
- `roboharn_evo/agent/hpk/hierarchical_knowledge.py` defines trajectory packages and
  Task/Action atomization. `vlm_hierarchical_reflector.py` performs model-based
  trajectory interpretation.
- `hierarchical_store.py`, `semantic_consolidator.py`, and `family_store.py`
  organize persistent Task/Action Knowledge and its Skill index. `incremental_maintainer.py` and
  `rgb_maintenance.py` implement evidence-aware maintenance.
- `hierarchical_retriever.py`, `family_router.py`, and `rgb_retrieval.py`
  retrieve knowledge for the current decision. `goal_consistency.py` connects
  subtask goals to action realizations.
- `roboharn_evo/agent/recovery/` implements tool dispatch, action-effect verification,
  and recovery. `roboharn_evo/agent/grasp_attachment_contract.py` handles grasp evidence.
- `roboharn_evo/models/` and `roboharn_evo/agent/vla/` define planner/executor interfaces.
  `roboharn_evo/services/` provides model and segmentation service entry points.
- `roboharn_evo/resources/skills/` contains the planning, perception, memory, and
  execution prompts used by the agent.


List available tasks without starting simulators:

```bash
python scripts/run_rmbench.py --list-tasks
python scripts/run_libero_pro.py --list-suites
```

RMBench experiment definitions are under `benchmarks/rmbench/experiments/`.
The frozen six-task comparison defines Off, flat reflection, Task-only,
Action-only, and Full conditions, including explicit source/evaluation seeds.
`scripts/evaluate_rmbench_hpk_rq1.py` evaluates hierarchical extraction;
`scripts/evaluate_hpk_v3_family_retrieval.py` evaluates retrieval; and
`scripts/maintain_hpk_v3_family_store.py` exposes incremental maintenance.
Experiment definitions retain their separate budgets and instruction conditions.

## Services and configuration

Service entry points are `scripts/serve_rmbench_agent_api.py`,
`scripts/serve_rmbench_pi05.py`, and `scripts/serve_rmbench_sam3.py`.
Use their `--help` output for provider, model, checkpoint, and endpoint arguments.
Credentials belong in environment variables or external credential files.
`configs/agent_api_key_pool.example.json` illustrates multiple independently
configured keys. Loopback addresses in example configurations refer to local
services. Replace `/path/to/` locations with your installed resources.

Set `RMBENCH_ASSETS_ROOT` to the licensed asset directory. Runtime outputs are
written under `eval_result/`. Cover Blocks service orchestration additionally
uses `RMBENCH_PYTHON`, `SAM3_PYTHON`, `SAM3_REPO`, `SAM3_CHECKPOINT`,
`SAM3_BPE_PATH`, `ROBOHARN_EVO_PROVIDER_AUTH_FILE`, and `ROBOHARN_EVO_PROVIDER_CONFIG_FILE`.

`agent.hpk_v3.mode` controls HPK use.
`knowledge_updates_enabled` controls persistent updates.
`agent.recovery.enable_execution_evidence` controls execution-evidence behavior;
disabling it also disables online knowledge updates and reflection. The supplied
deployment configuration currently disables that switch. Select explicit
experiment settings when evaluating verified maintenance or continued learning.

## License

RoboHarn-Evo original code is licensed under [Apache-2.0](LICENSE).
Copyright 2026 Shifeng Bao.

Third-party code retains its original copyright notices and licenses. See the
benchmark-specific LICENSE and NOTICE files. OpenPI source under
`benchmarks/rmbench/policy/pi05/` retains its Apache-2.0 license. Model weights,
simulation assets, and external datasets have their own license requirements.
