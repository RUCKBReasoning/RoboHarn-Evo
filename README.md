<div align="center">

# RoboHarn-Evo
### Evolving Hierarchical Physical Knowledge for Self-Improving Robotic Manipulation

**Turn physical experience into reusable knowledge.**

[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.37583)
[![Project Page](https://img.shields.io/badge/Project_Page-143F3B?style=for-the-badge&logo=googlechrome&logoColor=white)](https://ruckbreasoning.github.io/RoboHarn-Evo/)
[![Demos](https://img.shields.io/badge/Real_Robot-Demos-477F67?style=for-the-badge&logo=youtube&logoColor=white)](https://ruckbreasoning.github.io/RoboHarn-Evo/#demos)
[![License](https://img.shields.io/badge/License-Apache_2.0-52616B?style=for-the-badge)](LICENSE)

</div>

<details>
<summary>Full affiliations & contributions</summary>

**Shifeng Bao**<sup>1,3*</sup>, **Fanding Huang**<sup>2*</sup>,
**Yihan Lin**<sup>1,3</sup>, **Youhe Feng**<sup>1,3</sup>, **Guanlin Li**<sup>1,3</sup>,
**Chen Zhao**<sup>1,3</sup>, **Yang Li**<sup>1,3</sup>, **Jiawei He**<sup>5</sup>,
**Cheng Chi**<sup>1,4‡</sup>, **Jing Zhang**<sup>1,4†</sup>

<sup>1</sup> School of Information, Renmin University of China<br>
<sup>2</sup> Tsinghua University<br>
<sup>3</sup> Key Laboratory of Data Engineering and Knowledge Engineering, Beijing, China<br>
<sup>4</sup> Engineering Research Center of Database and Business Intelligence, Beijing, China<br>
<sup>5</sup> XYZ Embodied AI, Beijing, China

\* Equal contribution. † Corresponding author. ‡ Project leader.

</details>

<p align="center">
  <a href="https://ruckbreasoning.github.io/RoboHarn-Evo/#demos">
    <img src="https://ruckbreasoning.github.io/RoboHarn-Evo/assets/images/task-3.jpg" alt="Watch RoboHarn-Evo on a real robot: uncover blocks, count them, and press the matching buttons" width="100%">
  </a>
  <br>
  <a href="https://ruckbreasoning.github.io/RoboHarn-Evo/#demos"><strong>Watch the real-robot demonstrations →</strong></a>
</p>

## 🔥 News

- **2026.10.05:** Our [project page](https://ruckbreasoning.github.io/RoboHarn-Evo/) is live, with three real-robot demonstrations, method figures, and interactive results.

## 🧭 Overview

**Physical experience can improve what a robot knows.** RoboHarn-Evo develops **Hierarchical Physical Knowledge (HPK)** through interaction, with the vision-language model and low-level executor held fixed.

- **Task Knowledge** guides subtask selection, ordering, goals, and completion conditions.
- **Action Knowledge** guides object-relative geometry, interaction strategies, and physical effects.

An **inner execution loop** retrieves both levels of knowledge and grounds actions in the current scene. An **outer knowledge-update loop** uses recorded physical outcomes to revise, consolidate, and organize knowledge for later episodes. Action effects and subtask completion are assessed separately.

<p align="center">
  <img src="https://ruckbreasoning.github.io/RoboHarn-Evo/assets/images/method.png" alt="RoboHarn-Evo architecture: the inner loop uses Task and Action Knowledge during execution; the outer loop updates knowledge from physical evidence" width="100%">
</p>

## 📖 Abstract

Vision-language models can coordinate long-horizon robot manipulation, yet successful task reasoning still depends on whether local physical interactions produce the intended effects. We study how repeated interaction can improve this capability without updating the base model. We introduce **RoboHarn-Evo**, a dual-loop harness that evolves **Hierarchical Physical Knowledge (HPK)** from physical experience. HPK couples two levels of reusable knowledge: Task Knowledge captures which subtask should be executed and when it is complete, while Action Knowledge captures object-relative geometric strategies and their physical effects. During execution, the agent retrieves knowledge at the corresponding decision level and grounds it in the current scene under the task goal. Across episodes, physical feedback is used to revise historical knowledge, update its applicability, and organize reusable entries for subsequent retrieval. Experiments on RMBench show that HPK improves average success by up to **24.2 percentage points** across different agent models. With 80 interaction rollouts, held-out success rises from **48.3% to 75.0%** for GPT-5.5 and from **70.0% to 88.3%** for GPT-6. RoboHarn-Evo also resolves over 83% of historical knowledge errors while retaining 95.8% of valid knowledge, and transfers zero-shot from RMBench to RoboDojo with gains of 35.0 and 25.0 percentage points. These results demonstrate that physical interaction can be accumulated into reusable knowledge for improving subsequent manipulation.

## 🧩 Method

1. **Retrieve knowledge at each decision level.** Skill summaries route queries to related entries; Task Knowledge guides planning, while Action Knowledge supports action grounding.
2. **Ground actions in the current scene.** Object-relative strategies and the task goal constrain motion targets and expected effects.
3. **Check physical outcomes.** Execution evidence distinguishes the effect of an action from completion of the broader subtask.
4. **Update knowledge across episodes.** Reflect on trajectories, consolidate related entries, and revise their content and applicability using supporting, opposing, or unresolved evidence.

<details>
<summary>Skill-routed retrieval and evidence-driven maintenance</summary>

<p align="center">
  <img src="https://ruckbreasoning.github.io/RoboHarn-Evo/assets/images/knowledge.png" alt="Skill summaries route knowledge retrieval, while evidence from trajectories supports incremental knowledge maintenance" width="100%">
</p>

</details>

## 📊 Results

**RMBench: six-task mean success.** Each model is compared with and without HPK using the same executor. The paper reports 20 held-out episodes per task.

| Model | Without HPK | Full HPK | Gain |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B | 19.2% | **41.7%** | +22.5 pp |
| GPT-5.5 | 58.3% | **82.5%** | +24.2 pp |
| GPT-6 | 72.5% | **91.7%** | +19.2 pp |

**Continued interaction.** In the separate learning experiment, knowledge evolves over 80 source rollouts. Held-out evaluation uses frozen knowledge snapshots and contributes no feedback to updates.

| Model | Before interaction | After 80 rollouts |
| --- | ---: | ---: |
| GPT-5.5 | 48.3% | **75.0%** |
| GPT-6 | 70.0% | **88.3%** |

Values are mean held-out success over three learning histories. After 80 rollouts, 83.3% / 91.7% of initial knowledge errors are repaired or deactivated for GPT-5.5 / GPT-6, while both retain 95.8% of initially correct entries.

**Transfer to physical robots.** Across three tasks with eight scenes per task, mean success rises from **41.7% to 62.5%** after real-world knowledge updates; mean normalized task progress rises from **70.5% to 84.2%**. These paper statistics are separate from the illustrative recordings below. Frozen knowledge also transfers from RMBench to RoboDojo, improving zero-shot success by **35.0 pp** for GPT-5.5 and **25.0 pp** for GPT-6.

See the [paper](https://arxiv.org/abs/2609.37583) and [interactive results](https://ruckbreasoning.github.io/RoboHarn-Evo/#results) for task-level comparisons and evaluation details.

## 🎬 Real-Robot Demonstrations

| Task | Demonstration | Video |
| --- | --- | --- |
| Cover blocks | Apply simulation-derived knowledge to real-world covering | [Watch MP4](https://ruckbreasoning.github.io/RoboHarn-Evo/assets/videos/task-1.mp4) |
| Press by number | Use accumulated experience for ordered button presses | [Watch MP4](https://ruckbreasoning.github.io/RoboHarn-Evo/assets/videos/task-2.mp4) |
| Uncover, count & press | Reuse knowledge in a composed manipulation task | [Watch MP4](https://ruckbreasoning.github.io/RoboHarn-Evo/assets/videos/task-3.mp4) |

Clips are edited excerpts at **3× recorded speed**, with waiting intervals omitted. They illustrate knowledge use rather than a controlled success-rate comparison. The [project page](https://ruckbreasoning.github.io/RoboHarn-Evo/#demos) provides chapters, knowledge summaries, and recorded outcome qualifications, including unresolved automatic verification of the blue-button effect in the third task.

## 🛠️ Installation

Use Python **3.10 or newer** in the environment for your selected benchmark:

```bash
git clone https://github.com/RUCKBReasoning/RoboHarn-Evo.git
cd RoboHarn-Evo
python -m pip install -e '.[api,expert]'
```

The distribution name is `roboharn-evo`; the import namespace is `roboharn_evo`. Simulator dependencies, model checkpoints, SAM3 source and weights, and licensed simulation assets are installed separately. Benchmark applications run from the source checkout.

Consult the [benchmark overview](benchmarks/README.md), [RMBench setup](benchmarks/rmbench/README.md), and [LIBERO-PRO setup](benchmarks/libero_pro/README.md) for their respective runtime requirements.

## 🧪 Evaluation and Configuration

### Discover tasks and inspect a run

These commands do not start a simulator:

```bash
python scripts/run_rmbench.py --list-tasks
python scripts/run_libero_pro.py --list-suites
python scripts/run_rmbench.py --dry-run --task cover_blocks --seed 0
```

After configuring the required services, model resources, and assets, use `scripts/run_rmbench.py --run` with explicit task, seed, deployment configuration, asset root, and output arguments. Runtime outputs are written under `eval_result/`.

### Model and segmentation services

Inspect provider, model, checkpoint, and endpoint arguments with:

```bash
python scripts/serve_rmbench_agent_api.py --help
python scripts/serve_rmbench_pi05.py --help
python scripts/serve_rmbench_sam3.py --help
```

Set `RMBENCH_ASSETS_ROOT` to the licensed asset directory. Credentials belong in environment variables or external credential files; [the key-pool example](configs/agent_api_key_pool.example.json) shows multiple independently configured keys. Loopback addresses refer to local services; replace `/path/to/` placeholders with installed resources.

Cover Blocks service orchestration additionally uses `RMBENCH_PYTHON`, `SAM3_PYTHON`, `SAM3_REPO`, `SAM3_CHECKPOINT`, `SAM3_BPE_PATH`, `ROBOHARN_EVO_PROVIDER_AUTH_FILE`, and `ROBOHARN_EVO_PROVIDER_CONFIG_FILE`.

### Knowledge use and experiment protocols

| Setting | Purpose |
| --- | --- |
| `agent.hpk_v3.mode` | Select HPK usage mode |
| `knowledge_updates_enabled` | Control persistent knowledge updates |
| `agent.recovery.enable_execution_evidence` | Enable execution evidence; disabling it also disables online knowledge updates and reflection |

**The supplied deployment configuration currently disables execution evidence.** Select explicit experiment settings when evaluating verified maintenance or continued learning.

[RMBench experiment definitions](benchmarks/rmbench/experiments/) specify source/evaluation seeds and Off, flat reflection, Task-only, Action-only, and Full conditions. Definitions retain their own budgets, instruction conditions, and split sizes; use the intended protocol rather than treating every YAML as the paper's main evaluation.

- [Hierarchical extraction evaluation](scripts/evaluate_rmbench_hpk_rq1.py)
- [Knowledge retrieval evaluation](scripts/evaluate_hpk_v3_family_retrieval.py)
- [Incremental knowledge maintenance](scripts/maintain_hpk_v3_family_store.py)

## 🗂️ Repository Structure

```text
roboharn_evo/
  agent/hpk/          Knowledge extraction, storage, retrieval and maintenance
  agent/recovery/     Tool dispatch, action-effect verification and recovery
  agent/vla/          Low-level executor interfaces
  models/            Planner and model interfaces
  services/          Model and segmentation service entry points
  benchmark_adapters/ Shared observation, action and feedback interfaces
  resources/         Agent configuration and planning/execution prompts
benchmarks/          RMBench and LIBERO-PRO applications and task definitions
configs/             Service and worker configuration examples
scripts/             Runners, services, knowledge tools and evaluations
```

<details>
<summary>Implementation map</summary>

- `roboharn_evo/agent/runtime.py` constructs the agent; `agent/core/img_agent.py` implements the interaction and control loops.
- Under `agent/hpk/`, `hierarchical_knowledge.py` defines trajectory packages and Task/Action atomization; `vlm_hierarchical_reflector.py` interprets trajectories with the model.
- `hierarchical_store.py`, `semantic_consolidator.py`, and `family_store.py` organize persistent knowledge and its Skill index; `incremental_maintainer.py` and `rgb_maintenance.py` perform evidence-aware maintenance.
- `hierarchical_retriever.py`, `family_router.py`, and `rgb_retrieval.py` retrieve knowledge; `goal_consistency.py` connects subtask goals to action realizations.
- `roboharn_evo/agent/grasp_attachment_contract.py` handles grasp evidence. Prompts for planning, perception, memory, and execution are in `roboharn_evo/resources/skills/`.
- HPK configuration uses the `agent.hpk_v3` namespace.

</details>

## 📝 Citation

If you use RoboHarn-Evo in your research, please cite:

```bibtex
@misc{bao2026roboharnevo,
  title={RoboHarn-Evo: Evolving Hierarchical Physical Knowledge for Self-Improving Robotic Manipulation},
  author={Bao, Shifeng and Huang, Fanding and Lin, Yihan and Feng, Youhe and Li, Guanlin and Zhao, Chen and Li, Yang and He, Jiawei and Chi, Cheng and Zhang, Jing},
  year={2026},
  eprint={2609.37583},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2609.37583}
}
```

## 📄 License and Third-Party Attribution

RoboHarn-Evo original code is licensed under [Apache-2.0](LICENSE). Copyright 2026 Shifeng Bao.

Third-party code retains its original copyright notices and licenses; see the benchmark-specific LICENSE and NOTICE files. OpenPI source under `benchmarks/rmbench/policy/pi05/` retains its Apache-2.0 license. Model weights, simulation assets, and external datasets have their own license requirements.
