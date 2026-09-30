#!/usr/bin/env python3
"""Run one Harness VLA cell through RMBench's existing evaluator and scorer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from roboharn_evo.benchmark_adapters.rmbench.harness_vla.campaign import (
    freeze_reference_memory, reference_history, remaining_reference_budget, write_source_receipt,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rmbench_harness_vla.yaml")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=["development", "bootstrap", "evaluation"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--reference-history", type=Path, action="append", default=[],
                        help="Earlier completed cells for this same reference seed; deduct their usage before continuing. Repeat for every prior session.")
    parser.add_argument("--assets-root", type=Path, default=Path("/path/to/RMBench/assets"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    profile = yaml.safe_load(args.config.read_text())
    if "ray_tracing_denoiser" in profile:
        os.environ["RMBENCH_RAY_TRACING_DENOISER"] = profile["ray_tracing_denoiser"]
    if profile.get("network_proxy"):
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[name] = profile["network_proxy"]
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    protocol = yaml.safe_load((ROOT / profile["evaluation_protocol"]).read_text())
    tasks = {item["name"]: item for item in protocol["tasks"]}
    if args.task not in tasks:
        parser.error("task is outside the frozen six-task matrix")
    if args.phase == "evaluation" and args.seed not in tasks[args.task]["evaluation_seeds"]:
        parser.error("seed is outside the existing paired evaluation matrix")
    if args.phase == "bootstrap" and args.seed != profile["bootstrap"]["reference_seed"]:
        parser.error("bootstrap uses one fixed reference seed, not a trajectory quota")
    if args.phase != "evaluation" and args.seed in tasks[args.task]["evaluation_seeds"]:
        parser.error("development/bootstrap cannot consume held-out evaluation seeds")
    if args.phase == "evaluation" and not args.dry_run:
        if not (args.memory / "reference_coverage.json").is_file():
            frozen_memory = args.output / "frozen_memory"
            freeze_reference_memory(args.memory, frozen_memory, [args.task])
            args.memory = frozen_memory
    budget = dict(profile["bootstrap" if args.phase == "bootstrap" else "evaluation"])
    if args.reference_history:
        if args.phase != "bootstrap":
            parser.error("reference continuation is unavailable during development or evaluation")
        budget = remaining_reference_budget(budget, args.task, args.seed,
                                            [reference_history(path) for path in args.reference_history])
    from benchmarks.rmbench.integration import build_run_config, run_evaluation
    config = build_run_config(config_path=args.config, task_name=args.task, task_config=protocol["task_config"],
                              seed=0, output=args.output, checkpoint_setting="harness_vla_gpt55_high",
                              instruction_type=protocol["instruction_type"], instruction_set=protocol["instruction_set"])
    config.update(policy_name="roboharn_evo.benchmark_adapters.rmbench.harness_vla.policy", eval_video_log=False,
                  eval={"test_num": 1, "start_seed": args.seed, "step_limit": budget["action_limit"],
                        "step_limit_mode": budget.get("step_limit_mode", "cap"),
                        "exact_seed_fail_closed": True},
                  data_type={"rgb": True, "depth": True, "endpose": True, "qpos": True,
                             "third_view": True, "pointcloud": False})
    config["harness_vla"] = {**profile, **budget, "phase": args.phase,
                              "output_dir": str(args.output.resolve()), "memory_dir": str(args.memory.resolve()),
                              "reference_history": [str(path.resolve()) for path in args.reference_history]}
    if args.dry_run:
        print(json.dumps({"task": args.task, "seed": args.seed, "phase": args.phase,
                          "policy": config["policy_name"], "instruction_set": config["instruction_set"],
                          "budget": budget, "model": profile["model"], "effort": profile["reasoning_effort"],
                          "memory_source": "RPent reference-seed exploration"}, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    args.memory.mkdir(parents=True, exist_ok=True)
    if args.phase in {"development", "bootstrap"} and not (args.memory / "MEMORY.md").exists():
        (args.memory / "MEMORY.md").write_text("# Global Memory\n\nNo RMBench experience has been recorded yet.\n")
    (args.output / "run_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    run_evaluation(config, assets=args.assets_root, output=args.output)
    if args.phase == "bootstrap":
        write_source_receipt(args.memory, args.output, args.task, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
