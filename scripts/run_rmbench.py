#!/usr/bin/env python3
"""Run the repository-owned RMBench integration.

Metadata, task discovery, and dry-run modes do not import SAPIEN or start a
rollout.  ``--run`` is the only mode that loads the copied evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


# Set this before importing any repository module so a source-checkout launch
# 避免在 RoboHarn-Evo 或 benchmark 源代码目录创建 __pycache__。
sys.dont_write_bytecode = True


if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.rmbench.integration import (  # noqa: E402
    build_run_config,
    discover_tasks,
    dry_run_report,
    run_evaluation,
    runtime_provenance,
)
from benchmarks.rmbench.paths import default_deploy_config, output_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RoboHarn-Evo RMBench runner (default: metadata-only dry-run)."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-tasks", action="store_true", help="list implemented task modules without importing the simulator")
    mode.add_argument("--provenance", action="store_true", help="print benchmark/RoboHarn-Evo/config provenance without importing the simulator")
    mode.add_argument("--dry-run", action="store_true", help="validate paths and configuration without importing the simulator")
    mode.add_argument("--run", action="store_true", help="start the real copied RMBench evaluator")
    parser.add_argument("--task", help="implemented RMBench task name")
    parser.add_argument("--config", type=Path, default=default_deploy_config(), help="RoboHarn-Evo deploy YAML")
    parser.add_argument("--task-config", default="demo_clean", help="task config name or YAML path")
    parser.add_argument("--seed", type=int, default=0, help="RMBench evaluator seed index")
    parser.add_argument("--output", type=Path, default=output_root(), help="evaluation output root")
    parser.add_argument("--assets-root", type=Path, help="external asset dir containing embodiments/ and objects/")
    parser.add_argument("--checkpoint-setting", default="roboharn_evo_agent", help="evaluation provenance label")
    parser.add_argument("--instruction-type", default="unseen", choices=("seen", "unseen"))
    parser.add_argument("--instruction-set", default="rmbench_original")
    return parser


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_tasks:
        _print_json({"benchmark": "rmbench", "tasks": list(discover_tasks())})
        return 0

    common = {
        "config_path": args.config,
        "assets": args.assets_root,
        "output": args.output,
    }
    if args.provenance:
        _print_json(runtime_provenance(**common))
        return 0

    if not args.run:
        _print_json(
            dry_run_report(
                **common,
                task_name=args.task,
                task_config=args.task_config,
                seed=args.seed,
            )
        )
        return 0

    if not args.task:
        parser.error("--run requires --task")
    config = build_run_config(
        config_path=args.config,
        task_name=args.task,
        task_config=args.task_config,
        seed=args.seed,
        output=args.output,
        checkpoint_setting=args.checkpoint_setting,
        instruction_type=args.instruction_type,
        instruction_set=args.instruction_set,
    )
    _print_json(runtime_provenance(**common))
    run_evaluation(config, assets=args.assets_root, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
