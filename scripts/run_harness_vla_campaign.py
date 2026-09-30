#!/usr/bin/env python3
"""Execute a frozen baseline matrix; never substitute seeds or retry episodes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roboharn_evo.benchmark_adapters.rmbench.harness_vla.campaign import (
    collect_cell, plan, reference_history, remaining_reference_budget,
    freeze_reference_memory, stop_worker, wait_worker, write_summary,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rmbench_harness_vla.yaml")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=["bootstrap", "evaluation"], required=True)
    parser.add_argument("--run", action="store_true", help="Without this flag, print the matrix only.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluation-seeds-per-task", type=int,
                        help="Use the first N preregistered seeds per task; omitted means all seeds.")
    parser.add_argument("--reference-memory", type=Path,
                        help="Optional prior reference lessons to freeze for evaluation, including failures.")
    parser.add_argument("--tasks", nargs="+", help="Run only these tasks from the preregistered matrix, in protocol order.")
    parser.add_argument("--reference-history-root", type=Path, action="append", default=[],
                        help="Explicitly continue a repaired reference campaign, charging earlier usage. Bootstrap only.")
    args = parser.parse_args()
    if args.phase != "evaluation" and (args.evaluation_seeds_per_task is not None or args.reference_memory is not None):
        parser.error("evaluation subset and reference-memory options require --phase evaluation")
    frozen = plan(args.config, evaluation_seeds_per_task=args.evaluation_seeds_per_task, tasks=args.tasks)
    if args.reference_memory is not None:
        frozen["reference_memory_source"] = str(args.reference_memory.resolve())
    cells = [c for c in frozen["cells"] if c["phase"] == args.phase]
    histories = []
    if args.reference_history_root:
        if args.phase != "bootstrap":
            parser.error("reference histories cannot retry formal evaluation")
        for previous in args.reference_history_root:
            if previous.resolve() == args.root.resolve():
                parser.error("use a new campaign root after an infrastructure repair")
            for config in sorted(previous.rglob("run_config.yaml")):
                record = reference_history(config.parent)
                if record.get("phase") is None:
                    raise ValueError(f"prior cell has no auditable outcome/usage: {config.parent}")
                if record.get("phase") == "bootstrap":
                    histories.append((config.parent.resolve(), record))
    if not args.run:
        print(json.dumps({"phase": args.phase, "count": len(cells), "cells": cells}, indent=2))
        return 0
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = args.root / "campaign_plan.json"
    if manifest.exists() and json.loads(manifest.read_text()) != frozen:
        raise ValueError("campaign profile changed; do not mix protocols in the same output root")
    manifest.write_text(json.dumps(frozen, indent=2))
    profile = frozen["profile"]
    source_memory = args.root / "reference_memory"
    if histories and not source_memory.exists():
        # 仅传递当前基线自身的记忆。
        previous_memory = args.reference_history_root[-1] / "reference_memory"
        shutil.copytree(previous_memory, source_memory)
    memory = source_memory
    if args.phase == "evaluation":
        memory = args.root / "frozen_memory"
        if not memory.exists():
            freeze_reference_memory(args.reference_memory or source_memory, memory, {c["task"] for c in cells})
        elif not args.resume:
            raise ValueError("evaluation memory already frozen; use --resume without changing the protocol")
    records = []
    status = args.root / f"{args.phase}_status.json"
    process = None
    write_summary(args.root, records)

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        for ordinal, cell in enumerate(cells):
            # A run-local stop request is checked only between workers, so the
            # current episode and its video/result finalization can finish.
            if (args.root / "STOP_AFTER_CURRENT").exists():
                status.write_text(json.dumps({"stage": "stopped_after_current",
                                               "completed": len(records), "planned": len(cells),
                                               "next_cell_not_started": cell}, indent=2))
                return 0
            # NVRTC has a short temporary-directory-name limit. Keep physical
            # job roots short; the manifest/receipt retain full task and seed.
            directory = args.root / ("b" if args.phase == "bootstrap" else "e") / f"c{ordinal:02d}"
            receipt = directory / "cell_result.json"
            if receipt.exists() and args.resume:
                record = json.loads(receipt.read_text())
                if any(record.get(key) != value for key, value in cell.items()):
                    raise ValueError("saved cell identity differs from the frozen task/seed/phase")
                if not record.get("denominator_eligible"):
                    raise RuntimeError("previous infrastructure-invalid cell needs explicit repair; no automatic rerun")
                records.append(record)
                continue
            if directory.exists():
                raise RuntimeError(f"existing unfinished cell requires inspection, not automatic restart: {directory}")
            directory.mkdir(parents=True)
            command = [profile["simulator_python"], str(ROOT / "scripts/run_rmbench_harness_vla.py"),
                       "--config", str(args.config.resolve()), "--task", cell["task"], "--seed", str(cell["seed"]),
                       "--phase", args.phase, "--output", str(directory), "--memory", str(memory)]
            for earlier, record in histories:
                if (record.get("task"), record.get("seed")) == (cell["task"], cell["seed"]):
                    command.extend(["--reference-history", str(earlier)])
            budget = profile[args.phase]
            if args.phase == "bootstrap":
                budget = remaining_reference_budget(budget, cell["task"], cell["seed"],
                    [record for _, record in histories if (record.get("task"), record.get("seed")) == (cell["task"], cell["seed"])])
            with (directory / "launcher.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT)
                status.write_text(json.dumps({"stage": "running", "pid": os.getpid(), "worker_pid": process.pid,
                                               "current_cell": cell, "completed": len(records), "planned": len(cells)}, indent=2))
                returncode = wait_worker(process, budget["wall_timeout_sec"])
            collected = collect_cell(directory)
            if any(collected.get(key) is not None and collected[key] != value for key, value in cell.items()):
                raise ValueError("native outcome does not belong to the requested cell")
            record = {**collected, **cell, "returncode": returncode}
            receipt.write_text(json.dumps(record, indent=2))
            records.append(record)
            write_summary(args.root, records)
            if returncode or not record["denominator_eligible"]:
                status.write_text(json.dumps({"stage": "infrastructure_stopped", "cell": cell,
                                               "completed": len(records), "planned": len(cells)}, indent=2))
                return 1
        status.write_text(json.dumps({"stage": "finished", "completed": len(records), "planned": len(cells)}, indent=2))
    except KeyboardInterrupt:
        status.write_text(json.dumps({"stage": "interrupted", "completed": len(records),
                                       "planned": len(cells), "worker_pid": None if process is None else process.pid}, indent=2))
        raise
    finally:
        if process is not None and process.poll() is None:
            stop_worker(process)
        write_summary(args.root, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
