#!/usr/bin/env python3
"""Run one task-method Q1 shard with fresh exclusive Agent and SAM3 services."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import yaml
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_rmbench_hpk_v31_gate_with_services import (  # noqa: E402
    _health,
    _start,
    _stop,
    _wait_health,
    agent_service_command,
)
from roboharn_evo.benchmark_adapters.rmbench.formal_q1 import (  # noqa: E402
    build_matrix,
    load_protocol,
)
from roboharn_evo.benchmark_adapters.rmbench.formal_runner import run_cell  # noqa: E402


def _sam3_command(
    *,
    python: Path,
    port: int,
    sam3_repo: Path,
    checkpoint: Path,
    bpe_path: Path,
    cache_root: Path,
    allowed_input_root: Path,
    allowed_output_root: Path,
    output_root: Path,
    instance_id: str,
) -> list[str]:
    return [
        str(python),
        "-u",
        str(REPO_ROOT / "scripts/serve_rmbench_sam3.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--sam3-repo",
        str(sam3_repo),
        "--checkpoint",
        str(checkpoint),
        "--bpe-path",
        str(bpe_path),
        "--device",
        "cuda",
        "--amp-dtype",
        "bfloat16",
        "--confidence-threshold",
        "0.1",
        "--cache-root",
        str(cache_root),
        "--output-root",
        str(output_root),
        "--allowed-output-root",
        str(allowed_output_root),
        "--allowed-input-root",
        str(allowed_input_root),
        "--instance-id",
        instance_id,
    ]


def run_shard(args: argparse.Namespace) -> list[dict]:
    protocol = load_protocol(args.protocol)
    run_config = yaml.safe_load(
        (args.preparation_root.resolve(strict=True) / "run_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    rows = build_matrix(protocol, code_commit=str(run_config["code_commit"]))
    cells = [
        row for row in rows if row["task"] == args.task and row["method"] == args.method
    ]
    if len(cells) != 10:
        raise RuntimeError("a formal task-method shard must contain exactly 10 cells")
    experiment_root = args.experiment_root.resolve()
    service_root = experiment_root / "services" / args.shard_id
    if service_root.exists() or service_root.is_symlink():
        raise RuntimeError(f"service shard already exists: {service_root}")
    service_root.mkdir(parents=True)
    cells_root = experiment_root / "cells"
    cells_root.mkdir(parents=True, exist_ok=True)
    agent_url = f"http://127.0.0.1:{args.agent_port}"
    sam_url = f"http://127.0.0.1:{args.sam3_port}"
    if _health(agent_url) is not None or _health(sam_url) is not None:
        raise RuntimeError("exclusive shard service port is already in use")
    common_environment = dict(os.environ)
    common_environment.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    agent_runtime = service_root / "agent/runtime"
    agent_runtime.mkdir(parents=True)
    agent_command = agent_service_command(
        python=args.rmbench_python,
        project_root=REPO_ROOT,
        port=args.agent_port,
        contract={"agent_model": "gpt-5.5", "reasoning_effort": "xhigh"},
        codex_bin=args.codex_bin,
        codex_auth_file=args.codex_auth_file,
        codex_config_file=args.codex_config_file,
        runtime_root=agent_runtime,
    )
    sam_input = service_root / "sam3/io/inputs"
    sam_output = service_root / "sam3/io/masks"
    sam_input.mkdir(parents=True)
    sam_output.mkdir(parents=True)
    sam_command = _sam3_command(
        python=args.sam3_python,
        port=args.sam3_port,
        sam3_repo=args.sam3_repo,
        checkpoint=args.sam3_checkpoint,
        bpe_path=args.sam3_bpe_path,
        cache_root=service_root / "sam3/cache",
        allowed_input_root=sam_input,
        allowed_output_root=sam_output,
        output_root=sam_output,
        instance_id=f"hpk-q1-{args.shard_id}",
    )
    sam_environment = dict(common_environment)
    sam_environment.update(
        {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": args.gpu}
    )
    agent_process: subprocess.Popen[bytes] | None = None
    sam_process: subprocess.Popen[bytes] | None = None
    agent_stream: BinaryIO | None = None
    sam_stream: BinaryIO | None = None
    results: list[dict] = []
    try:
        agent_process, agent_stream = _start(
            agent_command,
            cwd=REPO_ROOT,
            environment=common_environment,
            log_path=service_root / "agent/service.log",
        )
        agent_health = _wait_health(agent_process, agent_url, timeout_sec=90)
        sam_process, sam_stream = _start(
            sam_command,
            cwd=REPO_ROOT,
            environment=sam_environment,
            log_path=service_root / "sam3/service.log",
        )
        sam_health = _wait_health(sam_process, sam_url, timeout_sec=300)
        (service_root / "service_identity.json").write_text(
            json.dumps(
                {
                    "agent": agent_health,
                    "sam3": sam_health,
                    "agent_command": agent_command,
                    "sam3_command": sam_command,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for cell in cells:
            try:
                result = run_cell(
                    protocol_path=args.protocol,
                    preparation_root=args.preparation_root,
                    source_root=args.source_root,
                    experiment_root=experiment_root,
                    cell_id=cell["cell_id"],
                    launcher=args.launcher,
                    assets_root=args.assets_root,
                    agent_api_base_url=agent_url,
                    sam3_service_url=sam_url,
                    gpu=args.gpu,
                    expected_agent_concurrency=1,
                    agent_service_log=service_root / "agent/service.log",
                    sam3_input_root=sam_input,
                    sam3_output_root=sam_output,
                )
            except Exception as exc:  # keep the preregistered shard moving
                result = {
                    "cell_id": cell["cell_id"],
                    "task": cell["task"],
                    "method": cell["method"],
                    "evaluation_seed": cell["evaluation_seed"],
                    "status": "infrastructure_invalid",
                    "automatic_retry": False,
                    "runner_error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            with (service_root / "shard_results.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        _stop(sam_process)
        _stop(agent_process)
        if sam_stream is not None:
            sam_stream.close()
        if agent_stream is not None:
            agent_stream.close()
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "benchmarks/rmbench/experiments/hpk_formal_q1_v1.yaml",
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=(
            REPO_ROOT
            / "benchmarks/rmbench/policy/roboharn_evo/scripts/run_gpt55_pure_tool_control_8way.sh"
        ),
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("/path/to/RMBench/assets"),
    )
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--agent-port", type=int, required=True)
    parser.add_argument("--sam3-port", type=int, required=True)
    parser.add_argument(
        "--rmbench-python",
        type=Path,
        default=Path("/path/to/simulator/environment/bin/python"),
    )
    parser.add_argument(
        "--sam3-python",
        type=Path,
        default=Path("/path/to/SAM3/environment/bin/python"),
    )
    parser.add_argument("--codex-bin", type=Path, default=Path("/usr/bin/codex"))
    parser.add_argument(
        "--codex-auth-file", type=Path, default=Path("/path/to/provider/auth.json")
    )
    parser.add_argument(
        "--codex-config-file", type=Path, default=Path("/path/to/provider/config.toml")
    )
    parser.add_argument(
        "--sam3-repo", type=Path, default=Path("/path/to/SAM3")
    )
    parser.add_argument(
        "--sam3-checkpoint",
        type=Path,
        default=Path("/path/to/checkpoints/sam3.1_multiplex.pt"),
    )
    parser.add_argument(
        "--sam3-bpe-path",
        type=Path,
        default=Path(
            "/path/to/SAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    results = run_shard(args)
    print(json.dumps({"cells": len(results)}, sort_keys=True))
    if any(result.get("status") == "infrastructure_invalid" for result in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
