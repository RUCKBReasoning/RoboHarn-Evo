#!/usr/bin/env python3
"""Build all six frozen Q1 source pools with one fresh GPT-5.5 service."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_rmbench_hpk_v31_gate_with_services import (  # noqa: E402
    _health,
    _start,
    _stop,
    _wait_health,
)
from roboharn_evo.benchmark_adapters.rmbench.formal_q1 import load_protocol  # noqa: E402


def _agent_command(args: argparse.Namespace, runtime_root: Path) -> list[str]:
    return [
        str(args.python),
        "-u",
        str(REPO_ROOT / "scripts/serve_rmbench_agent_api.py"),
        "--provider",
        "openai",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.agent_port),
        "--backend",
        "codex-account",
        "--model",
        "gpt-5.5",
        "--reasoning-effort",
        "xhigh",
        "--timeout-sec",
        "600",
        "--max-concurrent-requests",
        str(args.parallel_tasks),
        "--request-queue-timeout-sec",
        "30",
        "--responses-max-images",
        "32",
        "--max-retries",
        "0",
        "--planner-prompt-mode",
        "legacy_duplicate",
        "--codex-bin",
        str(args.codex_bin),
        "--codex-auth-file",
        str(args.codex_auth_file),
        "--config-file",
        str(args.codex_config_file),
        "--codex-runtime-root",
        str(runtime_root),
        "--codex-workdir",
        "/tmp",
    ]


def _run_wave(commands: Sequence[list[str]], *, environment: dict[str, str]) -> None:
    processes = [
        subprocess.Popen(command, cwd=REPO_ROOT, env=environment)
        for command in commands
    ]
    failures = []
    for command, process in zip(commands, processes, strict=True):
        status = process.wait()
        if status != 0:
            failures.append({"command": command, "exit_code": status})
    if failures:
        raise RuntimeError(f"formal source stage failed without retry: {failures}")


def _source_command(
    args: argparse.Namespace,
    *,
    stage: str,
    task: str,
    source_root: Path,
) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts/build_rmbench_hpk_formal_q1_source.py"),
        stage,
        "--task",
        task,
        "--protocol",
        str(args.protocol),
        "--asset-root",
        str(args.assets_root),
        "--output-root",
        str(source_root),
        "--gpt-url",
        f"http://127.0.0.1:{args.agent_port}",
        "--max-images",
        "32",
    ]


def run(args: argparse.Namespace) -> dict:
    protocol = load_protocol(args.protocol)
    tasks = [task["name"] for task in protocol["tasks"]]
    root = args.output_root.resolve()
    if root.exists() or root.is_symlink():
        raise RuntimeError(f"formal source output already exists: {root}")
    source_root = root / "source_pools"
    service_root = root / "source_service"
    runtime_root = service_root / "runtime"
    runtime_root.mkdir(parents=True)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    prepare_commands = [
        _source_command(
            args,
            stage="prepare",
            task=task,
            source_root=source_root,
        )
        for task in tasks
    ]
    _run_wave(prepare_commands, environment=environment)
    service_url = f"http://127.0.0.1:{args.agent_port}"
    if _health(service_url) is not None:
        raise RuntimeError("exclusive formal source service port is already in use")
    command = _agent_command(args, runtime_root)
    process = None
    stream = None
    try:
        process, stream = _start(
            command,
            cwd=REPO_ROOT,
            environment=environment,
            log_path=service_root / "service.log",
        )
        health = _wait_health(process, service_url, timeout_sec=90)
        (service_root / "service_identity.json").write_text(
            json.dumps({"health": health, "command": command}, indent=2) + "\n",
            encoding="utf-8",
        )
        for stage in ("reflect", "consolidate"):
            commands = [
                _source_command(
                    args,
                    stage=stage,
                    task=task,
                    source_root=source_root,
                )
                for task in tasks
            ]
            for offset in range(0, len(commands), args.parallel_tasks):
                _run_wave(
                    commands[offset : offset + args.parallel_tasks],
                    environment=environment,
                )
    finally:
        _stop(process)
        if stream is not None:
            stream.close()
    result = {
        "schema": "roboharn_evo/rmbench/hpk_formal_q1_source_build/v1",
        "tasks": tasks,
        "source_trajectories_per_task": 10,
        "source_trajectories_total": 60,
        "reflection_calls": 60,
        "consolidation_calls": 6,
        "sam3_calls": 0,
        "automatic_retries": 0,
        "all_action_chunk_boundaries_presented": True,
        "max_images_per_reflection": 32,
    }
    (root / "source_build_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "benchmarks/rmbench/experiments/hpk_formal_q1_v1.yaml",
    )
    parser.add_argument(
        "--assets-root", type=Path, default=Path("/path/to/RMBench")
    )
    parser.add_argument("--agent-port", type=int, default=19134)
    parser.add_argument("--parallel-tasks", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument(
        "--python", type=Path, default=Path("/path/to/simulator/environment/bin/python")
    )
    parser.add_argument("--codex-bin", type=Path, default=Path("/usr/bin/codex"))
    parser.add_argument(
        "--codex-auth-file", type=Path, default=Path("/path/to/provider/auth.json")
    )
    parser.add_argument(
        "--codex-config-file", type=Path, default=Path("/path/to/provider/config.toml")
    )
    return parser.parse_args(argv)


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
