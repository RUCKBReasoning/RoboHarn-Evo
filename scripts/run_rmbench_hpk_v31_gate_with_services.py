#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_rmbench_hpk_v31_utility_gate import (  # noqa: E402
    CONDITIONS,
    UtilityGateRunError,
    load_run_plan,
    run_cell,
)

_LOOPBACK_HTTP = build_opener(ProxyHandler({}))


class UtilityGateServiceError(RuntimeError):
    """A fresh exclusive service could not be established for the Gate."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )


def _health(url: str, *, timeout_sec: float = 2.0) -> dict[str, Any] | None:
    try:
        request = Request(
            url.rstrip("/") + "/health", headers={"Accept": "application/json"}
        )
        with _LOOPBACK_HTTP.open(request, timeout=timeout_sec) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _wait_health(
    process: subprocess.Popen[bytes],
    url: str,
    *,
    timeout_sec: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise UtilityGateServiceError(
                f"service exited before health check: exit={process.returncode}"
            )
        value = _health(url)
        if value is not None and value.get("status") == "ok":
            return value
        time.sleep(0.5)
    raise UtilityGateServiceError(f"service did not become healthy at {url}")


def _stop(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=60)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _start(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
) -> tuple[subprocess.Popen[bytes], BinaryIO]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("xb")
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        stream.close()
        raise
    return process, stream


def segmentation_artifact_dir(cell: Mapping[str, Any]) -> Path:
    root = Path(cell["cell_root"])
    seed = int(cell["seed"])
    return root / "segmentation_artifacts" / f"worker_0_gpu_0_seed_0_e{seed}"


def agent_service_command(
    *,
    python: Path,
    project_root: Path,
    port: int,
    contract: Mapping[str, Any],
    codex_bin: Path,
    codex_auth_file: Path,
    codex_config_file: Path,
    runtime_root: Path,
) -> list[str]:
    return [
        str(python),
        "-u",
        str(project_root / "scripts/serve_rmbench_agent_api.py"),
        "--provider",
        "openai",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--backend",
        "codex-account",
        "--model",
        str(contract["agent_model"]),
        "--reasoning-effort",
        str(contract["reasoning_effort"]),
        "--timeout-sec",
        "600",
        "--max-concurrent-requests",
        "1",
        "--request-queue-timeout-sec",
        "5",
        "--responses-max-images",
        "16",
        "--max-retries",
        "0",
        "--planner-prompt-mode",
        "legacy_duplicate",
        "--codex-bin",
        str(codex_bin),
        "--codex-auth-file",
        str(codex_auth_file),
        "--config-file",
        str(codex_config_file),
        "--codex-runtime-root",
        str(runtime_root),
        "--codex-workdir",
        "/tmp",
    ]


def sam3_service_command(
    *,
    python: Path,
    project_root: Path,
    port: int,
    sam3_repo: Path,
    checkpoint: Path,
    bpe_path: Path,
    cache_root: Path,
    artifact_dir: Path,
    instance_id: str,
) -> list[str]:
    return [
        str(python),
        "-u",
        str(project_root / "scripts/serve_rmbench_sam3.py"),
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
        str(artifact_dir / "masks"),
        "--allowed-output-root",
        str(artifact_dir / "masks"),
        "--allowed-input-root",
        str(artifact_dir / "inputs"),
        "--instance-id",
        instance_id,
    ]


def run_with_services(
    *,
    plan_path: Path,
    launcher: Path,
    assets_root: Path,
    gpu: str,
    agent_port: int,
    sam3_port: int,
    rmbench_python: Path,
    sam3_python: Path,
    codex_bin: Path,
    codex_auth_file: Path,
    codex_config_file: Path,
    sam3_repo: Path,
    checkpoint: Path,
    bpe_path: Path,
    boundary_id: str | None = None,
    condition: str | None = None,
) -> list[dict[str, Any]]:
    plan = plan_path.resolve(strict=True)
    project_root = Path(__file__).resolve().parents[1]
    contract, cells = load_run_plan(plan)
    selected = [
        cell
        for cell in cells
        if (boundary_id is None or cell["boundary_id"] == boundary_id)
        and (condition is None or cell["condition"] == condition)
    ]
    if not selected:
        raise UtilityGateServiceError("cell selection is empty")
    agent_url = f"http://127.0.0.1:{agent_port}"
    sam3_url = f"http://127.0.0.1:{sam3_port}"
    if _health(agent_url) is not None or _health(sam3_url) is not None:
        raise UtilityGateServiceError(
            "exclusive Agent/SAM3 ports are already serving; refusing reuse"
        )
    service_root = plan.parent / "services"
    if service_root.exists() or service_root.is_symlink():
        raise UtilityGateServiceError(
            "service artifact root already exists; refusing an implicit rerun"
        )
    service_root.mkdir(parents=True, exist_ok=False)
    agent_log = service_root / "agent" / "service.log"
    agent_runtime_root = service_root / "agent" / "runtime"
    agent_runtime_root.mkdir(parents=True)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(project_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    agent_process: subprocess.Popen[bytes] | None = None
    agent_stream: BinaryIO | None = None
    results: list[dict[str, Any]] = []
    try:
        command = agent_service_command(
            python=rmbench_python,
            project_root=project_root,
            port=agent_port,
            contract=contract,
            codex_bin=codex_bin,
            codex_auth_file=codex_auth_file,
            codex_config_file=codex_config_file,
            runtime_root=agent_runtime_root,
        )
        agent_process, agent_stream = _start(
            command,
            cwd=project_root,
            environment=environment,
            log_path=agent_log,
        )
        agent_health = _wait_health(agent_process, agent_url, timeout_sec=60)
        _write_json(
            service_root / "agent/launch.json",
            {"command": command, "health": agent_health},
        )
        for cell in selected:
            artifact_dir = segmentation_artifact_dir(cell)
            (artifact_dir / "inputs").mkdir(parents=True, exist_ok=False)
            (artifact_dir / "masks").mkdir(parents=True, exist_ok=False)
            cell_service_root = Path(cell["cell_root"]) / "sam3_service"
            sam_log = cell_service_root / "service.log"
            sam_cache = service_root / "sam3_cache"
            sam_cache.mkdir(exist_ok=True)
            sam_environment = dict(environment)
            sam_environment.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONUNBUFFERED": "1",
                }
            )
            sam_command = sam3_service_command(
                python=sam3_python,
                project_root=project_root,
                port=sam3_port,
                sam3_repo=sam3_repo,
                checkpoint=checkpoint,
                bpe_path=bpe_path,
                cache_root=sam_cache,
                artifact_dir=artifact_dir,
                instance_id=(f"hpk-v31-{cell['boundary_id']}-{cell['condition']}-sam3"),
            )
            sam_process: subprocess.Popen[bytes] | None = None
            sam_stream: BinaryIO | None = None
            try:
                sam_process, sam_stream = _start(
                    sam_command,
                    cwd=project_root,
                    environment=sam_environment,
                    log_path=sam_log,
                )
                sam_health = _wait_health(sam_process, sam3_url, timeout_sec=300)
                _write_json(
                    cell_service_root / "launch.json",
                    {"command": sam_command, "health": sam_health},
                )
                results.append(
                    run_cell(
                        cell,
                        launcher=launcher,
                        service_audit_log=agent_log,
                        contract=contract,
                        assets_root=assets_root,
                        agent_api_base_url=agent_url,
                        sam3_service_url=sam3_url,
                        gpu=gpu,
                    )
                )
            finally:
                _stop(sam_process)
                if sam_stream is not None:
                    sam_stream.close()
    except (UtilityGateRunError, OSError, subprocess.SubprocessError) as exc:
        raise UtilityGateServiceError(str(exc)) from exc
    finally:
        _stop(agent_process)
        if agent_stream is not None:
            agent_stream.close()
    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--agent-port", type=int, default=19114)
    parser.add_argument("--sam3-port", type=int, default=19314)
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
    parser.add_argument("--sam3-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bpe-path", type=Path, required=True)
    parser.add_argument("--boundary-id")
    parser.add_argument("--condition", choices=CONDITIONS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    values = run_with_services(
        plan_path=args.run_plan,
        launcher=args.launcher.resolve(strict=True),
        assets_root=args.assets_root.resolve(strict=True),
        gpu=str(args.gpu).strip(),
        agent_port=args.agent_port,
        sam3_port=args.sam3_port,
        rmbench_python=args.rmbench_python.resolve(strict=True),
        sam3_python=args.sam3_python.resolve(strict=True),
        codex_bin=args.codex_bin.resolve(strict=True),
        codex_auth_file=args.codex_auth_file.resolve(strict=True),
        codex_config_file=args.codex_config_file.resolve(strict=True),
        sam3_repo=args.sam3_repo.resolve(strict=True),
        checkpoint=args.checkpoint.resolve(strict=True),
        bpe_path=args.bpe_path.resolve(strict=True),
        boundary_id=args.boundary_id,
        condition=args.condition,
    )
    print(json.dumps(values, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UtilityGateServiceError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
