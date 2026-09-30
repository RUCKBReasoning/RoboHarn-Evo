"""执行并汇总预注册的 RMBench HPK Q1 实验。"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from roboharn_evo.benchmark_adapters.rmbench.formal_binding import CELL_BINDING_SCHEMA
from roboharn_evo.benchmark_adapters.rmbench.formal_q1 import (
    RUN_CONFIG_SCHEMA,
    build_matrix,
    current_git_commit,
    load_protocol,
)


CELL_RESULT_SCHEMA = "roboharn_evo/rmbench/hpk_formal_q1_cell_result/v1"
STALL_RECORD_SCHEMA = "roboharn_evo/rmbench/launcher_stall/v1"


class FormalQ1RunError(RuntimeError):
    """One formal cell cannot be launched or validated as preregistered."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalQ1RunError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise FormalQ1RunError(f"{path} must contain one JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_bytes(root: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[str(path.relative_to(root))] = path.read_bytes()
    if not result:
        raise FormalQ1RunError("formal source pool is empty")
    return result


def _records(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FormalQ1RunError(f"invalid trace JSON at line {line_number}") from exc
        if not isinstance(value, dict):
            raise FormalQ1RunError(f"trace line {line_number} is not an object")
        result.append(value)
    return result


def _one(values: list[Any], *, label: str) -> Any:
    if len(values) != 1:
        raise FormalQ1RunError(f"expected exactly one {label}; found {len(values)}")
    return values[0]


def load_cell(
    *,
    protocol_path: str | Path,
    preparation_root: str | Path,
    cell_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_protocol(protocol_path)
    preparation = Path(preparation_root).resolve(strict=True)
    run_config = yaml.safe_load(
        (preparation / "run_config.yaml").read_text(encoding="utf-8")
    )
    if (
        not isinstance(run_config, Mapping)
        or run_config.get("schema") != RUN_CONFIG_SCHEMA
    ):
        raise FormalQ1RunError("formal run configuration schema mismatch")
    code_commit = str(run_config.get("code_commit", ""))
    if current_git_commit(Path(__file__).resolve().parents[3]) != code_commit:
        raise FormalQ1RunError("runtime HEAD differs from the frozen formal commit")
    if run_config.get("protocol") != protocol:
        raise FormalQ1RunError(
            "embedded protocol differs from the frozen protocol file"
        )
    expected = build_matrix(protocol, code_commit=code_commit)
    selected = _one(
        [row for row in expected if row["cell_id"] == cell_id],
        label="preregistered cell",
    )
    with (preparation / "method_task_seed_matrix.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        written = list(csv.DictReader(stream))
    if len(written) != 300 or {row["cell_id"] for row in written} != {
        row["cell_id"] for row in expected
    }:
        raise FormalQ1RunError("written method-task-seed matrix differs from protocol")
    return dict(run_config), selected


def build_cell_binding(
    *,
    run_config: Mapping[str, Any],
    cell: Mapping[str, Any],
    source_root: str | Path,
    path: str | Path,
) -> Path:
    source = Path(source_root).resolve(strict=True)
    views = (source / str(cell["task"]) / "method_views").resolve(strict=True)
    binding = {
        "schema": CELL_BINDING_SCHEMA,
        "protocol_id": cell["protocol_id"],
        "code_commit": run_config["code_commit"],
        "cell_id": cell["cell_id"],
        "task": cell["task"],
        "method": cell["method"],
        "evaluation_seed": cell["evaluation_seed"],
        "source_pool": cell["source_pool"],
        "method_views_root": str(views),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(binding, ensure_ascii=False, indent=2) + "\n")
    return destination


def launcher_environment(
    *,
    cell: Mapping[str, Any],
    cell_root: Path,
    project_root: Path,
    benchmark_root: Path,
    assets_root: Path,
    agent_api_base_url: str,
    sam3_service_url: str,
    gpu: str,
    expected_agent_concurrency: int,
    instruction_set: str = "rmbench_original",
) -> dict[str, str]:
    cache_variant = "original" if instruction_set == "rmbench_original" else "custom"
    runtime_cache = (
        project_root / "eval_result/rmbench/q"
        / cache_variant
        / str(cell["task"])
        / str(cell["method"])
        / str(cell["evaluation_seed"])
    )
    environment = dict(os.environ)
    environment.update(
        {
            "REPO_ROOT": str(benchmark_root),
            "ROBOHARN_EVO_PROJECT_ROOT": str(project_root),
            "RMBENCH_ASSETS_ROOT": str(assets_root.resolve(strict=True)),
            "RMBENCH_OUTPUT_ROOT": str(cell_root / "run_output"),
            "LOG_ROOT": str(cell_root / "launcher"),
            "SEGMENTATION_ARTIFACT_ROOT": str(cell_root / "segmentation_artifacts"),
            "RMBENCH_RUNTIME_CONFIG_ROOT": str(cell_root / "runtime_configs"),
            "ROBOHARN_EVO_WORKSPACE_ROOT": str(cell_root / "runtime_workspace"),
            "RUNTIME_CACHE_ROOT": str(runtime_cache),
            "RUNTIME_PROVENANCE_PATH": str(cell_root / "runtime_provenance.json"),
            "AGENT_SERVICE_IDENTITY_PATH": str(
                cell_root / "agent_service_identity.json"
            ),
            "NUM_WORKERS": "1",
            "GPU_IDS": "0",
            "RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN": str(gpu),
            # The launcher exposes exactly one physical GPU to this process.
            # Bind SAPIEN to logical cuda:0 inside that namespace instead of
            # allowing Vulkan to choose a renderer independently.
            "RMBENCH_RENDER_DEVICE": "cuda:0",
            "RMBENCH_RENDER_DEVICE_STRICT": "1",
            "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
            "RMBENCH_EXPECTED_PHYSICAL_GPU": str(gpu),
            "RMBENCH_RAY_TRACING_DENOISER": "none",
            "RMBENCH_SLOW_ACTION_TRACE_SEC": "120",
            "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH": str(
                cell_root / "run_output" / "renderer_device_provenance.json"
            ),
            "TASK_NAME": str(cell["task"]),
            "TASK_CONFIG": str(cell["task_config"]),
            "INSTRUCTION_SET": instruction_set,
            "N_PER_WORKER": "1",
            "SEED_OFFSET": "0",
            "EVAL_START_SEEDS": str(cell["evaluation_seed"]),
            "REQUIRE_EXPLICIT_EVAL_START_SEEDS": "1",
            "PERCEPTION_CONDITION": "no_oracle",
            "NON_FORMAL_DIAGNOSTIC": "0",
            "RETRY_BUDGET": "0",
            "BACKEND_ERROR_BUDGET": "0",
            "MAX_ROUNDS": "10",
            "MAX_CONTROL_TURNS": "64",
            "MAX_NO_PROGRESS_CONTROL_TURNS": "10",
            "BASE_CKPT": str(cell["cell_id"]),
            "AGENT_API_BASE_URL": agent_api_base_url.rstrip("/"),
            "SAM3_SERVICE_URL": sam3_service_url.rstrip("/"),
            "EXPECTED_AGENT_MODEL": "gpt-5.5",
            "EXPECTED_AGENT_API_MODE": "responses_compat",
            "EXPECTED_REASONING_EFFORT": "xhigh",
            "EXPECTED_RESPONSE_STORAGE": "account_default",
            "EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS": str(expected_agent_concurrency),
            "REQUIRE_AGENT_INFERENCE_PREFLIGHT": "1",
            "REQUIRE_SAM3_PREFLIGHT": "1",
            "RECORD_RUNTIME_PROVENANCE": "1",
            "DETACH": "0",
            "WORKER_START_DELAY_SEC": "0",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    return environment


def _trace_progress(root: Path) -> tuple[tuple[tuple[str, int, int], ...], int | None]:
    maximum: int | None = None
    activity: list[tuple[str, int, int]] = []
    for trace in root.rglob("episode_*_agent_trace.jsonl"):
        try:
            stat = trace.stat()
            lines = trace.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        activity.append((str(trace), stat.st_mtime_ns, stat.st_size))
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = record.get("env_step")
            if isinstance(step, int) and not isinstance(step, bool):
                maximum = step if maximum is None else max(maximum, step)
    return tuple(sorted(activity)), maximum


def _stop_process_group(
    process: subprocess.Popen[bytes], *, grace_sec: float = 10.0
) -> int:
    if process.poll() is not None:
        return int(process.returncode)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return int(process.wait(timeout=grace_sec))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return int(process.wait())


def _run_with_progress_watchdog(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    output_root: Path,
    startup_timeout_sec: float,
    no_trace_timeout_sec: float,
    poll_interval_sec: float,
) -> tuple[int, dict[str, Any]]:
    if (
        startup_timeout_sec <= 0
        or no_trace_timeout_sec <= 0
        or poll_interval_sec <= 0
    ):
        raise ValueError("watchdog durations must be positive")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=dict(environment),
        start_new_session=True,
    )
    last_step: int | None = None
    last_activity: tuple[tuple[str, int, int], ...] = ()
    trace_seen = False
    last_progress = time.monotonic()
    try:
        while True:
            try:
                return_code = process.wait(timeout=poll_interval_sec)
                return int(return_code), {
                    "triggered": False,
                    "startup_timeout_sec": startup_timeout_sec,
                    "no_trace_timeout_sec": no_trace_timeout_sec,
                    "last_environment_step": last_step,
                }
            except subprocess.TimeoutExpired:
                activity, current_step = _trace_progress(output_root)
                if activity and activity != last_activity:
                    last_activity = activity
                    trace_seen = True
                    last_step = current_step
                    last_progress = time.monotonic()
                stalled_for = time.monotonic() - last_progress
                timeout_sec = (
                    no_trace_timeout_sec if trace_seen else startup_timeout_sec
                )
                if stalled_for < timeout_sec:
                    continue
                record = {
                    "schema": STALL_RECORD_SCHEMA,
                    "reason": "no_agent_trace_activity",
                    "phase": "runtime" if trace_seen else "startup",
                    "timeout_sec": timeout_sec,
                    "last_environment_step": last_step,
                    "detected_at_unix_sec": time.time(),
                }
                _write_json(output_root / "launcher_stall.json", record)
                return_code = _stop_process_group(process)
                return return_code, {"triggered": True, **record}
    except BaseException:
        _stop_process_group(process)
        raise


def summarize_trace(
    trace_path: Path,
    *,
    cell: Mapping[str, Any],
) -> dict[str, Any]:
    records = _records(trace_path)
    start = _one(
        [record for record in records if record.get("event") == "episode_start"],
        label="episode_start",
    )
    end = _one(
        [record for record in records if record.get("event") == "episode_end"],
        label="episode_end",
    )
    if (
        start.get("task_name") != cell["task"]
        or start.get("seed") != cell["evaluation_seed"]
        or start.get("formal_protocol") is not True
        or start.get("formal_protocol_version") != 3
        or start.get("exact_seed_fail_closed") is not True
        or start.get("step_limit") != 150
    ):
        raise FormalQ1RunError("episode_start differs from the formal cell contract")
    if end.get("seed") != cell["evaluation_seed"] or end.get("step_limit") != 150:
        raise FormalQ1RunError("episode_end differs from the formal cell contract")
    validity = end.get("episode_validity")
    if not isinstance(validity, Mapping):
        raise FormalQ1RunError("episode_end lacks its validity classification")
    events = Counter(str(record.get("event", "")) for record in records)
    return {
        "trace_path": str(trace_path),
        "success": end.get("success") is True,
        "official_progress": end.get("max_reward"),
        "environment_actions": end.get("total_steps"),
        "control_turns": end.get("control_turn_count"),
        "episode_validity": dict(validity),
        "failure_reason": end.get("failure_reason"),
        "wall_time_sec": round(float(end["timestamp"]) - float(start["timestamp"]), 3),
        "event_counts": dict(sorted(events.items())),
    }


def run_cell(
    *,
    protocol_path: str | Path,
    preparation_root: str | Path,
    source_root: str | Path,
    experiment_root: str | Path,
    cell_id: str,
    launcher: str | Path,
    assets_root: str | Path,
    agent_api_base_url: str,
    sam3_service_url: str,
    gpu: str,
    expected_agent_concurrency: int,
    instruction_set: str = "rmbench_original",
    agent_service_log: str | Path | None = None,
    sam3_input_root: str | Path | None = None,
    sam3_output_root: str | Path | None = None,
    startup_timeout_sec: float = 1800.0,
    no_trace_timeout_sec: float = 900.0,
    watchdog_poll_interval_sec: float = 10.0,
) -> dict[str, Any]:
    run_config, cell = load_cell(
        protocol_path=protocol_path,
        preparation_root=preparation_root,
        cell_id=cell_id,
    )
    project_root = Path(__file__).resolve().parents[3]
    benchmark_root = project_root / "benchmarks" / "rmbench"
    root = Path(experiment_root).resolve() / "cells" / cell_id
    if root.exists() or root.is_symlink():
        raise FormalQ1RunError(f"formal cell already exists; refusing rerun: {root}")
    root.mkdir(parents=True)
    if (sam3_input_root is None) != (sam3_output_root is None):
        raise FormalQ1RunError("SAM3 input and output roots must be provided together")
    if sam3_input_root is not None and sam3_output_root is not None:
        input_target = Path(sam3_input_root).resolve(strict=True) / cell_id
        output_target = Path(sam3_output_root).resolve(strict=True) / cell_id
        input_target.mkdir()
        output_target.mkdir()
        worker_artifacts = (
            root
            / "segmentation_artifacts"
            / f"worker_0_gpu_0_seed_0_e{cell['evaluation_seed']}"
        )
        worker_artifacts.mkdir(parents=True)
        (worker_artifacts / "inputs").symlink_to(input_target, target_is_directory=True)
        (worker_artifacts / "masks").symlink_to(output_target, target_is_directory=True)
    source_task_root = Path(source_root).resolve(strict=True) / str(cell["task"])
    source_before = _source_bytes(source_task_root)
    binding = build_cell_binding(
        run_config=run_config,
        cell=cell,
        source_root=source_root,
        path=root / "cell_binding.json",
    )
    launcher_path = Path(launcher).resolve(strict=True)
    environment = launcher_environment(
        cell=cell,
        cell_root=root,
        project_root=project_root,
        benchmark_root=benchmark_root,
        assets_root=Path(assets_root),
        agent_api_base_url=agent_api_base_url,
        sam3_service_url=sam3_service_url,
        gpu=gpu,
        expected_agent_concurrency=expected_agent_concurrency,
        instruction_set=str(run_config["protocol"]["instruction_set"]),
    )
    command = [
        "/bin/bash",
        str(launcher_path),
        "--eval.exact_seed_fail_closed",
        "True",
        "--eval.step_limit",
        "150",
        "--rmbench_formal.config_path",
        str(binding),
    ]
    service_log = None if agent_service_log is None else Path(agent_service_log)
    service_offset = service_log.stat().st_size if service_log is not None else 0
    started = time.time()
    launcher_exit_code, watchdog = _run_with_progress_watchdog(
        command,
        cwd=benchmark_root,
        environment=environment,
        output_root=root / "run_output",
        startup_timeout_sec=startup_timeout_sec,
        no_trace_timeout_sec=no_trace_timeout_sec,
        poll_interval_sec=watchdog_poll_interval_sec,
    )
    if service_log is not None:
        with service_log.open("rb") as stream:
            stream.seek(service_offset)
            (root / "agent_service_audit_window.log").write_bytes(stream.read())
    source_unchanged = source_before == _source_bytes(source_task_root)
    traces = sorted((root / "run_output").rglob("episode_*_agent_trace.jsonl"))
    result: dict[str, Any] = {
        "schema": CELL_RESULT_SCHEMA,
        "cell_id": cell_id,
        "task": cell["task"],
        "method": cell["method"],
        "evaluation_seed": cell["evaluation_seed"],
        "launcher_exit_code": launcher_exit_code,
        "automatic_retry": False,
        "source_pool_unchanged": source_unchanged,
        "runner_wall_time_sec": round(time.time() - started, 3),
        "step_watchdog": watchdog,
    }
    try:
        if not source_unchanged:
            raise FormalQ1RunError("frozen source pool changed during evaluation")
        if watchdog["triggered"]:
            raise FormalQ1RunError(
                "formal launcher produced no Agent trace activity for "
                f"{watchdog['timeout_sec']:g} seconds"
            )
        if launcher_exit_code != 0:
            raise FormalQ1RunError(
                f"formal launcher exited with status {launcher_exit_code}"
            )
        trace_path = _one(traces, label="episode trace")
        result.update(summarize_trace(trace_path, cell=cell))
        result["status"] = (
            "complete"
            if result["episode_validity"].get("benchmark_denominator_eligible") is True
            else "infrastructure_invalid"
        )
    except FormalQ1RunError as exc:
        result["status"] = "infrastructure_invalid"
        result["validation_error"] = str(exc)
    _write_json(root / "cell_result.json", result)
    return result


__all__ = [
    "CELL_RESULT_SCHEMA",
    "STALL_RECORD_SCHEMA",
    "FormalQ1RunError",
    "build_cell_binding",
    "launcher_environment",
    "load_cell",
    "run_cell",
    "summarize_trace",
]
