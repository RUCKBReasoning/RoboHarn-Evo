#!/usr/bin/env python3
"""Run a statically sharded GPT-5.5 rollout grid with isolated resources.

This is an orchestration layer only.  It does not import or modify the RoboHarn-Evo
agent: every slot delegates its ordered task/seed subset to the existing
single-GPU sequential launcher.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from typing import Any, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_ROBOHARN_PROJECT_ROOT = DEFAULT_REPO_ROOT.parents[1]
SEQUENTIAL_SCRIPT_REL = Path("policy/roboharn_evo/scripts/run_gpt55_six_tasks_sequential.sh")
SAM3_SERVER_REL = Path("policy/roboharn_evo/scripts/serve_sam3_segmentation.py")
MANIFEST_NAME = "parallel_manifest.json"
SAM3_READY_MANIFEST_NAME = "sam3_ready_manifest.json"
PLAN_NAME = "parallel_plan.tsv"
RUNTIME_CACHE_ENVIRONMENT_LAYOUT = {
    "XDG_CACHE_HOME": "xdg",
    "CUDA_CACHE_PATH": "cuda",
    "MESA_SHADER_CACHE_DIR": "mesa",
    "TORCH_HOME": "torch",
    "HF_HOME": "huggingface",
    "TRITON_CACHE_DIR": "triton",
    "NUMBA_CACHE_DIR": "numba",
    "MPLCONFIGDIR": "matplotlib",
}
RUNTIME_TEMP_ENVIRONMENT_VARIABLES = ("TMPDIR", "TMP", "TEMP")


class ConfigError(RuntimeError):
    """A user-visible configuration or preflight error (exit status 2)."""


@dataclass(frozen=True)
class Job:
    job_index: int
    slot: int
    slot_ordinal: int
    task: str
    eval_start_seed: int
    eval_gpu: int
    sam3_gpu: int
    sam3_url: str


@dataclass(frozen=True)
class Config:
    repo_root: Path
    roboharn_project_root: Path
    output_root: Path
    assets_root: Path | None
    pythonpath: str
    runtime_config_root: Path
    runtime_workspace_root: Path
    runtime_cache_root: Path
    runtime_temp_root: Path
    slots: int
    eval_gpus: tuple[int, ...]
    sam3_gpus: tuple[int, ...]
    sam3_base_port: int
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    job_assignment_mode: str
    allow_shared_gpu: bool
    allow_occupied_gpu: bool
    allow_verified_causalwam_gpu_occupancy: bool
    allowed_causalwam_pids: tuple[int, ...]
    causalwam_root: Path | None
    gpu_lock_root: Path
    render_device: str
    batch_stamp: str
    run_label: str
    batch_log_root: Path
    task_config: str
    instruction_set: str
    policy_name: str
    perception_condition: str
    agent_api_base_url: str
    max_objects: int
    eval_step_limit: int
    max_rounds: int
    max_control_turns: int
    max_no_progress_control_turns: int
    continue_on_error: bool
    sam3_python: Path
    sam3_repo: Path | None
    sam3_checkpoint: Path | None
    sam3_bpe_path: Path | None
    sam3_confidence_threshold: float
    sam3_amp_dtype: str
    sam3_startup_timeout_sec: float
    sam3_health_interval_sec: float
    sam3_real_probe: bool
    sam3_probe_timeout_sec: float
    slot_start_delay_sec: float
    shutdown_sigint_grace_sec: float
    tmux_session: str


@dataclass
class OwnedProcess:
    kind: str
    slot: int
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: TextIO
    start_ticks: int | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cleanup_notice(message: str) -> None:
    """Best-effort diagnostics that can never break the ownership barrier."""
    try:
        print(message, file=sys.stderr, flush=True)
    except BaseException:
        pass


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean (0/1), got: {raw!r}")


def env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got: {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got: {value}")
    return value


def env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be numeric, got: {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got: {value}")
    return value


def parse_csv_ints(name: str, raw: str, *, exact_length: int | None = None) -> tuple[int, ...]:
    pieces = [piece.strip() for piece in raw.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ConfigError(f"{name} must be a non-empty comma-separated integer list")
    try:
        values = tuple(int(piece, 10) for piece in pieces)
    except ValueError as exc:
        raise ConfigError(f"{name} contains a non-integer value: {raw!r}") from exc
    if any(value < 0 for value in values):
        raise ConfigError(f"{name} values must be non-negative")
    if len(set(values)) != len(values):
        raise ConfigError(f"{name} contains a duplicate resource: {raw}")
    if exact_length is not None and len(values) != exact_length:
        raise ConfigError(f"{name} must contain exactly {exact_length} values, got {len(values)}")
    return values


def parse_optional_pid_csv(name: str, raw: str) -> tuple[int, ...]:
    """Parse an explicit PID allowlist; an empty value means no exceptions."""
    if not raw.strip():
        return ()
    values = parse_csv_ints(name, raw)
    if any(value == 0 for value in values):
        raise ConfigError(f"{name} values must be positive process IDs")
    return values


def parse_tasks(raw: str) -> tuple[str, ...]:
    tasks = tuple(piece.strip() for piece in raw.split(","))
    if not tasks or any(not task for task in tasks):
        raise ConfigError("TASKS_CSV must contain at least one non-empty task")
    for task in tasks:
        if not task.replace("_", "").isalnum():
            raise ConfigError(f"TASKS_CSV contains an invalid task name: {task!r}")
    if len(set(tasks)) != len(tasks):
        raise ConfigError("TASKS_CSV contains a duplicate task")
    return tasks


def composed_pythonpath(*, project_root: Path, repo_root: Path) -> str:
    """返回 RoboHarn-Evo 包目录和 benchmark 包目录。"""

    return os.pathsep.join((str(project_root), str(repo_root)))


def explicit_external_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def short_runtime_temp_root(*, output_root: Path, batch_root: Path) -> Path:
    """Return a short, deterministic, batch-isolated temp directory."""

    batch_token = hashlib.sha256(
        os.fsencode(str(batch_root.resolve()))
    ).hexdigest()[:12]
    return (output_root / "tmp" / batch_token).resolve()


def runtime_environment(
    config: Config,
    *,
    workspace_root: Path | None = None,
    cache_root: Path | None = None,
    runtime_config_root: Path | None = None,
    temp_root: Path | None = None,
) -> dict[str, str]:
    """Return the copied-chain import and write-boundary environment."""

    workspace = (workspace_root or config.runtime_workspace_root).resolve()
    caches = (cache_root or config.runtime_cache_root).resolve()
    runtime_configs = (runtime_config_root or config.runtime_config_root).resolve()
    temporary = (temp_root or config.runtime_temp_root).resolve()
    result = {
        "PYTHONPATH": config.pythonpath,
        "PYTHONDONTWRITEBYTECODE": "1",
        "RMBENCH_ROOT": str(config.repo_root),
        "RMBENCH_OUTPUT_ROOT": str(config.output_root),
        "ROBOHARN_EVO_OUTPUT_ROOT": str(config.output_root),
        "ROBOHARN_EVO_WORKSPACE_ROOT": str(workspace),
        "RMBENCH_RUNTIME_CONFIG_ROOT": str(runtime_configs),
    }
    if config.assets_root is not None:
        result["RMBENCH_ASSETS_ROOT"] = str(config.assets_root)
    result.update(
        {
            variable: str(caches / relative_path)
            for variable, relative_path in RUNTIME_CACHE_ENVIRONMENT_LAYOUT.items()
        }
    )
    result.update(
        {
            variable: str(temporary)
            for variable in RUNTIME_TEMP_ENVIRONMENT_VARIABLES
        }
    )
    return result


def load_config(batch_log_root_override: str | None) -> Config:
    repo_root = Path(os.environ.get("REPO_ROOT", str(DEFAULT_REPO_ROOT))).expanduser().resolve()
    if repo_root != DEFAULT_REPO_ROOT:
        raise ConfigError(
            "REPO_ROOT must be the benchmark containing this copied scheduler "
            f"and may not select a donor checkout: {DEFAULT_REPO_ROOT}, got {repo_root}"
        )
    roboharn_project_root = Path(
        os.environ.get("ROBOHARN_EVO_PROJECT_ROOT", str(repo_root.parents[1]))
    ).expanduser().resolve()
    expected_project_root = repo_root.parents[1]
    if roboharn_project_root != expected_project_root:
        raise ConfigError(
            "ROBOHARN_EVO_PROJECT_ROOT must be the project containing the copied "
            f"benchmark: {expected_project_root}, got {roboharn_project_root}"
        )
    output_root = Path(
        os.environ.get(
            "RMBENCH_OUTPUT_ROOT",
            str(roboharn_project_root / "eval_result" / "rmbench"),
        )
    ).expanduser().resolve()
    assets_root = explicit_external_path("RMBENCH_ASSETS_ROOT")
    slots = env_int("PARALLEL_SLOTS", 4, minimum=1)
    eval_gpus = parse_csv_ints(
        "GPU_IDS", os.environ.get("GPU_IDS", "4,5,6,7"), exact_length=slots
    )
    sam3_gpus = parse_csv_ints(
        "SAM3_GPU_IDS", os.environ.get("SAM3_GPU_IDS", "0,1,2,3"), exact_length=slots
    )
    allow_shared_gpu = env_bool("ALLOW_SHARED_GPU", False)
    allow_occupied_gpu = env_bool("ALLOW_OCCUPIED_GPU", False)
    allow_verified_causalwam_gpu_occupancy = env_bool(
        "ALLOW_VERIFIED_CAUSALWAM_GPU_OCCUPANCY", True
    )
    allowed_causalwam_pids = parse_optional_pid_csv(
        "GPU_OCCUPANCY_ALLOWED_CAUSALWAM_PIDS_CSV",
        os.environ.get("GPU_OCCUPANCY_ALLOWED_CAUSALWAM_PIDS_CSV", ""),
    )
    overlap = sorted(set(eval_gpus) & set(sam3_gpus))
    if overlap and not allow_shared_gpu:
        raise ConfigError(
            "Eval/SAM3 GPU pools overlap at "
            f"{overlap}; resource isolation forbids shared GPUs by default. "
            "Set ALLOW_SHARED_GPU=1 only for a non-isolated diagnostic run."
        )

    base_port = env_int("SAM3_BASE_PORT", 9311, minimum=1)
    if base_port > 65535 or base_port + slots - 1 > 65535:
        raise ConfigError(
            f"SAM3_BASE_PORT port range {base_port}..{base_port + slots - 1} is invalid"
        )
    tasks = parse_tasks(
        os.environ.get(
            "TASKS_CSV", "rearrange_blocks,swap_blocks,swap_T,battery_try"
        )
    )
    seeds = parse_csv_ints(
        "EVAL_START_SEEDS_CSV", os.environ.get("EVAL_START_SEEDS_CSV", "0,1")
    )
    job_assignment_mode = os.environ.get(
        "JOB_ASSIGNMENT_MODE", "round_robin"
    ).strip()
    if job_assignment_mode not in {"round_robin", "task_affinity"}:
        raise ConfigError(
            "JOB_ASSIGNMENT_MODE must be round_robin or task_affinity"
        )

    stamp = os.environ.get("BATCH_STAMP", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    default_root = (
        output_root
        / "formal_runs"
        / f"gpt55_parallel4_isolated_{stamp}"
    )
    batch_root = Path(
        batch_log_root_override
        or os.environ.get("BATCH_LOG_ROOT", str(default_root))
    ).expanduser().resolve()
    gpu_lock_root = Path(
        os.environ.get(
            "GPU_LOCK_ROOT",
            str(output_root / "runtime_locks" / "gpus"),
        )
    ).expanduser().resolve()
    allowed_runtime_root = (roboharn_project_root / "eval_result" / "rmbench").resolve()
    for label, path in (
        ("RMBENCH_OUTPUT_ROOT", output_root),
        ("BATCH_LOG_ROOT", batch_root),
        ("GPU_LOCK_ROOT", gpu_lock_root),
    ):
        if path != allowed_runtime_root and allowed_runtime_root not in path.parents:
            raise ConfigError(
                f"{label} must stay under {allowed_runtime_root}: {path}"
            )
    if paths_overlap(output_root, repo_root):
        raise ConfigError(
            "RMBENCH_OUTPUT_ROOT must not overlap the copied benchmark source: "
            f"{output_root} vs {repo_root}"
        )
    if assets_root is not None and paths_overlap(output_root, assets_root):
        raise ConfigError(
            "RMBENCH_OUTPUT_ROOT must not overlap RMBENCH_ASSETS_ROOT: "
            f"{output_root} vs {assets_root}"
        )
    run_label = os.environ.get(
        "RUN_LABEL", f"formalv3_parallel4_isolated_9104_{stamp}"
    ).strip()
    if not run_label or any(not (ch.isalnum() or ch in "._-") for ch in run_label):
        raise ConfigError("RUN_LABEL may contain only letters, digits, '.', '_', and '-'")

    render_device = (
        os.environ.get("RMBENCH_RENDER_DEVICE", "pci:auto").strip()
        or "pci:auto"
    )
    if render_device != "pci:auto":
        raise ConfigError(
            "The isolated multi-GPU scheduler requires "
            "RMBENCH_RENDER_DEVICE=pci:auto; each slot derives its own exact "
            "pci:<bus-id> alias after the nvidia-smi preflight"
        )

    amp_dtype = os.environ.get("SAM3_AMP_DTYPE", "bfloat16").strip()
    if amp_dtype not in {"bfloat16", "float16", "none"}:
        raise ConfigError("SAM3_AMP_DTYPE must be bfloat16, float16, or none")
    perception = os.environ.get("PERCEPTION_CONDITION", "no_oracle").strip()
    if perception not in {"no_oracle", "oracle"}:
        raise ConfigError("PERCEPTION_CONDITION must be no_oracle or oracle")

    continue_on_error = env_bool("CONTINUE_ON_ERROR", False)
    if continue_on_error:
        raise ConfigError(
            "CONTINUE_ON_ERROR=1 is incompatible with this isolated formal scheduler; "
            "infrastructure/trace failures must stop all slots"
        )

    return Config(
        repo_root=repo_root,
        roboharn_project_root=roboharn_project_root,
        output_root=output_root,
        assets_root=assets_root,
        pythonpath=composed_pythonpath(
            project_root=roboharn_project_root,
            repo_root=repo_root,
        ),
        runtime_config_root=(batch_root / "runtime_configs").resolve(),
        runtime_workspace_root=(batch_root / "runtime_workspace").resolve(),
        runtime_cache_root=(batch_root / "runtime_caches").resolve(),
        runtime_temp_root=short_runtime_temp_root(
            output_root=output_root,
            batch_root=batch_root,
        ),
        slots=slots,
        eval_gpus=eval_gpus,
        sam3_gpus=sam3_gpus,
        sam3_base_port=base_port,
        tasks=tasks,
        seeds=seeds,
        job_assignment_mode=job_assignment_mode,
        allow_shared_gpu=allow_shared_gpu,
        allow_occupied_gpu=allow_occupied_gpu,
        allow_verified_causalwam_gpu_occupancy=(
            allow_verified_causalwam_gpu_occupancy
        ),
        allowed_causalwam_pids=allowed_causalwam_pids,
        causalwam_root=explicit_external_path("CAUSALWAM_ROOT"),
        gpu_lock_root=gpu_lock_root,
        render_device=render_device,
        batch_stamp=stamp,
        run_label=run_label,
        batch_log_root=batch_root,
        task_config=os.environ.get("TASK_CONFIG", "demo_clean").strip(),
        instruction_set=os.environ.get("INSTRUCTION_SET", "rmbench_original").strip(),
        policy_name=os.environ.get("POLICY_NAME", "policy.roboharn_evo.deploy_policy").strip(),
        perception_condition=perception,
        agent_api_base_url=os.environ.get("AGENT_API_BASE_URL", "http://127.0.0.1:9104").rstrip("/"),
        max_objects=env_int("MAX_OBJECTS", 8, minimum=1),
        eval_step_limit=env_int("EVAL_STEP_LIMIT", 150, minimum=1),
        max_rounds=env_int("MAX_ROUNDS", 10, minimum=1),
        max_control_turns=env_int("MAX_CONTROL_TURNS", 64, minimum=1),
        max_no_progress_control_turns=env_int(
            "MAX_NO_PROGRESS_CONTROL_TURNS", 10, minimum=1
        ),
        continue_on_error=continue_on_error,
        sam3_python=Path(
            os.environ.get("SAM3_PYTHON", "python")
        ).expanduser().resolve(),
        sam3_repo=explicit_external_path("ROBOHARN_EVO_SAM3_REPO"),
        sam3_checkpoint=explicit_external_path("ROBOHARN_EVO_SAM3_CHECKPOINT"),
        sam3_bpe_path=explicit_external_path("ROBOHARN_EVO_SAM3_BPE_PATH"),
        sam3_confidence_threshold=env_float("SAM3_CONFIDENCE_THRESHOLD", 0.1),
        sam3_amp_dtype=amp_dtype,
        sam3_startup_timeout_sec=env_float("SAM3_STARTUP_TIMEOUT_SEC", 240.0, minimum=1.0),
        sam3_health_interval_sec=env_float("SAM3_HEALTH_INTERVAL_SEC", 1.0, minimum=0.1),
        sam3_real_probe=env_bool("SAM3_REAL_PROBE", True),
        sam3_probe_timeout_sec=env_float("SAM3_PROBE_TIMEOUT_SEC", 180.0, minimum=1.0),
        slot_start_delay_sec=env_float("SLOT_START_DELAY_SEC", 0.0),
        shutdown_sigint_grace_sec=env_float(
            "SHUTDOWN_SIGINT_GRACE_SEC", 30.0, minimum=0.1
        ),
        tmux_session=os.environ.get("BATCH_SESSION", f"gpt55_parallel4_{stamp}"),
    )


def build_jobs(config: Config) -> list[Job]:
    jobs: list[Job] = []
    slot_ordinals = [0] * config.slots
    for task_index, task in enumerate(config.tasks):
        for seed in config.seeds:
            index = len(jobs)
            if config.job_assignment_mode == "task_affinity":
                # Keep every seed of one task on the same isolated renderer,
                # evaluator and SAM3 service.  If there are more tasks than
                # slots, whole tasks share a slot rather than being split.
                slot = task_index % config.slots
            else:
                slot = index % config.slots
            jobs.append(
                Job(
                    job_index=index,
                    slot=slot,
                    slot_ordinal=slot_ordinals[slot],
                    task=task,
                    eval_start_seed=seed,
                    eval_gpu=config.eval_gpus[slot],
                    sam3_gpu=config.sam3_gpus[slot],
                    sam3_url=f"http://127.0.0.1:{config.sam3_base_port + slot}",
                )
            )
            slot_ordinals[slot] += 1
    return jobs


def print_plan(config: Config, jobs: list[Job]) -> None:
    print(
        f"Parallel isolated plan: jobs={len(jobs)} slots={config.slots} "
        f"condition={config.perception_condition}"
    )
    for job in jobs:
        print(
            "[job "
            f"job_index={job.job_index} slot={job.slot} slot_ordinal={job.slot_ordinal} "
            f"task={job.task} eval_start_seed={job.eval_start_seed} "
            f"eval_gpu={job.eval_gpu} sam3_gpu={job.sam3_gpu} sam3_url={job.sam3_url}]"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run a static four-slot rollout grid with per-slot GPU/SAM3 isolation."
    )
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print the mapping without writes.")
    mode.add_argument("--foreground", action="store_true", help="Run in this terminal (default).")
    mode.add_argument("--detach", action="store_true", help="Run the scheduler in a tmux session.")
    mode.add_argument("--status", action="store_true", help="Read the manifest and slot state.")
    mode.add_argument("--skip-slot", type=int, metavar="N", help="Gracefully skip slot N's current rollout.")
    result.add_argument("--batch-log-root", help="Parallel batch log/control root.")
    return result


def require_batch_root_argument(args: argparse.Namespace) -> Path:
    raw = args.batch_log_root
    if not raw:
        raise ConfigError("--batch-log-root is required with --status or --skip-slot")
    return Path(raw).expanduser().resolve()


def proc_start_ticks(pid: int) -> int | None:
    try:
        stat_path = Path("/proc/self/stat") if pid == os.getpid() else Path(f"/proc/{pid}/stat")
        tail = stat_path.read_text(encoding="utf-8").rsplit(") ", 1)[1]
        fields = tail.split()
        return int(fields[19])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def current_boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def owner_is_live(owner: dict[str, Any]) -> bool:
    pid = owner.get("pid")
    start_ticks = owner.get("start_ticks")
    boot_id = owner.get("boot_id")
    return (
        isinstance(pid, int)
        and pid > 0
        and isinstance(start_ticks, int)
        and start_ticks > 0
        and isinstance(boot_id, str)
        and bool(boot_id)
        and boot_id == current_boot_id()
        and proc_start_ticks(pid) == start_ticks
    )


def read_lock_owner(lock_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


class GpuLockSet:
    """Stable-inode flock locks plus PID-reuse-safe diagnostic ownership."""

    def __init__(self, root: Path, gpu_ids: list[int], token: str) -> None:
        self.root = root
        self.gpu_ids = sorted(set(gpu_ids))
        self.token = token
        self.owner = {
            "schema": "roboharn_evo/gpu_resource_lock/v1",
            "pid": os.getpid(),
            "start_ticks": proc_start_ticks(os.getpid()),
            "boot_id": current_boot_id(),
            "token": token,
            "script": str(SCRIPT_PATH),
            "created_at": utc_now(),
        }
        if self.owner["start_ticks"] is None or not self.owner["boot_id"]:
            raise ConfigError(
                "Could not establish PID/start_ticks/boot_id identity for GPU lock owner"
            )
        self.acquired: list[tuple[int, int]] = []

    def lock_path(self, gpu_id: int) -> Path:
        return self.root / f"gpu_{gpu_id}.lock"

    def _acquire_one(self, gpu_id: int) -> tuple[bool, dict[str, Any] | None]:
        lock_path = self.lock_path(gpu_id)
        try:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            os.set_inheritable(fd, False)
        except OSError as exc:
            raise ConfigError(f"Could not open GPU lock {lock_path}: {exc}") from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                raw = os.pread(fd, 64 * 1024, 0).decode("utf-8")
                previous = json.loads(raw) if raw.strip() else None
                if not isinstance(previous, dict):
                    previous = None
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                previous = None
            os.close(fd)
            return False, previous
        except OSError as exc:
            os.close(fd)
            raise ConfigError(f"Could not lock GPU resource {lock_path}: {exc}") from exc
        metadata = {**self.owner, "gpu_id": gpu_id, "lock_path": str(lock_path)}
        encoded = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
        try:
            os.ftruncate(fd, 0)
            os.pwrite(fd, encoded, 0)
            os.fsync(fd)
        except OSError:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            raise
        self.acquired.append((gpu_id, fd))
        return True, None

    def acquire(self) -> list[dict[str, Any]]:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            for gpu_id in self.gpu_ids:
                acquired, previous = self._acquire_one(gpu_id)
                if not acquired:
                    state = "live" if previous is not None and owner_is_live(previous) else "held"
                    raise ConfigError(
                        f"GPU {gpu_id} has a {state} cross-batch lock at "
                        f"{self.lock_path(gpu_id)}; owner={previous}"
                    )
        except Exception:
            self.release()
            raise
        return [
            {
                "gpu_id": gpu_id,
                "path": str(self.lock_path(gpu_id)),
                "owner": {**self.owner, "gpu_id": gpu_id},
            }
            for gpu_id, _fd in self.acquired
        ]

    def release(self) -> None:
        for _gpu_id, fd in reversed(self.acquired):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
        self.acquired.clear()


def process_identity(process: subprocess.Popen[str]) -> dict[str, Any]:
    return {
        "pid": process.pid,
        "start_ticks": proc_start_ticks(process.pid),
        "returncode": process.poll(),
    }


def json_request(url: str, timeout: float) -> dict[str, Any]:
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object from {url}")
    return payload


def json_post_request(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    opener = build_opener(ProxyHandler({}))
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"expected a JSON object from {url}")
    return result


def preflight_agent(config: Config) -> dict[str, Any]:
    health = json_request(f"{config.agent_api_base_url}/health", timeout=5.0)
    if health.get("status") != "ok":
        raise ConfigError(f"9104 planner is not healthy: status={health.get('status')!r}")
    capacity = health.get("max_concurrent_requests")
    if not isinstance(capacity, int) or capacity < config.slots:
        raise ConfigError(
            f"9104 max_concurrent_requests={capacity!r}, but {config.slots} isolated slots require >= {config.slots}"
        )
    expected = {
        "model": "gpt-5.5",
        "api_mode": "responses_compat",
        "reasoning_effort": "xhigh",
    }
    mismatches = [
        f"{key}={health.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if health.get(key) != value
    ]
    if mismatches:
        raise ConfigError("9104 formal configuration mismatch: " + "; ".join(mismatches))
    return health


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise ConfigError(f"Missing {description}: {path}")


def path_is_within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    boundary = root.resolve()
    return resolved == boundary or boundary in resolved.parents


def paths_overlap(first: Path, second: Path) -> bool:
    return path_is_within(first, second) or path_is_within(second, first)


def preflight_paths(config: Config) -> None:
    required_external_paths = {
        "RMBENCH_ASSETS_ROOT": config.assets_root,
        "ROBOHARN_EVO_SAM3_REPO": config.sam3_repo,
        "ROBOHARN_EVO_SAM3_CHECKPOINT": config.sam3_checkpoint,
        "ROBOHARN_EVO_SAM3_BPE_PATH": config.sam3_bpe_path,
        "CAUSALWAM_ROOT": config.causalwam_root,
    }
    missing_external_paths = [
        name for name, path in required_external_paths.items() if path is None
    ]
    if missing_external_paths:
        raise ConfigError(
            "The copied formal chain requires explicit read-only external paths "
            "and has no donor fallback; set: "
            + ", ".join(missing_external_paths)
        )
    assert config.assets_root is not None
    assert config.sam3_repo is not None
    assert config.sam3_checkpoint is not None
    assert config.sam3_bpe_path is not None
    assert config.causalwam_root is not None
    require_path(config.assets_root / "embodiments", "RMBench embodiment assets")
    require_path(config.assets_root / "objects", "RMBench object assets")
    eval_result_root = (config.roboharn_project_root / "eval_result").resolve()
    if not path_is_within(config.output_root, eval_result_root):
        raise ConfigError(
            "RMBENCH_OUTPUT_ROOT must stay under the RoboHarn-Evo eval_result tree: "
            f"{config.output_root} vs {eval_result_root}"
        )
    for label, path in (
        ("batch log root", config.batch_log_root),
        ("GPU lock root", config.gpu_lock_root),
        ("runtime config root", config.runtime_config_root),
        ("runtime workspace root", config.runtime_workspace_root),
        ("runtime cache root", config.runtime_cache_root),
        ("runtime temp root", config.runtime_temp_root),
    ):
        if not path_is_within(path, config.output_root):
            raise ConfigError(
                f"{label} must stay under RMBENCH_OUTPUT_ROOT "
                f"{config.output_root}: {path}"
            )
    for label, readonly_root in (
        ("copied benchmark source", config.repo_root),
        ("RMBench assets", config.assets_root),
        ("SAM3 repository", config.sam3_repo),
        ("SAM3 checkpoint", config.sam3_checkpoint),
        ("SAM3 BPE vocabulary", config.sam3_bpe_path),
        ("CausalWAM repository", config.causalwam_root),
    ):
        if paths_overlap(config.output_root, readonly_root):
            raise ConfigError(
                f"RMBENCH_OUTPUT_ROOT overlaps read-only {label}: "
                f"{config.output_root} vs {readonly_root}"
            )
    require_path(config.repo_root / SEQUENTIAL_SCRIPT_REL, "sequential launcher")
    require_path(config.repo_root / SAM3_SERVER_REL, "SAM3 service script")
    require_path(config.repo_root / f"task_config/{config.task_config}.yml", "task config")
    require_path(config.sam3_python, "SAM3 Python")
    require_path(config.sam3_repo, "SAM3 repository")
    require_path(config.sam3_checkpoint, "SAM3 checkpoint")
    require_path(config.sam3_bpe_path, "SAM3 BPE vocabulary")
    for task in config.tasks:
        require_path(config.repo_root / f"envs/{task}.py", f"{task} environment")
        require_path(
            config.repo_root / f"data/data/{task}/{config.task_config}",
            f"{task} task data",
        )
        if config.instruction_set == "rmbench_original":
            instruction = config.repo_root / f"description/task_instruction/{task}.json"
        else:
            instruction = (
                config.repo_root
                / f"description/task_instruction_sets/{config.instruction_set}/{task}.json"
            )
        require_path(instruction, f"{task} instruction")


def preflight_gpus(config: Config) -> dict[int, dict[str, str]]:
    try:
        output = run_nvidia_smi(
            [
                "--query-gpu=index,uuid,pci.bus_id",
                "--format=csv,noheader,nounits",
            ]
        )
    except ConfigError as exc:
        raise ConfigError(f"Could not enumerate GPUs: {exc}") from exc

    inventory: dict[int, dict[str, str]] = {}
    seen_uuids: set[str] = set()
    seen_pci_bus_ids: set[str] = set()
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",", 2)]
        if (
            len(fields) != 3
            or not fields[0].isdigit()
            or not fields[1]
            or not fields[2]
        ):
            raise ConfigError(f"Could not parse nvidia-smi GPU inventory row: {raw_line!r}")
        gpu_id = int(fields[0])
        gpu_uuid = fields[1]
        if gpu_uuid.lower() in {"n/a", "[n/a]"}:
            raise ConfigError(f"nvidia-smi GPU {gpu_id} has no usable UUID")
        if gpu_id in inventory:
            raise ConfigError(f"nvidia-smi reported duplicate GPU index: {gpu_id}")
        if gpu_uuid in seen_uuids:
            raise ConfigError(f"nvidia-smi reported duplicate GPU UUID: {gpu_uuid}")
        pci_bus_id = normalize_pci_bus_id(fields[2])
        if pci_bus_id is None:
            raise ConfigError(
                f"nvidia-smi GPU {gpu_id} has an invalid PCI bus ID: {fields[2]!r}"
            )
        if pci_bus_id in seen_pci_bus_ids:
            raise ConfigError(
                f"nvidia-smi reported duplicate GPU PCI bus ID: {pci_bus_id}"
            )
        seen_uuids.add(gpu_uuid)
        seen_pci_bus_ids.add(pci_bus_id)
        inventory[gpu_id] = {
            "uuid": gpu_uuid,
            "pci_bus_id": pci_bus_id,
        }
    if not inventory:
        raise ConfigError("nvidia-smi returned an empty GPU inventory")

    available = set(inventory)
    requested = set(config.eval_gpus) | set(config.sam3_gpus)
    missing = sorted(requested - available)
    if missing:
        raise ConfigError(f"Requested GPU indices are unavailable: {missing}; available={sorted(available)}")
    return {gpu_id: inventory[gpu_id] for gpu_id in sorted(requested)}


def run_nvidia_smi(arguments: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise ConfigError(f"GPU occupancy preflight failed for nvidia-smi {arguments}: {exc}") from exc
    return completed.stdout


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def inspect_causalwam_process(
    pid: int,
    causalwam_root: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Verify one explicitly allowlisted PID against exact /proc evidence.

    NVIDIA normally reports only a generic executable name such as
    ``python``.  That name is not sufficient to identify CausalWAM.  A PID is
    eligible only when the operator listed it explicitly *and* its live
    command contains an existing Python/shell code path inside the configured
    CausalWAM checkout.  No substring matching is used.
    """
    if pid <= 0:
        return {"verified": False, "reason": "pid_not_positive", "pid": pid}
    root = causalwam_root.expanduser().resolve()
    if not root.is_dir():
        return {
            "verified": False,
            "reason": "causalwam_root_not_directory",
            "pid": pid,
            "causalwam_root": str(root),
        }
    proc_dir = proc_root / str(pid)
    try:
        stat_text = (proc_dir / "stat").read_text(encoding="utf-8")
        stat_fields = stat_text.rsplit(") ", 1)[1].split()
        start_ticks_before = int(stat_fields[19])
        raw_cmdline = (proc_dir / "cmdline").read_bytes()
        cwd = (proc_dir / "cwd").resolve(strict=True)
        executable = (proc_dir / "exe").resolve(strict=True)
    except (FileNotFoundError, IndexError, OSError, UnicodeError, ValueError) as exc:
        return {
            "verified": False,
            "reason": "proc_identity_unreadable",
            "pid": pid,
            "error_type": type(exc).__name__,
        }
    argv = [
        part.decode("utf-8", errors="surrogateescape")
        for part in raw_cmdline.split(b"\0")
        if part
    ]
    if not argv:
        return {"verified": False, "reason": "empty_cmdline", "pid": pid}

    matched_code_path: Path | None = None
    for argument in argv:
        # Deliberately accept only a real code-file argument.  Strings such as
        # "CausalWAM worker" and generic process names can never grant access.
        if argument.startswith("-"):
            continue
        candidate = Path(argument)
        if candidate.suffix not in {".py", ".sh"}:
            continue
        if not candidate.is_absolute():
            candidate = cwd / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file() and _path_is_within(resolved, root):
            matched_code_path = resolved
            break
    try:
        current_stat_text = (proc_dir / "stat").read_text(encoding="utf-8")
        current_fields = current_stat_text.rsplit(") ", 1)[1].split()
        start_ticks_after = int(current_fields[19])
    except (FileNotFoundError, IndexError, OSError, UnicodeError, ValueError):
        return {"verified": False, "reason": "process_changed_during_check", "pid": pid}
    if start_ticks_before != start_ticks_after:
        return {"verified": False, "reason": "process_changed_during_check", "pid": pid}
    if matched_code_path is None:
        return {
            "verified": False,
            "reason": "no_causalwam_code_path_in_cmdline",
            "pid": pid,
            "start_ticks": start_ticks_before,
            "cwd": str(cwd),
            "executable": str(executable),
            "cmdline_sha256": hashlib.sha256(raw_cmdline).hexdigest(),
        }
    return {
        "verified": True,
        "reason": "exact_repo_code_path",
        "pid": pid,
        "start_ticks": start_ticks_before,
        "cwd": str(cwd),
        "executable": str(executable),
        "matched_code_path": str(matched_code_path),
        "causalwam_root": str(root),
        "cmdline_sha256": hashlib.sha256(raw_cmdline).hexdigest(),
    }


def preflight_gpu_occupancy(config: Config) -> dict[str, Any]:
    """Reject unknown pre-existing GPU clients and audit explicit exceptions."""
    inventory_text = run_nvidia_smi(
        ["--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    )
    uuid_to_index: dict[str, int] = {}
    for raw_line in inventory_text.splitlines():
        fields = [field.strip() for field in raw_line.split(",", 1)]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1]:
            raise ConfigError(f"Could not parse nvidia-smi GPU inventory row: {raw_line!r}")
        uuid_to_index[fields[1]] = int(fields[0])

    requested = sorted(set(config.eval_gpus) | set(config.sam3_gpus))
    occupancy: dict[int, dict[int, dict[str, Any]]] = {gpu_id: {} for gpu_id in requested}
    compute_text = run_nvidia_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    for raw_line in compute_text.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",", 2)]
        if len(fields) < 2 or fields[0] not in uuid_to_index or not fields[1].isdigit():
            raise ConfigError(f"Could not parse nvidia-smi compute process row: {raw_line!r}")
        gpu_id = uuid_to_index[fields[0]]
        if gpu_id not in occupancy:
            continue
        pid = int(fields[1])
        occupancy[gpu_id][pid] = {
            "pid": pid,
            "types": ["compute"],
            "process_name": fields[2] if len(fields) > 2 else "",
            "sources": ["query-compute-apps"],
        }

    # pmon reports graphics-only and mixed C/G clients that are absent from
    # query-compute-apps.  We fail closed if it cannot be queried or parsed.
    pmon_outputs = [
        run_nvidia_smi(["pmon", "-i", str(gpu_id), "-c", "1"])
        for gpu_id in requested
    ]
    pmon_text = "\n".join(pmon_outputs)
    for raw_line in pmon_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3 or not fields[0].isdigit():
            raise ConfigError(f"Could not parse nvidia-smi pmon row: {raw_line!r}")
        gpu_id = int(fields[0])
        if gpu_id not in occupancy:
            continue
        if fields[1] == "-":
            continue
        if not fields[1].isdigit():
            raise ConfigError(f"Could not parse nvidia-smi pmon PID: {raw_line!r}")
        pid = int(fields[1])
        process_type = fields[2].lower()
        if process_type in {"-", "none"}:
            continue
        record = occupancy[gpu_id].setdefault(
            pid,
            {
                "pid": pid,
                "types": [],
                "process_name": fields[-1] if len(fields) > 3 else "",
                "sources": [],
            },
        )
        if process_type not in record["types"]:
            record["types"].append(process_type)
        if "pmon" not in record["sources"]:
            record["sources"].append("pmon")

    configured_causalwam_pids = set(config.allowed_causalwam_pids)
    identity_cache: dict[int, dict[str, Any]] = {}
    serializable: dict[str, list[dict[str, Any]]] = {}
    ignored: dict[str, list[dict[str, Any]]] = {}
    blocking: dict[str, list[dict[str, Any]]] = {}
    for gpu_id, records in occupancy.items():
        gpu_records: list[dict[str, Any]] = []
        for record in sorted(records.values(), key=lambda item: item["pid"]):
            pid = int(record["pid"])
            enriched = dict(record)
            explicitly_allowlisted = pid in configured_causalwam_pids
            eligible_for_causalwam_check = (
                explicitly_allowlisted
                or config.allow_verified_causalwam_gpu_occupancy
            )
            if eligible_for_causalwam_check:
                identity = identity_cache.setdefault(
                    pid,
                    inspect_causalwam_process(pid, config.causalwam_root),
                )
                enriched["causalwam_identity"] = identity
                if identity.get("verified") is True:
                    enriched["occupancy_disposition"] = (
                        "ignored_explicit_causalwam"
                        if explicitly_allowlisted
                        else "ignored_auto_verified_causalwam"
                    )
                    ignored.setdefault(str(gpu_id), []).append(enriched)
                else:
                    enriched["occupancy_disposition"] = (
                        "blocked_identity_mismatch"
                        if explicitly_allowlisted
                        else "blocked_not_verified_causalwam"
                    )
                    blocking.setdefault(str(gpu_id), []).append(enriched)
            else:
                enriched["occupancy_disposition"] = "blocked_not_allowlisted"
                blocking.setdefault(str(gpu_id), []).append(enriched)
            gpu_records.append(enriched)
        serializable[str(gpu_id)] = gpu_records
    occupied_gpu_ids = [int(gpu_id) for gpu_id, records in serializable.items() if records]
    blocking_gpu_ids = [int(gpu_id) for gpu_id in blocking]
    result = {
        "checked_at": utc_now(),
        "requested_gpu_ids": requested,
        "allow_occupied_gpu": config.allow_occupied_gpu,
        "causalwam_exception_policy": {
            "mode": "exact_repo_code_path",
            "auto_verify_all_occupants": (
                config.allow_verified_causalwam_gpu_occupancy
            ),
            "configured_pids": sorted(configured_causalwam_pids),
            "causalwam_root": str(config.causalwam_root),
        },
        "processes_by_gpu": serializable,
        "ignored_causalwam_processes_by_gpu": ignored,
        "blocking_processes_by_gpu": blocking,
        "occupied_gpu_ids": occupied_gpu_ids,
        "blocking_gpu_ids": blocking_gpu_ids,
        "passed": not blocking or config.allow_occupied_gpu,
    }
    if blocking and not config.allow_occupied_gpu:
        raise ConfigError(
            "Requested GPUs have external compute/graphics processes; refusing to share: "
            + json.dumps(blocking, sort_keys=True)
        )
    return result


def preflight_ports(config: Config) -> None:
    sockets: list[socket.socket] = []
    try:
        for slot in range(config.slots):
            port = config.sam3_base_port + slot
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                sock.close()
                raise ConfigError(f"SAM3 port {port} is already occupied: {exc}") from exc
            sockets.append(sock)
    finally:
        for sock in sockets:
            sock.close()


def check_new_batch_root(path: Path) -> None:
    if path.exists():
        try:
            next(path.iterdir())
        except StopIteration:
            return
        raise ConfigError(f"Batch log root already exists and is non-empty: {path}")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_plan(path: Path, jobs: list[Job]) -> None:
    columns = (
        "job_index",
        "slot",
        "slot_ordinal",
        "task",
        "eval_start_seed",
        "eval_gpu",
        "sam3_gpu",
        "sam3_url",
    )
    lines = ["\t".join(columns)]
    for job in jobs:
        row = asdict(job)
        lines.append("\t".join(str(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_pci_bus_id(value: Any) -> str | None:
    """Normalize NVIDIA PCI spellings for independent audit checks."""

    text = str(value or "").strip().lower()
    match = re.fullmatch(
        r"(?P<domain>[0-9a-f]{1,8}):(?P<bus>[0-9a-f]{1,2}):"
        r"(?P<device>[0-9a-f]{1,2})\.(?P<function>[0-7])",
        text,
    )
    if match is None:
        return None
    domain = int(match.group("domain"), 16)
    if domain > 0xFFFF:
        return None
    return (
        f"{domain:04x}:{int(match.group('bus'), 16):02x}:"
        f"{int(match.group('device'), 16):02x}.{match.group('function')}"
    )


def config_manifest(config: Config) -> dict[str, Any]:
    result = asdict(config)
    for key, value in list(result.items()):
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
    return result


def tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


class Scheduler:
    def __init__(
        self,
        config: Config,
        jobs: list[Job],
        *,
        token: str | None = None,
        gpu_locks: GpuLockSet | None = None,
    ) -> None:
        self.config = config
        self.jobs = jobs
        self.sam_processes: list[OwnedProcess] = []
        self.slot_processes: list[OwnedProcess] = []
        boot_path = Path("/proc/sys/kernel/random/boot_id")
        self.boot_id = boot_path.read_text(encoding="utf-8").strip() if boot_path.exists() else ""
        self.token = token or f"{config.batch_stamp}_{os.getpid()}_{time.time_ns()}"
        self.gpu_locks = gpu_locks
        # Populated only by the host-side nvidia-smi preflight.  launch_slots
        # fails closed if the expected PCI address for an eval GPU is absent.
        self.gpu_inventory: dict[int, dict[str, str]] = {}
        self.manifest: dict[str, Any] = {
            "schema": "roboharn_evo/parallel4_isolated_batch/v1",
            "schema_version": 1,
            "state": "initializing",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "batch_token": self.token,
            "owner": {
                "pid": os.getpid(),
                "start_ticks": proc_start_ticks(os.getpid()),
                "boot_id": self.boot_id,
                "script": str(SCRIPT_PATH),
            },
            "config": config_manifest(config),
            "plan_file": str(config.batch_log_root / PLAN_NAME),
            "jobs": [asdict(job) for job in jobs],
            "sam3_services": [],
            "slots": [],
            "gpu_resource_locks": [],
            "gpu_occupancy_preflight": {},
            "cleanup": {},
        }

    @property
    def manifest_path(self) -> Path:
        return self.config.batch_log_root / MANIFEST_NAME

    @property
    def sam3_ready_manifest_path(self) -> Path:
        return self.config.batch_log_root / SAM3_READY_MANIFEST_NAME

    def persist(self, state: str | None = None) -> None:
        if state is not None:
            self.manifest["state"] = state
        self.manifest["updated_at"] = utc_now()
        write_json_atomic(self.manifest_path, self.manifest)

    def sam3_ready_evidence(self) -> dict[str, Any]:
        """Return a secret-free, immutable summary of the existing SAM3 gates."""

        services: list[dict[str, Any]] = []
        for service in self.manifest["sam3_services"]:
            health = service.get("health", {})
            probe = service.get("inference_probe", {})
            services.append(
                {
                    "slot": service.get("slot"),
                    "service_url": service.get("url"),
                    "port": service.get("port"),
                    "sam3_gpu_id": service.get("gpu"),
                    "sam3_gpu_uuid": service.get("gpu_uuid"),
                    "service_output_root": service.get("output_root"),
                    "service_state": service.get("state"),
                    "health_verified": service.get("health_verified") is True,
                    "backend": health.get("backend"),
                    "checkpoint_path": health.get("checkpoint"),
                    "device": health.get("device"),
                    "cuda_available": health.get("cuda_available"),
                    "cuda_current_device": health.get("cuda_current_device"),
                    "visible_gpu_uuid": health.get("cuda_visible_devices"),
                    "cuda_device_order": health.get("cuda_device_order"),
                    "confidence_threshold": health.get("confidence_threshold"),
                    "process_pid": health.get("pid"),
                    "real_probe": {
                        "required": self.config.sam3_real_probe,
                        "success": probe.get("success"),
                        "num_detections": probe.get("num_detections"),
                        "completed_at": probe.get("completed_at"),
                    },
                }
            )
        return {
            "schema": "roboharn_evo/sam3_scheduler_readiness/v1",
            "schema_version": 1,
            "recorded_at": utc_now(),
            "batch_stamp": self.config.batch_stamp,
            "state": "sam3_ready",
            "sam3_amp_dtype": self.config.sam3_amp_dtype,
            "sam3_confidence_threshold": self.config.sam3_confidence_threshold,
            "services": services,
        }

    def prepare(self) -> None:
        self.config.batch_log_root.mkdir(parents=True, exist_ok=True)
        (self.config.batch_log_root / "sam3").mkdir()
        (self.config.batch_log_root / "slots").mkdir()
        (self.config.batch_log_root / "segmentation_artifacts").mkdir()
        self.config.runtime_config_root.mkdir(parents=True, exist_ok=True)
        self.config.runtime_workspace_root.mkdir(parents=True, exist_ok=True)
        self.config.runtime_temp_root.mkdir(parents=True, exist_ok=True)
        for relative_path in RUNTIME_CACHE_ENVIRONMENT_LAYOUT.values():
            (self.config.runtime_cache_root / relative_path).mkdir(
                parents=True,
                exist_ok=True,
            )
        write_plan(self.config.batch_log_root / PLAN_NAME, self.jobs)
        self.persist("prepared")

    def launch_sam3(self) -> None:
        for slot in range(self.config.slots):
            physical_gpu = self.config.sam3_gpus[slot]
            gpu_identity = self.gpu_inventory.get(physical_gpu)
            gpu_uuid = "" if gpu_identity is None else str(gpu_identity.get("uuid", "")).strip()
            if not gpu_uuid:
                raise ConfigError(
                    "SAM3 launch is missing the host GPU UUID identity "
                    f"for physical GPU {physical_gpu}; the nvidia-smi preflight "
                    "must run first"
                )
            service_root = self.config.batch_log_root / "sam3" / f"slot_{slot}"
            service_root.mkdir(parents=True)
            # The service-enforced write root is the same per-slot artifact
            # tree used by the rollout client.  Keeping these identical lets
            # the server reject accidental cross-slot output paths.
            output_root = (
                self.config.batch_log_root / "segmentation_artifacts" / f"slot_{slot}"
            ).resolve()
            service_root.mkdir(parents=True, exist_ok=True)
            output_root.mkdir(parents=True)
            log_path = service_root / "service.log"
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            service_temp_root = self.config.runtime_temp_root / f"sam{slot}"
            service_temp_root.mkdir(parents=True, exist_ok=True)
            instance_id = f"{self.token}:slot_{slot}"
            port = self.config.sam3_base_port + slot
            command = [
                str(self.config.sam3_python),
                "-u",
                str(self.config.repo_root / SAM3_SERVER_REL),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--sam3-repo",
                str(self.config.sam3_repo),
                "--checkpoint",
                str(self.config.sam3_checkpoint),
                "--bpe-path",
                str(self.config.sam3_bpe_path),
                "--device",
                "cuda",
                "--amp-dtype",
                self.config.sam3_amp_dtype,
                "--confidence-threshold",
                str(self.config.sam3_confidence_threshold),
                "--output-root",
                str(output_root),
                "--allowed-output-root",
                str(output_root),
                "--instance-id",
                instance_id,
            ]
            env = os.environ.copy()
            env.update(
                runtime_environment(
                    self.config,
                    temp_root=service_temp_root,
                )
            )
            env.update(
                {
                    # UUID selection is invariant to CUDA's numeric device
                    # ordering.  Inside the child this physical GPU is still
                    # exposed as logical cuda:0.
                    "CUDA_VISIBLE_DEVICES": gpu_uuid,
                    # Make CUDA ordinals deterministic relative to the
                    # nvidia-smi PCI inventory used by this scheduler.
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "PYTHONUNBUFFERED": "1",
                    "RMBENCH_ROOT": str(self.config.repo_root),
                    "ROBOHARN_EVO_SAM3_REPO": str(self.config.sam3_repo),
                    "ROBOHARN_EVO_SAM3_CHECKPOINT": str(self.config.sam3_checkpoint),
                    "ROBOHARN_EVO_SAM3_BPE_PATH": str(self.config.sam3_bpe_path),
                }
            )
            process = subprocess.Popen(
                command,
                cwd=self.config.repo_root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            owned = OwnedProcess(
                kind="sam3",
                slot=slot,
                process=process,
                log_path=log_path,
                log_handle=log_handle,
                start_ticks=proc_start_ticks(process.pid),
            )
            self.sam_processes.append(owned)
            self.manifest["sam3_services"].append(
                {
                    "slot": slot,
                    "gpu": physical_gpu,
                    "gpu_uuid": gpu_uuid,
                    "cuda_visible_device_token": gpu_uuid,
                    "port": port,
                    "url": f"http://127.0.0.1:{port}",
                    "instance_id": instance_id,
                    "output_root": str(output_root),
                    "log_path": str(log_path),
                    "process": process_identity(process),
                    "state": "starting",
                    "health_verified": False,
                }
            )
        self.persist("starting_sam3")
        self.wait_for_sam3_health()

    def wait_for_sam3_health(self) -> None:
        deadline = time.monotonic() + self.config.sam3_startup_timeout_sec
        pending = {item.slot for item in self.sam_processes}
        while pending and time.monotonic() < deadline:
            for item in self.sam_processes:
                if item.slot not in pending:
                    continue
                rc = item.process.poll()
                if rc is not None:
                    raise ConfigError(
                        f"SAM3 slot {item.slot} exited during startup with rc={rc}.\n{tail(item.log_path)}"
                    )
                expected = self.manifest["sam3_services"][item.slot]
                try:
                    health = json_request(f"{expected['url']}/health", timeout=2.0)
                except RuntimeError:
                    continue
                actual_output = str(Path(str(health.get("output_root", ""))).expanduser().resolve())
                checks = {
                    "status": health.get("status") == "ok",
                    "instance_id": health.get("instance_id") == expected["instance_id"],
                    "pid": health.get("pid") == item.process.pid,
                    "output_root": actual_output == expected["output_root"],
                    "device": health.get("device") == "cuda",
                    "cuda_available": health.get("cuda_available") is True,
                    "cuda_current_device": health.get("cuda_current_device") == 0,
                    "cuda_visible_device_token": health.get(
                        "cuda_visible_devices"
                    )
                    == expected["cuda_visible_device_token"],
                    "cuda_device_order": health.get("cuda_device_order")
                    == "PCI_BUS_ID",
                }
                expected["health_checks"] = checks
                if not all(checks.values()):
                    raise ConfigError(
                        f"SAM3 slot {item.slot} health identity mismatch: checks={checks}, health={health}"
                    )
                expected["health_verified"] = True
                expected["health"] = health
                expected["process"] = process_identity(item.process)
                expected["state"] = "health_verified"
                pending.remove(item.slot)
                self.persist()
            if pending:
                time.sleep(self.config.sam3_health_interval_sec)
        if pending:
            details = "\n".join(
                f"slot {item.slot}: {tail(item.log_path, 8)}"
                for item in self.sam_processes
                if item.slot in pending
            )
            raise ConfigError(f"Timed out waiting for SAM3 slots {sorted(pending)}.\n{details}")
        if self.config.sam3_real_probe:
            self.run_sam3_inference_probes()
        self.persist("sam3_ready")
        # Freeze the post-probe identity before slot lifecycle updates mutate
        # the live scheduler manifest.  Rollout provenance hashes this atomic
        # snapshot; it is evidence only and is not a new launch gate.
        write_json_atomic(
            self.sam3_ready_manifest_path,
            self.sam3_ready_evidence(),
        )

    def run_sam3_inference_probes(self) -> None:
        # A tiny local PPM avoids any image-library dependency in this
        # orchestrator.  Each request still exercises the real processor on
        # its assigned CUDA device after /health identity verification.
        image_path = (self.config.batch_log_root / "sam3" / "startup_probe.ppm").resolve()
        width = 32
        height = 32
        pixels = bytes([127, 127, 127]) * width * height
        image_path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + pixels)

        def probe(slot: int) -> tuple[int, dict[str, Any]]:
            service = self.manifest["sam3_services"][slot]
            service_output_root = Path(
                str(
                    service.get("output_root")
                    or (
                        self.config.batch_log_root
                        / "segmentation_artifacts"
                        / f"slot_{slot}"
                    )
                )
            ).resolve()
            probe_root = (service_output_root / "_startup_probe").resolve()
            result = json_post_request(
                f"{service['url']}/segment_image",
                {
                    "image_path": str(image_path),
                    "text_prompt": "object",
                    "object_id": f"startup_probe_slot_{slot}",
                    "top_k": 1,
                    "output_dir": str(probe_root),
                },
                timeout=self.config.sam3_probe_timeout_sec,
            )
            if result.get("success") is not True:
                raise RuntimeError(f"SAM3 slot {slot} probe returned failure: {result}")
            return slot, {
                "success": True,
                "num_detections": result.get("num_detections"),
                "completed_at": utc_now(),
            }

        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=self.config.slots) as executor:
            futures = {executor.submit(probe, slot): slot for slot in range(self.config.slots)}
            for future in as_completed(futures):
                slot = futures[future]
                try:
                    completed_slot, summary = future.result()
                except Exception as exc:  # preserve all slot diagnostics
                    failures.append(f"slot {slot}: {exc}")
                else:
                    self.manifest["sam3_services"][completed_slot]["inference_probe"] = summary
                    self.persist()
        if failures:
            raise ConfigError("SAM3 startup inference probe failed: " + "; ".join(failures))
        for service in self.manifest["sam3_services"]:
            service["state"] = "ready"
        self.persist()

    def launch_slots(self) -> None:
        by_slot: list[list[Job]] = [[] for _ in range(self.config.slots)]
        for job in self.jobs:
            by_slot[job.slot].append(job)
        sequential = self.config.repo_root / SEQUENTIAL_SCRIPT_REL
        for slot, slot_jobs in enumerate(by_slot):
            slot_root = (self.config.batch_log_root / "slots" / f"slot_{slot}").resolve()
            artifact_root = (
                self.config.batch_log_root / "segmentation_artifacts" / f"slot_{slot}"
            ).resolve()
            slot_root.mkdir(parents=True)
            # launch_sam3 creates the same per-slot artifact root so that the
            # service can enforce it before rollout clients start.
            artifact_root.mkdir(parents=True, exist_ok=True)
            (slot_root / "runtime_workspace").mkdir(parents=True, exist_ok=True)
            (slot_root / "runtime_configs").mkdir(parents=True, exist_ok=True)
            slot_temp_root = self.config.runtime_temp_root / f"s{slot}"
            slot_temp_root.mkdir(parents=True, exist_ok=True)
            for relative_path in RUNTIME_CACHE_ENVIRONMENT_LAYOUT.values():
                (slot_root / "runtime_caches" / relative_path).mkdir(
                    parents=True,
                    exist_ok=True,
                )
            slot_record: dict[str, Any] = {
                "slot": slot,
                "jobs": [job.job_index for job in slot_jobs],
                "batch_log_root": str(slot_root),
                "segmentation_artifact_root": str(artifact_root),
                "state": "empty" if not slot_jobs else "starting",
            }
            self.manifest["slots"].append(slot_record)
            if not slot_jobs:
                continue
            eval_gpu = self.config.eval_gpus[slot]
            gpu_identity = self.gpu_inventory.get(eval_gpu)
            expected_pci_bus_id = (
                "" if gpu_identity is None else str(gpu_identity.get("pci_bus_id", "")).strip()
            )
            expected_gpu_uuid = (
                "" if gpu_identity is None else str(gpu_identity.get("uuid", "")).strip()
            )
            normalized_pci_bus_id = normalize_pci_bus_id(expected_pci_bus_id)
            if normalized_pci_bus_id is None or not expected_gpu_uuid:
                raise ConfigError(
                    "Renderer-device launch is missing the host GPU UUID/PCI identity "
                    f"for eval GPU {eval_gpu}; the nvidia-smi preflight must run first"
                )
            requested_render_alias = f"pci:{normalized_pci_bus_id}"
            per_job_renderer_provenance = [
                {
                    "job_index": job.job_index,
                    "task": job.task,
                    "eval_start_seed": job.eval_start_seed,
                    "path": str(
                        slot_root
                        / job.task
                        / f"episode_{job.eval_start_seed:06d}"
                        / "renderer_device_provenance.json"
                    ),
                }
                for job in slot_jobs
            ]
            log_path = slot_root / "slot_launcher.log"
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            run_jobs_csv = ",".join(
                f"{job.task}:{job.eval_start_seed}" for job in slot_jobs
            )
            env = os.environ.copy()
            # SAM3_BASE_PORT belongs only to this outer service pool.  The
            # delegated single-worker launcher gives it precedence over
            # SAM3_SERVICE_URL, so inheriting it would silently route every
            # slot to the first service port.
            env.pop("SAM3_BASE_PORT", None)
            env.update(
                runtime_environment(
                    self.config,
                    workspace_root=slot_root / "runtime_workspace",
                    cache_root=slot_root / "runtime_caches",
                    runtime_config_root=slot_root / "runtime_configs",
                    temp_root=slot_temp_root,
                )
            )
            external_evidence_specs = env.get(
                "RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS", ""
            ).strip()
            sam3_ready_spec = (
                "sam3_scheduler_ready_manifest="
                f"{self.sam3_ready_manifest_path}"
            )
            env["RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS"] = "\n".join(
                item for item in (external_evidence_specs, sam3_ready_spec) if item
            )
            env.update(
                {
                    "REPO_ROOT": str(self.config.repo_root),
                    "GPU_ID": str(self.config.eval_gpus[slot]),
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    # GPU_ID remains the stable physical index used in result
                    # names.  The CUDA selector uses the immutable UUID so a
                    # numeric ordinal cannot silently bind another card.
                    "RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN": expected_gpu_uuid,
                    # SAPIEN/Vulkan is bound by the slot's exact host PCI
                    # address.  CUDA remains independently isolated by the
                    # immutable UUID above, which maps to logical cuda:0.
                    "RMBENCH_RENDER_DEVICE": requested_render_alias,
                    "RMBENCH_RENDER_DEVICE_STRICT": "1",
                    "RMBENCH_EXPECTED_RENDER_CUDA_ID": "0",
                    "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID": normalized_pci_bus_id,
                    "RMBENCH_EXPECTED_PHYSICAL_GPU": str(eval_gpu),
                    # The sequential child replaces this placeholder with the
                    # concrete per-task/per-seed run-log path before eval.
                    "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH": str(
                        slot_root / "renderer_device_provenance.pending.json"
                    ),
                    "TASK_CONFIG": self.config.task_config,
                    "INSTRUCTION_SET": self.config.instruction_set,
                    "POLICY_NAME": self.config.policy_name,
                    "PERCEPTION_CONDITION": self.config.perception_condition,
                    "AGENT_API_BASE_URL": self.config.agent_api_base_url,
                    "SAM3_SERVICE_URL": f"http://127.0.0.1:{self.config.sam3_base_port + slot}",
                    "MAX_OBJECTS": str(self.config.max_objects),
                    "EVAL_STEP_LIMIT": str(self.config.eval_step_limit),
                    "MAX_ROUNDS": str(self.config.max_rounds),
                    "MAX_CONTROL_TURNS": str(self.config.max_control_turns),
                    "MAX_NO_PROGRESS_CONTROL_TURNS": str(
                        self.config.max_no_progress_control_turns
                    ),
                    "BATCH_STAMP": f"{self.config.batch_stamp}_slot{slot}",
                    "RUN_LABEL": self.config.run_label,
                    "BATCH_LOG_ROOT": str(slot_root),
                    "SEGMENTATION_ARTIFACT_ROOT": str(artifact_root),
                    "BATCH_SESSION": f"{self.config.tmux_session}_slot{slot}",
                    "BATCH_DETACH": "0",
                    "CONTINUE_ON_ERROR": "1" if self.config.continue_on_error else "0",
                    "TASKS_CSV": ",".join(self.config.tasks),
                    "EVAL_START_SEEDS_CSV": ",".join(str(seed) for seed in self.config.seeds),
                    "RUN_JOBS_CSV": run_jobs_csv,
                    "NO_PROXY": "127.0.0.1,localhost",
                    "no_proxy": "127.0.0.1,localhost",
                }
            )
            process = subprocess.Popen(
                ["/bin/bash", str(sequential), "--foreground"],
                cwd=self.config.repo_root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            owned = OwnedProcess(
                kind="slot",
                slot=slot,
                process=process,
                log_path=log_path,
                log_handle=log_handle,
                start_ticks=proc_start_ticks(process.pid),
            )
            self.slot_processes.append(owned)
            slot_record.update(
                {
                    "state": "running",
                    "run_jobs_csv": run_jobs_csv,
                    "log_path": str(log_path),
                    "process": process_identity(process),
                    "renderer_device_contract": {
                        "render_device_policy": self.config.render_device,
                        "requested_render_alias": requested_render_alias,
                        "expected_logical_cuda_id": 0,
                        "expected_physical_gpu": eval_gpu,
                        "expected_gpu_uuid": expected_gpu_uuid,
                        "expected_pci_bus_id": normalized_pci_bus_id,
                        "cuda_visible_device_token": expected_gpu_uuid,
                        "cuda_device_order": "PCI_BUS_ID",
                        "provenance_scope": "per_task_seed_run_log",
                        "per_job_provenance": per_job_renderer_provenance,
                    },
                }
            )
            self.persist()
            print(
                f"slot={slot} pid={process.pid} eval_gpu={self.config.eval_gpus[slot]} "
                f"sam3_gpu={self.config.sam3_gpus[slot]} jobs={run_jobs_csv}",
                flush=True,
            )
            if self.config.slot_start_delay_sec > 0:
                time.sleep(self.config.slot_start_delay_sec)
        self.persist("running")

    def audit_slot_renderer_provenance(
        self,
        slot: int,
        slot_record: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify every completed job's independently written device record."""

        status_path = Path(str(slot_record["batch_log_root"])) / "batch_status.tsv"
        errors: list[str] = []
        entries: list[dict[str, Any]] = []
        rows_by_job: dict[tuple[str, int], dict[str, str]] = {}
        if not status_path.is_file():
            errors.append(f"missing slot status ledger: {status_path}")
        else:
            try:
                with status_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    required = {"task", "eval_start_seed", "status"}
                    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                        errors.append(
                            "slot status ledger is missing required columns: "
                            f"{sorted(required)}"
                        )
                    else:
                        for row in reader:
                            task = str(row.get("task", ""))
                            raw_seed = str(row.get("eval_start_seed", ""))
                            try:
                                seed = int(raw_seed, 10)
                            except ValueError:
                                errors.append(
                                    f"slot status ledger has invalid seed {raw_seed!r}"
                                )
                                continue
                            key = (task, seed)
                            if key in rows_by_job:
                                errors.append(
                                    f"slot status ledger repeats task/seed {task}:{seed}"
                                )
                                continue
                            rows_by_job[key] = dict(row)
            except (OSError, csv.Error) as exc:
                errors.append(f"could not read slot status ledger {status_path}: {exc}")

        expected_jobs = [job for job in self.jobs if job.slot == slot]
        expected_keys = {(job.task, job.eval_start_seed) for job in expected_jobs}
        extra_keys = sorted(set(rows_by_job) - expected_keys)
        if extra_keys:
            errors.append(f"slot status ledger contains unassigned jobs: {extra_keys}")

        contract = slot_record.get("renderer_device_contract", {})
        paths_by_job = {
            int(item["job_index"]): Path(str(item["path"]))
            for item in contract.get("per_job_provenance", [])
            if isinstance(item, dict)
            and isinstance(item.get("job_index"), int)
            and item.get("path")
        }
        expected_pci = normalize_pci_bus_id(contract.get("expected_pci_bus_id"))
        expected_visible_token = str(
            contract.get("cuda_visible_device_token") or ""
        ).strip()
        expected_render_alias = str(
            contract.get("requested_render_alias") or ""
        ).strip()
        for job in expected_jobs:
            row = rows_by_job.get((job.task, job.eval_start_seed))
            if row is None:
                errors.append(
                    f"slot status ledger is missing assigned job "
                    f"{job.task}:{job.eval_start_seed}"
                )
                continue
            status_value = str(row.get("status", ""))
            entry: dict[str, Any] = {
                "job_index": job.job_index,
                "task": job.task,
                "eval_start_seed": job.eval_start_seed,
                "status": status_value,
            }
            if status_value == "skipped":
                entry["provenance_required"] = False
                entry["passed"] = True
                entries.append(entry)
                continue
            if status_value != "complete":
                entry["provenance_required"] = False
                entry["passed"] = False
                entries.append(entry)
                errors.append(
                    f"job {job.task}:{job.eval_start_seed} has unexpected "
                    f"successful-slot status {status_value!r}"
                )
                continue

            entry["provenance_required"] = True
            path = paths_by_job.get(job.job_index)
            entry["path"] = None if path is None else str(path)
            item_errors: list[str] = []
            payload: dict[str, Any] | None = None
            if path is None:
                item_errors.append("manifest has no renderer provenance path")
            else:
                try:
                    raw_provenance = path.read_bytes()
                    loaded = json.loads(raw_provenance.decode("utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    item_errors.append(f"could not read renderer provenance: {exc}")
                else:
                    entry["file_sha256"] = hashlib.sha256(raw_provenance).hexdigest()
                    if not isinstance(loaded, dict):
                        item_errors.append("renderer provenance is not a JSON object")
                    else:
                        payload = loaded
            if payload is not None:
                requested = payload.get("requested_device_info")
                if not isinstance(requested, dict):
                    requested = {}
                selected = payload.get("selected_device_info")
                if not isinstance(selected, dict):
                    selected = {}
                checks = {
                    "schema": payload.get("schema")
                    == "rmbench/renderer_device_binding/v1",
                    "mode": payload.get("mode") == "explicit",
                    "strict": payload.get("strict") is True,
                    "validation_passed": payload.get("validation_passed") is True,
                    "cuda_device_order": payload.get("cuda_device_order")
                    == "PCI_BUS_ID",
                    "cuda_visible_device_token": payload.get(
                        "cuda_visible_device_tokens"
                    )
                    == [expected_visible_token],
                    "requested_render_alias": payload.get("requested_device")
                    == expected_render_alias
                    and expected_render_alias == f"pci:{expected_pci}",
                    "expected_physical_gpu": str(
                        payload.get("expected_physical_gpu") or ""
                    )
                    == str(job.eval_gpu),
                    "expected_logical_cuda_id": payload.get(
                        "expected_logical_cuda_id"
                    )
                    == 0,
                    "selected_logical_cuda_id": selected.get("cuda_id") == 0,
                    "selected_is_cuda": selected.get("is_cuda") is True,
                    "selected_can_render": selected.get("can_render") is True,
                    "requested_is_cuda": requested.get("is_cuda") is True,
                    "requested_can_render": requested.get("can_render") is True,
                    "requested_pci_bus_id": expected_pci is not None
                    and requested.get("pci_bus_id_normalized") == expected_pci,
                    "selected_pci_bus_id": expected_pci is not None
                    and selected.get("pci_bus_id_normalized") == expected_pci,
                    "requested_selected_cuda_id": requested.get("cuda_id")
                    == selected.get("cuda_id"),
                    "requested_selected_pci_bus_id": requested.get(
                        "pci_bus_id_normalized"
                    )
                    == selected.get("pci_bus_id_normalized"),
                }
                failed_checks = [name for name, passed in checks.items() if not passed]
                if failed_checks:
                    item_errors.append(
                        "renderer provenance checks failed: " + ",".join(failed_checks)
                    )
                entry["checks"] = checks
            entry["passed"] = not item_errors
            entry["errors"] = item_errors
            entries.append(entry)
            errors.extend(
                f"job {job.task}:{job.eval_start_seed}: {message}"
                for message in item_errors
            )

        return {
            "schema": "roboharn_evo/renderer_device_audit/v1",
            "checked_at": utc_now(),
            "required": True,
            "status_ledger_path": str(status_path),
            "passed": not errors,
            "entries": entries,
            "errors": errors,
        }

    def monitor(self) -> int:
        live = {item.slot: item for item in self.slot_processes}
        first_failure = 0
        sigint_started_at: float | None = None
        sigint_grace_reported = False
        while live:
            dead_services = [
                item
                for item in self.sam_processes
                if item.process.poll() is not None
                and self.manifest["sam3_services"][item.slot].get("state") != "failed"
            ]
            if dead_services:
                for item in dead_services:
                    service = self.manifest["sam3_services"][item.slot]
                    service["state"] = "failed"
                    service["process"] = process_identity(item.process)
                self.persist()
                if first_failure == 0:
                    first_failure = 75
                    slots = ",".join(str(item.slot) for item in dead_services)
                    print(
                        f"SAM3 service failure in slot(s) {slots}; stopping all rollout slots with SIGINT",
                        file=sys.stderr,
                        flush=True,
                    )
                    self.signal_owned(list(live.values()), signal.SIGINT)
                    sigint_started_at = time.monotonic()
            for slot, item in list(live.items()):
                rc = item.process.poll()
                if rc is None:
                    continue
                item.process.wait()
                item.log_handle.close()
                record = self.manifest["slots"][slot]
                effective_rc = rc
                if rc == 0:
                    renderer_audit = self.audit_slot_renderer_provenance(slot, record)
                    record["renderer_device_audit"] = renderer_audit
                    if not renderer_audit["passed"]:
                        effective_rc = 76
                        print(
                            f"slot={slot} renderer provenance audit failed: "
                            + "; ".join(renderer_audit["errors"]),
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    record["renderer_device_audit"] = {
                        "schema": "roboharn_evo/renderer_device_audit/v1",
                        "checked_at": utc_now(),
                        "required": False,
                        "passed": None,
                        "reason": "slot_process_failed_before_success_audit",
                    }
                record["state"] = "complete" if effective_rc == 0 else "failed"
                record["process"] = process_identity(item.process)
                record["returncode"] = rc
                record["effective_returncode"] = effective_rc
                self.persist()
                print(
                    f"slot={slot} exited rc={rc} effective_rc={effective_rc}",
                    flush=True,
                )
                del live[slot]
                if effective_rc != 0 and first_failure == 0:
                    first_failure = effective_rc
                    print(
                        f"slot={slot} had an infrastructure/trace failure; stopping other slots with SIGINT",
                        file=sys.stderr,
                        flush=True,
                    )
                    self.signal_owned(list(live.values()), signal.SIGINT)
                    sigint_started_at = time.monotonic()
            if (
                live
                and sigint_started_at is not None
                and not sigint_grace_reported
                and time.monotonic() - sigint_started_at
                >= self.config.shutdown_sigint_grace_sec
            ):
                sigint_grace_reported = True
                self.record_cleanup_grace_expired_noexcept(
                    "monitor_slots", list(live.values())
                )
            if live:
                time.sleep(1.0)
        if sigint_grace_reported:
            self.mark_cleanup_complete("monitor_slots")
        return first_failure

    @staticmethod
    def signal_owned(items: list[OwnedProcess], sig: signal.Signals) -> None:
        for item in items:
            try:
                returncode = item.process.poll()
            except BaseException as exc:
                # Unknown liveness is treated as live.  This function is used
                # only while draining processes already owned by this batch.
                returncode = None
                print(
                    f"Could not poll {item.kind}[{item.slot}] pid={item.process.pid} "
                    f"before {sig.name}: {exc}; treating it as live",
                    file=sys.stderr,
                    flush=True,
                )
            if returncode is not None:
                continue
            try:
                item.process.send_signal(sig)
            except BaseException as exc:
                # A failed signal must never make cleanup fall through to
                # unlock while the process may still be alive.  The
                # subsequent wait deliberately remains in place.
                print(
                    f"Could not send {sig.name} to {item.kind}[{item.slot}] "
                    f"pid={item.process.pid}: {exc}; continuing to wait",
                    file=sys.stderr,
                    flush=True,
                )

    @staticmethod
    def wait_owned(
        items: list[OwnedProcess],
        *,
        grace_sec: float = 30.0,
        on_grace_expired: Any | None = None,
    ) -> None:
        """Wait for SIGINT-owned children without escalating or releasing early.

        The grace interval is an observability deadline, not a kill deadline.
        Once it expires, the caller is notified exactly once, but this method
        keeps polling until every child has really exited.  Therefore the
        scheduler retains its GPU locks and logs while a slow child handles
        Ctrl+C; it never falls through to SIGTERM/SIGKILL or starts later work
        on resources that may still be live.
        """
        pending = list(items)
        deadline = time.monotonic() + max(0.0, grace_sec)
        grace_reported = False
        poll_errors_reported: set[tuple[str, int, int]] = set()
        while pending:
            still_live: list[OwnedProcess] = []
            for item in pending:
                try:
                    returncode = item.process.poll()
                except BaseException as exc:
                    returncode = None
                    error_key = (item.kind, item.slot, item.process.pid)
                    if error_key not in poll_errors_reported:
                        poll_errors_reported.add(error_key)
                        print(
                            f"Could not poll {item.kind}[{item.slot}] "
                            f"pid={item.process.pid} during cleanup: {exc}; "
                            "treating it as live",
                            file=sys.stderr,
                            flush=True,
                        )
                if returncode is None:
                    still_live.append(item)
                    continue
                # poll() already established that wait() cannot block here.
                try:
                    item.process.wait()
                except Exception as exc:
                    # The process is already known to have exited.  Failure
                    # to reap/inspect it must not reopen the resource-safety
                    # decision or abandon another still-live child.
                    print(
                        f"Could not finalize exited {item.kind}[{item.slot}] "
                        f"pid={item.process.pid}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                try:
                    item.log_handle.close()
                except Exception as exc:
                    print(
                        f"Could not close log for exited {item.kind}[{item.slot}] "
                        f"pid={item.process.pid}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            pending = still_live
            if not pending:
                break
            now = time.monotonic()
            if not grace_reported and now >= deadline:
                grace_reported = True
                if on_grace_expired is not None:
                    try:
                        on_grace_expired(list(pending))
                    except BaseException as exc:
                        # Cleanup evidence is secondary to ownership safety.
                        # Even an interrupted or failed manifest write cannot
                        # let the scheduler return while a child is alive.
                        print(
                            "Could not record SIGINT cleanup grace expiry; "
                            f"continuing to wait with resources held: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
            # Wake exactly at the grace boundary when it is still pending;
            # afterwards use a bounded polling interval.
            sleep_sec = 0.25
            if not grace_reported:
                sleep_sec = min(sleep_sec, max(0.001, deadline - now))
            try:
                time.sleep(sleep_sec)
            except BaseException as exc:
                # A repeated terminal signal or an interrupted sleep must not
                # punch through the ownership barrier.  Keep polling until
                # every process has actually exited.
                print(
                    f"Cleanup wait interrupted ({exc}); continuing with GPU locks held",
                    file=sys.stderr,
                    flush=True,
                )

    def record_cleanup_grace_expired(
        self, phase: str, live: list[OwnedProcess]
    ) -> None:
        cleanup = self.manifest.setdefault("cleanup", {})
        cleanup[phase] = {
            "state": "sigint_grace_expired_waiting_no_escalation",
            "grace_sec": self.config.shutdown_sigint_grace_sec,
            "reported_at": utc_now(),
            "live_processes": [
                {
                    "kind": item.kind,
                    "slot": item.slot,
                    "pid": item.process.pid,
                    "start_ticks": item.start_ticks,
                    "current_identity": process_identity(item.process),
                    "log_path": str(item.log_path),
                }
                for item in live
            ],
        }
        self.persist("cleanup_waiting_after_sigint_grace")
        details = ", ".join(
            f"{item.kind}[{item.slot}]=pid:{item.process.pid}" for item in live
        )
        print(
            f"SIGINT cleanup grace expired for {phase}: {details}. "
            "Continuing to wait without TERM/KILL; GPU locks remain held.",
            file=sys.stderr,
            flush=True,
        )

    def record_cleanup_grace_expired_noexcept(
        self, phase: str, live: list[OwnedProcess]
    ) -> None:
        try:
            self.record_cleanup_grace_expired(phase, live)
        except BaseException as exc:
            print(
                f"Could not persist SIGINT cleanup state for {phase}; "
                f"continuing to wait with GPU locks held: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def mark_cleanup_complete(self, phase: str) -> None:
        record = self.manifest.setdefault("cleanup", {}).get(phase)
        if not isinstance(record, dict):
            return
        record["state"] = "exited_after_sigint_grace"
        record["completed_at"] = utc_now()
        try:
            self.persist("cleanup_resumed_after_sigint_grace")
        except BaseException as exc:
            print(
                f"Could not persist completed SIGINT cleanup state for {phase}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def stop_slots(self) -> None:
        live = self.live_owned(self.slot_processes)
        if live:
            print("Forwarding SIGINT to owned slot schedulers...", file=sys.stderr, flush=True)
            self.signal_owned(live, signal.SIGINT)
            self.wait_owned(
                live,
                grace_sec=self.config.shutdown_sigint_grace_sec,
                on_grace_expired=lambda remaining: (
                    self.record_cleanup_grace_expired_noexcept("slots", remaining)
                ),
            )
            self.mark_cleanup_complete("slots")
        for item in self.slot_processes:
            record = self.manifest["slots"][item.slot]
            record["process"] = process_identity(item.process)
            if record.get("state") == "running":
                record["state"] = "interrupted"

    def stop_sam3(self) -> None:
        live = self.live_owned(self.sam_processes)
        if live:
            print("Stopping owned SAM3 services with SIGINT...", flush=True)
            self.signal_owned(live, signal.SIGINT)
            self.wait_owned(
                live,
                grace_sec=self.config.shutdown_sigint_grace_sec,
                on_grace_expired=lambda remaining: (
                    self.record_cleanup_grace_expired_noexcept("sam3", remaining)
                ),
            )
            self.mark_cleanup_complete("sam3")
        for item in self.sam_processes:
            service = self.manifest["sam3_services"][item.slot]
            service["process"] = process_identity(item.process)
            if service.get("state") != "failed":
                service["state"] = (
                    "stopped" if item.process.poll() is not None else "stop_failed"
                )

    def shutdown(self, final_state: str) -> None:
        # shutdown() is also used by non-KeyboardInterrupt failure paths.
        # Make cleanup immune to a second terminal Ctrl+C/SIGTERM/HUP until
        # all owned children have exited, then restore exactly the handlers
        # that were active on entry.  The restoration matters when Scheduler
        # is embedded or tested without run_foreground's outer signal scope.
        previous_handlers: dict[signal.Signals, Any] = {}
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                previous_handlers[sig] = signal.signal(sig, signal.SIG_IGN)
            except (OSError, ValueError):
                pass
        try:
            self._shutdown_owned(final_state)
        finally:
            for sig, previous in previous_handlers.items():
                try:
                    signal.signal(sig, previous)
                except (OSError, ValueError) as exc:
                    cleanup_notice(
                        f"Could not restore {sig.name} after cleanup: {exc}"
                    )

    def _shutdown_owned(self, final_state: str) -> None:
        first_error: BaseException | None = None
        try:
            self.stop_slots()
        except BaseException as exc:
            first_error = exc
            print(f"Slot cleanup error: {exc}", file=sys.stderr, flush=True)
        try:
            self.stop_sam3()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            print(f"SAM3 cleanup error: {exc}", file=sys.stderr, flush=True)

        # Last-resort non-escalating drain.  This is intentionally free of
        # manifest writes: even if the normal cleanup path failed, all owned
        # children must exit before run_foreground can release GPU locks.
        self.drain_owned_before_unlock()
        try:
            self.persist(final_state)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            print(f"Final cleanup manifest write failed: {exc}", file=sys.stderr, flush=True)
        if first_error is not None:
            raise RuntimeError("parallel scheduler cleanup encountered an error") from first_error

    def drain_owned_before_unlock(self) -> None:
        live = self.live_owned([*self.slot_processes, *self.sam_processes])
        if not live:
            return
        print(
            "Final cleanup safety drain: forwarding SIGINT and retaining GPU locks "
            "until all owned processes exit.",
            file=sys.stderr,
            flush=True,
        )
        self.signal_owned(live, signal.SIGINT)
        self.wait_owned(
            live,
            grace_sec=self.config.shutdown_sigint_grace_sec,
            on_grace_expired=lambda remaining: (
                self.record_cleanup_grace_expired_noexcept(
                    "final_safety_drain", remaining
                )
            ),
        )
        self.mark_cleanup_complete("final_safety_drain")

    @staticmethod
    def live_owned(items: list[OwnedProcess]) -> list[OwnedProcess]:
        """Return known/possibly-live children; uncertainty fails closed."""
        live: list[OwnedProcess] = []
        for item in items:
            try:
                returncode = item.process.poll()
            except BaseException as exc:
                returncode = None
                print(
                    f"Could not poll owned {item.kind}[{item.slot}] "
                    f"pid={item.process.pid}: {exc}; treating it as live",
                    file=sys.stderr,
                    flush=True,
                )
            if returncode is None:
                live.append(item)
        return live


def load_manifest(batch_root: Path) -> dict[str, Any]:
    path = batch_root / MANIFEST_NAME
    if not path.is_file():
        raise ConfigError(f"Parallel manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Could not read parallel manifest {path}: {exc}") from exc
    if payload.get("schema") not in {"roboharn_evo/parallel4_isolated_batch/v1", "tcm/parallel4_isolated_batch/v1"}:
        raise ConfigError(f"Unsupported parallel manifest schema in {path}")
    return payload


def status(batch_root: Path) -> int:
    manifest = load_manifest(batch_root)
    owner = manifest.get("owner", {})
    owner_pid = owner.get("pid")
    owner_alive = (
        isinstance(owner_pid, int)
        and proc_start_ticks(owner_pid) == owner.get("start_ticks")
    )
    print(
        f"batch_root={batch_root} state={manifest.get('state')} "
        f"owner_pid={owner_pid} owner_alive={str(owner_alive).lower()}"
    )
    for slot in manifest.get("slots", []):
        process = slot.get("process", {})
        pid = process.get("pid")
        alive = isinstance(pid, int) and proc_start_ticks(pid) == process.get("start_ticks")
        current_path = Path(str(slot.get("batch_log_root", ""))) / "control/current.tsv"
        current = "unavailable"
        if current_path.is_file():
            try:
                current = current_path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
        print(
            f"slot={slot.get('slot')} state={slot.get('state')} pid={pid} "
            f"alive={str(alive).lower()} current={current}"
        )
    return 0


def skip_slot(batch_root: Path, slot_number: int) -> int:
    manifest = load_manifest(batch_root)
    slots = manifest.get("slots", [])
    selected = next((slot for slot in slots if slot.get("slot") == slot_number), None)
    if selected is None:
        raise ConfigError(f"Slot {slot_number} is not present in {batch_root / MANIFEST_NAME}")
    slot_root = Path(str(selected.get("batch_log_root", ""))).expanduser().resolve()
    expected_parent = (batch_root / "slots").resolve()
    if slot_root.parent != expected_parent:
        raise ConfigError(f"Refusing unexpected slot control path from manifest: {slot_root}")
    script_raw = manifest.get("owner", {}).get("script")
    if not script_raw:
        raise ConfigError("Manifest is missing owner script identity")
    repo_root = Path(str(manifest.get("config", {}).get("repo_root", ""))).resolve()
    sequential = repo_root / SEQUENTIAL_SCRIPT_REL
    require_path(sequential, "sequential launcher")
    completed = subprocess.run(
        [
            "/bin/bash",
            str(sequential),
            "--skip-current",
            "--batch-log-root",
            str(slot_root),
        ],
        cwd=repo_root,
        check=False,
    )
    return completed.returncode


def detach(config: Config) -> int:
    check_new_batch_root(config.batch_log_root)
    required_external_paths = {
        "RMBENCH_ASSETS_ROOT": config.assets_root,
        "ROBOHARN_EVO_SAM3_REPO": config.sam3_repo,
        "ROBOHARN_EVO_SAM3_CHECKPOINT": config.sam3_checkpoint,
        "ROBOHARN_EVO_SAM3_BPE_PATH": config.sam3_bpe_path,
        "CAUSALWAM_ROOT": config.causalwam_root,
    }
    missing_external_paths = [
        name for name, path in required_external_paths.items() if path is None
    ]
    if missing_external_paths:
        raise ConfigError(
            "The copied formal chain requires explicit read-only external paths "
            "before --detach and has no donor fallback; set: "
            + ", ".join(missing_external_paths)
        )
    if not shutil_which("tmux"):
        raise ConfigError("--detach requires tmux")
    session = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in config.tmux_session)
    if not session.strip("_"):
        raise ConfigError("BATCH_SESSION must contain an alphanumeric character")
    exists = subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if exists.returncode == 0:
        raise ConfigError(f"tmux session already exists: {session}")
    values = {
        **runtime_environment(config),
        "REPO_ROOT": str(config.repo_root),
        "ROBOHARN_EVO_PROJECT_ROOT": str(config.roboharn_project_root),
        "PARALLEL_SLOTS": str(config.slots),
        "GPU_IDS": ",".join(map(str, config.eval_gpus)),
        "SAM3_GPU_IDS": ",".join(map(str, config.sam3_gpus)),
        "SAM3_BASE_PORT": str(config.sam3_base_port),
        "TASKS_CSV": ",".join(config.tasks),
        "EVAL_START_SEEDS_CSV": ",".join(map(str, config.seeds)),
        "JOB_ASSIGNMENT_MODE": config.job_assignment_mode,
        "ALLOW_SHARED_GPU": "1" if config.allow_shared_gpu else "0",
        "ALLOW_OCCUPIED_GPU": "1" if config.allow_occupied_gpu else "0",
        "ALLOW_VERIFIED_CAUSALWAM_GPU_OCCUPANCY": (
            "1" if config.allow_verified_causalwam_gpu_occupancy else "0"
        ),
        "GPU_OCCUPANCY_ALLOWED_CAUSALWAM_PIDS_CSV": ",".join(
            map(str, config.allowed_causalwam_pids)
        ),
        "CAUSALWAM_ROOT": str(config.causalwam_root),
        "GPU_LOCK_ROOT": str(config.gpu_lock_root),
        "RMBENCH_RENDER_DEVICE": config.render_device,
        "BATCH_STAMP": config.batch_stamp,
        "RUN_LABEL": config.run_label,
        "TASK_CONFIG": config.task_config,
        "INSTRUCTION_SET": config.instruction_set,
        "POLICY_NAME": config.policy_name,
        "PERCEPTION_CONDITION": config.perception_condition,
        "AGENT_API_BASE_URL": config.agent_api_base_url,
        "MAX_OBJECTS": str(config.max_objects),
        "EVAL_STEP_LIMIT": str(config.eval_step_limit),
        "MAX_ROUNDS": str(config.max_rounds),
        "MAX_CONTROL_TURNS": str(config.max_control_turns),
        "MAX_NO_PROGRESS_CONTROL_TURNS": str(config.max_no_progress_control_turns),
        "CONTINUE_ON_ERROR": "1" if config.continue_on_error else "0",
        "SAM3_PYTHON": str(config.sam3_python),
        "ROBOHARN_EVO_SAM3_REPO": str(config.sam3_repo),
        "ROBOHARN_EVO_SAM3_CHECKPOINT": str(config.sam3_checkpoint),
        "ROBOHARN_EVO_SAM3_BPE_PATH": str(config.sam3_bpe_path),
        "SAM3_CONFIDENCE_THRESHOLD": str(config.sam3_confidence_threshold),
        "SAM3_AMP_DTYPE": config.sam3_amp_dtype,
        "SAM3_STARTUP_TIMEOUT_SEC": str(config.sam3_startup_timeout_sec),
        "SAM3_HEALTH_INTERVAL_SEC": str(config.sam3_health_interval_sec),
        "SAM3_REAL_PROBE": "1" if config.sam3_real_probe else "0",
        "SAM3_PROBE_TIMEOUT_SEC": str(config.sam3_probe_timeout_sec),
        "SLOT_START_DELAY_SEC": str(config.slot_start_delay_sec),
        "SHUTDOWN_SIGINT_GRACE_SEC": str(config.shutdown_sigint_grace_sec),
        "BATCH_SESSION": session,
        # tmux servers can retain an older PATH than the invoking shell.  Put
        # the interpreter that launched this scheduler first so child Bash
        # launchers resolve the same environment's `python` executable.
        "PATH": os.pathsep.join(
            [str(Path(sys.executable).resolve().parent), os.environ.get("PATH", os.defpath)]
        ),
    }
    external_evidence_specs = os.environ.get(
        "RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS", ""
    )
    if external_evidence_specs:
        values["RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS"] = (
            external_evidence_specs
        )
    command = [
        "/usr/bin/env",
        "-u",
        "ALL_PROXY",
        "-u",
        "all_proxy",
        "-u",
        "HTTP_PROXY",
        "-u",
        "HTTPS_PROXY",
        "-u",
        "http_proxy",
        "-u",
        "https_proxy",
        "NO_PROXY=127.0.0.1,localhost",
        "no_proxy=127.0.0.1,localhost",
        *(f"{key}={value}" for key, value in values.items()),
    ]
    command.extend(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--foreground",
            "--batch-log-root",
            str(config.batch_log_root),
        ]
    )
    completed = subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-c",
            str(config.repo_root),
            shlex.join(command),
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise ConfigError(f"tmux new-session failed with rc={completed.returncode}")
    print(f"Parallel batch started in tmux: {session}")
    print(f"Attach: tmux attach -t {session}")
    print(f"Status: {sys.executable} {SCRIPT_PATH} --status --batch-log-root {config.batch_log_root}")
    for slot in range(config.slots):
        print(
            f"Skip slot {slot}: {sys.executable} {SCRIPT_PATH} --skip-slot {slot} "
            f"--batch-log-root {config.batch_log_root}"
        )
    return 0


def shutil_which(command: str) -> str | None:
    """Tiny stdlib-only equivalent that keeps the import surface explicit."""
    for directory in os.environ.get("PATH", os.defpath).split(os.pathsep):
        candidate = Path(directory) / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def run_foreground(config: Config, jobs: list[Job]) -> int:
    check_new_batch_root(config.batch_log_root)
    preflight_paths(config)
    preflight_ports(config)
    health = preflight_agent(config)
    token = f"{config.batch_stamp}_{os.getpid()}_{time.time_ns()}"
    gpu_locks = GpuLockSet(
        config.gpu_lock_root,
        list(config.eval_gpus) + list(config.sam3_gpus),
        token,
    )
    lock_records: list[dict[str, Any]] = []
    scheduler: Scheduler | None = None
    scheduler_prepared = False
    interrupted = False

    def request_shutdown(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt(f"received signal {signum}")

    previous_term = signal.signal(signal.SIGTERM, request_shutdown)
    previous_hup = signal.signal(signal.SIGHUP, request_shutdown)
    previous_int = signal.signal(signal.SIGINT, request_shutdown)
    try:
        lock_records = gpu_locks.acquire()
        scheduler = Scheduler(config, jobs, token=token, gpu_locks=gpu_locks)
        scheduler.prepare()
        scheduler_prepared = True
        scheduler.manifest["agent_health"] = health
        scheduler.manifest["gpu_resource_locks"] = lock_records
        scheduler.persist("gpu_locks_acquired")
        # Occupancy is checked only after cross-batch locks are held, closing
        # the race in which two conforming schedulers both see idle GPUs.
        try:
            gpu_inventory = preflight_gpus(config)
            scheduler.gpu_inventory = gpu_inventory
            scheduler.manifest["gpu_inventory"] = {
                str(gpu_id): identity
                for gpu_id, identity in sorted(gpu_inventory.items())
            }
            occupancy = preflight_gpu_occupancy(config)
        except ConfigError as exc:
            scheduler.manifest["gpu_occupancy_preflight"] = {
                "checked_at": utc_now(),
                "requested_gpu_ids": sorted(
                    set(config.eval_gpus) | set(config.sam3_gpus)
                ),
                "passed": False,
                "error": str(exc),
            }
            scheduler.persist("gpu_preflight_failed")
            raise
        scheduler.manifest["gpu_occupancy_preflight"] = occupancy
        scheduler.persist("preflight_complete")
        scheduler.launch_sam3()
        scheduler.launch_slots()
        rc = scheduler.monitor()
        scheduler.stop_sam3()
        scheduler.persist("complete" if rc == 0 else "failed")
        return rc
    except KeyboardInterrupt:
        interrupted = True
        # Cleanup is deliberately non-escalating.  Ignore repeated terminal
        # signals while waiting for owned children to honor their first
        # SIGINT, rather than aborting cleanup and orphaning a slot.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        print("Parallel batch interrupted; stopping owned slots before SAM3.", file=sys.stderr)
        if scheduler is not None and scheduler_prepared:
            scheduler.shutdown("interrupted")
        return 130
    except Exception:
        if scheduler is not None and scheduler_prepared:
            scheduler.shutdown("failed")
        raise
    finally:
        # This is the final lock-release gate, independent of normal cleanup
        # and manifest persistence.  No code path may unlock a GPU while an
        # owned direct child is still live.  The sequential/8-way wrappers do
        # not exit until their eval children have handled the same SIGINT.
        if scheduler is not None and scheduler_prepared:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                try:
                    signal.signal(sig, signal.SIG_IGN)
                except (OSError, ValueError):
                    pass
            while scheduler.live_owned(
                [*scheduler.slot_processes, *scheduler.sam_processes]
            ):
                try:
                    scheduler.drain_owned_before_unlock()
                except BaseException as exc:
                    cleanup_notice(
                        "Final pre-unlock process drain failed; retrying without "
                        f"releasing GPU locks: {exc}"
                    )
                    try:
                        time.sleep(0.25)
                    except BaseException:
                        pass
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGHUP, previous_hup)
        gpu_locks.release()
        if interrupted:
            sys.stdout.flush()
            sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.status:
        return status(require_batch_root_argument(args))
    if args.skip_slot is not None:
        root = require_batch_root_argument(args)
        if args.skip_slot < 0:
            raise ConfigError("--skip-slot must be a non-negative slot number")
        return skip_slot(root, args.skip_slot)

    config = load_config(args.batch_log_root)
    jobs = build_jobs(config)
    if args.dry_run:
        print_plan(config, jobs)
        return 0
    if args.detach:
        return detach(config)
    print_plan(config, jobs)
    return run_foreground(config, jobs)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
