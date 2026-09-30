"""Simulator-free composition helpers for the copied RMBench source tree."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import yaml

from benchmarks.rmbench.paths import (
    ASSETS_ROOT_ENV,
    OUTPUT_ROOT_ENV,
    assets_root,
    benchmark_root,
    default_deploy_config,
    output_root,
    project_root,
    resolve_task_config,
)
from benchmarks.rmbench.runtime_assets import (
    RUNTIME_CONFIG_ROOT_ENV,
    require_writable_root_outside,
    runtime_config_root,
)


CANONICAL_UPSTREAM_REPOSITORY = "https://github.com/RoboTwin-Platform/RMBench"
UPSTREAM_REPOSITORY = CANONICAL_UPSTREAM_REPOSITORY
ROBOHARN_POLICY_MODULE = "benchmarks.rmbench.policy.roboharn_evo.deploy_policy"
_LEGACY_RMBENCH_ROOT_ENV = "RMBENCH_ROOT"
_ROBOHARN_OUTPUT_ROOT_ENV = "ROBOHARN_EVO_OUTPUT_ROOT"
_ROBOHARN_WORKSPACE_ROOT_ENV = "ROBOHARN_EVO_WORKSPACE_ROOT"
_RENDER_DEVICE_PROVENANCE_PATH_ENV = (
    "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH"
)
_RENDER_DEVICE_PROVENANCE_FILENAME = "renderer_device_provenance.json"
_RUNTIME_CACHE_ENVIRONMENT_LAYOUT = {
    "XDG_CACHE_HOME": "xdg",
    "CUDA_CACHE_PATH": "cuda",
    "MESA_SHADER_CACHE_DIR": "mesa",
    "TORCH_HOME": "torch",
    "HF_HOME": "huggingface",
    "TRITON_CACHE_DIR": "triton",
    "NUMBA_CACHE_DIR": "numba",
    "MPLCONFIGDIR": "matplotlib",
    "TMPDIR": "tmp",
    "TMP": "tmp",
    "TEMP": "tmp",
}
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_tasks() -> tuple[str, ...]:
    """Discover implemented task modules without importing SAPIEN."""

    ignored = {"__init__", "_base_task", "_GLOBAL_CONFIGS", "renderer_device_contract"}
    tasks = {
        path.stem
        for path in (benchmark_root() / "envs").glob("*.py")
        if path.stem not in ignored and not path.stem.startswith("_")
    }
    return tuple(sorted(tasks))


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"RMBench config does not exist: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"RMBench config must contain a YAML mapping: {path}")
    return payload


def _deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _runtime_config_root_for_output(output_dir: Path) -> Path:
    configured = os.environ.get(RUNTIME_CONFIG_ROOT_ENV, "").strip()
    return runtime_config_root(configured or output_dir / "runtime_configs")


def build_run_config(
    *,
    config_path: str | os.PathLike[str] | None = None,
    task_name: str,
    task_config: str = "demo_clean",
    seed: int = 0,
    output: str | os.PathLike[str] | None = None,
    checkpoint_setting: str = "roboharn_evo_agent",
    instruction_type: str = "unseen",
    instruction_set: str = "rmbench_original",
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from roboharn_evo.agent.hpk.compatibility import normalize_agent_knowledge_config

    tasks = discover_tasks()
    if task_name not in tasks:
        raise ValueError(f"unknown RMBench task {task_name!r}; available: {', '.join(tasks)}")
    deploy_config = Path(config_path).expanduser().resolve() if config_path else default_deploy_config()
    task_config_path = resolve_task_config(task_config)
    if not task_config_path.is_file():
        raise FileNotFoundError(f"RMBench task config does not exist: {task_config_path}")
    config = _read_yaml_mapping(deploy_config)
    if "agent" in config:
        config["agent"] = normalize_agent_knowledge_config(config["agent"])
    config.update(
        {
            "policy_name": ROBOHARN_POLICY_MODULE,
            "task_name": task_name,
            "task_config": task_config_path.stem,
            "task_config_path": str(task_config_path),
            "ckpt_setting": str(checkpoint_setting),
            "seed": int(seed),
            "instruction_type": str(instruction_type),
            "instruction_set": str(instruction_set),
            "output_root": str(output_root(output)),
        }
    )
    if overrides:
        normalized_overrides = dict(overrides)
        if "agent" in normalized_overrides:
            normalized_overrides["agent"] = normalize_agent_knowledge_config(normalized_overrides["agent"])
        _deep_update(config, normalized_overrides)
    return config


def runtime_provenance(
    *,
    config_path: str | os.PathLike[str] | None = None,
    assets: str | os.PathLike[str] | None = None,
    output: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    import roboharn_evo

    config = Path(config_path).expanduser().resolve() if config_path else default_deploy_config()
    asset_dir = assets_root(assets)
    output_dir = output_root(output)
    configured_runtime_root = _runtime_config_root_for_output(output_dir)
    return {
        "schema_version": 1,
        "benchmark": "rmbench",
        "benchmark_module": "benchmarks.rmbench",
        "benchmark_root": str(benchmark_root()),
        "canonical_upstream_repository": CANONICAL_UPSTREAM_REPOSITORY,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "policy_module": ROBOHARN_POLICY_MODULE,
        "roboharn_evo_version": roboharn_evo.__version__,
        "roboharn_evo_module": str(Path(roboharn_evo.__file__).resolve()),
        "config": str(config),
        "config_sha256": _sha256(config) if config.is_file() else None,
        "assets_root": str(asset_dir),
        "assets_available": (asset_dir / "embodiments").is_dir() and (asset_dir / "objects").is_dir(),
        "output_root": str(output_dir),
        "runtime_config_root": str(configured_runtime_root),
        "runtime_workspace_root": str(output_dir / "runtime_workspace"),
        "runtime_cache_root": str(output_dir / "runtime_caches"),
    }


@contextmanager
def _runtime_process_boundary(
    *,
    asset_dir: Path,
    output_dir: Path,
    runtime_configs: Path,
    workspace_dir: Path,
    cache_dir: Path,
):
    """Pin relative writes and third-party caches to validated output storage."""

    previous_cwd = Path.cwd()
    temporary_parent = project_root() / "eval_result" / "rt"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f"r{os.getpid()}_", dir=temporary_parent))
    updates = {
        ASSETS_ROOT_ENV: str(asset_dir),
        OUTPUT_ROOT_ENV: str(output_dir),
        RUNTIME_CONFIG_ROOT_ENV: str(runtime_configs),
        _ROBOHARN_OUTPUT_ROOT_ENV: str(output_dir),
        _ROBOHARN_WORKSPACE_ROOT_ENV: str(workspace_dir),
        _RENDER_DEVICE_PROVENANCE_PATH_ENV: str(
            output_dir / _RENDER_DEVICE_PROVENANCE_FILENAME
        ),
        **{
            variable: str(cache_dir / relative_path)
            for variable, relative_path in _RUNTIME_CACHE_ENVIRONMENT_LAYOUT.items()
        },
    }
    # NVRTC 对临时目录长度有限制，每个运行使用项目内的独立短路径。
    updates.update({name: str(temporary_dir) for name in ("TMPDIR", "TMP", "TEMP")})
    previous_env = {name: os.environ.get(name) for name in updates}
    previous_dont_write_bytecode = sys.dont_write_bytecode
    workspace_dir.mkdir(parents=True, exist_ok=True)
    for path in dict.fromkeys(updates[name] for name in _RUNTIME_CACHE_ENVIRONMENT_LAYOUT):
        Path(path).mkdir(parents=True, exist_ok=True)
    try:
        sys.dont_write_bytecode = True
        os.environ.update(updates)
        os.chdir(workspace_dir)
        yield
    finally:
        os.chdir(previous_cwd)
        sys.dont_write_bytecode = previous_dont_write_bytecode
        for name, previous in previous_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def dry_run_report(
    *,
    config_path: str | os.PathLike[str] | None = None,
    task_name: str | None = None,
    task_config: str = "demo_clean",
    seed: int = 0,
    assets: str | os.PathLike[str] | None = None,
    output: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    config = Path(config_path).expanduser().resolve() if config_path else default_deploy_config()
    # Validate YAML without importing the simulator, renderer, or policy stack.
    _read_yaml_mapping(config)
    task_config_path = resolve_task_config(task_config)
    if not task_config_path.is_file():
        raise FileNotFoundError(f"RMBench task config does not exist: {task_config_path}")
    tasks = discover_tasks()
    if task_name is not None and task_name not in tasks:
        raise ValueError(f"unknown RMBench task {task_name!r}; available: {', '.join(tasks)}")
    report = runtime_provenance(config_path=config, assets=assets, output=output)
    report.update(
        {
            "mode": "dry-run",
            "simulator_imported": False,
            "rollout_started": False,
            "task": task_name,
            "tasks": list(tasks),
            "task_config": str(task_config_path),
            "seed": int(seed),
        }
    )
    return report


def load_evaluator() -> ModuleType:
    """Import the copied evaluator only when an actual run is requested."""

    return importlib.import_module("benchmarks.rmbench.script.eval_policy")


def run_evaluation(
    config: Mapping[str, Any],
    *,
    assets: str | os.PathLike[str] | None = None,
    output: str | os.PathLike[str] | None = None,
) -> Any:
    asset_dir = assets_root(assets)
    missing = [name for name in ("embodiments", "objects") if not (asset_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            "RMBench simulator assets are unavailable under "
            f"{asset_dir}; expected: {', '.join(missing)}. "
            f"Set {ASSETS_ROOT_ENV} or pass --assets-root to a legally obtained asset directory."
        )
    output_dir = output_root(output or config.get("output_root"))
    runtime_configs = _runtime_config_root_for_output(output_dir)
    project_dir = project_root().resolve()
    runtime_base = (project_dir / "eval_result").resolve()
    workspace_dir = (output_dir / "runtime_workspace").resolve()
    cache_dir = (output_dir / "runtime_caches").resolve()
    for label, writable in (
        ("RMBench output root", output_dir),
        ("RMBench runtime config root", runtime_configs),
        ("RoboHarn-Evo runtime workspace", workspace_dir),
        ("RoboHarn-Evo runtime cache", cache_dir),
    ):
        resolved = Path(writable).expanduser().resolve()
        if resolved != runtime_base and runtime_base not in resolved.parents:
            raise ValueError(
                f"{label} must be inside the dedicated RoboHarn-Evo eval_result tree: "
                f"{resolved} vs {runtime_base}"
            )
    readonly_roots: list[str | os.PathLike[str]] = [asset_dir]
    configured_assets_root = os.environ.get(ASSETS_ROOT_ENV, "").strip()
    if configured_assets_root:
        readonly_roots.append(configured_assets_root)
    legacy_root = os.environ.get(_LEGACY_RMBENCH_ROOT_ENV, "").strip()
    if legacy_root:
        readonly_roots.append(legacy_root)
    for label, writable in (
        ("RMBench output root", output_dir),
        ("RMBench runtime config root", runtime_configs),
        ("RoboHarn-Evo runtime workspace", workspace_dir),
        ("RoboHarn-Evo runtime cache", cache_dir),
    ):
        require_writable_root_outside(
            writable,
            readonly_roots=tuple(readonly_roots),
            label=label,
        )

    evaluator_config = dict(config)
    evaluator_config["output_root"] = str(output_dir)

    with _runtime_process_boundary(
        asset_dir=asset_dir,
        output_dir=output_dir,
        runtime_configs=runtime_configs,
        workspace_dir=workspace_dir,
        cache_dir=cache_dir,
    ):
        evaluator = load_evaluator()
        render_preflight = importlib.import_module(
            "benchmarks.rmbench.script.test_render"
        ).Sapien_TEST
        render_preflight()
        evaluator._install_termination_signal_handlers()
        return evaluator.main(evaluator_config)


def provenance_json(**kwargs: Any) -> str:
    return json.dumps(runtime_provenance(**kwargs), ensure_ascii=False, indent=2, sort_keys=True)


__all__ = [
    "CANONICAL_UPSTREAM_REPOSITORY",
    "ROBOHARN_POLICY_MODULE",
    "UPSTREAM_REPOSITORY",
    "build_run_config",
    "discover_tasks",
    "dry_run_report",
    "load_evaluator",
    "provenance_json",
    "run_evaluation",
    "runtime_provenance",
]
