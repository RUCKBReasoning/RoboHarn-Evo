"""Caller-CWD-independent paths for the copied RMBench integration."""

from __future__ import annotations

import os
from pathlib import Path


ASSETS_ROOT_ENV = "RMBENCH_ASSETS_ROOT"
OUTPUT_ROOT_ENV = "RMBENCH_OUTPUT_ROOT"
_LEGACY_RMBENCH_ROOT_ENV = "RMBENCH_ROOT"


def benchmark_root() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return benchmark_root().parents[1]


def _configured_path(value: str | os.PathLike[str] | None, env_name: str, fallback: Path) -> Path:
    explicit = "" if value is None else os.fspath(value).strip()
    configured = explicit or os.environ.get(env_name, "").strip()
    if not configured:
        return fallback.resolve()
    return Path(configured).expanduser().resolve()


def assets_root(value: str | os.PathLike[str] | None = None) -> Path:
    """Return the external RMBench asset directory.

    The directory is expected to contain ``embodiments/`` and ``objects/``.
    Downloaded assets are not redistributed with this repository.
    """

    return _configured_path(value, ASSETS_ROOT_ENV, benchmark_root() / "assets")


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def output_root(value: str | os.PathLike[str] | None = None) -> Path:
    runtime_base = (project_root() / "eval_result").resolve()
    resolved = _configured_path(
        value,
        OUTPUT_ROOT_ENV,
        runtime_base / "rmbench",
    )
    if resolved != runtime_base and runtime_base not in resolved.parents:
        raise ValueError(
            "RMBench output root must be inside the dedicated RoboHarn-Evo "
            f"eval_result tree: {resolved} vs {runtime_base}"
        )
    readonly_roots = [assets_root()]
    configured_legacy = os.environ.get(_LEGACY_RMBENCH_ROOT_ENV, "").strip()
    if configured_legacy:
        readonly_roots.append(Path(configured_legacy).expanduser().resolve())
    for readonly in readonly_roots:
        if _paths_overlap(resolved, readonly):
            raise ValueError(
                "RMBench output root must not overlap read-only source/assets "
                f"root: {resolved} vs {readonly}"
            )
    return resolved


def task_config_root() -> Path:
    return benchmark_root() / "task_config"


def description_root() -> Path:
    return benchmark_root() / "description"


def policy_root() -> Path:
    return benchmark_root() / "policy"


def default_deploy_config() -> Path:
    return policy_root() / "roboharn_evo" / "deploy_policy.yml"


def resolve_task_config(name_or_path: str | os.PathLike[str]) -> Path:
    candidate = Path(name_or_path).expanduser()
    if candidate.suffix in {".yml", ".yaml"} or candidate.parent != Path("."):
        return candidate.resolve()
    return (task_config_root() / f"{candidate.name}.yml").resolve()


def resolve_asset_reference(value: str | os.PathLike[str]) -> Path:
    """Resolve a legacy ``./assets/...`` reference under ``assets_root``."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    parts = candidate.parts
    while parts and parts[0] in {".", ""}:
        parts = parts[1:]
    if parts and parts[0] == "assets":
        parts = parts[1:]
    return assets_root().joinpath(*parts).resolve()


__all__ = [
    "ASSETS_ROOT_ENV",
    "OUTPUT_ROOT_ENV",
    "assets_root",
    "benchmark_root",
    "default_deploy_config",
    "description_root",
    "output_root",
    "policy_root",
    "project_root",
    "resolve_asset_reference",
    "resolve_task_config",
    "source_manifest",
    "task_config_root",
]
