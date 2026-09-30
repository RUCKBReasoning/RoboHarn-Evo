from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _configured_path(env_name: str, fallback: Path) -> Path:
    value = os.getenv(env_name, "").strip()
    return Path(value).expanduser().resolve() if value else fallback.resolve()


def pi05_repository_dir() -> Path:
    return _configured_path("ROBOHARN_EVO_PI05_REPO", project_root() / "external" / "pi05")


def pi05_checkpoint_root() -> Path:
    return _configured_path("ROBOHARN_EVO_PI05_CHECKPOINT_ROOT", project_root() / "checkpoints" / "pi05")


def pi05_assets_root() -> Path:
    return _configured_path("ROBOHARN_EVO_PI05_ASSETS_ROOT", project_root() / "assets" / "pi05")


def pi05_base_params_dir() -> Path:
    return _configured_path(
        "ROBOHARN_EVO_PI05_BASE_PARAMS",
        project_root() / "checkpoints" / "pi05_base" / "params",
    )


def openpi_repository_dir() -> Path:
    return _configured_path("ROBOHARN_EVO_OPENPI_ROOT", pi05_repository_dir())


def robobrain_inference_path() -> Path:
    return _configured_path(
        "ROBOHARN_EVO_ROBOBRAIN_INFERENCE",
        project_root() / "external" / "RoboBrain2.5" / "inference.py",
    )


def agentic_model_path() -> str:
    configured = os.getenv("ROBOHARN_EVO_AGENTIC_MODEL", "").strip()
    if configured:
        return configured
    return str(project_root() / "checkpoints" / "qwen_vl" / "Qwen3-VL-8B-Thinking")


__all__ = [
    "agentic_model_path",
    "openpi_repository_dir",
    "pi05_assets_root",
    "pi05_base_params_dir",
    "pi05_checkpoint_root",
    "pi05_repository_dir",
    "project_root",
    "robobrain_inference_path",
]
