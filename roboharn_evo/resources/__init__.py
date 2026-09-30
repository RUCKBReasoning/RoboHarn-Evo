from __future__ import annotations

from pathlib import Path


def resource_root() -> Path:
    """返回 RoboHarn-Evo 的安装资源目录。"""
    return Path(__file__).resolve().parent


def config_path(name: str = "default.yaml") -> Path:
    """Resolve a packaged configuration by file name."""
    path = resource_root() / "configs" / name
    if not path.is_file():
        raise FileNotFoundError(f"Packaged RoboHarn-Evo config not found: {name}")
    return path


def skills_path() -> Path:
    """返回 RoboHarn-Evo 的 Skill 资源目录。"""
    path = resource_root() / "skills"
    if not path.is_dir():
        raise FileNotFoundError("Packaged RoboHarn-Evo Skill library is missing")
    return path


__all__ = ["config_path", "resource_root", "skills_path"]
