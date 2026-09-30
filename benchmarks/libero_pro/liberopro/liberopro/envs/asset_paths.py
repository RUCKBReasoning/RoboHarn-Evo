"""Resolve non-wheel LIBERO-PRO assets without a package symlink."""

from __future__ import annotations

import os
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ASSETS_ROOT = _PACKAGE_ROOT / "assets"
_DEFAULT_CUSTOM_ASSETS_ROOT = _PACKAGE_ROOT.parent.parent / "notebooks" / "custom_assets"


def assets_root() -> Path:
    configured = os.environ.get("LIBERO_PRO_ASSETS_ROOT", "").strip()
    return Path(configured).expanduser().resolve() if configured else _DEFAULT_ASSETS_ROOT


def asset_path(*parts: str) -> str:
    return str(assets_root().joinpath(*parts))


def custom_asset_path(*parts: str) -> str:
    configured = os.environ.get("LIBERO_PRO_CUSTOM_ASSETS_ROOT", "").strip()
    root = Path(configured).expanduser().resolve() if configured else _DEFAULT_CUSTOM_ASSETS_ROOT
    return str(root.joinpath(*parts))


__all__ = ["asset_path", "assets_root", "custom_asset_path"]
