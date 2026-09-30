"""Small, simulator-free helpers shared by benchmark composition roots."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_summary(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return {
        "path": str(path.resolve()),
        "files": len(lines),
        "aggregate_sha256": hashlib.sha256(payload).hexdigest(),
    }


def dependency_available(module_name: str) -> bool:
    """Check a top-level dependency without importing it."""
    if "." in module_name:
        raise ValueError("dependency_available accepts top-level module names only")
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def module_was_imported(prefix: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for name in sys.modules)


def assert_path_within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} resolved outside TCM: {resolved}") from exc
    return resolved


__all__ = [
    "REPOSITORY_ROOT",
    "assert_path_within",
    "dependency_available",
    "manifest_summary",
    "module_was_imported",
    "sha256_file",
]
