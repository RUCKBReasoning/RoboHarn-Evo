"""Read-only asset materialization for RMBench runtime-generated files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import uuid

from benchmarks.rmbench.paths import assets_root, output_root


RUNTIME_CONFIG_ROOT_ENV = "RMBENCH_RUNTIME_CONFIG_ROOT"
_LEGACY_REPOSITORY_ROOT_ENV = "RMBENCH_ROOT"
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def runtime_config_root(
    value: str | os.PathLike[str] | None = None,
) -> Path:
    explicit = "" if value is None else os.fspath(value).strip()
    configured = explicit or os.environ.get(RUNTIME_CONFIG_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (output_root() / "runtime_configs").resolve()


def paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""

    left = first.expanduser().resolve()
    right = second.expanduser().resolve()
    return left == right or left in right.parents or right in left.parents


def require_writable_root_outside(
    writable_root: str | os.PathLike[str],
    *,
    readonly_roots: tuple[str | os.PathLike[str], ...],
    label: str,
) -> Path:
    """Reject a writable root that overlaps an asset or donor source tree."""

    resolved = Path(writable_root).expanduser().resolve()
    for readonly_root in readonly_roots:
        raw = os.fspath(readonly_root).strip()
        if not raw:
            continue
        readonly = Path(raw).expanduser().resolve()
        if paths_overlap(resolved, readonly):
            raise ValueError(
                f"{label} must not overlap read-only source/assets root: "
                f"{resolved} vs {readonly}"
            )
    return resolved


def readonly_asset_boundaries(
    asset_root: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    """Return asset plus discoverable legacy repository boundaries."""

    asset_dir = assets_root(asset_root)
    roots = [asset_dir]
    configured_legacy = os.environ.get(_LEGACY_REPOSITORY_ROOT_ENV, "").strip()
    if configured_legacy:
        roots.append(Path(configured_legacy).expanduser().resolve())
    candidate_repository = asset_dir.parent
    if (
        asset_dir.name == "assets"
        and (candidate_repository / "policy").is_dir()
        and (candidate_repository / "script").is_dir()
    ):
        roots.append(candidate_repository.resolve())
    return tuple(dict.fromkeys(root.resolve() for root in roots))


def _source_template(config_path: Path) -> Path | None:
    template = config_path.with_name(
        f"{config_path.stem}_tmp{config_path.suffix}"
    )
    if template.is_file():
        return template
    if config_path.is_file():
        return config_path
    return None


def materialized_curobo_path(
    config_path: str | os.PathLike[str],
    *,
    destination_root: str | os.PathLike[str] | None = None,
) -> Path:
    source = Path(config_path).expanduser().resolve()
    digest = hashlib.sha256(os.fsencode(source)).hexdigest()[:16]
    parent_label = _SAFE_COMPONENT_RE.sub("_", source.parent.name).strip("._-")
    if not parent_label:
        parent_label = "embodiment"
    return runtime_config_root(destination_root) / (
        f"{parent_label}-{digest}"
    ) / source.name


def materialize_curobo_config(
    config_path: str | os.PathLike[str],
    *,
    asset_root: str | os.PathLike[str] | None = None,
    destination_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Copy and expand one CuRobo YAML without modifying its asset tree."""

    source = Path(config_path).expanduser().resolve()
    template = _source_template(source)
    if template is None:
        return source

    readonly_assets = assets_root(asset_root)
    destination = materialized_curobo_path(
        source,
        destination_root=destination_root,
    )
    require_writable_root_outside(
        destination.parent,
        readonly_roots=readonly_asset_boundaries(readonly_assets),
        label="RMBench runtime config root",
    )

    replacement = os.fspath(readonly_assets)
    content = template.read_text(encoding="utf-8")
    for token in (
        "${ASSETS_PATH}/assets",
        "$ASSETS_PATH/assets",
        "${ASSETS_PATH}",
        "$ASSETS_PATH",
    ):
        content = content.replace(token, replacement)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.read_text(encoding="utf-8") == content:
        return destination

    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "RUNTIME_CONFIG_ROOT_ENV",
    "materialize_curobo_config",
    "materialized_curobo_path",
    "paths_overlap",
    "readonly_asset_boundaries",
    "require_writable_root_outside",
    "runtime_config_root",
]
