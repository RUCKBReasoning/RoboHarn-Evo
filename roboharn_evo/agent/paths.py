from __future__ import annotations

import os
from pathlib import Path

from roboharn_evo.resources import skills_path


WORKSPACE_ROOT_ENV = "ROBOHARN_EVO_WORKSPACE_ROOT"
OUTPUT_ROOT_ENV = "ROBOHARN_EVO_OUTPUT_ROOT"
SEGMENTATION_ARTIFACT_ROOT_ENV = "ROBOHARN_EVO_SEGMENTATION_ARTIFACT_DIR"
_READONLY_ROOT_ENVS = ("RMBENCH_ROOT", "RMBENCH_ASSETS_ROOT")


def project_root() -> Path:
    """返回 RoboHarn-Evo 源代码安装的项目目录。"""
    return Path(__file__).resolve().parents[2]


def rmbench_root() -> Path:
    """返回 RoboHarn-Evo 源代码与资源目录。"""

    return project_root()


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _readonly_roots() -> tuple[Path, ...]:
    roots = []
    for env_name in _READONLY_ROOT_ENVS:
        configured = os.getenv(env_name, "").strip()
        if configured:
            roots.append(Path(configured).expanduser().resolve())
    return tuple(dict.fromkeys(roots))


def _validated_runtime_root(value: Path, *, label: str) -> Path:
    """Resolve one runtime root and prove it cannot alias source/donor data."""

    resolved = value.expanduser().resolve()
    runtime_base = (project_root() / "eval_result").resolve()
    if resolved != runtime_base and runtime_base not in resolved.parents:
        raise ValueError(
            f"{label} must be inside the dedicated RoboHarn-Evo eval_result tree: "
            f"{resolved} vs {runtime_base}"
        )
    for readonly_root in _readonly_roots():
        if _paths_overlap(resolved, readonly_root):
            raise ValueError(
                f"{label} must not overlap read-only RMBench source/assets root: "
                f"{resolved} vs {readonly_root}"
            )
    return resolved


def writable_workspace_root() -> Path:
    """Return the isolated root allowed for model/tool-created artifacts."""

    configured = os.getenv(WORKSPACE_ROOT_ENV, "").strip()
    if configured:
        candidate = Path(configured)
    else:
        candidate = project_root() / "eval_result" / "runtime_workspace"
    return _validated_runtime_root(candidate, label=WORKSPACE_ROOT_ENV)


def writable_output_root() -> Path:
    """Return the output-owned artifact root used by deployment adapters."""

    configured = os.getenv(OUTPUT_ROOT_ENV, "").strip()
    candidate = (
        Path(configured)
        if configured
        else project_root() / "eval_result"
    )
    return _validated_runtime_root(candidate, label=OUTPUT_ROOT_ENV)


def writable_segmentation_artifact_root() -> Path | None:
    """Return the scheduler-assigned per-worker segmentation artifact root."""

    configured = os.getenv(SEGMENTATION_ARTIFACT_ROOT_ENV, "").strip()
    if not configured:
        return None
    return _validated_runtime_root(
        Path(configured),
        label=SEGMENTATION_ARTIFACT_ROOT_ENV,
    )


def rmbench_base_dir() -> Path:
    configured = os.getenv("RMBENCH_BASE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return rmbench_root().parent


def roboharn_skills_dir() -> Path:
    return skills_path()


def eval_result_dir(*parts: str) -> Path:
    return writable_workspace_root().joinpath(*parts)


def writable_roots() -> tuple[Path, ...]:
    """Return the explicitly trusted roots available to agent write tools."""

    roots = [writable_workspace_root(), writable_output_root()]
    segmentation_root = writable_segmentation_artifact_root()
    if segmentation_root is not None:
        roots.append(segmentation_root)
    return tuple(dict.fromkeys(root.resolve() for root in roots))


def resolve_writable_path(
    value: str | os.PathLike[str],
    *,
    base: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a path and fail closed unless it is under a trusted write root."""

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        base_path = (
            Path(base).expanduser().resolve()
            if base is not None
            else writable_workspace_root()
        )
        candidate = base_path / candidate
    resolved = candidate.resolve()
    allowed = writable_roots()
    if not any(resolved == root or root in resolved.parents for root in allowed):
        roots_text = ", ".join(os.fspath(root) for root in allowed)
        raise ValueError(
            f"write path is outside trusted RoboHarn-Evo roots: {resolved}; "
            f"allowed roots: {roots_text}"
        )
    for readonly_root in _readonly_roots():
        if resolved == readonly_root or readonly_root in resolved.parents:
            raise ValueError(
                "write path is inside a read-only RMBench source/assets root: "
                f"{resolved} vs {readonly_root}"
            )
    return resolved


def external_path(env_name: str, *default_parts: str) -> Path:
    configured = os.getenv(env_name, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return rmbench_base_dir().joinpath(*default_parts)


def sam3_repo_dir() -> Path:
    return external_path("ROBOHARN_EVO_SAM3_REPO", "sam3")


def sam3_checkpoint_path() -> Path:
    return external_path("ROBOHARN_EVO_SAM3_CHECKPOINT", "sam3ckpt", "sam3.1_multiplex.pt")


def sam3_bpe_path() -> Path:
    return external_path("ROBOHARN_EVO_SAM3_BPE_PATH", "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")


def sam2_repo_dir() -> Path:
    return external_path("ROBOHARN_EVO_SAM2_REPO", "sam", "sam2")


def sam2_checkpoint_path() -> Path:
    return external_path("ROBOHARN_EVO_SAM2_CHECKPOINT", "sam", "sam2", "checkpoints", "sam2.1_hiera_tiny.pt")
