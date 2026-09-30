from __future__ import annotations

import os
from pathlib import Path


def rmbench_root() -> Path:
    configured = os.getenv("RMBENCH_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def rmbench_base_dir() -> Path:
    configured = os.getenv("RMBENCH_BASE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return rmbench_root().parent


def roboharn_skills_dir() -> Path:
    return rmbench_root() / "policy" / "roboharn_evo" / "skills"


def eval_result_dir(*parts: str) -> Path:
    return rmbench_root().joinpath("eval_result", *parts)


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
