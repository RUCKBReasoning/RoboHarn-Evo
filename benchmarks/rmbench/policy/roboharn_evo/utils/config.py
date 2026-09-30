from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from .io import load_yaml


def load_config(path: str) -> dict[str, Any]:
    return load_yaml(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_cli_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    dataset_cfg = config.setdefault("dataset", {})
    trainer_cfg = config.setdefault("trainer", {})

    if getattr(args, "rmbench_root", None):
        dataset_cfg["rmbench_root"] = args.rmbench_root
    if getattr(args, "annotation_paths", None):
        dataset_cfg["annotation_paths"] = list(args.annotation_paths)
    if getattr(args, "output_dir", None):
        trainer_cfg["output_dir"] = args.output_dir
    if getattr(args, "epochs", None) is not None:
        trainer_cfg["epochs"] = int(args.epochs)
    if getattr(args, "batch_size", None) is not None:
        trainer_cfg["batch_size"] = int(args.batch_size)
    return config
