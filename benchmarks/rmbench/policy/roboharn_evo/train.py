from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from policy.roboharn_evo.datasets.dataset import RMBenchRoboHarnDataset, split_rows_by_episode
from policy.roboharn_evo.models.policy import build_policy_from_config
from policy.roboharn_evo.trainers.trainer import RoboHarnTrainer
from policy.roboharn_evo.utils.config import apply_cli_overrides, load_config, set_seed
from policy.roboharn_evo.utils.io import load_jsonl
from policy.roboharn_evo.utils.text import build_vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RoboHarn-Evo hierarchical planner-executor policy.")
    parser.add_argument("--config", type=str, default="policy/roboharn_evo/configs/default.yaml")
    parser.add_argument("--rmbench-root", type=str, default=None)
    parser.add_argument("--annotation-paths", type=str, nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    set_seed(int(config["seed"]))
    planner_backend = str(config.get("planner", {}).get("backend", "lightweight"))
    if planner_backend != "lightweight":
        raise NotImplementedError(
            "External planner backends require their own training programs. "
            "Use the RoboBrain deployment/service for the RoboBrain planner VLM."
        )

    dataset_cfg = config["dataset"]
    trainer_cfg = config["trainer"]
    executor_cfg = config.get("executor", {"backend": "lightweight"})
    rows = load_jsonl(list(dataset_cfg["annotation_paths"]))
    train_rows, val_rows = split_rows_by_episode(
        rows=rows,
        train_ratio=float(dataset_cfg["train_ratio"]),
        seed=int(config["seed"]),
    )
    if not val_rows:
        val_rows = []

    vocab = build_vocabulary(train_rows if train_rows else rows, config=config["text"])
    train_dataset = RMBenchRoboHarnDataset(
        rows=train_rows,
        rmbench_root=str(dataset_cfg["rmbench_root"]),
        task_config=str(dataset_cfg["task_config"]),
        vocab=vocab,
        image_size=int(dataset_cfg["image_size"]),
        planner_camera_key=str(dataset_cfg["planner_camera_key"]),
        executor_camera_keys=list(dataset_cfg["executor_camera_keys"]),
        state_dim=int(dataset_cfg["state_dim"]),
        action_dim=int(dataset_cfg["action_dim"]),
        action_horizon=int(dataset_cfg["action_horizon"]),
        action_stride=int(dataset_cfg["action_stride"]),
        evidence_state_dim=int(dataset_cfg["evidence_state_dim"]),
        max_memory_candidates=int(dataset_cfg["max_memory_candidates"]),
        include_hard_negatives=bool(dataset_cfg["include_hard_negatives"]),
    )

    val_loader = None
    if val_rows:
        val_dataset = RMBenchRoboHarnDataset(
            rows=val_rows,
            rmbench_root=str(dataset_cfg["rmbench_root"]),
            task_config=str(dataset_cfg["task_config"]),
            vocab=vocab,
            image_size=int(dataset_cfg["image_size"]),
            planner_camera_key=str(dataset_cfg["planner_camera_key"]),
            executor_camera_keys=list(dataset_cfg["executor_camera_keys"]),
            state_dim=int(dataset_cfg["state_dim"]),
            action_dim=int(dataset_cfg["action_dim"]),
            action_horizon=int(dataset_cfg["action_horizon"]),
            action_stride=int(dataset_cfg["action_stride"]),
            evidence_state_dim=int(dataset_cfg["evidence_state_dim"]),
            max_memory_candidates=int(dataset_cfg["max_memory_candidates"]),
            include_hard_negatives=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(trainer_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(trainer_cfg["num_workers"]),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(trainer_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(trainer_cfg["num_workers"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and bool(trainer_cfg["use_cuda"]) else "cpu")
    model = build_policy_from_config(config=config, vocab=vocab).to(device)
    if str(executor_cfg.get("backend", "lightweight")) == "pi05_adapter":
        print("[tcm] executor backend is pi05_adapter; train.py will optimize planner loss only.")
    trainer = RoboHarnTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        vocab=vocab,
        device=device,
    )
    trainer.run()


if __name__ == "__main__":
    main()
