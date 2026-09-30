from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..utils.text import VocabularySet


class RoboHarnTrainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: dict[str, Any],
        vocab: VocabularySet,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.vocab = vocab
        self.device = device

        trainer_cfg = config["trainer"]
        self.epochs = int(trainer_cfg["epochs"])
        self.log_interval = int(trainer_cfg["log_interval"])
        self.grad_clip_norm = float(trainer_cfg["grad_clip_norm"])
        self.output_dir = Path(str(trainer_cfg["output_dir"]))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        optimizer_cfg = config["optimizer"]
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_cfg["lr"]),
            weight_decay=float(optimizer_cfg["weight_decay"]),
        )
        self.best_val_loss = float("inf")

    def _move_batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        moved: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device)
        return moved

    def _run_epoch(self, *, loader: DataLoader, training: bool) -> dict[str, float]:
        running: dict[str, float] = {
            "total_loss": 0.0,
            "planner_loss": 0.0,
            "commit_loss": 0.0,
            "write_loss": 0.0,
            "subtask_loss": 0.0,
            "exec_loss": 0.0,
        }
        steps = 0
        if training:
            self.model.train()
        else:
            self.model.eval()

        for step_index, batch in enumerate(loader, start=1):
            batch = self._move_batch_to_device(batch)
            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                outputs = self.model(batch)
                loss_dict = self.model.compute_losses(batch=batch, outputs=outputs)
                total_loss = loss_dict["total_loss"]

            if training:
                total_loss.backward()
                if self.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()

            steps += 1
            for key in running.keys():
                running[key] += float(loss_dict[key].detach().cpu().item())

            if training and step_index % self.log_interval == 0:
                print(
                    f"[train] step={step_index:04d} "
                    f"total={loss_dict['total_loss'].item():.4f} "
                    f"commit={loss_dict['commit_loss'].item():.4f} "
                    f"write={loss_dict['write_loss'].item():.4f} "
                    f"subtask={loss_dict['subtask_loss'].item():.4f} "
                    f"exec={loss_dict['exec_loss'].item():.4f}"
                )

        if steps == 0:
            return running
        for key in running.keys():
            running[key] /= float(steps)
        return running

    def _save_checkpoint(self, *, epoch: int, val_metrics: dict[str, float]) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "config": self.config,
            "vocab": self.vocab.to_dict(),
            "val_metrics": val_metrics,
        }
        checkpoint_path = self.output_dir / "best.pt"
        torch.save(checkpoint, checkpoint_path)

    def run(self) -> None:
        for epoch in range(1, self.epochs + 1):
            train_metrics = self._run_epoch(loader=self.train_loader, training=True)
            print(
                f"[epoch {epoch:03d}] train "
                f"total={train_metrics['total_loss']:.4f} "
                f"commit={train_metrics['commit_loss']:.4f} "
                f"write={train_metrics['write_loss']:.4f} "
                f"subtask={train_metrics['subtask_loss']:.4f} "
                f"exec={train_metrics['exec_loss']:.4f}"
            )

            val_metrics: dict[str, float] = {"total_loss": float("nan")}
            if self.val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_epoch(loader=self.val_loader, training=False)
                print(
                    f"[epoch {epoch:03d}] val "
                    f"total={val_metrics['total_loss']:.4f} "
                    f"commit={val_metrics['commit_loss']:.4f} "
                    f"write={val_metrics['write_loss']:.4f} "
                    f"subtask={val_metrics['subtask_loss']:.4f} "
                    f"exec={val_metrics['exec_loss']:.4f}"
                )

                if val_metrics["total_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["total_loss"]
                    self._save_checkpoint(epoch=epoch, val_metrics=val_metrics)
            else:
                self._save_checkpoint(epoch=epoch, val_metrics=val_metrics)
