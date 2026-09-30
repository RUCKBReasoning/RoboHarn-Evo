from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .backend_factory import build_planner_backend
from .executor import ExecutorModule
from .pi05_adapter import Pi05ExecutorAdapter, Pi05ExecutorConfig
from .planner import PlannerModule
from ..utils.io import image_to_tensor
from ..utils.text import VocabularySet


class RoboHarnPolicy(nn.Module):
    def __init__(self, *, config: dict[str, Any], vocab: VocabularySet) -> None:
        super().__init__()
        self.config = config
        self.vocab = vocab
        model_cfg = config["model"]
        dataset_cfg = config["dataset"]
        planner_cfg = config["planner"]
        executor_cfg = config.get("executor", {"backend": "lightweight"})
        self.executor_backend = str(executor_cfg.get("backend", "lightweight"))
        self.planner_backend_name = str(planner_cfg.get("backend", "robobrain_adapter"))
        self.planner_runtime = build_planner_backend(planner_cfg)

        self.planner = PlannerModule(
            vocab_size=len(vocab.text_tokenizer.id_to_token),
            pad_id=vocab.text_tokenizer.pad_id,
            commit_vocab_size=len(vocab.id_to_commit),
            subtask_vocab_size=len(vocab.id_to_subtask),
            image_hidden_dim=int(model_cfg["image_hidden_dim"]),
            text_hidden_dim=int(model_cfg["text_hidden_dim"]),
            state_input_dim=int(dataset_cfg["evidence_state_dim"]),
            model_dim=int(model_cfg["planner_dim"]),
        )
        self.executor: ExecutorModule | None = None
        self.pi05_executor: Pi05ExecutorAdapter | None = None
        if self.executor_backend == "lightweight":
            self.executor = ExecutorModule(
                vocab_size=len(vocab.text_tokenizer.id_to_token),
                pad_id=vocab.text_tokenizer.pad_id,
                state_dim=int(dataset_cfg["state_dim"]),
                action_dim=int(dataset_cfg["action_dim"]),
                text_hidden_dim=int(model_cfg["text_hidden_dim"]),
                image_hidden_dim=int(model_cfg["image_hidden_dim"]),
                model_dim=int(model_cfg["executor_dim"]),
                num_layers=int(model_cfg["executor_layers"]),
                num_heads=int(model_cfg["executor_heads"]),
                max_horizon=int(dataset_cfg["action_horizon"]),
            )
        elif self.executor_backend == "pi05_adapter":
            pi05_cfg = executor_cfg["pi05"]
            self.pi05_executor = Pi05ExecutorAdapter(
                Pi05ExecutorConfig(
                    train_config_name=str(pi05_cfg["train_config_name"]),
                    model_name=str(pi05_cfg["model_name"]),
                    checkpoint_id=int(pi05_cfg["checkpoint_id"]),
                    pi0_step=int(pi05_cfg["pi0_step"]),
                    prompt_template=str(pi05_cfg["prompt_template"]),
                )
            )
        else:
            raise ValueError(f"Unsupported executor backend: {self.executor_backend}")

        memory_bank_tokens = vocab.memory_bank_token_tensor()
        subtask_bank_tokens = vocab.subtask_bank_token_tensor()
        # Potential failure case:
        # if a deployment episode requires a memory sentence never seen during training,
        # the closed-set memory bank can only choose the nearest known ledger entry.
        self.register_buffer("memory_bank_tokens", memory_bank_tokens, persistent=True)
        self.register_buffer("subtask_bank_tokens", subtask_bank_tokens, persistent=True)
        self.register_buffer(
            "memory_bank_mask",
            torch.ones(memory_bank_tokens.shape[0], dtype=torch.bool),
            persistent=True,
        )
        self.no_update_id = int(vocab.no_update_id)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        planner_outputs = self.planner(
            task_tokens=batch["task_tokens"],
            prev_memory_tokens=batch["prev_memory_tokens"],
            evidence_start_image=batch["evidence_start_image"],
            evidence_end_image=batch["evidence_end_image"],
            evidence_state=batch["evidence_state"],
            memory_candidate_tokens=batch["memory_candidate_tokens"],
            memory_candidate_mask=batch["memory_candidate_mask"],
            updated_memory_tokens=batch["committed_memory_tokens"],
        )
        executor_outputs: dict[str, torch.Tensor] = {}
        if self.executor is not None:
            executor_outputs = self.executor(
                obs_images=batch["obs_images"],
                state_seq=batch["state_seq"],
                time_seq=batch["time_seq"],
                task_tokens=batch["task_tokens"],
                subtask_tokens=batch["subtask_tokens"],
                memory_tokens=batch["committed_memory_tokens"],
                sequence_mask=batch["sequence_mask"],
            )
        return {**planner_outputs, **executor_outputs}

    @torch.no_grad()
    def predict_planner_step(self, obs: dict[str, Any], lang: str) -> dict[str, Any]:
        prev_memory_text = str(obs.get("prev_memory_text", ""))
        prediction = self.planner_runtime.predict_planner_step(
            task=lang,
            previous_memory_text=prev_memory_text,
            planner_start_image=np.asarray(obs["planner_start_image"], dtype=np.uint8),
            planner_end_image=np.asarray(obs["planner_end_image"], dtype=np.uint8),
            planner_state=np.asarray(obs["planner_state"], dtype=np.float32),
        )
        commit_label = str(prediction["commit_label"])
        commit_id = self.vocab.commit_to_id.get(commit_label, self.no_update_id)
        return {
            "commit_id": int(commit_id),
            "commit_label": commit_label,
            "memory_id": -1,
            "memory_text": str(prediction["memory_text"]),
            "subtask_id": -1,
            "subtask_text": str(prediction["subtask_text"]),
            "raw_output": prediction.get("raw_output"),
        }

    def compute_losses(
        self,
        *,
        batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        loss_cfg = self.config["loss"]
        commit_loss = F.cross_entropy(outputs["commit_logits"], batch["commit_label"])

        write_mask = batch["apply_write_loss"].bool()
        if write_mask.any():
            write_scores = outputs["memory_scores"][write_mask]
            write_targets = batch["memory_target_index"][write_mask]
            write_loss = F.cross_entropy(write_scores, write_targets)
        else:
            write_loss = commit_loss.new_zeros(())

        subtask_mask = batch["apply_subtask_loss"].bool()
        if subtask_mask.any():
            subtask_logits = outputs["subtask_logits"][subtask_mask]
            subtask_targets = batch["subtask_label"][subtask_mask]
            subtask_loss = F.cross_entropy(subtask_logits, subtask_targets)
        else:
            subtask_loss = commit_loss.new_zeros(())

        if "predicted_action" in outputs:
            action_error = F.smooth_l1_loss(
                outputs["predicted_action"],
                batch["action_seq"],
                reduction="none",
            )
            # sequence_mask: (B, T) -> (B, T, D_a)
            expanded_mask = batch["sequence_mask"].unsqueeze(-1).expand(
                batch["sequence_mask"].shape[0],
                batch["sequence_mask"].shape[1],
                batch["action_seq"].shape[-1],
            )
            action_mask = expanded_mask.to(dtype=action_error.dtype)
            masked_action_error = action_error * action_mask
            normalizer = action_mask.sum().clamp_min(1.0)
            exec_loss = masked_action_error.sum() / normalizer
        else:
            # pi05 使用独立的 openpi 配置进行训练。
            exec_loss = commit_loss.new_zeros(())

        planner_loss = (
            commit_loss
            + (float(loss_cfg["lambda_write"]) * write_loss)
            + (float(loss_cfg["lambda_subtask"]) * subtask_loss)
        )
        total_loss = planner_loss + (float(loss_cfg["lambda_exec"]) * exec_loss)
        return {
            "total_loss": total_loss,
            "planner_loss": planner_loss,
            "commit_loss": commit_loss,
            "write_loss": write_loss,
            "subtask_loss": subtask_loss,
            "exec_loss": exec_loss,
        }

    @torch.no_grad()
    def infer_action_sequence(self, obs: dict[str, Any], lang: str) -> np.ndarray:
        planner_prediction = self.predict_planner_step(obs=obs, lang=lang)

        if self.pi05_executor is not None:
            return self.pi05_executor.predict_action_chunk(
                observation=obs,
                task=lang,
                subtask=str(planner_prediction["subtask_text"]),
                memory=str(planner_prediction["memory_text"]),
            )

        device = next(self.parameters()).device
        image_size = int(self.config["dataset"]["image_size"])
        state_dim = int(self.config["dataset"]["state_dim"])
        executor_images = self._coerce_sequence_images(obs["executor_images"], image_size=image_size, device=device)
        state_seq = torch.tensor(obs["executor_state"], dtype=torch.float32, device=device)
        if state_seq.ndim == 1:
            state_seq = state_seq.unsqueeze(0)
        state_seq = state_seq[:, :state_dim]
        state_seq = state_seq.unsqueeze(0)
        task_tokens = torch.tensor(
            [self.vocab.encode_task(lang)],
            dtype=torch.long,
            device=device,
        )
        committed_memory_tokens = torch.tensor(
            [self.vocab.encode_memory(str(planner_prediction["memory_text"]))],
            dtype=torch.long,
            device=device,
        )
        predicted_subtask_tokens = torch.tensor(
            [self.vocab.encode_subtask(str(planner_prediction["subtask_text"]))],
            dtype=torch.long,
            device=device,
        )

        horizon = state_seq.shape[1]
        if "time" in obs:
            time_seq = torch.tensor(obs["time"], dtype=torch.float32, device=device)
            if time_seq.ndim == 1:
                time_seq = time_seq.unsqueeze(-1)
        else:
            time_values = torch.linspace(0.0, 1.0, steps=horizon, device=device)
            time_seq = time_values.unsqueeze(-1)
        time_seq = time_seq.unsqueeze(0)

        sequence_mask = torch.ones((1, horizon), dtype=torch.bool, device=device)
        executor_outputs = self.executor(
            obs_images=executor_images,
            state_seq=state_seq,
            time_seq=time_seq,
            task_tokens=task_tokens,
            subtask_tokens=predicted_subtask_tokens,
            memory_tokens=committed_memory_tokens,
            sequence_mask=sequence_mask,
        )
        action_sequence = executor_outputs["predicted_action"][0].detach().cpu().numpy()
        return action_sequence

    def _coerce_single_image(self, image: Any, *, image_size: int, device: torch.device) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            return tensor.to(device=device)
        image_tensor = image_to_tensor(np.asarray(image, dtype=np.uint8), image_size=image_size)
        return image_tensor.unsqueeze(0).to(device=device)

    def _coerce_sequence_images(self, images: Any, *, image_size: int, device: torch.device) -> torch.Tensor:
        array = np.asarray(images, dtype=np.uint8)
        if array.ndim == 3:
            array = array[None, None, ...]
        elif array.ndim == 4:
            array = array[:, None, ...]
        assert array.ndim == 5, f"executor_images must have shape (T, V, H, W, C) or compatible, got {array.shape}"

        horizon, num_views = array.shape[0], array.shape[1]
        tensors = torch.zeros((1, horizon, num_views, 3, image_size, image_size), dtype=torch.float32, device=device)
        for step_index in range(horizon):
            for view_index in range(num_views):
                tensors[0, step_index, view_index] = image_to_tensor(array[step_index, view_index], image_size=image_size)
        return tensors


def build_policy_from_config(config: dict[str, Any], vocab: VocabularySet) -> RoboHarnPolicy:
    return RoboHarnPolicy(config=config, vocab=vocab)


def inference(model: RoboHarnPolicy, obs: dict[str, Any], lang: str) -> np.ndarray:
    """
    Returns an action sequence.

    Expected obs keys:
    - planner_start_image: HxWx3 uint8
    - planner_end_image: HxWx3 uint8
    - planner_state: (D_e,)
    - executor_images: (T, V, H, W, 3) or compatible
    - executor_state: (T, D_s)
    - prev_memory_text: optional str
    """
    model.eval()
    return model.infer_action_sequence(obs=obs, lang=lang)


def load_policy_from_checkpoint(checkpoint_path: str, device: torch.device | str = "cpu") -> RoboHarnPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = VocabularySet.from_dict(checkpoint["vocab"])
    model = build_policy_from_config(config=checkpoint["config"], vocab=vocab)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model
