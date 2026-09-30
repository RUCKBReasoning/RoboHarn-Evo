from __future__ import annotations

import torch
from torch import nn

from .encoder import MLPEncoder, SequenceTextEncoder, SmallImageEncoder


class PlannerModule(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        commit_vocab_size: int,
        subtask_vocab_size: int,
        image_hidden_dim: int,
        text_hidden_dim: int,
        state_input_dim: int,
        model_dim: int,
    ) -> None:
        super().__init__()
        self.task_encoder = SequenceTextEncoder(vocab_size=vocab_size, embed_dim=text_hidden_dim, hidden_dim=text_hidden_dim, pad_id=pad_id)
        self.memory_encoder = SequenceTextEncoder(vocab_size=vocab_size, embed_dim=text_hidden_dim, hidden_dim=text_hidden_dim, pad_id=pad_id)
        self.candidate_memory_encoder = SequenceTextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_hidden_dim,
            hidden_dim=text_hidden_dim,
            pad_id=pad_id,
        )
        self.image_encoder = SmallImageEncoder(in_channels=3, hidden_dim=image_hidden_dim)
        self.state_encoder = MLPEncoder(input_dim=state_input_dim, hidden_dim=model_dim, output_dim=model_dim)
        planner_input_dim = (2 * text_hidden_dim) + (2 * image_hidden_dim) + model_dim
        self.context_fuser = nn.Sequential(
            nn.Linear(planner_input_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
        )
        self.commit_head = nn.Linear(model_dim, commit_vocab_size)
        self.memory_query = nn.Linear(model_dim, text_hidden_dim)
        self.subtask_head = nn.Sequential(
            nn.Linear(model_dim + text_hidden_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, subtask_vocab_size),
        )

    def encode_context(
        self,
        *,
        task_tokens: torch.Tensor,
        prev_memory_tokens: torch.Tensor,
        evidence_start_image: torch.Tensor,
        evidence_end_image: torch.Tensor,
        evidence_state: torch.Tensor,
    ) -> torch.Tensor:
        # task_tokens: (B, L_task)
        # prev_memory_tokens: (B, L_mem)
        # evidence_start_image: (B, 3, H, W)
        # evidence_end_image: (B, 3, H, W)
        # evidence_state: (B, D_e)
        task_feature = self.task_encoder(task_tokens)
        prev_memory_feature = self.memory_encoder(prev_memory_tokens)
        start_image_feature = self.image_encoder(evidence_start_image)
        end_image_feature = self.image_encoder(evidence_end_image)
        evidence_state_feature = self.state_encoder(evidence_state)
        fused = torch.cat(
            [
                task_feature,
                prev_memory_feature,
                start_image_feature,
                end_image_feature,
                evidence_state_feature,
            ],
            dim=-1,
        )
        # fused: (B, D_fused)
        return self.context_fuser(fused)

    def score_memory_candidates(
        self,
        *,
        context_feature: torch.Tensor,
        memory_candidate_tokens: torch.Tensor,
        memory_candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # context_feature: (B, D_model)
        # memory_candidate_tokens: (B, K, L_mem)
        # memory_candidate_mask: (B, K)
        batch_size, candidate_count, seq_len = memory_candidate_tokens.shape
        flattened_tokens = memory_candidate_tokens.reshape(batch_size * candidate_count, seq_len)
        candidate_features = self.candidate_memory_encoder(flattened_tokens)
        # candidate_features: (B*K, D_text)
        candidate_features = candidate_features.reshape(batch_size, candidate_count, -1)
        query = self.memory_query(context_feature).unsqueeze(-1)
        # query: (B, D_text, 1)
        memory_scores = torch.bmm(candidate_features, query).squeeze(-1)
        # memory_scores: (B, K)
        memory_scores = memory_scores.masked_fill(~memory_candidate_mask.bool(), float("-inf"))
        return memory_scores, candidate_features

    def predict_subtask(
        self,
        *,
        context_feature: torch.Tensor,
        updated_memory_feature: torch.Tensor,
    ) -> torch.Tensor:
        # context_feature: (B, D_model)
        # updated_memory_feature: (B, D_text)
        fused = torch.cat([context_feature, updated_memory_feature], dim=-1)
        return self.subtask_head(fused)

    def forward(
        self,
        *,
        task_tokens: torch.Tensor,
        prev_memory_tokens: torch.Tensor,
        evidence_start_image: torch.Tensor,
        evidence_end_image: torch.Tensor,
        evidence_state: torch.Tensor,
        memory_candidate_tokens: torch.Tensor,
        memory_candidate_mask: torch.Tensor,
        updated_memory_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context_feature = self.encode_context(
            task_tokens=task_tokens,
            prev_memory_tokens=prev_memory_tokens,
            evidence_start_image=evidence_start_image,
            evidence_end_image=evidence_end_image,
            evidence_state=evidence_state,
        )
        commit_logits = self.commit_head(context_feature)
        memory_scores, candidate_features = self.score_memory_candidates(
            context_feature=context_feature,
            memory_candidate_tokens=memory_candidate_tokens,
            memory_candidate_mask=memory_candidate_mask,
        )

        if updated_memory_tokens is not None:
            updated_memory_feature = self.memory_encoder(updated_memory_tokens)
        else:
            memory_choice = torch.argmax(memory_scores, dim=-1)
            choice_index = memory_choice.view(memory_choice.shape[0], 1, 1).expand(
                memory_choice.shape[0],
                1,
                candidate_features.shape[-1],
            )
            updated_memory_feature = torch.gather(candidate_features, dim=1, index=choice_index).squeeze(1)

        subtask_logits = self.predict_subtask(
            context_feature=context_feature,
            updated_memory_feature=updated_memory_feature,
        )
        return {
            "planner_context": context_feature,
            "commit_logits": commit_logits,
            "memory_scores": memory_scores,
            "subtask_logits": subtask_logits,
            "candidate_features": candidate_features,
            "updated_memory_feature": updated_memory_feature,
        }
