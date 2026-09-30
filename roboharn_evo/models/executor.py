from __future__ import annotations

import torch
from torch import nn

from roboharn_evo.models.encoder import (
    MLPEncoder,
    PositionalEncoding,
    ScalarTimeEmbedding,
    SequenceTextEncoder,
    SmallImageEncoder,
)


class ExecutorModule(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        state_dim: int,
        action_dim: int,
        text_hidden_dim: int,
        image_hidden_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        max_horizon: int,
    ) -> None:
        super().__init__()
        self.task_encoder = SequenceTextEncoder(vocab_size=vocab_size, embed_dim=text_hidden_dim, hidden_dim=text_hidden_dim, pad_id=pad_id)
        self.subtask_encoder = SequenceTextEncoder(vocab_size=vocab_size, embed_dim=text_hidden_dim, hidden_dim=text_hidden_dim, pad_id=pad_id)
        self.memory_encoder = SequenceTextEncoder(vocab_size=vocab_size, embed_dim=text_hidden_dim, hidden_dim=text_hidden_dim, pad_id=pad_id)
        self.image_encoder = SmallImageEncoder(in_channels=3, hidden_dim=image_hidden_dim)
        self.state_encoder = MLPEncoder(input_dim=state_dim, hidden_dim=model_dim, output_dim=model_dim)
        self.time_encoder = ScalarTimeEmbedding(output_dim=model_dim)
        self.context_projection = nn.Sequential(
            nn.Linear((3 * text_hidden_dim), model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
        )
        fusion_dim = image_hidden_dim + model_dim + model_dim + model_dim
        self.token_projection = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.position = PositionalEncoding(model_dim=model_dim, max_len=max_horizon)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.action_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, action_dim),
        )

    def forward(
        self,
        *,
        obs_images: torch.Tensor,
        state_seq: torch.Tensor,
        time_seq: torch.Tensor,
        task_tokens: torch.Tensor,
        subtask_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        sequence_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # obs_images: (B, T, V, C, H, W)
        # state_seq: (B, T, D_s)
        # time_seq: (B, T, 1)
        batch_size, horizon, num_views, channels, height, width = obs_images.shape
        flattened_images = obs_images.reshape(batch_size * horizon * num_views, channels, height, width)
        image_features = self.image_encoder(flattened_images)
        # image_features: (B*T*V, D_img)
        image_features = image_features.reshape(batch_size, horizon, num_views, -1)
        # Average over camera views explicitly. This avoids hiding camera fusion inside broadcasting.
        image_features = image_features.mean(dim=2)
        # image_features: (B, T, D_img)

        flattened_states = state_seq.reshape(batch_size * horizon, state_seq.shape[-1])
        state_features = self.state_encoder(flattened_states).reshape(batch_size, horizon, -1)
        time_features = self.time_encoder(time_seq)

        task_feature = self.task_encoder(task_tokens)
        subtask_feature = self.subtask_encoder(subtask_tokens)
        memory_feature = self.memory_encoder(memory_tokens)
        context_feature = torch.cat([task_feature, subtask_feature, memory_feature], dim=-1)
        context_feature = self.context_projection(context_feature)
        context_feature = context_feature.unsqueeze(1).expand(batch_size, horizon, context_feature.shape[-1])
        # context_feature: (B, T, D_ctx)

        fused = torch.cat([image_features, state_features, time_features, context_feature], dim=-1)
        fused = self.token_projection(fused)
        fused = self.position(fused)
        padding_mask = None
        if sequence_mask is not None:
            padding_mask = ~sequence_mask.bool()
        latent = self.temporal_encoder(fused, src_key_padding_mask=padding_mask)
        # latent z: (B, T, D_z)
        predicted_action = self.action_head(latent)
        # predicted_action a: (B, T, D_a)
        return {
            "latent": latent,
            "predicted_action": predicted_action,
        }
