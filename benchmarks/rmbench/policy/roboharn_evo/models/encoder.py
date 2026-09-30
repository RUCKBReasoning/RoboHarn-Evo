from __future__ import annotations

import math

import torch
from torch import nn


class SequenceTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_id: int) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=self.pad_id)
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, L)
        embedded = self.embedding(token_ids)
        # embedded: (B, L, D_embed)
        _, hidden = self.rnn(embedded)
        # hidden: (1, B, D_hidden)
        return hidden.squeeze(0)


class SmallImageEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, C, H, W)
        features = self.backbone(images)
        # features: (B, D_hidden, 1, 1)
        return features.flatten(start_dim=1)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, D_in)
        return self.network(inputs)


class ScalarTimeEmbedding(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, time_values: torch.Tensor) -> torch.Tensor:
        # time_values: (B, T, 1)
        batch_size, horizon, _ = time_values.shape
        flattened = time_values.reshape(batch_size * horizon, 1)
        embedded = self.projection(flattened)
        # embedded: (B*T, D_time)
        return embedded.reshape(batch_size, horizon, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / float(model_dim))
        )
        positional = torch.zeros(max_len, model_dim, dtype=torch.float32)
        positional[:, 0::2] = torch.sin(positions * div_term)
        positional[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("positional", positional.unsqueeze(0), persistent=False)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: (B, T, D_model)
        sequence_len = sequence.shape[1]
        positional = self.positional[:, :sequence_len, :]
        return sequence + positional

