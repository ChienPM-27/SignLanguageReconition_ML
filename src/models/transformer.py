"""Small full-clip Transformer encoder for the shared landmark feature contract."""
from __future__ import annotations

import math

import torch
from torch import nn

from src.models.gru import TemporalAttention


class ASLTransformerClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 num_classes: int = 20, dropout: float = 0.3, pooling: str = "mean_max",
                 attention_heads: int = 4, feedforward_size: int = 256):
        super().__init__()
        if hidden_size % attention_heads or hidden_size % 2:
            raise ValueError("Transformer hidden size must be even and divisible by attention heads.")
        if pooling not in ("mean_max", "mean", "last", "attention"):
            raise ValueError(f"Unsupported pooling: {pooling}")
        self.input_size, self.hidden_size, self.num_layers = input_size, hidden_size, num_layers
        self.num_classes = num_classes
        self.pooling, self.rnn_type, self.bidirectional = pooling, "TRANSFORMER", False
        self.attention_heads, self.feedforward_size = attention_heads, feedforward_size
        self.dropout = dropout
        self.input_projection = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size))
        layer = nn.TransformerEncoderLayer(hidden_size, attention_heads, feedforward_size,
                                           dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.position_dropout = nn.Dropout(dropout)
        if pooling == "attention":
            self.attn_pool = TemporalAttention(hidden_size)
        self.fc = nn.Linear(hidden_size * (2 if pooling == "mean_max" else 1), num_classes)
        self.fc_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != self.input_size or x.shape[1] == 0:
            raise ValueError("Expected nonempty (batch, time, input_size) feature tensor.")
        h = self.input_projection(x)
        # Positions follow real resampled timesteps, including frames with missing modalities.
        positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, self.hidden_size, 2, device=x.device) *
                                (-math.log(10000.0) / self.hidden_size))
        encoding = torch.zeros((x.shape[1], self.hidden_size), device=x.device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        h = self.encoder(self.position_dropout(h + encoding.to(h.dtype)))
        if self.pooling == "mean_max":
            pooled = torch.cat((h.mean(dim=1), h.max(dim=1).values), dim=-1)
        elif self.pooling == "attention":
            pooled = self.attn_pool(h)
        elif self.pooling == "last":
            pooled = h[:, -1]
        else:
            pooled = h.mean(dim=1)
        return self.fc(self.fc_dropout(pooled))
