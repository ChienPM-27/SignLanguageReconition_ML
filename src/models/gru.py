"""Kiến trúc mô hình GRU / LSTM phân loại chuỗi thời gian nhận diện cử chỉ ASL.

Hỗ trợ:
- Cấu hình chuẩn: 2 lớp GRU / LSTM, hidden 128, dropout 0.3.
- Cấu hình ablation nhỏ: 1 lớp GRU / LSTM, hidden 64, dropout 0.2.
- Pooling nâng cao:
    + 'mean_max': Kết hợp Global Average Pooling và Global Max Pooling (chụp cả context và apex).
    + 'attention': Temporal Attention Pooling qua các frame.
    + 'mean': Global Average Pooling.
    + 'last': Hidden state ở frame cuối cùng (baseline).
- Input: tensor (Batch, T=32, D=143 hoặc 160). Output: logits (Batch, num_classes).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Mapping, Optional, Sequence


class TemporalAttention(nn.Module):
    """Temporal attention pooling over sequence length T."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, T, hidden_dim)
        scores = self.attn(x)  # (Batch, T, 1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)  # (Batch, hidden_dim)
        return context


class ASLGRUClassifier(nn.Module):
    """Mô hình phân loại chuỗi landmarks ASL sử dụng GRU hoặc LSTM với pooling nâng cao."""

    def __init__(
        self,
        input_size: int = 160,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 20,
        dropout: float = 0.3,
        bidirectional: bool = False,
        pooling: str = "mean_max",
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.pooling = pooling.lower()
        self.rnn_type = rnn_type.upper()

        gru_dropout = dropout if num_layers > 1 else 0.0

        if self.rnn_type == "LSTM":
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=gru_dropout,
                bidirectional=bidirectional,
            )
        else:
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=gru_dropout,
                bidirectional=bidirectional,
            )

        direction_factor = 2 if bidirectional else 1
        rnn_out_dim = hidden_size * direction_factor

        if self.pooling == "mean_max":
            fc_in_dim = rnn_out_dim * 2
        elif self.pooling == "attention":
            fc_in_dim = rnn_out_dim
            self.attn_pool = TemporalAttention(rnn_out_dim)
        elif self.pooling in ("mean", "last"):
            fc_in_dim = rnn_out_dim
        else:
            raise ValueError(
                f"Unknown pooling '{pooling}'. Allowed: 'mean_max', 'attention', 'mean', 'last'."
            )

        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor shape (Batch, T, input_size)

        Returns:
            logits: Tensor shape (Batch, num_classes)
        """
        # Lựa chọn submodule tương ứng
        rnn = self.lstm if self.rnn_type == "LSTM" else self.gru
        out, _ = rnn(x)  # (Batch, T, hidden_size * directions)

        if self.pooling == "mean_max":
            mean_pool = torch.mean(out, dim=1)
            max_pool, _ = torch.max(out, dim=1)
            pooled = torch.cat([mean_pool, max_pool], dim=-1)
        elif self.pooling == "attention":
            pooled = self.attn_pool(out)
        elif self.pooling == "mean":
            pooled = torch.mean(out, dim=1)
        elif self.pooling == "last":
            pooled = out[:, -1, :]
        else:
            pooled = out[:, -1, :]

        dropped = self.fc_dropout(pooled)
        logits = self.fc(dropped)
        return logits


# Alias cho phép khởi tạo class ASLLSTMClassifier trực tiếp
class ASLLSTMClassifier(ASLGRUClassifier):
    """Mô hình phân loại chuỗi landmarks ASL sử dụng Long Short-Term Memory (LSTM)."""

    def __init__(
        self,
        input_size: int = 160,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 20,
        dropout: float = 0.3,
        bidirectional: bool = False,
        pooling: str = "mean_max",
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
            pooling=pooling,
            rnn_type="LSTM",
        )


def create_model(
    input_size: int = 160,
    num_classes: int = 20,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    bidirectional: bool = False,
    pooling: str = "mean_max",
    rnn_type: str = "GRU",
    is_ablation: bool = False,
    attention_heads: int = 4,
    feedforward_size: int = 256,
    region_fusion: bool = False,
    regions: Optional[Sequence[str]] = None,
    region_weights: Optional[Mapping[str, float]] = None,
    region_normalization_epsilon: float = 0.000001,
) -> nn.Module:
    """Hàm tiện ích tạo mô hình GRU hoặc LSTM chuẩn hoặc mô hình ablation nhỏ gọn."""
    if region_fusion:
        from src.models.region_fusion import RegionFusionClassifier
        from src.utils.config import ModelConfig
        from src.features.schema import FeatureLayout
        fusion_cfg = ModelConfig(region_fusion=True, region_normalization_epsilon=region_normalization_epsilon)
        if regions is not None:
            fusion_cfg.regions = list(regions)
        if region_weights is not None:
            fusion_cfg.region_weights = dict(region_weights)
        fusion_cfg.validate()
        if input_size != FeatureLayout.build(feature_set="hands_pose", face=True).total_dim or is_ablation:
            raise ValueError("Region fusion requires the 196-channel base layout and explicit architecture settings.")
        core = create_model(hidden_size, num_classes, hidden_size, num_layers, dropout,
                            bidirectional, pooling, rnn_type, False, attention_heads, feedforward_size)
        return RegionFusionClassifier(core, fusion_cfg.regions, fusion_cfg.region_weights,
                                      fusion_cfg.region_normalization_epsilon)
    if rnn_type.upper() == "TRANSFORMER":
        from src.models.transformer import ASLTransformerClassifier
        if is_ablation or bidirectional:
            raise ValueError("Transformer does not support the RNN ablation preset or bidirectional flag.")
        return ASLTransformerClassifier(input_size, hidden_size, num_layers, num_classes,
                                        dropout, pooling, attention_heads, feedforward_size)
    if rnn_type.upper() not in ("GRU", "LSTM"):
        raise ValueError(f"Unsupported model type: {rnn_type}")
    if is_ablation:
        return ASLGRUClassifier(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            num_classes=num_classes,
            dropout=0.2,
            bidirectional=bidirectional,
            pooling=pooling,
            rnn_type=rnn_type,
        )
    return ASLGRUClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
        pooling=pooling,
        rnn_type=rnn_type,
    )
