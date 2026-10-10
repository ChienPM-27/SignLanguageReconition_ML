"""Kiến trúc mô hình GRU phân loại chuỗi thời gian nhận diện cử chỉ ASL.

Hỗ trợ:
- Cấu hình chuẩn: 2 lớp GRU, hidden 128, dropout 0.3.
- Cấu hình ablation nhỏ: 1 lớp GRU, hidden 64, dropout 0.2 (chống overfit trên tập dữ liệu nhỏ).
- Input: tensor (Batch, T=32, 143). Output: logits (Batch, 20).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ASLGRUClassifier(nn.Module):
    """Mô hình phân loại chuỗi landmarks ASL sử dụng Gated Recurrent Unit (GRU)."""

    def __init__(
        self,
        input_size: int = 143,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 20,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # GRU Dropout chỉ có hiệu lực khi num_layers > 1
        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * direction_factor, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor shape (Batch, T, input_size)

        Returns:
            logits: Tensor shape (Batch, num_classes)
        """
        # out shape: (Batch, T, hidden_size * directions)
        out, _ = self.gru(x)

        # Lấy hidden state tại frame cuối cùng của chuỗi (T=-1)
        last_step_feature = out[:, -1, :]

        # Dropout và chiếu qua Linear classifier
        dropped = self.fc_dropout(last_step_feature)
        logits = self.fc(dropped)
        return logits


def create_model(
    input_size: int = 143,
    num_classes: int = 20,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    is_ablation: bool = False,
) -> ASLGRUClassifier:
    """Hàm tiện ích tạo mô hình GRU chuẩn hoặc mô hình ablation nhỏ gọn."""
    if is_ablation:
        return ASLGRUClassifier(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            num_classes=num_classes,
            dropout=0.2,
        )
    return ASLGRUClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )
