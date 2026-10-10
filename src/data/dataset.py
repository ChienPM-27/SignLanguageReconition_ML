"""PyTorch Dataset cho tập dữ liệu Landmark ASL-Citizen.

Tính năng:
- Đọc chuỗi đặc trưng 143 chiều từ file .npy đã trích xuất.
- Tự động áp dụng Data Augmentation khi ở chế độ huấn luyện (train).
- Trả về tensor x shape (T=32, 143) và nhãn integer y trong [0, num_classes-1].
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import apply_random_augmentation


class ASLLandmarkDataset(Dataset):
    """PyTorch Dataset đọc các file landmark .npy cho 20 gloss ASL-Citizen."""

    def __init__(
        self,
        split_csv: str | Path,
        landmarks_dir: str | Path,
        is_train: bool = False,
        augment: bool = False,
        sequence_length: int = 32,
        feature_dim: int = 143,
    ):
        self.split_csv = Path(split_csv)
        self.landmarks_dir = Path(landmarks_dir)
        self.is_train = is_train
        self.augment = augment and is_train
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim

        self.samples = []
        with open(self.split_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "video_id": row["video_id"],
                    "gloss": row["gloss"],
                    "label": int(row["label"]),
                    "signer_id": row.get("signer_id", ""),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        vid_id = sample["video_id"]
        label = sample["label"]

        npy_path = self.landmarks_dir / f"{vid_id}.npy"

        if npy_path.exists():
            features = np.load(str(npy_path)).astype(np.float32)
        else:
            # Fallback tensor 0 nếu file chưa được extract
            features = np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)

        # Đảm bảo đúng shape (T, feature_dim)
        if features.shape != (self.sequence_length, self.feature_dim):
            padded = np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)
            cur_t = min(self.sequence_length, features.shape[0])
            cur_d = min(self.feature_dim, features.shape[1])
            padded[:cur_t, :cur_d] = features[:cur_t, :cur_d]
            features = padded

        # Áp dụng Augmentation nếu bật
        if self.augment:
            features = apply_random_augmentation(features)

        x_tensor = torch.from_numpy(features).float()
        y_tensor = torch.tensor(label, dtype=torch.long)

        return x_tensor, y_tensor
