"""PyTorch Dataset cho tập dữ liệu chuỗi đặc trưng ISLR.

Đặc tính:
- Đọc chuỗi đặc trưng .npy từ manifest / split CSV.
- Kiểm tra tính toàn vẹn của file: nếu thiếu file, ném FileNotFoundError hoặc
  lọc bỏ kèm cảnh báo ĐẾM RÕ; TUYỆT ĐỐI không trả tensor 0 âm thầm.
- Hỗ trợ Data Augmentation qua AugmentationConfig / dict.
- Suy luận số chiều feature_dim tự động, không hardcode.
- Trả về tensor x shape (sequence_length, feature_dim) và nhãn integer y.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augmentation import apply_augmentation
from src.utils.logging import get_logger

logger = get_logger("islr.dataset")


class ISRLandmarkDataset(Dataset):
    """PyTorch Dataset đọc các file landmark .npy cho ISLR."""

    def __init__(
        self,
        split_csv: str | Path,
        features_dir: Optional[str | Path] = None,
        is_train: bool = False,
        augment: bool = False,
        augmentation_config: Optional[Any] = None,
        sequence_length: int = 32,
        feature_dim: Optional[int] = None,
        strict: bool = True,
    ):
        self.split_csv = Path(split_csv)
        self.features_dir = Path(features_dir) if features_dir is not None else None
        self.is_train = is_train
        self.augment = augment and is_train
        self.augmentation_config = augmentation_config
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.strict = strict

        if not self.split_csv.is_file():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv}")

        self.samples: List[Dict[str, Any]] = []
        raw_rows: List[Dict[str, str]] = []
        with open(self.split_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows.append(row)

        total_rows = len(raw_rows)
        missing_count = 0

        for r in raw_rows:
            # Hỗ trợ cả ISLR schema (sequence_id, participant_id) và legacy ASL-Citizen (video_id, signer_id)
            seq_id = r.get("sequence_id") or r.get("video_id") or ""
            part_id = r.get("participant_id") or r.get("signer_id") or ""
            gloss = r.get("gloss") or ""
            label_val = r.get("label")
            if label_val is None or label_val == "":
                continue
            label = int(label_val)

            # Xác định đường dẫn file .npy
            path_str = r.get("path")
            if path_str:
                candidate_path = Path(path_str)
                if not candidate_path.is_file() and self.features_dir is not None:
                    candidate_path = self.features_dir / candidate_path.name
            elif self.features_dir is not None and seq_id:
                candidate_path = self.features_dir / f"{seq_id}.npy"
            else:
                candidate_path = self.split_csv.parent / f"{seq_id}.npy"

            if not candidate_path.is_file():
                missing_count += 1
                if self.strict:
                    raise FileNotFoundError(
                        f"Missing feature .npy file for sequence '{seq_id}': {candidate_path}. "
                        "Set strict=False to filter missing files with an explicit count warning."
                    )
                continue

            self.samples.append({
                "sequence_id": seq_id,
                "participant_id": part_id,
                "gloss": gloss,
                "label": label,
                "path": candidate_path,
            })

        if missing_count > 0:
            logger.warning(
                f"Dataset initialized with missing files filtered: {missing_count}/{total_rows} "
                f"missing samples removed from {self.split_csv.name}. Retained: {len(self.samples)} samples."
            )

        if len(self.samples) == 0:
            raise ValueError(f"No valid feature files found from {self.split_csv} (total entries: {total_rows}).")

        # Suy luận feature_dim nếu chưa cung cấp
        if self.feature_dim is None:
            first_arr = np.load(str(self.samples[0]["path"]))
            self.feature_dim = first_arr.shape[1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        npy_path = sample["path"]

        if not npy_path.is_file():
            raise FileNotFoundError(
                f"Feature file vanished or missing during getitem for sequence '{sample['sequence_id']}': {npy_path}. "
                "Silent zero tensors are strictly prohibited."
            )

        features = np.load(str(npy_path)).astype(np.float32)

        # Kiểm tra shape
        if features.ndim != 2:
            raise ValueError(f"Feature array must be 2D (T, D), got shape {features.shape} at {npy_path}")

        t_cur, d_cur = features.shape
        if d_cur != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, got {d_cur} at {npy_path}"
            )

        # Đảm bảo đúng chiều thời gian sequence_length
        if t_cur != self.sequence_length:
            padded = np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)
            valid_t = min(t_cur, self.sequence_length)
            padded[:valid_t, :] = features[:valid_t, :]
            features = padded

        # Áp dụng Augmentation nếu ở chế độ train và bật augment
        if self.augment:
            features = apply_augmentation(features, cfg=self.augmentation_config)

        x_tensor = torch.from_numpy(features).float()
        y_tensor = torch.tensor(sample["label"], dtype=torch.long)

        return x_tensor, y_tensor


# Alias tương thích ngược
ASLLandmarkDataset = ISRLandmarkDataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """Tạo DataLoader tiện ích cho ISLR."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
