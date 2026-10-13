"""PyTorch Dataset cho tập dữ liệu chuỗi đặc trưng ISLR.

Đặc tính:
- Đọc chuỗi đặc trưng .npy từ manifest / split CSV.
- Kiểm tra tính toàn vẹn của file: nếu thiếu file, ném FileNotFoundError hoặc
  lọc bỏ kèm cảnh báo ĐẾM RÕ; TUYỆT ĐỐI không trả tensor 0 âm thầm.
- Hỗ trợ nạp trước dữ liệu (Preload: 'none', 'ram', 'device') với cơ chế bảo vệ bộ nhớ.
- Hỗ trợ Data Augmentation qua AugmentationConfig / dict.
- Suy luận số chiều feature_dim tự động, không hardcode.
- Trả về tensor x shape (sequence_length, feature_dim) và nhãn integer y.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augmentation import apply_augmentation
from src.data.device_loader import DeviceDataLoader
from src.utils.logging import get_logger

logger = get_logger("islr.dataset")

MAX_PRELOAD_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB giới hạn an toàn


def load_and_validate_feature_file(
    npy_path: str | Path,
    sequence_length: int,
    feature_dim: int,
    sequence_id: str = "",
) -> np.ndarray:
    """Helper đọc file .npy, kiểm tra tính toàn vẹn (tồn tại, ndim==2, feature_dim) và căn chỉnh thời gian.

    Đảm bảo nguyên tắc: không bao giờ trả tensor 0 âm thầm.
    """
    path = Path(npy_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature file vanished or missing during load for sequence '{sequence_id}': {path}. "
            "Silent zero tensors are strictly prohibited."
        )

    features = np.load(str(path)).astype(np.float32)

    # Kiểm tra shape
    if features.ndim != 2:
        raise ValueError(f"Feature array must be 2D (T, D), got shape {features.shape} at {path}")

    t_cur, d_cur = features.shape
    if d_cur != feature_dim:
        raise ValueError(
            f"Feature dimension mismatch: expected {feature_dim}, got {d_cur} at {path}"
        )

    # Đảm bảo đúng chiều thời gian sequence_length
    if t_cur != sequence_length:
        padded = np.zeros((sequence_length, feature_dim), dtype=np.float32)
        valid_t = min(t_cur, sequence_length)
        padded[:valid_t, :] = features[:valid_t, :]
        features = padded

    return features


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
        preload: str = "none",
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.split_csv = Path(split_csv)
        self.features_dir = Path(features_dir) if features_dir is not None else None
        self.is_train = is_train
        self.augment = augment and is_train
        self.augmentation_config = augmentation_config
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.strict = strict
        self.preload = preload

        if self.preload not in {"none", "ram", "device"}:
            raise ValueError(f"Invalid preload mode '{self.preload}'. Allowed: ['none', 'ram', 'device'].")

        if device is not None:
            self.device: Optional[torch.device] = torch.device(device)
        elif self.preload == "device":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

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
            if first_arr.ndim != 2:
                raise ValueError(
                    f"Feature array must be 2D (T, D), got shape {first_arr.shape} at {self.samples[0]['path']}"
                )
            self.feature_dim = first_arr.shape[1]

        # Nạp trước dữ liệu vào bộ nhớ nếu preload != 'none'
        self._preloaded_features: Optional[torch.Tensor] = None
        self._preloaded_labels: Optional[torch.Tensor] = None
        self.augment_on_device: bool = False

        if self.preload in {"ram", "device"}:
            expected_bytes = len(self.samples) * self.sequence_length * self.feature_dim * 4
            logger.info(
                f"Preloading {len(self.samples)} sequences into {self.preload} memory "
                f"(estimated feature size: {expected_bytes / (1024 * 1024):.2f} MB)..."
            )
            if expected_bytes > MAX_PRELOAD_BYTES:
                raise ValueError(
                    f"Dung lượng dữ liệu dự kiến ({expected_bytes / (1024**3):.2f} GiB) vượt quá giới hạn an toàn 4 GiB "
                    f"cho training.preload='{self.preload}'. Vui lòng giảm kích thước tập dữ liệu hoặc sử dụng "
                    f"'--set training.preload=none' để đọc trực tiếp từ đĩa."
                )

            features_tensor = torch.empty(
                (len(self.samples), self.sequence_length, self.feature_dim),
                dtype=torch.float32,
            )
            labels_tensor = torch.empty((len(self.samples),), dtype=torch.long)

            for idx, sample in enumerate(self.samples):
                feat = load_and_validate_feature_file(
                    sample["path"],
                    sequence_length=self.sequence_length,
                    feature_dim=self.feature_dim,
                    sequence_id=sample["sequence_id"],
                )
                features_tensor[idx] = torch.from_numpy(feat)
                labels_tensor[idx] = sample["label"]

            if self.preload == "device":
                assert self.device is not None
                self._preloaded_features = features_tensor.to(self.device)
                self._preloaded_labels = labels_tensor.to(self.device)
                # Dataset ở chế độ "device" phải tắt augmentation per-sample (tránh augment hai lần)
                self.augment_on_device = self.augment
                self.augment = False
            else:
                self._preloaded_features = features_tensor
                self._preloaded_labels = labels_tensor

    @property
    def preloaded_features(self) -> Optional[torch.Tensor]:
        """Tensor đặc trưng đã nạp trước (hoặc None nếu preload='none')."""
        return self._preloaded_features

    @property
    def preloaded_labels(self) -> Optional[torch.Tensor]:
        """Tensor nhãn đã nạp trước (hoặc None nếu preload='none')."""
        return self._preloaded_labels

    @property
    def preload_mode(self) -> str:
        """Chế độ preload hiện tại ('none', 'ram', 'device')."""
        return self.preload

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # Phục vụ từ bộ nhớ nếu đã preload
        if self._preloaded_features is not None and self._preloaded_labels is not None:
            if self.augment:
                feat_np = self._preloaded_features[idx].cpu().numpy().copy()
                feat_np = apply_augmentation(feat_np, cfg=self.augmentation_config)
                x_tensor = torch.from_numpy(feat_np).float()
            else:
                x_tensor = self._preloaded_features[idx]
            y_tensor = self._preloaded_labels[idx]
            return x_tensor, y_tensor

        # Preload là 'none': đọc lười (lazy load) từ đĩa
        features = load_and_validate_feature_file(
            sample["path"],
            sequence_length=self.sequence_length,
            feature_dim=self.feature_dim,
            sequence_id=sample["sequence_id"],
        )

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
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Union[DataLoader, DeviceDataLoader]:
    """Tạo DataLoader tiện ích cho ISLR, tự động hỗ trợ DeviceDataLoader khi preload='device'."""
    if getattr(dataset, "preload", "none") == "device":
        return DeviceDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            generator=generator,
            augment=getattr(dataset, "augment_on_device", False),
            augmentation_config=getattr(dataset, "augmentation_config", None),
        )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        generator=generator,
    )
