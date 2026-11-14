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
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augmentation import apply_augmentation
from src.data.device_loader import DeviceDataLoader
from src.features.schema import (
    PREPROCESS_SCHEMA_VERSION,
    compute_preprocess_identity,
    infer_feature_layout_from_dim,
)
from src.utils.logging import get_logger

logger = get_logger("islr.dataset")

MAX_PRELOAD_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB giới hạn an toàn


def load_and_validate_feature_file(
    npy_path: str | Path,
    sequence_length: int,
    feature_dim: Optional[int] = None,
    sequence_id: str = "",
) -> np.ndarray:
    """Helper đọc file .npy, kiểm tra tính toàn vẹn (tồn tại, ndim==2, feature_dim, finite, binary masks) và căn chỉnh thời gian.

    Đảm bảo nguyên tắc: không bao giờ silent pad/truncate hoặc trả tensor 0 âm thầm.
    """
    path = Path(npy_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature file vanished or missing during load for sequence '{sequence_id}': {path}. "
            "Silent zero tensors are strictly prohibited."
        )

    features = np.load(str(path)).astype(np.float32)

    # 1. Kiểm tra shape 2D
    if features.ndim != 2:
        raise ValueError(f"Feature array must be 2D (T, D), got shape {features.shape} at {path} for sequence '{sequence_id}'")

    t_cur, d_cur = features.shape

    # 2. Kiểm tra non-finite (NaN / Inf)
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite values (NaN or Inf) detected in feature file {path} for sequence '{sequence_id}'.")

    # 3. Kiểm tra feature_dim
    if feature_dim is not None and d_cur != feature_dim:
        raise ValueError(
            f"Feature dimension mismatch in {path} for sequence '{sequence_id}': expected {feature_dim}, got {d_cur}"
        )
    layout = infer_feature_layout_from_dim(d_cur)

    # 4. Kiểm tra đúng chiều thời gian sequence_length (Strict check: không silent pad/truncate)
    if t_cur != sequence_length:
        raise ValueError(
            f"Feature sequence length mismatch in {path} for sequence '{sequence_id}': "
            f"expected {sequence_length}, got {t_cur}. Silent padding or truncation is strictly prohibited."
        )

    # 5. Kiểm tra tính nhị phân của các mask hỗ trợ (support masks must be strictly binary 0.0 or 1.0)
    hand_support = features[:, layout.slice_hand_masks]
    if not np.all((hand_support == 0.0) | (hand_support == 1.0)):
        raise ValueError(
            f"Non-binary hand support mask values detected in {path} for sequence '{sequence_id}'. "
            "Support masks must strictly contain 0.0 or 1.0."
        )
    if layout.has_pose and layout.slice_pose_mask is not None:
        pose_support = features[:, layout.slice_pose_mask]
        if not np.all((pose_support == 0.0) | (pose_support == 1.0)):
            raise ValueError(
                f"Non-binary pose support mask values detected in {path} for sequence '{sequence_id}'. "
                "Pose masks must strictly contain 0.0 or 1.0."
            )
    if layout.slice_geometry_support is not None:
        geom_support = features[:, layout.slice_geometry_support]
        if not np.all((geom_support == 0.0) | (geom_support == 1.0)):
            raise ValueError(
                f"Non-binary geometry support mask values detected in {path} for sequence '{sequence_id}'. "
                "Geometry support masks must strictly contain 0.0 or 1.0."
            )
    if layout.slice_face_mask is not None:
        face_support = features[:, layout.slice_face_mask]
        if not np.all((face_support == 0.0) | (face_support == 1.0)):
            raise ValueError(f"Non-binary face support mask values detected in {path} for sequence '{sequence_id}'.")

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
        expected_preprocess_identity: Optional[str] = None,
        normalization_epsilon: float = 1e-4,
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
        self.expected_preprocess_identity = expected_preprocess_identity
        self.normalization_epsilon = normalization_epsilon

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

        label_to_gloss: Dict[int, str] = {}
        for s in self.samples:
            lbl = s["label"]
            gls = s["gloss"]
            if lbl not in label_to_gloss and gls:
                label_to_gloss[lbl] = gls

        if label_to_gloss:
            max_lbl = max(label_to_gloss.keys())
            if set(label_to_gloss.keys()) == set(range(max_lbl + 1)):
                self.class_names: List[str] = [label_to_gloss[i] for i in range(max_lbl + 1)]
                self.label_map: Dict[str, int] = {cname: idx for idx, cname in enumerate(self.class_names)}
            else:
                self.class_names = [label_to_gloss[k] for k in sorted(label_to_gloss.keys())]
                self.label_map = {cname: idx for idx, cname in enumerate(self.class_names)}
        else:
            self.class_names = []
            self.label_map = {}

        # Strict validation of preprocess_meta.json across all unique feature directories
        feature_cache_dirs: Set[Path] = {s["path"].parent.resolve() for s in self.samples}
        if self.features_dir is not None:
            feature_cache_dirs.add(self.features_dir.resolve())

        self.preprocess_identity: Optional[str] = None
        for cdir in sorted(feature_cache_dirs):
            meta_path = cdir / "preprocess_meta.json"
            if not meta_path.is_file() and cdir.name == "features":
                meta_path = cdir.parent / "preprocess_meta.json"

            if not meta_path.is_file():
                raise FileNotFoundError(
                    f"Missing canonical 'preprocess_meta.json' in feature cache directory: {cdir}. "
                    "All feature directories must contain valid preprocessing metadata."
                )

            try:
                with open(meta_path, mode="r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as exc:
                raise ValueError(f"Corrupted or unreadable preprocess_meta.json at {meta_path}: {exc}") from exc

            if not isinstance(meta, dict):
                raise ValueError(f"Invalid preprocess_meta.json format at {meta_path}: root must be a JSON object.")

            if "config_hash" not in meta or not meta["config_hash"]:
                raise ValueError(
                    f"Missing canonical 'config_hash' in {meta_path}. Legacy 'hash' fallback is strictly prohibited."
                )

            params = meta.get("parameters")
            if not isinstance(params, dict):
                raise ValueError(f"Missing canonical 'parameters' dictionary in {meta_path}.")

            meta_schema = meta.get("schema_version") or params.get("schema_version")
            if meta_schema != PREPROCESS_SCHEMA_VERSION:
                raise ValueError(
                    f"Preprocessing schema version mismatch in {meta_path}: "
                    f"dataset has version {meta_schema}, but codebase requires version {PREPROCESS_SCHEMA_VERSION}. "
                    "Features must be regenerated."
                )

            computed_id, _ = compute_preprocess_identity(params)
            if computed_id != meta["config_hash"]:
                raise ValueError(
                    f"Corrupted metadata in {meta_path}: recomputed identity hash '{computed_id}' "
                    f"does not match declared config_hash '{meta['config_hash']}'."
                )

            current_id = str(meta["config_hash"])
            if self.preprocess_identity is not None and self.preprocess_identity != current_id:
                raise ValueError(
                    f"Mixed feature caches detected! Cache directory {cdir} has identity '{current_id}', "
                    f"conflicting with previously validated identity '{self.preprocess_identity}'."
                )
            self.preprocess_identity = current_id

        if expected_preprocess_identity is not None and self.preprocess_identity != expected_preprocess_identity:
            raise ValueError(
                f"Dataset preprocess identity mismatch: cache has '{self.preprocess_identity}', "
                f"but expected '{expected_preprocess_identity}'. Features must be regenerated."
            )

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
                feat_np = apply_augmentation(
                    feat_np, cfg=self.augmentation_config, normalization_epsilon=self.normalization_epsilon
                )
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
            features = apply_augmentation(
                features, cfg=self.augmentation_config, normalization_epsilon=self.normalization_epsilon
            )

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
        from src.data.augmentation_gpu import apply_augmentation_batch

        return DeviceDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            generator=generator,
            augment=getattr(dataset, "augment_on_device", False),
            augmentation_config=getattr(dataset, "augmentation_config", None),
            augmentation_fn=apply_augmentation_batch,
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
