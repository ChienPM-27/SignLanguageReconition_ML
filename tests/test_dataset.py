"""Tests for ISLR PyTorch Dataset and Data Augmentation (Step 5).

Covers:
- ISRLandmarkDataset reading .npy features.
- Strict missing file detection (raises FileNotFoundError loudly).
- Non-strict missing file filtering (counts and logs warning, never silent zeros).
- Augmentation toggles (flip, jitter, rotate, time_stretch).
- DataLoader batching and dimension checks.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.augmentation import (
    add_jitter,
    apply_augmentation,
    horizontal_flip,
    random_rotate_2d,
    time_stretch,
)
from src.data.dataset import ISRLandmarkDataset, create_dataloader


def _create_synthetic_split_dataset(tmp_path: Path, num_samples: int = 6, num_classes: int = 3, feature_dim: int = 143):
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_csv = split_dir / "train.csv"

    feat_dir = tmp_path / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(num_samples):
        seq_id = f"seq_{i:03d}"
        gloss = f"gloss_{i % num_classes}"
        lbl = i % num_classes
        npy_path = feat_dir / f"{seq_id}.npy"
        # Generate random array of shape (32, feature_dim)
        arr = np.random.randn(32, feature_dim).astype(np.float32)
        np.save(str(npy_path), arr)
        rows.append([seq_id, f"signer_{i % 2}", gloss, lbl, str(npy_path)])

    with open(split_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "participant_id", "gloss", "label", "path"])
        writer.writerows(rows)

    return split_csv, feat_dir


def test_dataset_loading_clean(tmp_path: Path) -> None:
    split_csv, feat_dir = _create_synthetic_split_dataset(tmp_path, num_samples=6, num_classes=3)

    ds = ISRLandmarkDataset(
        split_csv=split_csv,
        is_train=True,
        augment=False,
        sequence_length=32,
        feature_dim=143,
        strict=True,
    )

    assert len(ds) == 6
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (32, 143)
    assert y.dtype == torch.long


def test_dataset_strict_missing_file_raises(tmp_path: Path) -> None:
    split_csv, feat_dir = _create_synthetic_split_dataset(tmp_path, num_samples=4, num_classes=2)

    # Delete one file to trigger missing file detection
    missing_file = feat_dir / "seq_002.npy"
    missing_file.unlink()

    with pytest.raises(FileNotFoundError, match="Missing feature .npy file"):
        ISRLandmarkDataset(
            split_csv=split_csv,
            is_train=True,
            strict=True,
        )


def test_dataset_non_strict_filters_with_count(tmp_path: Path) -> None:
    split_csv, feat_dir = _create_synthetic_split_dataset(tmp_path, num_samples=6, num_classes=2)

    # Delete two files
    (feat_dir / "seq_001.npy").unlink()
    (feat_dir / "seq_003.npy").unlink()

    ds = ISRLandmarkDataset(
        split_csv=split_csv,
        is_train=False,
        strict=False,
    )

    # Should retain 4 valid samples
    assert len(ds) == 4


def test_dataset_dataloader_batching(tmp_path: Path) -> None:
    split_csv, feat_dir = _create_synthetic_split_dataset(tmp_path, num_samples=8, num_classes=4)

    ds = ISRLandmarkDataset(split_csv=split_csv, sequence_length=32, feature_dim=143)
    loader = create_dataloader(ds, batch_size=4, shuffle=False)

    batches = list(loader)
    assert len(batches) == 2
    x_b, y_b = batches[0]
    assert x_b.shape == (4, 32, 143)
    assert y_b.shape == (4,)


def test_augmentation_functions() -> None:
    arr = np.random.randn(32, 143).astype(np.float32)

    # Jitter
    jittered = add_jitter(arr, sigma=0.01)
    assert jittered.shape == arr.shape
    assert not np.array_equal(arr, jittered)

    # Rotate
    rotated = random_rotate_2d(arr, max_angle_deg=10.0)
    assert rotated.shape == arr.shape

    # Flip
    flipped = horizontal_flip(arr)
    assert flipped.shape == arr.shape

    # Time stretch
    stretched = time_stretch(arr, max_stretch=0.2)
    assert stretched.shape == arr.shape

    # Config-driven apply_augmentation
    aug_cfg = {
        "flip": {"enabled": True, "prob": 1.0},
        "jitter": {"enabled": True, "prob": 1.0, "sigma": 0.01},
        "rotate": {"enabled": True, "prob": 1.0, "max_angle_deg": 5.0},
        "time_stretch": {"enabled": True, "prob": 1.0, "max_stretch": 0.1},
    }
    augmented = apply_augmentation(arr, cfg=aug_cfg)
    assert augmented.shape == arr.shape
