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
import math
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
from src.data.augmentation_gpu import (
    apply_augmentation_batch,
    batch_add_jitter,
    batch_horizontal_flip,
    batch_random_rotate_2d,
    batch_time_stretch,
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
    arr[:, 141:143] = 1.0

    # Jitter
    jittered = add_jitter(arr, sigma=0.01)
    assert jittered.shape == arr.shape
    assert not np.array_equal(arr, jittered)

    # Rotate
    rotated = random_rotate_2d(arr, max_angle_deg=10.0)
    assert rotated.shape == arr.shape

    # Flip without hand swapping (swap_hands=False, default)
    flipped_no_swap = horizontal_flip(arr, swap_hands=False)
    assert flipped_no_swap.shape == arr.shape
    # Mask is NOT swapped
    assert np.allclose(flipped_no_swap[:, 141:143], arr[:, 141:143])
    # Slot 0 local x is inverted in place
    orig_s0_x = arr[:, :126].reshape(32, 2, 21, 3)[:, 0, :, 0]
    flip_s0_x = flipped_no_swap[:, :126].reshape(32, 2, 21, 3)[:, 0, :, 0]
    assert np.allclose(flip_s0_x, -orig_s0_x)

    # Flip with hand swapping (swap_hands=True)
    flipped_swap = horizontal_flip(arr, swap_hands=True)
    assert flipped_swap.shape == arr.shape
    # Mask IS swapped
    assert np.allclose(flipped_swap[:, 141:143], arr[:, [142, 141]])
    # Slot 0 gets inverted Slot 1
    orig_s1_x = arr[:, :126].reshape(32, 2, 21, 3)[:, 1, :, 0]
    flip_swap_s0_x = flipped_swap[:, :126].reshape(32, 2, 21, 3)[:, 0, :, 0]
    assert np.allclose(flip_swap_s0_x, -orig_s1_x)

    # Time stretch
    stretched = time_stretch(arr, max_stretch=0.2)
    assert stretched.shape == arr.shape

    # Config-driven apply_augmentation
    aug_cfg = {
        "flip": {"enabled": True, "prob": 1.0, "swap_hands": False},
        "jitter": {"enabled": True, "prob": 1.0, "sigma": 0.01},
        "rotate": {"enabled": True, "prob": 1.0, "max_angle_deg": 5.0},
        "time_stretch": {"enabled": True, "prob": 1.0, "max_stretch": 0.1},
    }
    augmented = apply_augmentation(arr, cfg=aug_cfg)
    assert augmented.shape == arr.shape


def test_preload_modes_parity(tmp_path: Path) -> None:
    split_csv, _ = _create_synthetic_split_dataset(tmp_path, num_samples=6, num_classes=3)

    ds_none = ISRLandmarkDataset(split_csv=split_csv, preload="none", sequence_length=32, feature_dim=143)
    ds_ram = ISRLandmarkDataset(split_csv=split_csv, preload="ram", sequence_length=32, feature_dim=143)
    ds_dev = ISRLandmarkDataset(split_csv=split_csv, preload="device", device="cpu", sequence_length=32, feature_dim=143)

    assert ds_none.preloaded_features is None
    assert ds_none.preloaded_labels is None
    assert ds_ram.preloaded_features is not None
    assert ds_ram.preloaded_labels is not None
    assert ds_dev.preloaded_features is not None
    assert ds_dev.preloaded_labels is not None

    assert ds_ram.preloaded_features.shape == (6, 32, 143)
    assert ds_dev.preloaded_features.shape == (6, 32, 143)

    for i in range(len(ds_none)):
        x_none, y_none = ds_none[i]
        x_ram, y_ram = ds_ram[i]
        x_dev, y_dev = ds_dev[i]

        assert torch.allclose(x_none, x_ram)
        assert torch.allclose(x_none, x_dev)
        assert y_none == y_ram == y_dev


def test_device_dataloader_behavior(tmp_path: Path) -> None:
    split_csv, _ = _create_synthetic_split_dataset(tmp_path, num_samples=7, num_classes=2)
    ds = ISRLandmarkDataset(split_csv=split_csv, preload="device", device="cpu", sequence_length=32, feature_dim=143)

    # 1. Non-drop-last loader
    loader = create_dataloader(ds, batch_size=3, shuffle=False, drop_last=False)
    assert len(loader) == 3
    assert loader.dataset is ds

    batches = list(loader)
    assert len(batches) == 3
    assert batches[0][0].shape == (3, 32, 143)
    assert batches[1][0].shape == (3, 32, 143)
    assert batches[2][0].shape == (1, 32, 143)

    # Multiple iterations have identical deterministic order
    batches_again = list(loader)
    assert torch.allclose(batches[0][0], batches_again[0][0])
    assert torch.equal(batches[0][1], batches_again[0][1])

    # 2. Drop-last loader
    loader_dl = create_dataloader(ds, batch_size=3, shuffle=False, drop_last=True)
    assert len(loader_dl) == 2
    batches_dl = list(loader_dl)
    assert len(batches_dl) == 2
    assert batches_dl[0][0].shape == (3, 32, 143)
    assert batches_dl[1][0].shape == (3, 32, 143)

    # 3. Reproducible shuffling with seed
    l_seed1 = create_dataloader(ds, batch_size=2, shuffle=True, seed=42)
    l_seed2 = create_dataloader(ds, batch_size=2, shuffle=True, seed=42)
    b1 = list(l_seed1)
    b2 = list(l_seed2)
    for (x1, y1), (x2, y2) in zip(b1, b2):
        assert torch.allclose(x1, x2)
        assert torch.equal(y1, y2)


def test_preload_memory_limit_exceeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split_csv, _ = _create_synthetic_split_dataset(tmp_path, num_samples=4, num_classes=2)
    import src.data.dataset as ds_module

    # Monkeypatch limit to 100 bytes so 4*32*143*4 bytes exceeds it
    monkeypatch.setattr(ds_module, "MAX_PRELOAD_BYTES", 100)

    with pytest.raises(ValueError, match="vượt quá giới hạn an toàn 4 GiB"):
        ISRLandmarkDataset(split_csv=split_csv, preload="ram", sequence_length=32, feature_dim=143)


def test_preload_corrupted_file_raises(tmp_path: Path) -> None:
    split_csv, feat_dir = _create_synthetic_split_dataset(tmp_path, num_samples=3, num_classes=2)

    # Corrupt one file to have ndim == 1
    np.save(str(feat_dir / "seq_001.npy"), np.ones(50, dtype=np.float32))

    with pytest.raises(ValueError, match="Feature array must be 2D"):
        ISRLandmarkDataset(split_csv=split_csv, preload="ram", sequence_length=32, feature_dim=143)


def test_batch_augmentation_invariants() -> None:
    # Shape, dtype, non-inplace, finite
    B, T, D = 4, 32, 143
    x = torch.randn(B, T, D, dtype=torch.float32)
    x_copy = x.clone()

    aug_cfg = {
        "flip": {"enabled": True, "prob": 1.0, "swap_hands": False},
        "jitter": {"enabled": True, "prob": 1.0, "sigma": 0.01},
        "rotate": {"enabled": True, "prob": 1.0, "max_angle_deg": 5.0},
        "time_stretch": {"enabled": True, "prob": 1.0, "max_stretch": 0.1},
    }

    gen = torch.Generator().manual_seed(42)
    out = apply_augmentation_batch(x, cfg=aug_cfg, generator=gen)

    # 1. Output properties
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()

    # 2. Input tensor NOT modified in-place
    assert torch.equal(x, x_copy)

    # 3. Seed reproducibility
    gen1 = torch.Generator().manual_seed(123)
    gen2 = torch.Generator().manual_seed(123)
    out1 = apply_augmentation_batch(x, cfg=aug_cfg, generator=gen1)
    out2 = apply_augmentation_batch(x, cfg=aug_cfg, generator=gen2)
    assert torch.allclose(out1, out2)

    # 4. prob = 0.0 returns exact identical tensor
    zero_cfg = {
        "flip": {"enabled": True, "prob": 0.0},
        "jitter": {"enabled": True, "prob": 0.0},
        "rotate": {"enabled": True, "prob": 0.0},
        "time_stretch": {"enabled": True, "prob": 0.0},
    }
    assert torch.equal(apply_augmentation_batch(x, cfg=zero_cfg), x)


def test_batch_flip_matches_numpy() -> None:
    np.random.seed(99)
    # Test on both 143 and 160 dims
    for D in [143, 160]:
        arr_np = np.random.randn(32, D).astype(np.float32)
        x_tensor = torch.from_numpy(arr_np).unsqueeze(0)  # (1, 32, D)

        for swap in [False, True]:
            expected_np = horizontal_flip(arr_np, swap_hands=swap)
            actual_torch = batch_horizontal_flip(x_tensor, swap_hands=swap)
            actual_np = actual_torch.squeeze(0).cpu().numpy()

            assert np.allclose(actual_np, expected_np, atol=1e-6)


def test_batch_rotate_matches_numpy() -> None:
    # Use fixed angle to test exact numeric match with 2D rotation formula
    angle = 7.5
    for D in [143, 160]:
        arr_np = np.random.randn(32, D).astype(np.float32)
        arr_np[:, 141:143] = 1.0
        x_tensor = torch.from_numpy(arr_np).unsqueeze(0)  # (1, 32, D)

        # Compute manual numpy rotation with angle 7.5 deg
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        expected_np = arr_np.copy()
        shapes = expected_np[:, :126].reshape(32, 2, 21, 3)
        xy = shapes[:, :, :, :2].reshape(-1, 2)
        shapes[:, :, :, :2] = np.dot(xy, rot_mat.T).reshape(32, 2, 21, 2)
        expected_np[:, :126] = shapes.reshape(32, 126)

        wrists = expected_np[:, 126:132].reshape(32, 2, 3)
        w_xy = wrists[:, :, :2].reshape(-1, 2) - 0.5
        wrists[:, :, :2] = (np.dot(w_xy, rot_mat.T) + 0.5).reshape(32, 2, 2)
        expected_np[:, 126:132] = wrists.reshape(32, 6)

        actual_torch = batch_random_rotate_2d(x_tensor, angles_deg=torch.tensor([angle]))
        actual_np = actual_torch.squeeze(0).cpu().numpy()

        assert np.allclose(actual_np, expected_np, atol=1e-6)


def test_batch_jitter_dimension_boundary() -> None:
    # Jitter must ONLY touch min(132, D) dimensions on present slots
    x = torch.zeros(2, 32, 160, dtype=torch.float32)
    x[:, :, 141:143] = 1.0
    out = batch_add_jitter(x, sigma=0.05)

    # First 132 dims have noise
    assert not torch.equal(out[:, :, :132], x[:, :, :132])
    # Dims 132..160 must remain strictly 0
    assert torch.equal(out[:, :, 132:], x[:, :, 132:])


def test_batch_time_stretch_identity() -> None:
    x = torch.randn(2, 32, 143, dtype=torch.float32)
    # rate = 1.0 must be the exact identity mapping
    out = batch_time_stretch(x, rates=torch.tensor([1.0, 1.0]))
    assert torch.allclose(out, x, atol=1e-6)


def test_device_loader_with_batch_augmentation(tmp_path: Path) -> None:
    split_csv, _ = _create_synthetic_split_dataset(tmp_path, num_samples=4, num_classes=2)

    aug_cfg = {
        "jitter": {"enabled": True, "prob": 1.0, "sigma": 0.05},
    }

    # Train dataset with augmentation
    ds_train = ISRLandmarkDataset(
        split_csv=split_csv,
        is_train=True,
        augment=True,
        augmentation_config=aug_cfg,
        preload="device",
        device="cpu",
    )
    # In device mode, per-sample augment is False, augment_on_device is True
    assert ds_train.augment is False
    assert ds_train.augment_on_device is True

    loader = create_dataloader(ds_train, batch_size=4, shuffle=False)
    batch_x, batch_y = next(iter(loader))

    # Preloaded feature should remain clean
    raw_x = ds_train.preloaded_features
    assert not torch.equal(batch_x, raw_x)
    assert not torch.allclose(batch_x[:, :, :132], raw_x[:, :, :132])
    # The non-jittered dims remain identical
    assert torch.equal(batch_x[:, :, 132:], raw_x[:, :, 132:])


