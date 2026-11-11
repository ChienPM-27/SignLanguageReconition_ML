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
import json
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


def _create_synthetic_split_dataset(
    tmp_path: Path,
    num_samples: int = 6,
    num_classes: int = 3,
    feature_dim: int = 143,
    create_meta: bool = True,
):
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
        # Generate array of shape (32, feature_dim) with strictly binary support masks
        arr = np.random.randn(32, feature_dim).astype(np.float32)
        arr[:, 141:143] = 1.0
        if feature_dim >= 160:
            arr[:, 159:160] = 1.0
        np.save(str(npy_path), arr)
        rows.append([seq_id, f"signer_{i % 2}", gloss, lbl, str(npy_path)])

    with open(split_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "participant_id", "gloss", "label", "path"])
        writer.writerows(rows)

    if create_meta:
        from src.features.schema import compute_preprocess_identity
        from src.utils.config import load_config
        cfg = load_config()
        cfg.features.set = "hands_pose" if feature_dim >= 160 else "hands"
        identity, params = compute_preprocess_identity(cfg)
        with open(feat_dir / "preprocess_meta.json", "w", encoding="utf-8") as f:
            json.dump({"config_hash": identity, "parameters": params}, f, indent=2)

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

    # Time stretch is disabled in Phase 0
    with pytest.raises(ValueError, match="time_stretch is disabled in Phase 0"):
        time_stretch(arr, max_stretch=0.2)
    assert time_stretch(arr, max_stretch=0.0).shape == arr.shape

    # Config-driven apply_augmentation
    aug_cfg = {
        "flip": {"enabled": True, "prob": 1.0, "swap_hands": False},
        "jitter": {"enabled": True, "prob": 1.0, "sigma": 0.01},
        "rotate": {"enabled": True, "prob": 1.0, "max_angle_deg": 5.0},
        "time_stretch": {"enabled": False, "prob": 0.0, "max_stretch": 0.0},
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
        "time_stretch": {"enabled": False, "prob": 0.0, "max_stretch": 0.0},
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

        expected_np = random_rotate_2d(arr_np, angle_deg=angle)
        actual_torch = batch_random_rotate_2d(x_tensor, angles_deg=torch.tensor([angle]))
        actual_np = actual_torch.squeeze(0).cpu().numpy()

        assert np.allclose(actual_np, expected_np, atol=1e-5)


def test_batch_jitter_dimension_boundary() -> None:
    # Jitter touches local hand shapes [0:126], global wrists [126:132], and recomputes motion [132:141].
    # Local wrist landmark 0 must remain identically zero.
    # Pose channels [143:160] must remain zero when pose is absent.
    x = torch.zeros(2, 32, 160, dtype=torch.float32)
    x[:, :, 141:143] = 1.0
    out = batch_add_jitter(x, sigma=0.05)

    # First 132 dims have noise on non-origin landmarks
    assert not torch.equal(out[:, :, :132], x[:, :, :132])
    # Local wrist origins (landmark 0 of both hands: indices 0:3 and 63:66) remain exactly zero
    assert torch.equal(out[:, :, 0:3], torch.zeros(2, 32, 3))
    assert torch.equal(out[:, :, 63:66], torch.zeros(2, 32, 3))
    # Absent pose channels [143:160] remain strictly zero
    assert torch.equal(out[:, :, 143:], torch.zeros(2, 32, 17))


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
    # Hand masks remain identical
    assert torch.equal(batch_x[:, :, 141:], raw_x[:, :, 141:])


def test_dataset_rejects_silent_pad_truncate(tmp_path: Path) -> None:
    """Ensure silent padding or truncation is strictly prohibited when T != sequence_length."""
    feat_file = tmp_path / "bad_len.npy"
    # Create array with 25 timesteps instead of expected 32
    bad_arr = np.ones((25, 143), dtype=np.float32)
    bad_arr[:, 141:143] = 1.0
    np.save(str(feat_file), bad_arr)

    from src.data.dataset import load_and_validate_feature_file

    with pytest.raises(ValueError, match="Silent padding or truncation is strictly prohibited"):
        load_and_validate_feature_file(feat_file, sequence_length=32, feature_dim=143)


def test_dataset_rejects_nonfinite_values(tmp_path: Path) -> None:
    """Ensure feature loader rejects arrays containing NaN or Inf."""
    from src.data.dataset import load_and_validate_feature_file

    feat_file = tmp_path / "nan_feat.npy"
    arr = np.ones((32, 143), dtype=np.float32)
    arr[5, 10] = np.nan
    np.save(str(feat_file), arr)

    with pytest.raises(ValueError, match="Non-finite values"):
        load_and_validate_feature_file(feat_file, sequence_length=32, feature_dim=143)


def test_dataset_rejects_nonbinary_support_masks(tmp_path: Path) -> None:
    """Ensure feature loader rejects arrays with non-binary values in presence masks."""
    from src.data.dataset import load_and_validate_feature_file

    feat_file = tmp_path / "nonbinary_mask.npy"
    arr = np.ones((32, 143), dtype=np.float32)
    arr[:, 141:143] = 0.5  # Invalid non-binary value
    np.save(str(feat_file), arr)

    with pytest.raises(ValueError, match="Non-binary hand support mask values detected"):
        load_and_validate_feature_file(feat_file, sequence_length=32, feature_dim=143)


def test_dataset_rejects_missing_preprocess_meta(tmp_path: Path) -> None:
    """Ensure dataset rejects cache missing canonical preprocess_meta.json."""
    split_csv, feat_dir = _create_synthetic_split_dataset(tmp_path, num_samples=4, num_classes=2, create_meta=False)
    with pytest.raises(FileNotFoundError, match="Missing canonical 'preprocess_meta.json'"):
        ISRLandmarkDataset(split_csv=split_csv, sequence_length=32, feature_dim=143)


def test_dataset_rejects_corrupted_or_legacy_meta(tmp_path: Path) -> None:
    """Ensure dataset rejects metadata with legacy 'hash' fallback or mismatched parameters."""
    split_csv, feat_dir = _create_synthetic_split_dataset(tmp_path, num_samples=4, num_classes=2, create_meta=False)

    # 1. Test legacy 'hash' key without 'config_hash'
    legacy_meta = feat_dir / "preprocess_meta.json"
    with open(legacy_meta, "w", encoding="utf-8") as f:
        json.dump({"hash": "legacy_abc123", "parameters": {}}, f)
    with pytest.raises(ValueError, match="Missing canonical 'config_hash'"):
        ISRLandmarkDataset(split_csv=split_csv, sequence_length=32, feature_dim=143)

    # 2. Test schema version mismatch
    with open(legacy_meta, "w", encoding="utf-8") as f:
        json.dump({"config_hash": "abc", "parameters": {"schema_version": 1}}, f)
    with pytest.raises(ValueError, match="Preprocessing schema version mismatch"):
        ISRLandmarkDataset(split_csv=split_csv, sequence_length=32, feature_dim=143)

    # 3. Test parameter hash mismatch with declared config_hash
    from src.features.schema import compute_preprocess_identity
    from src.utils.config import load_config
    cfg = load_config()
    _, params = compute_preprocess_identity(cfg)
    with open(legacy_meta, "w", encoding="utf-8") as f:
        json.dump({"config_hash": "tampered_fake_hash", "parameters": params}, f)
    with pytest.raises(ValueError, match="Corrupted metadata"):
        ISRLandmarkDataset(split_csv=split_csv, sequence_length=32, feature_dim=143)


def test_dataset_rejects_mixed_cache_directories(tmp_path: Path) -> None:
    """Ensure dataset fails loudly if samples come from directories with conflicting identities."""
    dir_a = tmp_path / "cache_a"
    dir_b = tmp_path / "cache_b"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    from src.features.schema import compute_preprocess_identity
    from src.utils.config import load_config
    cfg_a = load_config()
    cfg_a.preprocess.max_gap_frames = 3
    id_a, params_a = compute_preprocess_identity(cfg_a)
    with open(dir_a / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump({"config_hash": id_a, "parameters": params_a}, f)

    cfg_b = load_config()
    cfg_b.preprocess.max_gap_frames = 5
    id_b, params_b = compute_preprocess_identity(cfg_b)
    with open(dir_b / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump({"config_hash": id_b, "parameters": params_b}, f)

    # File in A
    arr_a = np.ones((32, 143), dtype=np.float32)
    arr_a[:, 141:143] = 1.0
    path_a = dir_a / "seq_a.npy"
    np.save(str(path_a), arr_a)

    # File in B
    arr_b = np.ones((32, 143), dtype=np.float32)
    arr_b[:, 141:143] = 1.0
    path_b = dir_b / "seq_b.npy"
    np.save(str(path_b), arr_b)

    split_csv = tmp_path / "mixed.csv"
    with open(split_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "participant_id", "gloss", "label", "path"])
        writer.writerow(["seq_a", "signer_1", "sign_0", 0, str(path_a)])
        writer.writerow(["seq_b", "signer_2", "sign_1", 1, str(path_b)])

    with pytest.raises(ValueError, match="Mixed feature caches detected"):
        ISRLandmarkDataset(split_csv=split_csv, sequence_length=32, feature_dim=143)


def test_dataset_expected_preprocess_identity(tmp_path: Path) -> None:
    """Ensure expected_preprocess_identity strictly validates cache identity."""
    split_csv, feat_dir = _create_synthetic_split_dataset(tmp_path, num_samples=4, num_classes=2, create_meta=True)

    with open(feat_dir / "preprocess_meta.json", "r", encoding="utf-8") as f:
        actual_id = json.load(f)["config_hash"]

    # Matching identity succeeds
    ds = ISRLandmarkDataset(split_csv=split_csv, sequence_length=32, feature_dim=143, expected_preprocess_identity=actual_id)
    assert ds.preprocess_identity == actual_id

    # Mismatched identity raises
    with pytest.raises(ValueError, match="Dataset preprocess identity mismatch"):
        ISRLandmarkDataset(split_csv=split_csv, sequence_length=32, feature_dim=143, expected_preprocess_identity="wrong_id")
