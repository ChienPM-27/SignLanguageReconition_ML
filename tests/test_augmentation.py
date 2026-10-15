"""Unit tests for Phase A augmentation fixes and new transformations.

Tests cover:
- Preservation of absent hand slots (mask <= 0.5) across all augmentations (numpy & torch),
  specifically including non-zero interpolated wrists from process_video_landmarks.
- Parity of recompute_derived_features against compute_feature_vector_143 on real pipeline features.
- Double horizontal flip (swap_hands=False) is exact identity on present slots (including rel and vel).
- Exact numeric parity between numpy and torch implementations for every augmentation.
- Preservation of tensor shape (T, 143) and float32 dtype.
- Training smoke test for 2 epochs on synthetic data with augment=True for both preload='ram' and preload='device'.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch

from src.data.augmentation import (
    add_jitter,
    apply_augmentation,
    horizontal_flip,
    random_rotate_2d,
    recompute_derived_features,
    scale_translate,
    shape_scale,
)
from src.data.augmentation_gpu import (
    apply_augmentation_batch,
    batch_add_jitter,
    batch_horizontal_flip,
    batch_random_rotate_2d,
    batch_recompute_derived_features,
    batch_scale_translate,
    batch_shape_scale,
)
from src.data.dataset import ISRLandmarkDataset, create_dataloader
from src.features.normalization import process_video_landmarks
from src.models.gru import ASLGRUClassifier
from src.training.train import train_gru
from src.utils.config import load_config


def _create_synthetic_video_landmarks_with_gaps(
    raw_len: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate raw landmarks with gaps to produce interpolated non-zero coords in absent slots."""
    np.random.seed(123)
    raw_coords = np.zeros((raw_len, 2, 21, 3), dtype=np.float32)
    hand_masks = np.zeros((raw_len, 2), dtype=bool)

    # Hand 0: present across all frames with valid hand coordinates
    hand_masks[:, 0] = True
    raw_coords[:, 0, 0, :] = np.linspace([0.3, 0.4, 0.0], [0.4, 0.5, 0.1], raw_len)
    for pt in range(1, 21):
        raw_coords[:, 0, pt, :] = raw_coords[:, 0, 0, :] + np.random.uniform(0.01, 0.05, size=(raw_len, 3))

    # Hand 1: present in two separated segments [8:14] and [26:34]
    # The gap of 12 frames (> max_gap_frames=3) is NOT filled by interpolate_bounded_gaps,
    # but temporal resampling will linearly interpolate coords across the gap into absent frames!
    hand_masks[8:14, 1] = True
    raw_coords[8:14, 1, 0, :] = np.linspace([0.7, 0.4, 0.0], [0.75, 0.45, 0.05], 6)
    for pt in range(1, 21):
        raw_coords[8:14, 1, pt, :] = raw_coords[8:14, 1, 0, :] + np.random.uniform(0.01, 0.05, size=(6, 3))

    hand_masks[26:34, 1] = True
    raw_coords[26:34, 1, 0, :] = np.linspace([0.8, 0.5, 0.05], [0.85, 0.6, 0.1], 8)
    for pt in range(1, 21):
        raw_coords[26:34, 1, pt, :] = raw_coords[26:34, 1, 0, :] + np.random.uniform(0.01, 0.05, size=(8, 3))

    features_143, resampled_masks = process_video_landmarks(raw_coords, hand_masks, target_len=32)
    return features_143, resampled_masks, raw_coords


def test_absent_slot_preservation_numpy() -> None:
    """Verify absent hand slots (mask <= 0.5) are strictly unchanged by all numpy augmentations."""
    features_143, resampled_masks, _ = _create_synthetic_video_landmarks_with_gaps()

    absent_hand1 = ~resampled_masks[:, 1]
    assert absent_hand1.any(), "Test requires at least one absent frame for hand 1"

    # Verify that in some absent frames, wrist coords are non-zero due to interpolation
    absent_wrist1 = features_143[absent_hand1, 129:132]
    assert np.any(absent_wrist1 != 0.0), "Interpolated wrist in absent slot should have non-zero coords"

    aug_cfg = {
        "flip": {"enabled": True, "prob": 1.0, "swap_hands": False},
        "rotate": {"enabled": True, "prob": 1.0, "max_angle_deg": 15.0},
        "shape_scale": {"enabled": True, "prob": 1.0, "r": 0.20},
        "scale_translate": {"enabled": True, "prob": 1.0, "scale_min": 0.8, "scale_max": 1.2, "translate": 0.15},
        "jitter": {"enabled": True, "prob": 1.0, "sigma": 0.05},
        "time_stretch": {"enabled": False, "prob": 0.0},
    }

    # 1. Flip
    flipped = horizontal_flip(features_143, swap_hands=False)
    assert np.array_equal(flipped[absent_hand1, 63:126], features_143[absent_hand1, 63:126])
    assert np.array_equal(flipped[absent_hand1, 129:132], features_143[absent_hand1, 129:132])

    # 2. Rotate
    rotated = random_rotate_2d(features_143, max_angle_deg=15.0)
    assert np.array_equal(rotated[absent_hand1, 63:126], features_143[absent_hand1, 63:126])
    assert np.array_equal(rotated[absent_hand1, 129:132], features_143[absent_hand1, 129:132])

    # 3. Shape scale
    sh_scaled = shape_scale(features_143, r=0.20)
    assert np.array_equal(sh_scaled[absent_hand1, 63:126], features_143[absent_hand1, 63:126])

    # 4. Scale translate
    sc_trans = scale_translate(features_143, scale_min=0.8, scale_max=1.2, translate=0.15)
    assert np.array_equal(sc_trans[absent_hand1, 129:132], features_143[absent_hand1, 129:132])

    # 5. Jitter
    jittered = add_jitter(features_143, sigma=0.05)
    assert np.array_equal(jittered[absent_hand1, 63:126], features_143[absent_hand1, 63:126])
    assert np.array_equal(jittered[absent_hand1, 129:132], features_143[absent_hand1, 129:132])

    # 6. Full pipeline
    full_aug = apply_augmentation(features_143, cfg=aug_cfg)
    assert np.array_equal(full_aug[absent_hand1, 63:126], features_143[absent_hand1, 63:126])
    assert np.array_equal(full_aug[absent_hand1, 129:132], features_143[absent_hand1, 129:132])


def test_absent_slot_preservation_torch() -> None:
    """Verify absent hand slots (mask <= 0.5) are strictly unchanged by all torch batch augmentations."""
    features_143, resampled_masks, _ = _create_synthetic_video_landmarks_with_gaps()
    absent_hand1 = ~resampled_masks[:, 1]
    assert absent_hand1.any()

    x = torch.from_numpy(features_143).unsqueeze(0)  # (1, 32, 143)
    abs_indices = np.where(absent_hand1)[0]

    aug_cfg = {
        "flip": {"enabled": True, "prob": 1.0, "swap_hands": False},
        "rotate": {"enabled": True, "prob": 1.0, "max_angle_deg": 15.0},
        "shape_scale": {"enabled": True, "prob": 1.0, "r": 0.20},
        "scale_translate": {"enabled": True, "prob": 1.0, "scale_min": 0.8, "scale_max": 1.2, "translate": 0.15},
        "jitter": {"enabled": True, "prob": 1.0, "sigma": 0.05},
        "time_stretch": {"enabled": False, "prob": 0.0},
    }

    # 1. Flip
    flipped_t = batch_horizontal_flip(x, swap_hands=False)
    assert torch.equal(flipped_t[0, abs_indices, 63:126], x[0, abs_indices, 63:126])
    assert torch.equal(flipped_t[0, abs_indices, 129:132], x[0, abs_indices, 129:132])

    # 2. Rotate
    rotated_t = batch_random_rotate_2d(x, max_angle_deg=15.0)
    assert torch.equal(rotated_t[0, abs_indices, 63:126], x[0, abs_indices, 63:126])
    assert torch.equal(rotated_t[0, abs_indices, 129:132], x[0, abs_indices, 129:132])

    # 3. Shape scale
    sh_scaled_t = batch_shape_scale(x, r=0.20)
    assert torch.equal(sh_scaled_t[0, abs_indices, 63:126], x[0, abs_indices, 63:126])

    # 4. Scale translate
    sc_trans_t = batch_scale_translate(x, scale_min=0.8, scale_max=1.2, translate=0.15)
    assert torch.equal(sc_trans_t[0, abs_indices, 129:132], x[0, abs_indices, 129:132])

    # 5. Jitter
    jittered_t = batch_add_jitter(x, sigma=0.05)
    assert torch.equal(jittered_t[0, abs_indices, 63:126], x[0, abs_indices, 63:126])
    assert torch.equal(jittered_t[0, abs_indices, 129:132], x[0, abs_indices, 129:132])

    # 6. Full pipeline
    full_aug_t = apply_augmentation_batch(x, cfg=aug_cfg)
    assert torch.equal(full_aug_t[0, abs_indices, 63:126], x[0, abs_indices, 63:126])
    assert torch.equal(full_aug_t[0, abs_indices, 129:132], x[0, abs_indices, 129:132])


def test_recompute_derived_features_parity() -> None:
    """Verify recompute_derived_features reproduces exact [132:141] features from process_video_landmarks."""
    features_143, _, _ = _create_synthetic_video_landmarks_with_gaps()

    # 1. Numpy recompute
    derived_np = recompute_derived_features(features_143)
    assert np.allclose(derived_np[:, 132:141], features_143[:, 132:141], atol=1e-5)

    # 2. Torch batch recompute
    x_tensor = torch.from_numpy(features_143).unsqueeze(0)
    derived_torch = batch_recompute_derived_features(x_tensor)
    derived_torch_np = derived_torch.squeeze(0).cpu().numpy()
    assert np.allclose(derived_torch_np[:, 132:141], features_143[:, 132:141], atol=1e-5)


def test_double_flip_identity() -> None:
    """Verify flipping twice with swap_hands=False is the exact identity mapping on present slots."""
    features_143, _, _ = _create_synthetic_video_landmarks_with_gaps()

    # 1. Numpy
    flip1 = horizontal_flip(features_143, swap_hands=False)
    flip2 = horizontal_flip(flip1, swap_hands=False)
    assert np.allclose(flip2, features_143, atol=1e-5)

    # 2. Torch
    x_tensor = torch.from_numpy(features_143).unsqueeze(0)
    flip1_t = batch_horizontal_flip(x_tensor, swap_hands=False)
    flip2_t = batch_horizontal_flip(flip1_t, swap_hands=False)
    assert torch.allclose(flip2_t, x_tensor, atol=1e-5)


def test_augmentation_parity_numpy_vs_torch() -> None:
    """Verify exact numerical parity between numpy and torch batch for each augmentation with fixed parameters."""
    features_143, _, _ = _create_synthetic_video_landmarks_with_gaps()
    x_tensor = torch.from_numpy(features_143).unsqueeze(0)

    # 1. Flip (both swap_hands=False and swap_hands=True)
    for swap in [False, True]:
        np_flip = horizontal_flip(features_143, swap_hands=swap)
        torch_flip = batch_horizontal_flip(x_tensor, swap_hands=swap).squeeze(0).cpu().numpy()
        assert np.allclose(np_flip, torch_flip, atol=1e-6)

    # 2. Rotate with fixed angle
    angle = 6.25
    np_rot = random_rotate_2d(features_143, angle_deg=angle)
    torch_rot = batch_random_rotate_2d(x_tensor, angles_deg=torch.tensor([angle])).squeeze(0).cpu().numpy()
    assert np.allclose(np_rot, torch_rot, atol=1e-6)

    # 3. Shape scale with fixed factors
    sx, sy = 1.08, 0.94
    np_sh = shape_scale(features_143, sx=sx, sy=sy)
    torch_sh = batch_shape_scale(
        x_tensor,
        sx=torch.tensor([sx], dtype=torch.float32),
        sy=torch.tensor([sy], dtype=torch.float32),
    ).squeeze(0).cpu().numpy()
    assert np.allclose(np_sh, torch_sh, atol=1e-6)

    # 4. Scale translate with fixed factors
    s_val, tx, ty = 1.12, 0.05, -0.06
    np_sc = scale_translate(features_143, s=s_val, tx=tx, ty=ty)
    torch_sc = batch_scale_translate(
        x_tensor,
        s=torch.tensor([s_val], dtype=torch.float32),
        tx=torch.tensor([tx], dtype=torch.float32),
        ty=torch.tensor([ty], dtype=torch.float32),
    ).squeeze(0).cpu().numpy()
    assert np.allclose(np_sc, torch_sc, atol=1e-6)

    # 5. Jitter with identical noise tensors
    noise_s = np.random.randn(32, 2, 21, 3).astype(np.float32) * 0.01
    noise_w = np.random.randn(32, 2, 3).astype(np.float32) * 0.01
    np_jit = add_jitter(features_143, sigma=0.01, noise_shape=noise_s, noise_wrist=noise_w)
    torch_jit = batch_add_jitter(
        x_tensor,
        sigma=0.01,
        noise_shape=torch.from_numpy(noise_s).unsqueeze(0),
        noise_wrist=torch.from_numpy(noise_w).unsqueeze(0),
    ).squeeze(0).cpu().numpy()
    assert np.allclose(np_jit, torch_jit, atol=1e-6)


def test_shape_and_dtype_preservation() -> None:
    """Verify all augmentations preserve (T, 143) shape and float32 dtype."""
    T, D = 32, 143
    arr = np.random.randn(T, D).astype(np.float32)
    arr[:, 141:143] = 1.0
    tensor = torch.from_numpy(arr).unsqueeze(0)

    for aug_fn in [
        lambda a: horizontal_flip(a, swap_hands=False),
        lambda a: horizontal_flip(a, swap_hands=True),
        lambda a: random_rotate_2d(a, max_angle_deg=8.0),
        lambda a: shape_scale(a, r=0.10),
        lambda a: scale_translate(a, scale_min=0.85, scale_max=1.15, translate=0.10),
        lambda a: add_jitter(a, sigma=0.01),
        lambda a: recompute_derived_features(a),
    ]:
        out = aug_fn(arr)
        assert out.shape == (T, D)
        assert out.dtype == np.float32

    for batch_fn in [
        lambda x: batch_horizontal_flip(x, swap_hands=False),
        lambda x: batch_horizontal_flip(x, swap_hands=True),
        lambda x: batch_random_rotate_2d(x, max_angle_deg=8.0),
        lambda x: batch_shape_scale(x, r=0.10),
        lambda x: batch_scale_translate(x, scale_min=0.85, scale_max=1.15, translate=0.10),
        lambda x: batch_add_jitter(x, sigma=0.01),
        lambda x: batch_recompute_derived_features(x),
    ]:
        out_t = batch_fn(tensor)
        assert out_t.shape == (1, T, D)
        assert out_t.dtype == torch.float32


def _setup_smoke_train_env(tmp_path: Path, n_samples: int = 8, n_classes: int = 2) -> Tuple[Path, Path]:
    """Helper to set up synthetic training environment for 2-epoch smoke test."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = tmp_path / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(csv_path: Path, count: int) -> None:
        rows = []
        for i in range(count):
            lbl = i % n_classes
            seq_id = f"{csv_path.stem}_{i:03d}"
            npy_path = feat_dir / f"{seq_id}.npy"
            # Valid feature matrix with active presence masks
            arr = np.random.randn(32, 143).astype(np.float32) * 0.1 + (lbl + 1.0)
            arr[:, 141:143] = 1.0
            np.save(str(npy_path), arr)
            rows.append([seq_id, f"signer_{i % 2}", f"sign_{lbl}", lbl, str(npy_path)])

        with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence_id", "participant_id", "gloss", "label", "path"])
            writer.writerows(rows)

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    write_csv(train_csv, n_samples)
    write_csv(val_csv, max(4, n_samples // 2))

    return train_csv, val_csv


@pytest.mark.parametrize("preload_mode", ["ram", "device"])
def test_smoke_train_2_epochs_with_augmentation(tmp_path: Path, preload_mode: str) -> None:
    """Smoke test training 2 epochs with augmentation enabled on synthetic data."""
    train_csv, val_csv = _setup_smoke_train_env(tmp_path / preload_mode, n_samples=8, n_classes=2)
    cfg = load_config()

    device = torch.device("cpu")
    train_ds = ISRLandmarkDataset(
        split_csv=train_csv,
        is_train=True,
        augment=True,
        augmentation_config=cfg.augmentation,
        sequence_length=32,
        feature_dim=143,
        strict=True,
        preload=preload_mode,
        device=device,
    )
    val_ds = ISRLandmarkDataset(
        split_csv=val_csv,
        is_train=False,
        augment=False,
        sequence_length=32,
        feature_dim=143,
        strict=True,
        preload=preload_mode,
        device=device,
    )

    train_loader = create_dataloader(train_ds, batch_size=4, shuffle=True, seed=42)
    val_loader = create_dataloader(val_ds, batch_size=4, shuffle=False)

    model = ASLGRUClassifier(
        input_size=143,
        hidden_size=16,
        num_layers=1,
        num_classes=2,
        dropout=0.1,
    )
    ckpt_path = tmp_path / preload_mode / "best_model.pt"
    metrics_path = tmp_path / preload_mode / "train_metrics.csv"

    history = train_gru(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        lr=0.01,
        patience=5,
        early_stopping_min_delta=0.001,
        early_stopping_min_epochs=0,
        device=device,
        checkpoint_path=ckpt_path,
        metrics_csv_path=metrics_path,
        verbose=False,
    )

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert ckpt_path.is_file()
    assert metrics_path.is_file()
    for ep in range(2):
        assert np.isfinite(history["train_loss"][ep])
        assert 0.0 <= history["train_acc"][ep] <= 1.0
        assert np.isfinite(history["val_loss"][ep])
        assert 0.0 <= history["val_acc"][ep] <= 1.0
