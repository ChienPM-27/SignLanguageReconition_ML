"""Wave B regression tests: validity across pipeline, augmentations, and quality gates.

Addresses Astra review findings:
- B-01: Transition validity preservation across dropouts and geometric transforms
- B-02: Canonical normalization epsilon propagation across loaders and augmentation helpers
- B-03: Body wrist velocity suppression across pose dropouts in torso_xy mode
- B-04: Post-resampling quality gating and predictor zero-forward prevention
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.data.augmentation import (
    apply_augmentation,
    horizontal_flip,
    recompute_derived_features,
    scale_translate,
)
from src.data.augmentation_gpu import (
    apply_augmentation_batch,
    batch_horizontal_flip,
)
from src.data.dataset import ISRLandmarkDataset, create_dataloader
from src.features.pipeline import prepare_landmark_sequence
from src.features.schema import (
    PREPROCESS_SCHEMA_VERSION,
    compute_preprocess_identity,
    get_feature_dim,
    get_feature_layout_from_config,
)
from src.inference.predictor import ISLRPredictor
from src.models.gru import ASLGRUClassifier
from src.utils.config import Config


def _base_config() -> Config:
    cfg = Config()
    cfg.features.set = "hands"
    cfg.model.hidden_size = 16
    cfg.model.num_layers = 1
    cfg.model.dropout = 0.0
    cfg.model.pooling = "last"
    cfg.preprocess.sequence_length = 4
    cfg.preprocess.trim_no_hand = False
    cfg.preprocess.dominant_hand = "none"
    cfg.training.preload = "none"
    cfg.training.device = "cpu"
    cfg.training.epochs = 1
    cfg.training.seeds = [42]
    cfg.training.num_seeds = 1
    for name in ("flip", "rotate", "jitter", "shape_scale", "scale_translate", "time_stretch"):
        getattr(cfg.augmentation, name)["enabled"] = False
    cfg.validate()
    return cfg


def _raw_fixture(n: int = 100):
    t = np.arange(n, dtype=np.float32)
    hands = np.zeros((n, 2, 21, 3), dtype=np.float32)
    for h in range(2):
        wrist = np.column_stack((0.3 + 0.3 * h + 0.002 * t, 0.55 + 0.001 * t, np.zeros(n)))
        offsets = np.column_stack((np.arange(21) * 0.001, np.arange(21) * 0.0015, np.arange(21) * 0.0005))
        hands[:, h] = wrist[:, None, :] + offsets[None, :, :]
        hands[:, h, 0] = wrist
        hands[:, h, 9] = wrist + [0.1, 0.0, 0.0]
        hands[:, h, 8, 0] += 0.0005 * t
    pose = np.zeros((n, 4, 3), dtype=np.float32)
    pose[:, 0] = np.column_stack((0.4 + 0.001 * t, np.full(n, 0.3), np.zeros(n)))
    pose[:, 1] = pose[:, 0] + [0.2, 0, 0]
    pose[:, 2] = pose[:, 0] + [-0.05, 0.2, 0]
    pose[:, 3] = pose[:, 1] + [0.05, 0.2, 0]
    return hands, np.ones((n, 2), dtype=bool), pose, np.ones((n, 1), dtype=bool)


def _make_model(cfg: Config) -> ASLGRUClassifier:
    return ASLGRUClassifier(
        input_size=get_feature_dim(cfg),
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_classes=2,
        dropout=cfg.model.dropout,
        pooling=cfg.model.pooling,
        bidirectional=cfg.model.bidirectional,
        rnn_type=cfg.model.name,
    )


def test_b01_hand_dropout_preserves_zero_motion_across_gap() -> None:
    """B-01: Zero motion across long gap must be preserved after augmentation flip on CPU and Torch."""
    cfg = _base_config()
    cfg.features.fingertip_displacement = True
    raw, masks, _, _ = _raw_fixture(100)
    masks[40:61] = False  # Long gap > 5 frames

    result = prepare_landmark_sequence(raw_coords=raw, hand_masks=masks, cfg=cfg)
    assert result.is_usable is True
    assert "hand_transitions" in result.provenance
    assert "pose_transitions" in result.provenance

    layout = get_feature_layout_from_config(cfg)
    clean_tip_max = float(np.abs(result.features[2, layout.slice_fingertip_displacement]).max())
    clean_wrist_vel_max = float(np.abs(result.features[2, layout.slice_wrist_vel]).max())
    assert clean_tip_max == 0.0
    assert clean_wrist_vel_max == 0.0

    # CPU horizontal flip
    cpu_flip = horizontal_flip(result.features)
    cpu_tip_max = float(np.abs(cpu_flip[2, layout.slice_fingertip_displacement]).max())
    cpu_wrist_vel_max = float(np.abs(cpu_flip[2, layout.slice_wrist_vel]).max())
    assert cpu_tip_max == 0.0
    assert cpu_wrist_vel_max == 0.0

    # Torch batch horizontal flip
    t_feat = torch.from_numpy(result.features).unsqueeze(0)
    device_flip = batch_horizontal_flip(t_feat)[0].numpy()
    torch_tip_max = float(np.abs(device_flip[2, layout.slice_fingertip_displacement]).max())
    torch_wrist_vel_max = float(np.abs(device_flip[2, layout.slice_wrist_vel]).max())
    assert torch_tip_max == 0.0
    assert torch_wrist_vel_max == 0.0


def test_b03_pose_dropout_suppresses_body_wrist_velocity() -> None:
    """B-03: Body wrist velocity in torso_xy mode must be gated by pose transitions."""
    cfg = _base_config()
    cfg.features.set = "hands_pose"
    cfg.features.wrist_reference = "torso_xy"
    raw, masks, pose, pose_masks = _raw_fixture(100)
    pose_masks[40:61] = False  # Long pose gap

    result = prepare_landmark_sequence(
        raw_coords=raw, hand_masks=masks, pose_coords=pose, pose_masks=pose_masks, cfg=cfg
    )
    assert result.is_usable is True

    # At timestep 2 (across pose gap), body wrist velocity and torso velocity must be 0
    body_wrist_vel = result.features[2, 135:141]
    torso_vel = result.features[2, 155:158]
    assert np.all(body_wrist_vel == 0.0)
    assert np.all(torso_vel == 0.0)


def test_b02_canonical_epsilon_propagated_to_helpers() -> None:
    """B-02: Configured normalization_epsilon (0.01) must propagate to shape_scale and scale_translate."""
    cfg = _base_config()
    cfg.features.set = "hands_pose"
    cfg.features.bend_cosines = True
    cfg.preprocess.normalization_epsilon = 0.01
    cfg.validate()

    features = np.zeros((4, get_feature_dim(cfg)), dtype=np.float32)
    features[:, 141:143] = 1.0
    features[:, 159] = 1.0
    features[:, 158] = 0.2
    features[:, 143:146] = [-0.5, 0, 0]
    features[:, 146:149] = [0.5, 0, 0]
    shapes = features[:, :126].reshape(4, 2, 21, 3)
    shapes[:, :, 9] = [1, 0, 0]
    for k, index in enumerate((5, 6, 7, 8)):
        shapes[:, :, index] = [k * 0.005, 0, 0]
    features = recompute_derived_features(features, normalization_epsilon=0.01)

    layout = get_feature_layout_from_config(cfg)
    bend_idx = layout.slice_bend_cosines.start + 2

    # Clean bend cosine under epsilon 0.01 with bone length 0.005 is degenerate (0.0)
    clean_bend = float(features[0, bend_idx])
    assert clean_bend == 0.0

    # CPU shape_scale with canonical epsilon 0.01
    shape_cfg = copy.deepcopy(cfg.augmentation)
    shape_cfg.shape_scale.update(enabled=True, prob=1.0, r=0.1)
    cpu_shape = apply_augmentation(features, shape_cfg, normalization_epsilon=0.01)
    assert float(cpu_shape[0, bend_idx]) == 0.0

    # Torch shape_scale with canonical epsilon 0.01
    t_feat = torch.from_numpy(features).unsqueeze(0)
    torch_shape = apply_augmentation_batch(
        t_feat, shape_cfg, generator=torch.Generator().manual_seed(42), normalization_epsilon=0.01
    )[0].numpy()
    assert float(torch_shape[0, bend_idx]) == 0.0

    # CPU scale_translate with canonical epsilon 0.01
    cfg_image = _base_config()
    cfg_image.features.bend_cosines = True
    cfg_image.preprocess.normalization_epsilon = 0.01
    cfg_image.validate()
    feat_image = np.zeros((4, get_feature_dim(cfg_image)), dtype=np.float32)
    feat_image[:, 141:143] = 1.0
    feat_image[:, 126:132] = 0.5
    shapes_img = feat_image[:, :126].reshape(4, 2, 21, 3)
    shapes_img[:, :, 9] = [1, 0, 0]
    for k, index in enumerate((5, 6, 7, 8)):
        shapes_img[:, :, index] = [k * 0.005, 0, 0]
    feat_image = recompute_derived_features(feat_image, normalization_epsilon=0.01)

    trans_cfg = copy.deepcopy(cfg_image.augmentation)
    trans_cfg.scale_translate.update(enabled=True, prob=1.0)
    cpu_trans = scale_translate(feat_image, normalization_epsilon=0.01)
    layout_img = get_feature_layout_from_config(cfg_image)
    assert float(cpu_trans[0, layout_img.slice_bend_cosines.start + 2]) == 0.0


def test_b02_loader_epsilon_modes(tmp_path: Path) -> None:
    """B-02: Normalization epsilon is honored across preload='none', 'ram', 'device'."""
    cfg = _base_config()
    cfg.features.set = "hands_pose"
    cfg.features.bend_cosines = True
    cfg.preprocess.normalization_epsilon = 0.01
    cfg.augmentation.flip.update(enabled=True, prob=1.0, swap_hands=False)
    cfg.validate()

    features = np.zeros((4, get_feature_dim(cfg)), dtype=np.float32)
    features[:, 141:143] = 1.0
    features[:, 159] = 1.0
    features[:, 158] = 0.2
    features[:, 143:146] = [-0.5, 0, 0]
    features[:, 146:149] = [0.5, 0, 0]
    shapes = features[:, :126].reshape(4, 2, 21, 3)
    shapes[:, :, 9] = [1, 0, 0]
    for k, index in enumerate((5, 6, 7, 8)):
        shapes[:, :, index] = [k * 0.005, 0, 0]
    features = recompute_derived_features(features, normalization_epsilon=0.01)

    layout = get_feature_layout_from_config(cfg)
    bend_idx = layout.slice_bend_cosines.start + 2

    for mode in ("none", "ram", "device"):
        dir_p = tmp_path / f"eps_{mode}"
        cache = dir_p / "features"
        cache.mkdir(parents=True, exist_ok=True)
        identity, params = compute_preprocess_identity(cfg)
        (cache / "preprocess_meta.json").write_text(
            json.dumps({"schema_version": PREPROCESS_SCHEMA_VERSION, "config_hash": identity, "parameters": params}),
            encoding="utf-8",
        )
        np.save(cache / "0.npy", features)

        split_csv = dir_p / "train.csv"
        with open(split_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sequence_id", "participant_id", "gloss", "label", "path"])
            w.writerow(["0", "p0", "hello", 0, str(cache / "0.npy")])

        dataset = ISRLandmarkDataset(
            split_csv=split_csv,
            is_train=True,
            augment=True,
            augmentation_config=cfg.augmentation,
            sequence_length=4,
            feature_dim=features.shape[1],
            preload=mode,
            device="cpu",
            expected_preprocess_identity=identity,
            normalization_epsilon=cfg.preprocess.normalization_epsilon,
        )
        batch = next(iter(create_dataloader(dataset, batch_size=1)))[0][0].numpy()
        assert float(batch[0, bend_idx]) == 0.0
        assert dataset.normalization_epsilon == 0.01


def test_b04_post_resample_quality_gate(tmp_path: Path) -> None:
    """B-04: Resampled sequence with 0 target support is unusable and rejected by predictor without model forward."""
    cfg = _base_config()
    raw, masks, _, _ = _raw_fixture(100)
    masks[:] = False
    masks[10:13] = True  # 3 valid frames, but at 0, 33, 66, 99 all are false

    result = prepare_landmark_sequence(raw_coords=raw, hand_masks=masks, cfg=cfg)
    assert result.is_usable is False
    assert result.rejection_reason == "zero_target_hand_support"

    # Set up predictor
    glosses_csv = tmp_path / "glosses.csv"
    glosses_csv.write_text("gloss,label\nhello,0\nworld,1\n", encoding="utf-8")
    model = _make_model(cfg)
    ckpt_path = tmp_path / "model.pt"
    identity, _ = compute_preprocess_identity(cfg)
    payload = {
        "epoch": 1,
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "preprocess_identity": identity,
        "resolved_config": cfg.to_dict(),
        "class_names": ["hello", "world"],
        "label_map": {"hello": 0, "world": 1},
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, str(ckpt_path))

    with patch("src.inference.predictor.MediaPipeHandExtractor"):
        predictor = ISLRPredictor(ckpt_path, glosses_csv=glosses_csv, device="cpu")

    with patch.object(predictor.model, "forward", wraps=predictor.model.forward) as mock_forward:
        with pytest.raises(ValueError, match="Unusable sequence: zero_target_hand_support"):
            predictor.predict_raw_coordinates(raw, masks)
        assert mock_forward.call_count == 0

    predictor.close()
