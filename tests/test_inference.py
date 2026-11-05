"""Tests for inference predictor and realtime prediction functionality."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import torch

from src.features.schema import PREPROCESS_SCHEMA_VERSION
from src.inference.predictor import ISLRPredictor, PredictionResult
from src.models.gru import ASLLSTMClassifier, create_model


def _create_mock_model_and_glosses(
    tmp_path: Path,
    input_size: int = 143,
    pooling: str = "mean_max",
    rnn_type: str = "GRU",
    save_resolved_config: bool = True,
    sequence_length: int = 32,
    custom_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    glosses_csv = tmp_path / "glosses.csv"
    with open(glosses_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gloss", "label", "n_samples"])
        writer.writerow(["hello", "0", "10"])
        writer.writerow(["world", "1", "10"])

    model = create_model(
        input_size=input_size,
        hidden_size=32,
        num_layers=1,
        num_classes=2,
        dropout=0.0,
        pooling=pooling,
        rnn_type=rnn_type,
    )
    model_pt = tmp_path / "best_model.pt"

    if save_resolved_config:
        feature_set = "hands_pose" if input_size >= 160 else "hands"
        cfg_dict = custom_cfg or {
            "features": {"set": feature_set},
            "preprocess": {
                "sequence_length": sequence_length,
                "pad_frames_around": 2,
                "max_gap_frames": 5,
                "min_valid_frames": 5,
                "dominant_hand": True,
                "normalization_epsilon": 0.0001,
            },
        }
        from src.features.schema import compute_preprocess_identity
        from src.utils.config import build_config_from_dict

        cfg_obj = build_config_from_dict(cfg_dict)
        identity_hash, _ = compute_preprocess_identity(cfg_obj)
        payload = {
            "model_state_dict": model.state_dict(),
            "resolved_config": cfg_dict,
            "schema_version": PREPROCESS_SCHEMA_VERSION,
            "preprocess_identity": identity_hash,
        }
        torch.save(payload, str(model_pt))
    else:
        torch.save(model.state_dict(), str(model_pt))

    return model_pt, glosses_csv


def test_predictor_initialization_and_prediction(tmp_path: Path) -> None:
    model_pt, glosses_csv = _create_mock_model_and_glosses(tmp_path, input_size=143, pooling="last")

    predictor = ISLRPredictor(model_path=model_pt, glosses_csv=glosses_csv, device="cpu")
    assert predictor.num_classes == 2
    assert predictor.label_to_gloss[0] == "hello"
    assert predictor.label_to_gloss[1] == "world"
    assert predictor.schema_version == PREPROCESS_SCHEMA_VERSION
    assert predictor.is_legacy is False

    # Test features prediction
    dummy_feats = np.random.randn(32, 143).astype(np.float32)
    res = predictor.predict_features(dummy_feats, top_k=2)
    assert isinstance(res, PredictionResult)
    assert res.predicted_gloss in {"hello", "world"}
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.top_k) == 2
    assert "hello" in res.probabilities
    assert "world" in res.probabilities

    # Test raw coordinates prediction
    raw_coords = np.random.randn(20, 2, 21, 3).astype(np.float32)
    hand_masks = np.ones((20, 2), dtype=bool)
    res_coords = predictor.predict_raw_coordinates(raw_coords, hand_masks)
    assert res_coords.predicted_gloss in {"hello", "world"}
    predictor.close()


def test_predictor_160_dims_and_pose_support(tmp_path: Path) -> None:
    """Verify predictor seamlessly runs on 160-dim hands+pose model with mean_max pooling."""
    model_pt, glosses_csv = _create_mock_model_and_glosses(
        tmp_path, input_size=160, pooling="mean_max", rnn_type="GRU"
    )

    predictor = ISLRPredictor(model_path=model_pt, glosses_csv=glosses_csv, device="cpu")
    assert predictor.input_size == 160
    assert predictor.pooling == "mean_max"

    # Test prediction with raw coords and pose coords
    raw_coords = np.random.randn(25, 2, 21, 3).astype(np.float32)
    hand_masks = np.ones((25, 2), dtype=bool)
    pose_coords = np.random.randn(25, 33, 3).astype(np.float32)
    pose_masks = np.ones((25, 1), dtype=bool)

    res = predictor.predict_raw_coordinates(
        raw_coords=raw_coords,
        hand_masks=hand_masks,
        pose_coords=pose_coords,
        pose_masks=pose_masks,
    )
    assert isinstance(res, PredictionResult)
    assert res.predicted_gloss in {"hello", "world"}

    # Test graceful fallback when pose is None (should auto zero-pad to 160)
    res_no_pose = predictor.predict_raw_coordinates(
        raw_coords=raw_coords,
        hand_masks=hand_masks,
        pose_coords=None,
    )
    assert isinstance(res_no_pose, PredictionResult)
    assert res_no_pose.predicted_gloss in {"hello", "world"}
    predictor.close()


def test_legacy_checkpoint_blocks_raw_coordinates_inference(tmp_path: Path) -> None:
    """Verify legacy checkpoints without resolved_config reject raw coordinates prediction."""
    model_pt, glosses_csv = _create_mock_model_and_glosses(
        tmp_path, input_size=143, save_resolved_config=False
    )
    predictor = ISLRPredictor(model_path=model_pt, glosses_csv=glosses_csv, device="cpu")
    assert predictor.is_legacy is True
    assert predictor.cfg is None

    # Precomputed features work
    dummy_feats = np.random.randn(32, 143).astype(np.float32)
    res = predictor.predict_features(dummy_feats)
    assert res.predicted_gloss in {"hello", "world"}

    # Raw coordinates raise ValueError with clear message
    raw_coords = np.random.randn(20, 2, 21, 3).astype(np.float32)
    hand_masks = np.ones((20, 2), dtype=bool)
    with pytest.raises(ValueError, match="Model checkpoint lacks resolved preprocessing configuration"):
        predictor.predict_raw_coordinates(raw_coords, hand_masks)
    predictor.close()


def test_checkpoint_resolved_config_drives_parameters(tmp_path: Path) -> None:
    """Verify checkpoint resolved_config determines sequence_length and feature_set."""
    custom_cfg = {
        "features": {"set": "hands"},
        "preprocess": {
            "sequence_length": 48,
            "pad_frames_around": 5,
            "max_gap_frames": 0,
            "min_valid_frames": 3,
            "dominant_hand": True,
            "normalization_epsilon": 0.0001,
        },
    }
    model_pt, glosses_csv = _create_mock_model_and_glosses(
        tmp_path, input_size=143, custom_cfg=custom_cfg
    )
    predictor = ISLRPredictor(model_path=model_pt, glosses_csv=glosses_csv, device="cpu")
    assert predictor.cfg is not None
    assert predictor.cfg.preprocess.sequence_length == 48
    assert predictor.cfg.preprocess.pad_frames_around == 5
    assert predictor.cfg.preprocess.max_gap_frames == 0

    # Prediction expecting 48 frames: passing 32 frames should raise ValueError
    wrong_feats = np.random.randn(32, 143).astype(np.float32)
    with pytest.raises(ValueError, match=r"Expected feature tensor of shape \(48, 143\)"):
        predictor.predict_features(wrong_feats)

    # Correct length 48 succeeds
    correct_feats = np.random.randn(48, 143).astype(np.float32)
    res = predictor.predict_features(correct_feats)
    assert res.predicted_gloss in {"hello", "world"}
    predictor.close()


def test_mismatched_model_input_size_raises(tmp_path: Path) -> None:
    """Verify mismatch between resolved_config feature_set and model input_size raises ValueError."""
    # Config claims "hands" (143), but model is 160 dims
    custom_cfg = {
        "features": {"set": "hands"},
        "preprocess": {
            "sequence_length": 32,
            "pad_frames_around": 2,
            "max_gap_frames": 5,
            "min_valid_frames": 5,
            "dominant_hand": True,
            "normalization_epsilon": 0.0001,
        },
    }
    model_pt, glosses_csv = _create_mock_model_and_glosses(
        tmp_path, input_size=160, custom_cfg=custom_cfg
    )
    with pytest.raises(ValueError, match="Checkpoint feature dimension mismatch"):
        ISLRPredictor(model_path=model_pt, glosses_csv=glosses_csv, device="cpu")


def test_model_pooling_and_lstm_variants() -> None:
    """Verify all pooling options and LSTM variant compute valid outputs."""
    x_143 = torch.randn(2, 32, 143)
    x_160 = torch.randn(2, 32, 160)

    for p in ["mean_max", "attention", "mean", "last"]:
        m_gru = create_model(input_size=160, num_classes=5, pooling=p, rnn_type="GRU")
        out_gru = m_gru(x_160)
        assert out_gru.shape == (2, 5)

        m_lstm = create_model(input_size=143, num_classes=5, pooling=p, rnn_type="LSTM")
        out_lstm = m_lstm(x_143)
        assert out_lstm.shape == (2, 5)

    # Test direct ASLLSTMClassifier class
    m_direct_lstm = ASLLSTMClassifier(input_size=160, num_classes=5, pooling="mean_max")
    assert m_direct_lstm(x_160).shape == (2, 5)


def test_predictor_missing_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Specified model_path does not exist"):
        ISLRPredictor(model_path=tmp_path / "nonexistent.pt")


def test_predictor_with_explicit_config_matching_and_mismatch(tmp_path: Path) -> None:
    """Verify that passing explicit cfg with matching identity succeeds, while mismatch fails loudly."""
    from src.utils.config import build_config_from_dict

    model_pt, glosses_csv = _create_mock_model_and_glosses(tmp_path, input_size=143)

    matching_cfg = build_config_from_dict({
        "features": {"set": "hands"},
        "preprocess": {
            "sequence_length": 32,
            "pad_frames_around": 2,
            "max_gap_frames": 5,
            "min_valid_frames": 5,
            "dominant_hand": True,
            "normalization_epsilon": 0.0001,
        },
    })

    # 1. Matching config succeeds
    predictor = ISLRPredictor(model_path=model_pt, glosses_csv=glosses_csv, device="cpu", cfg=matching_cfg)
    assert predictor.cfg is not None
    predictor.close()

    # 2. Mismatched config (e.g. sequence_length=48) raises ValueError
    mismatched_cfg = build_config_from_dict({
        "features": {"set": "hands"},
        "preprocess": {
            "sequence_length": 48,
            "pad_frames_around": 2,
            "max_gap_frames": 5,
            "min_valid_frames": 5,
            "dominant_hand": True,
            "normalization_epsilon": 0.0001,
        },
    })
    with pytest.raises(ValueError, match="Explicit config identity .* does not match checkpoint preprocess_identity"):
        ISLRPredictor(model_path=model_pt, glosses_csv=glosses_csv, device="cpu", cfg=mismatched_cfg)

