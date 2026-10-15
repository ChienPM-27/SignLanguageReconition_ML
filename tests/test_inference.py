"""Tests for inference predictor and realtime prediction functionality."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import pytest
import torch

from src.inference.predictor import ISLRPredictor, PredictionResult
from src.models.gru import ASLGRUClassifier


def _create_mock_model_and_glosses(tmp_path: Path) -> tuple[Path, Path]:
    glosses_csv = tmp_path / "glosses.csv"
    with open(glosses_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gloss", "label", "n_samples"])
        writer.writerow(["hello", "0", "10"])
        writer.writerow(["world", "1", "10"])

    model = ASLGRUClassifier(input_size=143, hidden_size=32, num_layers=1, num_classes=2, dropout=0.0)
    model_pt = tmp_path / "best_model.pt"
    torch.save(model.state_dict(), str(model_pt))
    return model_pt, glosses_csv


def test_predictor_initialization_and_prediction(tmp_path: Path) -> None:
    model_pt, glosses_csv = _create_mock_model_and_glosses(tmp_path)

    predictor = ISLRPredictor(model_path=model_pt, glosses_csv=glosses_csv, device="cpu")
    assert predictor.num_classes == 2
    assert predictor.label_to_gloss[0] == "hello"
    assert predictor.label_to_gloss[1] == "world"

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


def test_predictor_missing_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Specified model_path does not exist"):
        ISLRPredictor(model_path=tmp_path / "nonexistent.pt")
