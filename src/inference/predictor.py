"""ISLR Inference Engine for video and landmark sequence prediction.

Supports:
- Direct video prediction (.mp4, .avi, .mov, .webm) via MediaPipe Holistic/Hands.
- Parquet landmark catalog prediction.
- Raw landmarks array prediction.
- Real-time frame list prediction.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.features.landmark_extractor import MediaPipeHandExtractor
from src.features.normalization import (
    compute_feature_vector_143,
    interpolate_bounded_gaps,
    resample_temporal,
    trim_idle_frames,
)
from src.models.gru import ASLGRUClassifier


@dataclass
class PredictionResult:
    """Structured container for prediction output."""
    predicted_gloss: str
    confidence: float
    top_k: List[Tuple[str, float]]
    probabilities: Dict[str, float]
    num_frames: int
    hand_detection_rate: float
    stats: Dict[str, Any]


class ISLRPredictor:
    """Full-pipeline inference predictor for Isolated Sign Language Recognition."""

    DEFAULT_MODEL_CANDIDATES = [
        Path("outputs/Kaggle_test/islr_phase2_artifacts_15.10.2026/outputs/runs/20261015_032408_gru_baseline/seed_42/best_model.pt"),
        Path("outputs/runs/20261015_032408_gru_baseline/seed_42/best_model.pt"),
    ]

    DEFAULT_GLOSS_CANDIDATES = [
        Path("outputs/Kaggle_test/islr_phase2_artifacts_15.10.2026/data/selection/glosses.csv"),
        Path("data/selection/glosses.csv"),
    ]

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        glosses_csv: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model_path = self._resolve_model_path(model_path)
        self.glosses_csv = self._resolve_glosses_csv(glosses_csv)

        # 1. Load gloss mapping
        self.label_to_gloss: Dict[int, str] = {}
        self.gloss_to_label: Dict[str, int] = {}
        self._load_gloss_mapping(self.glosses_csv)
        self.num_classes = len(self.label_to_gloss)

        # 2. Load model architecture & weights dynamically from checkpoint
        self._load_checkpoint(self.model_path)

        # 3. Extractor helper
        self.extractor = MediaPipeHandExtractor()

    def _resolve_model_path(self, path: Optional[Union[str, Path]]) -> Path:
        if path is not None:
            p = Path(path)
            if p.is_file():
                return p
            raise FileNotFoundError(f"Specified model_path does not exist: {path}")

        for cand in self.DEFAULT_MODEL_CANDIDATES:
            if cand.is_file():
                return cand

        # Fallback search in outputs/
        found = list(Path("outputs").rglob("best_model.pt"))
        if found:
            return found[0]

        raise FileNotFoundError(
            "Could not locate best_model.pt checkpoint. Please provide an explicit model_path."
        )

    def _resolve_glosses_csv(self, path: Optional[Union[str, Path]]) -> Path:
        if path is not None:
            p = Path(path)
            if p.is_file():
                return p
            raise FileNotFoundError(f"Specified glosses_csv does not exist: {path}")

        for cand in self.DEFAULT_GLOSS_CANDIDATES:
            if cand.is_file():
                return cand

        found = list(Path(".").rglob("glosses.csv"))
        if found:
            return found[0]

        raise FileNotFoundError(
            "Could not locate glosses.csv. Please provide an explicit glosses_csv."
        )

    def _load_gloss_mapping(self, csv_path: Path) -> None:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gloss = row["gloss"].strip()
                label = int(row["label"])
                self.label_to_gloss[label] = gloss
                self.gloss_to_label[gloss] = label

    def _load_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state

        # Infer dimensions from state_dict
        input_size = 143
        hidden_size = 128
        num_layers = 2
        bidirectional = False

        if "gru.weight_ih_l0" in state_dict:
            input_size = state_dict["gru.weight_ih_l0"].shape[1]
        if "gru.weight_hh_l0" in state_dict:
            hidden_size = state_dict["gru.weight_hh_l0"].shape[1]
        if "gru.weight_ih_l0_reverse" in state_dict:
            bidirectional = True

        layer_indices = [
            int(k.split("_l")[1].split("_")[0])
            for k in state_dict.keys()
            if "gru.weight_hh_l" in k and "_reverse" not in k
        ]
        if layer_indices:
            num_layers = max(layer_indices) + 1

        self.model = ASLGRUClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=self.num_classes,
            dropout=0.3,
            bidirectional=bidirectional,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_features(
        self,
        features: np.ndarray,
        stats: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> PredictionResult:
        """Predict from normalized feature matrix of shape (32, 143)."""
        tensor = torch.from_numpy(features).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top1_gloss = self.label_to_gloss.get(top1_idx, f"unknown_{top1_idx}")
        top1_conf = float(probs[top1_idx])

        top_k_results: List[Tuple[str, float]] = []
        for idx in sorted_indices[:top_k]:
            g = self.label_to_gloss.get(int(idx), f"unknown_{idx}")
            top_k_results.append((g, float(probs[idx])))

        all_probs: Dict[str, float] = {
            self.label_to_gloss.get(i, f"class_{i}"): float(probs[i])
            for i in range(len(probs))
        }

        stat_dict = stats or {}
        return PredictionResult(
            predicted_gloss=top1_gloss,
            confidence=top1_conf,
            top_k=top_k_results,
            probabilities=all_probs,
            num_frames=stat_dict.get("n_frames", 32),
            hand_detection_rate=stat_dict.get("det_rate", 1.0),
            stats=stat_dict,
        )

    def predict_raw_coordinates(
        self,
        raw_coords: np.ndarray,
        hand_masks: np.ndarray,
        stats: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> PredictionResult:
        """Process raw coordinates (T, 2, 21, 3) through normalization to prediction."""
        # 1. Trim idle frames
        trimmed_coords, trimmed_masks = trim_idle_frames(raw_coords, hand_masks, margin=2)
        if len(trimmed_coords) == 0:
            trimmed_coords, trimmed_masks = raw_coords, hand_masks

        # 2. Interpolate short gaps
        filled_coords, filled_masks = interpolate_bounded_gaps(
            trimmed_coords, trimmed_masks, max_gap_frames=3
        )

        # 3. Resample to standard length T=32
        resampled_coords, resampled_masks = resample_temporal(
            filled_coords, filled_masks, target_len=32
        )

        # 4. Compute 143-dim features
        features = compute_feature_vector_143(resampled_coords, resampled_masks)

        return self.predict_features(features=features, stats=stats, top_k=top_k)

    def predict_video(
        self,
        video_path: Union[str, Path],
        top_k: int = 3,
    ) -> PredictionResult:
        """Extract landmarks from video file and predict sign gloss."""
        raw_coords, hand_masks, stats = self.extractor.extract_from_video(video_path)
        if not stats.get("ok", False) or len(raw_coords) == 0:
            raise ValueError(f"Could not read video or extract frames from: {video_path}")
        return self.predict_raw_coordinates(raw_coords, hand_masks, stats=stats, top_k=top_k)

    def close(self) -> None:
        """Release extractor resources."""
        self.extractor.close()
