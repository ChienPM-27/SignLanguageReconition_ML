"""ISLR Inference Engine for video and landmark sequence prediction.

Supports:
- Direct video prediction (.mp4, .avi, .mov, .webm) via MediaPipe Hands & Pose.
- Parquet landmark catalog prediction.
- Raw landmarks array prediction (both hands 143 dims and hands_pose 160 dims).
- Dynamic pooling (mean_max, attention, last) and model type (GRU, LSTM).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.features.landmark_extractor import MediaPipeHandExtractor
from src.features.pipeline import prepare_landmark_sequence
from src.features.schema import PREPROCESS_SCHEMA_VERSION, get_feature_dim
from src.models.gru import ASLGRUClassifier
from src.utils.config import Config, build_config_from_dict


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
        Path("outputs/Kaggle_test/islr_phaseA_generalization_15.10.2026/outputs/runs/20261015_104434_exp_r2_aug/seed_42/best_model.pt"),
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
        cfg: Optional[Config] = None,
    ):
        self.device = torch.device(device)
        self.model_path = self._resolve_model_path(model_path)
        self.glosses_csv = self._resolve_glosses_csv(glosses_csv)

        # 1. Load gloss mapping
        self.label_to_gloss: Dict[int, str] = {}
        self.gloss_to_label: Dict[str, int] = {}
        self._load_gloss_mapping(self.glosses_csv)
        self.num_classes = len(self.label_to_gloss)

        # 2. State attributes
        self.input_size = 143
        self.pooling = "last"
        self.rnn_type = "GRU"
        self.cfg: Optional[Config] = None
        self.is_legacy: bool = False
        self.schema_version: Optional[int] = None
        self.preprocess_identity: Optional[str] = None

        # 3. Load model architecture & weights dynamically from checkpoint
        self._load_checkpoint(self.model_path)

        if cfg is not None:
            if self.preprocess_identity is not None:
                from src.features.schema import compute_preprocess_identity

                passed_id, _ = compute_preprocess_identity(cfg)
                if passed_id != self.preprocess_identity:
                    raise ValueError(
                        f"Explicit config identity '{passed_id}' does not match checkpoint "
                        f"preprocess_identity '{self.preprocess_identity}'."
                    )
            self.cfg = cfg

        # 4. Extractor helper
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
        state = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
            raw_ckpt = state
        else:
            state_dict = state
            raw_ckpt = {}

        # Preprocessing metadata & resolved config
        resolved_cfg_dict = raw_ckpt.get("resolved_config")
        schema_version = raw_ckpt.get("schema_version")
        preprocess_identity = raw_ckpt.get("preprocess_identity")

        if resolved_cfg_dict is not None and isinstance(resolved_cfg_dict, dict):
            self.cfg = build_config_from_dict(resolved_cfg_dict)
            self.schema_version = schema_version if schema_version is not None else PREPROCESS_SCHEMA_VERSION
            self.preprocess_identity = preprocess_identity
            self.is_legacy = False
        else:
            self.cfg = None
            self.schema_version = None
            self.preprocess_identity = None
            self.is_legacy = True

        # Infer dimensions from state_dict
        input_size = 143
        hidden_size = 128
        num_layers = 2
        bidirectional = False
        rnn_type = "GRU"

        rnn_prefix = "gru"
        if "lstm.weight_ih_l0" in state_dict:
            rnn_prefix = "lstm"
            rnn_type = "LSTM"
        elif "gru.weight_ih_l0" in state_dict:
            rnn_prefix = "gru"
            rnn_type = "GRU"
        elif "rnn.weight_ih_l0" in state_dict:
            rnn_prefix = "rnn"

        weight_ih = state_dict.get(f"{rnn_prefix}.weight_ih_l0")
        if weight_ih is not None:
            input_size = weight_ih.shape[1]

        weight_hh = state_dict.get(f"{rnn_prefix}.weight_hh_l0")
        if weight_hh is not None:
            hidden_size = weight_hh.shape[1]

        if f"{rnn_prefix}.weight_ih_l0_reverse" in state_dict:
            bidirectional = True

        layer_indices = [
            int(k.split("_l")[1].split("_")[0])
            for k in state_dict.keys()
            if f"{rnn_prefix}.weight_hh_l" in k and "_reverse" not in k
        ]
        if layer_indices:
            num_layers = max(layer_indices) + 1

        direction_factor = 2 if bidirectional else 1
        fc_in = state_dict["fc.weight"].shape[1] if "fc.weight" in state_dict else hidden_size * direction_factor
        if fc_in == hidden_size * direction_factor * 2:
            pooling = "mean_max"
        elif "attn_pool.attn.0.weight" in state_dict:
            pooling = "attention"
        else:
            pooling = "last"

        self.input_size = input_size
        self.pooling = pooling
        self.rnn_type = rnn_type

        # Validate checkpoint configuration dimensions if resolved_config is present
        if self.cfg is not None:
            expected_dim = get_feature_dim(self.cfg.features.set)
            if expected_dim != self.input_size:
                raise ValueError(
                    f"Checkpoint feature dimension mismatch: resolved_config features.set='{self.cfg.features.set}' "
                    f"expects {expected_dim} dims, but model input_size is {self.input_size}."
                )

        self.model = ASLGRUClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=self.num_classes,
            dropout=0.3,
            bidirectional=bidirectional,
            pooling=pooling,
            rnn_type=rnn_type,
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
        """Predict from normalized feature matrix of shape (sequence_length, input_size)."""
        if features.ndim != 2:
            raise ValueError(f"Features must be 2D (sequence_length, feature_dim), got shape {features.shape}")

        if self.cfg is not None:
            expected_seq_len = self.cfg.preprocess.sequence_length
            if features.shape != (expected_seq_len, self.input_size):
                raise ValueError(
                    f"Expected feature tensor of shape ({expected_seq_len}, {self.input_size}), got {features.shape}"
                )
        else:
            if features.shape[1] != self.input_size:
                raise ValueError(
                    f"Expected feature tensor with width {self.input_size}, got {features.shape[1]}"
                )

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
            num_frames=stat_dict.get("n_frames", features.shape[0]),
            hand_detection_rate=stat_dict.get("det_rate", 1.0),
            stats=stat_dict,
        )

    def predict_raw_coordinates(
        self,
        raw_coords: np.ndarray,
        hand_masks: np.ndarray,
        pose_coords: Optional[np.ndarray] = None,
        pose_masks: Optional[np.ndarray] = None,
        stats: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> PredictionResult:
        """Process raw coordinates through canonical preprocessing to prediction."""
        if self.cfg is None:
            raise ValueError(
                "Model checkpoint lacks resolved preprocessing configuration (legacy checkpoint). "
                "Raw coordinate inference requires retraining with corrected preprocessing schema version 2. "
                "Only precomputed feature inference (predict_features) is supported for legacy checkpoints."
            )

        if pose_coords is None and stats and "pose_coords" in stats:
            pose_coords = stats["pose_coords"]
            pose_masks = stats.get("pose_masks")

        result = prepare_landmark_sequence(
            raw_coords=raw_coords,
            hand_masks=hand_masks,
            pose_coords=pose_coords,
            pose_masks=pose_masks,
            cfg=self.cfg,
        )

        if not result.is_usable:
            raise ValueError(f"Unusable sequence: {result.rejection_reason}")

        return self.predict_features(features=result.features, stats=stats, top_k=top_k)

    def predict_video(
        self,
        video_path: Union[str, Path],
        top_k: int = 3,
    ) -> PredictionResult:
        """Extract landmarks from video file and predict sign gloss."""
        raw_coords, hand_masks, stats = self.extractor.extract_from_video(video_path)
        if not stats.get("ok", False) or len(raw_coords) == 0:
            raise ValueError(f"Could not read video or extract frames from: {video_path}")
        pose_coords = stats.get("pose_coords")
        pose_masks = stats.get("pose_masks")
        return self.predict_raw_coordinates(
            raw_coords=raw_coords,
            hand_masks=hand_masks,
            pose_coords=pose_coords,
            pose_masks=pose_masks,
            stats=stats,
            top_k=top_k,
        )

    def close(self) -> None:
        """Release extractor resources."""
        self.extractor.close()
