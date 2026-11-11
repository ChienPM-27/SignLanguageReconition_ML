"""Tests for standalone model evaluation script (scripts/21_evaluate.py)."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import pytest
import torch

eval_module = importlib.import_module("scripts.21_evaluate")
run_standalone_evaluation = eval_module.run_standalone_evaluation
from src.features.schema import PREPROCESS_SCHEMA_VERSION, compute_preprocess_identity, get_feature_dim
from src.models.gru import ASLGRUClassifier
from src.utils.config import load_config
from tests.test_train import _setup_synthetic_train_environment


def test_standalone_evaluation_cli(tmp_path: Path) -> None:
    splits_dir, _, val_csv, test_csv, class_names = _setup_synthetic_train_environment(
        tmp_path, n_samples=6, n_classes=2
    )

    cfg = load_config()
    cfg.features.set = "hands"
    dim = get_feature_dim(cfg.features.set)
    identity, _ = compute_preprocess_identity(cfg)

    # Build and save a valid checkpoint
    model = ASLGRUClassifier(
        input_size=dim,
        hidden_size=16,
        num_layers=1,
        num_classes=len(class_names),
        dropout=0.0,
    )
    ckpt_path = tmp_path / "model_best.pt"
    ckpt_dict = {
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "preprocess_identity": identity,
        "resolved_config": {
            "features": {"set": "hands"},
            "preprocess": {"sequence_length": 32},
            "model": {
                "name": "GRU",
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "pooling": "mean_max",
                "bidirectional": False,
            },
        },
        "class_names": class_names,
        "label_map": {c: i for i, c in enumerate(class_names)},
    }
    torch.save(ckpt_dict, str(ckpt_path))

    # Run evaluation on test split
    out_dir = tmp_path / "eval_out"
    overrides = [f"env.local_root={tmp_path}"]
    ret = run_standalone_evaluation(
        checkpoint_path=ckpt_path,
        split="test",
        overrides=overrides,
        output_dir=out_dir,
        device_name="cpu",
    )
    assert ret == 0

    report_path = out_dir / "evaluation_test_report.json"
    assert report_path.is_file()
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert "top1_accuracy" in rep
    assert "macro_f1" in rep
    assert "per_participant_accuracy" in rep
    assert (out_dir / "confusion_matrix.png").is_file()


def test_standalone_evaluation_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        run_standalone_evaluation(checkpoint_path=tmp_path / "ghost.pt")
