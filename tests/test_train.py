"""Tests for ISLR model training pipeline, overfit check, and RF baseline (Step 5).

Offline synthetic fixtures test GRU model training, multi-seed aggregation,
overfit verification (acc >= 95%), Random Forest baseline, and CLI integration.
"""
from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

train_module = importlib.import_module("scripts.20_train")
run_baseline_rf = train_module.run_baseline_rf
run_overfit_check = train_module.run_overfit_check
run_training_pipeline = train_module.run_training_pipeline

from src.data.dataset import ISRLandmarkDataset
from src.models.gru import ASLGRUClassifier
from src.training.train import create_scheduler, evaluate_epoch, train_one_epoch
from src.utils.config import load_config


def _setup_synthetic_train_environment(tmp_path: Path, n_samples: int = 12, n_classes: int = 3):
    splits_dir = tmp_path / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = tmp_path / "cache" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    class_names = [f"sign_{c}" for c in range(n_classes)]
    label_map = {f"sign_{c}": c for c in range(n_classes)}
    with open(splits_dir / "label_map.json", mode="w", encoding="utf-8") as f:
        json.dump(label_map, f)

    with open(splits_dir / "classes.csv", mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "gloss"])
        for c in range(n_classes):
            writer.writerow([c, f"sign_{c}"])

    def create_split_file(filename: str, samples_count: int) -> Path:
        p = splits_dir / filename
        rows = []
        for i in range(samples_count):
            lbl = i % n_classes
            seq_id = f"{filename}_{i:03d}"
            npy_path = feat_dir / f"{seq_id}.npy"
            # Distinct pattern per class to allow fast overfit / learning
            arr = np.ones((32, 143), dtype=np.float32) * (lbl + 1.0) * 0.5
            np.save(str(npy_path), arr)
            rows.append([seq_id, f"signer_{i % 3}", f"sign_{lbl}", lbl, str(npy_path)])

        with open(p, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence_id", "participant_id", "gloss", "label", "path"])
            writer.writerows(rows)
        return p

    train_csv = create_split_file("train.csv", n_samples)
    val_csv = create_split_file("val.csv", max(3, n_samples // 3))
    test_csv = create_split_file("test.csv", max(3, n_samples // 3))

    return splits_dir, train_csv, val_csv, test_csv, class_names


def test_train_one_epoch_and_evaluate(tmp_path: Path) -> None:
    _, train_csv, val_csv, _, _ = _setup_synthetic_train_environment(tmp_path, n_samples=6, n_classes=2)

    device = torch.device("cpu")
    train_ds = ISRLandmarkDataset(split_csv=train_csv, sequence_length=32, feature_dim=143)
    val_ds = ISRLandmarkDataset(split_csv=val_csv, sequence_length=32, feature_dim=143)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    model = ASLGRUClassifier(input_size=143, hidden_size=16, num_layers=1, num_classes=2, dropout=0.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0

    val_loss, top1, f1, y_pred, y_probs, y_true = evaluate_epoch(model, val_loader, criterion, device, num_classes=2)
    assert val_loss >= 0.0
    assert 0.0 <= top1 <= 1.0
    assert len(y_true) == len(val_ds)
    assert len(y_pred) == len(val_ds)


def test_create_scheduler_types() -> None:
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    sched_plat, mode_plat = create_scheduler(optimizer, {"type": "plateau", "factor": 0.5, "patience": 2}, epochs=10)
    assert mode_plat == "metric"
    optimizer.step()
    sched_plat.step(0.5)

    sched_cos, mode_cos = create_scheduler(optimizer, {"type": "cosine"}, epochs=10)
    assert mode_cos == "epoch"
    optimizer.step()
    sched_cos.step()

    sched_step, mode_step = create_scheduler(optimizer, {"type": "step", "factor": 0.5, "patience": 2}, epochs=10)
    assert mode_step == "epoch"
    optimizer.step()
    sched_step.step()


def test_run_overfit_check(tmp_path: Path) -> None:
    _, train_csv, _, _, class_names = _setup_synthetic_train_environment(tmp_path, n_samples=6, n_classes=3)
    cfg = load_config(overrides=[f"env.local_root={tmp_path}"])
    device = torch.device("cpu")
    run_dir = tmp_path / "runs" / "test_overfit"

    report = run_overfit_check(
        cfg=cfg,
        train_csv=train_csv,
        device=device,
        run_dir=run_dir,
        class_names=class_names,
    )

    assert report["passed"] is True
    assert report["final_train_acc"] >= 0.95
    assert (run_dir / "overfit_check_report.json").is_file()


def test_run_baseline_rf(tmp_path: Path) -> None:
    _, train_csv, _, test_csv, class_names = _setup_synthetic_train_environment(tmp_path, n_samples=12, n_classes=3)
    cfg = load_config(overrides=[f"env.local_root={tmp_path}"])
    run_dir = tmp_path / "runs" / "test_rf"

    report = run_baseline_rf(
        cfg=cfg,
        train_csv=train_csv,
        test_csv=test_csv,
        run_dir=run_dir,
        class_names=class_names,
        seed=42,
    )

    assert report["top1_accuracy"] >= 0.0
    assert (run_dir / "baseline_rf_report.json").is_file()


def test_cli_train_pipeline_end_to_end(tmp_path: Path) -> None:
    _setup_synthetic_train_environment(tmp_path, n_samples=6, n_classes=2)

    overrides = [
        f"env.local_root={tmp_path}",
        "training.epochs=2",
        "training.batch_size=2",
        "model.hidden_size=16",
        "model.num_layers=1",
        "training.seeds=[42, 43]",
    ]

    ret = run_training_pipeline(overrides=overrides)
    assert ret == 0

    run_dirs = list((tmp_path / "outputs" / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "resolved_config.yaml").is_file()
    assert (run_dir / "multi_seed_summary.json").is_file()
    assert (run_dir / "seed_42" / "best_model.pt").is_file()
    assert (run_dir / "seed_42" / "test_report.json").is_file()
    assert (run_dir / "seed_42" / "confusion_matrix.png").is_file()
    assert (run_dir / "seed_42" / "learning_curves.png").is_file()
