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
from src.training.train import (
    create_scheduler,
    evaluate_epoch,
    load_checkpoint,
    save_checkpoint,
    train_gru,
    train_one_epoch,
)
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
    assert (run_dir / "seed_42" / "train_metrics.csv").is_file()
    assert (run_dir / "seed_42" / "test_report.json").is_file()
    assert (run_dir / "seed_42" / "confusion_matrix.png").is_file()
    assert (run_dir / "seed_42" / "learning_curves.png").is_file()


def test_device_loader_train_one_epoch_smoke(tmp_path: Path) -> None:
    _, train_csv, val_csv, _, _ = _setup_synthetic_train_environment(tmp_path, n_samples=6, n_classes=2)

    device = torch.device("cpu")
    train_ds = ISRLandmarkDataset(
        split_csv=train_csv,
        sequence_length=32,
        feature_dim=143,
        preload="device",
        device=device,
    )
    val_ds = ISRLandmarkDataset(
        split_csv=val_csv,
        sequence_length=32,
        feature_dim=143,
        preload="device",
        device=device,
    )

    from src.data.dataset import create_dataloader
    train_loader = create_dataloader(train_ds, batch_size=2, shuffle=True, seed=42)
    val_loader = create_dataloader(val_ds, batch_size=2, shuffle=False)

    model = ASLGRUClassifier(input_size=143, hidden_size=16, num_layers=1, num_classes=2, dropout=0.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0

    val_loss, top1, f1, y_pred, y_probs, y_true = evaluate_epoch(
        model, val_loader, criterion, device, num_classes=2
    )
    assert val_loss >= 0.0
    assert 0.0 <= top1 <= 1.0
    assert len(y_true) == len(val_ds)


def test_train_metrics_csv_streaming_and_flush(tmp_path: Path) -> None:
    _, train_csv, val_csv, _, _ = _setup_synthetic_train_environment(tmp_path, n_samples=6, n_classes=2)
    device = torch.device("cpu")
    train_ds = ISRLandmarkDataset(split_csv=train_csv, sequence_length=32, feature_dim=143)
    val_ds = ISRLandmarkDataset(split_csv=val_csv, sequence_length=32, feature_dim=143)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    model = ASLGRUClassifier(input_size=143, hidden_size=16, num_layers=1, num_classes=2, dropout=0.0)
    csv_path = tmp_path / "custom_metrics.csv"
    ckpt_path = tmp_path / "custom_best.pt"

    history = train_gru(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        device=device,
        checkpoint_path=ckpt_path,
        metrics_csv_path=csv_path,
        verbose=False,
    )
    assert len(history["train_loss"]) == 3
    assert csv_path.is_file()
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert len(reader) == 4  # Header + 3 epochs
    assert reader[0] == ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr"]
    assert reader[1][0] == "1"
    assert reader[3][0] == "3"
    assert ckpt_path.is_file()


def test_save_load_checkpoint_and_resume(tmp_path: Path) -> None:
    _, train_csv, val_csv, _, _ = _setup_synthetic_train_environment(tmp_path, n_samples=6, n_classes=2)
    device = torch.device("cpu")
    train_ds = ISRLandmarkDataset(split_csv=train_csv, sequence_length=32, feature_dim=143)
    val_ds = ISRLandmarkDataset(split_csv=val_csv, sequence_length=32, feature_dim=143)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    model = ASLGRUClassifier(input_size=143, hidden_size=16, num_layers=1, num_classes=2, dropout=0.0)
    ckpt_path = tmp_path / "stage1_best.pt"
    csv_path = tmp_path / "stage1_metrics.csv"

    # Train for 2 epochs
    h1 = train_gru(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        device=device,
        checkpoint_path=ckpt_path,
        metrics_csv_path=csv_path,
        verbose=False,
    )
    assert len(h1["train_loss"]) == 2
    assert ckpt_path.is_file()

    # Explicit direct save_checkpoint test
    direct_ckpt_path = tmp_path / "direct_save.pt"
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    saved_p = save_checkpoint(
        checkpoint_path=direct_ckpt_path,
        model=model,
        optimizer=opt,
        scheduler=None,
        scaler=None,
        epoch=2,
        best_val_loss=0.5,
        best_val_acc=0.8,
        patience_counter=0,
        early_stopping_best=0.5,
        history=h1,
    )
    assert saved_p.is_file()
    direct_loaded = load_checkpoint(direct_ckpt_path)
    assert direct_loaded["epoch"] == 2
    assert direct_loaded["best_val_loss"] == 0.5

    # Load and verify stateful checkpoint structure
    ckpt_dict = load_checkpoint(ckpt_path)
    assert "epoch" in ckpt_dict
    assert "model_state_dict" in ckpt_dict
    assert "optimizer_state_dict" in ckpt_dict
    assert "best_val_loss" in ckpt_dict

    # Resume training for 2 more epochs (target epochs=4)
    model2 = ASLGRUClassifier(input_size=143, hidden_size=16, num_layers=1, num_classes=2, dropout=0.0)
    h2 = train_gru(
        model=model2,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=4,
        device=device,
        checkpoint_path=ckpt_path,
        resume_checkpoint_path=ckpt_path,
        metrics_csv_path=csv_path,
        verbose=False,
    )
    assert len(h2["train_loss"]) == 4

    # Check metrics CSV has 4 epochs recorded
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert len(reader) == 5  # header + 4 epochs
    assert reader[4][0] == "4"


def test_early_stopping_min_delta_and_min_epochs(tmp_path: Path) -> None:
    _, train_csv, val_csv, _, _ = _setup_synthetic_train_environment(tmp_path, n_samples=6, n_classes=2)
    device = torch.device("cpu")
    train_ds = ISRLandmarkDataset(split_csv=train_csv, sequence_length=32, feature_dim=143)
    val_ds = ISRLandmarkDataset(split_csv=val_csv, sequence_length=32, feature_dim=143)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    model = ASLGRUClassifier(input_size=143, hidden_size=16, num_layers=1, num_classes=2, dropout=0.0)

    # Test early_stopping_min_epochs: patience=1, min_epochs=4, should not stop before epoch 4
    h = train_gru(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=8,
        patience=1,
        early_stopping_min_delta=100.0,  # virtually impossible to improve by 100
        early_stopping_min_epochs=4,
        device=device,
        verbose=False,
    )
    # Should run at least 4 epochs because min_epochs=4
    assert len(h["train_loss"]) >= 4

