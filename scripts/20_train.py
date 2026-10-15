"""CLI script for ISLR model training, multi-seed aggregation, overfit check, and RF baseline.

Usage:
    python scripts/20_train.py --config configs/islr.yaml [--set key=value ...]
    python scripts/20_train.py --config configs/islr.yaml --overfit-check
    python scripts/20_train.py --config configs/islr.yaml --baseline-rf
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import ISRLandmarkDataset, create_dataloader
from src.features.normalization import get_feature_dim
from src.models.gru import ASLGRUClassifier
from src.training.evaluate import evaluate_test_set
from src.training.metrics import compute_macro_f1, compute_top_k_accuracy, plot_learning_curves
from src.training.train import train_gru, train_one_epoch
from src.utils.config import Config, load_config, save_config
from src.utils.env import detect_env, resolve_paths
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("islr.train_cli")


def get_git_commit_hash() -> str:
    """Retrieve current HEAD git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def select_device(device_str: str) -> torch.device:
    """Select compute device based on config string ('auto', 'cuda', 'cpu')."""
    dev = device_str.lower().strip()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def load_class_metadata(splits_dir: Path) -> Tuple[List[str], Dict[str, int]]:
    """Load ordered class names and label mapping from splits directory."""
    label_map_file = splits_dir / "label_map.json"
    classes_file = splits_dir / "classes.csv"

    if label_map_file.is_file():
        with open(label_map_file, mode="r", encoding="utf-8") as f:
            lmap = json.load(f)
        # Sort by integer label
        sorted_pairs = sorted(lmap.items(), key=lambda item: int(item[1]))
        class_names = [k for k, _ in sorted_pairs]
        return class_names, {k: int(v) for k, v in lmap.items()}

    if classes_file.is_file():
        class_names = []
        lmap = {}
        with open(classes_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = sorted(list(reader), key=lambda r: int(r["label"]))
            for r in rows:
                gloss = r["gloss"]
                lbl = int(r["label"])
                class_names.append(gloss)
                lmap[gloss] = lbl
        return class_names, lmap

    raise FileNotFoundError(
        f"Neither label_map.json nor classes.csv found in {splits_dir}. Run prepare_data stage=split first."
    )


def run_overfit_check(
    cfg: Config,
    train_csv: Path,
    device: torch.device,
    run_dir: Path,
    class_names: List[str],
) -> Dict[str, Any]:
    """Execute overfit check on 1 sample per gloss (dropout=0, aug=off, target acc >= 95%)."""
    logger.info("Executing Overfit Verification Check (1 sample per gloss, 0 dropout, no aug)...")
    run_dir.mkdir(parents=True, exist_ok=True)

    expected_dim = get_feature_dim(cfg.features.set)
    train_ds = ISRLandmarkDataset(
        split_csv=train_csv,
        is_train=True,
        augment=False,
        sequence_length=cfg.preprocess.sequence_length,
        feature_dim=expected_dim,
        strict=True,
        preload="ram",
    )

    # Pick exactly 1 sample per gloss
    chosen_indices: List[int] = []
    seen_labels: set[int] = set()
    for idx, sample in enumerate(train_ds.samples):
        lbl = sample["label"]
        if lbl not in seen_labels:
            seen_labels.add(lbl)
            chosen_indices.append(idx)
        if len(seen_labels) == len(class_names):
            break

    subset_ds = Subset(train_ds, chosen_indices)
    loader = DataLoader(subset_ds, batch_size=len(chosen_indices), shuffle=True)

    num_classes = len(class_names)
    model = ASLGRUClassifier(
        input_size=expected_dim,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_classes=num_classes,
        dropout=0.0,
        bidirectional=cfg.model.bidirectional,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0)

    # Train for 50 epochs
    epochs = 50
    final_loss = 1.0
    final_acc = 0.0
    for _ in range(epochs):
        final_loss, final_acc = train_one_epoch(
            model=model,
            dataloader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=1.0,
        )

    passed = final_acc >= 0.95
    report = {
        "mode": "overfit_check",
        "num_samples": len(chosen_indices),
        "num_classes": num_classes,
        "epochs": epochs,
        "final_train_loss": round(float(final_loss), 4),
        "final_train_acc": round(float(final_acc), 4),
        "passed": bool(passed),
        "git_commit_hash": get_git_commit_hash(),
    }

    report_path = run_dir / "overfit_check_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        f"Overfit Check Complete: Acc={final_acc:.2%}, Loss={final_loss:.4f}, Passed={passed}. "
        f"Report saved to {report_path}"
    )
    return report


def run_baseline_rf(
    cfg: Config,
    train_csv: Path,
    test_csv: Path,
    run_dir: Path,
    class_names: List[str],
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate Random Forest baseline on feature summary statistics (mean, std, min, max)."""
    logger.info("Executing Random Forest Tabular Baseline (mean, std, min, max across sequence)...")
    run_dir.mkdir(parents=True, exist_ok=True)

    expected_dim = get_feature_dim(cfg.features.set)

    def extract_rf_features(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        ds = ISRLandmarkDataset(
            split_csv=csv_path,
            is_train=False,
            augment=False,
            sequence_length=cfg.preprocess.sequence_length,
            feature_dim=expected_dim,
            strict=True,
            preload="ram",
        )
        x_list, y_list = [], []
        for idx in range(len(ds)):
            x_tensor, y_tensor = ds[idx]
            arr = x_tensor.numpy()  # (T, D)
            # Statistical aggregation across time
            feat_mean = np.mean(arr, axis=0)
            feat_std = np.std(arr, axis=0)
            feat_min = np.min(arr, axis=0)
            feat_max = np.max(arr, axis=0)
            stat_feat = np.concatenate([feat_mean, feat_std, feat_min, feat_max])
            x_list.append(stat_feat)
            y_list.append(y_tensor.item())
        return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    x_train, y_train = extract_rf_features(train_csv)
    x_test, y_test = extract_rf_features(test_csv)

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(x_train, y_train)

    probs = rf.predict_proba(x_test)
    preds = np.argmax(probs, axis=1)

    top1 = compute_top_k_accuracy(y_test, probs, k=1)
    top3 = compute_top_k_accuracy(y_test, probs, k=min(3, len(class_names)))
    f1 = compute_macro_f1(y_test, preds, num_classes=len(class_names))

    report = {
        "mode": "baseline_rf",
        "num_train_samples": len(x_train),
        "num_test_samples": len(x_test),
        "feature_dim_summary": int(x_train.shape[1]),
        "top1_accuracy": round(float(top1), 4),
        "top3_accuracy": round(float(top3), 4),
        "macro_f1": round(float(f1), 4),
        "seed": seed,
        "git_commit_hash": get_git_commit_hash(),
    }

    report_path = run_dir / "baseline_rf_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        f"RF Baseline Complete: Top-1={top1:.4f}, Top-3={top3:.4f}, Macro-F1={f1:.4f}. "
        f"Report saved to {report_path}"
    )
    return report


def train_single_seed(
    cfg: Config,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    class_names: List[str],
    seed: int,
    device: torch.device,
    run_dir: Path,
    commit_hash: str,
) -> Dict[str, Any]:
    """Train a single GRU model on given seed and output artifacts."""
    logger.info(f"--- Starting Training Run (Seed: {seed}) ---")
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    expected_dim = get_feature_dim(cfg.features.set)

    train_ds = ISRLandmarkDataset(
        split_csv=train_csv,
        is_train=True,
        augment=True,
        augmentation_config=cfg.augmentation,
        sequence_length=cfg.preprocess.sequence_length,
        feature_dim=expected_dim,
        strict=True,
        preload=cfg.training.preload,
        device=device,
    )
    val_ds = ISRLandmarkDataset(
        split_csv=val_csv,
        is_train=False,
        augment=False,
        sequence_length=cfg.preprocess.sequence_length,
        feature_dim=expected_dim,
        strict=True,
        preload=cfg.training.preload,
        device=device,
    )
    test_ds = ISRLandmarkDataset(
        split_csv=test_csv,
        is_train=False,
        augment=False,
        sequence_length=cfg.preprocess.sequence_length,
        feature_dim=expected_dim,
        strict=True,
        preload=cfg.training.preload,
        device=device,
    )

    train_loader = create_dataloader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        seed=seed,
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    test_loader = create_dataloader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    num_classes = len(class_names)
    model = ASLGRUClassifier(
        input_size=expected_dim,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_classes=num_classes,
        dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional,
    )

    ckpt_path = run_dir / "best_model.pt"

    history = train_gru(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.training.epochs,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        label_smoothing=cfg.training.label_smoothing,
        patience=cfg.training.patience,
        device=device,
        checkpoint_path=ckpt_path,
        verbose=True,
        grad_clip=cfg.training.grad_clip,
        scheduler_cfg=cfg.training.scheduler,
        amp=cfg.training.amp,
    )

    # Save history.json
    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Plot learning curves
    lc_path = run_dir / "learning_curves.png"
    plot_learning_curves(history, lc_path)

    # Evaluate on test set
    test_report_path = run_dir / "test_report.json"
    test_report = evaluate_test_set(
        model=model,
        test_loader=test_loader,
        test_csv=test_csv,
        class_names=class_names,
        device=device,
        plots_dir=run_dir,
        report_json_path=test_report_path,
        git_commit_hash=commit_hash,
    )

    logger.info(
        f"Seed {seed} Completed: Top-1={test_report['top1_accuracy']:.4f}, "
        f"Top-3={test_report['top3_accuracy']:.4f}, Macro-F1={test_report['macro_f1']:.4f}"
    )
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "test_report": test_report,
        "best_val_loss": history.get("best_val_loss"),
        "best_val_acc": history.get("best_val_acc"),
    }


def run_training_pipeline(
    config_path: str = "configs/islr.yaml",
    overrides: Optional[List[str]] = None,
    overfit_check: bool = False,
    baseline_rf: bool = False,
    seeds_override: Optional[List[int]] = None,
) -> int:
    """Execute training workflow based on validated config."""
    cfg = load_config(config_path=config_path, overrides=overrides)
    paths = resolve_paths(detect_env(), cfg, create_dirs=True)
    device = select_device(cfg.training.device)
    commit_hash = get_git_commit_hash()

    logger.info(f"ISLR Training Pipeline | Env: {paths.env_name} | Device: {device} | Git: {commit_hash[:8]}")

    splits_dir = paths.persistent_root / "data" / "splits"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    for req in [train_csv, val_csv, test_csv]:
        if not req.is_file():
            raise FileNotFoundError(f"Missing required split file {req}. Run prepare_data first.")

    class_names, _ = load_class_metadata(splits_dir)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_name = f"{timestamp_str}_{cfg.run.name}"
    run_dir = paths.persistent_root / cfg.run.output_dir / base_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    resolved_cfg_path = run_dir / "resolved_config.yaml"
    save_config(cfg, resolved_cfg_path)

    # Mode 1: Overfit Check
    if overfit_check:
        run_overfit_check(
            cfg=cfg,
            train_csv=train_csv,
            device=device,
            run_dir=run_dir,
            class_names=class_names,
        )
        return 0

    # Mode 2: Random Forest Baseline
    if baseline_rf:
        seed = cfg.training.seeds[0] if cfg.training.seeds else 42
        run_baseline_rf(
            cfg=cfg,
            train_csv=train_csv,
            test_csv=test_csv,
            run_dir=run_dir,
            class_names=class_names,
            seed=seed,
        )
        return 0

    # Mode 3: Multi-seed (or single-seed) GRU Training
    target_seeds = seeds_override if seeds_override is not None else cfg.training.seeds
    if not target_seeds:
        target_seeds = [42]

    results = []
    for s in target_seeds:
        seed_run_dir = run_dir if len(target_seeds) == 1 else run_dir / f"seed_{s}"
        res = train_single_seed(
            cfg=cfg,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            class_names=class_names,
            seed=s,
            device=device,
            run_dir=seed_run_dir,
            commit_hash=commit_hash,
        )
        results.append(res)

    # Aggregate multi-seed summary
    if len(results) > 1:
        top1_vals = [r["test_report"]["top1_accuracy"] for r in results]
        top3_vals = [r["test_report"]["top3_accuracy"] for r in results]
        f1_vals = [r["test_report"]["macro_f1"] for r in results]

        summary = {
            "num_seeds": len(results),
            "seeds": target_seeds,
            "git_commit_hash": commit_hash,
            "top1_accuracy_mean": round(float(np.mean(top1_vals)), 4),
            "top1_accuracy_std": round(float(np.std(top1_vals)), 4),
            "top3_accuracy_mean": round(float(np.mean(top3_vals)), 4),
            "top3_accuracy_std": round(float(np.std(top3_vals)), 4),
            "macro_f1_mean": round(float(np.mean(f1_vals)), 4),
            "macro_f1_std": round(float(np.std(f1_vals)), 4),
            "individual_runs": results,
        }
        sum_path = run_dir / "multi_seed_summary.json"
        sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(
            f"Multi-Seed Summary: Top-1={summary['top1_accuracy_mean']:.4f} ± {summary['top1_accuracy_std']:.4f} | "
            f"Macro-F1={summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}. Saved to {sum_path}"
        )

    return 0


def main() -> int:
    """CLI entrypoint for islr model training."""
    parser = argparse.ArgumentParser(description="ISLR Phase 2 Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/islr.yaml", help="Path to config yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Config overrides (key=value)")
    parser.add_argument("--overfit-check", action="store_true", help="Run 1 sample/gloss sanity overfit test")
    parser.add_argument("--baseline-rf", action="store_true", help="Run Random Forest tabular baseline")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Override list of training seeds")

    args = parser.parse_args()

    try:
        return run_training_pipeline(
            config_path=args.config,
            overrides=args.overrides,
            overfit_check=args.overfit_check,
            baseline_rf=args.baseline_rf,
            seeds_override=args.seeds,
        )
    except Exception as exc:
        logger.error(f"Training pipeline execution failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
