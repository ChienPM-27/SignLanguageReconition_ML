"""Standalone evaluation CLI for ISLR models on explicit splits (test, val, train).

Usage:
    python scripts/21_evaluate.py --checkpoint outputs/runs/<run_id>/best_model.pt --split test
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.data.dataset import ISRLandmarkDataset, create_dataloader
from src.features.schema import compute_preprocess_identity, get_feature_dim
from src.models.gru import create_model
from src.training.evaluate import evaluate_test_set
from src.utils.checkpoint import validate_checkpoint_metadata
from src.utils.config import apply_overrides, build_config_from_dict
from src.utils.env import detect_env, resolve_paths
from src.utils.logging import get_logger

logger = get_logger("islr.evaluate_cli")


def get_git_commit_hash() -> str:
    """Safely obtain current git commit hash."""
    try:
        import subprocess
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def select_device(device_str: str) -> torch.device:
    """Select compute device ('auto', 'cuda', 'cpu')."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def run_standalone_evaluation(
    checkpoint_path: str | Path,
    split: str = "test",
    config_path: str = "configs/islr.yaml",
    overrides: Optional[List[str]] = None,
    output_dir: Optional[str | Path] = None,
    device_name: str = "auto",
) -> int:
    """Execute evaluation for a selected checkpoint against a split CSV."""
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_p}")

    device = select_device(device_name)
    commit_hash = get_git_commit_hash()

    # Load checkpoint payload
    ckpt_payload = torch.load(str(ckpt_p), map_location=device, weights_only=False)
    if not isinstance(ckpt_payload, dict) or "model_state_dict" not in ckpt_payload:
        raise ValueError(f"Invalid checkpoint at {ckpt_p}: missing 'model_state_dict'.")

    # 1. Strictly validate checkpoint metadata, classes, and label map
    saved_cfg, classes_list, ckpt_label_map = validate_checkpoint_metadata(
        ckpt_payload=ckpt_payload,
        checkpoint_path=ckpt_p,
        allow_legacy=False,
    )

    # 2. Guard against model semantics overrides during standalone evaluation
    FORBIDDEN_EVAL_OVERRIDE_PREFIXES = ("model.", "features.", "preprocess.", "data.gloss_classes")
    if overrides:
        for ovr in overrides:
            key = ovr.split("=")[0].strip()
            if any(key.startswith(p) for p in FORBIDDEN_EVAL_OVERRIDE_PREFIXES):
                raise ValueError(
                    f"Cannot override '{key}' during standalone evaluation: "
                    "altering model architecture or preprocessing semantics of a trained checkpoint is strictly prohibited."
                )
        cfg_dict = copy.deepcopy(saved_cfg.to_dict())
        apply_overrides(cfg_dict, overrides)
        cfg = build_config_from_dict(cfg_dict)
    else:
        cfg = saved_cfg

    preprocess_id = ckpt_payload["preprocess_identity"]
    recomputed_id, _ = compute_preprocess_identity(cfg)
    if preprocess_id != recomputed_id:
        raise ValueError(
            f"Checkpoint preprocess_identity '{preprocess_id}' does not match recomputed config identity '{recomputed_id}'."
        )

    paths = resolve_paths(detect_env(), cfg, create_dirs=True)

    # Resolve split CSV
    splits_dir = paths.persistent_root / "data" / "splits"
    split_csv = splits_dir / f"{split}.csv"
    if not split_csv.is_file():
        raise FileNotFoundError(f"Split file not found: {split_csv}")

    num_classes = len(classes_list)
    expected_dim = get_feature_dim(cfg)

    # Instantiate model from validated configuration
    model = create_model(
        input_size=expected_dim,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_classes=num_classes,
        dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional,
        pooling=cfg.model.pooling,
        rnn_type=cfg.model.name,
        attention_heads=cfg.model.attention_heads,
        feedforward_size=cfg.model.feedforward_size,
        region_fusion=cfg.model.region_fusion,
        regions=cfg.model.regions,
        region_weights=cfg.model.region_weights,
        region_normalization_epsilon=cfg.model.region_normalization_epsilon,
    )
    model.load_state_dict(ckpt_payload["model_state_dict"])
    model.to(device)
    model.eval()

    # Load dataset
    dataset = ISRLandmarkDataset(
        split_csv=split_csv,
        is_train=False,
        augment=False,
        sequence_length=cfg.preprocess.sequence_length,
        feature_dim=expected_dim,
        strict=True,
        preload="none",
        expected_preprocess_identity=preprocess_id,
    )

    data_loader = create_dataloader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Target output directory
    out_dir = Path(output_dir) if output_dir is not None else ckpt_p.parent / f"eval_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = out_dir / f"evaluation_{split}_report.json"
    report = evaluate_test_set(
        model=model,
        test_loader=data_loader,
        test_csv=split_csv,
        class_names=classes_list,
        device=device,
        plots_dir=out_dir,
        report_json_path=report_json_path,
        git_commit_hash=commit_hash,
        preprocess_identity=preprocess_id,
        split_identity=split,
    )

    logger.info(
        f"Evaluation Complete on '{split}' ({report['num_test_samples']} samples): "
        f"Top-1={report['top1_accuracy']:.4f}, Top-3={report['top3_accuracy']:.4f}, "
        f"Macro-F1={report['macro_f1']:.4f}, Loss={report['loss']:.4f}"
    )
    logger.info(f"Report saved to {report_json_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ISLR Standalone Split Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Split to evaluate")
    parser.add_argument("--config", type=str, default="configs/islr.yaml", help="Path to config yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Config overrides (key=value)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports and plots")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")

    args = parser.parse_args()
    try:
        return run_standalone_evaluation(
            checkpoint_path=args.checkpoint,
            split=args.split,
            config_path=args.config,
            overrides=args.overrides,
            output_dir=args.output_dir,
            device_name=args.device,
        )
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
