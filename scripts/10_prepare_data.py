#!/usr/bin/env python3
"""ISLR Data Preparation CLI: Selection, Download, Preprocess, Split.

Executes stages sequentially or individually via `--stage [selection|download|preprocess|split|all]`.
Supports arbitrary depth configuration overrides via `--set key=value`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.download import download_target_gloss_data
from src.data.preprocess import run_preprocessing
from src.data.selection import select_20_glosses
from src.data.split import create_signer_splits
from src.utils.config import DEFAULT_CONFIG_PATH, load_config, save_config
from src.utils.env import detect_env, resolve_paths
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("islr.prepare_data")

ALLOWED_STAGES = ("selection", "download", "preprocess", "split", "all")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISLR Data Preparation Pipeline (Selection, Download, Preprocess, Split)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=ALLOWED_STAGES,
        help="Stage to execute (default: all).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Config overrides in key=value format (e.g. --set selection.n_glosses=10).",
    )
    return parser.parse_args(args)


def run_pipeline(
    config_path: str = DEFAULT_CONFIG_PATH,
    stage: str = "all",
    overrides: list[str] | None = None,
) -> int:
    cfg = load_config(config_path=config_path, overrides=overrides)
    set_seed(cfg.env.seed)

    env_name = detect_env() if cfg.env.name == "auto" else cfg.env.name
    paths = resolve_paths(env_name=env_name, cfg=cfg, create_dirs=True)

    logger.info(f"Initialized data pipeline in environment '{env_name}' (stage: {stage})")

    # Stage: Selection
    if stage in ("selection", "all"):
        logger.info("Executing Stage: Selection (20 glosses)")
        selected_glosses, glosses_csv, report_path = select_20_glosses(cfg=cfg, paths=paths)
        logger.info(f"Selection complete: {len(selected_glosses)} glosses -> {glosses_csv}")

    # Stage: Download
    if stage in ("download", "all"):
        logger.info("Executing Stage: Download (Target gloss parquets)")
        download_manifest, download_records = download_target_gloss_data(cfg=cfg, paths=paths)
        logger.info(f"Download complete: {len(download_records)} records -> {download_manifest}")

    # Stage: Preprocess
    if stage in ("preprocess", "all"):
        logger.info("Executing Stage: Preprocess (Parquet -> .npy features)")
        feat_manifest, excluded_csv, prep_report = run_preprocessing(cfg=cfg, paths=paths)
        logger.info(f"Preprocessing complete: manifest -> {feat_manifest}, report -> {prep_report}")

    # Stage: Split
    if stage in ("split", "all"):
        logger.info("Executing Stage: Split (Signer-independent holdout/kfold)")
        split_files, split_report = create_signer_splits(cfg=cfg, paths=paths)
        logger.info(f"Split complete: files -> {list(split_files.keys())}, report -> {split_report}")

    # Save resolved config
    if cfg.output.save_resolved_config:
        res_cfg_path = paths.output_dir / "resolved_config.yaml"
        save_config(cfg, res_cfg_path)
        logger.info(f"Saved resolved configuration to {res_cfg_path}")

    return 0


def main() -> None:
    args = parse_args()
    exit_code = run_pipeline(
        config_path=args.config,
        stage=args.stage,
        overrides=args.overrides,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
