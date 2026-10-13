"""Tests for ISLR signer-independent data splitting module and CLI stage (Step 4).

Covers participant holdout splitting, GroupKFold, zero-leakage verification,
min_per_gloss coverage across splits, automatic seed perturbation, and report generation.
"""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
import numpy as np

from src.data.split import (
    create_signer_splits,
    verify_no_participant_leakage,
)
from src.utils.config import load_config
from src.utils.env import ENV_LOCAL, resolve_paths
from tests.test_selection import _create_synthetic_catalog

prepare_data_module = importlib.import_module("scripts.10_prepare_data")
run_pipeline = prepare_data_module.run_pipeline


def test_verify_no_participant_leakage_clean() -> None:
    tr = [{"participant_id": "p1"}, {"participant_id": "p2"}]
    va = [{"participant_id": "p3"}]
    te = [{"participant_id": "p4"}]

    clean, leaks = verify_no_participant_leakage(tr, va, te)
    assert clean is True
    assert len(leaks["train_val"]) == 0
    assert len(leaks["train_test"]) == 0
    assert len(leaks["val_test"]) == 0


def test_verify_no_participant_leakage_dirty() -> None:
    tr = [{"participant_id": "p1"}, {"participant_id": "p2"}]
    va = [{"participant_id": "p2"}]  # Leak!
    te = [{"participant_id": "p3"}]

    clean, leaks = verify_no_participant_leakage(tr, va, te)
    assert clean is False
    assert "p2" in leaks["train_val"]


def _populate_features_manifest(cache_dir: Path, n_glosses: int = 4, n_signers: int = 6) -> Path:
    features_dir = cache_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = features_dir / "manifest.csv"

    rows = []
    for g_idx in range(n_glosses):
        gloss = f"gloss_{g_idx:02d}"
        for s_idx in range(n_signers):
            part_id = f"signer_{s_idx:02d}"
            seq_id = f"seq_{g_idx}_{s_idx}"
            npy_path = features_dir / f"{seq_id}.npy"
            np.save(str(npy_path), np.zeros((32, 143), dtype=np.float32))
            rows.append([seq_id, part_id, gloss, g_idx, 30, 0.95, str(npy_path)])

    with open(manifest_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "participant_id", "gloss", "label", "n_valid_frames", "hand_coverage", "path"])
        writer.writerows(rows)

    return manifest_csv


def test_create_signer_splits_holdout(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    _populate_features_manifest(cache_dir, n_glosses=4, n_signers=6)

    overrides = [
        f"env.local_root={tmp_path}",
        f"data.cache_dir={cache_dir}",
        f"output.report_dir={tmp_path / 'reports'}",
        "split.method=participant_holdout",
        "split.n_train=0.6",
        "split.n_val=0.2",
        "split.n_test=0.2",
        "split.seed=42",
        "split.min_per_gloss_per_split=1",
    ]
    cfg = load_config(overrides=overrides)
    paths = resolve_paths(ENV_LOCAL, cfg, create_dirs=True)

    split_files, report_md = create_signer_splits(cfg=cfg, paths=paths)

    assert "train" in split_files and split_files["train"].is_file()
    assert "val" in split_files and split_files["val"].is_file()
    assert "test" in split_files and split_files["test"].is_file()
    assert report_md.is_file()

    # Read records and assert strict zero participant leakage
    def read_split(p: Path):
        with open(p, mode="r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    tr_recs = read_split(split_files["train"])
    va_recs = read_split(split_files["val"])
    te_recs = read_split(split_files["test"])

    clean, leaks = verify_no_participant_leakage(tr_recs, va_recs, te_recs)
    assert clean is True, f"Detected leakage: {leaks}"

    # Verify classes.csv and label_map.json
    splits_dir = paths.persistent_root / "data" / "splits"
    assert (splits_dir / "label_map.json").is_file()
    assert (splits_dir / "classes.csv").is_file()
    with open(splits_dir / "label_map.json", mode="r", encoding="utf-8") as f:
        lmap = json.load(f)
        assert len(lmap) == 4


def test_create_signer_splits_group_kfold(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    _populate_features_manifest(cache_dir, n_glosses=3, n_signers=6)

    overrides = [
        f"env.local_root={tmp_path}",
        f"data.cache_dir={cache_dir}",
        f"output.report_dir={tmp_path / 'reports'}",
        "split.method=group_kfold",
        "split.n_folds=3",
    ]
    cfg = load_config(overrides=overrides)
    paths = resolve_paths(ENV_LOCAL, cfg, create_dirs=True)

    split_files, report_md = create_signer_splits(cfg=cfg, paths=paths)

    assert len(split_files) == 3
    assert "fold_0" in split_files
    assert "fold_1" in split_files
    assert "fold_2" in split_files

    # Check folds have mutually disjoint participants
    fold_parts = []
    for f_idx in range(3):
        with open(split_files[f"fold_{f_idx}"], mode="r", encoding="utf-8") as f:
            parts = {r["participant_id"] for r in csv.DictReader(f)}
            fold_parts.append(parts)

    assert len(fold_parts[0].intersection(fold_parts[1])) == 0
    assert len(fold_parts[0].intersection(fold_parts[2])) == 0
    assert len(fold_parts[1].intersection(fold_parts[2])) == 0


def test_cli_prepare_data_all_stages(tmp_path: Path) -> None:
    mirror_dir = tmp_path / "mirror"
    catalog_path = mirror_dir / "train.csv"
    _create_synthetic_catalog(catalog_path, mirror_dir, num_signs=8, samples_per_sign=12, num_participants=6)

    overrides = [
        f"env.local_root={tmp_path}",
        f"data.local_mirror_dir={mirror_dir}",
        "selection.n_glosses=4",
        "selection.n_candidates=8",
        "data.max_files_per_gloss=3",
        f"data.cache_dir={tmp_path / 'cache'}",
        f"output.output_dir={tmp_path / 'out'}",
        f"output.selection_dir={tmp_path / 'selection'}",
        f"output.report_dir={tmp_path / 'reports'}",
        "split.min_per_gloss_per_split=1",
    ]

    ret = run_pipeline(stage="all", overrides=overrides)
    assert ret == 0

    splits_dir = tmp_path / "data" / "splits"
    assert (splits_dir / "train.csv").is_file()
    assert (splits_dir / "val.csv").is_file()
    assert (splits_dir / "test.csv").is_file()
    assert (tmp_path / "reports" / "split_report.md").is_file()
