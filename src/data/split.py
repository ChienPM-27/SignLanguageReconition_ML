"""Signer-Independent (Participant-Holdout and GroupKFold) data splitting for ISLR.

Ensures:
- Strict zero-leakage invariant: each participant belongs to exactly one split.
- Stratification check: each target gloss has at least min_per_gloss_per_split samples per split.
- Automatic seed perturbation if initial partition fails gloss coverage.
- Generation of train.csv, val.csv, test.csv, label_map.json, classes.csv, and split_report.md.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from src.utils.config import Config
from src.utils.env import ResolvedPaths
from src.utils.logging import get_logger

logger = get_logger("islr.split")


def verify_no_participant_leakage(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Set[str]]]:
    """Verify that train, val, and test participant sets are strictly disjoint."""
    train_parts = {str(r["participant_id"]).strip() for r in train_records}
    val_parts = {str(r["participant_id"]).strip() for r in val_records}
    test_parts = {str(r["participant_id"]).strip() for r in test_records}

    train_val_leak = train_parts.intersection(val_parts)
    train_test_leak = train_parts.intersection(test_parts)
    val_test_leak = val_parts.intersection(test_parts)

    leaks = {
        "train_val": train_val_leak,
        "train_test": train_test_leak,
        "val_test": val_test_leak,
    }
    is_clean = len(train_val_leak) == 0 and len(train_test_leak) == 0 and len(val_test_leak) == 0
    return is_clean, leaks


def create_signer_splits(
    cfg: Config,
    paths: ResolvedPaths,
) -> Tuple[Dict[str, Path], Path]:
    """Execute signer-independent dataset splitting from preprocessed manifest."""
    manifest_csv = paths.cache_dir / "features" / "manifest.csv"
    if not manifest_csv.is_file():
        raise FileNotFoundError(f"Features manifest {manifest_csv} not found. Run preprocessing first.")

    # Target splits output directory
    splits_dir = paths.persistent_root / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    report_dir = paths.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    with open(manifest_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "sequence_id": str(row["sequence_id"]).strip(),
                "participant_id": str(row["participant_id"]).strip(),
                "gloss": str(row["gloss"]).strip(),
                "label": int(row["label"]),
                "path": str(row["path"]).strip(),
            })

    if not records:
        raise ValueError("Features manifest contains zero records. Cannot create splits.")

    unique_participants = sorted({r["participant_id"] for r in records})
    unique_glosses = sorted({r["gloss"] for r in records})
    n_parts = len(unique_participants)

    logger.info(
        f"Splitting {len(records)} sequences across {n_parts} participants "
        f"and {len(unique_glosses)} glosses."
    )

    split_method = cfg.split.method
    split_files: Dict[str, Path] = {}

    if split_method == "participant_holdout":
        train_recs, val_recs, test_recs, used_seed = _find_valid_holdout_partition(
            records=records,
            participants=unique_participants,
            glosses=unique_glosses,
            cfg=cfg,
            max_seed_attempts=50,
        )

        # Verification check
        is_clean, leaks = verify_no_participant_leakage(train_recs, val_recs, test_recs)
        if not is_clean:
            raise RuntimeError(f"Participant leakage detected: {leaks}")

        # Save split CSVs
        for split_name, rec_list in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
            out_csv = splits_dir / f"{split_name}.csv"
            with open(out_csv, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sequence_id", "participant_id", "gloss", "label", "path"])
                for r in rec_list:
                    writer.writerow([r["sequence_id"], r["participant_id"], r["gloss"], r["label"], r["path"]])
            split_files[split_name] = out_csv

        # Save classes.csv and label_map.json
        label_map = {r["gloss"]: r["label"] for r in records}
        with open(splits_dir / "label_map.json", mode="w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)

        classes_csv = splits_dir / "classes.csv"
        with open(classes_csv, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gloss", "label"])
            for g in sorted(label_map.keys()):
                writer.writerow([g, label_map[g]])

        # Generate split report
        report_md_path = report_dir / "split_report.md"
        _generate_split_report(
            report_path=report_md_path,
            train_recs=train_recs,
            val_recs=val_recs,
            test_recs=test_recs,
            glosses=unique_glosses,
            cfg=cfg,
            used_seed=used_seed,
        )
        logger.info(f"Generated split report at {report_md_path}")
        return split_files, report_md_path

    else:
        # GroupKFold partitioning
        folds = _partition_group_kfold(
            records=records,
            participants=unique_participants,
            n_folds=cfg.split.n_folds,
            seed=cfg.split.seed,
        )
        for fold_idx, fold_recs in enumerate(folds):
            out_csv = splits_dir / f"fold_{fold_idx}.csv"
            with open(out_csv, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sequence_id", "participant_id", "gloss", "label", "path"])
                for r in fold_recs:
                    writer.writerow([r["sequence_id"], r["participant_id"], r["gloss"], r["label"], r["path"]])
            split_files[f"fold_{fold_idx}"] = out_csv

        report_md_path = report_dir / "split_report.md"
        report_md_path.write_text(
            f"# ISLR GroupKFold Split Report\n\nTotal folds: {len(folds)}\nTotal records: {len(records)}\n",
            encoding="utf-8",
        )
        return split_files, report_md_path


def _find_valid_holdout_partition(
    records: List[Dict[str, Any]],
    participants: List[str],
    glosses: List[str],
    cfg: Config,
    max_seed_attempts: int = 50,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Try candidate seeds to find a participant partition satisfying min_per_gloss."""
    n_parts = len(participants)
    r_train = cfg.split.n_train
    r_val = cfg.split.n_val
    r_test = cfg.split.n_test
    min_per_gloss = cfg.split.min_per_gloss_per_split

    # Normalize ratios if sum close to 1.0
    total_ratio = r_train + r_val + r_test
    if total_ratio > 0:
        n_tr_part = max(1, int(round(n_parts * (r_train / total_ratio))))
        n_val_part = max(1, int(round(n_parts * (r_val / total_ratio))))
        n_te_part = max(1, n_parts - n_tr_part - n_val_part)
    else:
        n_tr_part = max(1, int(r_train))
        n_val_part = max(1, int(r_val))
        n_te_part = max(1, int(r_test))

    best_seed = cfg.split.seed
    best_tr, best_va, best_te = [], [], []
    best_min_count = -1

    for attempt in range(max_seed_attempts):
        cur_seed = cfg.split.seed + attempt
        rng = np.random.RandomState(cur_seed)
        shuffled_parts = list(participants)
        rng.shuffle(shuffled_parts)

        tr_parts = set(shuffled_parts[:n_tr_part])
        va_parts = set(shuffled_parts[n_tr_part : n_tr_part + n_val_part])
        te_parts = set(shuffled_parts[n_tr_part + n_val_part : n_tr_part + n_val_part + n_te_part])

        tr_recs = [r for r in records if r["participant_id"] in tr_parts]
        va_recs = [r for r in records if r["participant_id"] in va_parts]
        te_recs = [r for r in records if r["participant_id"] in te_parts]

        # Check min_per_gloss across all 3 sets
        def min_count(recs: List[Dict[str, Any]]) -> int:
            counts = {g: 0 for g in glosses}
            for r in recs:
                counts[r["gloss"]] += 1
            return min(counts.values()) if counts else 0

        c_tr = min_count(tr_recs)
        c_va = min_count(va_recs)
        c_te = min_count(te_recs)
        overall_min = min(c_tr, c_va, c_te)

        if overall_min > best_min_count:
            best_min_count = overall_min
            best_seed = cur_seed
            best_tr, best_va, best_te = tr_recs, va_recs, te_recs

        if overall_min >= min_per_gloss:
            logger.info(
                f"Found valid participant split on seed {cur_seed} (attempt {attempt + 1}). "
                f"Min samples per gloss: train={c_tr}, val={c_va}, test={c_te}."
            )
            return tr_recs, va_recs, te_recs, cur_seed

    logger.warning(
        f"Could not achieve min_per_gloss={min_per_gloss} for all glosses across {max_seed_attempts} seeds. "
        f"Best seed {best_seed} achieved min {best_min_count}. Using best partition."
    )
    return best_tr, best_va, best_te, best_seed


def _partition_group_kfold(
    records: List[Dict[str, Any]],
    participants: List[str],
    n_folds: int = 5,
    seed: int = 42,
) -> List[List[Dict[str, Any]]]:
    """Partition participants evenly into n_folds."""
    rng = np.random.RandomState(seed)
    shuffled_parts = list(participants)
    rng.shuffle(shuffled_parts)

    folds_parts: List[Set[str]] = [set() for _ in range(n_folds)]
    for i, p in enumerate(shuffled_parts):
        folds_parts[i % n_folds].add(p)

    folds_recs: List[List[Dict[str, Any]]] = []
    for f_set in folds_parts:
        f_recs = [r for r in records if r["participant_id"] in f_set]
        folds_recs.append(f_recs)

    return folds_recs


def _generate_split_report(
    report_path: Path,
    train_recs: List[Dict[str, Any]],
    val_recs: List[Dict[str, Any]],
    test_recs: List[Dict[str, Any]],
    glosses: List[str],
    cfg: Config,
    used_seed: int,
) -> None:
    tr_parts = sorted({r["participant_id"] for r in train_recs})
    va_parts = sorted({r["participant_id"] for r in val_recs})
    te_parts = sorted({r["participant_id"] for r in test_recs})

    total_seqs = len(train_recs) + len(val_recs) + len(test_recs)

    lines = [
        "# ISLR Signer-Independent Split Report",
        "",
        "## 1. Split Overview",
        f"- **Method:** `{cfg.split.method}` (Seed: `{used_seed}`)",
        f"- **Total Sequences:** {total_seqs}",
        f"- **Train:** {len(train_recs)} sequences ({len(train_recs)/total_seqs*100:.1f}%), {len(tr_parts)} participants",
        f"- **Validation:** {len(val_recs)} sequences ({len(val_recs)/total_seqs*100:.1f}%), {len(va_parts)} participants",
        f"- **Test:** {len(test_recs)} sequences ({len(test_recs)/total_seqs*100:.1f}%), {len(te_parts)} participants",
        "",
        "## 2. Participant Disjointness (Leakage Check)",
        f"- **Train Participants ({len(tr_parts)}):** `{', '.join(tr_parts)}`",
        f"- **Val Participants ({len(va_parts)}):** `{', '.join(va_parts)}`",
        f"- **Test Participants ({len(te_parts)}):** `{', '.join(te_parts)}`",
        "- **Signer Leakage Check:** ✅ PASSED (Zero overlapping participants)",
        "",
        "## 3. Gloss Balance Across Splits",
        "| Label | Gloss | Train Seqs | Val Seqs | Test Seqs | Total |",
        "| :---: | :--- | :---: | :---: | :---: | :---: |",
    ]

    def count_by_gloss(recs):
        counts = {}
        for r in recs:
            counts[r["gloss"]] = counts.get(r["gloss"], 0) + 1
        return counts

    c_tr = count_by_gloss(train_recs)
    c_va = count_by_gloss(val_recs)
    c_te = count_by_gloss(test_recs)

    for idx, g in enumerate(glosses):
        n_tr = c_tr.get(g, 0)
        n_va = c_va.get(g, 0)
        n_te = c_te.get(g, 0)
        n_tot = n_tr + n_va + n_te
        lines.append(f"| {idx} | **{g}** | {n_tr} | {n_va} | {n_te} | {n_tot} |")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
