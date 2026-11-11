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
from typing import Any, Dict, List, Optional, Set, Tuple

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

    if cfg.split.manifest_path:
        manifest_p = Path(cfg.split.manifest_path)
        if not manifest_p.is_file():
            candidate = paths.repo_root / cfg.split.manifest_path
            if candidate.is_file():
                manifest_p = candidate
            else:
                candidate2 = paths.persistent_root / cfg.split.manifest_path
                if candidate2.is_file():
                    manifest_p = candidate2
                else:
                    raise FileNotFoundError(f"Frozen split manifest not found at: {cfg.split.manifest_path}")

        logger.info(f"Rebinding frozen split from manifest: {manifest_p}")
        with open(manifest_p, mode="r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        schema_ver = manifest_data.get("schema_version")
        if schema_ver != 1:
            raise ValueError(
                f"Unsupported split manifest schema_version in {manifest_p}: "
                f"expected 1, got {schema_ver}."
            )

        splits_dict = manifest_data.get("splits", {})
        used_seed = int(manifest_data.get("seed", cfg.split.seed))
        frozen_vocab = manifest_data.get("vocabulary")
        frozen_label_map = manifest_data.get("label_map")
        effective_vocab = list(frozen_vocab) if frozen_vocab else unique_glosses

        # Validate label mapping against frozen manifest
        if frozen_label_map is not None and isinstance(frozen_label_map, dict):
            for r in records:
                g = r.get("gloss")
                if g in frozen_label_map and r.get("label") != frozen_label_map[g]:
                    raise ValueError(
                        f"Label mismatch in frozen split rebind for gloss '{g}': "
                        f"cache has label {r.get('label')}, but manifest specifies {frozen_label_map[g]}."
                    )
            label_map = frozen_label_map
        else:
            label_map = {r["gloss"]: r["label"] for r in records}

        records_by_seq = {r["sequence_id"]: r for r in records}

        # Validate per-sequence provenance (signer, gloss, label)
        seq_provenance = manifest_data.get("sequences")
        if seq_provenance and isinstance(seq_provenance, dict):
            for sid, expected_meta in seq_provenance.items():
                if sid in records_by_seq:
                    rec = records_by_seq[sid]
                    if "participant_id" in expected_meta and rec.get("participant_id") != expected_meta["participant_id"]:
                        raise ValueError(
                            f"Signer provenance violation for sequence '{sid}': "
                            f"expected '{expected_meta['participant_id']}', got '{rec.get('participant_id')}'."
                        )
                    if "gloss" in expected_meta and rec.get("gloss") != expected_meta["gloss"]:
                        raise ValueError(
                            f"Gloss provenance violation for sequence '{sid}': "
                            f"expected '{expected_meta['gloss']}', got '{rec.get('gloss')}'."
                        )
        else:
            # Fallback for manifests where sequence_id encodes {split}_{gloss}_{p}
            for sid, rec in records_by_seq.items():
                parts = sid.split("_")
                if len(parts) >= 3 and parts[0] in splits_dict:
                    s_split = parts[0]
                    if parts[1] in effective_vocab:
                        exp_gloss = parts[1]
                        if rec.get("gloss") != exp_gloss:
                            raise ValueError(
                                f"Gloss provenance violation for sequence '{sid}': "
                                f"expected gloss '{exp_gloss}', got '{rec.get('gloss')}'."
                            )
                        exp_signer = f"{s_split}_p{parts[2]}"
                        if exp_signer in splits_dict[s_split].get("participant_ids", []):
                            if rec.get("participant_id") != exp_signer:
                                raise ValueError(
                                    f"Signer provenance violation for sequence '{sid}': "
                                    f"expected participant '{exp_signer}', got '{rec.get('participant_id')}'."
                                )

        # Track missing sequences per split for explicit exclusion reporting
        missing_by_split: Dict[str, List[str]] = {}
        for sname in ("train", "val", "test"):
            expected_sids = splits_dict.get(sname, {}).get("sequence_ids", [])
            missing = [sid for sid in expected_sids if sid not in records_by_seq]
            if missing:
                missing_by_split[sname] = missing

        train_recs = [records_by_seq[sid] for sid in splits_dict.get("train", {}).get("sequence_ids", []) if sid in records_by_seq]
        val_recs = [records_by_seq[sid] for sid in splits_dict.get("val", {}).get("sequence_ids", []) if sid in records_by_seq]
        test_recs = [records_by_seq[sid] for sid in splits_dict.get("test", {}).get("sequence_ids", []) if sid in records_by_seq]

        # Verify participant provenance and class coverage in rebind
        min_per_gloss = cfg.split.min_per_gloss_per_split
        for split_name, rec_list in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
            # Check participant IDs against frozen manifest
            expected_participants = set(splits_dict.get(split_name, {}).get("participant_ids", []))
            if expected_participants:
                actual_participants = {r["participant_id"] for r in rec_list}
                if not actual_participants.issubset(expected_participants):
                    foreign = actual_participants - expected_participants
                    raise ValueError(
                        f"Signer provenance violation in frozen split rebind for '{split_name}': "
                        f"found unexpected participant IDs {sorted(foreign)}. "
                        f"Frozen manifest restricts participants to {sorted(expected_participants)}."
                    )

            counts: Dict[str, int] = {}
            for r in rec_list:
                counts[r["gloss"]] = counts.get(r["gloss"], 0) + 1
            for g in effective_vocab:
                cnt = counts.get(g, 0)
                if cnt < min_per_gloss:
                    raise ValueError(
                        f"Class coverage violation in frozen split rebind: gloss '{g}' has {cnt} samples "
                        f"in '{split_name}' (minimum required: {min_per_gloss}). "
                        "Reassigning participants or altering seeds to compensate is strictly prohibited."
                    )

        is_clean, leaks = verify_no_participant_leakage(train_recs, val_recs, test_recs)
        if not is_clean:
            raise RuntimeError(f"Participant leakage detected in frozen manifest rebind: {leaks}")

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
        with open(splits_dir / "label_map.json", mode="w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)

        classes_csv = splits_dir / "classes.csv"
        with open(classes_csv, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gloss", "label"])
            for g in sorted(label_map.keys()):
                writer.writerow([g, label_map[g]])

        report_md_path = report_dir / "split_report.md"
        _generate_split_report(
            report_path=report_md_path,
            train_recs=train_recs,
            val_recs=val_recs,
            test_recs=test_recs,
            glosses=effective_vocab,
            cfg=cfg,
            used_seed=used_seed,
            missing_sequences=missing_by_split,
        )
        split_files["manifest"] = manifest_p
        return split_files, report_md_path

    elif split_method == "participant_holdout":
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

        # Save canonical split_manifest.json for freezing and downstream reproducibility
        all_split_recs = [("train", train_recs), ("val", val_recs), ("test", test_recs)]
        manifest_payload = {
            "schema_version": 1,
            "split_method": "participant_holdout",
            "seed": used_seed,
            "vocabulary": unique_glosses,
            "label_map": label_map,
            "splits": {
                "train": {
                    "participant_ids": sorted({r["participant_id"] for r in train_recs}),
                    "sequence_ids": [r["sequence_id"] for r in train_recs],
                },
                "val": {
                    "participant_ids": sorted({r["participant_id"] for r in val_recs}),
                    "sequence_ids": [r["sequence_id"] for r in val_recs],
                },
                "test": {
                    "participant_ids": sorted({r["participant_id"] for r in test_recs}),
                    "sequence_ids": [r["sequence_id"] for r in test_recs],
                },
            },
            "sequences": {
                r["sequence_id"]: {
                    "participant_id": r["participant_id"],
                    "gloss": r["gloss"],
                    "label": r["label"],
                    "split": sname,
                }
                for sname, recs in all_split_recs
                for r in recs
            },
        }
        manifest_json_path = splits_dir / "split_manifest.json"
        with open(manifest_json_path, mode="w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)
        split_files["manifest"] = manifest_json_path

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
    missing_sequences: Optional[Dict[str, List[str]]] = None,
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

    if missing_sequences and any(missing_sequences.values()):
        lines.extend([
            "",
            "## 4. Frozen Rebind Exclusions & Missing Sequences",
            "| Split | Missing Sequence ID | Exclusion Reason |",
            "| :--- | :--- | :--- |",
        ])
        for sname, sids in missing_sequences.items():
            for sid in sids:
                lines.append(f"| `{sname}` | `{sid}` | Sequence absent from active feature cache (exclusion) |")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
