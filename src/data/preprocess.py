"""ISLR Landmark Preprocessing Module: Parquet -> Feature .npy.

Pipeline steps per sequence:
1. Pivot parquet to (T_raw, 543, 3) across MediaPipe Holistic landmarks.
2. Determine dominant hand by empirical data (valid frames & motion energy).
3. Trim idle frames at sequence edges; exclude sequences below min_valid_frames.
4. Linearly interpolate short NaN gaps (<= max_gap_frames); fill remaining with 0.
5. Resample temporal sequence to fixed T (default 32).
6. Extract normalized features (143 dims for hands, 160 dims for hands_pose).
7. Save .npy to cache_dir/features and generate manifest.csv & excluded.csv.
8. Generate detailed preprocessing report with distributions and plots.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from src.features.normalization import (
    compute_feature_vector,
    get_feature_dim,
    interpolate_bounded_gaps,
    resample_temporal,
)
from src.utils.config import Config
from src.utils.env import ResolvedPaths
from src.utils.logging import get_logger

logger = get_logger("islr.preprocess")

# Landmark index offsets in 543-point MediaPipe Holistic array
FACE_OFFSET = 0
FACE_COUNT = 468
LH_OFFSET = 468
LH_COUNT = 21
POSE_OFFSET = 489
POSE_COUNT = 33
RH_OFFSET = 522
RH_COUNT = 21
TOTAL_LANDMARKS = 543


@dataclass
class PreprocessItemResult:
    sequence_id: str
    participant_id: str
    gloss: str
    label: int
    n_valid_frames: int
    hand_coverage: float
    npy_path: str
    status: str  # 'kept' or 'excluded'
    reason: str = ""


def pivot_parquet_to_raw_landmarks(parquet_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load parquet and pivot into (T_raw, 543, 3) and array of unique frame numbers."""
    table = pq.read_table(
        parquet_path,
        columns=["frame", "type", "landmark_index", "x", "y", "z"],
    )
    frames_col = table.column("frame").to_numpy()
    types_col = table.column("type").to_pylist()
    indices_col = table.column("landmark_index").to_numpy()
    x_col = table.column("x").to_numpy()
    y_col = table.column("y").to_numpy()
    z_col = table.column("z").to_numpy()

    unique_frames, frame_inverse = np.unique(frames_col, return_inverse=True)
    T_raw = len(unique_frames)
    if T_raw == 0:
        return np.empty((0, TOTAL_LANDMARKS, 3), dtype=np.float32), np.empty((0,), dtype=int)

    # Compute global landmark index for each row
    global_indices = np.zeros(len(frames_col), dtype=int)
    for i, (t, idx) in enumerate(zip(types_col, indices_col)):
        if t == "face":
            global_indices[i] = FACE_OFFSET + idx
        elif t == "left_hand":
            global_indices[i] = LH_OFFSET + idx
        elif t == "pose":
            global_indices[i] = POSE_OFFSET + idx
        elif t == "right_hand":
            global_indices[i] = RH_OFFSET + idx
        else:
            global_indices[i] = idx

    landmarks = np.full((T_raw, TOTAL_LANDMARKS, 3), np.nan, dtype=np.float32)
    valid_coords = ~np.isnan(x_col)
    f_idx = frame_inverse[valid_coords]
    g_idx = global_indices[valid_coords]

    landmarks[f_idx, g_idx, 0] = x_col[valid_coords]
    landmarks[f_idx, g_idx, 1] = y_col[valid_coords]
    landmarks[f_idx, g_idx, 2] = z_col[valid_coords]

    return landmarks, unique_frames


def compute_preprocess_hash(cfg: Config) -> Tuple[str, Dict[str, Any]]:
    """Compute deterministic SHA256 hash of configuration parameters affecting feature output."""
    meta_params: Dict[str, Any] = {
        "preprocess.dominant_hand": str(cfg.preprocess.dominant_hand),
        "preprocess.max_gap_frames": int(cfg.preprocess.max_gap_frames),
        "preprocess.min_valid_frames": int(cfg.preprocess.min_valid_frames),
        "preprocess.pad_frames_around": int(cfg.preprocess.pad_frames_around),
        "preprocess.sequence_length": int(cfg.preprocess.sequence_length),
        "preprocess.trim_no_hand": bool(cfg.preprocess.trim_no_hand),
        "features.set": str(cfg.features.set),
    }
    if hasattr(cfg.features, "use_pose"):
        meta_params["features.use_pose"] = bool(cfg.features.use_pose)
    serialized = json.dumps(meta_params, sort_keys=True)
    cfg_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return cfg_hash, meta_params


def preprocess_single_sequence(
    parquet_path: Path,
    sequence_id: str,
    participant_id: str,
    gloss: str,
    label: int,
    cfg: Config,
    out_dir: Path,
    reusable_manifest: Optional[Dict[str, Tuple[int, float]]] = None,
) -> PreprocessItemResult:
    """Execute end-to-end preprocessing for a single landmark sequence."""
    target_npy = out_dir / f"{sequence_id}.npy"

    # Idempotent resume check backed by existing valid manifest entry (zero fake values)
    if (
        reusable_manifest is not None
        and sequence_id in reusable_manifest
        and target_npy.is_file()
        and target_npy.stat().st_size > 0
    ):
        try:
            arr = np.load(str(target_npy))
            expected_dim = get_feature_dim(cfg.features.set, getattr(cfg.features, "use_pose", False))
            if arr.shape == (cfg.preprocess.sequence_length, expected_dim):
                n_valid, cov = reusable_manifest[sequence_id]
                return PreprocessItemResult(
                    sequence_id=sequence_id,
                    participant_id=participant_id,
                    gloss=gloss,
                    label=label,
                    n_valid_frames=n_valid,
                    hand_coverage=cov,
                    npy_path=str(target_npy),
                    status="kept",
                )
        except Exception:
            pass  # Recompute if file was corrupted

    # 1. Pivot parquet
    try:
        raw_landmarks, _ = pivot_parquet_to_raw_landmarks(parquet_path)
    except Exception as exc:
        return PreprocessItemResult(
            sequence_id=sequence_id,
            participant_id=participant_id,
            gloss=gloss,
            label=label,
            n_valid_frames=0,
            hand_coverage=0.0,
            npy_path="",
            status="excluded",
            reason=f"read_error: {exc}",
        )

    T_raw = len(raw_landmarks)
    if T_raw == 0:
        return PreprocessItemResult(
            sequence_id=sequence_id,
            participant_id=participant_id,
            gloss=gloss,
            label=label,
            n_valid_frames=0,
            hand_coverage=0.0,
            npy_path="",
            status="excluded",
            reason="empty_parquet",
        )

    lh_raw = raw_landmarks[:, LH_OFFSET : LH_OFFSET + LH_COUNT, :]  # (T_raw, 21, 3)
    rh_raw = raw_landmarks[:, RH_OFFSET : RH_OFFSET + RH_COUNT, :]  # (T_raw, 21, 3)
    pose_raw = raw_landmarks[:, POSE_OFFSET : POSE_OFFSET + POSE_COUNT, :]  # (T_raw, 33, 3)

    lh_mask = ~np.isnan(lh_raw[:, 0, 0])  # Wrist non-nan
    rh_mask = ~np.isnan(rh_raw[:, 0, 0])

    # 2. Dominant hand detection by empirical data
    lh_valid_count = int(np.sum(lh_mask))
    rh_valid_count = int(np.sum(rh_mask))

    def calc_motion(hand_pts: np.ndarray, mask: np.ndarray) -> float:
        if np.sum(mask) < 2:
            return 0.0
        wrists = hand_pts[mask, 0, :]
        deltas = np.linalg.norm(wrists[1:] - wrists[:-1], axis=-1)
        return float(np.sum(deltas))

    lh_motion = calc_motion(lh_raw, lh_mask)
    rh_motion = calc_motion(rh_raw, rh_mask)

    use_dominant = cfg.preprocess.dominant_hand not in (False, "false", "disabled", "none")
    if use_dominant and (rh_valid_count > lh_valid_count or (rh_valid_count == lh_valid_count and rh_motion > lh_motion)):
        # Hand 0 = Right hand, Hand 1 = Left hand
        hands_coords = np.stack([rh_raw, lh_raw], axis=1)  # (T_raw, 2, 21, 3)
        hands_masks = np.stack([rh_mask, lh_mask], axis=1)  # (T_raw, 2)
    else:
        # Hand 0 = Left hand, Hand 1 = Right hand
        hands_coords = np.stack([lh_raw, rh_raw], axis=1)
        hands_masks = np.stack([lh_mask, rh_mask], axis=1)

    has_any_hand = hands_masks.any(axis=1)
    n_valid_hand_frames = int(np.sum(has_any_hand))
    hand_coverage = float(n_valid_hand_frames / T_raw) if T_raw > 0 else 0.0

    # 3. Trim edge idle frames & validate min_valid_frames
    if n_valid_hand_frames < cfg.preprocess.min_valid_frames:
        return PreprocessItemResult(
            sequence_id=sequence_id,
            participant_id=participant_id,
            gloss=gloss,
            label=label,
            n_valid_frames=n_valid_hand_frames,
            hand_coverage=hand_coverage,
            npy_path="",
            status="excluded",
            reason=f"insufficient_hand_frames: {n_valid_hand_frames} < {cfg.preprocess.min_valid_frames}",
        )

    if cfg.preprocess.trim_no_hand:
        valid_indices = np.where(has_any_hand)[0]
        margin = cfg.preprocess.pad_frames_around
        start_t = max(0, int(valid_indices[0]) - margin)
        end_t = min(T_raw, int(valid_indices[-1]) + margin + 1)
        hands_coords = hands_coords[start_t:end_t]
        hands_masks = hands_masks[start_t:end_t]
        pose_raw = pose_raw[start_t:end_t]

    # 4. Interpolate short gaps and zero-fill remaining
    hands_filled, hands_masks_filled = interpolate_bounded_gaps(
        hands_coords,
        hands_masks,
        max_gap_frames=cfg.preprocess.max_gap_frames,
    )
    pose_filled = np.nan_to_num(pose_raw, nan=0.0)
    pose_mask = ~np.isnan(pose_raw[:, 11, 0]) if len(pose_raw) > 0 else np.zeros((len(hands_coords),), dtype=bool)

    # 5. Resample to fixed sequence_length
    T_target = cfg.preprocess.sequence_length
    resampled_hands, resampled_hand_masks = resample_temporal(
        hands_filled,
        hands_masks_filled,
        target_len=T_target,
    )
    resampled_pose, resampled_pose_masks = resample_temporal(
        pose_filled,
        pose_mask[:, np.newaxis],
        target_len=T_target,
    )

    # 6. Extract normalized features
    features = compute_feature_vector(
        hands_coords=resampled_hands,
        hands_masks=resampled_hand_masks,
        pose_coords=resampled_pose,
        pose_masks=resampled_pose_masks,
        use_pose=cfg.features.use_pose or (cfg.features.set == "hands_pose"),
    )

    # 7. Save feature .npy
    target_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(target_npy), features.astype(np.float32))

    return PreprocessItemResult(
        sequence_id=sequence_id,
        participant_id=participant_id,
        gloss=gloss,
        label=label,
        n_valid_frames=n_valid_hand_frames,
        hand_coverage=hand_coverage,
        npy_path=str(target_npy),
        status="kept",
    )


def run_preprocessing(
    cfg: Config,
    paths: ResolvedPaths,
) -> Tuple[Path, Path, Path]:
    """Execute batch preprocessing across all downloaded sequences for selected glosses."""
    features_dir = paths.cache_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    report_dir = paths.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    # Read selected glosses and labels
    glosses_csv = paths.selection_dir / "glosses.csv"
    if not glosses_csv.is_file():
        raise FileNotFoundError(f"Selection file {glosses_csv} not found. Run selection first.")

    label_map: Dict[str, int] = {}
    with open(glosses_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_map[row["gloss"].strip()] = int(row["label"])

    # Read download manifest (or fallback to catalog if local mirror)
    download_manifest_path = paths.data_dir / "download_manifest.csv"
    items_to_process: List[Dict[str, Any]] = []

    if download_manifest_path.is_file():
        with open(download_manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"] != "failed" and row["local_path"]:
                    items_to_process.append({
                        "path": Path(row["local_path"]),
                        "sequence_id": row["sequence_id"],
                        "participant_id": row["participant_id"],
                        "gloss": row["gloss"],
                        "label": label_map.get(row["gloss"], -1),
                    })
    else:
        # Fallback to catalog
        catalog_path = paths.data_dir / cfg.data.train_csv_name
        if not catalog_path.is_file() and cfg.data.local_mirror_dir:
            catalog_path = Path(cfg.data.local_mirror_dir) / cfg.data.train_csv_name
        if not catalog_path.is_file():
            raise FileNotFoundError(f"Neither download_manifest.csv nor {catalog_path} was found.")

        with open(catalog_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sign = row["sign"].strip()
                if sign in label_map:
                    local_p = paths.data_dir / row["path"].strip()
                    if not local_p.is_file() and cfg.data.local_mirror_dir:
                        local_p = Path(cfg.data.local_mirror_dir) / row["path"].strip()
                    if local_p.is_file():
                        items_to_process.append({
                            "path": local_p,
                            "sequence_id": str(row["sequence_id"]).strip(),
                            "participant_id": str(row["participant_id"]).strip(),
                            "gloss": sign,
                            "label": label_map[sign],
                        })

    logger.info(f"Preparing to preprocess {len(items_to_process)} sequences.")
    num_workers = max(1, cfg.preprocess.num_workers)

    # Cache validation via preprocess_meta.json
    meta_json_path = features_dir / "preprocess_meta.json"
    manifest_csv = features_dir / "manifest.csv"
    current_hash, meta_params = compute_preprocess_hash(cfg)

    reusable_manifest: Dict[str, Tuple[int, float]] = {}
    cache_is_valid = False

    if meta_json_path.is_file():
        try:
            with open(meta_json_path, mode="r", encoding="utf-8") as f:
                old_meta = json.load(f)
            if old_meta.get("config_hash") == current_hash:
                cache_is_valid = True
                logger.info("Matching preprocess_meta.json found; cache is valid for resume.")
            else:
                logger.info("Preprocess parameters changed; invalidating existing features cache.")
        except Exception:
            logger.warning("Could not parse preprocess_meta.json; invalidating cache.")

    if cache_is_valid and manifest_csv.is_file():
        try:
            with open(manifest_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    sid = r.get("sequence_id", "").strip()
                    if sid and "n_valid_frames" in r and "hand_coverage" in r:
                        reusable_manifest[sid] = (
                            int(r["n_valid_frames"]),
                            float(r["hand_coverage"]),
                        )
            logger.info(f"Loaded {len(reusable_manifest)} existing sequences from manifest for resume.")
        except Exception:
            reusable_manifest.clear()
    else:
        # Cache is invalid or absent: clear old manifest and .npy files
        if manifest_csv.is_file():
            try:
                manifest_csv.unlink()
            except OSError:
                pass
        for old_npy in features_dir.glob("*.npy"):
            try:
                old_npy.unlink()
            except OSError:
                pass

    results: List[PreprocessItemResult] = []
    if num_workers > 1 and len(items_to_process) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_item = {
                executor.submit(
                    preprocess_single_sequence,
                    item["path"],
                    item["sequence_id"],
                    item["participant_id"],
                    item["gloss"],
                    item["label"],
                    cfg,
                    features_dir,
                    reusable_manifest,
                ): item
                for item in items_to_process
            }
            for future in as_completed(future_to_item):
                results.append(future.result())
    else:
        for item in items_to_process:
            res = preprocess_single_sequence(
                item["path"],
                item["sequence_id"],
                item["participant_id"],
                item["gloss"],
                item["label"],
                cfg,
                features_dir,
                reusable_manifest,
            )
            results.append(res)

    # Sort results
    results.sort(key=lambda r: (r.gloss, r.sequence_id))
    kept_items = [r for r in results if r.status == "kept"]
    excluded_items = [r for r in results if r.status == "excluded"]

    logger.info(
        f"Preprocessing complete: {len(kept_items)} kept, {len(excluded_items)} excluded "
        f"out of {len(results)} total."
    )

    # 7. Write manifest.csv
    with open(manifest_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sequence_id",
            "participant_id",
            "gloss",
            "label",
            "n_valid_frames",
            "hand_coverage",
            "path",
        ])
        for r in kept_items:
            writer.writerow([
                r.sequence_id,
                r.participant_id,
                r.gloss,
                r.label,
                r.n_valid_frames,
                f"{r.hand_coverage:.4f}",
                r.npy_path,
            ])

    # Write preprocess_meta.json
    meta_record = {
        "config_hash": current_hash,
        "parameters": meta_params,
    }
    with open(meta_json_path, mode="w", encoding="utf-8") as f:
        json.dump(meta_record, f, indent=2)

    # Write excluded.csv
    excluded_csv = features_dir / "excluded.csv"
    with open(excluded_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "participant_id", "gloss", "reason"])
        for r in excluded_items:
            writer.writerow([r.sequence_id, r.participant_id, r.gloss, r.reason])

    # 8. Generate report and plots
    report_md_path = report_dir / "preprocess_report.md"
    _generate_preprocess_report(
        report_path=report_md_path,
        kept=kept_items,
        excluded=excluded_items,
        cfg=cfg,
        report_dir=report_dir,
    )

    return manifest_csv, excluded_csv, report_md_path


def _generate_preprocess_report(
    report_path: Path,
    kept: List[PreprocessItemResult],
    excluded: List[PreprocessItemResult],
    cfg: Config,
    report_dir: Path,
) -> None:
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_img_path = plots_dir / "hand_coverage_distribution.png"

    total = len(kept) + len(excluded)
    cov_values = [r.hand_coverage for r in kept]
    mean_cov = float(np.mean(cov_values)) if cov_values else 0.0

    # Group statistics by gloss
    gloss_stats: Dict[str, List[float]] = {}
    for r in kept:
        gloss_stats.setdefault(r.gloss, []).append(r.hand_coverage)

    # Group statistics by participant
    part_stats: Dict[str, List[float]] = {}
    for r in kept:
        part_stats.setdefault(r.participant_id, []).append(r.hand_coverage)

    # Render coverage histogram plot
    if cov_values:
        plt.figure(figsize=(8, 4))
        plt.hist(cov_values, bins=20, color="#1f77b4", edgecolor="black", alpha=0.7)
        plt.title("Distribution of Hand Coverage Across Sequences")
        plt.xlabel("Hand Coverage Fraction")
        plt.ylabel("Number of Sequences")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(str(plot_img_path), dpi=100)
        plt.close()

    # Markdown report
    lines = [
        "# ISLR Preprocessing and Landmark Feature Report",
        "",
        "## 1. Executive Summary",
        f"- **Total Sequences Processed:** {total}",
        f"- **Retained (Kept) Sequences:** {len(kept)} ({len(kept)/total*100:.1f}%)" if total else "- Kept: 0",
        f"- **Excluded Sequences:** {len(excluded)}",
        f"- **Feature Dimension Inferred:** `{get_feature_dim(cfg.features.set, cfg.features.use_pose)}`",
        f"- **Target Temporal Length:** `{cfg.preprocess.sequence_length}`",
        f"- **Mean Hand Coverage (Kept):** `{mean_cov:.3f}`",
        "",
        "## 2. Exclusion Breakdown",
    ]
    if excluded:
        reasons: Dict[str, int] = {}
        for e in excluded:
            reasons[e.reason] = reasons.get(e.reason, 0) + 1
        lines.append("| Exclusion Reason | Count |")
        lines.append("| :--- | :---: |")
        for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append(f"| `{r}` | {cnt} |")
    else:
        lines.append("No sequences were excluded during preprocessing.")

    lines.extend([
        "",
        "## 3. Hand Coverage by Gloss (Mean, Min, Max)",
        "| Gloss | Sequences | Mean Coverage | Min Coverage | Max Coverage | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for g, vals in sorted(gloss_stats.items()):
        g_mean = float(np.mean(vals))
        g_min = float(np.min(vals))
        g_max = float(np.max(vals))
        status = "⚠️ LOW" if g_mean < 0.3 else "✅ OK"
        lines.append(f"| **{g}** | {len(vals)} | {g_mean:.3f} | {g_min:.3f} | {g_max:.3f} | {status} |")

    lines.extend([
        "",
        "## 4. Problematic Participants & Glosses",
    ])
    low_glosses = [g for g, vals in gloss_stats.items() if np.mean(vals) < 0.3]
    low_parts = [p for p, vals in part_stats.items() if np.mean(vals) < 0.3]
    if low_glosses:
        lines.append(f"- **Low Coverage Glosses (< 30%):** {', '.join(low_glosses)}")
    else:
        lines.append("- All glosses maintain adequate hand visibility (>= 30%).")
    if low_parts:
        lines.append(f"- **Low Coverage Participants (< 30%):** {', '.join(low_parts)}")
    else:
        lines.append("- All participants maintain adequate hand visibility.")

    lines.extend([
        "",
        f"![Hand Coverage Distribution]({plot_img_path.as_posix()})",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
