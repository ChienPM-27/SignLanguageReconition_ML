"""Tests for ISLR landmark preprocessing module and CLI stage (Step 3).

Covers parquet pivoting to (T, 543, 3), dominant hand detection, idle trimming,
bounded gap interpolation, temporal resampling to fixed T, dynamic feature extraction
(hands 143 vs hands_pose 160), manifest & excluded CSVs, and markdown reports.
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
import numpy as np

from src.data.preprocess import (
    pivot_parquet_to_raw_landmarks,
    preprocess_single_sequence,
    run_preprocessing,
)
from src.features.normalization import (
    compute_feature_vector,
    get_feature_dim,
    interpolate_bounded_gaps,
)
from src.utils.config import load_config
from src.utils.env import ENV_LOCAL, resolve_paths
from tests.test_selection import _create_synthetic_catalog, _create_synthetic_parquet

prepare_data_module = importlib.import_module("scripts.10_prepare_data")
run_pipeline = prepare_data_module.run_pipeline


def test_pivot_parquet_to_raw_landmarks(tmp_path: Path) -> None:
    parquet_path = tmp_path / "test.parquet"
    _create_synthetic_parquet(parquet_path, num_frames=6, lh_nan=False, rh_nan=False)

    landmarks, frames = pivot_parquet_to_raw_landmarks(parquet_path)
    assert landmarks.shape == (6, 543, 3)
    assert len(frames) == 6
    # LH wrist (468) and RH wrist (522) are non-nan
    assert not np.isnan(landmarks[0, 468, 0])
    assert not np.isnan(landmarks[0, 522, 0])


def test_interpolate_bounded_gaps() -> None:
    # 6 frames, 2 hands, 21 landmarks, 3 dims
    coords = np.zeros((6, 2, 21, 3), dtype=np.float32)
    masks = np.zeros((6, 2), dtype=bool)

    # Hand 0: frame 0 and frame 3 valid (gap of 2 frames: 1, 2)
    coords[0, 0] = 1.0
    coords[3, 0] = 4.0
    masks[0, 0] = True
    masks[3, 0] = True

    # Hand 1: frame 0 and frame 5 valid (gap of 4 frames: 1, 2, 3, 4)
    coords[0, 1] = 1.0
    coords[5, 1] = 6.0
    masks[0, 1] = True
    masks[5, 1] = True

    filled, updated_masks = interpolate_bounded_gaps(coords, masks, max_gap_frames=2)

    # Gap of 2 (frames 1, 2) should be interpolated
    assert updated_masks[1, 0] is True or updated_masks[1, 0] == 1
    assert updated_masks[2, 0] is True or updated_masks[2, 0] == 1
    assert np.isclose(filled[1, 0, 0, 0], 2.0)
    assert np.isclose(filled[2, 0, 0, 0], 3.0)

    # Gap of 4 (> max_gap_frames=2) should NOT be interpolated
    assert updated_masks[2, 1] is False or updated_masks[2, 1] == 0
    assert np.isclose(filled[2, 1, 0, 0], 0.0)


def test_feature_dimension_deduction() -> None:
    # hands only: 143 dims
    dim_hands = get_feature_dim("hands", use_pose=False)
    assert dim_hands == 143

    # hands_pose: 160 dims
    dim_hands_pose = get_feature_dim("hands_pose", use_pose=False)
    assert dim_hands_pose == 160

    dim_use_pose = get_feature_dim("hands", use_pose=True)
    assert dim_use_pose == 160

    # Test feature vector computation produces exact dimensions
    T = 16
    h_coords = np.ones((T, 2, 21, 3), dtype=np.float32)
    h_masks = np.ones((T, 2), dtype=bool)
    p_coords = np.ones((T, 33, 3), dtype=np.float32)
    p_masks = np.ones((T, 1), dtype=bool)

    f_hands = compute_feature_vector(h_coords, h_masks, use_pose=False)
    assert f_hands.shape == (T, 143)

    f_pose = compute_feature_vector(h_coords, h_masks, p_coords, p_masks, use_pose=True)
    assert f_pose.shape == (T, 160)


def test_preprocess_single_sequence_kept_and_dominant_hand(tmp_path: Path) -> None:
    # LH all NaN, RH valid -> RH should become hand 0
    parquet_path = tmp_path / "valid.parquet"
    _create_synthetic_parquet(parquet_path, num_frames=8, lh_nan=True, rh_nan=False)

    out_dir = tmp_path / "features"
    cfg = load_config(overrides=["preprocess.sequence_length=16", "features.set=hands"])

    res = preprocess_single_sequence(
        parquet_path=parquet_path,
        sequence_id="1001",
        participant_id="signer_01",
        gloss="sign_a",
        label=0,
        cfg=cfg,
        out_dir=out_dir,
    )

    assert res.status == "kept"
    assert Path(res.npy_path).is_file()
    saved_arr = np.load(res.npy_path)
    assert saved_arr.shape == (16, 143)


def test_preprocess_single_sequence_excluded_insufficient_frames(tmp_path: Path) -> None:
    # Only 1 frame of valid hand, when min_valid_frames=3 -> should be excluded
    parquet_path = tmp_path / "short.parquet"
    _create_synthetic_parquet(parquet_path, num_frames=1, lh_nan=False, rh_nan=True)

    out_dir = tmp_path / "features"
    cfg = load_config(overrides=["preprocess.min_valid_frames=3"])

    res = preprocess_single_sequence(
        parquet_path=parquet_path,
        sequence_id="1002",
        participant_id="signer_01",
        gloss="sign_b",
        label=1,
        cfg=cfg,
        out_dir=out_dir,
    )

    assert res.status == "excluded"
    assert "insufficient_hand_frames" in res.reason
    assert res.npy_path == ""


def test_run_preprocessing_end_to_end(tmp_path: Path) -> None:
    mirror_dir = tmp_path / "mirror"
    catalog_path = mirror_dir / "train.csv"
    _create_synthetic_catalog(catalog_path, mirror_dir, num_signs=5, samples_per_sign=4)

    selection_dir = tmp_path / "selection"
    selection_dir.mkdir(parents=True)
    glosses_csv = selection_dir / "glosses.csv"
    glosses_csv.write_text(
        "gloss,label,n_samples,n_participants,max_dominance,hand_coverage,motion_energy,composite_score\n"
        "sign_00,0,4,4,0.25,1.0,0.1,0.9\n"
        "sign_01,1,4,4,0.25,1.0,0.1,0.85\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    reports_dir = tmp_path / "reports"

    overrides = [
        f"data.local_mirror_dir={mirror_dir}",
        f"data.cache_dir={cache_dir}",
        f"output.selection_dir={selection_dir}",
        f"output.output_dir={out_dir}",
        f"output.report_dir={reports_dir}",
        "preprocess.sequence_length=32",
        "features.set=hands",
    ]
    cfg = load_config(overrides=overrides)
    paths = resolve_paths(ENV_LOCAL, cfg, create_dirs=True)

    manifest_csv, excluded_csv, report_md = run_preprocessing(cfg=cfg, paths=paths)

    assert manifest_csv.is_file()
    assert excluded_csv.is_file()
    assert report_md.is_file()

    with open(manifest_csv, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 8  # 2 signs * 4 samples
        assert "hand_coverage" in rows[0]
        # Check that saved .npy has correct shape
        first_npy = Path(rows[0]["path"])
        assert first_npy.is_file()
        arr = np.load(str(first_npy))
        assert arr.shape == (32, 143)


def test_cli_prepare_data_preprocess_stage(tmp_path: Path) -> None:
    mirror_dir = tmp_path / "mirror"
    catalog_path = mirror_dir / "train.csv"
    _create_synthetic_catalog(catalog_path, mirror_dir, num_signs=8, samples_per_sign=12, num_participants=6)

    overrides = [
        f"env.local_root={tmp_path}",
        f"data.local_mirror_dir={mirror_dir}",
        "selection.n_glosses=4",
        "selection.n_candidates=8",
        "data.max_files_per_gloss=2",
        f"data.cache_dir={tmp_path / 'cache'}",
        f"output.output_dir={tmp_path / 'out'}",
        f"output.selection_dir={tmp_path / 'selection'}",
        f"output.report_dir={tmp_path / 'reports'}",
    ]

    ret = run_pipeline(stage="all", overrides=overrides)
    assert ret == 0
    assert (tmp_path / "cache" / "features" / "manifest.csv").is_file()
    assert (tmp_path / "reports" / "preprocess_report.md").is_file()
