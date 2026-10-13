"""Tests for ISLR gloss selection module and CLI stage (Step 1).

Covers catalog parsing, candidate prefiltering, force_include/exclude overrides,
hand coverage & motion energy metric computation on synthetic parquets with NaNs,
greedy diversity vs top_score selection, and CSV & markdown report generation.
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

prepare_data_module = importlib.import_module("scripts.10_prepare_data")
run_pipeline = prepare_data_module.run_pipeline
from src.data.selection import (
    compute_parquet_hand_metrics,
    read_catalog_records,
    select_20_glosses,
)
from src.utils.config import load_config
from src.utils.env import ENV_LOCAL, resolve_paths

EXPECTED_COLUMNS = ["frame", "row_id", "type", "landmark_index", "x", "y", "z"]


def _create_synthetic_parquet(
    file_path: Path,
    num_frames: int = 5,
    lh_nan: bool = False,
    rh_nan: bool = True,
) -> Path:
    """Helper to generate synthetic landmark parquet for unit tests."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    types_per_frame = ["face"] * 468 + ["left_hand"] * 21 + ["pose"] * 33 + ["right_hand"] * 21
    indices_per_frame = list(range(468)) + list(range(21)) + list(range(33)) + list(range(21))

    frames = [f for f in range(num_frames) for _ in range(543)]
    all_types = types_per_frame * num_frames
    all_indices = indices_per_frame * num_frames
    row_ids = [f"{f}-{t}-{idx}" for f in range(num_frames) for t, idx in zip(types_per_frame, indices_per_frame)]

    x_coords, y_coords, z_coords = [], [], []
    for f, t, idx in zip(frames, all_types, all_indices):
        if t == "left_hand" and lh_nan:
            x_coords.append(float("nan"))
            y_coords.append(float("nan"))
            z_coords.append(float("nan"))
        elif t == "right_hand" and rh_nan:
            x_coords.append(float("nan"))
            y_coords.append(float("nan"))
            z_coords.append(float("nan"))
        else:
            # Vary wrist coordinates slightly across frames to generate motion
            offset = 0.05 * f if idx == 0 else 0.0
            x_coords.append(0.5 + offset)
            y_coords.append(0.5 + offset)
            z_coords.append(0.5)

    table = pa.Table.from_arrays(
        [
            pa.array(frames, type=pa.int16()),
            pa.array(row_ids, type=pa.string()),
            pa.array(all_types, type=pa.string()),
            pa.array(all_indices, type=pa.int16()),
            pa.array(x_coords, type=pa.float32()),
            pa.array(y_coords, type=pa.float32()),
            pa.array(z_coords, type=pa.float32()),
        ],
        names=EXPECTED_COLUMNS,
    )
    pq.write_table(table, file_path)
    return file_path


def _create_synthetic_catalog(
    catalog_path: Path,
    mirror_dir: Path,
    num_signs: int = 25,
    samples_per_sign: int = 12,
    num_participants: int = 6,
) -> Path:
    """Create a synthetic train.csv and matching probed parquets in mirror_dir."""
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [["path", "participant_id", "sequence_id", "sign"]]

    for s_idx in range(num_signs):
        sign_name = f"sign_{s_idx:02d}"
        for samp_idx in range(samples_per_sign):
            p_id = f"signer_{samp_idx % num_participants:02d}"
            seq_id = f"seq_{s_idx}_{samp_idx}"
            rel_path = f"train_landmark_files/{p_id}/{seq_id}.parquet"
            rows.append([rel_path, p_id, seq_id, sign_name])

            # Pre-populate parquet in mirror_dir so no network is required
            parquet_path = mirror_dir / rel_path
            _create_synthetic_parquet(parquet_path, num_frames=4, lh_nan=False, rh_nan=True)

    with open(catalog_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return catalog_path


def test_read_catalog_records_valid(tmp_path: Path) -> None:
    csv_file = tmp_path / "train.csv"
    csv_file.write_text(
        "path,participant_id,sequence_id,sign\n"
        "train_landmark_files/1/100.parquet,1,100,hello\n",
        encoding="utf-8",
    )
    records = read_catalog_records(csv_file)
    assert len(records) == 1
    assert records[0]["sign"] == "hello"
    assert records[0]["participant_id"] == "1"


def test_read_catalog_records_missing_cols(tmp_path: Path) -> None:
    csv_file = tmp_path / "train.csv"
    csv_file.write_text("path,participant_id,sign\nfoo,1,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        read_catalog_records(csv_file)


def test_compute_parquet_hand_metrics(tmp_path: Path) -> None:
    parquet_path = tmp_path / "test.parquet"
    _create_synthetic_parquet(parquet_path, num_frames=4, lh_nan=False, rh_nan=True)

    # LH has valid landmarks, RH is all NaN
    cov_any, energy_any = compute_parquet_hand_metrics(parquet_path, require_hands="any")
    assert cov_any == 1.0
    assert energy_any > 0.0

    cov_both, _ = compute_parquet_hand_metrics(parquet_path, require_hands="both")
    assert cov_both == 0.0

    cov_dom, _ = compute_parquet_hand_metrics(parquet_path, require_hands="dominant_only")
    assert cov_dom == 1.0


def test_select_20_glosses_end_to_end(tmp_path: Path) -> None:
    mirror_dir = tmp_path / "mirror"
    catalog_path = mirror_dir / "train.csv"
    _create_synthetic_catalog(catalog_path, mirror_dir, num_signs=25)

    output_dir = tmp_path / "outputs"
    selection_dir = tmp_path / "selection"

    overrides = [
        f"data.local_mirror_dir={mirror_dir}",
        "selection.n_glosses=20",
        "selection.n_candidates=25",
        "selection.exclude_signs=['sign_00']",
        "selection.force_include_signs=['sign_24']",
        "selection.diversity_method=greedy_maxmin",
        f"output.output_dir={output_dir}",
        f"output.selection_dir={selection_dir}",
    ]
    cfg = load_config(overrides=overrides)
    paths = resolve_paths(ENV_LOCAL, cfg, create_dirs=True)

    selected, glosses_csv, report_path = select_20_glosses(cfg=cfg, paths=paths)

    assert len(selected) == 20
    assert glosses_csv.is_file()
    assert report_path.is_file()

    selected_signs = [c.sign for c in selected]
    # sign_00 was excluded
    assert "sign_00" not in selected_signs
    # sign_24 was forced
    assert "sign_24" in selected_signs

    # Check labels are 0 to 19 contiguous
    labels = [c.label for c in selected]
    assert labels == list(range(20))

    # Read glosses.csv back and verify content
    with open(glosses_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 20
        assert "composite_score" in rows[0]
        assert "hand_coverage" in rows[0]


def test_cli_prepare_data_selection_stage(tmp_path: Path) -> None:
    mirror_dir = tmp_path / "mirror"
    catalog_path = mirror_dir / "train.csv"
    _create_synthetic_catalog(catalog_path, mirror_dir, num_signs=15)

    overrides = [
        f"data.local_mirror_dir={mirror_dir}",
        "selection.n_glosses=10",
        "selection.n_candidates=15",
        "selection.diversity_method=top_score",
        f"output.output_dir={tmp_path / 'out'}",
        f"output.selection_dir={tmp_path / 'selection'}",
    ]

    ret = run_pipeline(stage="selection", overrides=overrides)
    assert ret == 0
    assert (tmp_path / "selection" / "glosses.csv").is_file()
    assert (tmp_path / "selection" / "selection_report.md").is_file()
    assert (tmp_path / "out" / "resolved_config.yaml").is_file()
