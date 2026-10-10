"""Tests for Phase 1 schema verification and report generation (src/data/islr_schema.py).

Covers catalog summary extraction, single parquet inspection, mismatch detection,
fail-loudly behavior on missing columns or invalid counts, and markdown report generation.
"""

from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.kaggle_access import FetchResult
from src.data.islr_schema import (
    EXPECTED_PARQUET_COLUMNS,
    generate_schema_markdown_report,
    inspect_catalog_csv,
    inspect_parquet_file,
    verify_schema,
)

EXPECTED_LANDMARK_COUNT = 543


def _create_mock_parquet(
    file_path: Path,
    num_frames: int = 2,
    lh_nan: bool = True,
    corrupt_col: bool = False,
    extra_col: bool = False,
) -> Path:
    """Helper to generate a valid or intentionally corrupted landmark parquet file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    types_per_frame = ["face"] * 468 + ["left_hand"] * 21 + ["pose"] * 33 + ["right_hand"] * 21
    indices_per_frame = list(range(468)) + list(range(21)) + list(range(33)) + list(range(21))

    frames = [f for f in range(num_frames) for _ in range(EXPECTED_LANDMARK_COUNT)]
    all_types = types_per_frame * num_frames
    all_indices = indices_per_frame * num_frames
    row_ids = [f"{f}-{t}-{idx}" for f in range(num_frames) for t, idx in zip(types_per_frame, indices_per_frame)]

    x_coords = []
    y_coords = []
    z_coords = []
    for t in all_types:
        if t == "left_hand" and lh_nan:
            x_coords.append(float("nan"))
            y_coords.append(float("nan"))
            z_coords.append(float("nan"))
        else:
            x_coords.append(0.5)
            y_coords.append(0.5)
            z_coords.append(0.5)

    col_names = list(EXPECTED_PARQUET_COLUMNS)
    arrays = [
        pa.array(frames, type=pa.int16()),
        pa.array(row_ids, type=pa.string()),
        pa.array(all_types, type=pa.string()),
        pa.array(all_indices, type=pa.int16()),
        pa.array(x_coords, type=pa.float32()),
        pa.array(y_coords, type=pa.float32()),
        pa.array(z_coords, type=pa.float32()),
    ]

    if corrupt_col:
        # Drop z coordinate
        col_names = col_names[:-1]
        arrays = arrays[:-1]

    if extra_col:
        col_names.append("extra_col")
        arrays.append(pa.array([1] * len(frames), type=pa.int32()))

    table = pa.Table.from_arrays(arrays, names=col_names)
    pq.write_table(table, file_path)
    return file_path


def _create_mock_catalog(
    file_path: Path,
    parquet_rel_path: str = "train_landmark_files/26734/1000035562.parquet",
    missing_col: bool = False,
    empty_cell: bool = False,
) -> Path:
    """Helper to generate a valid or invalid catalog CSV file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if missing_col:
        content = "path,participant_id,sign\n" f"{parquet_rel_path},26734,blow\n"
    elif empty_cell:
        content = "path,participant_id,sequence_id,sign\n" f"{parquet_rel_path},  ,1000035562,blow\n"
    else:
        content = (
            "path,participant_id,sequence_id,sign\n"
            f"{parquet_rel_path},26734,1000035562,blow\n"
            f"train_landmark_files/26734/1000035563.parquet,26734,1000035563,wait\n"
            f"train_landmark_files/12345/2000035564.parquet,12345,2000035564,blow\n"
        )
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_inspect_catalog_csv_valid(tmp_path: Path) -> None:
    """Verify catalog inspection returns accurate statistics and finds no mismatches."""
    cat_file = _create_mock_catalog(tmp_path / "train.csv")
    summary = inspect_catalog_csv(cat_file)

    assert summary.total_records == 3
    assert summary.num_unique_participants == 2
    assert summary.num_unique_signs == 2
    assert summary.num_unique_sequences == 3
    assert summary.column_mismatches == []
    assert summary.has_empty_values is False


def test_inspect_catalog_csv_column_mismatch(tmp_path: Path) -> None:
    """Verify catalog inspection detects missing columns."""
    cat_file = _create_mock_catalog(tmp_path / "bad_train.csv", missing_col=True)
    summary = inspect_catalog_csv(cat_file)
    assert len(summary.column_mismatches) > 0
    assert "Missing expected columns" in summary.column_mismatches[0]


def test_inspect_parquet_file_valid(tmp_path: Path) -> None:
    """Verify parquet inspection returns 543 landmarks per frame and NaN rates."""
    pq_file = _create_mock_parquet(tmp_path / "sample.parquet", num_frames=3, lh_nan=True)
    summary = inspect_parquet_file(pq_file)

    assert summary.num_frames == 3
    assert summary.num_rows == 3 * EXPECTED_LANDMARK_COUNT
    assert summary.landmark_counts_by_type["face"] == 468
    assert summary.landmark_counts_by_type["left_hand"] == 21
    assert summary.landmark_counts_by_type["pose"] == 33
    assert summary.landmark_counts_by_type["right_hand"] == 21
    assert summary.left_hand_nan_rate == 1.0
    assert summary.right_hand_nan_rate == 0.0
    assert summary.is_valid is True


def test_inspect_parquet_file_column_mismatch(tmp_path: Path) -> None:
    """Verify parquet file with missing column is marked invalid with mismatches."""
    pq_file = _create_mock_parquet(tmp_path / "corrupt.parquet", corrupt_col=True)
    summary = inspect_parquet_file(pq_file)
    assert summary.is_valid is False
    assert any("Missing columns" in m for m in summary.column_mismatches)


def test_verify_schema_end_to_end_and_report_generation(tmp_path: Path) -> None:
    """Verify end-to-end schema verification and markdown report formatting."""
    cat_file = _create_mock_catalog(tmp_path / "train.csv")
    pq_file = _create_mock_parquet(tmp_path / "p1.parquet", num_frames=4, lh_nan=True)

    report = verify_schema(
        catalog_path=cat_file,
        parquet_paths=[pq_file],
        nested_path_behavior="Flattened files relocated to canonical hierarchy",
    )

    assert report.all_expectations_met is True
    assert report.discrepancies == []
    assert report.frames_per_sequence_stats["min"] == 4.0
    assert report.frames_per_sequence_stats["max"] == 4.0

    # Test markdown report rendering
    md = generate_schema_markdown_report(report)
    assert "# ISLR Dataset Schema Verification Report" in md
    assert "ALL EXPECTATIONS MET (PASS)" in md
    assert "Unique Participants" in md
    assert "Unique Signs" in md
    assert "468" in md
    assert "Left Hand" in md
    assert "100.0%" in md
    assert "Flattened files relocated to canonical hierarchy" in md


def test_verify_schema_flags_discrepancies(tmp_path: Path) -> None:
    """Verify verify_schema aggregates discrepancies when catalog or parquets are mismatched."""
    bad_cat = _create_mock_catalog(tmp_path / "bad.csv", missing_col=True)
    bad_pq = _create_mock_parquet(tmp_path / "bad.parquet", corrupt_col=True)

    report = verify_schema(catalog_path=bad_cat, parquet_paths=[bad_pq])
    assert report.all_expectations_met is False
    assert len(report.discrepancies) >= 2
    md = generate_schema_markdown_report(report)
    assert "MISMATCH DETECTED (INVESTIGATE)" in md
    assert "⚠️" in md


def test_inspect_parquet_file_hand_presence_exact_fractions(tmp_path: Path) -> None:
    """Verify hand presence frame rate where known frames have all NaNs for left/right hand."""
    num_frames = 4
    types_per_frame = ["face"] * 468 + ["left_hand"] * 21 + ["pose"] * 33 + ["right_hand"] * 21
    indices_per_frame = list(range(468)) + list(range(21)) + list(range(33)) + list(range(21))

    frames = [f for f in range(num_frames) for _ in range(EXPECTED_LANDMARK_COUNT)]
    all_types = types_per_frame * num_frames
    all_indices = indices_per_frame * num_frames
    row_ids = [f"{f}-{t}-{idx}" for f in range(num_frames) for t, idx in zip(types_per_frame, indices_per_frame)]

    x_coords = []
    y_coords = []
    z_coords = []

    for f in range(num_frames):
        for t in types_per_frame:
            if t == "left_hand":
                is_nan = f not in (0, 1)  # Valid in frames 0, 1 -> 2/4 = 0.5
            elif t == "right_hand":
                is_nan = f not in (1, 2)  # Valid in frames 1, 2 -> 2/4 = 0.5
            else:
                is_nan = False

            if is_nan:
                x_coords.append(float("nan"))
                y_coords.append(float("nan"))
                z_coords.append(float("nan"))
            else:
                x_coords.append(0.5)
                y_coords.append(0.5)
                z_coords.append(0.5)

    arrays = [
        pa.array(frames, type=pa.int16()),
        pa.array(row_ids, type=pa.string()),
        pa.array(all_types, type=pa.string()),
        pa.array(all_indices, type=pa.int16()),
        pa.array(x_coords, type=pa.float32()),
        pa.array(y_coords, type=pa.float32()),
        pa.array(z_coords, type=pa.float32()),
    ]
    table = pa.Table.from_arrays(arrays, names=list(EXPECTED_PARQUET_COLUMNS))
    pq_file = tmp_path / "presence_test.parquet"
    pq.write_table(table, pq_file)

    summary = inspect_parquet_file(pq_file)
    assert summary.num_frames == 4
    assert summary.left_hand_present_frame_rate == 0.5
    assert summary.right_hand_present_frame_rate == 0.5
    assert summary.left_hand_nan_rate == 0.5
    assert summary.right_hand_nan_rate == 0.5

    # Check report rendering of Section 5
    cat_file = _create_mock_catalog(tmp_path / "train.csv")
    report = verify_schema(catalog_path=cat_file, parquet_paths=[pq_file])
    md = generate_schema_markdown_report(report)
    assert "Frames with Hand Present" in md
    assert "| Left Hand | 50.0% | 50.0% |" in md
    assert "| Right Hand | 50.0% | 50.0% |" in md


def test_schema_report_section_7_all_mirror_or_cache(tmp_path: Path) -> None:
    """Verify Section 7 prints exact required string when all files come from mirror/cache."""
    cat_file = _create_mock_catalog(tmp_path / "train.csv")
    pq_file = _create_mock_parquet(tmp_path / "p1.parquet")

    obs = [
        FetchResult(path=pq_file, source="local_mirror", delivered_as=None, relocated=False),
        FetchResult(path=pq_file, source="cache", delivered_as=None, relocated=False),
    ]
    report = verify_schema(
        catalog_path=cat_file,
        parquet_paths=[pq_file],
        nested_path_behavior=obs,
    )
    md = generate_schema_markdown_report(report)
    assert "Not observed: all files came from local mirror/cache; run once on a clean Colab cache." in md
    assert "verified" not in report.nested_path_behavior.lower()


def test_schema_report_section_7_downloaded_facts(tmp_path: Path) -> None:
    """Verify Section 7 prints observed facts when files were downloaded."""
    cat_file = _create_mock_catalog(tmp_path / "train.csv")
    pq_file = _create_mock_parquet(tmp_path / "1000035562.parquet")

    obs = [
        FetchResult(path=pq_file, source="downloaded", delivered_as="flat", relocated=True),
    ]
    report = verify_schema(
        catalog_path=cat_file,
        parquet_paths=[pq_file],
        nested_path_behavior=obs,
    )
    md = generate_schema_markdown_report(report)
    assert "- 1000035562.parquet: source=downloaded, delivered_as=flat, relocated=True" in md


def test_schema_report_section_6_probe_warning(tmp_path: Path) -> None:
    """Verify Section 6 prints probed file count and warns when < 5 files are probed."""
    cat_file = _create_mock_catalog(tmp_path / "train.csv")
    pqs_1 = [_create_mock_parquet(tmp_path / "p1.parquet")]

    report_1 = verify_schema(catalog_path=cat_file, parquet_paths=pqs_1)
    md_1 = generate_schema_markdown_report(report_1)
    assert "- **Probed Files**: 1" in md_1
    assert "Fewer than 5 files were probed" in md_1

    # 5 files probed: no warning
    pqs_5 = [_create_mock_parquet(tmp_path / f"p_{i}.parquet") for i in range(5)]
    report_5 = verify_schema(catalog_path=cat_file, parquet_paths=pqs_5)
    md_5 = generate_schema_markdown_report(report_5)
    assert "- **Probed Files**: 5" in md_5
    assert "Fewer than 5 files were probed" not in md_5

