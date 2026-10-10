"""Tests for Phase 0 setup check script and schema verification (R5).

Covers catalog CSV reading, parquet schema validation, and end-to-end setup verification.
"""

import importlib
from pathlib import Path
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.config import load_config

setup_check_module = importlib.import_module("scripts.00_setup_check")
read_catalog_csv = setup_check_module.read_catalog_csv
verify_parquet_schema = setup_check_module.verify_parquet_schema
run_setup_check = setup_check_module.run_setup_check
EXIT_CODE_SUCCESS = setup_check_module.EXIT_CODE_SUCCESS

EXPECTED_COLUMNS = ["frame", "row_id", "type", "landmark_index", "x", "y", "z"]
EXPECTED_LANDMARK_COUNT = 543


def test_read_catalog_csv(tmp_path: Path) -> None:
    """Verify reading and parsing of valid train.csv catalog."""
    csv_file = tmp_path / "train.csv"
    csv_file.write_text(
        "path,participant_id,sequence_id,sign\n"
        "train_landmark_files/26734/1000035562.parquet,26734,1000035562,blow\n",
        encoding="utf-8",
    )
    records = read_catalog_csv(csv_file)
    assert len(records) == 1
    assert records[0]["participant_id"] == "26734"
    assert records[0]["sign"] == "blow"


def test_read_catalog_csv_missing_column(tmp_path: Path) -> None:
    """Verify missing required catalog column raises ValueError."""
    csv_file = tmp_path / "train.csv"
    csv_file.write_text("path,participant_id,sign\nfoo,1,bar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing expected column 'sequence_id'"):
        read_catalog_csv(csv_file)


def test_verify_parquet_schema_success(tmp_path: Path) -> None:
    """Verify parquet schema inspection with valid 543 landmarks per frame."""
    target_parquet = tmp_path / "sample.parquet"
    num_frames = 2
    total_landmarks = EXPECTED_LANDMARK_COUNT * num_frames

    frames = [0] * EXPECTED_LANDMARK_COUNT + [1] * EXPECTED_LANDMARK_COUNT
    row_ids = [f"{f}-type-{idx}" for f in (0, 1) for idx in range(EXPECTED_LANDMARK_COUNT)]
    types = ["face"] * total_landmarks
    indices = list(range(EXPECTED_LANDMARK_COUNT)) * num_frames
    coords = [0.5] * total_landmarks

    table = pa.Table.from_arrays(
        [
            pa.array(frames, type=pa.int16()),
            pa.array(row_ids, type=pa.string()),
            pa.array(types, type=pa.string()),
            pa.array(indices, type=pa.int16()),
            pa.array(coords, type=pa.float32()),
            pa.array(coords, type=pa.float32()),
            pa.array(coords, type=pa.float32()),
        ],
        names=EXPECTED_COLUMNS,
    )
    pq.write_table(table, target_parquet)

    summary = verify_parquet_schema(
        parquet_path=target_parquet,
        expected_columns=EXPECTED_COLUMNS,
        num_expected_landmarks=EXPECTED_LANDMARK_COUNT,
    )
    assert summary["num_rows"] == total_landmarks
    assert summary["landmarks_per_frame"] == EXPECTED_LANDMARK_COUNT
    assert summary["schema_verified"] is True


def test_verify_parquet_schema_missing_column(tmp_path: Path) -> None:
    """Verify missing column in parquet raises ValueError."""
    target_parquet = tmp_path / "sample.parquet"
    table = pa.Table.from_arrays(
        [pa.array([0]), pa.array(["row_0"])],
        names=["frame", "row_id"],
    )
    pq.write_table(table, target_parquet)

    with pytest.raises(ValueError, match="missing expected columns"):
        verify_parquet_schema(
            parquet_path=target_parquet,
            expected_columns=EXPECTED_COLUMNS,
            num_expected_landmarks=EXPECTED_LANDMARK_COUNT,
        )


def test_verify_parquet_schema_invalid_landmark_count(tmp_path: Path) -> None:
    """Verify mismatched landmarks per frame raises ValueError."""
    target_parquet = tmp_path / "sample.parquet"
    wrong_count = 100
    table = pa.Table.from_arrays(
        [
            pa.array([0] * wrong_count, type=pa.int16()),
            pa.array([f"0-face-{i}" for i in range(wrong_count)]),
            pa.array(["face"] * wrong_count),
            pa.array(list(range(wrong_count)), type=pa.int16()),
            pa.array([0.5] * wrong_count, type=pa.float32()),
            pa.array([0.5] * wrong_count, type=pa.float32()),
            pa.array([0.5] * wrong_count, type=pa.float32()),
        ],
        names=EXPECTED_COLUMNS,
    )
    pq.write_table(table, target_parquet)

    with pytest.raises(ValueError, match="landmarks per frame"):
        verify_parquet_schema(
            parquet_path=target_parquet,
            expected_columns=EXPECTED_COLUMNS,
            num_expected_landmarks=EXPECTED_LANDMARK_COUNT,
        )


def test_verify_parquet_schema_unequal_frame_sizes(tmp_path: Path) -> None:
    """Verify uneven landmark counts across frames (e.g. 544 in f0, 542 in f1) raises ValueError."""
    target_parquet = tmp_path / "unequal.parquet"
    frames = [0] * 544 + [1] * 542
    total = len(frames)
    table = pa.Table.from_arrays(
        [
            pa.array(frames, type=pa.int16()),
            pa.array([f"row-{i}" for i in range(total)]),
            pa.array(["face"] * total),
            pa.array([0] * total, type=pa.int16()),
            pa.array([0.5] * total, type=pa.float32()),
            pa.array([0.5] * total, type=pa.float32()),
            pa.array([0.5] * total, type=pa.float32()),
        ],
        names=EXPECTED_COLUMNS,
    )
    pq.write_table(table, target_parquet)

    with pytest.raises(ValueError, match="Inconsistent landmarks per frame"):
        verify_parquet_schema(
            parquet_path=target_parquet,
            expected_columns=EXPECTED_COLUMNS,
            num_expected_landmarks=EXPECTED_LANDMARK_COUNT,
        )


def test_verify_parquet_schema_invalid_column_type(tmp_path: Path) -> None:
    """Verify non-float coordinate column type raises TypeError."""
    target_parquet = tmp_path / "bad_types.parquet"
    count = EXPECTED_LANDMARK_COUNT
    table = pa.Table.from_arrays(
        [
            pa.array([0] * count, type=pa.int16()),
            pa.array([f"row-{i}" for i in range(count)]),
            pa.array(["face"] * count),
            pa.array([0] * count, type=pa.int16()),
            pa.array(["not_a_float"] * count),  # Invalid string type for x coordinate
            pa.array([0.5] * count, type=pa.float32()),
            pa.array([0.5] * count, type=pa.float32()),
        ],
        names=EXPECTED_COLUMNS,
    )
    pq.write_table(table, target_parquet)

    with pytest.raises(TypeError, match="Coordinate column 'x' must be floating type"):
        verify_parquet_schema(
            parquet_path=target_parquet,
            expected_columns=EXPECTED_COLUMNS,
            num_expected_landmarks=EXPECTED_LANDMARK_COUNT,
        )


def test_verify_parquet_schema_unknown_landmark_type(tmp_path: Path) -> None:
    """Verify unsupported landmark type raises ValueError."""
    target_parquet = tmp_path / "bad_landmark_type.parquet"
    count = EXPECTED_LANDMARK_COUNT
    table = pa.Table.from_arrays(
        [
            pa.array([0] * count, type=pa.int16()),
            pa.array([f"row-{i}" for i in range(count)]),
            pa.array(["alien_limb"] * count),  # Unknown landmark type
            pa.array([0] * count, type=pa.int16()),
            pa.array([0.5] * count, type=pa.float32()),
            pa.array([0.5] * count, type=pa.float32()),
            pa.array([0.5] * count, type=pa.float32()),
        ],
        names=EXPECTED_COLUMNS,
    )
    pq.write_table(table, target_parquet)

    with pytest.raises(ValueError, match="Unknown landmark type 'alien_limb'"):
        verify_parquet_schema(
            parquet_path=target_parquet,
            expected_columns=EXPECTED_COLUMNS,
            num_expected_landmarks=EXPECTED_LANDMARK_COUNT,
        )


def test_run_setup_check_end_to_end(tmp_path: Path) -> None:
    """Verify full end-to-end execution of setup check against self-contained local mirror."""
    out_dir = tmp_path / "outputs"
    rep_dir = tmp_path / "reports"
    mirror_dir = tmp_path / "mirror"
    mirror_dir.mkdir(parents=True, exist_ok=True)

    rel_parquet_path = "train_landmark_files/26734/1000035562.parquet"
    train_csv = mirror_dir / "train.csv"
    train_csv.write_text(
        "path,participant_id,sequence_id,sign\n"
        f"{rel_parquet_path},26734,1000035562,blow\n",
        encoding="utf-8",
    )

    parquet_file = mirror_dir / Path(rel_parquet_path)
    parquet_file.parent.mkdir(parents=True, exist_ok=True)

    # 543 landmarks per frame: face=468, left_hand=21, pose=33, right_hand=21
    num_frames = 2
    types_per_frame = ["face"] * 468 + ["left_hand"] * 21 + ["pose"] * 33 + ["right_hand"] * 21
    indices_per_frame = list(range(468)) + list(range(21)) + list(range(33)) + list(range(21))

    frames = [0] * 543 + [1] * 543
    all_types = types_per_frame * num_frames
    all_indices = indices_per_frame * num_frames
    row_ids = [f"{f}-{t}-{idx}" for f in (0, 1) for t, idx in zip(types_per_frame, indices_per_frame)]

    # NaN allowed on one hand (left_hand coordinates set to NaN)
    x_coords = []
    y_coords = []
    z_coords = []
    for t in all_types:
        if t == "left_hand":
            x_coords.append(float("nan"))
            y_coords.append(float("nan"))
            z_coords.append(float("nan"))
        else:
            x_coords.append(0.5)
            y_coords.append(0.5)
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
    pq.write_table(table, parquet_file)

    cfg = load_config(
        overrides=[
            f"output.output_dir={out_dir}",
            f"output.report_dir={rep_dir}",
            f"data.local_mirror_dir={mirror_dir}",
            "eda.num_landmark_probe_files=1",
        ]
    )
    exit_code = run_setup_check(cfg)
    assert exit_code == EXIT_CODE_SUCCESS
    assert (rep_dir / "setup_check_report.yaml").is_file()
    assert (rep_dir / "islr_schema.md").is_file()
    schema_report_text = (rep_dir / "islr_schema.md").read_text(encoding="utf-8")
    assert "ISLR Dataset Schema Verification Report" in schema_report_text
    assert "Left Hand" in schema_report_text
    assert "Right Hand" in schema_report_text
    assert (out_dir / "resolved_config.yaml").is_file()


def test_verify_parquet_schema_accepts_str_path(tmp_path: Path) -> None:
    """Verify verify_parquet_schema accepts str path as well as Path object."""
    target_parquet = tmp_path / "str_path.parquet"
    count = EXPECTED_LANDMARK_COUNT
    table = pa.Table.from_arrays(
        [
            pa.array([0] * count, type=pa.int16()),
            pa.array([f"0-face-{i}" for i in range(count)]),
            pa.array(["face"] * count),
            pa.array(list(range(count)), type=pa.int16()),
            pa.array([0.5] * count, type=pa.float32()),
            pa.array([0.5] * count, type=pa.float32()),
            pa.array([0.5] * count, type=pa.float32()),
        ],
        names=EXPECTED_COLUMNS,
    )
    pq.write_table(table, target_parquet)

    summary = verify_parquet_schema(
        parquet_path=str(target_parquet),
        expected_columns=EXPECTED_COLUMNS,
        num_expected_landmarks=EXPECTED_LANDMARK_COUNT,
    )
    assert summary["schema_verified"] is True


def test_read_catalog_csv_accepts_str_path(tmp_path: Path) -> None:
    """Verify read_catalog_csv accepts str path as well as Path object."""
    csv_file = tmp_path / "catalog.csv"
    csv_file.write_text(
        "path,participant_id,sequence_id,sign\n"
        "train_landmark_files/26734/1000035562.parquet,26734,1000035562,blow\n",
        encoding="utf-8",
    )
    records = read_catalog_csv(str(csv_file))
    assert len(records) == 1


def test_read_catalog_csv_empty_path_raises_value_error(tmp_path: Path) -> None:
    """Verify catalog row with empty path column raises ValueError."""
    csv_file = tmp_path / "empty_path.csv"
    csv_file.write_text(
        "path,participant_id,sequence_id,sign\n"
        "   ,26734,1000035562,blow\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contains empty or missing 'path'"):
        read_catalog_csv(csv_file)


def test_verify_parquet_schema_null_values_in_structural_column(tmp_path: Path) -> None:
    """Verify null/missing values in structural columns (e.g. type) raises ValueError."""
    target_parquet = tmp_path / "null_type.parquet"
    count = EXPECTED_LANDMARK_COUNT
    types_with_null = ["face"] * (count - 1) + [None]
    table = pa.Table.from_arrays(
        [
            pa.array([0] * count, type=pa.int16()),
            pa.array([f"0-face-{i}" for i in range(count)]),
            pa.array(types_with_null, type=pa.string()),
            pa.array(list(range(count)), type=pa.int16()),
            pa.array([0.5] * count, type=pa.float32()),
            pa.array([0.5] * count, type=pa.float32()),
            pa.array([0.5] * count, type=pa.float32()),
        ],
        names=EXPECTED_COLUMNS,
    )
    pq.write_table(table, target_parquet)

    with pytest.raises(ValueError, match="Column 'type' in .* contains 1 null/missing values"):
        verify_parquet_schema(
            parquet_path=target_parquet,
            expected_columns=EXPECTED_COLUMNS,
            num_expected_landmarks=EXPECTED_LANDMARK_COUNT,
        )


def test_verify_parquet_schema_negative_frame_index(tmp_path: Path) -> None:
    """Verify negative frame values in parquet raises ValueError."""
    target_parquet = tmp_path / "negative_frame.parquet"
    count = EXPECTED_LANDMARK_COUNT
    table = pa.Table.from_arrays(
        [
            pa.array([-1] * count, type=pa.int16()),
            pa.array([f"-1-face-{i}" for i in range(count)]),
            pa.array(["face"] * count),
            pa.array(list(range(count)), type=pa.int16()),
            pa.array([0.5] * count, type=pa.float32()),
            pa.array([0.5] * count, type=pa.float32()),
            pa.array([0.5] * count, type=pa.float32()),
        ],
        names=EXPECTED_COLUMNS,
    )
    pq.write_table(table, target_parquet)

    with pytest.raises(ValueError, match="Negative frame index found"):
        verify_parquet_schema(
            parquet_path=target_parquet,
            expected_columns=EXPECTED_COLUMNS,
            num_expected_landmarks=EXPECTED_LANDMARK_COUNT,
        )


def test_verify_parquet_schema_negative_landmark_index(tmp_path: Path) -> None:
    """Verify negative landmark index in parquet raises ValueError."""
    target_parquet = tmp_path / "negative_landmark_index.parquet"
    count = EXPECTED_LANDMARK_COUNT
    table = pa.Table.from_arrays(
        [
            pa.array([0] * count, type=pa.int16()),
            pa.array([f"0-face-{i}" for i in range(count)]),
            pa.array(["face"] * count),
            pa.array([-1] + list(range(count - 1)), type=pa.int16()),
            pa.array([0.5] * count, type=pa.float32()),
            pa.array([0.5] * count, type=pa.float32()),
            pa.array([0.5] * count, type=pa.float32()),
        ],
        names=EXPECTED_COLUMNS,
    )
    pq.write_table(table, target_parquet)

    with pytest.raises(ValueError, match="Negative landmark_index found"):
        verify_parquet_schema(
            parquet_path=target_parquet,
            expected_columns=EXPECTED_COLUMNS,
            num_expected_landmarks=EXPECTED_LANDMARK_COUNT,
        )


def test_verify_parquet_schema_dictionary_encoded_string_column(tmp_path: Path) -> None:
    """Verify verify_parquet_schema accepts dictionary-encoded string columns (e.g. type/row_id)."""
    target_parquet = tmp_path / "dict_type.parquet"
    count = EXPECTED_LANDMARK_COUNT
    table = pa.table({
        "frame": pa.array([0] * count, type=pa.int16()),
        "row_id": pa.array([f"0-face-{i}" for i in range(count)]),
        "type": pa.DictionaryArray.from_arrays(pa.array([0] * count), pa.array(["face"])),
        "landmark_index": pa.array(list(range(count)), type=pa.int16()),
        "x": pa.array([0.5] * count, type=pa.float32()),
        "y": pa.array([0.5] * count, type=pa.float32()),
        "z": pa.array([0.5] * count, type=pa.float32()),
    })
    pq.write_table(table, target_parquet)

    summary = verify_parquet_schema(
        parquet_path=target_parquet,
        expected_columns=EXPECTED_COLUMNS,
        num_expected_landmarks=EXPECTED_LANDMARK_COUNT,
    )
    assert summary["schema_verified"] is True


def test_verify_parquet_schema_duplicate_row_ids_raises_value_error(tmp_path: Path) -> None:
    """Verify duplicate row_id values in landmark parquet file raises ValueError."""
    target_parquet = tmp_path / "dup_row_ids.parquet"
    count = EXPECTED_LANDMARK_COUNT
    table = pa.table({
        "frame": pa.array([0] * count, type=pa.int16()),
        "row_id": pa.array(["duplicated_id"] * count),
        "type": pa.array(["face"] * count),
        "landmark_index": pa.array(list(range(count)), type=pa.int16()),
        "x": pa.array([0.5] * count, type=pa.float32()),
        "y": pa.array([0.5] * count, type=pa.float32()),
        "z": pa.array([0.5] * count, type=pa.float32()),
    })
    pq.write_table(table, target_parquet)

    with pytest.raises(ValueError, match="Duplicate row_id values detected"):
        verify_parquet_schema(
            parquet_path=target_parquet,
            expected_columns=EXPECTED_COLUMNS,
            num_expected_landmarks=EXPECTED_LANDMARK_COUNT,
        )


def test_read_catalog_csv_missing_metadata_field_raises_value_error(tmp_path: Path) -> None:
    """Verify catalog row with empty participant_id or sign raises ValueError."""
    csv_file = tmp_path / "missing_meta.csv"
    csv_file.write_text(
        "path,participant_id,sequence_id,sign\n"
        "train_landmark_files/26734/1000035562.parquet,   ,1000035562,blow\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contains empty or missing 'participant_id'"):
        read_catalog_csv(csv_file)

