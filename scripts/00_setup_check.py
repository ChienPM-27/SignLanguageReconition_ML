"""ISLR Phase 0 Setup Check and Environment Verification Script.

Inspects runtime environment, canonical paths, GPU availability, and Kaggle credential status.
Idempotently fetches train.csv and exactly one landmark parquet file to verify schema
and nested remote path resolution.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Union
import yaml

SYS_PATH_INSERT_INDEX = 0

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(SYS_PATH_INSERT_INDEX, str(REPO_ROOT))

from src.data.kaggle_access import (
    fetch_file,
    get_credential_status,
)
from src.utils.config import (
    Config,
    parse_cli_args,
    save_config,
)
from src.utils.env import (
    detect_environment,
    resolve_paths,
)
from src.utils.logging import get_logger
from src.utils.seed import set_seed

REPORT_FILENAME = "setup_check_report.yaml"
CONFIG_OUT_FILENAME = "resolved_config.yaml"
FIRST_SAMPLE_INDEX = 0
PATH_COLUMN_NAME = "path"
EXPECTED_CSV_COLUMNS = ("path", "participant_id", "sequence_id", "sign")
EXPECTED_LANDMARK_TYPES = frozenset({"face", "left_hand", "pose", "right_hand"})
EXPECTED_LANDMARK_COLUMNS = frozenset({
    "frame",
    "row_id",
    "type",
    "landmark_index",
    "x",
    "y",
    "z",
})

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1
CUDA_DEVICE_INDEX = 0

REPORT_INDENT = 2
SEPARATOR_LINE_LENGTH = 70
EXPECTED_PROBE_FILE_COUNT = 1
EMPTY_COLLECTION_LENGTH = 0
NO_GPU_DEVICE_COUNT = 0
MIN_VALID_INDEX = 0
SETUP_LOGGER_NAME = "islr_setup"
ERROR_LOGGER_NAME = "islr_setup_error"


def _is_string_type(field_type: Any, pa_module: Any) -> bool:
    """Check if PyArrow field type represents a string, large string, string view, or dictionary string."""
    if pa_module.types.is_string(field_type) or pa_module.types.is_large_string(field_type):
        return True
    if hasattr(pa_module.types, "is_string_view") and pa_module.types.is_string_view(field_type):
        return True
    if pa_module.types.is_dictionary(field_type):
        return _is_string_type(field_type.value_type, pa_module)
    return False


def inspect_gpu() -> Dict[str, Any]:
    """Check GPU availability and device name via PyTorch."""
    try:
        import torch
        available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(CUDA_DEVICE_INDEX) if available else "None (CPU)"
        return {
            "available": available,
            "device_name": device_name,
            "device_count": torch.cuda.device_count() if available else NO_GPU_DEVICE_COUNT,
        }
    except ImportError:
        return {
            "available": False,
            "device_name": "None (PyTorch not available)",
            "device_count": NO_GPU_DEVICE_COUNT,
        }


def read_catalog_csv(csv_path: Union[str, Path]) -> List[Dict[str, str]]:
    """Read train.csv catalog and validate headers and paths."""
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.is_file():
        raise FileNotFoundError(f"Catalog file does not exist: {csv_path_obj}")

    rows: List[Dict[str, str]] = []
    with open(csv_path_obj, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for expected_col in EXPECTED_CSV_COLUMNS:
            if expected_col not in fieldnames:
                raise ValueError(
                    f"Catalog CSV '{csv_path_obj}' missing expected column '{expected_col}'. "
                    f"Found columns: {fieldnames}"
                )
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Catalog CSV '{csv_path_obj}' contains no records.")

    for idx, row in enumerate(rows):
        for expected_col in EXPECTED_CSV_COLUMNS:
            val = row.get(expected_col)
            if val is None or not val.strip():
                raise ValueError(
                    f"Catalog CSV '{csv_path_obj}' row {idx} contains empty or missing '{expected_col}'."
                )

    return rows


def verify_parquet_schema(
    parquet_path: Union[str, Path],
    expected_columns: List[str],
    num_expected_landmarks: int,
) -> Dict[str, Any]:
    """Inspect and verify schema of single probed parquet file."""
    parquet_path_obj = Path(parquet_path)
    if not parquet_path_obj.is_file():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path_obj}")

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path_obj)
    column_names = table.column_names
    num_rows = table.num_rows

    # 1. Verify required competition columns and expected columns are present
    missing_required = [col for col in EXPECTED_LANDMARK_COLUMNS if col not in column_names]
    if missing_required:
        raise ValueError(
            f"Parquet file '{parquet_path_obj}' is missing expected columns: {missing_required}. "
            f"Present columns: {column_names}"
        )

    missing_columns = [col for col in expected_columns if col not in column_names]
    if missing_columns:
        raise ValueError(
            f"Parquet file '{parquet_path_obj}' is missing expected columns: {missing_columns}. "
            f"Present columns: {column_names}"
        )

    # 2. Check for null values in structural columns
    for structural_col in ("frame", "row_id", "type", "landmark_index"):
        col_data = table.column(structural_col)
        if col_data.null_count > EMPTY_COLLECTION_LENGTH:
            raise ValueError(
                f"Column '{structural_col}' in '{parquet_path_obj}' contains {col_data.null_count} null/missing values."
            )

    # 3. Verify column data types
    schema = table.schema
    types_dict = {
        name: str(schema.field(name).type)
        for name in column_names
    }

    if not pa.types.is_integer(schema.field("frame").type):
        raise TypeError(
            f"Column 'frame' must be integer type, got {schema.field('frame').type}"
        )
    if not _is_string_type(schema.field("row_id").type, pa):
        raise TypeError(
            f"Column 'row_id' must be string type, got {schema.field('row_id').type}"
        )
    if not _is_string_type(schema.field("type").type, pa):
        raise TypeError(
            f"Column 'type' must be string type, got {schema.field('type').type}"
        )
    if not pa.types.is_integer(schema.field("landmark_index").type):
        raise TypeError(
            f"Column 'landmark_index' must be integer type, got {schema.field('landmark_index').type}"
        )
    for coord_col in ("x", "y", "z"):
        if not pa.types.is_floating(schema.field(coord_col).type):
            raise TypeError(
                f"Coordinate column '{coord_col}' must be floating type, got {schema.field(coord_col).type}"
            )

    # 4. Verify non-negative indices
    min_frame = pc.min(table.column("frame")).as_py()
    if min_frame is not None and min_frame < MIN_VALID_INDEX:
        raise ValueError(f"Negative frame index found in '{parquet_path_obj}': {min_frame}")

    min_landmark_idx = pc.min(table.column("landmark_index")).as_py()
    if min_landmark_idx is not None and min_landmark_idx < MIN_VALID_INDEX:
        raise ValueError(f"Negative landmark_index found in '{parquet_path_obj}': {min_landmark_idx}")

    # 5. Check landmarks per frame using PyArrow compute value_counts
    frame_col = table.column("frame")
    if len(frame_col) == EMPTY_COLLECTION_LENGTH:
        raise ValueError(f"Parquet file '{parquet_path_obj}' has no rows or frames.")

    frame_value_counts = pc.value_counts(frame_col)
    num_unique_frames = len(frame_value_counts)
    if num_unique_frames == EMPTY_COLLECTION_LENGTH:
        raise ValueError(f"Parquet file '{parquet_path_obj}' has no frames.")

    counts_arr = frame_value_counts.field("counts")
    min_count = pc.min(counts_arr).as_py()
    max_count = pc.max(counts_arr).as_py()

    if min_count != num_expected_landmarks or max_count != num_expected_landmarks:
        raise ValueError(
            f"Inconsistent landmarks per frame in '{parquet_path_obj}': "
            f"expected exactly {num_expected_landmarks} landmarks per frame, "
            f"found frame counts between {min_count} and {max_count}."
        )

    expected_total_rows = num_unique_frames * num_expected_landmarks
    if num_rows != expected_total_rows:
        raise ValueError(
            f"Total rows ({num_rows}) does not match expected {num_unique_frames} frames * "
            f"{num_expected_landmarks} landmarks ({expected_total_rows})."
        )

    # 6. Verify row_id uniqueness
    unique_row_ids = pc.unique(table.column("row_id"))
    if len(unique_row_ids) != num_rows:
        raise ValueError(
            f"Duplicate row_id values detected in '{parquet_path_obj}': "
            f"found {len(unique_row_ids)} unique row_ids across {num_rows} total rows."
        )

    # 6. Check landmark types present
    type_col = table.column("type")
    distinct_types = sorted(pc.unique(type_col).to_pylist())
    for lt in distinct_types:
        if lt not in EXPECTED_LANDMARK_TYPES:
            raise ValueError(
                f"Unknown landmark type '{lt}' in '{parquet_path_obj}'. "
                f"Expected one of: {sorted(EXPECTED_LANDMARK_TYPES)}"
            )

    return {
        "num_rows": num_rows,
        "num_columns": len(column_names),
        "column_names": column_names,
        "column_types": types_dict,
        "num_frames": num_unique_frames,
        "landmarks_per_frame": num_expected_landmarks,
        "landmark_types": distinct_types,
        "schema_verified": True,
    }


def run_setup_check(cfg: Config) -> int:
    """Execute setup verification check."""
    logger = get_logger(name=SETUP_LOGGER_NAME, level=cfg.env.log_level)
    applied_seed = set_seed(cfg.env.seed)

    effective_env = cfg.env.name if cfg.env.name != "auto" else detect_environment()
    paths = resolve_paths(effective_env, cfg)
    gpu_info = inspect_gpu()
    cred_status = get_credential_status()

    logger.info("=" * SEPARATOR_LINE_LENGTH)
    logger.info("ISLR Phase 0 Setup Check")
    logger.info("=" * SEPARATOR_LINE_LENGTH)
    logger.info("Runtime Environment:  %s", paths.env_name)
    logger.info("Random Seed:          %d", applied_seed)
    logger.info("GPU Available:        %s (%s)", gpu_info["available"], gpu_info["device_name"])
    logger.info("Kaggle Credentials:   %s", cred_status)
    logger.info("Repo Root:            %s", paths.repo_root)
    logger.info("Data Directory:       %s", paths.data_dir)
    logger.info("Cache Directory:      %s", paths.cache_dir)
    logger.info("Output Directory:     %s", paths.output_dir)
    logger.info("Report Directory:     %s", paths.report_dir)
    logger.info("=" * SEPARATOR_LINE_LENGTH)

    # 1. Fetch train.csv idempotently
    logger.info("Fetching '%s' catalog...", cfg.data.train_csv_name)
    train_csv_path = fetch_file(
        remote_path=cfg.data.train_csv_name,
        dest_dir=paths.data_dir,
        competition_id=cfg.data.competition_id,
        local_mirror_dir=cfg.data.local_mirror_dir,
        logger=logger,
    )
    catalog_rows = read_catalog_csv(train_csv_path)
    logger.info(
        "Successfully loaded '%s' catalog (%d records).",
        train_csv_path.name,
        len(catalog_rows),
    )

    # 2. Probe exactly 1 parquet file
    num_probe_files = cfg.eda.num_landmark_probe_files
    if num_probe_files != EXPECTED_PROBE_FILE_COUNT:
        raise ValueError(
            f"Phase 0 setup check requires exactly {EXPECTED_PROBE_FILE_COUNT} probe file, configured: {num_probe_files}"
        )

    sample_record = catalog_rows[FIRST_SAMPLE_INDEX]
    sample_remote_path = sample_record[PATH_COLUMN_NAME]
    logger.info(
        "Probing landmark file [1 of %d]: %s",
        num_probe_files,
        sample_remote_path,
    )

    parquet_file_path = fetch_file(
        remote_path=sample_remote_path,
        dest_dir=paths.data_dir,
        competition_id=cfg.data.competition_id,
        local_mirror_dir=cfg.data.local_mirror_dir,
        logger=logger,
    )
    logger.info("Nested path resolved: '%s' -> '%s'", sample_remote_path, parquet_file_path)

    # 3. Verify parquet schema
    schema_summary = verify_parquet_schema(
        parquet_path=parquet_file_path,
        expected_columns=cfg.eda.expected_columns,
        num_expected_landmarks=cfg.eda.num_expected_landmarks,
    )

    logger.info("-" * SEPARATOR_LINE_LENGTH)
    logger.info("Parquet Landmark Schema Summary:")
    logger.info("  Rows:                 %d", schema_summary["num_rows"])
    logger.info("  Frames:               %d", schema_summary["num_frames"])
    logger.info("  Landmarks per Frame:  %d", schema_summary["landmarks_per_frame"])
    logger.info("  Distinct Types:       %s", schema_summary["landmark_types"])
    logger.info("  Columns:              %s", schema_summary["column_names"])
    logger.info("  Types:                %s", schema_summary["column_types"])
    logger.info("  Schema Verified:      %s", schema_summary["schema_verified"])
    logger.info("-" * SEPARATOR_LINE_LENGTH)

    # 4. Save artifacts
    report_data = {
        "setup_check": {
            "status": "SUCCESS",
            "environment": paths.to_dict(),
            "gpu_info": gpu_info,
            "kaggle_credentials_status": cred_status,
            "catalog_path": str(train_csv_path),
            "catalog_records": len(catalog_rows),
            "probed_file": {
                "remote_path": sample_remote_path,
                "local_path": str(parquet_file_path),
                "schema_summary": schema_summary,
            },
        }
    }

    report_file = paths.report_dir / REPORT_FILENAME
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        yaml.dump(report_data, f, indent=REPORT_INDENT, sort_keys=False)
    logger.info("Saved setup verification report to '%s'.", report_file)

    if cfg.output.save_resolved_config:
        resolved_cfg_file = paths.output_dir / CONFIG_OUT_FILENAME
        save_config(cfg, resolved_cfg_file)
        logger.info("Saved resolved configuration to '%s'.", resolved_cfg_file)

    logger.info("Setup check completed successfully!")
    return EXIT_CODE_SUCCESS


def main() -> None:
    cfg, _ = parse_cli_args()
    try:
        exit_code = run_setup_check(cfg)
        sys.exit(exit_code)
    except Exception as exc:
        logger = get_logger(name=ERROR_LOGGER_NAME)
        logger.error("Setup check failed: %s", exc, exc_info=True)
        sys.exit(EXIT_CODE_FAILURE)


if __name__ == "__main__":
    main()
