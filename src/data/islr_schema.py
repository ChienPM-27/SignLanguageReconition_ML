"""ISLR Phase 1 Schema Verification and Inspection Module.

Inspects train.csv catalog and probed parquet landmark files, compares them
against competition expectations, computes missing rates and distribution metrics,
and generates reports/islr_schema.md.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

EXPECTED_CSV_COLUMNS: Tuple[str, ...] = ("path", "participant_id", "sequence_id", "sign")
EXPECTED_PARQUET_COLUMNS: Tuple[str, ...] = (
    "frame",
    "row_id",
    "type",
    "landmark_index",
    "x",
    "y",
    "z",
)
EXPECTED_LANDMARK_TYPES: Tuple[str, ...] = ("face", "left_hand", "pose", "right_hand")
EXPECTED_LANDMARK_COUNTS_PER_TYPE: Dict[str, int] = {
    "face": 468,
    "left_hand": 21,
    "pose": 33,
    "right_hand": 21,
}
TOTAL_EXPECTED_LANDMARKS: int = 543

SCHEMA_LOGGER_NAME = "islr_schema"


@dataclass(frozen=True)
class CatalogSummary:
    """Statistical summary of the train.csv catalog."""
    total_records: int
    num_unique_participants: int
    num_unique_signs: int
    num_unique_sequences: int
    columns: List[str]
    column_mismatches: List[str]
    has_empty_values: bool


@dataclass(frozen=True)
class ParquetSummary:
    """Detailed structural and landmark inspection of a single landmark parquet file."""
    file_path: str
    num_rows: int
    num_frames: int
    columns: List[str]
    column_types: Dict[str, str]
    column_mismatches: List[str]
    landmark_counts_by_type: Dict[str, int]
    landmark_count_mismatches: List[str]
    left_hand_nan_rate: float
    right_hand_nan_rate: float
    face_nan_rate: float
    pose_nan_rate: float
    left_hand_present_frame_rate: float
    right_hand_present_frame_rate: float
    unique_types: List[str]
    type_mismatches: List[str]
    is_valid: bool


@dataclass(frozen=True)
class SchemaVerificationReport:
    """Aggregate comparison between observed data and competition specifications."""
    catalog_summary: CatalogSummary
    parquet_summaries: List[ParquetSummary]
    frames_per_sequence_stats: Dict[str, float]
    nested_path_behavior: str
    all_expectations_met: bool
    discrepancies: List[str]


def inspect_catalog_csv(csv_path: Union[str, Path]) -> CatalogSummary:
    """Read train.csv catalog and extract structural schema and distribution metrics."""
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.is_file():
        raise FileNotFoundError(f"Catalog file does not exist: {csv_path_obj}")

    participants = set()
    signs = set()
    sequences = set()
    has_empty = False
    total_rows = 0

    with open(csv_path_obj, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])

        # Check for column mismatches
        missing_cols = [c for c in EXPECTED_CSV_COLUMNS if c not in columns]
        extra_cols = [c for c in columns if c not in EXPECTED_CSV_COLUMNS]
        col_mismatches = []
        if missing_cols:
            col_mismatches.append(f"Missing expected columns: {missing_cols}")
        if extra_cols:
            col_mismatches.append(f"Unexpected extra columns: {extra_cols}")

        for row in reader:
            total_rows += 1
            for exp_col in EXPECTED_CSV_COLUMNS:
                val = row.get(exp_col)
                if val is None or not str(val).strip():
                    has_empty = True
            p_id = row.get("participant_id")
            s_id = row.get("sequence_id")
            sign = row.get("sign")
            if p_id:
                participants.add(p_id)
            if s_id:
                sequences.add(s_id)
            if sign:
                signs.add(sign)

    return CatalogSummary(
        total_records=total_rows,
        num_unique_participants=len(participants),
        num_unique_signs=len(signs),
        num_unique_sequences=len(sequences),
        columns=columns,
        column_mismatches=col_mismatches,
        has_empty_values=has_empty,
    )


def inspect_parquet_file(parquet_path: Union[str, Path]) -> ParquetSummary:
    """Inspect landmark parquet file for schema, types, landmark counts, and NaN rates."""
    parquet_path_obj = Path(parquet_path)
    if not parquet_path_obj.is_file():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path_obj}")

    table = pq.read_table(parquet_path_obj)
    columns = table.column_names
    num_rows = table.num_rows

    col_types = {name: str(table.schema.field(name).type) for name in columns}

    missing_cols = [c for c in EXPECTED_PARQUET_COLUMNS if c not in columns]
    extra_cols = [c for c in columns if c not in EXPECTED_PARQUET_COLUMNS]
    col_mismatches = []
    if missing_cols:
        col_mismatches.append(f"Missing columns: {missing_cols}")
    if extra_cols:
        col_mismatches.append(f"Extra columns: {extra_cols}")

    # Check frame counts
    frame_col = table.column("frame")
    unique_frames = pc.unique(frame_col)
    num_frames = len(unique_frames)

    # Inspect types and landmark counts per type
    type_col = table.column("type")
    unique_types = sorted(pc.unique(type_col).to_pylist())

    type_mismatches = []
    for t in unique_types:
        if t not in EXPECTED_LANDMARK_TYPES:
            type_mismatches.append(f"Unexpected landmark type: {t}")
    for exp_t in EXPECTED_LANDMARK_TYPES:
        if exp_t not in unique_types:
            type_mismatches.append(f"Missing expected landmark type: {exp_t}")

    # Counts per type per frame (sampling 1 frame or averaging across frames)
    landmark_counts_by_type: Dict[str, int] = {}
    landmark_count_mismatches: List[str] = []

    type_counts_table = pc.value_counts(type_col)
    type_counts_dict = {
        row["values"]: row["counts"]
        for row in type_counts_table.to_pylist()
    }

    for t, total_type_rows in type_counts_dict.items():
        count_per_frame = total_type_rows // max(1, num_frames)
        landmark_counts_by_type[t] = count_per_frame
        expected_count = EXPECTED_LANDMARK_COUNTS_PER_TYPE.get(t)
        if expected_count is not None and count_per_frame != expected_count:
            landmark_count_mismatches.append(
                f"Type '{t}' has {count_per_frame} landmarks/frame, expected {expected_count}"
            )

    # Compute NaN rate per landmark type for coordinate x, y, z
    coord_cols = [c for c in ("x", "y", "z") if c in table.column_names]

    def _compute_nan_rate_for_type(type_name: str) -> float:
        if not coord_cols:
            return 1.0
        mask = pc.equal(table.column("type"), type_name)
        type_filtered = table.filter(mask)
        if type_filtered.num_rows == 0:
            return 1.0
        total_cells = type_filtered.num_rows * len(coord_cols)
        total_nans = 0
        for col_name in coord_cols:
            col = type_filtered.column(col_name)
            null_count = col.null_count
            nan_count = pc.sum(pc.is_nan(col)).as_py() or 0
            total_nans += null_count + nan_count
        return float(total_nans / total_cells)

    lh_nan_rate = _compute_nan_rate_for_type("left_hand")
    rh_nan_rate = _compute_nan_rate_for_type("right_hand")
    face_nan_rate = _compute_nan_rate_for_type("face")
    pose_nan_rate = _compute_nan_rate_for_type("pose")

    def _compute_hand_present_frame_rate(type_name: str) -> float:
        if num_frames <= 0 or not coord_cols or len(coord_cols) < 3:
            return 0.0
        mask = pc.equal(table.column("type"), type_name)
        hand_table = table.filter(mask)
        if hand_table.num_rows == 0:
            return 0.0

        is_valid_row = None
        for coord in ("x", "y", "z"):
            col = hand_table.column(coord)
            not_null = pc.invert(pc.is_null(col))
            not_nan = pc.invert(pc.is_nan(col))
            is_valid_coord = pc.and_(not_null, not_nan)
            is_valid_row = is_valid_coord if is_valid_row is None else pc.and_(is_valid_row, is_valid_coord)

        valid_rows_table = hand_table.filter(is_valid_row)
        if valid_rows_table.num_rows == 0:
            return 0.0

        frame_counts = pc.value_counts(valid_rows_table.column("frame"))
        counts = frame_counts.field("counts")
        expected_cnt = EXPECTED_LANDMARK_COUNTS_PER_TYPE.get(type_name, 21)
        present_mask = pc.equal(counts, expected_cnt)
        present_frames = pc.sum(pc.cast(present_mask, pa.int64())).as_py() or 0
        return float(present_frames / num_frames)

    lh_present_rate = _compute_hand_present_frame_rate("left_hand")
    rh_present_rate = _compute_hand_present_frame_rate("right_hand")

    is_valid = (
        len(col_mismatches) == 0
        and len(type_mismatches) == 0
        and len(landmark_count_mismatches) == 0
    )

    return ParquetSummary(
        file_path=str(parquet_path_obj),
        num_rows=num_rows,
        num_frames=num_frames,
        columns=columns,
        column_types=col_types,
        column_mismatches=col_mismatches,
        landmark_counts_by_type=landmark_counts_by_type,
        landmark_count_mismatches=landmark_count_mismatches,
        left_hand_nan_rate=round(lh_nan_rate, 4),
        right_hand_nan_rate=round(rh_nan_rate, 4),
        face_nan_rate=round(face_nan_rate, 4),
        pose_nan_rate=round(pose_nan_rate, 4),
        left_hand_present_frame_rate=round(lh_present_rate, 4),
        right_hand_present_frame_rate=round(rh_present_rate, 4),
        unique_types=unique_types,
        type_mismatches=type_mismatches,
        is_valid=is_valid,
    )


def format_nested_path_behavior(nested_path_behavior: Any) -> str:
    """Format observed nested-path download facts for Section 7 of schema report."""
    if isinstance(nested_path_behavior, str):
        return nested_path_behavior

    from src.data.kaggle_access import FetchResult

    if isinstance(nested_path_behavior, FetchResult):
        results = [nested_path_behavior]
    elif isinstance(nested_path_behavior, (list, tuple)):
        results = [r for r in nested_path_behavior if isinstance(r, FetchResult)]
        if not results:
            return str(nested_path_behavior)
    else:
        return str(nested_path_behavior)

    if not results or all(r.source in {"local_mirror", "cache"} for r in results):
        return "Not observed: all files came from local mirror/cache; run once on a clean Colab cache."

    downloaded = [r for r in results if r.source == "downloaded"]
    lines = []
    for r in downloaded:
        lines.append(
            f"- {r.path.name}: source={r.source}, delivered_as={r.delivered_as}, relocated={r.relocated}"
        )
    return "\n".join(lines)


def verify_schema(
    catalog_path: Union[str, Path],
    parquet_paths: List[Union[str, Path]],
    nested_path_behavior: Union[str, Any] = "Not observed: all files came from local mirror/cache; run once on a clean Colab cache.",
) -> SchemaVerificationReport:
    """Perform end-to-end schema verification comparing catalog and parquets to specifications."""
    cat_summary = inspect_catalog_csv(catalog_path)
    parquet_summaries = [inspect_parquet_file(p) for p in parquet_paths]

    frame_counts = [ps.num_frames for ps in parquet_summaries]
    if frame_counts:
        frames_stats = {
            "min": float(np.min(frame_counts)),
            "median": float(np.median(frame_counts)),
            "max": float(np.max(frame_counts)),
            "mean": float(np.mean(frame_counts)),
            "std": float(np.std(frame_counts)),
        }
    else:
        frames_stats = {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

    discrepancies: List[str] = []
    if cat_summary.column_mismatches:
        discrepancies.extend(cat_summary.column_mismatches)
    if cat_summary.has_empty_values:
        discrepancies.append("train.csv contains empty or whitespace-only cells in expected columns.")

    for ps in parquet_summaries:
        if ps.column_mismatches:
            discrepancies.extend([f"{ps.file_path}: {m}" for m in ps.column_mismatches])
        if ps.type_mismatches:
            discrepancies.extend([f"{ps.file_path}: {m}" for m in ps.type_mismatches])
        if ps.landmark_count_mismatches:
            discrepancies.extend([f"{ps.file_path}: {m}" for m in ps.landmark_count_mismatches])

    all_expectations_met = len(discrepancies) == 0
    behavior_str = format_nested_path_behavior(nested_path_behavior)

    return SchemaVerificationReport(
        catalog_summary=cat_summary,
        parquet_summaries=parquet_summaries,
        frames_per_sequence_stats=frames_stats,
        nested_path_behavior=behavior_str,
        all_expectations_met=all_expectations_met,
        discrepancies=discrepancies,
    )


def generate_schema_markdown_report(report: SchemaVerificationReport) -> str:
    """Generate reports/islr_schema.md formatted markdown string."""
    cat = report.catalog_summary
    lines = [
        "# ISLR Dataset Schema Verification Report",
        "",
        "## 1. Executive Summary",
        f"- **Verification Status**: {'ALL EXPECTATIONS MET (PASS)' if report.all_expectations_met else 'MISMATCH DETECTED (INVESTIGATE)'}",
        f"- **Catalog Records in train.csv**: {cat.total_records:,}",
        f"- **Unique Participants**: {cat.num_unique_participants}",
        f"- **Unique Signs**: {cat.num_unique_signs}",
        f"- **Unique Sequences**: {cat.num_unique_sequences:,}",
        f"- **Probed Parquet Files**: {len(report.parquet_summaries)}",
        "",
        "## 2. Catalog (train.csv) Schema Verification",
        "| Column | Expected | Observed | Status |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for col in EXPECTED_CSV_COLUMNS:
        status = "Match" if col in cat.columns else "Missing"
        lines.append(f"| `{col}` | Required | `{col}` | {status} |")

    lines.extend([
        "",
        "## 3. Landmark Parquet Schema Verification",
        "| Column | Expected Type | Observed Type | Status |",
        "| :--- | :--- | :--- | :--- |",
    ])

    sample_pq = report.parquet_summaries[0] if report.parquet_summaries else None
    if sample_pq:
        for col in EXPECTED_PARQUET_COLUMNS:
            obs_type = sample_pq.column_types.get(col, "Missing")
            lines.append(f"| `{col}` | Competition Spec | `{obs_type}` | {'Match' if col in sample_pq.columns else 'Mismatch'} |")

        lines.extend([
            "",
            "## 4. Landmark Types & Counts per Frame",
            "| Type | Expected Landmarks/Frame | Observed Landmarks/Frame | Status |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for t, exp_cnt in EXPECTED_LANDMARK_COUNTS_PER_TYPE.items():
            obs_cnt = sample_pq.landmark_counts_by_type.get(t, 0)
            status = "Match" if obs_cnt == exp_cnt else f"Mismatch ({obs_cnt} vs {exp_cnt})"
            lines.append(f"| `{t}` | {exp_cnt} | {obs_cnt} | {status} |")
        lines.append(f"| **Total per Frame** | **{TOTAL_EXPECTED_LANDMARKS}** | **{sum(sample_pq.landmark_counts_by_type.values())}** | **{'Match' if sum(sample_pq.landmark_counts_by_type.values()) == TOTAL_EXPECTED_LANDMARKS else 'Mismatch'}** |")

        lines.extend([
            "",
            "## 5. Missing Data / NaN Rate Analysis",
            "| Subsystem | NaN Rate | Frames with Hand Present | Notes |",
            "| :--- | :--- | :--- | :--- |",
            f"| Left Hand | {sample_pq.left_hand_nan_rate * 100:.1f}% | {sample_pq.left_hand_present_frame_rate * 100:.1f}% | Missing frames allowed when left hand is non-dominant |",
            f"| Right Hand | {sample_pq.right_hand_nan_rate * 100:.1f}% | {sample_pq.right_hand_present_frame_rate * 100:.1f}% | Missing frames allowed when right hand is non-dominant |",
            f"| Face | {sample_pq.face_nan_rate * 100:.1f}% | N/A | Face landmark coverage |",
            f"| Pose | {sample_pq.pose_nan_rate * 100:.1f}% | N/A | Body pose landmark coverage |",
        ])

    lines.extend([
        "",
        "## 6. Frames per Sequence Distribution (Sampled)",
        f"- **Probed Files**: {len(report.parquet_summaries)}",
        f"- **Min Frames**: {report.frames_per_sequence_stats['min']:.1f}",
        f"- **Median Frames**: {report.frames_per_sequence_stats['median']:.1f}",
        f"- **Mean Frames**: {report.frames_per_sequence_stats['mean']:.1f} ± {report.frames_per_sequence_stats['std']:.1f}",
        f"- **Max Frames**: {report.frames_per_sequence_stats['max']:.1f}",
    ])

    if len(report.parquet_summaries) < 5:
        lines.append(
            f"- ⚠️ **Warning**: Fewer than 5 files were probed ({len(report.parquet_summaries)} file(s)). "
            "Sequence length distribution statistics may be limited."
        )

    lines.extend([
        "",
        "## 7. Remote Nested-Path CLI Behavior",
        f"{report.nested_path_behavior}",
        "",
        "## 8. Discrepancies and Mismatches",
    ])

    if report.discrepancies:
        for disc in report.discrepancies:
            lines.append(f"- ⚠️ {disc}")
    else:
        lines.append("- None detected. Observed data strictly conforms to all competition invariants.")

    lines.append("")
    return "\n".join(lines)
