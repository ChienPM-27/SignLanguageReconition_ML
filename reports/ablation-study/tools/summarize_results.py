#!/usr/bin/env python3
"""Tổng hợp bảng ablation từ artifacts thật; chỉ dùng Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

RUNS = {
    "F0": ("A", 160), "F0-control": ("A", 160),
    "F1": ("B", 162), "F2": ("B", 162),
    "F3": ("C", 192), "F4": ("C", 212),
}
SEEDS = {42, 43, 44}
METRICS = {
    "top1_accuracy": "val_top1_accuracy",
    "top3_accuracy": "val_top3_accuracy",
    "macro_f1": "val_macro_f1",
    "loss": "val_loss",
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON phải là object: {path}")
    return value


def number(value: object, name: str, probability: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Metric không hợp lệ: {name}={value!r}")
    result = float(value)
    if result < 0 or (probability and result > 1):
        raise ValueError(f"Metric ngoài khoảng: {name}={result}")
    return result


def collect_attempt(path: Path) -> dict:
    meta = read_json(path / "metadata.json")
    run_id = meta.get("run_id")
    if run_id not in RUNS or meta.get("expected_dim") != RUNS[run_id][1]:
        raise ValueError(f"Run/dimension không thuộc campaign: {path}")
    status = meta.get("status")
    if status not in {"COMPLETED", "FAILED", "RUNNING"}:
        raise ValueError(f"Status không hợp lệ: {path}")
    record = {
        "run_id": run_id, "owner": RUNS[run_id][0], "dim": RUNS[run_id][1],
        "status": status, "attempt_path": str(path.resolve()),
        "metadata": meta, "seeds_completed": 0,
    }
    if status != "COMPLETED":
        record["seeds_completed"] = len({p.parent.name for p in (path / "runs").rglob("val_report.json") if p.parent.name in {"seed_42", "seed_43", "seed_44"}})
        return record
    for stage in ("preprocess", "split", "train"):
        if meta.get("stages", {}).get(stage, {}).get("exit_code") != 0:
            raise ValueError(f"Run COMPLETED thiếu stage thành công: {path}/{stage}")
        if not (path / "logs" / (stage + ".log")).is_file():
            raise FileNotFoundError(f"Thiếu log {stage}: {path}")
    for key in ("source_commit", "checkout_commit", "config_sha256", "frozen_split_sha256", "selection_sha256"):
        if not isinstance(meta.get(key), str) or not meta[key]:
            raise ValueError(f"Thiếu provenance {key}: {path}")
    required = (
        "inputs/config.yaml", "inputs/split_manifest.json", "inputs/selection/glosses.csv",
        "effective_config.yaml", "cache_audit/preprocess_meta.json",
        "cache_audit/manifest.csv", "cache_audit/excluded.csv",
        "split_snapshot/train.csv", "split_snapshot/val.csv", "split_snapshot/test.csv",
    )
    for name in required:
        if not (path / name).is_file():
            raise FileNotFoundError(f"Thiếu artifact: {path / name}")
    for name, key in (
        ("inputs/config.yaml", "config_sha256"),
        ("inputs/split_manifest.json", "frozen_split_sha256"),
        ("inputs/selection/glosses.csv", "selection_sha256"),
    ):
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != meta[key]:
            raise ValueError(f"Shared artifact hash lệch: {path / name}")
    summaries = list((path / "runs").rglob("multi_seed_summary.json"))
    if len(summaries) != 1:
        raise ValueError(f"Cần đúng một multi_seed_summary.json trong attempt: {path}")
    summary_path = summaries[0]
    summary = read_json(summary_path)
    items = summary.get("individual_runs", [])
    if not isinstance(items, list) or len(items) != 3:
        raise ValueError(f"Run phải có đủ 3 seed độc lập: {summary_path}")
    seeds = [item.get("seed") for item in items]
    if any(type(seed) is not int for seed in seeds) or set(seeds) != SEEDS:
        raise ValueError(f"Seeds phải đúng 42/43/44, không trùng: {summary_path}")
    if summary.get("num_seeds") != 3 or summary.get("seeds") != [42, 43, 44]:
        raise ValueError(f"Summary seed metadata lệch: {summary_path}")
    if summary.get("git_commit_hash") != meta["checkout_commit"]:
        raise ValueError(f"Git SHA trong summary lệch: {summary_path}")
    values = {key: [] for key in METRICS}
    times = []
    memberships = []
    preprocess_ids = set()
    classes = []
    seed_rows = []
    for seed in sorted(SEEDS):
        seed_dir = summary_path.parent / f"seed_{seed}"
        for name in ("best_model.pt", "history.json", "train_metrics.csv", "val_report.json"):
            if not (seed_dir / name).is_file():
                raise FileNotFoundError(f"Thiếu seed artifact: {seed_dir / name}")
        if (seed_dir / "test_report.json").exists():
            raise ValueError(f"Feature campaign đã evaluate test: {seed_dir}")
        report = read_json(seed_dir / "val_report.json")
        history = read_json(seed_dir / "history.json")
        if report.get("git_commit_hash") != meta["checkout_commit"]:
            raise ValueError(f"Git SHA trong validation report lệch: {seed_dir}")
        prep_id = report.get("preprocess_identity")
        membership = report.get("sample_membership")
        if not isinstance(prep_id, str) or prep_id in ("", "unknown"):
            raise ValueError(f"Thiếu preprocess identity: {seed_dir}")
        if not isinstance(membership, list) or not membership or len(membership) != len(set(membership)):
            raise ValueError(f"Sample membership rỗng/trùng/thiếu: {seed_dir}")
        if not isinstance(report.get("class_names"), list) or not report["class_names"]:
            raise ValueError(f"Thiếu ordered class_names: {seed_dir}")
        memberships.append(sorted(membership))
        preprocess_ids.add(prep_id)
        classes.append(report["class_names"])
        row = {"seed": seed}
        for key in METRICS:
            value = number(report.get(key), key, probability=key != "loss")
            values[key].append(value)
            row[key] = value
        seconds = number(history.get("training_time_sec"), "training_time_sec")
        times.append(seconds)
        row["training_time_sec"] = seconds
        seed_rows.append(row)
    if len(preprocess_ids) != 1 or any(m != memberships[0] for m in memberships) or any(c != classes[0] for c in classes):
        raise ValueError(f"Cohort/schema/vocabulary lệch giữa seeds: {path}")
    cache_meta = read_json(path / "cache_audit/preprocess_meta.json")
    if cache_meta.get("config_hash") not in preprocess_ids:
        raise ValueError(f"Cache identity không khớp val reports: {path}")
    aggregates = {}
    for key, summary_key in METRICS.items():
        mean, std = statistics.mean(values[key]), statistics.pstdev(values[key])
        for suffix, expected in (("_mean", mean), ("_std", std)):
            actual = number(summary.get(summary_key + suffix), summary_key + suffix)
            if abs(actual - expected) > 0.00011:
                raise ValueError(f"Aggregate không khớp per-seed metrics: {summary_path}/{summary_key + suffix}")
        aggregates[key] = {"mean": mean, "std": std}
    record.update({
        "seeds_completed": 3, "metrics": aggregates,
        "training_time_sec_total": sum(times),
        "seed_results": seed_rows, "class_names": classes[0],
        "sample_membership": memberships[0], "preprocess_identity": next(iter(preprocess_ids)),
        "summary_path": str(summary_path.resolve()),
    })
    return record


def generate_report(input_roots: list[Path], output: Path) -> dict:
    attempt_paths = set()
    for root in input_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Input directory không tồn tại: {root}")
        candidates = [root / "metadata.json"] if (root / "metadata.json").is_file() else root.rglob("metadata.json")
        attempt_paths.update(p.parent.resolve() for p in candidates)
    records = {}
    for path in sorted(attempt_paths):
        record = collect_attempt(path)
        run_id = record["run_id"]
        if run_id in records:
            raise ValueError(f"Nhiều attempts cho {run_id}; truyền các attempt paths được team chọn rõ ràng.")
        records[run_id] = record
    if not records:
        raise ValueError("Không tìm thấy artifacts được run_ablation.py tạo.")
    completed = [r for r in records.values() if r["status"] == "COMPLETED"]
    for key in ("source_commit", "config_sha256", "frozen_split_sha256", "selection_sha256"):
        if len({r["metadata"][key] for r in completed}) > 1:
            raise ValueError(f"Campaign trộn {key}; không tổng hợp các protocol khác nhau.")
    if completed and any(r["class_names"] != completed[0]["class_names"] for r in completed):
        raise ValueError("Ordered vocabulary lệch giữa cấu hình.")
    cohort_warning = completed and any(r["sample_membership"] != completed[0]["sample_membership"] for r in completed)
    lines = [
        "# Bảng kết quả ablation tự động", "",
        "Metrics validation [0,1], mean ± population std qua seeds 42/43/44. Training time là tổng 3 seeds, đơn vị phút.",
        "PENDING/FAILED/RUNNING không có mean ± std; bảng không chứa test metrics.", "",
        "| Run | Người | D | Seeds | Val Top-1 | Val Top-3 | Val Macro-F1 | Val loss | Train phút | Trạng thái |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run_id, (owner, dim) in RUNS.items():
        record = records.get(run_id)
        status = record["status"] if record else "PENDING"
        count = record["seeds_completed"] if record else 0
        cells = []
        for key in METRICS:
            aggregate = record["metrics"][key] if status == "COMPLETED" else None
            cells.append(f"{aggregate['mean']:.4f} ± {aggregate['std']:.4f}" if aggregate else "PENDING")
        minutes = f"{record['training_time_sec_total'] / 60:.2f}" if status == "COMPLETED" else "PENDING"
        lines.append(f"| {run_id} | {owner} | {dim} | {count}/3 | " + " | ".join(cells) + f" | {minutes} | {status} |")
    lines.extend(["", "## So sánh Δ Validation Macro-F1", "", "| Comparison | Δ |", "| --- | --- |"])
    for current, previous in (("F0-control", "F0"), ("F1", "F0-control"), ("F2", "F1"), ("F3", "F2"), ("F4", "F3")):
        a, b = records.get(current), records.get(previous)
        delta = "PENDING"
        if a and b and a["status"] == b["status"] == "COMPLETED":
            delta = f"{a['metrics']['macro_f1']['mean'] - b['metrics']['macro_f1']['mean']:+.4f}"
        lines.append(f"| {current} − {previous} | {delta} |")
    lines.extend(["", "## Provenance / artifacts", ""])
    for run_id, record in records.items():
        lines.append(f"- {run_id}: {record['attempt_path']}; status={record['status']}.")
    if completed:
        meta = completed[0]["metadata"]
        for key in ("source_commit", "config_sha256", "frozen_split_sha256", "selection_sha256"):
            lines.append(f"- {key}: {meta[key]}.")
    if cohort_warning:
        lines.extend(["", "**Cohort validation khác nhau giữa cấu hình.** Các số ở trên là full-cohort; cần supplementary matched-cohort comparison trước khi kết luận về feature effect."])
    lines.extend(["", "Chưa chọn winning configuration tự động; cần review per-gloss/signer, exclusions và protocol.", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    result = {"runs": records, "cohort_warning": bool(cohort_warning)}
    output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Bảng Markdown output; tạo JSON bên cạnh")
    args = parser.parse_args()
    try:
        generate_report(args.input_roots, args.output)
    except (ValueError, OSError, TypeError) as error:
        parser.exit(1, f"Không tạo bảng: {error}\n")
    print(f"Đã tạo bảng: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
