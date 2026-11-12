"""Tổng hợp ma trận V2 từ artifacts; không chọn winner khi ma trận/cohort chưa đầy đủ."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

from study import FEATURE_IDS, FEATURES, PROFILES, SEEDS, config_for, matrix, read_json, sha, write_json


def finite(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Metric không hợp lệ: {value}")
    return float(value)


def aggregate(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values), "per_seed": values}


def collect(path: Path) -> dict:
    meta = read_json(path / "metadata.json")
    if meta.get("campaign") != "v2" or meta.get("profile") not in PROFILES or meta.get("feature") not in FEATURE_IDS:
        raise ValueError(f"Artifact ngoài campaign V2: {path}")
    if meta.get("run_id") != meta["profile"] + "__" + meta["feature"]:
        raise ValueError(f"Run ID lệch: {path}")
    if meta.get("status") != "COMPLETED":
        return {"run_id": meta["run_id"], "status": meta.get("status"), "path": str(path)}
    for stage in ("preprocess", "split", "train"):
        if meta.get("stages", {}).get(stage, {}).get("exit_code") != 0 or not (path / "logs" / (stage + ".log")).is_file():
            raise ValueError(f"Run thiếu stage thành công: {path}/{stage}")
    for filename, key in (("config.yaml", "config_sha256"), ("split_manifest.json", "split_sha256"),
                          ("selection/glosses.csv", "vocabulary_sha256"), ("source_manifest.json", "source_manifest_sha256")):
        if sha(path / "inputs" / filename) != meta.get(key):
            raise ValueError(f"Input provenance lệch: {path}/{filename}")
    if sha(path / "effective_config.yaml") != meta.get("effective_config_sha256"):
        raise ValueError(f"Effective config bị thay đổi: {path}")
    if "planned_config_sha256" in meta and sha(path / "inputs/planned_config.yaml") != meta["planned_config_sha256"]:
        raise ValueError(f"Planned config bị thay đổi: {path}")
    from src.utils.config import load_config
    from src.features.schema import compute_preprocess_identity, get_feature_dim
    from study import validate_protocol
    cfg = load_config(str(path / "effective_config.yaml"))
    validate_protocol(cfg)
    expected = config_for(path / "inputs/config.yaml", meta["profile"], meta["feature"])
    actual_dict, expected_dict = cfg.to_dict(), expected.to_dict()
    for section in ("features", "preprocess", "model", "training", "augmentation", "evaluation"):
        if actual_dict[section] != expected_dict[section]:
            raise ValueError(f"Protocol/config {section} lệch: {path}")
    if get_feature_dim(cfg) != meta.get("dimension"):
        raise ValueError(f"Dimension lệch: {path}")
    identity = compute_preprocess_identity(cfg)[0]
    cache = read_json(path / "cache_audit/preprocess_meta.json")
    if cache.get("config_hash") != identity:
        raise ValueError(f"Cache identity lệch: {path}")
    frozen = read_json(path / "inputs/split_manifest.json")
    memberships = {}
    for split in ("train", "val", "test"):
        with (path / "split_snapshot" / (split + ".csv")).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        ids = [r["sequence_id"] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(frozen["splits"][split]["sequence_ids"]):
            raise ValueError(f"Cohort lệch: {path}/{split}")
        memberships[split] = sorted(ids)
    reports = list((path / "runs").rglob("val_report.json"))
    if len(reports) != len(SEEDS) or any((path / "runs").rglob("test_report.json")):
        raise ValueError(f"Cần đúng 3 validation reports và chưa evaluate test: {path}")
    ordered = {}
    provenance = {}
    for report_path in reports:
        seed = int(report_path.parent.name.removeprefix("seed_"))
        if seed not in SEEDS or seed in ordered:
            raise ValueError(f"Seed lệch/trùng: {report_path}")
        for artifact in ("best_model.pt", "history.json", "train_metrics.csv"):
            if not (report_path.parent / artifact).is_file():
                raise FileNotFoundError(report_path.parent / artifact)
        report = read_json(report_path)
        predictions = report.get("predictions", [])
        if report.get("preprocess_identity") != identity or sorted(report.get("sample_membership", [])) != memberships["val"]:
            raise ValueError(f"Validation identity/cohort lệch: {report_path}")
        ids = [p["sequence_id"] for p in predictions]
        if len(ids) != len(set(ids)) or sorted(ids) != memberships["val"]:
            raise ValueError(f"Thiếu prediction đối chiếu theo sequence: {report_path}")
        if report.get("class_names") != frozen["vocabulary"]:
            raise ValueError(f"Vocabulary lệch: {report_path}")
        truth = {p["sequence_id"]: (p["participant_id"], p["y_true"]) for p in predictions}
        for sid, (participant, label) in truth.items():
            source = frozen["sequences"][sid]
            if participant != source["participant_id"] or label != source["label"]:
                raise ValueError(f"Prediction provenance lệch: {sid}")
        if provenance and provenance != truth:
            raise ValueError(f"Ground truth lệch giữa seeds: {path}")
        provenance = truth
        for name in ("macro_f1", "top1_accuracy", "top3_accuracy"):
            if not 0 <= finite(report[name]) <= 1:
                raise ValueError(f"Metric ngoài khoảng: {report_path}")
        ordered[seed] = {name: finite(report[name]) for name in ("macro_f1", "top1_accuracy", "top3_accuracy", "loss")}
        ordered[seed]["parameters"] = finite(report["model_parameters"])
        ordered[seed]["training_time_sec"] = finite(read_json(report_path.parent / "history.json")["training_time_sec"])
    if set(ordered) != set(SEEDS):
        raise ValueError(f"Thiếu seed: {path}")
    return {"run_id": meta["run_id"], "profile": meta["profile"], "feature": meta["feature"],
            "status": "COMPLETED", "path": str(path.resolve()), "dimension": meta["dimension"],
            "metrics": {name: aggregate([ordered[s][name] for s in SEEDS]) for name in ordered[SEEDS[0]]},
            "provenance": {key: meta[key] for key in ("source_manifest_sha256", "config_sha256", "split_sha256", "vocabulary_sha256")},
            "memberships": memberships}


def summarize(roots: list[Path], output: Path) -> dict:
    # Ensure lazy source imports are available when invoked as a standalone script.
    from study import imports
    imports()
    records = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        candidates = [root / "metadata.json"] if (root / "metadata.json").is_file() else root.rglob("metadata.json")
        for item in candidates:
            record = collect(item.parent)
            if record["run_id"] in records:
                raise ValueError(f"Duplicate attempt {record['run_id']}; truyền rõ attempt đã chọn.")
            records[record["run_id"]] = record
    expected = [row["run_id"] for row in matrix()]
    done = [r for r in records.values() if r["status"] == "COMPLETED"]
    if done and any(r["provenance"] != done[0]["provenance"] or r["memberships"] != done[0]["memberships"] for r in done):
        raise ValueError("Không so sánh artifacts khác source/config/split/vocabulary/cohort.")
    missing = [run_id for run_id in expected if run_id not in records or records[run_id]["status"] != "COMPLETED"]
    ranked = sorted(done, key=lambda r: (-r["metrics"]["macro_f1"]["mean"], r["metrics"]["macro_f1"]["std"],
                                         r["metrics"]["parameters"]["mean"], r["run_id"]))
    deltas = {}
    def difference(a, b):
        return aggregate([x - y for x, y in zip(records[a]["metrics"]["macro_f1"]["per_seed"], records[b]["metrics"]["macro_f1"]["per_seed"])])
    for profile in PROFILES:
        for base in FEATURES:
            a, b = profile + "__" + base + "-face", profile + "__" + base
            if a in records and b in records and records[a]["status"] == records[b]["status"] == "COMPLETED":
                deltas[a + " minus " + b] = difference(a, b)
        for suffix in ("", "-face"):
            for lower, upper in zip(list(FEATURES)[:-1], list(FEATURES)[1:]):
                a, b = profile + "__" + upper + suffix, profile + "__" + lower + suffix
                if a in records and b in records and records[a]["status"] == records[b]["status"] == "COMPLETED":
                    deltas[a + " minus " + b] = difference(a, b)
    interactions = {}
    for profile in PROFILES:
        if profile == "G0":
            continue
        for feature in FEATURE_IDS:
            a, b = profile + "__" + feature, "G0__" + feature
            if a in records and b in records and records[a]["status"] == records[b]["status"] == "COMPLETED":
                deltas[a + " minus " + b] = difference(a, b)
        for base in FEATURES:
            first = profile + "__" + base + "-face minus " + profile + "__" + base
            second = "G0__" + base + "-face minus G0__" + base
            if first in deltas and second in deltas:
                interactions[profile + " versus G0: face on " + base] = aggregate(
                    [x - y for x, y in zip(deltas[first]["per_seed"], deltas[second]["per_seed"])])
    result = {"status": "COMPLETED" if not missing else "PENDING", "completed_cells": len(done),
              "expected_cells": len(expected), "missing": missing, "winner": ranked[0]["run_id"] if not missing else None,
              "records": records, "paired_macro_f1_deltas": deltas,
              "model_face_interactions": interactions}
    write_json(output.with_suffix(".json"), result)
    lines = ["# Ma trận kiểm chứng chéo V2", "", f"Trạng thái: **{result['status']}** — {len(done)}/60 cells.", "",
             "Winner theo Validation Macro-F1 mean: " + (result["winner"] or "chưa chọn; cần đủ ma trận."), "",
             "| Profile | " + " | ".join(FEATURE_IDS) + " |", "|---|" + "---|" * len(FEATURE_IDS)]
    for profile in PROFILES:
        cells = []
        for feature in FEATURE_IDS:
            record = records.get(profile + "__" + feature)
            metric = record["metrics"]["macro_f1"] if record and record["status"] == "COMPLETED" else None
            cells.append(f"{metric['mean']:.4f} ± {metric['std']:.4f}" if metric else "PENDING")
        lines.append("| " + profile + " | " + " | ".join(cells) + " |")
    lines += ["", "Mỗi cell = 3 seeds42/43/44. Std là population std của 3 seeds, không phải confidence interval.",
              "Winner chỉ tốt nhất trong 60 cấu hình với protocol đã freeze; không chứng minh tối ưu toàn cục.",
              "Per-seed delta và đường dẫn artifacts trong JSON; prediction theo signer trong val_report.json."]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.roots, args.output)
