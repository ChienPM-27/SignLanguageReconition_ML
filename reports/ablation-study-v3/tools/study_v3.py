"""Frozen region ablation protocol, auditable runner and cross-model acceptance."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from shutil import copy2, make_archive
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features.schema import compute_preprocess_identity, get_feature_dim
from src.utils.config import load_config, save_config

SEEDS = (42, 43, 44)
REGIONS = {"H": ["hands"], "B": ["body"], "F": ["face"],
           "HB": ["hands", "body"], "HF": ["hands", "face"],
           "BF": ["body", "face"], "HBF": ["hands", "body", "face"]}
PROFILES = {"G0": ("GRU", "mean_max", False), "G-mean": ("GRU", "mean", False),
            "G-attn": ("GRU", "attention", False), "G-bi": ("GRU", "mean_max", True),
            "L0": ("LSTM", "mean_max", False), "T0": ("Transformer", "mean_max", False)}
SHARED_FILES = ("config.yaml", "split_manifest.json", "selection/glosses.csv", "source_manifest.json",
                "matrix.json", "protocol.json", "freeze.json")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict:
    files = [REPO / "requirements.txt", REPO / "configs/islr.yaml"]
    for folder in ("src", "scripts", "configs/ablation-study-v3", "reports/ablation-study-v3/tools"):
        files.extend(p for p in (REPO / folder).rglob("*") if p.is_file() and p.suffix in (".py", ".yaml", ".json"))
    return {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(set(files))}


def matrix() -> list[dict]:
    result = []
    for profile in PROFILES:
        for feature, regions in REGIONS.items():
            owner = "A" if profile == "G0" else "C" if profile in ("L0", "T0") else "B"
            result.append({"run_id": profile + "__" + feature, "profile": profile, "feature": feature,
                           "regions": regions, "owner": owner, "dimension": 196, "seeds": list(SEEDS),
                           "phase": 1 if profile in ("G0", "L0", "T0") or feature == "HBF" else 2,
                           "config": profile + "__" + feature + ".yaml"})
    return result


def config_for(base: Path, profile: str, feature: str):
    name, pooling, bidirectional = PROFILES[profile]
    overrides = {"features.set": "hands_pose", "features.face": True, "features.wrist_reference": "image",
                 "features.palm_scale": "frame", "features.fingertip_displacement": False,
                 "features.bend_cosines": False, "preprocess.trim_no_hand": False,
                 "preprocess.dominant_hand": "disabled", "model.region_fusion": True,
                 "model.regions": REGIONS[feature], "model.name": name, "model.pooling": pooling,
                 "model.bidirectional": bidirectional, "training.resume_checkpoint": None,
                 "evaluation.evaluate_test": False, "augmentation.scale_translate.enabled": False,
                 "augmentation.flip.swap_hands": True}
    cfg = load_config(str(base), overrides=[f"{k}={json.dumps(v)}" for k, v in overrides.items()])
    cfg.run.name = profile + "__" + feature
    return cfg


def validate_config(cfg, expected) -> None:
    a, b = cfg.to_dict(), expected.to_dict()
    for section in ("features", "preprocess", "model", "training", "augmentation", "evaluation", "selection", "eda", "split"):
        left, right = dict(a[section]), dict(b[section])
        if section == "split":
            left.pop("manifest_path")
            right.pop("manifest_path")
        if left != right:
            raise ValueError(f"Config {section} khác protocol V3 đã freeze.")
    for key in b["data"]:
        if key not in ("cache_dir", "local_mirror_dir") and a["data"][key] != b["data"][key]:
            raise ValueError(f"Config data.{key} khác protocol V3.")
    if cfg.run.name != expected.run.name or cfg.env.seed != expected.env.seed:
        raise ValueError("Run name/seed khác protocol V3.")


def verify_shared(shared: Path) -> dict:
    frozen = read(shared / "freeze.json")
    hashes = {name: sha(shared / name) for name in SHARED_FILES if name != "freeze.json"}
    if frozen.get("campaign") != "v3-regions-701020" or hashes != frozen.get("files"):
        raise ValueError("Shared package hash/protocol không hợp lệ.")
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]
    if frozen.get("freeze_id") != identity or read(shared / "source_manifest.json") != source_manifest():
        raise ValueError("Source hoặc Freeze ID khác snapshot V3.")
    if read(shared / "matrix.json") != matrix():
        raise ValueError("Matrix khác protocol V3.")
    return {"freeze_id": identity, **{name.replace("/", "_").replace(".", "_") + "_sha256": value
                                     for name, value in hashes.items()}}


def check_cohort(folder: Path, frozen: Path) -> dict:
    manifest = read(frozen)
    memberships = {}
    for split in ("train", "val", "test"):
        with (folder / (split + ".csv")).open(encoding="utf-8", newline="") as stream:
            ids = [row["sequence_id"] for row in csv.DictReader(stream)]
        if len(ids) != len(set(ids)) or set(ids) != set(manifest["splits"][split]["sequence_ids"]):
            raise ValueError(f"Cohort {split} khác frozen split.")
        memberships[split] = sorted(ids)
    return memberships


def logged(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", buffering=1) as stream:
        stream.write("COMMAND_JSON: " + json.dumps(command) + "\n")
        with subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                print(line, end="", flush=True)
            code = process.wait()
        stream.write(f"EXIT_CODE: {code}\n")
    if code:
        raise RuntimeError(f"Stage exit={code}; xem {log}")


def run(row: dict, shared: Path, persistent_root: Path, mirror: Path) -> Path:
    provenance = verify_shared(shared)
    cfg = load_config(str(REPO / "configs/ablation-study-v3" / ("member-" + row["owner"].lower()) / row["config"]))
    validate_config(cfg, config_for(shared / "config.yaml", row["profile"], row["feature"]))
    cfg.env.local_root = str(persistent_root)
    cfg.data.local_mirror_dir = str(mirror)
    runtime = persistent_root / "ablation_v3_runtime" / provenance["freeze_id"]
    runtime.mkdir(parents=True, exist_ok=True)
    lock = runtime / ".runtime.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(row["run_id"])
    attempt = runtime / row["run_id"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    meta = {"campaign": "v3-regions-701020", **row, "status": "RUNNING", "provenance": provenance, "stages": {}}
    try:
        attempt.mkdir(parents=True)
        for name in SHARED_FILES:
            destination = attempt / "inputs" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(shared / name, destination)
        identity = compute_preprocess_identity(cfg)[0]
        cfg.data.cache_dir = str(persistent_root / "ablation_v3_cache" / provenance["freeze_id"] / identity)
        cfg.output.output_dir = str(attempt / "outputs")
        cfg.output.report_dir = str(attempt / "reports")
        cfg.output.selection_dir = str(attempt / "inputs/selection")
        cfg.run.output_dir = str(attempt / "runs")
        cfg.split.manifest_path = str(attempt / "inputs/split_manifest.json")
        save_config(cfg, attempt / "effective_config.yaml")
        import torch
        meta["hardware"] = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        for stage in ("preprocess", "split", "train"):
            write(attempt / "metadata.json", meta)
            command = [sys.executable, "-X", "utf8", "-u", "scripts/20_train.py" if stage == "train" else "scripts/10_prepare_data.py",
                       "--config", str(attempt / "effective_config.yaml")]
            if stage != "train":
                command.extend(["--stage", stage])
            logged(command, attempt / "logs" / (stage + ".log"))
            meta["stages"][stage] = {"exit_code": 0}
            if stage == "preprocess":
                for filename in ("preprocess_meta.json", "manifest.csv", "excluded.csv"):
                    (attempt / "cache_audit").mkdir(exist_ok=True)
                    copy2(Path(cfg.data.cache_dir) / "features" / filename, attempt / "cache_audit" / filename)
            elif stage == "split":
                folder = persistent_root / "data/splits"
                check_cohort(folder, shared / "split_manifest.json")
                (attempt / "split_snapshot").mkdir()
                for filename in ("train.csv", "val.csv", "test.csv", "classes.csv", "label_map.json"):
                    copy2(folder / filename, attempt / "split_snapshot" / filename)
        inventory = {p.relative_to(attempt).as_posix(): sha(p) for p in sorted(attempt.rglob("*"))
                     if p.is_file() and p.name != "metadata.json"}
        write(attempt / "ARTIFACTS.json", inventory)
        meta["artifact_inventory_sha256"] = sha(attempt / "ARTIFACTS.json")
        meta["status"] = "COMPLETED"
        write(attempt / "metadata.json", meta)
        collect(attempt)
        return attempt
    except BaseException as exc:
        meta["status"], meta["error"] = "FAILED", f"{type(exc).__name__}: {exc}"
        if attempt.exists():
            write(attempt / "metadata.json", meta)
        raise
    finally:
        if attempt.exists():
            make_archive(str(attempt), "zip", attempt)
        lock.unlink(missing_ok=True)


def aggregate(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values), "per_seed": values}


def collect(path: Path) -> dict:
    import numpy as np
    from src.training.metrics import compute_macro_f1, compute_top_k_accuracy
    meta = read(path / "metadata.json")
    row = next((r for r in matrix() if r["run_id"] == meta.get("run_id")), None)
    if meta.get("campaign") != "v3-regions-701020" or row is None or any(meta.get(k) != v for k, v in row.items()):
        raise ValueError("Attempt khác ma trận V3.")
    if meta.get("status") != "COMPLETED":
        raise ValueError("Attempt chưa COMPLETED.")
    provenance = verify_shared(path / "inputs")
    if meta.get("provenance") != provenance or sha(path / "ARTIFACTS.json") != meta.get("artifact_inventory_sha256"):
        raise ValueError("Attempt provenance/inventory lệch.")
    for relative, checksum in read(path / "ARTIFACTS.json").items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path.resolve()) or sha(target) != checksum:
            raise ValueError(f"Artifact bị thay đổi: {relative}")
    cfg = load_config(str(path / "effective_config.yaml"))
    validate_config(cfg, config_for(path / "inputs/config.yaml", row["profile"], row["feature"]))
    identity = compute_preprocess_identity(cfg)[0]
    if get_feature_dim(cfg) != row["dimension"] or read(path / "cache_audit/preprocess_meta.json")["config_hash"] != identity:
        raise ValueError("Cache identity/dimension lệch.")
    memberships = check_cohort(path / "split_snapshot", path / "inputs/split_manifest.json")
    frozen = read(path / "inputs/split_manifest.json")
    reports = list((path / "runs").rglob("val_report.json"))
    if len(reports) != 3 or any((path / "runs").rglob("test_report.json")):
        raise ValueError("Cần 3 validation reports, chưa mở test.")
    for stage in ("preprocess", "split", "train"):
        if meta["stages"].get(stage, {}).get("exit_code") != 0 or not (path / "logs" / (stage + ".log")).is_file():
            raise ValueError(f"Thiếu stage thành công: {stage}")
    ordered = {}
    for report_path in reports:
        seed = int(report_path.parent.name.removeprefix("seed_"))
        if seed not in SEEDS or seed in ordered:
            raise ValueError("Seed thiếu/trùng/lệch.")
        for artifact in ("best_model.pt", "history.json", "train_metrics.csv"):
            if not (report_path.parent / artifact).is_file():
                raise FileNotFoundError(report_path.parent / artifact)
        report = read(report_path)
        predictions = report.get("predictions", [])
        if report.get("preprocess_identity") != identity or sorted(report.get("sample_membership", [])) != memberships["val"]:
            raise ValueError("Report preprocess identity/membership lệch.")
        ids = [p["sequence_id"] for p in predictions]
        if sorted(ids) != memberships["val"] or len(ids) != len(set(ids)) or report.get("class_names") != frozen["vocabulary"]:
            raise ValueError("Thiếu prediction hoặc vocabulary lệch.")
        for prediction in predictions:
            source = frozen["sequences"][prediction["sequence_id"]]
            if prediction["participant_id"] != source["participant_id"] or prediction["y_true"] != source["label"]:
                raise ValueError("Prediction ground truth lệch frozen cohort.")
        probabilities = np.asarray([p["probabilities"] for p in predictions], dtype=np.float64)
        labels = np.asarray([p["y_true"] for p in predictions])
        predicted = np.asarray([p["y_pred"] for p in predictions])
        if (probabilities.shape != (len(predictions), len(frozen["vocabulary"]))
                or not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=0.00001)
                or not np.array_equal(probabilities.argmax(axis=1), predicted)):
            raise ValueError("Prediction probabilities/argmax không hợp lệ.")
        recomputed = {"macro_f1": compute_macro_f1(labels, predicted, len(frozen["vocabulary"])),
                      "top1_accuracy": compute_top_k_accuracy(labels, probabilities, 1),
                      "top3_accuracy": compute_top_k_accuracy(labels, probabilities, min(3, len(frozen["vocabulary"])))}
        if any(abs(report[key] - value) > 0.000051 for key, value in recomputed.items()):
            raise ValueError("Report metrics lệch predictions.")
        metrics = {key: report[key] for key in ("macro_f1", "top1_accuracy", "top3_accuracy", "loss", "model_parameters")}
        if any(type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value) for value in metrics.values()):
            raise ValueError("Metric không finite.")
        if any(not 0 <= metrics[key] <= 1 for key in ("macro_f1", "top1_accuracy", "top3_accuracy")):
            raise ValueError("Metric ngoài khoảng.")
        metrics["training_time_sec"] = read(report_path.parent / "history.json")["training_time_sec"]
        if not math.isfinite(metrics["training_time_sec"]) or metrics["training_time_sec"] < 0:
            raise ValueError("Training time không hợp lệ.")
        ordered[seed] = metrics
    return {**row, "path": str(path.resolve()), "status": "COMPLETED", "provenance": provenance,
            "memberships": memberships, "metrics": {key: aggregate([ordered[s][key] for s in SEEDS]) for key in ordered[42]}}


def rank(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda r: (-r["metrics"]["macro_f1"]["mean"], r["metrics"]["macro_f1"]["std"],
                                           r["metrics"]["model_parameters"]["mean"], r["run_id"]))


def summarize(attempts: list[Path], output: Path) -> dict:
    rows = [collect(path) for path in attempts]
    records = {row["run_id"]: row for row in rows}
    if len(records) != len(rows):
        raise ValueError("Duplicate selected run; không train/đếm lại baseline.")
    if rows and any(r["provenance"] != rows[0]["provenance"] or r["memberships"] != rows[0]["memberships"] for r in rows):
        raise ValueError("Kết quả khác source/cohort/protocol.")
    required = [row["run_id"] for row in matrix() if row["phase"] == 1]
    missing = [run_id for run_id in required if run_id not in records]
    cross = None
    if not missing:
        chosen = rank([records[p + "__HBF"] for p in PROFILES if p.startswith("G")])[0]["profile"]
        cross_ids = [] if chosen == "G0" else [chosen + "__" + f for f in REGIONS if f != "HBF"]
        cross = {"freeze_id": rows[0]["provenance"]["freeze_id"], "profile": chosen, "run_ids": cross_ids,
                 "selection_rule": "Full-region GRU: mean Macro-F1, lower population std, fewer parameters, run ID",
                 "baseline_metrics": {p: records[p + "__HBF"]["metrics"] for p in PROFILES if p.startswith("G")}}
        missing += [run_id for run_id in cross_ids if run_id not in records]
        required += cross_ids
    if any(run_id not in required for run_id in records):
        raise ValueError("Có phase-2 run chưa được kế hoạch kiểm chứng chọn.")
    deltas = {}
    for profile in PROFILES:
        for smaller, small_regions in REGIONS.items():
            for larger, large_regions in REGIONS.items():
                a, b = profile + "__" + larger, profile + "__" + smaller
                if set(small_regions) < set(large_regions) and a in records and b in records:
                    deltas[a + " minus " + b] = aggregate([x - y for x, y in zip(records[a]["metrics"]["macro_f1"]["per_seed"],
                                                                                                records[b]["metrics"]["macro_f1"]["per_seed"])])
    model_deltas, interactions = {}, {}
    for profile in PROFILES:
        if profile == "G0":
            continue
        for feature in REGIONS:
            a, b = profile + "__" + feature, "G0__" + feature
            if a in records and b in records:
                model_deltas[a + " minus " + b] = aggregate([x - y for x, y in zip(records[a]["metrics"]["macro_f1"]["per_seed"],
                                                                                                records[b]["metrics"]["macro_f1"]["per_seed"])])
        for feature in REGIONS:
            first = profile + "__HBF minus " + profile + "__" + feature
            second = "G0__HBF minus G0__" + feature
            if first in deltas and second in deltas:
                interactions[profile + " versus G0: HBF minus " + feature] = aggregate(
                    [x - y for x, y in zip(deltas[first]["per_seed"], deltas[second]["per_seed"])])
    result = {"status": "COMPLETED" if not missing else "PENDING", "phase1_expected": 24,
              "completed_cells": len(rows), "expected_cells": len(required), "missing": missing,
              "cross_plan": cross, "winner": rank(rows)[0]["run_id"] if not missing else None,
              "records": records, "paired_region_deltas": deltas, "paired_model_deltas": model_deltas,
              "model_region_interactions": interactions}
    write(output.with_suffix(".json"), result)
    lines = ["# Nghiệm thu ablation V3", "", f"{result['status']}: {len(rows)}/{len(required)} configs.", "",
             "Winner: " + (result["winner"] or "chưa đủ vòng đầu/kiểm chứng chéo"), "",
             "| Profile | " + " | ".join(REGIONS) + " |", "|---|" + "---|" * len(REGIONS)]
    for profile in PROFILES:
        cells = []
        for feature in REGIONS:
            item = records.get(profile + "__" + feature)
            metric = item["metrics"]["macro_f1"] if item else None
            cells.append(f"{metric['mean']:.4f} ± {metric['std']:.4f}" if metric else "—")
        lines.append("| " + profile + " | " + " | ".join(cells) + " |")
    lines += ["", "Std là population std của 3 seeds, không phải confidence interval.",
              "Winner chỉ tốt nhất trong các configs của protocol; test chỉ dùng sau khi khóa winner."]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
