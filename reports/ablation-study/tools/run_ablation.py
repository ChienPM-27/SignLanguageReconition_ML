#!/usr/bin/env python3
"""Chạy một cấu hình ablation và giữ stdout/stderr, provenance, artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2, make_archive

EXPERIMENTS = {
    "F0": ("A", 160, "image", "frame", False, False, True),
    "F0-control": ("A", 160, "image", "frame", False, False, False),
    "F1": ("B", 162, "torso_xy", "frame", False, False, False),
    "F2": ("B", 162, "torso_xy", "clip_median", False, False, False),
    "F3": ("C", 192, "torso_xy", "clip_median", True, False, False),
    "F4": ("C", 212, "torso_xy", "clip_median", True, True, False),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_logged(command: list[str], log_path: Path, cwd: Path) -> int:
    """Ghi cả stdout/stderr và exit code, kể cả khi subprocess thất bại."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write("COMMAND_JSON: " + json.dumps(command, ensure_ascii=False) + "\n")
        try:
            with subprocess.Popen(
                command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            ) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                code = process.wait()
        except BaseException as error:
            log.write(f"PROCESS_ERROR: {type(error).__name__}: {error}\n")
            raise
        log.write(f"EXIT_CODE: {code}\n")
        return code


def build_overrides(run_id: str, shared: Path, relative: str) -> list[str]:
    _, _, wrist, palm, tips, bends, scale = EXPERIMENTS[run_id]
    values = {
        "features.set": "hands_pose",
        "features.wrist_reference": wrist,
        "features.palm_scale": palm,
        "features.fingertip_displacement": str(tips).lower(),
        "features.bend_cosines": str(bends).lower(),
        "augmentation.scale_translate.enabled": str(scale).lower(),
        "split.manifest_path": str(shared / "split_manifest.json"),
        "output.selection_dir": str(shared / "selection"),
        "data.cache_dir": relative + "/cache",
        "output.output_dir": relative + "/outputs",
        "output.report_dir": relative + "/reports",
        "run.output_dir": relative + "/runs",
        "run.name": "ablation_" + run_id,
        "training.resume_checkpoint": "null",
        "evaluation.evaluate_test": "false",
    }
    return [item for key, value in values.items() for item in ("--set", f"{key}={value}")]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", choices=EXPERIMENTS)
    parser.add_argument("shared_dir", type=Path, help="Shared config, split, selection và source commit")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    from src.features.schema import PREPROCESS_SCHEMA_VERSION, get_feature_dim
    from src.utils.config import load_config, save_config
    from src.utils.env import detect_env, resolve_paths
    from summarize_results import generate_report

    shared = args.shared_dir.resolve()
    config_path = shared / "config.yaml"
    base_cfg = load_config(config_path=str(config_path))
    base_paths = resolve_paths(detect_env(), base_cfg, create_dirs=False)
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    relative = f"ablation_runtime/{args.run_id}/{attempt_id}"
    attempt = base_paths.persistent_root / relative
    attempt.mkdir(parents=True, exist_ok=False)
    logs = attempt / "logs"
    logs.mkdir()
    metadata = {
        "run_id": args.run_id, "owner": EXPERIMENTS[args.run_id][0],
        "expected_dim": EXPERIMENTS[args.run_id][1], "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt_id": attempt_id, "stages": {},
        "python": sys.version, "platform": platform.platform(),
    }
    stage = "preflight"
    try:
        for relative_input in ("config.yaml", "source_commit.txt", "split_manifest.json", "selection/glosses.csv"):
            source = shared / relative_input
            if not source.is_file():
                raise FileNotFoundError(f"Thiếu shared artifact: {source}")
            destination = attempt / "inputs" / relative_input
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, destination)
        source_commit = (shared / "source_commit.txt").read_text(encoding="utf-8").strip()
        if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
            raise ValueError("source_commit.txt phải chứa full Git SHA gồm 40 ký tự hex.")
        checkout_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        source_paths = ["src", "scripts", "configs", "requirements.txt"]
        subprocess.run(["git", "diff", "--quiet", source_commit, "--", *source_paths], cwd=repo, check=True)
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *source_paths], cwd=repo, text=True)
        if dirty.strip():
            raise ValueError("Source train còn modified/untracked; cần checkout source đã freeze.")
        if PREPROCESS_SCHEMA_VERSION != 4:
            raise ValueError(f"Campaign yêu cầu schema v4, source hiện là v{PREPROCESS_SCHEMA_VERSION}.")
        expected_model = {
            "name": "GRU", "hidden_size": 128, "num_layers": 2,
            "dropout": 0.3, "bidirectional": False, "pooling": "mean_max",
        }
        expected_training = {
            "batch_size": 32, "lr": 0.001, "weight_decay": 0.0001,
            "label_smoothing": 0.1, "epochs": 60, "patience": 15,
            "early_stopping_min_epochs": 5, "seeds": [42, 43, 44], "num_seeds": 3,
        }
        for section, expected in ((base_cfg.model, expected_model), (base_cfg.training, expected_training)):
            for key, value in expected.items():
                if getattr(section, key) != value:
                    raise ValueError(f"Snapshot lệch protocol ở {key}: {getattr(section, key)!r} != {value!r}")
        if base_cfg.preprocess.sequence_length != 32:
            raise ValueError("Snapshot phải dùng preprocess.sequence_length=32.")
        overrides = build_overrides(args.run_id, shared, relative)
        cfg = load_config(config_path=str(config_path), overrides=overrides[1::2])
        if get_feature_dim(cfg) != metadata["expected_dim"]:
            raise ValueError("Feature dimension không khớp cấu hình ablation.")
        metadata.update({
            "source_commit": source_commit, "checkout_commit": checkout_commit,
            "config_sha256": sha256(config_path),
            "frozen_split_sha256": sha256(shared / "split_manifest.json"),
            "selection_sha256": sha256(shared / "selection/glosses.csv"),
            "schema_version": PREPROCESS_SCHEMA_VERSION,
        })
        versions = {}
        for package in ("torch", "numpy", "pandas", "pyarrow", "scikit-learn", "PyYAML"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "NOT_INSTALLED"
        metadata["package_versions"] = versions
        import torch
        metadata["cuda_available"] = torch.cuda.is_available()
        metadata["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        metadata["configured_device"] = cfg.training.device
        save_config(cfg, attempt / "effective_config.yaml")
        (logs / "preflight.log").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_json(attempt / "metadata.json", metadata)
        common = ["--config", str(config_path), *overrides]
        for stage in ("preprocess", "split", "train"):
            script = "scripts/20_train.py" if stage == "train" else "scripts/10_prepare_data.py"
            specific = ["--seeds", "42", "43", "44"] if stage == "train" else ["--stage", stage]
            command = [sys.executable, "-X", "utf8", "-u", script, *specific, *common]
            stage_record = {"command": command, "started_at_utc": datetime.now(timezone.utc).isoformat()}
            metadata["stages"][stage] = stage_record
            write_json(attempt / "metadata.json", metadata)
            code = run_logged(command, logs / f"{stage}.log", repo)
            stage_record.update({"exit_code": code, "ended_at_utc": datetime.now(timezone.utc).isoformat()})
            write_json(attempt / "metadata.json", metadata)
            if code:
                raise RuntimeError(f"Stage {stage} thất bại (exit={code}); log: {logs / (stage + '.log')}")
            paths = resolve_paths(detect_env(), cfg, create_dirs=False)
            if stage == "preprocess":
                source_dir = paths.cache_dir / "features"
                destination_dir = attempt / "cache_audit"
                destination_dir.mkdir()
                for name in ("preprocess_meta.json", "manifest.csv", "excluded.csv"):
                    copy2(source_dir / name, destination_dir / name)
            elif stage == "split":
                source_dir = paths.persistent_root / "data" / "splits"
                destination_dir = attempt / "split_snapshot"
                destination_dir.mkdir()
                for name in ("train.csv", "val.csv", "test.csv", "classes.csv", "label_map.json"):
                    copy2(source_dir / name, destination_dir / name)
        stage = "report"
        metadata["status"] = "COMPLETED"
        metadata["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(attempt / "metadata.json", metadata)
        generate_report([attempt], attempt / "RESULTS.md")
        stage = "archive"
        archive = make_archive(str(attempt), "zip", root_dir=attempt.parent, base_dir=attempt.name)
        print(f"Đã lưu log, artifacts và bảng kết quả: {attempt}", flush=True)
        print(f"Bundle nộp team: {archive}", flush=True)
        return 0
    except BaseException as error:
        metadata.update({
            "status": "FAILED", "failed_stage": stage,
            "error": f"{type(error).__name__}: {error}",
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_json(attempt / "metadata.json", metadata)
        with (logs / "failure.log").open("a", encoding="utf-8") as log:
            log.write(metadata["error"] + "\n")
        try:
            make_archive(str(attempt), "zip", root_dir=attempt.parent, base_dir=attempt.name)
        except OSError as archive_error:
            print(f"Không zip được failed run; thư mục log vẫn giữ: {archive_error}", file=sys.stderr)
        print(f"Run thất bại; log vẫn giữ tại {attempt}: {metadata['error']}", file=sys.stderr)
        if isinstance(error, KeyboardInterrupt):
            return 130
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
