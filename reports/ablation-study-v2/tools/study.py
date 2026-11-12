"""Ma trận V2, config canonical, freeze source và runner giữ provenance/log/artifact."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from shutil import copy2, make_archive
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
SEEDS = (42, 43, 44)
FEATURES = {
    "F0": ("image", "frame", False, False),
    "F1": ("torso_xy", "frame", False, False),
    "F2": ("torso_xy", "clip_median", False, False),
    "F3": ("torso_xy", "clip_median", True, False),
    "F4": ("torso_xy", "clip_median", True, True),
}
PROFILES = {
    "G0": ("GRU", "mean_max", False),
    "G-mean": ("GRU", "mean", False),
    "G-attn": ("GRU", "attention", False),
    "G-bi": ("GRU", "mean_max", True),
    "L0": ("LSTM", "mean_max", False),
    "T0": ("Transformer", "mean_max", False),
}
FEATURE_IDS = tuple(name + suffix for name in FEATURES for suffix in ("", "-face"))


def owner(profile: str, feature: str) -> str:
    if profile == "G0":
        return "C" if feature in ("F0", "F0-face") else "A"
    return "B" if profile.startswith("G-") else "C"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict:
    # Normalize line endings so Windows/Linux checkouts can verify the same text source.
    files = {REPO / "requirements.txt"}
    for folder in ("src", "scripts", "configs", "tests", "reports/ablation-study-v2/tools"):
        files.update(p for p in (REPO / folder).rglob("*") if p.is_file() and p.suffix in (".py", ".yaml"))
    return {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(files)}


def imports():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.utils.config import load_config, save_config
    from src.features.schema import get_feature_dim
    return load_config, save_config, get_feature_dim


def config_for(base_path: Path, profile: str, feature: str):
    load_config, _, _ = imports()
    wrist, palm, tips, bends = FEATURES[feature.removesuffix("-face")]
    model, pooling, bidirectional = PROFILES[profile]
    overrides = {
        "features.set": "hands_pose", "features.wrist_reference": wrist,
        "features.palm_scale": palm, "features.fingertip_displacement": tips,
        "features.bend_cosines": bends, "features.face": feature.endswith("-face"),
        "augmentation.scale_translate.enabled": False,
        "model.name": model, "model.pooling": pooling, "model.bidirectional": bidirectional,
        "training.resume_checkpoint": None, "evaluation.evaluate_test": False,
    }
    return load_config(str(base_path), overrides=[f"{k}={json.dumps(v)}" for k, v in overrides.items()])


def matrix() -> list[dict]:
    _, _, get_dim = imports()
    return [{"run_id": profile + "__" + feature, "profile": profile, "feature": feature,
             "owner": owner(profile, feature), "dimension": get_dim(config_for(REPO / "configs/islr.yaml", profile, feature)),
             "seeds": list(SEEDS)} for profile in PROFILES for feature in FEATURE_IDS]


def member_configs(output: Path, base: Path) -> None:
    """Export immutable, canonical run configs for all three member queues."""
    _, save_config, get_dim = imports()
    if output.exists() and any(p.name != "README.md" for p in output.iterdir()):
        raise FileExistsError(f"Dùng thư mục config mới: {output}")
    rows = matrix()
    prepared = []
    for row in rows:
        cfg = config_for(base, row["profile"], row["feature"])
        validate_protocol(cfg)
        cfg.run.name = row["run_id"]
        cfg.env.project_dir_name += "-v2-member-" + row["owner"].lower()
        prepared.append((dict(row, dimension=get_dim(cfg)), cfg))
    output.mkdir(parents=True, exist_ok=True)
    for member in ("A", "B", "C"):
        folder = output / ("member-" + member.lower())
        folder.mkdir()
        queue = []
        for row, cfg in prepared:
            if row["owner"] == member:
                filename = row["run_id"] + ".yaml"
                save_config(cfg, folder / filename)
                queue.append(dict(row, config=filename))
        write_json(folder / "QUEUE.json", queue)
        lines = [f"# Member {member}: ablation V2", "",
                 f"Branch: `codex/ablation-member-{member.lower()}`.", "",
                 f"{len(queue)} tổ hợp, {len(queue) * len(SEEDS)} lượt training; mỗi YAML chạy đủ seeds42/43/44.", "",
                 "Dùng shared package đã freeze. Chạy các configs tuần tự trên runtime của mình.", "",
                 "| Run ID | D | Config |", "|---|---:|---|"]
        lines += [f"| {row['run_id']} | {row['dimension']} | [{row['config']}]({row['config']}) |" for row in queue]
        first = queue[0]["config"]
        lines += ["", "```bash",
                  f"python reports/ablation-study-v2/tools/study.py run-config configs/ablation-study-v2/member-{member.lower()}/{first} artifacts/ablation-study-v2/shared",
                  "```", "", "Xem hướng dẫn môi trường, freeze và nghiệm thu ở [README chung](../README.md).", ""]
        (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Đã xuất 60 configs / 180 lượt training: {output}")


def planned_cell(cfg) -> tuple[str, str]:
    parts = cfg.run.name.split("__")
    if len(parts) != 2 or parts[0] not in PROFILES or parts[1] not in FEATURE_IDS:
        raise ValueError("run.name phải là profile__feature thuộc ma trận V2.")
    return parts[0], parts[1]


def validate_planned(cfg, expected) -> None:
    validate_protocol(cfg)
    actual, frozen = cfg.to_dict(), expected.to_dict()
    if actual["env"]["seed"] != frozen["env"]["seed"]:
        raise ValueError("Config env.seed khác shared đã freeze.")
    for section in ("features", "preprocess", "model", "training", "augmentation", "evaluation", "split", "selection", "eda"):
        # The runner always binds the frozen split manifest, independent of its runtime path.
        left, right = dict(actual[section]), dict(frozen[section])
        if section == "split":
            left.pop("manifest_path")
            right.pop("manifest_path")
        if left != right:
            raise ValueError(f"Config {section} khác protocol shared đã freeze.")
    for key, value in frozen["data"].items():
        if key not in ("cache_dir", "local_mirror_dir") and actual["data"][key] != value:
            raise ValueError(f"Config data.{key} khác shared đã freeze.")


def run_config(config_path: Path, shared: Path) -> Path:
    load_config, _, _ = imports()
    profile, feature = planned_cell(load_config(str(config_path)))
    return run(profile, feature, shared, planned_config=config_path)


def freeze(shared: Path, config_path: Path, split: Path, selection: Path) -> None:
    _, save_config, _ = imports()
    cfg = config_for(config_path, "G0", "F0")
    validate_protocol(cfg)
    if shared.exists():
        raise FileExistsError(f"Dùng thư mục shared mới để giữ lịch sử: {shared}")
    for path in (split, selection / "glosses.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)
    shared.mkdir(parents=True)
    (shared / "selection").mkdir()
    save_config(cfg, shared / "config.yaml")
    copy2(split, shared / "split_manifest.json")
    copy2(selection / "glosses.csv", shared / "selection/glosses.csv")
    write_json(shared / "source_manifest.json", source_manifest())
    write_json(shared / "matrix.json", matrix())
    print(f"Đã freeze config/source/split/vocabulary vào {shared}")


def validate_protocol(cfg) -> None:
    expected = {"hidden_size": 128, "num_layers": 2, "dropout": 0.3,
                "attention_heads": 4, "feedforward_size": 256}
    if any(getattr(cfg.model, key) != value for key, value in expected.items()):
        raise ValueError("Protocol V2 yêu cầu hidden128, layers2, dropout.3, heads4, feedforward256.")
    if cfg.preprocess.sequence_length != 32 or cfg.training.seeds != list(SEEDS) or cfg.training.num_seeds != 3:
        raise ValueError("Protocol V2 yêu cầu T32 và seeds42/43/44.")
    if cfg.evaluation.evaluate_test or cfg.training.resume_checkpoint:
        raise ValueError("Campaign dùng validation và initialization mới.")


def verify_shared(shared: Path) -> dict:
    for name in ("config.yaml", "split_manifest.json", "selection/glosses.csv", "source_manifest.json"):
        if not (shared / name).is_file():
            raise FileNotFoundError(shared / name)
    if read_json(shared / "source_manifest.json") != source_manifest():
        raise ValueError("Source khác snapshot đã freeze. Không trộn source vào campaign.")
    return {"source_manifest_sha256": sha(shared / "source_manifest.json"),
            "config_sha256": sha(shared / "config.yaml"),
            "split_sha256": sha(shared / "split_manifest.json"),
            "vocabulary_sha256": sha(shared / "selection/glosses.csv")}


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write("COMMAND_JSON: " + json.dumps(command, ensure_ascii=False) + "\n")
        with subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            code = process.wait()
        log.write(f"EXIT_CODE: {code}\n")
    if code:
        raise RuntimeError(f"Stage exit={code}; log: {log_path}")


def check_cohort(split_folder: Path, frozen: Path) -> None:
    manifest = read_json(frozen)
    for name in ("train", "val", "test"):
        with (split_folder / (name + ".csv")).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        ids = [r["sequence_id"] for r in rows]
        expected = manifest["splits"][name]["sequence_ids"]
        if len(ids) != len(set(ids)) or set(ids) != set(expected):
            raise ValueError(f"Cohort {name} khác frozen manifest. Chuẩn bị intersection trước khi train.")


def common_cohort(base: Path, manifests: list[Path], output: Path) -> dict:
    """Intersection of all five geometry levels, preserving original signer assignments."""
    if len(manifests) != len(FEATURES):
        raise ValueError("Cần 5 manifest.csv từ F0/F1/F2/F3/F4 (face off).")
    frozen = read_json(base)
    if frozen.get("schema_version") != 1 or not frozen.get("sequences") or not frozen.get("vocabulary"):
        raise ValueError("Cần split manifest canonical có sequences provenance và vocabulary.")
    intersection = set(frozen["sequences"])
    layouts = set()
    for path in manifests:
        params = read_json(path.parent / "preprocess_meta.json")["parameters"]
        if params.get("schema_version") != 5 or params.get("features.face", False):
            raise ValueError("Cohort dùng cache v5, face off.")
        layouts.add(tuple(params.get("features." + key) for key in ("wrist_reference", "palm_scale", "fingertip_displacement", "bend_cosines")))
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        ids = [r["sequence_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sequence trong {path}")
        for row in rows:
            expected = frozen["sequences"].get(row["sequence_id"])
            if expected and any(str(row[key]) != str(expected[key]) for key in ("participant_id", "gloss", "label") if key in expected):
                raise ValueError(f"Provenance lệch trong {path}: {row['sequence_id']}")
        intersection &= set(ids)
    if layouts != set(FEATURES.values()):
        raise ValueError("Cần đúng 5 layout F0/F1/F2/F3/F4, không lặp cache.")
    cfg = imports()[0](str(REPO / "configs/islr.yaml"))
    for name in ("train", "val", "test"):
        split = frozen["splits"][name]
        selected = [sid for sid in split["sequence_ids"] if sid in intersection]
        participants = {frozen["sequences"][sid]["participant_id"] for sid in selected}
        if participants != set(split["participant_ids"]):
            raise ValueError(f"Intersection làm mất signer ở {name}; bổ sung dữ liệu, không đổi split seed.")
        for gloss in frozen["vocabulary"]:
            count = sum(frozen["sequences"][sid]["gloss"] == gloss for sid in selected)
            if count < cfg.split.min_per_gloss_per_split:
                raise ValueError(f"Intersection thiếu class {gloss} trong {name}.")
        split["sequence_ids"] = selected
    frozen["sequences"] = {sid: value for sid, value in frozen["sequences"].items() if sid in intersection}
    frozen["cohort_audit"] = {"base_manifest_sha256": sha(base),
                              "feature_manifest_sha256": [sha(path) for path in manifests],
                              "kept_sequences": len(intersection)}
    if output.exists():
        raise FileExistsError(output)
    write_json(output, frozen)
    return frozen


def run(profile: str, feature: str, shared: Path, planned_config: Path | None = None) -> Path:
    _, save_config, get_dim = imports()
    from src.utils.env import detect_env, resolve_paths
    from src.features.schema import PREPROCESS_SCHEMA_VERSION, compute_preprocess_identity
    if PREPROCESS_SCHEMA_VERSION != 5:
        raise ValueError("Campaign V2 yêu cầu schema v5.")
    provenance = verify_shared(shared)
    cfg = config_for(shared / "config.yaml", profile, feature)
    if planned_config is not None:
        load_config, _, _ = imports()
        planned = load_config(str(planned_config))
        if planned_cell(planned) != (profile, feature):
            raise ValueError("Config run.name lệch tổ hợp được yêu cầu.")
        validate_planned(planned, cfg)
        cfg = planned
    validate_protocol(cfg)
    paths = resolve_paths(detect_env(), cfg, create_dirs=False)
    runtime = paths.persistent_root / "ablation_v2_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    # Existing split CSVs are shared within a runtime: one study process at a time.
    lock = runtime / ".runtime.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(profile + "__" + feature)
    attempt = runtime / (profile + "__" + feature) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    meta = {"campaign": "v2", "run_id": profile + "__" + feature, "profile": profile,
            "feature": feature, "owner": owner(profile, feature), "dimension": get_dim(cfg),
            "status": "RUNNING", "seeds": list(SEEDS), "stages": {}, **provenance}
    try:
        attempt.mkdir(parents=True)
        (attempt / "logs").mkdir()
        for name in ("config.yaml", "split_manifest.json", "selection/glosses.csv", "source_manifest.json"):
            destination = attempt / "inputs" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(shared / name, destination)
        if planned_config is not None:
            copy2(planned_config, attempt / "inputs/planned_config.yaml")
            meta["planned_config_sha256"] = sha(attempt / "inputs/planned_config.yaml")
        cache_key = hashlib.sha256((provenance["source_manifest_sha256"] + compute_preprocess_identity(cfg)[0]).encode()).hexdigest()[:16]
        cfg.data.cache_dir = str(paths.persistent_root / "ablation_v2_cache" / cache_key / feature)
        meta["cache_dir"] = cfg.data.cache_dir
        cfg.output.output_dir, cfg.output.report_dir = str(attempt / "outputs"), str(attempt / "reports")
        cfg.output.selection_dir = str(attempt / "inputs/selection")
        cfg.run.output_dir, cfg.run.name = str(attempt / "runs"), meta["run_id"]
        cfg.split.manifest_path = str(attempt / "inputs/split_manifest.json")
        save_config(cfg, attempt / "effective_config.yaml")
        meta["effective_config_sha256"] = sha(attempt / "effective_config.yaml")
        import torch
        meta["hardware"] = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        for stage in ("preprocess", "split", "train"):
            write_json(attempt / "metadata.json", meta)
            script = "scripts/20_train.py" if stage == "train" else "scripts/10_prepare_data.py"
            command = [sys.executable, "-X", "utf8", "-u", script, "--config", str(attempt / "effective_config.yaml")]
            if stage != "train":
                command += ["--stage", stage]
            run_logged(command, attempt / "logs" / (stage + ".log"))
            meta["stages"][stage] = {"exit_code": 0}
            if stage == "preprocess":
                (attempt / "cache_audit").mkdir()
                for item in ("preprocess_meta.json", "manifest.csv", "excluded.csv"):
                    copy2(Path(cfg.data.cache_dir) / "features" / item, attempt / "cache_audit" / item)
            if stage == "split":
                split_folder = paths.persistent_root / "data/splits"
                check_cohort(split_folder, attempt / "inputs/split_manifest.json")
                (attempt / "split_snapshot").mkdir()
                for item in ("train.csv", "val.csv", "test.csv", "classes.csv", "label_map.json"):
                    copy2(split_folder / item, attempt / "split_snapshot" / item)
        if verify_shared(shared) != provenance:
            raise ValueError("Shared source/config thay đổi trong lúc train.")
        meta["status"] = "COMPLETED"
    except BaseException as error:
        meta.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if attempt.is_dir():
                write_json(attempt / "metadata.json", meta)
                make_archive(str(attempt), "zip", root_dir=attempt.parent, base_dir=attempt.name)
        finally:
            lock.unlink()
    print(f"Đã lưu attempt và ZIP: {attempt}")
    return attempt


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    m = sub.add_parser("matrix")
    m.add_argument("--owner", choices=("A", "B", "C"))
    m.add_argument("--output", type=Path)
    c = sub.add_parser("config")
    c.add_argument("profile", choices=PROFILES)
    c.add_argument("feature", choices=FEATURE_IDS)
    c.add_argument("output", type=Path)
    c.add_argument("--base", type=Path, default=REPO / "configs/islr.yaml")
    e = sub.add_parser("member-configs")
    e.add_argument("output", type=Path)
    e.add_argument("--base", type=Path, default=REPO / "configs/islr.yaml")
    f = sub.add_parser("freeze")
    f.add_argument("shared", type=Path)
    f.add_argument("--config", type=Path, default=REPO / "configs/islr.yaml")
    f.add_argument("--split", type=Path, required=True)
    f.add_argument("--selection", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("profile", choices=PROFILES)
    r.add_argument("feature", choices=FEATURE_IDS)
    r.add_argument("shared", type=Path)
    p = sub.add_parser("run-config")
    p.add_argument("config", type=Path)
    p.add_argument("shared", type=Path)
    h = sub.add_parser("cohort")
    h.add_argument("base_split", type=Path)
    h.add_argument("output", type=Path)
    h.add_argument("manifests", nargs=5, type=Path)
    args = parser.parse_args()
    if args.action == "matrix":
        rows = [row for row in matrix() if args.owner is None or row["owner"] == args.owner]
        if args.output:
            write_json(args.output, rows)
            print(f"Đã lưu {len(rows)} cells, {len(rows) * len(SEEDS)} lần train: {args.output}")
        else:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
    elif args.action == "config":
        _, save_config, _ = imports()
        save_config(config_for(args.base, args.profile, args.feature), args.output)
    elif args.action == "freeze":
        freeze(args.shared.resolve(), args.config.resolve(), args.split.resolve(), args.selection.resolve())
    elif args.action == "member-configs":
        member_configs(args.output.resolve(), args.base.resolve())
    elif args.action == "run-config":
        run_config(args.config.resolve(), args.shared.resolve())
    elif args.action == "cohort":
        common_cohort(args.base_split, args.manifests, args.output)
        print(f"Đã lưu cohort chung: {args.output}")
    else:
        run(args.profile, args.feature, args.shared.resolve())


if __name__ == "__main__":
    main()
