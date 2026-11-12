"""Notebook workflow: frozen data, member queues, persistent results and audit bundles."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from shutil import copy2, copytree
import stat
import sys
import zipfile

import study

SHARED_FILES = ("config.yaml", "split_manifest.json", "selection/glosses.csv", "source_manifest.json", "matrix.json")
RUNTIME_KEYS = {"env.use_drive", "env.project_dir_name", "env.local_root", "env.log_level", "data.local_mirror_dir"}


def _relative(root: Path, path: str) -> Path:
    normalized = PurePosixPath(path.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts or (normalized.parts and ":" in normalized.parts[0]):
        raise ValueError(f"Đường dẫn archive/state không hợp lệ: {path}")
    result = (root / str(normalized)).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Đường dẫn nằm ngoài folder kết quả: {path}")
    return result


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def discover(inputs: list[Path], pattern: str) -> Path:
    candidates = sorted({p.resolve() for root in inputs if root.exists() for p in root.rglob(pattern) if p.is_file()})
    if len(candidates) != 1:
        raise FileNotFoundError(f"Cần đúng một input {pattern}; tìm thấy {len(candidates)}. Đặt đường dẫn input rõ ràng. Candidates: {candidates}")
    return candidates[0]


def _package_payloads(source: Path, names: tuple[str, ...]) -> dict[str, bytes]:
    if source.is_dir():
        return {name: (source / name).read_bytes() for name in names}
    with zipfile.ZipFile(source) as archive:
        candidates = [name.removesuffix(names[0]) for name in archive.namelist()
                      if name.endswith(names[0]) and all(name.removesuffix(names[0]) + item in archive.namelist() for item in names)]
        if len(candidates) != 1:
            raise ValueError("ZIP cần chứa đúng một shared package V2 đầy đủ.")
        prefix = candidates[0]
        return {name: archive.read(prefix + name) for name in names}


def _extract_results(source: Path, target: Path) -> None:
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Không ghi đè folder kết quả đã có: {target}")
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        paths = [_relative(target, item.filename) for item in members]
        if len(paths) != len(set(paths)):
            raise ValueError("ZIP có đường dẫn trùng nhau.")
        if any(stat.S_ISLNK(item.external_attr >> 16) for item in members):
            raise ValueError("Không nhận symlink trong result ZIP.")
        for item, path in zip(members, paths):
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(item))


def _read_baseline(source: Path) -> dict:
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            payloads = {archive.read(name) for name in archive.namelist() if name.endswith("split_manifest.json")}
        if len(payloads) != 1:
            raise ValueError("Baseline ZIP cần có một frozen split duy nhất; đặt file split_manifest.json rõ ràng nếu có nhiều splits.")
        return json.loads(payloads.pop())
    return study.read_json(source)


def validate_dataset_manifest(data: dict, cfg) -> None:
    vocabulary = data.get("vocabulary", [])
    if data.get("schema_version") != 1 or len(vocabulary) != cfg.selection.n_glosses or len(vocabulary) != len(set(vocabulary)):
        raise ValueError("Frozen split cần schema1 và đủ ordered vocabulary theo shared config.")
    if data.get("label_map") != {gloss: index for index, gloss in enumerate(vocabulary)}:
        raise ValueError("Label map/vocabulary order không canonical.")
    sequences = data.get("sequences", {})
    used_ids, used_signers = set(), set()
    for name in ("train", "val", "test"):
        split = data.get("splits", {}).get(name, {})
        ids = split.get("sequence_ids", [])
        if not ids or len(ids) != len(set(ids)) or used_ids.intersection(ids):
            raise ValueError(f"Membership trống/trùng/leakage ở {name}.")
        rows = [sequences[sid] for sid in ids]
        signers = {str(row["participant_id"]) for row in rows}
        if signers != {str(signer) for signer in split.get("participant_ids", [])} or used_signers.intersection(signers):
            raise ValueError(f"Signer assignment/leakage ở {name}.")
        for index, gloss in enumerate(vocabulary):
            matched = [row for row in rows if row["gloss"] == gloss]
            if len(matched) < cfg.split.min_per_gloss_per_split or any(row["label"] != index for row in matched):
                raise ValueError(f"Thiếu class hoặc label lệch: {name}/{gloss}.")
        if any(row["gloss"] not in vocabulary for row in rows):
            raise ValueError("Frozen split có gloss ngoài vocabulary.")
        used_ids.update(ids)
        used_signers.update(signers)
    if used_ids != set(sequences):
        raise ValueError("Sequences provenance khác membership train/val/test.")


class MemberSession:
    def __init__(self, member: str, runtime_overrides: dict | None = None):
        if member not in ("A", "B", "C"):
            raise ValueError("Member phải là A/B/C.")
        self.member = member
        self.queue = study.read_json(study.REPO / f"configs/ablation-study-v2/member-{member.lower()}/QUEUE.json")
        self.runtime_overrides = dict(runtime_overrides or {})
        load_config, _, _ = study.imports()
        # Validate unknown keys canonically before restricting known overrides to runtime paths.
        load_config(overrides=[f"{key}={json.dumps(value)}" for key, value in self.runtime_overrides.items()])
        if set(self.runtime_overrides) - RUNTIME_KEYS:
            raise ValueError("Notebook chỉ nhận runtime path overrides; hyperparameters dùng YAML đã chuẩn hóa.")
        from src.utils.env import detect_env, resolve_paths
        self.env_name = detect_env()
        defaults = {}
        if self.env_name == "colab":
            defaults["env.use_drive"] = True
        elif self.env_name == "local":
            defaults["env.local_root"] = f"artifacts/ablation-study-v2/member-{member.lower()}"
        defaults.update(self.runtime_overrides)
        self.runtime_overrides = defaults
        base = study.REPO / f"configs/ablation-study-v2/member-{member.lower()}/{self.queue[0]['config']}"
        cfg = load_config(str(base), overrides=[f"{key}={json.dumps(value)}" for key, value in defaults.items()])
        paths = resolve_paths(self.env_name, cfg, create_dirs=True)
        self.persistent_root = paths.persistent_root
        code_json = json.dumps(study.source_manifest(), sort_keys=True, separators=(",", ":"))
        self.code_id = hashlib.sha256(code_json.encode()).hexdigest()[:12]
        self.root = self.persistent_root / "ablation_v2_results" / f"member-{member.lower()}" / self.code_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.shared = self.root / "shared"
        self.state_path = self.root / "queue_status.json"
        self.state = {"member": member, "code_id": self.code_id, "selected": {}, "failed": {}, "provenance": None}
        if self.state_path.is_file():
            self.state = study.read_json(self.state_path)
        self._validate_state()
        print(f"Member {member}: {len(self.queue)} tổ hợp / {len(self.queue) * 3} lượt training")
        print(f"Runtime: {self.env_name}; results folder: {self.root}")

    def _validate_state(self) -> None:
        allowed = {row["run_id"] for row in self.queue}
        if self.state.get("member") != self.member or self.state.get("code_id") != self.code_id:
            raise ValueError("Result package khác member/source; không trộn campaign.")
        if (set(self.state.get("selected", {})) | set(self.state.get("failed", {}))) - allowed:
            raise ValueError("Queue state có run ngoài phân công.")
        for path in self.state.get("selected", {}).values():
            _relative(self.root, path)
        for paths in self.state.get("failed", {}).values():
            for path in paths:
                _relative(self.root, path)

    def runtime_config(self, base: Path, destination: Path):
        load_config, save_config, _ = study.imports()
        cfg = load_config(str(base), overrides=[f"{key}={json.dumps(value)}" for key, value in self.runtime_overrides.items()])
        save_config(cfg, destination)
        return cfg

    def check_shared(self) -> dict:
        provenance = study.verify_shared(self.shared)
        cfg = study.imports()[0](str(self.shared / "config.yaml"))
        study.validate_protocol(cfg)
        split = study.read_json(self.shared / "split_manifest.json")
        validate_dataset_manifest(split, cfg)
        with (self.shared / "selection/glosses.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [(row["gloss"], int(row["label"])) for row in rows] != list(zip(split["vocabulary"], range(len(split["vocabulary"])))):
            raise ValueError("Shared vocabulary CSV khác split/class order.")
        if study.read_json(self.shared / "matrix.json") != study.matrix():
            raise ValueError("Shared matrix khác ma trận V2 hiện tại.")
        if self.state["provenance"] is not None and self.state["provenance"] != provenance:
            raise ValueError("Shared config/split/vocabulary khác kết quả đã lưu.")
        self.state["provenance"] = provenance
        _atomic_json(self.state_path, self.state)
        return provenance

    def import_shared(self, source: Path) -> Path:
        payloads = _package_payloads(source, SHARED_FILES)
        if self.shared.exists():
            if any((self.shared / name).read_bytes() != content for name, content in payloads.items()):
                raise FileExistsError("Shared package đã có khác nội dung input; không ghi đè.")
        else:
            for name, content in payloads.items():
                target = _relative(self.shared, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        self.check_shared()
        print(f"Shared package hợp lệ: {self.shared}")
        return self.shared

    def ensure_data(self, base: Path, manifest: dict) -> None:
        from src.utils.env import resolve_paths
        cfg = self.runtime_config(base, self.root / "runtime_configs/data.yaml")
        if self.shared.exists():
            cfg.output.selection_dir = str(self.shared / "selection")
        study.imports()[1](cfg, self.root / "runtime_configs/data.yaml")
        paths = resolve_paths(self.env_name, cfg, create_dirs=True)
        mirror = Path(cfg.data.local_mirror_dir) if cfg.data.local_mirror_dir else paths.data_dir
        catalog = mirror / cfg.data.train_csv_name
        available = {}
        if catalog.is_file():
            with catalog.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    sid = str(row["sequence_id"]).strip()
                    if sid in manifest["sequences"]:
                        if sid in available:
                            raise ValueError(f"Duplicate catalog sequence: {sid}")
                        expected = manifest["sequences"][sid]
                        if str(row["participant_id"]).strip() != str(expected["participant_id"]) or row["sign"].strip() != expected["gloss"]:
                            raise ValueError(f"Raw catalog provenance lệch: {sid}")
                        path = mirror / row["path"].strip()
                        if path.is_file():
                            available[sid] = path
        if set(available) == set(manifest["sequences"]):
            print(f"Raw data sẵn có: {len(available)} frozen sequences.")
            return
        # No SDK import occurs until credentials are validated by the canonical helper.
        from src.data.kaggle_access import KaggleCredentialsNotFoundError, get_kaggle_credentials
        credentials = get_kaggle_credentials(allow_interactive=False)
        if credentials is None:
            missing = len(set(manifest["sequences"]) - set(available))
            raise KaggleCredentialsNotFoundError(f"Thiếu {missing} raw sequences. Attach asl-signs đầy đủ hoặc cấu hình Kaggle Secrets để download.")
        import os
        if credentials.token:
            os.environ["KAGGLE_API_TOKEN"] = credentials.token
        else:
            os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = credentials.username, credentials.key
        self._stage(self.root / "runtime_configs/data.yaml", "download", self.root / "logs/download.log")
        download_manifest = paths.data_dir / "download_manifest.csv"
        if not download_manifest.is_file():
            download_manifest = paths.persistent_root / "download_manifest.csv"
        with download_manifest.open(encoding="utf-8", newline="") as stream:
            downloaded = {row["sequence_id"] for row in csv.DictReader(stream) if row["status"] != "failed" and Path(row["local_path"]).is_file()}
        if set(manifest["sequences"]) - downloaded:
            raise FileNotFoundError("Download chưa đủ raw frozen cohort; chưa bắt đầu training.")

    def _stage(self, config: Path, stage: str, log: Path) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        study.run_logged([sys.executable, "-X", "utf8", "-u", "scripts/10_prepare_data.py", "--config", str(config), "--stage", stage], log)

    def prepare_shared(self, baseline: Path) -> Path:
        if self.member != "A":
            raise ValueError("Chỉ A chuẩn bị common cohort/shared package; B/C nhận cùng package.")
        data = _read_baseline(baseline)
        cfg = study.config_for(study.REPO / "configs/islr.yaml", "G0", "F0")
        validate_dataset_manifest(data, cfg)
        preparation = self.root / "preparation"
        preparation.mkdir(exist_ok=True)
        base = preparation / "base_split_manifest.json"
        study.write_json(base, data)
        selection = preparation / "selection"
        selection.mkdir(exist_ok=True)
        with (selection / "glosses.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["gloss", "label"])
            writer.writerows((gloss, index) for index, gloss in enumerate(data["vocabulary"]))
        if self.shared.exists():
            self.check_shared()
            audit = study.read_json(self.shared / "split_manifest.json").get("cohort_audit", {})
            if audit.get("base_manifest_sha256") != study.sha(base):
                raise ValueError("Baseline split khác package đã chuẩn bị.")
            return self.export_shared()
        manifests = []
        for feature in study.FEATURES:
            probe = preparation / feature
            probe.mkdir(exist_ok=True)
            generated = probe / "config.yaml"
            current = study.config_for(study.REPO / "configs/islr.yaml", "G0", feature)
            current.env.project_dir_name = study.imports()[0](str(study.REPO / "configs/ablation-study-v2/member-a/G0__F1.yaml")).env.project_dir_name
            study.imports()[1](current, generated)
            current = self.runtime_config(generated, generated)
            current.data.cache_dir = str(probe / "cache")
            current.output.output_dir, current.output.report_dir = str(probe / "outputs"), str(probe / "reports")
            current.output.selection_dir = str(selection)
            study.imports()[1](current, generated)
            if not manifests:
                self.ensure_data(generated, data)
            self._stage(generated, "preprocess", probe / "preprocess.log")
            manifests.append(probe / "cache/features/manifest.csv")
        common = preparation / "common_split_manifest.json"
        if common.exists():
            audit = study.read_json(common).get("cohort_audit", {})
            if audit.get("base_manifest_sha256") != study.sha(base) or audit.get("feature_manifest_sha256") != [study.sha(path) for path in manifests]:
                raise ValueError("Common cohort đã lưu khác probes/baseline; không ghi đè lịch sử chuẩn bị.")
        else:
            study.common_cohort(base, manifests, common)
        study.freeze(self.shared, study.REPO / "configs/islr.yaml", common, selection)
        self.check_shared()
        return self.export_shared()

    def export_shared(self) -> Path:
        self.check_shared()
        target = self.root.parent / f"ablation_v2_shared_{self.code_id}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in SHARED_FILES:
                archive.write(self.shared / name, name)
        print(f"Shared ZIP giao A/B/C: {target}")
        return target

    def selected_attempts(self) -> list[Path]:
        from summarize import collect
        result = []
        for run_id, relative in self.state["selected"].items():
            path = _relative(self.root, relative)
            record = collect(path)
            if record["status"] != "COMPLETED" or record["run_id"] != run_id or record["provenance"] != self.state["provenance"]:
                raise ValueError(f"Selected attempt không hợp lệ: {run_id}")
            result.append(path)
        return result

    def export_results(self) -> Path:
        self._validate_state()
        target = self.root.parent / f"member_{self.member.lower()}_{self.code_id}_results.zip"
        temporary = target.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.root.rglob("*")):
                relative = path.relative_to(self.root)
                if path.is_file() and relative.parts[0] not in ("preparation", "bundles", "aggregate_imports") and not path.name.endswith(".tmp"):
                    archive.write(path, relative.as_posix())
        temporary.replace(target)
        print(f"Results ZIP: {target}")
        return target

    def restore_results(self, source: Path) -> None:
        if self.state["selected"] or self.state["failed"] or self.shared.exists():
            raise FileExistsError("Restore vào runtime mới trước khi nhận shared/training; không ghi đè kết quả.")
        _extract_results(source, self.root)
        self.state = study.read_json(self.state_path)
        self._validate_state()
        self.check_shared()
        self.selected_attempts()
        print(f"Đã restore {len(self.state['selected'])} tổ hợp hợp lệ.")

    def report(self) -> dict:
        self.check_shared()
        selected = self.selected_attempts()
        if selected:
            from summarize import summarize
            summarize(selected, self.root / "MEMBER_RESULTS.md")
        rows = [{"run_id": row["run_id"], "feature": row["feature"], "profile": row["profile"],
                 "status": "COMPLETED" if row["run_id"] in self.state["selected"] else "FAILED" if row["run_id"] in self.state["failed"] else "PENDING"}
                for row in self.queue]
        result = {"member": self.member, "completed": len(selected), "expected": len(self.queue), "queue": rows}
        study.write_json(self.root / "MEMBER_STATUS.json", result)
        return result

    def run_queue(self, run_ids: list[str] | None = None, retry_failed: bool = False) -> dict:
        self.check_shared()
        allowed = {row["run_id"] for row in self.queue}
        selected_ids = allowed if run_ids is None else set(run_ids)
        if selected_ids - allowed or (run_ids is not None and len(run_ids) != len(selected_ids)):
            raise ValueError("RUN_IDS trùng hoặc ngoài phân công của member.")
        self.selected_attempts()
        for row in self.queue:
            run_id = row["run_id"]
            if run_id not in selected_ids or run_id in self.state["selected"]:
                continue
            if self.state["failed"].get(run_id) and not retry_failed:
                raise RuntimeError(f"{run_id} đã lỗi kỹ thuật. Kiểm tra logs rồi bật RETRY_FAILED nếu muốn chạy lại.")
            base = study.REPO / f"configs/ablation-study-v2/member-{self.member.lower()}/{row['config']}"
            planned = self.root / "runtime_configs" / row["config"]
            self.runtime_config(base, planned)
            study.validate_planned(study.imports()[0](str(planned)), study.config_for(self.shared / "config.yaml", row["profile"], row["feature"]))
            runtime = self.persistent_root / "ablation_v2_runtime" / run_id
            before = set(runtime.glob("*/metadata.json")) if runtime.exists() else set()
            try:
                print(f"Training {run_id}: seeds42/43/44")
                attempt = study.run_config(planned, self.shared)
                destination = self.root / "runs" / run_id / attempt.name
                copytree(attempt, destination)
                bundles = self.root / "bundles"
                bundles.mkdir(exist_ok=True)
                copy2(Path(str(attempt) + ".zip"), bundles / f"{run_id}__{attempt.name}.zip")
                from summarize import collect
                record = collect(destination)
                if record["status"] != "COMPLETED" or record["provenance"] != self.state["provenance"]:
                    raise ValueError("Attempt chưa đủ artifacts/provenance để nghiệm thu.")
                self.state["selected"][run_id] = destination.relative_to(self.root).as_posix()
                self.state["failed"].pop(run_id, None)
            except BaseException:
                for metadata in set(runtime.glob("*/metadata.json")) - before:
                    attempt = metadata.parent
                    destination = self.root / "failed_attempts" / run_id / attempt.name
                    if not destination.exists():
                        copytree(attempt, destination)
                    archive = Path(str(attempt) + ".zip")
                    if archive.is_file():
                        bundles = self.root / "bundles"
                        bundles.mkdir(exist_ok=True)
                        copy2(archive, bundles / f"{run_id}__{attempt.name}__FAILED.zip")
                    self.state["failed"].setdefault(run_id, []).append(destination.relative_to(self.root).as_posix())
                raise
            finally:
                _atomic_json(self.state_path, self.state)
                self.report()
                self.export_results()
        return self.report()

    def aggregate(self, sources: list[Path]) -> dict:
        if self.member != "A":
            raise ValueError("A nhận các result ZIP để tổng hợp ma trận chung.")
        self.check_shared()
        selected = self.selected_attempts()
        seen_members = {"A"}
        for source in sources:
            if source.is_dir():
                imported = source
            else:
                imported = self.root / "aggregate_imports" / study.sha(source)[:16]
                if not imported.exists():
                    _extract_results(source, imported)
            state = study.read_json(imported / "queue_status.json")
            if state["member"] in seen_members or state["member"] not in ("B", "C") or state["code_id"] != self.code_id or state["provenance"] != self.state["provenance"]:
                raise ValueError("Result input trùng member hoặc lệch source/cohort/protocol.")
            seen_members.add(state["member"])
            expected = {row["run_id"] for row in study.matrix() if row["owner"] == state["member"]}
            if set(state["selected"]) - expected:
                raise ValueError("Result input có selected run ngoài phân công.")
            for run_id, relative in state["selected"].items():
                path = _relative(imported, relative)
                from summarize import collect
                if collect(path)["run_id"] != run_id:
                    raise ValueError("Selected path không khớp run ID.")
                selected.append(path)
        from summarize import summarize
        return summarize(selected, self.root / "ALL_MEMBERS_RESULTS.md")
