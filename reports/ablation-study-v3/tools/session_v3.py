"""Notebook queues, safe ZIP restore, automatic result bundles and cross-plan handoff."""
from __future__ import annotations

import csv
from pathlib import Path
from shutil import copytree
import zipfile

import study_v3 as study


def relative(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()) or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive/state path: {name}")
    return target


def extract(source: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Restore phải vào thư mục trống.")
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip():
            raise ValueError("ZIP trùng entry hoặc CRC lỗi.")
        for entry in archive.infolist():
            relative(destination, entry.filename)
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP symlink không được hỗ trợ.")
        archive.extractall(destination)


class MemberSession:
    def __init__(self, member: str, shared: Path, mirror: Path, persistent_root: Path):
        if member not in ("A", "B", "C"):
            raise ValueError("Member phải là A/B/C.")
        self.member, self.shared, self.mirror = member, shared, mirror
        self.persistent_root = persistent_root
        self.provenance = study.verify_shared(shared)
        self.freeze_id = self.provenance["freeze_id"]
        self.root = persistent_root / "ablation_v3_results" / ("member-" + member.lower()) / self.freeze_id
        self.state_path = self.root / "queue_status.json"
        self.state = {"member": member, "freeze_id": self.freeze_id, "provenance": self.provenance,
                      "selected": {}, "failed": {}, "cross_plan": None}
        if self.state_path.is_file():
            self.state = study.read(self.state_path)
        self.validate_state()
        print(f"Member {member}; Freeze ID: {self.freeze_id}; results: {self.root}")

    def queue(self) -> list[dict]:
        ids = self.state["cross_plan"]["run_ids"] if self.state["cross_plan"] else []
        return [r for r in study.matrix() if r["owner"] == self.member and (r["phase"] == 1 or r["run_id"] in ids)]

    def validate_state(self) -> None:
        if (self.state["member"] != self.member or self.state["freeze_id"] != self.freeze_id
                or self.state["provenance"] != self.provenance):
            raise ValueError("Results khác member/source/protocol; không trộn V2/V3.")
        allowed = {r["run_id"] for r in self.queue()}
        if (set(self.state["selected"]) | set(self.state["failed"])) - allowed:
            raise ValueError("State có run ngoài phân công.")
        for name in self.state["selected"].values():
            relative(self.root, name)

    def accepted(self) -> list[Path]:
        self.validate_state()
        result = []
        for run_id, name in self.state["selected"].items():
            path = relative(self.root, name)
            record = study.collect(path)
            if record["run_id"] != run_id or record["provenance"] != self.provenance:
                raise ValueError("Selected attempt lệch provenance/run ID.")
            result.append(path)
        return result

    def check_data(self) -> None:
        frozen = study.read(self.shared / "split_manifest.json")
        available = set()
        with (self.mirror / "train.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                sid = row["sequence_id"]
                if sid in frozen["sequences"]:
                    expected = frozen["sequences"][sid]
                    if (sid in available or str(row["participant_id"]) != str(expected["participant_id"])
                            or row["sign"] != expected["gloss"]):
                        raise ValueError(f"Raw catalog provenance lệch: {sid}")
                    if (self.mirror / row["path"]).is_file():
                        available.add(sid)
        missing = set(frozen["sequences"]) - available
        if missing:
            raise FileNotFoundError(f"Attach đầy đủ asl-signs; thiếu {len(missing)} frozen sequences.")
        print(f"Raw cohort hợp lệ: {len(available)} sequences.")

    def report(self) -> dict:
        self.accepted()
        rows = [{**row, "status": "COMPLETED" if row["run_id"] in self.state["selected"] else
                 "FAILED" if row["run_id"] in self.state["failed"] else "PENDING"} for row in self.queue()]
        result = {"member": self.member, "freeze_id": self.freeze_id, "completed": len(self.state["selected"]),
                  "expected": len(rows), "queue": rows}
        study.write(self.root / "MEMBER_STATUS.json", result)
        study.write(self.state_path, self.state)
        return result

    def export_results(self) -> Path:
        self.report()
        destination = self.root.parent / f"member_{self.member.lower()}_{self.freeze_id}_v3_results.zip"
        temporary = destination.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.root.rglob("*")):
                name = path.relative_to(self.root)
                if path.is_file() and name.parts[0] != "aggregate_imports" and not path.name.endswith(".tmp"):
                    archive.write(path, name.as_posix())
        temporary.replace(destination)
        print(f"Results ZIP: {destination}")
        return destination

    def restore_results(self, source: Path) -> None:
        if self.state["selected"] or self.state["failed"] or self.state_path.exists():
            raise FileExistsError("Restore vào session mới trước khi training/report.")
        extract(source, self.root)
        self.state = study.read(self.state_path)
        self.accepted()
        print(f"Restored {len(self.state['selected'])} configs; configs COMPLETED sẽ được bỏ qua.")

    def load_cross_plan(self, source: Path) -> None:
        if self.member != "B":
            raise ValueError("Chỉ B chạy vòng kiểm chứng GRU architecture.")
        if source.suffix == ".zip":
            with zipfile.ZipFile(source) as archive:
                if archive.namelist() != ["cross_plan.json"]:
                    raise ValueError("Cross-plan ZIP không hợp lệ.")
                import json
                plan = json.loads(archive.read("cross_plan.json"))
        else:
            plan = study.read(source)
        if plan.get("freeze_id") != self.freeze_id:
            raise ValueError("Cross-plan khác freeze.")
        candidates = plan.get("baseline_metrics", {})
        if set(candidates) != {p for p in study.PROFILES if p.startswith("G")}:
            raise ValueError("Cross-plan thiếu GRU baselines.")
        ranked = study.rank([{"run_id": p + "__HBF", "metrics": metrics} for p, metrics in candidates.items()])
        chosen = ranked[0]["run_id"].split("__")[0]
        expected = [] if chosen == "G0" else [chosen + "__" + f for f in study.REGIONS if f != "HBF"]
        if plan.get("profile") != chosen or plan.get("run_ids") != expected:
            raise ValueError("Cross-plan không đúng selection rule đã freeze.")
        if self.state["cross_plan"] and self.state["cross_plan"] != plan:
            raise ValueError("Không đổi cross-plan của results đã lưu.")
        self.state["cross_plan"] = plan
        study.write(self.state_path, self.state)
        print(f"Cross-plan: {chosen}; thêm {len(expected)} configs.")

    def run_queue(self, run_ids: list[str] | None = None, retry_failed: bool = False) -> dict:
        self.accepted()
        self.check_data()
        allowed = {r["run_id"] for r in self.queue()}
        chosen = allowed if run_ids is None else set(run_ids)
        if chosen - allowed or run_ids is not None and len(chosen) != len(run_ids):
            raise ValueError("RUN_IDS ngoài queue hoặc trùng.")
        for row in self.queue():
            run_id = row["run_id"]
            if run_id not in chosen or run_id in self.state["selected"]:
                continue
            if run_id in self.state["failed"] and not retry_failed:
                raise RuntimeError(f"{run_id} đã lỗi; kiểm tra logs rồi bật RETRY_FAILED.")
            try:
                print(f"Training {run_id}: seeds42/43/44")
                attempt = study.run(row, self.shared, self.persistent_root, self.mirror)
                destination = self.root / "runs" / run_id / attempt.name
                copytree(attempt, destination)
                study.collect(destination)
                self.state["selected"][run_id] = destination.relative_to(self.root).as_posix()
                self.state["failed"].pop(run_id, None)
            except BaseException as exc:
                self.state["failed"][run_id] = f"{type(exc).__name__}: {exc}"
                runtime = self.persistent_root / "ablation_v3_runtime" / self.freeze_id / run_id
                for metadata in runtime.glob("*/metadata.json"):
                    if study.read(metadata).get("status") == "FAILED":
                        target = self.root / "failed_attempts" / run_id / metadata.parent.name
                        if not target.exists():
                            copytree(metadata.parent, target)
                raise
            finally:
                study.write(self.state_path, self.state)
                self.export_results()
        return self.report()

    def aggregate(self, sources: list[Path]) -> dict:
        if self.member != "A":
            raise ValueError("A tổng hợp results của ba thành viên.")
        attempts = self.accepted()
        members = {"A"}
        for source in sources:
            folder = source if source.is_dir() else self.root / "aggregate_imports" / study.sha(source)[:16]
            if not source.is_dir() and not folder.exists():
                extract(source, folder)
            state = study.read(folder / "queue_status.json")
            member = state["member"]
            if member not in ("B", "C") or member in members or state["provenance"] != self.provenance:
                raise ValueError("Result package khác provenance hoặc trùng member.")
            members.add(member)
            allowed = {r["run_id"] for r in study.matrix() if r["owner"] == member}
            for run_id, name in state["selected"].items():
                path = relative(folder, name)
                record = study.collect(path)
                if run_id not in allowed or record["run_id"] != run_id:
                    raise ValueError("Selected run ngoài ownership.")
                attempts.append(path)
        result = study.summarize(attempts, self.root / "ALL_MEMBERS_RESULTS.md")
        if result["cross_plan"]:
            study.write(self.root / "cross_plan.json", result["cross_plan"])
            with zipfile.ZipFile(self.root / "cross_plan.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(self.root / "cross_plan.json", "cross_plan.json")
            print(f"Gửi B cross-plan: {self.root / 'cross_plan.zip'}")
        self.export_results()
        return result

    def finalize_test(self, summary: dict) -> dict:
        if self.member != "A" or summary.get("status") != "COMPLETED" or not summary.get("winner"):
            raise ValueError("Chỉ mở test sau khi A nghiệm thu đủ ma trận và khóa winner.")
        self.check_data()
        winner = summary["winner"]
        selected = Path(summary["records"][winner]["path"])
        record = study.collect(selected)
        if record["provenance"] != self.provenance:
            raise ValueError("Winner khác freeze.")
        checkpoints = sorted((selected / "runs").rglob("best_model.pt"))
        if len(checkpoints) != 3 or {p.parent.name for p in checkpoints} != {"seed_42", "seed_43", "seed_44"}:
            raise ValueError("Winner cần đúng checkpoints seeds42/43/44.")
        lock_record = {"freeze_id": self.freeze_id, "winner": winner,
                       "checkpoints": {p.parent.name: study.sha(p) for p in checkpoints}}
        target = self.root / "FINAL_TEST"
        if (target / "WINNER_LOCK.json").is_file() and study.read(target / "WINNER_LOCK.json") != lock_record:
            raise ValueError("Winner đã khóa khác lựa chọn mới.")
        study.write(target / "WINNER_LOCK.json", lock_record)
        cfg = study.config_for(self.shared / "config.yaml", record["profile"], record["feature"])
        cfg.env.local_root = str(self.persistent_root)
        cfg.data.local_mirror_dir = str(self.mirror)
        identity = study.compute_preprocess_identity(cfg)[0]
        cfg.data.cache_dir = str(self.persistent_root / "ablation_v3_cache" / self.freeze_id / identity)
        cfg.output.output_dir, cfg.output.report_dir = str(target / "outputs"), str(target / "reports")
        cfg.output.selection_dir = str(self.shared / "selection")
        cfg.split.manifest_path = str(self.shared / "split_manifest.json")
        study.save_config(cfg, target / "evaluation_config.yaml")
        runtime = self.persistent_root / "ablation_v3_runtime" / self.freeze_id
        runtime.mkdir(parents=True, exist_ok=True)
        lock = runtime / ".runtime.lock"
        with lock.open("x", encoding="utf-8") as stream:
            stream.write("final_test")
        reports = []
        try:
            import sys
            for stage in ("preprocess", "split"):
                study.logged([sys.executable, "-X", "utf8", "scripts/10_prepare_data.py", "--config",
                              str(target / "evaluation_config.yaml"), "--stage", stage], target / "logs" / (stage + ".log"))
            study.check_cohort(self.persistent_root / "data/splits", self.shared / "split_manifest.json")
            for checkpoint in checkpoints:
                output = target / checkpoint.parent.name
                study.logged([sys.executable, "-X", "utf8", "scripts/21_evaluate.py", "--checkpoint", str(checkpoint),
                              "--split", "test", "--set", "env.local_root=" + str(self.persistent_root),
                              "--output-dir", str(output)], target / "logs" / (checkpoint.parent.name + ".log"))
                reports.append(study.read(output / "evaluation_test_report.json"))
            result = {**lock_record, "metrics": {key: study.aggregate([r[key] for r in reports])
                      for key in ("macro_f1", "top1_accuracy", "top3_accuracy", "loss")}}
            study.write(target / "TEST_SUMMARY.json", result)
            return result
        finally:
            lock.unlink(missing_ok=True)
            self.export_results()
