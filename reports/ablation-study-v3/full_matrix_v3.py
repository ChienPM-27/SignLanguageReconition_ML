"""Explicit full-42 orchestration over unchanged frozen V3 training code."""
from __future__ import annotations

from pathlib import Path
from shutil import copytree
import zipfile

from session_v3 import MemberSession, extract, relative
import study_v3 as study


def summarize_full_matrix(attempts: list[Path], output: Path) -> dict:
    rows = [study.collect(path) for path in attempts]
    records = {r["run_id"]: r for r in rows}
    if len(records) != len(rows):
        raise ValueError("Duplicate selected run in full-42 results.")
    if rows and any(r["provenance"] != rows[0]["provenance"] or r["memberships"] != rows[0]["memberships"] for r in rows):
        raise ValueError("Full-42 results differ in frozen source/cohort/protocol.")
    required = [r["run_id"] for r in study.matrix()]
    if set(records) - set(required):
        raise ValueError("Run outside frozen 42-cell matrix.")
    missing = [run_id for run_id in required if run_id not in records]
    deltas = {}
    for profile in study.PROFILES:
        for smaller, small_regions in study.REGIONS.items():
            for larger, large_regions in study.REGIONS.items():
                a, b = profile + "__" + larger, profile + "__" + smaller
                if set(small_regions) < set(large_regions) and a in records and b in records:
                    deltas[a + " minus " + b] = study.aggregate([
                        x - y for x, y in zip(records[a]["metrics"]["macro_f1"]["per_seed"],
                                             records[b]["metrics"]["macro_f1"]["per_seed"])])
    result = {"scope": "v3-full42", "freeze_id": rows[0]["provenance"]["freeze_id"] if rows else None,
              "status": "PENDING" if missing else "COMPLETED", "completed_cells": len(rows),
              "expected_cells": 42, "completed_fits": len(rows) * 3, "expected_fits": 126,
              "missing": missing, "cross_plan": None,
              "winner": study.rank(rows)[0]["run_id"] if rows and not missing else None,
              "records": records, "paired_region_deltas": deltas}
    study.write(output.with_suffix(".json"), result)
    lines = ["# Nghiệm thu đủ 42 configs V3", "", f"{result['status']}: {len(rows)}/42 configs.",
             "Winner: " + (result["winner"] or "chờ đủ 42 configs"), "",
             "| Profile | " + " | ".join(study.REGIONS) + " |", "|---|" + "---|" * len(study.REGIONS)]
    for profile in study.PROFILES:
        cells = []
        for feature in study.REGIONS:
            row = records.get(profile + "__" + feature)
            metric = row["metrics"]["macro_f1"] if row else None
            cells.append(f"{metric['mean']:.4f} ± {metric['std']:.4f}" if metric else "—")
        lines.append("| " + profile + " | " + " | ".join(cells) + " |")
    lines += ["", "Mở rộng phạm vi chạy thành đủ 42 configs; giữ nguyên frozen source/config/cohort.",
              "Rule chọn winner không đổi: mean Validation Macro-F1, std thấp hơn, parameters ít hơn, run ID.",
              "Test chỉ mở sau nghiệm thu đủ 42 configs và khóa checkpoints; std là population std của 3 seeds."]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


class FullMatrixSession(MemberSession):
    """Run every assigned YAML, reuse accepted results, and require all 42 cells."""

    def queue(self) -> list[dict]:
        return [r for r in study.matrix() if r["owner"] == self.member]

    def validate_state(self) -> None:
        super().validate_state()
        scope_path = self.root / "FULL_MATRIX_SCOPE.json"
        if scope_path.is_file():
            scope = study.read(scope_path)
            if scope.get("scope") != "v3-full42" or scope.get("freeze_id") != self.freeze_id:
                raise ValueError("Restored full-matrix scope differs from this launcher.")

    def load_cross_plan(self, source: Path) -> None:
        raise ValueError("Full-42 launcher runs all assigned configs; CROSS_PLAN_INPUT is unnecessary.")

    def restore_results(self, source: Path) -> None:
        if self.state["selected"] or self.state["failed"] or self.state_path.exists():
            raise FileExistsError("Restore requires an empty results session.")
        if source.is_dir():
            for path in source.rglob("*"):
                if path.is_symlink():
                    raise ValueError("Result folder contains a symlink.")
                relative(source, path.relative_to(source).as_posix())
            copytree(source, self.root)
        else:
            extract(source, self.root)
        self.state = study.read(self.state_path)
        self.accepted()
        print(f"Restored {len(self.state['selected'])} accepted configs; completed runs will be skipped.")

    def report(self) -> dict:
        result = super().report()
        scope = {"scope": "v3-full42", "freeze_id": self.freeze_id,
                 "training_source_provenance": self.provenance,
                 "selection_scope_change": "Run all 42 existing frozen configurations instead of conditional 24/30.",
                 "expected_cells": 42, "expected_fits": 126,
                 "member_expected_cells": len(self.queue()),
                 "orchestrator_sha256": study.sha(Path(__file__))}
        study.write(self.root / "FULL_MATRIX_SCOPE.json", scope)
        result["scope"] = scope["scope"]
        result["remaining"] = [r["run_id"] for r in result["queue"] if r["status"] != "COMPLETED"]
        study.write(self.root / "MEMBER_STATUS.json", result)
        return result

    def export_results(self) -> Path:
        self.report()
        destination = self.root.parent / f"member_{self.member.lower()}_{self.freeze_id}_v3_full42_results.zip"
        temporary = destination.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.root.rglob("*")):
                name = path.relative_to(self.root)
                if path.is_file() and name.parts[0] != "aggregate_imports" and not path.name.endswith(".tmp"):
                    archive.write(path, name.as_posix())
        temporary.replace(destination)
        print(f"Results ZIP: {destination}")
        return destination

    def aggregate(self, sources: list[Path]) -> dict:
        if self.member != "A":
            raise ValueError("Only member A aggregates full-42 results.")
        attempts = self.accepted()
        members = {"A"}
        for source in sources:
            folder = source if source.is_dir() else self.root / "aggregate_imports" / study.sha(source)[:16]
            if not source.is_dir() and not folder.exists():
                extract(source, folder)
            state = study.read(folder / "queue_status.json")
            member = state["member"]
            if member not in ("B", "C") or member in members or state["provenance"] != self.provenance:
                raise ValueError("Different provenance or duplicate member.")
            members.add(member)
            allowed = {r["run_id"] for r in study.matrix() if r["owner"] == member}
            for run_id, name in state["selected"].items():
                path = relative(folder, name)
                record = study.collect(path)
                if run_id not in allowed or record["run_id"] != run_id:
                    raise ValueError("Selected run differs from assigned owner/run ID.")
                attempts.append(path)
        result = summarize_full_matrix(attempts, self.root / "ALL_MEMBERS_FULL42_RESULTS.md")
        self.export_results()
        return result

    def finalize_test(self, summary: dict) -> dict:
        if (summary.get("scope") != "v3-full42" or summary.get("status") != "COMPLETED"
                or summary.get("expected_cells") != 42 or summary.get("completed_cells") != 42
                or set(summary.get("records", {})) != {r["run_id"] for r in study.matrix()}):
            raise ValueError("Hold out test until the complete full-42 matrix is accepted.")
        return super().finalize_test(summary)
