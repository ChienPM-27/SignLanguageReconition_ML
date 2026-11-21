"""Face-required scope over the unchanged frozen V2 configs and artifacts.

Training/source/data provenance stays canonical. The required matrix is the
30 face-on cells; legacy face-off artifacts remain archived and cannot rank.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from shutil import copy2
from types import FunctionType

import notebook_support as workflow
import study
import summarize as summary

SCOPE_ID = "v2-face-required-v1"
FACE_IDS = tuple(name for name in study.FEATURE_IDS if name.endswith("-face"))


@lru_cache(maxsize=1)
def _matrix_template() -> list[dict]:
    rows = [row for row in study.matrix() if row["feature"] in FACE_IDS]
    if len(rows) != 30 or len({row["run_id"] for row in rows}) != 30:
        raise ValueError("Face-required scope must contain exactly 30 canonical cells.")
    counts = {member: sum(row["owner"] == member for row in rows) for member in "ABC"}
    if counts != {"A": 4, "B": 15, "C": 11}:
        raise ValueError("Face-required ownership differs from the frozen matrix.")
    return rows


def face_matrix() -> list[dict]:
    # The base workflow still verifies frozen source/shared on every run. Cache
    # only this immutable scope projection; callers cannot mutate the template.
    return deepcopy(_matrix_template())


def descriptor() -> dict:
    encoded = json.dumps(face_matrix(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"id": SCOPE_ID, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "matrix_sha256": hashlib.sha256(encoded).hexdigest()}


def validate_scope(path: Path) -> None:
    if path.is_file():
        audit = study.read_json(path)
        if any(audit.get(key) != value for key, value in descriptor().items()) or audit.get("matrix") != face_matrix():
            raise ValueError("Face-required scope/protocol differs from this notebook.")


def summarize_faces(roots: list[Path], output: Path) -> dict:
    """Reuse strict canonical acceptance/ranking with a private face-only context."""
    allowed = {row["run_id"] for row in face_matrix()}
    historical = {row["run_id"] for row in study.matrix()} - allowed
    active, excluded = [], []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        candidates = [root / "metadata.json"] if (root / "metadata.json").is_file() else root.rglob("metadata.json")
        for item in candidates:
            meta = study.read_json(item)
            run_id = meta.get("run_id")
            if run_id in allowed:
                active.append(item.parent)
            elif run_id in historical and meta.get("campaign") == "v2":
                excluded.append(run_id)
            else:
                raise ValueError(f"Artifact outside frozen V2 scope: {item}")
    # Do not patch study.matrix: shared validation must still check its original
    # frozen superset. Only this function gets the narrowed matrix/column globals.
    context = {**summary.summarize.__globals__, "matrix": face_matrix, "FEATURE_IDS": FACE_IDS}
    scoped_summary = FunctionType(summary.summarize.__code__, context,
                                  summary.summarize.__name__, summary.summarize.__defaults__)
    result = scoped_summary(active, output)
    result["scope"] = descriptor()
    result["excluded_historical_run_ids"] = sorted(set(excluded))
    study.write_json(output.with_suffix(".json"), result)
    text = output.read_text(encoding="utf-8").replace("/60 cells", "/30 cells")
    text = text.replace("trong 60 cấu hình", "trong 30 cấu hình có face")
    text += "\nFace là yêu cầu thiết kế cố định; bảng này không kiểm định tác dụng bật/tắt face.\n"
    output.write_text(text, encoding="utf-8")
    return result


class FaceOnlySession(workflow.MemberSession):
    """Keep the original state archive, execute/count only assigned face runs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep a new root empty until RESTORE_INPUT has been imported.
        print(f"Active face-only queue: {len(self.face_queue)} configs / {len(self.face_queue) * 3} fits")

    @classmethod
    def adopt(cls, existing: workflow.MemberSession):
        """Adopt an idle session after its current training cell has finished."""
        if not isinstance(existing, workflow.MemberSession):
            raise TypeError("Expected the existing frozen V2 MemberSession.")
        existing._validate_state()
        existing.check_shared()
        session = cls.__new__(cls)
        session.__dict__.update(existing.__dict__)
        session.enable_scope()
        return session

    @property
    def face_queue(self) -> list[dict]:
        result = [row for row in self.queue if row["feature"] in FACE_IDS]
        expected = {row["run_id"] for row in face_matrix() if row["owner"] == self.member}
        if {row["run_id"] for row in result} != expected:
            raise ValueError("Member face queue differs from the canonical scoped matrix.")
        return result

    def enable_scope(self) -> dict:
        path = self.root / "FACE_ONLY_PROTOCOL.json"
        validate_scope(path)
        audit = {**descriptor(), "matrix": face_matrix(),
                 "shared_provenance": self.state.get("provenance"),
                 "face_policy": "Required by design; no face on/off efficacy comparison"}
        workflow._atomic_json(path, audit)
        payload = Path(__file__).read_bytes()
        destination = self.root / "compatibility/face_only_v1.py"
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError("Archived face scope module differs from this notebook.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        print(f"Face-only scope: Member {self.member}, {len(self.face_queue)} configs / {len(self.face_queue) * 3} fits")
        return audit

    def selected_attempts(self) -> list[Path]:
        # Validate every archived selection, including historical face-off runs.
        all_attempts = super().selected_attempts()
        allowed = {row["run_id"] for row in self.face_queue}
        return [path for path in all_attempts if study.read_json(path / "metadata.json")["run_id"] in allowed]

    def report(self) -> dict:
        self.check_shared()
        selected = self.selected_attempts()
        summarize_faces(selected, self.root / "MEMBER_RESULTS.md")
        rows = [{"run_id": row["run_id"], "feature": row["feature"], "profile": row["profile"],
                 "status": "COMPLETED" if row["run_id"] in self.state["selected"] else
                           "FAILED" if row["run_id"] in self.state["failed"] else "PENDING"}
                for row in self.face_queue]
        historical = sorted(run_id for run_id in self.state["selected"] if run_id not in {r["run_id"] for r in rows})
        result = {"member": self.member, "scope": descriptor(), "completed": len(selected),
                  "expected": len(rows), "queue": rows, "historical_selected_run_ids": historical}
        study.write_json(self.root / "MEMBER_STATUS.json", result)
        return result

    def run_queue(self, run_ids: list[str] | None = None, retry_failed: bool = False) -> dict:
        allowed = {row["run_id"] for row in self.face_queue}
        selected = [row["run_id"] for row in self.face_queue] if run_ids is None else run_ids
        if set(selected) - allowed or len(selected) != len(set(selected)):
            raise ValueError("RUN_IDS must be unique face-only runs assigned to this member.")
        self.enable_scope()
        return super().run_queue(selected, retry_failed=retry_failed)

    def export_results(self) -> Path:
        self.enable_scope()
        original = super().export_results()
        target = original.with_name(f"member_{self.member.lower()}_{self.code_id}_face_only_results.zip")
        copy2(original, target)
        print("Face-only results ZIP:", target)
        return target

    def aggregate(self, sources: list[Path]) -> dict:
        if self.member != "A":
            raise ValueError("Only A aggregates the three member result packages.")
        self.check_shared()
        selected = self.selected_attempts()
        seen = {"A"}
        for source in sources:
            imported = source
            if not source.is_dir():
                imported = self.root / "aggregate_imports" / study.sha(source)[:16]
                if not imported.exists():
                    workflow._extract_results(source, imported)
            state = study.read_json(imported / "queue_status.json")
            member = state.get("member")
            if member not in ("B", "C") or member in seen or state.get("code_id") != self.code_id or state.get("provenance") != self.state["provenance"]:
                raise ValueError("Imported member/source/cohort/protocol differs or is duplicated.")
            validate_scope(imported / "FACE_ONLY_PROTOCOL.json")
            seen.add(member)
            canonical = {r["run_id"] for r in study.matrix() if r["owner"] == member}
            if set(state["selected"]) - canonical:
                raise ValueError("Imported selected run is outside member assignment.")
            for run_id, relative in state["selected"].items():
                path = workflow._relative(imported, relative)
                record = summary.collect(path)
                if record["run_id"] != run_id or record["status"] != "COMPLETED" or record["provenance"] != self.state["provenance"]:
                    raise ValueError("Imported selected attempt is invalid.")
                if record["feature"] in FACE_IDS:
                    selected.append(path)
        return summarize_faces(selected, self.root / "ALL_MEMBERS_RESULTS.md")
