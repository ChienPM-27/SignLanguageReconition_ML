"""Self-contained face-scope migration, reuse, completion and ranking tests."""
from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[4]
HOTFIX_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("canonical_workflow_fixtures", REPO / "tests/test_ablation_notebooks.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
workflow = fixtures.workflow


@pytest.fixture
def scope(workflow, monkeypatch):
    monkeypatch.syspath_prepend(str(HOTFIX_DIR))
    return importlib.import_module("face_only_v1")


def runner(session, study, calls):
    def run(planned, shared):
        cfg = study.imports()[0](str(planned))
        calls.append(cfg.run.name)
        return fixtures.complete_attempt(session.persistent_root / "ablation_v2_runtime" / cfg.run.name / "fixture01",
                                         planned, shared, study)
    return run


def test_scope_counts_config_identity_and_shared_superset(scope, workflow):
    study, _, shared, _ = workflow
    rows = scope.face_matrix()
    assert len(rows) == 30 and len({r["run_id"] for r in rows}) == 30
    assert {m: sum(r["owner"] == m for r in rows) for m in "ABC"} == {"A": 4, "B": 15, "C": 11}
    assert sum(len(r["seeds"]) for r in rows) == 90
    assert len(study.matrix()) == 60
    assert study.read_json(shared / "matrix.json") == study.matrix()
    for row in rows:
        cfg = study.imports()[0](str(study.REPO / f"configs/ablation-study-v2/member-{row['owner'].lower()}/{row['run_id']}.yaml"))
        assert cfg.features.face
        study.validate_planned(cfg, study.config_for(shared / "config.yaml", row["profile"], row["feature"]))


def test_adopt_keeps_face_artifacts_and_skips_all_off_training(scope, workflow, monkeypatch):
    study, helper, shared, _ = workflow
    old = helper.MemberSession("A")
    old.import_shared(shared)
    calls = []
    monkeypatch.setattr(study, "run_config", runner(old, study, calls))
    old.run_queue(["G0__F1", "G0__F1-face"])
    saved = deepcopy(old.state)
    face_path = old.root / old.state["selected"]["G0__F1-face"]
    checkpoints = {p: p.read_bytes() for p in face_path.rglob("best_model.pt")}
    session = scope.FaceOnlySession.adopt(old)
    assert session.state == saved
    assert session.report()["completed"] == 1 and session.report()["expected"] == 4
    assert session.report()["historical_selected_run_ids"] == ["G0__F1"]
    result = session.run_queue()
    assert result["completed"] == result["expected"] == 4
    assert calls == ["G0__F1", "G0__F1-face", "G0__F2-face", "G0__F3-face", "G0__F4-face"]
    assert "G0__F1" in session.state["selected"]
    assert all(p.read_bytes() == content for p, content in checkpoints.items())
    with pytest.raises(ValueError, match="face-only"):
        session.run_queue(["G0__F2"])
    before = list(calls)
    session.run_queue()
    assert calls == before
    package = session.export_results()
    assert package.name.endswith("_face_only_results.zip")
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        assert "FACE_ONLY_PROTOCOL.json" in archive.namelist()
        assert archive.read("compatibility/face_only_v1.py") == Path(scope.__file__).read_bytes()
        assert any(name.startswith("runs/G0__F1/") for name in archive.namelist())


def test_legacy_zip_restore_leaves_fresh_root_empty_and_reuses_face(scope, workflow, monkeypatch, tmp_path):
    study, helper, shared, _ = workflow
    old = helper.MemberSession("A")
    old.import_shared(shared)
    calls = []
    monkeypatch.setattr(study, "run_config", runner(old, study, calls))
    old.run_queue(["G0__F1", "G0__F1-face"])
    package = old.export_results()
    import src.utils.env as environment
    monkeypatch.setattr(environment, "resolve_paths", lambda *args, **kwargs: SimpleNamespace(
        persistent_root=tmp_path / "restored", data_dir=tmp_path / "raw", repo_root=study.REPO))
    session = scope.FaceOnlySession("A")
    assert not list(session.root.iterdir())
    session.restore_results(package)
    assert session.report()["completed"] == 1 and session.report()["expected"] == 4
    session.run_queue(["G0__F1-face"])
    assert calls == ["G0__F1", "G0__F1-face"]
    assert session.check_shared() == old.check_shared()


def test_scope_aggregate_requires_30_valid_cells_and_excludes_off_winner(scope, workflow, monkeypatch, tmp_path):
    study, helper, shared, _ = workflow
    sessions, packages, calls = {}, {}, []
    for member in "ABC":
        session = scope.FaceOnlySession(member)
        session.import_shared(shared)
        monkeypatch.setattr(study, "run_config", runner(session, study, calls))
        status = session.run_queue()
        assert status["completed"] == status["expected"] == {"A": 4, "B": 15, "C": 11}[member]
        sessions[member] = session
        packages[member] = session.export_results()
    a = sessions["A"]
    pending = a.aggregate([packages["B"]])
    assert pending["status"] == "PENDING" and pending["winner"] is None
    assert pending["expected_cells"] == 30 and len(pending["missing"]) == 11
    complete = a.aggregate([packages["B"], packages["C"]])
    assert complete["status"] == "COMPLETED" and complete["completed_cells"] == 30
    assert complete["winner"].endswith("-face")
    assert complete["model_face_interactions"] == {}
    assert len(calls) == len(set(calls)) == 30
    with pytest.raises(ValueError, match="duplicated"):
        a.aggregate([packages["B"], packages["B"]])
    # A historical off run can have a higher score; it still cannot rank.
    old = helper.MemberSession("A")
    monkeypatch.setattr(study, "run_config", runner(old, study, calls))
    old.run_queue(["G0__F1"])
    off = old.root / old.state["selected"]["G0__F1"]
    for path in off.rglob("val_report.json"):
        report = study.read_json(path); report["macro_f1"] = 1
        study.write_json(path, report)
    roots = [path for session in sessions.values() for path in session.selected_attempts()]
    result = scope.summarize_faces([*roots, off], tmp_path / "faces.md")
    assert result["completed_cells"] == result["expected_cells"] == 30
    assert "G0__F1" not in result["records"] and result["winner"].endswith("-face")
    assert result["excluded_historical_run_ids"] == ["G0__F1"]
    text = (tmp_path / "faces.md").read_text(encoding="utf-8")
    assert "30/30 cells" in text and "60" not in text
    assert "| F0-face | F1-face | F2-face | F3-face | F4-face |" in text


def test_changed_scope_and_corrupt_face_report_fail_loudly(scope, workflow, monkeypatch):
    study, _, shared, _ = workflow
    session = scope.FaceOnlySession("A")
    session.import_shared(shared)
    monkeypatch.setattr(study, "run_config", runner(session, study, []))
    session.run_queue(["G0__F1-face"])
    audit = session.root / "FACE_ONLY_PROTOCOL.json"
    content = study.read_json(audit); content["matrix_sha256"] = "wrong"
    study.write_json(audit, content)
    with pytest.raises(ValueError, match="scope/protocol"):
        session.enable_scope()
    path = session.root / session.state["selected"]["G0__F1-face"]
    report = next(path.rglob("val_report.json"))
    content = study.read_json(report); content["predictions"].pop()
    study.write_json(report, content)
    with pytest.raises(ValueError, match="prediction"):
        scope.summarize_faces([path], session.root / "bad.md")


def test_embedded_finalize_cell_runs_without_any_training(scope, workflow, monkeypatch, tmp_path):
    study, helper, shared, _ = workflow
    old = helper.MemberSession("A")
    old.import_shared(shared)
    calls = []
    monkeypatch.setattr(study, "run_logged", lambda *args: calls.append("stage"))
    monkeypatch.setattr(study, "run_config", lambda *args: calls.append("train"))
    builder = importlib.import_module("build_face_only_launchers")
    # Generate from tracked module sources, independent of untracked artifacts.
    code = builder.module_code() + '''from face_only_v1 import FaceOnlySession
import campaign_compat_v2 as campaign_compat
session = FaceOnlySession.adopt(session)
campaign_compat.install(session)
status = session.report()
result_zip = session.export_results()
'''
    namespace = {"session": old, "WORKSPACE": tmp_path}
    exec(compile(code, "finalize-test", "exec"), namespace)
    assert isinstance(namespace["session"], scope.FaceOnlySession)
    assert namespace["status"]["completed"] == 0 and namespace["status"]["expected"] == 4
    assert namespace["result_zip"].is_file() and calls == []
