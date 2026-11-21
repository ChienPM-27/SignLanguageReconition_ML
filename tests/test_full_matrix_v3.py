"""Self-contained full-42 dispatch, resume and acceptance safety checks."""
from copy import deepcopy
import ast
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPORT = Path(__file__).resolve().parents[1] / "reports/ablation-study-v3"
sys.path.insert(0, str(REPORT / "tools"))
sys.path.insert(0, str(REPORT))
from full_matrix_v3 import FullMatrixSession, summarize_full_matrix
from recover_full42_split_cell import recover_split_paths
import study_v3 as study
from src.data.split import create_signer_splits
from src.utils.config import Config
from src.utils import env


def records():
    return {r["run_id"]: {**r, "status": "COMPLETED", "path": r["run_id"],
                         "provenance": {"freeze_id": "fixture"},
                         "memberships": {"train": ["one"], "val": ["two"], "test": ["three"]},
                         "metrics": {"macro_f1": study.aggregate([.7, .71, .72]),
                                     "model_parameters": study.aggregate([100, 100, 100])}}
            for r in study.matrix()}


def test_run_all_missing18_skip_baselines_and_resume_without_duplicate_training(monkeypatch, tmp_path):
    fixture = records()
    monkeypatch.setattr(study, "verify_shared", lambda _: {"freeze_id": "fixture"})
    monkeypatch.setattr(study, "collect", lambda p: fixture[p.parent.name])
    session = FullMatrixSession("B", tmp_path, tmp_path, tmp_path / "storage")
    monkeypatch.setattr(session, "check_data", lambda: None)
    baselines = [r for r in study.matrix() if r["owner"] == "B" and r["feature"] == "HBF"]
    for row in baselines:
        path = session.root / "runs" / row["run_id"] / "original"
        path.mkdir(parents=True)
        (path / "payload.txt").write_text("preserved", encoding="utf-8")
        session.state["selected"][row["run_id"]] = path.relative_to(session.root).as_posix()
    called = []

    def run(row, shared, persistent_root, mirror):
        called.append(row["run_id"])
        path = tmp_path / "synthetic-training" / row["run_id"] / "new"
        path.mkdir(parents=True)
        (path / "payload.txt").write_text("new", encoding="utf-8")
        return path

    monkeypatch.setattr(study, "run", run)
    expected = [r["run_id"] for r in study.matrix() if r["owner"] == "B" and r["feature"] != "HBF"]
    result = session.run_queue(expected)
    assert result["completed"] == result["expected"] == 21
    assert called == expected and len(called) == 18
    for row in baselines:
        assert (session.root / session.state["selected"][row["run_id"]] / "payload.txt").read_text() == "preserved"
    bundle = session.export_results()
    restored = FullMatrixSession("B", tmp_path, tmp_path, tmp_path / "fresh")
    monkeypatch.setattr(restored, "check_data", lambda: None)
    restored.restore_results(bundle)
    before = list(called)
    assert restored.run_queue(expected)["completed"] == 21
    assert called == before
    assert study.read(restored.root / "FULL_MATRIX_SCOPE.json")["scope"] == "v3-full42"


def test_full42_acceptance_requires_all_variants_and_rejects_duplicate_or_mixed_source(monkeypatch, tmp_path):
    fixture = records()
    monkeypatch.setattr(study, "collect", lambda p: fixture[p.name])
    phase1 = [Path(r["run_id"]) for r in study.matrix() if r["phase"] == 1]
    pending = summarize_full_matrix(phase1, tmp_path / "pending.md")
    assert pending["status"] == "PENDING" and pending["winner"] is None
    assert pending["completed_cells"] == 24 and len(pending["missing"]) == 18
    full = [Path(r["run_id"]) for r in study.matrix()]
    completed = summarize_full_matrix(full, tmp_path / "full.md")
    assert completed["status"] == "COMPLETED" and completed["expected_cells"] == 42
    assert completed["completed_fits"] == 126
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_full_matrix(full + [full[0]], tmp_path / "duplicates.md")
    changed = deepcopy(fixture[full[-1].name])
    changed["provenance"] = {"freeze_id": "other-source"}
    fixture[full[-1].name] = changed
    with pytest.raises(ValueError, match="differ"):
        summarize_full_matrix(full, tmp_path / "different.md")


def test_folder_restore_for_kaggle_auto_extracted_results(monkeypatch, tmp_path):
    fixture = records()
    monkeypatch.setattr(study, "verify_shared", lambda _: {"freeze_id": "fixture"})
    monkeypatch.setattr(study, "collect", lambda p: fixture[p.parent.name])
    source = tmp_path / "kaggle-input"
    state = {"member": "B", "freeze_id": "fixture", "provenance": {"freeze_id": "fixture"},
             "selected": {}, "failed": {}, "cross_plan": None}
    for row in study.matrix():
        if row["owner"] == "B" and row["feature"] == "HBF":
            name = "runs/" + row["run_id"] + "/original"
            (source / name).mkdir(parents=True)
            state["selected"][row["run_id"]] = name
    study.write(source / "queue_status.json", state)
    session = FullMatrixSession("B", tmp_path, tmp_path, tmp_path / "storage")
    session.restore_results(source)
    report = session.report()
    assert report["completed"] == 3 and report["expected"] == 21
    assert len(report["remaining"]) == 18
    assert study.read(source / "queue_status.json") == state
    with pytest.raises(FileExistsError):
        session.restore_results(source)


def test_full42_guard_rejects_old30_cell_completion_before_test():
    session = object.__new__(FullMatrixSession)
    with pytest.raises(ValueError, match="complete full-42"):
        session.finalize_test({"status": "COMPLETED", "winner": "G-bi__H", "completed_cells": 30})


def test_actual_kaggle_split_writer_and_same_session_recovery_preserve_cache(monkeypatch, tmp_path):
    working = tmp_path / "kaggle-working"
    legacy_root = working / "ablation_v3_full42"
    shared = tmp_path / "shared"
    shared.mkdir()
    cfg = Config()
    cfg.env.name = "kaggle"
    cfg.env.local_root = str(legacy_root)
    cfg.split.min_per_gloss_per_split = 1
    cfg.data.cache_dir = str(legacy_root / "cache")
    cfg.output.report_dir = str(legacy_root / "reports")
    cfg.split.manifest_path = str(shared / "split_manifest.json")
    frozen = {"schema_version": 1, "seed": 42, "vocabulary": ["example"], "label_map": {"example": 0},
              "splits": {name: {"sequence_ids": [str(i)], "participant_ids": [str(i)]}
                         for i, name in enumerate(("train", "val", "test"), 1)}}
    study.write(shared / "split_manifest.json", frozen)
    monkeypatch.setattr(env, "KAGGLE_WORKING_DIR", str(working))
    paths = env.resolve_paths("kaggle", cfg, create_dirs=False)
    assert paths.persistent_root == working and paths.cache_dir == Path(cfg.data.cache_dir)
    features = paths.cache_dir / "features"
    features.mkdir(parents=True)
    marker = features / "reusable-cache.npy"
    marker.write_bytes(b"existing cached features")
    with (features / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sequence_id", "participant_id", "gloss", "label", "path"])
        writer.writeheader()
        for i in range(1, 4):
            writer.writerow({"sequence_id": str(i), "participant_id": str(i), "gloss": "example",
                             "label": 0, "path": str(marker)})
    before = marker.read_bytes()
    create_signer_splits(cfg, paths)
    assert (working / "data/splits/train.csv").is_file()
    with pytest.raises(FileNotFoundError):
        study.check_cohort(legacy_root / "data/splits", shared / "split_manifest.json")
    session = SimpleNamespace(persistent_root=legacy_root, shared=shared,
                              root=legacy_root / "results", freeze_id="fixture")
    restored = recover_split_paths(session, working)
    study.check_cohort(restored, shared / "split_manifest.json")
    for filename in ("train.csv", "val.csv", "test.csv", "classes.csv", "label_map.json"):
        assert (restored / filename).read_bytes() == (working / "data/splits" / filename).read_bytes()
    assert marker.read_bytes() == before
    assert not study.read(session.root / "PATH_RECOVERY.json")["source_training_changed"]
    recover_split_paths(session, working)
    (restored / "train.csv").write_text("foreign split", encoding="utf-8")
    with pytest.raises(ValueError, match="ghi đè"):
        recover_split_paths(session, working)
    session.persistent_root = tmp_path / "outside"
    with pytest.raises(ValueError, match="working root"):
        recover_split_paths(session, working)


def test_exported_notebook_root_matches_actual_kaggle_resolution():
    notebook = json.loads((REPORT.parents[1] / "notebooks/ablation_v3_complete_18_kaggle.ipynb").read_text(encoding="utf-8"))
    roots = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PERSISTENT_ROOT" for t in statement.targets):
                roots.append(Path(ast.literal_eval(statement.value.args[0])))
    cfg = Config()
    cfg.env.local_root = "/kaggle/working/ablation_v3_full42"
    canonical = env.resolve_paths("kaggle", cfg, create_dirs=False)
    assert roots == [canonical.persistent_root]
