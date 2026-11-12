"""Notebook entrypoints and data/result workflows without cloud or real datasets."""
import ast
import csv
import importlib
import importlib.util
from pathlib import Path
from shutil import copy2, make_archive
from types import SimpleNamespace
import zipfile

import pytest

from src.features.schema import compute_preprocess_identity, get_feature_dim
from src.utils.config import load_config, save_config


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset():
    vocabulary = [f"gloss{index}" for index in range(20)]
    data = {"schema_version": 1, "seed": 42, "vocabulary": vocabulary,
            "label_map": {gloss: index for index, gloss in enumerate(vocabulary)}, "sequences": {}, "splits": {}}
    for name in ("train", "val", "test"):
        ids = [f"{name}-{index}" for index in range(20)]
        signer = name + "_signer"
        data["splits"][name] = {"sequence_ids": ids, "participant_ids": [signer]}
        data["sequences"].update({sid: {"participant_id": signer, "gloss": vocabulary[index], "label": index}
                                  for index, sid in enumerate(ids)})
    return data


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "reports/ablation-study-v2/tools"
    monkeypatch.syspath_prepend(str(folder))
    study = importlib.import_module("study")
    helper = importlib.import_module("notebook_support")
    monkeypatch.setattr(study, "source_manifest", lambda: {"src/frozen.py": "same"})
    import src.utils.env
    monkeypatch.setattr(src.utils.env, "detect_env", lambda: "local")
    monkeypatch.setattr(src.utils.env, "resolve_paths", lambda *args, **kwargs: SimpleNamespace(
        persistent_root=tmp_path / "runtime", data_dir=tmp_path / "raw", repo_root=study.REPO))
    split = tmp_path / "baseline.json"
    study.write_json(split, dataset())
    selection = tmp_path / "selection"
    write_csv(selection / "glosses.csv", [{"gloss": gloss, "label": index} for index, gloss in enumerate(dataset()["vocabulary"])])
    shared = tmp_path / "shared"
    study.freeze(shared, study.REPO / "configs/islr.yaml", split, selection)
    return study, helper, shared, split


def complete_attempt(path, planned, shared, study):
    path.mkdir(parents=True)
    cfg = load_config(str(planned))
    profile, feature = study.planned_cell(cfg)
    for name in ("config.yaml", "split_manifest.json", "selection/glosses.csv", "source_manifest.json"):
        target = path / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(shared / name, target)
    copy2(planned, path / "inputs/planned_config.yaml")
    save_config(cfg, path / "effective_config.yaml")
    identity, parameters = compute_preprocess_identity(cfg)
    study.write_json(path / "cache_audit/preprocess_meta.json", {"config_hash": identity, "parameters": parameters})
    data = study.read_json(shared / "split_manifest.json")
    for name, split in data["splits"].items():
        write_csv(path / "split_snapshot" / (name + ".csv"), [{"sequence_id": sid, **data["sequences"][sid]} for sid in split["sequence_ids"]])
    stages = {}
    for stage in ("preprocess", "split", "train"):
        log = path / "logs" / (stage + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("EXIT_CODE: 0\n")
        stages[stage] = {"exit_code": 0}
    for seed in study.SEEDS:
        run = path / "runs" / f"seed_{seed}"
        run.mkdir(parents=True)
        (run / "best_model.pt").write_bytes(b"synthetic artifact")
        (run / "train_metrics.csv").write_text("epoch,loss\n1,1\n")
        study.write_json(run / "history.json", {"training_time_sec": 1})
        predictions = [{"sequence_id": sid, "participant_id": data["sequences"][sid]["participant_id"],
                        "y_true": index, "y_pred": index, "probabilities": [0.05] * 20}
                       for index, sid in enumerate(data["splits"]["val"]["sequence_ids"])]
        study.write_json(run / "val_report.json", {"preprocess_identity": identity,
            "sample_membership": data["splits"]["val"]["sequence_ids"], "class_names": data["vocabulary"],
            "predictions": predictions, "macro_f1": 0.5, "top1_accuracy": 0.5, "top3_accuracy": 0.8,
            "loss": 1, "model_parameters": 100})
    meta = {"campaign": "v2", "run_id": cfg.run.name, "profile": profile, "feature": feature,
            "owner": study.owner(profile, feature), "dimension": get_feature_dim(cfg), "status": "COMPLETED",
            "stages": stages, "effective_config_sha256": study.sha(path / "effective_config.yaml"),
            "planned_config_sha256": study.sha(path / "inputs/planned_config.yaml")}
    for name, key in (("config.yaml", "config_sha256"), ("split_manifest.json", "split_sha256"),
                      ("selection/glosses.csv", "vocabulary_sha256"), ("source_manifest.json", "source_manifest_sha256")):
        meta[key] = study.sha(path / "inputs" / name)
    study.write_json(path / "metadata.json", meta)
    make_archive(str(path), "zip", root_dir=path.parent, base_dir=path.name)
    return path


@pytest.mark.parametrize("member", ["A", "B", "C"])
def test_notebook_format_syntax_and_safe_defaults(member):
    script = Path(__file__).resolve().parents[1] / "scripts/22_create_ablation_notebooks.py"
    spec = importlib.util.spec_from_file_location("notebook_builder", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    notebook = builder.build(member)
    assert notebook == builder.build(member)
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and not cell["outputs"]
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert f'MEMBER = "{member}"' in source
    assert 'MODE = "prepare"' in source if member == "A" else 'MODE = "train"' in source
    assert 'SOURCE_REF = "ablation-v2-notebooks-v1"' in source
    assert "evaluate_test=True" not in source
    try:
        import nbformat
    except ImportError:
        return
    nbformat.validate(notebook)


def test_shared_validation_runtime_guards_and_safe_archives(workflow, tmp_path):
    study, helper, shared, _ = workflow
    session = helper.MemberSession("C")
    with pytest.raises(KeyError):
        helper.MemberSession("C", {"env.unknown_runtime_key": True})
    with pytest.raises(ValueError, match="runtime"):
        helper.MemberSession("C", {"training.lr": 0.5})
    package = Path(make_archive(str(tmp_path / "shared_bundle"), "zip", root_dir=shared))
    session.import_shared(package)
    assert session.state["provenance"] == study.verify_shared(shared)
    changed = dataset()
    changed["splits"]["val"]["participant_ids"] = ["train_signer"]
    with pytest.raises(ValueError, match="Signer"):
        helper.validate_dataset_manifest(changed, load_config())
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="Đường dẫn"):
        helper._extract_results(archive, tmp_path / "extract")
    assert not (tmp_path / "outside.txt").exists()


def test_raw_preflight_without_optional_clients(workflow, tmp_path, monkeypatch):
    _, helper, shared, _ = workflow
    session = helper.MemberSession("B")
    session.import_shared(shared)
    mirror = tmp_path / "mirror"
    rows = []
    for sid, meta in dataset()["sequences"].items():
        path = mirror / "train_landmark_files" / (sid + ".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"raw fixture presence only")
        rows.append({"sequence_id": sid, "participant_id": meta["participant_id"], "sign": meta["gloss"], "path": str(path.relative_to(mirror))})
    write_csv(mirror / "train.csv", rows)
    session.runtime_overrides["data.local_mirror_dir"] = str(mirror)
    import src.data.kaggle_access
    def forbidden_credentials(**kwargs):
        pytest.fail("Mounted raw data must bypass clients/credentials.")
    monkeypatch.setattr(src.data.kaggle_access, "get_kaggle_credentials", forbidden_credentials)
    session.ensure_data(shared / "config.yaml", dataset())
    (mirror / rows[0]["path"]).unlink()
    monkeypatch.setattr(src.data.kaggle_access, "get_kaggle_credentials", lambda **kwargs: None)
    with pytest.raises(src.data.kaggle_access.KaggleCredentialsNotFoundError, match="Thiếu 1"):
        session.ensure_data(shared / "config.yaml", dataset())


def test_a_prepare_keeps_baseline_and_runs_only_five_preprocess_probes(workflow, monkeypatch):
    study, helper, _, baseline = workflow
    session = helper.MemberSession("A")
    monkeypatch.setattr(session, "ensure_data", lambda *args: None)
    stages = []
    def preprocess(config, stage, log):
        assert stage == "preprocess"
        stages.append(stage)
        cfg = load_config(str(config))
        assert not cfg.features.face
        folder = Path(cfg.data.cache_dir) / "features"
        write_csv(folder / "manifest.csv", [{"sequence_id": sid, **meta} for sid, meta in dataset()["sequences"].items()])
        identity, parameters = compute_preprocess_identity(cfg)
        study.write_json(folder / "preprocess_meta.json", {"config_hash": identity, "parameters": parameters})
    monkeypatch.setattr(session, "_stage", preprocess)
    bundle = session.prepare_shared(baseline)
    assert stages == ["preprocess"] * 5
    actual = study.read_json(session.shared / "split_manifest.json")
    assert actual["splits"] == dataset()["splits"] and actual["vocabulary"] == dataset()["vocabulary"]
    assert session.prepare_shared(baseline) == bundle
    assert len(stages) == 5
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == set(helper.SHARED_FILES)


def test_queue_skip_restore_and_result_folder_isolation(workflow, tmp_path, monkeypatch):
    study, helper, shared, _ = workflow
    session = helper.MemberSession("C")
    session.import_shared(shared)
    calls = []
    def fake_run(planned, actual_shared):
        cfg = load_config(str(planned))
        assert cfg.model.name == "Transformer" and cfg.features.face
        calls.append(cfg.run.name)
        return complete_attempt(session.persistent_root / "ablation_v2_runtime" / cfg.run.name / "attempt01", planned, actual_shared, study)
    monkeypatch.setattr(study, "run_config", fake_run)
    result = session.run_queue(["T0__F0-face"])
    assert result["completed"] == 1 and result["expected"] == 22
    session.run_queue(["T0__F0-face"])
    assert calls == ["T0__F0-face"]
    bundle = session.export_results()
    with zipfile.ZipFile(bundle) as archive:
        assert "queue_status.json" in archive.namelist()
        assert any(name.endswith("val_report.json") for name in archive.namelist())
        assert not any(name.endswith(".zip") or name.startswith("preparation/") for name in archive.namelist())
    import src.utils.env
    monkeypatch.setattr(src.utils.env, "resolve_paths", lambda *args, **kwargs: SimpleNamespace(persistent_root=tmp_path / "fresh_runtime"))
    restored = helper.MemberSession("C")
    restored.restore_results(bundle)
    assert all(path.is_relative_to(restored.root) for path in restored.selected_attempts())
    restored.run_queue(["T0__F0-face"])
    assert calls == ["T0__F0-face"]
    with pytest.raises(ValueError, match="ngoài phân công"):
        restored.run_queue(["G0__F4"])


def test_failed_queue_requires_technical_retry_and_keeps_history(workflow, monkeypatch):
    study, helper, shared, _ = workflow
    session = helper.MemberSession("C")
    session.import_shared(shared)
    calls = []
    def fake_run(planned, actual_shared):
        cfg = load_config(str(planned))
        calls.append(cfg.run.name)
        path = complete_attempt(session.persistent_root / "ablation_v2_runtime" / cfg.run.name / f"attempt{len(calls)}", planned, actual_shared, study)
        if len(calls) == 1:
            metadata = study.read_json(path / "metadata.json")
            metadata.update(status="FAILED", error="synthetic technical failure")
            study.write_json(path / "metadata.json", metadata)
            make_archive(str(path), "zip", root_dir=path.parent, base_dir=path.name)
            raise RuntimeError("synthetic technical failure")
        return path
    monkeypatch.setattr(study, "run_config", fake_run)
    with pytest.raises(RuntimeError, match="synthetic"):
        session.run_queue(["G0__F0"])
    assert session.state["failed"]["G0__F0"] and session.export_results().is_file()
    with pytest.raises(RuntimeError, match="RETRY_FAILED"):
        session.run_queue(["G0__F0"])
    assert len(calls) == 1
    session.run_queue(["G0__F0"], retry_failed=True)
    assert len(calls) == 2 and len(session.selected_attempts()) == 1
    assert list((session.root / "failed_attempts").rglob("metadata.json"))


def test_cross_member_aggregate_uses_selected_paths_only(workflow, monkeypatch):
    study, helper, shared, _ = workflow
    sessions = {member: helper.MemberSession(member) for member in "ABC"}
    for session in sessions.values():
        session.import_shared(shared)
    def fake_run(planned, actual_shared):
        cfg = load_config(str(planned))
        return complete_attempt(sessions["A"].persistent_root / "ablation_v2_runtime" / cfg.run.name / "complete", planned, actual_shared, study)
    monkeypatch.setattr(study, "run_config", fake_run)
    for member, run in (("A", "G0__F1"), ("B", "G-mean__F0"), ("C", "L0__F0")):
        sessions[member].run_queue([run])
    b, c = sessions["B"].export_results(), sessions["C"].export_results()
    result = sessions["A"].aggregate([b, c])
    assert result["completed_cells"] == 3 and result["winner"] is None
    assert "L0__F0 minus G0__F0" not in result["paired_macro_f1_deltas"]
    with pytest.raises(ValueError, match="trùng member"):
        sessions["A"].aggregate([b, b])
