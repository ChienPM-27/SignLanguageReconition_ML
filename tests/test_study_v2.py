"""Experiment coverage, cohort preservation, artifact guards and cross comparisons."""
import csv
import importlib
import json
from pathlib import Path
from collections import Counter
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.features.schema import compute_preprocess_identity
from src.utils.config import load_config, save_config


@pytest.fixture
def study_tools(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "reports/ablation-study-v2/tools"
    monkeypatch.syspath_prepend(str(folder))
    return importlib.import_module("study"), importlib.import_module("summarize")


def split_fixture():
    data = {"schema_version": 1, "seed": 42, "vocabulary": ["hello", "bye"],
            "label_map": {"hello": 0, "bye": 1}, "sequences": {}, "splits": {}}
    for split in ("train", "val", "test"):
        ids = [split + "0", split + "1"]
        data["splits"][split] = {"sequence_ids": ids, "participant_ids": [split + "_person"]}
        for label, sid in enumerate(ids):
            data["sequences"][sid] = {"participant_id": split + "_person", "gloss": data["vocabulary"][label], "label": label}
    return data


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def attempt_fixture(folder, tools, profile="G0", feature="F0", score=1.0):
    study, _ = tools
    split = split_fixture()
    inputs = folder / "inputs"
    inputs.mkdir(parents=True)
    cfg = load_config()
    save_config(cfg, inputs / "config.yaml")
    study.write_json(inputs / "split_manifest.json", split)
    study.write_json(inputs / "source_manifest.json", {"src/example.py": "abc"})
    csv_file(inputs / "selection/glosses.csv", [{"gloss": g, "label": i} for i, g in enumerate(split["vocabulary"])])
    effective = study.config_for(inputs / "config.yaml", profile, feature)
    save_config(effective, folder / "effective_config.yaml")
    identity, params = compute_preprocess_identity(effective)
    study.write_json(folder / "cache_audit/preprocess_meta.json", {"config_hash": identity, "parameters": params})
    for name in split["splits"]:
        csv_file(folder / "split_snapshot" / (name + ".csv"), [{"sequence_id": sid, **split["sequences"][sid]} for sid in split["splits"][name]["sequence_ids"]])
    stages = {}
    (folder / "logs").mkdir()
    for stage in ("preprocess", "split", "train"):
        stages[stage] = {"exit_code": 0}
        (folder / "logs" / (stage + ".log")).write_text("EXIT_CODE: 0\n")
    for seed in study.SEEDS:
        directory = folder / "runs" / f"seed_{seed}"
        directory.mkdir(parents=True)
        (directory / "best_model.pt").write_bytes(b"fixture checkpoint")
        (directory / "train_metrics.csv").write_text("fixture log")
        study.write_json(directory / "history.json", {"training_time_sec": 1})
        predictions = [{"sequence_id": sid, "participant_id": split["sequences"][sid]["participant_id"],
                        "y_true": i, "y_pred": i if score else 1-i, "probabilities": [0.5, 0.5]}
                       for i, sid in enumerate(split["splits"]["val"]["sequence_ids"])]
        study.write_json(directory / "val_report.json", {
            "preprocess_identity": identity, "sample_membership": split["splits"]["val"]["sequence_ids"],
            "class_names": split["vocabulary"], "predictions": predictions, "macro_f1": score,
            "top1_accuracy": score, "top3_accuracy": 1, "loss": 1, "model_parameters": 100,
        })
    meta = {"campaign": "v2", "run_id": profile + "__" + feature, "profile": profile, "feature": feature,
            "dimension": study.imports()[2](effective), "status": "COMPLETED", "stages": stages,
            "effective_config_sha256": study.sha(folder / "effective_config.yaml")}
    for filename, key in (("config.yaml", "config_sha256"), ("split_manifest.json", "split_sha256"),
                          ("selection/glosses.csv", "vocabulary_sha256"), ("source_manifest.json", "source_manifest_sha256")):
        meta[key] = study.sha(inputs / filename)
    study.write_json(folder / "metadata.json", meta)
    return folder


def test_full_matrix_and_single_factor_architecture(study_tools):
    study, _ = study_tools
    rows = study.matrix()
    assert len(rows) == 60 and len({r["run_id"] for r in rows}) == 60
    assert Counter(r["owner"] for r in rows) == {"A": 8, "B": 30, "C": 22}
    assert sum(len(r["seeds"]) for r in rows) == 180
    base = study.config_for(study.REPO / "configs/islr.yaml", "G0", "F0")
    for profile, changed in (("G-mean", "pooling"), ("G-attn", "pooling"), ("G-bi", "bidirectional"), ("L0", "name"), ("T0", "name")):
        cfg = study.config_for(study.REPO / "configs/islr.yaml", profile, "F0")
        assert {k for k, v in cfg.to_dict()["model"].items() if v != base.to_dict()["model"][k]} == {changed}
    dims = {row["feature"]: row["dimension"] for row in rows}
    assert dims == {"F0": 160, "F0-face": 196, "F1": 162, "F1-face": 198,
                    "F2": 162, "F2-face": 198, "F3": 192, "F3-face": 228, "F4": 212, "F4-face": 248}


def test_common_cohort_rejects_class_loss_and_duplicate_caches(tmp_path, study_tools):
    study, _ = study_tools
    split = split_fixture()
    base = tmp_path / "base.json"
    study.write_json(base, split)
    manifests = []
    for name in study.FEATURES:
        folder = tmp_path / name
        cfg = study.config_for(study.REPO / "configs/islr.yaml", "G0", name)
        csv_file(folder / "manifest.csv", [{"sequence_id": sid, **meta} for sid, meta in split["sequences"].items()])
        study.write_json(folder / "preprocess_meta.json", {"parameters": compute_preprocess_identity(cfg)[1]})
        manifests.append(folder / "manifest.csv")
    result = study.common_cohort(base, manifests, tmp_path / "common.json")
    assert result["splits"] == split["splits"]
    with pytest.raises(ValueError, match="không lặp"):
        study.common_cohort(base, [manifests[0]] * 5, tmp_path / "duplicate.json")
    csv_file(manifests[-1], [{"sequence_id": sid, **meta} for sid, meta in split["sequences"].items() if sid != "val1"])
    with pytest.raises(ValueError, match="thiếu class"):
        study.common_cohort(base, manifests, tmp_path / "bad.json")


def test_summary_pending_deltas_and_artifact_guards(tmp_path, study_tools):
    study, summary = study_tools
    off = attempt_fixture(tmp_path / "off", study_tools, score=0)
    on = attempt_fixture(tmp_path / "on", study_tools, feature="F0-face", score=1)
    result = summary.summarize([off, on], tmp_path / "RESULTS.md")
    assert result["winner"] is None and result["completed_cells"] == 2 and result["status"] == "PENDING"
    assert result["paired_macro_f1_deltas"]["G0__F0-face minus G0__F0"]["mean"] == 1
    with pytest.raises(ValueError, match="Duplicate attempt"):
        summary.summarize([off, off], tmp_path / "bad.md")
    (on / "runs/seed_42/test_report.json").write_text("{}")
    with pytest.raises(ValueError, match="chưa evaluate test"):
        summary.collect(on)
    (on / "runs/seed_42/test_report.json").unlink()
    report = on / "runs/seed_42/val_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["predictions"][0]["participant_id"] = "foreign"
    study.write_json(report, payload)
    with pytest.raises(ValueError, match="provenance"):
        summary.collect(on)


def test_summary_complete_selects_across_all_profiles(tmp_path, study_tools):
    study, summary = study_tools
    selected = []
    for row in study.matrix():
        best = row["profile"] == "L0" and row["feature"] == "F0"
        selected.append(attempt_fixture(tmp_path / row["run_id"], study_tools,
                                        row["profile"], row["feature"], score=float(best)))
    result = summary.summarize(selected, tmp_path / "RESULTS.md")
    assert result["status"] == "COMPLETED" and result["winner"] == "L0__F0"
    assert result["model_face_interactions"]["L0 versus G0: face on F0"]["mean"] == -1


def test_source_freeze_and_failed_runner_release_lock(tmp_path, study_tools, monkeypatch):
    study, _ = study_tools
    state = {"src/example.py": "same"}
    monkeypatch.setattr(study, "source_manifest", lambda: dict(state))
    split = tmp_path / "base_split.json"
    study.write_json(split, split_fixture())
    selection = tmp_path / "selection"
    csv_file(selection / "glosses.csv", [{"gloss": "hello", "label": 0}, {"gloss": "bye", "label": 1}])
    shared = tmp_path / "shared"
    study.freeze(shared, study.REPO / "configs/islr.yaml", split, selection)
    assert len(study.read_json(shared / "matrix.json")) == 60
    assert study.verify_shared(shared)["source_manifest_sha256"]
    with pytest.raises(FileExistsError):
        study.freeze(shared, study.REPO / "configs/islr.yaml", split, selection)
    state["src/example.py"] = "changed"
    with pytest.raises(ValueError, match="Source khác"):
        study.verify_shared(shared)
    state["src/example.py"] = "same"
    import src.utils.env
    monkeypatch.setattr(src.utils.env, "resolve_paths", lambda *args, **kwargs: SimpleNamespace(persistent_root=tmp_path))
    def fail(command, log):
        raise RuntimeError("fixture stage failure")
    monkeypatch.setattr(study, "run_logged", fail)
    with pytest.raises(RuntimeError, match="fixture stage failure"):
        study.run("T0", "F0-face", shared)
    assert not (tmp_path / "ablation_v2_runtime/.runtime.lock").exists()
    attempts = list((tmp_path / "ablation_v2_runtime").rglob("metadata.json"))
    assert len(attempts) == 1 and study.read_json(attempts[0])["status"] == "FAILED"
    assert list((tmp_path / "ablation_v2_runtime").rglob("*.zip"))


def test_matrix_cli_utf8_even_with_windows_console_encoding(tmp_path, study_tools):
    study, _ = study_tools
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    output = tmp_path / "matrix.json"
    result = subprocess.run([sys.executable, str(Path(study.__file__)), "matrix", "--output", str(output)],
                            env=env, capture_output=True, check=True)
    assert "Đã lưu 60 cells" in result.stdout.decode("utf-8")
    assert len(json.loads(output.read_text(encoding="utf-8"))) == 60


def test_export_member_configs_complete_and_isolated(tmp_path, study_tools):
    study, _ = study_tools
    output = tmp_path / "configs"
    study.member_configs(output, study.REPO / "configs/islr.yaml")
    configs = list(output.rglob("*.yaml"))
    assert len(configs) == 60
    roots = set()
    for member, count in (("A", 8), ("B", 30), ("C", 22)):
        folder = output / ("member-" + member.lower())
        queue = study.read_json(folder / "QUEUE.json")
        assert len(queue) == count
        for row in queue:
            cfg = load_config(str(folder / row["config"]))
            profile, feature = study.planned_cell(cfg)
            assert (profile, feature) == (row["profile"], row["feature"])
            assert study.owner(profile, feature) == member
            assert cfg.features.face == feature.endswith("-face")
            assert cfg.training.seeds == [42, 43, 44] and not cfg.evaluation.evaluate_test
            assert cfg.env.project_dir_name.endswith("member-" + member.lower())
            roots.add(cfg.env.project_dir_name)
            study.validate_planned(cfg, study.config_for(study.REPO / "configs/islr.yaml", profile, feature))
    assert len(roots) == 3
    with pytest.raises(FileExistsError):
        study.member_configs(output, study.REPO / "configs/islr.yaml")


@pytest.mark.parametrize("section,key,value", [("training", "lr", 0.5), ("features", "face", False), ("env", "seed", 43)])
def test_run_config_rejects_protocol_drift_before_runtime(tmp_path, study_tools, monkeypatch, section, key, value):
    study, _ = study_tools
    shared = tmp_path / "shared"
    shared.mkdir()
    save_config(study.config_for(study.REPO / "configs/islr.yaml", "G0", "F0"), shared / "config.yaml")
    monkeypatch.setattr(study, "verify_shared", lambda path: {})
    cfg = study.config_for(shared / "config.yaml", "T0", "F0-face")
    cfg.run.name = "T0__F0-face"
    setattr(getattr(cfg, section), key, value)
    planned = tmp_path / "planned.yaml"
    save_config(cfg, planned)
    import src.utils.env
    def unexpected_runtime(*args, **kwargs):
        pytest.fail("Protocol drift must be rejected before touching runtime/data.")
    monkeypatch.setattr(src.utils.env, "resolve_paths", unexpected_runtime)
    with pytest.raises(ValueError, match=f"Config {section}"):
        study.run_config(planned, shared)


def test_run_config_reads_face_transformer_and_archives_plan(tmp_path, study_tools, monkeypatch):
    study, _ = study_tools
    monkeypatch.setattr(study, "source_manifest", lambda: {"src/example.py": "same"})
    split = tmp_path / "split.json"
    study.write_json(split, split_fixture())
    selection = tmp_path / "selection"
    csv_file(selection / "glosses.csv", [{"gloss": "hello", "label": 0}, {"gloss": "bye", "label": 1}])
    shared = tmp_path / "shared"
    study.freeze(shared, study.REPO / "configs/islr.yaml", split, selection)
    cfg = study.config_for(shared / "config.yaml", "T0", "F0-face")
    cfg.run.name = "T0__F0-face"
    cfg.env.project_dir_name = "isolated-member-c"
    cfg.data.local_mirror_dir = str(tmp_path / "mirror")
    planned = tmp_path / "planned.yaml"
    save_config(cfg, planned)
    import src.utils.env
    def resolved(env, actual, **kwargs):
        assert actual.env.project_dir_name == "isolated-member-c"
        assert actual.data.local_mirror_dir == str(tmp_path / "mirror")
        return SimpleNamespace(persistent_root=tmp_path / "runtime-c")
    monkeypatch.setattr(src.utils.env, "resolve_paths", resolved)
    def inspect_command(command, log):
        actual = load_config(command[command.index("--config") + 1])
        assert actual.model.name == "Transformer" and actual.features.face
        assert actual.training.seeds == [42, 43, 44]
        assert Path(actual.split.manifest_path).parts[-2:] == ("inputs", "split_manifest.json")
        raise RuntimeError("stop synthetic preflight")
    monkeypatch.setattr(study, "run_logged", inspect_command)
    with pytest.raises(RuntimeError, match="synthetic preflight"):
        study.run_config(planned, shared)
    metadata = next((tmp_path / "runtime-c").rglob("metadata.json"))
    assert study.read_json(metadata)["planned_config_sha256"] == study.sha(metadata.parent / "inputs/planned_config.yaml")
    assert not (tmp_path / "runtime-c/ablation_v2_runtime/.runtime.lock").exists()


def test_summary_rejects_changed_planned_config(tmp_path, study_tools):
    study, summary = study_tools
    attempt = attempt_fixture(tmp_path / "attempt", study_tools)
    plan = attempt / "inputs/planned_config.yaml"
    save_config(study.config_for(study.REPO / "configs/islr.yaml", "G0", "F0"), plan)
    meta = study.read_json(attempt / "metadata.json")
    meta["planned_config_sha256"] = study.sha(plan)
    study.write_json(attempt / "metadata.json", meta)
    summary.collect(attempt)
    plan.write_text(plan.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Planned config"):
        summary.collect(attempt)
