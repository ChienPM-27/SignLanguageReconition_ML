"""Self-contained anatomical ablation, CLI export and frozen cross-plan checks."""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.data.dataset import ISRLandmarkDataset, create_dataloader, load_and_validate_feature_file
from src.features.pipeline import prepare_landmark_sequence
from src.features.schema import compute_preprocess_identity
from src.models.gru import create_model
from src.utils.checkpoint import validate_model_architecture
from src.utils.config import Config, build_config_from_dict, load_config, save_config

TOOLS = Path(__file__).resolve().parents[1] / "reports/ablation-study-v3/tools"
sys.path.insert(0, str(TOOLS))
import study_v3
from session_v3 import MemberSession, extract


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(tmp_path, profile="G0", feature="HBF"):
    path = tmp_path / "base.yaml"
    save_config(Config(), path)
    return study_v3.config_for(path, profile, feature)


def model(cfg):
    return create_model(input_size=196, hidden_size=cfg.model.hidden_size,
                        num_layers=cfg.model.num_layers, num_classes=3, dropout=cfg.model.dropout,
                        bidirectional=cfg.model.bidirectional, pooling=cfg.model.pooling,
                        rnn_type=cfg.model.name, region_fusion=cfg.model.region_fusion,
                        regions=cfg.model.regions, region_weights=cfg.model.region_weights,
                        region_normalization_epsilon=cfg.model.region_normalization_epsilon)


def geometry():
    hands = np.tile(np.linspace(.1, .3, 63, dtype=np.float32).reshape(1, 1, 21, 3), (12, 2, 1, 1))
    hands[:, 1, :, 0] += .2
    pose = np.zeros((12, 33, 3), dtype=np.float32)
    pose[:, 11], pose[:, 12] = (.3, .4, 0), (.7, .4, 0)
    face = np.tile(np.array([[.5,.5], [.3,.4], [.7,.4], [.4,.4], [.6,.4], [.35,.38],
                            [.65,.37], [.35,.42], [.65,.43], [.4,.6], [.6,.6], [.5,.58],
                            [.5,.62], [.35,.3], [.65,.3], [.5,.8]], dtype=np.float32)[None], (12, 1, 1))
    return hands, pose, face


@pytest.mark.parametrize("row", study_v3.matrix(), ids=lambda r: r["run_id"])
def test_all_42_eligible_cells_forward_backward_and_region_isolation(tmp_path, row):
    cfg = config(tmp_path, row["profile"], row["feature"])
    hands, pose, face = geometry()
    result = prepare_landmark_sequence(raw_coords=hands, pose_coords=pose, face_coords=face, cfg=cfg)
    assert result.is_usable and result.features.shape == (32, 196)
    network = model(cfg).eval()
    x = torch.from_numpy(result.features).unsqueeze(0)
    before = network(x)
    mutated = x.clone()
    for region in set(study_v3.REGIONS["HBF"]) - set(cfg.model.regions):
        mutated[..., network.slices[region]] = float("nan")
    torch.testing.assert_close(before, network(mutated), rtol=0, atol=0)
    before.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    for region in set(study_v3.REGIONS["HBF"]) - set(cfg.model.regions):
        assert network.projections[region].weight.grad is None
    rebuilt = model(cfg).eval()
    rebuilt.load_state_dict(network.state_dict())
    validate_model_architecture(cfg, rebuilt)
    torch.testing.assert_close(before, rebuilt(x), rtol=0, atol=0)


def test_exact_weights_missing_support_and_initialization(tmp_path):
    cfg = config(tmp_path)
    torch.manual_seed(42)
    full = model(cfg)
    x = torch.ones(1, 2, 196)
    for projection in full.projections.values():
        torch.nn.init.ones_(projection.weight)
    expected = torch.full((1, 2, 128), 1 / np.sqrt(128), dtype=torch.float32)
    torch.testing.assert_close(full.fuse(x), expected)
    x[..., 195] = 0
    torch.testing.assert_close(full.fuse(x), expected * .9)
    cfg.model.regions = ["hands", "body"]
    pair = model(cfg)
    assert pair.fusion_weights == pytest.approx({"hands": 7/9, "body": 2/9})
    cfg.model.regions = ["face"]
    face = model(cfg)
    assert face.fusion_weights == {"face": 1.0}
    torch.testing.assert_close(face.fuse(torch.zeros(1, 2, 196)), torch.zeros(1, 2, 128))
    cfg.model.regions = ["hands", "body", "face"]
    torch.manual_seed(44)
    first = model(cfg)
    cfg.model.regions = ["face"]
    torch.manual_seed(44)
    second = model(cfg)
    for a, b in zip(first.state_dict().values(), second.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_strict_configs_and_independent_region_preprocessing(tmp_path):
    cfg = config(tmp_path)
    raw = cfg.to_dict()
    for key in ("torso", "hand"):
        changed = deepcopy(raw)
        changed["model"]["region_weights"][key] = .1
        with pytest.raises(KeyError):
            build_config_from_dict(changed)
    path = tmp_path / "canonical.yaml"
    save_config(cfg, path)
    with pytest.raises(KeyError):
        load_config(path, overrides=["model.region_weights.unknown=0.2"])
    with pytest.raises(KeyError):
        load_config(path, overrides=["model.region_weights.body.foo=0.2"])
    for values in (["face", "face"], [], ["unknown"]):
        changed = deepcopy(raw)
        changed["model"]["regions"] = values
        with pytest.raises(ValueError):
            build_config_from_dict(changed)
    for key in ("region_normalization_epsilon",):
        changed = deepcopy(raw); changed["model"][key] = float("nan")
        with pytest.raises(ValueError):
            build_config_from_dict(changed)
    identities = {compute_preprocess_identity(config(tmp_path, "G0", feature))[0] for feature in study_v3.REGIONS}
    assert len(identities) == 1
    hands, pose, face = geometry()
    original = prepare_landmark_sequence(raw_coords=hands, pose_coords=pose, face_coords=face, cfg=cfg).features
    modified = hands.copy(); modified[:, :, :, 0] += .15
    changed = prepare_landmark_sequence(raw_coords=modified, pose_coords=pose, face_coords=face, cfg=cfg).features
    np.testing.assert_array_equal(original[:, 143:], changed[:, 143:])
    bad = deepcopy(cfg); bad.model.regions = ["face"]
    with pytest.raises(ValueError, match="region settings"):
        validate_model_architecture(bad, model(cfg))


@pytest.mark.parametrize("profile", ["G0", "L0", "T0"])
def test_actual_cli_three_seeds_export_and_checkpoint(tmp_path, profile, monkeypatch):
    cfg = config(tmp_path, profile)
    cfg.env.local_root = str(tmp_path)
    cfg.training.device = "cpu"
    cfg.training.epochs = 1
    cfg.training.early_stopping_min_epochs = 0
    cfg.run.output_dir = "runs"
    cfg.validate()
    cache = tmp_path / "features"; cache.mkdir()
    identity, parameters = compute_preprocess_identity(cfg)
    (cache / "preprocess_meta.json").write_text(json.dumps({"schema_version": 5, "config_hash": identity, "parameters": parameters}))
    hands, pose, face = geometry()
    features = prepare_landmark_sequence(raw_coords=hands, pose_coords=pose, face_coords=face, cfg=cfg).features
    rows = []
    for i in range(6):
        path = cache / f"sample-{i}.npy"; np.save(path, features)
        rows.append({"sequence_id": f"sample-{i}", "participant_id": str(i % 2), "gloss": f"class-{i % 3}", "label": i % 3, "path": str(path)})
    splits = tmp_path / "data/splits"; splits.mkdir(parents=True)
    for split, selection in (("train", rows), ("val", rows[:3]), ("test", rows[3:])):
        with (splits / (split + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(selection)
    (splits / "label_map.json").write_text(json.dumps({f"class-{i}": i for i in range(3)}))
    config_path = tmp_path / "train.yaml"; save_config(cfg, config_path)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
    completed = subprocess.run([sys.executable, "-X", "utf8", "scripts/20_train.py", "--config", str(config_path)],
                               cwd=TOOLS.parents[2], env=env, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    reports = list((tmp_path / "runs").rglob("val_report.json"))
    assert len(reports) == 3 and not list((tmp_path / "runs").rglob("test_report.json"))
    for report in reports:
        payload = json.loads(report.read_text())
        assert [p["sequence_id"] for p in payload["predictions"]] == [f"sample-{i}" for i in range(3)]
        checkpoint = torch.load(report.parent / "best_model.pt", map_location="cpu", weights_only=False)
        restored_cfg = build_config_from_dict(checkpoint["resolved_config"])
        restored = model(restored_cfg)
        restored.load_state_dict(checkpoint["model_state_dict"])
        validate_model_architecture(restored_cfg, restored)
        assert checkpoint["seed"] in study_v3.SEEDS
    evaluated = subprocess.run([sys.executable, "-X", "utf8", "scripts/21_evaluate.py",
                               "--checkpoint", str(reports[0].parent / "best_model.pt"), "--split", "val",
                               "--device", "cpu", "--output-dir", str(tmp_path / "standalone")],
                               cwd=TOOLS.parents[2], env=env, capture_output=True, text=True, timeout=60)
    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
    assert (tmp_path / "standalone/evaluation_val_report.json").is_file()
    dataset = ISRLandmarkDataset(splits / "val.csv", preload="device", device="cpu", feature_dim=196)
    assert len(list(create_dataloader(dataset, batch_size=2, shuffle=False).sampler)) == 3
    broken = features.copy(); broken[0, 195] = .5; np.save(cache / "bad.npy", broken)
    with pytest.raises(ValueError, match="Non-binary face"):
        load_and_validate_feature_file(cache / "bad.npy", 32)
    from shutil import copy2, copytree
    import hashlib
    attempt = tmp_path / "attempt"; inputs = attempt / "inputs"; inputs.mkdir(parents=True)
    row = next(r for r in study_v3.matrix() if r["run_id"] == profile + "__HBF")
    save_config(cfg, inputs / "config.yaml")
    frozen = {"vocabulary": [f"class-{i}" for i in range(3)],
              "sequences": {r["sequence_id"]: {"participant_id": r["participant_id"], "label": r["label"]} for r in rows},
              "splits": {name: {"sequence_ids": [r["sequence_id"] for r in selected]}
                         for name, selected in (("train", rows), ("val", rows[:3]), ("test", rows[3:]))}}
    study_v3.write(inputs / "split_manifest.json", frozen)
    (inputs / "selection").mkdir(); (inputs / "selection/glosses.csv").write_text("gloss,label\nclass-0,0\nclass-1,1\nclass-2,2\n")
    source = {"fixture.py": "self-contained"}
    monkeypatch.setattr(study_v3, "source_manifest", lambda: source)
    study_v3.write(inputs / "source_manifest.json", source)
    study_v3.write(inputs / "matrix.json", study_v3.matrix())
    study_v3.write(inputs / "protocol.json", {"fixture": True})
    hashes = {name: study_v3.sha(inputs / name) for name in study_v3.SHARED_FILES if name != "freeze.json"}
    freeze_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]
    study_v3.write(inputs / "freeze.json", {"campaign": "v3-regions-701020", "freeze_id": freeze_id, "files": hashes})
    provenance = study_v3.verify_shared(inputs)
    save_config(cfg, attempt / "effective_config.yaml")
    copytree(tmp_path / "runs", attempt / "runs")
    copytree(splits, attempt / "split_snapshot")
    (attempt / "cache_audit").mkdir(); copy2(cache / "preprocess_meta.json", attempt / "cache_audit/preprocess_meta.json")
    (attempt / "logs").mkdir()
    for stage in ("train", "split", "preprocess"):
        (attempt / "logs" / (stage + ".log")).write_text(completed.stdout if stage == "train" else "fixture stage\n")
    inventory = {p.relative_to(attempt).as_posix(): study_v3.sha(p) for p in attempt.rglob("*") if p.is_file()}
    study_v3.write(attempt / "ARTIFACTS.json", inventory)
    metadata = {"campaign": "v3-regions-701020", **row, "status": "COMPLETED", "provenance": provenance,
                "stages": {s: {"exit_code": 0} for s in ("train", "split", "preprocess")},
                "artifact_inventory_sha256": study_v3.sha(attempt / "ARTIFACTS.json")}
    study_v3.write(attempt / "metadata.json", metadata)
    assert study_v3.collect(attempt)["run_id"] == row["run_id"]
    session = MemberSession(row["owner"], inputs, tmp_path, tmp_path / "storage")
    destination = session.root / "runs" / row["run_id"] / "fixture"
    copytree(attempt, destination)
    session.state["selected"][row["run_id"]] = destination.relative_to(session.root).as_posix()
    exported = session.export_results()
    restored = MemberSession(row["owner"], inputs, tmp_path, tmp_path / "fresh")
    restored.restore_results(exported)
    monkeypatch.setattr(restored, "check_data", lambda: None)
    monkeypatch.setattr(study_v3, "run", lambda *args: pytest.fail("Accepted run was trained again"))
    assert restored.run_queue([row["run_id"]])["completed"] == 1
    corrupted = destination / "runs" / reports[0].relative_to(tmp_path / "runs")
    payload = json.loads(corrupted.read_text()); payload["macro_f1"] = .1234
    study_v3.write(corrupted, payload)
    updated = study_v3.read(destination / "ARTIFACTS.json")
    updated[corrupted.relative_to(destination).as_posix()] = study_v3.sha(corrupted)
    study_v3.write(destination / "ARTIFACTS.json", updated)
    metadata["artifact_inventory_sha256"] = study_v3.sha(destination / "ARTIFACTS.json")
    study_v3.write(destination / "metadata.json", metadata)
    with pytest.raises(ValueError, match="metrics lệch predictions"):
        study_v3.collect(destination)


def record(row, value):
    return {**row, "status": "COMPLETED", "provenance": {"freeze_id": "fixture"}, "memberships": {"val": ["1"]},
            "metrics": {"macro_f1": study_v3.aggregate([value]*3), "model_parameters": study_v3.aggregate([100]*3)}}


def test_cross_plan_completion_and_no_duplicate_training(monkeypatch, tmp_path):
    rows = [r for r in study_v3.matrix() if r["phase"] == 1]
    fixtures = {r["run_id"]: record(r, .9 if r["profile"] == "G-attn" else .8) for r in rows}
    monkeypatch.setattr(study_v3, "collect", lambda path: fixtures[str(path)])
    result = study_v3.summarize([Path(name) for name in fixtures], tmp_path / "results.md")
    assert result["expected_cells"] == 30 and result["winner"] is None
    assert result["cross_plan"]["profile"] == "G-attn" and len(result["missing"]) == 6
    for row in study_v3.matrix():
        if row["run_id"] in result["missing"]:
            fixtures[row["run_id"]] = record(row, .91)
    final = study_v3.summarize([Path(name) for name in fixtures], tmp_path / "results.md")
    assert final["status"] == "COMPLETED" and final["completed_cells"] == 30
    with pytest.raises(ValueError, match="Duplicate"):
        study_v3.summarize([Path("G0__H"), Path("G0__H")], tmp_path / "bad.md")
    session = object.__new__(MemberSession)
    session.member, session.freeze_id, session.root = "B", "fixture", tmp_path
    session.state_path = tmp_path / "state.json"
    session.state = {"cross_plan": None}
    study_v3.write(tmp_path / "plan.json", result["cross_plan"])
    session.load_cross_plan(tmp_path / "plan.json")
    corrupt = deepcopy(result["cross_plan"]); corrupt["profile"] = "G-bi"
    study_v3.write(tmp_path / "bad_plan.json", corrupt)
    with pytest.raises(ValueError):
        session.load_cross_plan(tmp_path / "bad_plan.json")


def test_zip_restore_rejects_traversal(tmp_path):
    import zipfile
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as archive:
        archive.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError):
        extract(tmp_path / "bad.zip", tmp_path / "destination")
    assert not (tmp_path / "outside.txt").exists()


def test_test_set_stays_locked_before_campaign_completion():
    session = object.__new__(MemberSession)
    session.member = "A"
    with pytest.raises(ValueError, match="khóa winner"):
        session.finalize_test({"status": "PENDING", "winner": None})
