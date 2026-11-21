"""Real Dataset/device-loader/model/report coverage for all 60 frozen cells."""
import csv
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.data.dataset as datasets
from src.features.pipeline import prepare_landmark_sequence
from src.features.schema import compute_preprocess_identity, get_feature_layout_from_config
from src.models.gru import create_model
import src.training.evaluate as evaluation
from src.utils.config import load_config, save_config

REPO = Path(__file__).resolve().parents[4]
HOTFIX_DIR = Path(__file__).resolve().parents[1]
MATRIX = json.loads((REPO / "reports/ablation-study-v2/MATRIX.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def compat(monkeypatch):
    monkeypatch.syspath_prepend(str(HOTFIX_DIR))
    spec = importlib.util.spec_from_file_location("campaign_compat_v2_test", HOTFIX_DIR / "campaign_compat_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(datasets, "load_and_validate_feature_file", datasets.load_and_validate_feature_file)
    monkeypatch.setattr(evaluation, "evaluate_test_set", evaluation.evaluate_test_set)
    monkeypatch.setattr(datasets.DeviceDataLoader, "_reporting_hotfix", None, raising=False)
    monkeypatch.setattr(datasets.DeviceDataLoader, "sampler", None, raising=False)
    monkeypatch.delattr(datasets.DeviceDataLoader, "sampler")
    return module


def sample_geometry():
    t = 12
    hands = np.tile(np.linspace(.1, .3, 63, dtype=np.float32).reshape(1, 1, 21, 3), (t, 2, 1, 1))
    hands[:, 1, :, 0] += .2
    pose = np.zeros((t, 33, 3), dtype=np.float32)
    pose[:, 11], pose[:, 12] = (.3, .4, 0), (.7, .4, 0)
    points = np.array([[.5,.5], [.3,.4], [.7,.4], [.4,.4], [.6,.4], [.35,.38],
                       [.65,.37], [.35,.42], [.65,.43], [.4,.6], [.6,.6], [.5,.58],
                       [.5,.62], [.35,.3], [.65,.3], [.5,.8]], dtype=np.float32)
    face = np.tile(points[None], (t, 1, 1))
    face[:, 12, 1] += np.arange(t) * .001
    return hands, pose, face


def make_cache(root, cfg, count=4):
    cache = root / "features"
    cache.mkdir(parents=True)
    hands, pose, face = sample_geometry()
    result = prepare_landmark_sequence(raw_coords=hands, pose_coords=pose, face_coords=face, cfg=cfg)
    assert result.is_usable
    identity, parameters = compute_preprocess_identity(cfg)
    (cache / "preprocess_meta.json").write_text(json.dumps({"schema_version": 5,
        "config_hash": identity, "parameters": parameters}), encoding="utf-8")
    split = root / "samples.csv"
    with split.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence_id", "participant_id", "gloss", "label", "path"])
        writer.writeheader()
        for i in range(count):
            path = cache / f"sample-{i}.npy"
            np.save(path, result.features)
            writer.writerow({"sequence_id": f"sample-{i}", "participant_id": f"signer-{i % 2}",
                             "gloss": f"class-{i % 3}", "label": i % 3, "path": str(path)})
    return split, identity, result.features


@pytest.mark.parametrize("row", MATRIX, ids=lambda row: row["run_id"])
def test_all_frozen_cells_dataset_augmentation_backward_and_export(compat, tmp_path, row):
    cfg = load_config(str(REPO / f"configs/ablation-study-v2/member-{row['owner'].lower()}/{row['run_id']}.yaml"))
    split, identity, original = make_cache(tmp_path, cfg)
    assert original.shape == (32, row["dimension"])
    if not cfg.features.face:
        before = datasets.load_and_validate_feature_file(tmp_path / "features/sample-0.npy", 32)
    compat.install_process_compatibility()
    compat.install_process_compatibility()
    dataset = datasets.ISRLandmarkDataset(split, sequence_length=32, feature_dim=row["dimension"],
        preload="device", device="cpu", expected_preprocess_identity=identity,
        is_train=True, augment=True, augmentation_config=cfg.augmentation)
    np.testing.assert_array_equal(dataset.preloaded_features[0].numpy(), original)
    if not cfg.features.face:
        np.testing.assert_array_equal(before, dataset.preloaded_features[0].numpy())
    train_loader = datasets.create_dataloader(dataset, batch_size=2, shuffle=True, seed=42)
    features, labels = next(iter(train_loader))
    assert torch.isfinite(features).all()
    model = create_model(input_size=row["dimension"], hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers, num_classes=20, dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional, pooling=cfg.model.pooling, rnn_type=cfg.model.name,
        attention_heads=cfg.model.attention_heads, feedforward_size=cfg.model.feedforward_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    loss = torch.nn.functional.cross_entropy(model(features), labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    validation = datasets.ISRLandmarkDataset(split, sequence_length=32, feature_dim=row["dimension"],
        preload="device", device="cpu", expected_preprocess_identity=identity)
    loader = datasets.create_dataloader(validation, batch_size=3, shuffle=False)
    report = evaluation.evaluate_test_set(model, loader, split, [f"class-{i}" for i in range(20)],
        torch.device("cpu"), report_json_path=tmp_path / "report.json")
    assert [p["sequence_id"] for p in report["predictions"]] == [f"sample-{i}" for i in range(4)]
    assert [p["y_true"] for p in report["predictions"]] == [0, 1, 2, 0]
    assert report["preprocess_identity"] == identity
    assert report["feature_loader_hotfix"]["id"] == compat.HOTFIX_ID
    assert report["reporting_hotfix"]["id"] == compat.reporting.HOTFIX_ID


def test_reproduces_198_rejection_and_retains_strict_validation(compat, tmp_path):
    cfg = load_config(str(REPO / "configs/ablation-study-v2/member-a/G0__F1-face.yaml"))
    _, _, features = make_cache(tmp_path, cfg)
    path = tmp_path / "features/sample-0.npy"
    with pytest.raises(ValueError, match="Unsupported feature dimension 198"):
        datasets.load_and_validate_feature_file(path, 32)
    compat.install_feature_loader_compatibility()
    np.testing.assert_array_equal(datasets.load_and_validate_feature_file(path, 32), features)
    layout = get_feature_layout_from_config(cfg)
    for mask in (layout.slice_hand_masks, layout.slice_pose_mask, layout.slice_geometry_support, layout.slice_face_mask):
        broken = features.copy()
        broken[0, mask] = .5
        np.save(path, broken)
        with pytest.raises(ValueError, match="Non-binary"):
            datasets.load_and_validate_feature_file(path, 32)
    broken = features.copy(); broken[0, 0] = np.nan
    np.save(path, broken)
    with pytest.raises(ValueError, match="Non-finite"):
        datasets.load_and_validate_feature_file(path, 32)
    np.save(path, features)
    with pytest.raises(ValueError, match="sequence length mismatch"):
        datasets.load_and_validate_feature_file(path, 31)
    with pytest.raises(ValueError, match="dimension mismatch"):
        datasets.load_and_validate_feature_file(path, 32, feature_dim=162)
    np.save(path, np.zeros((32, 161), dtype=np.float32))
    with pytest.raises(ValueError, match="Cannot uniquely infer"):
        datasets.load_and_validate_feature_file(path, 32)


def test_live_upgrade_preserves_queue_source_and_arguments(compat, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(REPO / "reports/ablation-study-v2/tools"))
    import study
    calls = []
    monkeypatch.setattr(study, "run_logged", lambda cmd, log: calls.append(cmd))
    state = {"selected": {"G0__F1": "existing-accepted-attempt"}, "failed": {"G0__F1-face": ["failed-attempt"]}}
    saved_state = deepcopy(state)
    session = SimpleNamespace(root=tmp_path, shared=tmp_path / "shared", state=state)
    manifest = study.source_manifest()
    compat.reporting.install(session)
    compat.install(session)
    compat.install(session)
    compat.reporting.install(session)
    command = [sys.executable, "-X", "utf8", "-u", "scripts/20_train.py", "--config", "same.yaml"]
    study.run_logged(command, tmp_path / "train.log")
    assert calls[0] == [*command[:4], str(HOTFIX_DIR / "campaign_compat_v2.py"), "--bootstrap", *command[4:]]
    prepare = [sys.executable, "scripts/10_prepare_data.py", "--stage", "preprocess"]
    study.run_logged(prepare, tmp_path / "prepare.log")
    assert calls[1] == prepare
    assert session.state == saved_state and study.source_manifest() == manifest
    assert (tmp_path / "FEATURE_LOADER_HOTFIX.json").is_file()
    assert (tmp_path / "compatibility/campaign_compat_v2.py").read_bytes() == Path(compat.__file__).read_bytes()


def test_real_training_cli_child_three_seeds_face_and_reports(compat, tmp_path):
    cfg = load_config(str(REPO / "configs/ablation-study-v2/member-a/G0__F1-face.yaml"))
    cfg.env.local_root = str(tmp_path)
    cfg.data.cache_dir = str(tmp_path / "cache")
    cfg.training.device = "cpu"
    cfg.training.epochs = 1  # Smoke fixture only; frozen campaign configs stay unchanged.
    cfg.training.early_stopping_min_epochs = 0
    cfg.run.output_dir = "runs"
    cfg.validate()
    split, _, _ = make_cache(tmp_path, cfg, count=6)
    splits = tmp_path / "data/splits"
    splits.mkdir(parents=True)
    rows = list(csv.DictReader(split.open(encoding="utf-8")))
    for name, selected in (("train", rows), ("val", rows[:3]), ("test", rows[3:])):
        with (splits / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(selected)
    (splits / "label_map.json").write_text(json.dumps({f"class-{i}": i for i in range(3)}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    save_config(cfg, config)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
    result = subprocess.run([sys.executable, "-X", "utf8", str(HOTFIX_DIR / "campaign_compat_v2.py"),
        "--bootstrap", str(REPO / "scripts/20_train.py"), "--config", str(config)],
        cwd=REPO, env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    reports = sorted((tmp_path / "runs").rglob("val_report.json"))
    assert len(reports) == 3
    assert len(list((tmp_path / "runs").rglob("best_model.pt"))) == 3
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        assert [p["sequence_id"] for p in report["predictions"]] == [f"sample-{i}" for i in range(3)]
        assert report["feature_loader_hotfix"]["id"] == compat.HOTFIX_ID
        assert report["reporting_hotfix"]["id"] == compat.reporting.HOTFIX_ID
    assert not list((tmp_path / "runs").rglob("test_report.json"))
