"""Regression test suite for Wave A review findings.

Guarantees:
1. Strict resume validation with zero model parameter mutation on failure.
2. Predictor rejects invalid checkpoint schema/identity.
3. Dataset rejects foreign cache without local preprocess_meta.json.
4. Training CLI strictly rejects mismatch between runtime config, train cache, and val cache.
5. Split creation rejects signer drift and class disappearance under frozen manifests.
6. Evaluation CLI rejects checkpoints with missing or corrupted metadata.
"""
import copy
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.data.dataset import ISRLandmarkDataset
from src.data.split import create_signer_splits
from src.features.schema import PREPROCESS_SCHEMA_VERSION, compute_preprocess_identity
from src.inference.predictor import ISLRPredictor
from src.models.gru import ASLGRUClassifier
from src.training.train import train_gru
from src.utils.config import Config
from src.utils.env import resolve_paths


def _create_base_config() -> Config:
    cfg = Config()
    cfg.features.set = "hands"
    cfg.model.hidden_size = 16
    cfg.model.num_layers = 1
    cfg.model.dropout = 0.0
    cfg.training.epochs = 1
    cfg.training.preload = "none"
    cfg.training.batch_size = 2
    cfg.training.seeds = [42]
    cfg.training.num_seeds = 1
    cfg.training.device = "cpu"
    for name in ["flip", "scale_translate", "rotate", "jitter", "time_stretch", "shape_scale"]:
        getattr(cfg.augmentation, name)["enabled"] = False
    cfg.validate()
    return cfg


def _make_model(cfg: Config) -> ASLGRUClassifier:
    return ASLGRUClassifier(
        input_size=143,
        hidden_size=16,
        num_layers=1,
        num_classes=2,
        dropout=0.0,
        pooling=cfg.model.pooling,
    )


def _write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence_id", "participant_id", "gloss", "label", "path"])
        w.writerows(rows)


def _write_cache_meta(path: Path, cfg: Config) -> str:
    path.mkdir(parents=True, exist_ok=True)
    identity, params = compute_preprocess_identity(cfg)
    meta = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "config_hash": identity,
        "parameters": params,
    }
    (path / "preprocess_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return identity


def _write_dummy_feature(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ones((32, 143), dtype=np.float32)
    np.save(str(path), arr)


@pytest.mark.parametrize(
    "case",
    [
        "unknown_config",
        "config_identity_mismatch",
        "reversed_labels",
        "extra_tensor",
        "bad_optimizer",
        "same_shape_pooling_mismatch",
        "missing_classes_and_labels",
        "missing_optimizer_state",
        "invalid_epoch_after_commit",
    ],
)
def test_resume_checkpoint_rejection_and_atomic_rollback(tmp_path: Path, case: str) -> None:
    """Finding 3 & 4: Resume checkpoint must validate strictly and never mutate model state on failure."""
    cfg = _create_base_config()
    identity, _ = compute_preprocess_identity(cfg)
    model = _make_model(cfg)
    before_state = copy.deepcopy(model.state_dict())

    payload = {
        "epoch": 1,
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "preprocess_identity": identity,
        "resolved_config": cfg.to_dict(),
        "class_names": ["hello", "world"],
        "label_map": {"hello": 0, "world": 1},
        "model_state_dict": {k: v.clone() + 3 for k, v in before_state.items()},
        "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 0.001}]},
    }

    if case == "unknown_config":
        payload["resolved_config"]["features"]["removed_alias"] = True
    elif case == "config_identity_mismatch":
        payload["resolved_config"]["preprocess"]["normalization_epsilon"] *= 10
    elif case == "reversed_labels":
        payload["class_names"] = ["world", "hello"]
        payload["label_map"] = {"world": 0, "hello": 1}
    elif case == "extra_tensor":
        payload["model_state_dict"]["unexpected.weight"] = torch.ones(1)
    elif case == "bad_optimizer":
        payload["optimizer_state_dict"] = {"state": {}, "param_groups": []}
    elif case == "same_shape_pooling_mismatch":
        payload["resolved_config"]["model"]["pooling"] = "mean"
    elif case == "missing_classes_and_labels":
        payload.pop("class_names")
        payload.pop("label_map")
    elif case == "missing_optimizer_state":
        payload.pop("optimizer_state_dict")
    elif case == "invalid_epoch_after_commit":
        payload["epoch"] = "invalid-epoch"

    ckpt_path = tmp_path / f"{case}.pt"
    torch.save(payload, ckpt_path)

    with pytest.raises((ValueError, KeyError)):
        train_gru(
            model=model,
            train_loader=None,
            val_loader=None,
            epochs=0,
            resume_checkpoint_path=ckpt_path,
            preprocess_identity=identity,
            resolved_config=cfg.to_dict(),
            class_names=["hello", "world"],
            label_map={"hello": 0, "world": 1},
            verbose=False,
        )

    # Invariant: weights must be 100% identical to before_state
    for k, v in model.state_dict().items():
        assert torch.equal(v, before_state[k]), f"Model weight {k} was mutated on failed resume!"


def test_foreign_cache_cannot_be_certified_by_features_dir(tmp_path: Path) -> None:
    """Finding 7: Features dir fallback must not certify untrusted foreign directories."""
    cfg = _create_base_config()
    trusted = tmp_path / "trusted"
    _write_cache_meta(trusted, cfg)

    untrusted = tmp_path / "untrusted"
    untrusted.mkdir(parents=True)
    npy = untrusted / "a.npy"
    _write_dummy_feature(npy)

    split = tmp_path / "foreign.csv"
    _write_csv(split, [["a", "p0", "hello", 0, str(npy)]])

    # Should raise FileNotFoundError or ValueError because untrusted has no preprocess_meta.json
    with pytest.raises((FileNotFoundError, ValueError), match="Missing canonical 'preprocess_meta.json'"):
        ds = ISRLandmarkDataset(split, features_dir=trusted, sequence_length=32, feature_dim=143)
        _ = ds[0]


def test_frozen_manifest_rejects_signer_drift_and_class_loss(tmp_path: Path) -> None:
    """Finding 5: Frozen manifest must reject changed signers and class disappearance."""
    cfg = _create_base_config()
    cfg.env.local_root = str(tmp_path)
    cfg.data.cache_dir = str(tmp_path / "cache")
    paths = resolve_paths("local", cfg, create_dirs=True)

    frozen = tmp_path / "frozen.json"
    frozen_data = {
        "schema_version": 1,
        "seed": 42,
        "vocabulary": ["hello", "world"],
        "label_map": {"hello": 0, "world": 1},
        "splits": {
            name: {
                "participant_ids": [f"{name}_original"],
                "sequence_ids": [f"{name}_hello", f"{name}_world"],
            }
            for name in ["train", "val", "test"]
        },
    }
    frozen.write_text(json.dumps(frozen_data), encoding="utf-8")
    cfg.split.manifest_path = str(frozen)

    # 1. Changed signers must raise ValueError
    drift_rows = []
    for name in ["train", "val", "test"]:
        for gloss, label in [("hello", 0), ("world", 1)]:
            drift_rows.append([f"{name}_{gloss}", f"{name}_replacement", gloss, label, "new_cache.npy"])
    _write_csv(paths.cache_dir / "features/manifest.csv", drift_rows)

    with pytest.raises(ValueError, match="Signer provenance violation"):
        create_signer_splits(cfg, paths)

    # 2. Whole class disappearance must raise ValueError
    missing_rows = []
    for name in ["train", "val", "test"]:
        missing_rows.append([f"{name}_hello", f"{name}_original", "hello", 0, "cache.npy"])
    _write_csv(paths.cache_dir / "features/manifest.csv", missing_rows)

    with pytest.raises(ValueError, match="Class coverage violation in frozen split rebind"):
        create_signer_splits(cfg, paths)


def test_frozen_split_diagnostics_suite(tmp_path: Path) -> None:
    """A-03: Verify exact diagnostic cases from review for frozen splits."""
    cfg = _create_base_config()
    cfg.env.local_root = str(tmp_path)
    cfg.data.cache_dir = str(tmp_path / "cache")
    paths = resolve_paths("local", cfg, create_dirs=True)

    # 1. Unsupported manifest schema
    _write_csv(paths.cache_dir / "features/manifest.csv", [["seq1", "p0", "hello", 0, "path.npy"]])
    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text(json.dumps({"schema_version": 999, "splits": {}}), encoding="utf-8")
    cfg.split.manifest_path = str(bad_schema)
    with pytest.raises(ValueError, match="Unsupported split manifest schema_version"):
        create_signer_splits(cfg, paths)

    # 2. Same split signer changed for a sequence
    frozen = tmp_path / "frozen.json"
    rows = []
    payload = {
        "schema_version": 1,
        "seed": 42,
        "vocabulary": ["hello", "world"],
        "label_map": {"hello": 0, "world": 1},
        "splits": {},
    }
    for split in ("train", "val", "test"):
        split_ids = []
        for gloss, label in (("hello", 0), ("world", 1)):
            for p in range(2):
                sid = f"{split}_{gloss}_{p}"
                split_ids.append(sid)
                rows.append([sid, f"{split}_p{p}", gloss, label, "cache.npy"])
        payload["splits"][split] = {
            "participant_ids": [f"{split}_p0", f"{split}_p1"],
            "sequence_ids": split_ids,
        }
    frozen.write_text(json.dumps(payload), encoding="utf-8")
    cfg.split.manifest_path = str(frozen)

    # Sequence train_hello_0 changes signer to train_p1
    drift_rows = copy.deepcopy(rows)
    drift_rows[0][1] = "train_p1"
    _write_csv(paths.cache_dir / "features/manifest.csv", drift_rows)
    with pytest.raises(ValueError, match="Signer provenance violation"):
        create_signer_splits(cfg, paths)

    # Sequence train_hello_0 changes gloss to world
    gloss_rows = copy.deepcopy(rows)
    gloss_rows[0][2:4] = ["world", 1]
    _write_csv(paths.cache_dir / "features/manifest.csv", gloss_rows)
    with pytest.raises(ValueError, match="Gloss provenance violation"):
        create_signer_splits(cfg, paths)

    # Missing sequence generates explicit exclusion report
    missing_one = rows[1:]  # Missing 1 sequence
    _write_csv(paths.cache_dir / "features/manifest.csv", missing_one)
    files, report_p = create_signer_splits(cfg, paths)
    report_content = report_p.read_text(encoding="utf-8").lower()
    assert "exclusion" in report_content


def test_standalone_evaluation_diagnostics_suite(tmp_path: Path) -> None:
    """A-04 & E-01: Verify standalone evaluation rejects corrupted metadata and populates audit fields."""
    import importlib
    eval_cli = importlib.import_module("scripts.21_evaluate")
    run_standalone_evaluation = eval_cli.run_standalone_evaluation

    cfg = _create_base_config()
    cfg.env.local_root = str(tmp_path)
    cfg.data.cache_dir = str(tmp_path / "cache")
    _ = resolve_paths("local", cfg, create_dirs=True)

    cache = tmp_path / "cache/features"
    identity = _write_cache_meta(cache, cfg)

    rows = []
    for label, gloss in enumerate(["hello", "world"]):
        path = cache / f"{label}.npy"
        _write_dummy_feature(path)
        rows.append([str(label), "eval_signer", gloss, label, str(path)])
    _write_csv(tmp_path / "data/splits/val.csv", rows)

    model = _make_model(cfg)
    valid_payload = {
        "epoch": 1,
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "preprocess_identity": identity,
        "resolved_config": cfg.to_dict(),
        "class_names": ["hello", "world"],
        "label_map": {"hello": 0, "world": 1},
        "model_state_dict": model.state_dict(),
    }

    # 1. Missing resolved_config
    p_no_cfg = copy.deepcopy(valid_payload)
    p_no_cfg.pop("resolved_config")
    ckpt_no_cfg = tmp_path / "no_cfg.pt"
    torch.save(p_no_cfg, ckpt_no_cfg)
    with pytest.raises(ValueError, match="missing canonical 'resolved_config'"):
        run_standalone_evaluation(ckpt_no_cfg, split="val", output_dir=tmp_path / "eval_no_cfg")

    # 2. Duplicate class names
    p_dup_classes = copy.deepcopy(valid_payload)
    p_dup_classes["class_names"] = ["hello", "hello"]
    ckpt_dup = tmp_path / "dup_classes.pt"
    torch.save(p_dup_classes, ckpt_dup)
    with pytest.raises(ValueError, match="Duplicate class names"):
        run_standalone_evaluation(ckpt_dup, split="val", output_dir=tmp_path / "eval_dup")

    # 3. Class order / label map conflict
    p_conflict = copy.deepcopy(valid_payload)
    p_conflict["class_names"] = ["world", "hello"]
    ckpt_conflict = tmp_path / "conflict.pt"
    torch.save(p_conflict, ckpt_conflict)
    with pytest.raises(ValueError, match="Class order / label map conflict"):
        run_standalone_evaluation(ckpt_conflict, split="val", output_dir=tmp_path / "eval_conflict")

    # 4. Non-contiguous label indices
    p_noncontig = copy.deepcopy(valid_payload)
    p_noncontig["label_map"] = {"hello": 0, "world": 2}
    ckpt_noncontig = tmp_path / "noncontig.pt"
    torch.save(p_noncontig, ckpt_noncontig)
    with pytest.raises(ValueError, match="Non-contiguous label indices"):
        run_standalone_evaluation(ckpt_noncontig, split="val", output_dir=tmp_path / "eval_noncontig")

    # 5. Overriding model semantics (e.g. pooling) during evaluation
    ckpt_valid = tmp_path / "valid.pt"
    torch.save(valid_payload, ckpt_valid)
    with pytest.raises(ValueError, match="altering model architecture or preprocessing semantics"):
        run_standalone_evaluation(
            ckpt_valid,
            split="val",
            output_dir=tmp_path / "eval_override",
            overrides=["model.pooling=mean"],
        )

    # 6. Valid evaluation report contains all required audit fields
    out_valid = tmp_path / "eval_valid"
    with patch("src.training.evaluate.plot_confusion_matrix"):
        ret = run_standalone_evaluation(ckpt_valid, split="val", output_dir=out_valid)
    assert ret == 0
    report_json = json.loads((out_valid / "evaluation_val_report.json").read_text(encoding="utf-8"))
    for required_key in ("sample_membership", "preprocess_identity", "split_identity", "confusion_matrix"):
        assert required_key in report_json, f"Missing required audit field: '{required_key}'"
    assert len(report_json["sample_membership"]) == 2
    assert report_json["preprocess_identity"] == identity


def test_eval_cli_rejects_missing_metadata(tmp_path: Path) -> None:
    """Finding 6: Standalone evaluation CLI must strictly reject checkpoint missing metadata."""
    import importlib
    eval_cli = importlib.import_module("scripts.21_evaluate")
    run_standalone_evaluation = eval_cli.run_standalone_evaluation

    cfg = _create_base_config()
    cfg.env.local_root = str(tmp_path)
    cfg.data.cache_dir = str(tmp_path / "cache")
    eval_features = tmp_path / "cache/features"
    _write_cache_meta(eval_features, cfg)

    rows = []
    for i, gloss in enumerate(["hello", "world"]):
        path = eval_features / f"{i}.npy"
        _write_dummy_feature(path)
        rows.append([str(i), "eval_signer", gloss, i, str(path)])
    _write_csv(tmp_path / "data/splits/train.csv", rows)

    model = _make_model(cfg)
    # Checkpoint missing schema_version and preprocess_identity
    bad_ckpt = {
        "model_state_dict": model.state_dict(),
        "resolved_config": cfg.to_dict(),
        "class_names": ["hello", "world"],
    }
    ckpt_path = tmp_path / "eval_bad.pt"
    torch.save(bad_ckpt, ckpt_path)

    with pytest.raises(ValueError, match="Unsupported checkpoint schema_version"):
        with patch("src.training.evaluate.plot_confusion_matrix"):
            run_standalone_evaluation(ckpt_path, split="train", output_dir=tmp_path / "evaluation")


def test_predictor_validates_checkpoint_metadata_strictly(tmp_path: Path) -> None:
    """Finding 1 & 6: Predictor validates class names and canonical preprocess metadata."""
    cfg = _create_base_config()
    identity, _ = compute_preprocess_identity(cfg)
    model = _make_model(cfg)

    glosses_csv = tmp_path / "glosses.csv"
    glosses_csv.write_text("gloss,label\nhello,0\nworld,1\n", encoding="utf-8")

    valid_payload = {
        "model_state_dict": model.state_dict(),
        "resolved_config": cfg.to_dict(),
        "preprocess_identity": identity,
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "class_names": ["hello", "world"],
        "label_map": {"hello": 0, "world": 1},
    }
    valid_ckpt = tmp_path / "valid.pt"
    torch.save(valid_payload, valid_ckpt)

    predictor = ISLRPredictor(valid_ckpt, glosses_csv, device="cpu")
    assert predictor.class_names == ["hello", "world"]
    assert predictor.is_legacy is False
    predictor.close()

    # Wrong schema version must be rejected
    bad_schema_payload = copy.deepcopy(valid_payload)
    bad_schema_payload["schema_version"] = 999
    bad_schema_ckpt = tmp_path / "bad_schema.pt"
    torch.save(bad_schema_payload, bad_schema_ckpt)
    with pytest.raises(ValueError, match="Unsupported checkpoint schema_version"):
        ISLRPredictor(bad_schema_ckpt, glosses_csv, device="cpu")

    # Wrong identity hash must be rejected
    bad_id_payload = copy.deepcopy(valid_payload)
    bad_id_payload["preprocess_identity"] = "corrupted-hash"
    bad_id_ckpt = tmp_path / "bad_id.pt"
    torch.save(bad_id_payload, bad_id_ckpt)
    with pytest.raises(ValueError, match="does not match recomputed config identity"):
        ISLRPredictor(bad_id_ckpt, glosses_csv, device="cpu")
