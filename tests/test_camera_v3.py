"""Regressions for full-gesture capture and required face inputs in the V3 demo."""
from __future__ import annotations

from types import SimpleNamespace
import hashlib
import json
import zipfile

import numpy as np
import pytest

from demos import camera_v3
from demos.v3_models import CameraModel, load_catalog, materialize_model
from src.features.pipeline import prepare_landmark_sequence
from src.features.schema import FACE_LANDMARK_INDICES
from src.utils.config import load_config


def test_recorder_keeps_full_gesture_and_original_frames(tmp_path, monkeypatch):
    frames = []
    writer = SimpleNamespace(isOpened=lambda: True, write=lambda frame: frames.append(frame.copy()), release=lambda: None)
    monkeypatch.setattr(camera_v3.cv2, "VideoWriter", lambda *args: writer)
    recorder = camera_v3.GestureRecorder(tmp_path, 30)
    original = np.arange(18 * 24 * 3, dtype=np.uint8).reshape(18, 24, 3)
    recorder.start(original)
    for _ in range(79):
        recorder.append(original)
    clip = recorder.finish()
    assert clip.frame_count == len(frames) == 80
    assert not recorder.active
    np.testing.assert_array_equal(frames[-1], original)
    recorder.start(original)
    assert recorder.finish().frame_count == 1


def geometry(count=48):
    hands = np.zeros((count, 2, 21, 3), np.float32)
    for side in range(2):
        hands[:, side, :, 0] = .2 + side * .4 + np.arange(21) * .003
        hands[:, side, :, 1] = .3 + np.arange(21) * .006
    hands[..., 0] += np.arange(count)[:, None, None] * .001
    face = np.zeros((count, 468, 3), np.float32)
    face[:, FACE_LANDMARK_INDICES, 0] = np.linspace(.3, .6, 16)
    face[:, FACE_LANDMARK_INDICES, 1] = np.linspace(.2, .5, 16)
    pose = np.zeros((count, 33, 3), np.float32)
    pose[:, 11] = [.3, .4, 0]
    pose[:, 12] = [.7, .4, 0]
    pose[:, 13] = [.2, .6, 0]
    pose[:, 14] = [.8, .6, 0]
    return hands, np.ones((count, 2), bool), {
        "face_coords": face, "face_mask": np.ones(count, bool),
        "pose_coords": pose, "pose_masks": np.ones(count, bool),
    }


def predictor_stub(hands, masks):
    cfg = load_config(overrides=["features.face=true", "model.region_fusion=true", 'model.regions=["hands","face"]',
                                "model.name=Transformer", "preprocess.trim_no_hand=false", "preprocess.dominant_hand=disabled"])
    calls = []

    def predict_features(features, stats, top_k):
        calls.append((features, stats, top_k))
        return "prediction"

    return SimpleNamespace(cfg=cfg, extractor=SimpleNamespace(extract_from_video=lambda path: (
        hands, masks, {"ok": True, "n_frames": len(hands)})), predict_features=predict_features), calls


def test_camera_features_match_canonical_pipeline(tmp_path, monkeypatch):
    hands, masks, auxiliary = geometry()
    predictor, calls = predictor_stub(hands, masks)
    monkeypatch.setattr(camera_v3, "extract_auxiliary_video", lambda *args, **kwargs: auxiliary)
    assert camera_v3.infer_clip(predictor, tmp_path / "clip.avi") == "prediction"
    expected = prepare_landmark_sequence(raw_coords=hands, hand_masks=masks, pose_coords=auxiliary["pose_coords"],
                                         pose_masks=auxiliary["pose_masks"], face_coords=auxiliary["face_coords"],
                                         face_mask=auxiliary["face_mask"], cfg=predictor.cfg)
    np.testing.assert_array_equal(calls[0][0], expected.features)
    assert calls[0][0].shape == (32, 196)
    assert calls[0][1]["n_frames"] == 48
    assert calls[0][1]["face_detection_rate"] > 0


def test_missing_face_is_rejected_before_model(tmp_path, monkeypatch):
    hands, masks, auxiliary = geometry()
    auxiliary["face_mask"][:] = False
    predictor, calls = predictor_stub(hands, masks)
    monkeypatch.setattr(camera_v3, "extract_auxiliary_video", lambda *args, **kwargs: auxiliary)
    with pytest.raises(ValueError, match="khuôn mặt"):
        camera_v3.infer_clip(predictor, tmp_path / "clip.avi")
    assert calls == []


def test_mismatched_face_frames_are_rejected(tmp_path, monkeypatch):
    hands, masks, auxiliary = geometry()
    auxiliary["face_coords"] = auxiliary["face_coords"][:-1]
    predictor, calls = predictor_stub(hands, masks)
    monkeypatch.setattr(camera_v3, "extract_auxiliary_video", lambda *args, **kwargs: auxiliary)
    with pytest.raises(ValueError, match="cùng số frames"):
        camera_v3.infer_clip(predictor, tmp_path / "clip.avi")
    assert calls == []


@pytest.mark.parametrize("regions", [
    ["hands"], ["body"], ["face"], ["hands", "body"],
    ["hands", "face"], ["body", "face"], ["hands", "body", "face"],
])
def test_all_region_subsets_use_checkpoint_pipeline(tmp_path, monkeypatch, regions):
    hands, masks, auxiliary = geometry()
    predictor, calls = predictor_stub(hands, masks)
    predictor.cfg.model.regions = regions
    if "face" not in regions:
        auxiliary["face_mask"][:] = False
    if "body" not in regions:
        auxiliary["pose_masks"][:] = False
    monkeypatch.setattr(camera_v3, "extract_auxiliary_video", lambda *args, **kwargs: auxiliary)
    assert camera_v3.infer_clip(predictor, tmp_path / "clip.avi") == "prediction"
    expected = prepare_landmark_sequence(raw_coords=hands, hand_masks=masks, cfg=predictor.cfg,
                                         pose_coords=auxiliary["pose_coords"], pose_masks=auxiliary["pose_masks"],
                                         face_coords=auxiliary["face_coords"], face_mask=auxiliary["face_mask"])
    np.testing.assert_array_equal(calls[0][0], expected.features)
    assert calls[0][0].shape == (32, 196)


def test_missing_body_is_rejected_before_model(tmp_path, monkeypatch):
    hands, masks, auxiliary = geometry()
    predictor, calls = predictor_stub(hands, masks)
    predictor.cfg.model.regions = ["hands", "body"]
    auxiliary["pose_masks"][:] = False
    monkeypatch.setattr(camera_v3, "extract_auxiliary_video", lambda *args, **kwargs: auxiliary)
    with pytest.raises(ValueError, match="thân trên"):
        camera_v3.infer_clip(predictor, tmp_path / "clip.avi")
    assert calls == []


def model_fixture(tmp_path):
    checkpoint, glosses = b"synthetic checkpoint", b"label,gloss\n0,hello\n"
    archive = tmp_path / "results.zip"
    prefix = "runs/G0__H/attempt"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(prefix + "/runs/training/seed_42/best_model.pt", checkpoint)
        z.writestr(prefix + "/inputs/selection/glosses.csv", glosses)
        z.writestr("../../outside.txt", b"do not extract")
    model = CameraModel("G0__H", 42, ("hands",), 196, .71, "freeze", archive, prefix,
                        hashlib.sha256(archive.read_bytes()).hexdigest(),
                        hashlib.sha256(checkpoint).hexdigest(), hashlib.sha256(glosses).hexdigest())
    return model


def test_materialize_checks_and_reuses_exact_selected_artifacts(tmp_path):
    model = model_fixture(tmp_path)
    deployment = tmp_path / "deployment"
    checkpoint, glosses = materialize_model(model, deployment)
    assert checkpoint.read_bytes() == b"synthetic checkpoint"
    assert glosses.read_bytes() == b"label,gloss\n0,hello\n"
    assert len(list(deployment.rglob("*.*"))) == 2
    model.archive.unlink()
    assert materialize_model(model, deployment) == (checkpoint, glosses)
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum checkpoint"):
        materialize_model(model, deployment)


def test_materialize_rejects_changed_archive_before_writing(tmp_path):
    model = model_fixture(tmp_path)
    model.archive.write_bytes(b"tampered")
    deployment = tmp_path / "deployment"
    with pytest.raises(ValueError, match="Checksum ZIP"):
        materialize_model(model, deployment)
    assert not deployment.exists()


def test_catalog_uses_reviewed_seed_hashes(tmp_path):
    model = model_fixture(tmp_path)
    row = {"archive": str(model.archive), "archive_prefix": model.archive_prefix,
           "status": "COMPLETED", "owner": "A", "regions": ["hands"], "dimension": 196,
           "seeds": [42], "metrics": {"macro_f1": {"mean": .71}},
           "provenance": {"freeze_id": "freeze", "selection_glosses_csv_sha256": model.glosses_sha256}}
    data = {"scope": "v3-full42", "status": "COMPLETED", "ranking": ["G0__H"],
            "freeze_id": "freeze", "records": {"G0__H": row},
            "details": {"G0__H": {"epoch_details": [{"seed": 42, "checkpoint_sha256": model.checkpoint_sha256}]}},
            "archive_sha256": {"A": model.archive_sha256}}
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_catalog(path) == {("G0__H", 42): model}


def test_runtime_rejects_wrong_checkpoint_config_and_closes(tmp_path, monkeypatch):
    model = model_fixture(tmp_path)
    closed = []
    predictor = SimpleNamespace(cfg=load_config(), input_size=196, close=lambda: closed.append(True))
    monkeypatch.setattr(camera_v3, "materialize_model", lambda selected: (tmp_path / "model.pt", tmp_path / "glosses.csv"))
    monkeypatch.setattr(camera_v3, "ISLRPredictor", lambda **kwargs: predictor)
    with pytest.raises(ValueError, match="Cấu hình checkpoint"):
        camera_v3.load_runtime(model)
    assert closed == [True]


def test_runtime_seed_selection_preserves_campaign_config(tmp_path, monkeypatch):
    from dataclasses import replace

    model = replace(model_fixture(tmp_path), seed=43)
    cfg = load_config(overrides=["run.name=G0__H", "features.face=true", "model.region_fusion=true",
                                 'model.regions=["hands"]', "preprocess.trim_no_hand=false"])
    assert cfg.env.seed == 42
    predictor = SimpleNamespace(cfg=cfg, input_size=196, close=lambda: None,
                                extractor=SimpleNamespace(_get_hands=lambda: None))
    monkeypatch.setattr(camera_v3, "materialize_model", lambda selected: (tmp_path / "model.pt", tmp_path / "glosses.csv"))
    monkeypatch.setattr(camera_v3, "ISLRPredictor", lambda **kwargs: predictor)
    monkeypatch.setattr(camera_v3, "_task_asset", lambda kind: None)
    assert camera_v3.load_runtime(model) is predictor


def test_switch_is_blocked_during_recording_or_inference():
    app = camera_v3.CameraApp.__new__(camera_v3.CameraApp)
    app.future = SimpleNamespace()
    app.switch_model()  # Must return without reading selectors or replacing runtime.
    app.future = None
    app.recorder = SimpleNamespace(active=True)
    app.switch_model()


def test_replay_submits_same_saved_clip(tmp_path):
    app = camera_v3.CameraApp.__new__(camera_v3.CameraApp)
    app.predictor = SimpleNamespace()
    app.future = None
    app.recorder = SimpleNamespace(active=False)
    app.saved_clip = camera_v3.CapturedClip(tmp_path / "saved.avi", 80, 2.7)
    calls = []
    app.submit_clip = calls.append
    app.replay_clip()
    assert calls == [app.saved_clip]
    app.recorder.active = True
    app.replay_clip()
    assert len(calls) == 1
