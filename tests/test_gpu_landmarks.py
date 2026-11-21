"""GPU adapter contracts; no browser, network, optional SDK, or local data required."""
from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from demos import camera_v3, setup_gpu_landmarks
from demos.gpu_landmarks import (
    GPULandmarksUnavailableError, GpuLandmarkExtractor, SharedGpuExtractor, SharedGpuLandmarks,
    browser_path, packet_from_result, validate_hardware,
)
from demos.live_landmarks import LandmarkFrame
from src.features.pipeline import prepare_landmark_sequence
from src.features.schema import FACE_LANDMARK_INDICES, compute_preprocess_identity
from src.utils.config import CameraConfig, load_config


def raw_result(labels):
    return {"hands": [np.full((21, 3), index + .1).tolist() for index in range(len(labels))],
            "handedness": labels, "pose": None, "face": None}


@pytest.mark.parametrize("labels,order", [(["Left", "Right"], [0, 1]), (["Right", "Left"], [1, 0]),
    (["Left", "Left"], [0, 1]), (["Right", "Unknown"], [0, 1]), (["Unknown", "Right"], [0, 1])])
def test_anatomical_slots_and_unknown_fallback_match_native_extractor(labels, order):
    result = raw_result(labels)
    frame = packet_from_result(result, 1.5)
    np.testing.assert_array_equal(frame.hands, np.asarray(result["hands"], np.float32)[order])
    assert frame.hand_masks.all() and frame.captured_at == 1.5
    assert not frame.pose_present and not frame.face_present
    assert np.isnan(frame.pose).all() and np.isnan(frame.face).all()


def test_single_right_hand_stays_in_right_slot():
    frame = packet_from_result(raw_result(["Right"]), 0)
    assert frame.hand_masks.tolist() == [False, True]
    assert not frame.hands[0].any()


@pytest.mark.parametrize("kind,shape", [("hands", (20, 3)), ("face", (478, 3)), ("pose", (32, 3))])
def test_malformed_gpu_outputs_fail_before_preprocessing(kind, shape):
    result = raw_result(["Left"])
    result[kind] = [np.zeros(shape).tolist()] if kind == "hands" else np.zeros(shape).tolist()
    with pytest.raises(ValueError, match="invalid shape"):
        packet_from_result(result, 0)


def test_hardware_contract_rejects_cpu_and_software_fallback():
    hardware = {name: {"delegate": "GPU", "renderer": "ANGLE (NVIDIA, GeForce GTX 1050)"}
                for name in ("hands", "face", "pose")}
    validate_hardware(hardware)
    for invalid in ("Google SwiftShader", "Microsoft Basic Render Driver", "llvmpipe"):
        wrong = dict(hardware)
        wrong["face"] = {"delegate": "GPU", "renderer": invalid}
        with pytest.raises(GPULandmarksUnavailableError):
            validate_hardware(wrong)
    with pytest.raises(GPULandmarksUnavailableError):
        validate_hardware({"hands": hardware["hands"]})


def test_missing_browser_is_explicit_before_importing_gpu_sdk(tmp_path):
    with pytest.raises(GPULandmarksUnavailableError, match="Chrome/Edge"):
        browser_path(str(tmp_path / "missing.exe"))


def test_shared_worker_owns_gpu_calls_and_borrowed_close_does_not_destroy_it(monkeypatch):
    import threading
    from demos import gpu_landmarks

    threads = []
    closed = []
    core = SimpleNamespace(hardware={}, initialize=lambda: threads.append(threading.get_ident()),
        extract=lambda frame, stamp: threads.append(threading.get_ident()),
        close=lambda: closed.append(threading.get_ident()))
    monkeypatch.setattr(gpu_landmarks, "GpuLandmarkExtractor", lambda cfg: core)
    service = SharedGpuLandmarks(CameraConfig())
    try:
        service.initialize()
        service.extract(None, 0)
        SharedGpuExtractor(service).close()
        assert not closed
    finally:
        service.close()
    assert threads[0] == threads[1] == closed[0] != threading.get_ident()
    service.close()
    assert len(closed) == 1


def test_gpu_video_uses_all_original_frames_and_canonical_feature_pipeline(monkeypatch):
    count = 53
    hands = np.zeros((2, 21, 3), np.float32)
    hands[:, :, 0] = .2 + np.arange(21) * .005
    hands[:, :, 1] = .3 + np.arange(21) * .005
    pose = np.zeros((33, 3), np.float32)
    pose[11:15] = [[.3, .4, 0], [.7, .4, 0], [.2, .6, 0], [.8, .6, 0]]
    face = np.zeros((468, 3), np.float32)
    face[FACE_LANDMARK_INDICES, 0] = np.linspace(.3, .6, 16)
    face[FACE_LANDMARK_INDICES, 1] = np.linspace(.2, .5, 16)
    frames = [np.full((16, 24, 3), index, np.uint8) for index in range(count)]
    remaining = iter(frames)
    release = []
    def read():
        frame = next(remaining, None)
        return (False, None) if frame is None else (True, frame)

    capture = SimpleNamespace(isOpened=lambda: True, get=lambda key: 30,
        read=read, release=lambda: release.append(True))
    seen, packets = [], []

    def extract(frame, stamp):
        seen.append(frame.copy())
        packet = LandmarkFrame(stamp, hands + len(seen) * .001, np.ones(2, bool), pose, True, face, True)
        packets.append(packet)
        return packet

    monkeypatch.setattr(camera_v3.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(camera_v3, "extract_auxiliary_video", lambda *args, **kwargs: pytest.fail("GPU must not invoke CPU extraction"))
    cfg = load_config(overrides=["features.face=true", "model.region_fusion=true",
        'model.regions=["hands","body","face"]', "preprocess.trim_no_hand=false"])
    calls = []
    service = SimpleNamespace(core=SimpleNamespace(cfg=CameraConfig()),
        extract_batch=lambda frames, stamps: [extract(frame, stamp) for frame, stamp in zip(frames, stamps)])
    predictor = SimpleNamespace(cfg=cfg, extractor=SharedGpuExtractor(service),
        predict_features=lambda features, **kwargs: calls.append((features, kwargs)) or "prediction")
    assert camera_v3.infer_clip(predictor, Path("synthetic.avi")) == "prediction"
    expected = prepare_landmark_sequence(cfg=cfg, raw_coords=np.stack([p.hands for p in packets]),
        hand_masks=np.stack([p.hand_masks for p in packets]), pose_coords=np.stack([p.pose for p in packets]),
        pose_masks=np.ones(count, bool), face_coords=np.stack([p.face for p in packets]), face_mask=np.ones(count, bool))
    np.testing.assert_array_equal(calls[0][0], expected.features)
    assert calls[0][0].shape == (32, 196)
    assert calls[0][1]["stats"]["n_frames"] == count
    assert calls[0][1]["stats"]["landmark_device"] == "GPU"
    np.testing.assert_array_equal(seen, frames)
    assert release == [True]


def test_gpu_backend_keys_are_canonical_and_identity_remains_frozen():
    gpu = load_config()
    cpu = load_config(overrides=["camera.landmark_backend=cpu"])
    assert gpu.camera.landmark_backend == "webgl_gpu"
    assert compute_preprocess_identity(gpu)[0] == compute_preprocess_identity(cpu)[0]
    with pytest.raises(KeyError):
        load_config(overrides=["camera.gpu_delegate=true"])
    with pytest.raises(ValueError):
        load_config(overrides=["camera.landmark_backend=gpu"])


def test_asset_install_checks_integrity_and_ignores_unselected_tar_paths(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("package/vision_bundle.mjs", "../../outside.txt"):
            entry = tarfile.TarInfo(name)
            entry.size = 4
            archive.addfile(entry, io.BytesIO(b"test"))
    payload = buffer.getvalue()
    archive_path = tmp_path / "sdk.tgz"
    archive_path.write_bytes(payload)
    destination = tmp_path / "sdk"
    monkeypatch.setattr(setup_gpu_landmarks, "ARCHIVE", archive_path)
    monkeypatch.setattr(setup_gpu_landmarks, "SDK_ROOT", destination)
    monkeypatch.setattr(setup_gpu_landmarks, "FILES", ("vision_bundle.mjs",))
    monkeypatch.setattr(setup_gpu_landmarks, "INTEGRITY", base64.b64encode(hashlib.sha512(payload).digest()).decode())
    assert setup_gpu_landmarks.ensure_sdk() == destination
    assert (destination / "vision_bundle.mjs").read_bytes() == b"test"
    assert not (tmp_path / "outside.txt").exists()
    (destination / "MANIFEST.json").unlink()
    monkeypatch.setattr(setup_gpu_landmarks, "INTEGRITY", "invalid")
    with pytest.raises(ValueError, match="integrity"):
        setup_gpu_landmarks.ensure_sdk()


def test_gpu_runtime_enforces_thread_ownership_without_opening_browser():
    extractor = GpuLandmarkExtractor(CameraConfig())
    extractor.page = SimpleNamespace()
    extractor.owner = -1
    with pytest.raises(RuntimeError, match="owning thread"):
        extractor.initialize()
