"""Face invariance, cache ownership, augmentation parity, and model integration."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.data.augmentation import horizontal_flip, random_rotate_2d
from src.data.augmentation_gpu import batch_horizontal_flip, batch_random_rotate_2d
from src.data.preprocess import preprocess_single_sequence
from src.features.face import prepare_face_sequence
from src.features.pipeline import prepare_landmark_sequence
from src.features.schema import (
    FACE_LANDMARK_INDICES, PREPROCESS_SCHEMA_VERSION, compute_preprocess_identity,
    get_feature_dim, get_feature_layout_from_config, infer_feature_layout_from_dim,
)
from src.features.validity import load_validity, save_validity
from src.inference.predictor import ISLRPredictor
from src.models.gru import create_model
from src.training.train import train_gru
from src.training.evaluate import evaluate_test_set
from src.utils.checkpoint import validate_model_architecture
from src.utils.config import Config, load_config


def geometry(t=12):
    hands = np.tile(np.linspace(0.1, 0.3, 63, dtype=np.float32).reshape(1, 1, 21, 3), (t, 2, 1, 1))
    hands[:, 1, :, 0] += 0.2
    pose = np.zeros((t, 33, 3), dtype=np.float32)
    pose[:, 11], pose[:, 12] = (0.3, 0.4, 0), (0.7, 0.4, 0)
    points = np.array([
        [0.5, .5], [.3, .4], [.7, .4], [.4, .4], [.6, .4],
        [.35, .38], [.65, .37], [.35, .42], [.65, .43], [.4, .6],
        [.6, .6], [.5, .58], [.5, .62], [.35, .3], [.65, .3], [.5, .8],
    ], dtype=np.float32)
    face = np.tile(points[None], (t, 1, 1))
    face[:, 12, 1] += np.arange(t) * .001
    return hands, pose, face


def config(face=True):
    cfg = Config()
    cfg.features.set = "hands_pose"
    cfg.features.face = face
    cfg.model.hidden_size = 16
    cfg.model.num_layers = 1
    cfg.model.dropout = 0.0
    cfg.model.feedforward_size = 32
    cfg.validate()
    return cfg


def prepare(cfg, hands, pose, face):
    return prepare_landmark_sequence(raw_coords=hands, pose_coords=pose, face_coords=face, cfg=cfg)


def test_face_geometry_invariance_and_missing_support():
    _, _, face = geometry()
    args = (None, len(face), (0, len(face)), 12, 1, 1e-4)
    original = prepare_face_sequence(face, *args)[0]
    moved = prepare_face_sequence(face * 2.5 + [.2, -.3], *args)[0]
    np.testing.assert_allclose(original, moved, atol=2e-6)
    missing = face.copy()
    missing[3:5] = np.nan  # exceeds max_gap_frames
    features, observed, usable = prepare_face_sequence(missing, *args)
    assert not usable[3:5].any() and not observed[3:5].any()
    assert not features[3:5].any() and np.isfinite(features).all()
    missing[3:5] = face[3:5]
    missing[4] = np.nan  # bounded single frame gap is imputed, never observed
    _, observed, usable = prepare_face_sequence(missing, *args)
    assert usable[4, 0] and not observed[4, 0]
    with pytest.raises(ValueError, match="aligned"):
        prepare_face_sequence(face[:-1], *args)


def test_face_toggle_prefix_hash_dimensions_and_sidecar(tmp_path):
    hands, pose, face = geometry()
    cfg = config()
    on, off = prepare(cfg, hands, pose, face), prepare(config(False), hands, pose, face)
    assert on.is_usable and off.is_usable and on.features.shape == (32, 196)
    np.testing.assert_array_equal(on.features[:, :160], off.features)
    unavailable = prepare(cfg, hands, pose, None)
    assert unavailable.is_usable and not unavailable.features[:, 160:].any()
    assert compute_preprocess_identity(cfg)[0] != compute_preprocess_identity(config(False))[0]
    for wrist in ("image", "torso_xy"):
        for tips in (False, True):
            for bends in (False, True):
                cfg.features.wrist_reference = wrist
                cfg.features.fingertip_displacement, cfg.features.bend_cosines = tips, bends
                layout = get_feature_layout_from_config(cfg)
                assert infer_feature_layout_from_dim(layout.total_dim) == layout
    cfg = config()
    identity = compute_preprocess_identity(cfg)[0]
    path = tmp_path / "sequence.npy"
    np.save(path, on.features)
    save_validity(path, on.provenance, on.features, identity, "sequence", "a" * 64)
    assert load_validity(path, on.features, identity, "sequence")["usable_face"].all()
    with pytest.raises(ValueError, match="ownership"):
        load_validity(path, on.features, compute_preprocess_identity(config(False))[0], "sequence")


def test_face_cpu_tensor_augmentation_parity():
    hands, pose, face = geometry()
    x = prepare(config(), hands, pose, face).features
    flipped = horizontal_flip(x)
    np.testing.assert_allclose(batch_horizontal_flip(torch.from_numpy(x)[None])[0].numpy(), flipped, atol=1e-6)
    np.testing.assert_allclose(horizontal_flip(flipped), x, atol=1e-6)
    rotated = random_rotate_2d(x, angle_deg=17)
    np.testing.assert_allclose(batch_random_rotate_2d(torch.from_numpy(x)[None], angles_deg=17)[0].numpy(), rotated, atol=2e-6)
    layout = infer_feature_layout_from_dim(196)
    np.testing.assert_allclose(rotated[:, layout.slice_face_ratios], x[:, layout.slice_face_ratios], atol=2e-6)


def test_parquet_raw_inference_face_parity(tmp_path: Path):
    hands, pose, face = geometry()
    rows = []
    for frame in range(len(hands)):
        for kind, coords in (("left_hand", hands[frame, 0]), ("right_hand", hands[frame, 1]), ("pose", pose[frame])):
            rows.extend(dict(frame=frame, type=kind, landmark_index=i, x=float(p[0]), y=float(p[1]), z=float(p[2])) for i, p in enumerate(coords))
        rows.extend(dict(frame=frame, type="face", landmark_index=i, x=float(p[0]), y=float(p[1]), z=float("nan")) for i, p in zip(FACE_LANDMARK_INDICES, face[frame]))
    path = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    cfg = config()
    result = preprocess_single_sequence(path, "s1", "p1", "hello", 0, cfg, tmp_path / "features")
    assert result.status == "kept"
    np.testing.assert_allclose(np.load(result.npy_path), prepare(cfg, hands, pose, face).features, atol=1e-6)


@pytest.mark.parametrize("kind", ["GRU", "LSTM", "Transformer"])
def test_model_face_backward_and_checkpoint_roundtrip(tmp_path, kind):
    cfg = config()
    cfg.model.name = kind
    cfg.validate()
    model = create_model(input_size=get_feature_dim(cfg), hidden_size=16, num_layers=1,
                         num_classes=2, dropout=0.0, rnn_type=kind, feedforward_size=32)
    x = torch.randn(2, 32, get_feature_dim(cfg))
    logits = model(x)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    validate_model_architecture(cfg, model)
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "resolved_config": cfg.to_dict(),
                "schema_version": PREPROCESS_SCHEMA_VERSION,
                "preprocess_identity": compute_preprocess_identity(cfg)[0],
                "class_names": ["hello", "bye"], "label_map": {"hello": 0, "bye": 1}}, path)
    vocabulary = tmp_path / "glosses.csv"
    vocabulary.write_text("gloss,label\nhello,0\nbye,1\n", encoding="utf-8")
    predictor = ISLRPredictor(model_path=path, glosses_csv=vocabulary, device="cpu")
    model.eval()
    np.testing.assert_allclose(torch.softmax(model(x[:1]), dim=-1).detach().numpy()[0],
                               list(predictor.predict_features(x[0].numpy()).probabilities.values()), atol=1e-6)
    predictor.close()


def test_transformer_config_and_positions():
    cfg = load_config(overrides=["features.face=true", "model.name=Transformer"])
    assert cfg.features.face and cfg.model.attention_heads == 4
    with pytest.raises(KeyError):
        load_config(overrides=["features.face_weight=0.1"])
    cfg.model.hidden_size = 127
    with pytest.raises(ValueError, match="divisible"):
        cfg.validate()
    torch.manual_seed(42)
    model = create_model(input_size=196, hidden_size=16, num_layers=1, num_classes=2,
                         rnn_type="Transformer", pooling="mean", dropout=0.0, feedforward_size=32).eval()
    x = torch.randn(1, 8, 196)
    assert not torch.allclose(model(x), model(x.flip(1)))


@pytest.mark.parametrize("kind", ["GRU", "LSTM", "Transformer"])
def test_training_evaluation_with_face_and_predictions(tmp_path, kind):
    """Real optimizer/checkpoint/evaluation path, not just a forward-shape check."""
    from torch.utils.data import DataLoader, TensorDataset
    cfg = config()
    cfg.model.name = kind
    model = create_model(input_size=196, hidden_size=16, num_layers=1, num_classes=2,
                         dropout=0.0, rnn_type=kind, feedforward_size=32)
    dataset = TensorDataset(torch.randn(4, 32, 196), torch.tensor([0, 1, 0, 1]))
    dataset.samples = [{"sequence_id": str(i), "participant_id": "p" + str(i % 2), "label": i % 2} for i in range(4)]
    dataset.preprocess_identity = compute_preprocess_identity(cfg)[0]
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    history = train_gru(model, loader, loader, epochs=2, device=torch.device("cpu"),
                        checkpoint_path=tmp_path / "best.pt", verbose=False,
                        resolved_config=cfg.to_dict(), preprocess_identity=dataset.preprocess_identity,
                        class_names=["hello", "bye"], label_map={"hello": 0, "bye": 1}, seed=42)
    assert len(history["train_loss"]) == 2 and (tmp_path / "best.pt").is_file()
    report = evaluate_test_set(model, loader, tmp_path / "unused.csv", ["hello", "bye"], torch.device("cpu"))
    assert report["model_parameters"] > 0 and len(report["predictions"]) == 4
    assert [p["y_true"] for p in report["predictions"]] == [0, 1, 0, 1]
    shuffled = DataLoader(dataset, batch_size=2, shuffle=True)
    assert "predictions" not in evaluate_test_set(model, shuffled, tmp_path / "unused.csv", ["hello", "bye"], torch.device("cpu"))


def test_auxiliary_video_without_optional_mediapipe(monkeypatch):
    """Backend mocks exercise frame alignment, missing face and resource cleanup."""
    import sys
    from types import SimpleNamespace
    from src.features import video_auxiliary
    flags = {"closed": False, "released": False}
    landmarks = [SimpleNamespace(x=.5, y=.4, z=0) for _ in range(468)]
    class Face:
        count = 0
        def process(self, rgb):
            self.count += 1
            return SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=landmarks)] if self.count == 1 else [])
        def close(self):
            flags["closed"] = True
    class Capture:
        count = 0
        def isOpened(self):
            return True
        def read(self):
            self.count += 1
            return (True, np.zeros((4, 4, 3), dtype=np.uint8)) if self.count <= 2 else (False, None)
        def release(self):
            flags["released"] = True
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(solutions=SimpleNamespace(face_mesh=SimpleNamespace(FaceMesh=lambda **kwargs: Face()))))
    monkeypatch.setattr(video_auxiliary.cv2, "VideoCapture", lambda path: Capture())
    result = video_auxiliary.extract_auxiliary_video("fixture.mp4", face=True, pose=False)
    assert result["face_coords"].shape == (2, 468, 3)
    assert result["face_mask"].tolist() == [True, False]
    assert np.isnan(result["face_coords"][1]).all() and all(flags.values())
