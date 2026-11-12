"""Schema-v5 reliability metadata. Feature values never imply continuity."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.features.schema import PREPROCESS_SCHEMA_VERSION, infer_feature_layout_from_dim
from src.utils.artifacts import content_hash

MASK_WIDTHS = {
    "observed_hands": 2, "usable_hands": 2, "hand_transitions": 2,
    "observed_pose": 1, "usable_pose": 1, "pose_transitions": 1,
    "valid_timesteps": 1,
}


def validate_validity(metadata: dict[str, Any], features: np.ndarray) -> None:
    t, d = features.shape
    layout = infer_feature_layout_from_dim(d)
    widths = dict(MASK_WIDTHS)
    if layout.face:
        widths.update(observed_face=1, usable_face=1)
    for name, width in widths.items():
        arr = metadata.get(name)
        if not isinstance(arr, np.ndarray) or arr.dtype != np.bool_ or arr.shape != (t, width):
            raise ValueError(f"Invalid validity mask {name}: expected bool ({t}, {width}).")
    slots = list(metadata.get("slot_to_anatomical", []))
    if sorted(slots) != ["left", "right"]:
        raise ValueError("Invalid slot_to_anatomical permutation.")
    valid = metadata["valid_timesteps"]
    if not valid.all():
        raise ValueError("Fixed-resampled cache must contain real timesteps, not padding.")
    if not np.array_equal(metadata["usable_hands"], features[:, layout.slice_hand_masks]):
        raise ValueError("Validity hand support disagrees with feature channels.")
    if layout.has_pose and not np.array_equal(metadata["usable_pose"], features[:, layout.slice_pose_mask]):
        raise ValueError("Validity pose support disagrees with feature channels.")
    if layout.face and not np.array_equal(metadata["usable_face"], features[:, layout.slice_face_mask]):
        raise ValueError("Validity face support disagrees with feature channels.")
    if layout.face and np.any(metadata["observed_face"] & ~metadata["usable_face"]):
        raise ValueError("Observed face support must be a subset of usable support.")
    for part in ("hands", "pose"):
        usable = metadata[f"usable_{part}"]
        if np.any(metadata[f"observed_{part}"] & ~usable):
            raise ValueError("Observed support must be a subset of usable support.")
        transition = metadata["hand_transitions" if part == "hands" else "pose_transitions"]
        if transition[0].any() or np.any(transition[1:] & ~(usable[1:] & usable[:-1])):
            raise ValueError("Transition support requires two usable endpoints and a zero first row.")


def save_validity(npy_path: str | Path, metadata: dict[str, Any], features: np.ndarray,
                  preprocess_identity: str, sequence_id: str, source_hash: str) -> None:
    validate_validity(metadata, features)
    payload = {key: metadata[key] for key in MASK_WIDTHS}
    if infer_feature_layout_from_dim(features.shape[1]).face:
        payload.update(observed_face=metadata["observed_face"], usable_face=metadata["usable_face"])
    payload.update(
        slot_to_anatomical=np.asarray(metadata["slot_to_anatomical"]),
        schema_version=np.asarray(PREPROCESS_SCHEMA_VERSION),
        preprocess_identity=np.asarray(preprocess_identity),
        sequence_id=np.asarray(str(sequence_id)),
        source_hash=np.asarray(source_hash),
        feature_hash=np.asarray(content_hash(npy_path)),
    )
    np.savez_compressed(Path(npy_path).with_suffix(".validity.npz"), **payload)


def load_validity(npy_path: str | Path, features: np.ndarray,
                  preprocess_identity: str, sequence_id: str) -> dict[str, Any]:
    path = Path(npy_path).with_suffix(".validity.npz")
    if not path.is_file():
        raise FileNotFoundError(f"Missing schema-v{PREPROCESS_SCHEMA_VERSION} validity sidecar: {path}. Regenerate cache.")
    with np.load(path, allow_pickle=False) as archive:
        metadata = {key: archive[key] for key in archive.files}
    for key, expected in {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "preprocess_identity": preprocess_identity,
        "sequence_id": str(sequence_id),
        "feature_hash": content_hash(npy_path),
    }.items():
        value = metadata.get(key)
        if not isinstance(value, np.ndarray) or value.shape != () or value.item() != expected:
            raise ValueError(f"Validity sidecar ownership mismatch: {key} in {path}.")
    source = metadata.get("source_hash")
    if source is None or source.shape != () or len(str(source.item())) != 64:
        raise ValueError(f"Missing source content hash in {path}.")
    validate_validity(metadata, features)
    return metadata
