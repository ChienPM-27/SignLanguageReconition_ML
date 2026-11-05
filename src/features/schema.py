"""Canonical feature layout, coordinate space definitions, and schema constants for ISLR.

Defines format invariants and preprocessing schema versions for:
- Hands-only feature set (143 dimensions)
- Hands+Pose feature set (160 dimensions)
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

# Preprocessing schema identifier (version 2 removes legacy interpolation/reflection defects)
PREPROCESS_SCHEMA_VERSION: int = 2

# Supported feature sets
FEATURE_SET_HANDS: str = "hands"
FEATURE_SET_HANDS_POSE: str = "hands_pose"
ALLOWED_FEATURE_SETS: Tuple[str, ...] = (FEATURE_SET_HANDS, FEATURE_SET_HANDS_POSE)

# Feature vector dimensions
FEATURE_DIM_HANDS: int = 143
FEATURE_DIM_HANDS_POSE: int = 160

# Landmark geometry counts
NUM_HANDS: int = 2
NUM_HAND_LANDMARKS: int = 21
COORDS_PER_LANDMARK: int = 3
TOTAL_HAND_SHAPE_DIMS: int = NUM_HANDS * NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK  # 126

NUM_POSE_LANDMARKS: int = 33
NUM_SELECTED_POSE_LANDMARKS: int = 4

# MediaPipe Pose keypoint indices for upper body:
# 11: Left Shoulder, 12: Right Shoulder, 13: Left Elbow, 14: Right Elbow
POSE_INDEX_LEFT_SHOULDER: int = 11
POSE_INDEX_RIGHT_SHOULDER: int = 12
POSE_INDEX_LEFT_ELBOW: int = 13
POSE_INDEX_RIGHT_ELBOW: int = 14
POSE_SELECTED_LANDMARKS: Tuple[int, ...] = (
    POSE_INDEX_LEFT_SHOULDER,
    POSE_INDEX_RIGHT_SHOULDER,
    POSE_INDEX_LEFT_ELBOW,
    POSE_INDEX_RIGHT_ELBOW,
)

# Anatomical swap permutation on selected pose points [L_sh, R_sh, L_el, R_el] -> [R_sh, L_sh, R_el, L_el]
POSE_FLIP_PERMUTATION: Tuple[int, ...] = (1, 0, 3, 2)

# Feature slice definitions (end-exclusive)
SLICE_LOCAL_HAND_SHAPES: slice = slice(0, 126)
SLICE_GLOBAL_WRISTS: slice = slice(126, 132)
SLICE_REL_WRIST: slice = slice(132, 135)
SLICE_WRIST_VEL: slice = slice(135, 141)
SLICE_HAND_MASKS: slice = slice(141, 143)

SLICE_NORM_POSE: slice = slice(143, 155)
SLICE_TORSO_VEL: slice = slice(155, 158)
SLICE_SHOULDER_SCALE: slice = slice(158, 159)
SLICE_POSE_MASK: slice = slice(159, 160)
SLICE_POSE_FEATURES: slice = slice(143, 160)


def get_feature_dim(feature_set: str = FEATURE_SET_HANDS) -> int:
    """Deterministically derive feature vector dimension from canonical feature set.

    Args:
        feature_set: 'hands' (143) or 'hands_pose' (160).

    Returns:
        int: Feature vector dimension.

    Raises:
        ValueError: If feature_set is not supported.
    """
    if feature_set == FEATURE_SET_HANDS:
        return FEATURE_DIM_HANDS
    elif feature_set == FEATURE_SET_HANDS_POSE:
        return FEATURE_DIM_HANDS_POSE
    raise ValueError(
        f"Unsupported feature_set '{feature_set}'. Allowed: {list(ALLOWED_FEATURE_SETS)}"
    )


def compute_preprocess_identity(cfg: Any) -> Tuple[str, Dict[str, Any]]:
    """Compute deterministic SHA256 identity hash from canonical preprocessing settings.

    Payload includes schema version and all settings affecting feature output or clip acceptance.
    """
    prep = getattr(cfg, "preprocess", cfg)
    feat = getattr(cfg, "features", cfg)

    meta_params: Dict[str, Any] = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "preprocess.dominant_hand": str(getattr(prep, "dominant_hand", "auto")),
        "preprocess.max_gap_frames": int(getattr(prep, "max_gap_frames", 3)),
        "preprocess.min_valid_frames": int(getattr(prep, "min_valid_frames", 3)),
        "preprocess.normalization_epsilon": float(getattr(prep, "normalization_epsilon", 0.0001)),
        "preprocess.pad_frames_around": int(getattr(prep, "pad_frames_around", 2)),
        "preprocess.sequence_length": int(getattr(prep, "sequence_length", 32)),
        "preprocess.trim_no_hand": bool(getattr(prep, "trim_no_hand", True)),
        "features.set": str(getattr(feat, "set", "hands")),
    }
    serialized = json.dumps(meta_params, sort_keys=True, separators=(",", ":"))
    identity_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return identity_hash, meta_params
