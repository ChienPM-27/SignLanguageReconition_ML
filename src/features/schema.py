"""Canonical feature layout, coordinate space definitions, and schema constants for ISLR.

Defines format invariants and preprocessing schema versions for:
- Hands-only feature set (143 dimensions)
- Hands+Pose feature set (160 dimensions)
- Wave B body-relative torso geometry (162 dimensions)
- Wave B compact fingertip displacements (190 / 192 dimensions)
- Wave B finger articulation bend cosines (210 / 212 dimensions)
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Optional, Tuple

# Version 5 adds optional compact face channels and face reliability sidecars.
PREPROCESS_SCHEMA_VERSION: int = 5

# Supported feature sets
FEATURE_SET_HANDS: str = "hands"
FEATURE_SET_HANDS_POSE: str = "hands_pose"
ALLOWED_FEATURE_SETS: Tuple[str, ...] = (FEATURE_SET_HANDS, FEATURE_SET_HANDS_POSE)

# Feature vector base dimensions
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

# Fingertip indices (Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20)
FINGERTIP_LANDMARK_INDICES: Tuple[int, ...] = (4, 8, 12, 16, 20)

# Finger chains for articulation angles (consecutive bone-pairs)
FINGER_CHAINS: Tuple[Tuple[int, int, int, int], ...] = (
    (1, 2, 3, 4),       # Thumb chain
    (5, 6, 7, 8),       # Index chain
    (9, 10, 11, 12),    # Middle chain
    (13, 14, 15, 16),   # Ring chain
    (17, 18, 19, 20),   # Pinky chain
)

NUM_GEOMETRY_SUPPORT_DIMS: int = 2
NUM_FINGERTIP_DISPLACEMENT_DIMS: int = 30  # 2 hands * 5 tips * 3 coords
NUM_BEND_COSINES_DIMS: int = 20            # 2 hands * 5 chains * 2 bone-pairs

# Compact face topology: nose, eye corners/lids, lip corners/lids, brows, chin.
# Fixed schema semantics, not a configurable landmark subset.
FACE_LANDMARK_INDICES: Tuple[int, ...] = (
    1, 33, 263, 133, 362, 159, 386, 145, 374, 61, 291, 13, 14, 70, 300, 152,
)
FACE_FLIP_PERMUTATION: Tuple[int, ...] = (
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 11, 12, 14, 13, 15,
)
NUM_FACE_SHAPE_DIMS: int = len(FACE_LANDMARK_INDICES) * 2
NUM_FACE_FEATURE_DIMS: int = NUM_FACE_SHAPE_DIMS + 3 + 1
FACE_OPENNESS_PAIRS: Tuple[Tuple[int, int, int, int], ...] = (
    (11, 12, 9, 10), (5, 7, 1, 3), (6, 8, 2, 4),
)

# Base feature slice definitions (end-exclusive)
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


@dataclass(frozen=True)
class FeatureLayout:
    """Canonical descriptor for feature matrix slicing and channel semantics."""

    total_dim: int
    feature_set: str
    wrist_reference: str
    palm_scale: str
    fingertip_displacement: bool
    bend_cosines: bool
    face: bool = False

    # Base Hand Slices
    slice_local_hand_shapes: slice = slice(0, 126)
    slice_wrists: slice = slice(126, 132)
    slice_rel_wrist: slice = slice(132, 135)
    slice_wrist_vel: slice = slice(135, 141)
    slice_hand_masks: slice = slice(141, 143)

    # Pose Slices (present if feature_set == 'hands_pose')
    slice_norm_pose: Optional[slice] = None
    slice_torso_vel: Optional[slice] = None
    slice_shoulder_scale: Optional[slice] = None
    slice_pose_mask: Optional[slice] = None
    slice_pose_features: Optional[slice] = None

    # Wave B Extension Slices
    slice_geometry_support: Optional[slice] = None
    slice_fingertip_displacement: Optional[slice] = None
    slice_bend_cosines: Optional[slice] = None
    slice_face_shapes: Optional[slice] = None
    slice_face_ratios: Optional[slice] = None
    slice_face_mask: Optional[slice] = None

    @property
    def has_pose(self) -> bool:
        """True if layout contains torso/pose feature slices."""
        return self.feature_set == FEATURE_SET_HANDS_POSE

    @classmethod
    def build(
        cls,
        feature_set: str = FEATURE_SET_HANDS,
        wrist_reference: str = "image",
        palm_scale: str = "frame",
        fingertip_displacement: bool = False,
        bend_cosines: bool = False,
        face: bool = False,
    ) -> FeatureLayout:
        if feature_set not in ALLOWED_FEATURE_SETS:
            raise ValueError(
                f"Unsupported feature_set '{feature_set}'. Allowed: {list(ALLOWED_FEATURE_SETS)}"
            )
        if wrist_reference == "torso_xy" and feature_set != FEATURE_SET_HANDS_POSE:
            raise ValueError("features.wrist_reference='torso_xy' requires features.set='hands_pose'.")

        curr = 143
        slice_norm_pose = None
        slice_torso_vel = None
        slice_shoulder_scale = None
        slice_pose_mask = None
        slice_pose_features = None

        if feature_set == FEATURE_SET_HANDS_POSE:
            slice_norm_pose = slice(143, 155)
            slice_torso_vel = slice(155, 158)
            slice_shoulder_scale = slice(158, 159)
            slice_pose_mask = slice(159, 160)
            slice_pose_features = slice(143, 160)
            curr = 160

        slice_geom = None
        if wrist_reference == "torso_xy":
            slice_geom = slice(curr, curr + NUM_GEOMETRY_SUPPORT_DIMS)
            curr += NUM_GEOMETRY_SUPPORT_DIMS

        slice_tips = None
        if fingertip_displacement:
            slice_tips = slice(curr, curr + NUM_FINGERTIP_DISPLACEMENT_DIMS)
            curr += NUM_FINGERTIP_DISPLACEMENT_DIMS

        slice_bends = None
        if bend_cosines:
            slice_bends = slice(curr, curr + NUM_BEND_COSINES_DIMS)
            curr += NUM_BEND_COSINES_DIMS

        slice_face_shapes = slice_face_ratios = slice_face_mask = None
        if face:
            slice_face_shapes = slice(curr, curr + NUM_FACE_SHAPE_DIMS)
            slice_face_ratios = slice(curr + NUM_FACE_SHAPE_DIMS, curr + NUM_FACE_SHAPE_DIMS + 3)
            slice_face_mask = slice(curr + NUM_FACE_FEATURE_DIMS - 1, curr + NUM_FACE_FEATURE_DIMS)
            curr += NUM_FACE_FEATURE_DIMS

        return cls(
            total_dim=curr,
            feature_set=feature_set,
            wrist_reference=wrist_reference,
            palm_scale=palm_scale,
            fingertip_displacement=fingertip_displacement,
            bend_cosines=bend_cosines,
            face=face,
            slice_norm_pose=slice_norm_pose,
            slice_torso_vel=slice_torso_vel,
            slice_shoulder_scale=slice_shoulder_scale,
            slice_pose_mask=slice_pose_mask,
            slice_pose_features=slice_pose_features,
            slice_geometry_support=slice_geom,
            slice_fingertip_displacement=slice_tips,
            slice_bend_cosines=slice_bends,
            slice_face_shapes=slice_face_shapes,
            slice_face_ratios=slice_face_ratios,
            slice_face_mask=slice_face_mask,
        )


def infer_feature_layout_from_dim(dim: int) -> FeatureLayout:
    """Infer canonical layout for standard feature dimensions."""
    face_base_dims = (143, 160, 162, 163, 173, 180, 182, 190, 192, 193, 210, 212)
    if dim - NUM_FACE_FEATURE_DIMS in face_base_dims:
        base = infer_feature_layout_from_dim(dim - NUM_FACE_FEATURE_DIMS)
        return FeatureLayout.build(base.feature_set, base.wrist_reference, base.palm_scale,
                                   base.fingertip_displacement, base.bend_cosines, face=True)
    if dim == 143:
        return FeatureLayout.build(feature_set=FEATURE_SET_HANDS)
    elif dim == 160:
        return FeatureLayout.build(feature_set=FEATURE_SET_HANDS_POSE)
    elif dim == 162:
        return FeatureLayout.build(feature_set=FEATURE_SET_HANDS_POSE, wrist_reference="torso_xy")
    elif dim == 163:
        return FeatureLayout.build(feature_set=FEATURE_SET_HANDS, bend_cosines=True)
    elif dim == 173:
        return FeatureLayout.build(feature_set=FEATURE_SET_HANDS, fingertip_displacement=True)
    elif dim == 180:
        return FeatureLayout.build(feature_set=FEATURE_SET_HANDS_POSE, bend_cosines=True)
    elif dim == 182:
        return FeatureLayout.build(
            feature_set=FEATURE_SET_HANDS_POSE, wrist_reference="torso_xy", bend_cosines=True
        )
    elif dim == 190:
        return FeatureLayout.build(feature_set=FEATURE_SET_HANDS_POSE, fingertip_displacement=True)
    elif dim == 192:
        return FeatureLayout.build(
            feature_set=FEATURE_SET_HANDS_POSE, wrist_reference="torso_xy", fingertip_displacement=True
        )
    elif dim == 193:
        return FeatureLayout.build(
            feature_set=FEATURE_SET_HANDS, fingertip_displacement=True, bend_cosines=True
        )
    elif dim == 210:
        return FeatureLayout.build(
            feature_set=FEATURE_SET_HANDS_POSE, fingertip_displacement=True, bend_cosines=True
        )
    elif dim == 212:
        return FeatureLayout.build(
            feature_set=FEATURE_SET_HANDS_POSE,
            wrist_reference="torso_xy",
            fingertip_displacement=True,
            bend_cosines=True,
        )
    raise ValueError(f"Cannot uniquely infer feature layout from dimension {dim}.")


def _extract_config_val(cfg_obj: Any, section: str, key: str, default: Any) -> Any:
    """Helper to extract configuration value from Config object, nested dict, or flat dict."""
    flat_key = f"{section}.{key}"
    if isinstance(cfg_obj, dict):
        if flat_key in cfg_obj:
            return cfg_obj[flat_key]
        if section in cfg_obj and isinstance(cfg_obj[section], dict) and key in cfg_obj[section]:
            return cfg_obj[section][key]
        if key in cfg_obj:
            return cfg_obj[key]
    sec_obj = getattr(cfg_obj, section, None)
    if sec_obj is not None:
        if isinstance(sec_obj, dict) and key in sec_obj:
            return sec_obj[key]
        if hasattr(sec_obj, key):
            return getattr(sec_obj, key)
    if hasattr(cfg_obj, key):
        return getattr(cfg_obj, key)
    return default


def get_feature_layout_from_config(cfg: Any) -> FeatureLayout:
    """Instantiate exact FeatureLayout from configuration."""
    feature_set = str(_extract_config_val(cfg, "features", "set", FEATURE_SET_HANDS))
    wrist_reference = str(_extract_config_val(cfg, "features", "wrist_reference", "image"))
    palm_scale = str(_extract_config_val(cfg, "features", "palm_scale", "frame"))
    fingertip_displacement = bool(_extract_config_val(cfg, "features", "fingertip_displacement", False))
    bend_cosines = bool(_extract_config_val(cfg, "features", "bend_cosines", False))
    face = bool(_extract_config_val(cfg, "features", "face", False))
    return FeatureLayout.build(
        feature_set=feature_set,
        wrist_reference=wrist_reference,
        palm_scale=palm_scale,
        fingertip_displacement=fingertip_displacement,
        bend_cosines=bend_cosines,
        face=face,
    )


def get_feature_dim(
    feature_set: Any = FEATURE_SET_HANDS,
    wrist_reference: str = "image",
    fingertip_displacement: bool = False,
    bend_cosines: bool = False,
    face: bool = False,
) -> int:
    """Deterministically derive feature vector dimension from arguments or config object."""
    if isinstance(feature_set, str):
        return FeatureLayout.build(
            feature_set=feature_set,
            wrist_reference=wrist_reference,
            fingertip_displacement=fingertip_displacement,
            bend_cosines=bend_cosines,
            face=face,
        ).total_dim
    layout = get_feature_layout_from_config(feature_set)
    return layout.total_dim


def compute_preprocess_identity(cfg: Any) -> Tuple[str, Dict[str, Any]]:
    """Compute deterministic SHA256 identity hash from canonical preprocessing settings.

    Payload includes schema version and all settings affecting feature output or clip acceptance.
    Supports Config objects, nested dictionary representations, and flat parameter dicts.
    """
    raw_schema = _extract_config_val(cfg, "preprocess", "schema_version", PREPROCESS_SCHEMA_VERSION)
    schema_version = int(raw_schema) if raw_schema is not None else PREPROCESS_SCHEMA_VERSION

    meta_params: Dict[str, Any] = {
        "schema_version": schema_version,
        "preprocess.dominant_hand": str(_extract_config_val(cfg, "preprocess", "dominant_hand", "auto")),
        "preprocess.max_gap_frames": int(_extract_config_val(cfg, "preprocess", "max_gap_frames", 3)),
        "preprocess.min_valid_frames": int(_extract_config_val(cfg, "preprocess", "min_valid_frames", 3)),
        "preprocess.normalization_epsilon": float(_extract_config_val(cfg, "preprocess", "normalization_epsilon", 0.0001)),
        "preprocess.pad_frames_around": int(_extract_config_val(cfg, "preprocess", "pad_frames_around", 2)),
        "preprocess.sequence_length": int(_extract_config_val(cfg, "preprocess", "sequence_length", 32)),
        "preprocess.trim_no_hand": bool(_extract_config_val(cfg, "preprocess", "trim_no_hand", True)),
        "features.set": str(_extract_config_val(cfg, "features", "set", FEATURE_SET_HANDS)),
        "features.wrist_reference": str(_extract_config_val(cfg, "features", "wrist_reference", "image")),
        "features.palm_scale": str(_extract_config_val(cfg, "features", "palm_scale", "frame")),
        "features.fingertip_displacement": bool(_extract_config_val(cfg, "features", "fingertip_displacement", False)),
        "features.bend_cosines": bool(_extract_config_val(cfg, "features", "bend_cosines", False)),
        "features.face": bool(_extract_config_val(cfg, "features", "face", False)),
    }
    serialized = json.dumps(meta_params, sort_keys=True, separators=(",", ":"))
    identity_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return identity_hash, meta_params
