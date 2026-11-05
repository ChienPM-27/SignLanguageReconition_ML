"""Data augmentation module for ISLR landmark feature sequences.

Provides:
- horizontal_flip: Reflect local and torso coordinates, swap anatomical pose sides,
                   and preserve absent zeros.
- random_rotate_2d: 2D rotation of hands, wrists, and pose with recomputed motion.
- add_jitter: Gaussian perturbation of shape (with fixed wrist origin) and wrists,
              with recomputed motion.
- scale_translate: Scale and translation of global wrist coordinates with recomputed motion.
- shape_scale: Independent x, y scaling of local hand shapes.
- time_stretch: Explicitly disabled in Phase 0.
- apply_augmentation: High-level config-driven transformation.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional
import numpy as np

from src.features.schema import (
    SLICE_GLOBAL_WRISTS,
    SLICE_HAND_MASKS,
    SLICE_LOCAL_HAND_SHAPES,
    SLICE_NORM_POSE,
    SLICE_POSE_MASK,
    SLICE_REL_WRIST,
    SLICE_SHOULDER_SCALE,
    SLICE_TORSO_VEL,
    SLICE_WRIST_VEL,
)


def recompute_derived_features(features: np.ndarray) -> np.ndarray:
    """Recompute relative wrist [132:135] and wrist velocity [135:141] using gated valid endpoints.

    Contract:
    - w[t, h] = coords if m[t, h] else 0.0
    - r[t] = w[t, 0] - w[t, 1] if m[t, 0] and m[t, 1] else 0.0
    - v[0, h] = 0.0
    - v[t, h] = w[t, h] - w[t-1, h] if m[t, h] and m[t-1, h] else 0.0
    - If D >= 160: sanitize unsupported pose channels and gate torso velocity u[t] on p[t] and p[t-1].
    """
    out = features.copy()
    T, D = out.shape
    if D >= 143:
        m = out[:, SLICE_HAND_MASKS] > 0.5  # (T, 2)
        wrists = out[:, SLICE_GLOBAL_WRISTS].reshape(T, 2, 3)
        # Ensure unsupported wrists are strictly zeroed
        wrists = np.where(m[:, :, np.newaxis], wrists, 0.0)
        out[:, SLICE_GLOBAL_WRISTS] = wrists.reshape(T, 6)

        two_hand_mask = (m[:, 0] & m[:, 1])[:, np.newaxis]
        rel_wrist = wrists[:, 0, :] - wrists[:, 1, :]
        out[:, SLICE_REL_WRIST] = np.where(two_hand_mask, rel_wrist, 0.0)

        wrist_vel = np.zeros((T, 2, 3), dtype=out.dtype)
        if T > 1:
            vel_valid = (m[1:, :] & m[:-1, :])[:, :, np.newaxis]
            wrist_vel[1:] = np.where(vel_valid, wrists[1:] - wrists[:-1], 0.0)
        out[:, SLICE_WRIST_VEL] = wrist_vel.reshape(T, 6)

    if D >= 160:
        p = out[:, SLICE_POSE_MASK].flatten() > 0.5  # (T,)
        out[:, SLICE_NORM_POSE] = np.where(p[:, np.newaxis], out[:, SLICE_NORM_POSE], 0.0)
        out[:, SLICE_SHOULDER_SCALE] = np.where(p[:, np.newaxis], out[:, SLICE_SHOULDER_SCALE], 0.0)
        u_valid = np.zeros(T, dtype=bool)
        if T > 1:
            u_valid[1:] = p[1:] & p[:-1]
        out[:, SLICE_TORSO_VEL] = np.where(u_valid[:, np.newaxis], out[:, SLICE_TORSO_VEL], 0.0)

    return out


def add_jitter(
    features: np.ndarray,
    sigma: float = 0.01,
    noise_shape: Optional[np.ndarray] = None,
    noise_wrist: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Add Gaussian noise to local hand shapes and wrists on usable slots, then recompute motion.

    Local hand shape jitter keeps landmark 0 (wrist origin) identically 0.
    """
    out = features.copy()
    T, D = out.shape
    if sigma <= 0.0:
        return out

    m = out[:, SLICE_HAND_MASKS] > 0.5 if D >= 143 else np.ones((T, 2), dtype=bool)

    # 1. Local shapes [0:126] -> (T, 2, 21, 3)
    if D >= 126:
        shapes = out[:, SLICE_LOCAL_HAND_SHAPES].reshape(T, 2, 21, 3)
        if noise_shape is None:
            noise_s = np.random.normal(0.0, sigma, size=shapes.shape).astype(np.float32)
        else:
            noise_s = np.array(noise_shape, dtype=np.float32).copy()
            if noise_s.ndim == 2 and noise_s.shape == (T, 126):
                noise_s = noise_s.reshape(T, 2, 21, 3)
        # Keep local wrist landmark 0 zero
        noise_s[:, :, 0, :] = 0.0
        shapes = np.where(m[:, :, np.newaxis, np.newaxis], shapes + noise_s, 0.0)
        out[:, SLICE_LOCAL_HAND_SHAPES] = shapes.reshape(T, 126)

    # 2. Global wrists [126:132] -> (T, 2, 3)
    if D >= 132:
        wrists = out[:, SLICE_GLOBAL_WRISTS].reshape(T, 2, 3)
        if noise_wrist is None:
            noise_w = np.random.normal(0.0, sigma, size=wrists.shape).astype(np.float32)
        else:
            noise_w = np.array(noise_wrist, dtype=np.float32).copy()
            if noise_w.ndim == 2 and noise_w.shape == (T, 6):
                noise_w = noise_w.reshape(T, 2, 3)
        wrists = np.where(m[:, :, np.newaxis], wrists + noise_w, 0.0)
        out[:, SLICE_GLOBAL_WRISTS] = wrists.reshape(T, 6)

    # 3. Recompute motion to maintain internal consistency
    if D >= 143:
        out = recompute_derived_features(out)

    return out


def random_rotate_2d(
    features: np.ndarray,
    max_angle_deg: float = 8.0,
    angle_deg: Optional[float] = None,
) -> np.ndarray:
    """Rotate points in the (x, y) plane for usable slots and recompute motion.

    - Local shapes rotate around (0, 0).
    - Global wrists rotate around (0.5, 0.5).
    - Torso-normalized pose rotates around (0, 0).
    - Torso velocity rotates in the xy plane.
    """
    if angle_deg is None:
        angle_deg = float(np.random.uniform(-max_angle_deg, max_angle_deg))
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    out = features.copy()
    T, D = out.shape
    m = out[:, SLICE_HAND_MASKS] > 0.5 if D >= 143 else np.ones((T, 2), dtype=bool)

    # 1. Local shapes [0:126] -> (T, 2, 21, 3)
    if D >= 126:
        shapes = out[:, SLICE_LOCAL_HAND_SHAPES].reshape(T, 2, 21, 3)
        xy = shapes[:, :, :, :2].reshape(-1, 2)
        xy_rot = (xy @ rot_matrix.T).reshape(T, 2, 21, 2)
        shapes[:, :, :, :2] = np.where(m[:, :, np.newaxis, np.newaxis], xy_rot, 0.0)
        shapes[:, :, :, 2] = np.where(m[:, :, np.newaxis], shapes[:, :, :, 2], 0.0)
        out[:, SLICE_LOCAL_HAND_SHAPES] = shapes.reshape(T, 126)

    # 2. Global wrists [126:132] -> (T, 2, 3)
    if D >= 132:
        wrists = out[:, SLICE_GLOBAL_WRISTS].reshape(T, 2, 3)
        w_xy = wrists[:, :, :2].reshape(-1, 2) - 0.5
        w_xy_rot = (w_xy @ rot_matrix.T + 0.5).reshape(T, 2, 2)
        wrists[:, :, :2] = np.where(m[:, :, np.newaxis], w_xy_rot, 0.0)
        wrists[:, :, 2] = np.where(m, wrists[:, :, 2], 0.0)
        out[:, SLICE_GLOBAL_WRISTS] = wrists.reshape(T, 6)

    # 3. Recompute wrist derived features
    if D >= 143:
        out = recompute_derived_features(out)

    # 4. Pose rotation [143:155] (4 points) and torso velocity [155:158]
    if D >= 160:
        p = out[:, SLICE_POSE_MASK].flatten() > 0.5
        pose = out[:, SLICE_NORM_POSE].reshape(T, 4, 3)
        p_xy = pose[:, :, :2].reshape(-1, 2)
        p_xy_rot = (p_xy @ rot_matrix.T).reshape(T, 4, 2)
        pose[:, :, :2] = np.where(p[:, np.newaxis, np.newaxis], p_xy_rot, 0.0)
        pose[:, :, 2] = np.where(p[:, np.newaxis], pose[:, :, 2], 0.0)
        out[:, SLICE_NORM_POSE] = pose.reshape(T, 12)

        u = out[:, SLICE_TORSO_VEL]
        u_xy = u[:, :2]
        u_xy_rot = u_xy @ rot_matrix.T
        u_valid = np.zeros(T, dtype=bool)
        if T > 1:
            u_valid[1:] = p[1:] & p[:-1]
        u[:, :2] = np.where(u_valid[:, np.newaxis], u_xy_rot, 0.0)
        u[:, 2] = np.where(u_valid, u[:, 2], 0.0)
        out[:, SLICE_TORSO_VEL] = u
        out[:, SLICE_SHOULDER_SCALE] = np.where(p[:, np.newaxis], out[:, SLICE_SHOULDER_SCALE], 0.0)

    return out


def horizontal_flip(features: np.ndarray, swap_hands: bool = False) -> np.ndarray:
    """Horizontal mirror reflection across x axis.

    Exact mathematical contract:
    - H'[t, h, j] = diag(-1, 1, 1) H[t, h, j]
    - w'[t, h] = (1 - w_x, w_y, w_z)
    - q'[t, k] = diag(-1, 1, 1) q[t, pi[k]] where pi = [1, 0, 3, 2]
    - u'[t] = diag(-1, 1, 1) u[t]
    - b'[t] = b[t]; p'[t] = p[t]
    - Unsupported slots strictly remain 0.
    """
    out = features.copy()
    T, D = out.shape

    m = out[:, SLICE_HAND_MASKS] > 0.5 if D >= 143 else np.ones((T, 2), dtype=bool)

    # 1. Local Shapes (126 dims)
    if D >= 126:
        shapes = out[:, SLICE_LOCAL_HAND_SHAPES].reshape(T, 2, 21, 3)
        shapes[:, :, :, 0] = np.where(m[:, :, np.newaxis], -shapes[:, :, :, 0], 0.0)
        shapes[:, :, :, 1:] = np.where(m[:, :, np.newaxis, np.newaxis], shapes[:, :, :, 1:], 0.0)
        if swap_hands:
            shapes = shapes[:, [1, 0], :, :]
        out[:, SLICE_LOCAL_HAND_SHAPES] = shapes.reshape(T, 126)

    # 2. Global Wrists (6 dims)
    if D >= 132:
        wrists = out[:, SLICE_GLOBAL_WRISTS].reshape(T, 2, 3)
        wrists[:, :, 0] = np.where(m, 1.0 - wrists[:, :, 0], 0.0)
        wrists[:, :, 1:] = np.where(m[:, :, np.newaxis], wrists[:, :, 1:], 0.0)
        if swap_hands:
            wrists = wrists[:, [1, 0], :]
        out[:, SLICE_GLOBAL_WRISTS] = wrists.reshape(T, 6)

    # 3. Swap Hand Masks if swap_hands is requested
    if D >= 143 and swap_hands:
        out[:, SLICE_HAND_MASKS] = out[:, [142, 141]]

    # 4. Recompute wrist motion features
    if D >= 143:
        out = recompute_derived_features(out)

    # 5. Pose landmarks (12 coords torso-centered, 3 coords torso velocity)
    if D >= 160:
        p = out[:, SLICE_POSE_MASK].flatten() > 0.5
        pose = out[:, SLICE_NORM_POSE].reshape(T, 4, 3)
        # Torso-centered coordinates: negate x (NOT 1 - x!)
        pose[:, :, 0] = np.where(p[:, np.newaxis], -pose[:, :, 0], 0.0)
        pose[:, :, 1:] = np.where(p[:, np.newaxis, np.newaxis], pose[:, :, 1:], 0.0)
        # Swap anatomical sides: L_shoulder <-> R_shoulder, L_elbow <-> R_elbow
        pose_flipped = pose[:, [1, 0, 3, 2], :]
        out[:, SLICE_NORM_POSE] = pose_flipped.reshape(T, 12)

        # Torso velocity: negate x
        u = out[:, SLICE_TORSO_VEL]
        u_valid = np.zeros(T, dtype=bool)
        if T > 1:
            u_valid[1:] = p[1:] & p[:-1]
        u[:, 0] = np.where(u_valid, -u[:, 0], 0.0)
        u[:, 1:] = np.where(u_valid[:, np.newaxis], u[:, 1:], 0.0)
        out[:, SLICE_TORSO_VEL] = u

        out[:, SLICE_SHOULDER_SCALE] = np.where(p[:, np.newaxis], out[:, SLICE_SHOULDER_SCALE], 0.0)

    return out


def shape_scale(
    features: np.ndarray,
    r: float = 0.10,
    sx: Optional[float] = None,
    sy: Optional[float] = None,
) -> np.ndarray:
    """Scale x and y axes of local hand shapes [0:126] for usable slots."""
    out = features.copy()
    T, D = out.shape
    if D < 126 or r <= 0.0:
        return out

    if sx is None:
        sx = float(np.random.uniform(1.0 - r, 1.0 + r))
    if sy is None:
        sy = float(np.random.uniform(1.0 - r, 1.0 + r))

    m = out[:, SLICE_HAND_MASKS] > 0.5 if D >= 143 else np.ones((T, 2), dtype=bool)

    shapes = out[:, SLICE_LOCAL_HAND_SHAPES].reshape(T, 2, 21, 3)
    sh_x = shapes[:, :, :, 0] * sx
    sh_y = shapes[:, :, :, 1] * sy
    shapes[:, :, :, 0] = np.where(m[:, :, np.newaxis], sh_x, 0.0)
    shapes[:, :, :, 1] = np.where(m[:, :, np.newaxis], sh_y, 0.0)
    shapes[:, :, :, 2] = np.where(m[:, :, np.newaxis], shapes[:, :, :, 2], 0.0)
    out[:, SLICE_LOCAL_HAND_SHAPES] = shapes.reshape(T, 126)

    return out


def scale_translate(
    features: np.ndarray,
    scale_min: float = 0.85,
    scale_max: float = 1.15,
    translate: float = 0.10,
    s: Optional[float] = None,
    t: Optional[Any] = None,
    tx: Optional[float] = None,
    ty: Optional[float] = None,
) -> np.ndarray:
    """Scale and translate global wrist coordinates for usable slots, then recompute motion."""
    out = features.copy()
    T, D = out.shape
    if D < 132:
        return out

    if t is not None:
        if isinstance(t, (tuple, list, np.ndarray)) and len(t) == 2:
            tx = float(t[0])
            ty = float(t[1])
        else:
            raise ValueError(f"t must be a 2-element sequence (tx, ty), got {t}")

    if s is None:
        s = float(np.random.uniform(scale_min, scale_max))
    if tx is None:
        tx = float(np.random.uniform(-translate, translate))
    if ty is None:
        ty = float(np.random.uniform(-translate, translate))

    m = out[:, SLICE_HAND_MASKS] > 0.5 if D >= 143 else np.ones((T, 2), dtype=bool)

    wrists = out[:, SLICE_GLOBAL_WRISTS].reshape(T, 2, 3)
    w_x = (wrists[:, :, 0] - 0.5) * s + 0.5 + tx
    w_y = (wrists[:, :, 1] - 0.5) * s + 0.5 + ty
    wrists[:, :, 0] = np.where(m, w_x, 0.0)
    wrists[:, :, 1] = np.where(m, w_y, 0.0)
    wrists[:, :, 2] = np.where(m, wrists[:, :, 2], 0.0)
    out[:, SLICE_GLOBAL_WRISTS] = wrists.reshape(T, 6)

    if D >= 143:
        out = recompute_derived_features(out)

    return out


def time_stretch(features: np.ndarray, max_stretch: float = 0.2) -> np.ndarray:
    """Time-stretch operation (explicitly disabled in Phase 0)."""
    if max_stretch > 0.0:
        raise ValueError(
            "time_stretch is disabled in Phase 0 because current implementations "
            "do not share a validity-aware temporal contract."
        )
    return features.copy()


def _get_aug_param(cfg_dict: Dict[str, Any], key: str, default: Any) -> Any:
    val = cfg_dict.get(key)
    return val if val is not None else default


def apply_augmentation(features: np.ndarray, cfg: Any = None) -> np.ndarray:
    """Apply sequence of spatial augmentations according to AugmentationConfig or dict."""
    if cfg is None:
        return apply_random_augmentation(features)

    flip_cfg = getattr(cfg, "flip", {}) if not isinstance(cfg, dict) else cfg.get("flip", {})
    rotate_cfg = getattr(cfg, "rotate", {}) if not isinstance(cfg, dict) else cfg.get("rotate", {})
    shape_scale_cfg = getattr(cfg, "shape_scale", {}) if not isinstance(cfg, dict) else cfg.get("shape_scale", {})
    scale_translate_cfg = getattr(cfg, "scale_translate", {}) if not isinstance(cfg, dict) else cfg.get("scale_translate", {})
    jitter_cfg = getattr(cfg, "jitter", {}) if not isinstance(cfg, dict) else cfg.get("jitter", {})
    stretch_cfg = getattr(cfg, "time_stretch", {}) if not isinstance(cfg, dict) else cfg.get("time_stretch", {})

    out = features.copy()

    applied = False

    # 1. Flip
    if _get_aug_param(flip_cfg, "enabled", False):
        prob = float(_get_aug_param(flip_cfg, "prob", 0.5))
        swap_hands = bool(_get_aug_param(flip_cfg, "swap_hands", False))
        if np.random.rand() < prob:
            out = horizontal_flip(out, swap_hands=swap_hands)
            applied = True

    # 2. Rotate
    if _get_aug_param(rotate_cfg, "enabled", False):
        prob = float(_get_aug_param(rotate_cfg, "prob", 0.5))
        max_angle = float(_get_aug_param(rotate_cfg, "max_angle_deg", 8.0))
        if np.random.rand() < prob:
            out = random_rotate_2d(out, max_angle_deg=max_angle)
            applied = True

    # 3. Shape Scale
    if _get_aug_param(shape_scale_cfg, "enabled", False):
        prob = float(_get_aug_param(shape_scale_cfg, "prob", 0.5))
        r = float(_get_aug_param(shape_scale_cfg, "r", 0.10))
        if np.random.rand() < prob:
            out = shape_scale(out, r=r)
            applied = True

    # 4. Scale & Translate
    if _get_aug_param(scale_translate_cfg, "enabled", False):
        prob = float(_get_aug_param(scale_translate_cfg, "prob", 0.7))
        scale_min = float(_get_aug_param(scale_translate_cfg, "scale_min", 0.85))
        scale_max = float(_get_aug_param(scale_translate_cfg, "scale_max", 1.15))
        translate = float(_get_aug_param(scale_translate_cfg, "translate", 0.10))
        if np.random.rand() < prob:
            out = scale_translate(out, scale_min=scale_min, scale_max=scale_max, translate=translate)
            applied = True

    # 5. Jitter
    if _get_aug_param(jitter_cfg, "enabled", False):
        prob = float(_get_aug_param(jitter_cfg, "prob", 0.5))
        sigma = float(_get_aug_param(jitter_cfg, "sigma", 0.01))
        if np.random.rand() < prob:
            out = add_jitter(out, sigma=sigma)
            applied = True

    # 6. Time Stretch (must fail if enabled with positive prob and stretch)
    if _get_aug_param(stretch_cfg, "enabled", False):
        prob = float(_get_aug_param(stretch_cfg, "prob", 0.5))
        max_stretch = float(_get_aug_param(stretch_cfg, "max_stretch", 0.2))
        if prob > 0.0 and max_stretch > 0.0:
            time_stretch(out, max_stretch=max_stretch)
            applied = True

    # Final defensive recomputation of derived features
    if applied:
        out = recompute_derived_features(out)
    return out


def apply_random_augmentation(features: np.ndarray) -> np.ndarray:
    """Default augmentation composition for backward compatibility."""
    default_cfg = {
        "flip": {"enabled": True, "prob": 0.5, "swap_hands": False},
        "rotate": {"enabled": True, "prob": 0.5, "max_angle_deg": 8.0},
        "shape_scale": {"enabled": True, "prob": 0.5, "r": 0.10},
        "scale_translate": {"enabled": True, "prob": 0.7, "scale_min": 0.85, "scale_max": 1.15, "translate": 0.10},
        "jitter": {"enabled": True, "prob": 0.5, "sigma": 0.01},
        "time_stretch": {"enabled": False, "prob": 0.0, "max_stretch": 0.0},
    }
    return apply_augmentation(features, cfg=default_cfg)
