"""Module biến đổi dữ liệu (Data Augmentation) cho chuỗi đặc trưng ISLR.

Hỗ trợ:
- horizontal_flip: Lật gương quanh trục x và hoán đổi hai bàn tay.
- add_jitter: Thêm nhiễu Gaussian nhẹ vào tọa độ landmarks.
- random_rotate_2d: Xoay nhẹ các điểm trong mặt phẳng (x, y).
- time_stretch: Co giãn trục thời gian của chuỗi cử chỉ.
- apply_augmentation: Điều khiển qua AugmentationConfig / dict.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional
import numpy as np


def recompute_derived_features(features: np.ndarray) -> np.ndarray:
    """Tính lại relative wrist [132:135] và wrist velocity [135:141] từ [126:132] và mask [141:143].

    Đúng chuẩn compute_feature_vector_143 trong src/features/normalization.py:
    - rel = wrist0 - wrist1 (chỉ khi CẢ hai tay có mặt: mask0 > 0.5 & mask1 > 0.5, else 0.0)
    - velocity: vel[t] = w[t] - w[t-1] với vel[0] = 0.0 (không mask).
    """
    out = features.copy()
    T, D = out.shape
    if D >= 143:
        wrists = out[:, 126:132].reshape(T, 2, 3)
        two_hand_mask = ((out[:, 141] > 0.5) & (out[:, 142] > 0.5))[:, np.newaxis]
        rel_wrist = wrists[:, 0, :] - wrists[:, 1, :]
        out[:, 132:135] = np.where(two_hand_mask, rel_wrist, 0.0)

        wrist_vel = np.zeros((T, 2, 3), dtype=out.dtype)
        if T > 1:
            wrist_vel[1:] = wrists[1:] - wrists[:-1]
        out[:, 135:141] = wrist_vel.reshape(T, 6)
    return out


def add_jitter(
    features: np.ndarray,
    sigma: float = 0.01,
    noise_shape: Optional[np.ndarray] = None,
    noise_wrist: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Thêm nhiễu Gaussian nhẹ vào tọa độ landmarks CHỈ cho slot có mặt (shape + wrist).

    Không recompute velocity (giữ velocity sạch).
    """
    out = features.copy()
    T, D = out.shape
    if sigma <= 0.0:
        return out

    mask = out[:, 141:143] > 0.5 if D >= 143 else np.ones((T, 2), dtype=bool)

    # 1. Local shapes [0:126] -> (T, 2, 21, 3)
    if D >= 126:
        shapes = out[:, :126].reshape(T, 2, 21, 3)
        if noise_shape is None:
            noise_shape = np.random.normal(0.0, sigma, size=shapes.shape).astype(np.float32)
        shapes = np.where(mask[:, :, None, None], shapes + noise_shape, shapes)
        out[:, :126] = shapes.reshape(T, 126)

    # 2. Global wrists [126:132] -> (T, 2, 3)
    if D >= 132:
        wrists = out[:, 126:132].reshape(T, 2, 3)
        if noise_wrist is None:
            noise_wrist = np.random.normal(0.0, sigma, size=wrists.shape).astype(np.float32)
        wrists = np.where(mask[:, :, None], wrists + noise_wrist, wrists)
        out[:, 126:132] = wrists.reshape(T, 6)

    return out


def random_rotate_2d(
    features: np.ndarray,
    max_angle_deg: float = 8.0,
    angle_deg: Optional[float] = None,
) -> np.ndarray:
    """Xoay nhẹ các điểm trong mặt phẳng (x, y).

    Wrists xoay quanh (0.5, 0.5) CHỈ cho tay có mặt (mask > 0.5).
    Không recompute velocity.
    """
    if angle_deg is None:
        angle_deg = float(np.random.uniform(-max_angle_deg, max_angle_deg))
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    out = features.copy()
    T, D = out.shape
    mask = out[:, 141:143] > 0.5 if D >= 143 else np.ones((T, 2), dtype=bool)

    # 1. Xoay local shapes [0:126] -> (T, 2 tay, 21 điểm, 3) quanh gốc (0, 0)
    if D >= 126:
        shapes = out[:, :126].reshape(T, 2, 21, 3)
        xy = shapes[:, :, :, :2].reshape(-1, 2)
        xy_rot = np.dot(xy, rot_matrix.T).reshape(T, 2, 21, 2)
        shapes[:, :, :, :2] = np.where(mask[:, :, None, None], xy_rot, shapes[:, :, :, :2])
        out[:, :126] = shapes.reshape(T, 126)

    # 2. Xoay global wrists [126:132] -> (T, 2 tay, 3) quanh tâm (0.5, 0.5) CHỈ cho tay có mặt
    if D >= 132:
        wrists = out[:, 126:132].reshape(T, 2, 3)
        w_xy = wrists[:, :, :2].reshape(-1, 2) - 0.5
        w_xy_rot = (np.dot(w_xy, rot_matrix.T) + 0.5).reshape(T, 2, 2)
        wrists[:, :, :2] = np.where(mask[:, :, None], w_xy_rot, wrists[:, :, :2])
        out[:, 126:132] = wrists.reshape(T, 6)

    return out


def horizontal_flip(features: np.ndarray, swap_hands: bool = False) -> np.ndarray:
    """Lật gương quanh trục x.

    Args:
        features: Mảng shape (T, D).
        swap_hands: Nếu False, chỉ lật trục x và giữ nguyên slot 0 là tay trội.
                    Nếu True, hoán đổi slot tay 0 <-> tay 1.
    """
    out = features.copy()
    T, D = out.shape

    mask = out[:, 141:143] > 0.5 if D >= 143 else np.ones((T, 2), dtype=bool)

    # 1. Local Shapes (126 dims: 63 dims per hand)
    if D >= 126:
        shapes = out[:, :126].reshape(T, 2, 21, 3)
        shapes[:, :, :, 0] = np.where(mask[:, :, None], -shapes[:, :, :, 0], shapes[:, :, :, 0])
        if swap_hands:
            shapes = shapes[:, [1, 0], :, :]
        out[:, :126] = shapes.reshape(T, 126)

    # 2. Global Wrists (6 dims: 3 per hand)
    if D >= 132:
        wrists = out[:, 126:132].reshape(T, 2, 3)
        wrists[:, :, 0] = np.where(mask, 1.0 - wrists[:, :, 0], wrists[:, :, 0])
        if swap_hands:
            wrists = wrists[:, [1, 0], :]
        out[:, 126:132] = wrists.reshape(T, 6)

    # 3. Mask (2 dims)
    if D >= 143 and swap_hands:
        out[:, 141:143] = out[:, [142, 141]]

    # 4. Recompute derived features (rel [132:135] & vel [135:141])
    if D >= 141:
        out = recompute_derived_features(out)

    # 5. Pose shoulder & elbow points if present (12 coords: left/right shoulder, left/right elbow)
    if D >= 155:
        pose = out[:, 143:155].reshape(T, 4, 3)
        pose[:, :, 0] = 1.0 - pose[:, :, 0]
        pose_flipped = pose[:, [1, 0, 3, 2], :]
        out[:, 143:155] = pose_flipped.reshape(T, 12)

    return out


def shape_scale(
    features: np.ndarray,
    r: float = 0.10,
    sx: Optional[float] = None,
    sy: Optional[float] = None,
) -> np.ndarray:
    """Co giãn độc lập trục x, y của local hand shapes [0:126] cho tay có mặt.

    sx, sy ~ U(1 - r, 1 + r).
    """
    out = features.copy()
    T, D = out.shape
    if D < 126 or r <= 0.0:
        return out

    if sx is None:
        sx = float(np.random.uniform(1.0 - r, 1.0 + r))
    if sy is None:
        sy = float(np.random.uniform(1.0 - r, 1.0 + r))

    mask = out[:, 141:143] > 0.5 if D >= 143 else np.ones((T, 2), dtype=bool)

    shapes = out[:, :126].reshape(T, 2, 21, 3)
    sh_x = shapes[:, :, :, 0] * sx
    sh_y = shapes[:, :, :, 1] * sy
    shapes[:, :, :, 0] = np.where(mask[:, :, None], sh_x, shapes[:, :, :, 0])
    shapes[:, :, :, 1] = np.where(mask[:, :, None], sh_y, shapes[:, :, :, 1])
    out[:, :126] = shapes.reshape(T, 126)

    return out


def scale_translate(
    features: np.ndarray,
    scale_min: float = 0.85,
    scale_max: float = 1.15,
    translate: float = 0.10,
    s: Optional[float] = None,
    tx: Optional[float] = None,
    ty: Optional[float] = None,
) -> np.ndarray:
    """Co giãn và dịch chuyển cổ tay tuyệt đối (xy) cho tay có mặt, dùng chung cho cả chuỗi.

    wrist_xy' = (wrist_xy - 0.5) * s + 0.5 + (tx, ty)
    z giữ nguyên. Sau đó recompute_derived_features.
    """
    out = features.copy()
    T, D = out.shape
    if D < 132:
        return out

    if s is None:
        s = float(np.random.uniform(scale_min, scale_max))
    if tx is None:
        tx = float(np.random.uniform(-translate, translate))
    if ty is None:
        ty = float(np.random.uniform(-translate, translate))

    mask = out[:, 141:143] > 0.5 if D >= 143 else np.ones((T, 2), dtype=bool)

    wrists = out[:, 126:132].reshape(T, 2, 3)
    w_x = (wrists[:, :, 0] - 0.5) * s + 0.5 + tx
    w_y = (wrists[:, :, 1] - 0.5) * s + 0.5 + ty
    wrists[:, :, 0] = np.where(mask, w_x, wrists[:, :, 0])
    wrists[:, :, 1] = np.where(mask, w_y, wrists[:, :, 1])
    out[:, 126:132] = wrists.reshape(T, 6)

    if D >= 141:
        out = recompute_derived_features(out)

    return out


def time_stretch(features: np.ndarray, max_stretch: float = 0.2) -> np.ndarray:
    """Co giãn trục thời gian của chuỗi cử chỉ."""
    T, D = features.shape
    if T <= 2 or max_stretch <= 0.0:
        return features.copy()

    rate = np.random.uniform(1.0 - max_stretch, 1.0 + max_stretch)
    t_orig = np.linspace(0.0, 1.0, T)
    stretched_len = max(2, int(round(T * rate)))
    stretched_t = np.linspace(0.0, 1.0, stretched_len)

    out = np.empty((T, D), dtype=features.dtype)
    for d in range(D):
        vals = np.interp(stretched_t, t_orig, features[:, d])
        out[:, d] = np.interp(t_orig, np.linspace(0.0, 1.0, len(vals)), vals)

    return out


def _get_aug_param(cfg_dict: Dict[str, Any], key: str, default: Any) -> Any:
    val = cfg_dict.get(key)
    return val if val is not None else default


def apply_augmentation(features: np.ndarray, cfg: Any = None) -> np.ndarray:
    """Áp dụng các phép biến đổi dữ liệu theo cấu hình AugmentationConfig hoặc dict.

    Thứ tự áp dụng chuẩn: flip -> rotate -> shape_scale -> scale_translate -> jitter.
    """
    if cfg is None:
        return apply_random_augmentation(features)

    # Extract dictionary representations
    flip_cfg = getattr(cfg, "flip", {}) if not isinstance(cfg, dict) else cfg.get("flip", {})
    rotate_cfg = getattr(cfg, "rotate", {}) if not isinstance(cfg, dict) else cfg.get("rotate", {})
    shape_scale_cfg = getattr(cfg, "shape_scale", {}) if not isinstance(cfg, dict) else cfg.get("shape_scale", {})
    scale_translate_cfg = getattr(cfg, "scale_translate", {}) if not isinstance(cfg, dict) else cfg.get("scale_translate", {})
    jitter_cfg = getattr(cfg, "jitter", {}) if not isinstance(cfg, dict) else cfg.get("jitter", {})
    stretch_cfg = getattr(cfg, "time_stretch", {}) if not isinstance(cfg, dict) else cfg.get("time_stretch", {})

    out = features.copy()

    # 1. Flip
    if _get_aug_param(flip_cfg, "enabled", False):
        prob = float(_get_aug_param(flip_cfg, "prob", 0.5))
        swap_hands = bool(_get_aug_param(flip_cfg, "swap_hands", False))
        if np.random.rand() < prob:
            out = horizontal_flip(out, swap_hands=swap_hands)

    # 2. Rotate
    if _get_aug_param(rotate_cfg, "enabled", False):
        prob = float(_get_aug_param(rotate_cfg, "prob", 0.5))
        max_angle = float(_get_aug_param(rotate_cfg, "max_angle_deg", 8.0))
        if np.random.rand() < prob:
            out = random_rotate_2d(out, max_angle_deg=max_angle)

    # 3. Shape Scale
    if _get_aug_param(shape_scale_cfg, "enabled", False):
        prob = float(_get_aug_param(shape_scale_cfg, "prob", 0.5))
        r = float(_get_aug_param(shape_scale_cfg, "r", 0.10))
        if np.random.rand() < prob:
            out = shape_scale(out, r=r)

    # 4. Scale & Translate
    if _get_aug_param(scale_translate_cfg, "enabled", False):
        prob = float(_get_aug_param(scale_translate_cfg, "prob", 0.7))
        scale_min = float(_get_aug_param(scale_translate_cfg, "scale_min", 0.85))
        scale_max = float(_get_aug_param(scale_translate_cfg, "scale_max", 1.15))
        translate = float(_get_aug_param(scale_translate_cfg, "translate", 0.10))
        if np.random.rand() < prob:
            out = scale_translate(out, scale_min=scale_min, scale_max=scale_max, translate=translate)

    # 5. Jitter
    if _get_aug_param(jitter_cfg, "enabled", False):
        prob = float(_get_aug_param(jitter_cfg, "prob", 0.5))
        sigma = float(_get_aug_param(jitter_cfg, "sigma", 0.01))
        if np.random.rand() < prob:
            out = add_jitter(out, sigma=sigma)

    # 6. Time Stretch
    if _get_aug_param(stretch_cfg, "enabled", False):
        prob = float(_get_aug_param(stretch_cfg, "prob", 0.3))
        max_stretch = float(_get_aug_param(stretch_cfg, "max_stretch", 0.2))
        if np.random.rand() < prob:
            out = time_stretch(out, max_stretch=max_stretch)

    return out


def apply_random_augmentation(features: np.ndarray) -> np.ndarray:
    """Áp dụng ngẫu nhiên một tổ hợp các phép biến đổi mặc định (backwards compatibility)."""
    default_cfg = {
        "flip": {"enabled": True, "prob": 0.5, "swap_hands": False},
        "rotate": {"enabled": True, "prob": 0.5, "max_angle_deg": 8.0},
        "shape_scale": {"enabled": True, "prob": 0.5, "r": 0.10},
        "scale_translate": {"enabled": True, "prob": 0.7, "scale_min": 0.85, "scale_max": 1.15, "translate": 0.10},
        "jitter": {"enabled": True, "prob": 0.5, "sigma": 0.01},
        "time_stretch": {"enabled": False, "prob": 0.0, "max_stretch": 0.2},
    }
    return apply_augmentation(features, cfg=default_cfg)
