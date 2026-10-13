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
from typing import Any, Dict
import numpy as np


def add_jitter(features: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Thêm nhiễu Gaussian nhẹ vào tọa độ landmarks."""
    out = features.copy()
    # Nhiễu vào local shapes [0:126] và global wrists [126:132]
    coord_dim = min(132, out.shape[1])
    noise = np.random.normal(0.0, sigma, size=out[:, :coord_dim].shape).astype(np.float32)
    out[:, :coord_dim] += noise
    return out


def random_rotate_2d(features: np.ndarray, max_angle_deg: float = 8.0) -> np.ndarray:
    """Xoay nhẹ các điểm trong mặt phẳng (x, y)."""
    angle_rad = math.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    out = features.copy()
    T = len(out)
    D = out.shape[1]

    # 1. Xoay local shapes [0:126] -> (T, 2 tay, 21 điểm, 3)
    if D >= 126:
        shapes = out[:, :126].reshape(T, 2, 21, 3)
        xy = shapes[:, :, :, :2].reshape(-1, 2)
        xy_rot = np.dot(xy, rot_matrix.T)
        shapes[:, :, :, :2] = xy_rot.reshape(T, 2, 21, 2)
        out[:, :126] = shapes.reshape(T, 126)

    # 2. Xoay global wrists [126:132] -> (T, 2 tay, 3) quanh tâm (0.5, 0.5)
    if D >= 132:
        wrists = out[:, 126:132].reshape(T, 2, 3)
        w_xy = wrists[:, :, :2].reshape(-1, 2) - 0.5
        w_xy_rot = np.dot(w_xy, rot_matrix.T) + 0.5
        wrists[:, :, :2] = w_xy_rot.reshape(T, 2, 2)
        out[:, 126:132] = wrists.reshape(T, 6)

    return out


def horizontal_flip(features: np.ndarray) -> np.ndarray:
    """Lật gương quanh trục x=0.5 và hoán đổi tay 1 <-> tay 2."""
    out = features.copy()
    T = len(out)
    D = out.shape[1]

    # 1. Local Shapes (126 dims: 63 dims per hand)
    if D >= 126:
        shapes = out[:, :126].reshape(T, 2, 21, 3)
        shapes[:, :, :, 0] = -shapes[:, :, :, 0]
        shapes_flipped = shapes[:, [1, 0], :, :]
        out[:, :126] = shapes_flipped.reshape(T, 126)

    # 2. Global Wrists (6 dims: 3 per hand)
    if D >= 132:
        wrists = out[:, 126:132].reshape(T, 2, 3)
        wrists[:, :, 0] = 1.0 - wrists[:, :, 0]
        wrists_flipped = wrists[:, [1, 0], :]
        out[:, 126:132] = wrists_flipped.reshape(T, 6)

    # 3. Relative Wrist Vector: rel = wrist_1 - wrist_2
    if D >= 135:
        out[:, 132:135] = -out[:, 132:135]
        out[:, 132] = -out[:, 132]

    # 4. Wrist Velocities (6 dims: 3 per hand)
    if D >= 141:
        vel = out[:, 135:141].reshape(T, 2, 3)
        vel[:, :, 0] = -vel[:, :, 0]
        vel_flipped = vel[:, [1, 0], :]
        out[:, 135:141] = vel_flipped.reshape(T, 6)

    # 5. Mask (2 dims)
    if D >= 143:
        out[:, 141:143] = out[:, [142, 141]]

    # 6. Pose shoulder & elbow points if present (12 coords: left/right shoulder, left/right elbow)
    if D >= 155:
        pose = out[:, 143:155].reshape(T, 4, 3)
        pose[:, :, 0] = 1.0 - pose[:, :, 0]
        # Swap left shoulder (0) with right shoulder (1), and left elbow (2) with right elbow (3)
        pose_flipped = pose[:, [1, 0, 3, 2], :]
        out[:, 143:155] = pose_flipped.reshape(T, 12)

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
    """Áp dụng các phép biến đổi dữ liệu theo cấu hình AugmentationConfig hoặc dict."""
    if cfg is None:
        return apply_random_augmentation(features)

    # Extract dictionary representations
    flip_cfg = getattr(cfg, "flip", {}) if not isinstance(cfg, dict) else cfg.get("flip", {})
    jitter_cfg = getattr(cfg, "jitter", {}) if not isinstance(cfg, dict) else cfg.get("jitter", {})
    rotate_cfg = getattr(cfg, "rotate", {}) if not isinstance(cfg, dict) else cfg.get("rotate", {})
    stretch_cfg = getattr(cfg, "time_stretch", {}) if not isinstance(cfg, dict) else cfg.get("time_stretch", {})

    out = features.copy()

    # 1. Flip
    if _get_aug_param(flip_cfg, "enabled", False):
        prob = _get_aug_param(flip_cfg, "prob", 0.5)
        if np.random.rand() < prob:
            out = horizontal_flip(out)

    # 2. Rotate
    if _get_aug_param(rotate_cfg, "enabled", False):
        prob = _get_aug_param(rotate_cfg, "prob", 0.5)
        max_angle = _get_aug_param(rotate_cfg, "max_angle_deg", 8.0)
        if np.random.rand() < prob:
            out = random_rotate_2d(out, max_angle_deg=max_angle)

    # 3. Jitter
    if _get_aug_param(jitter_cfg, "enabled", False):
        prob = _get_aug_param(jitter_cfg, "prob", 0.5)
        sigma = _get_aug_param(jitter_cfg, "sigma", 0.01)
        if np.random.rand() < prob:
            out = add_jitter(out, sigma=sigma)

    # 4. Time Stretch
    if _get_aug_param(stretch_cfg, "enabled", False):
        prob = _get_aug_param(stretch_cfg, "prob", 0.3)
        max_stretch = _get_aug_param(stretch_cfg, "max_stretch", 0.2)
        if np.random.rand() < prob:
            out = time_stretch(out, max_stretch=max_stretch)

    return out


def apply_random_augmentation(features: np.ndarray) -> np.ndarray:
    """Áp dụng ngẫu nhiên một tổ hợp các phép biến đổi mặc định (backwards compatibility)."""
    default_cfg = {
        "flip": {"enabled": True, "prob": 0.5},
        "rotate": {"enabled": True, "prob": 0.5, "max_angle_deg": 8.0},
        "jitter": {"enabled": True, "prob": 0.5, "sigma": 0.008},
        "time_stretch": {"enabled": False, "prob": 0.0, "max_stretch": 0.2},
    }
    return apply_augmentation(features, cfg=default_cfg)
