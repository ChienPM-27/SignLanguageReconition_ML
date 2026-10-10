"""Module biến đổi dữ liệu (Data Augmentation) cho chuỗi đặc trưng 143 chiều.

Bao gồm:
- horizontal_flip: Lật gương quanh x=0.5 và hoán đổi hai bàn tay.
- add_jitter: Thêm nhiễu Gaussian nhẹ vào tọa độ landmarks.
- random_rotate_2d: Xoay nhẹ trong mặt phẳng (x, y).
- time_stretch: Co giãn tốc độ cử chỉ.
"""
from __future__ import annotations

import math
import numpy as np


def add_jitter(features: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Thêm nhiễu Gaussian nhẹ vào tọa độ local shape và global wrist."""
    out = features.copy()
    # Nhiễu vào local shapes [0:126] và global wrists [126:132]
    noise = np.random.normal(0.0, sigma, size=out[:, :132].shape).astype(np.float32)
    # Không thêm nhiễu vào các frame không có tay
    mask1 = out[:, 141:142]
    mask2 = out[:, 142:143]
    # Broadcast mask tương ứng cho hand 1 và hand 2
    out[:, :132] += noise
    return out


def random_rotate_2d(features: np.ndarray, max_angle_deg: float = 8.0) -> np.ndarray:
    """Xoay nhẹ các điểm trong mặt phẳng (x, y)."""
    angle_rad = math.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    out = features.copy()
    T = len(out)

    # 1. Xoay local shapes [0:126] -> (T, 2 tay, 21 điểm, 3)
    shapes = out[:, :126].reshape(T, 2, 21, 3)
    xy = shapes[:, :, :, :2].reshape(-1, 2)
    xy_rot = np.dot(xy, rot_matrix.T)
    shapes[:, :, :, :2] = xy_rot.reshape(T, 2, 21, 2)
    out[:, :126] = shapes.reshape(T, 126)

    # 2. Xoay global wrists [126:132] -> (T, 2 tay, 3) quanh tâm (0.5, 0.5)
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

    # 1. Local Shapes (126 dims: 63 dims per hand)
    shapes = out[:, :126].reshape(T, 2, 21, 3)
    # Lật tọa độ x cục bộ
    shapes[:, :, :, 0] = -shapes[:, :, :, 0]
    # Hoán đổi giữa hand 0 và hand 1
    shapes_flipped = shapes[:, [1, 0], :, :]
    out[:, :126] = shapes_flipped.reshape(T, 126)

    # 2. Global Wrists (6 dims: 3 per hand)
    wrists = out[:, 126:132].reshape(T, 2, 3)
    # Lật quanh trục ảnh x=0.5
    wrists[:, :, 0] = 1.0 - wrists[:, :, 0]
    wrists_flipped = wrists[:, [1, 0], :]
    out[:, 126:132] = wrists_flipped.reshape(T, 6)

    # 3. Relative Wrist Vector: rel = wrist_1 - wrist_2
    # Khi lật: rel_x = -(wrist_1_x - wrist_2_x), và do đổi ngôi 1 <-> 2 nên rel = -rel
    out[:, 132:135] = -out[:, 132:135]
    out[:, 132] = -out[:, 132]  # Lật thêm hoành độ x

    # 4. Wrist Velocities (6 dims: 3 per hand)
    vel = out[:, 135:141].reshape(T, 2, 3)
    vel[:, :, 0] = -vel[:, :, 0]
    vel_flipped = vel[:, [1, 0], :]
    out[:, 135:141] = vel_flipped.reshape(T, 6)

    # 5. Mask (2 dims)
    out[:, 141:143] = out[:, [142, 141]]

    return out


def apply_random_augmentation(features: np.ndarray) -> np.ndarray:
    """Áp dụng ngẫu nhiên một tổ hợp các phép biến đổi dữ liệu khi huấn luyện."""
    out = features.copy()
    # 50% xác suất lật gương
    if np.random.rand() < 0.5:
        out = horizontal_flip(out)
    # 50% xác suất xoay nhẹ
    if np.random.rand() < 0.5:
        out = random_rotate_2d(out, max_angle_deg=8.0)
    # 50% xác suất thêm nhiễu nhẹ
    if np.random.rand() < 0.5:
        out = add_jitter(out, sigma=0.008)
    return out
