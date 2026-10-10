"""Module chuẩn hóa không gian, thời gian và trích xuất vector đặc trưng 143 chiều.

Thiết kế đặc trưng bảo toàn thông tin toàn diện:
1. Cắt đoạn nghỉ đầu/cuối (Trim idle frames).
2. Điền frame mất tay bằng nội suy giữa các frame hợp lệ lân cận (tránh nội suy vào zero).
3. Resample thời gian về T = 32 frames (tọa độ nội suy tuyến tính, mask dùng nearest neighbor).
4. Vector đặc trưng 143 chiều per frame:
   - Local Hand Shape (126 dims): Dời về cổ tay, chia cho palm size.
   - Global Wrist Coords (6 dims): Tọa độ cổ tay toàn cục trong khung hình.
   - Relative Wrist Vector (3 dims): Vector tương đối wrist_1 - wrist_2.
   - Wrist Velocities (6 dims): Vận tốc chuyển động cổ tay delta giữa các frame.
   - Hand Mask (2 dims): Cờ boolean biểu thị frame có tay thật sự.
"""
from __future__ import annotations

import numpy as np


def trim_idle_frames(
    raw_coords: np.ndarray,
    hand_masks: np.ndarray,
    margin: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Cắt bỏ các frame tay đứng yên/nghỉ ở đầu và cuối video."""
    T = len(raw_coords)
    if T == 0:
        return raw_coords, hand_masks

    has_hand = hand_masks.any(axis=1)
    if not has_hand.any():
        return raw_coords, hand_masks

    valid_idx = np.where(has_hand)[0]
    start = max(0, valid_idx[0] - margin)
    end = min(T, valid_idx[-1] + margin + 1)
    return raw_coords[start:end], hand_masks[start:end]


def interpolate_missing_hands(
    raw_coords: np.ndarray,
    hand_masks: np.ndarray
) -> np.ndarray:
    """Nội suy tọa độ các frame mất tay ngắn hạn từ các frame hợp lệ lân cận."""
    T, num_hands, num_lms, dims = raw_coords.shape
    filled_coords = raw_coords.copy()

    for h in range(num_hands):
        valid_t = np.where(hand_masks[:, h])[0]
        # Nếu tay này hoàn toàn không xuất hiện trong cả video -> giữ 0
        if len(valid_t) == 0:
            continue
        # Nếu chỉ có 1 frame hợp lệ -> gán frame đó cho mọi vị trí
        if len(valid_t) == 1:
            filled_coords[:, h] = filled_coords[valid_t[0], h]
            continue

        # Nội suy tuyến tính từng landmark (x, y, z) qua trục thời gian
        for lm in range(num_lms):
            for d in range(dims):
                xp = valid_t
                fp = raw_coords[valid_t, h, lm, d]
                # np.interp tự động nội suy ở giữa và ngoại suy bằng constant ở 2 đầu
                filled_coords[:, h, lm, d] = np.interp(np.arange(T), xp, fp)

    return filled_coords


def resample_temporal(
    coords: np.ndarray,
    masks: np.ndarray,
    target_len: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Resample chuỗi thời gian về độ dài cố định T=32 frames."""
    T = len(coords)
    if T == 0:
        return np.zeros((target_len, *coords.shape[1:]), dtype=np.float32), np.zeros((target_len, masks.shape[1]), dtype=bool)

    if T == target_len:
        return coords.astype(np.float32), masks.astype(bool)

    orig_t = np.linspace(0.0, 1.0, T)
    target_t = np.linspace(0.0, 1.0, target_len)

    # 1. Nội suy tọa độ liên tục bằng Linear Interpolation
    flat_coords = coords.reshape(T, -1)
    resampled_flat = np.zeros((target_len, flat_coords.shape[1]), dtype=np.float32)
    for i in range(flat_coords.shape[1]):
        resampled_flat[:, i] = np.interp(target_t, orig_t, flat_coords[:, i])
    resampled_coords = resampled_flat.reshape(target_len, *coords.shape[1:])

    # 2. Resample Mask bằng Nearest Neighbor (bảo toàn giá trị boolean)
    nearest_idx = np.searchsorted(orig_t, target_t)
    nearest_idx = np.clip(nearest_idx, 0, T - 1)
    resampled_masks = masks[nearest_idx]

    return resampled_coords.astype(np.float32), resampled_masks.astype(bool)


def compute_feature_vector_143(
    coords: np.ndarray,
    masks: np.ndarray
) -> np.ndarray:
    """Biến đổi tọa độ (T, 2, 21, 3) thành ma trận đặc trưng (T, 143).

    Cấu trúc 143 đặc trưng:
    - [0:126]: Local Hand Shapes (21 x 3 x 2 tay) sau khi dời cổ tay và chia palm size
    - [126:132]: Global Wrist Coordinates (x, y, z x 2 tay)
    - [132:135]: Relative Wrist Vector (wrist_1 - wrist_2)
    - [135:141]: Wrist Velocities (delta wrist giữa 2 frame liên tiếp x 2 tay)
    - [141:143]: Hand Presence Masks (has_hand1, has_hand2)
    """
    T = len(coords)
    features = np.zeros((T, 143), dtype=np.float32)

    # Lưu tọa độ cổ tay toàn cục (global wrist coords)
    global_wrists = coords[:, :, 0, :].copy()  # shape (T, 2, 3)

    # 1. Local Hand Shapes (126 dims)
    local_shapes = np.zeros((T, 2, 21, 3), dtype=np.float32)
    for h in range(2):
        for t in range(T):
            if masks[t, h]:
                hand_pts = coords[t, h].copy()  # (21, 3)
                wrist = hand_pts[0].copy()
                hand_centered = hand_pts - wrist
                # Palm size = khoảng cách từ cổ tay (0) đến gốc ngón giữa (9)
                palm_size = float(np.linalg.norm(hand_centered[9]))
                if palm_size > 1e-4:
                    hand_centered /= palm_size
                local_shapes[t, h] = hand_centered

    features[:, :126] = local_shapes.reshape(T, 126)

    # 2. Global Wrist Coordinates (6 dims: [126:132])
    features[:, 126:132] = global_wrists.reshape(T, 6)

    # 3. Relative Wrist Vector (3 dims: [132:135])
    rel_wrist = global_wrists[:, 0, :] - global_wrists[:, 1, :]
    # Chỉ giữ vector tương đối nếu cả 2 tay đều xuất hiện
    two_hand_mask = (masks[:, 0] & masks[:, 1])[:, np.newaxis]
    features[:, 132:135] = np.where(two_hand_mask, rel_wrist, 0.0)

    # 4. Wrist Velocities (6 dims: [135:141])
    wrist_vel = np.zeros((T, 2, 3), dtype=np.float32)
    if T > 1:
        wrist_vel[1:] = global_wrists[1:] - global_wrists[:-1]
    features[:, 135:141] = wrist_vel.reshape(T, 6)

    # 5. Hand Presence Mask (2 dims: [141:143])
    features[:, 141:143] = masks.astype(np.float32)

    return features


def process_video_landmarks(
    raw_coords: np.ndarray,
    hand_masks: np.ndarray,
    target_len: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Pipeline chuẩn hóa hoàn chỉnh: Trim -> Interpolate -> Resample -> 143-dim Features."""
    # Bước 1: Cắt đoạn nghỉ
    trimmed_coords, trimmed_masks = trim_idle_frames(raw_coords, hand_masks)
    if len(trimmed_coords) == 0:
        return np.zeros((target_len, 143), dtype=np.float32), np.zeros((target_len, 2), dtype=bool)

    # Bước 2: Điền frame mất tay bằng nội suy giữa các frame có tay
    filled_coords = interpolate_missing_hands(trimmed_coords, trimmed_masks)

    # Bước 3: Resample thời gian về target_len = 32
    resampled_coords, resampled_masks = resample_temporal(filled_coords, trimmed_masks, target_len=target_len)

    # Bước 4: Trích xuất vector đặc trưng 143 chiều
    features_143 = compute_feature_vector_143(resampled_coords, resampled_masks)

    return features_143, resampled_masks
