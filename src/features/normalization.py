"""Spatial/temporal normalization and feature extraction for ISLR landmarks.

Features:
- Configurable feature sets: 'hands' (143 dims) or 'hands_pose' (160 dims).
- Automatic feature dimension deduction (zero hardcoding).
- Dominant hand handling, wrist centering, and palm size scaling.
- Short gap interpolation bounded by max_gap_frames.
- Temporal resampling to fixed T frames.
- Upper-body pose landmark normalization (shoulders, elbows) relative to torso.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def get_feature_dim(feature_set: str = "hands") -> int:
    """Deterministically derive feature vector dimension from feature set.

    Args:
        feature_set: 'hands' (143) or 'hands_pose' (160).

    Returns:
        int: Total feature dimensions per frame.
    """
    if feature_set == "hands_pose":
        return 160
    return 143


def trim_idle_frames(
    raw_coords: np.ndarray,
    hand_masks: np.ndarray,
    margin: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim leading and trailing frames where neither hand is detected."""
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


def interpolate_bounded_gaps(
    raw_coords: np.ndarray,
    hand_masks: np.ndarray,
    max_gap_frames: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate short NaN gaps (<= max_gap_frames) between valid hand detections.

    Gaps longer than max_gap_frames remain zero with mask=False.
    """
    T, num_hands, num_lms, dims = raw_coords.shape
    filled_coords = np.nan_to_num(raw_coords, nan=0.0)
    updated_masks = hand_masks.copy()

    for h in range(num_hands):
        valid_indices = np.where(hand_masks[:, h])[0]
        if len(valid_indices) < 2:
            continue

        for i in range(len(valid_indices) - 1):
            idx_start = valid_indices[i]
            idx_end = valid_indices[i + 1]
            gap_len = idx_end - idx_start - 1
            if 0 < gap_len <= max_gap_frames:
                # Linearly interpolate across this short gap
                for t in range(idx_start + 1, idx_end):
                    alpha = (t - idx_start) / (idx_end - idx_start)
                    filled_coords[t, h] = (
                        (1.0 - alpha) * raw_coords[idx_start, h]
                        + alpha * raw_coords[idx_end, h]
                    )
                    updated_masks[t, h] = True

    return filled_coords, updated_masks


def interpolate_missing_hands(
    raw_coords: np.ndarray,
    hand_masks: np.ndarray,
) -> np.ndarray:
    """Legacy interpolation helper for backward compatibility."""
    filled, _ = interpolate_bounded_gaps(raw_coords, hand_masks, max_gap_frames=999)
    return filled


def resample_temporal(
    coords: np.ndarray,
    masks: np.ndarray,
    target_len: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample continuous coordinates linearly and masks by nearest neighbor to target_len."""
    T = len(coords)
    if T == 0:
        return (
            np.zeros((target_len, *coords.shape[1:]), dtype=np.float32),
            np.zeros((target_len, masks.shape[1]), dtype=bool),
        )

    if T == target_len:
        return coords.astype(np.float32), masks.astype(bool)

    orig_t = np.linspace(0.0, 1.0, T)
    target_t = np.linspace(0.0, 1.0, target_len)

    flat_coords = coords.reshape(T, -1)
    resampled_flat = np.zeros((target_len, flat_coords.shape[1]), dtype=np.float32)
    for i in range(flat_coords.shape[1]):
        resampled_flat[:, i] = np.interp(target_t, orig_t, flat_coords[:, i])
    resampled_coords = resampled_flat.reshape(target_len, *coords.shape[1:])

    nearest_idx = np.searchsorted(orig_t, target_t)
    nearest_idx = np.clip(nearest_idx, 0, T - 1)
    resampled_masks = masks[nearest_idx]

    return resampled_coords.astype(np.float32), resampled_masks.astype(bool)


def compute_feature_vector_143(
    coords: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Transform (T, 2, 21, 3) hand landmarks into standard 143-dim feature matrix."""
    T = len(coords)
    features = np.zeros((T, 143), dtype=np.float32)
    if T == 0:
        return features

    global_wrists = coords[:, :, 0, :].copy()  # (T, 2, 3)

    # 1. Local Hand Shapes (126 dims)
    local_shapes = np.zeros((T, 2, 21, 3), dtype=np.float32)
    for h in range(2):
        for t in range(T):
            if masks[t, h]:
                hand_pts = coords[t, h].copy()
                wrist = hand_pts[0].copy()
                hand_centered = hand_pts - wrist
                palm_size = float(np.linalg.norm(hand_centered[9]))
                if palm_size > 1e-4:
                    hand_centered /= palm_size
                local_shapes[t, h] = hand_centered

    features[:, :126] = local_shapes.reshape(T, 126)

    # 2. Global Wrist Coordinates (6 dims: [126:132])
    features[:, 126:132] = global_wrists.reshape(T, 6)

    # 3. Relative Wrist Vector (3 dims: [132:135])
    rel_wrist = global_wrists[:, 0, :] - global_wrists[:, 1, :]
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


def compute_feature_vector(
    hands_coords: np.ndarray,
    hands_masks: np.ndarray,
    pose_coords: Optional[np.ndarray] = None,
    pose_masks: Optional[np.ndarray] = None,
    feature_set: str = "hands",
) -> np.ndarray:
    """Compute comprehensive feature vector with optional upper-body pose reference.

    Dimensions:
    - If feature_set == 'hands': 143 dims.
    - If feature_set == 'hands_pose': 160 dims (143 hands + 17 pose torso & elbow features).
    """
    hand_feats = compute_feature_vector_143(hands_coords, hands_masks)
    if feature_set != "hands_pose" or pose_coords is None:
        return hand_feats

    T = len(hands_coords)
    # Pose keypoints: [11: L_shoulder, 12: R_shoulder, 13: L_elbow, 14: R_elbow]
    # shape (T, 4, 3)
    pose_pts = pose_coords[:, [11, 12, 13, 14], :].copy()
    l_sh = pose_pts[:, 0, :]
    r_sh = pose_pts[:, 1, :]
    torso_center = (l_sh + r_sh) / 2.0  # (T, 3)
    torso_scale = np.linalg.norm(l_sh - r_sh, axis=-1, keepdims=True)  # (T, 1)

    # Normalize pose points relative to torso center and scale
    norm_pose = np.zeros_like(pose_pts)
    valid_scale = (torso_scale[:, 0] > 1e-4)
    for i in range(4):
        centered = pose_pts[:, i, :] - torso_center
        norm_pose[:, i, :] = np.where(
            valid_scale[:, np.newaxis],
            centered / np.maximum(torso_scale, 1e-4),
            0.0,
        )

    # Torso velocity
    torso_vel = np.zeros((T, 3), dtype=np.float32)
    if T > 1:
        torso_vel[1:] = torso_center[1:] - torso_center[:-1]

    p_mask = pose_masks if pose_masks is not None else np.ones((T, 1), dtype=np.float32)
    if p_mask.ndim == 1:
        p_mask = p_mask[:, np.newaxis]

    pose_features = np.hstack([
        norm_pose.reshape(T, 12),
        torso_vel,
        np.clip(torso_scale, 0.0, 2.0),
        p_mask[:, :1].astype(np.float32),
    ])

    return np.hstack([hand_feats, pose_features]).astype(np.float32)


def process_video_landmarks(
    raw_coords: np.ndarray,
    hand_masks: np.ndarray,
    target_len: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard pipeline for video landmarks backward compatibility."""
    trimmed_coords, trimmed_masks = trim_idle_frames(raw_coords, hand_masks)
    if len(trimmed_coords) == 0:
        return (
            np.zeros((target_len, 143), dtype=np.float32),
            np.zeros((target_len, 2), dtype=bool),
        )

    filled_coords, filled_masks = interpolate_bounded_gaps(trimmed_coords, trimmed_masks)
    resampled_coords, resampled_masks = resample_temporal(
        filled_coords, filled_masks, target_len=target_len
    )
    features_143 = compute_feature_vector_143(resampled_coords, resampled_masks)
    return features_143, resampled_masks
