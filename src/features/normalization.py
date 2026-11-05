"""Spatial/temporal normalization and feature extraction for ISLR landmarks.

Features:
- Configurable feature sets: 'hands' (143 dims) or 'hands_pose' (160 dims).
- Schema-derived dimension deduction (zero hardcoding).
- Dominant hand handling, wrist centering, and palm size scaling.
- Bounded short-gap interpolation (<= max_gap_frames), leaving long/unbounded gaps zeroed.
- Temporal resampling to fixed T frames using explicit nearest-neighbor support tie rules.
- Upper-body pose landmark normalization (shoulders, elbows) relative to torso center and scale.
- End-point gated velocity and relative motion on valid observation support.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from src.features.schema import (
    FEATURE_DIM_HANDS,
    FEATURE_DIM_HANDS_POSE,
    FEATURE_SET_HANDS,
    FEATURE_SET_HANDS_POSE,
    get_feature_dim,
)

__all__ = [
    "FEATURE_DIM_HANDS",
    "FEATURE_DIM_HANDS_POSE",
    "FEATURE_SET_HANDS",
    "FEATURE_SET_HANDS_POSE",
    "get_feature_dim",
    "get_trim_indices",
    "trim_idle_frames",
    "interpolate_bounded_gaps",
    "interpolate_missing_hands",
    "resample_temporal",
    "compute_feature_vector_143",
    "compute_feature_vector",
    "process_video_landmarks",
]


def get_trim_indices(
    hand_masks: np.ndarray,
    margin: int = 2,
) -> Tuple[int, int]:
    """Get start and end frame indices where hands are active.

    Args:
        hand_masks: Array of shape (T, 2) indicating active hands.
        margin: Number of padding frames around the active hand window.

    Returns:
        Tuple[int, int]: Half-open interval [start, end).
    """
    T = len(hand_masks)
    if T == 0:
        return 0, 0

    has_hand = hand_masks.any(axis=1)
    if not has_hand.any():
        return 0, T

    valid_idx = np.where(has_hand)[0]
    start = max(0, int(valid_idx[0]) - margin)
    end = min(T, int(valid_idx[-1]) + margin + 1)
    return start, end


def trim_idle_frames(
    raw_coords: np.ndarray,
    hand_masks: np.ndarray,
    margin: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim leading and trailing frames where neither hand is detected."""
    start, end = get_trim_indices(hand_masks, margin=margin)
    if end > start:
        return raw_coords[start:end], hand_masks[start:end]
    return raw_coords, hand_masks


def interpolate_bounded_gaps(
    coords: np.ndarray,
    masks: np.ndarray,
    max_gap_frames: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate short missing gaps (0 < gap <= max_gap_frames) between valid endpoints.

    Rules:
    - An observed endpoint requires mask=True and all coordinates in that slot to be finite.
    - Missing runs with length 0 < g <= max_gap_frames are linearly interpolated.
    - Leading gaps, trailing gaps, and gaps > max_gap_frames remain unavailable (mask=False, coords=0).
    - Coordinates are never interpolated toward or from an invalid/missing zero sentinel.
    - Zero-valued observed coordinates remain valid; NaN/Inf values are sanitized.

    Args:
        coords: Landmark coordinates of shape (T, num_slots, ...) or (T, ...).
        masks: Boolean support masks of shape (T, num_slots) or (T,).
        max_gap_frames: Maximum number of consecutive missing frames to fill.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (filled_coords, updated_masks).
    """
    T = len(coords)
    if T == 0:
        return np.zeros_like(coords, dtype=np.float32), np.zeros_like(masks, dtype=bool)

    # Standardize masks shape to (T, num_slots)
    orig_masks_ndim = masks.ndim
    if orig_masks_ndim == 1:
        masks_2d = masks[:, np.newaxis].astype(bool)
    else:
        masks_2d = masks.astype(bool)

    num_slots = masks_2d.shape[1]

    # Standardize coords shape to (T, num_slots, -1)
    orig_coords_shape = coords.shape
    coords_reshaped = coords.reshape(T, num_slots, -1)

    filled_reshaped = np.zeros_like(coords_reshaped, dtype=np.float32)
    updated_masks = np.zeros((T, num_slots), dtype=bool)

    for h in range(num_slots):
        # Identify valid observed endpoints: mask is True and all coordinates are finite
        slot_coords = coords_reshaped[:, h, :]
        slot_finite = np.all(np.isfinite(slot_coords), axis=1)
        slot_valid = masks_2d[:, h] & slot_finite

        valid_indices = np.where(slot_valid)[0]
        if len(valid_indices) == 0:
            continue

        # Valid observed endpoints are preserved
        for idx in valid_indices:
            filled_reshaped[idx, h] = slot_coords[idx]
            updated_masks[idx, h] = True

        if max_gap_frames <= 0 or len(valid_indices) < 2:
            continue

        # Linearly interpolate bounded interior gaps
        for i in range(len(valid_indices) - 1):
            idx_start = valid_indices[i]
            idx_end = valid_indices[i + 1]
            gap_len = idx_end - idx_start - 1

            if 0 < gap_len <= max_gap_frames:
                c_start = slot_coords[idx_start]
                c_end = slot_coords[idx_end]
                denom = float(idx_end - idx_start)
                for t in range(idx_start + 1, idx_end):
                    alpha = (t - idx_start) / denom
                    filled_reshaped[t, h] = (1.0 - alpha) * c_start + alpha * c_end
                    updated_masks[t, h] = True

    filled_coords = filled_reshaped.reshape(orig_coords_shape).astype(np.float32)
    if orig_masks_ndim == 1:
        out_masks = updated_masks[:, 0]
    else:
        out_masks = updated_masks

    return filled_coords, out_masks


def interpolate_missing_hands(
    raw_coords: np.ndarray,
    hand_masks: np.ndarray,
) -> np.ndarray:
    """Legacy helper for bounded interpolation."""
    filled, _ = interpolate_bounded_gaps(raw_coords, hand_masks, max_gap_frames=3)
    return filled


def resample_temporal(
    coords: np.ndarray,
    masks: np.ndarray,
    target_len: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample landmark coordinates and support masks to a fixed sequence length.

    Exact mathematical contract:
    - z[j] = j * (N - 1) / (L - 1) for j in 0..L-1
    - i0 = floor(z[j]), i1 = min(i0 + 1, N - 1), alpha = z[j] - i0
    - n[j] = i0 if alpha <= 0.5 else i1 (midpoint ties go to earlier frame)
    - m_out[j, h] = m[n[j], h]
    - If m[i0, h] and m[i1, h]: c_out[j, h] = (1 - alpha) * c[i0, h] + alpha * c[i1, h]
    - Elif m[n[j], h]: c_out[j, h] = c[n[j], h]
    - Else: c_out[j, h] = 0

    Args:
        coords: Coordinates of shape (N, ...).
        masks: Masks of shape (N, num_slots) or (N,).
        target_len: Target sequence length L (must be >= 1).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (resampled_coords, resampled_masks).
    """
    if target_len <= 0:
        raise ValueError(f"target_len must be a positive integer >= 1, got {target_len}")

    N = len(coords)
    orig_masks_ndim = masks.ndim
    if orig_masks_ndim == 1:
        masks_2d = masks[:, np.newaxis].astype(bool)
    else:
        masks_2d = masks.astype(bool)

    num_slots = masks_2d.shape[1]
    orig_coords_shape = coords.shape

    out_coords_shape = (target_len, *orig_coords_shape[1:])
    out_masks_shape = (target_len, num_slots)

    if N == 0:
        c_zero = np.zeros(out_coords_shape, dtype=np.float32)
        m_zero = np.zeros(target_len, dtype=bool) if orig_masks_ndim == 1 else np.zeros(out_masks_shape, dtype=bool)
        return c_zero, m_zero

    coords_reshaped = coords.reshape(N, num_slots, -1)
    resampled_flat = np.zeros((target_len, num_slots, coords_reshaped.shape[-1]), dtype=np.float32)
    resampled_masks = np.zeros((target_len, num_slots), dtype=bool)

    if target_len == 1:
        # Select first frame explicitly
        for h in range(num_slots):
            if masks_2d[0, h]:
                resampled_masks[0, h] = True
                resampled_flat[0, h] = coords_reshaped[0, h]
        out_c = resampled_flat.reshape(out_coords_shape)
        out_m = resampled_masks[:, 0] if orig_masks_ndim == 1 else resampled_masks
        return out_c, out_m

    if N == 1:
        # Repeat single source frame across target_len
        for h in range(num_slots):
            if masks_2d[0, h]:
                resampled_masks[:, h] = True
                resampled_flat[:, h] = coords_reshaped[0, h]
        out_c = resampled_flat.reshape(out_coords_shape)
        out_m = resampled_masks[:, 0] if orig_masks_ndim == 1 else resampled_masks
        return out_c, out_m

    # Uniform grid over source indices
    j_indices = np.arange(target_len, dtype=np.float64)
    z = j_indices * (N - 1) / (target_len - 1)
    i0 = np.floor(z).astype(int)
    i1 = np.minimum(i0 + 1, N - 1)
    alpha = z - i0
    # True nearest neighbor with tie-breaking to earlier frame
    n = np.where(alpha <= 0.5, i0, i1)

    alpha_expanded = alpha[:, np.newaxis]  # (target_len, 1)

    for h in range(num_slots):
        m_slot = masks_2d[:, h]
        c_slot = coords_reshaped[:, h, :]

        m_out_slot = m_slot[n]
        resampled_masks[:, h] = m_out_slot

        both_valid = m_slot[i0] & m_slot[i1]
        single_valid = m_out_slot & (~both_valid)

        c_both = (1.0 - alpha_expanded) * c_slot[i0] + alpha_expanded * c_slot[i1]
        c_single = c_slot[n]

        # Apply coordinates: blend if both endpoints valid, hold nearest if one valid, else 0
        c_interp = np.where(
            both_valid[:, np.newaxis],
            c_both,
            np.where(single_valid[:, np.newaxis], c_single, 0.0),
        )
        resampled_flat[:, h] = c_interp

    out_c = resampled_flat.reshape(out_coords_shape).astype(np.float32)
    out_m = resampled_masks[:, 0] if orig_masks_ndim == 1 else resampled_masks
    return out_c, out_m


def compute_feature_vector_143(
    coords: np.ndarray,
    masks: np.ndarray,
    normalization_epsilon: float = 1e-4,
) -> np.ndarray:
    """Transform (T, 2, 21, 3) hand landmarks into standard 143-dim feature matrix.

    Features:
    - [0:126]: Local hand shapes (2 hands * 21 points * 3 coords), centered on wrist,
               normalized by palm size (norm of landmark 9 from wrist). Landmark 0 is strictly 0.
    - [126:132]: Global wrist coordinates (2 hands * 3 coords), zero where unsupported.
    - [132:135]: Relative wrist vector w[0] - w[1], gated on both hands valid.
    - [135:141]: Wrist velocity w[t] - w[t-1], gated on both temporal endpoints valid.
    - [141:143]: Binary hand presence mask.
    """
    T = len(coords)
    features = np.zeros((T, 143), dtype=np.float32)
    if T == 0:
        return features

    m = masks.astype(bool)
    if m.ndim == 1:
        m = m[:, np.newaxis]

    # Global wrists (T, 2, 3), strictly zero for unsupported slots
    raw_wrists = coords[:, :, 0, :].copy()
    global_wrists = np.where(m[:, :, np.newaxis], raw_wrists, 0.0)

    # 1. Local Hand Shapes (126 dims)
    local_shapes = np.zeros((T, 2, 21, 3), dtype=np.float32)
    for h in range(2):
        valid_t = np.where(m[:, h])[0]
        for t in valid_t:
            hand_pts = coords[t, h].copy()
            wrist = hand_pts[0].copy()
            hand_centered = hand_pts - wrist
            # Landmark 0 is identically zero
            hand_centered[0] = 0.0
            palm_size = float(np.linalg.norm(hand_centered[9]))
            if palm_size > normalization_epsilon:
                hand_centered /= palm_size
            else:
                hand_centered = np.zeros_like(hand_centered)
            local_shapes[t, h] = hand_centered

    features[:, :126] = local_shapes.reshape(T, 126)

    # 2. Global Wrist Coordinates (6 dims: [126:132])
    features[:, 126:132] = global_wrists.reshape(T, 6)

    # 3. Relative Wrist Vector (3 dims: [132:135])
    two_hand_mask = (m[:, 0] & m[:, 1])[:, np.newaxis]
    rel_wrist = global_wrists[:, 0, :] - global_wrists[:, 1, :]
    features[:, 132:135] = np.where(two_hand_mask, rel_wrist, 0.0)

    # 4. Wrist Velocities (6 dims: [135:141])
    wrist_vel = np.zeros((T, 2, 3), dtype=np.float32)
    if T > 1:
        vel_mask = (m[1:, :] & m[:-1, :])[:, :, np.newaxis]
        wrist_vel[1:] = np.where(vel_mask, global_wrists[1:] - global_wrists[:-1], 0.0)
    features[:, 135:141] = wrist_vel.reshape(T, 6)

    # 5. Hand Presence Mask (2 dims: [141:143])
    features[:, 141:143] = m[:, :2].astype(np.float32)

    return features


def compute_feature_vector(
    hands_coords: np.ndarray,
    hands_masks: np.ndarray,
    pose_coords: Optional[np.ndarray] = None,
    pose_masks: Optional[np.ndarray] = None,
    feature_set: str = FEATURE_SET_HANDS,
    normalization_epsilon: float = 1e-4,
) -> np.ndarray:
    """Compute comprehensive feature vector with optional upper-body pose reference.

    Dimensions:
    - 'hands': 143 dims.
    - 'hands_pose': 160 dims (143 hands + 17 pose torso & elbow features).
    """
    if feature_set not in (FEATURE_SET_HANDS, FEATURE_SET_HANDS_POSE):
        raise ValueError(
            f"Unsupported feature_set '{feature_set}'. Allowed: ['{FEATURE_SET_HANDS}', '{FEATURE_SET_HANDS_POSE}']"
        )

    hand_feats = compute_feature_vector_143(
        coords=hands_coords,
        masks=hands_masks,
        normalization_epsilon=normalization_epsilon,
    )
    if feature_set == FEATURE_SET_HANDS:
        return hand_feats

    T = len(hands_coords)
    if pose_coords is None or T == 0:
        pose_pad = np.zeros((T, 17), dtype=np.float32)
        return np.hstack([hand_feats, pose_pad]).astype(np.float32)

    # Extract 4 upper-body joints: [11: L_shoulder, 12: R_shoulder, 13: L_elbow, 14: R_elbow]
    if pose_coords.shape[1] == 33:
        pose_pts = pose_coords[:, [11, 12, 13, 14], :].copy()
    elif pose_coords.shape[1] == 4:
        pose_pts = pose_coords.copy()
    else:
        raise ValueError(f"Expected pose_coords to have 33 or 4 joints, got {pose_coords.shape}")

    all_finite = np.all(np.isfinite(pose_pts), axis=(1, 2))
    if pose_masks is not None:
        p_resampled = pose_masks.flatten() > 0.5
    else:
        p_resampled = np.ones(T, dtype=bool)

    l_sh = pose_pts[:, 0, :]
    r_sh = pose_pts[:, 1, :]
    torso_center = (l_sh + r_sh) / 2.0
    torso_scale = np.linalg.norm(l_sh - r_sh, axis=-1)

    # Usable pose support requires finite points, resampled support, and non-degenerate shoulders
    p_usable = all_finite & p_resampled & (torso_scale > normalization_epsilon)

    norm_pose = np.zeros((T, 4, 3), dtype=np.float32)
    for i in range(4):
        centered = pose_pts[:, i, :] - torso_center
        norm_pose[:, i, :] = np.where(
            p_usable[:, np.newaxis],
            centered / np.maximum(torso_scale[:, np.newaxis], normalization_epsilon),
            0.0,
        )

    # Torso velocity: gated on both adjacent frames being usable
    torso_vel = np.zeros((T, 3), dtype=np.float32)
    if T > 1:
        u_valid = (p_usable[1:] & p_usable[:-1])[:, np.newaxis]
        torso_vel[1:] = np.where(u_valid, torso_center[1:] - torso_center[:-1], 0.0)

    clipped_scale = np.where(p_usable, np.clip(torso_scale, 0.0, 2.0), 0.0)[:, np.newaxis]
    p_mask_channel = p_usable.astype(np.float32)[:, np.newaxis]

    pose_features = np.hstack([
        norm_pose.reshape(T, 12),
        torso_vel,
        clipped_scale,
        p_mask_channel,
    ]).astype(np.float32)

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
            np.zeros((target_len, FEATURE_DIM_HANDS), dtype=np.float32),
            np.zeros((target_len, 2), dtype=bool),
        )

    filled_coords, filled_masks = interpolate_bounded_gaps(trimmed_coords, trimmed_masks)
    resampled_coords, resampled_masks = resample_temporal(
        filled_coords, filled_masks, target_len=target_len
    )
    features_143 = compute_feature_vector_143(resampled_coords, resampled_masks)
    return features_143, resampled_masks
