"""Spatial/temporal normalization and feature extraction for ISLR landmarks.

Features:
- Configurable feature sets: 'hands' (143 dims) or 'hands_pose' (160 dims).
- Wave B semantic feature extensions:
  - 'torso_xy' body-relative wrists (+2 geometry support channels)
  - 'clip_median' robust palm scale normalization
  - Fingertip local displacements (+30 channels)
  - Finger articulation bend cosines (+20 channels)
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
    FINGER_CHAINS,
    FINGERTIP_LANDMARK_INDICES,
    FeatureLayout,
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
    "compute_clip_palm_scales",
    "compute_clip_shoulder_width",
    "compute_fingertip_displacements",
    "compute_bend_cosines",
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
    """
    T = len(coords)
    orig_coords_shape = coords.shape
    orig_masks_ndim = masks.ndim

    if orig_masks_ndim == 1:
        masks_2d = masks[:, np.newaxis].astype(bool)
    else:
        masks_2d = masks.astype(bool)

    num_slots = masks_2d.shape[1]

    if T == 0:
        out_c = np.zeros_like(coords, dtype=np.float32)
        out_m = np.zeros(0, dtype=bool) if orig_masks_ndim == 1 else np.zeros((0, num_slots), dtype=bool)
        return out_c, out_m

    coords_reshaped = coords.reshape(T, num_slots, -1).copy()
    coords_reshaped = np.nan_to_num(coords_reshaped, nan=0.0, posinf=0.0, neginf=0.0)

    # Endpoint validity: mask is True and non-zero/finite
    out_coords = np.zeros_like(coords_reshaped, dtype=np.float32)
    out_masks = np.zeros((T, num_slots), dtype=bool)

    for h in range(num_slots):
        m_slot = masks_2d[:, h].copy()
        c_slot = coords_reshaped[:, h, :].copy()

        # Sanitize missing frames to 0
        c_slot[~m_slot] = 0.0

        valid_indices = np.where(m_slot)[0]
        if len(valid_indices) == 0:
            continue

        # Keep original observed frames
        out_coords[valid_indices, h, :] = c_slot[valid_indices]
        out_masks[valid_indices, h] = True

        if max_gap_frames <= 0:
            continue

        # Find interior missing runs
        for k in range(len(valid_indices) - 1):
            idx_start = valid_indices[k]
            idx_end = valid_indices[k + 1]
            gap_len = idx_end - idx_start - 1

            if 0 < gap_len <= max_gap_frames:
                v0 = c_slot[idx_start]
                v1 = c_slot[idx_end]
                for step in range(1, gap_len + 1):
                    alpha = float(step) / float(gap_len + 1)
                    fill_idx = idx_start + step
                    out_coords[fill_idx, h, :] = (1.0 - alpha) * v0 + alpha * v1
                    out_masks[fill_idx, h] = True

    final_coords = out_coords.reshape(orig_coords_shape).astype(np.float32)
    final_masks = out_masks[:, 0] if orig_masks_ndim == 1 else out_masks
    return final_coords, final_masks


def interpolate_missing_hands(
    raw_coords: np.ndarray,
    hand_masks: np.ndarray,
) -> np.ndarray:
    """Convenience wrapper for bounded interpolation (max_gap_frames=3)."""
    filled, _ = interpolate_bounded_gaps(raw_coords, hand_masks, max_gap_frames=3)
    return filled


def resample_temporal(
    coords: np.ndarray,
    masks: np.ndarray,
    target_len: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample landmark coordinates and support masks to a fixed sequence length."""
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
        for h in range(num_slots):
            if masks_2d[0, h]:
                resampled_masks[0, h] = True
                resampled_flat[0, h] = coords_reshaped[0, h]
        out_c = resampled_flat.reshape(out_coords_shape)
        out_m = resampled_masks[:, 0] if orig_masks_ndim == 1 else resampled_masks
        return out_c, out_m

    if N == 1:
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
    n = np.where(alpha <= 0.5, i0, i1)

    alpha_expanded = alpha[:, np.newaxis]

    for h in range(num_slots):
        m_slot = masks_2d[:, h]
        c_slot = coords_reshaped[:, h, :]

        m_out_slot = m_slot[n]
        resampled_masks[:, h] = m_out_slot

        both_valid = m_slot[i0] & m_slot[i1]
        single_valid = m_out_slot & (~both_valid)

        c_both = (1.0 - alpha_expanded) * c_slot[i0] + alpha_expanded * c_slot[i1]
        c_single = c_slot[n]

        c_interp = np.where(
            both_valid[:, np.newaxis],
            c_both,
            np.where(single_valid[:, np.newaxis], c_single, 0.0),
        )
        resampled_flat[:, h] = c_interp

    out_c = resampled_flat.reshape(out_coords_shape).astype(np.float32)
    out_m = resampled_masks[:, 0] if orig_masks_ndim == 1 else resampled_masks
    return out_c, out_m


def compute_clip_palm_scales(
    hands_coords: np.ndarray,
    masks: np.ndarray,
    normalization_epsilon: float = 1e-4,
) -> Tuple[float, float]:
    """Compute median observed non-degenerate palm scale for each hand slot across valid frames.

    Args:
        hands_coords: Landmark coordinates (T, 2, 21, 3).
        masks: Boolean support masks (T, 2).
        normalization_epsilon: Minimum scale threshold.

    Returns:
        Tuple[float, float]: (scale_slot_0, scale_slot_1).
    """
    T = len(hands_coords)
    scales = [0.0, 0.0]
    if T == 0:
        return (0.0, 0.0)

    m = masks.astype(bool)
    if m.ndim == 1:
        m = m[:, np.newaxis]

    for h in range(min(2, hands_coords.shape[1])):
        valid_t = np.where(m[:, h])[0]
        if len(valid_t) == 0:
            continue
        wrist = hands_coords[valid_t, h, 0, :]
        knuckle = hands_coords[valid_t, h, 9, :]
        dists = np.linalg.norm(knuckle - wrist, axis=-1)
        valid_dists = dists[dists > normalization_epsilon]
        if len(valid_dists) > 0:
            scales[h] = float(np.median(valid_dists))

    return (scales[0], scales[1])


def compute_clip_shoulder_width(
    pose_coords: Optional[np.ndarray],
    masks: Optional[np.ndarray],
    normalization_epsilon: float = 1e-4,
) -> float:
    """Compute median observed valid non-degenerate shoulder width across sequence frames.

    Args:
        pose_coords: Upper-body pose points (T, 4, 3) or (T, 33, 3).
        masks: Pose support masks (T,) or (T, 1).
        normalization_epsilon: Minimum width threshold.

    Returns:
        float: Median shoulder width, or 0.0 if invalid/insufficient.
    """
    if pose_coords is None or len(pose_coords) == 0:
        return 0.0

    T = len(pose_coords)
    if pose_coords.shape[1] == 33:
        l_sh = pose_coords[:, 11, :2]
        r_sh = pose_coords[:, 12, :2]
    elif pose_coords.shape[1] == 4:
        l_sh = pose_coords[:, 0, :2]
        r_sh = pose_coords[:, 1, :2]
    else:
        return 0.0

    p_finite = np.all(np.isfinite(l_sh), axis=-1) & np.all(np.isfinite(r_sh), axis=-1)
    if masks is not None:
        p_mask = masks.flatten() > 0.5
    else:
        p_mask = np.ones(T, dtype=bool)

    dists = np.linalg.norm(l_sh - r_sh, axis=-1)
    valid_mask = p_finite & p_mask & (dists > normalization_epsilon)
    valid_dists = dists[valid_mask]

    if len(valid_dists) > 0:
        return float(np.median(valid_dists))
    return 0.0


def compute_resampled_transitions(
    source_masks: np.ndarray,
    target_len: int,
) -> np.ndarray:
    """Compute (target_len, num_slots) transition validity mask across temporal resampling.

    A transition at target step j (j >= 1) is valid if and only if the source sequence was
    continuously supported across all source frames between floor(z[j-1]) and ceil(z[j]).
    Transition at j = 0 is always False.
    """
    if source_masks is None or len(source_masks) <= 1 or target_len <= 1:
        num_slots = source_masks.shape[1] if (source_masks is not None and source_masks.ndim > 1) else 1
        return np.zeros((target_len, num_slots), dtype=bool)

    N = len(source_masks)
    s_masks = source_masks.reshape(N, -1)
    num_slots = s_masks.shape[1]

    transitions = np.zeros((target_len, num_slots), dtype=bool)
    j_indices = np.arange(target_len, dtype=np.float64)
    z = j_indices * (N - 1) / (target_len - 1)

    for j in range(1, target_len):
        k_start = int(np.floor(z[j - 1]))
        k_end = min(int(np.ceil(z[j])) + 1, N)
        for h in range(num_slots):
            transitions[j, h] = bool(np.all(s_masks[k_start:k_end, h]))

    return transitions


def compute_fingertip_displacements(
    local_shapes: np.ndarray,
    masks: np.ndarray,
    transitions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute 30-dim local fingertip displacements (2 hands * 5 tips * 3 coordinates).

    Tips: Thumb 4, Index 8, Middle 12, Ring 16, Pinky 20.
    Gated on adjacent valid observation endpoints (m[t, h] and m[t-1, h]).
    Displacement at t=0 is identically 0.0.
    """
    T = len(local_shapes)
    if T == 0:
        return np.zeros((0, 30), dtype=np.float32)

    m = masks.astype(bool)
    if m.ndim == 1:
        m = m[:, np.newaxis]

    # Tips: indices (4, 8, 12, 16, 20)
    tips = local_shapes[:, :, list(FINGERTIP_LANDMARK_INDICES), :]  # (T, 2, 5, 3)
    tip_disp = np.zeros((T, 2, 5, 3), dtype=np.float32)

    if T > 1:
        if transitions is not None:
            t_valid = transitions[1:, :2] if len(transitions) == T else (transitions[:, :2] if transitions.ndim > 1 else transitions[:, np.newaxis])
            valid_adjacent = (t_valid & m[1:, :2] & m[:-1, :2])[:, :, np.newaxis, np.newaxis]
        else:
            valid_adjacent = (m[1:, :2] & m[:-1, :2])[:, :, np.newaxis, np.newaxis]
        diff = tips[1:] - tips[:-1]
        tip_disp[1:] = np.where(valid_adjacent, diff, 0.0)

    return tip_disp.reshape(T, 30).astype(np.float32)


def compute_bend_cosines(
    local_shapes: np.ndarray,
    masks: np.ndarray,
    normalization_epsilon: float = 1e-4,
) -> np.ndarray:
    """Compute 20-dim finger articulation bend cosines (2 hands * 5 chains * 2 bone-pairs).

    For consecutive bone vectors u (i -> j) and v (j -> k), cos(theta) = clip(u . v / (|u| * |v|), -1, 1).
    Straight finger: cos(theta) = +1.0.
    Right angle bend: cos(theta) = 0.0.
    Folded back: cos(theta) = -1.0.
    Degenerate or unsupported bones: 0.0.
    """
    T = len(local_shapes)
    if T == 0:
        return np.zeros((0, 20), dtype=np.float32)

    m = masks.astype(bool)
    if m.ndim == 1:
        m = m[:, np.newaxis]

    bend_cosines = np.zeros((T, 2, 5, 2), dtype=np.float32)

    for h in range(min(2, local_shapes.shape[1])):
        valid_t = np.where(m[:, h])[0]
        if len(valid_t) == 0:
            continue

        hand_frames = local_shapes[valid_t, h]  # (N_valid, 21, 3)

        for c_idx, chain in enumerate(FINGER_CHAINS):
            i0, i1, i2, i3 = chain
            p0 = hand_frames[:, i0, :]
            p1 = hand_frames[:, i1, :]
            p2 = hand_frames[:, i2, :]
            p3 = hand_frames[:, i3, :]

            # Pair 0: bone (i0 -> i1) and bone (i1 -> i2)
            u0 = p1 - p0
            v0 = p2 - p1
            len_u0 = np.linalg.norm(u0, axis=-1)
            len_v0 = np.linalg.norm(v0, axis=-1)
            valid_pair0 = (len_u0 > normalization_epsilon) & (len_v0 > normalization_epsilon)
            dot0 = np.sum(u0 * v0, axis=-1)
            denom0 = np.where(valid_pair0, len_u0 * len_v0, 1.0)
            cos0 = np.where(
                valid_pair0,
                np.clip(dot0 / denom0, -1.0, 1.0),
                0.0,
            )
            bend_cosines[valid_t, h, c_idx, 0] = cos0

            # Pair 1: bone (i1 -> i2) and bone (i2 -> i3)
            u1 = p2 - p1
            v1 = p3 - p2
            len_u1 = np.linalg.norm(u1, axis=-1)
            len_v1 = np.linalg.norm(v1, axis=-1)
            valid_pair1 = (len_u1 > normalization_epsilon) & (len_v1 > normalization_epsilon)
            dot1 = np.sum(u1 * v1, axis=-1)
            denom1 = np.where(valid_pair1, len_u1 * len_v1, 1.0)
            cos1 = np.where(
                valid_pair1,
                np.clip(dot1 / denom1, -1.0, 1.0),
                0.0,
            )
            bend_cosines[valid_t, h, c_idx, 1] = cos1

    return bend_cosines.reshape(T, 20).astype(np.float32)


def compute_feature_vector_143(
    coords: np.ndarray,
    masks: np.ndarray,
    normalization_epsilon: float = 1e-4,
) -> np.ndarray:
    """Transform (T, 2, 21, 3) hand landmarks into standard 143-dim feature matrix."""
    T = len(coords)
    features = np.zeros((T, 143), dtype=np.float32)
    if T == 0:
        return features

    m = masks.astype(bool)
    if m.ndim == 1:
        m = m[:, np.newaxis]

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
            hand_centered[0] = 0.0
            palm_size = float(np.linalg.norm(hand_centered[9]))
            if palm_size > normalization_epsilon:
                hand_centered /= palm_size
            else:
                hand_centered = np.zeros_like(hand_centered)
            local_shapes[t, h] = hand_centered

    features[:, :126] = local_shapes.reshape(T, 126)
    features[:, 126:132] = global_wrists.reshape(T, 6)

    two_hand_mask = (m[:, 0] & m[:, 1])[:, np.newaxis]
    rel_wrist = global_wrists[:, 0, :] - global_wrists[:, 1, :]
    features[:, 132:135] = np.where(two_hand_mask, rel_wrist, 0.0)

    wrist_vel = np.zeros((T, 2, 3), dtype=np.float32)
    if T > 1:
        vel_mask = (m[1:, :] & m[:-1, :])[:, :, np.newaxis]
        wrist_vel[1:] = np.where(vel_mask, global_wrists[1:] - global_wrists[:-1], 0.0)
    features[:, 135:141] = wrist_vel.reshape(T, 6)
    features[:, 141:143] = m[:, :2].astype(np.float32)

    return features


def compute_feature_vector(
    hands_coords: np.ndarray,
    hands_masks: np.ndarray,
    pose_coords: Optional[np.ndarray] = None,
    pose_masks: Optional[np.ndarray] = None,
    feature_set: str = FEATURE_SET_HANDS,
    wrist_reference: str = "image",
    palm_scale: str = "frame",
    fingertip_displacement: bool = False,
    bend_cosines: bool = False,
    clip_palm_scales: Optional[Tuple[float, float]] = None,
    clip_shoulder_width: Optional[float] = None,
    normalization_epsilon: float = 1e-4,
    hand_transitions: Optional[np.ndarray] = None,
    pose_transitions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute canonical feature vector according to exact FeatureLayout contract."""
    layout = FeatureLayout.build(
        feature_set=feature_set,
        wrist_reference=wrist_reference,
        palm_scale=palm_scale,
        fingertip_displacement=fingertip_displacement,
        bend_cosines=bend_cosines,
    )

    T = len(hands_coords)
    if T == 0:
        return np.zeros((0, layout.total_dim), dtype=np.float32)

    m = hands_masks.astype(bool)
    if m.ndim == 1:
        m = m[:, np.newaxis]

    # 1. Local Hand Shapes (126 dims)
    local_shapes = np.zeros((T, 2, 21, 3), dtype=np.float32)
    for h in range(min(2, hands_coords.shape[1])):
        valid_t = np.where(m[:, h])[0]
        for t in valid_t:
            hand_pts = hands_coords[t, h].copy()
            wrist = hand_pts[0].copy()
            hand_centered = hand_pts - wrist
            hand_centered[0] = 0.0

            if palm_scale == "clip_median" and clip_palm_scales is not None:
                psize = clip_palm_scales[h]
            else:
                psize = float(np.linalg.norm(hand_centered[9]))

            if psize > normalization_epsilon:
                hand_centered /= psize
            else:
                hand_centered = np.zeros_like(hand_centered)
            local_shapes[t, h] = hand_centered

    # 2. Wrists, Relative Wrist, and Velocities
    if wrist_reference == "torso_xy":
        # Extract upper body pose
        if pose_coords is not None and len(pose_coords) > 0:
            if pose_coords.shape[1] == 33:
                pose_pts = pose_coords[:, [11, 12, 13, 14], :].copy()
            elif pose_coords.shape[1] == 4:
                pose_pts = pose_coords.copy()
            else:
                pose_pts = np.zeros((T, 4, 3), dtype=np.float32)
        else:
            pose_pts = np.zeros((T, 4, 3), dtype=np.float32)

        l_sh = pose_pts[:, 0, :]
        r_sh = pose_pts[:, 1, :]
        torso_center = (l_sh + r_sh) / 2.0
        torso_scale = np.linalg.norm(l_sh - r_sh, axis=-1)

        p_finite = np.all(np.isfinite(pose_pts), axis=(1, 2))
        p_resampled = pose_masks.flatten() > 0.5 if pose_masks is not None else np.ones(T, dtype=bool)
        p_usable = p_finite & p_resampled & (torso_scale > normalization_epsilon)

        ell = clip_shoulder_width if clip_shoulder_width is not None else 0.0
        ell_valid = ell > normalization_epsilon

        # Geometry support mask g[t, h]
        g = np.zeros((T, 2), dtype=bool)
        for h in range(2):
            g[:, h] = m[:, h] & p_usable & ell_valid

        w_image = hands_coords[:, :, 0, :].copy()
        wrists = np.zeros((T, 2, 3), dtype=np.float32)
        for h in range(2):
            if ell_valid:
                wrists[:, h, :2] = np.where(g[:, h, np.newaxis], (w_image[:, h, :2] - torso_center[:, :2]) / ell, 0.0)
            wrists[:, h, 2] = 0.0  # Torso-relative z strictly 0

        # Relative wrist: gated on both hands geometry valid
        both_g = (g[:, 0] & g[:, 1])[:, np.newaxis]
        rel_wrist = np.zeros((T, 3), dtype=np.float32)
        rel_wrist[:, :2] = np.where(both_g, wrists[:, 0, :2] - wrists[:, 1, :2], 0.0)
        rel_wrist[:, 2] = 0.0

        # Wrist velocity: gated on adjacent frames both geometry valid AND transitions valid
        wrist_vel = np.zeros((T, 2, 3), dtype=np.float32)
        if T > 1:
            h_trans = hand_transitions[1:, :2] if (hand_transitions is not None and hand_transitions.ndim > 1) else (hand_transitions[1:, np.newaxis] if hand_transitions is not None else np.ones((T - 1, 2), dtype=bool))
            p_trans = pose_transitions[1:, 0:1] if (pose_transitions is not None and pose_transitions.ndim > 1) else (pose_transitions[1:, np.newaxis] if pose_transitions is not None else np.ones((T - 1, 1), dtype=bool))
            vel_g = (g[1:, :] & g[:-1, :] & h_trans & p_trans)[:, :, np.newaxis]
            diff_w = wrists[1:, :, :2] - wrists[:-1, :, :2]
            wrist_vel[1:, :, :2] = np.where(vel_g[:, :, :2], diff_w, 0.0)
            wrist_vel[:, :, 2] = 0.0
    else:
        # Standard image wrists
        raw_wrists = hands_coords[:, :, 0, :].copy()
        wrists = np.where(m[:, :, np.newaxis], raw_wrists, 0.0)

        two_hand_mask = (m[:, 0] & m[:, 1])[:, np.newaxis]
        rel_wrist = np.where(two_hand_mask, wrists[:, 0, :] - wrists[:, 1, :], 0.0)

        wrist_vel = np.zeros((T, 2, 3), dtype=np.float32)
        if T > 1:
            if hand_transitions is not None:
                h_trans = hand_transitions[1:, :2] if hand_transitions.ndim > 1 else hand_transitions[1:, np.newaxis]
                vel_mask = (m[1:, :] & m[:-1, :] & h_trans)[:, :, np.newaxis]
            else:
                vel_mask = (m[1:, :] & m[:-1, :])[:, :, np.newaxis]
            wrist_vel[1:] = np.where(vel_mask, wrists[1:] - wrists[:-1], 0.0)
        g = None

    # Base Hand Feature Matrix (143 dims)
    feature_blocks = [
        local_shapes.reshape(T, 126),
        wrists.reshape(T, 6),
        rel_wrist,
        wrist_vel.reshape(T, 6),
        m[:, :2].astype(np.float32),
    ]

    # Pose Channels (17 dims if hands_pose)
    if feature_set == FEATURE_SET_HANDS_POSE:
        if pose_coords is None or len(pose_coords) == 0:
            pose_features = np.zeros((T, 17), dtype=np.float32)
        else:
            if pose_coords.shape[1] == 33:
                pose_pts = pose_coords[:, [11, 12, 13, 14], :].copy()
            elif pose_coords.shape[1] == 4:
                pose_pts = pose_coords.copy()
            else:
                pose_pts = np.zeros((T, 4, 3), dtype=np.float32)

            l_sh = pose_pts[:, 0, :]
            r_sh = pose_pts[:, 1, :]
            torso_center = (l_sh + r_sh) / 2.0
            torso_scale = np.linalg.norm(l_sh - r_sh, axis=-1)

            p_finite = np.all(np.isfinite(pose_pts), axis=(1, 2))
            p_resampled = pose_masks.flatten() > 0.5 if pose_masks is not None else np.ones(T, dtype=bool)
            p_usable = p_finite & p_resampled & (torso_scale > normalization_epsilon)

            norm_pose = np.zeros((T, 4, 3), dtype=np.float32)
            for i in range(4):
                centered = pose_pts[:, i, :] - torso_center
                norm_pose[:, i, :] = np.where(
                    p_usable[:, np.newaxis],
                    centered / np.maximum(torso_scale[:, np.newaxis], normalization_epsilon),
                    0.0,
                )

            torso_vel = np.zeros((T, 3), dtype=np.float32)
            if T > 1:
                if pose_transitions is not None:
                    p_trans = pose_transitions[1:, 0] if pose_transitions.ndim > 1 else pose_transitions[1:]
                    u_valid = (p_usable[1:] & p_usable[:-1] & p_trans)[:, np.newaxis]
                else:
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

        feature_blocks.append(pose_features)

    # Geometry Support Channels (2 dims if torso_xy)
    if layout.slice_geometry_support is not None:
        geom_support = g.astype(np.float32) if g is not None else np.zeros((T, 2), dtype=np.float32)
        feature_blocks.append(geom_support)

    # Fingertip Displacements (30 dims if enabled)
    if layout.slice_fingertip_displacement is not None:
        tip_disps = compute_fingertip_displacements(
            local_shapes, m[:, :2], transitions=hand_transitions
        )
        feature_blocks.append(tip_disps)

    # Finger Articulation Bend Cosines (20 dims if enabled)
    if layout.slice_bend_cosines is not None:
        bends = compute_bend_cosines(local_shapes, m[:, :2], normalization_epsilon=normalization_epsilon)
        feature_blocks.append(bends)

    features = np.hstack(feature_blocks).astype(np.float32)
    assert features.shape == (T, layout.total_dim), (
        f"Feature dimension mismatch: expected {layout.total_dim}, got {features.shape[1]}"
    )
    return features


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
