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

from src.features.normalization import (
    compute_bend_cosines,
    compute_fingertip_displacements,
)
from src.features.schema import (
    SLICE_GLOBAL_WRISTS,
    SLICE_HAND_MASKS,
    SLICE_LOCAL_HAND_SHAPES,
    SLICE_NORM_POSE,
    SLICE_POSE_MASK,
    SLICE_SHOULDER_SCALE,
    SLICE_TORSO_VEL,
    infer_feature_layout_from_dim,
)


def recompute_derived_features(
    features: np.ndarray,
    normalization_epsilon: float = 1e-4,
    hand_transitions: Optional[np.ndarray] = None,
    pose_transitions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Recompute relative wrist, velocities, fingertip displacements, and bend cosines.

    Layout-aware contract:
    - If layout.wrist_reference == 'torso_xy': gated on geometry support mask g[t, h], wrists z strictly 0.
    - If layout.wrist_reference == 'image': gated on hand support mask m[t, h].
    - Pose velocity gated on adjacent valid pose frames and pose transitions.
    - Local fingertip displacements and bend cosines recomputed from current local shapes.
    - Preserves zero motion across long gaps where transitions are invalid.
    """
    out = features.copy()
    T, D = out.shape
    if D < 143:
        return out

    layout = infer_feature_layout_from_dim(D)
    m = out[:, layout.slice_hand_masks] > 0.5  # (T, 2)
    wrists = out[:, layout.slice_wrists].reshape(T, 2, 3)

    if hand_transitions is not None:
        h_trans = hand_transitions[1:, :2] if len(hand_transitions) == T else (hand_transitions[:, :2] if hand_transitions.ndim > 1 else hand_transitions[:, np.newaxis])
    else:
        # Infer transition validity from input features: if input had 0 velocity/displacement while delta was non-zero,
        # it represents a long dropout gap where transitions must NOT be revived.
        initial_wrist_vel = features[:, layout.slice_wrist_vel].reshape(T, 2, 3)
        old_wrists = features[:, layout.slice_wrists].reshape(T, 2, 3)
        old_delta = old_wrists[1:, :, :2] - old_wrists[:-1, :, :2]
        has_delta = np.any(np.abs(old_delta) > 1e-6, axis=-1)
        is_zero_vel = np.all(np.abs(initial_wrist_vel[1:, :, :2]) < 1e-6, axis=-1)
        h_trans = ~(has_delta & is_zero_vel)
        if layout.slice_fingertip_displacement is not None:
            initial_tips = features[:, layout.slice_fingertip_displacement].reshape(T, 2, 5, 3)
            old_shapes = features[:, layout.slice_local_hand_shapes].reshape(T, 2, 21, 3)
            old_tip_pts = old_shapes[:, :, [4, 8, 12, 16, 20], :]
            old_tip_delta = old_tip_pts[1:] - old_tip_pts[:-1]
            has_tip_delta = np.any(np.abs(old_tip_delta) > 1e-6, axis=(2, 3))
            is_zero_tip = np.all(np.abs(initial_tips[1:]) < 1e-6, axis=(2, 3))
            h_trans = h_trans & ~(has_tip_delta & is_zero_tip)

    if pose_transitions is not None:
        p_trans = pose_transitions[1:, 0:1] if len(pose_transitions) == T else (pose_transitions[:, 0:1] if pose_transitions.ndim > 1 else pose_transitions[:, np.newaxis])
    elif layout.has_pose and layout.slice_torso_vel is not None:
        initial_torso_vel = features[:, layout.slice_torso_vel]
        old_pose = features[:, layout.slice_norm_pose].reshape(T, 4, 3)
        old_torso_delta = old_pose[1:, 0, :2] - old_pose[:-1, 0, :2]
        has_torso_delta = np.any(np.abs(old_torso_delta) > 1e-6, axis=-1)
        is_zero_torso_vel = np.all(np.abs(initial_torso_vel[1:]) < 1e-6, axis=-1)
        p_trans = ~(has_torso_delta & is_zero_torso_vel)[:, np.newaxis]
    else:
        p_trans = np.ones((T - 1, 1), dtype=bool)

    if layout.wrist_reference == "torso_xy" and layout.slice_geometry_support is not None:
        g = out[:, layout.slice_geometry_support] > 0.5  # (T, 2)
        wrists = np.where(g[:, :, np.newaxis], wrists, 0.0)
        wrists[:, :, 2] = 0.0  # Invariant: Torso-relative wrist z strictly 0
        out[:, layout.slice_wrists] = wrists.reshape(T, 6)

        both_g = (g[:, 0] & g[:, 1])[:, np.newaxis]
        rel_wrist = np.zeros((T, 3), dtype=out.dtype)
        rel_wrist[:, :2] = np.where(both_g, wrists[:, 0, :2] - wrists[:, 1, :2], 0.0)
        rel_wrist[:, 2] = 0.0
        out[:, layout.slice_rel_wrist] = rel_wrist

        wrist_vel = np.zeros((T, 2, 3), dtype=out.dtype)
        if T > 1:
            vel_g = (g[1:, :] & g[:-1, :] & h_trans & p_trans)[:, :, np.newaxis]
            diff = wrists[1:, :, :2] - wrists[:-1, :, :2]
            wrist_vel[1:, :, :2] = np.where(vel_g[:, :, :2], diff, 0.0)
            wrist_vel[:, :, 2] = 0.0
        out[:, layout.slice_wrist_vel] = wrist_vel.reshape(T, 6)
    else:
        wrists = np.where(m[:, :, np.newaxis], wrists, 0.0)
        out[:, layout.slice_wrists] = wrists.reshape(T, 6)

        two_hand_mask = (m[:, 0] & m[:, 1])[:, np.newaxis]
        rel_wrist = wrists[:, 0, :] - wrists[:, 1, :]
        out[:, layout.slice_rel_wrist] = np.where(two_hand_mask, rel_wrist, 0.0)

        wrist_vel = np.zeros((T, 2, 3), dtype=out.dtype)
        if T > 1:
            vel_valid = (m[1:, :] & m[:-1, :] & h_trans)[:, :, np.newaxis]
            wrist_vel[1:] = np.where(vel_valid, wrists[1:] - wrists[:-1], 0.0)
        out[:, layout.slice_wrist_vel] = wrist_vel.reshape(T, 6)

    if layout.slice_norm_pose is not None:
        p = out[:, layout.slice_pose_mask].flatten() > 0.5
        out[:, layout.slice_norm_pose] = np.where(p[:, np.newaxis], out[:, layout.slice_norm_pose], 0.0)
        out[:, layout.slice_shoulder_scale] = np.where(p[:, np.newaxis], out[:, layout.slice_shoulder_scale], 0.0)
        u_valid = np.zeros(T, dtype=bool)
        if T > 1:
            u_valid[1:] = p[1:] & p[:-1] & p_trans[:, 0]
        out[:, layout.slice_torso_vel] = np.where(u_valid[:, np.newaxis], out[:, layout.slice_torso_vel], 0.0)

    # Recompute fingertip displacement if present
    if layout.slice_fingertip_displacement is not None:
        shapes = out[:, layout.slice_local_hand_shapes].reshape(T, 2, 21, 3)
        tip_disp = compute_fingertip_displacements(shapes, m[:, :2], transitions=h_trans)
        out[:, layout.slice_fingertip_displacement] = tip_disp

    # Recompute bend cosines if present
    if layout.slice_bend_cosines is not None:
        shapes = out[:, layout.slice_local_hand_shapes].reshape(T, 2, 21, 3)
        bends = compute_bend_cosines(shapes, m[:, :2], normalization_epsilon=normalization_epsilon)
        out[:, layout.slice_bend_cosines] = bends

    return out


def add_jitter(
    features: np.ndarray,
    sigma: float = 0.01,
    noise_shape: Optional[np.ndarray] = None,
    noise_wrist: Optional[np.ndarray] = None,
    normalization_epsilon: float = 1e-4,
    hand_transitions: Optional[np.ndarray] = None,
    pose_transitions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Add Gaussian noise to local hand shapes and wrists on usable slots, then recompute motion.

    Local hand shape jitter keeps landmark 0 (wrist origin) identically 0.
    In torso_xy mode, wrist z jitter is strictly 0.0.
    """
    out = features.copy()
    T, D = out.shape
    if sigma <= 0.0:
        return out

    layout = infer_feature_layout_from_dim(D) if D >= 143 else None
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
        noise_s[:, :, 0, :] = 0.0
        shapes = np.where(m[:, :, np.newaxis, np.newaxis], shapes + noise_s, 0.0)
        out[:, SLICE_LOCAL_HAND_SHAPES] = shapes.reshape(T, 126)

    # 2. Wrists [126:132] -> (T, 2, 3)
    if D >= 132:
        wrists = out[:, SLICE_GLOBAL_WRISTS].reshape(T, 2, 3)
        if noise_wrist is None:
            noise_w = np.random.normal(0.0, sigma, size=wrists.shape).astype(np.float32)
        else:
            noise_w = np.array(noise_wrist, dtype=np.float32).copy()
            if noise_w.ndim == 2 and noise_w.shape == (T, 6):
                noise_w = noise_w.reshape(T, 2, 3)
        if layout and layout.wrist_reference == "torso_xy":
            noise_w[:, :, 2] = 0.0
            g_mask = (
                out[:, layout.slice_geometry_support] > 0.5
                if layout.slice_geometry_support is not None
                else m
            )
            wrists = np.where(g_mask[:, :, np.newaxis], wrists + noise_w, 0.0)
            wrists[:, :, 2] = 0.0
        else:
            wrists = np.where(m[:, :, np.newaxis], wrists + noise_w, 0.0)
        out[:, SLICE_GLOBAL_WRISTS] = wrists.reshape(T, 6)

    # 3. Recompute motion to maintain internal consistency
    if D >= 143:
        out = recompute_derived_features(
            out,
            normalization_epsilon=normalization_epsilon,
            hand_transitions=hand_transitions,
            pose_transitions=pose_transitions,
        )

    return out


def random_rotate_2d(
    features: np.ndarray,
    max_angle_deg: float = 8.0,
    angle_deg: Optional[float] = None,
    normalization_epsilon: float = 1e-4,
    hand_transitions: Optional[np.ndarray] = None,
    pose_transitions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rotate points in the (x, y) plane for usable slots and recompute motion.

    - Local shapes rotate around (0, 0).
    - Image wrists rotate around (0.5, 0.5).
    - Torso-relative wrists rotate around (0, 0).
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
    layout = infer_feature_layout_from_dim(D) if D >= 143 else None
    m = out[:, SLICE_HAND_MASKS] > 0.5 if D >= 143 else np.ones((T, 2), dtype=bool)

    # 1. Local shapes [0:126] -> (T, 2, 21, 3)
    if D >= 126:
        shapes = out[:, SLICE_LOCAL_HAND_SHAPES].reshape(T, 2, 21, 3)
        xy = shapes[:, :, :, :2].reshape(-1, 2)
        xy_rot = (xy @ rot_matrix.T).reshape(T, 2, 21, 2)
        shapes[:, :, :, :2] = np.where(m[:, :, np.newaxis, np.newaxis], xy_rot, 0.0)
        shapes[:, :, :, 2] = np.where(m[:, :, np.newaxis], shapes[:, :, :, 2], 0.0)
        out[:, SLICE_LOCAL_HAND_SHAPES] = shapes.reshape(T, 126)

    # 2. Wrists [126:132] -> (T, 2, 3)
    if D >= 132:
        wrists = out[:, SLICE_GLOBAL_WRISTS].reshape(T, 2, 3)
        if layout and layout.wrist_reference == "torso_xy":
            g_mask = (
                out[:, layout.slice_geometry_support] > 0.5
                if layout.slice_geometry_support is not None
                else m
            )
            w_xy = wrists[:, :, :2].reshape(-1, 2)
            w_xy_rot = (w_xy @ rot_matrix.T).reshape(T, 2, 2)
            wrists[:, :, :2] = np.where(g_mask[:, :, np.newaxis], w_xy_rot, 0.0)
            wrists[:, :, 2] = 0.0
        else:
            w_xy = wrists[:, :, :2].reshape(-1, 2) - 0.5
            w_xy_rot = (w_xy @ rot_matrix.T + 0.5).reshape(T, 2, 2)
            wrists[:, :, :2] = np.where(m[:, :, np.newaxis], w_xy_rot, 0.0)
            wrists[:, :, 2] = np.where(m, wrists[:, :, 2], 0.0)
        out[:, SLICE_GLOBAL_WRISTS] = wrists.reshape(T, 6)

    # 3. Pose rotation [143:155] (4 points) and torso velocity [155:158]
    if layout and layout.has_pose:
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

    # 4. Recompute derived features
    if D >= 143:
        out = recompute_derived_features(
            out,
            normalization_epsilon=normalization_epsilon,
            hand_transitions=hand_transitions,
            pose_transitions=pose_transitions,
        )

    return out


def horizontal_flip(
    features: np.ndarray,
    swap_hands: bool = False,
    normalization_epsilon: float = 1e-4,
    hand_transitions: Optional[np.ndarray] = None,
    pose_transitions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Horizontal mirror reflection across x axis.

    Exact mathematical contract:
    - H'[t, h, j] = diag(-1, 1, 1) H[t, h, j]
    - w'[t, h] = (-w_x, w_y, 0) if torso_xy else (1 - w_x, w_y, w_z)
    - q'[t, k] = diag(-1, 1, 1) q[t, pi[k]] where pi = [1, 0, 3, 2]
    - u'[t] = diag(-1, 1, 1) u[t]
    - Unsupported slots strictly remain 0.
    """
    out = features.copy()
    T, D = out.shape

    layout = infer_feature_layout_from_dim(D) if D >= 143 else None
    m = out[:, SLICE_HAND_MASKS] > 0.5 if D >= 143 else np.ones((T, 2), dtype=bool)

    # 1. Local Shapes (126 dims)
    if D >= 126:
        shapes = out[:, SLICE_LOCAL_HAND_SHAPES].reshape(T, 2, 21, 3)
        shapes[:, :, :, 0] = np.where(m[:, :, np.newaxis], -shapes[:, :, :, 0], 0.0)
        shapes[:, :, :, 1:] = np.where(m[:, :, np.newaxis, np.newaxis], shapes[:, :, :, 1:], 0.0)
        if swap_hands:
            shapes = shapes[:, [1, 0], :, :]
        out[:, SLICE_LOCAL_HAND_SHAPES] = shapes.reshape(T, 126)

    # 2. Wrists (6 dims)
    if D >= 132:
        wrists = out[:, SLICE_GLOBAL_WRISTS].reshape(T, 2, 3)
        if layout and layout.wrist_reference == "torso_xy":
            g_mask = (
                out[:, layout.slice_geometry_support] > 0.5
                if layout.slice_geometry_support is not None
                else m
            )
            wrists[:, :, 0] = np.where(g_mask, -wrists[:, :, 0], 0.0)
            wrists[:, :, 1:] = np.where(g_mask[:, :, np.newaxis], wrists[:, :, 1:], 0.0)
            wrists[:, :, 2] = 0.0
        else:
            wrists[:, :, 0] = np.where(m, 1.0 - wrists[:, :, 0], 0.0)
            wrists[:, :, 1:] = np.where(m[:, :, np.newaxis], wrists[:, :, 1:], 0.0)
        if swap_hands:
            wrists = wrists[:, [1, 0], :]
        out[:, SLICE_GLOBAL_WRISTS] = wrists.reshape(T, 6)

    # 3. Swap Hand Masks, Geometry Support, and hand slots if swap_hands is requested
    if D >= 143 and swap_hands:
        out[:, SLICE_HAND_MASKS] = out[:, [142, 141]]
        if layout and layout.slice_geometry_support is not None:
            geom_s = layout.slice_geometry_support
            out[:, geom_s] = out[:, [geom_s.start + 1, geom_s.start]]
        if layout and layout.slice_wrist_vel is not None:
            w_vel = out[:, layout.slice_wrist_vel].reshape(T, 2, 3)
            out[:, layout.slice_wrist_vel] = w_vel[:, [1, 0], :].reshape(T, 6)
        if layout and layout.slice_fingertip_displacement is not None:
            f_disp = out[:, layout.slice_fingertip_displacement].reshape(T, 2, 5, 3)
            out[:, layout.slice_fingertip_displacement] = f_disp[:, [1, 0], :, :].reshape(T, 30)
        if layout and layout.slice_bend_cosines is not None:
            b_cos = out[:, layout.slice_bend_cosines].reshape(T, 2, 5, 2)
            out[:, layout.slice_bend_cosines] = b_cos[:, [1, 0], :, :].reshape(T, 20)
        if hand_transitions is not None and hand_transitions.ndim > 1:
            hand_transitions = hand_transitions[:, [1, 0]]

    # 4. Pose landmarks (12 coords torso-centered, 3 coords torso velocity)
    if layout and layout.has_pose:
        p = out[:, SLICE_POSE_MASK].flatten() > 0.5
        pose = out[:, SLICE_NORM_POSE].reshape(T, 4, 3)
        # Torso-centered coordinates: negate x
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

    # 5. Recompute derived features (relative wrist, velocities, fingertips, bends)
    if D >= 143:
        out = recompute_derived_features(
            out,
            normalization_epsilon=normalization_epsilon,
            hand_transitions=hand_transitions,
            pose_transitions=pose_transitions,
        )

    return out


def shape_scale(
    features: np.ndarray,
    r: float = 0.10,
    sx: Optional[float] = None,
    sy: Optional[float] = None,
    normalization_epsilon: float = 1e-4,
    hand_transitions: Optional[np.ndarray] = None,
    pose_transitions: Optional[np.ndarray] = None,
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

    if D >= 143:
        out = recompute_derived_features(
            out,
            normalization_epsilon=normalization_epsilon,
            hand_transitions=hand_transitions,
            pose_transitions=pose_transitions,
        )

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
    normalization_epsilon: float = 1e-4,
    hand_transitions: Optional[np.ndarray] = None,
    pose_transitions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Scale and translate global wrist coordinates for usable slots, then recompute motion."""
    out = features.copy()
    T, D = out.shape
    if D < 132:
        return out

    layout = infer_feature_layout_from_dim(D) if D >= 143 else None
    if layout and layout.wrist_reference == "torso_xy":
        raise ValueError(
            "scale_translate is not supported when features.wrist_reference='torso_xy'. "
            "Wrist-only image scale and translation corrupts torso-relative coordinate geometry."
        )

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
        out = recompute_derived_features(
            out,
            normalization_epsilon=normalization_epsilon,
            hand_transitions=hand_transitions,
            pose_transitions=pose_transitions,
        )

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


def apply_augmentation(
    features: np.ndarray,
    cfg: Any = None,
    normalization_epsilon: float = 1e-4,
    hand_transitions: Optional[np.ndarray] = None,
    pose_transitions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply stochastic augmentation pipeline based on Config object or dictionary."""
    if cfg is None:
        return features

    aug_cfg = getattr(cfg, "augmentation", cfg)
    if isinstance(aug_cfg, dict):
        flip_cfg = aug_cfg.get("flip", {})
        rotate_cfg = aug_cfg.get("rotate", {})
        shape_scale_cfg = aug_cfg.get("shape_scale", {})
        scale_trans_cfg = aug_cfg.get("scale_translate", {})
        jitter_cfg = aug_cfg.get("jitter", {})
    else:
        flip_cfg = getattr(aug_cfg, "flip", {})
        rotate_cfg = getattr(aug_cfg, "rotate", {})
        shape_scale_cfg = getattr(aug_cfg, "shape_scale", {})
        scale_trans_cfg = getattr(aug_cfg, "scale_translate", {})
        jitter_cfg = getattr(aug_cfg, "jitter", {})

    out = features.copy()

    # 1. Flip
    if flip_cfg.get("enabled", False):
        prob = float(_get_aug_param(flip_cfg, "prob", 0.5))
        if np.random.rand() < prob:
            swap_hands = bool(_get_aug_param(flip_cfg, "swap_hands", False))
            out = horizontal_flip(
                out,
                swap_hands=swap_hands,
                normalization_epsilon=normalization_epsilon,
                hand_transitions=hand_transitions,
                pose_transitions=pose_transitions,
            )

    # 2. Rotate
    if rotate_cfg.get("enabled", False):
        prob = float(_get_aug_param(rotate_cfg, "prob", 0.5))
        if np.random.rand() < prob:
            max_angle = float(_get_aug_param(rotate_cfg, "max_angle_deg", 8.0))
            out = random_rotate_2d(
                out,
                max_angle_deg=max_angle,
                normalization_epsilon=normalization_epsilon,
                hand_transitions=hand_transitions,
                pose_transitions=pose_transitions,
            )

    # 3. Shape Scale
    if shape_scale_cfg.get("enabled", False):
        prob = float(_get_aug_param(shape_scale_cfg, "prob", 0.5))
        r = float(_get_aug_param(shape_scale_cfg, "r", 0.10))
        if np.random.rand() < prob and r > 0.0:
            out = shape_scale(
                out,
                r=r,
                normalization_epsilon=normalization_epsilon,
                hand_transitions=hand_transitions,
                pose_transitions=pose_transitions,
            )

    # 4. Scale & Translate
    if scale_trans_cfg.get("enabled", False):
        prob = float(_get_aug_param(scale_trans_cfg, "prob", 0.7))
        if np.random.rand() < prob:
            s_min = float(_get_aug_param(scale_trans_cfg, "scale_min", 0.85))
            s_max = float(_get_aug_param(scale_trans_cfg, "scale_max", 1.15))
            trans = float(_get_aug_param(scale_trans_cfg, "translate", 0.10))
            out = scale_translate(
                out,
                scale_min=s_min,
                scale_max=s_max,
                translate=trans,
                normalization_epsilon=normalization_epsilon,
                hand_transitions=hand_transitions,
                pose_transitions=pose_transitions,
            )

    # 5. Jitter
    if jitter_cfg.get("enabled", False):
        prob = float(_get_aug_param(jitter_cfg, "prob", 0.5))
        if np.random.rand() < prob:
            sigma = float(_get_aug_param(jitter_cfg, "sigma", 0.01))
            out = add_jitter(
                out,
                sigma=sigma,
                normalization_epsilon=normalization_epsilon,
                hand_transitions=hand_transitions,
                pose_transitions=pose_transitions,
            )
        if np.random.rand() < prob:
            sigma = float(_get_aug_param(jitter_cfg, "sigma", 0.01))
            out = add_jitter(
                out, sigma=sigma, normalization_epsilon=normalization_epsilon
            )

    # Enforce torso_xy wrist z invariant at shared boundary
    T, D = out.shape
    if D >= 143:
        layout = infer_feature_layout_from_dim(D)
        if layout and layout.wrist_reference == "torso_xy":
            out[:, [128, 131]] = 0.0

    return out
