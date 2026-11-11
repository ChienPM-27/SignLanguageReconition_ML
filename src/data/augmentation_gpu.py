"""Batched PyTorch tensor augmentation running directly on device (GPU / CPU).

Supports parallel batched transformations on tensor (B, T, D):
1. batch_horizontal_flip: Horizontal reflection with corrected torso-pose reflection and anatomical swap.
2. batch_random_rotate_2d: 2D rotation of hands, wrists, and pose with recomputed motion.
3. batch_add_jitter: Gaussian perturbation of shape (with fixed wrist origin) and wrists with recomputed motion.
4. batch_shape_scale: Independent x, y scaling of local hand shapes on usable slots.
5. batch_scale_translate: Scale and translation of global wrist coordinates with recomputed motion.
6. batch_time_stretch: Explicitly disabled in Phase 0.
7. apply_augmentation_batch: Config-driven batched pipeline.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Tuple
import torch

from src.features.schema import (
    NUM_BEND_COSINES_DIMS,
    NUM_FINGERTIP_DISPLACEMENT_DIMS,
    SLICE_GLOBAL_WRISTS,
    SLICE_HAND_MASKS,
    SLICE_LOCAL_HAND_SHAPES,
    SLICE_NORM_POSE,
    SLICE_SHOULDER_SCALE,
    SLICE_TORSO_VEL,
    infer_feature_layout_from_dim,
)


def _rand(
    size: Tuple[int, ...],
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate uniform random tensor [0, 1) compatible with cross-device generator."""
    if generator is not None:
        if str(generator.device) == str(device):
            return torch.rand(size, generator=generator, device=device)
        return torch.rand(size, generator=generator, device=generator.device).to(device)
    return torch.rand(size, device=device)


def _randn(
    size: Tuple[int, ...],
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate standard normal random tensor N(0, 1) compatible with cross-device generator."""
    if generator is not None:
        if str(generator.device) == str(device):
            return torch.randn(size, generator=generator, device=device)
        return torch.randn(size, generator=generator, device=generator.device).to(device)
    return torch.randn(size, device=device)


def _get_aug_param(cfg_obj: Any, key: str, default: Any) -> Any:
    """Extract parameter from dict or config dataclass."""
    if isinstance(cfg_obj, dict):
        val = cfg_obj.get(key)
    else:
        val = getattr(cfg_obj, key, None)
    return val if val is not None else default


def batch_recompute_derived_features(
    x: torch.Tensor,
    normalization_epsilon: float = 1e-4,
    hand_transitions: Optional[torch.Tensor] = None,
    pose_transitions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Recompute relative wrist, velocities, fingertip displacements, and bend cosines on batch tensor (B, T, D).

    Layout-aware contract:
    - If layout.wrist_reference == 'torso_xy': gated on geometry support mask g[t, h], wrists z strictly 0.
    - If layout.wrist_reference == 'image': gated on hand support mask m[t, h].
    - Pose velocity gated on adjacent valid pose frames and pose transitions.
    - Local fingertip displacements and bend cosines recomputed from current local shapes.
    - Preserves zero motion across long gaps where transitions are invalid.
    """
    out = x.clone()
    B, T, D = out.shape
    if D < 143:
        return out

    layout = infer_feature_layout_from_dim(D)
    m = out[:, :, layout.slice_hand_masks] > 0.5  # (B, T, 2)
    wrists = out[:, :, layout.slice_wrists].reshape(B, T, 2, 3)

    if hand_transitions is not None:
        if hand_transitions.shape[1] == T:
            h_trans = hand_transitions[:, 1:, :2]
        else:
            h_trans = hand_transitions[:, :, :2] if hand_transitions.ndim > 2 else hand_transitions.unsqueeze(-1)
    else:
        # Infer transition validity from input x
        initial_wrist_vel = x[:, :, layout.slice_wrist_vel].reshape(B, T, 2, 3)
        old_wrists = x[:, :, layout.slice_wrists].reshape(B, T, 2, 3)
        old_delta = old_wrists[:, 1:, :, :2] - old_wrists[:, :-1, :, :2]
        has_delta = torch.any(torch.abs(old_delta) > 1e-6, dim=-1)
        is_zero_vel = torch.all(torch.abs(initial_wrist_vel[:, 1:, :, :2]) < 1e-6, dim=-1)
        h_trans = ~(has_delta & is_zero_vel)
        if layout.slice_fingertip_displacement is not None:
            initial_tips = x[:, :, layout.slice_fingertip_displacement].reshape(B, T, 2, 5, 3)
            old_shapes = x[:, :, layout.slice_local_hand_shapes].reshape(B, T, 2, 21, 3)
            old_tip_pts = old_shapes[:, :, :, [4, 8, 12, 16, 20], :]
            old_tip_delta = old_tip_pts[:, 1:] - old_tip_pts[:, :-1]
            has_tip_delta = torch.any(torch.abs(old_tip_delta) > 1e-6, dim=(-1, -2))
            is_zero_tip = torch.all(torch.abs(initial_tips[:, 1:]) < 1e-6, dim=(-1, -2))
            h_trans = h_trans & ~(has_tip_delta & is_zero_tip)

    if pose_transitions is not None:
        if pose_transitions.shape[1] == T:
            p_trans = pose_transitions[:, 1:, 0:1]
        else:
            p_trans = pose_transitions[:, :, 0:1] if pose_transitions.ndim > 2 else pose_transitions.unsqueeze(-1)
    elif layout.has_pose and layout.slice_torso_vel is not None:
        initial_torso_vel = x[:, :, layout.slice_torso_vel]
        old_pose = x[:, :, layout.slice_norm_pose].reshape(B, T, 4, 3)
        old_torso_delta = old_pose[:, 1:, 0, :2] - old_pose[:, :-1, 0, :2]
        has_torso_delta = torch.any(torch.abs(old_torso_delta) > 1e-6, dim=-1)
        is_zero_torso_vel = torch.all(torch.abs(initial_torso_vel[:, 1:]) < 1e-6, dim=-1)
        p_trans = ~(has_torso_delta & is_zero_torso_vel).unsqueeze(-1)
    else:
        p_trans = torch.ones((B, T - 1, 1), dtype=torch.bool, device=out.device)

    if layout.wrist_reference == "torso_xy" and layout.slice_geometry_support is not None:
        g = out[:, :, layout.slice_geometry_support] > 0.5  # (B, T, 2)
        wrists = torch.where(g.unsqueeze(-1), wrists, torch.zeros_like(wrists))
        wrists[:, :, :, 2] = 0.0  # Invariant: Torso-relative wrist z strictly 0
        out[:, :, layout.slice_wrists] = wrists.reshape(B, T, 6)

        both_g = (g[:, :, 0] & g[:, :, 1]).unsqueeze(-1)
        rel_wrist = torch.zeros((B, T, 3), dtype=out.dtype, device=out.device)
        rel_wrist[:, :, :2] = torch.where(
            both_g,
            wrists[:, :, 0, :2] - wrists[:, :, 1, :2],
            torch.zeros_like(rel_wrist[:, :, :2]),
        )
        rel_wrist[:, :, 2] = 0.0
        out[:, :, layout.slice_rel_wrist] = rel_wrist

        wrist_vel = torch.zeros((B, T, 2, 3), dtype=out.dtype, device=out.device)
        if T > 1:
            vel_valid = (g[:, 1:, :] & g[:, :-1, :] & h_trans & p_trans).unsqueeze(-1)
            diff = wrists[:, 1:, :, :2] - wrists[:, :-1, :, :2]
            wrist_vel[:, 1:, :, :2] = torch.where(vel_valid, diff, torch.zeros_like(diff))
            wrist_vel[:, :, :, 2] = 0.0
        out[:, :, layout.slice_wrist_vel] = wrist_vel.reshape(B, T, 6)
    else:
        wrists = torch.where(m.unsqueeze(-1), wrists, torch.zeros_like(wrists))
        out[:, :, layout.slice_wrists] = wrists.reshape(B, T, 6)

        two_hand_mask = (m[:, :, 0] & m[:, :, 1]).unsqueeze(-1)
        rel_wrist = wrists[:, :, 0, :] - wrists[:, :, 1, :]
        out[:, :, layout.slice_rel_wrist] = torch.where(two_hand_mask, rel_wrist, torch.zeros_like(rel_wrist))

        wrist_vel = torch.zeros((B, T, 2, 3), dtype=out.dtype, device=out.device)
        if T > 1:
            vel_valid = (m[:, 1:, :] & m[:, :-1, :] & h_trans).unsqueeze(-1)
            diff = wrists[:, 1:] - wrists[:, :-1]
            wrist_vel[:, 1:] = torch.where(vel_valid, diff, torch.zeros_like(diff))
        out[:, :, layout.slice_wrist_vel] = wrist_vel.reshape(B, T, 6)

    if layout.slice_norm_pose is not None:
        p = out[:, :, layout.slice_pose_mask].squeeze(-1) > 0.5  # (B, T)
        p_u = p.unsqueeze(-1)
        out[:, :, layout.slice_norm_pose] = torch.where(
            p_u, out[:, :, layout.slice_norm_pose], torch.zeros_like(out[:, :, layout.slice_norm_pose])
        )
        out[:, :, layout.slice_shoulder_scale] = torch.where(
            p_u, out[:, :, layout.slice_shoulder_scale], torch.zeros_like(out[:, :, layout.slice_shoulder_scale])
        )
        u_valid = torch.zeros((B, T), dtype=torch.bool, device=out.device)
        if T > 1:
            u_valid[:, 1:] = p[:, 1:] & p[:, :-1] & p_trans.squeeze(-1)
        out[:, :, layout.slice_torso_vel] = torch.where(
            u_valid.unsqueeze(-1), out[:, :, layout.slice_torso_vel], torch.zeros_like(out[:, :, layout.slice_torso_vel])
        )

    # Recompute fingertip displacement if present
    if layout.slice_fingertip_displacement is not None:
        shapes = out[:, :, layout.slice_local_hand_shapes].reshape(B, T, 2, 21, 3)
        tips = shapes[:, :, :, [4, 8, 12, 16, 20], :]  # (B, T, 2, 5, 3)
        tip_disp = torch.zeros((B, T, 2, 5, 3), dtype=out.dtype, device=out.device)
        if T > 1:
            valid_t = (m[:, 1:, :2] & m[:, :-1, :2] & h_trans).unsqueeze(-1).unsqueeze(-1)
            tip_disp[:, 1:] = torch.where(valid_t, tips[:, 1:] - tips[:, :-1], torch.zeros_like(tips[:, 1:]))
        out[:, :, layout.slice_fingertip_displacement] = tip_disp.reshape(B, T, NUM_FINGERTIP_DISPLACEMENT_DIMS)

    # Recompute bend cosines if present
    if layout.slice_bend_cosines is not None:
        shapes = out[:, :, layout.slice_local_hand_shapes].reshape(B, T, 2, 21, 3)
        bends = torch.zeros((B, T, 2, 5, 2), dtype=out.dtype, device=out.device)
        chains = [
            (1, 2, 3, 4),
            (5, 6, 7, 8),
            (9, 10, 11, 12),
            (13, 14, 15, 16),
            (17, 18, 19, 20),
        ]
        eps = normalization_epsilon
        for c_idx, (i0, i1, i2, i3) in enumerate(chains):
            p0, p1, p2, p3 = shapes[:, :, :, i0], shapes[:, :, :, i1], shapes[:, :, :, i2], shapes[:, :, :, i3]
            u0, v0 = p1 - p0, p2 - p1
            len_u0 = torch.norm(u0, dim=-1)
            len_v0 = torch.norm(v0, dim=-1)
            valid0 = (len_u0 > eps) & (len_v0 > eps) & m
            denom0 = torch.where(valid0, len_u0 * len_v0, torch.ones_like(len_u0))
            cos0 = torch.clamp(torch.sum(u0 * v0, dim=-1) / denom0, -1.0, 1.0)
            bends[:, :, :, c_idx, 0] = torch.where(valid0, cos0, torch.zeros_like(cos0))

            u1, v1 = p2 - p1, p3 - p2
            len_u1 = torch.norm(u1, dim=-1)
            len_v1 = torch.norm(v1, dim=-1)
            valid1 = (len_u1 > eps) & (len_v1 > eps) & m
            denom1 = torch.where(valid1, len_u1 * len_v1, torch.ones_like(len_u1))
            cos1 = torch.clamp(torch.sum(u1 * v1, dim=-1) / denom1, -1.0, 1.0)
            bends[:, :, :, c_idx, 1] = torch.where(valid1, cos1, torch.zeros_like(cos1))
        out[:, :, layout.slice_bend_cosines] = bends.reshape(B, T, NUM_BEND_COSINES_DIMS)

    return out


def batch_horizontal_flip(
    x: torch.Tensor,
    swap_hands: bool = False,
    normalization_epsilon: float = 1e-4,
    hand_transitions: Optional[torch.Tensor] = None,
    pose_transitions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Horizontal mirror reflection on batch tensor (B, T, D).

    Corrected contract:
    - Local shapes negate x on usable slots.
    - Wrists: negate x if torso_xy else map x -> 1 - x on usable slots.
    - Torso-centered pose negates x and swaps anatomical sides [1, 0, 3, 2].
    - Torso velocity negates x on valid adjacent frames.
    - Unsupported slots strictly remain 0.
    """
    out = x.clone()
    B, T, D = out.shape

    layout = infer_feature_layout_from_dim(D) if D >= 143 else None
    m = out[:, :, SLICE_HAND_MASKS] > 0.5 if D >= 143 else torch.ones((B, T, 2), dtype=torch.bool, device=out.device)

    # 1. Local Shapes (126 dims)
    if D >= 126:
        shapes = out[:, :, SLICE_LOCAL_HAND_SHAPES].reshape(B, T, 2, 21, 3).clone()
        m_s = m.unsqueeze(-1)
        shapes[:, :, :, :, 0] = torch.where(m_s, -shapes[:, :, :, :, 0], torch.zeros_like(shapes[:, :, :, :, 0]))
        shapes[:, :, :, :, 1:] = torch.where(m.unsqueeze(-1).unsqueeze(-1), shapes[:, :, :, :, 1:], torch.zeros_like(shapes[:, :, :, :, 1:]))
        if swap_hands:
            shapes = shapes[:, :, [1, 0], :, :]
        out[:, :, SLICE_LOCAL_HAND_SHAPES] = shapes.reshape(B, T, 126)

    # 2. Wrists (6 dims)
    if D >= 132:
        wrists = out[:, :, SLICE_GLOBAL_WRISTS].reshape(B, T, 2, 3).clone()
        if layout and layout.wrist_reference == "torso_xy":
            g_mask = (
                out[:, :, layout.slice_geometry_support] > 0.5
                if layout.slice_geometry_support is not None
                else m
            )
            wrists[:, :, :, 0] = torch.where(g_mask, -wrists[:, :, :, 0], torch.zeros_like(wrists[:, :, :, 0]))
            wrists[:, :, :, 1:] = torch.where(g_mask.unsqueeze(-1), wrists[:, :, :, 1:], torch.zeros_like(wrists[:, :, :, 1:]))
            wrists[:, :, :, 2] = 0.0
        else:
            wrists[:, :, :, 0] = torch.where(m, 1.0 - wrists[:, :, :, 0], torch.zeros_like(wrists[:, :, :, 0]))
            wrists[:, :, :, 1:] = torch.where(m.unsqueeze(-1), wrists[:, :, :, 1:], torch.zeros_like(wrists[:, :, :, 1:]))
        if swap_hands:
            wrists = wrists[:, :, [1, 0], :]
        out[:, :, SLICE_GLOBAL_WRISTS] = wrists.reshape(B, T, 6)

    # 3. Swap Hand Masks, Geometry Support, and hand slots if swap_hands is requested
    if D >= 143 and swap_hands:
        out[:, :, SLICE_HAND_MASKS] = out[:, :, [142, 141]]
        if layout and layout.slice_geometry_support is not None:
            geom_s = layout.slice_geometry_support
            out[:, :, geom_s] = out[:, :, [geom_s.start + 1, geom_s.start]]
        if layout and layout.slice_wrist_vel is not None:
            w_vel = out[:, :, layout.slice_wrist_vel].reshape(B, T, 2, 3)
            out[:, :, layout.slice_wrist_vel] = w_vel[:, :, [1, 0], :].reshape(B, T, 6)
        if layout and layout.slice_fingertip_displacement is not None:
            f_disp = out[:, :, layout.slice_fingertip_displacement].reshape(B, T, 2, 5, 3)
            out[:, :, layout.slice_fingertip_displacement] = f_disp[:, :, [1, 0], :, :].reshape(B, T, NUM_FINGERTIP_DISPLACEMENT_DIMS)
        if layout and layout.slice_bend_cosines is not None:
            b_cos = out[:, :, layout.slice_bend_cosines].reshape(B, T, 2, 5, 2)
            out[:, :, layout.slice_bend_cosines] = b_cos[:, :, [1, 0], :, :].reshape(B, T, NUM_BEND_COSINES_DIMS)
        if hand_transitions is not None and hand_transitions.ndim > 2:
            hand_transitions = hand_transitions[:, :, [1, 0]]

    # 4. Pose landmarks
    if layout and layout.has_pose:
        p = out[:, :, layout.slice_pose_mask].squeeze(-1) > 0.5 if layout else out[:, :, 159] > 0.5
        pose = out[:, :, SLICE_NORM_POSE].reshape(B, T, 4, 3).clone()
        # Torso-centered coordinates: negate x
        pose[:, :, :, 0] = torch.where(p.unsqueeze(-1), -pose[:, :, :, 0], torch.zeros_like(pose[:, :, :, 0]))
        pose[:, :, :, 1:] = torch.where(p.unsqueeze(-1).unsqueeze(-1), pose[:, :, :, 1:], torch.zeros_like(pose[:, :, :, 1:]))
        # Swap anatomical sides: L_shoulder <-> R_shoulder, L_elbow <-> R_elbow
        pose_flipped = pose[:, :, [1, 0, 3, 2], :]
        out[:, :, SLICE_NORM_POSE] = pose_flipped.reshape(B, T, 12)

        # Torso velocity: negate x
        u = out[:, :, SLICE_TORSO_VEL].clone()
        u_valid = torch.zeros((B, T), dtype=torch.bool, device=out.device)
        if T > 1:
            u_valid[:, 1:] = p[:, 1:] & p[:, :-1]
        u[:, :, 0] = torch.where(u_valid, -u[:, :, 0], torch.zeros_like(u[:, :, 0]))
        u[:, :, 1:] = torch.where(u_valid.unsqueeze(-1), u[:, :, 1:], torch.zeros_like(u[:, :, 1:]))
        out[:, :, SLICE_TORSO_VEL] = u

        out[:, :, SLICE_SHOULDER_SCALE] = torch.where(
            p.unsqueeze(-1), out[:, :, SLICE_SHOULDER_SCALE], torch.zeros_like(out[:, :, SLICE_SHOULDER_SCALE])
        )

    # 5. Recompute derived features
    if D >= 143:
        out = batch_recompute_derived_features(
            out,
            normalization_epsilon=normalization_epsilon,
            hand_transitions=hand_transitions,
            pose_transitions=pose_transitions,
        )

    return out


def batch_random_rotate_2d(
    x: torch.Tensor,
    max_angle_deg: float = 8.0,
    angles_deg: Optional[Any] = None,
    generator: Optional[torch.Generator] = None,
    normalization_epsilon: float = 1e-4,
) -> torch.Tensor:
    """Rotate points in (x, y) plane on batch (B, T, D) and recompute motion."""
    out = x.clone()
    B, T, D = out.shape
    if B == 0 or max_angle_deg <= 0.0:
        return out

    layout = infer_feature_layout_from_dim(D) if D >= 143 else None

    if angles_deg is None:
        rand_u = _rand((B,), out.device, generator)
        angles_deg = rand_u * (2.0 * max_angle_deg) - max_angle_deg
    else:
        angles_deg = torch.as_tensor(angles_deg, device=out.device, dtype=torch.float32)
        if angles_deg.numel() == 1:
            angles_deg = angles_deg.reshape(1).expand(B)
        elif angles_deg.shape != (B,):
            raise ValueError(f"angles_deg must have shape ({B},) or be a scalar, got {angles_deg.shape}")

    angles_rad = angles_deg * (math.pi / 180.0)
    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)

    m = out[:, :, SLICE_HAND_MASKS] > 0.5 if D >= 143 else torch.ones((B, T, 2), dtype=torch.bool, device=out.device)

    # 1. Local shapes [0:126] -> (B, T, 2, 21, 3)
    if D >= 126:
        shapes = out[:, :, SLICE_LOCAL_HAND_SHAPES].reshape(B, T, 2, 21, 3).clone()
        cos_m = cos_a.view(B, 1, 1, 1)
        sin_m = sin_a.view(B, 1, 1, 1)
        sx = shapes[:, :, :, :, 0].clone()
        sy = shapes[:, :, :, :, 1].clone()
        rot_x = sx * cos_m - sy * sin_m
        rot_y = sx * sin_m + sy * cos_m
        m_s = m.unsqueeze(-1)
        shapes[:, :, :, :, 0] = torch.where(m_s, rot_x, torch.zeros_like(rot_x))
        shapes[:, :, :, :, 1] = torch.where(m_s, rot_y, torch.zeros_like(rot_y))
        shapes[:, :, :, :, 2] = torch.where(m_s, shapes[:, :, :, :, 2], torch.zeros_like(shapes[:, :, :, :, 2]))
        out[:, :, SLICE_LOCAL_HAND_SHAPES] = shapes.reshape(B, T, 126)

    # 2. Wrists [126:132] -> (B, T, 2, 3)
    if D >= 132:
        wrists = out[:, :, SLICE_GLOBAL_WRISTS].reshape(B, T, 2, 3).clone()
        cos_w = cos_a.view(B, 1, 1)
        sin_w = sin_a.view(B, 1, 1)
        if layout and layout.wrist_reference == "torso_xy":
            g_mask = (
                out[:, :, layout.slice_geometry_support] > 0.5
                if layout.slice_geometry_support is not None
                else m
            )
            wx = wrists[:, :, :, 0].clone()
            wy = wrists[:, :, :, 1].clone()
            rot_wx = wx * cos_w - wy * sin_w
            rot_wy = wx * sin_w + wy * cos_w
            wrists[:, :, :, 0] = torch.where(g_mask, rot_wx, torch.zeros_like(rot_wx))
            wrists[:, :, :, 1] = torch.where(g_mask, rot_wy, torch.zeros_like(rot_wy))
            wrists[:, :, :, 2] = 0.0
        else:
            wx = wrists[:, :, :, 0].clone() - 0.5
            wy = wrists[:, :, :, 1].clone() - 0.5
            rot_wx = wx * cos_w - wy * sin_w + 0.5
            rot_wy = wx * sin_w + wy * cos_w + 0.5
            wrists[:, :, :, 0] = torch.where(m, rot_wx, torch.zeros_like(rot_wx))
            wrists[:, :, :, 1] = torch.where(m, rot_wy, torch.zeros_like(rot_wy))
            wrists[:, :, :, 2] = torch.where(m, wrists[:, :, :, 2], torch.zeros_like(wrists[:, :, :, 2]))
        out[:, :, SLICE_GLOBAL_WRISTS] = wrists.reshape(B, T, 6)

    # 3. Pose rotation [143:155] and torso velocity [155:158]
    if layout and layout.has_pose:
        p = out[:, :, layout.slice_pose_mask].squeeze(-1) > 0.5 if layout else out[:, :, 159] > 0.5
        pose = out[:, :, SLICE_NORM_POSE].reshape(B, T, 4, 3).clone()
        cos_p = cos_a.view(B, 1, 1)
        sin_p = sin_a.view(B, 1, 1)
        px = pose[:, :, :, 0].clone()
        py = pose[:, :, :, 1].clone()
        rot_px = px * cos_p - py * sin_p
        rot_py = px * sin_p + py * cos_p
        p_u = p.unsqueeze(-1)
        pose[:, :, :, 0] = torch.where(p_u, rot_px, torch.zeros_like(rot_px))
        pose[:, :, :, 1] = torch.where(p_u, rot_py, torch.zeros_like(rot_py))
        pose[:, :, :, 2] = torch.where(p_u, pose[:, :, :, 2], torch.zeros_like(pose[:, :, :, 2]))
        out[:, :, SLICE_NORM_POSE] = pose.reshape(B, T, 12)

        u = out[:, :, SLICE_TORSO_VEL].clone()
        cos_u = cos_a.view(B, 1)
        sin_u = sin_a.view(B, 1)
        ux = u[:, :, 0].clone()
        uy = u[:, :, 1].clone()
        rot_ux = ux * cos_u - uy * sin_u
        rot_uy = ux * sin_u + uy * cos_u
        u_valid = torch.zeros((B, T), dtype=torch.bool, device=out.device)
        if T > 1:
            u_valid[:, 1:] = p[:, 1:] & p[:, :-1]
        u[:, :, 0] = torch.where(u_valid, rot_ux, torch.zeros_like(rot_ux))
        u[:, :, 1] = torch.where(u_valid, rot_uy, torch.zeros_like(rot_uy))
        u[:, :, 2] = torch.where(u_valid, u[:, :, 2], torch.zeros_like(u[:, :, 2]))
        out[:, :, SLICE_TORSO_VEL] = u
        out[:, :, SLICE_SHOULDER_SCALE] = torch.where(
            p.unsqueeze(-1), out[:, :, SLICE_SHOULDER_SCALE], torch.zeros_like(out[:, :, SLICE_SHOULDER_SCALE])
        )

    # 4. Recompute derived features
    if D >= 143:
        out = batch_recompute_derived_features(out, normalization_epsilon=normalization_epsilon)

    return out


def batch_add_jitter(
    x: torch.Tensor,
    sigma: float = 0.01,
    noise_shape: Optional[Any] = None,
    noise_wrist: Optional[Any] = None,
    generator: Optional[torch.Generator] = None,
    normalization_epsilon: float = 1e-4,
) -> torch.Tensor:
    """Add Gaussian perturbation to local shapes and wrists on batch (B, T, D).

    In torso_xy mode, wrist z jitter is strictly zeroed.
    """
    out = x.clone()
    B, T, D = out.shape
    if B == 0 or sigma <= 0.0:
        return out

    layout = infer_feature_layout_from_dim(D) if D >= 143 else None
    m = out[:, :, SLICE_HAND_MASKS] > 0.5 if D >= 143 else torch.ones((B, T, 2), dtype=torch.bool, device=out.device)

    # 1. Local shapes [0:126] -> (B, T, 2, 21, 3)
    if D >= 126:
        shapes = out[:, :, SLICE_LOCAL_HAND_SHAPES].reshape(B, T, 2, 21, 3).clone()
        if noise_shape is not None:
            noise_shape_t = torch.as_tensor(noise_shape, device=out.device, dtype=out.dtype).clone()
            if noise_shape_t.ndim == 2 and noise_shape_t.shape == (T, 126):
                noise_shape_t = noise_shape_t.reshape(T, 2, 21, 3).unsqueeze(0).expand(B, -1, -1, -1, -1).clone()
            elif noise_shape_t.ndim == 4 and noise_shape_t.shape == (T, 2, 21, 3):
                noise_shape_t = noise_shape_t.unsqueeze(0).expand(B, -1, -1, -1, -1).clone()
            elif noise_shape_t.ndim == 3 and noise_shape_t.shape == (B, T, 126):
                noise_shape_t = noise_shape_t.reshape(B, T, 2, 21, 3).clone()
            elif noise_shape_t.ndim == 4 and noise_shape_t.shape == (B, T, 2, 21, 3):
                pass
        else:
            noise_shape_t = _randn((B, T, 2, 21, 3), out.device, generator) * sigma
        noise_shape_t[:, :, :, 0, :] = 0.0
        m_s = m.unsqueeze(-1).unsqueeze(-1)
        shapes = torch.where(m_s, shapes + noise_shape_t, torch.zeros_like(shapes))
        out[:, :, SLICE_LOCAL_HAND_SHAPES] = shapes.reshape(B, T, 126)

    # 2. Wrists [126:132] -> (B, T, 2, 3)
    if D >= 132:
        wrists = out[:, :, SLICE_GLOBAL_WRISTS].reshape(B, T, 2, 3).clone()
        if noise_wrist is not None:
            noise_wrist_t = torch.as_tensor(noise_wrist, device=out.device, dtype=out.dtype).clone()
            if noise_wrist_t.ndim == 2 and noise_wrist_t.shape == (T, 6):
                noise_wrist_t = noise_wrist_t.reshape(T, 2, 3).unsqueeze(0).expand(B, -1, -1, -1).clone()
            elif noise_wrist_t.ndim == 3 and noise_wrist_t.shape == (T, 2, 3):
                noise_wrist_t = noise_wrist_t.unsqueeze(0).expand(B, -1, -1, -1).clone()
            elif noise_wrist_t.ndim == 3 and noise_wrist_t.shape == (B, T, 6):
                noise_wrist_t = noise_wrist_t.reshape(B, T, 2, 3).clone()
        else:
            noise_wrist_t = _randn((B, T, 2, 3), out.device, generator) * sigma
        if layout and layout.wrist_reference == "torso_xy":
            noise_wrist_t[:, :, :, 2] = 0.0
            g_mask = (
                out[:, :, layout.slice_geometry_support] > 0.5
                if layout.slice_geometry_support is not None
                else m
            )
            wrists = torch.where(g_mask.unsqueeze(-1), wrists + noise_wrist_t, torch.zeros_like(wrists))
            wrists[:, :, :, 2] = 0.0
        else:
            wrists = torch.where(m.unsqueeze(-1), wrists + noise_wrist_t, torch.zeros_like(wrists))
        out[:, :, SLICE_GLOBAL_WRISTS] = wrists.reshape(B, T, 6)

    # 3. Recompute derived features
    if D >= 143:
        out = batch_recompute_derived_features(out, normalization_epsilon=normalization_epsilon)

    return out


def batch_shape_scale(
    x: torch.Tensor,
    r: float = 0.10,
    sx: Optional[Any] = None,
    sy: Optional[Any] = None,
    generator: Optional[torch.Generator] = None,
    normalization_epsilon: float = 1e-4,
) -> torch.Tensor:
    """Scale local hand shapes [0:126] for usable slots on batch (B, T, D)."""
    out = x.clone()
    B, T, D = out.shape
    if B == 0 or D < 126 or r <= 0.0:
        return out

    if sx is None:
        sx = _rand((B,), out.device, generator) * (2.0 * r) + (1.0 - r)
    else:
        sx = torch.as_tensor(sx, device=out.device, dtype=torch.float32)
        if sx.numel() == 1:
            sx = sx.reshape(1).expand(B)
        elif sx.shape != (B,):
            raise ValueError(f"sx must have shape ({B},) or be a scalar, got {sx.shape}")

    if sy is None:
        sy = _rand((B,), out.device, generator) * (2.0 * r) + (1.0 - r)
    else:
        sy = torch.as_tensor(sy, device=out.device, dtype=torch.float32)
        if sy.numel() == 1:
            sy = sy.reshape(1).expand(B)
        elif sy.shape != (B,):
            raise ValueError(f"sy must have shape ({B},) or be a scalar, got {sy.shape}")

    sx_m = sx.view(B, 1, 1, 1)
    sy_m = sy.view(B, 1, 1, 1)

    m = out[:, :, SLICE_HAND_MASKS] > 0.5 if D >= 143 else torch.ones((B, T, 2), dtype=torch.bool, device=out.device)
    m_s = m.unsqueeze(-1)

    shapes = out[:, :, SLICE_LOCAL_HAND_SHAPES].reshape(B, T, 2, 21, 3).clone()
    sh_x = shapes[:, :, :, :, 0] * sx_m
    sh_y = shapes[:, :, :, :, 1] * sy_m
    shapes[:, :, :, :, 0] = torch.where(m_s, sh_x, torch.zeros_like(sh_x))
    shapes[:, :, :, :, 1] = torch.where(m_s, sh_y, torch.zeros_like(sh_y))
    shapes[:, :, :, :, 2] = torch.where(m_s, shapes[:, :, :, :, 2], torch.zeros_like(shapes[:, :, :, :, 2]))
    out[:, :, SLICE_LOCAL_HAND_SHAPES] = shapes.reshape(B, T, 126)

    if D >= 143:
        out = batch_recompute_derived_features(out, normalization_epsilon=normalization_epsilon)

    return out


def batch_scale_translate(
    x: torch.Tensor,
    scale_min: float = 0.85,
    scale_max: float = 1.15,
    translate: float = 0.10,
    s: Optional[Any] = None,
    t: Optional[Any] = None,
    tx: Optional[Any] = None,
    ty: Optional[Any] = None,
    generator: Optional[torch.Generator] = None,
    normalization_epsilon: float = 1e-4,
) -> torch.Tensor:
    """Scale and translate global wrist coordinates on batch (B, T, D) and recompute motion."""
    out = x.clone()
    B, T, D = out.shape
    if B == 0 or D < 132:
        return out

    layout = infer_feature_layout_from_dim(D) if D >= 143 else None
    if layout and layout.wrist_reference == "torso_xy":
        raise ValueError(
            "batch_scale_translate is not supported when features.wrist_reference='torso_xy'. "
            "Wrist-only image scale and translation corrupts torso-relative coordinate geometry."
        )

    if t is not None:
        t_t = torch.as_tensor(t, device=out.device, dtype=torch.float32)
        if t_t.ndim == 2 and t_t.shape == (B, 2):
            tx, ty = t_t[:, 0], t_t[:, 1]
        elif t_t.numel() == 2:
            t_flat = t_t.flatten()
            tx, ty = t_flat[0], t_flat[1]
        elif isinstance(t, (tuple, list)) and len(t) == 2:
            tx, ty = t[0], t[1]
        else:
            raise ValueError(f"t tensor must have shape (2,) or ({B}, 2), got {t_t.shape}")

    if s is None:
        s = _rand((B,), out.device, generator) * (scale_max - scale_min) + scale_min
    else:
        s = torch.as_tensor(s, device=out.device, dtype=torch.float32)
        if s.numel() == 1:
            s = s.reshape(1).expand(B)
        elif s.shape != (B,):
            raise ValueError(f"s must have shape ({B},) or be a scalar, got {s.shape}")

    if tx is None:
        tx = _rand((B,), out.device, generator) * (2.0 * translate) - translate
    else:
        tx = torch.as_tensor(tx, device=out.device, dtype=torch.float32)
        if tx.numel() == 1:
            tx = tx.reshape(1).expand(B)
        elif tx.shape != (B,):
            raise ValueError(f"tx must have shape ({B},) or be a scalar, got {tx.shape}")

    if ty is None:
        ty = _rand((B,), out.device, generator) * (2.0 * translate) - translate
    else:
        ty = torch.as_tensor(ty, device=out.device, dtype=torch.float32)
        if ty.numel() == 1:
            ty = ty.reshape(1).expand(B)
        elif ty.shape != (B,):
            raise ValueError(f"ty must have shape ({B},) or be a scalar, got {ty.shape}")

    s_m = s.view(B, 1, 1)
    tx_m = tx.view(B, 1, 1)
    ty_m = ty.view(B, 1, 1)

    m = out[:, :, SLICE_HAND_MASKS] > 0.5 if D >= 143 else torch.ones((B, T, 2), dtype=torch.bool, device=out.device)

    wrists = out[:, :, SLICE_GLOBAL_WRISTS].reshape(B, T, 2, 3).clone()
    w_x = (wrists[:, :, :, 0] - 0.5) * s_m + 0.5 + tx_m
    w_y = (wrists[:, :, :, 1] - 0.5) * s_m + 0.5 + ty_m
    wrists[:, :, :, 0] = torch.where(m, w_x, torch.zeros_like(w_x))
    wrists[:, :, :, 1] = torch.where(m, w_y, torch.zeros_like(w_y))
    wrists[:, :, :, 2] = torch.where(m, wrists[:, :, :, 2], torch.zeros_like(wrists[:, :, :, 2]))
    out[:, :, SLICE_GLOBAL_WRISTS] = wrists.reshape(B, T, 6)

    if D >= 143:
        out = batch_recompute_derived_features(out, normalization_epsilon=normalization_epsilon)

    return out


def batch_time_stretch(
    x: torch.Tensor,
    max_stretch: float = 0.2,
    rates: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Time-stretch operation (explicitly disabled in Phase 0)."""
    if rates is not None:
        if not torch.allclose(rates, torch.ones_like(rates)):
            raise ValueError(
                "batch_time_stretch is disabled in Phase 0 because current implementations "
                "do not share a validity-aware temporal contract."
            )
        return x.clone()
    if max_stretch > 0.0:
        raise ValueError(
            "batch_time_stretch is disabled in Phase 0 because current implementations "
            "do not share a validity-aware temporal contract."
        )
    return x.clone()


def apply_augmentation_batch(
    x: torch.Tensor,
    cfg: Any = None,
    generator: Optional[torch.Generator] = None,
    normalization_epsilon: float = 1e-4,
) -> torch.Tensor:
    """Apply sequence of augmentations per sample in batch on device."""
    if cfg is None or x.shape[0] == 0:
        return x.clone()

    out = x.clone()
    B = out.shape[0]
    device = out.device

    flip_cfg = getattr(cfg, "flip", {}) if not isinstance(cfg, dict) else cfg.get("flip", {})
    rotate_cfg = getattr(cfg, "rotate", {}) if not isinstance(cfg, dict) else cfg.get("rotate", {})
    shape_scale_cfg = getattr(cfg, "shape_scale", {}) if not isinstance(cfg, dict) else cfg.get("shape_scale", {})
    scale_translate_cfg = getattr(cfg, "scale_translate", {}) if not isinstance(cfg, dict) else cfg.get("scale_translate", {})
    jitter_cfg = getattr(cfg, "jitter", {}) if not isinstance(cfg, dict) else cfg.get("jitter", {})
    stretch_cfg = getattr(cfg, "time_stretch", {}) if not isinstance(cfg, dict) else cfg.get("time_stretch", {})

    applied = False

    # 1. Flip
    if _get_aug_param(flip_cfg, "enabled", False):
        prob = float(_get_aug_param(flip_cfg, "prob", 0.5))
        swap_hands = bool(_get_aug_param(flip_cfg, "swap_hands", False))
        if prob > 0.0:
            mask = _rand((B,), device, generator) < prob
            indices = torch.nonzero(mask).squeeze(1)
            if len(indices) > 0:
                out[indices] = batch_horizontal_flip(
                    out[indices],
                    swap_hands=swap_hands,
                    normalization_epsilon=normalization_epsilon,
                )
                applied = True

    # 2. Rotate
    if _get_aug_param(rotate_cfg, "enabled", False):
        prob = float(_get_aug_param(rotate_cfg, "prob", 0.5))
        max_angle = float(_get_aug_param(rotate_cfg, "max_angle_deg", 8.0))
        if prob > 0.0 and max_angle > 0.0:
            mask = _rand((B,), device, generator) < prob
            indices = torch.nonzero(mask).squeeze(1)
            if len(indices) > 0:
                out[indices] = batch_random_rotate_2d(
                    out[indices],
                    max_angle_deg=max_angle,
                    generator=generator,
                    normalization_epsilon=normalization_epsilon,
                )
                applied = True

    # 3. Shape Scale
    if _get_aug_param(shape_scale_cfg, "enabled", False):
        prob = float(_get_aug_param(shape_scale_cfg, "prob", 0.5))
        r = float(_get_aug_param(shape_scale_cfg, "r", 0.10))
        if prob > 0.0 and r > 0.0:
            mask = _rand((B,), device, generator) < prob
            indices = torch.nonzero(mask).squeeze(1)
            if len(indices) > 0:
                out[indices] = batch_shape_scale(
                    out[indices],
                    r=r,
                    generator=generator,
                    normalization_epsilon=normalization_epsilon,
                )
                applied = True

    # 4. Scale & Translate
    if _get_aug_param(scale_translate_cfg, "enabled", False):
        prob = float(_get_aug_param(scale_translate_cfg, "prob", 0.7))
        scale_min = float(_get_aug_param(scale_translate_cfg, "scale_min", 0.85))
        scale_max = float(_get_aug_param(scale_translate_cfg, "scale_max", 1.15))
        translate = float(_get_aug_param(scale_translate_cfg, "translate", 0.10))
        if prob > 0.0:
            mask = _rand((B,), device, generator) < prob
            indices = torch.nonzero(mask).squeeze(1)
            if len(indices) > 0:
                out[indices] = batch_scale_translate(
                    out[indices],
                    scale_min=scale_min,
                    scale_max=scale_max,
                    translate=translate,
                    generator=generator,
                    normalization_epsilon=normalization_epsilon,
                )
                applied = True

    # 5. Jitter
    if _get_aug_param(jitter_cfg, "enabled", False):
        prob = float(_get_aug_param(jitter_cfg, "prob", 0.5))
        sigma = float(_get_aug_param(jitter_cfg, "sigma", 0.01))
        if prob > 0.0 and sigma > 0.0:
            mask = _rand((B,), device, generator) < prob
            indices = torch.nonzero(mask).squeeze(1)
            if len(indices) > 0:
                out[indices] = batch_add_jitter(
                    out[indices],
                    sigma=sigma,
                    generator=generator,
                    normalization_epsilon=normalization_epsilon,
                )
                applied = True

    # 6. Time Stretch (must fail if enabled with positive prob and stretch)
    if _get_aug_param(stretch_cfg, "enabled", False):
        prob = float(_get_aug_param(stretch_cfg, "prob", 0.5))
        max_stretch = float(_get_aug_param(stretch_cfg, "max_stretch", 0.2))
        if prob > 0.0 and max_stretch > 0.0:
            batch_time_stretch(out, max_stretch=max_stretch, generator=generator)
            applied = True

    # Final defensive recomputation and boundary invariant enforcement
    if applied:
        out = batch_recompute_derived_features(out, normalization_epsilon=normalization_epsilon)
        D = out.shape[-1]
        layout = infer_feature_layout_from_dim(D) if D >= 143 else None
        if layout and layout.wrist_reference == "torso_xy":
            out[:, :, [128, 131]] = 0.0
    return out
