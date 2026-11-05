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


def batch_recompute_derived_features(x: torch.Tensor) -> torch.Tensor:
    """Recompute relative wrist [132:135] and wrist velocity [135:141] on batch tensor (B, T, D).

    Gated on valid endpoints:
    - rel = wrist0 - wrist1 (only when BOTH hands present: mask0 > 0.5 & mask1 > 0.5, else 0.0)
    - velocity: vel[t] = w[t] - w[t-1] gated on adjacent valid frames, with vel[0] = 0.0.
    - If D >= 160: sanitize unsupported pose channels and gate torso velocity on adjacent valid frames.
    """
    out = x.clone()
    B, T, D = out.shape
    if D >= 143:
        m = out[:, :, SLICE_HAND_MASKS] > 0.5  # (B, T, 2)
        wrists = out[:, :, SLICE_GLOBAL_WRISTS].reshape(B, T, 2, 3)
        wrists = torch.where(m.unsqueeze(-1), wrists, torch.zeros_like(wrists))
        out[:, :, SLICE_GLOBAL_WRISTS] = wrists.reshape(B, T, 6)

        two_hand_mask = (m[:, :, 0] & m[:, :, 1]).unsqueeze(-1)
        rel_wrist = wrists[:, :, 0, :] - wrists[:, :, 1, :]
        out[:, :, SLICE_REL_WRIST] = torch.where(two_hand_mask, rel_wrist, torch.zeros_like(rel_wrist))

        wrist_vel = torch.zeros((B, T, 2, 3), dtype=out.dtype, device=out.device)
        if T > 1:
            vel_valid = (m[:, 1:, :] & m[:, :-1, :]).unsqueeze(-1)
            diff = wrists[:, 1:] - wrists[:, :-1]
            wrist_vel[:, 1:] = torch.where(vel_valid, diff, torch.zeros_like(diff))
        out[:, :, SLICE_WRIST_VEL] = wrist_vel.reshape(B, T, 6)

    if D >= 160:
        p = out[:, :, SLICE_POSE_MASK].squeeze(-1) > 0.5  # (B, T)
        p_u = p.unsqueeze(-1)
        out[:, :, SLICE_NORM_POSE] = torch.where(p_u, out[:, :, SLICE_NORM_POSE], torch.zeros_like(out[:, :, SLICE_NORM_POSE]))
        out[:, :, SLICE_SHOULDER_SCALE] = torch.where(p_u, out[:, :, SLICE_SHOULDER_SCALE], torch.zeros_like(out[:, :, SLICE_SHOULDER_SCALE]))
        u_valid = torch.zeros((B, T), dtype=torch.bool, device=out.device)
        if T > 1:
            u_valid[:, 1:] = p[:, 1:] & p[:, :-1]
        out[:, :, SLICE_TORSO_VEL] = torch.where(u_valid.unsqueeze(-1), out[:, :, SLICE_TORSO_VEL], torch.zeros_like(out[:, :, SLICE_TORSO_VEL]))

    return out


def batch_horizontal_flip(x: torch.Tensor, swap_hands: bool = False) -> torch.Tensor:
    """Horizontal mirror reflection on batch tensor (B, T, D).

    Corrected contract:
    - Local shapes negate x on usable slots.
    - Global wrists map x -> 1 - x on usable slots.
    - Torso-centered pose negates x and swaps anatomical sides [1, 0, 3, 2].
    - Torso velocity negates x on valid adjacent frames.
    - Unsupported slots strictly remain 0.
    """
    out = x.clone()
    B, T, D = out.shape

    m = out[:, :, SLICE_HAND_MASKS] > 0.5 if D >= 143 else torch.ones((B, T, 2), dtype=torch.bool, device=out.device)

    # 1. Local Shapes (126 dims)
    if D >= 126:
        shapes = out[:, :, SLICE_LOCAL_HAND_SHAPES].reshape(B, T, 2, 21, 3).clone()
        m_s = m.unsqueeze(-1)  # (B, T, 2, 1)
        shapes[:, :, :, :, 0] = torch.where(m_s, -shapes[:, :, :, :, 0], torch.zeros_like(shapes[:, :, :, :, 0]))
        shapes[:, :, :, :, 1:] = torch.where(m.unsqueeze(-1).unsqueeze(-1), shapes[:, :, :, :, 1:], torch.zeros_like(shapes[:, :, :, :, 1:]))
        if swap_hands:
            shapes = shapes[:, :, [1, 0], :, :]
        out[:, :, SLICE_LOCAL_HAND_SHAPES] = shapes.reshape(B, T, 126)

    # 2. Global Wrists (6 dims)
    if D >= 132:
        wrists = out[:, :, SLICE_GLOBAL_WRISTS].reshape(B, T, 2, 3).clone()
        wrists[:, :, :, 0] = torch.where(m, 1.0 - wrists[:, :, :, 0], torch.zeros_like(wrists[:, :, :, 0]))
        wrists[:, :, :, 1:] = torch.where(m.unsqueeze(-1), wrists[:, :, :, 1:], torch.zeros_like(wrists[:, :, :, 1:]))
        if swap_hands:
            wrists = wrists[:, :, [1, 0], :]
        out[:, :, SLICE_GLOBAL_WRISTS] = wrists.reshape(B, T, 6)

    # 3. Swap Hand Masks if swap_hands is requested
    if D >= 143 and swap_hands:
        out[:, :, SLICE_HAND_MASKS] = out[:, :, [142, 141]]

    # 4. Recompute derived features
    if D >= 143:
        out = batch_recompute_derived_features(out)

    # 5. Pose landmarks
    if D >= 160:
        p = out[:, :, SLICE_POSE_MASK].squeeze(-1) > 0.5  # (B, T)
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

        out[:, :, SLICE_SHOULDER_SCALE] = torch.where(p.unsqueeze(-1), out[:, :, SLICE_SHOULDER_SCALE], torch.zeros_like(out[:, :, SLICE_SHOULDER_SCALE]))

    return out


def batch_random_rotate_2d(
    x: torch.Tensor,
    max_angle_deg: float = 8.0,
    angles_deg: Optional[Any] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Rotate points in (x, y) plane on batch (B, T, D) and recompute motion."""
    out = x.clone()
    B, T, D = out.shape
    if B == 0 or max_angle_deg <= 0.0:
        return out

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

    # 2. Global wrists [126:132] -> (B, T, 2, 3)
    if D >= 132:
        wrists = out[:, :, SLICE_GLOBAL_WRISTS].reshape(B, T, 2, 3).clone()
        cos_w = cos_a.view(B, 1, 1)
        sin_w = sin_a.view(B, 1, 1)
        wx = wrists[:, :, :, 0].clone() - 0.5
        wy = wrists[:, :, :, 1].clone() - 0.5
        rot_wx = wx * cos_w - wy * sin_w + 0.5
        rot_wy = wx * sin_w + wy * cos_w + 0.5
        wrists[:, :, :, 0] = torch.where(m, rot_wx, torch.zeros_like(rot_wx))
        wrists[:, :, :, 1] = torch.where(m, rot_wy, torch.zeros_like(rot_wy))
        wrists[:, :, :, 2] = torch.where(m, wrists[:, :, :, 2], torch.zeros_like(wrists[:, :, :, 2]))
        out[:, :, SLICE_GLOBAL_WRISTS] = wrists.reshape(B, T, 6)

    # 3. Recompute derived features
    if D >= 143:
        out = batch_recompute_derived_features(out)

    # 4. Pose rotation [143:155] and torso velocity [155:158]
    if D >= 160:
        p = out[:, :, SLICE_POSE_MASK].squeeze(-1) > 0.5  # (B, T)
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

        out[:, :, SLICE_SHOULDER_SCALE] = torch.where(p.unsqueeze(-1), out[:, :, SLICE_SHOULDER_SCALE], torch.zeros_like(out[:, :, SLICE_SHOULDER_SCALE]))

    return out


def batch_add_jitter(
    x: torch.Tensor,
    sigma: float = 0.01,
    noise_shape: Optional[Any] = None,
    noise_wrist: Optional[Any] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Add Gaussian noise to shape and wrist on usable slots and recompute motion."""
    out = x.clone()
    B, T, D = out.shape
    if B == 0 or sigma <= 0.0:
        return out

    m = out[:, :, SLICE_HAND_MASKS] > 0.5 if D >= 143 else torch.ones((B, T, 2), dtype=torch.bool, device=out.device)

    # 1. Local shapes [0:126] -> (B, T, 2, 21, 3)
    if D >= 126:
        shapes = out[:, :, SLICE_LOCAL_HAND_SHAPES].reshape(B, T, 2, 21, 3).clone()
        if noise_shape is not None:
            noise_shape_t = torch.as_tensor(noise_shape, device=out.device, dtype=out.dtype).clone()
            if noise_shape_t.ndim == 2 and noise_shape_t.shape == (T, 126):
                noise_shape_t = noise_shape_t.reshape(T, 2, 21, 3).unsqueeze(0).expand(B, -1, -1, -1, -1).clone()
            elif noise_shape_t.ndim == 3 and noise_shape_t.shape == (B, T, 126):
                noise_shape_t = noise_shape_t.reshape(B, T, 2, 21, 3).clone()
            elif noise_shape_t.ndim == 4 and noise_shape_t.shape == (T, 2, 21, 3):
                noise_shape_t = noise_shape_t.unsqueeze(0).expand(B, -1, -1, -1, -1).clone()
        else:
            noise_shape_t = _randn((B, T, 2, 21, 3), out.device, generator) * sigma
        # Landmark 0 noise is identically zero
        noise_shape_t[:, :, :, 0, :] = 0.0
        m_s = m.unsqueeze(-1).unsqueeze(-1)
        shapes = torch.where(m_s, shapes + noise_shape_t, torch.zeros_like(shapes))
        out[:, :, SLICE_LOCAL_HAND_SHAPES] = shapes.reshape(B, T, 126)

    # 2. Global wrists [126:132] -> (B, T, 2, 3)
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
        wrists = torch.where(m.unsqueeze(-1), wrists + noise_wrist_t, torch.zeros_like(wrists))
        out[:, :, SLICE_GLOBAL_WRISTS] = wrists.reshape(B, T, 6)

    # 3. Recompute derived features
    if D >= 143:
        out = batch_recompute_derived_features(out)

    return out


def batch_shape_scale(
    x: torch.Tensor,
    r: float = 0.10,
    sx: Optional[Any] = None,
    sy: Optional[Any] = None,
    generator: Optional[torch.Generator] = None,
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
) -> torch.Tensor:
    """Scale and translate global wrist coordinates on batch (B, T, D) and recompute motion."""
    out = x.clone()
    B, T, D = out.shape
    if B == 0 or D < 132:
        return out

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
        out = batch_recompute_derived_features(out)

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
                out[indices] = batch_horizontal_flip(out[indices], swap_hands=swap_hands)
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
                    out[indices], max_angle_deg=max_angle, generator=generator
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
                out[indices] = batch_shape_scale(out[indices], r=r, generator=generator)
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
                out[indices] = batch_add_jitter(out[indices], sigma=sigma, generator=generator)
                applied = True

    # 6. Time Stretch (must fail if enabled with positive prob and stretch)
    if _get_aug_param(stretch_cfg, "enabled", False):
        prob = float(_get_aug_param(stretch_cfg, "prob", 0.5))
        max_stretch = float(_get_aug_param(stretch_cfg, "max_stretch", 0.2))
        if prob > 0.0 and max_stretch > 0.0:
            batch_time_stretch(out, max_stretch=max_stretch, generator=generator)
            applied = True

    # Final defensive recomputation
    if applied:
        out = batch_recompute_derived_features(out)
    return out
