"""Batched PyTorch tensor augmentation running directly on device (GPU / CPU).

Hỗ trợ các phép biến đổi song song theo batch trên tensor (B, T, D):
1. batch_horizontal_flip: Lật gương trục x, hỗ trợ tùy chọn swap_hands.
2. batch_random_rotate_2d: Xoay ngẫu nhiên độc lập từng mẫu trong [-max_angle, max_angle].
3. batch_add_jitter: Thêm nhiễu Gaussian nhẹ vào min(132, D) chiều tọa độ đầu.
4. batch_time_stretch: Co giãn tốc độ cử chỉ rate ~ U(1-max_stretch, 1+max_stretch)
   và nội suy tuyến tính theo thời gian.
5. apply_augmentation_batch: Áp dụng chuỗi biến đổi theo cấu hình AugmentationConfig / dict.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Tuple
import torch


def _rand(
    size: Tuple[int, ...],
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sinh số ngẫu nhiên đều [0, 1) tương thích generator cross-device."""
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
    """Sinh số ngẫu nhiên chuẩn N(0, 1) tương thích generator cross-device."""
    if generator is not None:
        if str(generator.device) == str(device):
            return torch.randn(size, generator=generator, device=device)
        return torch.randn(size, generator=generator, device=generator.device).to(device)
    return torch.randn(size, device=device)


def _get_aug_param(cfg_obj: Any, key: str, default: Any) -> Any:
    """Trích xuất tham số từ dict hoặc dataclass cấu hình."""
    if isinstance(cfg_obj, dict):
        val = cfg_obj.get(key)
    else:
        val = getattr(cfg_obj, key, None)
    return val if val is not None else default


def batch_horizontal_flip(x: torch.Tensor, swap_hands: bool = False) -> torch.Tensor:
    """Lật gương quanh trục x trên batch tensor shape (B, T, D).

    Args:
        x: Tensor shape (B, T, D).
        swap_hands: Nếu False, chỉ lật trục x và giữ nguyên slot 0 là tay trội.
                    Nếu True, hoán đổi slot tay 0 <-> tay 1.
    """
    out = x.clone()
    B, T, D = out.shape

    # 1. Local Shapes (126 dims: 63 dims per hand)
    if D >= 126:
        shapes = out[:, :, :126].reshape(B, T, 2, 21, 3).clone()
        shapes[:, :, :, :, 0] = -shapes[:, :, :, :, 0]
        if swap_hands:
            shapes = shapes[:, :, [1, 0], :, :]
        out[:, :, :126] = shapes.reshape(B, T, 126)

    # 2. Global Wrists (6 dims: 3 per hand)
    if D >= 132:
        wrists = out[:, :, 126:132].reshape(B, T, 2, 3).clone()
        wrists[:, :, :, 0] = 1.0 - wrists[:, :, :, 0]
        if swap_hands:
            wrists = wrists[:, :, [1, 0], :]
        out[:, :, 126:132] = wrists.reshape(B, T, 6)

    # 3. Relative Wrist Vector: rel = wrist_0 - wrist_1 (3 dims)
    if D >= 135:
        rel = out[:, :, 132:135].clone()
        if swap_hands:
            rel = -rel
            rel[:, :, 0] = -rel[:, :, 0]
        else:
            rel[:, :, 0] = -rel[:, :, 0]
        out[:, :, 132:135] = rel

    # 4. Wrist Velocities (6 dims: 3 per hand)
    if D >= 141:
        vel = out[:, :, 135:141].reshape(B, T, 2, 3).clone()
        vel[:, :, :, 0] = -vel[:, :, :, 0]
        if swap_hands:
            vel = vel[:, :, [1, 0], :]
        out[:, :, 135:141] = vel.reshape(B, T, 6)

    # 5. Mask (2 dims)
    if D >= 143:
        if swap_hands:
            out[:, :, 141:143] = out[:, :, [142, 141]]

    # 6. Pose landmarks nếu có (12 coords: left/right shoulder, left/right elbow)
    if D >= 155:
        pose = out[:, :, 143:155].reshape(B, T, 4, 3).clone()
        pose[:, :, :, 0] = 1.0 - pose[:, :, :, 0]
        pose_flipped = pose[:, :, [1, 0, 3, 2], :]
        out[:, :, 143:155] = pose_flipped.reshape(B, T, 12)

    return out


def batch_random_rotate_2d(
    x: torch.Tensor,
    max_angle_deg: float = 8.0,
    angles_deg: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Xoay ngẫu nhiên độc lập từng mẫu trong mặt phẳng (x, y) trên batch (B, T, D)."""
    out = x.clone()
    B, T, D = out.shape
    if B == 0 or max_angle_deg <= 0.0:
        return out

    if angles_deg is None:
        rand_u = _rand((B,), out.device, generator)
        angles_deg = rand_u * (2.0 * max_angle_deg) - max_angle_deg

    angles_rad = angles_deg * (math.pi / 180.0)
    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)

    # 1. Local shapes [0:126] -> (B, T, 2 tay, 21 điểm, 3) quanh gốc (0, 0)
    if D >= 126:
        shapes = out[:, :, :126].reshape(B, T, 2, 21, 3).clone()
        cos_m = cos_a.view(B, 1, 1, 1)
        sin_m = sin_a.view(B, 1, 1, 1)
        sx = shapes[:, :, :, :, 0].clone()
        sy = shapes[:, :, :, :, 1].clone()
        shapes[:, :, :, :, 0] = sx * cos_m - sy * sin_m
        shapes[:, :, :, :, 1] = sx * sin_m + sy * cos_m
        out[:, :, :126] = shapes.reshape(B, T, 126)

    # 2. Global wrists [126:132] -> (B, T, 2 tay, 3) quanh tâm (0.5, 0.5)
    if D >= 132:
        wrists = out[:, :, 126:132].reshape(B, T, 2, 3).clone()
        cos_w = cos_a.view(B, 1, 1)
        sin_w = sin_a.view(B, 1, 1)
        wx = wrists[:, :, :, 0].clone() - 0.5
        wy = wrists[:, :, :, 1].clone() - 0.5
        wrists[:, :, :, 0] = wx * cos_w - wy * sin_w + 0.5
        wrists[:, :, :, 1] = wx * sin_w + wy * cos_w + 0.5
        out[:, :, 126:132] = wrists.reshape(B, T, 6)

    return out


def batch_add_jitter(
    x: torch.Tensor,
    sigma: float = 0.01,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Thêm nhiễu Gaussian nhẹ vào min(132, D) chiều đầu trên batch (B, T, D)."""
    out = x.clone()
    B, T, D = out.shape
    if B == 0 or sigma <= 0.0:
        return out

    coord_dim = min(132, D)
    noise = _randn((B, T, coord_dim), out.device, generator) * sigma
    out[:, :, :coord_dim] = out[:, :, :coord_dim] + noise
    return out


def batch_time_stretch(
    x: torch.Tensor,
    max_stretch: float = 0.2,
    rates: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Co giãn trục thời gian với tốc độ rate ~ U(1-max_stretch, 1+max_stretch) và nội suy tuyến tính."""
    out = x.clone()
    B, T, D = out.shape
    if B == 0 or T <= 2 or max_stretch <= 0.0:
        return out

    if rates is None:
        rand_u = _rand((B,), out.device, generator)
        rates = rand_u * (2.0 * max_stretch) + (1.0 - max_stretch)

    t_grid = torch.arange(T, device=out.device, dtype=torch.float32)
    p = t_grid.unsqueeze(0) * rates.unsqueeze(1)
    p = torch.clamp(p, 0.0, float(T - 1))

    i0 = torch.floor(p).long()
    i1 = torch.clamp(i0 + 1, max=T - 1)
    alpha = (p - i0.float()).unsqueeze(-1)  # (B, T, 1)

    i0_expanded = i0.unsqueeze(-1).expand(B, T, D)
    i1_expanded = i1.unsqueeze(-1).expand(B, T, D)

    val0 = torch.gather(out, dim=1, index=i0_expanded)
    val1 = torch.gather(out, dim=1, index=i1_expanded)

    out = (1.0 - alpha) * val0 + alpha * val1
    return out


def apply_augmentation_batch(
    x: torch.Tensor,
    cfg: Any = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Áp dụng chuỗi biến đổi dữ liệu theo batch (B, T, D) trực tiếp trên device.

    Thứ tự áp dụng: Flip -> Rotate -> Jitter -> Time Stretch.
    Mỗi phép chọn độc lập theo từng mẫu qua mặt nạ Bernoulli(prob).
    """
    if cfg is None or x.shape[0] == 0:
        return x.clone()

    out = x.clone()
    B = out.shape[0]
    device = out.device

    flip_cfg = getattr(cfg, "flip", {}) if not isinstance(cfg, dict) else cfg.get("flip", {})
    rotate_cfg = getattr(cfg, "rotate", {}) if not isinstance(cfg, dict) else cfg.get("rotate", {})
    jitter_cfg = getattr(cfg, "jitter", {}) if not isinstance(cfg, dict) else cfg.get("jitter", {})
    stretch_cfg = getattr(cfg, "time_stretch", {}) if not isinstance(cfg, dict) else cfg.get("time_stretch", {})

    # 1. Flip
    if _get_aug_param(flip_cfg, "enabled", False):
        prob = float(_get_aug_param(flip_cfg, "prob", 0.5))
        swap_hands = bool(_get_aug_param(flip_cfg, "swap_hands", False))
        if prob > 0.0:
            mask = _rand((B,), device, generator) < prob
            indices = torch.nonzero(mask).squeeze(1)
            if len(indices) > 0:
                out[indices] = batch_horizontal_flip(out[indices], swap_hands=swap_hands)

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

    # 3. Jitter
    if _get_aug_param(jitter_cfg, "enabled", False):
        prob = float(_get_aug_param(jitter_cfg, "prob", 0.5))
        sigma = float(_get_aug_param(jitter_cfg, "sigma", 0.01))
        if prob > 0.0 and sigma > 0.0:
            mask = _rand((B,), device, generator) < prob
            indices = torch.nonzero(mask).squeeze(1)
            if len(indices) > 0:
                out[indices] = batch_add_jitter(out[indices], sigma=sigma, generator=generator)

    # 4. Time Stretch
    if _get_aug_param(stretch_cfg, "enabled", False):
        prob = float(_get_aug_param(stretch_cfg, "prob", 0.3))
        max_stretch = float(_get_aug_param(stretch_cfg, "max_stretch", 0.2))
        if prob > 0.0 and max_stretch > 0.0:
            mask = _rand((B,), device, generator) < prob
            indices = torch.nonzero(mask).squeeze(1)
            if len(indices) > 0:
                out[indices] = batch_time_stretch(
                    out[indices], max_stretch=max_stretch, generator=generator
                )

    return out
