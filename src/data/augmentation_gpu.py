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


def batch_recompute_derived_features(x: torch.Tensor) -> torch.Tensor:
    """Tính lại relative wrist [132:135] và wrist velocity [135:141] trên batch tensor (B, T, D).

    Đúng công thức chuẩn hóa của compute_feature_vector_143:
    - rel = wrist0 - wrist1 (chỉ khi CẢ hai tay có mặt: mask0 > 0.5 & mask1 > 0.5, else 0.0)
    - velocity: vel[t] = w[t] - w[t-1] với vel[0] = 0.0 (không mask).
    """
    out = x.clone()
    B, T, D = out.shape
    if D >= 143:
        wrists = out[:, :, 126:132].reshape(B, T, 2, 3)
        two_hand_mask = ((out[:, :, 141] > 0.5) & (out[:, :, 142] > 0.5)).unsqueeze(-1)
        rel_wrist = wrists[:, :, 0, :] - wrists[:, :, 1, :]
        out[:, :, 132:135] = torch.where(two_hand_mask, rel_wrist, torch.zeros_like(rel_wrist))

        wrist_vel = torch.zeros((B, T, 2, 3), dtype=out.dtype, device=out.device)
        if T > 1:
            wrist_vel[:, 1:] = wrists[:, 1:] - wrists[:, :-1]
        out[:, :, 135:141] = wrist_vel.reshape(B, T, 6)
    return out


def batch_horizontal_flip(x: torch.Tensor, swap_hands: bool = False) -> torch.Tensor:
    """Lật gương quanh trục x trên batch tensor shape (B, T, D).

    Args:
        x: Tensor shape (B, T, D).
        swap_hands: Nếu False, chỉ lật trục x và giữ nguyên slot 0 là tay trội.
                    Nếu True, hoán đổi slot tay 0 <-> tay 1.
    """
    out = x.clone()
    B, T, D = out.shape

    mask = out[:, :, 141:143] > 0.5 if D >= 143 else torch.ones((B, T, 2), dtype=torch.bool, device=out.device)

    # 1. Local Shapes (126 dims: 63 dims per hand)
    if D >= 126:
        shapes = out[:, :, :126].reshape(B, T, 2, 21, 3).clone()
        m_s = mask.unsqueeze(-1)  # (B, T, 2, 1)
        shapes[:, :, :, :, 0] = torch.where(m_s, -shapes[:, :, :, :, 0], shapes[:, :, :, :, 0])
        if swap_hands:
            shapes = shapes[:, :, [1, 0], :, :]
        out[:, :, :126] = shapes.reshape(B, T, 126)

    # 2. Global Wrists (6 dims: 3 per hand)
    if D >= 132:
        wrists = out[:, :, 126:132].reshape(B, T, 2, 3).clone()
        wrists[:, :, :, 0] = torch.where(mask, 1.0 - wrists[:, :, :, 0], wrists[:, :, :, 0])
        if swap_hands:
            wrists = wrists[:, :, [1, 0], :]
        out[:, :, 126:132] = wrists.reshape(B, T, 6)

    # 3. Mask (2 dims)
    if D >= 143 and swap_hands:
        out[:, :, 141:143] = out[:, :, [142, 141]]

    # 4. Recompute derived features (rel [132:135] & vel [135:141])
    if D >= 143:
        out = batch_recompute_derived_features(out)

    # 5. Pose landmarks nếu có (12 coords: left/right shoulder, left/right elbow)
    if D >= 155:
        pose = out[:, :, 143:155].reshape(B, T, 4, 3).clone()
        pose[:, :, :, 0] = 1.0 - pose[:, :, :, 0]
        pose_flipped = pose[:, :, [1, 0, 3, 2], :]
        out[:, :, 143:155] = pose_flipped.reshape(B, T, 12)

    return out


def batch_random_rotate_2d(
    x: torch.Tensor,
    max_angle_deg: float = 8.0,
    angles_deg: Optional[Any] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Xoay ngẫu nhiên độc lập từng mẫu trong mặt phẳng (x, y) trên batch (B, T, D).

    Wrists xoay quanh (0.5, 0.5) CHỈ cho tay có mặt (mask > 0.5).
    Không recompute velocity.
    """
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

    mask = out[:, :, 141:143] > 0.5 if D >= 143 else torch.ones((B, T, 2), dtype=torch.bool, device=out.device)

    # 1. Local shapes [0:126] -> (B, T, 2 tay, 21 điểm, 3) quanh gốc (0, 0)
    if D >= 126:
        shapes = out[:, :, :126].reshape(B, T, 2, 21, 3).clone()
        cos_m = cos_a.view(B, 1, 1, 1)
        sin_m = sin_a.view(B, 1, 1, 1)
        sx = shapes[:, :, :, :, 0].clone()
        sy = shapes[:, :, :, :, 1].clone()
        rot_x = sx * cos_m - sy * sin_m
        rot_y = sx * sin_m + sy * cos_m
        m_s = mask.unsqueeze(-1)  # (B, T, 2, 1)
        shapes[:, :, :, :, 0] = torch.where(m_s, rot_x, sx)
        shapes[:, :, :, :, 1] = torch.where(m_s, rot_y, sy)
        out[:, :, :126] = shapes.reshape(B, T, 126)

    # 2. Global wrists [126:132] -> (B, T, 2 tay, 3) quanh tâm (0.5, 0.5) CHỈ cho tay có mặt
    if D >= 132:
        wrists = out[:, :, 126:132].reshape(B, T, 2, 3).clone()
        cos_w = cos_a.view(B, 1, 1)
        sin_w = sin_a.view(B, 1, 1)
        wx = wrists[:, :, :, 0].clone() - 0.5
        wy = wrists[:, :, :, 1].clone() - 0.5
        rot_wx = wx * cos_w - wy * sin_w + 0.5
        rot_wy = wx * sin_w + wy * cos_w + 0.5
        wrists[:, :, :, 0] = torch.where(mask, rot_wx, wrists[:, :, :, 0])
        wrists[:, :, :, 1] = torch.where(mask, rot_wy, wrists[:, :, :, 1])
        out[:, :, 126:132] = wrists.reshape(B, T, 6)

    return out


def batch_add_jitter(
    x: torch.Tensor,
    sigma: float = 0.01,
    noise_shape: Optional[Any] = None,
    noise_wrist: Optional[Any] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Thêm nhiễu Gaussian nhẹ vào shape và wrist CHỈ cho slot có mặt trên batch (B, T, D).

    Không recompute velocity.
    """
    out = x.clone()
    B, T, D = out.shape
    if B == 0 or sigma <= 0.0:
        return out

    mask = out[:, :, 141:143] > 0.5 if D >= 143 else torch.ones((B, T, 2), dtype=torch.bool, device=out.device)

    # 1. Local shapes [0:126] -> (B, T, 2, 21, 3)
    if D >= 126:
        shapes = out[:, :, :126].reshape(B, T, 2, 21, 3).clone()
        if noise_shape is not None:
            noise_shape_t = torch.as_tensor(noise_shape, device=out.device, dtype=out.dtype)
            if noise_shape_t.ndim == 4 and noise_shape_t.shape == (T, 2, 21, 3):
                noise_shape_t = noise_shape_t.unsqueeze(0)
        else:
            noise_shape_t = _randn((B, T, 2, 21, 3), out.device, generator) * sigma
        m_s = mask.unsqueeze(-1).unsqueeze(-1)  # (B, T, 2, 1, 1)
        shapes = torch.where(m_s, shapes + noise_shape_t, shapes)
        out[:, :, :126] = shapes.reshape(B, T, 126)

    # 2. Global wrists [126:132] -> (B, T, 2, 3)
    if D >= 132:
        wrists = out[:, :, 126:132].reshape(B, T, 2, 3).clone()
        if noise_wrist is not None:
            noise_wrist_t = torch.as_tensor(noise_wrist, device=out.device, dtype=out.dtype)
            if noise_wrist_t.ndim == 2 and noise_wrist_t.shape == (T, 6):
                noise_wrist_t = noise_wrist_t.reshape(T, 2, 3).unsqueeze(0)
            elif noise_wrist_t.ndim == 3 and noise_wrist_t.shape == (T, 2, 3):
                noise_wrist_t = noise_wrist_t.unsqueeze(0)
        else:
            noise_wrist_t = _randn((B, T, 2, 3), out.device, generator) * sigma
        wrists = torch.where(mask.unsqueeze(-1), wrists + noise_wrist_t, wrists)
        out[:, :, 126:132] = wrists.reshape(B, T, 6)

    return out


def batch_shape_scale(
    x: torch.Tensor,
    r: float = 0.10,
    sx: Optional[Any] = None,
    sy: Optional[Any] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Co giãn độc lập trục x, y của local hand shapes [0:126] trên batch (B, T, D) cho tay có mặt.

    sx, sy ~ U(1 - r, 1 + r).
    """
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

    mask = out[:, :, 141:143] > 0.5 if D >= 143 else torch.ones((B, T, 2), dtype=torch.bool, device=out.device)
    m_s = mask.unsqueeze(-1)  # (B, T, 2, 1)

    shapes = out[:, :, :126].reshape(B, T, 2, 21, 3).clone()
    sh_x = shapes[:, :, :, :, 0] * sx_m
    sh_y = shapes[:, :, :, :, 1] * sy_m
    shapes[:, :, :, :, 0] = torch.where(m_s, sh_x, shapes[:, :, :, :, 0])
    shapes[:, :, :, :, 1] = torch.where(m_s, sh_y, shapes[:, :, :, :, 1])
    out[:, :, :126] = shapes.reshape(B, T, 126)

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
    """Co giãn và dịch chuyển cổ tay tuyệt đối (xy) trên batch (B, T, D) cho tay có mặt.

    wrist_xy' = (wrist_xy - 0.5) * s + 0.5 + (tx, ty)
    Sau đó batch_recompute_derived_features.
    """
    out = x.clone()
    B, T, D = out.shape
    if B == 0 or D < 132:
        return out

    if t is not None:
        if isinstance(t, (tuple, list)):
            if len(t) != 2:
                raise ValueError(f"t must have 2 elements (tx, ty), got {len(t)}")
            tx, ty = t[0], t[1]
        else:
            t_t = torch.as_tensor(t, device=out.device, dtype=torch.float32)
            if t_t.numel() == 2:
                t_flat = t_t.flatten()
                tx, ty = t_flat[0], t_flat[1]
            elif t_t.ndim == 2 and t_t.shape == (B, 2):
                tx, ty = t_t[:, 0], t_t[:, 1]
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

    mask = out[:, :, 141:143] > 0.5 if D >= 143 else torch.ones((B, T, 2), dtype=torch.bool, device=out.device)

    wrists = out[:, :, 126:132].reshape(B, T, 2, 3).clone()
    w_x = (wrists[:, :, :, 0] - 0.5) * s_m + 0.5 + tx_m
    w_y = (wrists[:, :, :, 1] - 0.5) * s_m + 0.5 + ty_m
    wrists[:, :, :, 0] = torch.where(mask, w_x, wrists[:, :, :, 0])
    wrists[:, :, :, 1] = torch.where(mask, w_y, wrists[:, :, :, 1])
    out[:, :, 126:132] = wrists.reshape(B, T, 6)

    if D >= 143:
        out = batch_recompute_derived_features(out)

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

    Thứ tự áp dụng: Flip -> Rotate -> Shape Scale -> Scale Translate -> Jitter -> Time Stretch.
    Mỗi phép chọn độc lập theo từng mẫu qua mặt nạ Bernoulli(prob).
    """
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

    # 3. Shape Scale
    if _get_aug_param(shape_scale_cfg, "enabled", False):
        prob = float(_get_aug_param(shape_scale_cfg, "prob", 0.5))
        r = float(_get_aug_param(shape_scale_cfg, "r", 0.10))
        if prob > 0.0 and r > 0.0:
            mask = _rand((B,), device, generator) < prob
            indices = torch.nonzero(mask).squeeze(1)
            if len(indices) > 0:
                out[indices] = batch_shape_scale(out[indices], r=r, generator=generator)

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

    # 5. Jitter
    if _get_aug_param(jitter_cfg, "enabled", False):
        prob = float(_get_aug_param(jitter_cfg, "prob", 0.5))
        sigma = float(_get_aug_param(jitter_cfg, "sigma", 0.01))
        if prob > 0.0 and sigma > 0.0:
            mask = _rand((B,), device, generator) < prob
            indices = torch.nonzero(mask).squeeze(1)
            if len(indices) > 0:
                out[indices] = batch_add_jitter(out[indices], sigma=sigma, generator=generator)

    # 6. Time Stretch
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
