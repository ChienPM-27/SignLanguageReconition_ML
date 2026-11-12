"""Compact face geometry shared by Parquet and raw/video inference.

Translation is removed using the eye midpoint. Scale is the median observed
outer-eye distance within the trimmed clip, estimated before gap interpolation.
Only XY is used: face/hand MediaPipe Z values do not share a metric depth origin.
"""
from __future__ import annotations

import numpy as np

from src.features.normalization import interpolate_bounded_gaps, resample_temporal
from src.features.schema import FACE_LANDMARK_INDICES, FACE_OPENNESS_PAIRS, NUM_FACE_FEATURE_DIMS


def face_openness(points: np.ndarray, masks: np.ndarray, epsilon: float) -> np.ndarray:
    """Mouth, first-eye, second-eye aperture/width ratios; invalid ratios are zero."""
    out = np.zeros((*points.shape[:-2], 3), dtype=np.float32)
    for column, (top, bottom, left, right) in enumerate(FACE_OPENNESS_PAIRS):
        width = np.linalg.norm(points[..., left, :] - points[..., right, :], axis=-1)
        aperture = np.linalg.norm(points[..., top, :] - points[..., bottom, :], axis=-1)
        valid = masks & np.isfinite(width) & np.isfinite(aperture) & (width > epsilon)
        np.divide(aperture, width, out=out[..., column], where=valid)
    return out


def prepare_face_sequence(
    coords: np.ndarray | None, mask: np.ndarray | None, raw_length: int,
    trim: tuple[int, int], target_length: int, max_gap_frames: int, epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce 36 channels and observed/usable masks without changing clip acceptance."""
    n_points = len(FACE_LANDMARK_INDICES)
    start, end = trim
    points = np.zeros((end - start, n_points, 2), dtype=np.float32)
    observed = np.zeros(end - start, dtype=bool)
    if coords is not None:
        if coords.ndim != 3 or len(coords) != raw_length or coords.shape[2] not in (2, 3):
            raise ValueError("face_coords must have shape (T, 16/468/478, 2/3) aligned with hands.")
        if coords.shape[1] == n_points:
            selected = coords[start:end, :, :2].astype(np.float32, copy=True)
        elif coords.shape[1] in (468, 478):
            selected = coords[start:end, FACE_LANDMARK_INDICES, :2].astype(np.float32, copy=True)
        else:
            raise ValueError("face_coords must contain the canonical 16 points or 468/478 MediaPipe points.")
        if mask is not None and np.asarray(mask).shape not in ((raw_length,), (raw_length, 1)):
            raise ValueError("face_mask must have shape (T,) or (T, 1).")
        observed = np.all(np.isfinite(selected), axis=(1, 2))
        if mask is not None:
            observed &= np.asarray(mask).reshape(-1)[start:end] > 0.5
        eye_width = np.linalg.norm(selected[:, 1] - selected[:, 2], axis=-1)
        observed &= eye_width > epsilon
        if observed.any():
            scale = float(np.median(eye_width[observed]))
            center = (selected[:, 1] + selected[:, 2]) / 2.0
            points = np.where(observed[:, None, None], (selected - center[:, None]) / scale, 0.0)
    filled, support = interpolate_bounded_gaps(points, observed, max_gap_frames)
    resampled, usable = resample_temporal(filled, support, target_length)
    observed_target = resample_temporal(points, observed, target_length)[1]
    features = np.zeros((target_length, NUM_FACE_FEATURE_DIMS), dtype=np.float32)
    features[:, :n_points * 2] = resampled.reshape(target_length, n_points * 2)
    features[:, n_points * 2:n_points * 2 + 3] = face_openness(resampled, usable.reshape(-1), epsilon)
    features[:, -1] = usable.reshape(-1)
    return features, observed_target.reshape(-1, 1), usable.reshape(-1, 1)
