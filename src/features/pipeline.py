"""Shared preprocessing pipeline for both training and inference.

Guarantees identical feature extraction across Parquet files, extracted videos, and webcam inputs.
Enforces:
- Consistent dominant hand selection
- Quality gating on minimum valid frames
- Synchronized idle trimming across hands and pose
- Pre-imputation estimation of clip median palm scale and shoulder width
- Bounded gap imputation
- Temporal resampling on a shared uniform grid
- Exact mathematical feature extraction with gated derivatives
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.features.normalization import (
    compute_clip_palm_scales,
    compute_clip_shoulder_width,
    compute_feature_vector,
    compute_resampled_transitions,
    interpolate_bounded_gaps,
    resample_temporal,
)
from src.features.schema import (
    FEATURE_SET_HANDS_POSE,
    FeatureLayout,
    get_feature_dim,
)
from src.features.face import prepare_face_sequence


@dataclass
class PreprocessResult:
    """Structured result of landmark sequence preprocessing."""

    features: np.ndarray  # (sequence_length, feature_dim) float32
    hand_masks: np.ndarray  # (sequence_length, 2) bool
    pose_mask: np.ndarray  # (sequence_length, 1) bool
    n_valid_hand_frames: int
    hand_coverage: float
    dominant_hand: str
    trim_indices: Tuple[int, int]
    is_usable: bool
    rejection_reason: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)


def prepare_landmark_sequence(
    lh_coords: Optional[np.ndarray] = None,
    rh_coords: Optional[np.ndarray] = None,
    lh_mask: Optional[np.ndarray] = None,
    rh_mask: Optional[np.ndarray] = None,
    pose_coords: Optional[np.ndarray] = None,
    pose_mask: Optional[np.ndarray] = None,
    cfg: Optional[Any] = None,
    sequence_length: int = 32,
    max_gap_frames: int = 3,
    pad_frames_around: int = 2,
    min_valid_frames: int = 3,
    trim_no_hand: bool = True,
    dominant_hand: str = "auto",
    feature_set: str = FEATURE_SET_HANDS_POSE,
    wrist_reference: str = "image",
    palm_scale: str = "frame",
    fingertip_displacement: bool = False,
    bend_cosines: bool = False,
    normalization_epsilon: float = 0.0001,
    raw_coords: Optional[np.ndarray] = None,
    hand_masks: Optional[np.ndarray] = None,
    pose_masks: Optional[np.ndarray] = None,
    face_coords: Optional[np.ndarray] = None,
    face_mask: Optional[np.ndarray] = None,
    face: bool = False,
) -> PreprocessResult:
    """Execute end-to-end landmark preprocessing according to canonical contract.

    Supports:
    1. Separate anatomical hands: lh_coords (T, 21, 3), rh_coords (T, 21, 3)
    2. Combined raw arrays: raw_coords (T, 2, 21, 3), hand_masks (T, 2) [slot 0: Left, slot 1: Right]
    """
    if raw_coords is not None:
        if raw_coords.ndim == 4 and raw_coords.shape[1] == 2:
            lh_coords = raw_coords[:, 0]
            rh_coords = raw_coords[:, 1]
        elif raw_coords.ndim == 3 and raw_coords.shape[1] == 42:
            lh_coords = raw_coords[:, :21]
            rh_coords = raw_coords[:, 21:42]

    if hand_masks is not None:
        if hand_masks.ndim == 2 and hand_masks.shape[1] >= 2:
            lh_mask = hand_masks[:, 0]
            rh_mask = hand_masks[:, 1]

    if pose_masks is not None and pose_mask is None:
        pose_mask = pose_masks

    if cfg is not None:
        prep = getattr(cfg, "preprocess", cfg)
        feat = getattr(cfg, "features", cfg)
        sequence_length = int(getattr(prep, "sequence_length", sequence_length))
        max_gap_frames = int(getattr(prep, "max_gap_frames", max_gap_frames))
        pad_frames_around = int(getattr(prep, "pad_frames_around", pad_frames_around))
        min_valid_frames = int(getattr(prep, "min_valid_frames", min_valid_frames))
        trim_no_hand = bool(getattr(prep, "trim_no_hand", trim_no_hand))
        dominant_hand = str(getattr(prep, "dominant_hand", dominant_hand))
        normalization_epsilon = float(getattr(prep, "normalization_epsilon", normalization_epsilon))
        feature_set = str(getattr(feat, "set", feature_set))
        wrist_reference = str(getattr(feat, "wrist_reference", wrist_reference))
        palm_scale = str(getattr(feat, "palm_scale", palm_scale))
        fingertip_displacement = bool(getattr(feat, "fingertip_displacement", fingertip_displacement))
        bend_cosines = bool(getattr(feat, "bend_cosines", bend_cosines))
        face = bool(getattr(feat, "face", face))

    target_dim = get_feature_dim(
        feature_set=feature_set,
        wrist_reference=wrist_reference,
        fingertip_displacement=fingertip_displacement,
        bend_cosines=bend_cosines,
        face=face,
    )

    if lh_coords is None or rh_coords is None or len(lh_coords) == 0 or len(rh_coords) != len(lh_coords):
        return PreprocessResult(
            features=np.zeros((sequence_length, target_dim), dtype=np.float32),
            hand_masks=np.zeros((sequence_length, 2), dtype=bool),
            pose_mask=np.zeros((sequence_length, 1), dtype=bool),
            n_valid_hand_frames=0,
            hand_coverage=0.0,
            dominant_hand="none",
            trim_indices=(0, 0),
            is_usable=False,
            rejection_reason="empty_or_mismatched_sequence",
        )

    T_raw = len(lh_coords)

    if T_raw == 0 or len(rh_coords) != T_raw:
        return PreprocessResult(
            features=np.zeros((sequence_length, target_dim), dtype=np.float32),
            hand_masks=np.zeros((sequence_length, 2), dtype=bool),
            pose_mask=np.zeros((sequence_length, 1), dtype=bool),
            n_valid_hand_frames=0,
            hand_coverage=0.0,
            dominant_hand="none",
            trim_indices=(0, 0),
            is_usable=False,
            rejection_reason="empty_or_mismatched_sequence",
        )

    # 1. Observation masks & coordinate sanitization
    lh_finite = np.all(np.isfinite(lh_coords), axis=(1, 2))
    rh_finite = np.all(np.isfinite(rh_coords), axis=(1, 2))

    o_lh = (lh_mask.flatten() > 0.5 if lh_mask is not None else np.ones(T_raw, dtype=bool)) & lh_finite
    o_rh = (rh_mask.flatten() > 0.5 if rh_mask is not None else np.ones(T_raw, dtype=bool)) & rh_finite

    lh_clean = np.where(o_lh[:, np.newaxis, np.newaxis], lh_coords, 0.0)
    rh_clean = np.where(o_rh[:, np.newaxis, np.newaxis], rh_coords, 0.0)

    # 2. Dominant hand detection by observed counts and motion energy
    n_L = int(np.sum(o_lh))
    n_R = int(np.sum(o_rh))
    E_L = 0.0
    E_R = 0.0
    if T_raw > 1:
        valid_adj_L = o_lh[1:] & o_lh[:-1]
        if np.any(valid_adj_L):
            diff_L = lh_clean[1:, 0, :] - lh_clean[:-1, 0, :]
            E_L = float(np.sum(np.linalg.norm(diff_L[valid_adj_L], axis=-1)))
        valid_adj_R = o_rh[1:] & o_rh[:-1]
        if np.any(valid_adj_R):
            diff_R = rh_clean[1:, 0, :] - rh_clean[:-1, 0, :]
            E_R = float(np.sum(np.linalg.norm(diff_R[valid_adj_R], axis=-1)))

    right_is_dominant = (n_R > n_L) or (n_R == n_L and E_R > E_L)
    use_dominant = dominant_hand not in (False, "false", "disabled", "none")

    if use_dominant:
        if right_is_dominant:
            hands_coords = np.stack([rh_clean, lh_clean], axis=1)  # Slot 0 = RH, Slot 1 = LH
            hands_obs = np.stack([o_rh, o_lh], axis=1)
            dom_str = "right"
            slot_to_anatomical = ["right", "left"]
        else:
            hands_coords = np.stack([lh_clean, rh_clean], axis=1)  # Slot 0 = LH, Slot 1 = RH
            hands_obs = np.stack([o_lh, o_rh], axis=1)
            dom_str = "left"
            slot_to_anatomical = ["left", "right"]
    else:
        hands_coords = np.stack([lh_clean, rh_clean], axis=1)
        hands_obs = np.stack([o_lh, o_rh], axis=1)
        dom_str = "anatomical"
        slot_to_anatomical = ["left", "right"]

    # 3. Quality gate: count observed frames with any hand active
    has_any_hand = hands_obs.any(axis=1)
    valid_frame_indices = np.where(has_any_hand)[0]
    n_valid_hand_frames = len(valid_frame_indices)
    hand_coverage = float(n_valid_hand_frames / T_raw)

    if n_valid_hand_frames < min_valid_frames:
        return PreprocessResult(
            features=np.zeros((sequence_length, target_dim), dtype=np.float32),
            hand_masks=np.zeros((sequence_length, 2), dtype=bool),
            pose_mask=np.zeros((sequence_length, 1), dtype=bool),
            n_valid_hand_frames=n_valid_hand_frames,
            hand_coverage=hand_coverage,
            dominant_hand=dom_str,
            trim_indices=(0, T_raw),
            is_usable=False,
            rejection_reason=f"insufficient_hand_frames: {n_valid_hand_frames} < {min_valid_frames}",
        )

    # 4. Trimming: slice hands and pose identically
    if trim_no_hand and n_valid_hand_frames > 0:
        start_t = max(0, int(valid_frame_indices[0]) - pad_frames_around)
        end_t = min(T_raw, int(valid_frame_indices[-1]) + pad_frames_around + 1)
    else:
        start_t = 0
        end_t = T_raw

    hands_trimmed = hands_coords[start_t:end_t]
    hands_obs_trimmed = hands_obs[start_t:end_t]

    # Pre-imputation estimation: clip median palm scales
    clip_palm_scales = compute_clip_palm_scales(
        hands_trimmed, hands_obs_trimmed, normalization_epsilon
    )

    # 5. Pose processing on same trimmed window
    if feature_set == FEATURE_SET_HANDS_POSE and pose_coords is not None and len(pose_coords) > 0:
        trimmed_pose = pose_coords[start_t:end_t]
        if trimmed_pose.shape[1] == 33:
            pose_selected = trimmed_pose[:, [11, 12, 13, 14], :].copy()
        elif trimmed_pose.shape[1] == 4:
            pose_selected = trimmed_pose.copy()
        else:
            pose_selected = np.zeros((len(hands_trimmed), 4, 3), dtype=np.float32)

        if pose_mask is not None and len(pose_mask) > 0:
            trimmed_p_mask = pose_mask[start_t:end_t].flatten() > 0.5
        else:
            trimmed_p_mask = np.ones(len(hands_trimmed), dtype=bool)

        finite_pose = np.all(np.isfinite(pose_selected), axis=(1, 2))
        l_sh = pose_selected[:, 0, :]
        r_sh = pose_selected[:, 1, :]
        shoulder_dist = np.linalg.norm(l_sh - r_sh, axis=-1)

        o_pose = trimmed_p_mask & finite_pose & (shoulder_dist > normalization_epsilon)
        pose_clean = np.where(o_pose[:, np.newaxis, np.newaxis], pose_selected, 0.0)

        # Pre-imputation estimation: clip shoulder width ell on validated observations o_pose
        clip_shoulder_width = compute_clip_shoulder_width(
            pose_clean, o_pose, normalization_epsilon
        )
    else:
        pose_clean = np.zeros((len(hands_trimmed), 4, 3), dtype=np.float32)
        o_pose = np.zeros(len(hands_trimmed), dtype=bool)
        clip_shoulder_width = 0.0

    # Policy for torso_xy wrist reference: strictly requires valid torso geometry
    if wrist_reference == "torso_xy":
        if clip_shoulder_width <= normalization_epsilon or np.sum(o_pose) == 0:
            layout = FeatureLayout.build(
                feature_set=feature_set,
                wrist_reference=wrist_reference,
                palm_scale=palm_scale,
                fingertip_displacement=fingertip_displacement,
                bend_cosines=bend_cosines,
                face=face,
            )
            return PreprocessResult(
                features=np.zeros((sequence_length, layout.total_dim), dtype=np.float32),
                hand_masks=np.zeros((sequence_length, 2), dtype=bool),
                pose_mask=np.zeros((sequence_length, 1), dtype=bool),
                n_valid_hand_frames=n_valid_hand_frames,
                hand_coverage=hand_coverage,
                dominant_hand=dom_str,
                trim_indices=(start_t, end_t),
                is_usable=False,
                rejection_reason="Missing or degenerate torso pose geometry required for wrist_reference='torso_xy'",
                provenance={
                    "n_raw_frames": T_raw,
                    "n_L": n_L,
                    "n_R": n_R,
                    "E_L": round(E_L, 4),
                    "E_R": round(E_R, 4),
                    "slot_to_anatomical": slot_to_anatomical,
                    "clip_palm_scales": clip_palm_scales,
                    "clip_shoulder_width": clip_shoulder_width,
                },
            )

    # 6. Bounded interpolation
    hands_filled, hands_support = interpolate_bounded_gaps(
        coords=hands_trimmed,
        masks=hands_obs_trimmed,
        max_gap_frames=max_gap_frames,
    )
    if feature_set == FEATURE_SET_HANDS_POSE:
        pose_filled, pose_support = interpolate_bounded_gaps(
            coords=pose_clean,
            masks=o_pose[:, np.newaxis],
            max_gap_frames=max_gap_frames,
        )
    else:
        pose_filled = None
        pose_support = None

    # 7. Resample to fixed sequence_length on a single uniform grid
    resampled_hands, resampled_hand_masks = resample_temporal(
        coords=hands_filled,
        masks=hands_support,
        target_len=sequence_length,
    )
    hand_transitions = compute_resampled_transitions(
        source_masks=hands_support,
        target_len=sequence_length,
    )

    if feature_set == FEATURE_SET_HANDS_POSE and pose_filled is not None and pose_support is not None:
        resampled_pose, resampled_pose_masks = resample_temporal(
            coords=pose_filled,
            masks=pose_support,
            target_len=sequence_length,
        )
        pose_transitions = compute_resampled_transitions(
            source_masks=pose_support,
            target_len=sequence_length,
        )
    else:
        resampled_pose = None
        resampled_pose_masks = None
        pose_transitions = None

    # 8. Feature extraction with canonical equations
    features = compute_feature_vector(
        hands_coords=resampled_hands,
        hands_masks=resampled_hand_masks,
        pose_coords=resampled_pose,
        pose_masks=resampled_pose_masks,
        feature_set=feature_set,
        wrist_reference=wrist_reference,
        palm_scale=palm_scale,
        fingertip_displacement=fingertip_displacement,
        bend_cosines=bend_cosines,
        clip_palm_scales=clip_palm_scales,
        clip_shoulder_width=clip_shoulder_width,
        normalization_epsilon=normalization_epsilon,
        hand_transitions=hand_transitions,
        pose_transitions=pose_transitions,
    )
    face_observed = face_usable = np.zeros((sequence_length, 1), dtype=bool)
    if face:
        face_features, face_observed, face_usable = prepare_face_sequence(
            face_coords, face_mask, T_raw, (start_t, end_t), sequence_length,
            max_gap_frames, normalization_epsilon,
        )
        features = np.concatenate((features, face_features), axis=1)

    out_pose_mask = (
        resampled_pose_masks
        if resampled_pose_masks is not None
        else np.zeros((sequence_length, 1), dtype=bool)
    )
    if out_pose_mask.ndim == 1:
        out_pose_mask = out_pose_mask[:, np.newaxis]

    # Post-resample Quality Gate (B-04): ensure sequence retains required target support
    target_hand_support = int(np.sum(resampled_hand_masks))
    is_usable = True
    rejection_reason = ""
    if target_hand_support == 0:
        is_usable = False
        rejection_reason = "zero_target_hand_support"
    elif feature_set == FEATURE_SET_HANDS_POSE and wrist_reference == "torso_xy":
        layout = FeatureLayout.build(feature_set, wrist_reference, palm_scale,
                                     fingertip_displacement, bend_cosines)
        if layout.slice_geometry_support is not None:
            geom_support = features[:, layout.slice_geometry_support]
            if np.sum(geom_support) == 0:
                is_usable = False
                rejection_reason = "zero_target_geometry_support"

    return PreprocessResult(
        features=features,
        hand_masks=resampled_hand_masks,
        pose_mask=out_pose_mask,
        n_valid_hand_frames=n_valid_hand_frames,
        hand_coverage=hand_coverage,
        dominant_hand=dom_str,
        trim_indices=(start_t, end_t),
        is_usable=is_usable,
        rejection_reason=rejection_reason,
        provenance={
            "n_raw_frames": T_raw,
            "n_L": n_L,
            "n_R": n_R,
            "E_L": round(E_L, 4),
            "E_R": round(E_R, 4),
            "slot_to_anatomical": slot_to_anatomical,
            "clip_palm_scales": clip_palm_scales,
            "clip_shoulder_width": round(clip_shoulder_width, 4) if clip_shoulder_width is not None else 0.0,
            "hand_transitions": hand_transitions,
            "pose_transitions": pose_transitions if pose_transitions is not None else np.zeros((sequence_length, 1), dtype=bool),
            "observed_hands": resample_temporal(hands_trimmed, hands_obs_trimmed, sequence_length)[1],
            "usable_hands": resampled_hand_masks,
            "observed_pose": resample_temporal(pose_clean, o_pose[:, None], sequence_length)[1],
            "usable_pose": out_pose_mask,
            "valid_timesteps": np.ones((sequence_length, 1), dtype=bool),
            "observed_face": face_observed,
            "usable_face": face_usable,
            "target_face_coverage": float(face_usable.mean()),
            "target_hand_coverage": float(resampled_hand_masks.any(axis=1).mean()),
            "target_pose_coverage": float(out_pose_mask.mean()),
        },
    )
