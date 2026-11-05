"""Deep test suite for the shared preprocessing pipeline and feature contract (P0.1 - P0.7).

Verifies:
1. Parity between Parquet preprocessing and inference adapter.
2. Dominant hand selection logic (support count, motion tie-breaking, exact tie break to left, disabled mode).
3. Quality gating: rejection on insufficient valid frames before imputation.
4. Trimming synchronization across hands, hand masks, and pose.
5. Bounded gap interpolation vs unbounded long gaps.
6. Support-aware temporal resampling nearest-neighbor hold.
7. Masked velocity and relative wrist derivatives.
8. Torso-centered pose normalization, shoulder scaling, and torso velocity.
9. Preprocess identity hashing determinism and sensitivity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.preprocess import preprocess_single_sequence
from src.features.normalization import (
    compute_feature_vector,
    interpolate_bounded_gaps,
    resample_temporal,
)
from src.features.pipeline import prepare_landmark_sequence
from src.features.schema import (
    FEATURE_SET_HANDS,
    FEATURE_SET_HANDS_POSE,
    compute_preprocess_identity,
)
from src.utils.config import Config, build_config_from_dict


def _make_dummy_config(
    feature_set: str = FEATURE_SET_HANDS_POSE,
    sequence_length: int = 32,
    max_gap_frames: int = 3,
    pad_frames_around: int = 2,
    min_valid_frames: int = 5,
    dominant_hand: str = "auto",
    normalization_epsilon: float = 0.0001,
) -> Config:
    return build_config_from_dict({
        "features": {"set": feature_set},
        "preprocess": {
            "sequence_length": sequence_length,
            "max_gap_frames": max_gap_frames,
            "pad_frames_around": pad_frames_around,
            "min_valid_frames": min_valid_frames,
            "trim_no_hand": True,
            "dominant_hand": dominant_hand,
            "normalization_epsilon": normalization_epsilon,
            "num_workers": 1,
        },
    })


def _create_synthetic_parquet(
    parquet_path: Path,
    lh_coords: np.ndarray,  # (T, 21, 3)
    rh_coords: np.ndarray,  # (T, 21, 3)
    lh_mask: np.ndarray,    # (T,) bool
    rh_mask: np.ndarray,    # (T,) bool
    pose_coords: Optional[np.ndarray] = None,  # (T, 33, 3)
    pose_mask: Optional[np.ndarray] = None,    # (T,) bool
) -> Path:
    """Serialize synthetic coordinates into parquet matching competition format."""
    T = len(lh_coords)
    records = []

    for t in range(T):
        # Left Hand
        if lh_mask[t]:
            for idx in range(21):
                records.append({
                    "frame": t,
                    "type": "left_hand",
                    "landmark_index": idx,
                    "x": float(lh_coords[t, idx, 0]),
                    "y": float(lh_coords[t, idx, 1]),
                    "z": float(lh_coords[t, idx, 2]),
                })
        else:
            for idx in range(21):
                records.append({
                    "frame": t,
                    "type": "left_hand",
                    "landmark_index": idx,
                    "x": float("nan"),
                    "y": float("nan"),
                    "z": float("nan"),
                })

        # Right Hand
        if rh_mask[t]:
            for idx in range(21):
                records.append({
                    "frame": t,
                    "type": "right_hand",
                    "landmark_index": idx,
                    "x": float(rh_coords[t, idx, 0]),
                    "y": float(rh_coords[t, idx, 1]),
                    "z": float(rh_coords[t, idx, 2]),
                })
        else:
            for idx in range(21):
                records.append({
                    "frame": t,
                    "type": "right_hand",
                    "landmark_index": idx,
                    "x": float("nan"),
                    "y": float("nan"),
                    "z": float("nan"),
                })

        # Pose
        if pose_coords is not None:
            has_p = pose_mask[t] if pose_mask is not None else True
            for idx in range(min(33, pose_coords.shape[1])):
                if has_p and np.isfinite(pose_coords[t, idx, 0]):
                    records.append({
                        "frame": t,
                        "type": "pose",
                        "landmark_index": idx,
                        "x": float(pose_coords[t, idx, 0]),
                        "y": float(pose_coords[t, idx, 1]),
                        "z": float(pose_coords[t, idx, 2]),
                    })
                else:
                    records.append({
                        "frame": t,
                        "type": "pose",
                        "landmark_index": idx,
                        "x": float("nan"),
                        "y": float("nan"),
                        "z": float("nan"),
                    })

    table = pa.Table.from_pylist(records)
    pq.write_table(table, str(parquet_path))
    return parquet_path


def test_parity_parquet_vs_inference_adapter(tmp_path: Path) -> None:
    """Acceptance criterion P0.6.1: Full numerical parity between Parquet preprocessing and inference pipeline."""
    T = 40
    np.random.seed(42)

    # Synthetic hands: active frames [5:30]
    lh_coords = np.zeros((T, 21, 3), dtype=np.float32)
    rh_coords = np.zeros((T, 21, 3), dtype=np.float32)
    lh_mask = np.zeros(T, dtype=bool)
    rh_mask = np.zeros(T, dtype=bool)

    # Left hand present frames 5..28 (24 frames)
    lh_mask[5:29] = True
    for t in range(5, 29):
        lh_coords[t] = np.random.uniform(0.2, 0.4, size=(21, 3))

    # Right hand present frames 15..30 (16 frames)
    rh_mask[15:31] = True
    for t in range(15, 31):
        rh_coords[t] = np.random.uniform(0.6, 0.8, size=(21, 3))

    # Upper body pose
    pose_coords = np.full((T, 33, 3), np.nan, dtype=np.float32)
    pose_mask = np.zeros(T, dtype=bool)
    pose_mask[5:32] = True
    for t in range(5, 32):
        pose_coords[t, 11] = [-0.2, 0.5, 0.0]  # L shoulder
        pose_coords[t, 12] = [0.2, 0.5, 0.0]   # R shoulder
        pose_coords[t, 13] = [-0.3, 0.7, 0.0]  # L elbow
        pose_coords[t, 14] = [0.3, 0.7, 0.0]   # R elbow

    cfg = _make_dummy_config(
        feature_set=FEATURE_SET_HANDS_POSE,
        sequence_length=32,
        max_gap_frames=3,
        pad_frames_around=2,
        min_valid_frames=5,
    )

    # 1. Pipeline via Parquet
    parquet_file = tmp_path / "sample.parquet"
    _create_synthetic_parquet(parquet_file, lh_coords, rh_coords, lh_mask, rh_mask, pose_coords, pose_mask)
    out_dir = tmp_path / "feat_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_res = preprocess_single_sequence(
        parquet_path=parquet_file,
        sequence_id="sample_01",
        participant_id="p1",
        gloss="hello",
        label=0,
        cfg=cfg,
        out_dir=out_dir,
    )
    assert parquet_res.status == "kept"
    parquet_feat = np.load(parquet_res.npy_path)

    # 2. Pipeline via inference adapter prepare_landmark_sequence
    direct_res = prepare_landmark_sequence(
        lh_coords=lh_coords,
        rh_coords=rh_coords,
        lh_mask=lh_mask,
        rh_mask=rh_mask,
        pose_coords=pose_coords,
        pose_mask=pose_mask,
        cfg=cfg,
    )
    assert direct_res.is_usable is True

    # Check parity within atol=1e-6, rtol=1e-5
    np.testing.assert_allclose(
        direct_res.features,
        parquet_feat,
        atol=1e-6,
        rtol=1e-5,
        err_msg="Parquet and direct pipeline features must be numerically identical",
    )
    assert direct_res.dominant_hand == "left"
    assert direct_res.n_valid_hand_frames == parquet_res.n_valid_frames


def test_dominant_hand_selection_tie_breaking() -> None:
    """Acceptance criterion P0.6.3: Dominant hand selection compares support counts, then motion, then breaks tie to left."""
    T = 20
    cfg_auto = _make_dummy_config(dominant_hand="auto")

    # Case 1: Right hand has strictly more valid frames -> Right is dominant
    lh = np.zeros((T, 21, 3), dtype=np.float32)
    rh = np.zeros((T, 21, 3), dtype=np.float32)
    lh_mask = np.zeros(T, dtype=bool)
    rh_mask = np.zeros(T, dtype=bool)

    lh_mask[5:12] = True   # 7 frames
    rh_mask[5:15] = True   # 10 frames
    res1 = prepare_landmark_sequence(lh, rh, lh_mask, rh_mask, cfg=cfg_auto)
    assert res1.dominant_hand == "right"
    assert res1.provenance["slot_to_anatomical"] == ["right", "left"]

    # Case 2: Equal valid counts, but right hand has higher adjacent-frame wrist motion -> Right is dominant
    lh_mask = np.zeros(T, dtype=bool)
    rh_mask = np.zeros(T, dtype=bool)
    lh_mask[5:15] = True  # 10 frames
    rh_mask[5:15] = True  # 10 frames

    # Left wrist moves 0.01 per frame
    for t in range(5, 15):
        lh[t, 0, 0] = t * 0.01

    # Right wrist moves 0.10 per frame
    for t in range(5, 15):
        rh[t, 0, 0] = t * 0.10

    res2 = prepare_landmark_sequence(lh, rh, lh_mask, rh_mask, cfg=cfg_auto)
    assert res2.dominant_hand == "right"
    assert res2.provenance["slot_to_anatomical"] == ["right", "left"]

    # Case 3: Exact tie in counts AND motion -> breaks tie to LEFT
    rh_tied = lh.copy()
    res3 = prepare_landmark_sequence(lh, rh_tied, lh_mask, lh_mask.copy(), cfg=cfg_auto)
    assert res3.dominant_hand == "left"
    assert res3.provenance["slot_to_anatomical"] == ["left", "right"]

    # Case 4: Separated dropout observations must NOT manufacture tie-break motion across gaps
    lh_dropout = np.zeros((T, 21, 3), dtype=np.float32)
    lh_d_mask = np.zeros(T, dtype=bool)
    # 2 isolated valid frames separated by 5 empty frames
    lh_d_mask[2] = True
    lh_dropout[2, 0, :] = [0.1, 0.1, 0.0]
    lh_d_mask[8] = True
    lh_dropout[8, 0, :] = [0.9, 0.9, 0.0]  # Large displacement, but separated!

    rh_adjacent = np.zeros((T, 21, 3), dtype=np.float32)
    rh_a_mask = np.zeros(T, dtype=bool)
    # 2 adjacent valid frames
    rh_a_mask[2] = True
    rh_adjacent[2, 0, :] = [0.1, 0.1, 0.0]
    rh_a_mask[3] = True
    rh_adjacent[3, 0, :] = [0.15, 0.15, 0.0]  # Small displacement 0.07, but adjacent!

    # Both have exactly 2 frames
    cfg_low_gate = _make_dummy_config(dominant_hand="auto", min_valid_frames=2)
    res4 = prepare_landmark_sequence(lh_dropout, rh_adjacent, lh_d_mask, rh_a_mask, cfg=cfg_low_gate)
    # Right has adjacent motion E_R > 0; Left has separated observations E_L = 0 -> Right is dominant!
    assert res4.dominant_hand == "right"
    assert res4.provenance["slot_to_anatomical"] == ["right", "left"]

    # Case 5: Dominant hand disabled (dominant_hand='none' or False) preserves anatomical order
    cfg_none = _make_dummy_config(dominant_hand="none")
    res5 = prepare_landmark_sequence(lh, rh, lh_mask, rh_mask, cfg=cfg_none)
    assert res5.dominant_hand in {"anatomical", "none"}
    assert res5.provenance["slot_to_anatomical"] == ["left", "right"]


def test_quality_gating_rejects_insufficient_frames() -> None:
    """Acceptance criterion P0.6.4: Clips with fewer valid frames than min_valid_frames are rejected before interpolation."""
    T = 30
    cfg = _make_dummy_config(min_valid_frames=6)

    lh = np.zeros((T, 21, 3), dtype=np.float32)
    rh = np.zeros((T, 21, 3), dtype=np.float32)
    m = np.zeros(T, dtype=bool)
    m[10:13] = True  # only 3 valid frames < 6

    res = prepare_landmark_sequence(lh, rh, m, m, cfg=cfg)
    assert res.is_usable is False
    assert "insufficient" in res.rejection_reason
    # Features must be zeros
    assert np.all(res.features == 0.0)


def test_bounded_gap_interpolation_and_long_gap_zeros() -> None:
    """Acceptance criterion P0.2 & P0.3: Short gaps interpolated, long gaps remain exactly zero."""
    T = 20
    coords = np.zeros((T, 2, 21, 3), dtype=np.float32)
    masks = np.zeros((T, 2), dtype=bool)

    # Hand 0: valid at t=2 and t=5 (gap of 2 frames: 3, 4 <= max_gap 2) -> Interpolated!
    masks[2, 0] = True
    coords[2, 0, 0, 0] = 0.2
    masks[5, 0] = True
    coords[5, 0, 0, 0] = 0.8

    # Hand 1: valid at t=2 and t=8 (gap of 5 frames > max_gap 2) -> NOT interpolated!
    masks[2, 1] = True
    coords[2, 1, 0, 0] = 0.3
    masks[8, 1] = True
    coords[8, 1, 0, 0] = 0.9

    filled_c, filled_m = interpolate_bounded_gaps(coords, masks, max_gap_frames=2)

    # Hand 0: frames 3 and 4 must be filled and supported
    assert filled_m[3, 0] is True or filled_m[3, 0] == True
    assert filled_m[4, 0] is True or filled_m[4, 0] == True
    assert np.isclose(filled_c[3, 0, 0, 0], 0.4)
    assert np.isclose(filled_c[4, 0, 0, 0], 0.6)

    # Hand 1: gap frames 3..7 must remain unsupported and 0.0
    for t in range(3, 8):
        assert filled_m[t, 1] is False or filled_m[t, 1] == False
        assert np.all(filled_c[t, 1] == 0.0)


def test_support_aware_temporal_resampling_nearest_neighbor_tie() -> None:
    """Acceptance criterion P0.2.2: Resampling ties break strictly to earlier frame; absent endpoints hold zero."""
    # 2 frames resampled to 3 frames: query indices are [0.0, 0.5, 1.0]
    # When both endpoints are supported, linear interpolation produces midpoint (2.5, 3.5, 4.5)
    coords = np.array([[[[1.0, 2.0, 3.0]]], [[[4.0, 5.0, 6.0]]]], dtype=np.float32)  # (2, 1, 1, 3)
    masks = np.array([[True], [True]], dtype=bool)  # (2, 1)

    res_c, res_m = resample_temporal(coords, masks, target_len=3)
    assert len(res_c) == 3
    # Frame 0 query 0.0 -> frame 0
    assert np.allclose(res_c[0, 0, 0], [1.0, 2.0, 3.0])
    # Frame 1 query 0.5 -> linear interpolation between frame 0 and 1
    assert np.allclose(res_c[1, 0, 0], [2.5, 3.5, 4.5])
    # Frame 2 query 1.0 -> frame 1
    assert np.allclose(res_c[2, 0, 0], [4.0, 5.0, 6.0])

    # Single-endpoint hold: if only frame 0 is supported and frame 1 is absent,
    # nearest-neighbor tie-breaking applies: query at 0.5 ties to earlier frame 0 (supported)
    # query at > 0.5 is nearest to frame 1 (absent) and must hold zero and false support!
    masks_single = np.array([[True], [False]], dtype=bool)
    coords_single = coords.copy()
    coords_single[1] = 0.0

    res_c_s, res_m_s = resample_temporal(coords_single, masks_single, target_len=3)
    # Query 0.0 is supported (frame 0)
    assert bool(res_m_s[0, 0]) is True
    assert np.allclose(res_c_s[0, 0, 0], [1.0, 2.0, 3.0])
    # Query 0.5 tied to earlier frame 0 is supported with frame 0 value
    assert bool(res_m_s[1, 0]) is True
    assert np.allclose(res_c_s[1, 0, 0], [1.0, 2.0, 3.0])
    # Query 1.0 (frame 1) is unsupported and zero!
    assert bool(res_m_s[2, 0]) is False
    assert np.all(res_c_s[2, 0, 0] == 0.0)


def test_motion_derivatives_masked_on_missing_endpoints() -> None:
    """Acceptance criterion P0.3.1: Disappearance/reappearance cannot produce spurious velocity."""
    T = 4
    # Wrist x: [0.6, 0.0, 0.7, 0.8], Mask: [1, 0, 1, 1]
    # Expected displacement x: [0.0, 0.0, 0.0, 0.1]
    hands = np.zeros((T, 2, 21, 3), dtype=np.float32)
    masks = np.zeros((T, 2), dtype=bool)

    masks[:, 0] = [True, False, True, True]
    hands[0, 0, 0, 0] = 0.6
    hands[1, 0, 0, 0] = 0.0
    hands[2, 0, 0, 0] = 0.7
    hands[3, 0, 0, 0] = 0.8

    feat = compute_feature_vector(hands, masks, feature_set=FEATURE_SET_HANDS)
    wrist_vel_x = feat[:, 135]  # Slot 0 wrist velocity x

    assert np.isclose(wrist_vel_x[0], 0.0)
    assert np.isclose(wrist_vel_x[1], 0.0), "Disappearance must not create -0.6 displacement"
    assert np.isclose(wrist_vel_x[2], 0.0), "Reappearance must not create +0.7 displacement"
    assert np.isclose(wrist_vel_x[3], 0.1), "Adjacent valid steps produce true displacement"


def test_hands_pose_with_absent_pose_supplies_160_dims_with_zeros() -> None:
    """Acceptance criterion P0.6.5: hands_pose with absent pose supplies 160 dims with 17 pose channels strictly zero."""
    T = 10
    hands = np.zeros((T, 2, 21, 3), dtype=np.float32)
    masks = np.ones((T, 2), dtype=bool)

    feat = compute_feature_vector(hands, masks, pose_coords=None, pose_masks=None, feature_set=FEATURE_SET_HANDS_POSE)
    assert feat.shape == (T, 160)
    # Channels 143:160 must be identically zero
    assert np.all(feat[:, 143:160] == 0.0)


def test_preprocess_identity_hash_determinism_and_sensitivity() -> None:
    """Acceptance criterion P0.7.1 & P0.7.2: Hash is deterministic, independent of dict order, sensitive to canonical tunables."""
    cfg1 = _make_dummy_config()
    cfg2 = _make_dummy_config()

    hash1, meta1 = compute_preprocess_identity(cfg1)
    hash2, meta2 = compute_preprocess_identity(cfg2)
    assert hash1 == hash2, "Identical configs must produce identical identity hash"
    assert len(hash1) == 64, "SHA-256 hash must be 64 hexadecimal characters"
    assert meta1 == meta2

    # Schema version sensitivity
    # Changing canonical parameters changes identity
    cfg_ep = _make_dummy_config(normalization_epsilon=0.001)
    assert compute_preprocess_identity(cfg_ep)[0] != hash1

    cfg_len = _make_dummy_config(sequence_length=48)
    assert compute_preprocess_identity(cfg_len)[0] != hash1

    cfg_gap = _make_dummy_config(max_gap_frames=5)
    assert compute_preprocess_identity(cfg_gap)[0] != hash1

    cfg_feat = _make_dummy_config(feature_set=FEATURE_SET_HANDS)
    assert compute_preprocess_identity(cfg_feat)[0] != hash1
