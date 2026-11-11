"""Comprehensive test suite for Wave B: Feature semantics, body-relative geometry,
fingertip displacement, articulation bend cosines, and layout-aware augmentations.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.augmentation import (
    horizontal_flip,
    scale_translate,
)
from src.data.augmentation_gpu import (
    batch_horizontal_flip,
    batch_scale_translate,
)
from src.features.normalization import (
    compute_bend_cosines,
    compute_clip_palm_scales,
    compute_clip_shoulder_width,
    compute_feature_vector,
    compute_fingertip_displacements,
)
from src.features.schema import (
    FeatureLayout,
    compute_preprocess_identity,
    get_feature_dim,
    infer_feature_layout_from_dim,
)
from src.utils.config import FeaturesConfig, load_config


# =========================================================================
# B0: Canonical Options & Layout Registry Tests
# =========================================================================

def test_feature_dimensions_registry() -> None:
    """Verify exact feature vector dimensions across all canonical Wave B combinations."""
    # 1. Base hands-only
    assert get_feature_dim("hands") == 143
    # 2. Base hands_pose image wrists
    assert get_feature_dim("hands_pose", wrist_reference="image") == 160
    # 3. Torso-relative wrists (160 + 2 geom support)
    assert get_feature_dim("hands_pose", wrist_reference="torso_xy") == 162
    # 4. Hands_pose image + fingertips (160 + 30)
    assert get_feature_dim("hands_pose", wrist_reference="image", fingertip_displacement=True) == 190
    # 5. Torso-relative + fingertips (162 + 30)
    assert get_feature_dim("hands_pose", wrist_reference="torso_xy", fingertip_displacement=True) == 192
    # 6. Hands_pose image + fingertips + bends (190 + 20)
    assert get_feature_dim(
        "hands_pose", wrist_reference="image", fingertip_displacement=True, bend_cosines=True
    ) == 210
    # 7. Full Wave B layout: torso-relative + fingertips + bends (192 + 20)
    assert get_feature_dim(
        "hands_pose", wrist_reference="torso_xy", fingertip_displacement=True, bend_cosines=True
    ) == 212


def test_feature_layout_slices() -> None:
    """Verify that FeatureLayout builds contiguous, non-overlapping slices matching total_dim."""
    for fset in ["hands", "hands_pose"]:
        for wref in ["image", "torso_xy"] if fset == "hands_pose" else ["image"]:
            for tips in [False, True]:
                for bends in [False, True]:
                    layout = FeatureLayout.build(
                        feature_set=fset,
                        wrist_reference=wref,
                        fingertip_displacement=tips,
                        bend_cosines=bends,
                    )
                    dim = layout.total_dim
                    assert layout.slice_local_hand_shapes == slice(0, 126)
                    assert layout.slice_wrists == slice(126, 132)
                    assert layout.slice_rel_wrist == slice(132, 135)
                    assert layout.slice_wrist_vel == slice(135, 141)
                    assert layout.slice_hand_masks == slice(141, 143)

                    if fset == "hands_pose":
                        assert layout.slice_norm_pose == slice(143, 155)
                        assert layout.slice_torso_vel == slice(155, 158)
                        assert layout.slice_shoulder_scale == slice(158, 159)
                        assert layout.slice_pose_mask == slice(159, 160)
                        if wref == "torso_xy":
                            assert layout.slice_geometry_support == slice(160, 162)
                        else:
                            assert layout.slice_geometry_support is None
                    else:
                        assert layout.slice_norm_pose is None
                        assert layout.slice_geometry_support is None

                    if tips:
                        assert layout.slice_fingertip_displacement is not None
                        assert (layout.slice_fingertip_displacement.stop - layout.slice_fingertip_displacement.start) == 30
                    if bends:
                        assert layout.slice_bend_cosines is not None
                        assert (layout.slice_bend_cosines.stop - layout.slice_bend_cosines.start) == 20
                        assert layout.slice_bend_cosines.stop == dim


def test_features_config_validation() -> None:
    """Verify strict validation of features configuration options."""
    # Valid
    cfg_f = FeaturesConfig(set="hands_pose", wrist_reference="torso_xy", palm_scale="clip_median")
    cfg_f.validate()

    # torso_xy requires hands_pose
    with pytest.raises(ValueError, match="requires features.set='hands_pose'"):
        FeaturesConfig(set="hands", wrist_reference="torso_xy").validate()

    # Invalid wrist_reference
    with pytest.raises(ValueError, match="Invalid features.wrist_reference"):
        FeaturesConfig(set="hands_pose", wrist_reference="camera_3d").validate()

    # Invalid palm_scale
    with pytest.raises(ValueError, match="Invalid features.palm_scale"):
        FeaturesConfig(set="hands", palm_scale="running_mean").validate()

    # Conflicting scale_translate with torso_xy in root Config
    cfg = load_config()
    cfg.features.set = "hands_pose"
    cfg.features.wrist_reference = "torso_xy"
    cfg.augmentation.scale_translate["enabled"] = True
    with pytest.raises(ValueError, match="augmentation.scale_translate cannot be enabled"):
        cfg.validate()


def test_preprocess_identity_wave_b_sensitivity() -> None:
    """Verify that Wave B features change preprocessing identity deterministically."""
    cfg = load_config()
    base_id, _ = compute_preprocess_identity(cfg)

    cfg.features.wrist_reference = "torso_xy"
    id_torso, _ = compute_preprocess_identity(cfg)
    assert id_torso != base_id

    cfg.features.palm_scale = "clip_median"
    id_palm, _ = compute_preprocess_identity(cfg)
    assert id_palm != id_torso

    cfg.features.fingertip_displacement = True
    id_tips, _ = compute_preprocess_identity(cfg)
    assert id_tips != id_palm

    cfg.features.bend_cosines = True
    id_bends, _ = compute_preprocess_identity(cfg)
    assert id_bends != id_tips


# =========================================================================
# B1: Body-Relative Wrists & Robust Scale Tests
# =========================================================================

def test_body_relative_wrist_translation_invariance() -> None:
    """Rigid camera translation of hands and shoulders must leave body-relative wrists invariant."""
    T = 16
    hands = np.zeros((T, 2, 21, 3), dtype=np.float32)
    # Slot 0 wrist at (0.4, 0.6, 0.0), Slot 1 wrist at (0.6, 0.6, 0.0)
    hands[:, 0, 0, :2] = [0.4, 0.6]
    hands[:, 1, 0, :2] = [0.6, 0.6]
    # Palm landmark 9 offset by 0.1
    hands[:, 0, 9, :2] = [0.4, 0.7]
    hands[:, 1, 9, :2] = [0.6, 0.7]
    h_mask = np.ones((T, 2), dtype=bool)

    pose = np.zeros((T, 4, 3), dtype=np.float32)
    # L_shoulder at (0.35, 0.4), R_shoulder at (0.65, 0.4) -> width = 0.3, center = (0.5, 0.4)
    pose[:, 0, :2] = [0.35, 0.4]
    pose[:, 1, :2] = [0.65, 0.4]
    p_mask = np.ones((T, 1), dtype=bool)

    sh_width = compute_clip_shoulder_width(pose, p_mask)
    assert abs(sh_width - 0.3) < 1e-5

    feats1 = compute_feature_vector(
        hands_coords=hands,
        hands_masks=h_mask,
        pose_coords=pose,
        pose_masks=p_mask,
        feature_set="hands_pose",
        wrist_reference="torso_xy",
        clip_shoulder_width=sh_width,
    )

    # Translate both hands and shoulders by (+0.12, -0.08)
    delta = np.array([0.12, -0.08, 0.0], dtype=np.float32)
    hands_shifted = hands + delta
    pose_shifted = pose + delta

    sh_width_shifted = compute_clip_shoulder_width(pose_shifted, p_mask)
    feats2 = compute_feature_vector(
        hands_coords=hands_shifted,
        hands_masks=h_mask,
        pose_coords=pose_shifted,
        pose_masks=p_mask,
        feature_set="hands_pose",
        wrist_reference="torso_xy",
        clip_shoulder_width=sh_width_shifted,
    )

    layout = infer_feature_layout_from_dim(feats1.shape[1])
    # Body-relative wrists must be exactly invariant
    w1 = feats1[:, layout.slice_wrists]
    w2 = feats2[:, layout.slice_wrists]
    np.testing.assert_allclose(w1, w2, atol=1e-5)


def test_body_relative_wrist_scale_invariance() -> None:
    """Uniform camera scaling of hands and shoulders must leave body-relative wrists invariant."""
    T = 16
    hands = np.zeros((T, 2, 21, 3), dtype=np.float32)
    hands[:, 0, 0, :2] = [0.4, 0.6]
    hands[:, 1, 0, :2] = [0.6, 0.6]
    hands[:, 0, 9, :2] = [0.4, 0.7]
    hands[:, 1, 9, :2] = [0.6, 0.7]
    h_mask = np.ones((T, 2), dtype=bool)

    pose = np.zeros((T, 4, 3), dtype=np.float32)
    pose[:, 0, :2] = [0.35, 0.4]
    pose[:, 1, :2] = [0.65, 0.4]
    p_mask = np.ones((T, 1), dtype=bool)

    sh_width1 = compute_clip_shoulder_width(pose, p_mask)
    feats1 = compute_feature_vector(
        hands_coords=hands,
        hands_masks=h_mask,
        pose_coords=pose,
        pose_masks=p_mask,
        feature_set="hands_pose",
        wrist_reference="torso_xy",
        clip_shoulder_width=sh_width1,
    )

    # Scale camera by 1.5x
    scale_factor = 1.5
    hands_scaled = hands * scale_factor
    pose_scaled = pose * scale_factor
    sh_width2 = compute_clip_shoulder_width(pose_scaled, p_mask)

    feats2 = compute_feature_vector(
        hands_coords=hands_scaled,
        hands_masks=h_mask,
        pose_coords=pose_scaled,
        pose_masks=p_mask,
        feature_set="hands_pose",
        wrist_reference="torso_xy",
        clip_shoulder_width=sh_width2,
    )

    layout = infer_feature_layout_from_dim(feats1.shape[1])
    w1 = feats1[:, layout.slice_wrists]
    w2 = feats2[:, layout.slice_wrists]
    np.testing.assert_allclose(w1, w2, atol=1e-5)


def test_missing_pose_marks_geometry_unsupported() -> None:
    """When pose is completely missing or degenerate, geometry support mask must be 0 and wrists zeroed."""
    T = 8
    hands = np.ones((T, 2, 21, 3), dtype=np.float32) * 0.5
    h_mask = np.ones((T, 2), dtype=bool)

    # Missing pose
    feats = compute_feature_vector(
        hands_coords=hands,
        hands_masks=h_mask,
        pose_coords=None,
        pose_masks=None,
        feature_set="hands_pose",
        wrist_reference="torso_xy",
        clip_shoulder_width=0.0,
    )
    layout = infer_feature_layout_from_dim(feats.shape[1])
    # Geometry support must be strictly zero
    g_support = feats[:, layout.slice_geometry_support]
    np.testing.assert_equal(g_support, 0.0)
    # Body-relative wrists must be zeroed
    wrists = feats[:, layout.slice_wrists]
    np.testing.assert_equal(wrists, 0.0)


def test_clip_median_palm_scale() -> None:
    """Clip median palm scale must be resistant to outlier frames and strictly finite."""
    T = 10
    hands = np.zeros((T, 2, 21, 3), dtype=np.float32)
    # Normal palm distance is 0.10
    hands[:, 0, 9, 0] = 0.10
    # Add an outlier in frame 3
    hands[3, 0, 9, 0] = 0.90
    m = np.ones((T, 2), dtype=bool)

    s0, s1 = compute_clip_palm_scales(hands, m)
    # Median of 9 frames of 0.10 and 1 frame of 0.90 is 0.10
    assert abs(s0 - 0.10) < 1e-4
    assert s1 == 0.0  # Slot 1 is all zero distance


# =========================================================================
# B2: Fingertip Displacement & Bend Cosines Tests
# =========================================================================

def test_fingertip_displacement_mathematics() -> None:
    """Verify local fingertip displacement contract (first timestep 0, gated on adjacent valid frames)."""
    T = 5
    local_shapes = np.zeros((T, 2, 21, 3), dtype=np.float32)
    # Move index fingertip (landmark 8) of Slot 0 linearly in x: t * 0.05
    for t in range(T):
        local_shapes[t, 0, 8, 0] = t * 0.05

    masks = np.ones((T, 2), dtype=bool)
    disps = compute_fingertip_displacements(local_shapes, masks)  # (T, 30)

    # Frame 0 must be 0.0
    assert np.all(disps[0] == 0.0)
    # Index tip is index 1 among (4, 8, 12, 16, 20), so offset in 30 dims is 1 * 3 = 3 (x coord)
    assert abs(disps[1, 3] - 0.05) < 1e-5
    assert abs(disps[2, 3] - 0.05) < 1e-5

    # If frame 2 is invalid, frame 2 and frame 3 displacements for Slot 0 must be 0
    masks[2, 0] = False
    disps_dropout = compute_fingertip_displacements(local_shapes, masks)
    assert disps_dropout[2, 3] == 0.0
    assert disps_dropout[3, 3] == 0.0  # frame 3 adjacent to invalid frame 2


def test_bend_cosines_analytical_angles() -> None:
    """Verify analytical angles: straight finger -> +1.0, right-angle bend -> 0.0, folded back -> -1.0."""
    T = 3
    local_shapes = np.zeros((T, 2, 21, 3), dtype=np.float32)
    masks = np.ones((T, 2), dtype=bool)

    # Chain 1: Index finger (5, 6, 7, 8)
    # Frame 0: Analytically straight finger along x-axis
    local_shapes[0, 0, 5] = [0.0, 0.0, 0.0]
    local_shapes[0, 0, 6] = [0.1, 0.0, 0.0]
    local_shapes[0, 0, 7] = [0.2, 0.0, 0.0]
    local_shapes[0, 0, 8] = [0.3, 0.0, 0.0]

    # Frame 1: 90-degree right angle bend between (6->7) and (7->8)
    local_shapes[1, 0, 5] = [0.0, 0.0, 0.0]
    local_shapes[1, 0, 6] = [0.1, 0.0, 0.0]
    local_shapes[1, 0, 7] = [0.2, 0.0, 0.0]
    local_shapes[1, 0, 8] = [0.2, 0.1, 0.0]  # turns up along y

    # Frame 2: Folded backwards 180 degrees between (6->7) and (7->8)
    local_shapes[2, 0, 5] = [0.0, 0.0, 0.0]
    local_shapes[2, 0, 6] = [0.1, 0.0, 0.0]
    local_shapes[2, 0, 7] = [0.2, 0.0, 0.0]
    local_shapes[2, 0, 8] = [0.1, 0.0, 0.0]  # folds back toward 6

    bends = compute_bend_cosines(local_shapes, masks)  # (T, 20)
    # Index finger is chain 1. Slot 0 chain 1 pair 0 is index (1 * 2 + 0) = 2; pair 1 is index 3.
    # Frame 0: Both pairs straight -> +1.0
    assert abs(bends[0, 2] - 1.0) < 1e-4
    assert abs(bends[0, 3] - 1.0) < 1e-4

    # Frame 1: Pair 0 straight -> +1.0; Pair 1 right angle -> 0.0
    assert abs(bends[1, 2] - 1.0) < 1e-4
    assert abs(bends[1, 3] - 0.0) < 1e-4

    # Frame 2: Pair 0 straight -> +1.0; Pair 1 folded back -> -1.0
    assert abs(bends[2, 2] - 1.0) < 1e-4
    assert abs(bends[2, 3] - (-1.0)) < 1e-4


# =========================================================================
# B3: Augmentation & Raw-Adapter Parity Tests
# =========================================================================

def test_horizontal_flip_torso_relative() -> None:
    """Torso-relative wrists must negate x around 0 (not 1 - x). Pose must negate x and swap sides."""
    T = 4
    D = 212
    feat = np.zeros((T, D), dtype=np.float32)
    layout = infer_feature_layout_from_dim(D)

    # Active hand and geometry masks
    feat[:, layout.slice_hand_masks] = 1.0
    feat[:, layout.slice_pose_mask] = 1.0
    feat[:, layout.slice_geometry_support] = 1.0

    # Wrists: slot 0 at (+0.25, 0.30, 0.0), slot 1 at (-0.25, 0.30, 0.0)
    feat[:, 126:129] = [0.25, 0.30, 0.0]
    feat[:, 129:132] = [-0.25, 0.30, 0.0]

    # Pose: Left shoulder at (-0.5, 0.0, 0.0), Right shoulder at (+0.5, 0.0, 0.0)
    # Pose indices: [L_sh, R_sh, L_el, R_el]
    feat[:, 143:146] = [-0.5, 0.0, 0.0]
    feat[:, 146:149] = [0.5, 0.0, 0.0]

    flipped = horizontal_flip(feat, swap_hands=False)

    # Torso wrist x negated around 0
    np.testing.assert_allclose(flipped[:, 126], -0.25, atol=1e-5)
    np.testing.assert_allclose(flipped[:, 129], 0.25, atol=1e-5)

    # Pose: L_sh (-0.5) negates to +0.5 and swaps with R_sh (+0.5 negated to -0.5)
    # Position of L_sh in flipped tensor:
    np.testing.assert_allclose(flipped[:, 143:146], np.tile([[-0.5, 0.0, 0.0]], (T, 1)), atol=1e-5)
    np.testing.assert_allclose(flipped[:, 146:149], np.tile([[0.5, 0.0, 0.0]], (T, 1)), atol=1e-5)


def test_scale_translate_rejects_torso_xy() -> None:
    """scale_translate must raise ValueError when applied to torso_xy features."""
    feat_torso = np.zeros((4, 162), dtype=np.float32)
    with pytest.raises(ValueError, match="scale_translate is not supported when features.wrist_reference='torso_xy'"):
        scale_translate(feat_torso)

    t_feat_torso = torch.zeros((2, 4, 162), dtype=torch.float32)
    with pytest.raises(ValueError, match="batch_scale_translate is not supported when features.wrist_reference='torso_xy'"):
        batch_scale_translate(t_feat_torso)


def test_cpu_gpu_augmentation_parity_wave_b() -> None:
    """Verify that PyTorch device augmentation and NumPy augmentation produce identical results on Wave B features."""
    T = 6
    D = 212
    feat_np = np.zeros((T, D), dtype=np.float32)
    layout = infer_feature_layout_from_dim(D)

    feat_np[:, layout.slice_hand_masks] = 1.0
    feat_np[:, layout.slice_pose_mask] = 1.0
    feat_np[:, layout.slice_geometry_support] = 1.0

    # Local shapes
    feat_np[:, 0:126] = np.random.uniform(-0.2, 0.2, size=(T, 126)).astype(np.float32)
    # Wrists
    feat_np[:, 126:129] = [0.15, 0.20, 0.0]
    feat_np[:, 129:132] = [-0.15, 0.20, 0.0]

    # CPU flip
    cpu_flipped = horizontal_flip(feat_np, swap_hands=False)

    # GPU flip
    gpu_tensor = torch.from_numpy(feat_np).unsqueeze(0)  # (1, T, D)
    gpu_flipped = batch_horizontal_flip(gpu_tensor, swap_hands=False).squeeze(0).numpy()

    np.testing.assert_allclose(cpu_flipped, gpu_flipped, atol=1e-5)
