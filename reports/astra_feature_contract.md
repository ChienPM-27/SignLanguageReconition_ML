# ISLR Preprocessing & Feature Layout Contract (Schema Version 3)

> **Implementation caveat — 2026-11-05:** the equations below describe the intended schema-v3 contract. Current code does not preserve source-interval transition masks through cache/augmentation, omits pose continuity from body-wrist motion, loses canonical epsilon on some paths, and may forward a clip with zero target support. Observed and imputed usable support are not interchangeable despite some historical labels below. See B-01–B-04 and Phase 2 in [the remediation plan](astra_remediation_plan.md). Schema migration, metadata and adapter work there are planned; this document does not certify Gate B.

**Document Version**: 3.0  
**Schema Version**: `3`  
**Date**: 2026-11-05  

This document defines the formal layout, coordinate reference spaces, normalization equations, channel slices, and transform rules for landmark feature representations in ISLR.

---

## 1. Feature Channel Layout & Dimension Registry

Features are represented as 2D floating-point tensors of shape `(sequence_length, total_dim)` with `dtype=float32`.

### Dimension Derivations

The canonical feature layout registry supports all 12 valid combinations derived from independent toggles:

| Configuration | Set | Wrist Reference | Tips (+30) | Bends (+20) | Dimension ($D$) | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Hands Baseline** | `hands` | `image` | No | No | **143** | Base hands-only |
| **Hands + Bends** | `hands` | `image` | No | Yes | **163** | Hands + Articulation bends |
| **Hands + Tips** | `hands` | `image` | Yes | No | **173** | Hands + Fingertip displacements |
| **Hands + Tips + Bends** | `hands` | `image` | Yes | Yes | **193** | Full hands cues without pose |
| **Hands + Pose Baseline** | `hands_pose` | `image` | No | No | **160** | Canonical Wave A baseline |
| **Torso-relative Wrists** | `hands_pose` | `torso_xy` | No | No | **162** | +2 geometry support channels |
| **Hands+Pose + Bends** | `hands_pose` | `image` | No | Yes | **180** | Independent bends ablation |
| **Torso-relative + Bends** | `hands_pose` | `torso_xy` | No | Yes | **182** | Body geometry + Articulation bends |
| **Hands+Pose + Tips** | `hands_pose` | `image` | Yes | No | **190** | Fingertip local velocities |
| **Torso-relative + Tips** | `hands_pose` | `torso_xy` | Yes | No | **192** | Geometry + Fingertip velocities |
| **Hands+Pose + Tips + Bends** | `hands_pose` | `image` | Yes | Yes | **210** | Image wrists + all Wave B cues |
| **Full Wave B Architecture** | `hands_pose` | `torso_xy` | Yes | Yes | **212** | Complete Wave B semantic layout |

---

## 2. Channel Slices & Semantics

### Base Hand Channels (0 .. 143)
1. `[0:126]` — **Local Handshapes (`slice(0, 126)`)**:
   - Two hands (Slot 0, Slot 1) $\times$ 21 landmarks $\times$ 3 coordinates $(x, y, z)$.
   - Wrist landmark 0 is shifted to $(0, 0, 0)$ and identically zeroed.
   - Normalized by palm size $L_{\text{palm}}$ (Euclidean distance from wrist to knuckle landmark 9):
     - Mode `frame`: computed per frame.
     - Mode `clip_median`: median of valid observed palm sizes across the shared trimmed sequence for each anatomical hand.
   - For missing or unobserved frames/hands: strictly $0.0$.
2. `[126:132]` — **Wrists (`slice(126, 132)`)**:
   - Two hands $\times$ 3 coordinates $(x, y, z)$.
   - Mode `image`: normalized image coordinates in $[0, 1]$ (or centered around $0.5$).
   - Mode `torso_xy`: body-relative coordinates $(x, y)$ normalized by clip median shoulder width $\ell$, with $z = 0$.
3. `[132:135]` — **Relative Wrist (`slice(132, 135)`)**:
   - Vector $w[0] - w[1]$ (Slot 0 minus Slot 1).
   - Gated on **both** hands present ($m[t, 0] = 1 \land m[t, 1] = 1$); otherwise strictly $0.0$.
4. `[135:141]` — **Wrist Velocities (`slice(135, 141)`)**:
   - Frame difference $w[t] - w[t-1]$ for each hand.
   - Gated on adjacent valid observation endpoints ($m[t, h] = 1 \land m[t-1, h] = 1$); $v[0, h] = 0.0$.
5. `[141:143]` — **Hand Observation Masks (`slice(141, 143)`)**:
   - Binary indicator ($1.0$ or $0.0$) for active/usable observations in Slot 0 and Slot 1.

### Pose Channels (143 .. 160, if `features.set == "hands_pose"`)
6. `[143:155]` — **Normalized Upper-Body Pose (`slice(143, 155)`)**:
   - 4 selected joints: Left Shoulder (11), Right Shoulder (12), Left Elbow (13), Right Elbow (14) $\times$ 3 coordinates.
   - Centered on torso midpoint $c[t] = (l\_sh[t] + r\_sh[t]) / 2$.
   - Scaled by instantaneous shoulder distance $\|l\_sh[t] - r\_sh[t]\|_2$.
7. `[155:158]` — **Torso Midpoint Velocity (`slice(155, 158)`)**:
   - $c[t] - c[t-1]$, gated on adjacent valid pose frames ($p[t] = 1 \land p[t-1] = 1$).
8. `[158:159]` — **Clipped Shoulder Scale (`slice(158, 159)`)**:
   - Torso scale clamped to $[0.0, 2.0]$.
9. `[159:160]` — **Pose Usability Mask (`slice(159, 160)`)**:
   - Binary indicator ($1.0$ or $0.0$) for active/usable pose observations.

### Wave B Extension Channels
10. `[160:162]` — **Geometry Support Mask (`slice(160, 162)`, if `wrist_reference == "torso_xy"`)**:
    - Two channels: $g[t, 0]$ and $g[t, 1]$ indicating joint usability:
      $$g[t, h] = m[t, h] \land p[t] \land \text{valid}(\ell)$$
    - Distinguishes between a hand at the torso center vs. missing pose geometry.
11. **Fingertip Displacements (`slice(curr, curr + 30)`, if `fingertip_displacement == True`)**:
    - 2 hands $\times$ 5 fingertips (Thumb 4, Index 8, Middle 12, Ring 16, Pinky 20) $\times$ 3 coordinates $(x, y, z)$.
    - Computed from **local normalized handshapes**:
      $$d_{\text{tip}}[t, h, k] = H[t, h, \text{tip}_k] - H[t-1, h, \text{tip}_k]$$
    - Gated on adjacent valid hand observations ($m[t, h] = 1 \land m[t-1, h] = 1$). $d_{\text{tip}}[0, h, k] = 0.0$.
12. **Finger Articulation Bend Cosines (`slice(curr, curr + 20)`, if `bend_cosines == True`)**:
    - 2 hands $\times$ 5 finger chains $\times$ 2 consecutive bone pairs.
    - Chains:
      - Thumb: `(1, 2, 3, 4)`
      - Index: `(5, 6, 7, 8)`
      - Middle: `(9, 10, 11, 12)`
      - Ring: `(13, 14, 15, 16)`
      - Pinky: `(17, 18, 19, 20)`
    - For consecutive bones $\vec{u} = p_j - p_i$ and $\vec{v} = p_k - p_j$:
      $$\cos(\theta) = \text{clip}\left(\frac{\vec{u} \cdot \vec{v}}{\text{denom}}, -1.0, 1.0\right)$$
      where $\text{denom} = \|\vec{u}\| \|\vec{v}\|$ if $(\|\vec{u}\| > \epsilon \land \|\vec{v}\| > \epsilon)$ else $1.0$.
      (Strictly avoids quadratic threshold clamping such as $\max(\|\vec{u}\|\|\vec{v}\|, \epsilon)$ that would distort short bone cosine angles).
    - Convention: straight finger points $+1.0$; $90^\circ$ bend gives $0.0$.
    - Gated on hand mask $m[t, h]$ and non-degenerate bone lengths ($> \epsilon$). Missing/degenerate bones yield $0.0$.

### Temporal Transition & Continuity Contract
- Time-difference features (wrist velocities, fingertip displacements, torso velocities) are valid at step $t$ if and only if observations were continuous across the entire resampling interval between $t-1$ and $t$.
- Continuous support is tracked via `compute_resampled_transitions(source_masks, target_len)`, preventing spurious velocity spikes when resampling spans across raw dropouts ($>\text{max\_gap\_frames}$).

---

## 3. Transformation & Augmentation Rules

1. **Reflection (Horizontal Flip)**:
   - Local handshape: $x \to -x$ (centered on wrist).
   - Image wrists: $x \to 1.0 - x$.
   - Torso-relative wrists (`torso_xy`): $x \to -x$ (centered on torso origin).
   - Pose joints: $x \to -x$, with anatomical swap $[1, 0, 3, 2]$ (Left $\leftrightarrow$ Right).
   - Torso velocity: $x \to -x$.
   - Fingertip displacements: $x \to -x$.
   - Bend cosines: Invariant under reflection (dot products and norms are preserved).
2. **Rotation (2D in XY plane)**:
   - Local handshapes rotate around $(0, 0)$.
   - Image wrists rotate around $(0.5, 0.5)$.
   - Torso-relative wrists rotate around $(0, 0)$.
   - Pose joints rotate around $(0, 0)$.
   - All derived velocities, relative vectors, fingertip displacements, and bend cosines are recomputed after rotation to ensure zero stale features.
3. **Wrist Scale & Translation**:
   - Incompatible with `torso_xy` (must be disabled when `features.wrist_reference == "torso_xy"`).
