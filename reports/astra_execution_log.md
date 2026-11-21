# Astra Execution Log & Audit Trail

## Status Update — 2026-11-05, 14:15 +07:00

**Gate 0: PASSED. Gate A: SYNTHETIC VERIFIED. Gate B: SYNTHETIC VERIFIED.**
All 8 critical P1 issues (A-01–A-04 and B-01–B-04) from Astra's review have been resolved, regression-tested, and independently verified via diagnostic probes.

- **Full Test Suite**: **249 passed, 1 warning (CUDA unavailable); 21.47s; exit 0**.
- **Pyflakes**: **0 findings, exit 0**.
- **Optional-Kaggle-absent Isolation Suite**: **72 passed; 4.30s; exit 0**.
- **Diagnostic Probes (`.review-wave-ab/probes.py`)**: **Exit 0 (all 8 P1 diagnostic probes verified)**:
  - `resume` (A-01, A-02): atomic rollback verified on all 4 failure modes (same-shape pooling mismatch, missing classes/labels, missing optimizer state, late invalid epoch). `accepted=False`, `model_mutated=False`.
  - `frozen_split` (A-03): signer provenance violation rejected, gloss mutation rejected, missing sequence generated explicit exclusion report, unsupported manifest schema rejected.
  - `standalone_evaluation` (A-04, E-01): verified required audit fields (`sample_membership`, `preprocess_identity`, `split_identity`, `confusion_matrix`), missing config rejected, duplicate/noncontiguous labels rejected, model pooling override strictly forbidden.
  - `device_loader_epsilon` & `cpu_helper_epsilon` (B-02): canonical epsilon 0.01 propagated cleanly across `none`, `ram`, `device` preloads and CPU `shape_scale`/`scale_translate`, yielding exact 0.0 for degenerate short bones.
  - `hand_dropout_recompute` (B-01): `clean_gap_tip_max=0.0`, `cpu_flip_gap_tip_max=0.0`, `torch_flip_gap_tip_max=0.0`, `clean_gap_wrist_velocity_max=0.0`, `cpu_flip_gap_wrist_velocity_max=0.0`, `provenance_has_transition_masks=True`.
  - `pose_dropout_body_velocity` (B-03): `body_wrist_velocity_across_pose_gap=[0, 0, 0, 0, 0, 0]`, `torso_velocity_across_pose_gap=[0, 0, 0]`.
  - `post_resample_quality` (B-04): `is_usable=False`, `raw_valid_frames=3`, `target_hand_support_count=0`, `model_forward_count=0`, `returned_closed_set_prediction=null`.
  - `layout_and_offline_raw_parity`: 24 configurations / 12 dimensions: offline raw max error `0.0`, supplied transform numpy vs torch max error `1.79e-7`, failed configurations `[]`.

### Phased Status Summary

| Phase | Description | Status | Evidence / Notes |
| :---: | :--- | :---: | :--- |
| **Phase 0** | Fix test self-containment (`T-01`), separate status | **VERIFIED** | All tests self-contained in `tmp_path`; `test_inference.py` T-01 fixed; pyflakes 0 findings. |
| **Phase 1** | Atomic checkpoint/resume rollback, frozen split provenance, evaluator metadata | **VERIFIED** | A-01, A-02, A-03, A-04, E-01, N-01 closed. Verified by 15 regression tests in `tests/test_wave_a_regression.py`. |
| **Phase 2** | Validity across pipeline, canonical epsilon, zero-support quality gate | **VERIFIED** | B-01, B-02, B-03, B-04 closed. Verified by 5 regression tests in `tests/test_wave_b_regression.py` and diagnostic probes. |
| **Phase 3** | Corrected GRU baseline (3 seeds) & feature ablations | **PENDING RUN ON GPU** | Pipeline is ready; real-data 86k Parquet training pending Kaggle/Colab GPU execution. |
| **Phase 4** | MaskedTCN and HandGraphTCN model integration | **PENDING** | Architectural models to be integrated into factory and benchmarked. |
| **Phase 5** | Deployment benchmarks (CPU/webcam/OOV) | **PENDING** | Real-time latency and OOV evaluation on hardware. |

---

**Historical log follows.** The VERIFIED/GATED claims in sections 1–4 below are preserved as historical development trail:

**Date**: 2026-11-05  
**Current Working Branch**: `main`  
**Head Commit**: `8e335b7`  
**Platform**: Windows 11 (AMD64)  
**Python Runtime**: `3.13.15` (v3.13.15:4061bc4)  
**PyTorch Version**: `2.14.1+cpu`  
**Local CUDA**: `False` (CPU execution locally; GPU T4 acceleration on Kaggle/Colab)  

---

## 1. Status of Implementation Waves

| Wave | Task Description | Status | Evidence / Notes |
| :---: | :--- | :---: | :--- |
| **A0** | Audit có bằng chứng & Execution Log | **VERIFIED** | Initial audit completed; test suite 201 pass; pyflakes clean; execution log created. |
| **A1** | Strict cache/checkpoint/config compatibility | **VERIFIED** | Addressed all Wave A Review findings: atomic state rollback on rejected resumes, strict config verification, schema validation, foreign cache isolation. |
| **A2** | Freeze splits & Separate validation from final test | **VERIFIED** | Split manifest snapshot & reload; signer provenance checks; effective vocabulary class preservation; standalone `scripts/21_evaluate.py` with sys.path bootstrap. |
| **A3** | Baseline GRU freeze & Synthetic smoke | **VERIFIED** | 238/238 unit tests passed; pyflakes 0 errors; full synthetic verification pass. Baseline 86k run pending Kaggle execution. |
| **B0** | Canonical options & Feature layout registry | **VERIFIED** | Schema v3; FeatureLayout registry covering all 12 canonical dimension combinations (143, 160, 162, 163, 173, 180, 182, 190, 192, 193, 210, 212); `reports/astra_feature_contract.md`. |
| **B1** | Body-relative wrists & Robust scales | **VERIFIED** | `torso_xy` mode with pre-imputation shoulder width $\ell$ restricted to validated $o_{\text{pose}}$; 2 geometry support channels; `clip_median` palm scale. Translation/scale invariance proven. |
| **B2** | Fingertip displacement & Articulation bend cosines | **VERIFIED** | 30-dim local fingertip displacements with temporal transition support; 20-dim bend cosines with fixed denominator clamp dividing only by bone product. |
| **B3** | Layout-aware augmentation & Raw adapter parity | **VERIFIED** | CPU/GPU mirror reflection with torso origin ($-x$); rotation with complete derivative recomputation; CPU/GPU shape scale parity; canonical epsilon propagation; strict torso wrist $z=0$ invariant preservation. |
| **C** | MaskedTCN & HandGraphTCN Model Factory | **GATED** | Gate A and Gate B review findings 100% resolved; ready to implement Wave C model architectures. |
| **D** | Architecture & Feature Ablations on Kaggle | **GATED** | Depends on Wave A-C completion. |
| **E** | Pretraining, SupCon, Targeted Face/Hybrid | **GATED** | Conditional on Wave D validation evidence. |

---

## 2. Gate A & Gate B Verification Evidence

### Test Suite Execution
* **Command**: `uv run pytest -q`
* **Result**: **238 passed, 1 warning in 20.91s** (Exit code: 0)
* **Regression Test Coverage (`tests/test_wave_a_regression.py` - 9 tests)**:
  * `test_resume_checkpoint_rejection_and_atomic_rollback`: Confirms strict rejection of unknown configs, identity mismatches, inverted labels, extra model tensors, and bad optimizer state dicts with **0% parameter mutation**.
  * `test_foreign_cache_cannot_be_certified_by_features_dir`: Confirms foreign feature files without local `preprocess_meta.json` are rejected, preventing false certification.
  * `test_frozen_manifest_rejects_signer_drift_and_class_loss`: Confirms split rebind rejects participant drift and catches disappeared glosses.
  * `test_eval_cli_rejects_missing_metadata`: Confirms standalone evaluator CLI strictly rejects checkpoints missing canonical schema or identity.
  * `test_predictor_validates_checkpoint_metadata_strictly`: Confirms `ISLRPredictor` correctly initializes `class_names`, rejects legacy or corrupted schema versions, and checks config identity hash.

### Static Code Analysis
* **Command**: `uv run pyflakes src scripts tests app.py`
* **Result**: **0 errors, 0 warnings** (Exit code: 0)

### Real-Data Training Verification Status
* **Baseline Run**: **PENDING RUN ON KAGGLE/COLAB GPU**.
* The synthetic test suite and CLI dry runs pass 100%. Real training on the 86,888 Parquet clips across 3 seeds (`seeds=[42, 43, 44]`) will be executed on GPU infrastructure as specified in the benchmark protocol. No real-data numbers are fabricated.

---

## 3. Implementation Details for Wave B

### B0: Canonical Options & Feature Layout Registry
1. **Configuration Extensions (`src/utils/config.py`, `configs/islr.yaml`)**:
   * Added `features.wrist_reference: str = "image"` (`image` or `torso_xy`).
   * Added `features.palm_scale: str = "frame"` (`frame` or `clip_median`).
   * Added `features.fingertip_displacement: bool = False`.
   * Added `features.bend_cosines: bool = False`.
   * Enforced validation: `torso_xy` strictly requires `features.set == "hands_pose"`; conflicting `scale_translate` with `torso_xy` raises `ValueError`.
2. **Layout Registry & Schema v3 (`src/features/schema.py`)**:
   * Bumped `PREPROCESS_SCHEMA_VERSION = 3`.
   * Created `FeatureLayout` descriptor with explicit slices for:
     * `slice_local_hand_shapes` (0..126)
     * `slice_wrists` (126..132)
     * `slice_rel_wrist` (132..135)
     * `slice_wrist_vel` (135..141)
     * `slice_hand_masks` (141..143)
     * `slice_norm_pose` (143..155, if hands_pose)
     * `slice_torso_vel` (155..158, if hands_pose)
     * `slice_shoulder_scale` (158..159, if hands_pose)
     * `slice_pose_mask` (159..160, if hands_pose)
     * `slice_geometry_support` (160..162, if torso_xy)
     * `slice_fingertip_displacement` (30 dims, if enabled)
     * `slice_bend_cosines` (20 dims, if enabled)
   * Created `reports/astra_feature_contract.md` documenting slices, units, masks, and transform rules.

### B1: Body-Relative Wrists & Robust Scales (`src/features/normalization.py`, `src/features/pipeline.py`)
1. **Pre-imputation Shoulder Width ($\ell$)**:
   * Computed as the median of observed valid non-degenerate shoulder distances $\|l\_sh[t] - r\_sh[t]\|_2$ in the shared trim interval.
   * If $\ell \le \epsilon$ or pose is unobserved, geometry is marked unsupported.
2. **Torso-relative Wrists ($w_{\text{body}}$)**:
   * $$w_{\text{body}}[t, h, xy] = \frac{w_{\text{image}}[t, h, xy] - c[t, xy]}{\ell}$$
   * Gated on geometry support $g[t, h] = m[t, h] \land p[t] \land (\ell > \epsilon)$.
   * $w_{\text{body}}[t, h, z]$ is strictly $0.0$ (hand $z$ is reserved for intra-hand shape).
   * Appends 2 geometry support channels ($g[t, 0]$, $g[t, 1]$).
3. **Clip-median Palm Scale**:
   * Computes the median observed non-degenerate palm size across valid frames for each hand slot in the trim window, stabilizing local shape normalization.

### B2: Fingertip Displacement & Articulation (`src/features/normalization.py`)
1. **Fingertip Displacements (30 dims)**:
   * Tracks local displacement of tips (Thumb 4, Index 8, Middle 12, Ring 16, Pinky 20) across consecutive timesteps:
     $$d_{\text{tip}}[t, h, k] = H[t, h, \text{tip}_k] - H[t-1, h, \text{tip}_k]$$
   * Gated on adjacent valid observation support ($m[t, h] \land m[t-1, h]$); frame 0 is strictly $0.0$.
2. **Articulation Bend Cosines (20 dims)**:
   * Evaluates consecutive bone pairs across 5 finger chains per hand:
     * Thumb: `(1, 2, 3, 4)`
     * Index: `(5, 6, 7, 8)`
     * Middle: `(9, 10, 11, 12)`
     * Ring: `(13, 14, 15, 16)`
     * Pinky: `(17, 18, 19, 20)`
   * Computes $\cos(\theta) = \text{clip}\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}, -1.0, 1.0\right)$.
   * Straight finger yields $+1.0$; $90^\circ$ bend yields $0.0$; folded back yields $-1.0$.

### B3: Layout-Aware Augmentations (`src/data/augmentation.py`, `src/data/augmentation_gpu.py`)
1. **Corrected Reflection**:
   * Local shapes: $x \to -x$.
   * Image wrists: $x \to 1 - x$.
   * Torso wrists: $x \to -x$ (centered on torso origin).
   * Pose points: $x \to -x$ with anatomical swap $[1, 0, 3, 2]$.
   * Fingertip displacements: $x \to -x$.
   * Bend cosines: Invariant under reflection.
2. **Rotation & Zero Stale Features**:
   * Rotates coordinates in the $xy$ plane according to each block's coordinate origin.
   * Recomputes all velocities, relative vectors, fingertip displacements, and bend cosines post-transform.
3. **Incompatible Transform Rejection**:
   * Calling `scale_translate` or `batch_scale_translate` on `torso_xy` features explicitly raises `ValueError`.

---

## 4. Gate B Completion & Wave C Unlock Criteria

Gate B is **VERIFIED** across all contracts:
* All 7 feature dimension layouts verified.
* Body-relative geometry and robust scaling verified.
* Fingertip displacement and articulation cosines verified.
* Layout-aware augmentations on CPU and GPU verified.
* 229/229 tests passing, pyflakes clean, graphify updated.

**Next Wave**: Wave C — Model Factory & Two Astra Architectures:
* **C0**: Model Factory & Boundary Contract (`src/models/factory.py`, `src/models/pooling.py`).
* **C1**: Residual Depthwise Temporal CNN (`src/models/tcn.py`, ~293k params).
* **C2**: Shared Hand Graph + Temporal CNN (`src/models/hand_graph_tcn.py`, ~250k params).
* **C3**: Training, checkpoint, and inference integration.
