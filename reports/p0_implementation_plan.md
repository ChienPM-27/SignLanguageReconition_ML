# Priority 0 implementation plan

Status: completed; all P0.1–P0.8 engineering tasks implemented, mathematically verified, and passing 100% of the test suite.

Objective: remove the four verified preprocessing, augmentation, and inference inconsistencies before comparing architectures. Preserve the 20-gloss task, signer-independent split, and existing GRU baseline. Support both `features.set=hands` (143 dimensions) and the current default, `hands_pose` (160 dimensions).

## Scope and verified starting points

| Issue | Current code | Required outcome |
| --- | --- | --- |
| 1. Incorrect pose reflection | `horizontal_flip` and `batch_horizontal_flip` in `src/data/augmentation.py` and `src/data/augmentation_gpu.py` apply `1-x` to torso-centered pose coordinates. | Reflect centered coordinates with `-x`, swap anatomical pose sides, transform torso velocity, and preserve absent pose zeros. |
| 2. Stale derived features | Rotation and jitter return modified wrists with old relative-position and velocity channels. All three implementations of wrist velocity use unmasked differences. | Every public spatial augmentation returns internally consistent features; NumPy and Torch implement the same equations. |
| 3. Missingness/resampling artifacts | `resample_temporal` in `src/features/normalization.py` uses insertion indices as nearest neighbors and interpolates through zero-filled missing hands. | Explicit observation/support masks, bounded interpolation, true nearest-neighbor mask selection, and no interpolation toward missing-data zeros. |
| 4. Training/inference divergence | `ISLRPredictor.predict_raw_coordinates` in `src/inference/predictor.py` hard-codes margin 2, gap 5, length 32, and selects the feature set using `input_size >= 155`. | One shared preprocessing function driven by the checkpoint's validated canonical configuration and schema. |

The dependency order differs from the issue numbering: fix validity and derivative semantics before wiring augmentation and inference to them.

| Order | Task | Depends on | Approximate effort |
| --- | --- | --- | --- |
| 1 | P0.1 — Freeze the feature contract and add synthetic fixtures | — | Small |
| 2 | P0.2 — Repair interpolation and resampling | P0.1 | Medium |
| 3 | P0.3 — Gate derived motion on valid endpoints | P0.2 | Small |
| 4 | P0.4 — Correct pose reflection in both backends | P0.3 | Small |
| 5 | P0.5 — Make spatial augmentation outputs consistent | P0.3–P0.4 | Medium |
| 6 | P0.6 — Share preprocessing across training and inference | P0.2–P0.5 | Medium |
| 7 | P0.7 — Bind caches and checkpoints to preprocessing semantics | P0.6 | Medium |
| 8 | P0.8 — Run regression checks and establish the corrected baseline | P0.1–P0.7 | Small engineering; training time separate |

All new tests must generate their own arrays, parquet files, manifests, configuration files, and checkpoints under `tmp_path`. Do not use downloaded data, existing output artifacts, a webcam, or network access as unit-test prerequisites.

## P0.1 — Freeze the feature contract and add synthetic fixtures

- [x] Document the channel layout, coordinate spaces, mask semantics, and derivative sign convention.
- [x] Introduce shared schema constants and reusable fixtures before modifying behavior.
- [x] Add regression tests with independently calculated expected values; confirm that the relevant tests expose the current failures.

**Target files**

- New `src/features/schema.py`: feature layout and preprocessing schema identifier; these are format constants, not experiment tunables.
- `src/features/normalization.py`: `get_feature_dim`, feature-function contracts.
- `configs/islr.yaml`, `src/utils/config.py`: canonical settings and validation.
- `tests/conftest.py`, `tests/test_preprocess.py`, `tests/test_augmentation.py`, `tests/test_config.py`.

**Exact contract**

| Slice, Python end-exclusive | Symbol | Meaning |
| --- | --- | --- |
| `0:126` | `H[t,h,j]` | Two wrist-centered, palm-normalized hand shapes; slot 0 is dominant when configured. |
| `126:132` | `w[t,h]` | Two original hand-wrist coordinate triplets. Only `xy` shares an image-space origin across hands. |
| `132:135` | `r[t]` | Existing convention: `w[t,0] - w[t,1]`, gated on both hands. Do not silently reverse it. |
| `135:141` | `v[t,h]` | First difference per resampled step, gated on both temporal endpoints. |
| `141:143` | `m[t,h]` | Binary usable-coordinate support after permitted imputation/resampling. |
| `143:155` | `q[t,k]` | Torso-normalized pose in anatomical order `[L shoulder, R shoulder, L elbow, R elbow]`. |
| `155:158` | `u[t]` | First difference of the original pose shoulder midpoint. |
| `158` | `b[t]` | Existing clipped shoulder-distance feature, `clip(s[t], 0, 2)`. |
| `159` | `p[t]` | Binary usable pose support. |

Distinguish `o` (observed) from `m` (usable after imputation). Preserve observation/imputation provenance as preprocessing metadata; do not add network input channels in P0. Padding validity, if introduced later, is a separate concept. A zero coordinate is not evidence of absence: masks determine absence.

For a conservative P0 policy, an observed hand requires the supplied detector mask and finite `xyz` for all 21 points. Treat a partial-NaN hand as unavailable for bounded imputation instead of silently inserting a zero joint. Pose support requires finite coordinates for the four selected joints, the supplied pose mask when present, and nondegenerate shoulders. Record any resulting change in excluded-clip counts.

Reuse `preprocess.trim_no_hand`, `pad_frames_around`, `min_valid_frames`, `max_gap_frames`, `sequence_length`, `dominant_hand`, and `features.set`. Consolidate the existing `1e-4` normalization threshold as `preprocess.normalization_epsilon: 0.0001`, mapped to `PreprocessConfig.normalization_epsilon`; require a finite positive value and pass it to both hand and pose normalization. Do not introduce `target_len`, `trim_margin`, or `feature_set` as duplicate configuration keys. Function argument names may differ from YAML keys, but must receive their values from the canonical configuration.

`get_feature_dim` must explicitly handle `hands` and `hands_pose` and reject unsupported values rather than falling back to 143. Unknown configuration **keys** must raise `KeyError` in YAML and `--set`; invalid values of known keys can raise `ValueError`.

**Test cases / acceptance**

1. Generate deterministic hands with distinguishable fingers, constant and linear wrist trajectories, one-hand/two-hand frames, and controlled missing intervals.
2. Generate valid symmetric shoulders, asymmetric elbows, a translating torso, and missing pose frames; derive normalized pose expectations analytically.
3. Assert exactly 143/160 dimensions and reject an unknown feature-set value or unsupported layout instead of accepting `D >= 155`.
4. Verify observed zero-valued coordinates remain valid; NaN/Inf coordinates never reach model features as NaN/Inf.
5. Verify changing each relevant canonical override reaches its dataclass, and unknown/deprecated keys fail in both YAML loading and CLI overrides.

## P0.2 — Repair bounded interpolation and temporal resampling

- [x] Sanitize coordinates using explicit validity, keeping observation masks separate from imputed support.
- [x] Fill only bounded short gaps; leave leading, trailing, and long gaps unavailable.
- [x] Replace insertion-index mask selection with true nearest-neighbor selection and a documented tie rule.
- [x] Use the same resampling grid and support policy for hands and pose.

**Target files**

- `src/features/normalization.py`: `interpolate_bounded_gaps`, `resample_temporal`, `process_video_landmarks`.
- `src/data/preprocess.py`: pose validity and interpolation/resampling calls, later moved into P0.6's shared pipeline.
- `tests/test_preprocess.py`, `tests/test_augmentation.py`.

**Exact mathematical fixes**

Let `N` be the number of input frames after trimming and `L = cfg.preprocess.sequence_length`.

For two observed endpoints `a < b`, the missing run has length `g = b-a-1`. Fill it only when `0 < g <= cfg.preprocess.max_gap_frames`:

```text
alpha = (t-a)/(b-a)
c_filled[t] = (1-alpha)*c[a] + alpha*c[b],  a < t < b
m[t] = true; o[t] remains false
```

Unbounded runs and runs longer than the limit have `m=false` and coordinates exactly zero. Do not interpolate from an invalid/partially nonfinite endpoint. Apply this independently to each hand and to the selected pose block. Validate shapes and mask lengths before interpolation.

For uniform sequence resampling, use source-index coordinates:

```text
z[j] = j*(N-1)/(L-1), j = 0,...,L-1
i0 = floor(z[j]); i1 = min(i0+1, N-1)
alpha = z[j]-i0
n[j] = i0 if alpha <= 0.5 else i1  # midpoint ties go to earlier frame
m_out[j,h] = m[n[j],h]
```

Use this coordinate rule for each independent hand/pose block:

```text
if m[i0,h] and m[i1,h]:
    c_out[j,h] = (1-alpha)*c[i0,h] + alpha*c[i1,h]
elif m[n[j],h]:
    c_out[j,h] = c[n[j],h]
else:
    c_out[j,h] = 0
```

The nearest-valid-endpoint branch is a deliberate zero-order hold within that endpoint's nearest-neighbor cell. It prevents blending real coordinates toward missing-data zeros. It does not fill the interior of a long gap or search arbitrarily far for an observation. Do not use unqualified `np.interp` across the entire zero-filled trajectory.

Handle `N=0` with all-zero coordinates/all-false masks at the low-level resampler; the high-level pipeline will reject unusable clips. For `N=1`, repeat its sanitized coordinates and support. For `N=L`, still sanitize unsupported coordinates; do not bypass invariants through an identity shortcut. Reject nonpositive target lengths. For a low-level `L=1` call, select the first source frame explicitly; production configuration continues to require length at least 4.

The P0 grid uses sample index, retaining current duration-normalization semantics. Preserve original frame IDs in metadata if supplied. Timestamp-aware velocity, source-gap continuity under aggressive downsampling, and anti-aliasing are separate temporal-modeling work; endpoint gating alone does not solve those problems.

**Test cases / acceptance**

1. Gap limits: lengths `0`, `1`, `max_gap_frames`, and `max_gap_frames+1`; test `max_gap_frames=0` explicitly.
2. Leading/trailing gaps remain unavailable. A fully missing hand remains zero across all output frames.
3. With short-gap filling disabled, source validity `[1,0,1]` at times `[0,.5,1]`, resampled to nine positions, must yield `[1,1,1,0,0,0,0,1,1]` under the earlier-frame tie rule. At time `.125`, support must be true.
4. For that example, source x values `[.6,0,.9]` must become `[.6,.6,.6,0,0,0,0,.9,.9]`; no artificial `.3`/`.45` transition toward a missing zero.
5. All-valid linear trajectories interpolate exactly, including both endpoints. Constant trajectories remain constant for upsampling and downsampling.
6. Cover empty, single-frame, identity-length, odd/even length, both feature sets, absent pose, and partial-NaN pose/hand inputs.
7. Replace the existing assertion in `tests/test_augmentation.py` that absent wrists produced by preprocessing must be nonzero. Assert canonical absence means zero coordinates and false support instead. Retain separate tests that spatial transforms do not create absent hands.

## P0.3 — Gate derived motion on valid endpoints

- [x] Apply the same derivative definition during feature extraction and recomputation.
- [x] Zero invalid wrists before deriving relationships, while avoiding arithmetic on NaN placeholders.
- [x] Gate torso motion using valid pose endpoints as well.

**Target files**

- `src/features/normalization.py`: `compute_feature_vector_143`, `compute_feature_vector`.
- `src/data/augmentation.py`: `recompute_derived_features`.
- `src/data/augmentation_gpu.py`: `batch_recompute_derived_features`.
- `tests/test_preprocess.py`, `tests/test_augmentation.py`.

**Exact mathematical fixes**

Using sanitized wrists and the support masks from P0.2:

```text
w[t,h] = raw_wrist[t,h] if m[t,h] else 0
r[t] = w[t,0]-w[t,1] if m[t,0] and m[t,1] else 0
v[0,h] = 0
v[t,h] = w[t,h]-w[t-1,h] if m[t,h] and m[t-1,h] else 0
```

Implement masked branches with `where` after sanitization; multiplying a NaN by zero does not remove it. Keep `v` as displacement per resampled step. Do not introduce an FPS multiplier or divide by clip duration in this repair. The z channels retain their current definition; cross-hand wrist z is not a physical global-depth measurement.

For pose, let `C[t]=(P_Lshoulder[t]+P_Rshoulder[t])/2` and `s[t]=||P_Lshoulder[t]-P_Rshoulder[t]||_2`. With epsilon from the canonical normalization setting:

```text
p[t] = selected_pose_is_finite[t] and resampled_pose_support[t] and s[t] > epsilon
q[t,k] = (P[t,k]-C[t])/s[t] if p[t] else 0
b[t] = clip(s[t],0,2) if p[t] else 0
u[0] = 0
u[t] = C[t]-C[t-1] if p[t] and p[t-1] else 0
```

`resampled_pose_support` comes from P0.2 and includes permitted imputation. Build the original observation mask from detector support and finite selected coordinates; when no detector mask is supplied, infer observations from finite selected coordinates before filling, never from a zero-filled array. A missing pose frame has all 17 pose channels zero. This preserves the existing pose normalization formula while preventing false motion at dropout boundaries.

**Test cases / acceptance**

1. Wrist x `[.6,0,.7,.8]`, mask `[1,0,1,1]`: x displacement `[0,0,0,.1]`; disappearance must not create `-.6` and reappearance must not create `+.7`.
2. Both hands valid: `w0=(.7,.4,0)`, `w1=(.2,.6,0)` gives `r=(.5,-.2,0)`. Either missing hand gives `r=0`.
3. A bounded imputed gap is usable and produces the expected interpolated displacement; a long unfilled gap does not.
4. Repeat disappearance/reappearance cases for torso velocity; degenerate shoulders produce zero pose features without division errors.
5. Compare independently calculated expected channels against normalization, NumPy recomputation, and Torch CPU recomputation. Add CUDA parity when available; CPU coverage is mandatory.
6. Recomputing twice is idempotent. Shape, dtype, device, masks, and caller-owned inputs are preserved.

## P0.4 — Correct normalized-pose reflection in both backends

- [x] Apply reflection according to each channel's coordinate space.
- [x] Swap anatomical pose sides separately from dominant-hand slot policy.
- [x] Reflect torso velocity and preserve invalid pose/hand zeros.

**Target files**

- `src/data/augmentation.py`: `horizontal_flip`.
- `src/data/augmentation_gpu.py`: `batch_horizontal_flip`.
- `src/features/schema.py`: explicit pose permutation and slices.
- `tests/test_augmentation.py`.

**Exact mathematical fixes**

Let `F=diag(-1,1,1)` and `pi=[1,0,3,2]`.

```text
H'[t,h,j] = F H[t,h,j]                       # local, centered
w'[t,h] = (1-w_x[t,h], w_y[t,h], w_z[t,h])    # image-space xy
q'[t,k] = F q[t,pi[k]]                       # torso-centered
u'[t] = F u[t]
b'[t] = b[t]; p'[t] = p[t]
```

Apply coordinate formulas only to usable slots and emit zeros for unavailable slots. For centered pose, `1-q_x` is incorrect: reflection of the raw torso midpoint cancels the translation during normalization.

Keep hand slots fixed for the current `augmentation.flip.swap_hands=false`. Anatomical pose swapping still occurs. If `swap_hands=true` is explicitly used, permute complete hand coordinate blocks and masks together, then recompute `r` and `v`; do not only flip the old relative vector. With dominant-hand canonicalization enabled, reject a conflicting `swap_hands=true` configuration so slot 0 retains its documented meaning.

**Test cases / acceptance**

1. Use shoulders `(-.5,0,0),(.5,0,0)` and asymmetric elbows. Assert every expected reflected/swapped point; the shoulder midpoint must remain `(0,0,0)`.
2. An all-zero absent pose remains exactly zero. This detects the current spurious nonzero pose coordinates.
3. Torso velocity `(.1,.2,0)` becomes `(-.1,.2,0)`; scale and support are unchanged.
4. For a wrist x of `.2`, expect `.8`; for a local joint x of `.2`, expect `-.2`.
5. Apply flip twice to a canonical valid feature tensor: recover the original, including derivatives, within float32 tolerance. Test both hand-slot modes at the helper level and configuration rejection of the contradictory mode.
6. Compare NumPy and Torch outputs for 143/160 dimensions, one-hand input, missing pose, and batched samples with different masks.

## P0.5 — Make spatial augmentation outputs internally consistent

- [x] Recompute wrist relationships after every public operation that changes wrists.
- [x] Rotate pose and torso motion with the same sampled angle as the hands.
- [x] Remove the accidental noisy-position/clean-velocity behavior from jitter.
- [x] Guard the currently disabled temporal augmentation until it can meet the same representation contract.

**Target files**

- `src/data/augmentation.py`: `random_rotate_2d`, `add_jitter`, `scale_translate`, `shape_scale`, `apply_augmentation`, `time_stretch`.
- `src/data/augmentation_gpu.py`: corresponding `batch_*` functions and `apply_augmentation_batch`.
- `configs/islr.yaml`, `src/utils/config.py`: validation of supported augmentation combinations.
- `tests/test_augmentation.py`, `tests/test_config.py`.

**Exact mathematical fixes**

For one angle per clip, constant over its frames, let:

```text
R(theta) = [[cos(theta),-sin(theta)], [sin(theta),cos(theta)]]
c = (.5,.5)
H'_xy = R H_xy
w'_xy = R(w_xy-c)+c
q'_xy = R q_xy
u'_xy = R u_xy
```

Keep z unchanged; rotation preserves the current shoulder norm and thus `b`. Recompute `r` and `v` from the transformed wrists using P0.3. For fully valid adjacent frames, this implies `r'_xy=R r_xy` and `v'_xy=R v_xy`. Do not rely on a later scale/translate operation to repair a rotation: that operation is probabilistic and may not run.

For wrist jitter `eta[t,h]`, use `w'=w+eta` on supported slots, then recompute:

```text
r' = (w0+eta0)-(w1+eta1)                  # both hands supported
v'[t,h] = v[t,h]+eta[t,h]-eta[t-1,h]      # both temporal endpoints supported
```

For hand-shape jitter, keep the local wrist joint exactly zero: use zero noise at landmark 0 and perturb other local joints. A future denoising experiment may deliberately pair noisy positions with clean motion, but that is a separate representation/loss decision, not the default P0 behavior.

Retain the existing wrist-only scale/translation operation as a documented augmentation of wrist location:

```text
w'_xy = a*(w_xy-c)+c+d
r'_xy = a*r_xy
v'_xy = a*v_xy
```

Here `a,d` are constant over a clip; the last two equalities apply only where their masks are valid. Pose channels remain unchanged for this wrist-only operation. Do not call it a coherent full-body camera transform: extending image-space scaling to the current 3D-normalized pose and clipped scale channel needs additional source geometry. Local shape scaling changes only `H_xy`, leaving wrist derivatives unchanged.

Public helpers must return consistent features themselves. A final recomputation in `apply_augmentation*` may be used defensively, but must not be the only repair. Update docstrings that currently promise unmasked velocities or no recomputation. Preserve deterministic supplied-angle/noise inputs and non-mutating behavior.

**Temporal augmentation containment**

The default already has `augmentation.time_stretch.enabled=false`. Keep it disabled for P0 and reject `enabled=true` with a clear `ValueError` explaining that the current NumPy/Torch implementations do not share a validity-aware temporal contract. Guard direct calls that request a nonidentity stretch too; do not silently execute the old behavior. This is a temporary, explicit restriction of a known configuration key, not an alias or an ignored override.

Simply recomputing wrist differences after interpolating all 160 channels is insufficient: it creates fractional masks and leaves torso velocity inconsistent. Re-enabling time warp requires a common endpoint-preserving grid, support-aware resampling, and access to source torso centers to derive torso velocity. That implementation belongs to the subsequent temporal-modeling task.

**Test cases / acceptance**

1. Rotation alone, with all other transforms disabled: a 90-degree mathematical xy rotation maps displacement `(.1,0,0)` to `(0,.1,0)`. Also check relative wrists, pose joints, and torso velocity.
2. Jitter alone with supplied noise: wrist x `[.2,.3,.4]` plus noise `[0,.02,-.01]` gives displacement `[0,.12,.07]`.
3. Constant translation leaves wrist derivatives unchanged. Scale `a=2` doubles supported xy derivatives and leaves z differences unchanged.
4. Flip + rotation + jitter: independently derive expected `r,v` from the final coordinates and compare; vary which transforms are selected to cover skipped scale/translation.
5. Local shape scaling leaves `w,r,v` unchanged. Shape jitter keeps both local wrist origins zero.
6. Missing slots remain zero; masks remain exactly binary; a pose dropout cannot acquire nonzero torso velocity through rotation/flip.
7. NumPy/Torch parity uses identical explicit angles, scales, offsets, noise, and transform selections, rather than assuming their RNGs emit identical sequences.
8. Verify the default time-stretch setting is disabled; enabling it or calling a nonidentity direct stretch fails explicitly until a correct implementation exists.

## P0.6 — Share preprocessing across training and inference

- [x] Extract a pure, I/O-free preprocessing function and one hand-ordering implementation.
- [x] Make parquet, extracted-video, and webcam inputs supply the same anatomical input contract.
- [x] Replace inference literals and dimension heuristics with canonical configuration values.
- [x] Test tensor parity at the model boundary, not merely matching predicted labels.

**Target files**

- New `src/features/pipeline.py`: proposed `prepare_landmark_sequence` and structured result containing features, support, provenance, ordering, and rejection reason.
- `src/data/preprocess.py`: `preprocess_single_sequence` becomes parquet/cache/report I/O around the shared function.
- `src/features/normalization.py`: delegate or remove duplicated legacy pipeline logic; no independent default-driven path.
- `src/features/landmark_extractor.py`: `extract_from_video` supplies anatomical `[left,right]` slots and explicit masks, without its own dominant-hand canonicalization.
- `src/inference/predictor.py`: `predict_raw_coordinates`, `predict_video`, checkpoint configuration use.
- `scripts/30_predict.py`, `scripts/31_realtime_demo.py`, `scripts/40_eval_samples.py`, `app.py`: update affected adapters/callers; inspect their current hand ordering before changing them.
- `tests/test_preprocess.py`, `tests/test_inference.py`; new `tests/test_pipeline.py` for pure parity tests.

**Exact algorithm**

Both paths must execute:

```text
validate raw shapes/masks and finite observations
-> choose one hand-slot permutation for the entire clip
-> count observed valid frames and apply the quality gate
-> trim using one shared interval
-> bounded imputation
-> support-aware resampling on one grid
-> feature extraction and masked derivatives
-> validate (L,D), float32, finite values, binary support
```

For dominant-hand selection, compare observed support counts first; break a tie by observed adjacent-frame wrist motion, and break an exact tie in favor of left:

```text
n_h = sum_t o[t,h]
E_h = sum_{t=1}^{N-1} 1[o[t,h] and o[t-1,h]] * ||w[t,h]-w[t-1,h]||_2
right_is_dominant = (n_R > n_L) or (n_R == n_L and E_R > E_L)
```

Do not compute the tie-break motion by dropping missing frames and joining separated observations. Apply ordering once; the extractor must not pre-sort and then allow inference to re-sort. When `preprocess.dominant_hand` is disabled using the existing accepted value convention, retain anatomical order. Persist the chosen slot-to-anatomical mapping in metadata.

Let `A={t: o[t,0] or o[t,1]}`. If `|A| < cfg.preprocess.min_valid_frames`, reject the clip before interpolation can manufacture extra valid frames. Training records an exclusion reason; inference returns/raises an explicit unusable-clip result before model execution. The UI can retain its waiting state instead of classifying an all-zero sequence.

When trimming is enabled and `A` is nonempty, use the same half-open interval for every array:

```text
start = max(0, min(A)-cfg.preprocess.pad_frames_around)
end   = min(N, max(A)+cfg.preprocess.pad_frames_around+1)
```

When disabled, use `[0,N)`. Slice hands, hand masks, pose, pose masks, and source frame IDs identically. Pass `cfg.preprocess.max_gap_frames` and `cfg.preprocess.sequence_length` to the shared helpers; select the feature layout solely from `cfg.features.set`.

**Test cases / acceptance**

1. Serialize one synthetic raw sequence into parquet and also pass its anatomical arrays through the inference adapter. Capture features immediately before model execution; assert agreement within `atol=1e-6, rtol=1e-5` and exact equality of masks/order/trim indices.
2. Parameterize over both feature sets, target lengths 16/32/48, gap limits 0/3/5, margins 0/2/5, trim enabled/disabled, and dominant-hand auto/disabled. Use targeted combinations with at least one all-nondefault case; a full Cartesian product is unnecessary.
3. Test dominant left, dominant right, equal counts with different motion, exact tie, and dropout-separated observations that must not create tie-break motion.
4. Test pose present/absent/partly missing, hand slots with opposite coverage, long missing intervals, and single/empty/insufficient-observation clips.
5. Verify rejected clips never invoke model forward. Verify `hands_pose` with absent pose still supplies exactly 160 dimensions with the 17 pose channels zero.
6. Stub the video/webcam detector to verify adapter ordering and pose propagation without importing or executing a real detector. Feature parity is assessed for identical landmarks; this does not claim identical landmarks across different MediaPipe versions.
7. Test that a feature tensor with a mismatched dimension or sequence length is rejected rather than padded/truncated to make a checkpoint appear compatible.

## P0.7 — Bind caches and checkpoints to preprocessing semantics

- [x] Version the changed feature semantics, even though D remains 143/160.
- [x] Store the validated resolved configuration with new checkpoints.
- [x] Validate cache identity and checkpoint compatibility before reuse, inference, or training resume.
- [x] Make legacy artifact handling explicit.

**Target files**

- `src/features/schema.py`: one immutable preprocessing schema version, proposed initial corrected identifier `2` (unversioned artifacts are legacy).
- `src/data/preprocess.py`: `compute_preprocess_hash`, `run_preprocessing`, resume/cache checks.
- `src/data/dataset.py`: metadata/shape validation for supplied caches.
- `src/training/train.py`: `save_checkpoint`, `load_checkpoint`, `train_gru` metadata propagation.
- `scripts/20_train.py`: pass validated configuration and preprocessing identity through training/checkpoint creation.
- `src/inference/predictor.py`: `_load_checkpoint` and preprocessing configuration setup.
- `src/utils/config.py`: a supported strict dictionary-to-dataclass loader if needed; reuse the same validation as YAML.
- `tests/test_preprocess.py`, `tests/test_dataset.py`, `tests/test_train.py`, `tests/test_inference.py`, `tests/test_config.py`.

**Exact identity and compatibility rules**

Extend the existing preprocessing hash payload with the schema version and every canonical setting that affects feature output or clip acceptance, including the normalization epsilon:

```text
identity = SHA256(UTF8(JSON(payload, sort_keys=true, fixed_separators)))
payload = {schema_version, canonical preprocessing settings, features.set}
```

Keep this payload defined in one function. Worker count and output/cache paths do not affect numerical features and should not change the identity. A schema bump must change the identity even when YAML and array shape are unchanged.

Save one `resolved_config` mapping with new checkpoints, using the existing canonical section/key structure, plus the preprocessing schema version and identity. Metadata snapshots are provenance, not alternative runtime configuration keys. The checkpoint is authoritative for preprocessing at inference; current repository defaults must not silently override it. If an explicit runtime preprocessing configuration is supplied, require an identity match or fail clearly before prediction.

Validate `get_feature_dim(checkpoint_cfg.features.set) == model_input_size` and the expected sequence length at the feature boundary. Reject unknown metadata schemas and unknown configuration keys. Keep label-map/class-count validation alongside these checks.

Caches with missing/mismatched identity must be regenerated before use. Do not accept a same-shaped `.npy` as proof of compatibility. Resume must compare the training checkpoint's preprocessing identity to the dataset identity; an old optimizer/model state cannot silently resume on corrected features.

Legacy raw state dictionaries do not contain enough information to reconstruct preprocessing. Do not infer schema or feature set from tensor width. They may remain usable through an explicitly documented precomputed-feature interface, but raw-coordinate inference and training resume under corrected semantics must fail with a clear migration/retraining instruction. Update synthetic checkpoint fixtures to create the new metadata rather than preserving silent legacy defaults. Do not relabel an old checkpoint as corrected without retraining.

**Test cases / acceptance**

1. Same configuration/schema produces the same hash independent of dictionary order. A schema-only change invalidates a same-shaped cached array.
2. Each numerical/acceptance override changes identity where appropriate; worker/path changes do not. Store test metadata and arrays entirely in `tmp_path`.
3. A checkpoint with `sequence_length=48`, `max_gap_frames=0`, `pad_frames_around=5`, and `features.set=hands` must drive those values even when repository defaults differ.
4. Reject configuration/schema/input-width/sequence-length mismatches before model forward; reject unknown YAML/override/checkpoint config keys with `KeyError`.
5. Missing legacy metadata blocks raw-coordinate inference with an actionable message; a valid new checkpoint round-trips its config and identity exactly.
6. Resume from a matching new checkpoint preserves the existing optimizer/scheduler/scaler tests. A mismatched dataset/checkpoint identity fails before updating any weights.
7. A stale cache with plausible manifest entries and `(L,D)` shape is still refused or rebuilt. Valid matching caches retain correct coverage/count metadata on resume.

## P0.8 — Regression checks and corrected-baseline handoff

- [x] Run focused mathematical and parity tests, then the full self-contained suite.
- [x] Verify pyflakes and optional-package import isolation.
- [x] Regenerate features with the corrected schema before the next training run.
- [x] Establish a corrected GRU baseline using the existing participant split and unchanged model/training settings.

**Target files**

- All test files touched above; `tests/test_kaggle_access.py`, `tests/test_env.py`, and the existing config tests remain regression coverage for project invariants.
- `reports/training_fix_notes.md`: append the corrected semantics, replaced test expectations, cache/checkpoint migration, and temporary time-stretch restriction.
- New `reports/p0_validation_report.md`: record commands, results, artifact identities, and later baseline results.

**Verification commands, from the repository root**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_preprocess.py tests/test_augmentation.py tests/test_pipeline.py tests/test_inference.py tests/test_config.py tests/test_dataset.py tests/test_train.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pyflakes src scripts tests app.py
```

Run mathematical tests on CPU. CUDA parity is additional coverage when available. Run the suite in a clean environment with declared core/test dependencies and without optional Kaggle client packages; stub optional detector dependencies for inference unit tests. Credentials, network access, and untracked model/data files must not be required. Preserve the credential precedence, SDK import isolation, and absolute-path/environment resolution rules in `AGENTS.md`.

**Mathematical acceptance criteria**

```text
unsupported coordinate blocks == 0
hand and pose support are binary
masked relative/velocity equations hold after every supported transform
double reflection recovers canonical input
offline_features(raw,cfg) ~= inference_features(raw,cfg)
cache_identity == checkpoint_identity == resolved preprocessing identity
```

Use exact equality for masks and zero sentinels, and float32 tolerances for arithmetic. Test expected mathematics independently; agreement between two implementations can preserve a shared bug.

**Baseline handoff, after engineering acceptance**

Keep the vocabulary and signer manifests fixed. Report exclusions caused by stricter validity; never reassign participants to compensate for exclusions. Retrain the same GRU from initialization on regenerated features, retaining existing training hyperparameters and seeds for a controlled comparison. Record validation Top-1, Top-3, Macro-F1, per-gloss confusion, and per-signer counts. Do not repeatedly consult held-out test signers during these fixes; reserve final test evaluation for the selected implementation.

Measure preprocessing latency separately and record end-to-end CPU p50/p95 latency on the target machine. P0 guarantees consistency when its tests pass; an accuracy improvement or 30 FPS deployment performance must be measured, not assumed.

**Definition of done**

All eight task checklists are implemented and verified; all four original regressions have direct mathematical tests; train/inference parity holds under nondefault canonical settings; caches and checkpoints cannot silently cross schema boundaries; full tests and pyflakes pass. Record any unavailable hardware measurements explicitly. Architectural changes and new feature channels begin only after this corrected baseline is available.
