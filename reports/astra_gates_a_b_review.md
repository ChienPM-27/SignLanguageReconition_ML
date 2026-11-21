# Gate A and Gate B code review — 2026-11-05

**Verdict: Gate A is not passed. Gate B is not passed.** Several earlier fixes work, but eight P1 issues remain. The execution log's claim that both gates are fully verified exceeds the current evidence.

**Recheck at 13:26 +07:00:** all 82 earlier snapshot file hashes matched the current tree. A new snapshot rerun reproduced all eight P1 findings and **237 passed / 1 failed / 1 warning in 22.21s**; pyflakes is clean; optional-Kaggle-absent checks pass 72 tests. New evidence is under `.review-wave-ab/20261105T062449161992Z/`. See [the phased remediation plan](astra_remediation_plan.md). The findings below remain open; no production fix was made by this recheck.

This review adds a report and diagnostic artifacts. Production source, configuration, notebooks and earlier reports were preserved.

## Scope and verification

- HEAD: 8e335b7a46f1e2e9d1b0c47850aa053c852b55f0; reviewed the uncommitted working tree.
- Captured **82 files at 2026-11-05 12:27:06 +07:00**. No file changed during capture or between capture and completion of the checks.
- Snapshot and hashes: .review-wave-ab/20261105T052706212692Z/snapshot/ and snapshot_manifest.json.
- Runtime: Windows 11; Python 3.13.15; PyTorch 2.14.1; NumPy 2.5.3; PyArrow 26.0.0; pytest 9.1.1; pyflakes 4.0.3. CUDA unavailable.
- Criteria: reports/astra_next_implementation_plan.md:108 and :178. Gate A requires A0–A2 and synthetic smoke; real baseline measurements may remain pending. Gate B requires feature mathematics, NumPy/Torch CPU parity, offline/raw parity under nondefault configuration, and rejection of unusable clips before forward.
- Graphify expansion: checkpoint resume cache identity frozen split layout geometry fingertip bend augmentation pipeline. With the CLI unavailable, a standard-library BFS over the existing graph located related modules; 1,183 nodes were visited. Source and synthetic probes independently confirmed all findings. No graph rebuild or LLM extraction ran; extraction token cost was 0.
- Fixtures, parquet files, feature arrays and checkpoints were generated within the review directory. Existing datasets, models, videos and detector assets were not used. Detector construction was stubbed for raw predictor probes; real MediaPipe/video/webcam behavior and CUDA execution were not tested.

| Check | Result |
| --- | --- |
| Full snapshot pytest suite | **237 passed, 1 failed, 1 CUDA warning; 22.16s; exit 1** |
| Pyflakes on snapshot src/scripts/tests/app.py | **0 findings; exit 0** |
| Credential/path/setup tests with all kaggle imports blocked | **72 passed; 4.99s; exit 0** |
| Standalone scripts/21_evaluate.py --help | **Exit 0** |
| Independent diagnostic probes | **Exit 0**, with contract failures recorded below |
| Offline parquet features vs raw predictor input | **24 configurations, all 12 dimensions; maximum absolute difference 0** |
| NumPy vs Torch CPU transforms with supplied parameters | **Maximum absolute difference 1.79e-7**, below 1e-5 |
| Unaugmented loading through none/ram/device | **24 configurations × 3 modes; maximum absolute difference 0** |

The parity matrix uses T=8, max_gap_frames=1, epsilon=0.002, trim_no_hand=false, anatomical slot ordering, both palm-scale modes and independent tips/bends toggles. It covers flip with/without slot swap, rotation, jitter, shape scaling and image wrist scale/translation. This establishes parity for those fixtures. The epsilon=0.01 and dropout probes expose contracts that the matrix does not satisfy.

Probe exit 0 means diagnostics completed and saved observations; it does not mean the gates passed. Synthetic losses below are diagnostic values, not baseline metrics.

## Gate A blockers

### A-01 · P1 — Rejected resume can still change model weights

**Source:** src/training/train.py:399, :408, :413.

Rollback protects state application, but epoch/best-loss/patience parsing happens afterward. A payload with valid schema, config, identity, labels and optimizer state but epoch="invalid-epoch" copies the checkpoint weights, then raises ValueError at int(epoch). The rejected resume leaves changed weights.

**Probe:** resume.invalid_epoch_after_commit → accepted=false, model_mutated=true. Epochs=0 and no loaders ensure no training step caused the mutation.

Validate counters/history and all states before application. Include the whole application in an atomic commit/rollback, including scheduler/scaler state or independently validated copies.

### A-02 · P1 — Resume accepts changed semantics and missing required metadata

**Source:** src/training/train.py:337, :353, :354, :382.

Saved config now passes strict parsing and identity recomputation, but its model semantics are not compared with the runtime model/config. Tensor shape checks cannot distinguish mean from last pooling. Class names/maps are compared only when present; missing optimizer state silently resets optimizer history.

**Probes accepted and replaced weights:**

- Saved model.pooling=mean, runtime model/config pooling=last.
- Missing both class_names and label_map while the caller supplied them.
- Missing optimizer_state_dict.

Require ordered class metadata and complete resume state. Compare model semantics explicitly and define permitted runtime/path/epoch overrides; tensor shape equality is insufficient.

### A-03 · P1 — Frozen splits do not freeze per-sequence provenance

**Source:** src/data/split.py:106, :127, :128, :136, :139, :234.

The fix checks each split's allowed signer set and whole-class coverage, but snapshots do not retain each sequence's original signer/gloss/label. Missing IDs are silently filtered. Snapshot schema is not validated; generated snapshots lack content hashes and membership identity.

**All four probes accepted:**

- A train sequence changed signer from train_p0 to the already allowed train_p1.
- Two sequence IDs exchanged gloss/label while retaining vocabulary and class counts.
- One frozen train sequence disappeared: output shrank from 4 to 3 with no explicit exclusion report.
- Frozen manifest schema_version=999.

Freeze per-sequence signer/gloss/label provenance and content hashes. Validate schema and mapping; report every exclusion with gloss/signer/reason. Rebind must preserve frozen membership semantics.

### A-04 · P1 — Final evaluator bypasses checkpoint validation contracts

**Source:** scripts/21_evaluate.py:78, :82, :108, :120, :134.

The evaluator validates schema/identity but falls back to external YAML if resolved_config is absent. It does not validate unique class names, class order against label_map, or contiguous indices. Arbitrary model overrides are applied before model construction; preprocessing identity does not protect model pooling.

**All evaluated successfully:**

- Missing resolved_config, supplied equivalent external YAML.
- Duplicate class names.
- Class order contradicting the label map.
- Noncontiguous label indices.
- Selected last-pooling checkpoint overridden to mean; synthetic loss changed from 0.7106 to 0.7066.

Use a shared checkpoint validator for resume, predictor and evaluation. Final evaluation should reconstruct the selected model's validated semantics and allow only defined runtime/path overrides.

## Gate B blockers

### B-01 · P1 — Augmentation restores motion across a long raw dropout

**Source:** src/features/pipeline.py:313, :347, :369; src/data/augmentation.py:71, :86, :102; src/data/augmentation_gpu.py:105, :120, :148; src/data/preprocess.py:210.

Initial extraction now computes transition validity across the whole source interval. These masks are not retained in result provenance/cache. Augmentation recomputes derivatives from adjacent target masks only.

For N=100, T=4 and hand dropout frames 40–60, target positions 33 and 66 are supported. Clean extraction zeros the gap-crossing transition; flip recomputation restores it.

| Gap-crossing channel | Clean | NumPy flip | Torch CPU flip |
| --- | --- | --- | --- |
| Maximum local fingertip displacement | 0 | **0.165** | **0.165** |
| Maximum wrist displacement | 0 | **0.066** | Recomputed from endpoint masks |

Agreement between implementations does not establish continuity. Preserve observation/usable/transition provenance through cache loading and augmentation, permute it on slot swaps, and use it for all derivative recomputation.

### B-02 · P1 — Canonical epsilon still does not propagate through augmentation

**Source:** src/data/dataset.py:127, :355, :374, :410; src/data/device_loader.py:86; src/data/augmentation.py:369, :422, :490, :499.

The dataset stores/passes epsilon for per-sample transforms. DeviceDataLoader calls augmentation with only batch/config/generator, so Torch uses its default 0.0001. CPU shape_scale and scale_translate also recompute with the default; neither helper accepts/passes the canonical value.

**With epsilon=0.01 and normalized straight bones of length 0.005:**

- Flip through none/ram: bend cosine **0**.
- Same config through the actual device loader: bend cosine **1**.
- CPU shape scaling or wrist scale/translation: clean bend **0 → 1**.
- Direct Torch shape scaling with canonical epsilon supplied: bend stays **0**.

The bones are below the configured validity threshold, so zero is correct. Pass the existing preprocessing epsilon through the device loader and every public transform/recomputation path; do not create an augmentation epsilon alias.

### B-03 · P1 — Body wrist velocity ignores pose continuity

**Source:** src/features/normalization.py:629, :631, :696.

Torso-relative velocity uses geometry at target endpoints and hand_transitions. Pose transitions are applied to torso velocity but omitted from body wrist velocity.

For continuous hands and pose dropout frames 40–60, N=100/T=4 yields supported endpoints on either side.

**Probe:** body wrist xy displacement across the missing pose interval ≈ **0.165** for each hand; torso velocity for the same interval is correctly **0**.

Geometry transitions must require continuous usable hand and pose support. Apply that joint transition mask during extraction and augmentation.

### B-04 · P1 — No supported target frames still permits model forward

**Source:** src/features/pipeline.py:198, :367; src/inference/predictor.py:407, :410.

Quality gating counts raw observed frames before resampling, without a final target-support check.

For N=100/T=4, trim_no_hand=false and observations only at frames 10–12, the raw minimum of three passes. Target positions [0,33,66,99] retain no hand support.

**Probe:** is_usable=true; target support count **0**; feature tensor entirely **0**; forward count **1**; closed-set prediction returned.

Reject sequences that lose all required support after resampling. Define torso geometry coverage separately, preserve rejection reasons, and verify predictor forward is never called for rejected sequences.

## Other gate gaps

**Test isolation:** tests/test_inference.py:303 and :310 create glosses_csv but omit it from predictor construction. The predictor recursively searches for an unrelated glosses.csv. In the clean snapshot, test_predictor_rejects_invalid_checkpoint_classes fails with FileNotFoundError before reaching its intended label assertions. Pass the fixture CSV explicitly to both calls. Do not add a global/repository gloss file. This violates AGENTS.md's self-contained-test rule and prevents certifying the logged 238/238 result.

**Evaluation audit:** src/training/evaluate.py:67 omits sample membership, preprocessing/split identities and the numerical confusion matrix with ordered glosses from JSON. All evaluator probes confirmed these omissions. The PNG and participant aggregates do not satisfy A2's audit contract.

**Adapters/capture:** the predictor now rejects missing pose for torso mode, closing the earlier silent-forward finding. MediaPipeHandExtractor still extracts only hands; actual video/webcam pose and mirrored-handedness parity remain incomplete. The realtime demo now derives capture buffer length directly from preprocessing T (scripts/31_realtime_demo.py:166, :180), coupling capture duration to resampling and truncating SPACEBAR clips. Use a separate canonical capture concept and resolve detector mirroring before claiming deployed parity.

**Baseline handoff:** the notebook still runs --stage all without a frozen-manifest override (notebooks/islr_phase2.ipynb:205), RF test evaluation (:241), and test evaluation of the newest discovered checkpoint (:284). Its default T4 preset changes batch/epochs from the controlled baseline. The baseline config/vocabulary/membership freeze and selection-provenance audit remain incomplete.

**Measurements/status:** real-data baseline measurements remain pending, as Gate A permits. This review establishes no model accuracy/latency or architecture comparison. The execution log references 201, 229 and 238 passing tests and claims GPU verification despite local CPU-only evidence. Preserve historical results and amend current status to match evidence.

## Earlier fixes confirmed or narrowed

| Earlier finding | Current evidence |
| --- | --- |
| Predictor crash with valid class_names | Fixed; construction and raw prediction work across the matrix |
| Training runtime/train/val cache identity comparison missing | Fixed in train_single_seed; negative checks pass |
| Unknown saved config / contradictory hash / reversed supplied labels accepted | Regression checks pass; same-shape semantics and absent metadata remain A-02 |
| Extra tensor / bad optimizer rejection mutates model | Regression checks pass; post-application scalar parsing remains A-01 |
| Foreign metadata certifies unrelated cache | Ownership regression passes |
| Corrected predictor accepts missing/wrong schema or identity | Strict predictor checks pass |
| Standalone evaluator import bootstrap | Direct --help exits 0 |
| Whole-class loss / wholly foreign signer set accepted on rebind | Rejected now; per-sequence freeze remains A-03 |
| Independent feature combinations rejected by loader | All 12 dimensions load in every preload mode |
| Short-bone cosine denominator clamp | Fixed; valid short straight bones yield 1 |
| Torso wrist z corrupted by jitter | Corrected boundary/transform paths; supplied-parameter parity passes |
| CPU ignores shape_scale | Operation restored; epsilon propagation remains B-02 |
| Shoulder median uses rejected pose observations | Validated o_pose is used now |
| Raw long hand dropout ignored in initial extraction | Clean derivative zero now; augmentation loses support in B-01 |
| Preprocess reuse/report derives base D only | Both derive D from full config now |

## Proposed status

| Task | Reviewed status |
| --- | --- |
| A0 | Checks rerun; suite fails test isolation |
| A1 | Needs A-01/A-02 and shared checkpoint validation |
| A2 | Needs A-03/A-04 and complete evaluation audit |
| A3 | Synthetic paths run; controlled handoff incomplete; real measurements pending |
| **Gate A** | **Not passed** |
| B0 | All dimensions exercised; registry/loader combinations work |
| B1/B2 | Main formulas work; joint pose continuity/final support checks need correction |
| B3 | Needs transition provenance/epsilon propagation; detector/capture work pending |
| **Gate B** | **Not passed** |

Correct A-01–A-04 and B-01–B-04, add self-contained regressions for these probes, repair the fixture failure, and rerun required checks. Current evidence does not support marking these gates verified or unlocking Wave C under the execution log's stated dependency.

## Reproduction artifacts

Review directory: .review-wave-ab/20261105T052706212692Z/.

- snapshot_manifest.json: timestamp, HEAD and hashes.
- snapshot/: reviewed source/config/tests/documents.
- suite_tests.xml and optional-absent_tests.xml: test evidence.
- probe_results.json: acceptance decisions and numerical observations.
- layout_parity_matrix.json: all 24 configurations.
- graph_query.json: vocabulary, seed nodes and BFS relationships.
- checks.json: commands, exit codes and captured output.
- probe-inputs/: generated synthetic parquet/cache/checkpoint artifacts.

Runners: .review-wave-ab/capture.py, run_checks.py and probes.py. They select the snapshot through latest.txt. Existing runs are preserved. Make a new capture before rerunning probes because their input directory deliberately refuses reuse.

