# Plan tiếp theo theo đề xuất Astra — handoff cho Gemini

Ngày lập: 2026-11-05. Trạng thái lúc lập: **planned**. **Cập nhật 13:26 +07:00: Wave A/B đã có implementation một phần nhưng cả hai gate chưa đạt; Wave C–E chưa hoàn tất.**

Đọc [audit và kế hoạch sửa theo phase](astra_remediation_plan.md) trước khi triển khai tiếp. Tài liệu đó đối chiếu current code, tái hiện tám P1 và thay thế thứ tự handoff/completion claims cũ. Các bảng “lúc lập plan”, checklist và phần giao việc dưới đây được giữ làm design/history, không phản ánh tiến độ hiện tại. Không triển khai lại Wave B từ đầu và không coi execution log cũ ghi VERIFIED là đủ để mở Wave C.

Mục tiêu: khép kín tính nhất quán của pipeline, thiết lập GRU baseline có thể tái lập, ablate geometry/finger-motion, rồi so sánh GRU với temporal CNN và hand-graph + temporal CNN trên cùng dữ liệu. Accuracy và latency là kết quả cần đo; không được ghi mức cải thiện dự đoán thành kết quả thực nghiệm.

## 1. Điểm bắt đầu và phạm vi

Đọc `AGENTS.md`, `reports/p0_implementation_plan.md`, `reports/p0_validation_report.md` và code hiện tại trước khi làm. Plan P0 cũ đánh dấu completed nhưng không đủ bằng chứng để coi toàn bộ P0.8 đã hoàn tất. Giữ lịch sử kết quả cũ và bổ sung trạng thái có bằng chứng.

| Hạng mục | Trạng thái tại lúc lập plan | Hành động tiếp |
| --- | --- | --- |
| Pose flip, bounded imputation, support-aware resampling, endpoint-gated derivatives | Đã có implementation và regression tests trong repo | Audit và chạy lại; không viết lại toàn bộ P0 |
| Shared preprocessing, dominant-slot mapping, schema v2 | Đã có | Khép kín validation và adapter parity |
| Cache/checkpoint/resume compatibility | Còn lỗ hổng thấy trực tiếp trong code | Wave A |
| GRU baseline | YAML hiện dùng `hands_pose`, T=32, GRU 128 × 2, unidirectional, mean/max, dropout 0.3, smoothing 0.1 | Freeze và retrain trên corrected cache |
| Body-relative wrists, fingertip displacement, bend cosines | Chưa tích hợp | Wave B |
| MaskedTCN, HandGraphTCN | Astra đưa standalone prototypes; repo chưa có | Wave C |
| Pretraining, SupCon, face/hybrid, adversarial objectives | Chưa tích hợp | Wave E có điều kiện |
| Synthetic checks / parameter counts của Astra | GRU 215,572; TCN 293,524; graph-TCN 250,212 tại D=160, 20 lớp | Dùng làm mốc kiểm tra cấu trúc; không phải benchmark accuracy/CPU |

Các vị trí cần kiểm chứng lại vì repo có thể đổi:

- `src/data/dataset.py::load_and_validate_feature_file`: đang tự pad/truncate khi T không khớp.
- `ISRLandmarkDataset`: metadata không bắt buộc; thiếu schema có thể được bỏ qua; còn fallback `hash`.
- `src/training/train.py::train_gru`: resume nạp model/optimizer trước khi so identity; thiếu saved identity có thể được bỏ qua; resume path không tồn tại có thể trở thành fresh run.
- `src/inference/predictor.py::_load_checkpoint`: có thể mặc định schema khi metadata thiếu; dựng model bằng suy luận từ weights thay vì hoàn toàn từ validated config.
- `scripts/20_train.py::train_single_seed`: tự tạo test loader và evaluate test sau mỗi seed; chưa có comprehensive validation report tương đương.
- `src/data/split.py::create_signer_splits`: có thể tìm seed mới để thỏa class coverage. Đây là hành vi cần tránh khi tái dựng feature cho một experiment đã freeze.
- `scripts/31_realtime_demo.py`: hand-only extraction và buffer 32 frame; T resampling không phải độ dài capture buffer. Body-relative geometry cần pose thật ở inference.

Graphify được dùng để định vị các module; graph có thể cũ. Mọi quyết định implementation phải được xác nhận bằng source/tests hiện tại. Report cũ ghi 200 tests pass; plan này chưa chạy lại suite.

## 2. Quy tắc xuyên suốt

1. Mọi tunable nằm ở `configs/islr.yaml`, map vào dataclass trong `src/utils/config.py`. Không thêm alias config. Key lạ phải `KeyError` cả YAML, dictionary loader và `--set`, kể cả nested key mới.
2. Giữ credential precedence, SDK import isolation và `env.resolve_paths`. Không log secrets; không dùng đường dẫn máy cá nhân làm default.
3. Tests tự tạo arrays/parquet/metadata/checkpoints trong `tmp_path`; không dùng cache, model hay video có sẵn. Optional detector/Kaggle libraries phải stub hoặc lazy-import.
4. Freeze 20 glosses, label map, participant assignment và sequence membership. Không lựa model, loss hay augmentation theo sample videos hoặc test accuracy.
5. Giữ raw inputs anatomical `[left,right]`; dominant ordering làm đúng một lần và lưu `slot_to_anatomical`. Pose luôn theo anatomical order.
6. Phân biệt observation support, imputed usable support, geometry support và padding. Một timestep mất tay không tự động trở thành padding.
7. T=32 là 32 phase steps của cả clip đã resample; không mặc định bằng 32 camera frames. Không gọi displacement per phase step là vận tốc vật lý.
8. Mỗi thay đổi semantics phải có identity mới. Không đổi tên/schema của artifact cũ để giả vờ đã retrain.
9. Synthetic pass chứng minh contract, không chứng minh model tốt hơn. Thiếu dataset/GPU/CPU target thì ghi pending cụ thể; tiếp tục phần engineering không phụ thuộc phần thiếu.

## 3. Wave và dependency

| Wave | Task | Depends on | Effort tương đối | Gate |
| --- | --- | --- | --- | --- |
| A | A0–A3: audit, compatibility, frozen protocol, baseline handoff | — | Medium | Source/tests đúng; protocol đã freeze |
| B | B0–B3: schema geometry, body-relative wrists, finger cues, augmentation/adapters | A1–A2 | Medium–high | Independent mathematics + train/inference parity |
| C | C0–C3: model factory, masked TCN, graph-TCN, integration | A1–A2; schema contract B0 | Medium | Forward/backward, padding invariance, checkpoint roundtrip |
| D | D0–D3: feature/architecture/T ablations, CPU benchmark, final evaluation | A3 baseline + B/C engineering | GPU time + medium engineering | Validation-led selection, measured performance |
| E | E0–E3: pretraining, SupCon, targeted face/hybrid, later objectives | D validation analysis | Separate work packages | Có evidence và comparison rõ ràng |

Gemini có thể chuẩn bị code B/C bằng synthetic inputs sau gate A. Không được tuyên bố kiến trúc tốt hơn hoặc tiến hành architecture comparison trước corrected baseline. Dùng commit/task nhỏ để review; không làm toàn bộ wave trong một diff khổng lồ.

## 4. Wave A — Khép kín P0 và baseline protocol

### A0 — Audit có bằng chứng

- [ ] Đọc working-tree diff; bảo toàn thay đổi người dùng; ghi commit hash, dependency versions và hardware.
- [ ] Chạy suite + pyflakes, ghi exit code và số tests thực tế. Đối chiếu các test được report cũ nhắc tới với tên test thật.
- [ ] Lập `reports/astra_execution_log.md`: task, changed files, commands/results, pending measurements. Sửa trạng thái P0/report nếu completion claim vượt bằng chứng; không xóa kết quả lịch sử.

**Acceptance:** phân biệt rõ implemented, verified engineering, measured experiment và pending. Không bịa cache identity hay baseline metrics.

### A1 — Strict cache/checkpoint/config compatibility

Targets: `src/features/schema.py`, `src/data/dataset.py`, `src/data/preprocess.py`, `src/training/train.py`, `src/inference/predictor.py`, `scripts/20_train.py`, config và related tests.

- [ ] Cache bắt buộc có `preprocess_meta.json` với canonical `config_hash` và `parameters`; bỏ fallback tên key `hash`. Validate exact supported schema, identity từ parameters và expected identity từ runtime config. Hash computation dùng một shared implementation.
- [ ] Định vị metadata ở **cache thật chứa từng feature file**, không chỉ thư mục split CSV. Explicit `features_dir` phải có semantics rõ; missing/conflicting metadata hoặc các cache trộn identity phải fail. Không lấy metadata của file đầu để chứng nhận toàn dataset.
- [ ] Reject T/D mismatch, nonfinite arrays, nonbinary support, unsupported layout; không pad/truncate ở fixed-length cache loader. Cover `none`, `ram`, `device` preload.
- [ ] Trước bất kỳ model/optimizer/scheduler/scaler state mutation khi resume, load payload để đọc, validate schema + strict resolved config + recomputed identity + train/val dataset identities + architecture/classes. Sau khi hợp lệ mới apply state.
- [ ] Resume thiếu path, thiếu identity/config/schema hoặc identity mismatch phải fail với hướng dẫn retrain. Không silently start fresh. Giữ optimizer/scheduler/scaler roundtrip với metadata hợp lệ.
- [ ] New checkpoint lưu validated `resolved_config`, explicit preprocessing schema/identity và ordered class names/label mapping. Snapshot metadata là provenance, không phải config alias.
- [ ] Inference không tự suy schema cho checkpoint có metadata thiếu. Instantiate new models từ validated config; weights chỉ dùng để verify consistency. Mapping trùng label, label không liên tục hoặc class count mismatch phải fail.
- [ ] Legacy raw state dict chỉ được qua precomputed-feature interface có contract rõ; raw inference và resume corrected training bị chặn. Không rộng hóa silent legacy fallback để tests cũ pass.
- [ ] `preprocess.normalization_epsilon` phải finite positive; audit finite/type/range validation cho keys liên quan. Canonical override phải tới cả pipeline, artifact identity và checkpoint.

**Tests bắt buộc:** missing/corrupt metadata; same shape nhưng khác gap/trim/epsilon/feature mode/schema; metadata ở feature parent; mixed caches; rejected resume giữ weights nguyên vẹn và không train step; valid metadata roundtrip; unknown checkpoint config key `KeyError`; label/schema mismatch fail trước forward. Chuyển synthetic fixtures sang tạo metadata đúng.

### A2 — Freeze splits và tách validation khỏi final test

Targets: `src/data/split.py`, `scripts/10_prepare_data.py`, `scripts/20_train.py`, `src/training/evaluate.py`, `src/training/metrics.py`; proposed `scripts/21_evaluate.py` và `tests/test_evaluate.py`.

- [ ] Snapshot ordered vocabulary/labels, participant-to-split assignment, sequence membership và content hashes. Identity của split membership không gồm path tuyệt đối hoặc model seed.
- [ ] Rebind frozen sequence IDs vào manifest/cache mới. Giữ participant assignment; báo exclusions theo gloss/signer/reason. Nếu mất class coverage thì fail/report; không đổi seed hoặc chuyển signer để bù.
- [ ] Audit selection provenance: held-out landmark separability/handshape statistics có tham gia chọn vocabulary hay không? Nếu không xác định được thì ghi limitation; không claim untouched test. Việc xây benchmark mới phải là quyết định riêng.
- [ ] Refactor evaluator dùng chung cho validation/final test: Top-1, Top-3, Macro-F1, confusion theo gloss, per-signer accuracy/counts, sample membership và identity. Metadata phải khớp order thật từ dataset sau filtering, không zip prediction với CSV chưa lọc.
- [ ] Default training chỉ dùng train/val; không đọc test features hay evaluate test. Thêm canonical `evaluation.evaluate_test: false`; explicit final evaluation qua script và config này sau khi chọn implementation.
- [ ] Multi-seed summary mặc định tổng hợp validation metrics. Best checkpoint vẫn chọn bằng validation loss; model/config selection dùng mean validation Macro-F1, rồi Top-1/CPU p95 nếu tương đương. Chốt quy tắc trước chạy.
- [ ] `--seeds` nếu giữ convenience CLI phải cập nhật canonical resolved `training.seeds` và `training.num_seeds`; lưu effective per-run seed. Mỗi checkpoint tái lập đúng seed đã chạy.

Proposed canonical key: `split.manifest_path: null` — null dành cho initial split generation; experiment đã freeze phải trỏ snapshot. Snapshot ghi membership/provenance, không là nguồn runtime tunables thứ hai.

**Acceptance:** không overlap signer; rebind không thay membership ngoài exclusions đã báo; training vẫn chạy được khi test feature path được stub để raise nếu truy cập; final evaluator dùng selected checkpoint và không train/tune; config dump phản ánh seed thật.

### A3 — Corrected GRU baseline và reproducible commands

- [ ] Freeze YAML/config snapshot hiện tại trước thêm feature/model options. Baseline: D=160, T=32, GRU hidden=128, layers=2, dropout=.3, bidirectional=false, mean_max, CE smoothing=.1, training/augmentation giữ nguyên snapshot, seeds 42/43/44.
- [ ] Regenerate cache bằng entrypoint có thật: `scripts/10_prepare_data.py --stage preprocess`. Rebind frozen split; không chạy lại selection/download nếu input đã có và không cần.
- [ ] Smoke training nhỏ bằng synthetic metadata hợp lệ; CPU đủ cho engineering. Có dữ liệu/hardware phù hợp mới chạy baseline thật từ initialization.
- [ ] Xuất baseline report: cache/schema identity, split hash, exclusions, class/signer counts, validation metrics từng seed, best epoch/loss, train–val gap, runtime/hardware. Test metrics chưa chạy ở wave này.
- [ ] Cập nhật notebook `notebooks/islr_phase2.ipynb`: commands thật, effective config, schema migration và validation-only mode. Không trộn T4 balanced preset với controlled baseline; batch/epochs khác là experiment riêng.

**Gate A:** A0–A2 và synthetic smoke pass. A3 chỉ đánh dấu measured khi có artifacts baseline thật; nếu thiếu data/GPU thì để pending. Không tái dùng Phase A checkpoint schema cũ làm corrected baseline.

## 5. Wave B — Feature semantics theo Astra

### B0 — Canonical options và feature layout registry

Giữ `features.set` làm lựa chọn `hands`/`hands_pose`; không thêm bộ alias `feature_set`/`input_dim` vào YAML. Thêm đúng các concepts sau vào YAML/dataclass khi triển khai wave B:

| Canonical key | Default để giữ baseline | Values / ý nghĩa |
| --- | --- | --- |
| `features.wrist_reference` | `image` | `image`, `torso_xy` |
| `features.palm_scale` | `frame` | `frame`, `clip_median` |
| `features.fingertip_displacement` | `false` | Append 30 local displacement dimensions |
| `features.bend_cosines` | `false` | Append 20 articulation dimensions |

- [ ] Layout registry định nghĩa slices/count/coordinate spaces cho mỗi combination; derive D từ config, không heuristic `D>=155`. Giữ API cũ `get_feature_dim(set)` nếu cần như wrapper cho base layout; runtime dùng full layout.
- [ ] `torso_xy` yêu cầu `hands_pose`. Hai geometry-support channels append sau base channels, trước fingertip/bend blocks. Đây là format rule, không toggle tunable.
- [ ] D chính xác: base 160; body-relative 162; body + tips 192; body + tips + bends 212. Astra nói khoảng 210 trước reliability; 2 channels thêm ở đây cần thiết để phân biệt mất torso với wrist tại origin. Với `image`, base/tips/bends là 160/190/210.
- [ ] Khi mở rộng semantics, bump canonical preprocessing schema lên v3; include mọi numerical/acceptance option mới trong identity. Legacy v2 artifacts giữ nguyên, không relabel. Regenerate cache và rerun baseline 160 trong v3; test chứng minh baseline-disabled output không đổi về numerical values.
- [ ] Ghi schema contract trong `reports/astra_feature_contract.md`: slices, units, masks, coordinate origin, transform rules, migration và detector assumptions.

### B1 — Body-relative wrists và robust scales

Targets: shared pipeline/normalization/schema, raw metadata và tests.

Định nghĩa deliberate extension cho mode `torso_xy`:

```text
c[t] = (left_shoulder_xy[t] + right_shoulder_xy[t]) / 2
ell = median observed valid nondegenerate shoulder widths trong clip
g[t,h] = hand_usable[t,h] AND pose_usable[t] AND valid(ell)
w_body[t,h,xy] = (w_image[t,h,xy] - c[t]) / ell nếu g, else 0
w_body[t,h,z] = 0
r_body[t,xy] = w_body[t,0,xy] - w_body[t,1,xy] nếu g của cả hai
v_body[t,h,xy] = w_body[t,h,xy] - w_body[t-1,h,xy] nếu g ở cả hai endpoint
r_body_z = v_body_z = 0
```

- [ ] Ước lượng `ell` từ observations đã validate, trong shared trim interval, trước imputation/resampling. Không estimate scale từ missing zeros hoặc pose đã normalized. Không có observed shoulder width hợp lệ thì geometry unsupported.
- [ ] Dùng usable torso center sau bounded resampling. Giữ local handshape khi pose mất; zero riêng body-location/derived channels và `g`. Không fallback image coordinates vào cùng body-relative channels.
- [ ] Giữ hand support ở 141:143 cho shape; geometry support là block riêng. Không đổi hand observation provenance khi torso missing.
- [ ] Hand z chỉ là intra-hand shape cue. Không trộn hand z và pose z thành shared 3D geometry; pose z hiện tại giữ trong pose-only branch có semantics riêng.
- [ ] `clip_median` palm scale: median observed nondegenerate wrist-to-landmark9 distance cho từng anatomical hand trong shared trim interval, rồi áp dụng slot permutation. Một scale mỗi hand/clip; giữ `frame` baseline mode.
- [ ] Mode clip-median dùng future context và dành cho completed isolated clips. Slowly updated streaming scale là extension riêng, không giả vờ hai modes tương đương.
- [ ] Không suy aspect ratio từ parquet khi không có image dimensions. Wave này dùng normalized image xy và ghi limitation. Nếu sau này thêm correction thì tạo canonical option/metadata/schema mới áp dụng đồng nhất mọi adapter.

**Independent tests:** cùng translation và scale của wrists/shoulders cho body xy bất biến; rigid torso translation không tạo hand-relative motion; left/right mapping đúng; missing/degenerate shoulders; one-hand/no-pose clips; imputed center; robust scale ít bị một shoulder outlier; clip-palm scale không đưa NaN/Inf vào features.

### B2 — Fingertip displacement và articulation

- [ ] Tip indices format constants `(4,8,12,16,20)`. Append differences của **local normalized handshape**: 2 hands × 5 tips × xyz = 30. First timestep zero, masked endpoints; không nối observation qua long dropout.
- [ ] Bend cosines: thumb chain `(1,2,3,4)`; finger chains `(5,6,7,8)`, `(9,10,11,12)`, `(13,14,15,16)`, `(17,18,19,20)`. Hai consecutive bone-pair cosines mỗi chain: 2 × 5 × 2 = 20.
- [ ] Dùng cosine, không `acos`. Validate both bone lengths với canonical epsilon; degenerate/unsupported pair zero; clamp numerical result vào [-1,1]. Convention bone directions thống nhất và ghi rõ straight finger -> +1.
- [ ] Reliability/provenance cho imputed coordinates và valid motion transitions giữ structured metadata; không tự append thêm arbitrary channels ngoài contract B0.

**Tests:** analytically straight/right-angle/bent fingers; local finger motion khi wrist đứng yên; local rigid translation triệt tiêu; constant shape -> displacement zero; dropout boundaries; anatomical ordering; finite gradients không cần thiết cho NumPy extractor nhưng output phải finite float32.

### B3 — Augmentation và raw-adapter parity cho layout mới

Targets: `augmentation.py`, `augmentation_gpu.py`, `pipeline.py`, `landmark_extractor.py`, predictor, CLI demos/eval và `app.py`.

- [ ] Chuyển public augmentation sang layout-aware slices. Shared recomputation nhận local shapes, wrist mode và geometry support; không derive body motion bằng hand mask đơn thuần.
- [ ] Reflection body xy là `-x` quanh torso origin; image wrist vẫn `1-x`; pose reflection và anatomical permutation giữ đúng. Rotate/scale vận hành theo coordinate origin của từng block.
- [ ] Wrist-only image scale/translation không mặc định hợp lệ với torso mode. Wave đầu reject enabled legacy operation khi torso mode; commands body-relative phải tắt nó. Full-camera joint raw transform là work package riêng.
- [ ] Recompute tips sau shape jitter/scale/rotation/flip; recompute bends sau transforms có thể thay angles. Tip xyz reflection/rotation đúng; absent hand/geometry blocks vẫn zero. Không giữ clean derived cues với noisy coordinates ngoài separate denoising experiment.
- [ ] Hoặc public transform hỗ trợ layout mới với đúng mathematics, hoặc fail explicit; không silently augment chỉ 160 channels đầu.
- [ ] Video/webcam input phải cung cấp hand + actual pose observations/masks khi model cần torso mode. Resolve MediaPipe mirrored handedness convention, không double-swap. Stub detectors trong tests.
- [ ] Nếu adapter không có pose, expose explicit unsupported/waiting state cho body geometry; không claim deployed parity. Handshape có thể còn usable theo policy đã ghi; report pose-coverage để người dùng biết classifier thiếu location cues.
- [ ] Capture buffer length/duration là concept riêng với `preprocess.sequence_length`. Khi cần configurable buffer dùng canonical realtime keys, không alias T. Không đổi segmented clip thành 32-frame rolling task chỉ vì T=32.

**Gate B:** all supported layouts có mathematical tests, NumPy/Torch CPU parity và offline/raw inference parity dưới nondefault config. Unusable clips không forward. CUDA parity chỉ additional khi có hardware.

## 6. Wave C — Hai kiến trúc Astra và integration

### C0 — Một model factory và model-boundary contract

Proposed `src/models/factory.py`, `src/models/pooling.py`.

- [ ] Factory đọc canonical `model.name`: giữ GRU/LSTM; thêm `TCN`, `HAND_GRAPH_TCN`. Training và new checkpoint inference dùng chung factory, derive input width từ layout và class count từ ordered labels.
- [ ] Các classifiers expose `input_size`, `num_classes` và `forward(x, valid=None)` -> logits. `valid` là bool `[B,T]` cho real timesteps. Fixed-resampled dataset truyền all-true; không suy valid bằng hand mask hoặc `x!=0`.
- [ ] Validate mask shape/device, finite active inputs và ít nhất một real timestep mỗi clip. Padding có thể chứa arbitrary/nonfinite values nhưng phải masked_fill trước projection; không dùng NaN × 0 để sanitize.
- [ ] GRU fixed-length all-true phải giữ baseline behavior. Nếu cung cấp right padding thì dùng packed sequence/lengths và masked pooling đúng; reject unsupported arbitrary-hole padding trong wave đầu.
- [ ] Train/eval loaders hiện `(x,y)` giữ hoạt động. Nếu thêm padded batches, update `DeviceDataLoader` và model-boundary adapter cùng contract; variable-length dataset/bucketing chưa bắt buộc ở wave này.

### C1 — Residual depthwise temporal CNN

Proposed `src/models/tcn.py`; cấu trúc theo Astra:

- Input linear projection -> width 128.
- Four residual blocks, dilations `(1,2,4,8)`, depthwise Conv1d kernel 5, symmetric padding, LayerNorm, FFN expansion 2, SiLU, dropout .15.
- Receptive field 61 phase steps; không temporal downsampling.
- Mask input, trước depthwise convolution và sau từng residual; masked mean/max pooling; linear classification head.

Canonical `model.tcn`: `width: 128`, `kernel_size: 5`, `dilations: [1,2,4,8]`, `ff_expansion: 2`, `dropout: 0.15`. Validate positive widths/dilations/expansion, odd kernel, dropout finite in [0,1). Không dùng `model.dropout` như alias cho TCN dropout; key đó giữ concept GRU/LSTM hiện tại.

**Acceptance:** D=160/default prototype gives 293,524 parameters nếu faithfully reproduced; khi khác phải giải thích layer diff. `[B,T,D]` -> `[B,C]` cho T=16/32/48/64; finite loss/gradients; eval right-padding invariance với masked noisy padding; active missing-hand frames không bị loại.

### C2 — Shared hand graph + temporal CNN

Proposed `src/models/hand_graph_tcn.py`:

- Fixed 21-joint anatomical edges + self-loops + palm edges `(5,9),(9,13),(13,17)`; adjacency row normalized, register buffer.
- Shared across both hands: xyz lift width 48; three residual message-passing layers combining self/neighbors; LayerNorm + ReLU.
- Joint-identity readout: flatten ordered 21 joints -> embedding 64 per hand. Không average toàn joints trước readout.
- Gate missing hand ngay sau lift, mỗi residual và readout để biases không tạo absent-hand embedding.
- Fuse 2 × 64 embeddings với toàn non-local-shape channels theo registry, gồm wrists/pose/geometry/tips/bends. Temporal core dùng cùng `model.tcn` contract, override width=96 cho graph preset.
- Không thêm learned adjacency, cross-hand 3D edges hoặc face branch trong prototype đầu. Pose/anatomical mapping không được nối elbow vào dominant slot sai.

Canonical `model.hand_graph`: `hidden_size: 48`, `num_layers: 3`, `readout_size: 64`. Topology là format constant; temporal width/dropout chỉ ở `model.tcn`, không duplicate `temporal_width`.

**Acceptance:** D=160, graph defaults + TCN width96 gives 250,212 parameters nếu faithful. Same shapes/backward/padding tests như C1; one-hand/missing hand output finite; anatomical hand permutation theo configured slot semantics; all extension channels reach fusion; adjacency/device serialization roundtrip.

### C3 — Training/checkpoint/inference integration

- [ ] New checkpoint instantiate đúng từng model từ validated canonical config; preserve layout/schema/identity/classes; wrong architecture hoặc width fail trước forward.
- [ ] Predict features/raw/video dùng cùng trained model factory; không infer TCN/graph bằng GRU weight-key heuristics.
- [ ] Synthetic end-to-end train -> save -> predict -> resume mỗi architecture; tests cover strict config nested keys, nondefault T/D và logits parity sau load ở eval.
- [ ] Notebook có independent presets/commands cho ba architectures; effective config lưu theo run. Không thay default GRU baseline sang architecture mới.

**Gate C:** focused suite + full suite + pyflakes pass. Chỉ ghi synthetic verified; accuracy/latency vẫn pending tới wave D.

## 7. Wave D — Controlled experiments và đo latency

### D0 — Feature ablation trên cùng GRU

| Run | Features | Mục đích |
| --- | --- | --- |
| F0 | corrected 160, image wrists, frame palm scale | Baseline |
| F0-control | F0, tắt wrist-only scale/translation | Kiểm soát augmentation restriction của torso mode |
| F1 | F0-control + torso_xy, D=162 | Location geometry effect |
| F2 | F1 + clip_median palm scale | Scale stability effect |
| F3 | F2 + 30 tip displacement, D=192 | Finger-motion effect |
| F4 | F3 + 20 bends, D=212 | Explicit articulation effect |

- [ ] Seeds 42/43/44; same frozen split, model/training protocol. T=32. Giữ cache riêng từng identity.
- [ ] Compare F1 với F0-control để không gộp geometry benefit với việc tắt augmentation. Nếu một cue không giúp validation, giữ best subset; không mặc định F4 thắng.
- [ ] Nếu feature mode thay eligible samples, báo full-cohort metrics và supplementary matched-cohort comparison; không giấu exclusions hoặc đổi signer.
- [ ] Label smoothing 0/.05/.1 là separate later ablation; không đổi cùng feature step. Sample videos chỉ sanity, không selection metric.

### D1 — Architecture rồi temporal length

- [ ] GRU/TCN/graph-TCN cùng winning feature layout, T=32, frozen protocol và seeds. Thêm comparison D=160 nếu cần phân biệt architecture × feature interaction.
- [ ] Báo parameter count, validation loss/Top-1/Top-3/Macro-F1, per-gloss/signer, training time, CPU classifier p50/p95.
- [ ] First pass giữ optimizer/batch/epochs/augmentation giống nhau; architecture-specific tuning có equal declared budget và report riêng. Không tuned TCN so với untuned GRU mà gọi fair comparison.
- [ ] Sau architecture selection, compare T=32 với 48; T=64 chỉ khi validation confusions về brief/repetitive signs hỗ trợ. Regenerate/rebind mỗi T; không pad T32 cache thành T48.
- [ ] Native-length + bucketing, anti-alias filtering, monotonic time warp là work package riêng nếu T48 chưa đủ; không bật lại time_stretch cũ.

### D2 — CPU và end-to-end benchmark

Proposed `scripts/50_benchmark.py`, `reports/astra_benchmark_report.md`.

- [ ] Batch1, eval/inference mode, warmup, synchronized timing khi có accelerator, declared thread count/device/versions; đủ repetitions và report p50/p95.
- [ ] Tách classifier-only, preprocessing-only, extraction và complete camera loop (buffer/render included). Ghi sign-completion-to-prediction delay riêng.
- [ ] Canonical `benchmark`: `warmup_runs: 50`, `measured_runs: 500`, `batch_size: 1`, `num_threads: 1`. Performance targets ghi report: classifier p95 <5ms và webcam incoming-frame budget khoảng 33ms là mục tiêu, không default guarantees.
- [ ] Benchmark webcam cần detector/hardware thật; nếu thiếu thì pending, không thay synthetic forward latency vào end-to-end FPS.
- [ ] Export/quantization là optional follow-up cho selected model; benchmark và accuracy parity riêng mỗi runtime. Không suy latency từ parameter count.
- [ ] Idle/out-of-vocabulary clips: measure false activations và rejection/delay. Threshold tuning chỉ dùng development data. Closed-set softmax confidence không tự chứng minh OOV rejection.

### D3 — Final evaluation và uncertainty

- [ ] Freeze selected feature/model/T/config, checkpoint selection rule và selected seed(s) dựa validation; sau đó evaluate untouched test signers trong một final evaluation campaign.
- [ ] Report Top-1/Top-3/Macro-F1, confusion, per-signer counts/accuracy, dataset/model identities và previous test-use limitation nếu có.
- [ ] Bootstrap theo signer clusters, giữ tất cả clips của signer được sample; không bootstrap individual clips. Proposed `evaluation.bootstrap_samples: 2000`, `evaluation.bootstrap_seed: 42`. Ghi rõ estimand, confidence intervals và số signers quá ít nếu có.
- [ ] Multi-seed final result theo protocol đã freeze; không chọn seed bằng test, không mở lại tuning sau final result.

## 8. Wave E — Roadmap có điều kiện, không triển khai mặc định

| Task | Trigger | Concrete experiment | Gate / risk |
| --- | --- | --- | --- |
| E0 — More-gloss pretraining | Có additional training-signers' data sau D analysis | Supervised/self-supervised pretrain encoder, reset head, finetune 20 glosses; optional training-only teacher distillation | Exclude val/test participants ở mọi stage; no extra data -> pending |
| E1 — Cross-signer SupCon | CE baseline ổn, signer gap còn lớn, labels/coverage đủ | Normalized projection64, 8 glosses × 4 clips, >=2 training signers/gloss khi có, temperature .1, weight .05 rồi .1, CE warmup | Different-signer same-gloss positives; mask anchors thiếu positive; stable logsumexp; projection bỏ deployment |
| E2 — Small face / Conv–Transformer | Validation confusions chỉ ra location/non-manual hoặc distant-phase issue | Face anchors/mouth/brow nhỏ hoặc conv stem + 2 pre-LN blocks width96–128 FFN×2 | Separate experiments, schemas/adapters parity; không all468 landmarks/RGB ngay |
| E3 — Adversarial/margin/adaptive graph | Strong baseline vẫn failure có evidence | Small class-conditioned signer adversary; ArcFace/CosFace/learned adjacency từng ablation | Negative-transfer monitoring; không gộp nhiều objectives |

Khi E1 được mở: tunables riêng canonical `training.supcon` trong YAML/dataclass, gồm enabled/default false, projection_dim, temperature, weight, warmup_epochs, glosses_per_batch, clips_per_gloss. Batch sampler phải giữ signer IDs, hoạt động với preload modes hoặc reject incompatible mode rõ ràng. SupCon negatives chỉ training data; chưa hard-negative mining. E0 ưu tiên xem xét trước specialized losses theo Astra, nhưng không download/pretrain tự động trong Wave A–D.

## 9. Verification, artifacts và definition of done

Commands engineering từ repo root (không yêu cầu training data):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_preprocess.py tests/test_augmentation.py tests/test_pipeline.py tests/test_inference.py tests/test_config.py tests/test_dataset.py tests/test_train.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pyflakes src scripts tests app.py
```

Bổ sung focused tests của evaluator/models/feature contract khi chúng được tạo. CUDA và webcam là additional checks; unit suite không phụ thuộc chúng. Minimal test environment phải pass khi optional Kaggle/detector SDK absent. Chạy lại broad checks sau meaningful changes; không ghi tests pass nếu chưa chạy.

Artifacts cần bàn giao theo wave:

| Artifact | Nội dung |
| --- | --- |
| `reports/astra_execution_log.md` | Task status, exact commands/results, file changes, remaining blockers |
| `reports/astra_feature_contract.md` | Versioned layouts, masks, geometry/augmentation mathematics |
| Frozen split snapshot dưới resolved data/output paths | Vocabulary/participant/sequence membership hashes, provenance, exclusions |
| Run directories dưới canonical `run.output_dir` | Effective config, schema/cache/split IDs, classes, checkpoint, validation reports/history |
| `reports/astra_ablation_report.md` | Feature/architecture/T comparisons, pending cells rõ ràng |
| `reports/astra_benchmark_report.md` | Hardware/settings, classifier/preprocess/end-to-end p50/p95 và delay |

Engineering done khi A/B/C contracts và integration pass toàn bộ tests + pyflakes. Experimental done khi corrected baseline, validation ablations, CPU measurements và final campaign thực sự có artifacts. Không đánh dấu D/E completed chỉ vì code hỗ trợ hoặc synthetic tests pass. Không tuyên bố accuracy gain/30FPS nếu chưa đo.

## 10. Thứ tự giao việc khuyến nghị

1. Giao Gemini **A0–A2 + A3 synthetic smoke và baseline commands** trước. Review identity/split/test-isolation gates.
2. Sau gate A, giao B0–B3 để chốt representation; có thể tích hợp C0–C3 bằng synthetic data nhưng chưa chạy architecture comparisons.
3. Khi có corrected baseline và GPU dataset, chạy D0 rồi D1. Không sweep cả roadmap cùng lúc.
4. D2 chọn model theo accuracy/latency tradeoff; D3 final campaign. E chỉ mở task cụ thể khi kết quả validation biện minh.

Lượt lập plan này chỉ tạo tài liệu handoff. Source implementation và training chưa được thực hiện theo các task mới.
