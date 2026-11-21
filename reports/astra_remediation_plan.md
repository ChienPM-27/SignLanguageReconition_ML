# Astra: đối chiếu implementation và kế hoạch sửa theo phase

Ngày kiểm tra: **2026-11-05, 13:24–13:26 +07:00**. HEAD: `8e335b7a46f1e2e9d1b0c47850aa053c852b55f0`; kết luận áp dụng cho working tree chưa commit, không chỉ HEAD.

**Kết luận: Gate A và Gate B chưa đạt.** Repo đã triển khai phần lớn P0 và Wave A/B, nhưng tám lỗi P1 trong [review Gate A/B](astra_gates_a_b_review.md) vẫn tái hiện. Hai kiến trúc Astra chưa được tích hợp. Chưa có bằng chứng corrected baseline, ablation hay benchmark deployment hoàn tất.

Lượt này thực hiện audit, chạy lại kiểm tra và cập nhật tài liệu kế hoạch/trạng thái. Không sửa production code, tests, YAML hay notebook; không train dữ liệu thật. Các task sửa code dưới đây đều **pending**, không phải kết quả đã thực hiện.

## 1. Đã làm gì theo Astra, chưa làm gì

| Đề xuất Astra | Implementation hiện tại | Kết luận có bằng chứng |
| --- | --- | --- |
| Sửa pose reflection | Local/torso x được negate; pose anatomical swap; giữ absent zeros | Đã có, parity synthetic hoạt động |
| Recompute derived features sau augmentation | Có ở NumPy/Torch | Chưa đúng toàn bộ: mất transition validity; epsilon không đi qua mọi loader/helper |
| Bounded imputation, nearest/support-aware resampling | Đã có; initial extraction chặn long hand gaps | Chưa kín: body wrist thiếu pose continuity; mất toàn bộ support sau resample vẫn được forward |
| Training/inference chung preprocessing | Cùng `prepare_landmark_sequence`, checkpoint config/schema/identity | Raw-array parity đạt cho fixtures; video/webcam vẫn chỉ extract hands |
| Cache/checkpoint contract | Strict T/D, metadata ownership, schema v3 và identity; predictor đã được harden | Resume/evaluator vẫn có đường bypass; Gate A1 chưa đóng |
| Frozen split, validation-only training | Có manifest/rebind và `evaluation.evaluate_test: false` | Manifest chưa freeze provenance từng sequence; notebook/RF còn đọc test |
| Body-relative wrists, robust palm scale | Có `torso_xy`, hai geometry-support channels, `clip_median` | Đã code, chưa đủ Gate B; YAML mặc định vẫn `image`/`frame` |
| Fingertip motion, bend cosines | Có +30 và +20 dimensions, registry 12 layouts | Đã code; cả hai mặc định `false`; chưa có measured ablation |
| Corrected GRU | Default D160, T32, hidden128 × 2, mean_max, dropout .3, smoothing .1, plateau | Cấu hình baseline có sẵn; freeze/retrain có provenance và metrics vẫn pending |
| MaskedTCN, HandGraphTCN | `src/models` chỉ có `gru.py` và `__init__.py` | Chưa tích hợp factory/model/config/training/inference/tests |
| Pretraining, SupCon, face/hybrid, adversarial/margin | Không thấy implementation trong model/training hiện tại | Roadmap có điều kiện, chưa làm |
| T32/48, architecture comparison, signer bootstrap, CPU/end-to-end/OOV | Chưa có campaign/report đáp ứng protocol | Chưa đo; không được khẳng định accuracy gain hay 30 FPS |

GRU hỗ trợ thêm LSTM qua cấu hình trong cùng module; điều này không tương đương hai prototype Astra. Đã instantiate model từ YAML hiện tại và xác nhận **215,572 tham số tại D160/20 lớp**. Mốc Astra cho TCN 293,524 và graph-TCN 250,212 là mốc prototype, không phải model đã train trong repo.

Các layout body-relative là **162 → 192 → 212**, vì có hai geometry-support channels; image-wrist tương ứng **160 → 190 → 210**. Số chiều không phải lỗi so với “khoảng 210” của Astra.

## 2. Kiểm tra chạy lại

Trước khi cập nhật tài liệu, cả **82/82 files** của snapshot review trước đều khớp hash working tree. Snapshot mới chứa 83 files, bổ sung báo cáo review; không có file đổi trong lúc capture.

Artifacts mới: `.review-wave-ab/20261105T062449161992Z/`, gồm `snapshot_manifest.json`, `checks.json`, `suite_tests.xml`, `optional-absent_tests.xml`, `probe_results.json`, `layout_parity_matrix.json` và synthetic inputs. Đây là artifacts review local; kết luận chính được lưu trong tài liệu này để không phụ thuộc thư mục ignored.

| Lệnh từ repo root | Kết quả lần này |
| --- | --- |
| `.\.venv\Scripts\python.exe .review-wave-ab/capture.py` | Snapshot 83 files; exit 0 |
| `.\.venv\Scripts\python.exe .review-wave-ab/run_checks.py suite` | **237 passed, 1 failed, 1 warning; 22.21s; exit 1** |
| `.\.venv\Scripts\python.exe .review-wave-ab/run_checks.py optional-absent` | **72 passed; 5.36s; exit 0**, imports Kaggle bị chặn |
| `.\.venv\Scripts\python.exe -m pyflakes src scripts tests app.py` | **0 findings; exit 0** |
| `.\.venv\Scripts\python.exe .review-wave-ab/probes.py` | Exit 0; **cả tám lỗi P1 vẫn tái hiện** |

Probe exit 0 chỉ nghĩa là diagnostic hoàn thành. Không được ghi nó thành “all contracts pass”. Test lỗi là `test_predictor_rejects_invalid_checkpoint_classes`: fixture tạo `glosses_csv` nhưng hai predictor calls ở `tests/test_inference.py:303` và `:310` không truyền file đó. Khi không có dữ liệu local, test nhận `FileNotFoundError` trước assertion về labels.

Parity matrix vẫn có giá trị: **24 configurations / 12 dimensions**, offline parquet vs raw-array inference max error **0**; NumPy vs Torch CPU supplied-parameter transforms max error **1.79e-7**; unaugmented `none`/`ram`/`device` loading max error **0**. Matrix này không chứng minh các dropout/epsilon edge cases đúng.

Runtime: Python 3.13.15, PyTorch 2.14.1 CPU, NumPy 2.5.3, PyArrow 26.0.0, pytest 9.1.1, pyflakes 4.0.3. **Không có CUDA**. Detector được stub; không kiểm tra webcam/MediaPipe thật. Optional-absent check chứng minh nhóm 72 tests nêu trên, không phải một fresh environment đầy đủ đã được provision.

Graphify dùng vocab `checkpoint resume cache identity frozen split layout geometry fingertip bend augmentation pipeline`; BFS CLI tìm 742 nodes, output giới hạn 2,000 tokens. Graph có cả snapshot cũ nên chỉ dùng định vị; findings dựa source hiện tại và probes. Không rebuild hay semantic extraction; extraction token cost 0.

## 3. Các lỗi phải sửa

ID giữ nguyên theo review để tránh mất dấu khi giao việc.

| ID / mức | Source hiện tại | Lỗi đã xác nhận và tác động | Phase |
| --- | --- | --- | --- |
| A-01 / P1 | `src/training/train.py:399`, `:413` | `epoch='invalid-epoch'` bị reject **sau khi weights đã đổi**; rollback chưa bao trùm scalar/history parsing | 1 |
| A-02 / P1 | `src/training/train.py:337`, `:353`, `:382` | Resume chấp nhận pooling mean/last khác nhau dù cùng tensor shapes; thiếu classes/labels hoặc optimizer state vẫn được nhận | 1 |
| A-03 / P1 | `src/data/split.py:106`, `:127`, `:234` | Đổi signer trong cùng split, đổi gloss của sequence, mất sequence không exclusion report, schema999 đều được nhận | 1 |
| A-04 / P1 | `scripts/21_evaluate.py:78`, `:108`, `:134` | Evaluator nhận thiếu resolved_config, duplicate/conflicting/noncontiguous labels và override pooling của selected checkpoint | 1 |
| B-01 / P1 | `pipeline.py:313`, `preprocess.py:210`, `augmentation.py:71`, `augmentation_gpu.py:105` | Long hand gap N100/T4: clean tip delta 0 nhưng sau flip **.165**, wrist delta **.066**; transition masks không tồn tại trong cache/augmentation | 2 |
| B-02 / P1 | `src/data/device_loader.py:86`, `augmentation.py:369`, `:422` | Epsilon .01 bị thay bằng default .0001 trên device loader/CPU helpers; cùng invalid short bone cho cosine **0 hoặc 1** tùy preload | 2 |
| B-03 / P1 | `src/features/normalization.py:629` | Body wrist delta qua long pose gap khoảng **.165** dù torso delta đúng bằng 0; thiếu pose transition trong geometry motion | 2 |
| B-04 / P1 | `src/features/pipeline.py:367`, `predictor.py:407` | Raw đủ 3 valid frames nhưng target support =0 vẫn `is_usable=true`; **model forward 1 lần trên toàn zeros** | 2 |
| T-01 / gate blocker | `tests/test_inference.py:303`, `:310` | Test phụ thuộc file ngoài fixture; full self-contained suite fail | 0 |
| E-01 / protocol gap | `src/training/evaluate.py:67` | JSON thiếu ordered confusion matrix, sample membership, preprocessing/split identities | 1 |
| I-01 / adapter gap | `src/features/landmark_extractor.py:18`, `scripts/31_realtime_demo.py:166`, `:180` | Hand-only detector; buffer dùng T làm capture length, mất đầu SPACEBAR clip dài | 2, 5 |
| N-01 / handoff gap | `notebooks/islr_phase2.ipynb:205`, `:241`, `:284` | `--stage all` không frozen override; RF/test và newest-checkpoint test; T4 preset đổi controlled baseline | 1, 3 |

Không gộp những thiếu sót này thành “chỉ cần thêm tests”. Một số code hiện có thực sự cho kết quả sai dù CPU/Torch giống nhau.

## 4. Kế hoạch thực hiện theo phase

### Phase 0 — Sửa trạng thái và tính tự chứa của kiểm tra

Phạm vi nhỏ; làm trước để các gate sau có số liệu tin cậy.

- [x] Đối chiếu plan/report với code và rerun snapshot suite/probes; lưu bằng chứng lần này.
- [x] Thêm current-status corrections trong các báo cáo cũ; giữ historical results.
- [ ] T-01: truyền `glosses_csv=glosses_csv` vào cả hai predictor calls. Không tạo gloss file toàn repo để che lỗi.
- [ ] Chạy lại suite từ snapshot sạch, pyflakes và optional-SDK isolation. Sau mỗi phase chỉ ghi số tests/exit codes thực chạy.

**Gate 0:** suite tự chứa pass; logs tách `implemented`, `synthetic verified`, `measured`, `pending`. Việc đạt Gate 0 không tự đóng tám P1.

### Phase 1 — Khóa artifact, resume và evaluation protocol (đóng Wave A)

Targets: `src/training/train.py`, `src/inference/predictor.py`, `scripts/21_evaluate.py`, `src/data/split.py`, `src/training/evaluate.py`, `scripts/20_train.py`, notebook và related tests. Tạo một shared checkpoint validator tại `src/utils/checkpoint.py` nếu phù hợp; không sao chép ba validator độc lập.

- [ ] A-01/A-02: parse/validate metadata, model semantics, epoch/counters/history và toàn bộ state trước application. Require ordered classes + bijective contiguous label map + optimizer state; scheduler/scaler state bắt buộc khi checkpoint protocol dùng chúng. Reject nonfinite/invalid state có ý nghĩa; chỉ cho phép sentinel được contract ghi rõ.
- [ ] So sánh `model.name`, pooling, width/layers/bidirectionality/dropout, input layout và class order với model/config đang resume. So tensor shapes là kiểm tra bổ sung. Định nghĩa allowlist runtime/path/epoch-extension overrides; không ngầm thay model semantics.
- [ ] Apply state atomically cho model/optimizer/scheduler/scaler/counters; lỗi ở bất kỳ bước nào giữ nguyên tất cả trạng thái trước resume. Thử cả malformed counters lẫn late state-loading error; valid resume vẫn roundtrip.
- [ ] A-04: evaluator/predictor/resume dùng chung validated checkpoint contract. Final evaluator không fallback external YAML thay config bị thiếu và không nhận model/preprocessing override. Cho phép relocation/device/report paths qua allowlist đã validate; schema/config key lạ vẫn `KeyError`.
- [ ] A-03: frozen manifest version mới lưu ordered vocabulary/map và record từng `sequence_id → participant_id, gloss, label, split, source content hash`. Membership hash độc lập absolute paths/model seed. Feature content hashes thuộc từng cache identity, không ép chúng giữ nguyên giữa ablations.
- [ ] Reject unsupported/missing manifest schema, duplicate IDs, changed per-sequence signer/label, overlap và conflicting membership. Rebind thiếu sequence phải xuất exclusion record với ID/gloss/signer/reason; kiểm tra coverage sau exclusions, không tìm seed/signer mới để bù.
- [ ] E-01: report JSON lưu ordered classes, numerical confusion counts, predictions/targets gắn exact sample IDs/order, per-signer counts, cache/schema/split/config/checkpoint identities. Không ghép predictions với CSV chưa lọc hay shuffled order.
- [ ] N-01: notebook/default workflow train+val only; RF baseline cũng không tự đọc test. Thay newest-checkpoint discovery bằng checkpoint/run được chọn rõ ràng; final test là thao tác riêng sau freeze. Bỏ `--stage all` khỏi controlled rerun; dùng preprocess + split rebind.
- [ ] Audit vocabulary selection và prior test use: code có landmark separability/handshape scoring trước split; xác định dữ liệu thực tế từng dùng. Nếu thiếu provenance, ghi limitation; không gọi test “untouched” và không silently chọn lại vocabulary/participants.

**Regression bắt buộc:** tất cả A-01–A-04 phải chuyển từ accepted/mutated sang rejected/unchanged; test read spy cho GRU và RF/notebook workflow; evaluator metadata roundtrip; strict nested config override checks. Các synthetic fixtures không cần Kaggle/GPU.

**Gate A:** Gate 0 + A-01–A-04/E-01/N-01 engineering checks + synthetic train/save/resume/evaluate pass. Real baseline vẫn pending tới Phase 3.

### Phase 2 — Validity xuyên suốt pipeline và adapters (đóng Wave B)

Targets: `src/features/{pipeline,normalization,schema}.py`, `src/data/{preprocess,dataset,device_loader,augmentation,augmentation_gpu}.py`, raw extractors/predictor/realtime/apps và config/tests.

- [ ] B-01: giữ observation support, imputed usable support, transition support và slot-to-anatomical mapping từ raw preparation tới cache, loaders, augmentation. Lưu per-sequence sidecar có version/hash, hoặc cấu trúc tương đương có contract rõ; không suy continuity từ adjacent target masks hoặc delta khác 0.
- [ ] Giữ D của model không đổi nếu reliability chỉ là metadata; model padding mask vẫn riêng với observation mask. Validate shape/dtype/binary masks, temporal length, hand-slot mapping và sidecar ownership ở cả `none`/`ram`/`device`. Missing sidecar trong schema mới phải fail.
- [ ] Mọi geometric transform nhận transition provenance, swap nó khi swap hand slots, rồi recompute derivatives với support đó. Identity/no-op transform cũng không được khôi phục motion qua long gaps. Không yêu cầu raw input được đọc lại trong từng minibatch.
- [ ] B-03: body wrist transition = continuous usable hand **AND** continuous usable pose, cùng endpoint geometry support. Wrist image/tips chỉ dùng hand continuity; torso velocity dùng pose continuity. Short imputed gaps theo policy đã freeze, long gaps không được nối.
- [ ] B-02: truyền đúng `preprocess.normalization_epsilon` qua dataset → device loader → public helper → recomputation. Thêm epsilon argument vào CPU `shape_scale`/`scale_translate`; không thêm alias augmentation epsilon.
- [ ] B-04: sau resampling, reject clip không còn required usable hand support; torso mode cần có joint hand/pose geometry support. Giữ missing-hand timesteps có pose trong clip hợp lệ, không biến chúng thành padding. Report raw/target coverage và rejection reason; predictor không forward clip rejected.
- [ ] Semantics/acceptance và cache payload đổi thì bump preprocessing schema **3 → 4**, bind vào identity và regenerate caches; không relabel v2/v3 artifacts. Freeze contract trước regenerate để không rebuild nhiều lần. Thêm numerical equality check cho unaffected baseline fixtures; giải thích khác biệt ở repaired edge cases.
- [ ] I-01: hands + actual pose cùng frame/time cho adapter cần pose; chốt mirrored-input/handedness convention và dominant permutation đúng một lần. Missing pose phải tạo unsupported/waiting state rõ ràng; không claim adapter parity từ hand-only detector.
- [ ] Tách capture khỏi T bằng canonical `realtime.max_capture_frames` trong YAML/dataclass; max là giới hạn bộ nhớ, không target resampling. SPACEBAR giữ toàn clip tới release; quá giới hạn thì báo overflow/reject rõ ràng, không silent deque truncation. Kiểm tra T32/48 không đổi raw capture duration.
- [ ] Cập nhật feature contract: phân biệt observed/usable/transition/padding; gọi motion là displacement per phase step; hand z chỉ intra-hand cue. Làm rõ shoulder scale của torso_xy có dùng pose-z hay XY-only và giữ decision trong schema; không mặc định image-normalized XY có metric aspect ratio khi parquet không lưu kích thước ảnh.

**Regression bắt buộc:** N100/T4 với hand-only dropout, pose-only dropout, cả hai; zero target support; epsilon .01 và bones .005; mọi deterministic transform/slot swap trên 12 layouts và ba preload modes; malformed/missing sidecars; nondefault trim/gap/T; synthetic offline/raw/video-adapter boundary parity. Không chỉ so NumPy với Torch: assert expected equations độc lập.

**Gate B:** các B-01–B-04 chuyển về expected zero/reject; fixture adapter/capture tests pass; full suite + pyflakes sạch. CUDA và actual detector/webcam measurements ghi pending riêng nếu chưa có hardware; không gọi Torch chạy CPU là “GPU verified”.

### Phase 3 — Corrected GRU và feature ablations có kiểm soát

Depends on Gate A/B core semantics. Adapter hardware benchmark có thể tiếp tục độc lập; không đổi feature contract giữa campaign.

- [ ] Freeze config snapshot baseline từ canonical YAML: D160/T32, GRU128×2, unidirectional, mean_max, dropout .3, smoothing .1; batch32, epochs60, min_epochs5, optimizer/scheduler/augmentation giữ snapshot; seeds42/43/44, initialization mới.
- [ ] T4 balanced batch128/epochs80/min_epochs15 là experiment riêng. Không dùng nó rồi gọi comparison với baseline “chỉ khác features”. Không coi 86,888 raw clips là số mẫu 20-gloss train: xuất actual class/signer/split counts sau exclusions.
- [ ] Dùng `scripts/10_prepare_data.py --stage preprocess` với config/cache riêng, sau đó `--stage split --set split.manifest_path=...` với snapshot hợp lệ; không rerun selection/download khi inputs đủ.
- [ ] Chạy corrected F0 ba seeds; ghi best epoch/loss, validation Top-1/Top-3/Macro-F1, per-gloss/signer, train–val gap, runtime và tất cả identities. Giữ original run như lịch sử riêng, không giả tạo measured comparison nếu artifact cũ thiếu.
- [ ] Feature sequence: **F0 D160 → F0-control tắt wrist scale/translation → F1 torso D162 → F2 clip_median → F3 tips D192 → F4 bends D212**. So F1 với F0-control; không gộp geometry với augmentation restriction.
- [ ] Giữ split/seeds/training budget; báo full-cohort và matched-cohort khi exclusions khác. Chọn subset theo validation, không mặc định full F4 thắng. Smoothing 0/.05/.1 là experiment riêng sau đó.

**Gate baseline/feature measurements:** mỗi completed row có config, manifest/hash, checkpoint, history và validation report. Không có data/GPU thì ghi pending; không thay synthetic accuracy vào bảng thật.

### Phase 4 — Hai kiến trúc Astra rồi temporal length

Synthetic implementation có thể bắt đầu sau Gate A/B trong khi Phase 3 đang chạy; **architecture accuracy comparison phải chờ corrected baseline và feature protocol**.

- [ ] C0: shared model factory và masked pooling; GRU/LSTM/TCN/HAND_GRAPH_TCN dùng canonical config. `forward(x, valid)` dùng real-timestep mask, fixed-resampled clip mặc định all-true; reject empty clip. Sanitize padding trước projection; GRU right-padding dùng lengths/packing.
- [ ] C1: width128, depthwise kernel5, dilations1/2/4/8, LayerNorm + FFN×2 + SiLU, dropout .15, mean/max. Receptive field61, không downsampling; target prototype count293,524 tại D160/20 lớp.
- [ ] C2: shared fixed 21-joint graph, hidden48, 3 layers, ordered-joint readout64 mỗi hand, missing-hand masking; fuse tất cả non-shape channels, temporal width96. Target250,212 tại D160; không thêm cross-part 3D edges dựa hand wrist z.
- [ ] Expose tunables trong `model.tcn`/`model.hand_graph`, YAML + dataclasses; shared temporal parameters không duplicate. Training/checkpoint/predictor/evaluator đều dùng factory và cùng validation contract.
- [ ] Test forward/backward/finite gradients, noisy/nonfinite right-padding invariance, missing-hand≠padding, nondefault D/T, every-extension-channel fusion, checkpoint logits roundtrip và resume cho cả ba architecture families.
- [ ] So architectures cùng feature layout/T32/split/seeds/budget. Sau đó T32 vs48, regenerate cache theo T; không pad cache32 thành48. Architecture-specific tuning có equal declared budget và report riêng.

**Gate C engineering:** synthetic suite + full suite + pyflakes pass. **Measured winner:** chỉ sau validation comparison và CPU latency; parameter count không chứng minh tốc độ.

### Phase 5 — Deployment benchmark và final evaluation

- [ ] Batch1 CPU p50/p95; ghi hardware, versions, thread count, warmup/repetitions. Separate classifier/preprocess/extractor/buffer/render; đo sign-completion delay. Tunables benchmark chỉ thêm qua canonical YAML/dataclass.
- [ ] Classifier p95 <5ms và incoming-frame budget khoảng33ms là mục tiêu cần đo. Export/quantization phải có numerical/accuracy parity và latency measurements riêng.
- [ ] Thử webcam/recorded clips với pose/mirror/capture path thật; idle/OOV false activations, delay và rejection coverage. Tune thresholds bằng development data, không bằng final test.
- [ ] Freeze selected model/features/T/checkpoint/seed policy bằng validation. Một final campaign; bootstrap theo signer clusters, report Top-1/Top-3/Macro-F1/confusion/per-signer và uncertainty. Không chọn best seed bằng test.
- [ ] Nếu historical test đã dùng hoặc selection provenance không đủ, công bố giới hạn đó; chỉ lập benchmark mới bằng quyết định riêng, không gọi cùng test “untouched”.

**Gate experimental done:** reports và artifacts thật cho baseline/ablations/latency/final campaign. “Code supports evaluation” chưa đủ.

### Phase 6 — Hướng nghiên cứu có điều kiện (Wave E)

Ưu tiên more-gloss pretraining/distillation chỉ từ training signers; tiếp theo cross-signer SupCon nếu signer gap còn lớn. SupCon starting experiment: projection64, 8 glosses×4 clips, ≥2 signers/gloss, temperature .1, weight .05–.1 sau CE warmup; exclude anchors thiếu cross-signer positive, bỏ head khi deploy. Face anchors/Conv–Transformer chỉ khi validation confusions biện minh; adversarial/margin/adaptive graph sau cùng. Mỗi hướng là một ablation, không gộp và không tự coi toàn Wave E là bắt buộc.

## 5. Quy tắc bàn giao

Một PR/commit nhỏ cho mỗi nhóm contract: test isolation → checkpoint validator/atomic resume → frozen provenance/evaluator/notebook → transition schema/cache → augmentation epsilon/continuity/quality → adapters → model factory/TCN → graph model → measured campaigns. Không commit hay reset working-tree changes của người dùng trong lượt lập kế hoạch này.

Mọi tunable dùng một canonical key trong `configs/islr.yaml` và `src/utils/config.py`; unknown YAML/dict/`--set` keys phải `KeyError`. Artifact metadata/snapshots là provenance, không là config alias. Giữ credential precedence/import isolation và absolute-path semantics của `AGENTS.md`. Fixtures tự sinh trong `tmp_path`, không dùng cache/model/video local.

Acceptance cuối mỗi phase: focused mathematical regressions, rồi full self-contained pytest + `pyflakes src scripts tests app.py`; ghi exact command/exit count và các measurement còn thiếu. [Plan gốc](astra_next_implementation_plan.md) giữ chi tiết thiết kế C/D/E; tài liệu này thay thế các completion claims và thứ tự triển khai đã lỗi thời. Ưu tiên tiếp theo là **Phase 0–2**, chưa phải train TCN ngay.
