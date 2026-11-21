# Source và validation V2

> Campaign V2 đã ngừng lưu local ngày 14/11/2026 theo yêu cầu chỉ giữ V3 `b4fd56f03a57`. Bundles/results và notebooks V2 đã chuyển vào Windows Recycle Bin; source/config/tools còn lại là tài liệu và công cụ cũ. Nội dung bên dưới là lịch sử kiểm tra trước khi dọn. Xem [V3 hiện tại](../ablation-study-v3/STATUS.md) và [ledger](../cleanup/2026-11-14.json).

Ngày kiểm tra: 2026-11-12, Asia/Saigon.

Implementation đã có:

- Canonical `features.face`, optional 36-channel compact face; schema v5.
- Shared preprocessing Parquet/raw, cache identity, face validity sidecars.
- Face flip/rotation và ratio recomputation trên NumPy/PyTorch.
- Transformer factory, CLI training/evaluation và checkpoint reconstruction.
- Prediction/probabilities theo sequence/signer và model parameter counts.
- Ma trận 60 cells, config generator, common-cohort builder, source freeze,
  runner với runtime lock/cache reuse/log/ZIP và summarizer kiểm tra provenance.
- 60 canonical configs chia folder A/B/C, queues và hướng dẫn branch.
- `run-config` đọc YAML thực tế, kiểm tra protocol/source/cohort, giữ planned config
  và hash trong attempt/ZIP; Colab Drive project roots tách theo member.
- 3 notebook V2 Kaggle/Colab pin tag `ablation-v2-notebooks-v1`, common workflow
  prepare/shared/raw preflight, queue skip/retry/restore và cross-member aggregation.
- Result folder tách theo member/source/run/attempt; per-run ZIP và member ZIP
  giữ đủ artifacts, bỏ caches/raw/probes và không lặp lại ZIPs đã chứa cùng nội dung.

Kết quả kiểm tra source hiện tại:

```text
282 passed, 1 warning
```

Đã chạy toàn bộ `tests` khi import hook chủ động chặn optional clients `kaggle`
và `mediapipe`. Tests dùng fixtures tự tạo; backend MediaPipe được mock trong
test auxiliary-video. Warning duy nhất từ test AMP khi CUDA không có trên máy
local; không có test fail.

Pyflakes:

```bash
python -m pyflakes src scripts tests app.py reports/ablation-study/tools reports/ablation-study-v2/tools
```

Exit 0, không có findings. `git diff --check` exit 0.

Tests mới kiểm tra:

- Face translation/scale invariance, missing support, bounded interpolation.
- Face-off prefix regression, hash/dimension registry và sidecar ownership.
- Parquet/raw parity; CPU/tensor flip/rotation parity và ratio invariance.
- GRU/LSTM/Transformer backward, checkpoint roundtrip và optimizer/evaluation
  thật trên synthetic data; positional encoding phân biệt thứ tự thời gian.
- Đủ 60 cells/180 seeds, thay một yếu tố cho architecture variants.
- Common cohort, class/signer provenance, source drift, failed-stage lock cleanup.
- Summarizer từ chối test leakage, duplicate attempts, missing predictions;
  không chọn winner trước khi đủ matrix. Fixtures có trường hợp LSTM/F0 thắng
  để kiểm tra lựa chọn không mặc định ưu tiên F4/face/Transformer.
- CLI xuất matrix bằng UTF-8 ngay cả khi console được đặt cp1252.
- Config queues đủ 8/30/22 tổ hợp; runtime roots tách theo member.
- `run-config` từ chối drift feature/training/env.seed trước khi chạm runtime;
  propagate Face/Transformer/mirror config và bind frozen split; archive planned
  config được summarizer kiểm tra hash.
- Notebook nbformat/Python syntax và defaults cho A/B/C được kiểm tra; shared
  validation, raw mounted-data bypass, 5 preprocess probes giữ signer/vocab,
  completed skip, relocated restore, technical retry/history, safe ZIP paths và
  aggregate selected attempts đã kiểm tra bằng fixtures tự tạo.

Trạng thái training:

- **Chưa chạy training V2 trên dataset thật, GPU/Kaggle/Colab.**
- **Chưa tạo shared data package/common cohort thật.** Bước này thực hiện sau
  khi có năm cache probes và frozen signer split theo PLAN.md.
- **Chưa smoke MediaPipe video inference với video/model assets thật.** Feature
  parity đã kiểm tra trên raw landmarks; cần smoke trước khi chứng nhận demo.
- **Chưa chạy notebook thực trên Kaggle/Colab GPU.** Notebook/source/workflow
  được kiểm tra offline; cần preflight GPU baseline trước campaign dài.
- Source v5 và bộ configs A/B/C được đóng gói chung cho các branch
  `codex/ablation-member-a`, `codex/ablation-member-b`, `codex/ablation-member-c`.
  `study.py freeze` tạo content manifest để khóa phiên bản trước training;
  snapshot v4 cũ không phải source cho V2. Branch có code không có nghĩa là đã
  có shared package dữ liệu thật hoặc kết quả training.

Không có kết quả/winner V2 được suy diễn từ synthetic tests hoặc campaign cũ.

## Cập nhật nghiệm thu 2026-11-13

- Shared thật `77a84a7b12f1` đã được A chuẩn bị trên Kaggle và đối chiếu:
  7.854 sequences, train/val/test 5.558/1.155/1.141, 21 signers, 20 glosses;
  vocabulary, frozen assignments và source manifest 142 files hợp lệ.
- ZIP A đã được kiểm tra: hoàn thành **1/8** configs (`G0__F1`), ba seeds42/43/44,
  mỗi seed có 1.155 aligned validation predictions và checkpoint. Mean Val Macro-F1
  0,7387667 ± 0,0148740. Đây chưa phải kết luận so sánh feature/model.
- Reporting v1 khắc phục DeviceDataLoader thiếu sampler metadata, đã được xác
  nhận trên F1 thật. Tiếp theo F1-face preprocess thành công nhưng bị Dataset
  whitelist cũ từ chối 198 chiều trước seed42 training. Test training cũ đi qua
  TensorDataset nên không bắt được thiếu sót validator; lần freeze trước thiếu
  preflight Dataset/device-loader cho face.
- Compatibility v2 dùng canonical FeatureLayout registry đã có và kiểm tra
  binary face mask, chạy cùng reporting v1 qua training subprocess. Source
  checkout và Shared giữ nguyên, addon có ID/hash riêng và lưu payload trong
  results ZIP. Existing valid F1 được giữ lại; retry F1-face lỗi kỹ thuật.
- Đã pass 69 regression checks, gồm all-60-cell preprocessing/Dataset/device
  augmentation/model backward/prediction export trên fixture tự tạo và real CLI
  F1-face ba seeds, một epoch trên CPU. Pyflakes, notebook/recovery syntax,
  nbformat, ZIP checksums/CRC, queues và frozen source manifest pass.
- Dùng notebooks/gói bàn giao trong `artifacts/ablation-study-v2/compat-v2/`.
  `KAGGLE_RESUME_CELL.py` tiếp tục ngay runtime đang lỗi, không xóa queue/cache.
  B/C dùng bản compat_v2 trước khi tiếp tục các configs face.
- Trong lần sửa này assistant chưa chạy campaign trên dataset thật/GPU, chưa
  đánh giá test hoặc chọn winner. A còn bảy configs; chưa có ZIP nghiệm thu B/C.

## Phạm vi mới: face bắt buộc, 2026-11-13

Nhóm quyết định luôn có face và bỏ so sánh face on/off. Kế hoạch đang áp dụng
là FACE_REQUIRED_PLAN.md: A4/B15/C11 configs, tổng 30 configs / 90 fits. B/C
chưa được bàn giao. A đang báo run cuối G0__F4-face, seed42 đã qua loader và
training; chưa có ZIP cuối để chứng nhận đủ 4/4 face runs.

Không yêu cầu train lại các face runs hợp lệ. Source, configs training và Shared
giữ nguyên; scoped notebook đọc lại state/checkpoints và chỉ nghiệm thu subset
face. Historical face-off runs không được tính hoặc rank trong bảng chính.
Dùng artifacts/ablation-study-v2/face-only-v1 để chốt A và bàn giao B/C.

Đã pass sáu checks mới trên fixtures tự tạo: scope counts/config identity,
adopt session và skip face runs đã valid, restore ZIP cũ, aggregate đủ 30 cells
và bỏ face-off winner, từ chối scope/report hỏng, chạy finalize cell không gọi
training. Pyflakes, notebook syntax/nbformat, embedded payload hashes, package
CRC/checksums, config copies và frozen source/shared đều đã xác minh. Đây là
kiểm tra kỹ thuật của chuyển phạm vi; chưa chứng nhận ZIP A cuối hoặc winner thật.

## Đã nghiệm thu phần A, 2026-11-13

ZIP A mới được cut từ Downloads và lưu theo SHA256 tại
`artifacts/ablation-study-v2/results/member-a/77a84a7b12f1/e1c6bb23f368/`.
Đã xác minh hash/CRC, canonical collect cho cả tám attempts cũ, ba seeds42/43/44,
provenance/membership/labels và required artifacts. Cả bốn face runs valid,
mỗi seed có 1.155 aligned validation predictions. A hoàn thành **4/4** theo
phạm vi face-required; tám selections lịch sử được giữ nguyên.

Đã tạo thêm ZIP face-only từ artifacts đã nghiệm thu, không training/inference:
`artifacts/ablation-study-v2/results/member-a/77a84a7b12f1/1017bac26f59/member_a_77a84a7b12f1_face_only_results.zip`.
ZIP có scope audit 30 cells, report member4/4, chỉ bốn face records trong bảng
chính và historical face-off artifacts. Cả hai version có REVIEW.json/ZIP_CHECK.json.

Val Macro-F1 mean/std của A: F1-face0.7960±0.00715, F2-face0.7847±0.00468,
F3-face0.7951±0.00517, F4-face0.7764±0.01612. Đây là kết quả A, chưa đủ
baseline F0-face của C hoặc các profiles B/C để chốt winner toàn bộ30 cells.

Đã phân tích lại từ predictions/history và tạo
`artifacts/ablation-study-v2/results/member-a/77a84a7b12f1/1017bac26f59/analysis/ANALYSIS.md`,
ANALYSIS.json cùng hai biểu đồ đã kiểm tra hình ảnh. Metrics Macro-F1/Top1/Top3
được tính lại từ predictions và khớp số đã báo trong sai số rounding4 decimals.
F2−F1 =−1.13 điểm %, giảm cả ba seeds; F3−F2 =+1.04, tăng cả ba seeds;
F4−F3 =−1.87, giảm hai seeds. F1/F3 chỉ chênh0.09 điểm %, chưa đủ kết luận
khác biệt có ý nghĩa thống kê. F1 là lựa chọn tạm thời trong bốn run A,
không phải winner toàn30 hoặc toàn bộ năm feature levels G0 khi thiếu F0-face.

Weak classes F1: awake/mouse/wake mean F1 khoảng36.32/48.84/50.79%;
confusions awake→wake61.0%, mouse→shhh40.6%, wake→awake30.6% trong lớp thật.
Validation support52–63/gloss, gần đều. Cần giữ kết luận trong20-gloss/3-val-signer
cohort; không có inference, train bổ sung hay test evaluation trong phân tích.
