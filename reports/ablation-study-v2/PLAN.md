# Kế hoạch ablation V2: feature × architecture × model

Ngày lập: 2026-11-12, Asia/Saigon. Source yêu cầu preprocessing schema v5.
Trạng thái: implementation và kế hoạch mới; chưa có kết quả training V2.

Mục tiêu là chọn tổ hợp tốt nhất **trong không gian thí nghiệm đã định trước**,
đồng thời kiểm tra feature có cải thiện khác nhau giữa GRU, LSTM và Transformer.
Không giả định F4 thắng F0 hoặc face luôn hữu ích.

## 1. Phạm vi và baseline mới

V2 train lại từ initialization mới. Artifacts của campaign v4 vẫn là lịch sử riêng.
Không nối checkpoints cũ, không trộn bảng kết quả cũ vào leaderboard V2.

Trong V2, **F0 dùng image/frame và tắt wrist scale/translation**: tương ứng lựa
chọn feature của F0-control cũ, nhưng chạy lại bằng source v5. Nhờ đó tất cả
F0–F4 có cùng augmentation policy; không trộn thay đổi augmentation vào phép
so feature. Các augmentation khác theo config shared đã freeze.

## 2. Mười bộ feature

| Level | Thay đổi cộng dồn | Face off: D | Face on: D |
|---|---|---:|---:|
| F0 | Hands + upper-body pose; wrist=image, palm=frame | 160 | 196 |
| F1 | F0, wrist=torso_xy; thêm 2 geometry-support masks | 162 | 198 |
| F2 | F1, palm=clip_median | 162 | 198 |
| F3 | F2 + 30 fingertip displacement channels | 192 | 228 |
| F4 | F3 + 20 finger-bend cosine channels | 212 | 248 |

ID có face dùng hậu tố `-face`: F0-face, F1-face, …, F4-face.
Canonical config chỉ có `features.face: true/false`; không có khóa tỷ trọng
tay/thân/mặt. Cùng một level luôn có cặp face off/on để đo tác dụng của mặt.

Face compact có 36 kênh:

- 16 điểm XY quanh mũi, mắt, môi, lông mày và cằm: 32 kênh.
- Tỷ lệ độ mở miệng và hai mắt: 3 kênh.
- Usable-face mask: 1 kênh.

Landmark indices cố định trong schema:
`[1,33,263,133,362,159,386,145,374,61,291,13,14,70,300,152]`.
Tâm là midpoint hai khóe mắt ngoài; scale là median khoảng cách hai khóe mắt
trên những frame thực sự quan sát được trong clip đã trim, trước imputation.
Feature hình dạng giữ hướng tương đối; RNN/Transformer học thay đổi theo thời gian.
Đây là biểu diễn hình dạng/biểu cảm local; vị trí tay so với vùng mặt là một
feature extension khác, chưa nằm trong 36 kênh này.

Face thiếu/NaN/degenerate được mask và giữ zero; gap ngắn được nội suy có giới
hạn như tay. Observed-face và usable-face lưu riêng trong validity sidecar.
Face không tạo thêm điều kiện loại clip. X/Y dùng cùng semantics ở Parquet và
raw/video inference. Flip đổi bên mắt/brow đúng topology; rotation áp cùng góc;
ratios được tính lại. Jitter/shape-scale hiện là augmentation riêng của tay.

## 3. Sáu cấu hình model/architecture

| Profile | Model | Layers / hidden | Pooling | Bidirectional | Người chạy |
|---|---|---|---|---|---|
| G0 | GRU | 2 / 128 | mean_max | false | A/C, phân công dưới đây |
| G-mean | GRU | 2 / 128 | mean | false | B |
| G-attn | GRU | 2 / 128 | attention | false | B |
| G-bi | GRU | 2 / 128 mỗi hướng | mean_max | true | B |
| L0 | LSTM | 2 / 128 | mean_max | false | C |
| T0 | Transformer encoder | 2 / 128 | mean_max | full-clip attention | C |

T0 có 4 attention heads, feedforward width256, sinusoidal positional encoding,
input projection + LayerNorm; dropout0.3. Cùng T32 và output class order.
Transformer dùng full-clip self-attention, phù hợp isolated sign recognition.

Các variant của B **đều so trực tiếp với G0**, không cộng dồn architecture.
G-bi có số parameters lớn hơn G0; G-mean/G-attn có pooling/head khác.
Đây là so sánh cấu hình cụ thể, không phải phép chứng minh capacity ngang nhau.
G0/L0/T0 cũng có parameter counts khác nhau dù cùng hidden128.

## 4. Ma trận đầy đủ và phân công A/B/C

Mỗi profile chạy trên **tất cả 10 bộ feature**. Có 6×10=60 cells; mỗi cell có
seeds42/43/44, tổng cộng **180 model fits**. MATRIX.json chứa ID/owner/D/seeds
cụ thể cho từng cell. Một fit là train một seed; một cell là ba fits.

| Member | Các cells phải chạy | Cells | Fits | Báo cáo phụ trách |
|---|---|---:|---:|---|
| A | G0 × F1/F2/F3/F4, mỗi level face off/on | 8 | 24 | Feature ablation, chuẩn bị cohort/cache, tổng hợp delta |
| B | G-mean/G-attn/G-bi × cả 10 feature sets | 30 | 90 | Architecture ablation và chi phí tính toán |
| C | G0 × F0/F0-face; L0/T0 × cả 10 feature sets | 22 | 66 | GRU baseline, LSTM/Transformer và model comparison |

C trực tiếp chạy cả GRU, LSTM và Transformer như yêu cầu. Hai baseline GRU
của C được A/B dùng chung; không train trùng để điền bảng. Tải không chia đều
theo số fits: A phụ trách chuẩn bị/analysis; Transformer của C có thể tốn thời
gian khác GRU. Nếu chuyển queue giữa người, giữ nguyên run ID và ghi actual
executor trong handoff; không tạo một attempt để chọn seed có kết quả tốt hơn.

Các cells độc lập, không cần checkpoint của cell trước. C nên chạy G0/F0 và
G0/F0-face đầu tiên để kiểm tra shared package. Sau preflight, A/B/C chạy trên
ba runtime riêng; mỗi runtime chạy các cells tuần tự. Runner khóa runtime để
tránh cùng ghi vào data/splits. Khi runtime bị ngắt đột ngột, kiểm tra process
còn sống trước khi xóa stale `.runtime.lock`.

Có thể chạy seed42 trước để kiểm tra kỹ thuật và ước lượng thời gian, nhưng
**không loại cells theo điểm seed42** nếu vẫn muốn kết luận trên đầy đủ 60 cells.
Runner chuẩn chạy cả ba seeds trong một attempt để giữ provenance đơn giản.

## 5. Shared data và cohort cố định trước training

1. Dùng cùng ordered 20-gloss vocabulary, label_map và raw landmark catalog.
   Giữ participant assignment của split hiện có; không chọn lại split seed dựa
   trên điểm validation. Nếu tạo split lần đầu, tạo đúng một lần rồi freeze.
2. Preprocess thử F0/F1/F2/F3/F4 face off bằng source v5, mỗi level một cache
   riêng. Thực hiện trên toàn bộ selected sequences, chưa train.
3. Lấy intersection của các sequence được giữ ở năm caches. Face on dùng cùng
   quy tắc nhận clip nên không giảm cohort; vẫn kiểm tra membership khi chạy.
4. Lọc sequence IDs của frozen split theo intersection, **giữ nguyên signer
   assignments và class order**. Công cụ `cohort` từ chối mất signer hoặc thiếu
   class trong một split. Nếu lỗi, bổ sung dữ liệu; không im lặng đổi split.
5. Freeze shared config/source manifest/common split/vocabulary. Sau bước này
   mọi run bắt buộc có đúng train/val/test sequence IDs trong shared split.

Source manifest hash normalize CRLF/LF để verify cùng source trên Windows/Linux,
bao gồm src/scripts/configs/tests và tools V2. Freeze bằng nội dung file nên
có thể kiểm chứng cả source chưa commit. Git SHA trong training reports chỉ
là thông tin checkout; source_manifest.json là snapshot content của campaign.
Nên commit source đã kiểm tra để handoff tiện hơn; copy nguyên source/tools và
shared package sang mỗi runtime, tránh sửa code trong lúc campaign đang chạy.

Cache v4/checkpoint v4 không được dùng bởi source v5. Tạo cache/output V2 riêng,
giữ artifacts cũ để đọc bằng source cũ nếu cần. Config/hash kiểm tra cả face toggle.

## 6. Protocol training và lựa chọn checkpoint

Base theo configs/islr.yaml; snapshot shared khóa mọi hyperparameter:
T32, batch32, lr0.001, weight_decay0.0001, smoothing0.1, dropout0.3,
epochs60, early-stopping minimum5/patience15, plateau scheduler theo config.
Ba seeds42/43/44; không resume checkpoint cũ. Mọi run `evaluate_test=false`.

Checkpoint mỗi seed được chọn bằng **validation loss**, theo training pipeline
hiện tại. Sau đó tính metrics trên validation ở checkpoint đã chọn. Cell được
so sánh bằng **Validation Macro-F1 mean của cả ba seeds**, không lấy best seed.
Training loss/accuracy không quyết định leaderboard.

Protocol cố định giúp so sánh 60 cấu hình cụ thể một cách nhất quán. Nó không
cho phép kết luận một họ thuật toán luôn tốt hơn họ khác sau mọi cách tuning.
Nếu làm tuning bổ sung, đăng ký vòng riêng với cùng số trials/thời gian mỗi
model family, chọn trên validation rồi chạy lại cả ba seeds; báo riêng khỏi V2.

## 7. Kiểm chứng chéo và tương tác

Các phép so sánh cần có trong báo cáo:

- Với từng profile, so F1−F0, F2−F1, F3−F2, F4−F3, riêng face off và face on.
- Với từng profile/level, so Fx-face−Fx: tác dụng của face ở cùng feature nền.
- Với từng feature set, so L0−G0 và T0−G0: model comparison với input giống nhau.
- Với từng feature set, so G-mean/G-attn/G-bi−G0: architecture comparison.
- Tương tác model×face: `[L0(Fx-face)−L0(Fx)]−[G0(Fx-face)−G0(Fx)]`;
  làm tương tự cho T0. Tương tác model×feature dùng các chênh lệch F4−F0.

Tính delta theo seed tương ứng rồi tổng hợp mean/std. Đọc confusion matrix,
per-gloss và per-signer để hiểu cải thiện/giảm sút. Validation report lưu
prediction/probabilities theo sequence_id, participant_id để đối chiếu cùng mẫu.
Không coi 3 seeds là đủ để mặc định kết luận statistical significance.

Nếu cần confidence interval, dùng paired cluster bootstrap theo signer trên
cùng predictions; giữ cùng nhóm resample cho hai cấu hình. Nếu có ít validation
signers, báo rõ CI kém ổn định và cần thêm signer-held-out folds để mở rộng kết luận.
Phân biệt uncertainty do samples/signers với variance do training seeds.
Nếu báo p-value cho nhiều phép so sánh, hiệu chỉnh multiple comparisons.

## 8. Chọn tổ hợp tốt nhất và test cuối

Chỉ chốt winner khi đủ 60 cells, mỗi cell đủ 3 seeds và đúng source/config/
vocabulary/cohort. Missing/failed cells ghi PENDING; không gọi kết quả tạm là
winner của toàn ma trận. Tool từ chối mixed provenance và duplicate attempts.

Quy tắc lựa chọn được đăng ký trước:

1. Mean Validation Macro-F1 cao nhất.
2. Nếu mean bằng nhau theo precision lưu report, ưu tiên std thấp hơn.
3. Nếu vẫn bằng, ưu tiên ít parameters hơn; cuối cùng run ID để deterministic.

Đây là selection policy; nếu delta nhỏ/CI chứa zero, viết là khác biệt chưa
đủ chắc chắn, dù policy vẫn chọn một tổ hợp để triển khai. Không tuyên bố
F4/face/Transformer bắt buộc thắng. Báo accuracy/Top3/loss, parameters,
training time, inference latency cùng hardware bên cạnh Macro-F1.

Sau lựa chọn, khóa winner và chọn seed42 làm checkpoint triển khai mặc định
(quy tắc trước khi xem test, không chọn seed tốt nhất). Evaluate test đúng một
lần cho mỗi checkpoint của winner ở ba seeds để báo mean/std; test không được
dùng để đổi winner, chọn hyperparameter hoặc chọn seed. Khi test không xác nhận
cải thiện, báo điều đó và giới hạn generalization; không quay lại tối ưu trên test.

Nếu muốn so final winner với baseline trên test, đăng ký baseline G0/F0 trước
khi mở test và evaluate cùng cohort; kết quả chỉ để báo cáo, không chọn lại model.

## 9. Commands và artifacts

Tools chạy từ repo root; dùng Python của runtime đã cài dependencies.

Ba notebook `notebooks/ablation_v2_member_a.ipynb`, `ablation_v2_member_b.ipynb`,
`ablation_v2_member_c.ipynb` là entrypoints cho Kaggle/Colab. Source pin bằng tag
`ablation-v2-notebooks-v1`; dùng shared package freeze từ snapshot notebook mới.
Hướng dẫn setup, prepare, queue/restore và folder kết quả:
`notebooks/ABLATION_V2_README.md`. Notebook A chỉ prepare khi MODE=prepare;
A/B/C cùng MODE=train sau khi đã nhận shared ZIP. Test metrics không mở ở notebook.

Ba branch `codex/ablation-member-a`, `codex/ablation-member-b`,
`codex/ablation-member-c` cùng chứa source v5 và đủ 60 configs chia theo owner
trong `configs/ablation-study-v2/member-a`, `member-b`, `member-c`.
Xem `configs/ablation-study-v2/README.md` để chạy bằng YAML và tách runtime.
Các folder config giống nhau trên ba branch để source manifest không lệch.
Chỉ khác branch name không tạo thêm GPU/runtime; vẫn cần máy/notebook riêng.

Mỗi YAML có run ID canonical và seeds42/43/44. `run-config` đọc YAML thực tế,
kiểm tra protocol so với shared, bind cohort rồi lưu planned/effective config.
Config runtime khác (Drive/mirror/local root) dùng bản copy ở ngoài source đã
freeze; không sửa YAML trong repo sau khi snapshot. Source/Face/Transformer đã
có không thay thế bước chuẩn bị common cohort và shared package dataset thật.

```bash
# Xem phân công cụ thể (không train)
python reports/ablation-study-v2/tools/study.py matrix --owner A
python reports/ablation-study-v2/tools/study.py matrix --owner B
python reports/ablation-study-v2/tools/study.py matrix --owner C

# Xuất canonical config để preprocess thử mỗi level; dùng --base nếu có config riêng
python reports/ablation-study-v2/tools/study.py config G0 F0 /path/to/probe/F0.yaml
python scripts/10_prepare_data.py --config /path/to/probe/F0.yaml --stage preprocess --set data.cache_dir=/path/to/probe/F0/cache
# Lặp F1/F2/F3/F4, mỗi level một config/cache; không dùng lại cache khác hash.

# Tạo common cohort; đúng thứ tự F0/F1/F2/F3/F4, mỗi cache có preprocess_meta.json
python reports/ablation-study-v2/tools/study.py cohort /path/to/base_split_manifest.json /path/to/common_split.json /path/to/F0/cache/features/manifest.csv /path/to/F1/cache/features/manifest.csv /path/to/F2/cache/features/manifest.csv /path/to/F3/cache/features/manifest.csv /path/to/F4/cache/features/manifest.csv

# Tạo shared directory mới sau khi source được kiểm tra
python reports/ablation-study-v2/tools/study.py freeze /path/to/shared_v2 --split /path/to/common_split.json --selection /path/to/selection

# Member C bắt đầu với GRU baseline; mỗi lệnh chạy đủ 3 seeds và lưu ZIP
python reports/ablation-study-v2/tools/study.py run-config configs/ablation-study-v2/member-c/G0__F0.yaml /path/to/shared_v2
python reports/ablation-study-v2/tools/study.py run-config configs/ablation-study-v2/member-c/T0__F4-face.yaml /path/to/shared_v2
# Cách chọn profile/feature trực tiếp vẫn được hỗ trợ
python reports/ablation-study-v2/tools/study.py run G0 F0 /path/to/shared_v2
python reports/ablation-study-v2/tools/study.py run G0 F0-face /path/to/shared_v2
python reports/ablation-study-v2/tools/study.py run L0 F0 /path/to/shared_v2
python reports/ablation-study-v2/tools/study.py run T0 F0-face /path/to/shared_v2

# Ví dụ A/B; tiếp tục hết cells trong MATRIX.json
python reports/ablation-study-v2/tools/study.py run G0 F4-face /path/to/shared_v2
python reports/ablation-study-v2/tools/study.py run G-attn F4-face /path/to/shared_v2

# Sau khi tập hợp và giải nén attempts A/B/C
python reports/ablation-study-v2/tools/summarize.py /path/to/selected_attempts --output reports/ablation-study-v2/RESULTS.md

# Sau khi khóa winner: rebind split CSV đúng feature của winner rồi evaluate test
python scripts/10_prepare_data.py --config /path/to/winner/effective_config.yaml --stage split
python scripts/21_evaluate.py --checkpoint /path/to/winner/runs/run_id/seed_42/best_model.pt --split test
# Lặp seed43/44 của cùng winner, không đổi tổ hợp theo điểm test.
```

`/path/to/...` là đường dẫn thực trên runtime; trên Windows đặt path có spaces
trong dấu nháy. Preprocess probes có thể ghi reports chung nếu chạy tuần tự;
đặt output.report_dir/output.output_dir riêng nếu muốn giữ cả năm probe reports.

Mỗi run có attempt riêng dưới persistent_root/ablation_v2_runtime:
inputs, effective_config, metadata, logs, cache_audit, split_snapshot, runs và ZIP.
Cache thật ở persistent_root/ablation_v2_cache, định danh theo source và feature
identity, dùng lại giữa các model trên cùng runtime. ZIP chứa audit/model/log,
không kèm toàn bộ raw/cache data; cache lớn không cần gửi lại cho mỗi cell.

Notebook giữ bản nghiệm thu riêng tại
`persistent_root/ablation_v2_results/member-<a|b|c>/<code-id>/runs/<run-id>/<attempt>`.
`queue_status.json` lưu selected paths tương đối để restore sang runtime khác;
metadata/configs/logs/checkpoints/predictions được giữ nguyên. Per-run ZIPs ở
`bundles`; member result ZIP loại cache/probe data và các per-run ZIP trùng nội dung.
Tổ hợp hoàn tất chỉ được skip sau artifact validation; failed attempts giữ riêng.
Notebook A tổng hợp ZIP của B/C bằng selected paths, không truyền toàn bộ folder
có cả retry histories vào summarizer.
Standalone evaluation dùng config trong checkpoint. Nếu chuyển sang runtime
khác, regenerate cache của đúng feature và rebind common split trước; cho
phép override đường dẫn runtime, không override model/feature semantics.
Không chọn lại attempt tốt nhất sau khi xem metrics; retry chỉ cho lỗi kỹ thuật,
giữ attempt thất bại và ghi lý do. Summarizer nhận đúng một selected attempt/cell.

Kaggle cần Save Version/Output hoặc tải ZIP về sau mỗi run; Colab cần vị trí
persistent phù hợp. Không dựa vào scratch cache để giữ artifacts cuối.

## 10. Hoàn thành

- Source v5 qua Unit/Integration Tests, pyflakes; face-off regression và
  face-on geometry/cache/augmentation/checkpoint parity được kiểm tra.
- Shared package được A/B/C verify; đủ common cohort cho mọi cell.
- Đủ 180 fits và logs/checkpoints/predictions/provenance.
- Bảng 6×10, delta/interaction và phân tích cost/generalization.
- Winner khóa bằng validation; test cuối được báo riêng.

Lưu ý validation hiện tại: test tự tạo kiểm tra Parquet/raw/predict_features;
MediaPipe video path có lazy face/pose extraction nhưng cần smoke trên video
thực ở runtime có MediaPipe/model assets trước khi chứng nhận demo video.
