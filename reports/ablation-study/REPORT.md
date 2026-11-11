# Kết quả feature ablation study

Campaign: F0–F4 + F0-control.
Trạng thái: **PREPARING — chưa đo**.
Ngày chuẩn bị: 2026-11-11 (+07:00).
Thành viên tổng hợp (kiêm train): PENDING. Cả 3 thành viên đều train.
Phân công: A = F0/F0-control; B = F1/F2; C = F3/F4. Mỗi người 6 model.
Bảng đo tự động sau train: RESULTS.md và RESULTS.json; dùng tools/summarize_results.py.

## Protocol đã chốt

| Trường | Giá trị |
| --- | --- |
| Source commit / ref | f6cf69fd4d22fece9c70a66a1b9f1e3461eb2fc2 / codex/ablation-study |
| Canonical config snapshot / SHA256 | shared/config.yaml / ef16f628cecce00be72cb8dd259140b0d21fbf448548bd8785a6d48f2baed0dd |
| Dataset và ordered 20-gloss vocabulary / SHA256 | PENDING |
| Frozen split manifest / SHA256 | PENDING |
| Selection provenance / historical test-use limitation | PENDING |
| Seeds | 42 / 43 / 44 |
| Model | GRU128×2, unidirectional, mean_max, dropout .3 |
| Temporal length | 32 |
| Training | batch32, epochs60, smoothing .1; các thông số khác theo snapshot |
| Checkpoint selection | Lowest validation loss mỗi seed |
| Feature selection | Mean Validation Macro-F1; tie-break Top-1 rồi CPU p95 nếu đã đo |
| Test policy | evaluate_test=false trong feature campaign |

## Bảng kết quả

Metrics mean ± std qua 3 seeds, đơn vị accuracy/F1 [0,1].
PENDING nghĩa là chưa đo; không điền 0 hoặc giá trị kỳ vọng thay kết quả.

| Run | Owner | D | Seeds hoàn tất | Val Top-1 | Val Top-3 | Val Macro-F1 | Val loss | Training time / GPU | Trạng thái |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| F0 | A | 160 | 0/3 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| F0-control | A | 160 | 0/3 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| F1 | B | 162 | 0/3 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| F2 | B | 162 | 0/3 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| F3 | C | 192 | 0/3 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| F4 | C | 212 | 0/3 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Controlled comparisons

| Comparison | Thành phần được đo | Δ Val Macro-F1 | Kết luận |
| --- | --- | --- | --- |
| F0-control − F0 | Tắt wrist-only scale/translation | PENDING | PENDING |
| F1 − F0-control | Body-relative wrist geometry | PENDING | PENDING |
| F2 − F1 | Clip-median palm scale | PENDING | PENDING |
| F3 − F2 | Fingertip displacement | PENDING | PENDING |
| F4 − F3 | Bend cosines | PENDING | PENDING |

## Cohort và integrity audit

- Train/val/test sample và signer counts từng cấu hình: PENDING.
- Exclusions theo gloss/signer/reason và class coverage: PENDING.
- Full-cohort / matched-cohort comparisons nếu cohort khác: PENDING.
- Cache/schema/split/vocabulary identities: PENDING.
- Source và protocol khớp giữa A/B/C: PENDING.
- Per-gloss confusion, per-signer metrics, best epochs, train–val gap: PENDING.
- Hardware/runtime/library versions; giới hạn khi so training time: PENDING.
- Source engineering verification: 249 tests passed; pyflakes sạch; SOURCE_STATUS.md.

Kiểm tra working tree ngày chuẩn bị: ast.parse 67 file Python, 0 lỗi cú pháp.
Đây không phải full tests/pyflakes hoặc chứng nhận source commit đã freeze.
Tools lưu log/tổng hợp: 16 checks pass trên fixtures tự tạo; xem TOOL_VALIDATION.md.
Working tree: full tests 249 passed, 1 warning (CUDA unavailable), 81.53s;
pyflakes trên tools/src/scripts/tests/app.py sạch. Source freeze: f6cf69fd4d22fece9c70a66a1b9f1e3461eb2fc2.
Chưa có model training metrics thực tế trong bảng này.

## Quyết định sau campaign

Selected feature layout: PENDING.
Lý do và validation evidence: PENDING.
Không mặc định F4 thắng. Campaign này dùng feature additions cộng dồn; các delta
là tác dụng khi thêm vào cấu hình trước, không chứng minh độc lập mọi interaction.
Nếu cần xác định interaction, lên thí nghiệm riêng sau bảng chính.

Architecture/T/label-smoothing ablations và final test report: chưa bắt đầu.
