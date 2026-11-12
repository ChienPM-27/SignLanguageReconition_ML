# Ablation study V2

Campaign mới train lại với optional compact face và Transformer.
Xem [kế hoạch đầy đủ](PLAN.md), [ma trận 60 cells](MATRIX.json) và
[source/validation status](STATUS.md).

Không trộn artifacts campaign v4 trong reports/ablation-study vào V2.
V2 F0 tương ứng feature của F0-control cũ: wrist image, palm frame,
wrist scale/translation tắt; mọi run được train lại bằng source v5.

Phân công: A feature + cohort/analysis; B GRU architecture; C GRU baseline,
LSTM và Transformer. Có 10 feature sets × 6 profiles × 3 seeds = 180 fits.
Mỗi cell có owner cụ thể trong MATRIX.json, không train baseline trùng.

Runner: tools/study.py. Tổng hợp có kiểm tra provenance: tools/summarize.py.
Chưa có training results V2; winner chỉ được chốt sau khi đủ ma trận.
