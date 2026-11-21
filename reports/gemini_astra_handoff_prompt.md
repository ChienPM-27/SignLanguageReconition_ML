# Prompt giao Gemini triển khai phần tiếp theo

Copy nội dung block dưới đây vào Gemini đang làm việc trong repo này:

**Cập nhật 2026-11-05:** Wave A/B đã có code nhưng chưa đạt gate. Phạm vi tiếp theo là sửa các findings đã tái hiện; không làm lại toàn bộ plan gốc. Current status và acceptance nằm trong `reports/astra_remediation_plan.md`.

```text
Bạn đang triển khai dự án D:\MyProject\SignLanguageReconition_ML.

Đọc AGENTS.md, reports/astra_remediation_plan.md và
reports/astra_gates_a_b_review.md trước. reports/astra_next_implementation_plan.md
giữ thiết kế chi tiết nhưng checklist/status lúc lập đã cũ.
Không mặc định reports/p0_implementation_plan.md ghi completed nghĩa là
corrected baseline và mọi compatibility gate đã hoàn tất.

Phạm vi lượt tiếp theo: Phase 0–1 trong remediation plan: T-01,
A-01–A-04, E-01 và N-01, rồi synthetic smoke. Source hiện tại đã có
Wave A/B; giữ implementation đúng và sửa gaps, không viết lại từ đầu.
Chưa tích hợp TCN/HandGraphTCN hoặc chạy real-data campaign trong lượt này.
Sau Gate A, làm Phase 2 để sửa B-01–B-04 và adapters, rồi review Gate B.

Ưu tiên regressions: malformed epoch reject mà weights/state không đổi;
same-shape pooling mismatch/missing labels/optimizer phải reject;
frozen per-sequence signer/gloss/content identity; evaluator không fallback
missing checkpoint config hoặc đổi model qua overrides; report có sample
membership/identities/confusion; test fixture phải truyền glosses_csv.

Thực hiện:
1. Audit source, working-tree diff và run tests/pyflakes. Bảo toàn changes
   có sẵn. Ghi exact commands, versions, exit codes và findings.
2. Khép kín metadata/schema/cache/config validation; bỏ silent pad/truncate;
   validate resume trước state mutation; reject missing/legacy metadata;
   validate checkpoint classes và instantiate new models từ canonical config.
3. Freeze vocabulary, participant assignment và sequence membership. Khi
   regenerate features, rebind frozen split, báo exclusions; không chọn seed
   split khác để bù mất coverage.
4. Giữ canonical evaluation.evaluate_test=false đã có; hoàn thiện report
   provenance và khóa cả RF/notebook về train/validation. Final test là
   separate explicit workflow sau model selection, không chọn newest checkpoint.
5. Tạo synthetic fixtures/checkpoints có metadata hợp lệ; regression tests
   phải phát hiện missing metadata, plausible-shape identity mismatch,
   rejected resume changing weights, wrong labels và accidental test access.
6. Freeze baseline GRU hiện tại: hands_pose D160, T32, hidden128, layers2,
   unidirectional, mean_max, dropout.3, smoothing.1, unchanged training/aug
   snapshot, seeds42/43/44. T4-balanced preset là experiment riêng.
7. Dùng script có thật scripts/10_prepare_data.py --stage preprocess.
   Không relabel checkpoint/cache cũ thành corrected schema. Không chạy
   selection/download lại nếu không cần. Thiếu real dataset/hardware thì
   để real baseline pending, hoàn thành synthetic smoke và commands trước.
8. Cập nhật reports/astra_execution_log.md và status có bằng chứng trong
   plan/report liên quan. Không xóa lịch sử kết quả cũ.

Giữ single canonical key cho mỗi config concept. Mọi tunable nằm trong
configs/islr.yaml và dataclass src/utils/config.py; unknown YAML/dict/--set
key phải KeyError. Preserve credential/import/path invariants. Tests tự tạo
mọi dữ liệu trong tmp_path và pass khi optional SDK absent.

Chạy focused regressions, full pytest và pyflakes sau implementation.
Không bịa accuracy gains, test results, artifacts hoặc latency. Synthetic
pass không phải training pass. Không dùng sample videos/test metrics để tune.

Kết thúc bằng: changed files + lý do; commands/results; từng task status;
pending baseline measurements; commands tái lập trên Kaggle/Colab; điều kiện
để chuyển sang Phase 2 sửa Gate B. Không tự mở các phase sau trong cùng lượt này.
```

Các lượt sau dùng Phase 2–6 của remediation plan theo gate có bằng chứng;
tham chiếu design C/D/E trong plan gốc để giữ schema/protocol. Không mở lại
Wave B như một implementation mới. Wave E cần task riêng có trigger từ validation.
