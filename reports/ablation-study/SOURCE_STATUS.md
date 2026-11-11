# Kiểm tra source cho ablation study

Ngày kiểm tra: 2026-11-11 (+07:00).

**Kết luận: source đủ cho F0–F4 chưa nằm trong main local hoặc main trên GitHub.**
Working tree có implementation mở rộng chưa commit; full tests mới chạy đạt
249 passed và pyflakes sạch. Source này vẫn cần commit/freeze trước handoff.

| Thành phần | main local tại 8e335b7 | Working tree hiện tại |
| --- | --- | --- |
| Preprocessing schema | v2 | v4 |
| wrist_reference / palm_scale / tips / bends | Chưa có trong canonical YAML | Có đủ 4 options |
| Frozen signer split | Chưa có split.manifest_path trong YAML | Có manifest/rebind support |
| Validation-only mặc định | Chưa có evaluation.evaluate_test trong YAML | evaluate_test=false |
| Reliability/artifact/checkpoint helpers | Chưa có 3 modules bên dưới | Có nhưng untracked |
| Standalone evaluator | Chưa có scripts/21_evaluate.py | Có nhưng untracked |
| GRU/LSTM | Có | Có |
| MaskedTCN / HandGraphTCN | Chưa tích hợp | Chưa tích hợp |

main local, cached origin/main và main trên GitHub cùng SHA:
`8e335b7a46f1e2e9d1b0c47850aa053c852b55f0`.

Đã xác minh GitHub main bằng git ls-remote origin refs/heads/main ngày 2026-11-11.

Các file phụ thuộc quan trọng chưa tracked tại thời điểm kiểm tra:

- `src/features/validity.py`
- `src/utils/artifacts.py`
- `src/utils/checkpoint.py`
- `scripts/21_evaluate.py`
- `tests/test_evaluate.py`
- `tests/test_features_wave_b.py`
- `tests/test_wave_a_regression.py`
- `tests/test_wave_b_regression.py`

Nhiều file configs/src/scripts/tests và notebook cũng modified.
Không chỉ commit YAML: phải đưa đầy đủ các thay đổi liên quan và modules mới
vào source commit được review trước handoff.

## Handoff source

| Trường | Giá trị |
| --- | --- |
| Source ref đã chia sẻ | PENDING |
| Source commit SHA dùng cho cả A/B/C | PENDING |
| Config snapshot SHA256 | PENDING |
| Shared frozen split SHA256 | PENDING |
| Ordered vocabulary/selection SHA256 | PENDING |
| Full tests command / result / artifact | Working tree: 249 passed, 1 warning, 81.53s; source freeze PENDING |
| Pyflakes command / result / artifact | Working tree + tools: 0 findings, exit 0; source freeze PENDING |
| Thành viên kiêm chốt source / reviewer | PENDING |

Source freeze có thể nằm trên branch implementation được review hoặc main sau
merge. Report branch lưu source ref/SHA thực tế; tên branch không thay cho SHA.
Đừng gửi team clone main hiện tại rồi chạy commands v4.

Các kiểm tra trong phiên này là Git tree/config/source inspection và syntax
inspection được ghi trong báo cáo tổng hợp. Không có GPU measurements mới.
Execution log cũ có synthetic verification, nhưng không dùng kết quả cũ để
certify một source commit mới.

Syntax inspection mới: Python ast.parse trên 67 file src/scripts/tests,
0 lỗi cú pháp. Runtime .venv đã dùng được sau khi quyền workspace được cấp.
Pyflakes: python -m pyflakes reports/ablation-study/tools src scripts tests app.py,
0 findings, exit 0. Full tests: python -m pytest tests -q -p no:cacheprovider
với basetemp riêng trong workspace: 249 passed, 1 warning, 81.53s, exit 0.
Warning do CUDA không có tại máy local; không có GPU measurements mới.
Kiểm tra working tree không đồng nghĩa source v4 đã có trên main.

## Trạng thái branch báo cáo

Đã tạo bộ báo cáo trong reports/ablation-study và 2 tools lưu log/tổng hợp bảng.
Đã tạo và checkout branch codex/ablation-study từ main tại 8e335b7.
Branch báo cáo chỉ commit tài liệu và tools trong reports/ablation-study.
Các thay đổi source có sẵn vẫn giữ nguyên và chưa được đưa vào commit mới.

Workflow cập nhật báo cáo tiếp theo:

```bash
git switch codex/ablation-study
git add -f -- "reports/ablation-study/*.md" "reports/ablation-study/runs/*.md" "reports/ablation-study/tools/*.py"
git commit -m "docs: prepare ablation study protocol and team reports"
git push -u origin codex/ablation-study
```

Commit báo cáo này không chứa corrected source đang modified/untracked.
Team cần source freeze riêng trước khi giao jobs cho A/B/C.

Các tools mới: run_ablation.py giữ stdout/stderr, metadata và ZIP; summarize_results.py
tạo bảng validation Markdown/JSON từ artifacts thật. Kiểm tra tools: 16 checks
pass trên fixtures self-contained; xem TOOL_VALIDATION.md. Kết quả này không
thay thế full source tests/pyflakes hoặc real-data training.
