# Kiểm tra tools ablation study

Ngày: 2026-11-11 (+07:00).
Kết quả: **16 checks pass** trên fixtures tự tạo trong temporary directory.

Các kiểm tra bao gồm:

- Sáu cấu hình có owner, canonical overrides và feature dimensions đúng.
- Bảng 6 runs / 18 seeds tính mean, population std và tổng training time đúng;
  artifacts tải về vẫn tổng hợp được dù đường dẫn máy train đã đổi.
- Từ chối thiếu log/checkpoint, duplicate seeds, NaN và aggregate không khớp.
- Từ chối shared artifact bị sửa, cohort khác giữa seeds và test_report.json.
- Từ chối trộn source commits và input attempts không hợp lệ.
- Run chưa hoàn tất không xuất mean ± std như kết quả COMPLETED.
- Giữ stdout/stderr và exit code của subprocess thất bại.
- Giữ lỗi launch subprocess trong log; cú pháp helper source hợp lệ.

CLI --help của cả hai tools chạy được; stdout/stderr hỗ trợ UTF-8 trên Windows.
Không train model hoặc dùng dữ liệu thật trong kiểm tra tools.
Không đưa số liệu fixtures vào REPORT.md hoặc bảng campaign.

Pyflakes trên tools/src/scripts/tests/app.py: 0 findings, exit 0 (.venv).
Full tests trên working tree: 249 passed, 1 warning do CUDA unavailable;
81.53s, exit 0. Dùng basetemp riêng và tắt pytest cache provider.
Source freeze: f6cf69fd4d22fece9c70a66a1b9f1e3461eb2fc2; config snapshot ở shared/.
Canonical config của 6 runs load thành công với D160/160/162/162/192/212.
Notebook JSON/Python syntax: 10 code cells hợp lệ (không thực thi notebook magics).
Frozen data split và real-data GPU campaign vẫn pending.
