# Báo cáo LaTeX ablation study

`main.tex` là tài liệu chính; `ablation_study.pdf` là bản PDF xuất bằng XeLaTeX.
Báo cáo viết bằng tiếng Việt, lấy kết quả từ `../RESULTS.json`, kiểm tra lại
artifacts thật và giữ các cấu hình chưa có kết quả ở trạng thái PENDING.

## Cập nhật khi nhận kết quả F1–F4

1. Lưu ZIP và giải nén từng attempt vào `artifacts/ablation-study/<run>/<attempt>/`.
2. Từ repo root, chạy:

```powershell
& ./reports/ablation-study/latex/build_report.ps1
```

Lệnh kiểm tra artifacts, cập nhật RESULTS.md/JSON, sinh bảng LaTeX và xuất lại PDF.
Nếu một run có nhiều attempts, chọn các attempt cụ thể để tránh chọn nhầm lần train:

```powershell
& ./reports/ablation-study/latex/build_report.ps1 -AttemptPaths @(
    'artifacts/ablation-study/F0/20261111T135519_479624Z',
    'artifacts/ablation-study/F0-control/20261111T140715_290931Z'
    # Thêm đường dẫn attempt đã chọn của F1/F2/F3/F4 khi có kết quả.
)
```

Yêu cầu Python standard library và XeLaTeX (MiKTeX/TeX Live với fontspec,
TeX Gyre Termes/Heros/Cursor). Script ưu tiên Python của `.venv`.
Không chạy lại training và không evaluate test.

## Cấu trúc

- `main.tex`: nội dung, protocol, lý do thí nghiệm, giới hạn và hướng dẫn cập nhật.
- `data.tex`, `tables/*.tex`: số liệu tự sinh; chỉnh ở nguồn dữ liệu, không sửa tay.
- `refresh_tables.py`: xác minh kết quả bằng công cụ tổng hợp hiện có rồi sinh bảng.
- `report_snapshot.json`: ngày cập nhật (+07:00) và danh sách attempts đã dùng.
- `build_report.ps1`: cập nhật số liệu và compile hai lần để hoàn thiện references.

Mở thư mục này trong LaTeX editor của Codex để chỉnh sửa và xem preview.
PDF xuất tại chỗ; log/auxiliary files ở `tmp/pdfs/ablation-study-build/`.
Sau khi hoàn tất cả campaign, review lại phần diễn giải trước khi dùng làm kết luận
cuối cùng; số liệu mới không tự chứng minh significance hay chất lượng final test.
