# Lưu artifacts và bảng ablation study

> Các bundles/results, bảng kết quả và PDF V1 mô tả bên dưới đã chuyển vào Windows Recycle Bin ngày 14/11/2026 theo yêu cầu chỉ giữ V3. Sơ đồ và lệnh bên dưới là tài liệu cũ, không mô tả nội dung local hiện tại. Xem [V3 hiện tại](../ablation-study-v3/STATUS.md) và [ledger](../cleanup/2026-11-14.json).

Trên máy hiện tại, repo root là `D:\MyProject\SignLanguageReconition_ML`.
Các đường dẫn trong sơ đồ bên dưới tương đối với repo root.

```text
artifacts/ablation-study/
  shared/
    ablation_shared.zip
    config.yaml
    source_commit.txt
    split_manifest.json
    selection/glosses.csv
  F0/
    20261111T135519_479624Z.zip
    20261111T135519_479624Z/       # logs, models, metadata và báo cáo
  F0-control/
    20261111T140715_290931Z.zip
    20261111T140715_290931Z/
  F1/                           # Trainer B bổ sung ZIP và attempt đã giải nén
  F2/
  F3/                           # Trainer C bổ sung ZIP và attempt đã giải nén
  F4/
  exports/trainer-a/
    RESULTS.md                  # Bản gốc tải từ Kaggle
    RESULTS.json
  import_manifest.json          # SHA256 của các file đã copy

reports/ablation-study/
  results/trainer-a/RESULTS.md
  results/trainer-a/RESULTS.json
  results/trainer-b/             # Bảng riêng sau khi nhận F1/F2
  results/trainer-c/             # Bảng riêng sau khi nhận F3/F4
  RESULTS.md                    # Bảng chung của 6 cấu hình; thiếu run thì PENDING
  RESULTS.json
  runs/F0.md                    # Mẫu ghi chú/review từng cấu hình
  runs/F0-control.md
  REPORT.md                     # Nhận xét và kết luận của study
```

Giữ nguyên tên ZIP và thư mục timestamp của từng attempt. Khi giải nén ZIP
vào thư mục run, giữ nguyên cấp thư mục timestamp mà ZIP đã chứa.
Không gom hoặc đổi tên các file `best_model.pt` của ba seeds.

`artifacts/ablation-study/` chứa bản lưu local của model/log/ZIP và được Git ignore.
Các bảng Markdown/JSON và ghi chú trong `reports/ablation-study/` dùng để review
và đưa lên Git khi cần. Giữ ZIP trên artifact storage của team để chia sẻ lâu dài.

## Tạo bảng riêng của A

Chạy từ repo root bằng môi trường Python của project:

```powershell
.\.venv\Scripts\python.exe reports/ablation-study/tools/summarize_results.py artifacts/ablation-study/F0/20261111T135519_479624Z artifacts/ablation-study/F0-control/20261111T140715_290931Z --output reports/ablation-study/results/trainer-a/RESULTS.md
```

Công cụ tạo JSON bên cạnh Markdown. B/C dùng cùng lệnh với hai attempt paths
của mình và output `results/trainer-b/RESULTS.md` hoặc `results/trainer-c/RESULTS.md`.

## Cập nhật bảng chung

Sau khi bổ sung các ZIP đã giải nén của B/C:

```powershell
.\.venv\Scripts\python.exe reports/ablation-study/tools/summarize_results.py artifacts/ablation-study --output reports/ablation-study/RESULTS.md
```

Nếu một cấu hình có nhiều attempts, truyền rõ các attempt paths được chọn.
Công cụ không tự chọn lần mới nhất hoặc seed có điểm cao nhất.

Hiện tại artifacts của A đã được kiểm tra và nhập; bảng chung chứa F0 và
F0-control COMPLETED. F1–F4 PENDING nghĩa là chưa có artifacts tương ứng trong
thư mục này, không phản ánh trạng thái runtime của các thành viên khác.

Shared ZIP để B/C dùng nằm ở `artifacts/ablation-study/shared/ablation_shared.zip`.
Bản tải ban đầu trong Downloads được giữ nguyên.
