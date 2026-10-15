# Google Isolated Sign Language Recognition (ISLR)

Hệ thống nhận diện ngôn ngữ ký hiệu Mỹ (Isolated Sign Language Recognition - ISLR) 20 gloss dựa trên dataset Kaggle `asl-signs` (MediaPipe Holistic landmarks trích xuất sẵn trong định dạng Parquet) và mô hình GRU (Gated Recurrent Unit).

## 1. Mục tiêu và Kiến trúc
- **Dataset:** Kaggle Competition `asl-signs` (543 landmarks/frame: face, pose, left_hand, right_hand; frame thiếu landmark = NaN).
- **Mục tiêu:** Nhận diện 20 gloss mục tiêu với phân chia người ký độc lập (Signer-Independent split).
- **Trích xuất đặc trưng & Tiền xử lý:**
  - Lọc bỏ frame không có tay hợp lệ (Trim idle frames).
  - Xác định dominant hand theo phân tích dữ liệu thực tế.
  - Nội suy các frame mất tay ngắn hạn (Interpolation), điền 0 và đặt mask.
  - Resample chuỗi thời gian về độ dài cố định $T=32$ frames.
  - Chuẩn hóa vector đặc trưng theo cấu hình config.
- **Mô hình core:** PyTorch GRU Classifier (2 lớp, hidden 128, dropout 0.3) với cấu hình hoàn toàn điều khiển bởi config.
- **Môi trường thực thi:** Tự động phát hiện và tương thích tối đa trên Google Colab, Kaggle Notebook và Local.

## 2. Cấu trúc thư mục
```
SignLanguageReconition_ML/
├── README.md
├── requirements.txt
├── requirements-inference.txt
├── requirements-colab.txt
├── pytest.ini
├── configs/
│   └── islr.yaml               # Canonical configuration file (Single Source of Truth)
├── data/
│   ├── cache/                  # Thư mục cache dữ liệu và parquet tải về
│   ├── selection/              # Báo cáo và danh sách 20 gloss được lựa chọn
│   └── splits/                 # File phân chia train/val/test signer-independent
├── notebooks/
│   └── islr_phase2.ipynb       # Notebook mỏng cho Colab / Kaggle
├── scripts/
│   ├── 00_setup_check.py       # Kiểm tra môi trường và schema parquet
│   ├── 10_prepare_data.py      # Pipeline dữ liệu: Selection, Download, Preprocess, Split
│   └── 20_train.py             # Huấn luyện mô hình GRU & Random Forest baseline
├── src/
│   ├── data/
│   │   ├── kaggle_access.py    # Tải dữ liệu an toàn từ Kaggle API với bộ nhớ đệm
│   │   ├── islr_schema.py      # Kiểm tra và phân tích schema Parquet
│   │   ├── dataset.py          # PyTorch Dataset đọc .npy từ manifest
│   │   └── augmentation.py     # Data augmentation cho tọa độ và đặc trưng
│   ├── features/
│   │   ├── normalization.py    # Chuẩn hóa landmarks và tính toán feature vector
│   │   └── landmark_extractor.py # Trích xuất landmark MediaPipe (dùng cho Phase 3)
│   ├── models/
│   │   └── gru.py              # Mô hình ASLGRUClassifier
│   ├── training/
│   │   ├── train.py            # Huấn luyện mô hình với Early Stopping & Scheduler
│   │   ├── evaluate.py         # Đánh giá Top-1, Top-3, Macro-F1 & Confusion Matrix
│   │   └── metrics.py          # Hàm tính độ chính xác và F1-score
│   └── utils/
│       ├── config.py           # Strictly typed config dataclasses & CLI override (--set)
│       ├── env.py              # Tự động nhận diện môi trường & resolve canonical paths
│       ├── logging.py          # Module ghi log chuẩn hóa
│       └── seed.py             # Cố định random seed tái lập kết quả
└── tests/                      # Bộ kiểm thử đơn vị hoàn toàn tự trị
```

## 3. Cài đặt và Chạy kiểm thử
```bash
# Cài đặt môi trường
pip install -r requirements.txt

# Chạy toàn bộ test suite (không cần kết nối mạng)
pytest -q tests/
```

## 4. Quy trình thực thi (Config-Driven Workflow)
Mọi thiết lập và tham số điều chỉnh được quản lý tại `configs/islr.yaml` và hỗ trợ ghi đè qua `--set key=value`:

1. **Kiểm tra môi trường và kiểm định Schema:**
   ```bash
   python scripts/00_setup_check.py --config configs/islr.yaml
   ```

2. **Chuẩn bị dữ liệu (Lựa chọn gloss, tải file, tiền xử lý, chia split):**
   ```bash
   # Chạy toàn bộ pipeline chuẩn bị dữ liệu
   python scripts/10_prepare_data.py --config configs/islr.yaml

   # Hoặc chạy từng giai đoạn riêng biệt:
   python scripts/10_prepare_data.py --config configs/islr.yaml --stage selection
   python scripts/10_prepare_data.py --config configs/islr.yaml --stage download
   python scripts/10_prepare_data.py --config configs/islr.yaml --stage preprocess
   python scripts/10_prepare_data.py --config configs/islr.yaml --stage split
   ```

3. **Huấn luyện mô hình:**
   ```bash
   # Huấn luyện chính thức theo config
   python scripts/20_train.py --config configs/islr.yaml

   # Chế độ kiểm tra quá khớp (Sanity check)
   python scripts/20_train.py --config configs/islr.yaml --overfit-check

   # Chế độ baseline Random Forest
   python scripts/20_train.py --config configs/islr.yaml --baseline-rf
   ```
