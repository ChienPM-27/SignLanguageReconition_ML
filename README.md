# ASL Sign Recognition

Hệ thống nhận diện ngôn ngữ ký hiệu Mỹ (American Sign Language - ASL) thời gian thực từ webcam sử dụng MediaPipe Hands và mô hình GRU.

## 1. Mục tiêu dự án
- Xây dựng pipeline nhận diện ASL từ video/webcam.
- Đầu vào: Hand landmarks (2 tay, 21 landmarks/tay, tọa độ 3D x, y, z).
- Mô hình nhận diện chuỗi thời gian: GRU (Gated Recurrent Unit).
- Tự động phân đoạn chuyển động (Automatic Motion Segmentation) bằng rule-based motion detector và rolling buffer.
- Dự đoán Top-5 kèm độ tin cậy (confidence score) và cơ chế threshold UNKNOWN.

## 2. Phạm vi phiên bản đầu tiên
- **Dataset:** ASL-Citizen (20 gloss mục tiêu được lọc kỹ theo tiêu chí cân bằng mẫu và đa dạng người ký).
- **Trích xuất đặc trưng:** MediaPipe Hands kết hợp Dominant Hand sorting, dời cổ tay, palm scale, tọa độ toàn cục, vận tốc và mask (143 chiều/frame).
- **Mô hình core:** PyTorch GRU (hidden 128, 2 lớp; kèm ablation hidden 64, 1 lớp).
- **Inference:** Realtime webcam với state machine (WAITING ⇄ SIGNING).

## 3. Cấu trúc thư mục
```
SignLanguageReconition_ML/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   ├── videos/
│   │   └── metadata/
│   ├── processed/
│   ├── landmarks/
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_landmark_extraction.ipynb
│   └── 03_model_experiment.ipynb
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── features/
│   │   ├── landmark_extractor.py
│   │   └── normalization.py
│   ├── segmentation/
│   │   ├── motion_detector.py
│   │   ├── buffer.py
│   │   └── state_machine.py
│   ├── models/
│   │   └── gru.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── metrics.py
│   ├── inference/
│   │   └── realtime.py
│   └── utils/
│       ├── config.py
│       └── seed.py
├── models/
│   └── checkpoints/
├── scripts/
│   ├── download_dataset.py
│   ├── extract_landmarks.py
│   ├── create_splits.py
│   └── train_model.py
└── app/
    └── main.py
```

## 4. Hướng dẫn cài đặt môi trường
Khuyến nghị dùng **Python 3.11**:
```bash
# Tạo môi trường ảo
py -3.11 -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Kaggle ISLR Quickstart (Phase 0)
1. Install dependencies: `pip install -r requirements.txt` (or `requirements-colab.txt` on Colab).
2. Configure Kaggle credentials via environment variables (`KAGGLE_USERNAME`, `KAGGLE_KEY`) or `~/.kaggle/kaggle.json`.
3. Verify setup and single-file landmark probe: `python scripts/00_setup_check.py --config configs/islr.yaml`
4. Apply CLI overrides if desired: `python scripts/00_setup_check.py --set env.log_level=DEBUG`
5. Inspect resolved config in `outputs/resolved_config.yaml` and summary report in `reports/setup_check_report.yaml`.
6. Run comprehensive verification test suite: `pytest -q tests/`
