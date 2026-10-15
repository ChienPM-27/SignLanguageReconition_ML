# Báo cáo kỹ thuật: Sửa Augmentation & Cấu hình Huấn luyện Tổng quát hóa (Phase A)

## 1. Tóm tắt Diff theo File

### `src/data/augmentation.py` (Numpy Pipeline)
- **Helper `recompute_derived_features(features)`**: Tính lại relative wrist `[132:135]` và wrist velocity `[135:141]` từ tọa độ cổ tay `[126:132]` và mask `[141:143]`, tuân thủ 100% logic chuẩn hóa từ `compute_feature_vector_143` (`src/features/normalization.py`).
- **Sửa `horizontal_flip`**:
  - Đổi dấu trục x của local hand shape `[0:126]` và lật cổ tay $x \to 1-x$ tại `[126:132]` **chỉ cho các slot tay có mặt** (`mask > 0.5`).
  - Hỗ trợ `swap_hands=True/False` (mặc định huấn luyện dùng `swap_hands=False` để giữ slot 0 là tay trội).
  - Gọi `recompute_derived_features` sau khi biến đổi tọa độ thay vì đổi dấu rel/vel thủ công (tránh lỗi tại khung hình tay xuất hiện/biến mất).
  - Giữ nguyên nhánh xử lý pose (`D >= 155`).
- **Sửa `random_rotate_2d`**: Xoay wrist quanh tâm $(0.5, 0.5)$ **chỉ cho tay có mặt**. Không biến đổi slot vắng (tránh làm lệch tọa độ nội suy gốc). Không tính lại velocity để giữ vận tốc sạch.
- **Sửa `add_jitter`**: Cộng nhiễu Gaussian nhẹ **chỉ cho slot có mặt** (cả local shape và wrist). Không cộng nhiễu vào velocity hay rel wrist.
- **Thêm `shape_scale`**: Tỷ lệ co giãn độc lập $s_x, s_y \sim U(1-r, 1+r)$ nhân vào local shape `[0:126]` cho tay có mặt. Mặc định $r=0.10$.
- **Thêm `scale_translate`**: Tỷ lệ $s \sim U(\text{scale\_min}, \text{scale\_max})$ và độ dịch chuyển $t_x, t_y \sim U(-\text{translate}, +\text{translate})$ áp dụng cho wrist của tay có mặt: $w_{xy}' = (w_{xy} - 0.5) \cdot s + 0.5 + (t_x, t_y)$, $z$ giữ nguyên, sau đó gọi `recompute_derived_features`. Mặc định $\text{scale\_min}=0.85, \text{scale\_max}=1.15, \text{translate}=0.10$.
- **Cập nhật `apply_augmentation`**: Áp dụng chuỗi biến đổi theo thứ tự chuẩn hóa:
  $$\text{flip} \longrightarrow \text{rotate} \longrightarrow \text{shape\_scale} \longrightarrow \text{scale\_translate} \longrightarrow \text{jitter}$$

### `src/data/augmentation_gpu.py` (PyTorch Batch GPU/CPU Pipeline)
- **Helper `batch_recompute_derived_features(x)`**: Bản tính toán song song theo batch trên tensor `(B, T, D)`.
- **Cập nhật `batch_horizontal_flip`**: Đồng bộ logic bảo toàn slot vắng, hoán đổi slot tay khi `swap_hands=True`, và gọi `batch_recompute_derived_features`.
- **Cập nhật `batch_random_rotate_2d`**: Chỉ xoay wrist quanh $(0.5, 0.5)$ cho tay có mặt.
- **Cập nhật `batch_add_jitter`**: Chỉ thêm nhiễu vào shape và wrist có mặt.
- **Thêm `batch_shape_scale` & `batch_scale_translate`**: Đầy đủ tham số tường minh tùy chọn (`s, tx, ty, sx, sy`) phục vụ kiểm tra parity số học giữa numpy và torch.
- **Cập nhật `apply_augmentation_batch`**: Điều khiển qua `AugmentationConfig` theo đúng thứ tự như bản numpy.

### `src/utils/config.py`
- Mở rộng `AugmentationConfig` với hai khối cấu hình dùng tham số vô hướng (scalars):
  - `scale_translate`: `enabled=False, prob=0.7, scale_min=0.85, scale_max=1.15, translate=0.10`
  - `shape_scale`: `enabled=False, prob=0.5, r=0.10`
- Cập nhật hàm `validate()` của `AugmentationConfig` để kiểm tra tính toàn vẹn kiểu `dict` cho tất cả các phép biến đổi. Hỗ trợ đầy đủ cú pháp CLI override `--set augmentation.<aug>.<key>=<val>`.

### `configs/islr.yaml`
- Bật `flip.enabled: true` (prob: 0.5, swap_hands: false).
- Bật `scale_translate.enabled: true` (prob: 0.7, scale_min: 0.85, scale_max: 1.15, translate: 0.10).
- Bật `shape_scale.enabled: true` (prob: 0.5, r: 0.10).
- Giữ nguyên các thông số model/training còn lại để làm baseline thử nghiệm.

### `scripts/40_eval_samples.py`
- Script đánh giá 20 video mẫu độc lập:
  - CLI: `--model <best_model.pt> --glosses <glosses.csv> --videos_dir data/sample_videos --out <json>`.
  - Tích hợp `MediaPipeHandExtractor` và `ISLRPredictor`.
  - Đánh giá song song Biến thể A (video mặc định với dominance swap) và Biến thể B (video + mirror x).
  - Định dạng hiển thị tương thích Windows console (tránh lỗi mã hóa UTF-8 charmap) và xuất báo cáo JSON chi tiết.

### `tests/test_dataset.py` & `tests/test_augmentation.py`
- Cập nhật `tests/test_dataset.py`: Đặt `mask=1.0` trên các mảng synthetic ngẫu nhiên để các kiểm thử cũ kiểm tra đúng logic toán học của phép biến đổi khi có tay.
- Tạo mới `tests/test_augmentation.py` (8 test suites) bao phủ toàn bộ các yêu cầu của T4.
- Tạo mới `tests/test_config.py::test_augmentation_config_and_overrides` xác thực việc phân tích cú pháp override lồng nhau.

---

## 2. Kết quả Pytest & Pyflakes

- **Pyflakes**: Không còn cảnh báo hoặc biến/import không sử dụng (exit code 0 trên toàn bộ repo).
- **Pytest**: Toàn bộ **185 tests** vượt qua thành công:
  ```text
  collected 185 items
  ........................................................................ [ 38%]
  ........................................................................ [ 77%]
  .........................................                                [100%]
  185 passed in 20.25s
  ```

### Chi tiết các kiểm thử trong `tests/test_augmentation.py`:
1. `test_absent_slot_preservation_numpy`: Bảo toàn giá trị slot vắng (kể cả khi wrist vắng có giá trị khác 0 do nội suy temporal resampling).
2. `test_absent_slot_preservation_torch`: Bảo toàn giá trị slot vắng trên batch PyTorch tensor.
3. `test_recompute_derived_features_parity`: Kết quả `recompute_derived_features` khớp chính xác `[132:141]` sinh từ pipeline chuẩn `process_video_landmarks`.
4. `test_double_flip_identity`: Lật gương 2 lần (`swap_hands=False`) là phép đồng nhất (identity mapping) trên slot có mặt, bảo toàn cả rel và vel.
5. `test_augmentation_parity_numpy_vs_torch`: Parity số học tuyệt đối giữa numpy và torch cho từng phép biến đổi với tham số cố định (tolerance `1e-6`).
6. `test_shape_and_dtype_preservation`: Đảm bảo kích thước `(T, 143)` và kiểu dữ liệu `float32` không bị biến đổi.
7. `test_smoke_train_2_epochs_with_augmentation[ram]`: Huấn luyện thử nghiệm 2 epochs trên dữ liệu synthetic với `preload='ram'` và augmentation bật.
8. `test_smoke_train_2_epochs_with_augmentation[device]`: Huấn luyện thử nghiệm 2 epochs trên dữ liệu synthetic với `preload='device'` và augmentation bật.

---

## 3. Rủi ro & Điểm Chưa Chắc Chắn

1. **Ghi chú về `time_stretch`**:
   - Hiện tại `time_stretch` trong `src/data/augmentation.py` đang nội suy tuyến tính chuỗi từ $T$ sang $T_{\text{stretched}}$, sau đó lại nội suy ngược ngay về $T$. Điều này làm triệt tiêu hiệu ứng co giãn thời gian thực sự (coi như no-op hoặc chỉ làm mờ nhẹ độ dốc).
   - Theo chỉ dẫn của Phase A, `time_stretch` được giữ nguyên **TẮT** (`enabled: false`) trong cấu hình và không sửa đổi trong đợt này.
2. **Độ nhạy của nhãn trên video mẫu HandSpeak**:
   - Tập 20 video trong `data/sample_videos` chỉ mang tính chất kiểm tra sơ bộ (sanity check). Một số ký hiệu HandSpeak có biến thể cử chỉ khác đôi chút so với phân phối của cuộc thi ASL Signs (Kaggle). Tuyệt đối không điều chỉnh siêu tham số dựa trên kết quả 20 video này.
3. **Hiệu năng thực tế trên Kaggle**:
   - Mặc dù code augmentation đã được kiểm thử parity chặt chẽ và không còn làm hỏng slot vắng, việc model có thực sự cải thiện khả năng tổng quát hóa trên signer mới hay không chỉ có thể xác nhận sau khi chạy huấn luyện thực tế trên Kaggle.

---

## 4. Ba Lệnh Huấn luyện Thực nghiệm Trên Kaggle (3 Seeds)

Đã xác nhận trước bằng `src.utils.config.load_config`: Tất cả các tham số `--set` dưới đây đều được parse chính xác theo đúng kiểu dữ liệu (`bool`, `float`, `int`, `str`).

### R1: Chỉ bật Flip (Flip-Only)
Kiểm tra tác động độc lập của việc sửa lỗi lật gương và bảo toàn slot có mặt:
```bash
python scripts/20_train.py --config configs/islr.yaml --seeds 42 43 44 --set run.name=exp_r1_flip --set augmentation.scale_translate.enabled=false --set augmentation.shape_scale.enabled=false
```

### R2: Bật tất cả Augmentation (All Aug)
Bật toàn bộ bộ augmentation mới (Flip + Rotate + Shape Scale + Scale & Translate + Jitter):
```bash
python scripts/20_train.py --config configs/islr.yaml --seeds 42 43 44 --set run.name=exp_r2_aug
```

### R3: R2 kèm Tăng cường Regularization (All Aug + Reg)
Tăng dropout lên 0.4 và weight decay lên 0.001 để đối phó với hiện tượng overfit signer:
```bash
python scripts/20_train.py --config configs/islr.yaml --seeds 42 43 44 --set run.name=exp_r3_reg --set model.dropout=0.4 --set training.weight_decay=0.001
```

---

## 5. Bảng Baseline Đối Chứng

| Chỉ số | Baseline Hiện Tại (Trước khi sửa) | Mục tiêu Kỳ vọng (Phase A) |
| :--- | :--- | :--- |
| **Val Accuracy (Seed 42)** | 0.665 | Cải thiện thu hẹp khoảng cách train-val |
| **Val Accuracy (Seed 43)** | 0.679 | Cải thiện độ ổn định giữa các signer |
| **Val Accuracy (Seed 44)** | 0.665 | Cải thiện độ ổn định giữa các signer |
| **Test Top-1 Accuracy** | 0.8107 ± 0.002 | Duy trì hoặc cải thiện |
| **Sample Videos Top-1 (Pipeline A)** | 7 / 20 (35.0%) | Tăng độ chính xác không phụ thuộc tay thuận |
| **Sample Videos Top-3 (Pipeline A)** | 14 / 20 (70.0%) | Tăng độ bao phủ ứng viên đúng |
