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
- **Thêm `scale_translate`**: Tỷ lệ $s \sim U(\text{scale\_min}, \text{scale\_max})$ và độ dịch chuyển $t_x, t_y \sim U(-\text{translate}, +\text{translate})$ (hoặc tham số $t=(t_x, t_y)$) áp dụng cho wrist của tay có mặt: $w_{xy}' = (w_{xy} - 0.5) \cdot s + 0.5 + (t_x, t_y)$, $z$ giữ nguyên, sau đó gọi `recompute_derived_features`. Mặc định $\text{scale\_min}=0.85, \text{scale\_max}=1.15, \text{translate}=0.10$.
- **Cập nhật `apply_augmentation`**: Áp dụng chuỗi biến đổi theo thứ tự chuẩn hóa:
  $$\text{flip} \longrightarrow \text{rotate} \longrightarrow \text{shape\_scale} \longrightarrow \text{scale\_translate} \longrightarrow \text{jitter}$$

### `src/data/augmentation_gpu.py` (PyTorch Batch GPU/CPU Pipeline)
- **Helper `batch_recompute_derived_features(x)`**: Bản tính toán song song theo batch trên tensor `(B, T, D)`.
- **Cập nhật `batch_horizontal_flip`**: Đồng bộ logic bảo toàn slot vắng, hoán đổi slot tay khi `swap_hands=True`, và gọi `batch_recompute_derived_features`.
- **Cập nhật `batch_random_rotate_2d`**: Chỉ xoay wrist quanh $(0.5, 0.5)$ cho tay có mặt. Tự động broadcast các giá trị góc `angles_deg` dạng `float` hoặc scalar tensor qua toàn bộ batch $B \ge 1$.
- **Cập nhật `batch_add_jitter`**: Chỉ thêm nhiễu vào shape và wrist có mặt. Hỗ trợ tự động chuyển đổi tensor nhiễu và broadcast nhiễu unbatched qua batch $B \ge 1$.
- **Thêm `batch_shape_scale` & `batch_scale_translate`**:
  - Đầy đủ tham số tường minh tùy chọn (`s, t, tx, ty, sx, sy`) phục vụ kiểm tra parity số học giữa numpy và torch.
  - Hỗ trợ truyền tham số dạng số thực (`float`), tuple `t=(tx, ty)`, hoặc scalar tensor, tự động broadcast tương thích hoàn toàn trên batch tensor kích thước $B \ge 1$.
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
- Tạo mới `tests/test_augmentation.py` (9 test suites) bao phủ toàn bộ các yêu cầu của T4, bổ sung kiểm tra vắng tay 0, vắng cả hai tay, lật gương 2 lần cho cả `swap_hands=True/False`, và broadcast tham số vô hướng trên batch tensor.
- Tạo mới `tests/test_config.py::test_augmentation_config_and_overrides` xác thực việc phân tích cú pháp override lồng nhau.

---

## 2. Kết quả Pytest & Pyflakes

- **Pyflakes**: Không còn cảnh báo hoặc biến/import không sử dụng (exit code 0 trên toàn bộ repo).
- **Pytest**: Toàn bộ **186 tests** vượt qua thành công:
  ```text
  collected 186 items
  ........................................................................ [ 38%]
  ........................................................................ [ 77%]
  ..........................................                               [100%]
  186 passed in 20.37s
  ```

### Chi tiết các kiểm thử trong `tests/test_augmentation.py`:
1. `test_absent_slot_preservation_numpy`: Bảo toàn giá trị slot vắng (tay 1 vắng, tay 0 vắng, cả hai tay vắng; kể cả khi wrist vắng có giá trị khác 0 do nội suy temporal resampling).
2. `test_absent_slot_preservation_torch`: Bảo toàn giá trị slot vắng trên batch PyTorch tensor (tay 1 vắng, tay 0 vắng, cả hai tay vắng).
3. `test_recompute_derived_features_parity`: Kết quả `recompute_derived_features` khớp chính xác `[132:141]` sinh từ pipeline chuẩn `process_video_landmarks`.
4. `test_double_flip_identity`: Lật gương 2 lần (cho cả `swap_hands=False` và `swap_hands=True`) là phép đồng nhất (identity mapping) trên slot có mặt, bảo toàn cả rel và vel.
5. `test_augmentation_parity_numpy_vs_torch`: Parity số học tuyệt đối giữa numpy và torch cho từng phép biến đổi với tham số cố định (tolerance `1e-6`).
6. `test_batch_scalar_and_broadcasting_inputs`: Kiểm tra batch augmentations hỗ trợ truyền tham số dạng số thực / scalar và broadcast đồng nhất trên toàn bộ batch $B > 1$.
7. `test_shape_and_dtype_preservation`: Đảm bảo kích thước `(T, 143)` và kiểu dữ liệu `float32` không bị biến đổi.
8. `test_smoke_train_2_epochs_with_augmentation[ram]`: Huấn luyện thử nghiệm 2 epochs trên dữ liệu synthetic với `preload='ram'` và augmentation bật.
9. `test_smoke_train_2_epochs_with_augmentation[device]`: Huấn luyện thử nghiệm 2 epochs trên dữ liệu synthetic với `preload='device'` và augmentation bật.

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


---

## 6. Giai đoạn P0: Chuẩn hóa Toàn diện Preprocessing, Augmentation & Inference Parity (P0.1 - P0.8)

### 6.1. Mục tiêu và Nguyên nhân Thực hiện
Trước khi thử nghiệm và so sánh các kiến trúc mô hình mới, hệ thống cần loại bỏ 4 điểm bất đồng bộ (inconsistencies) nghiêm trọng đã được kiểm chứng trong pipeline tiền xử lý và suy luận:
1. **Pose Reflection sai lệch**: Phép lật gương cũ dùng $1 - x$ trên tọa độ pose đã được chuẩn hóa theo tâm torso (thay vì đổi dấu $-x$), đồng thời không hoán đổi khớp vai/khuỷu tay trái-phải và làm rò rỉ giá trị khác 0 vào các slot pose vắng.
2. **Vận tốc giả mạo khi dropout (Stale Wrist / Torso Velocities)**: Hiệu số vận tốc giữa các frame không kiểm tra điều kiện hỗ trợ của cả hai đầu mút ($t$ và $t-1$), dẫn đến vận tốc nhảy đột biến khi tay hoặc thân xuất hiện/biến mất.
3. **Nội suy lấp đầy vô hạn (Unbounded Imputation Artifacts)**: Resampling cũ nội suy tuyến tính toàn bộ các khoảng trống kéo dài hàng chục frame, tạo ra các tọa độ tay nhân tạo trôi lơ lửng không có thực trong video.
4. **Bất đồng bộ giữa Huấn luyện và Suy luận (Train/Inference Divergence)**: Pipeline suy luận (inference) và trích xuất video trực tiếp dùng các magic numbers, heuristic xác định tay thuận và cắt gọt frame độc lập với cấu hình huấn luyện.

### 6.2. Chi tiết các Thay đổi Kỹ thuật

#### P0.1: Đóng băng Contract Đặc trưng & Schema Versioning
- Khởi tạo `src/features/schema.py`:
  - `PREPROCESS_SCHEMA_VERSION = 2`.
  - Hằng số kích thước đặc trưng: `FEATURE_DIM_HANDS = 143`, `FEATURE_DIM_HANDS_POSE = 160`.
  - Slice indexing rõ ràng cho cả hai bố cục đặc trưng (`HANDS_LANDMARKS_SLICE`, `POSE_LANDMARKS_SLICE`, `WRIST_COORDS_SLICE`, `RELATIVE_WRIST_SLICE`, `WRIST_VELOCITY_SLICE`, `TORSO_VELOCITY_SLICE`, `SUPPORT_MASKS_SLICE`).
  - Hoán vị giải phẫu pose khi lật gương: `POSE_FLIP_PERMUTATION = (1, 0, 3, 2)` (hoán đổi vai trái $\leftrightarrow$ vai phải, khuỷu trái $\leftrightarrow$ khuỷu phải).
  - Hàm `get_feature_dim(feature_set: str) -> int`.
  - Hàm `compute_preprocess_identity(cfg_or_dict: Union[ISLRConfig, dict]) -> str`: Mã hóa SHA256 canonical JSON của các tham số tiền xử lý.
- Cập nhật `configs/islr.yaml` và `src/utils/config.py`:
  - Thêm `normalization_epsilon: 0.0001` vào `PreprocessConfig`.
  - Kiểm tra `time_stretch`: Tạm thời vô hiệu hóa (`enabled: false`), cấm kích hoạt khi chưa có validity-aware temporal contract.
  - Kiểm tra xung đột: Cấm `swap_hands=True` khi `dominant_hand=True` trong cấu hình.
  - Thêm hàm `build_config_from_dict(d: dict) -> ISLRConfig` hỗ trợ nạp cấu hình đã lưu trong checkpoint.

#### P0.2 & P0.3: Sửa Bounded Interpolation, Temporal Resampling & Gating Vận tốc
- Cập nhật `src/features/normalization.py`:
  - **`interpolate_bounded_gaps`**: Chỉ nội suy các khoảng trống có độ dài $\le \text{max\_gap\_frames}$. Các khoảng trống lớn hơn bị xóa về 0 hoàn toàn và hạ mask về 0.
  - **`resample_temporal`**: Nội suy hỗ trợ (support-aware). Chỉ nội suy giữa hai đầu mút hợp lệ; nếu khoảng cách thời gian giữa hai quan sát quá lớn, giữ nguyên giá trị 0. Bẻ hòa (tie-break) chỉ số theo khung hình trước đó; giữ nguyên giá trị khi chỉ có một điểm đầu mút hợp lệ.
  - **`compute_feature_vector`**:
    - Vận tốc cổ tay: $v[t, h] = w[t, h] - w[t-1, h]$ nếu và chỉ nếu cả $mask[t, h]$ và $mask[t-1, h]$ đều hợp lệ; ngược lại $v[t, h] = 0$.
    - Cổ tay tương đối: $r[t] = w[t, 0] - w[t, 1]$ nếu và chỉ nếu cả hai tay cùng hợp lệ; ngược lại $r[t] = 0$.
    - Vận tốc torso: $u[t] = C[t] - C[t-1]$ nếu và chỉ nếu pose hợp lệ ở cả hai frame $t$ và $t-1$; ngược lại $u[t] = 0$.
    - Slot không được hỗ trợ luôn được zero hóa nghiêm ngặt.

#### P0.4: Chuẩn hóa Phép Lật Pose (Pose Reflection) trong NumPy & PyTorch
- Cập nhật `src/data/augmentation.py` (`horizontal_flip`) và `src/data/augmentation_gpu.py` (`batch_horizontal_flip`):
  - Tọa độ vai và khuỷu tay chuẩn hóa theo tâm thân: $q'_{xy} = (-q_x, q_y)$ (đổi dấu $x$, giữ nguyên $y$).
  - Hoán vị giải phẫu khớp: $q'[t, k] = F \cdot q[t, \pi[k]]$ với $\pi = [1, 0, 3, 2]$.
  - Vận tốc torso: $u'_x = -u_x$, $u'_y = u_y$.
  - Slot vắng giữ nguyên giá trị 0 tuyệt đối.

#### P0.5: Đồng bộ Tính Nhất quán Động học cho Augmentation (Internal Motion Consistency)
- Cập nhật `random_rotate_2d`, `add_jitter`, `scale_translate` trong cả hai backend:
  - Tự động gọi tính lại các đặc trưng phái sinh (`recompute_derived_features`) sau khi biến đổi tọa độ cổ tay.
  - `add_jitter`: Đặt nhiễu tại gốc cổ tay cục bộ (landmark 0) bằng 0 tuyệt đối để bảo toàn gốc tọa độ hình dạng bàn tay.
  - `apply_augmentation` / `apply_augmentation_batch`: Chỉ tính lại đặc trưng khi có ít nhất một phép biến đổi thực sự được áp dụng, bảo toàn dữ liệu khi `prob=0.0`.
  - `time_stretch` / `batch_time_stretch`: Bắt lỗi `ValueError` rõ ràng khi nhận yêu cầu stretch khác 1.0.

#### P0.6: Dùng Chung Pipeline Tiền Xử lý giữa Huấn luyện và Suy luận
- Tạo mới `src/features/pipeline.py`:
  - Hàm chuẩn hóa cốt lõi `prepare_landmark_sequence`: Không I/O, độc lập nền tảng.
  - Xác thực shape, số lượng frame hợp lệ (`min_valid_frames`), áp dụng gating loại bỏ clip không đủ chất lượng.
  - Xác định tay thuận thống nhất: Đếm số frame hợp lệ $n_h$, giải quyết hòa bằng tổng độ dài quỹ đạo di chuyển cổ tay $E_h$, ưu tiên tay trái khi hòa tuyệt đối.
  - Cắt gọt khoảng lặng đầu/cuối clip (`pad_frames_around`) trên cùng một khoảng chỉ số $[start, end)$ cho cả tay và pose.
- Cập nhật `src/data/preprocess.py`: Ủy thác toàn bộ xử lý chuỗi cho `prepare_landmark_sequence`.
- Cập nhật `src/features/landmark_extractor.py`: Xuất đúng slot giải phẫu (slot 0: Left, slot 1: Right), không tự ý hoán đổi tay thuận trước khi đưa vào pipeline.
- Cập nhật `src/inference/predictor.py`:
  - Nạp và xác thực `resolved_config`, `schema_version`, kiểm tra tương thích giữa `get_feature_dim` và `model.input_size`.
  - Sử dụng chung `prepare_landmark_sequence` cho dữ liệu raw coordinates và video trích xuất.
  - Từ chối suy luận trên tọa độ raw đối với checkpoint cũ chưa có metadata phiên bản và cấu hình.
- Cập nhật `app.py`: Kết nối trực tiếp `predict_video` qua `predictor`.

#### P0.7: Gắn Kết Caches và Checkpoints với Preprocessing Identity
- Cập nhật `src/training/train.py` và `scripts/20_train.py`:
  - Lưu trữ `resolved_config`, `schema_version = 2`, và `preprocess_identity` vào checkpoint `pt`.
  - Xác thực khớp `preprocess_identity` giữa checkpoint và dataset khi tiếp tục huấn luyện (resume).
- Cập nhật `src/data/dataset.py`:
  - Kiểm tra file `preprocess_meta.json` trong thư mục cache, xác thực `schema_version == PREPROCESS_SCHEMA_VERSION`.

### 6.3. Quy tắc Di chuyển (Migration Rules) cho Caches & Checkpoints
1. **Dữ liệu Features đã Cache (`features_dir`)**:
   - Mọi mảng `.npy` được trích xuất từ phiên bản cũ (schema version 1 hoặc không có `preprocess_meta.json`) bị từ chối tự động.
   - Khi chạy huấn luyện hoặc kiểm tra dataset, cần tạo lại cache bằng script `scripts/10_preprocess.py`.
2. **Model Checkpoints**:
   - Checkpoints cũ không chứa `schema_version` sẽ bị từ chối khi gọi `predict_raw_coordinates` hoặc `predict_video` trên `ISLRPredictor`.
   - Để tiếp tục sử dụng, mô hình cần được huấn luyện lại trên tập features mới có `schema_version = 2`.
