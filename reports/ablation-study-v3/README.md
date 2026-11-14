# Source campaign V3 `b4fd56f03a57`

Repo chứa source/config/tests và ba notebooks của campaign V3 đã freeze. Shared ZIP, checkpoints, raw data, cache, results ZIP và PDF được lưu riêng ngoài Git. Source snapshot trong Shared gồm 101 files; đừng chỉnh source/config đã freeze trong runtime của campaign đang chạy.

## Phân công và protocol

| Member | Vòng đầu | Configs / fits |
|---|---|---:|
| A | GRU G0 trên H, B, F, HB, HF, BF, HBF | 7 / 21 |
| B | GRU mean pooling, attention pooling, bidirectional trên HBF | 3 / 9 |
| C | LSTM và Transformer trên cùng 7 tổ hợp vùng | 14 / 42 |

H = bàn tay, B = thân trên, F = mặt. Vòng đầu có 24 configs/72 fits với seeds42/43/44. Nếu architecture GRU tốt nhất trên HBF khác G0, B chạy thêm sáu subsets theo `cross_plan.zip` do A tổng hợp. Tổng tối đa30 configs/90 fits; 42 YAML trong repo là các phương án hợp lệ, chỉ chạy queue được phân công.

Cache chung196 channels: tay143, thân17, mặt36. Mỗi vùng được project về hidden128, L2-normalize rồi cộng theo trọng số70/20/10; khi bỏ vùng, chỉ chuẩn hóa lại trọng số của các vùng bật. Frame thiếu quan sát đóng góp0. Temporal core xử lý chuỗi T32; hidden128/layers2/dropout0.3; batch32; Adam lr0.001/weight_decay0.0001; label_smoothing0.1; tối đa60 epochs; early stopping patience15/min_delta0.0001/min_epochs5. Transformer dùng heads4/feedforward256.

Cohort cố định20 glosses/7.854 sequences: train5.558/15 signers, validation1.155/3, test1.141/3. Checkpoint chọn theo Validation Loss; config chọn theo mean Validation Macro-F1, rồi std thấp hơn, parameters ít hơn và run ID. Test giữ kín tới khi đủ vòng đầu và cross-plan, khóa winner và checkpoint hashes. Trọng số70/20/10 là lựa chọn thiết kế, chưa được chứng minh tối ưu.

## Chạy trên Kaggle

1. Import `notebooks/ablation_v3_member_<a|b|c>_kaggle.ipynb`.
2. Attach competition `asl-signs` và Dataset chứa `ablation_v3_shared_b4fd56f03a57.zip` hoặc Shared đã giải nén có `freeze.json`, `config.yaml`, `code/`.
3. Bật GPU. Giữ `SHARED_INPUT = ""` để tự tìm; nếu attach nhiều Shared, chỉ định đúng ZIP hoặc thư mục gốc. Đặt `RAW_INPUT` tới thư mục có `train.csv` và `train_landmark_files/`; notebooks B/C đã sửa mặc định thành `/kaggle/input/competitions/asl-signs`.
4. Giữ `MODE = "train"`, `RESTORE_INPUT = ""`, `CROSS_PLAN_INPUT = ""`, `RUN_IDS = None`; Run All.
5. Sau mỗi config, tải Results ZIP tại `/kaggle/working/ablation_v3_results/member-<member>/` hoặc Save Version giữ Output.

Notebook kiểm source/config/split/protocol hashes và dùng code đóng gói trong Shared. Checkout Git trên Kaggle không cần thiết. Các notebooks được bàn giao là launchers của freeze hiện tại; `tools/build_freeze.py` là công cụ tạo snapshot mới từ Shared/cohort đã chuẩn bị, không chạy để thay thế campaign đang train.

Khôi phục: attach ZIP results V3 của chính member, đặt `RESTORE_INPUT` tới FILE ZIP. Queue bỏ qua configs đã nghiệm thu. A dùng `MODE = "aggregate"` và `RESULT_INPUTS` là ZIP B/C để tổng hợp. Nếu cần kiểm chứng GRU, B attach `cross_plan.zip` và results B vòng đầu, đặt `CROSS_PLAN_INPUT` cùng `RESTORE_INPUT`. Khi aggregate trả `COMPLETED`, A mới dùng `MODE = "test"`.

## Kiểm tra source

Cài dependencies trong `requirements.txt`, cùng `pytest`, `pyflakes` và `nbformat` cho kiểm tra phát triển. Chạy từ repo root:

```powershell
python -m pytest -q -p no:cacheprovider
python -m pyflakes src scripts tests reports/ablation-study-v3
```

Tests tạo dữ liệu synthetic trong thư mục tạm; không cần raw data, Shared hoặc checkpoints có sẵn. `verify_freeze.py` là kiểm tra gói bàn giao local và cần artifacts cùng cleanup ledger trên máy lưu trữ. `render_plan.py` tạo PDF local, cần `reportlab` và fonts Arial trên Windows. `analyze_member_a_results.py` kiểm ZIP results và tính lại metrics từ validation predictions, không chạy training/inference.

Member A đã nghiệm thu7/7 configs,21/21 fits. B/C đang chờ kết quả; chưa có global winner. Chi tiết: [STATUS.md](STATUS.md).
