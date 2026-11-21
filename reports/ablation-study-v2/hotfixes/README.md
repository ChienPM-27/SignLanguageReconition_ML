# Reporting compatibility v1 cho campaign V2 đã khóa

## Phạm vi mới: face-required

Nhóm đã chốt `features.face=true` cho mọi cell ngày 2026-11-13. Dùng
`face_only_v1.py` và `build_face_only_launchers.py`, notebooks/gói dưới
`artifacts/ablation-study-v2/face-only-v1`. A4/B15/C11, tổng 30 configs/90 fits.
Shared/configs/source frozen và các face runs hợp lệ giữ nguyên. Run face off
lưu lịch sử, không tính/rank trong bảng chính. Scope audit và module payload
lưu cùng ZIP. `KAGGLE_FACE_ONLY_FINALIZE_CELL.py` chỉ chốt/export A sau khi
cell training hiện tại kết thúc; không gọi training.

Sáu checks mới pass: canonical face subset, preserve/skip accepted face attempts,
legacy ZIP restore, 30-cell acceptance + exclude historical off winner, reject
corrupt scope/report, embedded finalization without training. Notebook/recovery
syntax/nbformat/pyflakes, embedded hashes, ZIP checksums/CRC, config copies và
unchanged frozen source/shared đã kiểm tra. Chưa có ZIP A cuối để nghiệm thu.

## Cập nhật 2026-11-13: dùng compatibility v2 để chạy face

Reporting v1 đã được xác nhận trên G0/F1 dataset thật, nhưng F1-face tiếp tục bị
Dataset whitelist cũ từ chối 198 chiều trước khi train seed42. Schema/layout,
preprocess và augmentation đã hỗ trợ face; test training trước đó dùng TensorDataset
nên không đi qua validator của ISRLandmarkDataset.

`campaign_compat_v2.py` bỏ guard dimension trùng bằng registry canonical đã có
trong source frozen và thêm binary face mask validation. Patch chỉ áp dụng lên đúng
helper frozen đã kiểm tra SHA256, không sửa files trong checkout hay feature arrays.
Nó chạy cùng payload reporting v1 giữ nguyên hash; hai addons được lưu trong
results ZIP và có audit trong từng report. Face-off F1 đã nghiệm thu được giữ lại.

`build_compat_v2_launchers.py` tạo `artifacts/ablation-study-v2/compat-v2/`:
notebooks A/B/C, handoff ZIPs và `KAGGLE_RESUME_CELL.py` dùng ngay trên runtime cũ.
Recovery cell cài launcher cả hai fixes, gọi full member queue với technical retry,
bỏ qua selected configs hợp lệ và giữ cache. A notebook mới mặc định full queue.
Shared 77a84a7b12f1 và tag frozen giữ nguyên; B/C dùng notebook compat_v2.

Kiểm tra mới: 69 checks pass, gồm cả 60 configs đúng model/hyperparameters với
synthetic raw landmarks -> preprocessing -> real Dataset preload=device trên CPU ->
device-loader augmentation -> optimizer backward -> aligned validation predictions.
Một subprocess chạy CLI 20_train.py thật với F1-face, ba seeds42/43/44, một epoch
trên fixture tự tạo, có ba checkpoints/reports và không evaluate test. Kiểm tra
negative masks/NaN/length/dimension, live launcher upgrade, reporting-v1 regressions
đều pass. Notebook/recovery syntax, nbformat, pyflakes, payload hashes, package CRC,
queues 8/30/22 và frozen source manifest đều đã xác minh. Chưa chạy campaign GPU thật
trong quá trình sửa; không suy diễn accuracy từ fixtures.

Các phần bên dưới ghi lại lần sửa reporting v1 trước đó.

Bug thực tế: configs V2 dùng training.preload=device. DeviceDataLoader không có sampler metadata, nên evaluate_test_set không ghi predictions dù loader shuffle=False và metadata/membership đúng. collect từ chối val_report.json sau khi ba seeds train xong. Các tests campaign trước đây tạo report fixtures trực tiếp, chưa bao phủ đường xuất report qua device loader.

report_predictions_v1.py bổ sung sampler metadata theo đúng thứ tự iterator hiện có và bọc training subprocess để exporter ghi audit. Shuffle/drop_last không được bypass. Không thay đổi weights, thứ tự batches, config, features, split, checkpoint selection hoặc metrics. Module và notebook launchers nằm ngoài phạm vi source_manifest; Git checkout của campaign tiếp tục pin tag ablation-v2-notebooks-v1 và khớp Shared 77a84a7b12f1. Reporting addon có ID/SHA256 riêng trong REPORTING_HOTFIX.json và từng val_report.json để tái lập đầy đủ.

build_launchers.py tạo notebooks bàn giao dưới artifacts/ablation-study-v2/reporting-hotfix-v1. Chúng nhúng nguyên addon, tự tìm Shared Input và cài wrapper sau preflight data/shared, trước training queue. Notebook nguồn trong notebooks/ và configs campaign được giữ nguyên.

Kiểm tra tự tạo:

```powershell
.venv/Scripts/python.exe -m pytest reports/ablation-study-v2/hotfixes/tests/test_report_predictions_v1.py -q
.venv/Scripts/python.exe -m pyflakes reports/ablation-study-v2/hotfixes
```

Đã chạy 5 checks ban đầu và một check subprocess thật sau đó: tổng 6 checks pass. Notebook syntax/nbformat/pyflakes và payload/hash ZIP đều đã kiểm tra. Đây là xác minh reporting compatibility, chưa phải validation campaign trên GPU/Kaggle thật.

Member A đã xác nhận runtime reset, Output trống và chưa lưu/tải result ZIP, nên không có checkpoints của F1 để sửa report tại chỗ. A launcher mặc định RUN_IDS=["G0__F1"] để chạy lại đúng config với ba seeds. Sau khi nghiệm thu/tải ZIP đầu tiên, RUN_IDS=None tiếp tục đủ phần A. Shared ZIP đã lưu trên máy vẫn hợp lệ và không cần prepare lại.
