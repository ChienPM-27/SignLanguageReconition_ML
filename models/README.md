# Model assets cho demo

Các file `.pt`, `.task` và `.wasm` được quản lý bằng Git LFS. Chạy `git lfs install` và `git lfs pull` sau khi clone/pull `main`; `git lfs fsck` kiểm tra object LFS local.

- **126 checkpoints V3**: `artifacts/ablation-study-v3/deployment/b4fd56f03a57/`, gồm 42 configs × seeds 42/43/44. Mỗi config có `glosses.csv`; inventory nằm tại `artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/7f86f6ac76de/ALL_MEMBERS_FULL42_RESULTS.json`.
- **16 checkpoints cũ**: `outputs/Kaggle_test/`, giữ đường dẫn và metadata cho các entry points GRU cũ.
- **3 MediaPipe task models**: `hand_landmarker.task`, `face_landmarker.task`, `pose_landmarker_lite.task`.
- **MediaPipe Web runtime 0.10.35**: `mediapipe-web/0.10.35/`; `MANIFEST.json` trong thư mục runtime ghi SHA256 các assets.

[MANIFEST.json](MANIFEST.json) liệt kê đường dẫn, kích thước và SHA256 của toàn bộ 145 model files. Tất cả 165 checkpoint entries trong các ZIP V3 local đều có cùng SHA256 với checkpoints đã đưa lên; ZIP training và các bản checkpoint trùng không cần tải để chạy demo.

Xem [hướng dẫn demo](../demos/README.md) để cài dependencies và chọn model/seed. Demo kiểm checkpoint và label map theo SHA256 của inventory nghiệm thu trước khi sử dụng.
