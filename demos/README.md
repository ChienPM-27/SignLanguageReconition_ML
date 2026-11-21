# Demo camera cho ablation V3

## Cài trên máy của thành viên

Cài Git LFS trước khi clone/pull `main`. Dùng Python 3.13 trên Windows cho cấu hình CUDA được pin; backend CPU dùng Python 3.11 trở lên.

```powershell
git lfs install
git clone --branch main https://github.com/ChienPM-27/SignLanguageReconition_ML.git
cd SignLanguageReconition_ML
git lfs pull
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-inference.txt
.\.venv\Scripts\python.exe -X utf8 demos/camera_v3.py --list-models
.\demos\RUN_CAMERA.cmd
```

Nếu đã có repo, chuyển sang `main`, chạy `git pull --ff-only` rồi `git lfs pull`. Bản `main` chứa đủ 126 checkpoints V3 và label maps trong `artifacts/ablation-study-v3/deployment/b4fd56f03a57/`, inventory nghiệm thu, 16 checkpoints cũ trong `outputs/Kaggle_test/`, 3 `.task` models và Web runtime trong `models/`. Các file binary được quản lý bằng Git LFS. Chạy `git lfs fsck` để kiểm file LFS sau khi tải.

Máy Windows cần Chrome hoặc Edge đã cài để chạy landmarks GPU mặc định. Chạy `.\demos\RUN_CAMERA.cmd --set camera.landmark_backend=cpu` nếu muốn dùng CPU. Dataset, cache và ZIP training không cần cho demo với các checkpoints đã có.

## Sử dụng camera

Mở `demos/RUN_CAMERA.cmd`. Mặc định **Auto detect** và **Hiện landmarks** đều bật: đưa tay vào hình, thực hiện một ký hiệu ASL, rồi dừng hoặc hạ tay để tự nhận diện.

- Có đủ 42 configs: `G0`, `G-mean`, `G-attn`, `G-bi`, `L0`, `T0` × `H`, `B`, `F`, `HB`, `HF`, `BF`, `HBF`; mỗi config có seeds `42`, `43`, `44`.
- H = tay, B = thân trên, F = mặt. Mặc định vẫn là `T0__HF`, seed `42`. Danh sách được xếp theo validation ranking đã nghiệm thu.
- `G0` = GRU mean/max pooling; `G-mean` = GRU mean pooling; `G-attn` = GRU attention pooling; `G-bi` = BiGRU; `L0` = LSTM; `T0` = Transformer.
- Sau khi nhận diện, đổi model/seed rồi bấm **Chạy lại đoạn vừa ghi** để so sánh trên cùng video. Lịch sử ghi rõ model và seed của từng kết quả.
- Tab **Bảng ablation · 42 models** hiển thị Validation Macro-F1 mean ± std, Top-1, Top-3 và parameters. Có lọc kiến trúc/vùng; chọn dòng rồi bấm **Dùng model được chọn** hoặc nhấp đúp. [Bảng đầy đủ để đọc riêng](ABLATION_STUDY.md).
- Auto detect tạm dừng khi xem bảng ablation; quay lại tab Camera để tiếp tục. Nếu đang ghi thủ công, kết thúc đoạn trước khi mở bảng.
- Landmarks được trích trực tiếp trên GPU: 21 điểm mỗi tay, 33 pose points và 468 face points. Preview vẽ skeleton tay, vai/khuỷu tay/cổ tay và 16 điểm mặt dùng trong feature layout; chỉ vẽ lên bản sao, video inference giữ pixels và chiều gốc.
- Auto detect bắt đầu khi thấy tay hoặc tay chuyển động, giữ đoạn trước khi bắt đầu, rồi kết thúc khi tay dừng/hạ tay. Có thời gian nghỉ giữa các lần nhận diện để tránh lặp ở tư thế đứng yên. Đây là tách ký hiệu theo chuyển động, không phải model nhận diện ngôn ngữ ký hiệu liên tục.
- Bấm **Ghi thủ công** hoặc **Space** để chuyển sang ghi bằng nút. Space lần nữa để kết thúc. Bật lại **Auto detect** để trở về tự động.
- Chỉ cho đổi model khi đã kết thúc ghi và inference. Khi nạp model lỗi, có thể chọn lại model/seed để thử lại.
- Đưa tay, mặt, vai và khuỷu tay vào khung hình. Pipeline giữ quality gate tay của training cho cả B/F; model có vùng face/body còn kiểm tra vùng đó có dữ liệu hợp lệ.

Chọn ngay từ terminal:

```powershell
.\demos\RUN_CAMERA.cmd --model G-bi__HBF --seed 43
.\.venv\Scripts\python.exe -X utf8 demos/camera_v3.py --list-models
.\.venv\Scripts\python.exe -X utf8 demos/camera_v3.py --model L0__H --seed 42 --video data/sample_videos/drink.mp4
.\demos\RUN_CAMERA.cmd --set camera.auto_detect=false
.\demos\RUN_CAMERA.cmd --set camera.end_hold_seconds=1.2 --set camera.landmark_width=480
```

Demo đọc inventory `ALL_MEMBERS_FULL42_RESULTS.json` của aggregate `b4fd56f03a57/7f86f6ac76de`, kiểm SHA256 checkpoint/label map đã nghiệm thu rồi dùng bản deployment được tải qua Git LFS. ZIP A/B/C local chỉ dùng để phục hồi checkpoint thiếu; khi đã có đủ file từ `main`, demo chạy mà không cần các ZIP này.

## GPU cho GTX 1050

Demo tự chọn CUDA khi có PyTorch CUDA hoạt động. Bản CUDA 11.8 cho Windows x64 / Python 3.13 được pin riêng để chạy trên GPU Pascal như GTX 1050:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-gtx1050.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Wheel cài đặt tạm đã được dọn. Lệnh cài trên tải wheel chính thức và kiểm SHA256 được pin trong `requirements-gpu-gtx1050.txt`.

Header và kết quả hiện **GPU / CPU** của model; dòng trạng thái landmarks hiện device riêng cho các detectors. Kết quả tách thời gian trích landmarks và forward model.

Đã kiểm trên GTX 1050 4 GB với PyTorch `2.7.1+cu118`: cả 126 checkpoints chạy CUDA. Đo `T0__HBF`, seed 42 trên cùng features của clip mẫu, median 30 lần sau warm-up: CPU **4,87 ms**, GPU **6,24 ms**; probabilities khớp với tolerance `1e-4`. Vì model nhỏ và batch chỉ có một clip, GPU ở phép đo này chưa nhanh hơn CPU. Report của bước chuyển classifier ở `artifacts/ablation-study-v3/deployment/GPU_VALIDATION.json`.

## Landmarks GPU trên Windows

Mặc định `camera.landmark_backend=webgl_gpu`: MediaPipe Web 0.10.35 dùng chính ba `.task` assets trong `models/`, chạy inference tay/mặt/pose qua WebGL/D3D11 trên GTX 1050. Runtime dùng Chrome/Edge chạy headless với profile riêng; chỉ tải assets/frame từ server loopback trong máy, không mở tab web trong phiên browser của bạn. Python MediaPipe 0.10.35 trên Windows báo `GPU processing is disabled in build flags`, nên backend này thay cho native Python detectors.

Một GPU worker dùng chung cho preview và clip inference; đổi model/seed không khởi tạo lại detectors. Frame được chuyển bằng PNG lossless trong RAM, theo nhóm `camera.landmark_batch_size`, rồi áp dụng `prepare_landmark_sequence` từ checkpoint như trước. Không giảm số frames, đổi handedness hay mirror video inference. GPU hardware được kiểm tra bằng renderer và các draw calls của cả ba tasks; software renderer bị từ chối.

Lần đầu có bước biên dịch shader/warm-up. Trên clip 53 frames đã kiểm, sau warm-up: CPU **4,53 s**, WebGL GPU **4,14 s** (hai lượt đo mỗi backend); lần đầu GPU **14,46 s**. Cả hai backend dự đoán `brown`, cùng feature shape `(32, 196)`; kiểm SHA256 pixels của đường truyền khớp hoàn toàn. Đây là phép đo một clip, không phải kết luận accuracy hay tốc độ cho mọi webcam. Report ở `artifacts/ablation-study-v3/deployment/LANDMARK_GPU_VALIDATION.json`.

```powershell
# Chuẩn bị Web runtime chính thức (tự kiểm npm integrity)
.\.venv\Scripts\python.exe -m demos.setup_gpu_landmarks

# GPU mặc định; có thể đặt executable Chrome/Edge riêng
.\demos\RUN_CAMERA.cmd --set camera.landmark_backend=webgl_gpu

# Backend CPU khi cần đối chiếu
.\demos\RUN_CAMERA.cmd --set camera.landmark_backend=cpu
```

Kiến trúc, preprocessing và region fusion lấy từ `resolved_config` của checkpoint; chọn seed là chọn một fit đã train. Các knobs demo nằm trong section `camera` của `configs/islr.yaml`, map tới `CameraConfig` ở `src/utils/config.py`; `--set` dùng canonical keys và fail với khóa lạ. Camera settings không thay đổi preprocessing identity của checkpoints cũ. Shared source/config đã đóng gói và các artifacts training của campaign giữ nguyên. Log camera ở `artifacts/ablation-study-v3/deployment/camera.log`.

Validation Macro-F1 hiển thị là mean của ba seeds trên dataset, còn điểm Top-3 là softmax của đoạn vừa ghi. Nhận diện camera không phải phép đo accuracy của ablation study; domain camera thực tế có thể khác dataset.
