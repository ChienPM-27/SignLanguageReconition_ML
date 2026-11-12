# Config training và branch A/B/C

Source chung đã có compact Face (+36 channels), schema v5 và Transformer encoder.
Ba branch cùng chứa source và cả 60 configs để source manifest giống nhau khi freeze.
Mỗi member chạy folder của mình; không thay source/config trong campaign đã freeze.

| Member | Branch | Folder | Tổ hợp | Lượt training |
|---|---|---|---:|---:|
| A | `codex/ablation-member-a` | [member-a](member-a/README.md) | 8 | 24 |
| B | `codex/ablation-member-b` | [member-b](member-b/README.md) | 30 | 90 |
| C | `codex/ablation-member-c` | [member-c](member-c/README.md) | 22 | 66 |

Mỗi YAML là một tổ hợp, chạy cả seeds42/43/44. `run.name` là run ID canonical.
Các configs là snapshot sinh từ `configs/islr.yaml` và ma trận `study.py`;
không thêm config aliases hay tự đổi hyperparameter giữa các members.
`QUEUE.json` liệt kê tất cả configs theo phân công; các tổ hợp độc lập.

## Trước khi training

1. A preprocess thử F0-F4 face off, lập common cohort giữ signer assignments.
2. A/B/C thống nhất config, source, split và vocabulary, rồi tạo đúng một shared package.
3. Copy nguyên shared package sang từng runtime. `study.py` verify source bằng nội dung file,
   không yêu cầu ba branch phải có cùng tên hoặc Git SHA.
4. C chạy G0/F0 và G0/F0-face để preflight; sau đó A/B/C chạy đồng thời trên runtime độc lập.

Không có common cohort/shared package dataset thật trong checkout này.
Không dùng shared package hoặc checkpoints schema v4 cho campaign v5.

## Chạy từ YAML

Chạy từ repo root với Python/runtime đã cài dependencies:

```bash
# Member A
python reports/ablation-study-v2/tools/study.py run-config configs/ablation-study-v2/member-a/G0__F1.yaml artifacts/ablation-study-v2/shared

# Member B
python reports/ablation-study-v2/tools/study.py run-config configs/ablation-study-v2/member-b/G-mean__F0.yaml artifacts/ablation-study-v2/shared

# Member C: GRU baseline, LSTM, Transformer
python reports/ablation-study-v2/tools/study.py run-config configs/ablation-study-v2/member-c/G0__F0.yaml artifacts/ablation-study-v2/shared
python reports/ablation-study-v2/tools/study.py run-config configs/ablation-study-v2/member-c/L0__F0.yaml artifacts/ablation-study-v2/shared
python reports/ablation-study-v2/tools/study.py run-config configs/ablation-study-v2/member-c/T0__F0-face.yaml artifacts/ablation-study-v2/shared
```

Runner đọc YAML thực tế, kiểm tra run ID/model/feature và protocol so với shared,
rồi tự chạy preprocess → split → train, bind common split và lưu effective config.
Không chạy trực tiếp `20_train.py` với các YAML này trước khi bind shared cohort.
Mỗi runtime chạy tuần tự; mỗi tổ hợp tạo một attempt và ZIP riêng.

## Runtime riêng và Colab Drive

Branch không tự tạo runtime/GPU riêng. Mỗi member dùng một máy hoặc notebook riêng.
Nếu notebooks cùng dùng một Drive, `env.project_dir_name` trong các YAML đã khác nhau:
`asl-signs-v2-member-a`, `asl-signs-v2-member-b`, `asl-signs-v2-member-c`.
Muốn lưu lên Drive, copy YAML ra thư mục ngoài source đã freeze và đặt canonical
`env.use_drive: true` trong bản copy, rồi đưa đường dẫn bản copy vào `run-config`.
`data.local_mirror_dir` có thể trỏ tới mirror thực trên runtime đó.
Không sửa các configs nằm trong repo sau khi freeze, vì chúng thuộc source manifest.
Các thay đổi protocol/features/model/selection/data sampling bị runner từ chối;
các runtime paths được xử lý bằng canonical config keys.

## Nghiệm thu

Giữ nguyên ZIP do runner tạo: metadata `COMPLETED`, logs ba stages, config/source
manifest, cache audit, frozen split/vocabulary, checkpoints/history/train metrics
và validation predictions cho đủ ba seeds. ZIP của attempt thất bại không được
coi là kết quả hoàn tất. Tập hợp đúng một selected attempt/cell; retry chỉ cho lỗi kỹ thuật.

Chỉ chốt winner khi đủ 60 tổ hợp hợp lệ, theo mean Validation Macro-F1 rồi std,
parameter count và run ID. Test chỉ mở sau khi khóa winner.
Xem [PLAN.md](../../reports/ablation-study-v2/PLAN.md) và tools/summarize.py.
