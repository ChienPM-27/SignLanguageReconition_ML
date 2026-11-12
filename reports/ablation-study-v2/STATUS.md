# Source và validation V2

Ngày kiểm tra: 2026-11-12, Asia/Saigon.

Implementation đã có:

- Canonical `features.face`, optional 36-channel compact face; schema v5.
- Shared preprocessing Parquet/raw, cache identity, face validity sidecars.
- Face flip/rotation và ratio recomputation trên NumPy/PyTorch.
- Transformer factory, CLI training/evaluation và checkpoint reconstruction.
- Prediction/probabilities theo sequence/signer và model parameter counts.
- Ma trận 60 cells, config generator, common-cohort builder, source freeze,
  runner với runtime lock/cache reuse/log/ZIP và summarizer kiểm tra provenance.
- 60 canonical configs chia folder A/B/C, queues và hướng dẫn branch.
- `run-config` đọc YAML thực tế, kiểm tra protocol/source/cohort, giữ planned config
  và hash trong attempt/ZIP; Colab Drive project roots tách theo member.
- 3 notebook V2 Kaggle/Colab pin tag `ablation-v2-notebooks-v1`, common workflow
  prepare/shared/raw preflight, queue skip/retry/restore và cross-member aggregation.
- Result folder tách theo member/source/run/attempt; per-run ZIP và member ZIP
  giữ đủ artifacts, bỏ caches/raw/probes và không lặp lại ZIPs đã chứa cùng nội dung.

Kết quả kiểm tra source hiện tại:

```text
282 passed, 1 warning
```

Đã chạy toàn bộ `tests` khi import hook chủ động chặn optional clients `kaggle`
và `mediapipe`. Tests dùng fixtures tự tạo; backend MediaPipe được mock trong
test auxiliary-video. Warning duy nhất từ test AMP khi CUDA không có trên máy
local; không có test fail.

Pyflakes:

```bash
python -m pyflakes src scripts tests app.py reports/ablation-study/tools reports/ablation-study-v2/tools
```

Exit 0, không có findings. `git diff --check` exit 0.

Tests mới kiểm tra:

- Face translation/scale invariance, missing support, bounded interpolation.
- Face-off prefix regression, hash/dimension registry và sidecar ownership.
- Parquet/raw parity; CPU/tensor flip/rotation parity và ratio invariance.
- GRU/LSTM/Transformer backward, checkpoint roundtrip và optimizer/evaluation
  thật trên synthetic data; positional encoding phân biệt thứ tự thời gian.
- Đủ 60 cells/180 seeds, thay một yếu tố cho architecture variants.
- Common cohort, class/signer provenance, source drift, failed-stage lock cleanup.
- Summarizer từ chối test leakage, duplicate attempts, missing predictions;
  không chọn winner trước khi đủ matrix. Fixtures có trường hợp LSTM/F0 thắng
  để kiểm tra lựa chọn không mặc định ưu tiên F4/face/Transformer.
- CLI xuất matrix bằng UTF-8 ngay cả khi console được đặt cp1252.
- Config queues đủ 8/30/22 tổ hợp; runtime roots tách theo member.
- `run-config` từ chối drift feature/training/env.seed trước khi chạm runtime;
  propagate Face/Transformer/mirror config và bind frozen split; archive planned
  config được summarizer kiểm tra hash.
- Notebook nbformat/Python syntax và defaults cho A/B/C được kiểm tra; shared
  validation, raw mounted-data bypass, 5 preprocess probes giữ signer/vocab,
  completed skip, relocated restore, technical retry/history, safe ZIP paths và
  aggregate selected attempts đã kiểm tra bằng fixtures tự tạo.

Trạng thái training:

- **Chưa chạy training V2 trên dataset thật, GPU/Kaggle/Colab.**
- **Chưa tạo shared data package/common cohort thật.** Bước này thực hiện sau
  khi có năm cache probes và frozen signer split theo PLAN.md.
- **Chưa smoke MediaPipe video inference với video/model assets thật.** Feature
  parity đã kiểm tra trên raw landmarks; cần smoke trước khi chứng nhận demo.
- **Chưa chạy notebook thực trên Kaggle/Colab GPU.** Notebook/source/workflow
  được kiểm tra offline; cần preflight GPU baseline trước campaign dài.
- Source v5 và bộ configs A/B/C được đóng gói chung cho các branch
  `codex/ablation-member-a`, `codex/ablation-member-b`, `codex/ablation-member-c`.
  `study.py freeze` tạo content manifest để khóa phiên bản trước training;
  snapshot v4 cũ không phải source cho V2. Branch có code không có nghĩa là đã
  có shared package dữ liệu thật hoặc kết quả training.

Không có kết quả/winner V2 được suy diễn từ synthetic tests hoặc campaign cũ.
