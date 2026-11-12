# Ablation study — cả 3 thành viên cùng train

> Campaign training lại với face, architecture và GRU/LSTM/Transformer dùng
> [kế hoạch V2](../ablation-study-v2/PLAN.md). Nội dung bên dưới ghi campaign v4
> trước đó; V2 có source v5, baseline và ma trận riêng.

Ngày chuẩn bị: 2026-11-11 (+07:00).
Branch báo cáo: `codex/ablation-study`.
Trạng thái campaign: **A đã hoàn tất F0/F0-control (6/6 model); artifacts F1–F4 đang PENDING.**
Branch chứa source v4, phân công, tools lưu log và mẫu báo cáo.
Source đã freeze tại `f6cf69fd4d22fece9c70a66a1b9f1e3461eb2fc2`;
shared/config.yaml và shared/source_commit.txt đã có trong repo. Xem SOURCE_STATUS.md.

Có 6 cấu hình × seeds42/43/44 = 18 lần train model.
**Mỗi người phụ trách 2 cấu hình = 6 model. Cả 3 đều train.**

## Phân công

| Thành viên | Cấu hình | Số model | Báo cáo phụ trách |
| --- | --- | ---: | --- |
| Trainer A | F0 → F0-control | 6 | `runs/F0.md`, `runs/F0-control.md` |
| Trainer B | F1 → F2 | 6 | `runs/F1.md`, `runs/F2.md` |
| Trainer C | F3 → F4 | 6 | `runs/F3.md`, `runs/F4.md` |

A kiêm chuẩn bị shared source/config/split trước campaign; B/C kiểm tra lại.
Sau khi cả 3 nộp artifacts, một thành viên chạy tổng hợp bảng chung.
Không dành riêng một người chỉ tổng hợp. Điền tên thật vào report từng run.

Các cấu hình độc lập: F1/F2/F3/F4 không cần checkpoint của run trước.
Tất cả train từ initialization mới. Cả 3 chạy song song trên **3 notebook/máy
riêng** sau shared handoff; trên mỗi runtime chạy 2 cấu hình tuần tự.
Số model chia đều, nhưng thời gian có thể khác theo GPU và D.

## Bảng cấu hình

Các thay đổi cộng dồn; mọi augmentation khác giữ theo snapshot chung.

| Run | Người | Wrist reference | Palm scale | Tips | Bends | Wrist scale/translation | D |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| F0 | A | image | frame | false | false | bật | 160 |
| F0-control | A | image | frame | false | false | tắt | 160 |
| F1 | B | torso_xy | frame | false | false | tắt | 162 |
| F2 | B | torso_xy | clip_median | false | false | tắt | 162 |
| F3 | C | torso_xy | clip_median | true | false | tắt | 192 |
| F4 | C | torso_xy | clip_median | true | true | tắt | 212 |

Protocol: hands_pose, T32, GRU128×2, unidirectional, mean_max, dropout .3;
batch32, lr .001, weight_decay .0001, smoothing .1, epochs60, early-stopping
minimum5/patience15. Các thông số khác theo snapshot của configs/islr.yaml.
Không dùng T4 balanced preset. Feature campaign dùng validation; evaluate_test=false.

## Shared handoff trước khi train

- Cả 3 checkout `codex/ablation-study`; dùng source SHA trong
  `reports/ablation-study/shared/source_commit.txt`. main vẫn là schema v2.
- Source đã đạt 249 tests passed và pyflakes sạch; evidence ở SOURCE_STATUS.md.
- Copy snapshot canonical từ `reports/ablation-study/shared/config.yaml`
  và source_commit.txt vào runtime shared package.
  Ordered 20-gloss vocabulary và label_map phải giống nhau cho cả 3.
- Preprocess F0 bằng source v4; tạo signer split đúng một lần với
  `scripts/10_prepare_data.py --stage split`, `split.manifest_path=null`.
  Giao generated `data/splits/split_manifest.json` cho B/C.
- Shared package gồm `config.yaml`, `source_commit.txt` (full Git SHA 40 ký tự),
  `split_manifest.json`, `selection/glosses.csv`.
- Mọi run rebind shared split. Ghi actual class/signer/sample counts,
  selection provenance và exclusions; không đổi signer hoặc tìm split seed mới.
- Cả 3 checkout source đã freeze. Report commits có thể khác nhau nếu
  src/scripts/configs/requirements.txt vẫn khớp source commit.

## Chạy và lưu log tự động

Runner: `tools/run_ablation.py`. Chạy tại repo, bằng Python đã cài dependencies.
Trên Kaggle attach dataset asl-signs và đặt shared package ở đường dẫn thật:

```bash
SHARED=/kaggle/working/ablation_shared
mkdir -p "$SHARED"
cp reports/ablation-study/shared/config.yaml reports/ablation-study/shared/source_commit.txt "$SHARED/"
# A bổ sung selection/glosses.csv và split_manifest.json; B/C nhận cùng package.

# Trainer A, runtime A:
python reports/ablation-study/tools/run_ablation.py F0 "$SHARED"
python reports/ablation-study/tools/run_ablation.py F0-control "$SHARED"

# Trainer B, runtime B:
python reports/ablation-study/tools/run_ablation.py F1 "$SHARED"
python reports/ablation-study/tools/run_ablation.py F2 "$SHARED"

# Trainer C, runtime C:
python reports/ablation-study/tools/run_ablation.py F3 "$SHARED"
python reports/ablation-study/tools/run_ablation.py F4 "$SHARED"
```

Runner dùng canonical config snapshot + --set overrides; kiểm tra protocol,
source và D. Runtime/cache paths tuân env.resolve_paths.
Output dưới `persistent_root/ablation_runtime/<run>/<attempt>/`:
Kaggle là `/kaggle/working/ablation_runtime/`, Colab/local theo canonical paths.

**Mỗi attempt có thư mục riêng; không ghi đè log của lần trước.** Không chạy
hai cấu hình cùng lúc trên cùng runtime vì split CSV của source hiện tại ghi
vào `persistent_root/data/splits`.

Runner tự lưu:

- `logs/preflight.log`: source/protocol/hardware/library metadata.
- `logs/preprocess.log`, `logs/split.log`, `logs/train.log`: stdout + stderr,
  exact commands và exit codes; lỗi không bị che bởi việc redirect log.
- `metadata.json`: run/owner/status/start/end/source/shared hashes và từng stage.
- `inputs/`, `effective_config.yaml`, `cache_audit/`, `split_snapshot/`:
  config, frozen split, vocabulary, cache metadata, exclusions và membership.
- `runs/<timestamp>_ablation_<run>/`: multi_seed_summary.json, mỗi seed có
  best_model.pt, history.json, train_metrics.csv, val_report.json và plots.
  Effective config theo seed nằm trong checkpoint metadata.
- `RESULTS.md` + `RESULTS.json`: bảng cho run vừa xong, các run chưa nhận ghi PENDING.
- Bundle `<attempt>.zip` nằm cạnh thư mục attempt, gồm log/model/report/provenance.

Nếu stage lỗi, runner trả exit code khác 0, giữ FAILED metadata/failure.log và
cố gắng zip artifacts đã có. Nếu runtime bị dừng đột ngột, log được flush nhưng
có thể chưa có ZIP/table; giữ thư mục và ghi PARTIAL/RUNNING, không gọi COMPLETED.

**Kaggle: Save Version và giữ Output, hoặc tải ZIP lên artifact storage của team
ngay sau job.** File trong /kaggle/working chỉ trở thành handoff bền vững sau bước
đó. Cache lớn trong /kaggle/temp không nằm trong ZIP; cache audit đã được giữ.

## Bảng kết quả sau khi train

Cấu trúc lưu ZIP/model, bảng riêng A/B/C và bảng chung: xem [STORAGE.md](STORAGE.md).

Tải và giải nén ZIP của 6 cấu hình vào một root chung, ví dụ
`artifacts/ablation-study/`. Giữ nguyên cấu trúc từng attempt.
Công cụ tổng hợp chỉ cần Python standard library:

```bash
python reports/ablation-study/tools/summarize_results.py artifacts/ablation-study --output reports/ablation-study/RESULTS.md
```

Có thể truyền nhiều input directories. Nếu một cấu hình có nhiều attempts,
truyền rõ 6 attempt paths đã chọn; công cụ không tự lấy latest/best seed.

Bảng tự tạo có Run, Owner, D, Seeds, Validation Top-1/Top-3/Macro-F1/loss
**mean ± std**, tổng training time và status; bảng Δ Macro-F1 cho từng feature step.
JSON bên cạnh giữ per-seed metrics/provenance/artifact paths.

Chỉ COMPLETED khi đủ 3 seeds, logs và artifacts. Công cụ từ chối thiếu artifacts,
NaN, duplicate seeds, test_report.json hoặc source/config/split/vocabulary bị trộn.
Nếu validation cohort khác nhau, bảng full-cohort có cảnh báo; team bổ sung
matched-cohort comparison trước kết luận. So F1 với F0-control.
Không mặc định F4 thắng; primary selection là mean Validation Macro-F1.
Best checkpoint mỗi seed vẫn chọn bằng validation loss.

Mỗi người điền thêm per-gloss/signer, counts, exclusions, limitations và links
ZIP trong report được phân công. REPORT.md lưu review/kết luận; RESULTS.md là
bảng đo tự động. Không điền số kỳ vọng thay số thực nghiệm.

## Workflow báo cáo Git

Sau khi branch chung được publish, mỗi người tạo branch riêng từ nó:

```bash
git fetch origin
git switch -c codex/ablation-trainer-a origin/codex/ablation-study
# B: codex/ablation-trainer-b; C: codex/ablation-trainer-c.
```

Mỗi người chỉ sửa run reports của mình. Tổng hợp results và report về
codex/ablation-study qua review. Reports hiện gitignored: thêm file mới bằng
`git add -f reports/ablation-study/<file>`.
Checkpoint/cache/ZIP lớn lưu artifact storage; Git lưu Markdown và result JSON
cần thiết. Source SHA và config/split hashes phải khớp giữa cả 3.

Architecture/T/label-smoothing ablations và final test là bước sau feature
campaign. TCN chưa tích hợp; không thuộc 6 jobs này.
