# Notebook training V2 cho A/B/C

| Member | Notebook | Training | Config queue |
|---|---|---:|---|
| A | [ablation_v2_member_a.ipynb](ablation_v2_member_a.ipynb) | 8 tổ hợp / 24 lượt | `configs/ablation-study-v2/member-a/QUEUE.json` |
| B | [ablation_v2_member_b.ipynb](ablation_v2_member_b.ipynb) | 30 / 90 | `configs/ablation-study-v2/member-b/QUEUE.json` |
| C | [ablation_v2_member_c.ipynb](ablation_v2_member_c.ipynb) | 22 / 66 | `configs/ablation-study-v2/member-c/QUEUE.json` |

Các notebook V2 pin source bằng tag `ablation-v2-notebooks-v1` và dùng YAML canonical
cùng runner V2. Notebook V4 cũ vẫn được giữ riêng. Việc thêm workflow notebook tạo
snapshot source mới; shared package cho campaign này phải freeze bằng snapshot mới.

## Chạy nhanh

1. **Kaggle:** import notebook, bật Internet + GPU, Add Input competition `asl-signs` đầy đủ.
   **Colab:** mở notebook, chọn GPU; session tự mount Drive để giữ kết quả.
2. **A chuẩn bị một lần:** `MODE="prepare"`, đặt `BASE_SPLIT_INPUT` tới frozen
   `split_manifest.json` hoặc ZIP baseline cũ. Notebook lấy đúng vocabulary/label_map
   từ manifest, preprocess F0-F4 face off, intersection cohort và freeze shared ZIP.
   Không tạo split mới hoặc selection mới. Cell prepare không train model.
3. Giao `ablation_v2_shared_<code-id>.zip` cho A/B/C. Kaggle: upload thành input dataset
   rồi attach cho cả ba; Colab: copy ZIP vào Drive và đặt `SHARED_INPUT`.
4. **Training:** cả ba dùng `MODE="train"`, Run All trên runtime riêng.
   Giữ `RUN_IDS=None` để chạy cả queue, hoặc chọn vài run IDs để chia theo phiên GPU.
   C nên chạy `RUN_IDS=["G0__F0", "G0__F0-face"]` trước để preflight trên GPU thật.
5. Lưu/tải member results ZIP. Kaggle cần Save Version/Output sau mỗi phiên;
   Colab có kết quả trong Drive. B/C không cần đợi A hoàn tất training.

Notebook kiểm tra source, config, vocabulary, signer assignment, cohort và raw
data trước khi bắt đầu queue. Nếu dataset không đủ, nó dừng hoặc dùng credentials
đã có để download; không im lặng train trên cohort nhỏ hơn.
Nếu repo GitHub private, cấu hình `GH_TOKEN` trong Secrets/environment.
Kaggle API dùng `KAGGLE_API_TOKEN` hoặc `KAGGLE_USERNAME`/`KAGGLE_KEY` theo helper canonical.
Không ghi tokens trực tiếp vào notebook hoặc gửi tokens trong gói kết quả.

## Config và runtime

Không cần gõ tên model, Face toggle, seeds, batch size hoặc learning rate ở notebook.
Các giá trị đã nằm trong 60 YAMLs và được đối chiếu với shared config.
`RUNTIME_OVERRIDES` chỉ nhận canonical keys cho Drive/local root/mirror/logging;
ví dụ `{"data.local_mirror_dir": "/content/drive/MyDrive/asl-signs-data"}`.
Bản runtime YAML được lưu ngoài source trong `runtime_configs/`.

## Thư mục kết quả riêng

Mỗi member có folder riêng dưới persistent root (Kaggle: `/kaggle/working`;
Colab: `/content/drive/MyDrive/asl-signs-v2-member-<a|b|c>`):

```text
ablation_v2_results/
  member-a/<code-id>/
    shared/                         # config/source/split/vocabulary đã freeze
    runtime_configs/                # YAML thực tế dùng trên runtime
    queue_status.json               # selected attempts và lỗi kỹ thuật
    MEMBER_STATUS.json              # tiến độ riêng: 8/8, 30/30 hoặc 22/22
    MEMBER_RESULTS.md/.json         # metrics validation; chưa chọn winner toàn ma trận
    runs/<run-id>/<attempt>/
      inputs/                       # shared + planned config và hashes
      logs/                         # preprocess, split, train
      cache_audit/
      split_snapshot/
      runs/<run-id>/seed_42/         # checkpoint, history, metrics, val predictions
      runs/<run-id>/seed_43/
      runs/<run-id>/seed_44/
      metadata.json
      effective_config.yaml
    failed_attempts/                # giữ lịch sử/logs retry kỹ thuật
    bundles/                        # ZIP riêng cho từng tổ hợp
  member-a/member_a_<code-id>_results.zip
  member-a/ablation_v2_shared_<code-id>.zip
  member-b/...
  member-c/...
```

`ablation_v2_cache/` và caches chuẩn bị cohort nằm ngoài gói nghiệm thu. ZIP member
chứa đầy đủ result artifacts, không kèm raw data/cache lớn hoặc lặp lại per-run ZIPs.
Kết quả được ghi sau mỗi tổ hợp; chỉ `COMPLETED` và qua artifact validation mới được
bỏ qua khi chạy lại. Failed attempts vẫn giữ trong folder/ZIP.

## Chạy tiếp và nghiệm thu

- Colab còn nguyên Drive: chạy lại notebook sẽ đọc queue state và kiểm tra từng
  selected attempt trước khi bỏ qua. Đây là tiếp tục queue, không resume checkpoint cũ.
- Kaggle/runtime mới: attach member results ZIP trước, đặt `RESTORE_INPUT` rồi Run All.
  Restore vào folder trống, giữ config/hashes và rebind các selected paths theo runtime mới.
- Khi lỗi: đọc logs; chỉ bật `RETRY_FAILED=True` sau khi xác nhận lỗi kỹ thuật.
  Không chọn seed/attempt có điểm cao hơn. Không tự xóa runtime lock nếu process còn chạy.
- B/C gửi nguyên member results ZIP. A đặt `AGGREGATE_INPUTS` tới ZIP của B/C;
  notebook đọc đúng selected attempts và gọi summarizer để kiểm chứng chéo.
  Ma trận thiếu cells giữ PENDING; đủ 60 cells × 3 seeds mới chọn winner bằng validation.
- Notebook không mở test set để chọn model. Test cuối thực hiện sau khi khóa winner.

Kiểm tra notebook trong repository gồm nbformat/syntax, shared/cohort guards,
queue skip/restore/retry, archive isolation và aggregation trên fixtures tự tạo.
Chưa có smoke trên GPU/Kaggle/Colab hoặc training dataset thật trong task này.
