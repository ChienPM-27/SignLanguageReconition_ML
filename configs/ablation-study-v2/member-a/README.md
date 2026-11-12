# Member A: ablation V2

Branch: `codex/ablation-member-a`.

8 tổ hợp, 24 lượt training; mỗi YAML chạy đủ seeds42/43/44.

Dùng shared package đã freeze. Chạy các configs tuần tự trên runtime của mình.

| Run ID | D | Config |
|---|---:|---|
| G0__F1 | 162 | [G0__F1.yaml](G0__F1.yaml) |
| G0__F1-face | 198 | [G0__F1-face.yaml](G0__F1-face.yaml) |
| G0__F2 | 162 | [G0__F2.yaml](G0__F2.yaml) |
| G0__F2-face | 198 | [G0__F2-face.yaml](G0__F2-face.yaml) |
| G0__F3 | 192 | [G0__F3.yaml](G0__F3.yaml) |
| G0__F3-face | 228 | [G0__F3-face.yaml](G0__F3-face.yaml) |
| G0__F4 | 212 | [G0__F4.yaml](G0__F4.yaml) |
| G0__F4-face | 248 | [G0__F4-face.yaml](G0__F4-face.yaml) |

```bash
python reports/ablation-study-v2/tools/study.py run-config configs/ablation-study-v2/member-a/G0__F1.yaml artifacts/ablation-study-v2/shared
```

Xem hướng dẫn môi trường, freeze và nghiệm thu ở [README chung](../README.md).
