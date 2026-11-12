# Member C: ablation V2

Branch: `codex/ablation-member-c`.

22 tổ hợp, 66 lượt training; mỗi YAML chạy đủ seeds42/43/44.

Dùng shared package đã freeze. Chạy các configs tuần tự trên runtime của mình.

| Run ID | D | Config |
|---|---:|---|
| G0__F0 | 160 | [G0__F0.yaml](G0__F0.yaml) |
| G0__F0-face | 196 | [G0__F0-face.yaml](G0__F0-face.yaml) |
| L0__F0 | 160 | [L0__F0.yaml](L0__F0.yaml) |
| L0__F0-face | 196 | [L0__F0-face.yaml](L0__F0-face.yaml) |
| L0__F1 | 162 | [L0__F1.yaml](L0__F1.yaml) |
| L0__F1-face | 198 | [L0__F1-face.yaml](L0__F1-face.yaml) |
| L0__F2 | 162 | [L0__F2.yaml](L0__F2.yaml) |
| L0__F2-face | 198 | [L0__F2-face.yaml](L0__F2-face.yaml) |
| L0__F3 | 192 | [L0__F3.yaml](L0__F3.yaml) |
| L0__F3-face | 228 | [L0__F3-face.yaml](L0__F3-face.yaml) |
| L0__F4 | 212 | [L0__F4.yaml](L0__F4.yaml) |
| L0__F4-face | 248 | [L0__F4-face.yaml](L0__F4-face.yaml) |
| T0__F0 | 160 | [T0__F0.yaml](T0__F0.yaml) |
| T0__F0-face | 196 | [T0__F0-face.yaml](T0__F0-face.yaml) |
| T0__F1 | 162 | [T0__F1.yaml](T0__F1.yaml) |
| T0__F1-face | 198 | [T0__F1-face.yaml](T0__F1-face.yaml) |
| T0__F2 | 162 | [T0__F2.yaml](T0__F2.yaml) |
| T0__F2-face | 198 | [T0__F2-face.yaml](T0__F2-face.yaml) |
| T0__F3 | 192 | [T0__F3.yaml](T0__F3.yaml) |
| T0__F3-face | 228 | [T0__F3-face.yaml](T0__F3-face.yaml) |
| T0__F4 | 212 | [T0__F4.yaml](T0__F4.yaml) |
| T0__F4-face | 248 | [T0__F4-face.yaml](T0__F4-face.yaml) |

```bash
python reports/ablation-study-v2/tools/study.py run-config configs/ablation-study-v2/member-c/G0__F0.yaml artifacts/ablation-study-v2/shared
```

Xem hướng dẫn môi trường, freeze và nghiệm thu ở [README chung](../README.md).
