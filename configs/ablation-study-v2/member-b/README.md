# Member B: ablation V2

Branch: `codex/ablation-member-b`.

30 tổ hợp, 90 lượt training; mỗi YAML chạy đủ seeds42/43/44.

Dùng shared package đã freeze. Chạy các configs tuần tự trên runtime của mình.

| Run ID | D | Config |
|---|---:|---|
| G-mean__F0 | 160 | [G-mean__F0.yaml](G-mean__F0.yaml) |
| G-mean__F0-face | 196 | [G-mean__F0-face.yaml](G-mean__F0-face.yaml) |
| G-mean__F1 | 162 | [G-mean__F1.yaml](G-mean__F1.yaml) |
| G-mean__F1-face | 198 | [G-mean__F1-face.yaml](G-mean__F1-face.yaml) |
| G-mean__F2 | 162 | [G-mean__F2.yaml](G-mean__F2.yaml) |
| G-mean__F2-face | 198 | [G-mean__F2-face.yaml](G-mean__F2-face.yaml) |
| G-mean__F3 | 192 | [G-mean__F3.yaml](G-mean__F3.yaml) |
| G-mean__F3-face | 228 | [G-mean__F3-face.yaml](G-mean__F3-face.yaml) |
| G-mean__F4 | 212 | [G-mean__F4.yaml](G-mean__F4.yaml) |
| G-mean__F4-face | 248 | [G-mean__F4-face.yaml](G-mean__F4-face.yaml) |
| G-attn__F0 | 160 | [G-attn__F0.yaml](G-attn__F0.yaml) |
| G-attn__F0-face | 196 | [G-attn__F0-face.yaml](G-attn__F0-face.yaml) |
| G-attn__F1 | 162 | [G-attn__F1.yaml](G-attn__F1.yaml) |
| G-attn__F1-face | 198 | [G-attn__F1-face.yaml](G-attn__F1-face.yaml) |
| G-attn__F2 | 162 | [G-attn__F2.yaml](G-attn__F2.yaml) |
| G-attn__F2-face | 198 | [G-attn__F2-face.yaml](G-attn__F2-face.yaml) |
| G-attn__F3 | 192 | [G-attn__F3.yaml](G-attn__F3.yaml) |
| G-attn__F3-face | 228 | [G-attn__F3-face.yaml](G-attn__F3-face.yaml) |
| G-attn__F4 | 212 | [G-attn__F4.yaml](G-attn__F4.yaml) |
| G-attn__F4-face | 248 | [G-attn__F4-face.yaml](G-attn__F4-face.yaml) |
| G-bi__F0 | 160 | [G-bi__F0.yaml](G-bi__F0.yaml) |
| G-bi__F0-face | 196 | [G-bi__F0-face.yaml](G-bi__F0-face.yaml) |
| G-bi__F1 | 162 | [G-bi__F1.yaml](G-bi__F1.yaml) |
| G-bi__F1-face | 198 | [G-bi__F1-face.yaml](G-bi__F1-face.yaml) |
| G-bi__F2 | 162 | [G-bi__F2.yaml](G-bi__F2.yaml) |
| G-bi__F2-face | 198 | [G-bi__F2-face.yaml](G-bi__F2-face.yaml) |
| G-bi__F3 | 192 | [G-bi__F3.yaml](G-bi__F3.yaml) |
| G-bi__F3-face | 228 | [G-bi__F3-face.yaml](G-bi__F3-face.yaml) |
| G-bi__F4 | 212 | [G-bi__F4.yaml](G-bi__F4.yaml) |
| G-bi__F4-face | 248 | [G-bi__F4-face.yaml](G-bi__F4-face.yaml) |

```bash
python reports/ablation-study-v2/tools/study.py run-config configs/ablation-study-v2/member-b/G-mean__F0.yaml artifacts/ablation-study-v2/shared
```

Xem hướng dẫn môi trường, freeze và nghiệm thu ở [README chung](../README.md).
