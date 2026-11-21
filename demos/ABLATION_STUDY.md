# Bảng ablation V3 để chọn model camera

Nguồn: inventory full42 đã nghiệm thu, freeze `b4fd56f03a57`. Tất cả chỉ số bảng là validation mean qua seeds 42/43/44; ± là population std, đơn vị điểm phần trăm.

Ưu tiên `T0__HF` nếu nhìn rõ tay và mặt: đứng đầu validation. Thử `T0__H` khi muốn nhận diện chỉ từ tay, và `G-bi__HBF` để đối chiếu với BiGRU tốt nhất. Model B/F/BF ít chính xác hơn rõ rệt trong campaign này.

H = tay, B = thân trên, F = mặt. Các vùng không được chọn không tham gia fusion. Auto detect vẫn dùng tay để xác định đầu/cuối động tác.

## Config tốt nhất của từng kiến trúc

| Kiến trúc | Config | Macro-F1 (%) | Top-1 (%) | Top-3 (%) | Parameters |
|---|---|---:|---:|---:|---:|
| GRU mean/max | `G0__H` | 71.20 ± 0.77 | 72.12 | 93.05 | 228,372 |
| GRU mean | `G-mean__HF` | 67.51 ± 0.13 | 68.08 | 90.71 | 225,812 |
| GRU attention | `G-attn__HBF` | 71.54 ± 1.07 | 72.55 | 92.84 | 234,133 |
| BiGRU | `G-bi__HBF` | 74.54 ± 0.32 | 75.07 | 94.03 | 529,940 |
| LSTM | `L0__HB` | 71.53 ± 0.75 | 72.26 | 91.58 | 294,420 |
| Transformer | `T0__HF` | 76.52 ± 0.56 | 77.26 | 93.48 | 311,956 |

## Toàn bộ 42 configs

| Hạng | Config | Vùng | Macro-F1 (%) | Top-1 (%) | Top-3 (%) | Parameters |
|---:|---|---|---:|---:|---:|---:|
| 1 | `T0__HF` | hands + face | 76.52 ± 0.56 | 77.26 | 93.48 | 311,956 |
| 2 | `T0__HBF` | hands + body + face | 75.92 ± 0.96 | 76.86 | 94.40 | 311,956 |
| 3 | `T0__HB` | hands + body | 75.45 ± 1.11 | 76.45 | 94.23 | 311,956 |
| 4 | `G-bi__HBF` | hands + body + face | 74.54 ± 0.32 | 75.07 | 94.03 | 529,940 |
| 5 | `G-bi__HB` | hands + body | 74.42 ± 0.15 | 74.84 | 94.03 | 529,940 |
| 6 | `T0__H` | hands | 74.34 ± 1.99 | 75.38 | 93.36 | 311,956 |
| 7 | `G-bi__H` | hands | 74.13 ± 0.77 | 74.69 | 93.28 | 529,940 |
| 8 | `G-bi__HF` | hands + face | 74.04 ± 0.87 | 74.75 | 93.71 | 529,940 |
| 9 | `G-attn__HBF` | hands + body + face | 71.54 ± 1.07 | 72.55 | 92.84 | 234,133 |
| 10 | `L0__HB` | hands + body | 71.53 ± 0.75 | 72.26 | 91.58 | 294,420 |
| 11 | `G-attn__HB` | hands + body | 71.30 ± 0.59 | 72.18 | 93.13 | 234,133 |
| 12 | `L0__HBF` | hands + body + face | 71.25 ± 0.32 | 71.98 | 91.69 | 294,420 |
| 13 | `G-attn__H` | hands | 71.24 ± 0.60 | 72.26 | 93.16 | 234,133 |
| 14 | `G-attn__HF` | hands + face | 71.23 ± 0.46 | 72.29 | 92.87 | 234,133 |
| 15 | `G0__H` | hands | 71.20 ± 0.77 | 72.12 | 93.05 | 228,372 |
| 16 | `G0__HB` | hands + body | 71.20 ± 1.04 | 71.98 | 92.61 | 228,372 |
| 17 | `L0__H` | hands | 70.80 ± 0.68 | 71.43 | 92.58 | 294,420 |
| 18 | `G0__HF` | hands + face | 70.72 ± 0.87 | 71.63 | 92.76 | 228,372 |
| 19 | `G0__HBF` | hands + body + face | 70.68 ± 0.73 | 71.43 | 92.73 | 228,372 |
| 20 | `L0__HF` | hands + face | 70.32 ± 2.19 | 70.91 | 92.76 | 294,420 |
| 21 | `G-mean__HF` | hands + face | 67.51 ± 0.13 | 68.08 | 90.71 | 225,812 |
| 22 | `G-mean__HBF` | hands + body + face | 67.47 ± 0.94 | 67.85 | 90.33 | 225,812 |
| 23 | `G-mean__H` | hands | 67.43 ± 0.68 | 68.17 | 90.68 | 225,812 |
| 24 | `G-mean__HB` | hands + body | 67.27 ± 1.18 | 67.97 | 90.36 | 225,812 |
| 25 | `T0__BF` | body + face | 23.16 ± 1.34 | 25.11 | 51.23 | 311,956 |
| 26 | `T0__F` | face | 22.59 ± 1.54 | 23.49 | 47.33 | 311,956 |
| 27 | `G-bi__BF` | body + face | 22.34 ± 1.57 | 23.20 | 49.61 | 529,940 |
| 28 | `G-bi__F` | face | 22.25 ± 2.21 | 22.74 | 45.77 | 529,940 |
| 29 | `G0__F` | face | 22.13 ± 0.52 | 22.66 | 45.37 | 228,372 |
| 30 | `G-attn__BF` | body + face | 22.07 ± 2.38 | 23.18 | 49.90 | 234,133 |
| 31 | `G0__BF` | body + face | 20.32 ± 2.23 | 21.36 | 47.24 | 228,372 |
| 32 | `L0__BF` | body + face | 20.09 ± 1.21 | 21.21 | 45.77 | 294,420 |
| 33 | `G-attn__F` | face | 19.93 ± 1.09 | 20.43 | 42.91 | 234,133 |
| 34 | `G-mean__BF` | body + face | 19.60 ± 2.69 | 20.92 | 46.06 | 225,812 |
| 35 | `G-mean__F` | face | 18.90 ± 0.79 | 19.42 | 40.15 | 225,812 |
| 36 | `L0__F` | face | 18.31 ± 0.87 | 18.76 | 40.95 | 294,420 |
| 37 | `G-bi__B` | body | 9.82 ± 0.81 | 11.26 | 31.43 | 529,940 |
| 38 | `G0__B` | body | 9.28 ± 0.28 | 11.28 | 29.87 | 228,372 |
| 39 | `G-mean__B` | body | 8.84 ± 0.83 | 11.49 | 29.90 | 225,812 |
| 40 | `G-attn__B` | body | 8.80 ± 0.72 | 10.97 | 29.26 | 234,133 |
| 41 | `L0__B` | body | 7.63 ± 1.14 | 10.77 | 28.22 | 294,420 |
| 42 | `T0__B` | body | 6.22 ± 0.71 | 10.19 | 27.10 | 311,956 |

## Cách đọc

Ranking dựa trên validation theo rule đã freeze: mean Macro-F1, std thấp hơn, parameters ít hơn, run ID. Số parameters giúp hiểu kích thước model; bảng chưa đo latency camera.

Winner `T0__HF` đã được test riêng: Macro-F1 87,30% ± 0,77, Top-1 88,20% ± 1,07, Top-3 98,34% ± 0,52 qua ba seeds. Các configs khác chưa được so sánh trên test cohort.

Điểm softmax trên camera và validation accuracy là hai phép đo khác nhau. Hãy ghi một clip rồi chạy lại cùng clip ở các model để so sánh hành vi thực tế.
