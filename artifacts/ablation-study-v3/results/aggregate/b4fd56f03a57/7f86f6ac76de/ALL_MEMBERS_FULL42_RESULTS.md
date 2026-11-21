# Kết quả ablation V3 đủ 42 configs

Đã nghiệm thu A 7/7, B 21/21, C 14/14 configs, đủ 126 fits với seeds 42/43/44.

Cấu hình được chọn theo Validation Macro-F1 là `T0__HF`, 76.52% ± 0.56 điểm phần trăm.

Freeze `b4fd56f03a57`, cùng 20 glosses và cohort 7.854 sequences. Train gồm 5.558 clips/15 signers, validation 1.155 clips/3 signers, test 1.141 clips/3 signers. Test evaluation còn chờ.

## Xếp hạng validation

| Hạng | Config | Macro-F1 mean ± std (%) | Top-1 (%) | Top-3 (%) | Parameters |
|---:|---|---:|---:|---:|---:|
| 1 | T0__HF | 76.52 ± 0.56 | 77.26 | 93.48 | 311956 |
| 2 | T0__HBF | 75.92 ± 0.96 | 76.86 | 94.40 | 311956 |
| 3 | T0__HB | 75.45 ± 1.11 | 76.45 | 94.23 | 311956 |
| 4 | G-bi__HBF | 74.54 ± 0.32 | 75.07 | 94.03 | 529940 |
| 5 | G-bi__HB | 74.42 ± 0.15 | 74.84 | 94.03 | 529940 |
| 6 | T0__H | 74.34 ± 1.99 | 75.38 | 93.36 | 311956 |
| 7 | G-bi__H | 74.13 ± 0.77 | 74.69 | 93.28 | 529940 |
| 8 | G-bi__HF | 74.04 ± 0.87 | 74.75 | 93.71 | 529940 |
| 9 | G-attn__HBF | 71.54 ± 1.07 | 72.55 | 92.84 | 234133 |
| 10 | L0__HB | 71.53 ± 0.75 | 72.26 | 91.58 | 294420 |
| 11 | G-attn__HB | 71.30 ± 0.59 | 72.18 | 93.13 | 234133 |
| 12 | L0__HBF | 71.25 ± 0.32 | 71.98 | 91.69 | 294420 |
| 13 | G-attn__H | 71.24 ± 0.60 | 72.26 | 93.16 | 234133 |
| 14 | G-attn__HF | 71.23 ± 0.46 | 72.29 | 92.87 | 234133 |
| 15 | G0__H | 71.20 ± 0.77 | 72.12 | 93.05 | 228372 |
| 16 | G0__HB | 71.20 ± 1.04 | 71.98 | 92.61 | 228372 |
| 17 | L0__H | 70.80 ± 0.68 | 71.43 | 92.58 | 294420 |
| 18 | G0__HF | 70.72 ± 0.87 | 71.63 | 92.76 | 228372 |
| 19 | G0__HBF | 70.68 ± 0.73 | 71.43 | 92.73 | 228372 |
| 20 | L0__HF | 70.32 ± 2.19 | 70.91 | 92.76 | 294420 |
| 21 | G-mean__HF | 67.51 ± 0.13 | 68.08 | 90.71 | 225812 |
| 22 | G-mean__HBF | 67.47 ± 0.94 | 67.85 | 90.33 | 225812 |
| 23 | G-mean__H | 67.43 ± 0.68 | 68.17 | 90.68 | 225812 |
| 24 | G-mean__HB | 67.27 ± 1.18 | 67.97 | 90.36 | 225812 |
| 25 | T0__BF | 23.16 ± 1.34 | 25.11 | 51.23 | 311956 |
| 26 | T0__F | 22.59 ± 1.54 | 23.49 | 47.33 | 311956 |
| 27 | G-bi__BF | 22.34 ± 1.57 | 23.20 | 49.61 | 529940 |
| 28 | G-bi__F | 22.25 ± 2.21 | 22.74 | 45.77 | 529940 |
| 29 | G0__F | 22.13 ± 0.52 | 22.66 | 45.37 | 228372 |
| 30 | G-attn__BF | 22.07 ± 2.38 | 23.18 | 49.90 | 234133 |
| 31 | G0__BF | 20.32 ± 2.23 | 21.36 | 47.24 | 228372 |
| 32 | L0__BF | 20.09 ± 1.21 | 21.21 | 45.77 | 294420 |
| 33 | G-attn__F | 19.93 ± 1.09 | 20.43 | 42.91 | 234133 |
| 34 | G-mean__BF | 19.60 ± 2.69 | 20.92 | 46.06 | 225812 |
| 35 | G-mean__F | 18.90 ± 0.79 | 19.42 | 40.15 | 225812 |
| 36 | L0__F | 18.31 ± 0.87 | 18.76 | 40.95 | 294420 |
| 37 | G-bi__B | 9.82 ± 0.81 | 11.26 | 31.43 | 529940 |
| 38 | G0__B | 9.28 ± 0.28 | 11.28 | 29.87 | 228372 |
| 39 | G-mean__B | 8.84 ± 0.83 | 11.49 | 29.90 | 225812 |
| 40 | G-attn__B | 8.80 ± 0.72 | 10.97 | 29.26 | 234133 |
| 41 | L0__B | 7.63 ± 1.14 | 10.77 | 28.22 | 294420 |
| 42 | T0__B | 6.22 ± 0.71 | 10.19 | 27.10 | 311956 |

## Ma trận đầy đủ

H = tay, B = thân trên, F = mặt. Mean Validation Macro-F1 (%).

| Profile | H | B | F | HB | HF | BF | HBF |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 71.20 | 9.28 | 22.13 | 71.20 | 70.72 | 20.32 | 70.68 |
| G-mean | 67.43 | 8.84 | 18.90 | 67.27 | 67.51 | 19.60 | 67.47 |
| G-attn | 71.24 | 8.80 | 19.93 | 71.30 | 71.23 | 22.07 | 71.54 |
| G-bi | 74.13 | 9.82 | 22.25 | 74.42 | 74.04 | 22.34 | 74.54 |
| L0 | 70.80 | 7.63 | 18.31 | 71.53 | 70.32 | 20.09 | 71.25 |
| T0 | 74.34 | 6.22 | 22.59 | 75.45 | 76.52 | 23.16 | 75.92 |

![Ma trận full42](full42_matrix.png)

## Config tốt nhất của từng profile

| Profile | Config | Macro-F1 mean ± std (%) | Seeds 42 / 43 / 44 (%) |
|---|---|---:|---|
| G0 | G0__H | 71.20 ± 0.77 | 70.85 / 70.48 / 72.26 |
| G-mean | G-mean__HF | 67.51 ± 0.13 | 67.50 / 67.36 / 67.67 |
| G-attn | G-attn__HBF | 71.54 ± 1.07 | 70.23 / 71.55 / 72.84 |
| G-bi | G-bi__HBF | 74.54 ± 0.32 | 74.52 / 74.94 / 74.15 |
| L0 | L0__HB | 71.53 ± 0.75 | 70.56 / 72.39 / 71.64 |
| T0 | T0__HF | 76.52 ± 0.56 | 76.17 / 76.08 / 77.30 |

## So architecture trên cùng vùng

Δ mean Macro-F1 so với G0, đơn vị điểm phần trăm. Ngoặc là số seeds tăng trên 3 seeds.

| Profile | H | HB | HF | HBF |
|---|---:|---:|---:|---:|
| G-mean | -3.77 (0/3) | -3.92 (0/3) | -3.21 (0/3) | -3.21 (0/3) |
| G-attn | +0.04 (2/3) | +0.10 (1/3) | +0.51 (2/3) | +0.86 (1/3) |
| G-bi | +2.94 (3/3) | +3.22 (3/3) | +3.31 (3/3) | +3.85 (3/3) |
| L0 | -0.39 (1/3) | +0.33 (1/3) | -0.41 (1/3) | +0.57 (2/3) |
| T0 | +3.15 (3/3) | +4.25 (3/3) | +5.79 (3/3) | +5.24 (3/3) |

## Thêm vùng vào H

Δ mean Macro-F1, điểm phần trăm, ngoặc là số seeds tăng trên 3 seeds.

| Profile | HB − H | HF − H | HBF − H |
|---|---:|---:|---:|
| G0 | +0.00 (1/3) | -0.47 (1/3) | -0.51 (1/3) |
| G-mean | -0.16 (1/3) | +0.08 (1/3) | +0.04 (2/3) |
| G-attn | +0.06 (1/3) | -0.01 (1/3) | +0.30 (1/3) |
| G-bi | +0.28 (2/3) | -0.10 (2/3) | +0.40 (2/3) |
| L0 | +0.73 (2/3) | -0.49 (2/3) | +0.45 (1/3) |
| T0 | +1.11 (2/3) | +2.17 (3/3) | +1.58 (1/3) |

## Ba hướng nhầm đã theo dõi

So sánh config tốt nhất của từng profile, đầu vào có thể khác nhau. Tỷ lệ là mean qua ba seeds, tính trên nhãn thật.

| Config | awake → wake (%) | mouse → shhh (%) | pretend → mouse (%) |
|---|---:|---:|---:|
| G0__H | 54.80 | 38.89 | 29.89 |
| G-mean__HF | 44.07 | 31.67 | 30.46 |
| G-attn__HBF | 63.28 | 38.33 | 29.89 |
| G-bi__HBF | 51.41 | 47.22 | 29.89 |
| L0__HB | 50.85 | 38.33 | 28.74 |
| T0__HF | 66.67 | 39.44 | 32.18 |

Số lỗi pretend → mouse theo signer, theo thứ tự seeds 42/43/44.

| Config | Signer 28656 | Signer 32319 | Signer 36257 |
|---|---|---|---|
| G0__H | 0/0/0 trên 18 | 17/18/17 trên 20 | 0/0/0 trên 20 |
| G-mean__HF | 0/0/0 trên 18 | 18/17/18 trên 20 | 0/0/0 trên 20 |
| G-attn__HBF | 0/0/0 trên 18 | 17/17/18 trên 20 | 0/0/0 trên 20 |
| G-bi__HBF | 0/0/0 trên 18 | 17/18/17 trên 20 | 0/0/0 trên 20 |
| L0__HB | 0/0/0 trên 18 | 16/17/17 trên 20 | 0/0/0 trên 20 |
| T0__HF | 0/0/0 trên 18 | 19/19/18 trên 20 | 0/0/0 trên 20 |

F1 từng gloss và recall của config được chọn.

| Gloss | Clips | Mean F1 (%) | Mean recall (%) |
|---|---:|---:|---:|
| awake | 59 | 25.03 | 21.47 |
| mouse | 60 | 41.46 | 36.67 |
| wake | 60 | 51.44 | 61.11 |
| pretend | 58 | 54.23 | 42.53 |
| listen | 57 | 61.46 | 66.67 |
| who | 60 | 69.99 | 66.67 |
| fireman | 57 | 77.77 | 73.10 |
| bye | 57 | 78.99 | 77.19 |
| drink | 59 | 79.16 | 83.62 |
| shhh | 59 | 81.43 | 96.61 |
| donkey | 63 | 83.47 | 78.84 |
| nuts | 57 | 86.40 | 87.13 |
| bird | 58 | 87.24 | 90.23 |
| doll | 59 | 88.00 | 97.18 |
| sleepy | 52 | 88.80 | 94.23 |
| brown | 57 | 91.79 | 94.74 |
| cow | 58 | 93.69 | 89.66 |
| duck | 56 | 95.94 | 98.21 |
| uncle | 57 | 96.84 | 97.66 |
| lips | 52 | 97.20 | 99.36 |

## Nghiệm thu và giới hạn

Cả 42 selected attempts pass canonical `study_v3.collect`, exact inventories, frozen source/config/split/cache identity, checkpoint hashes, successful stage logs, loss-selected histories và multi-seed summaries. Top-1/Top-3/Macro-F1 và confusion matrices được tính lại từ 126 bộ validation predictions.

Ba configs HBF B cũ được đối chiếu từng artifact với ZIP B vòng đầu và giữ nguyên. ZIP B mới còn lưu lịch sử attempt split thất bại trước hotfix, queue hiện tại có failed rỗng.

Rule xếp hạng dùng metrics làm tròn trong reports theo implementation đã freeze. Metrics chính xác từ predictions lưu riêng trong JSON. Mỗi seed dùng cùng validation cohort. Population std của ba seeds là thống kê mô tả.

Các so sánh kiến trúc có số parameters khác nhau. Ablation vùng đồng thời thay đổi thành phần và trọng số fusion sau chuẩn hóa. HF dùng 87,5% H / 12,5% F, HB dùng 77,78% H / 22,22% B, HBF dùng 70/20/10.

Nguyên nhân lỗi theo gloss/signer cần kiểm clip hoặc landmarks thực tế. Bước evaluation cuối dùng ba checkpoints đã chọn trên test cohort 1.141 clips.

[Metrics, hashes và audit JSON](ALL_MEMBERS_FULL42_RESULTS.json) · [Lựa chọn config và checkpoints](WINNER_SELECTION.json)
