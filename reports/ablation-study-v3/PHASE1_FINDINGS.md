# Phân tích kết quả A/B/C — 24 configs đã nghiệm thu

Transformer HF đang tốt nhất trong 24 configs đã chạy. BiGRU tốt nhất nhóm GRU trên cùng HBF. Tay là nguồn thông tin chính; face cải thiện rõ nhất ở Transformer khi ghép với tay, body có tác dụng nhỏ và phụ thuộc model.

Phạm vi hiện tại là 42 configs; còn18 configs GRU subsets đang chờ chạy. Chưa có global winner hoặc test metrics.

Dữ liệu so sánh: freeze b4fd56f03a57, cùng cohort 20 glosses/7.854 clips, train 5.558/15 signers, validation 1.155/3 signers, test 1.141/3 signers. A 7/B 3/C 14 =24 configs/72 fits, seeds42/43/44. Tất cả ZIP/source/config/split/inventory/checkpoint hashes và validation predictions đã nghiệm thu; lần phân tích này đọc lại archives để kiểm lỗi theo signer, không chạy inference hoặc test.

## Cấu hình dẫn đầu và tài nguyên đã quan sát

| Config | Macro-F1 mean ± std (%) | Top-1 (%) | Top-3 (%) | Parameters | Training phút/seed |
|---|---:|---:|---:|---:|---:|
| T0__HF | 76.52 ± 0.56 | 77.26 | 93.48 | 311956 | 3.05 |
| T0__HBF | 75.92 ± 0.96 | 76.86 | 94.40 | 311956 | 3.42 |
| T0__HB | 75.45 ± 1.11 | 76.45 | 94.23 | 311956 | 3.41 |
| G-bi__HBF | 74.54 ± 0.32 | 75.07 | 94.03 | 529940 | 3.90 |
| L0__HB | 71.53 ± 0.75 | 72.26 | 91.58 | 294420 | 3.64 |
| G0__H | 71.20 ± 0.77 | 72.12 | 93.05 | 228372 | 2.82 |

T0 HF thắng mean Macro-F1, còn T0 HBF có Top-3 và Validation Loss tốt hơn. Rule chọn config dùng Macro-F1 nên không đổi leader chỉ vì loss/Top-3. Parameters là tổng registered parameters, kể cả projectors của vùng bị tắt. Thời gian là training loop đã ghi trên Tesla T4, không gồm preprocessing và không phải latency/throughput benchmark.

## Member B: so architecture trên cùng HBF

| Profile | Macro-F1 (%) | Δ so G0 (điểm %) | Seeds tăng/3 |
|---|---:|---:|---:|
| G0 | 70.68 ± 0.73 | 0 | — |
| G-mean | 67.47 ± 0.94 | -3.21 | 0 |
| G-attn | 71.54 ± 1.07 | +0.86 | 1 |
| G-bi | 74.54 ± 0.32 | +3.85 | 3 |
| L0 | 71.25 ± 0.32 | +0.57 | 2 |
| T0 | 75.92 ± 0.96 | +5.24 | 3 |

BiGRU tăng 3,85 điểm Macro-F1 so GRU G0 trên HBF, tăng ở cả 3 seeds, std chỉ 0,32 điểm. G-attn tăng 0,86 điểm nhưng chỉ 1/3 seeds tăng; seed44 tăng 3,04 điểm, hai seeds còn lại giảm nhẹ. G-mean giảm 3,21 điểm và giảm cả 3 seeds. Mean pooling đơn thuần kém hơn mean_max trong thử nghiệm này; attention chưa vượt bidirectional. BiGRU có 529.940 parameters, khoảng 2,32 lần G0; cải thiện chưa tách riêng tác dụng bidirectional khỏi tăng capacity. T0 HBF vẫn hơn G-bi HBF1,39 điểm, tăng cả 3 seeds; đây là so sánh cùng feature subset, nhưng số parameters/kiến trúc vẫn khác.

## Member A/C: ablation theo vùng

H=tay, B=thân trên, F=mặt. Mean Validation Macro-F1 (%):

| Model | H | B | F | HB | HF | BF | HBF |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 71.20 | 9.28 | 22.13 | 71.20 | 70.72 | 20.32 | 70.68 |
| L0 | 70.80 | 7.63 | 18.31 | 71.53 | 70.32 | 20.09 | 71.25 |
| T0 | 74.34 | 6.22 | 22.59 | 75.45 | 76.52 | 23.16 | 75.92 |

| So với H-only | GRU G0 Δ điểm % | LSTM Δ điểm % | Transformer Δ điểm % |
|---|---:|---:|---:|
| Thêm body vào H | +0.00 | +0.73 | +1.11 |
| Thêm face vào H | -0.47 | -0.49 | +2.17 |
| Thêm body + face vào H | -0.51 | +0.45 | +1.58 |

A: H và HB có cùng mean Macro-F1 từ metrics làm tròn; HB biến động lớn hơn nên H được xếp trước. HF/HBF thấp hơn H khoảng 0,47/0,51 điểm. Thêm face vào HB làm G0 giảm 0,51 điểm ở cả 3 seeds.

C: LSTM HB tốt nhất nhóm LSTM 71,53%, nhưng chỉ hơn H0,73 điểm và không tăng nhất quán cả 3 seeds. Transformer HF tốt nhất 76,52%, hơn H2,17 điểm với per-seed gains[1,27;4,40;0,85] điểm: tăng cả 3 seeds. HBF thấp hơn HF0,59 điểm; giảm 2/3 seeds, do đó chưa kết luận body luôn có hại. Kết quả phản bác diễn giải rằng face luôn làm tốt hơn hoặc luôn làm xấu hơn: tác dụng phụ thuộc temporal model.

B-only/F-only/BF đều thấp hơn rõ rệt các subsets có tay. Tuy nhiên Top-1 B-only10,19–11,28% và F-only18,76–23,49% vẫn cao hơn majority-class baseline 5,45% trên validation này. Vùng phụ có tín hiệu nhưng chưa đủ phân biệt20 glosses khi dùng riêng.

Khi thêm/bỏ vùng, weights được renormalize: HF=87,5%H/12,5%F; HB=77,78%H/22,22%B; HBF=70/20/10. Do đó ablation đo toàn bộ cấu hình subset+fusion, không phải tác dụng thuần của một vùng giữ nguyên các weights khác. Chưa có sweep weights nên không chứng minh 70/20/10 tối ưu; ZIP không có raw feature arrays để kiểm định chất lượng landmarks trực tiếp.

## Lỗi theo gloss và signer

Các gloss yếu nhất ở T0 HF:

| Gloss | F1 mean (%) | Recall mean (%) | Validation clips |
|---|---:|---:|---:|
| awake | 25.03 | 21.47 | 59 |
| mouse | 41.46 | 36.67 | 60 |
| wake | 51.44 | 61.11 | 60 |
| pretend | 54.23 | 42.53 | 58 |
| listen | 61.46 | 66.67 | 57 |
| who | 69.99 | 66.67 | 60 |

Thêm face vào Transformer H cải thiện F1 `pretend`+13,03, `fireman`+8,19, `who`+6,36, `listen`+6,05 điểm; `mouse` giảm 8,50 điểm. Hiệu quả tăng tổng thể không đồng nghĩa mọi gloss đều tăng.

Confusions lớn nhất của T0 HF (mean count qua 3 seeds, không phải samples độc lập mới):

| Nhãn thật → dự đoán | Mean clips/seed | Tỷ lệ trong nhãn thật (%) |
|---|---:|---:|
| awake → wake | 39.33 | 66.67 |
| mouse → shhh | 23.67 | 39.44 |
| pretend → mouse | 18.67 | 32.18 |
| wake → awake | 16.67 | 27.78 |
| listen → drink | 11.67 | 20.47 |
| mouse → doll | 11.67 | 19.44 |

Accuracy theo held-out signer của các ứng viên (mean 3 seeds, %):

| Signer | Clips | G0 H | G-bi HBF | T0 HF |
|---|---:|---:|---:|---:|
| 28656 | 370 | 70.63 | 72.61 | 78.11 |
| 32319 | 384 | 70.83 | 73.52 | 76.56 |
| 36257 | 401 | 74.73 | 78.80 | 77.14 |

Accuracy theo signer phản ánh cả độ khó signer và phân bố gloss của signer đó; không tách riêng hai nguyên nhân. Chỉ có 3 held-out signers nên chưa suy rộng sang mọi người ký.

## Độ ổn định và hướng tiếp theo

T0 HF Macro-F1 seeds42/43/44 là 76,17/76,08/77,30%; std 0,56 điểm. Train–validation accuracy gap ở checkpoint được chọn: T0 HF 12.79 điểm, G-bi HBF 15.63 điểm. Có khoảng cách generalization đáng theo dõi; training accuracy không đại diện khả năng nhận diện trên signer mới.

Tiếp tục18 configs đã bàn giao để kiểm tra G-bi/G-attn/G-mean trên các subsets; chưa thể biết BiGRU HF/HB/H sẽ vượt T0 HF hay không. Giữ protocol hiện tại khi chạy bổ sung. Sau đủ 42 configs, mới chọn winner rồi mở test.

Ưu tiên kiểm video/nhãn và landmarks cho awake/wake, mouse, pretend, listen để tìm nguồn nhầm lẫn. Các confusions hiện tại chưa chứng minh nhãn sai hoặc preprocessing lỗi; một nghiên cứu sửa features/weights cần campaign riêng sau đợt này.

Các seed comparisons là mô tả paired gains; population std 3 seeds không phải confidence interval. Cùng1.155 clips được dùng qua nhiều fits, không xem72 fits là 72 datasets độc lập; chưa đưa ra kết luận statistical significance.

![Ablation và lỗi theo gloss](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/3e125a3e5bd9/findings/region_and_gloss.png)

[Dữ liệu phân tích JSON](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/3e125a3e5bd9/findings/FINDINGS.json) · [Bảng và audit 24 configs](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/3e125a3e5bd9/ALL_MEMBERS_RESULTS.json). JSON audit đầu tiên giữ snapshot scope 24/30 lịch sử; report này dùng scope 42 hiện tại và chỉ phân tích24 configs đã có.
