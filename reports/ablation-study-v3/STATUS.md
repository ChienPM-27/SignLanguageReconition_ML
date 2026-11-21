# Trạng thái V3

Campaign V3 `b4fd56f03a57` đã hoàn tất ngày 20/11/2026. Đã nghiệm thu **42/42 configs, 126/126 fits**, A7/B21/C14, và test evaluation của ba checkpoints `T0__HF` seeds 42/43/44. Validation Macro-F1 **76,52% ± 0,56 điểm phần trăm**, test Macro-F1 **87,30% ± 0,77 điểm phần trăm** trên 1.141 test clips. [Báo cáo cuối](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/b0cf901a9df3/analysis/FINAL_RESULTS.md) · [Ma trận validation full42](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/7f86f6ac76de/ALL_MEMBERS_FULL42_RESULTS.md).

[Phân tích vòng đầu 24 configs](PHASE1_FINDINGS.md) được giữ làm snapshot lịch sử. Báo cáo full42 hiện có ma trận 6 profiles × 7 subsets, ranking, paired region/model gains và lỗi theo gloss/signer từ validation predictions.

Hotfix notebook full42 ngày 20/11/2026 sửa persistent root về `/kaggle/working` để khớp split writer trên Kaggle. Có cell phục hồi phiên cũ giữ cache và retry, chi tiết trong [FULL42_HANDOFF.md](FULL42_HANDOFF.md). Bộ 6 regression tests và bootstrap với results B thật pass. ZIP B mới đã xác nhận 18 configs bổ sung hoàn tất sau hotfix, queue failed rỗng.

Đã freeze snapshot `b4fd56f03a57` ngày 13/11/2026. Bộ source gồm101 files; runtime kiểm hash source/config/matrix/split trước training. Mỗi gói member có notebook, Shared V3 ZIP giống nhau, config YAML riêng, hướng dẫn và PDF kế hoạch.

Vòng đầu A7/B3/C14 configs; tổng24/72 fits. Kiểm chứng GRU bổ sung0 hoặc6 configs, tối đa30/90 fits. Có42 YAML phương án sẵn trong snapshot, không chạy tất cả42.

Đã kiểm tra regression331 tests; bộ V3 cập nhật50 tests riêng; kiểm tra config/guards cuối42 tests riêng. Các lần có overlap, không cộng thành số tests độc lập. Pyflakes sạch cho src/scripts/tests và bộ tools V3. Notebook nbformat/syntax, CRC, canonical config cho42 cells, code deploy vào runtime mới, queue A/B/C và nguồn snapshot đã kiểm tra. Shared/results V2 đã được kiểm tra tính nguyên vẹn trước khi dọn theo yêu cầu bên dưới. PDF2 trang đã render và xem từng trang. Training CLI/standalone evaluation đã kiểm tra bằng dữ liệu synthetic trong tests. Campaign thật của A đã hoàn tất và nghiệm thu bên dưới; test thật vẫn chưa mở.

Files giao nhóm: [thư mục V3](../../artifacts/ablation-study-v3/). Ledger checksums: [VALIDATION.json](../../artifacts/ablation-study-v3/VALIDATION.json).

Ngày 14/11/2026, theo yêu cầu “Chỉ giữ campaign V3 hiện tại”, đã chuyển bundles/results V1/V2, các gói thử trong `drafts/`, launchers cũ, bảng/PDF kết quả V1, cache local F0 và thư mục review/test tạm vào Windows Recycle Bin. Tổng 12.549 files, 3.388.342.860 bytes đã được đưa ra khỏi workspace; chưa giải phóng dung lượng vật lý khi còn trong Recycle Bin. Shared V3, ba gói A/B/C và toàn bộ 7 configs/21 fits của A được giữ nguyên checksum. Source/config/data/dependencies được giữ lại, gồm các tools cũ vẫn phục vụ tests. [Ledger dọn dẹp](../cleanup/2026-11-14.json) ghi từng đường dẫn và trạng thái; các mục từng chuyển qua thư mục tạm trước khi recycle có `recovery_path`, dùng để đối chiếu khi khôi phục.

Không đổi hay publish các branches V2 cũ. Notebook V3 dùng source được đóng gói trong Shared nên không phụ thuộc checkout branch hoặc Internet để tải code.

## Test cuối đã nghiệm thu ngày 20/11/2026

Đã nhận `member_a_b4fd56f03a57_v3_full42_results.zip`, 71.648.051 bytes, 338 entries, SHA256 `b0cf901a9df3496ec22455ea37bb8b8f254ba99fc40ebd916cbb00a419fcc37c`. Lưu bản sao nguyên trạng theo member/freeze/checksum, bản Downloads giữ nguyên. ZIP chứa aggregate 42 records, winner lock và ba test reports.

Aggregate trong ZIP khớp cả 42 records đã nghiệm thu. A7 artifacts cũ khớp từng byte. Lock `T0__HF` khớp selection trước test và hashes của ba checkpoints C. Preprocess/split/test logs đều thành công. Ba reports có đúng 1.141 frozen test sequence IDs, labels, participants, vocabulary và preprocess identity. Test sequences/signers tách khỏi train/validation. Đã tính lại Top-1/Top-3/Macro-F1, cross-entropy, confusion matrices và per-signer accuracy từ predictions.

| Metric test | Mean ± population std qua 3 seeds |
|---|---:|
| Macro-F1 | 87,30% ± 0,77 điểm |
| Top-1 | 88,20% ± 1,07 điểm |
| Top-3 | 98,34% ± 0,52 điểm |
| Cross-entropy loss | 0,3207 ± 0,0333 |

Test signers 27610/37779/61333 có mean accuracy 85,17%/91,56%/87,53%. F1 `awake` còn 22,94%, `wake` 58,80%, `mouse` 80,38%, `pretend` 88,38%. `awake → wake` vẫn nhầm 80,12% mẫu awake trên test, `mouse → shhh` 7,60%, `pretend → mouse` 0/61 ở cả ba seeds. Validation và test dùng nhóm signer khác nhau nên các chênh lệch phản ánh cohort và generalization. Ranking configs dựa trên validation, test được dùng cho winner đã chọn.

[Final metrics và audit JSON](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/b0cf901a9df3/analysis/FINAL_RESULTS.json) · [Review ZIP](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/b0cf901a9df3/REVIEW.json). Script `analyze_final_test_results.py` pass pyflakes và audit bằng ZIP thật. Lượt review đọc các artifacts đã có.

## Nghiệm thu full42 ngày 20/11/2026

Đã nhận `member_b_b4fd56f03a57_v3_full42_results.zip`, 278.042.354 bytes, 966 entries, SHA256 `ca473a5e25c6e38ffb1bf27e76859292de066ee3b40828fbbb25ec9e40d3d1dc`. Bản sao nguyên trạng ở `artifacts/ablation-study-v3/results/member-b/b4fd56f03a57/ca473a5e25c6/`. Các artifacts của ba configs HBF cũ khớp từng byte với ZIP B vòng đầu. Lịch sử split attempt thất bại trước hotfix được lưu trong ZIP, selected queue đủ 21 configs và failed rỗng.

Cả 42 selected attempts pass canonical `study_v3.collect`, exact inventories, frozen source/config/matrix/protocol/split/cache identity, hashes của 126 checkpoints, successful logs, loss-selected histories và multi-seed summaries. Đã tính lại Top-1/Top-3/Macro-F1 và confusion matrices từ 126 bộ predictions, tất cả cùng 1.155 validation clips. Lần nghiệm thu đọc artifacts đã train.

| Member | Configs đã nghiệm thu | Fits | Trạng thái |
|---|---:|---:|---|
| A | 7/7 | 21/21 | ACCEPTED |
| B | 21/21 | 63/63 | ACCEPTED |
| C | 14/14 | 42/42 | ACCEPTED |

`T0__HF` được chọn trong toàn bộ 42 configs, Validation Macro-F1 76,52% ± 0,56, Top-1 77,26%, Top-3 93,48%. BiGRU tốt nhất là `G-bi__HBF` 74,54% ± 0,32. BiGRU tăng so với G0 ở cả 3 seeds trên từng subset có tay, mức tăng mean 2,94–3,85 điểm phần trăm. Các config tốt nhất của cả sáu profiles tiếp tục có lỗi `pretend → mouse` tập trung ở signer `32319`.

[Audit và metrics full42 JSON](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/7f86f6ac76de/ALL_MEMBERS_FULL42_RESULTS.json) · [Lựa chọn và checkpoint hashes](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/7f86f6ac76de/WINNER_SELECTION.json). Script `analyze_full42_results.py` tái lập nghiệm thu từ ba ZIP. Notebook `ablation_v3_final_test_kaggle.ipynb` aggregate đủ 42, kiểm winner/checkpoint hashes rồi gọi evaluation của ba seeds trên test cohort 1.141 clips. Test metrics thật đã được nghiệm thu ở mục trên.

Notebook test cuối pass nbformat/syntax, source 101 hashes, bootstrap với Shared và ba ZIP thật ở MODE `report`, restore A7, aggregate 42/42 và hashes ba checkpoints `T0__HF`. Script review và builder pass pyflakes. Lượt kiểm notebook ở local dùng runtime tạm có đường dẫn ngắn để phù hợp Windows. ZIP A mới xác nhận evaluation thật trên Kaggle đã hoàn tất.

## Member A đã nghiệm thu ngày 13/11/2026

Đã nhận `member_a_b4fd56f03a57_v3_results.zip`: 68.789.908 bytes, 317 ZIP entries; SHA256 `ae638dcade7c0e4c125a3015c12e71f170fd4c8b11c71bbb38bdb6fdf400cbf9`. Lưu nguyên trạng theo freeze và checksum tại `artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/ae638dcade7c/`; bản trong Downloads được giữ nguyên.

Canonical `study_v3.collect` pass cả 7 selected attempts, đủ 21 fits/seeds42/43/44. CRC, source/config/matrix/protocol/split/cache identity, exact inventory, checkpoints, successful stage logs, history selection và 21 bộ validation predictions đều đã kiểm tra. Mỗi bộ predictions có cùng 1.155 clips, 20 glosses và 3 held-out signers. Đã recompute Top-1/Top-3/Macro-F1/confusion matrices; không có test report hoặc winner lock.

| Member | Configs vòng đầu đã nghiệm thu | Fits vòng đầu | Trạng thái vòng đầu |
|---|---:|---:|---|
| A | 7/7 | 21/21 | ACCEPTED |
| B | 3/3 | 9/9 | ACCEPTED vòng đầu; đã hoàn tất mở rộng 21/21 ở full42 |
| C | 14/14 | 42/42 | ACCEPTED |

H và HB cùng mean Validation Macro-F1 `0.7120` theo report làm tròn bốn chữ số; H có std thấp hơn (`0.0077` so với `0.0104`) và được xếp trước theo tie-break đã freeze. HF `0.7072`; HBF `0.7068`; F `0.2213`; BF `0.2032`; B `0.0928`. Thêm body vào H gần như không đổi mean; thêm face vào HB giảm khoảng 0.51 điểm Macro-F1 %, giảm cả ba seeds. Kết luận giới hạn trong GRU G0/cohort/fixed weights; không chứng minh body/face vô dụng hoặc 70/20/10 tối ưu.

A không cần train lại. Ngày 20/11/2026 đã nhận và nghiệm thu B/C, đủ 24 configs vòng đầu. Protocol freeze trước đó chọn G-bi và cross-plan6 bên dưới; yêu cầu full42 hiện tại thay thế bước này bằng18 configs bổ sung của B. Test tiếp tục giữ kín.

Artifacts: [ZIP gốc](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/ae638dcade7c/member_a_b4fd56f03a57_v3_results.zip) · [REVIEW.json](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/ae638dcade7c/REVIEW.json) · [ZIP_CHECK.json](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/ae638dcade7c/ZIP_CHECK.json) · [STORAGE.json](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/ae638dcade7c/STORAGE.json) · [Phân tích chi tiết](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/ae638dcade7c/analysis/ANALYSIS.md).

Script phân tích nằm ở `reports/ablation-study-v3/analyze_member_a_results.py`, ngoài source scope đã freeze. Phân tích không chạy training, checkpoint inference hoặc test evaluation. Đã xem và kiểm tra hai biểu đồ Macro-F1/seeds và F1 từng gloss.

## Member B/C và tổng hợp vòng đầu ngày 20/11/2026

Phần tổng hợp dưới đây là snapshot của protocol điều kiện 24/30 trước khi người dùng yêu cầu chạy toàn bộ42. Cross-plan6 vẫn được lưu để đối chiếu; gói full42 bên trên thay thế hướng dẫn chạy bổ sung hiện tại.

ZIP B: 40.404.536 bytes, 137 entries, SHA256 `d2d48b98651a6c1c2deea50d35e5b49b94c1daf6e0340025352cc2e08c177518`. ZIP C: 173.453.052 bytes, 632 entries, SHA256 `9d6acb579e3f1f490dc5bb680269d45015df997ed20b24f3c230fbd48c6095a2`. Đã lưu bản sao nguyên trạng theo member/freeze/checksum; bản Downloads vẫn giữ nguyên. Mỗi member có `STORAGE.json`, `ZIP_CHECK.json`, `REVIEW.json` cạnh ZIP.

Canonical `study_v3.collect` pass cả 24 selected attempts. Đã kiểm CRC, exact inventory, source/config/matrix/protocol/split/cache identity, hashes của 72 checkpoints, successful stage logs, loss-selected histories và multi-seed summaries. Đã tính lại Top-1/Top-3/Macro-F1 và confusion matrices từ 72 bộ predictions; cả 72 dùng cùng 1.155 validation clips, 20 glosses, 3 held-out signers. Không chạy training, checkpoint inference hoặc test evaluation trong lần review.

| Dẫn đầu vòng đầu | Validation Macro-F1 mean ± population std | Top-1 | Top-3 |
|---|---:|---:|---:|
| T0__HF | 0.7652 ± 0.0056 | 0.7726 | 0.9348 |
| T0__HBF | 0.7592 ± 0.0096 | 0.7686 | 0.9440 |
| T0__HB | 0.7545 ± 0.0111 | 0.7645 | 0.9423 |
| G-bi__HBF | 0.7454 ± 0.0032 | 0.7507 | 0.9403 |

G-bi thắng nhóm GRU trên HBF theo rule đã freeze: mean Macro-F1, std thấp hơn, parameters ít hơn, run ID. B phải chạy `G-bi__H`, `G-bi__B`, `G-bi__F`, `G-bi__HB`, `G-bi__HF`, `G-bi__BF` với seeds42/43/44. G-bi trên HBF đã có nên không train lại. Campaign hiện là **PENDING 24/30 configs**, không phải lỗi nghiệm thu B/C. T0__HF chỉ là leader vòng đầu; chưa chốt global winner. Chênh lệch từ ba seeds chưa phải kết luận statistical significance.

[Báo cáo tổng hợp và bảng 24 configs](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/3e125a3e5bd9/ALL_MEMBERS_RESULTS.md) · [Metrics và audit JSON](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/3e125a3e5bd9/ALL_MEMBERS_RESULTS.json) · [cross_plan.zip cho B](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/3e125a3e5bd9/cross_plan.zip) · [Hướng dẫn B tiếp tục](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/3e125a3e5bd9/HUONG_DAN_B_CROSS_PLAN.md).

Cross-plan ZIP đã được load bằng `MemberSession.load_cross_plan` thật; queue phase2 khớp đúng sáu configs. Script review là `analyze_campaign_results.py`, ngoài source/config scope đã freeze; pyflakes và syntax pass. Biểu đồ so sánh đã render và xem trực tiếp.

Chuẩn bị push source ngày 14/11/2026: bản xuất riêng từ Git index đã pass 332 tests trong 393,20 giây, không có data/Shared/checkpoints/results local. Pyflakes sạch cho src/scripts/tests và reports V3; đủ 101 source hashes khớp Shared đã freeze. Ba notebooks pass nbformat/syntax và không có execution outputs; B/C dùng `RAW_INPUT = "/kaggle/input/competitions/asl-signs"`. Các lượt training/evaluation trong tests dùng dữ liệu synthetic; không mở test set thật của campaign.
