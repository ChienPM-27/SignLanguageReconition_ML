# Trạng thái V3

Đã freeze snapshot `b4fd56f03a57` ngày 13/11/2026. Bộ source gồm101 files; runtime kiểm hash source/config/matrix/split trước training. Mỗi gói member có notebook, Shared V3 ZIP giống nhau, config YAML riêng, hướng dẫn và PDF kế hoạch.

Vòng đầu A7/B3/C14 configs; tổng24/72 fits. Kiểm chứng GRU bổ sung0 hoặc6 configs, tối đa30/90 fits. Có42 YAML phương án sẵn trong snapshot, không chạy tất cả42.

Đã kiểm tra regression331 tests; bộ V3 cập nhật50 tests riêng; kiểm tra config/guards cuối42 tests riêng. Các lần có overlap, không cộng thành số tests độc lập. Pyflakes sạch cho src/scripts/tests và bộ tools V3. Notebook nbformat/syntax, CRC, canonical config cho42 cells, code deploy vào runtime mới, queue A/B/C và nguồn snapshot đã kiểm tra. Shared/results V2 đã được kiểm tra tính nguyên vẹn trước khi dọn theo yêu cầu bên dưới. PDF2 trang đã render và xem từng trang. Training CLI/standalone evaluation đã kiểm tra bằng dữ liệu synthetic trong tests. Campaign thật của A đã hoàn tất và nghiệm thu bên dưới; test thật vẫn chưa mở.

Files giao nhóm: [thư mục V3](../../artifacts/ablation-study-v3/). Ledger checksums: [VALIDATION.json](../../artifacts/ablation-study-v3/VALIDATION.json).

Ngày 14/11/2026, theo yêu cầu “Chỉ giữ campaign V3 hiện tại”, đã chuyển bundles/results V1/V2, các gói thử trong `drafts/`, launchers cũ, bảng/PDF kết quả V1, cache local F0 và thư mục review/test tạm vào Windows Recycle Bin. Tổng 12.549 files, 3.388.342.860 bytes đã được đưa ra khỏi workspace; chưa giải phóng dung lượng vật lý khi còn trong Recycle Bin. Shared V3, ba gói A/B/C và toàn bộ 7 configs/21 fits của A được giữ nguyên checksum. Source/config/data/dependencies được giữ lại, gồm các tools cũ vẫn phục vụ tests. [Ledger dọn dẹp](../cleanup/2026-11-14.json) ghi từng đường dẫn và trạng thái; các mục từng chuyển qua thư mục tạm trước khi recycle có `recovery_path`, dùng để đối chiếu khi khôi phục.

Không đổi hay publish các branches V2 cũ. Notebook V3 dùng source được đóng gói trong Shared nên không phụ thuộc checkout branch hoặc Internet để tải code.

## Member A đã nghiệm thu ngày 13/11/2026

Đã nhận `member_a_b4fd56f03a57_v3_results.zip`: 68.789.908 bytes, 317 ZIP entries; SHA256 `ae638dcade7c0e4c125a3015c12e71f170fd4c8b11c71bbb38bdb6fdf400cbf9`. Lưu nguyên trạng theo freeze và checksum tại `artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/ae638dcade7c/`; bản trong Downloads được giữ nguyên.

Canonical `study_v3.collect` pass cả 7 selected attempts, đủ 21 fits/seeds42/43/44. CRC, source/config/matrix/protocol/split/cache identity, exact inventory, checkpoints, successful stage logs, history selection và 21 bộ validation predictions đều đã kiểm tra. Mỗi bộ predictions có cùng 1.155 clips, 20 glosses và 3 held-out signers. Đã recompute Top-1/Top-3/Macro-F1/confusion matrices; không có test report hoặc winner lock.

| Member | Configs vòng đầu đã nghiệm thu | Fits | Trạng thái results |
|---|---:|---:|---|
| A | 7/7 | 21/21 | ACCEPTED |
| B | Chưa nhận | Chưa nghiệm thu | Pending 3 configs |
| C | Chưa nhận | Chưa nghiệm thu | Pending 14 configs |

H và HB cùng mean Validation Macro-F1 `0.7120` theo report làm tròn bốn chữ số; H có std thấp hơn (`0.0077` so với `0.0104`) và được xếp trước theo tie-break đã freeze. HF `0.7072`; HBF `0.7068`; F `0.2213`; BF `0.2032`; B `0.0928`. Thêm body vào H gần như không đổi mean; thêm face vào HB giảm khoảng 0.51 điểm Macro-F1 %, giảm cả ba seeds. Kết luận giới hạn trong GRU G0/cohort/fixed weights; không chứng minh body/face vô dụng hoặc 70/20/10 tối ưu.

A không cần train lại. Campaign đang có 7/24 configs vòng đầu đã nghiệm thu; chưa chọn GRU architecture/cross-plan hoặc global winner. Chờ results B/C; nếu GRU architecture thắng khác G0 mới bổ sung sáu subsets cho B. Test tiếp tục giữ kín.

Artifacts: [ZIP gốc](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/ae638dcade7c/member_a_b4fd56f03a57_v3_results.zip) · [REVIEW.json](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/ae638dcade7c/REVIEW.json) · [ZIP_CHECK.json](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/ae638dcade7c/ZIP_CHECK.json) · [STORAGE.json](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/ae638dcade7c/STORAGE.json) · [Phân tích chi tiết](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/ae638dcade7c/analysis/ANALYSIS.md).

Script phân tích nằm ở `reports/ablation-study-v3/analyze_member_a_results.py`, ngoài source scope đã freeze. Phân tích không chạy training, checkpoint inference hoặc test evaluation. Đã xem và kiểm tra hai biểu đồ Macro-F1/seeds và F1 từng gloss.

Chuẩn bị push source ngày 14/11/2026: bản xuất riêng từ Git index đã pass 332 tests trong 393,20 giây, không có data/Shared/checkpoints/results local. Pyflakes sạch cho src/scripts/tests và reports V3; đủ 101 source hashes khớp Shared đã freeze. Ba notebooks pass nbformat/syntax và không có execution outputs; B/C dùng `RAW_INPUT = "/kaggle/input/competitions/asl-signs"`. Các lượt training/evaluation trong tests dùng dữ liệu synthetic; không mở test set thật của campaign.
