# Bạn tự chạy 18 configs còn thiếu

Đã hoàn tất ngày 20/11/2026, ZIP B full42 được nghiệm thu đủ 21/21 configs, toàn nhóm đủ 42/42. [Kết quả full42](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/7f86f6ac76de/ALL_MEMBERS_FULL42_RESULTS.md) · [Notebook test cuối](../../notebooks/ablation_v3_final_test_kaggle.ipynb). Hướng dẫn chạy bổ sung bên dưới được giữ để đối chiếu.

Test cuối cũng đã nghiệm thu từ ZIP A full42, `T0__HF` đạt Macro-F1 87,30% ± 0,77 điểm phần trăm trên 1.141 test clips. [Báo cáo cuối](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/b0cf901a9df3/analysis/FINAL_RESULTS.md).

Theo yêu cầu ngày20/11/2026, chạy đủ42 configs đã chuẩn bị. A7 và C14 đã hoàn thành; B có3/21 configs HBF. Còn18 configs/54 fits:

| Profile | Các subsets còn thiếu | Configs / fits mới |
|---|---|---:|
| G-mean | H, B, F, HB, HF, BF | 6 / 18 |
| G-attn | H, B, F, HB, HF, BF | 6 / 18 |
| G-bi | H, B, F, HB, HF, BF | 6 / 18 |

Files:

- [Gói đầy đủ: notebook, configs18, Shared và ZIP results B](../../artifacts/ablation-study-v3/ablation_v3_complete_18_configs.zip).
- [Notebook Kaggle](../../notebooks/ablation_v3_complete_18_kaggle.ipynb).
- [18 YAML riêng](../../artifacts/ablation-study-v3/ablation_v3_missing_18_yaml.zip).
- [Hướng dẫn trong gói](../../artifacts/ablation-study-v3/supplements/full42-member-b/HUONG_DAN.md).
- [Ledger kiểm tra](../../artifacts/ablation-study-v3/supplements/full42-member-b/VALIDATION.json).

Giải nén gói, import notebook vào Kaggle; tạo Dataset từ Shared ZIP và ZIP results B có trong gói rồi Add Input dataset cùng competition `asl-signs`. Bật GPU và Run All. Notebook tự tìm inputs; nếu attach nhiều versions thì điền paths chính xác trong settings. Hỗ trợ cả ZIP và folder results đã được Kaggle giải nén.

Trước training phải thấy **3 COMPLETED + 18 PENDING**, tổng21 config B. `RUN_IDS` liệt kê tường minh cả18 configs; không còn gate phase2/cross-plan. ZIP được cập nhật sau mỗi config; nếu reset, attach ZIP full42 mới nhất và chạy lại cùng notebook để tiếp tục. Đích cuối **B21/21 configs, 63 fits** (9 cũ +54 mới). File tải về là `member_b_b4fd56f03a57_v3_full42_results.zip`.

Source training, 42 canonical configs, cohort, split, checkpoint selection và seeds42/43/44 giữ nguyên freeze `b4fd56f03a57`. `full_matrix_v3.py` là orchestration ngoài source scope đã freeze; nó ghi `FULL_MATRIX_SCOPE.json` để phân biệt việc mở rộng phạm vi chọn model thành42. Khi đủ kết quả, dùng `FullMatrixSession.aggregate` cho A để nghiệm thu42/42; aggregate24/30 của notebook cũ sẽ từ chối configs mở rộng. Test tiếp tục giữ kín tới đủ42 và khóa winner.

Kiểm tra gồm runner dispatch18/bỏ qua3/resume không train trùng, guard không chốt winner ở24/30, folder restore của Kaggle và nghiệm thu full42 bằng fixtures self-contained. Gói thật được kiểm CRC/hashes, notebook nbformat/syntax và bootstrap từ Shared. Review/local QA không chạy training hoặc test thật.

## Sửa lỗi Kaggle split path ngày20/11/2026

Notebook bổ sung ban đầu đặt runtime root ở `/kaggle/working/ablation_v3_full42`, trong khi `env.resolve_paths` luôn dùng `/kaggle/working` trên Kaggle. Do đó split writer đã ghi CSV thành công nhưng runner không tìm thấy file ở root lồng. Bản notebook và ZIP hiện tại đã sửa root về `/kaggle/working`, có guard so với canonical resolver. Shared/config/training source không thay đổi.

Nếu đang ở kernel cũ bị lỗi `ablation_v3_full42/data/splits/train.csv`, giữ phiên và chạy nội dung [cell phục hồi](recover_full42_split_cell.py) trong một Code cell mới. Cell kiểm frozen membership, sao chép năm CSV/JSON split tới đường dẫn runner cần mà không ghi đè file khác dữ liệu, giữ nguyên feature cache và configs COMPLETED, rồi retry các configs lỗi. Không cần reset phiên hoặc upload Shared lại.

Đã kiểm6 tests, gồm split writer thật dưới Kaggle path semantics, tái hiện lỗi root lồng, phục hồi/không ghi đè dữ liệu khác, cache preservation và canonical-root guard của notebook. Notebook mới bootstrap ở report mode với ZIP B thật: restore3, còn18, source101 hashes khớp. Chưa quan sát GPU training thật sau sửa; chờ output chạy tiếp của người dùng.
