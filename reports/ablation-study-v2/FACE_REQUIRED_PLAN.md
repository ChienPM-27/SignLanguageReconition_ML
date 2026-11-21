# Kế hoạch đang áp dụng: face bắt buộc, 30 configs / 90 lượt train

Quyết định của nhóm ngày 2026-11-13: luôn dùng face; không cần kiểm định face on/off.
Kế hoạch này thay phạm vi 60 cells trong PLAN.md đối với bảng kết quả chính.
Face là lựa chọn thiết kế, không được mô tả là cải thiện đã chứng minh qua ablation.

| Member | Configs phải nghiệm thu | Configs | Lượt train |
|---|---|---:|---:|
| A | G0 × F1-face/F2-face/F3-face/F4-face | 4 | 12 |
| B | G-mean/G-attn/G-bi × F0-face đến F4-face | 15 | 45 |
| C | G0__F0-face; L0/T0 × F0-face đến F4-face | 11 | 33 |
| Tổng | 6 profiles × 5 feature levels, mọi level có face | 30 | 90 |

Các members chạy song song trên runtime riêng. C cung cấp baseline GRU F0-face
cho A/B, không chạy lại baseline ở A. Mỗi config đủ seeds42/43/44.
So sánh các mức F0–F4 khi cố định profile, và các profiles khi cố định feature.
Không còn cặp face off/on hoặc phép kết luận tương tác của việc bật face.

F0 sử dụng policy F0-control cũ của V2 và thêm face. Các thay đổi F1 wrist=torso_xy,
F2 palm=clip_median, F3 fingertip displacement, F4 bend cosines vẫn cộng dồn.
Dimensions: F0-face196, F1-face198, F2-face198, F3-face228, F4-face248.

## Giữ kết quả đang có

Không train lại từ đầu chỉ vì thu hẹp phạm vi. Run face đã đủ artifacts và cùng
source/config/split/vocabulary/cohort giữ nguyên. Run face off đã chạy được lưu
trong lịch sử, nhưng không tính vào completion, ranking, winner hoặc bảng delta.
Không đổi tên/suffix hay sửa config metadata của run đã chạy.

A đang báo training G0__F4-face (seed42), run cuối trong queue tám config cũ.
Cho run đó hoàn tất đủ ba seeds và nghiệm thu, tải ZIP cũ rồi chạy cell chốt
face-only. Cell chỉ đọc/kiểm tra/export kết quả; không gọi training. Chỉ khi report
face-only xác nhận 4/4 hợp lệ mới coi A hoàn thành. Chưa có ZIP cuối để xác nhận.

## Notebook, source và Shared

Dùng artifacts/ablation-study-v2/face-only-v1, gồm notebooks A/B/C, ba gói bàn giao,
configs tham khảo, MATRIX.json 30 cells và KAGGLE_FACE_ONLY_FINALIZE_CELL.py.

Source/config training tiếp tục pin tag ablation-v2-notebooks-v1 và khớp Shared
77a84a7b12f1. Shared ZIP giữ nguyên byte-for-byte; không prepare lại split/cohort.
Matrix 60 trong Shared là superset canonical của source frozen. Module
face_only_v1.py xác định subset 30 và nghiệm thu riêng bằng FACE_ONLY_PROTOCOL.json.
Scope có ID v2-face-required-v1, module hash và matrix hash riêng. Các module
loader/reporting compatibility đã dùng giữ nguyên; cả ba module lưu trong ZIP.

FaceOnlySession giữ nguyên queue state archive của source, đọc các selections
cũ nhưng chỉ chạy/count/report các face IDs. Summarizer giữ canonical collect,
provenance/sequence checks, ranking và paired deltas; chỉ sử dụng context riêng
có 30 expected IDs và năm face columns. Không patch study.matrix hoặc sửa Shared
để tránh làm lệch provenance của các runs đã hoàn thành.

## Nghiệm thu cuối cùng

Files trả về: member_a_77a84a7b12f1_face_only_results.zip (4/4),
member_b_77a84a7b12f1_face_only_results.zip (15/15),
member_c_77a84a7b12f1_face_only_results.zip (11/11).

A dùng AGGREGATE_INPUTS trong notebook mới để tổng hợp. Cần đủ 30/30 configs
valid; thiếu cell thì chưa chốt winner. Chọn theo mean Validation Macro-F1,
lower population std của ba seeds, fewer parameters, run ID. Test giữ lại tới
khi winner đã khóa. Winner chỉ tốt nhất trong 30 cấu hình có face đã chạy.
