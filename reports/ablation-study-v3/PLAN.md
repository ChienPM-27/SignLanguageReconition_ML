# Ablation V3 đã chuẩn hóa theo vùng

Campaign đã hoàn tất ngày 20/11/2026, nghiệm thu **42 configs/126 fits**, A7/B21/C14, và test evaluation của ba checkpoints `T0__HF`. Validation Macro-F1 76,52% ± 0,56 điểm phần trăm, test Macro-F1 87,30% ± 0,77 điểm phần trăm. [Báo cáo cuối](../../artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/b0cf901a9df3/analysis/FINAL_RESULTS.md) · [Ma trận validation full42](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/7f86f6ac76de/ALL_MEMBERS_FULL42_RESULTS.md). Các đoạn 24/30 dưới đây ghi lại thiết kế điều kiện lúc freeze; source training/config/split giữ nguyên.

Hướng dẫn source, notebook và protocol: [README.md](README.md). Hướng dẫn trong gói bàn giao được lưu cùng Shared bên ngoài Git.

Vòng đầu: A7 configs, B3, C14; tổng24 configs/72 fits. A/B/C chạy song song. Sau vòng đầu, chọn architecture GRU theo HBF và bổ sung sáu subsets còn lại nếu architecture thắng khác G0; tổng tối đa30 configs/90 fits. Mọi YAML cho các phương án được chuẩn bị trước; runner chỉ chạy queue đã phân công và cross-plan được xác nhận từ kết quả vòng đầu.

Ba nhóm feature: hands143, upper-body17, face36. Đầu vào cache196; independent per-region projection128/L2 normalization; fixed fusion70/20/10; temporal core chung. V3 train fresh, không tái sử dụng checkpoints V2. Artifacts V1/V2 local đã chuyển vào Recycle Bin ngày 14/11/2026 theo yêu cầu chỉ giữ V3.

Configs ở `configs/ablation-study-v3/member-a|b|c/`; launchers ở `notebooks/ablation_v3_member_a|b|c_kaggle.ipynb`; source snapshot/Shared/handoff ZIP ở `artifacts/ablation-study-v3/`. Freeze ID và checksums ghi trong `FREEZE_STATUS.json`; trạng thái verification ghi riêng trong `VALIDATION.json`.
