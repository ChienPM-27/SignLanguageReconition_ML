# Source freeze cho ablation study

Ngày: 2026-11-11 (+07:00).
Branch: `codex/ablation-study`.
Source commit dùng chung: `f6cf69fd4d22fece9c70a66a1b9f1e3461eb2fc2`.

**Source v4 đã được commit trên branch ablation-study.**
Cả 3 thành viên checkout branch này và dùng source SHA ở shared/source_commit.txt.
Report commits có thể khác nhau nếu src/scripts/configs/requirements.txt vẫn khớp
source commit; runner kiểm tra điều này trước train.

## Nội dung source

| Thành phần | Trạng thái trên branch |
| --- | --- |
| Preprocessing schema | v4 |
| Feature options | wrist_reference, palm_scale, fingertip_displacement, bend_cosines |
| Reliability metadata | src/features/validity.py; .validity.npz cạnh .npy |
| Cache/artifact identity | src/utils/artifacts.py; hashes và strict ownership checks |
| Checkpoint/resume | src/utils/checkpoint.py; metadata validation và atomic rollback |
| Frozen signer split | split.manifest_path; provenance/rebind/coverage checks |
| Validation-only training | evaluation.evaluate_test=false |
| Standalone evaluator | scripts/21_evaluate.py |
| Regression tests | test_evaluate, test_features_wave_b, test_wave_a_regression, test_wave_b_regression |
| Model support | GRU/LSTM; TCN/HandGraphTCN chưa tích hợp |
| Notebook | Đồng bộ branch ablation-study, mô tả schema v4, explicit final-test checkpoint |

main local và main trên GitHub được kiểm tra vẫn ở
`8e335b7a46f1e2e9d1b0c47850aa053c852b55f0` (schema v2).
Team lấy source từ codex/ablation-study cho campaign này.

## Shared handoff

| Trường | Giá trị |
| --- | --- |
| Source ref | codex/ablation-study |
| Source SHA dùng cho A/B/C | f6cf69fd4d22fece9c70a66a1b9f1e3461eb2fc2 |
| Config snapshot | shared/config.yaml, lấy từ canonical configs/islr.yaml tại source SHA |
| Config snapshot SHA256 | ef16f628cecce00be72cb8dd259140b0d21fbf448548bd8785a6d48f2baed0dd |
| Frozen split manifest / SHA256 | PENDING — A chuẩn bị trên dữ liệu thật, B/C dùng cùng file |
| Selection/vocabulary / SHA256 | PENDING — giữ cùng ordered 20 glosses |
| Dataset/class/signer/sample counts, exclusions | PENDING — xuất sau preprocessing/rebind |
| Real-data model metrics / GPU measurements | PENDING |

Copy shared/config.yaml và shared/source_commit.txt vào runtime shared package.
A bổ sung selection/glosses.csv và split_manifest.json; giao cùng package cho B/C.
Không thay hashes bằng giá trị giả; không regenerate signer assignment riêng từng run.

## Verification

- Full tests: `python -m pytest tests -q -p no:cacheprovider` với basetemp riêng:
  **249 passed, 1 warning, 81.53s, exit 0**. Warning do local CUDA unavailable.
  Source Python được commit khớp working tree đã chạy tests; chỉ chuẩn hóa EOF.
- Pyflakes: `python -m pyflakes src scripts tests app.py reports/ablation-study/tools`:
  **0 findings, exit 0**.
- Canonical config của cả 6 runs load thành công và D đúng: 160/160/162/162/192/212.
- Notebook JSON và Python syntax của 10 code cells hợp lệ; notebook magics được
  loại khỏi syntax check. Notebook chưa được chạy end-to-end.
- Tools lưu log/tổng hợp: **16 checks pass** trên fixtures tự tạo; xem TOOL_VALIDATION.md.

Engineering checks không thay thế kết quả real-data training.

## Cập nhật báo cáo

Source/config snapshots đã freeze; report từng run và RESULTS.md/JSON cập nhật
từ artifacts thật. Mỗi người chỉ sửa 2 run reports được phân công. ZIP/model/cache
lớn lưu artifact storage; Git giữ báo cáo và numerical result JSON cần thiết.
