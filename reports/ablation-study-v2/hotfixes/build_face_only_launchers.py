"""Build face-required notebooks/handoffs without changing frozen training source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from build_compat_v2_launchers import build as build_compat, find_shared_bytes
from build_launchers import cell

REPO = Path(__file__).resolve().parents[3]
OUTPUT = REPO / "artifacts/ablation-study-v2/face-only-v1"
MODULES = ("report_predictions_v1.py", "campaign_compat_v2.py", "face_only_v1.py")


def module_code() -> str:
    text = '''import sys

FACE_MODULE_DIR = WORKSPACE / "ablation_v2_face_only_v1"
FACE_MODULE_DIR.mkdir(parents=True, exist_ok=True)
FACE_MODULE_PAYLOADS = {
'''
    for name in MODULES:
        payload = Path(__file__).with_name(name).read_text(encoding="utf-8")
        text += "    " + repr(name) + ": " + repr(payload) + ",\n"
    text += '''}
for face_module_name, face_module_source in FACE_MODULE_PAYLOADS.items():
    face_module_path = FACE_MODULE_DIR / face_module_name
    face_module_bytes = face_module_source.encode("utf-8")
    if face_module_path.exists() and face_module_path.read_bytes() != face_module_bytes:
        raise RuntimeError("Face-only module differs from this version: " + face_module_name)
    face_module_path.write_bytes(face_module_bytes)
if str(FACE_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(FACE_MODULE_DIR))
print("Face-only modules đã sẵn sàng; source/config training gốc giữ nguyên.")
'''
    return text


def scoped_rows() -> list[dict]:
    rows = json.loads((REPO / "reports/ablation-study-v2/MATRIX.json").read_text(encoding="utf-8"))
    return [row for row in rows if row["feature"].endswith("-face")]


def build(member: str) -> dict:
    notebook = build_compat(member)
    counts = {"A": 4, "B": 15, "C": 11}
    notebook["cells"][0]["source"] = [f"# Ablation V2 Face-required — Member {member}\n\n",
        "Face là feature bắt buộc theo quyết định của nhóm. Chỉ chạy F0-face đến F4-face.\n",
        f"Member {member}: **{counts[member]} configs / {counts[member] * 3} lượt train**. "
        "Toàn nhóm 30 configs / 90 lượt; không train trùng giữa A/B/C.\n\n",
        "Dùng dataset asl-signs, Shared 77a84a7b12f1 đã có và source tag đã khóa. "
        "Các run face hợp lệ đã chạy được giữ lại. Artifacts face off lưu lịch sử và không vào ranking.\n",
        "Kaggle: bật GPU/Internet, attach raw dataset + Shared, import notebook và Run All. "
        "Nếu runtime reset, RESTORE_INPUT nhận results ZIP cũ trước khi import Shared.\n"]
    new_cells = []
    inserted = False
    for item in notebook["cells"]:
        source = "".join(item["source"])
        if item["cell_type"] == "code" and "from notebook_support import MemberSession, discover" in source:
            new_cells.append(cell("markdown", "face-scope-module-guide",
                "## Nạp phạm vi face-required\n\n30 cells là subset đã đăng ký của source/config V2. "
                "Shared dữ liệu và các run IDs giữ nguyên.\n"))
            new_cells.append(cell("code", "face-scope-module-code", module_code()))
            source = source.replace("from notebook_support import MemberSession, discover",
                "from notebook_support import discover\nfrom face_only_v1 import FaceOnlySession")
            source = source.replace("session = MemberSession(", "session = FaceOnlySession(")
            source = source.replace("pd.DataFrame(session.queue)", "pd.DataFrame(session.face_queue)")
            inserted = True
        elif item["cell_type"] == "code" and item["id"] == "reporting-hotfix-code":
            source = '''import campaign_compat_v2 as campaign_compat
campaign_compat.install(session)
session.enable_scope()
'''
        elif item["cell_type"] == "code" and "MEMBER =" in source:
            source = source.replace('MODE = "prepare"', 'MODE = "train"')
        elif item["cell_type"] == "code" and "Ma trận chung:" in source:
            source = source.replace('"/ 60; winner:"', '"/ 30; winner:"')
        item["source"] = source.splitlines(keepends=True)
        new_cells.append(item)
    if not inserted:
        raise ValueError("Cannot find canonical member session initialization cell.")
    notebook["cells"] = new_cells
    notebook["metadata"]["face_required_scope"] = {"id": "v2-face-required-v1", "expected_cells": 30,
        "sha256": hashlib.sha256(Path(__file__).with_name("face_only_v1.py").read_bytes()).hexdigest()}
    for item in new_cells:
        if item["cell_type"] == "code":
            compile("".join(item["source"]), item["id"], "exec")
    return notebook


GUIDE = '''# Ablation: face là feature bắt buộc

Quyết định ngày 2026-11-13: không so sánh face on/off. Mọi cell được nghiệm thu
trong bảng mới đều có features.face=true. Không suy diễn rằng face đã được
chứng minh tốt hơn face off; đây là yêu cầu thiết kế của nhóm.

## Phân công mới: 30 configs / 90 lượt train

- A: G0__F1-face, G0__F2-face, G0__F3-face, G0__F4-face: 4 configs / 12 fits.
- B: G-mean/G-attn/G-bi × F0-face đến F4-face: 15 configs / 45 fits.
- C: G0__F0-face + LSTM(L0)/Transformer(T0) × F0-face đến F4-face: 11 configs / 33 fits.

C cung cấp GRU F0-face baseline cho A/B. Mỗi config chạy seeds42/43/44; không
train trùng. B/C có thể bắt đầu ngay trên runtime riêng, không đợi A xong.

F0 trong V2 dùng thiết lập F0-control cũ, giờ luôn thêm face. F1 thay wrist sang
torso_xy; F2 dùng clip_median palm scale; F3 thêm fingertip displacement; F4 thêm
bend cosines. Các mức cộng dồn. Dimensions F0/F1/F2/F3/F4-face: 196/198/198/228/248.

## A đang chạy queue cũ

A đã báo đang ở G0__F4-face, cell cuối trong queue tám configs. Để run hiện tại
hoàn tất đủ ba seeds, rồi tải member_a_77a84a7b12f1_results.zip và Save Version
có Output. Không dừng run face đang cần giữ, không train lại từ đầu.

Sau khi cell training kết thúc, mở KAGGLE_FACE_ONLY_FINALIZE_CELL.py, copy toàn
bộ vào một Code cell mới cuối notebook và chạy. Cell chỉ đổi phạm vi report/
nghiệm thu, giữ nguyên state/checkpoints/cache, không gọi training. Khi thấy
Face-only hoàn tất: 4 / 4, tải member_a_77a84a7b12f1_face_only_results.zip.
Nếu thiếu run face, cell liệt kê pending; dùng notebook A mới để tiếp tục riêng
face runs. Các run face đã valid tự động được bỏ qua.

Các run face off cũ vẫn nằm trong ZIP lịch sử, nhưng không tính vào completion,
ranking, deltas hoặc winner của bảng mới. Không đổi suffix để biến face off thành
face on. Face on đã chạy bằng config đúng có thể dùng lại nếu đủ provenance và
artifacts nghiệm thu. Log đang train chưa đủ chứng nhận config hoàn thành.

## Notebook/gói bàn giao mới cho A/B/C

Import ablation_v2_member_<member>_face_only.ipynb, bật GPU + Internet, attach
competition asl-signs đầy đủ và dataset Shared 77a84a7b12f1 hiện có. Giữ default
MODE=train, RUN_IDS=None, MEMBER/BRANCH/SOURCE_REF, rồi Run All. B/C lần đầu
không cần sửa settings. Mọi run được chọn đều có hậu tố -face.

Shared ZIP giữ nguyên byte-for-byte; không prepare lại cohort/split. Shared
matrix cũ là superset 60 configs; FACE_ONLY_PROTOCOL.json đăng ký subset 30 cần
nghiệm thu. Source, training config, seed policy và reporting/loader hotfix vẫn
giữ nguyên. Ba module thực chạy được nhúng và lưu trong results/compatibility/.
Configs trong gói là bản copy canonical để tham khảo, notebook tự đọc configs
frozen từ branch/tag. Không cần manual Git checkout trên Kaggle.

Nếu A chuyển notebook nhưng cùng runtime, queue state/checkpoints được đọc lại.
Nếu runtime reset, attach results ZIP đã lưu và đặt RESTORE_INPUT tới file ZIP
trước Run All. RESTORE_INPUT cần file ZIP; nếu Input unpack thành folder, đóng
gói folder results thành ZIP trong working rồi đặt đường dẫn đó.

## Nghiệm thu và tổng hợp

- A: 4/4, trả member_a_77a84a7b12f1_face_only_results.zip.
- B: 15/15, trả member_b_77a84a7b12f1_face_only_results.zip.
- C: 11/11, trả member_c_77a84a7b12f1_face_only_results.zip.

ZIP giữ scope audit, shared provenance, aligned validation predictions,
checkpoints/history/metrics/logs của ba seeds. Logs đơn lẻ không đủ.
A dùng notebook mới với AGGREGATE_INPUTS trỏ tới ZIP B/C để tổng hợp đúng 30 cells.
Winner chỉ chốt khi đủ 30/30; theo mean Val Macro-F1, lower std, fewer parameters,
run ID. Face off không tham gia, dù có điểm cao hơn. Test giữ lại tới khi khóa
winner. Kết luận chỉ áp dụng cho 30 cấu hình có face đã định trước.
'''


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    finalize = '''# Run in a NEW Code cell only AFTER the existing training cell has finished.
if "session" not in globals() or "WORKSPACE" not in globals():
    raise RuntimeError("Session variables missing: use the face_only notebook and restore results ZIP first.")
'''
    finalize += module_code()
    finalize += '''from face_only_v1 import FaceOnlySession
import campaign_compat_v2 as campaign_compat

session = FaceOnlySession.adopt(session)
campaign_compat.install(session)
status = session.report()
print("Face-only hoàn tất:", status["completed"], "/", status["expected"])
print("Pending face runs:", [row["run_id"] for row in status["queue"] if row["status"] != "COMPLETED"])
print("Historical face-off runs:", status["historical_selected_run_ids"])
print("Download:", session.export_results())
'''
    compile(finalize, "KAGGLE_FACE_ONLY_FINALIZE_CELL.py", "exec")
    (OUTPUT / "KAGGLE_FACE_ONLY_FINALIZE_CELL.py").write_text(finalize, encoding="utf-8")
    (OUTPUT / "HUONG_DAN.md").write_text(GUIDE, encoding="utf-8")
    (OUTPUT / "MATRIX.json").write_text(json.dumps(scoped_rows(), indent=2) + "\n", encoding="utf-8")
    shared = find_shared_bytes()
    for member in "ABC":
        filename = f"ablation_v2_member_{member.lower()}_face_only.ipynb"
        notebook = (json.dumps(build(member), ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        (OUTPUT / filename).write_bytes(notebook)
        source = REPO / f"configs/ablation-study-v2/member-{member.lower()}"
        queue = [r for r in json.loads((source / "QUEUE.json").read_text(encoding="utf-8")) if r["feature"].endswith("-face")]
        files = {filename: notebook, "ablation_v2_shared_77a84a7b12f1.zip": shared,
            "QUEUE.json": (json.dumps(queue, indent=2) + "\n").encode("utf-8"),
            "MATRIX.json": (OUTPUT / "MATRIX.json").read_bytes(), "HUONG_DAN.md": GUIDE.encode("utf-8"),
            "KAGGLE_FACE_ONLY_FINALIZE_CELL.py": finalize.encode("utf-8")}
        for row in queue:
            data = (source / row["config"]).read_bytes()
            destination = OUTPUT / f"configs/member-{member.lower()}" / row["config"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            files["configs/" + row["config"]] = data
        checksums = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
        files["CHECKSUMS.json"] = (json.dumps(checksums, indent=2) + "\n").encode("utf-8")
        target = OUTPUT / f"ban-giao-member-{member.lower()}-77a84a7b12f1-face-only-v1.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        print(target)


if __name__ == "__main__":
    main()
