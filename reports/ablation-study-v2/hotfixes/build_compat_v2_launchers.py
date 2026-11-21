"""Build self-contained recovery cell and member notebooks for loader/report fixes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from build_launchers import build as build_v1

REPO = Path(__file__).resolve().parents[3]
OUTPUT = REPO / "artifacts/ablation-study-v2/compat-v2"
MODULES = ("report_predictions_v1.py", "campaign_compat_v2.py")


def installer_code() -> str:
    code = '''import importlib.util
import sys

COMPAT_DIR = WORKSPACE / "ablation_v2_compat_v2"
COMPAT_DIR.mkdir(parents=True, exist_ok=True)
COMPAT_PAYLOADS = {
'''
    for name in MODULES:
        source = Path(__file__).with_name(name).read_text(encoding="utf-8")
        code += "    " + repr(name) + ": " + repr(source) + ",\n"
    code += '''}
for compat_name, compat_source in COMPAT_PAYLOADS.items():
    compat_path = COMPAT_DIR / compat_name
    compat_bytes = compat_source.encode("utf-8")
    if compat_path.exists() and compat_path.read_bytes() != compat_bytes:
        raise RuntimeError("Compatibility module differs from this reviewed version: " + compat_name)
    compat_path.write_bytes(compat_bytes)
if str(COMPAT_DIR) not in sys.path:
    sys.path.insert(0, str(COMPAT_DIR))
compat_spec = importlib.util.spec_from_file_location("ablation_v2_campaign_compat", COMPAT_DIR / "campaign_compat_v2.py")
campaign_compat = importlib.util.module_from_spec(compat_spec)
sys.modules[compat_spec.name] = campaign_compat
compat_spec.loader.exec_module(campaign_compat)
campaign_compat.install(session)
print("Đã cài feature loader + reporting compatibility; queue và feature cache được giữ lại.")
'''
    return code


def build(member: str) -> dict:
    notebook = build_v1(member)
    notebook["cells"][0]["source"] += ["\n**Compatibility v2:** sửa Dataset chặn chiều face; "
        "kèm reporting v1. A mặc định chạy đủ 8 configs và retry lỗi kỹ thuật đã xác định.\n"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and cell["id"] == "reporting-hotfix-code":
            cell["source"] = installer_code().splitlines(keepends=True)
        elif cell["cell_type"] == "markdown" and cell["id"] == "reporting-hotfix-guide":
            cell["source"] = ["### Compatibility v2: Dataset face + reporting\n\n",
                "FeatureLayout nhận chiều face 196/198/228/248; Dataset bỏ whitelist cũ và kiểm tra binary face mask. "
                "Reporting v1 giữ nguyên payload. Hai audit nằm trong từng val_report.json và folder kết quả. "
                "Git checkout, feature arrays, split, models và protocol tiếp tục dùng bộ đã khóa.\n"]
        elif member == "A" and cell["cell_type"] == "code" and "MEMBER =" in "".join(cell["source"]):
            settings = "".join(cell["source"]).replace('RUN_IDS = ["G0__F1"]', 'RUN_IDS = None')
            settings = settings.replace("RETRY_FAILED = False", "RETRY_FAILED = True")
            cell["source"] = settings.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    notebook["metadata"]["feature_loader_hotfix"] = {"id": "v2-feature-loader-face-v1",
        "sha256": hashlib.sha256(Path(__file__).with_name("campaign_compat_v2.py").read_bytes()).hexdigest()}
    return notebook


GUIDE = '''# Tiếp tục ablation với compatibility v2

Lỗi face thực tế: preprocess tạo F1-face 198 chiều, nhưng Dataset whitelist chỉ có chiều face off.
Run dừng trước seed42 training. Bản này dùng FeatureLayout registry đã có trong source frozen
và kiểm tra face support mask; không đổi feature arrays, model, seeds hay protocol.
Reporting v1 vẫn dùng payload cũ. Hai addon có SHA256 riêng trong kết quả và val_report.json.

## A còn session Kaggle đang bị lỗi

1. Tải results ZIP hiện tại để giữ F1 đã hoàn thành. Không reset session.
2. Mở KAGGLE_RESUME_CELL.py, copy toàn bộ nội dung vào một Code cell mới bên dưới cell bị lỗi.
3. Chạy cell mới đó. Nó cài cả hai addon, giữ queue/cache, gọi full queue với retry_failed=True.
   F1 đã nghiệm thu được bỏ qua. F1-face lỗi kỹ thuật được retry, rồi chạy các configs còn lại.
4. Khi hiện Hoàn tất: 8 / 8, tải lại member_a_77a84a7b12f1_results.zip và Save Version có Output.

Cell này dành cho runtime đã có biến session và WORKSPACE của notebook cũ.
Không dùng khi Output đã trống hoặc kernel mất biến. Trường hợp đó dùng notebook mới bên dưới.
Chạy lại cell nhiều lần vẫn bỏ qua configs hoàn thành và không cài bootstrap trùng.

## Notebook mới và B/C

- A: ablation_v2_member_a_compat_v2.ipynb, mặc định MODE=train, RUN_IDS=None, RETRY_FAILED=True.
- B: ablation_v2_member_b_compat_v2.ipynb, full queue 30 configs.
- C: ablation_v2_member_c_compat_v2.ipynb, full queue 22 configs.

Import đúng notebook, bật GPU + Internet, attach asl-signs và dataset Shared 77a84a7b12f1 hiện có,
rồi Run All. Giữ MEMBER/BRANCH/SOURCE_REF. Không prepare lại Shared.
Nếu B/C đã gặp cùng lỗi whitelist face, dùng recovery cell trên runtime cũ,
hoặc bật RETRY_FAILED=True trong notebook mới sau khi xác nhận đó là lỗi kỹ thuật này.
Nếu còn runtime files, notebook mới dùng lại queue state. Nếu runtime reset, attach results ZIP
và đặt RESTORE_INPUT tới file ZIP trước Run All. RESTORE_INPUT chỉ nhận ZIP; nếu Kaggle unpack
thành folder thì đóng gói folder results thành ZIP trong working rồi đặt đường dẫn file đó.

Source tag ablation-v2-notebooks-v1 và Shared ZIP giữ nguyên. Notebook nhúng hai module bên ngoài
Git checkout và áp dụng cả trong child process. F1 face-off đã nghiệm thu với reporting v1
được giữ lại: bản sửa Dataset không thay đổi dữ liệu/huấn luyện của các chiều cũ.
Các runs mới ghi thêm FEATURE_LOADER_HOTFIX.json và feature_loader_hotfix trong val_report.json.
Hai addon cũng được lưu dưới compatibility/ trong results ZIP để tái lập chính xác mã đã chạy.
Không chọn lại seed/attempt theo điểm; retry chỉ xử lý lỗi kỹ thuật.

A/B/C có thể chạy đồng thời trên runtime riêng. Chỉ đủ phần khi A 8/8, B 30/30, C 22/22.
Gửi full results ZIP để nghiệm thu cùng provenance, aligned predictions và checkpoints.
Logs đơn lẻ không đủ. Test vẫn giữ lại đến khi chọn winner theo Validation Macro-F1.
'''


def find_shared_bytes() -> bytes:
    previous = REPO / "artifacts/ablation-study-v2/reporting-hotfix-v1/ban-giao-member-a-77a84a7b12f1-reportfix-v1.zip"
    with zipfile.ZipFile(previous) as archive:
        matches = [name for name in archive.namelist() if name.endswith("ablation_v2_shared_77a84a7b12f1.zip")]
        if len(matches) != 1:
            raise ValueError("Expected exactly one verified shared ZIP in prior handoff.")
        shared = archive.read(matches[0])
    if hashlib.sha256(shared).hexdigest() != "9b14e9b2e1b9baea9c6ff7348d2071fe504c3da5e128016561ee5e277f99c5f0":
        raise ValueError("Shared archive differs from the reviewed original.")
    return shared


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    recovery = '''# Copy this entire file into one NEW Kaggle Code cell in the existing session.
if "session" not in globals() or "WORKSPACE" not in globals():
    raise RuntimeError("Session variables missing: import the compat_v2 notebook and restore results first.")
'''
    recovery += installer_code()
    recovery += '''RUN_IDS = None
RETRY_FAILED = True
status = session.run_queue(RUN_IDS, retry_failed=RETRY_FAILED)
print("Hoàn tất:", status["completed"], "/", status["expected"], "tổ hợp của member")
print("Results ZIP:", session.export_results())
'''
    compile(recovery, "KAGGLE_RESUME_CELL.py", "exec")
    (OUTPUT / "KAGGLE_RESUME_CELL.py").write_text(recovery, encoding="utf-8")
    (OUTPUT / "HUONG_DAN.md").write_text(GUIDE, encoding="utf-8")
    shared = find_shared_bytes()
    for member in ("A", "B", "C"):
        filename = f"ablation_v2_member_{member.lower()}_compat_v2.ipynb"
        payload = (json.dumps(build(member), ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        (OUTPUT / filename).write_bytes(payload)
        files = {filename: payload, "ablation_v2_shared_77a84a7b12f1.zip": shared,
            "QUEUE.json": (REPO / f"configs/ablation-study-v2/member-{member.lower()}/QUEUE.json").read_bytes(),
            "HUONG_DAN.md": GUIDE.encode("utf-8"), "KAGGLE_RESUME_CELL.py": recovery.encode("utf-8")}
        checksums = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
        files["CHECKSUMS.json"] = (json.dumps(checksums, indent=2) + "\n").encode("utf-8")
        archive_path = OUTPUT / f"ban-giao-member-{member.lower()}-77a84a7b12f1-compat-v2.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        print(archive_path)


if __name__ == "__main__":
    main()
