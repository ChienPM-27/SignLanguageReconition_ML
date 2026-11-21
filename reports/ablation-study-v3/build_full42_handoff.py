"""Build the standalone notebook and exact 18-YAML V3 completion handoff."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from shutil import copy2
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[2]
FREEZE = "b4fd56f03a57"
BASE = ROOT / "artifacts/ablation-study-v3"
OUTPUT = BASE / "supplements/full42-member-b"
PROFILES = ("G-mean", "G-attn", "G-bi")
FEATURES = ("H", "B", "F", "HB", "HF", "BF")
RUN_IDS = [p + "__" + f for p in PROFILES for f in FEATURES]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "reports/ablation-study-v3/full_matrix_v3.py"
    source_text = source.read_text(encoding="utf-8")
    existing = nbformat.read(ROOT / "notebooks/ablation_v3_member_b_kaggle.ipynb", as_version=4)
    bootstrap = existing.cells[2].source
    start = bootstrap.index("if not SHARED_INPUT:")
    end = bootstrap.index("source = Path(SHARED_INPUT)")
    bootstrap = bootstrap[:start] + '''if not SHARED_INPUT:
    candidates = list(SEARCH_ROOT.rglob("ablation_v3_shared_" + EXPECTED_FREEZE_ID + ".zip"))
    if not candidates:
        candidates = [p.parent for p in SEARCH_ROOT.rglob("freeze.json")
                      if (p.parent / "code/requirements.txt").is_file()
                      and json.loads(p.read_text()).get("freeze_id") == EXPECTED_FREEZE_ID]
    if len(candidates) != 1:
        raise ValueError(f"Tìm thấy {len(candidates)} Shared V3 inputs. Điền SHARED_INPUT tới đúng ZIP/thư mục Shared.")
    SHARED_INPUT = str(candidates[0])
''' + bootstrap[end:]
    settings = '''MODE = "train"  # train hoặc report
SHARED_INPUT = ""  # tự tìm Shared b4fd56f03a57
RAW_INPUT = ""  # tự tìm asl-signs; hoặc điền folder có train.csv và train_landmark_files/
RESTORE_INPUT = ""  # tự tìm results B trong Input; hỗ trợ FILE ZIP hoặc folder đã giải nén
RETRY_FAILED = False  # chỉ bật sau khi đã xử lý lỗi training trước đó
''' + "RUN_IDS = " + json.dumps(RUN_IDS, indent=4) + "\n"
    launcher = "ORCHESTRATOR_SOURCE = " + repr(source_text) + "\n" + '''LAUNCHER_ROOT = Path("/kaggle/working/islr_v3_full42_launcher")
LAUNCHER_ROOT.mkdir(parents=True, exist_ok=True)
launcher_file = LAUNCHER_ROOT / "full_matrix_v3.py"
launcher_file.write_text(ORCHESTRATOR_SOURCE, encoding="utf-8")
sys.path.insert(0, str(LAUNCHER_ROOT))
from full_matrix_v3 import FullMatrixSession
import study_v3
import pandas as pd
from IPython.display import display, FileLink

if not RAW_INPUT:
    default = Path("/kaggle/input/competitions/asl-signs")
    candidates = [default] if (default / "train.csv").is_file() and (default / "train_landmark_files").is_dir() else [
        p.parent for p in SEARCH_ROOT.rglob("train.csv") if (p.parent / "train_landmark_files").is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Tìm thấy {len(candidates)} asl-signs roots. Điền RAW_INPUT tới đúng competition.")
    RAW_INPUT = str(candidates[0])

if not RESTORE_INPUT:
    candidates = list(SEARCH_ROOT.rglob("member_b_" + EXPECTED_FREEZE_ID + "_v3_full42_results.zip"))
    if not candidates:
        candidates = list(SEARCH_ROOT.rglob("member_b_" + EXPECTED_FREEZE_ID + "_v3_results.zip"))
    if not candidates:
        candidates = []
        for path in SEARCH_ROOT.rglob("queue_status.json"):
            state = json.loads(path.read_text())
            if state.get("member") == "B" and state.get("freeze_id") == EXPECTED_FREEZE_ID:
                candidates.append(path.parent)
    if len(candidates) != 1:
        raise ValueError(f"Tìm thấy {len(candidates)} results B inputs. Attach ZIP results B trong gói hoặc điền RESTORE_INPUT.")
    RESTORE_INPUT = str(candidates[0])

PERSISTENT_ROOT = Path("/kaggle/working")
from src.utils.env import resolve_paths
root_cfg = study_v3.load_config(str(DEPLOY_ROOT / "config.yaml"))
root_cfg.env.local_root = str(PERSISTENT_ROOT)
canonical_root = resolve_paths(root_cfg.env.name, root_cfg, create_dirs=False).persistent_root
if canonical_root.resolve() != PERSISTENT_ROOT.resolve():
    raise ValueError(f"Runtime root {PERSISTENT_ROOT} differs from canonical root {canonical_root}.")
session = FullMatrixSession("B", DEPLOY_ROOT, Path(RAW_INPUT), PERSISTENT_ROOT)
if not session.state_path.exists():
    session.restore_results(Path(RESTORE_INPUT))
else:
    session.accepted()
expected_baselines = {p + "__HBF" for p in ("G-mean", "G-attn", "G-bi")}
if not expected_baselines.issubset(session.state["selected"]):
    raise ValueError("Results restore phải có đủ 3 config HBF đã hoàn thành của B.")
expected_remaining = {r["run_id"] for r in session.queue()} - expected_baselines
if set(RUN_IDS) != expected_remaining or len(RUN_IDS) != 18:
    raise ValueError("Notebook này phải chọn đúng toàn bộ 18 configs còn lại của B.")
for row in session.queue():
    cfg = study_v3.load_config(str(CODE_ROOT / "configs/ablation-study-v3/member-b" / row["config"]))
    study_v3.validate_config(cfg, study_v3.config_for(DEPLOY_ROOT / "config.yaml", row["profile"], row["feature"]))
status = session.report()
print("Đã có:", status["completed"], "/ 21 configs B; còn:", len(status["remaining"]))
print("Phạm vi toàn bộ nhóm: A7 + B21 + C14 = 42 configs / 126 fits.")
display(pd.DataFrame(status["queue"])[["run_id", "status"]])
if MODE == "train":
    session.check_data()
'''
    training = '''if MODE == "train":
    status = session.run_queue(RUN_IDS, retry_failed=RETRY_FAILED)
elif MODE == "report":
    status = session.report()
else:
    raise ValueError("MODE phải là train hoặc report.")
display(pd.DataFrame(status["queue"])[["run_id", "status"]])
print("Hoàn tất B:", status["completed"], "/ 21 configs")
if MODE == "train" and status["completed"] != 21:
    raise RuntimeError("Chưa hoàn thành cả 21 configs B; xem remaining/failed trước khi kết thúc.")
results_zip = session.export_results()
display(FileLink(str(results_zip)))
'''
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# Chạy đủ 18 configs còn thiếu — Ablation V3\n\n"
            "Attach competition `asl-signs`, Shared V3 và ZIP results B trong gói; bật GPU rồi **Run All**. "
            "Notebook tự restore 3 configs HBF đã có, chạy đủ 18 configs còn lại (54 fits), cập nhật ZIP sau mỗi config. "
            "Không cần cross_plan. Giữ nguyên source/config/split/seeds của freeze `b4fd56f03a57`."),
        nbformat.v4.new_code_cell(settings), nbformat.v4.new_code_cell(bootstrap),
        nbformat.v4.new_code_cell(existing.cells[3].source.replace('required = {', 'required = {"IPython": "ipython", ')),
        nbformat.v4.new_code_cell(launcher), nbformat.v4.new_code_cell(training),
        nbformat.v4.new_markdown_cell("## Kết quả cần tải\n\n"
            "Đích cuối là **B21/21 configs, 63 fits**, trong đó 9 fits HBF đã restore và 54 fits mới. "
            "Tải `member_b_b4fd56f03a57_v3_full42_results.zip` hoặc Save Version giữ Output. "
            "Nếu hết phiên GPU, attach ZIP mới nhất vào phiên mới và Run All; configs COMPLETED tự được bỏ qua. "
            "A/C không cần train lại. Giữ test kín; tổng hợp full42 bằng `FullMatrixSession.aggregate`, "
            "không dùng chế độ aggregate 24/30 của notebook cũ.")])
    notebook.metadata = dict(existing.metadata)
    notebook.metadata["islr_scope"] = {"freeze_id": FREEZE, "scope": "v3-full42", "missing_run_ids": RUN_IDS}
    nbformat.validate(notebook)
    notebook_path = OUTPUT / "ablation_v3_complete_18_kaggle.ipynb"
    nbformat.write(notebook, notebook_path)
    copy2(notebook_path, ROOT / "notebooks" / notebook_path.name)
    copy2(source, OUTPUT / source.name)
    shared = BASE / f"ablation_v3_shared_{FREEZE}.zip"
    restore = BASE / f"results/member-b/{FREEZE}/d2d48b98651a/member_b_{FREEZE}_v3_results.zip"
    for path in (shared, restore):
        target = OUTPUT / path.name
        if not target.exists():
            copy2(path, target)
        if sha(target) != sha(path):
            raise ValueError("Existing handoff input differs from immutable source.")
    (OUTPUT / "configs").mkdir(exist_ok=True)
    with zipfile.ZipFile(shared) as bundle:
        for run_id in RUN_IDS:
            filename = run_id + ".yaml"
            frozen = bundle.read("code/configs/ablation-study-v3/member-b/" + filename)
            current = (ROOT / "configs/ablation-study-v3/member-b" / filename).read_bytes()
            if current.replace(b"\r\n", b"\n") != frozen.replace(b"\r\n", b"\n"):
                raise ValueError("Config differs from frozen training config: " + filename)
            (OUTPUT / "configs" / filename).write_bytes(frozen)
    guide = """# Bạn tự chạy đủ configs còn lại

Thiếu đúng 18 configs/54 fits của B: G-mean, G-attn, G-bi × H/B/F/HB/HF/BF.
A đã đủ 7/7, C đủ 14/14. B có 3/21 configs HBF; notebook restore chúng, không train lại.
Đích toàn nhóm: 42 configs/126 fits. Source training, config, cohort, seeds giữ nguyên freeze b4fd56f03a57.
Phạm vi chạy được mở rộng thành full42 theo yêu cầu ngày 20/11/2026; runner bổ sung ghi FULL_MATRIX_SCOPE.json.

1. Giải nén gói; import ablation_v3_complete_18_kaggle.ipynb vào Kaggle.
2. Tạo một Kaggle Dataset từ Shared ZIP và ZIP results B có sẵn trong gói; Add Input dataset đó và competition asl-signs.
   Nếu Kaggle giải nén ZIP thành folder, notebook cũng hỗ trợ nhận diện Shared/results folder.
3. Bật GPU. Giữ các paths rỗng để tự tìm, rồi Run All. Nếu có nhiều phiên bản Input, điền SHARED_INPUT/RESTORE_INPUT chính xác.
4. Trước training, bảng phải có 21 configs B: 3 COMPLETED + 18 PENDING. Notebook chọn tường minh cả 18 run IDs.
5. Sau mỗi config, ZIP kết quả được cập nhật. Tải ZIP hoặc Save Version giữ Output, nhất là trước khi hết GPU/session.
6. Nếu reset, attach ZIP full42 mới nhất và dùng notebook này tiếp; configs đã COMPLETED tự được bỏ qua.
7. Kết thúc phải là B21/21. Tải member_b_b4fd56f03a57_v3_full42_results.zip trong
   /kaggle/working/ablation_v3_results/member-b/.

configs/ chứa đúng 18 YAML còn thiếu, byte-identical với config trong Shared. Notebook dùng các config đã kiểm hash trong Shared.
Không cần upload configs riêng nếu đã attach Shared. Không đổi hyperparameters cho đợt bổ sung để giữ khả năng so sánh A/B/C.
Không dùng cross_plan.zip hoặc notebook B cũ để chạy phần này: chúng chỉ mở 6 configs của variant được chọn.
full_matrix_v3.py là orchestration bổ sung, không sửa source training đã freeze.
Sau khi đủ B21, dùng FullMatrixSession cho A restore ZIP A và aggregate ZIP B full42 cùng C để nghiệm thu42/42.
Không dùng aggregate cũ (scope24/30), vì nó sẽ từ chối các configs mở rộng. Test vẫn giữ kín tới khi đủ42 và khóa winner.

Notebook đã sửa persistent root về /kaggle/working để khớp env.resolve_paths trên Kaggle.
Nếu đang ở phiên notebook cũ và gặp thiếu ablation_v3_full42/data/splits/train.csv,
chạy cell phục hồi recover_full42_split_cell.py trong cùng kernel: cell giữ cache/results và retry run lỗi.
"""
    (OUTPUT / "HUONG_DAN.md").write_text(guide, encoding="utf-8")
    recovery = ROOT / "reports/ablation-study-v3/recover_full42_split_cell.py"
    copy2(recovery, OUTPUT / recovery.name)
    ledger = {"freeze_id": FREEZE, "scope": "v3-full42", "missing_configs": 18, "new_fits": 54,
              "existing_b_configs": 3, "target_b_configs": 21, "target_all_configs": 42,
              "run_ids": RUN_IDS, "files": {p.relative_to(OUTPUT).as_posix(): sha(p)
                for p in sorted(OUTPUT.rglob("*")) if p.is_file() and p.name not in ("MANIFEST.json", "VALIDATION.json")}}
    (OUTPUT / "MANIFEST.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    transport = BASE / "ablation_v3_complete_18_configs.zip"
    with zipfile.ZipFile(transport, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_file() and path.name != "VALIDATION.json":
                bundle.write(path, path.relative_to(OUTPUT).as_posix())
    configs_zip = BASE / "ablation_v3_missing_18_yaml.zip"
    with zipfile.ZipFile(configs_zip, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted((OUTPUT / "configs").glob("*.yaml")):
            bundle.write(path, path.name)
    print(json.dumps({"notebook": str(notebook_path), "configs": str(OUTPUT / "configs"),
                      "configs_zip": str(configs_zip), "zip": str(transport), "bytes": transport.stat().st_size,
                      "sha256": sha(transport), "missing_configs": len(RUN_IDS)}, indent=2))


if __name__ == "__main__":
    main()
