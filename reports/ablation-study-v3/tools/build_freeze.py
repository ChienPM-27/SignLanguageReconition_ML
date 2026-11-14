"""Build canonical configs, source snapshot, shared ZIP and three Kaggle launchers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import study_v3 as study
from src.utils.config import save_config


BOOTSTRAP = '''from pathlib import Path
import hashlib, json, sys, zipfile

EXPECTED_FREEZE_ID = "__FREEZE_ID__"
SEARCH_ROOT = Path("/kaggle/input")
if not SHARED_INPUT:
    candidates = list(SEARCH_ROOT.rglob("ablation_v3_shared_" + EXPECTED_FREEZE_ID + ".zip"))
    candidates += [p.parent for p in SEARCH_ROOT.rglob("freeze.json")
                   if json.loads(p.read_text()).get("freeze_id") == EXPECTED_FREEZE_ID]
    if len(candidates) != 1:
        raise ValueError(f"Tìm thấy {len(candidates)} Shared V3 inputs. Đặt SHARED_INPUT vào ZIP hoặc thư mục có freeze.json.")
    SHARED_INPUT = str(candidates[0])
source = Path(SHARED_INPUT)
if source.is_file():
    with zipfile.ZipFile(source) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or archive.testzip():
            raise ValueError("Shared ZIP không hợp lệ.")
        payloads = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
else:
    payloads = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()}
frozen = json.loads(payloads["freeze.json"])
actual = {name: hashlib.sha256(payloads[name]).hexdigest() for name in frozen["files"]}
identity = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()[:12]
if actual != frozen["files"] or identity != EXPECTED_FREEZE_ID or frozen["freeze_id"] != identity:
    raise ValueError("Shared khác freeze của notebook này.")
manifest = json.loads(payloads["source_manifest.json"])
for name, checksum in manifest.items():
    if hashlib.sha256(payloads["code/" + name].replace(b"\\r\\n", b"\\n")).hexdigest() != checksum:
        raise ValueError(f"Source hash lệch: {name}")
DEPLOY_ROOT = Path("/kaggle/working") / ("islr_ablation_v3_" + identity)
for name, content in payloads.items():
    target = (DEPLOY_ROOT / name).resolve()
    if not target.is_relative_to(DEPLOY_ROOT.resolve()) or "\\\\" in name or ":" in name:
        raise ValueError("Unsafe Shared path.")
    if target.exists() and target.read_bytes() != content:
        raise ValueError(f"Runtime source/input bị sửa: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
CODE_ROOT = DEPLOY_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "reports/ablation-study-v3/tools"))
print("Đã xác minh source, config, matrix, split:", identity)
'''


def notebook(member: str, freeze_id: str) -> dict:
    def cell(kind: str, source: str) -> dict:
        value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        return value
    cells = [cell("markdown", f"# Member {member} - Ablation V3\n\nAttach `asl-signs` và Shared V3; bật GPU rồi Run All. Source nằm trong Shared, không cần checkout branch. Các results V2 được lưu riêng; V3 train từ đầu. Sau mỗi config, tải Results ZIP hoặc Save Version giữ Output.\n"),
             cell("code", f'MEMBER = "{member}"\nMODE = "train"  # train / report / aggregate / test (A)\nSHARED_INPUT = ""  # tự tìm Shared V3 ZIP hoặc thư mục đã giải nén\nRAW_INPUT = "/kaggle/input/asl-signs"\nRESTORE_INPUT = ""  # ZIP results V3 của chính member để tiếp tục configs còn thiếu\nCROSS_PLAN_INPUT = ""  # B: cross_plan.zip từ A, sau vòng đầu\nRESULT_INPUTS = []  # A: paths tới ZIP results B/C khi MODE="aggregate"\nRUN_IDS = None  # None chạy cả queue; hoặc list run IDs\nRETRY_FAILED = False\n'),
             cell("code", BOOTSTRAP.replace("__FREEZE_ID__", freeze_id)),
             cell("code", 'import subprocess, importlib.util\nrequired = {"yaml": "pyyaml", "pyarrow": "pyarrow", "sklearn": "scikit-learn", "matplotlib": "matplotlib", "pandas": "pandas", "tqdm": "tqdm"}\nmissing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]\nif missing:\n    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])\nimport torch\nprint("Torch:", torch.__version__, "GPU:", torch.cuda.is_available())\nif MODE == "train" and not torch.cuda.is_available():\n    raise RuntimeError("Bật GPU trong Kaggle Settings trước khi training.")\n'),
             cell("code", 'from session_v3 import MemberSession\nimport pandas as pd\nsession = MemberSession(MEMBER, DEPLOY_ROOT, Path(RAW_INPUT), Path("/kaggle/working"))\nif RESTORE_INPUT:\n    session.restore_results(Path(RESTORE_INPUT))\nif CROSS_PLAN_INPUT:\n    session.load_cross_plan(Path(CROSS_PLAN_INPUT))\nif MODE == "train":\n    session.check_data()\ndisplay(pd.DataFrame(session.report()["queue"]))\n'),
             cell("code", 'if MODE == "train":\n    status = session.run_queue(RUN_IDS, retry_failed=RETRY_FAILED)\n    display(pd.DataFrame(status["queue"]))\n    print("Hoàn tất:", status["completed"], "/", status["expected"])\nelif MODE in ("aggregate", "test"):\n    result = session.aggregate([Path(p) for p in RESULT_INPUTS])\n    print(result["status"], "winner:", result["winner"], "missing:", result["missing"])\n    if MODE == "test":\n        print(session.finalize_test(result))\nelif MODE == "report":\n    display(pd.DataFrame(session.report()["queue"]))\nelse:\n    raise ValueError("MODE phải là train/report/aggregate/test.")\n'),
             cell("code", 'results_zip = session.export_results()\nfrom IPython.display import FileLink, display\ndisplay(FileLink(str(results_zip)))\nif (session.root / "cross_plan.zip").is_file():\n    display(FileLink(str(session.root / "cross_plan.zip")))\n'),
             cell("markdown", "## Nghiệm thu\n\nMột config hoàn tất phải đủ 3 seed reports, predictions đúng cohort, checkpoints, history, metrics và stage logs. ZIP được tạo sau mỗi config; Kaggle reset sẽ mất Output chưa tải/Save Version. B giữ ZIP vòng đầu để restore khi chạy cross-plan. A tổng hợp đủ B/C, gửi cross-plan cho B nếu cần, rồi nghiệm thu lần cuối. Test giữ kín đến khi khóa winner.\n")]
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}}, "cells": [{**c, "id": f"v3-{member.lower()}-{i}"} for i, c in enumerate(cells)]}


def build(old_shared: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    shared = output / "shared"
    shared.mkdir(exist_ok=True)
    with zipfile.ZipFile(old_shared) as archive:
        for name in ("split_manifest.json", "selection/glosses.csv", "config.yaml"):
            target = shared / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    base = study.config_for(shared / "config.yaml", "G0", "HBF")
    base.env.project_dir_name = "asl-signs-ablation-v3"
    save_config(base, shared / "config.yaml")
    config_root = study.REPO / "configs/ablation-study-v3"
    for member in ("A", "B", "C"):
        folder = config_root / ("member-" + member.lower())
        folder.mkdir(parents=True, exist_ok=True)
        queue = [r for r in study.matrix() if r["owner"] == member]
        for row in queue:
            save_config(study.config_for(shared / "config.yaml", row["profile"], row["feature"]), folder / row["config"])
        study.write(folder / "QUEUE.json", queue)
    protocol = {"campaign": "v3-regions-701020", "weights": base.model.region_weights,
                "feature_groups": study.REGIONS, "phase1_cells": 24, "phase1_fits": 72,
                "max_required_cells": 30, "max_required_fits": 90, "eligible_cells": 42,
                "cross_rule": "Choose GRU architecture by full-region validation; complete its six remaining region subsets if not G0.",
                "winner_rule": "Validation Macro-F1 mean, lower population std, fewer parameters, run ID; only after phase1 and selected cross-plan complete.",
                "test_rule": "Hold out test until winner locked; evaluate existing three checkpoints without retraining.",
                "old_shared_sha256": study.sha(old_shared), "old_results_reuse": False,
                "cohort": "Keep the previous common cohort and signer split; disable hand-dependent trimming and torso-relative hand normalization."}
    study.write(shared / "matrix.json", study.matrix())
    study.write(shared / "protocol.json", protocol)
    manifest = study.source_manifest()
    study.write(shared / "source_manifest.json", manifest)
    hashes = {name: study.sha(shared / name) for name in study.SHARED_FILES if name != "freeze.json"}
    freeze_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]
    study.write(shared / "freeze.json", {"campaign": protocol["campaign"], "freeze_id": freeze_id, "files": hashes})
    study.verify_shared(shared)
    shared_zip = output / ("ablation_v3_shared_" + freeze_id + ".zip")
    with zipfile.ZipFile(shared_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in study.SHARED_FILES:
            archive.write(shared / name, name)
        for name in manifest:
            archive.writestr("code/" + name, (study.REPO / name).read_bytes().replace(b"\r\n", b"\n"))
    launchers = []
    for member in ("A", "B", "C"):
        name = f"ablation_v3_member_{member.lower()}_kaggle.ipynb"
        study.write(output / name, notebook(member, freeze_id))
        notebook_dir = study.REPO / "notebooks"
        study.write(notebook_dir / name, notebook(member, freeze_id))
        package = output / f"ban-giao-member-{member.lower()}-{freeze_id}-v3.zip"
        with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(shared_zip, shared_zip.name)
            archive.write(output / name, name)
            for row in study.matrix():
                if row["owner"] == member:
                    archive.write(config_root / ("member-" + member.lower()) / row["config"], "configs/" + row["config"])
            if (output / "HUONG_DAN.md").is_file():
                archive.write(output / "HUONG_DAN.md", "HUONG_DAN.md")
            if (output / "KE_HOACH_ABLATION_V3.pdf").is_file():
                archive.write(output / "KE_HOACH_ABLATION_V3.pdf", "KE_HOACH_ABLATION_V3.pdf")
        launchers.append(str(package.resolve()))
    result = {"freeze_id": freeze_id, "source_files": len(manifest), "shared_zip": str(shared_zip.resolve()),
              "shared_zip_sha256": study.sha(shared_zip), "handoffs": launchers, "phase1_cells": 24, "max_required_cells": 30}
    study.write(output / "FREEZE_STATUS.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-shared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.old_shared, args.output), ensure_ascii=False, indent=2))
