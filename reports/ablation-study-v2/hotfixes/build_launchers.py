"""Build transport notebooks with an embedded, audited reporting compatibility patch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

DISCOVERY = '''from pathlib import Path

if MODE == "train" and SHARED_INPUT is None and Path("/kaggle/input").exists():
    root = Path("/kaggle/input")
    required = ["config.yaml", "split_manifest.json", "source_manifest.json",
                "matrix.json", "selection/glosses.csv"]
    candidates = {p.parent.resolve() for p in root.rglob("source_manifest.json")
                  if all((p.parent / name).is_file() for name in required)}
    if not candidates:
        candidates = {p.resolve() for p in root.rglob("ablation_v2_shared*.zip")}
    print("Shared tìm thấy:", sorted(map(str, candidates)))
    if len(candidates) != 1:
        print("Input hiện có:", list(root.iterdir()))
        raise RuntimeError("Cần tìm được đúng một shared package V2.")
    SHARED_INPUT = str(next(iter(candidates)))
print("SHARED_INPUT =", SHARED_INPUT)
'''


def cell(kind: str, identity: str, text: str) -> dict:
    result = {"cell_type": kind, "id": identity, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(member: str) -> dict:
    source = REPO / f"notebooks/ablation_v2_member_{member.lower()}.ipynb"
    notebook = json.loads(source.read_text(encoding="utf-8"))
    original = notebook["cells"]
    original[0]["source"] += ["\n\n**Bản chạy có reporting hotfix v1.** Tự tìm đường dẫn Shared trên Kaggle. "
                               "Bản sửa chỉ bổ sung prediction export; source/model/features/protocol đã khóa giữ nguyên.\n"]
    settings = "".join(original[2]["source"])
    if member == "A":
        settings = settings.replace('MODE = "prepare"', 'MODE = "train"')
        settings = settings.replace('RUN_IDS = None', 'RUN_IDS = ["G0__F1"]')
    original[2]["source"] = settings.splitlines(keepends=True)
    payload = Path(__file__).with_name("report_predictions_v1.py").read_text(encoding="utf-8")
    payload_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    code = 'import importlib.util\nimport sys\nfrom pathlib import Path\n\n'
    code += "HOTFIX_SOURCE = " + repr(payload) + "\n"
    code += '''HOTFIX_PATH = WORKSPACE / "ablation_v2_reporting_hotfix" / "report_predictions_v1.py"
HOTFIX_PATH.parent.mkdir(parents=True, exist_ok=True)
hotfix_payload = HOTFIX_SOURCE.encode("utf-8")
if HOTFIX_PATH.exists() and HOTFIX_PATH.read_bytes() != hotfix_payload:
    raise RuntimeError("Reporting hotfix hiện có khác nội dung notebook; cần kiểm tra phiên bản.")
HOTFIX_PATH.write_bytes(hotfix_payload)
hotfix_spec = importlib.util.spec_from_file_location("ablation_v2_reporting_hotfix", HOTFIX_PATH)
reporting_hotfix = importlib.util.module_from_spec(hotfix_spec)
sys.modules[hotfix_spec.name] = reporting_hotfix
hotfix_spec.loader.exec_module(reporting_hotfix)
reporting_hotfix.install(session)
print("Source/config/cohort được giữ nguyên; reporting hotfix đã áp dụng cho training subprocess.")
'''
    notebook["cells"] = [*original[:8],
        cell("markdown", "shared-discovery-guide", "### Tự tìm Shared Input\n\nCell này xử lý folder mount khác nhau trên Kaggle.\n"),
        cell("code", "shared-discovery-code", DISCOVERY), original[8],
        cell("markdown", "reporting-hotfix-guide", "### Reporting hotfix v1\n\n"
             "Bổ sung sampler metadata để DeviceDataLoader xuất prediction theo sequence. "
             "Giữ nguyên thứ tự batch, model, features và seeds. Audit nằm trong val_report.json "
             "và REPORTING_HOTFIX.json. Source gốc được pin theo tag ablation-v2-notebooks-v1.\n"),
        cell("code", "reporting-hotfix-code", code), *original[9:]]
    notebook["metadata"]["reporting_hotfix"] = {"id": "v2-device-loader-predictions-v1", "sha256": payload_sha}
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            item["execution_count"], item["outputs"] = None, []
            compile("".join(item["source"]), item["id"], "exec")
    return notebook


def main() -> None:
    output = REPO / "artifacts/ablation-study-v2/reporting-hotfix-v1"
    output.mkdir(parents=True, exist_ok=True)
    for member in ("A", "B", "C"):
        target = output / f"ablation_v2_member_{member.lower()}_reportfix.ipynb"
        target.write_text(json.dumps(build(member), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
