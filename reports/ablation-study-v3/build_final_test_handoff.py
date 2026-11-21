"""Build the Kaggle final-test launcher from the accepted full-42 selection."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
AGGREGATE = ROOT / "artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/7f86f6ac76de"


def main() -> None:
    accepted = json.loads((AGGREGATE / "ALL_MEMBERS_FULL42_RESULTS.json").read_text(encoding="utf-8"))
    selection = json.loads((AGGREGATE / "WINNER_SELECTION.json").read_text(encoding="utf-8"))
    if accepted["status"] != "COMPLETED" or accepted["completed_cells"] != 42:
        raise ValueError("Final-test handoff requires the accepted full-42 matrix.")
    existing = nbformat.read(ROOT / "notebooks/ablation_v3_complete_18_kaggle.ipynb", as_version=4)
    settings = '''MODE = "test"  # test hoặc report
SHARED_INPUT = ""  # tự tìm Shared b4fd56f03a57
RAW_INPUT = ""  # tự tìm competition asl-signs
RESULT_INPUTS = {"A": "", "B": "", "C": ""}  # tự tìm ZIP hoặc folder results trong Kaggle Input
'''
    settings += "EXPECTED_ARCHIVE_SHA256 = " + repr(accepted["archive_sha256"]) + "\n"
    settings += "EXPECTED_WINNER = " + repr(selection["winner"]) + "\n"
    settings += "EXPECTED_CHECKPOINTS = " + repr(selection["checkpoints"]) + "\n"
    source = (ROOT / "reports/ablation-study-v3/full_matrix_v3.py").read_text(encoding="utf-8")
    launcher = "ORCHESTRATOR_SOURCE = " + repr(source) + "\n" + '''LAUNCHER_ROOT = Path("/kaggle/working/islr_v3_final_test_launcher")
LAUNCHER_ROOT.mkdir(parents=True, exist_ok=True)
(LAUNCHER_ROOT / "full_matrix_v3.py").write_text(ORCHESTRATOR_SOURCE, encoding="utf-8")
sys.path.insert(0, str(LAUNCHER_ROOT))
from full_matrix_v3 import FullMatrixSession
import study_v3
from src.utils.env import resolve_paths
from IPython.display import display, FileLink
import pandas as pd

if MODE not in ("test", "report"):
    raise ValueError("MODE phải là test hoặc report.")
if not RAW_INPUT:
    candidates = [p.parent for p in SEARCH_ROOT.rglob("train.csv")
                  if (p.parent / "train_landmark_files").is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Tìm thấy {len(candidates)} asl-signs roots. Điền RAW_INPUT tới đúng competition.")
    RAW_INPUT = str(candidates[0])

for member in ("A", "B", "C"):
    if not RESULT_INPUTS[member]:
        suffix = "_v3_full42_results.zip" if member == "B" else "_v3_results.zip"
        name = "member_" + member.lower() + "_" + EXPECTED_FREEZE_ID + suffix
        candidates = list(SEARCH_ROOT.rglob(name))
        if not candidates:
            candidates = []
            for path in SEARCH_ROOT.rglob("queue_status.json"):
                state = json.loads(path.read_text())
                expected = 21 if member == "B" else 7 if member == "A" else 14
                if (state.get("member") == member and state.get("freeze_id") == EXPECTED_FREEZE_ID
                        and len(state.get("selected", {})) == expected and not state.get("failed")):
                    candidates.append(path.parent)
        if len(candidates) != 1:
            raise ValueError(f"Tìm thấy {len(candidates)} results {member}. Điền RESULT_INPUTS[{member!r}] tới đúng ZIP/folder.")
        RESULT_INPUTS[member] = str(candidates[0])
    archive = Path(RESULT_INPUTS[member])
    if archive.is_file():
        with archive.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != EXPECTED_ARCHIVE_SHA256[member]:
            raise ValueError(f"ZIP results {member} khác bản đã nghiệm thu.")

PERSISTENT_ROOT = Path("/kaggle/working")
root_cfg = study_v3.load_config(str(DEPLOY_ROOT / "config.yaml"))
root_cfg.env.local_root = str(PERSISTENT_ROOT)
canonical_root = resolve_paths(root_cfg.env.name, root_cfg, create_dirs=False).persistent_root
if canonical_root.resolve() != PERSISTENT_ROOT.resolve():
    raise ValueError(f"Runtime root {PERSISTENT_ROOT} differs from canonical root {canonical_root}.")
session = FullMatrixSession("A", DEPLOY_ROOT, Path(RAW_INPUT), PERSISTENT_ROOT)
if not session.state_path.exists():
    session.restore_results(Path(RESULT_INPUTS["A"]))
else:
    session.accepted()
'''
    aggregate = '''summary = session.aggregate([Path(RESULT_INPUTS["B"]), Path(RESULT_INPUTS["C"])])
if summary["status"] != "COMPLETED" or summary["completed_cells"] != 42:
    raise RuntimeError("Aggregate cần đủ 42/42 configs trước evaluation.")
if summary["winner"] != EXPECTED_WINNER:
    raise ValueError("Winner khác lựa chọn validation đã nghiệm thu.")
winner = summary["records"][EXPECTED_WINNER]
checkpoints = sorted(Path(winner["path"]).joinpath("runs").rglob("best_model.pt"))
actual = {p.parent.name: study_v3.sha(p) for p in checkpoints}
if actual != EXPECTED_CHECKPOINTS:
    raise ValueError("Checkpoints khác seeds 42/43/44 đã chọn.")
print("Đã nghiệm thu", summary["completed_cells"], "/ 42 configs; winner:", summary["winner"])
display(pd.DataFrame([{ "run_id": r["run_id"], "macro_f1": r["metrics"]["macro_f1"]["mean"] }
                      for r in study_v3.rank(list(summary["records"].values()))]).head(10))
'''
    evaluate = '''if MODE == "test":
    test_result = session.finalize_test(summary)
    print("Test set 1.141 clips, checkpoints seeds 42/43/44")
    display(pd.DataFrame(test_result["metrics"]).T)
    display(FileLink(str(session.root / "FINAL_TEST/TEST_SUMMARY.json")))
results_zip = session.export_results()
display(FileLink(str(results_zip)))
'''
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# Đánh giá test cuối — Ablation V3 full42\n\n"
            "Attach competition `asl-signs`, Shared V3 và ba ZIP results A/B/C. "
            "B dùng `member_b_b4fd56f03a57_v3_full42_results.zip`. Bật GPU rồi **Run All**. "
            "Notebook nghiệm thu đủ 42 configs, kiểm lựa chọn `T0__HF` và hashes ba checkpoints, "
            "sau đó evaluation trên test cohort 1.141 clips."),
        nbformat.v4.new_code_cell(settings), nbformat.v4.new_code_cell(existing.cells[2].source),
        nbformat.v4.new_code_cell(existing.cells[3].source), nbformat.v4.new_code_cell(launcher),
        nbformat.v4.new_code_cell(aggregate), nbformat.v4.new_code_cell(evaluate),
        nbformat.v4.new_markdown_cell("Tải Results ZIP A mới có `FINAL_TEST/TEST_SUMMARY.json`, "
            "`WINNER_LOCK.json` và ba thư mục seed chứa test reports. "
            "MODE `report` dùng để nghiệm thu và xem ranking trước evaluation.")])
    notebook.metadata = dict(existing.metadata)
    notebook.metadata["islr_scope"] = {"scope": "v3-full42-final-test", "freeze_id": selection["freeze_id"],
                                      "winner": selection["winner"], "checkpoints": selection["checkpoints"]}
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    path = ROOT / "notebooks/ablation_v3_final_test_kaggle.ipynb"
    nbformat.write(notebook, path)
    print(path)


if __name__ == "__main__":
    main()
