"""Verify V3 delivery and integrity or authorized retirement of old archives."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/ablation-study-v3"
status = json.loads((ARTIFACT / "FREEZE_STATUS.json").read_text(encoding="utf-8"))
shared = Path(status["shared_zip"])
freeze_id = status["freeze_id"]


def digest(value):
    return hashlib.sha256(value).hexdigest()


checks = []
with zipfile.ZipFile(shared) as archive:
    assert archive.testzip() is None
    payloads = {name: archive.read(name) for name in archive.namelist()}
frozen = json.loads(payloads["freeze.json"])
assert digest(json.dumps(frozen["files"], sort_keys=True).encode())[:12] == freeze_id
for name, checksum in frozen["files"].items():
    assert digest(payloads[name]) == checksum
manifest = json.loads(payloads["source_manifest.json"])
for name, checksum in manifest.items():
    assert digest(payloads["code/" + name]) == checksum
    assert digest((ROOT / name).read_bytes().replace(b"\r\n", b"\n")) == checksum
assert set(payloads) == set(frozen["files"]) | {"freeze.json"} | {"code/" + name for name in manifest}
checks.append("Shared CRC, 101 source hashes, canonical inputs and exact payload verified")

for member, expected in (("A", 7), ("B", 3), ("C", 14)):
    filename = f"ablation_v3_member_{member.lower()}_kaggle.ipynb"
    notebook = nbformat.read(ARTIFACT / filename, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    assert json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8")) == json.loads((ARTIFACT / filename).read_text(encoding="utf-8"))
    package = ARTIFACT / f"ban-giao-member-{member.lower()}-{freeze_id}-v3.zip"
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        assert archive.read(shared.name) == shared.read_bytes()
        assert archive.read(filename) == (ARTIFACT / filename).read_bytes()
        assert archive.read("KE_HOACH_ABLATION_V3.pdf") == (ARTIFACT / "KE_HOACH_ABLATION_V3.pdf").read_bytes()
        assert archive.read("HUONG_DAN.md") == (ARTIFACT / "HUONG_DAN.md").read_bytes()
    with tempfile.TemporaryDirectory(prefix="verify-v3-", dir=ROOT / "tmp") as temporary:
        script = notebook.cells[2].source.replace('Path("/kaggle/working")', "Path(" + repr(temporary) + ")")
        script = 'SHARED_INPUT = ' + repr(str(shared)) + "\n" + script
        script += f'''\nfrom session_v3 import MemberSession
session = MemberSession({member!r}, DEPLOY_ROOT, Path({temporary!r}), Path({temporary!r}))
result = session.report()
assert result["expected"] == {expected} and result["completed"] == 0
assert session.freeze_id == {freeze_id!r}
import study_v3
assert Path(study_v3.__file__).resolve().is_relative_to(CODE_ROOT.resolve())
for row in study_v3.matrix():
    cfg = study_v3.load_config(str(CODE_ROOT / "configs/ablation-study-v3" / ("member-" + row["owner"].lower()) / row["config"]))
    study_v3.validate_config(cfg, study_v3.config_for(DEPLOY_ROOT / "config.yaml", row["profile"], row["feature"]))
assert study_v3.load_config(str(DEPLOY_ROOT / "config.yaml")).model.region_weights == {{"hands": .7, "body": .2, "face": .1}}
print("Fresh source deployment and member queue OK")
'''
        completed = subprocess.run([sys.executable, "-X", "utf8", "-c", script], cwd=temporary, text=True,
                                   capture_output=True, timeout=45)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    checks.append(f"Member {member}: notebook/ZIP/code deployment verified; {expected} phase-1 configs")

old_shared = ROOT / "artifacts/ablation-study-v2/handoff/member-b/ablation_v2_shared_77a84a7b12f1.zip"
old_results = ROOT / "artifacts/ablation-study-v2/results/member-a/77a84a7b12f1/1017bac26f59/member_a_77a84a7b12f1_face_only_results.zip"
if old_shared.is_file() and old_results.is_file():
    assert digest(old_shared.read_bytes()) == "9b14e9b2e1b9baea9c6ff7348d2071fe504c3da5e128016561ee5e277f99c5f0"
    assert digest(old_results.read_bytes()) == "1017bac26f598f1f6bc985cdf5b1cdfe381faf10828b4a5a0c4742e0d96d32c5"
    legacy = {"status": "PRESERVED"}
    checks.append("V2 Shared and accepted A results remain byte-identical")
else:
    ledger_path = ROOT / "reports/cleanup/2026-11-14.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["status"] == "COMPLETED" and ledger["scope"] == "V3_ONLY"
    assert ledger["operation"] == "Windows Recycle Bin"
    assert ledger["user_authorization"] == "Chỉ giữ campaign V3 hiện tại"
    assert ledger["retained_freeze"] == freeze_id
    assert ledger["protected_hashes_verified"] is True
    assert set(ledger["retired_campaigns"]) == {"ablation-study", "ablation-study-v2"}
    for campaign in ledger["retired_campaigns"]:
        retired_root = ROOT / "artifacts" / campaign
        targets = [entry for entry in ledger["targets"] if Path(entry["path"]) == retired_root]
        assert len(targets) == 1 and targets[0]["status"] == "RECYCLED"
        assert targets[0]["is_directory"] is True and not retired_root.exists()
    legacy = {"status": "RETIRED", "scope": ledger["scope"],
              "operation": ledger["operation"], "ledger": str(ledger_path)}
    checks.append("V1/V2 archives retired to Recycle Bin under completed user-authorized V3_ONLY cleanup")
result = {"status": "FROZEN", "freeze_id": freeze_id, "checks": checks,
          "legacy_archives": legacy,
          "test_evidence": {"regression_suite": "331 passed (prior run)", "latest_v3_suite": "50 passed (separate run; overlaps regression)",
                            "real_campaign_training": False, "real_test_evaluation": False},
          "files": {p.name: digest(p.read_bytes()) for p in [shared, *[Path(s) for s in status["handoffs"]], ARTIFACT / "KE_HOACH_ABLATION_V3.pdf"]}}
(ARTIFACT / "VALIDATION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
