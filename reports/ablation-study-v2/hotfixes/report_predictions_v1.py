"""Audited prediction-export compatibility for the frozen V2 campaign.

Adds sampler metadata to DeviceDataLoader without changing its iteration.
The frozen checkout is never edited. Training subprocesses run through this
module so validation reports retain the existing strict alignment checks.
"""
from __future__ import annotations

from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys

HOTFIX_ID = "v2-device-loader-predictions-v1"


def descriptor() -> dict:
    return {"id": HOTFIX_ID, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Sequential sampler metadata and prediction export only"}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def install_export_compatibility() -> dict:
    """Expose the actual sequential order; shuffled loaders remain ineligible."""
    from torch.utils.data import SequentialSampler
    from src.data.device_loader import DeviceDataLoader
    import src.training.evaluate as evaluation

    existing = getattr(DeviceDataLoader, "_reporting_hotfix", None)
    if existing is not None and existing != HOTFIX_ID:
        raise RuntimeError("A different reporting compatibility patch is already installed.")
    if existing is None:
        if hasattr(DeviceDataLoader, "sampler"):
            raise RuntimeError("DeviceDataLoader already defines sampler; review compatibility before patching.")

        def sampler(loader):
            return None if loader.shuffle else SequentialSampler(loader.dataset)

        DeviceDataLoader.sampler = property(sampler)
        DeviceDataLoader._reporting_hotfix = HOTFIX_ID

    original = evaluation.evaluate_test_set
    if getattr(original, "_reporting_hotfix", None) == HOTFIX_ID:
        return descriptor()

    @wraps(original)
    def evaluate_with_audit(*args, **kwargs):
        report = original(*args, **kwargs)
        report["reporting_hotfix"] = descriptor()
        bound = inspect.signature(original).bind_partial(*args, **kwargs)
        destination = bound.arguments.get("report_json_path")
        if destination:
            _write_json(Path(destination), report)
        return report

    evaluate_with_audit._reporting_hotfix = HOTFIX_ID
    evaluation.evaluate_test_set = evaluate_with_audit
    return descriptor()


def install(session) -> dict:
    """Install the audited launcher for every training subprocess in this session."""
    import study

    provenance = session.check_shared() if session.shared.exists() else None
    original = study.run_logged
    existing = getattr(original, "_reporting_hotfix", None)
    if existing is not None and existing != HOTFIX_ID:
        raise RuntimeError("A different training launcher patch is already installed.")
    if existing is None:
        launcher = str(Path(__file__).resolve())

        @wraps(original)
        def run_logged_with_reports(command, log_path):
            indices = [index for index, argument in enumerate(command)
                       if Path(str(argument)).as_posix().endswith("scripts/20_train.py")]
            if len(indices) > 1:
                raise ValueError("Ambiguous training command.")
            if indices:
                index = indices[0]
                command = [*command[:index], launcher, "--bootstrap", *command[index:]]
            return original(command, log_path)

        run_logged_with_reports._reporting_hotfix = HOTFIX_ID
        study.run_logged = run_logged_with_reports

    audit = {**descriptor(), "shared_provenance": provenance}
    _write_json(session.root / "REPORTING_HOTFIX.json", audit)
    print("Reporting hotfix:", audit["id"], audit["sha256"])
    return audit


def bootstrap(arguments: list[str]) -> None:
    if not arguments:
        raise ValueError("Bootstrap requires the frozen training script.")
    script = Path(arguments[0]).resolve()
    if script.name != "20_train.py" or script.parent.name != "scripts":
        raise ValueError("Reporting bootstrap only runs scripts/20_train.py.")
    sys.path.insert(0, str(script.parent.parent))
    install_export_compatibility()
    sys.argv = [str(script), *arguments[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    if sys.argv[1:2] != ["--bootstrap"]:
        raise SystemExit("Import install(session), or use --bootstrap scripts/20_train.py.")
    bootstrap(sys.argv[2:])
