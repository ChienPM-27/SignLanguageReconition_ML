"""Audited loader/report compatibility for the frozen ablation V2 campaign.

Accept dimensions already defined by the frozen FeatureLayout registry, and
validate the face support mask. No feature array or frozen checkout is edited.
The independent reporting v1 addon retains its original identity and payload.
"""
from __future__ import annotations

from functools import wraps
import hashlib
import inspect
from pathlib import Path
import runpy
import sys

import report_predictions_v1 as reporting

HOTFIX_ID = "v2-feature-loader-face-v1"
LAUNCHER_ID = "v2-loader-and-reporting-compat-v2"
FROZEN_LOADER_SHA256 = "eb11a1103727d8ce42c63ce5aec830fce2839d2050995c5974236ec4710fd1ac"
STALE_GUARD = '''    ALLOWED_FEATURE_DIMS = frozenset({143, 160, 162, 163, 173, 180, 182, 190, 192, 193, 210, 212})
    if d_cur not in ALLOWED_FEATURE_DIMS:
        raise ValueError(
            f"Unsupported feature dimension {d_cur} in {path} for sequence '{sequence_id}'. "
            f"Allowed dimensions: {sorted(ALLOWED_FEATURE_DIMS)}."
        )
'''
REGISTRY_GUARD = '''    # FeatureLayout is the canonical registry, including optional face channels.
    infer_feature_layout_from_dim(d_cur)
'''
FACE_VALIDATION = '''    if layout.face:
        face_support = features[:, layout.slice_face_mask]
        if not np.all((face_support == 0.0) | (face_support == 1.0)):
            raise ValueError(
                f"Non-binary face support mask values detected in {path} for sequence '{sequence_id}'. "
                "Face support masks must strictly contain 0.0 or 1.0."
            )

    return features
'''


def descriptor() -> dict:
    return {"id": HOTFIX_ID, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "frozen_loader_sha256": FROZEN_LOADER_SHA256,
            "scope": "Canonical feature dimension registry and binary face support validation"}


def install_feature_loader_compatibility() -> dict:
    """Patch exactly the reviewed frozen helper, retaining all its other checks."""
    import src.data.dataset as dataset

    original = dataset.load_and_validate_feature_file
    existing = getattr(original, "_feature_loader_hotfix", None)
    if existing == HOTFIX_ID:
        return descriptor()
    if existing is not None:
        raise RuntimeError("A different feature loader patch is already installed.")
    source = inspect.getsource(original).replace("\r\n", "\n")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != FROZEN_LOADER_SHA256:
        raise RuntimeError("Feature loader differs from the reviewed frozen source; refusing to patch.")
    if source.count(STALE_GUARD) != 1 or source.count("    return features\n") != 1:
        raise RuntimeError("Frozen loader guard/return structure differs; refusing to patch.")
    repaired = source.replace(STALE_GUARD, REGISTRY_GUARD).replace("    return features\n", FACE_VALIDATION)
    namespace = {}
    exec(compile(repaired, str(Path(__file__).resolve()) + ":feature_loader", "exec"),
         dataset.__dict__, namespace)
    helper = namespace["load_and_validate_feature_file"]
    helper._feature_loader_hotfix = HOTFIX_ID
    dataset.load_and_validate_feature_file = helper
    return descriptor()


def install_process_compatibility() -> None:
    """Install both audited fixes before the training script imports helpers."""
    install_feature_loader_compatibility()
    reporting.install_export_compatibility()
    import src.training.evaluate as evaluation

    original = evaluation.evaluate_test_set
    if getattr(original, "_feature_loader_hotfix", None) == HOTFIX_ID:
        return

    @wraps(original)
    def evaluate_with_loader_audit(*args, **kwargs):
        report = original(*args, **kwargs)
        report["feature_loader_hotfix"] = descriptor()
        destination = inspect.signature(original).bind_partial(*args, **kwargs).arguments.get("report_json_path")
        if destination:
            reporting._write_json(Path(destination), report)
        return report

    evaluate_with_loader_audit._feature_loader_hotfix = HOTFIX_ID
    evaluation.evaluate_test_set = evaluate_with_loader_audit


def install(session) -> dict:
    """Upgrade an existing reporting-v1 session without altering its queue state."""
    import study

    provenance = session.check_shared() if session.shared.exists() else None
    original = study.run_logged
    existing = getattr(original, "_campaign_compat", None)
    if existing is not None and existing != LAUNCHER_ID:
        raise RuntimeError("A different campaign compatibility launcher is installed.")
    if existing is None:
        previous = getattr(original, "_reporting_hotfix", None)
        if previous is not None:
            if previous != reporting.HOTFIX_ID or not hasattr(original, "__wrapped__"):
                raise RuntimeError("Unrecognized reporting launcher; refusing to replace it.")
            original = original.__wrapped__
        launcher = str(Path(__file__).resolve())

        @wraps(original)
        def run_logged_with_compatibility(command, log_path):
            indices = [i for i, argument in enumerate(command)
                       if Path(str(argument)).as_posix().endswith("scripts/20_train.py")]
            if len(indices) > 1:
                raise ValueError("Ambiguous training command.")
            if indices:
                i = indices[0]
                command = [*command[:i], launcher, "--bootstrap", *command[i:]]
            return original(command, log_path)

        run_logged_with_compatibility._campaign_compat = LAUNCHER_ID
        # An older notebook may reinstall reporting v1 after this upgrade.
        # Mark its functionality as present so it cannot nest a second bootstrap.
        run_logged_with_compatibility._reporting_hotfix = reporting.HOTFIX_ID
        study.run_logged = run_logged_with_compatibility

    audit = {**descriptor(), "shared_provenance": provenance,
             "reporting_hotfix": reporting.descriptor()}
    for name, source in (("campaign_compat_v2.py", Path(__file__)),
                         ("report_predictions_v1.py", Path(reporting.__file__))):
        destination = session.root / "compatibility" / name
        payload = source.read_bytes()
        if destination.exists() and destination.read_bytes() != payload:
            raise RuntimeError("Result bundle already contains a different compatibility payload.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    reporting._write_json(session.root / "FEATURE_LOADER_HOTFIX.json", audit)
    reporting._write_json(session.root / "REPORTING_HOTFIX.json",
                          {**reporting.descriptor(), "shared_provenance": provenance})
    print("Feature loader hotfix:", audit["id"], audit["sha256"])
    print("Reporting hotfix:", reporting.HOTFIX_ID, reporting.descriptor()["sha256"])
    return audit


def bootstrap(arguments: list[str]) -> None:
    if not arguments:
        raise ValueError("Bootstrap requires the frozen training script.")
    script = Path(arguments[0]).resolve()
    if script.name != "20_train.py" or script.parent.name != "scripts":
        raise ValueError("Compatibility bootstrap only runs scripts/20_train.py.")
    sys.path.insert(0, str(script.parent.parent))
    install_process_compatibility()
    sys.argv = [str(script), *arguments[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    if sys.argv[1:2] != ["--bootstrap"]:
        raise SystemExit("Import install(session), or use --bootstrap scripts/20_train.py.")
    bootstrap(sys.argv[2:])
