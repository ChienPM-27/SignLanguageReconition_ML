"""Run as one Kaggle cell to recover the existing full42 notebook session."""
from pathlib import Path
from shutil import copy2

import study_v3

def recover_split_paths(active_session, working: Path) -> Path:
    working = working.resolve()
    actual_splits = working / "data/splits"
    expected_splits = active_session.persistent_root / "data/splits"
    if not expected_splits.resolve().is_relative_to(working):
        raise ValueError("Đường dẫn phục hồi phải nằm trong Kaggle working root.")
    frozen_split = active_session.shared / "split_manifest.json"
    study_v3.check_cohort(actual_splits, frozen_split)
    filenames = ("train.csv", "val.csv", "test.csv", "classes.csv", "label_map.json")
    for filename in filenames:
        original, restored = actual_splits / filename, expected_splits / filename
        if not original.is_file():
            raise FileNotFoundError(original)
        if restored.exists() and restored.read_bytes() != original.read_bytes():
            raise ValueError(f"Không ghi đè split khác dữ liệu: {restored}")
    expected_splits.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        original, restored = actual_splits / filename, expected_splits / filename
        if not restored.exists():
            copy2(original, restored)
    study_v3.check_cohort(expected_splits, frozen_split)
    study_v3.write(active_session.root / "PATH_RECOVERY.json", {
        "reason": "Old full42 launcher nested its root while Kaggle split writer uses /kaggle/working.",
        "actual_splits": str(actual_splits), "runner_expected_splits": str(expected_splits),
        "freeze_id": active_session.freeze_id, "source_training_changed": False,
        "recovered_files_sha256": {name: study_v3.sha(expected_splits / name) for name in filenames},
    })
    return expected_splits


if __name__ == "__main__":
    active_session = globals().get("session")
    run_ids = globals().get("RUN_IDS")
    if active_session is None or run_ids is None:
        raise RuntimeError("Chạy cell này trong kernel notebook full42 đang gặp lỗi; cần session và RUN_IDS.")
    recover_split_paths(active_session, Path("/kaggle/working"))
    print("Đã khớp split với frozen cohort; giữ cache hiện tại và các configs COMPLETED.")
    status = active_session.run_queue(run_ids, retry_failed=True)
    print("Hoàn tất B:", status["completed"], "/", status["expected"])
    results_zip = active_session.export_results()
    print("Tải kết quả:", results_zip)
