"""Verify the selected V3 checkpoints' final test archive and summarize predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from shutil import copy2
import tempfile
import zipfile

from analyze_campaign_results import check, digest, write_json
import session_v3
import study_v3 as study

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/7f86f6ac76de"
METRICS = ("macro_f1", "top1_accuracy", "top3_accuracy", "loss")
PAIRS = (("awake", "wake"), ("mouse", "shhh"), ("pretend", "mouse"))


def inspect_final(archive: Path, validation: dict, selection: dict) -> dict:
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        check(len(names) == len(set(names)) and bundle.testzip() is None, "ZIP duplicates or CRC failure")
        for entry in entries:
            session_v3.relative(ROOT / "tmp/final-review", entry.filename)
            check((entry.external_attr >> 16) & 0o170000 != 0o120000, "ZIP symlink")
        state = json.loads(bundle.read("queue_status.json"))
        status = json.loads(bundle.read("MEMBER_STATUS.json"))
        aggregate = json.loads(bundle.read("ALL_MEMBERS_FULL42_RESULTS.json"))
        lock = json.loads(bundle.read("FINAL_TEST/WINNER_LOCK.json"))
        summary = json.loads(bundle.read("FINAL_TEST/TEST_SUMMARY.json"))
        expected_lock = {key: selection[key] for key in ("freeze_id", "winner", "checkpoints")}
        check(lock == expected_lock and {k: summary[k] for k in expected_lock} == expected_lock,
              "Test winner/checkpoint lock differs from validation selection")
        check(state["member"] == status["member"] == "A" and state["freeze_id"] == selection["freeze_id"]
              and state["provenance"] == selection["provenance"] and not state["failed"], "A identity/queue differs")
        expected_a = {r["run_id"] for r in study.matrix() if r["owner"] == "A"}
        check(set(state["selected"]) == expected_a and status["completed"] == status["expected"] == 7,
              "A selected queue is incomplete")
        check(aggregate["scope"] == "v3-full42" and aggregate["status"] == "COMPLETED"
              and aggregate["completed_cells"] == aggregate["expected_cells"] == 42
              and aggregate["completed_fits"] == aggregate["expected_fits"] == 126
              and aggregate["winner"] == selection["winner"] and aggregate["missing"] == [], "Full42 aggregate differs")
        check(set(aggregate["records"]) == set(validation["records"]), "Aggregate run IDs differ")
        for run_id, row in aggregate["records"].items():
            expected = {k: v for k, v in validation["records"][run_id].items() if k not in ("archive", "archive_prefix")}
            check({k: v for k, v in row.items() if k != "path"} == expected, "Accepted aggregate record changed: " + run_id)
        original = ROOT / "artifacts/ablation-study-v3/results/member-a/b4fd56f03a57/ae638dcade7c/member_a_b4fd56f03a57_v3_results.zip"
        check(digest(original) == validation["archive_sha256"]["A"], "Original A archive checksum differs")
        with zipfile.ZipFile(original) as previous:
            previous_state = json.loads(previous.read("queue_status.json"))
            check(state["selected"] == previous_state["selected"], "Original A attempts changed")
            for prefix in state["selected"].values():
                for name in previous.namelist():
                    if name.startswith(prefix + "/"):
                        check(previous.read(name) == bundle.read(name), "Original A artifact changed: " + name)
        expected_reports = {f"FINAL_TEST/seed_{s}/evaluation_test_report.json" for s in study.SEEDS}
        check({n for n in names if n.endswith("evaluation_test_report.json")} == expected_reports,
              "Final test reports missing, duplicated or unexpected")
        payload_hashes = {n: hashlib.sha256(bundle.read(n)).hexdigest() for n in names if n.startswith("FINAL_TEST/")}
        return {"entries": len(entries), "uncompressed_bytes": sum(e.file_size for e in entries),
                "crc_valid": True, "state": state, "summary": summary, "payload_sha256": payload_hashes}


def review_reports(folder: Path, manifest: dict, identity: str, parameters: int) -> dict:
    names = manifest["vocabulary"]
    expected_ids = set(manifest["splits"]["test"]["sequence_ids"])
    matrices, class_f1, recalls = [], [], []
    reported, exact = {key: [] for key in METRICS}, {key: [] for key in METRICS}
    signer_data, targeted, truth_hashes = {}, {}, []
    for seed in study.SEEDS:
        report = study.read(folder / f"seed_{seed}/evaluation_test_report.json")
        predictions = report["predictions"]
        ids = [p["sequence_id"] for p in predictions]
        check(len(ids) == len(set(ids)) and set(ids) == expected_ids
              and report["sample_membership"] == ids, "Test prediction membership/order differs")
        check(report["num_test_samples"] == len(expected_ids) == 1141 and report["num_classes"] == len(names) == 20
              and report["class_names"] == names and report["split_identity"] == "test"
              and report["preprocess_identity"] == identity and report["model_parameters"] == parameters,
              "Test report class/config/split identity differs")
        for p in predictions:
            frozen = manifest["sequences"][p["sequence_id"]]
            check(p["participant_id"] == frozen["participant_id"] and p["y_true"] == frozen["label"],
                  "Test ground truth differs from frozen cohort")
        truth = sorted((p["sequence_id"], p["participant_id"], p["y_true"]) for p in predictions)
        truth_hashes.append(hashlib.sha256(json.dumps(truth).encode()).hexdigest())
        y = np.asarray([p["y_true"] for p in predictions])
        predicted = np.asarray([p["y_pred"] for p in predictions])
        probabilities = np.asarray([p["probabilities"] for p in predictions], dtype=float)
        check(probabilities.shape == (1141, 20) and np.isfinite(probabilities).all() and (probabilities >= 0).all()
              and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
              and np.array_equal(probabilities.argmax(axis=1), predicted), "Test probabilities/argmax invalid")
        matrix = confusion_matrix(y, predicted, labels=np.arange(20))
        _, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=np.arange(20), zero_division=0)
        check(np.array_equal(matrix, report["confusion_matrix"]), "Test confusion matrix differs")
        truth_probabilities = probabilities[np.arange(len(y)), y]
        check((truth_probabilities > 0).all(), "Cannot recompute cross-entropy from zero true-class probabilities")
        recomputed = {"macro_f1": float(f1.mean()), "top1_accuracy": float(np.mean(y == predicted)),
                      "top3_accuracy": float(np.mean(np.any(np.argsort(probabilities, axis=1)[:, -3:] == y[:, None], axis=1))),
                      "loss": float(-np.log(truth_probabilities).mean())}
        for key, value in recomputed.items():
            check(abs(value - report[key]) <= .000051, "Test metric differs from predictions: " + key)
            reported[key].append(report[key])
            exact[key].append(value)
        matrices.append(matrix)
        class_f1.append(f1)
        recalls.append(recall)
        expected_signers = {p["participant_id"] for p in predictions}
        check(set(report["per_participant_accuracy"]) == expected_signers, "Reported signer IDs differ")
        for signer in sorted(expected_signers):
            clips = [p for p in predictions if p["participant_id"] == signer]
            correct = sum(p["y_true"] == p["y_pred"] for p in clips)
            accuracy = correct / len(clips)
            row = report["per_participant_accuracy"][signer]
            check(row["samples"] == len(clips) and row["correct"] == correct and abs(row["accuracy"] - accuracy) <= .000051,
                  "Reported signer accuracy differs")
            item = signer_data.setdefault(signer, {"samples": len(clips), "accuracy": [], "correct": []})
            item["accuracy"].append(accuracy)
            item["correct"].append(correct)
        for true_name, predicted_name in PAIRS:
            true, wrong = names.index(true_name), names.index(predicted_name)
            clips = [p for p in predictions if p["y_true"] == true]
            item = targeted.setdefault(true_name + "->" + predicted_name, {"support": len(clips), "wrong": [], "signers": {}})
            item["wrong"].append(sum(p["y_pred"] == wrong for p in clips))
            for signer in sorted({p["participant_id"] for p in clips}):
                subset = [p for p in clips if p["participant_id"] == signer]
                row = item["signers"].setdefault(signer, {"support": len(subset), "wrong": []})
                row["wrong"].append(sum(p["y_pred"] == wrong for p in subset))
    check(len(set(truth_hashes)) == 1, "Cross-seed test ground truth differs")
    return {"metrics": {key: study.aggregate(values) for key, values in reported.items()},
            "metrics_from_predictions": {key: study.aggregate(values) for key, values in exact.items()},
            "class_metrics": {name: {"support": int(support[i]), "f1": study.aggregate([float(v[i]) for v in class_f1]),
                                      "recall": study.aggregate([float(v[i]) for v in recalls])} for i, name in enumerate(names)},
            "per_signer": {k: {**v, "accuracy": study.aggregate(v["accuracy"])} for k, v in signer_data.items()},
            "targeted_confusions": targeted, "mean_confusion_matrix": np.mean(matrices, axis=0).tolist(),
            "ground_truth_sha256": truth_hashes[0]}


def write_report(data: dict, validation: dict, output: Path) -> None:
    winner = data["winner"]
    val = validation["records"][winner]["metrics"]
    metrics = data["metrics"]
    classes = data["class_metrics"]
    val_classes = validation["details"][winner]["class_metrics"]
    lines = ["# Kết quả cuối — Ablation V3 full42", "",
             "Đã nghiệm thu 42 configs/126 fits và test evaluation của ba checkpoints seeds 42/43/44 cho `T0__HF`.", "",
             "Test cohort gồm 1.141 clips, 20 glosses, 3 signer 27610/37779/61333. "
             "Cấu hình và checkpoints được chọn từ validation trước evaluation.", "",
             "| Metric | Validation mean ± std (%) | Test mean ± std (%) |", "|---|---:|---:|"]
    for key, label in (("macro_f1", "Macro-F1"), ("top1_accuracy", "Top-1"), ("top3_accuracy", "Top-3")):
        lines.append(f"| {label} | {val[key]['mean']*100:.2f} ± {val[key]['std']*100:.2f} "
                     f"| {metrics[key]['mean']*100:.2f} ± {metrics[key]['std']*100:.2f} |")
    lines += ["", "Std là population std qua ba seeds. Validation và test có các signer khác nhau, "
              "chênh lệch giữa hai tập phản ánh độ khó của từng cohort và mức tổng quát hóa.", "",
              "| Seed | Test Macro-F1 (%) | Test Top-1 (%) | Test Top-3 (%) | Test loss |", "|---:|---:|---:|---:|---:|"]
    for i, seed in enumerate(study.SEEDS):
        lines.append(f"| {seed} | {metrics['macro_f1']['per_seed'][i]*100:.2f} | {metrics['top1_accuracy']['per_seed'][i]*100:.2f} "
                     f"| {metrics['top3_accuracy']['per_seed'][i]*100:.2f} | {metrics['loss']['per_seed'][i]:.4f} |")
    lines += ["", "## Accuracy theo test signer", "", "| Signer | Clips | Mean accuracy (%) | Seeds 42 / 43 / 44 (%) |", "|---|---:|---:|---|"]
    for signer, row in data["per_signer"].items():
        lines.append(f"| {signer} | {row['samples']} | {row['accuracy']['mean']*100:.2f} | "
                     + " / ".join(f"{v*100:.2f}" for v in row["accuracy"]["per_seed"]) + " |")
    lines += ["", "## F1 từng gloss", "", "| Gloss | Test clips | Validation F1 (%) | Test F1 (%) | Test recall (%) |", "|---|---:|---:|---:|---:|"]
    for name, row in sorted(classes.items(), key=lambda p: p[1]["f1"]["mean"]):
        lines.append(f"| {name} | {row['support']} | {val_classes[name]['f1']['mean']*100:.2f} "
                     f"| {row['f1']['mean']*100:.2f} | {row['recall']['mean']*100:.2f} |")
    lines += ["", "![F1 từng gloss](validation_test_gloss_f1.png)", "", "## Ba hướng nhầm đã theo dõi", "",
              "Tỷ lệ tính trong nhãn thật, mean qua ba seeds.", "",
              "| Nhãn thật → dự đoán | Validation (%) | Test (%) | Test counts seeds 42/43/44 |", "|---|---:|---:|---|"]
    val_pairs = validation["prediction_diagnostics"][winner]["targeted_confusions"]
    for name, row in data["targeted_confusions"].items():
        previous = val_pairs[name]
        lines.append(f"| {name.replace('->', ' → ')} | {np.mean(previous['wrong'])/previous['support']*100:.2f} "
                     f"| {np.mean(row['wrong'])/row['support']*100:.2f} | " + " / ".join(str(v) for v in row["wrong"]) + f" trên {row['support']} |")
    lines += ["", "`awake → wake` tiếp tục là điểm yếu trên test. `pretend → mouse` có 0 mẫu ở cả ba seeds trên test, "
              "trong khi lỗi validation trước đó tập trung ở signer 32319. `mouse → shhh` giảm trên ba test signers. "
              "Các kết quả này phụ thuộc nhóm signer và cần clip/landmarks thực tế để xác định nguyên nhân.", "",
              "## Nghiệm thu", "",
              "ZIP pass CRC, duplicate/path/symlink checks. Aggregate trong ZIP khớp cả 42 records đã nghiệm thu. "
              "Artifacts A7 cũ khớp từng byte. Winner lock khớp selection và ba checkpoint SHA256 của C. "
              "Preprocess/split/test logs đều có exit code 0.", "",
              "Ba test reports có đúng frozen membership, labels, participants, vocabulary và preprocess identity. "
              "Test sequence IDs và signers tách khỏi train/validation. Probabilities hợp lệ và argmax khớp predictions. "
              "Top-1/Top-3/Macro-F1, cross-entropy loss, confusion matrices và signer accuracies được tính lại.", "",
              "Test đánh giá cấu hình T0__HF đã được chọn từ ma trận 42. Ranking giữa các configs vẫn dựa trên validation. "
              "Ba seeds dùng cùng 1.141 test clips, std là thống kê mô tả. Lượt review đọc predictions, source/config và checkpoints đã có.", "",
              "[Metrics và audit JSON](FINAL_RESULTS.json) · [ZIP đã lưu nguyên trạng](" + data["archive"].replace("\\", "/") + ")"]
    (output / "FINAL_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ordered = sorted(classes, key=lambda c: classes[c]["f1"]["mean"])
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(y - .18, [val_classes[c]["f1"]["mean"]*100 for c in ordered], height=.34, label="Validation", color="#8b99a7")
    ax.barh(y + .18, [classes[c]["f1"]["mean"]*100 for c in ordered], height=.34, label="Test", color="#187d73")
    ax.set_yticks(y, ordered)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("F1 từng gloss (%) — mean qua 3 seeds")
    ax.set_title("T0__HF — F1 trên validation và test của các nhóm signer khác nhau")
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "validation_test_gloss_f1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    source = args.archive.resolve()
    checksum = digest(source)
    folder = ROOT / "artifacts/ablation-study-v3/results/member-a/b4fd56f03a57" / checksum[:12]
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / source.name
    if not archive.exists():
        copy2(source, archive)
    check(digest(archive) == checksum, "Stored final archive differs")
    validation = study.read(VALIDATION / "ALL_MEMBERS_FULL42_RESULTS.json")
    selection = study.read(VALIDATION / "WINNER_SELECTION.json")
    audit = inspect_final(archive, validation, selection)
    output = folder / "analysis"
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="f", dir=ROOT / "tmp") as temporary:
        extracted = Path(temporary).resolve()
        check(extracted.is_relative_to((ROOT / "tmp").resolve()), "Unsafe extraction workspace")
        session_v3.extract(archive, extracted)
        for run_id, prefix in audit["state"]["selected"].items():
            record = study.collect(session_v3.relative(extracted, prefix))
            check(record["run_id"] == run_id and record["provenance"] == selection["provenance"], "A baseline differs")
        first = session_v3.relative(extracted, next(iter(audit["state"]["selected"].values())))
        manifest = study.read(first / "inputs/split_manifest.json")
        cfg = study.load_config(str(extracted / "FINAL_TEST/evaluation_config.yaml"))
        study.validate_config(cfg, study.config_for(first / "inputs/config.yaml", "T0", "HF"))
        identity = study.compute_preprocess_identity(cfg)[0]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            left, right = set(manifest["splits"][a]["sequence_ids"]), set(manifest["splits"][b]["sequence_ids"])
            check(not left & right, "Frozen sequences overlap")
            participants_a = {manifest["sequences"][sid]["participant_id"] for sid in left}
            participants_b = {manifest["sequences"][sid]["participant_id"] for sid in right}
            check(not participants_a & participants_b, "Frozen signer split overlaps")
        winner = validation["records"][selection["winner"]]
        with zipfile.ZipFile(winner["archive"]) as c_archive:
            for seed in study.SEEDS:
                paths = [n for n in c_archive.namelist() if n.startswith(winner["archive_prefix"] + "/")
                         and n.endswith(f"/seed_{seed}/best_model.pt")]
                check(len(paths) == 1 and hashlib.sha256(c_archive.read(paths[0])).hexdigest() == selection["checkpoints"][f"seed_{seed}"],
                      "Selected C checkpoint hash differs")
                log = (extracted / f"FINAL_TEST/logs/seed_{seed}.log").read_text(encoding="utf-8")
                command = json.loads(log.splitlines()[0].removeprefix("COMMAND_JSON: "))
                check(command[command.index("--split") + 1] == "test"
                      and command[command.index("--checkpoint") + 1].endswith(paths[0]), "Evaluation command checkpoint/split differs")
        for name in ("preprocess", "split", "seed_42", "seed_43", "seed_44"):
            check((extracted / f"FINAL_TEST/logs/{name}.log").read_text(encoding="utf-8").rstrip().endswith("EXIT_CODE: 0"),
                  "Final stage lacks successful exit")
        data = review_reports(extracted / "FINAL_TEST", manifest, identity, int(winner["metrics"]["model_parameters"]["mean"]))
    for key in METRICS:
        for moment in ("mean", "std", "per_seed"):
            check(np.allclose(data["metrics"][key][moment], audit["summary"]["metrics"][key][moment], rtol=0, atol=1e-10),
                  "Final summary differs from individual reports")
    timestamp = datetime.now(timezone(timedelta(hours=7))).isoformat()
    data.update({"status": "ACCEPTED", "campaign_status": "COMPLETED", "scope": "v3-full42-final-test",
                 "freeze_id": selection["freeze_id"], "winner": selection["winner"], "checkpoints": selection["checkpoints"],
                 "archive": str(archive), "archive_sha256": checksum, "validation_archive_sha256": validation["archive_sha256"],
                 "reviewed_at_asia_saigon": timestamp, "completed_training_configs": 42, "completed_training_fits": 126,
                 "test_checkpoint_evaluations": 3, "test_samples": 1141, "test_evaluated": True,
                 "training_executed_during_review": False, "inference_executed_during_review": False})
    write_json(output / "FINAL_RESULTS.json", data)
    write_json(folder / "STORAGE.json", {"source": str(source), "destination": str(archive), "sha256": checksum,
               "bytes": archive.stat().st_size, "source_removed": False, "hash_verified": True, "stored_at_asia_saigon": timestamp})
    write_json(folder / "ZIP_CHECK.json", {k: v for k, v in audit.items() if k not in ("state", "summary")})
    write_json(folder / "REVIEW.json", {"status": "ACCEPTED", "archive_sha256": checksum, "winner": data["winner"],
               "completed_configs": 42, "completed_fits": 126, "test_checkpoint_evaluations": 3,
               "test_samples": 1141, "test_evaluated": True, "report": str(output / "FINAL_RESULTS.md")})
    write_report(data, validation, output)
    check(digest(source) == digest(archive) == checksum, "Final archive changed during review")
    print(json.dumps({"status": data["status"], "winner": data["winner"], "test_metrics": data["metrics"],
                      "report": str(output / "FINAL_RESULTS.md")}, indent=2))


if __name__ == "__main__":
    main()
