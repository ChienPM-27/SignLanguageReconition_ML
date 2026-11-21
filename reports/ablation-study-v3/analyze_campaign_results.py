"""Review immutable V3 result archives and prepare the frozen GRU cross-plan."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from shutil import copy2
import sys
import tempfile
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reports/ablation-study-v3"))
from analyze_member_a_results import check, digest, write_json
import session_v3
import study_v3 as study


def inspect(archive: Path, member: str, *, full_matrix: bool = False) -> dict:
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        check(len(names) == len(set(names)), "Duplicate ZIP entries")
        check(bundle.testzip() is None, "ZIP CRC failure")
        for entry in entries:
            session_v3.relative(ROOT / "tmp/result_reviews/safety-check", entry.filename)
            check((entry.external_attr >> 16) & 0o170000 != 0o120000, "ZIP symlink")
        state = json.loads(bundle.read("queue_status.json"))
        status = json.loads(bundle.read("MEMBER_STATUS.json"))
        expected_rows = [r for r in study.matrix() if r["owner"] == member and (full_matrix or r["phase"] == 1)]
        expected = {r["run_id"] for r in expected_rows}
        check(state["member"] == member and state["freeze_id"] == "b4fd56f03a57", "Wrong member/freeze")
        check(set(state["selected"]) == expected and not state["failed"], "Incomplete member queue")
        check(state["cross_plan"] is None, "Expected original first-round archive")
        check(status["member"] == member and status["freeze_id"] == state["freeze_id"], "Status identity differs")
        check(status["completed"] == status["expected"] == len(expected), "Incomplete member status")
        check(len(status["queue"]) == len(expected), "Duplicate member status rows")
        check({r["run_id"] for r in status["queue"]} == expected, "Status queue differs")
        for row in status["queue"]:
            canonical = next(r for r in expected_rows if r["run_id"] == row["run_id"])
            check(row == {**canonical, "status": "COMPLETED"}, "Status row differs from frozen matrix")
        if full_matrix and member == "B":
            scope = json.loads(bundle.read("FULL_MATRIX_SCOPE.json"))
            check(scope["scope"] == status["scope"] == "v3-full42", "Wrong full-matrix scope")
            check(scope["freeze_id"] == state["freeze_id"]
                  and scope["training_source_provenance"] == state["provenance"], "Scope provenance differs")
            check(scope["expected_cells"] == 42 and scope["expected_fits"] == 126
                  and scope["member_expected_cells"] == 21 and status["remaining"] == [], "Scope counts differ")
        test_artifacts = [n for n in names if Path(n).name in
                          ("test_report.json", "evaluation_test_report.json", "TEST_SUMMARY.json", "WINNER_LOCK.json")]
        check(not test_artifacts, "Premature test artifacts")
        return {"status": "PASS", "entries": len(entries), "crc_valid": True,
                "uncompressed_bytes": sum(e.file_size for e in entries), "duplicate_entries": False,
                "unsafe_paths": False, "symlinks": False, "test_artifacts": [], "state": state}


def audit_attempt(attempt: Path, record: dict) -> dict:
    inventory = study.read(attempt / "ARTIFACTS.json")
    actual = {p.relative_to(attempt).as_posix() for p in attempt.rglob("*") if p.is_file()}
    check(actual == set(inventory) | {"metadata.json", "ARTIFACTS.json"}, "Uninventoried attempt payload")
    for stage in ("preprocess", "split", "train"):
        check((attempt / "logs" / (stage + ".log")).read_text(encoding="utf-8").rstrip().endswith("EXIT_CODE: 0"),
              "Stage log lacks successful exit code")
    with (attempt / "cache_audit/manifest.csv").open(encoding="utf-8", newline="") as stream:
        ids = [r["sequence_id"] for r in study.csv.DictReader(stream)]
    frozen = study.read(attempt / "inputs/split_manifest.json")
    check(len(ids) == len(set(ids)) and set(ids) == set(frozen["sequences"]), "Cache audit membership differs")
    reports = sorted((attempt / "runs").rglob("val_report.json"), key=lambda p: p.parent.name)
    check([int(p.parent.name.removeprefix("seed_")) for p in reports] == list(study.SEEDS), "Wrong seeds")
    class_scores, matrices, epochs, ground_truth = [], [], [], None
    metrics = {k: [] for k in ("macro_f1", "top1_accuracy", "top3_accuracy")}
    for report_path in reports:
        report = study.read(report_path)
        predictions = sorted(report["predictions"], key=lambda p: p["sequence_id"])
        truth = [(p["sequence_id"], p["participant_id"], p["y_true"]) for p in predictions]
        if ground_truth is None:
            ground_truth = truth
        check(truth == ground_truth, "Seed prediction alignment differs")
        labels = np.asarray([p["y_true"] for p in predictions])
        predicted = np.asarray([p["y_pred"] for p in predictions])
        probability = np.asarray([p["probabilities"] for p in predictions])
        _, _, f1, support = precision_recall_fscore_support(
            labels, predicted, labels=np.arange(len(frozen["vocabulary"])), zero_division=0)
        matrix = confusion_matrix(labels, predicted, labels=np.arange(len(frozen["vocabulary"])))
        check(np.array_equal(matrix, report["confusion_matrix"]), "Confusion matrix differs")
        recomputed = {"macro_f1": float(f1.mean()), "top1_accuracy": float(np.mean(labels == predicted)),
                      "top3_accuracy": float(np.mean(np.any(np.argsort(probability, axis=1)[:, -3:] == labels[:, None], axis=1)))}
        for key, value in recomputed.items():
            check(abs(value - report[key]) <= 0.000051, "Prediction metrics differ")
            metrics[key].append(value)
        class_scores.append(f1)
        matrices.append(matrix)
        history = study.read(report_path.parent / "history.json")
        best = int(np.argmin(history["val_loss"]))
        count = len(history["val_loss"])
        check(all(len(history[k]) == count for k in ("train_loss", "train_acc", "val_acc", "val_f1", "lr")),
              "History lengths differ")
        check(abs(history["val_acc"][best] - report["top1_accuracy"]) <= 0.000051
              and abs(history["val_f1"][best] - report["macro_f1"]) <= 0.000051, "Loss-selected history differs")
        check(abs(history["best_val_loss"] - history["val_loss"][best]) <= 1e-10, "Best loss differs")
        epochs.append({"seed": int(report_path.parent.name.removeprefix("seed_")), "epochs_run": count,
                       "best_epoch": best + 1, "checkpoint_sha256": digest(report_path.parent / "best_model.pt"),
                       "train_accuracy_at_best": history["train_acc"][best],
                       "val_accuracy_at_best": history["val_acc"][best],
                       "training_time_sec": history["training_time_sec"]})
    summaries = list((attempt / "runs").rglob("multi_seed_summary.json"))
    check(len(summaries) == 1, "Expected one multi-seed summary")
    summary = study.read(summaries[0])
    check(summary["seeds"] == list(study.SEEDS) and summary["num_seeds"] == 3, "Summary seeds differ")
    for key in ("macro_f1", "top1_accuracy", "top3_accuracy", "loss"):
        for moment in ("mean", "std"):
            check(abs(summary["val_" + key + "_" + moment] - record["metrics"][key][moment]) <= 0.000051,
                  "Multi-seed summary differs")
    return {"metrics_from_predictions": {k: study.aggregate(v) for k, v in metrics.items()},
            "class_metrics": {name: {"support": int(support[i]),
                                     "f1": study.aggregate([float(scores[i]) for scores in class_scores])}
                              for i, name in enumerate(frozen["vocabulary"])},
            "mean_confusion_matrix": np.mean(matrices, axis=0).tolist(), "epoch_details": epochs,
            "ground_truth_sha256": hashlib.sha256(json.dumps(ground_truth).encode()).hexdigest(),
            "hardware": study.read(attempt / "metadata.json")["hardware"]}


def make_plot(result: dict, output: Path) -> None:
    profiles = ("G0", "L0", "T0")
    features = tuple(study.REGIONS)
    values = np.asarray([[result["records"][p + "__" + f]["metrics"]["macro_f1"]["mean"] * 100
                          for f in features] for p in profiles])
    fig, (ax, bar) = plt.subplots(1, 2, figsize=(13, 4.7), gridspec_kw={"width_ratios": [1.5, 1]})
    heat = ax.imshow(values, vmin=0, vmax=80, cmap="YlGnBu")
    ax.set_xticks(range(len(features)), features)
    ax.set_yticks(range(len(profiles)), profiles)
    for i in range(len(profiles)):
        for j in range(len(features)):
            ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center",
                    color="white" if values[i,j] > 55 else "black")
    ax.set_title("Vòng đầu: Macro-F1 theo model và vùng (%)")
    fig.colorbar(heat, ax=ax, fraction=.025, pad=.04)
    gru = [p for p in study.PROFILES if p.startswith("G")]
    means = [result["records"][p + "__HBF"]["metrics"]["macro_f1"]["mean"] * 100 for p in gru]
    stds = [result["records"][p + "__HBF"]["metrics"]["macro_f1"]["std"] * 100 for p in gru]
    bar.bar(gru, means, yerr=stds, capsize=5, color=["#8797a5", "#8797a5", "#8797a5", "#167b75"])
    for i, mean in enumerate(means):
        bar.text(i, mean + stds[i] + 1, f"{mean:.2f}", ha="center")
    bar.set_ylim(0, 85)
    bar.set_ylabel("Validation Macro-F1 (%)")
    bar.set_title("GRU trên HBF; error bar = std 3 seeds")
    fig.tight_layout()
    fig.savefig(output / "phase1_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-a", type=Path, required=True)
    parser.add_argument("--member-b", type=Path, required=True)
    parser.add_argument("--member-c", type=Path, required=True)
    args = parser.parse_args()
    sources = {"A": args.member_a.resolve(), "B": args.member_b.resolve(), "C": args.member_c.resolve()}
    hashes = {m: digest(p) for m, p in sources.items()}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]
    output = ROOT / "artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57" / version
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone(timedelta(hours=7))).isoformat()
    archives, audits = {}, {}
    for member, source in sources.items():
        target = ROOT / "artifacts/ablation-study-v3/results" / ("member-" + member.lower()) / "b4fd56f03a57" / hashes[member][:12] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            copy2(source, target)
        check(digest(target) == hashes[member], "Stored archive differs from source")
        archives[member] = target
        audits[member] = inspect(target, member)
        if member != "A":
            write_json(target.parent / "STORAGE.json", {"operation": "copy", "source": str(source),
                       "destination": str(target), "sha256": hashes[member], "bytes": target.stat().st_size,
                       "stored_at_asia_saigon": timestamp, "hash_verified": True, "source_removed": False})
            write_json(target.parent / "ZIP_CHECK.json", {k: v for k, v in audits[member].items() if k != "state"}
                       | {"sha256": hashes[member], "bytes": target.stat().st_size})
    temp_root = ROOT / "tmp/result_reviews"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v3-campaign-", dir=temp_root) as temporary:
        temporary_path = Path(temporary).resolve()
        check(temporary_path.is_relative_to(temp_root.resolve()) and temporary_path != temp_root.resolve(), "Unsafe cleanup root")
        attempts, locations = [], {}
        for member, archive in archives.items():
            extracted = temporary_path / member
            session_v3.extract(archive, extracted)
            state = audits[member]["state"]
            for run_id, prefix in state["selected"].items():
                attempt = session_v3.relative(extracted, prefix)
                attempts.append(attempt)
                locations[run_id] = (member, prefix, state)
        result = study.summarize(attempts, temporary_path / "ALL_MEMBERS_RESULTS.md")
        detailed = {}
        for run_id, record in result["records"].items():
            member, prefix, state = locations[run_id]
            check(record["provenance"] == state["provenance"] and record["owner"] == member, "Selected provenance differs")
            detailed[run_id] = audit_attempt(Path(record["path"]), record)
            del record["path"]
            record.update({"archive": str(archives[member]), "archive_prefix": prefix})
            print("Accepted", run_id, record["metrics"]["macro_f1"], flush=True)
        check(len({d["ground_truth_sha256"] for d in detailed.values()}) == 1, "Cross-model prediction alignment differs")
        cross = result["cross_plan"]
        if cross:
            write_json(output / "cross_plan.json", cross)
            with zipfile.ZipFile(output / "cross_plan.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(output / "cross_plan.json", "cross_plan.json")
            b_attempt = next(Path(p) for p in attempts if study.read(p / "metadata.json")["owner"] == "B")
            session = session_v3.MemberSession("B", b_attempt / "inputs", temporary_path / "unused-mirror", temporary_path / "cross-check")
            session.load_cross_plan(output / "cross_plan.zip")
            check([r["run_id"] for r in session.queue() if r["phase"] == 2] == cross["run_ids"], "Cross queue differs")
        canonical_report = (temporary_path / "ALL_MEMBERS_RESULTS.md").read_text(encoding="utf-8")
    ranked = study.rank(list(result["records"].values()))
    result.update({"freeze_id": "b4fd56f03a57", "reviewed_at_asia_saigon": timestamp,
                   "archive_sha256": hashes, "completed_fits": len(ranked) * 3,
                   "first_round_leader": ranked[0]["run_id"], "ranking": [r["run_id"] for r in ranked],
                   "test_evaluated": False, "training_executed_during_review": False,
                   "inference_executed_during_review": False, "details": detailed,
                   "record_location_format": "Extract archive into an empty folder, then resolve archive_prefix within that folder."})
    write_json(output / "ALL_MEMBERS_RESULTS.json", result)
    make_plot(result, output)
    for member in ("B", "C"):
        records = {k: v for k, v in result["records"].items() if v["owner"] == member}
        write_json(archives[member].parent / "REVIEW.json", {
            "status": "ACCEPTED", "member": member, "freeze_id": result["freeze_id"],
            "configs": len(records), "fits": len(records) * 3, "archive": str(archives[member]),
            "archive_sha256": hashes[member], "provenance": next(iter(records.values()))["provenance"],
            "reviewed_at_asia_saigon": timestamp, "attempts": records, "aggregate_report": str(output / "ALL_MEMBERS_RESULTS.md"),
            "checks": ["ZIP CRC, ownership and complete first-round queue", "Canonical study_v3.collect for every selected attempt",
                       "Frozen source/config/matrix/protocol/split and exact artifact inventory",
                       "Checkpoint SHA256, stage log exits, history loss selection, multi-seed summaries",
                       "Aligned predictions; recomputed Top-1/Top-3/Macro-F1 and confusion matrices",
                       "No premature test report or winner lock"], "test_evaluated": False,
            "training_executed_during_review": False, "inference_executed_during_review": False})
    lines = [canonical_report, "## Nghiệm thu vòng đầu ngày 20/11/2026", "",
             "A: 7/7 configs, 21 fits; B: 3/3, 9 fits; C: 14/14, 42 fits. Tổng 24/24 configs, 72 fits.",
             "CRC, source/config/protocol/split/cache identity, checkpoints, inventories, logs, history và predictions đều pass.",
             "Mỗi fit có cùng 1.155 validation clips, 20 glosses, 3 held-out signers; test tiếp tục giữ kín.", "",
             "Ranking dùng metrics làm tròn trong reports theo implementation đã freeze; metrics chính xác từ predictions được lưu riêng trong JSON.",
             "ZIP được lưu nguyên trạng theo SHA256; B/C trong Downloads được giữ nguyên. Các đường dẫn trong JSON gồm archive + archive_prefix, không trỏ tới thư mục tạm.", "",
             "| Hạng vòng đầu | Config | Macro-F1 mean ± std | Top-1 | Top-3 | Parameters |", "|---:|---|---:|---:|---:|---:|"]
    for i, row in enumerate(ranked, 1):
        m = row["metrics"]
        lines.append(f"| {i} | {row['run_id']} | {m['macro_f1']['mean']:.4f} ± {m['macro_f1']['std']:.4f} | {m['top1_accuracy']['mean']:.4f} | {m['top3_accuracy']['mean']:.4f} | {m['model_parameters']['mean']:.0f} |")
    lines += ["", "![So sánh vòng đầu](phase1_comparison.png)", "", "## Vòng bổ sung cho B", "",
              f"GRU được chọn: `{cross['profile']}`. Cần thêm {len(cross['run_ids'])} configs/{len(cross['run_ids']) * 3} fits: "
              + ", ".join(f"`{r}`" for r in cross["run_ids"]) + ".",
              "B không cần train lại ba configs vòng đầu; A và C đã xong vòng training được phân công.",
              f"`{ranked[0]['run_id']}` dẫn đầu vòng đầu, chưa phải global winner: còn thiếu vòng GRU bổ sung theo protocol.",
              "Sau khi nhận results B bổ sung, tổng hợp lại 30 configs/90 fits rồi mới khóa winner và mở test.", "",
              "Body/face riêng thấp trên cả ba models; đây là kết quả của cohort và fixed fusion 70/20/10 này, không kết luận body/face vô dụng.",
              "Chênh lệch giữa các configs là mô tả từ ba seeds; chưa có kết luận statistical significance."]
    (output / "ALL_MEMBERS_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    handoff = ["# B chạy vòng GRU bổ sung", "", "Dùng notebook V3 B hiện tại và Shared freeze `b4fd56f03a57`.",
               "Attach competition `asl-signs`, Shared V3, ZIP results B vòng đầu và `cross_plan.zip` vào Kaggle; bật GPU.",
               "Điền đúng paths tới FILE ZIP trong notebook:", "", "```python", 'MODE = "train"',
               'RESTORE_INPUT = "/kaggle/input/<dataset-results-b>/member_b_b4fd56f03a57_v3_results.zip"',
               'CROSS_PLAN_INPUT = "/kaggle/input/<dataset-cross-plan>/cross_plan.zip"', "RUN_IDS = None", "```", "",
               "Hai paths là ví dụ: thay bằng paths thực tế hiển thị trong Kaggle Input.",
               "Giữ Shared/RAW input hiện tại đúng dataset, rồi Run All. Restore phải vào session mới/thư mục results trống.",
               "Runner tự giữ 3 configs đã hoàn thành và chạy đúng 6 configs `G-bi` bổ sung; không chạy toàn bộ 42 YAML.",
               "Đủ 9/9 configs B thì tải Results ZIP mới. Giữ ZIP vòng đầu để đối chiếu.",
               "A tổng hợp ZIP A gốc, ZIP C gốc và ZIP B mới; chỉ mở test khi aggregate trả COMPLETED."]
    (output / "HUONG_DAN_B_CROSS_PLAN.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    for member, source in sources.items():
        check(digest(source) == hashes[member] and digest(archives[member]) == hashes[member], "Archive changed during review")
    print(json.dumps({"status": result["status"], "completed_configs": len(ranked), "completed_fits": len(ranked) * 3,
                      "leader": result["first_round_leader"], "winner": result["winner"], "cross_plan": cross["run_ids"],
                      "report": str(output / "ALL_MEMBERS_RESULTS.md"), "handoff": str(output / "cross_plan.zip")}, indent=2))


if __name__ == "__main__":
    main()
