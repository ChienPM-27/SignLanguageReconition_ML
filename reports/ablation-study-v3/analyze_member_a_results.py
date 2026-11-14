"""Archive review and prediction-based analysis of completed member A V3 results."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reports/ablation-study-v3/tools"))
import session_v3
import study_v3 as study

SEEDS = (42, 43, 44)
FEATURES = ("H", "B", "F", "HB", "HF", "BF", "HBF")
COMPARISONS = (("HB", "H"), ("HF", "H"), ("HBF", "H"), ("HBF", "HB"),
               ("HBF", "HF"), ("BF", "F"), ("BF", "B"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def distribution(values) -> dict:
    values = [float(v) for v in values]
    return study.aggregate(values)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inspect_archive(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        check(len(names) == len(set(names)), "Duplicate ZIP entries")
        check(bundle.testzip() is None, "ZIP CRC failure")
        for entry in entries:
            session_v3.relative(ROOT / "tmp/result_reviews/safety-check", entry.filename)
            check((entry.external_attr >> 16) & 0o170000 != 0o120000, "ZIP contains symlink")
        state = json.loads(bundle.read("queue_status.json"))
        status = json.loads(bundle.read("MEMBER_STATUS.json"))
        expected = {"G0__" + feature for feature in FEATURES}
        check(state["member"] == "A" and state["freeze_id"] == "b4fd56f03a57", "Wrong member/freeze")
        check(set(state["selected"]) == expected and not state["failed"], "A queue is incomplete")
        check(state["cross_plan"] is None, "Unexpected cross-plan for A")
        check(status["member"] == "A" and status["freeze_id"] == state["freeze_id"], "Status identity mismatch")
        check(status["completed"] == status["expected"] == 7, "Status must report 7/7")
        check({r["run_id"] for r in status["queue"]} == expected
              and all(r["status"] == "COMPLETED" for r in status["queue"]), "Status queue mismatch")
        test_artifacts = [n for n in names if Path(n).name in ("test_report.json", "evaluation_test_report.json",
                                                            "TEST_SUMMARY.json", "WINNER_LOCK.json")]
        check(not test_artifacts, "Test has been opened before global winner selection")
        return {"status": "PASS", "entries": len(entries), "uncompressed_bytes": sum(e.file_size for e in entries),
                "crc_valid": True, "duplicate_entries": False, "unsafe_paths": False, "symlinks": False,
                "test_artifacts": test_artifacts, "state": state}


def analyze_extracted(extracted: Path, state: dict, archive: Path) -> tuple[dict, dict]:
    configs = {}
    accepted = []
    shared_truth = None
    vocabulary = None
    for feature in FEATURES:
        run_id = "G0__" + feature
        prefix = state["selected"][run_id]
        attempt = session_v3.relative(extracted, prefix)
        canonical = study.collect(attempt)
        check(canonical["run_id"] == run_id and canonical["provenance"] == state["provenance"], "Selected identity mismatch")
        inventory = study.read(attempt / "ARTIFACTS.json")
        actual_files = {p.relative_to(attempt).as_posix() for p in attempt.rglob("*") if p.is_file()}
        check(actual_files == set(inventory) | {"metadata.json", "ARTIFACTS.json"}, "Uninventoried attempt payload")
        for stage in ("preprocess", "split", "train"):
            log = (attempt / "logs" / (stage + ".log")).read_text(encoding="utf-8")
            check(log.rstrip().endswith("EXIT_CODE: 0"), "Stage log lacks successful exit code")
        accepted.append(canonical)
        report_paths = sorted((attempt / "runs").rglob("val_report.json"), key=lambda p: p.parent.name)
        check([int(p.parent.name.removeprefix("seed_")) for p in report_paths] == list(SEEDS), "Wrong seeds")
        matrices, class_f1s, precisions, recalls, epoch_details = [], [], [], [], []
        recomputed_metrics = {key: [] for key in ("macro_f1", "top1_accuracy", "top3_accuracy")}
        participants = {}
        for report_path, seed in zip(report_paths, SEEDS):
            report = study.read(report_path)
            predictions = sorted(report["predictions"], key=lambda p: p["sequence_id"])
            truth = [(p["sequence_id"], p["participant_id"], p["y_true"]) for p in predictions]
            if shared_truth is None:
                shared_truth, vocabulary = truth, report["class_names"]
            check(truth == shared_truth and report["class_names"] == vocabulary, "Prediction alignment differs across fits")
            y_true = np.asarray([p["y_true"] for p in predictions])
            y_pred = np.asarray([p["y_pred"] for p in predictions])
            probability = np.asarray([p["probabilities"] for p in predictions])
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=np.arange(len(vocabulary)), zero_division=0)
            matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(vocabulary)))
            check(np.array_equal(matrix, report["confusion_matrix"]), "Reported confusion matrix differs")
            actual_metrics = {"macro_f1": float(f1.mean()), "top1_accuracy": float(np.mean(y_true == y_pred)),
                              "top3_accuracy": float(np.mean(np.any(np.argsort(probability, axis=1)[:, -3:] == y_true[:, None], axis=1)))}
            for key, value in actual_metrics.items():
                check(abs(value - report[key]) <= 0.000051, "Prediction metric differs from rounded report")
                recomputed_metrics[key].append(value)
            matrices.append(matrix)
            class_f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            for signer in sorted({p["participant_id"] for p in predictions}):
                subset = [p for p in predictions if p["participant_id"] == signer]
                values = participants.setdefault(signer, {"samples": len(subset), "accuracy": []})
                values["accuracy"].append(float(np.mean([p["y_true"] == p["y_pred"] for p in subset])))
            history = study.read(report_path.parent / "history.json")
            best = int(np.argmin(history["val_loss"]))
            count = len(history["val_loss"])
            check(all(len(history[k]) == count for k in ("train_loss", "train_acc", "val_acc", "val_f1", "lr")), "History lengths differ")
            check(abs(history["val_acc"][best] - report["top1_accuracy"]) <= 0.000051
                  and abs(history["val_f1"][best] - report["macro_f1"]) <= 0.000051, "Loss-selected history differs from report")
            check(abs(history["best_val_loss"] - history["val_loss"][best]) <= 1e-10, "Best loss differs from history minimum")
            epoch_details.append({"seed": seed, "epochs_run": count, "best_epoch": best + 1,
                                  "train_accuracy_at_best": history["train_acc"][best],
                                  "val_accuracy_at_best": history["val_acc"][best],
                                  "gap_at_best": history["train_acc"][best] - history["val_acc"][best],
                                  "smoothed_loss_at_best": history["val_loss"][best],
                                  "evaluation_loss": report["loss"], "training_time_sec": history["training_time_sec"],
                                  "seconds_per_epoch": history["training_time_sec"] / count,
                                  "checkpoint_sha256": digest(report_path.parent / "best_model.pt")})
        summary_paths = list((attempt / "runs").rglob("multi_seed_summary.json"))
        check(len(summary_paths) == 1, "Expected one multi-seed summary")
        summary = study.read(summary_paths[0])
        check(summary["seeds"] == list(SEEDS) and summary["num_seeds"] == 3, "Multi-seed identity differs")
        for key in ("macro_f1", "top1_accuracy", "top3_accuracy", "loss"):
            for moment in ("mean", "std"):
                check(abs(summary["val_" + key + "_" + moment] - canonical["metrics"][key][moment]) <= 0.000051,
                      "Multi-seed summary differs from reports")
        with (attempt / "cache_audit/manifest.csv").open(encoding="utf-8", newline="") as stream:
            manifest_rows = list(study.csv.DictReader(stream))
        check({r["sequence_id"] for r in manifest_rows} == set(study.read(attempt / "inputs/split_manifest.json")["sequences"]),
              "Cache audit cohort differs")
        width = {"hands": 143, "body": 17, "face": 36}
        cfg = study.load_config(str(attempt / "effective_config.yaml"))
        denominator = sum(cfg.model.region_weights[r] for r in cfg.model.regions)
        configs[run_id] = {"run_id": run_id, "feature": feature, "archive_prefix": prefix,
                          "dimension_cache": canonical["dimension"],
                          "active_region_channels": sum(width[r] for r in cfg.model.regions),
                          "regions": cfg.model.regions,
                          "normalized_weights": {r: cfg.model.region_weights[r] / denominator for r in cfg.model.regions},
                          "metrics": canonical["metrics"], "metrics_from_predictions": {k: distribution(v) for k,v in recomputed_metrics.items()},
                          "class_metrics": {name: {"support": int(support[i]), "f1": distribution(np.asarray(class_f1s)[:, i]),
                                                    "precision_mean": float(np.mean(precisions, axis=0)[i]),
                                                    "recall_mean": float(np.mean(recalls, axis=0)[i])} for i,name in enumerate(vocabulary)},
                          "mean_confusion_matrix": np.mean(matrices, axis=0).tolist(),
                          "participants": {s: {"samples": v["samples"], "accuracy": distribution(v["accuracy"])} for s,v in participants.items()},
                          "epoch_details": epoch_details,
                          "mean_gap_at_best": float(np.mean([e["gap_at_best"] for e in epoch_details])),
                          "hardware": study.read(attempt / "metadata.json")["hardware"]}
        print("Accepted", run_id, "Macro-F1", canonical["metrics"]["macro_f1"], flush=True)
    check(all(r["provenance"] == accepted[0]["provenance"] and r["memberships"] == accepted[0]["memberships"] for r in accepted),
          "Shared provenance/cohort differs")
    ranking = [row["run_id"] for row in study.rank(accepted)]
    deltas = {}
    for upper, lower in COMPARISONS:
        a, b = configs["G0__" + upper], configs["G0__" + lower]
        difference = np.asarray(a["metrics"]["macro_f1"]["per_seed"]) - np.asarray(b["metrics"]["macro_f1"]["per_seed"])
        deltas[upper + " minus " + lower] = {**distribution(difference), "positive_seeds": int(np.sum(difference > 0)),
                                             "negative_seeds": int(np.sum(difference < 0))}
    class_changes = {}
    for upper, lower in COMPARISONS[:5]:
        changes = [{"gloss": c, "delta": configs["G0__" + upper]["class_metrics"][c]["f1"]["mean"]
                    - configs["G0__" + lower]["class_metrics"][c]["f1"]["mean"]} for c in vocabulary]
        class_changes[upper + " minus " + lower] = {"largest_gains": sorted(changes, key=lambda r: r["delta"], reverse=True)[:5],
                                                    "largest_losses": sorted(changes, key=lambda r: r["delta"])[:5]}
    confusions = {}
    for run_id, row in configs.items():
        matrix = np.asarray(row["mean_confusion_matrix"])
        pairs = [{"true": vocabulary[i], "predicted": vocabulary[j], "mean_count": float(matrix[i,j]),
                  "support": row["class_metrics"][vocabulary[i]]["support"], "rate": float(matrix[i,j] / matrix[i].sum())}
                 for i in range(len(vocabulary)) for j in range(len(vocabulary)) if i != j]
        confusions[run_id] = sorted(pairs, key=lambda p: p["mean_count"], reverse=True)[:10]
    data = {"status": "ACCEPTED", "campaign": "v3-regions-701020", "member": "A", "freeze_id": state["freeze_id"],
            "archive": str(archive.resolve()), "archive_sha256": digest(archive), "seeds": list(SEEDS),
            "completed_configs": 7, "completed_fits": 21, "provenance": state["provenance"],
            "classes": vocabulary, "split_counts": {k: len(v) for k,v in accepted[0]["memberships"].items()},
            "unique_validation_sequences": len(shared_truth), "validation_signers": sorted(configs["G0__H"]["participants"]),
            "configs": configs, "ranking_among_A": ranking, "best_among_A": ranking[0], "global_winner": None,
            "phase1_completed": 7, "phase1_expected": 24, "cross_plan": None,
            "missing_phase1_runs": [r["run_id"] for r in study.matrix() if r["phase"] == 1 and r["owner"] != "A"],
            "paired_region_deltas": deltas, "per_class_region_changes": class_changes, "top_confusions": confusions,
            "training_time_total_sec": sum(sum(r["metrics"]["training_time_sec"]["per_seed"]) for r in configs.values()),
            "statistical_note": "Population std của 3 seeds; không phải confidence interval. Các fits dùng chung 1.155 validation clips.",
            "limitations": ["Kết luận chỉ cho GRU G0 trên cohort và fixed-fusion protocol này.",
                            "Chưa nhận B/C; chưa chọn GRU architecture hoặc global winner, chưa mở test.",
                            "ZIP không chứa .npy feature cache hoặc raw clips nên chưa đo chất lượng body/face landmarks trực tiếp.",
                            "Không chạy training, checkpoint inference hoặc test evaluation trong lần phân tích này."]}
    review = {"status": "ACCEPTED", "member": "A", "freeze_id": state["freeze_id"], "configs": 7, "fits": 21,
              "archive": data["archive"], "archive_sha256": data["archive_sha256"], "provenance": state["provenance"],
              "split_counts": data["split_counts"], "unique_validation_sequences": len(shared_truth),
              "checks": ["7 selected COMPLETED attempts; exact A ownership; no failures",
                         "Canonical study_v3.collect passed for all seven attempts",
                         "Source/config/matrix/protocol/split hashes match freeze b4fd56f03a57",
                         "Exact artifact inventory and all 21 checkpoint checksums verified",
                         "Stage logs end with EXIT_CODE: 0; effective configs and cache identities match",
                         "21 aligned validation prediction sets; vocabulary/labels/probabilities/argmax verified",
                         "Top-1/Top-3/Macro-F1/confusion matrices recomputed from predictions",
                         "History loss-selected epochs and multi-seed summaries match reports",
                         "No test reports or global winner lock; B/C remain pending"],
              "training_executed_during_review": False, "inference_executed_during_review": False,
              "test_evaluated": False, "global_winner": None,
              "attempts": {r["run_id"]: {"archive_prefix": state["selected"][r["run_id"]], "status": r["status"],
                                        "metrics": r["metrics"]} for r in accepted}}
    return data, review


def plots(data: dict, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for i, feature in enumerate(FEATURES):
        metrics = data["configs"]["G0__" + feature]["metrics"]["macro_f1"]
        color = "#147d78" if "H" in feature else "#778291"
        ax.scatter(i + np.asarray([-.1, 0, .1]), np.asarray(metrics["per_seed"]) * 100, color=color, alpha=.65, s=32, zorder=3)
        ax.errorbar(i, metrics["mean"] * 100, yerr=metrics["std"] * 100, fmt="D", color=color, capsize=5, zorder=4)
        ax.annotate(f"{metrics['mean'] * 100:.2f}", (i, (metrics["mean"] + metrics["std"]) * 100),
                    xytext=(0, 9), textcoords="offset points", ha="center")
    ax.set_xticks(range(len(FEATURES)), FEATURES)
    ax.set_ylim(0, 81)
    ax.set_ylabel("Validation Macro-F1 (%)")
    ax.set_title("Member A V3 — GRU G0 trên bảy tổ hợp vùng", pad=18)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .025, "H: tay · B: thân · F: mặt. Chấm: seeds 42/43/44; hình thoi: mean ± population std (không phải CI).",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(output / "macro_f1_seed_comparison.png", dpi=180)
    plt.close(fig)

    classes = sorted(data["classes"], key=lambda c: data["configs"]["G0__H"]["class_metrics"][c]["f1"]["mean"])
    values = np.asarray([[data["configs"]["G0__" + f]["class_metrics"][c]["f1"]["mean"] for f in FEATURES] for c in classes])
    fig, ax = plt.subplots(figsize=(9.2, 8.4))
    heat = ax.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FEATURES)), FEATURES)
    ax.set_yticks(range(len(classes)), [f"{c} (n={data['configs']['G0__H']['class_metrics'][c]['support']})" for c in classes])
    for i in range(len(classes)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{values[i,j] * 100:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if values[i,j] < .6 else "black")
    ax.set_title("Validation F1 từng gloss — trung bình ba seeds (%)", pad=15)
    fig.colorbar(heat, ax=ax, fraction=.045, pad=.04, label="F1 (0–1)")
    fig.tight_layout()
    fig.savefig(output / "per_class_f1.png", dpi=180)
    plt.close(fig)


def report(data: dict, output: Path) -> str:
    lines = ["# Phân tích member A — Ablation V3", "",
             "Member A đã nghiệm thu 7/7 configs, đủ 21 fits. H và HB cùng mean Validation Macro-F1 0.7120 "
             "trên các report làm tròn bốn chữ số; H được xếp trước theo rule đã freeze vì population std thấp hơn. "
             "HBF không tăng mean so với H/HB trong GRU G0 và fixed-fusion protocol này. Chưa chọn winner toàn campaign.", "",
             f"Freeze `{data['freeze_id']}`; cohort 7.854 sequences; train/val/test = 5.558/1.155/1.141. "
             "Validation gồm 20 glosses và ba signers không có trong train; seeds 42/43/44.", "",
             "## Kết quả Validation", "",
             "| Vùng | Channels bật | Macro-F1 mean ± std | Top-1 | Top-3 | Loss đánh giá | Train TB/seed |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for feature in FEATURES:
        row = data["configs"]["G0__" + feature]
        m = row["metrics"]
        lines.append(f"| {feature} | {row['active_region_channels']} | {m['macro_f1']['mean']:.4f} ± {m['macro_f1']['std']:.4f} | "
                     f"{m['top1_accuracy']['mean'] * 100:.2f}% | {m['top3_accuracy']['mean'] * 100:.2f}% | "
                     f"{m['loss']['mean']:.4f} | {m['training_time_sec']['mean']:.1f}s |")
    lines += ["", "H = tay; B = thân trên; F = mặt. Cache chung 196 channels. Các branches được đăng ký ở mọi subset, "
              "nên tổng parameters đều 228.372; chỉ regions bật được đọc trong forward. Trọng số được chuẩn hóa trên "
              "các vùng bật: H/B/F đứng riêng có weight 1; HB = 7/9 và 2/9; HF = 7/8 và 1/8; BF = 2/3 và 1/3; HBF = 0.7/0.2/0.1.", "",
              "Số liệu mean/std và xếp hạng dùng report metrics đã làm tròn bốn chữ số, đúng canonical collector. "
              "ANALYSIS.json cũng lưu metrics recompute từ predictions ở độ chính xác đầy đủ. H/HB gần như ngang nhau; "
              "tie-break theo std không phải bằng chứng H tốt hơn có ý nghĩa thống kê.", "",
              "## Đóng góp khi thêm vùng", "",
              "| So sánh Macro-F1 | Delta mean (điểm %) | Delta seeds 42/43/44 (điểm %) | Seeds tăng |",
              "|---|---:|---|---:|"]
    for comparison, delta in data["paired_region_deltas"].items():
        mean = 0.0 if abs(delta["mean"]) < 1e-8 else delta["mean"] * 100
        points = ", ".join(f"{v * 100:+.2f}" for v in delta["per_seed"])
        lines.append(f"| {comparison} | {mean:+.2f} | {points} | {delta['positive_seeds']}/3 |")
    lines += ["", "Tay là nguồn tín hiệu mạnh nhất khi dùng riêng. B-only và F-only vẫn cao hơn mức đoán ngẫu nhiên "
              "Top-1 5%, nhưng thấp hơn rõ rệt các subsets có tay. B chỉ chứa vai/khuỷu tay và thông tin torso, "
              "không có hình dạng ngón tay; F là biểu diễn face compact. Điều này không chứng minh raw landmarks bị lỗi "
              "hoặc body/face vô dụng với mọi architecture.", "",
              "Thêm body vào H: mean gần 0, hướng chênh đảo giữa seeds và std tăng. Thêm face vào HB: mean giảm khoảng "
              "0.51 điểm %, giảm cả ba seeds. HF cũng thấp hơn H trung bình khoảng 0.47 điểm %. BF thấp hơn F khoảng "
              "1.81 điểm %. Kết luận giới hạn trong GRU G0, normalization/projection và fixed weights hiện tại; "
              "chưa kiểm định các tỷ lệ fusion khác hoặc architecture B/C.", "",
              "Đây là paired comparisons theo cùng training seeds, không phải causal estimate riêng của từng vùng: "
              "khi đổi subset, weights của những vùng còn lại cũng được chuẩn hóa lại và model được train mới.", "",
              "## Khác biệt theo gloss", "",
              "| Gloss khó với H | Clips | F1 H | F1 HB | F1 HF | F1 HBF |", "|---|---:|---:|---:|---:|---:|"]
    weakest = sorted(data["configs"]["G0__H"]["class_metrics"], key=lambda c: data["configs"]["G0__H"]["class_metrics"][c]["f1"]["mean"])[:6]
    for gloss in weakest:
        values = [data["configs"]["G0__" + f]["class_metrics"][gloss]["f1"]["mean"] * 100 for f in ("H", "HB", "HF", "HBF")]
        lines.append(f"| {gloss} | {data['configs']['G0__H']['class_metrics'][gloss]['support']} | "
                     + " | ".join(f"{v:.2f}%" for v in values) + " |")
    changes = data["per_class_region_changes"]["HBF minus H"]
    lines += ["", "HBF so với H, các gloss tăng nhiều nhất: " + "; ".join(f"{r['gloss']} {r['delta'] * 100:+.2f} điểm %" for r in changes["largest_gains"][:3]) + ".",
              "Các gloss giảm nhiều nhất: " + "; ".join(f"{r['gloss']} {r['delta'] * 100:+.2f} điểm %" for r in changes["largest_losses"][:3]) + ".", "",
              "| Nhầm lẫn chính của H | Số lỗi TB/seed | Tỷ lệ trong lớp thật |", "|---|---:|---:|"]
    for pair in data["top_confusions"]["G0__H"][:6]:
        lines.append(f"| {pair['true']} → {pair['predicted']} | {pair['mean_count']:.1f} | {pair['rate'] * 100:.1f}% |")
    lines += ["", "Tỷ lệ được tính trên cùng clips ở ba seeds; số lỗi trung bình có thể thập phân. "
              "Cần xem raw clip để diễn giải cặp nhầm, không kết luận nhãn dataset sai từ confusion matrix.", "",
              "## Khác biệt theo signer", "", "| Signer Validation | Clips | H Top-1 | HB Top-1 | HF Top-1 | HBF Top-1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for signer in data["validation_signers"]:
        n = data["configs"]["G0__H"]["participants"][signer]["samples"]
        values = [data["configs"]["G0__" + f]["participants"][signer]["accuracy"]["mean"] * 100 for f in ("H", "HB", "HF", "HBF")]
        lines.append(f"| {signer} | {n} | " + " | ".join(f"{v:.2f}%" for v in values) + " |")
    lines += ["", "## Training và khả năng tổng quát", "",
              "| Vùng | Epochs 42/43/44 | Best epochs theo Val Loss | Gap Train–Val tại best TB |", "|---|---|---|---:|"]
    for feature in FEATURES:
        row = data["configs"]["G0__" + feature]
        lines.append(f"| {feature} | " + "/".join(str(e["epochs_run"]) for e in row["epoch_details"]) + " | "
                     + "/".join(str(e["best_epoch"]) for e in row["epoch_details"]) + f" | {row['mean_gap_at_best'] * 100:.2f} điểm % |")
    lines += ["", f"Tổng training-loop time của 21 fits: {data['training_time_total_sec'] / 60:.2f} phút trên Tesla T4. "
              "Không tính preprocessing, split, report/plots/export hoặc session idle; không phải inference latency. "
              "Số epochs khác nhau do early stopping. Train accuracy có augmentation/dropout, Val ở eval mode, "
              "nên gap không tự xác định nguyên nhân overfitting.", "",
              "Checkpoint được chọn theo Val Loss ở vòng train có label_smoothing=0.1. Loss trong val_report dùng "
              "CrossEntropy không smoothing; hai thang loss khác nhau là đúng code hiện tại. Best epochs/accuracy/F1 "
              "đã được đối chiếu với predictions và reports.", "",
              "## Nghiệm thu, lưu trữ và bước tiếp theo", "",
              "Canonical study_v3.collect đã pass cả 7 attempts. Đã xác minh inventory, checkpoint hashes, successful "
              "stage logs, config/source/freeze/split/cache identities, đủ predictions cho 1.155 clips mỗi seed, "
              "probabilities/argmax/labels, confusion matrices và multi-seed summary. Không có test report hoặc winner lock.", "",
              "A hoàn tất; không cần train lại A cho campaign hiện tại. B/C tiếp tục đúng queue. Chờ đủ 24 configs vòng đầu, "
              "chọn GRU architecture theo HBF; chỉ khi winner architecture khác G0 mới gửi B chạy sáu subsets bổ sung. "
              "Winner toàn campaign và test tiếp tục giữ pending. Có thể ưu tiên kiểm tra chất lượng body/face cache "
              "ở một thí nghiệm chẩn đoán riêng, nhưng ZIP này chỉ chứa cache manifest/metadata, không chứa raw clips/.npy.", "",
              "Population std qua ba seeds không phải confidence interval; 21 fits vẫn dùng cùng 1.155 validation clips. "
              "Không suy luận rằng 70/20/10 tối ưu, hoặc cơ chế landmarks/noise đã được chứng minh.", "",
              f"ZIP lưu nguyên trạng: [{Path(data['archive']).name}]({Path(data['archive']).as_posix()}).",
              f"SHA256: `{data['archive_sha256']}`. Bản Downloads được giữ nguyên.", "",
              f"[Dữ liệu phân tích]({(output / 'ANALYSIS.json').as_posix()}) · "
              f"[Nghiệm thu]({(output.parent / 'REVIEW.json').as_posix()}) · "
              f"[So sánh Macro-F1]({(output / 'macro_f1_seed_comparison.png').as_posix()}) · "
              f"[F1 từng gloss]({(output / 'per_class_f1.png').as_posix()})"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    archive, source = args.archive.resolve(), args.source.resolve()
    archive_hash = digest(archive)
    check(digest(source) == archive_hash, "Stored ZIP differs from source")
    expected_root = ROOT / "artifacts/ablation-study-v3/results/member-a/b4fd56f03a57" / archive_hash[:12]
    check(archive.parent == expected_root.resolve(), "Archive must be stored by freeze and SHA256 prefix")
    audit = inspect_archive(archive)
    state = audit.pop("state")
    write_json(archive.parent / "ZIP_CHECK.json", {**audit, "sha256": archive_hash, "bytes": archive.stat().st_size})
    temp_root = ROOT / "tmp/result_reviews"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v3-member-a-", dir=temp_root) as temporary:
        extracted = Path(temporary).resolve()
        check(extracted.is_relative_to(temp_root.resolve()) and extracted != temp_root.resolve(), "Unsafe temporary cleanup root")
        session_v3.extract(archive, extracted)
        data, review = analyze_extracted(extracted, state, archive)
    output = archive.parent / "analysis"
    output.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone(timedelta(hours=7))).isoformat()
    review["reviewed_at_asia_saigon"] = timestamp
    write_json(archive.parent / "REVIEW.json", review)
    write_json(archive.parent / "STORAGE.json", {"operation": "copy", "source": str(source), "destination": str(archive),
                                               "sha256": archive_hash, "bytes": archive.stat().st_size,
                                               "stored_at_asia_saigon": timestamp, "hash_verified": True, "source_removed": False})
    write_json(output / "ANALYSIS.json", data)
    (output / "ANALYSIS.md").write_text(report(data, output), encoding="utf-8")
    plots(data, output)
    check(digest(archive) == archive_hash and digest(source) == archive_hash, "Original ZIP changed during review")
    print(json.dumps({"status": review["status"], "archive": str(archive), "report": str(output / "ANALYSIS.md"),
                      "ranking": data["ranking_among_A"], "paired_deltas": data["paired_region_deltas"],
                      "training_time_min": data["training_time_total_sec"] / 60}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
