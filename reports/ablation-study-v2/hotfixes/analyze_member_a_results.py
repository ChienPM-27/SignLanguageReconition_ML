"""Analyze accepted A predictions/history; never execute training or inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

REPO = Path(__file__).resolve().parents[3]
VERSION = REPO / "artifacts/ablation-study-v2/results/member-a/77a84a7b12f1/1017bac26f59"
ARCHIVE = VERSION / "member_a_77a84a7b12f1_face_only_results.zip"
OUTPUT = VERSION / "analysis"


def distribution(values) -> dict:
    data = np.asarray(values, dtype=float)
    return {"mean": float(data.mean()), "std": float(data.std()), "per_seed": data.tolist()}


def analyze() -> dict:
    review = json.loads((VERSION / "REVIEW.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == review["face_only_sha256"]
    results = {}
    shared_truth = None
    with zipfile.ZipFile(ARCHIVE) as archive:
        summary = json.loads(archive.read("MEMBER_RESULTS.json"))
        state = json.loads(archive.read("queue_status.json"))
        for run_id, record in sorted(summary["records"].items()):
            assert run_id.endswith("-face")
            relative = state["selected"][run_id]
            names = [name for name in archive.namelist()
                     if name.startswith(relative + "/runs/") and name.endswith("/val_report.json")]
            ordered = sorted(names, key=lambda name: int(name.split("/")[-2].removeprefix("seed_")))
            assert [int(n.split("/")[-2].removeprefix("seed_")) for n in ordered] == [42, 43, 44]
            f1s, precisions, recalls, matrices, epochs = [], [], [], [], []
            participant_values = {}
            for name in ordered:
                report = json.loads(archive.read(name))
                predictions = sorted(report["predictions"], key=lambda item: item["sequence_id"])
                truth = [(p["sequence_id"], p["participant_id"], p["y_true"]) for p in predictions]
                if shared_truth is None:
                    shared_truth = truth
                assert truth == shared_truth and len(truth) == 1155
                classes = report["class_names"]
                yt = np.array([p["y_true"] for p in predictions])
                yp = np.array([p["y_pred"] for p in predictions])
                probability = np.array([p["probabilities"] for p in predictions])
                assert probability.shape == (1155, len(classes))
                assert np.isfinite(probability).all() and (probability >= 0).all()
                np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)
                np.testing.assert_array_equal(probability.argmax(axis=1), yp)
                precision, recall, f1, support = precision_recall_fscore_support(
                    yt, yp, labels=np.arange(len(classes)), zero_division=0)
                top3 = np.mean(np.any(np.argsort(-probability, axis=1)[:, :3] == yt[:, None], axis=1))
                assert abs(f1.mean() - report["macro_f1"]) <= 0.000051
                assert abs(np.mean(yt == yp) - report["top1_accuracy"]) <= 0.000051
                assert abs(top3 - report["top3_accuracy"]) <= 0.000051
                f1s.append(f1); precisions.append(precision); recalls.append(recall)
                matrices.append(confusion_matrix(yt, yp, labels=np.arange(len(classes))))
                for signer in sorted({p["participant_id"] for p in predictions}):
                    subset = [p for p in predictions if p["participant_id"] == signer]
                    values = participant_values.setdefault(signer, {"samples": len(subset), "accuracy": []})
                    values["accuracy"].append(np.mean([p["y_true"] == p["y_pred"] for p in subset]))
                history = json.loads(archive.read(name.replace("val_report.json", "history.json")))
                best = int(np.argmin(history["val_loss"]))
                np.testing.assert_allclose(history["val_acc"][best], report["top1_accuracy"], atol=0.000051)
                epochs.append({"seed": int(name.split("/")[-2].removeprefix("seed_")),
                    "epochs_run": len(history["train_loss"]), "best_epoch": best + 1,
                    "train_accuracy_at_best": history["train_acc"][best],
                    "val_accuracy_at_best": history["val_acc"][best],
                    "gap_at_best": history["train_acc"][best] - history["val_acc"][best],
                    "training_time_sec": history["training_time_sec"],
                    "time_per_epoch_sec": history["training_time_sec"] / len(history["train_loss"])})
            mean_f1 = np.mean(f1s, axis=0)
            mean_matrix = np.mean(matrices, axis=0)
            results[run_id] = {"dimension": record["dimension"], "metrics": record["metrics"],
                "class_metrics": {name: {"support": int(support[i]),
                    "f1": distribution(np.array(f1s)[:, i]),
                    "precision_mean": float(np.mean(precisions, axis=0)[i]),
                    "recall_mean": float(np.mean(recalls, axis=0)[i])} for i, name in enumerate(classes)},
                "macro_f1_from_predictions": float(mean_f1.mean()), "mean_confusion_matrix": mean_matrix.tolist(),
                "participants": {s: {"samples": v["samples"], "accuracy": distribution(v["accuracy"])}
                                 for s, v in participant_values.items()},
                "epoch_details": epochs, "mean_gap_at_best": float(np.mean([e["gap_at_best"] for e in epochs])),
                "mean_time_per_epoch_sec": float(np.mean([e["time_per_epoch_sec"] for e in epochs]))}
    deltas = {}
    for upper, lower in (("F2", "F1"), ("F3", "F2"), ("F4", "F3"), ("F3", "F1"), ("F4", "F1")):
        a, b = f"G0__{upper}-face", f"G0__{lower}-face"
        difference = np.array(results[a]["metrics"]["macro_f1"]["per_seed"]) - np.array(results[b]["metrics"]["macro_f1"]["per_seed"])
        deltas[f"{upper}-face minus {lower}-face"] = {**distribution(difference),
            "positive_seeds": int(np.sum(difference > 0)), "negative_seeds": int(np.sum(difference < 0))}
    baseline = results["G0__F1-face"]
    matrix = np.array(baseline["mean_confusion_matrix"])
    pairs = sorted([{"true": classes[i], "predicted": classes[j], "mean_count": float(matrix[i, j]),
                     "true_class_support": baseline["class_metrics"][classes[i]]["support"],
                     "rate": float(matrix[i, j] / matrix[i].sum())}
                    for i in range(len(classes)) for j in range(len(classes)) if i != j],
                   key=lambda row: row["mean_count"], reverse=True)
    return {"scope": "Bốn config GRU G0 có face đã nghiệm thu của A; F0-face và các profiles khác đang chờ C/B",
            "source_archive": str(ARCHIVE), "sha256": review["face_only_sha256"],
            "seeds": [42, 43, 44], "unique_validation_sequences": 1155, "classes": classes,
            "validation_signers": sorted(baseline["participants"]), "configs": results,
            "paired_seed_deltas": deltas, "top_f1_confusions": pairs[:10],
            "best_among_A_by_mean": "G0__F1-face", "global_winner": None,
            "training_time_total_sec": sum(sum(r["metrics"]["training_time_sec"]["per_seed"]) for r in results.values()),
            "statistical_note": "Std của ba training seeds là population std, không phải confidence interval. Các seeds dùng lại cùng 1.155 clips validation, không tạo thêm mẫu độc lập."}


def plots(data: dict) -> None:
    ids = list(data["configs"])
    labels = [name.removeprefix("G0__") for name in ids]
    fig, ax = plt.subplots(figsize=(9, 4.9))
    colors = ["#168b82", "#6b7280", "#4769b0", "#ba6338"]
    all_points = []
    for index, name in enumerate(ids):
        values = data["configs"][name]["metrics"]["macro_f1"]
        points = np.array(values["per_seed"]) * 100
        all_points.extend(points)
        ax.scatter(index + np.array([-.1, 0, .1]), points, color=colors[index], alpha=.6, s=42, zorder=3)
        ax.errorbar(index, values["mean"] * 100, yerr=values["std"] * 100, fmt="D", color=colors[index],
                    markersize=7, capsize=7, linewidth=2, zorder=4)
        ax.annotate(f"{values['mean'] * 100:.2f}", (index, values["mean"] * 100 + values["std"] * 100),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=11)
    ax.set_xticks(range(len(ids)), labels)
    ax.set_ylabel("Validation Macro-F1 (%)")
    ax.set_title("Member A — GRU G0 với bốn bộ feature có face", fontsize=14, pad=15)
    ax.set_ylim(min(all_points) - 1, max(all_points) + 1.5)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .035, "Chấm nhỏ: seeds 42/43/44. Hình thoi và thanh: mean ± std; không phải confidence interval.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(OUTPUT / "macro_f1_seed_comparison.png", dpi=180)
    plt.close(fig)

    classes = sorted(data["classes"], key=lambda c: data["configs"][ids[0]]["class_metrics"][c]["f1"]["mean"])
    values = np.array([[data["configs"][name]["class_metrics"][c]["f1"]["mean"] for name in ids] for c in classes])
    fig, ax = plt.subplots(figsize=(7.4, 8.5))
    heat = ax.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ids)), labels)
    ax.set_yticks(range(len(classes)), [f"{c} (n={data['configs'][ids[0]]['class_metrics'][c]['support']})" for c in classes])
    for i in range(len(classes)):
        for j in range(len(ids)):
            ax.text(j, i, f"{values[i, j] * 100:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if values[i, j] < .6 else "black")
    ax.set_title("F1 từng lớp: trung bình qua ba seeds (%)", fontsize=13, pad=14)
    fig.colorbar(heat, ax=ax, fraction=.05, pad=.04, label="F1 (0–1)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "per_class_f1.png", dpi=180)
    plt.close(fig)


def report(data: dict) -> str:
    lines = ["# Phân tích kết quả nghiệm thu member A", "",
        "F1-face là lựa chọn tạm thời tốt nhất trong bốn config GRU G0 của A. F3-face rất sát, "
        "nhưng dùng thêm parameters và thời gian training. F4-face không cải thiện kết quả trong protocol này. "
        "Chưa có baseline G0/F0-face của C và các profiles B/C, nên chưa chọn winner toàn bộ 30 cells.", "",
        "Đã nghiệm thu 4/4 configs, mỗi config đủ seeds42/43/44; các metrics dưới đây được đối chiếu lại "
        "từ predictions. Có 1.155 clips validation, 20 glosses và ba signers chưa có trong training. "
        "Bốn config dùng cùng dataset/split/model/training policy và đều bật face.", "",
        "| Config | D | Val Macro-F1 mean ± std | Top-1 | Top-3 | Parameters | Train TB/seed |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in data["configs"].items():
        m = row["metrics"]
        lines.append(f"| {name.removeprefix('G0__')} | {row['dimension']} | {m['macro_f1']['mean']:.4f} ± {m['macro_f1']['std']:.4f} | "
            f"{m['top1_accuracy']['mean'] * 100:.2f}% | {m['top3_accuracy']['mean'] * 100:.2f}% | {int(m['parameters']['mean']):,} | "
            f"{m['training_time_sec']['mean']:.1f}s |")
    lines += ["", "Train TB/seed là thời gian trong training loop của history.json; không gồm preprocess, "
        "download, plots/report/export hoặc thời gian chờ của phiên Kaggle. Số epochs khác nhau do early stopping; "
        "đây chưa phải benchmark inference latency. Tổng training-loop time của 12 fits có face là "
        f"{data['training_time_total_sec'] / 60:.1f} phút.", "",
        "| So sánh Macro-F1 | Delta mean (điểm %) | Delta seeds42/43/44 (điểm %) | Seeds cải thiện |",
        "|---|---:|---|---:|"]
    for name, delta in data["paired_seed_deltas"].items():
        lines.append(f"| {name} | {delta['mean'] * 100:+.2f} | " + ", ".join(f"{v * 100:+.2f}" for v in delta["per_seed"]) + f" | {delta['positive_seeds']}/3 |")
    lines += ["",
        "F1 → F2 chỉ thay palm scale từ frame sang clip_median: mean Macro-F1 giảm 1.13 điểm %, "
        "và giảm trên cả ba seeds. F2 có std thấp nhất nhưng cũng có mean thấp hơn F1/F3. "
        "Điều này không chứng minh clip_median luôn kém trên mọi model/dataset.", "",
        "F2 → F3 thêm 30 fingertip displacement channels: mean tăng 1.04 điểm %, cải thiện cả ba seeds. "
        "F3 khôi phục gần mức F1; chênh F3–F1 chỉ −0.09 điểm %, trong khi hướng chênh thay đổi giữa seeds. "
        "Không khẳng định F1 vượt F3 có ý nghĩa thống kê chỉ từ ba runs này.", "",
        "F3 → F4 thêm 20 bend-cosine channels: mean giảm 1.87 điểm %, hai trong ba seeds giảm. "
        "F4 thấp hơn F1 trên cả ba seeds; std F4 là 1.61 điểm % so với 0.71 ở F1. "
        "Feature nhiều hơn không bảo đảm tốt hơn; nguyên nhân có thể liên quan redundancy/noise, "
        "generalization hoặc sự phù hợp với GRU/protocol. Chưa có thí nghiệm riêng để xác nhận nguyên nhân.", "",
        "F1 có mean Macro-F1, Top-1, Top-3 tốt nhất và loss đánh giá thấp nhất. F3 dùng thêm khoảng 5% parameters "
        "và training-loop time TB/seed cao hơn khoảng 35% so với F1. F4 dùng thêm khoảng 8.3% parameters "
        "và training-loop time TB/seed gấp khoảng 2.11 lần F1 trong các logs này.", "",
        "| Config | Epochs seeds42/43/44 | Best epochs theo val loss | Train–val accuracy gap TB tại best epoch |",
        "|---|---|---|---:|"]
    for name, row in data["configs"].items():
        lines.append(f"| {name.removeprefix('G0__')} | " + "/".join(str(e["epochs_run"]) for e in row["epoch_details"]) +
                     " | " + "/".join(str(e["best_epoch"]) for e in row["epoch_details"]) + f" | {row['mean_gap_at_best'] * 100:.2f} điểm % |")
    lines += ["", "Gap tăng từ khoảng 10.8 điểm % ở F1 tới 14.5 ở F4, gợi ý cần quan sát generalization. "
        "Train accuracy được đo khi training có augmentation/dropout; val accuracy đo ở eval mode, "
        "nên gap này không tự chứng minh overfitting hoặc xác định nguyên nhân. Checkpoint được chọn theo "
        "validation loss có label smoothing; loss trong val_report.json là loss đánh giá, không đối chiếu "
        "trực tiếp hai giá trị này như một lỗi training.", "",
        "Các lớp khó nhất của F1-face:", "",
        "| Gloss | Support clips | F1 TB | Recall TB |", "|---|---:|---:|---:|"]
    baseline = data["configs"]["G0__F1-face"]
    weakest = sorted(baseline["class_metrics"].items(), key=lambda item: item[1]["f1"]["mean"])[:5]
    for name, row in weakest:
        lines.append(f"| {name} | {row['support']} | {row['f1']['mean'] * 100:.2f}% | {row['recall_mean'] * 100:.2f}% |")
    lines += ["", "Support các lớp nằm trong khoảng 52–63 clips, khá đều. Nhóm lỗi này không thể giải thích "
        "đơn giản bằng việc một vài lớp có quá ít mẫu trong validation.", "",
        "Tác dụng của từng feature khác nhau theo lớp: F2 tăng F1 của awake khoảng 10.16 điểm % so với F1, "
        "nhưng giảm bird/nuts/who. F3 tăng mouse/pretend/doll khoảng 4.28/3.90/3.46 điểm % so với F2. "
        "F4 giảm pretend/awake/listen khoảng 9.36/9.28/6.27 điểm % so với F3, dù cải thiện wake/lips. "
        "Vì Macro-F1 cho mỗi gloss trọng số bằng nhau, gains cục bộ chưa đủ bù các lớp bị giảm.", "",
        "| Nhãn thật → nhãn dự đoán | Số lỗi TB/seed | Tỷ lệ trong lớp thật |", "|---|---:|---:|"]
    for pair in data["top_f1_confusions"][:5]:
        lines.append(f"| {pair['true']} → {pair['predicted']} | {pair['mean_count']:.1f} | {pair['rate'] * 100:.1f}% |")
    lines += ["", "Bảng nhầm lẫn là trung bình ba seeds trên cùng clips; số lỗi trung bình có thể là số thập phân. "
        "Cần xem landmark/clip tương ứng để hiểu nguyên nhân; bảng này không chứng minh nhãn dataset sai.", "",
        "| Validation signer | Clips | F1-face Top-1 TB | F3-face Top-1 TB | F4-face Top-1 TB |", "|---|---:|---:|---:|---:|"]
    for signer, row in baseline["participants"].items():
        lines.append(f"| {signer} | {row['samples']} | {row['accuracy']['mean'] * 100:.2f}% | "
            f"{data['configs']['G0__F3-face']['participants'][signer]['accuracy']['mean'] * 100:.2f}% | "
            f"{data['configs']['G0__F4-face']['participants'][signer]['accuracy']['mean'] * 100:.2f}% |")
    lines += ["", "Độ khó giữa signers có khác biệt. Chỉ ba signers validation nên kết luận cần giới hạn "
        "trong cohort hiện tại; 12 fits không tạo thành 13.860 clips validation độc lập.", "",
        "B/C tiếp tục đúng năm feature levels có face để kiểm tra sự phù hợp với architecture/model khác. "
        "C cung cấp G0/F0-face để hoàn thành so sánh F1–F0 và đo tác dụng của torso-relative wrist. "
        "Không cần train lại A; chỉ chốt winner khi đủ 30 cells và sau đó mới mở test.", "",
        f"Artifacts: [So sánh Macro-F1 và seeds]({(OUTPUT / 'macro_f1_seed_comparison.png').as_posix()}), "
        f"[F1 từng lớp]({(OUTPUT / 'per_class_f1.png').as_posix()}), "
        f"[Dữ liệu phân tích]({(OUTPUT / 'ANALYSIS.json').as_posix()}).", "",
        f"Nguồn: [ZIP đã nghiệm thu]({ARCHIVE.as_posix()}); SHA256 `{data['sha256']}`. Không chạy training hoặc inference trong phân tích này."]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = analyze()
    (OUTPUT / "ANALYSIS.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "ANALYSIS.md").write_text(report(data), encoding="utf-8")
    plots(data)
    print(json.dumps({"report": str(OUTPUT / "ANALYSIS.md"), "paired_deltas": data["paired_seed_deltas"],
                      "top_confusions": data["top_f1_confusions"][:5]}, indent=2))


if __name__ == "__main__":
    main()
