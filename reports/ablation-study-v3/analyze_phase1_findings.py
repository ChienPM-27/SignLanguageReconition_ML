"""Describe accepted A/B/C validation results without training or opening test."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/3e125a3e5bd9"
OUTPUT = SOURCE / "findings"
REPORT = ROOT / "reports/ablation-study-v3/PHASE1_FINDINGS.md"
FEATURES = ("H", "B", "F", "HB", "HF", "BF", "HBF")
SELECTED = ("G0__H", "G0__HBF", "G-bi__HBF", "L0__HB", "T0__H", "T0__HF", "T0__HBF")


def distribution(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "per_seed": a.tolist()}


def paired(data: dict, first: str, second: str) -> dict:
    a = np.asarray(data["records"][first]["metrics"]["macro_f1"]["per_seed"])
    b = np.asarray(data["records"][second]["metrics"]["macro_f1"]["per_seed"])
    delta = (a - b) * 100
    return {**distribution(delta), "positive_seeds": int((delta > 0).sum()),
            "negative_seeds": int((delta < 0).sum()), "units": "percentage points"}


def prediction_diagnostics(data: dict) -> dict:
    result = {}
    common_truth = None
    with ExitStack() as stack:
        archives = {}
        for row in data["records"].values():
            owner = row["owner"]
            if owner not in archives:
                path = Path(row["archive"])
                with path.open("rb") as stream:
                    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
                assert checksum == data["archive_sha256"][owner], "Source archive changed"
                archives[owner] = stack.enter_context(zipfile.ZipFile(path))
        for run_id in SELECTED:
            row = data["records"][run_id]
            archive = archives[row["owner"]]
            paths = sorted(n for n in archive.namelist() if n.startswith(row["archive_prefix"] + "/")
                           and n.endswith("/val_report.json"))
            assert len(paths) == 3
            signers, recall, errors = {}, [], []
            for path in paths:
                report = json.loads(archive.read(path))
                predictions = sorted(report["predictions"], key=lambda p: p["sequence_id"])
                truth = [(p["sequence_id"], p["participant_id"], p["y_true"]) for p in predictions]
                if common_truth is None:
                    common_truth = truth
                assert truth == common_truth, "Cross-model ground truth differs"
                for signer in sorted({p["participant_id"] for p in predictions}):
                    subset = [p for p in predictions if p["participant_id"] == signer]
                    values = signers.setdefault(signer, {"samples": len(subset), "accuracy": []})
                    values["accuracy"].append(float(np.mean([p["y_true"] == p["y_pred"] for p in subset])))
                matrix = np.asarray(report["confusion_matrix"], dtype=float)
                recall.append(np.diag(matrix) / matrix.sum(axis=1))
                errors.append(sum(p["y_true"] != p["y_pred"] for p in predictions))
            result[run_id] = {"signers": {str(k): {"samples": v["samples"], "accuracy": distribution(v["accuracy"])}
                                              for k, v in signers.items()},
                              "class_recall": {c: distribution(np.asarray(recall)[:, i])
                                               for i, c in enumerate(report["class_names"])},
                              "wrong_predictions": distribution(errors)}
    return result


def main() -> None:
    data = json.loads((SOURCE / "ALL_MEMBERS_RESULTS.json").read_text(encoding="utf-8"))
    assert len(data["records"]) == 24 and data["completed_fits"] == 72 and not data["test_evaluated"]
    OUTPUT.mkdir(exist_ok=True)
    records, details = data["records"], data["details"]
    diagnostics = prediction_diagnostics(data)
    region_deltas = {p: {a + " minus " + b: paired(data, p + "__" + a, p + "__" + b)
                         for a, b in (("HB", "H"), ("HF", "H"), ("HBF", "H"), ("HBF", "HB"), ("HBF", "HF"))}
                     for p in ("G0", "L0", "T0")}
    architecture_deltas = {p + " minus G0 on HBF": paired(data, p + "__HBF", "G0__HBF")
                           for p in ("G-mean", "G-attn", "G-bi", "L0", "T0")}
    architecture_deltas["T0 minus G-bi on HBF"] = paired(data, "T0__HBF", "G-bi__HBF")
    class_changes = {}
    for first, second in (("T0__HF", "T0__H"), ("T0__HBF", "T0__HF"), ("G-bi__HBF", "G0__HBF")):
        changes = [{"gloss": c, "f1_delta_pp": (details[first]["class_metrics"][c]["f1"]["mean"]
                    - details[second]["class_metrics"][c]["f1"]["mean"]) * 100}
                   for c in details[first]["class_metrics"]]
        class_changes[first + " minus " + second] = sorted(changes, key=lambda r: r["f1_delta_pp"], reverse=True)
    confusions = {}
    for run_id in SELECTED:
        names = list(details[run_id]["class_metrics"])
        matrix = np.asarray(details[run_id]["mean_confusion_matrix"])
        rows = [{"true": names[i], "predicted": names[j], "mean_count": float(matrix[i, j]),
                 "fraction_of_true_class": float(matrix[i, j] / matrix[i].sum())}
                for i in range(len(names)) for j in range(len(names)) if i != j]
        confusions[run_id] = sorted(rows, key=lambda r: r["mean_count"], reverse=True)[:10]
    training = {rid: {"mean_training_minutes": row["metrics"]["training_time_sec"]["mean"] / 60,
                     "mean_train_val_gap_pp": float(np.mean([(e["train_accuracy_at_best"] - e["val_accuracy_at_best"]) * 100
                                               for e in details[rid]["epoch_details"]])),
                     "epochs_run": [e["epochs_run"] for e in details[rid]["epoch_details"]],
                     "best_epochs": [e["best_epoch"] for e in details[rid]["epoch_details"]]}
                for rid, row in records.items()}
    result = {"scope": "24 accepted configurations out of current full42 target", "freeze_id": data["freeze_id"],
              "input_archive_sha256": data["archive_sha256"], "completed_configs": 24, "completed_fits": 72,
              "remaining_configs": 18, "target_configs": 42, "first_round_leader": "T0__HF", "global_winner": None,
              "region_deltas_pp": region_deltas, "architecture_deltas_pp": architecture_deltas,
              "class_changes": class_changes, "confusions": confusions, "prediction_diagnostics": diagnostics,
              "training": training, "training_executed": False, "test_evaluated": False,
              "statistical_note": "Population std of three paired seeds; descriptive, not a confidence interval or significance test."}
    (OUTPUT / "FINDINGS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(3)
    comparisons = (("HB minus H", "Thêm body vào H"), ("HF minus H", "Thêm face vào H"),
                   ("HBF minus H", "Thêm body + face vào H"))
    for i, (key, label) in enumerate(comparisons):
        values = [region_deltas[p][key]["mean"] for p in ("G0", "L0", "T0")]
        axes[0].bar(x + (i - 1) * .25, values, width=.24, label=label, color=("#8a97a3", "#1d8076", "#367fba")[i])
    axes[0].axhline(0, color="#333333", linewidth=.8)
    axes[0].set_xticks(x, ("GRU G0", "LSTM", "Transformer"))
    axes[0].set_ylabel("Thay đổi mean Validation Macro-F1 (điểm %)")
    axes[0].set_title("Tác dụng thêm vùng phụ thuộc temporal model")
    axes[0].legend(fontsize=9)
    names = list(details["T0__HF"]["class_metrics"])
    values = np.asarray([[details[rid]["class_metrics"][c]["f1"]["mean"] * 100 for rid in ("G0__H", "G-bi__HBF", "T0__HF")]
                         for c in names])
    heat = axes[1].imshow(values, aspect="auto", vmin=0, vmax=100, cmap="YlGnBu")
    axes[1].set_xticks(range(3), ("G0 H", "G-bi HBF", "T0 HF"))
    axes[1].set_yticks(range(len(names)), names, fontsize=8)
    axes[1].set_title("F1 từng gloss (%) — mean 3 seeds")
    fig.colorbar(heat, ax=axes[1], fraction=.035, pad=.03)
    fig.tight_layout()
    fig.savefig(OUTPUT / "region_and_gloss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = ["# Phân tích kết quả A/B/C — 24 configs đã nghiệm thu", "",
             "Transformer HF đang tốt nhất trong 24 configs đã chạy. BiGRU tốt nhất nhóm GRU trên cùng HBF. "
             "Tay là nguồn thông tin chính; face cải thiện rõ nhất ở Transformer khi ghép với tay, body có tác dụng nhỏ và phụ thuộc model.", "",
             "Phạm vi hiện tại là42 configs; còn18 configs GRU subsets đang chờ chạy. Chưa có global winner hoặc test metrics.", "",
             "Dữ liệu so sánh: freeze b4fd56f03a57, cùng cohort20 glosses/7.854 clips, train5.558/15 signers, "
             "validation1.155/3 signers, test1.141/3 signers. A7/B3/C14 =24 configs/72 fits, seeds42/43/44. "
             "Tất cả ZIP/source/config/split/inventory/checkpoint hashes và validation predictions đã nghiệm thu; "
             "lần phân tích này đọc lại archives để kiểm lỗi theo signer, không chạy inference hoặc test.", "",
             "## Cấu hình dẫn đầu và tài nguyên đã quan sát", "",
             "| Config | Macro-F1 mean ± std (%) | Top-1 (%) | Top-3 (%) | Parameters | Training phút/seed |", "|---|---:|---:|---:|---:|---:|"]
    for rid in ("T0__HF", "T0__HBF", "T0__HB", "G-bi__HBF", "L0__HB", "G0__H"):
        m = records[rid]["metrics"]
        lines.append(f"| {rid} | {m['macro_f1']['mean']*100:.2f} ± {m['macro_f1']['std']*100:.2f} | {m['top1_accuracy']['mean']*100:.2f} | {m['top3_accuracy']['mean']*100:.2f} | {m['model_parameters']['mean']:.0f} | {training[rid]['mean_training_minutes']:.2f} |")
    lines += ["", "T0 HF thắng mean Macro-F1, còn T0 HBF có Top-3 và Validation Loss tốt hơn. Rule chọn config dùng Macro-F1 nên không đổi leader chỉ vì loss/Top-3. "
              "Parameters là tổng registered parameters, kể cả projectors của vùng bị tắt. Thời gian là training loop đã ghi trên Tesla T4, "
              "không gồm preprocessing và không phải latency/throughput benchmark.", "",
              "## Member B: so architecture trên cùng HBF", "",
              "| Profile | Macro-F1 (%) | Δ so G0 (điểm %) | Seeds tăng/3 |", "|---|---:|---:|---:|"]
    for p in ("G0", "G-mean", "G-attn", "G-bi", "L0", "T0"):
        m = records[p + "__HBF"]["metrics"]["macro_f1"]
        delta = architecture_deltas.get(p + " minus G0 on HBF")
        lines.append(f"| {p} | {m['mean']*100:.2f} ± {m['std']*100:.2f} | {delta['mean']:+.2f} | {delta['positive_seeds']} |" if delta
                     else f"| {p} | {m['mean']*100:.2f} ± {m['std']*100:.2f} | 0 | — |")
    lines += ["", "BiGRU tăng3,85 điểm Macro-F1 so GRU G0 trên HBF, tăng ở cả3 seeds, std chỉ0,32 điểm. "
              "G-attn tăng0,86 điểm nhưng chỉ1/3 seeds tăng; seed44 tăng3,04 điểm, hai seeds còn lại giảm nhẹ. G-mean giảm3,21 điểm và giảm cả3 seeds. "
              "Mean pooling đơn thuần kém hơn mean_max trong thử nghiệm này; attention chưa vượt bidirectional. "
              "BiGRU có529.940 parameters, khoảng2,32 lần G0; cải thiện chưa tách riêng tác dụng bidirectional khỏi tăng capacity. "
              "T0 HBF vẫn hơn G-bi HBF1,39 điểm, tăng cả3 seeds; đây là so sánh cùng feature subset, nhưng số parameters/kiến trúc vẫn khác.", "",
              "## Member A/C: ablation theo vùng", "",
              "H=tay, B=thân trên, F=mặt. Mean Validation Macro-F1 (%):", "",
              "| Model | H | B | F | HB | HF | BF | HBF |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in ("G0", "L0", "T0"):
        lines.append("| " + p + " | " + " | ".join(f"{records[p+'__'+f]['metrics']['macro_f1']['mean']*100:.2f}" for f in FEATURES) + " |")
    lines += ["", "| So với H-only | GRU G0 Δ điểm % | LSTM Δ điểm % | Transformer Δ điểm % |", "|---|---:|---:|---:|"]
    for key, label in comparisons:
        lines.append("| " + label + " | " + " | ".join(f"{region_deltas[p][key]['mean']:+.2f}" for p in ("G0", "L0", "T0")) + " |")
    lines += ["", "A: H và HB có cùng mean Macro-F1 từ metrics làm tròn; HB biến động lớn hơn nên H được xếp trước. "
              "HF/HBF thấp hơn H khoảng0,47/0,51 điểm. Thêm face vào HB làm G0 giảm0,51 điểm ở cả3 seeds.", "",
              "C: LSTM HB tốt nhất nhóm LSTM71,53%, nhưng chỉ hơn H0,73 điểm và không tăng nhất quán cả3 seeds. "
              "Transformer HF tốt nhất76,52%, hơn H2,17 điểm với per-seed gains[1,27;4,40;0,85] điểm: tăng cả3 seeds. "
              "HBF thấp hơn HF0,59 điểm; giảm2/3 seeds, do đó chưa kết luận body luôn có hại. "
              "Kết quả phản bác diễn giải rằng face luôn làm tốt hơn hoặc luôn làm xấu hơn: tác dụng phụ thuộc temporal model.", "",
              "B-only/F-only/BF đều thấp hơn rõ rệt các subsets có tay. Tuy nhiên Top-1 B-only10,19–11,28% và F-only18,76–23,49% "
              "vẫn cao hơn majority-class baseline5,45% trên validation này. Vùng phụ có tín hiệu nhưng chưa đủ phân biệt20 glosses khi dùng riêng.", "",
              "Khi thêm/bỏ vùng, weights được renormalize: HF=87,5%H/12,5%F; HB=77,78%H/22,22%B; HBF=70/20/10. "
              "Do đó ablation đo toàn bộ cấu hình subset+fusion, không phải tác dụng thuần của một vùng giữ nguyên các weights khác. "
              "Chưa có sweep weights nên không chứng minh70/20/10 tối ưu; ZIP không có raw feature arrays để kiểm định chất lượng landmarks trực tiếp.", "",
              "## Lỗi theo gloss và signer", "",
              "Các gloss yếu nhất ở T0 HF:", "", "| Gloss | F1 mean (%) | Recall mean (%) | Validation clips |", "|---|---:|---:|---:|"]
    worst = sorted(details["T0__HF"]["class_metrics"].items(), key=lambda p: p[1]["f1"]["mean"])[:6]
    for c, item in worst:
        recall = diagnostics["T0__HF"]["class_recall"][c]["mean"]
        lines.append(f"| {c} | {item['f1']['mean']*100:.2f} | {recall*100:.2f} | {item['support']} |")
    lines += ["", "Thêm face vào Transformer H cải thiện F1 `pretend`+13,03, `fireman`+8,19, `who`+6,36, `listen`+6,05 điểm; "
              "`mouse` giảm8,50 điểm. Hiệu quả tăng tổng thể không đồng nghĩa mọi gloss đều tăng.", "",
              "Confusions lớn nhất của T0 HF (mean count qua3 seeds, không phải samples độc lập mới):", "",
              "| Nhãn thật → dự đoán | Mean clips/seed | Tỷ lệ trong nhãn thật (%) |", "|---|---:|---:|"]
    for pair in confusions["T0__HF"][:6]:
        lines.append(f"| {pair['true']} → {pair['predicted']} | {pair['mean_count']:.2f} | {pair['fraction_of_true_class']*100:.2f} |")
    lines += ["", "Accuracy theo held-out signer của các ứng viên (mean3 seeds, %):", "",
              "| Signer | Clips | G0 H | G-bi HBF | T0 HF |", "|---|---:|---:|---:|---:|"]
    for signer, item in diagnostics["T0__HF"]["signers"].items():
        lines.append(f"| {signer} | {item['samples']} | " + " | ".join(f"{diagnostics[rid]['signers'][signer]['accuracy']['mean']*100:.2f}" for rid in ("G0__H", "G-bi__HBF", "T0__HF")) + " |")
    lines += ["", "Accuracy theo signer phản ánh cả độ khó signer và phân bố gloss của signer đó; không tách riêng hai nguyên nhân. "
              "Chỉ có3 held-out signers nên chưa suy rộng sang mọi người ký.", "",
              "## Độ ổn định và hướng tiếp theo", "",
              f"T0 HF Macro-F1 seeds42/43/44 là76,17/76,08/77,30%; std0,56 điểm. "
              f"Train–validation accuracy gap ở checkpoint được chọn: T0 HF {training['T0__HF']['mean_train_val_gap_pp']:.2f} điểm, "
              f"G-bi HBF {training['G-bi__HBF']['mean_train_val_gap_pp']:.2f} điểm. "
              "Có khoảng cách generalization đáng theo dõi; training accuracy không đại diện khả năng nhận diện trên signer mới.", "",
              "Tiếp tục18 configs đã bàn giao để kiểm tra G-bi/G-attn/G-mean trên các subsets; chưa thể biết BiGRU HF/HB/H sẽ vượt T0 HF hay không. "
              "Giữ protocol hiện tại khi chạy bổ sung. Sau đủ42 configs, mới chọn winner rồi mở test.", "",
              "Ưu tiên kiểm video/nhãn và landmarks cho awake/wake, mouse, pretend, listen để tìm nguồn nhầm lẫn. "
              "Các confusions hiện tại chưa chứng minh nhãn sai hoặc preprocessing lỗi; một nghiên cứu sửa features/weights cần campaign riêng sau đợt này.", "",
              "Các seed comparisons là mô tả paired gains; population std3 seeds không phải confidence interval. "
              "Cùng1.155 clips được dùng qua nhiều fits, không xem72 fits là72 datasets độc lập; chưa đưa ra kết luận statistical significance.", "",
              "![Ablation và lỗi theo gloss](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/3e125a3e5bd9/findings/region_and_gloss.png)", "",
              "[Dữ liệu phân tích JSON](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/3e125a3e5bd9/findings/FINDINGS.json) · "
              "[Bảng và audit24 configs](../../artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/3e125a3e5bd9/ALL_MEMBERS_RESULTS.json). "
              "JSON audit đầu tiên giữ snapshot scope24/30 lịch sử; report này dùng scope42 hiện tại và chỉ phân tích24 configs đã có."]
    report_text = "\n".join(lines) + "\n"
    for word in ("là", "cohort", "train", "validation", "test", "A", "B", "C", "tăng", "giảm", "cả",
                 "chỉ", "khoảng", "có", "hơn", "thêm", "nhóm", "LSTM", "nhất", "gains", "baseline",
                 "minh", "qua", "mean", "std", "đủ", "tra", "đọc", "audit", "scope", "configs", "weights"):
        for digit in "0123456789":
            report_text = report_text.replace(word + digit, word + " " + digit)
    REPORT.write_text(report_text, encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "data": str(OUTPUT / "FINDINGS.json"),
                      "completed_configs": 24, "completed_fits": 72,
                      "top_confusions_T0_HF": confusions["T0__HF"][:3]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
