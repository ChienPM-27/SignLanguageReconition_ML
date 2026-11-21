"""Accept the exhaustive frozen V3 matrix and analyze its validation predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from shutil import copy2
import tempfile
import zipfile

from analyze_campaign_results import audit_attempt, check, digest, inspect, write_json
from analyze_phase1_findings import paired
from full_matrix_v3 import summarize_full_matrix
import session_v3
import study_v3 as study

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAIRS = (("awake", "wake"), ("mouse", "shhh"), ("pretend", "mouse"))


def diagnostics(attempt: Path) -> dict:
    signers, pairs = {}, {}
    for path in sorted((attempt / "runs").rglob("val_report.json")):
        report = study.read(path)
        predictions = report["predictions"]
        for signer in sorted({str(p["participant_id"]) for p in predictions}):
            subset = [p for p in predictions if str(p["participant_id"]) == signer]
            item = signers.setdefault(signer, {"samples": len(subset), "accuracy": []})
            item["accuracy"].append(float(np.mean([p["y_true"] == p["y_pred"] for p in subset])))
        names = report["class_names"]
        for true_name, predicted_name in PAIRS:
            true, predicted = names.index(true_name), names.index(predicted_name)
            subset = [p for p in predictions if p["y_true"] == true]
            item = pairs.setdefault(true_name + "->" + predicted_name,
                                    {"support": len(subset), "wrong": [], "signers": {}})
            item["wrong"].append(sum(p["y_pred"] == predicted for p in subset))
            for signer in sorted({str(p["participant_id"]) for p in subset}):
                clips = [p for p in subset if str(p["participant_id"]) == signer]
                counts = item["signers"].setdefault(signer, {"support": len(clips), "wrong": []})
                counts["wrong"].append(sum(p["y_pred"] == predicted for p in clips))
    return {"signers": {k: {"samples": v["samples"], "accuracy": study.aggregate(v["accuracy"])}
                        for k, v in signers.items()}, "targeted_confusions": pairs}


def verify_reused_b(archive: Path) -> None:
    old = ROOT / "artifacts/ablation-study-v3/results/member-b/b4fd56f03a57/d2d48b98651a/member_b_b4fd56f03a57_v3_results.zip"
    check(digest(old) == "d2d48b98651a6c1c2deea50d35e5b49b94c1daf6e0340025352cc2e08c177518",
          "Original B archive checksum differs")
    with zipfile.ZipFile(old) as previous, zipfile.ZipFile(archive) as current:
        selected = json.loads(previous.read("queue_status.json"))["selected"]
        new_selected = json.loads(current.read("queue_status.json"))["selected"]
        for run_id, prefix in selected.items():
            check(new_selected[run_id] == prefix, "Original B attempt was replaced")
            for name in previous.namelist():
                if name.startswith(prefix + "/"):
                    check(previous.read(name) == current.read(name), "Original B artifact changed: " + name)


def plot_matrix(data: dict, output: Path) -> None:
    profiles, features = tuple(study.PROFILES), tuple(study.REGIONS)
    values = np.asarray([[data["records"][p + "__" + f]["metrics"]["macro_f1"]["mean"] * 100
                          for f in features] for p in profiles])
    fig, ax = plt.subplots(figsize=(11, 5))
    heat = ax.imshow(values, vmin=0, vmax=100, cmap="YlGnBu")
    ax.set_xticks(range(len(features)), features)
    ax.set_yticks(range(len(profiles)), profiles)
    for i, profile in enumerate(profiles):
        for j, feature in enumerate(features):
            metric = data["records"][profile + "__" + feature]["metrics"]["macro_f1"]
            ax.text(j, i, f"{metric['mean']*100:.2f}\n± {metric['std']*100:.2f}", ha="center", va="center",
                    color="white" if metric["mean"] > .55 else "black", fontsize=9)
    ax.set_title("Validation Macro-F1 (%) — 42 configs, mean ± population std qua 3 seeds")
    fig.colorbar(heat, ax=ax, fraction=.025, pad=.04)
    fig.tight_layout()
    fig.savefig(output / "full42_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def report(data: dict, output: Path) -> None:
    records, details = data["records"], data["details"]
    ranking = study.rank(list(records.values()))
    best = {p: study.rank([r for r in ranking if r["profile"] == p])[0] for p in study.PROFILES}
    winner = records[data["winner"]]
    lines = ["# Kết quả ablation V3 đủ 42 configs", "",
             "Đã nghiệm thu A 7/7, B 21/21, C 14/14 configs, đủ 126 fits với seeds 42/43/44.", "",
             f"Cấu hình được chọn theo Validation Macro-F1 là `{data['winner']}`, "
             f"{winner['metrics']['macro_f1']['mean']*100:.2f}% ± {winner['metrics']['macro_f1']['std']*100:.2f} điểm phần trăm.", "",
             "Freeze `b4fd56f03a57`, cùng 20 glosses và cohort 7.854 sequences. Train gồm 5.558 clips/15 signers, "
             "validation 1.155 clips/3 signers, test 1.141 clips/3 signers. Test evaluation còn chờ.", "",
             "## Xếp hạng validation", "",
             "| Hạng | Config | Macro-F1 mean ± std (%) | Top-1 (%) | Top-3 (%) | Parameters |", "|---:|---|---:|---:|---:|---:|"]
    for i, row in enumerate(ranking, 1):
        m = row["metrics"]
        lines.append(f"| {i} | {row['run_id']} | {m['macro_f1']['mean']*100:.2f} ± {m['macro_f1']['std']*100:.2f} "
                     f"| {m['top1_accuracy']['mean']*100:.2f} | {m['top3_accuracy']['mean']*100:.2f} | {m['model_parameters']['mean']:.0f} |")
    lines += ["", "## Ma trận đầy đủ", "", "H = tay, B = thân trên, F = mặt. Mean Validation Macro-F1 (%).", "",
              "| Profile | " + " | ".join(study.REGIONS) + " |", "|---|" + "---:|" * 7]
    for profile in study.PROFILES:
        lines.append("| " + profile + " | " + " | ".join(
            f"{records[profile+'__'+feature]['metrics']['macro_f1']['mean']*100:.2f}" for feature in study.REGIONS) + " |")
    lines += ["", "![Ma trận full42](full42_matrix.png)", "", "## Config tốt nhất của từng profile", "",
              "| Profile | Config | Macro-F1 mean ± std (%) | Seeds 42 / 43 / 44 (%) |", "|---|---|---:|---|"]
    for profile, row in best.items():
        metric = row["metrics"]["macro_f1"]
        lines.append(f"| {profile} | {row['run_id']} | {metric['mean']*100:.2f} ± {metric['std']*100:.2f} | "
                     + " / ".join(f"{v*100:.2f}" for v in metric["per_seed"]) + " |")
    lines += ["", "## So architecture trên cùng vùng", "",
              "Δ mean Macro-F1 so với G0, đơn vị điểm phần trăm. Ngoặc là số seeds tăng trên 3 seeds.", "",
              "| Profile | H | HB | HF | HBF |", "|---|---:|---:|---:|---:|"]
    for profile in study.PROFILES:
        if profile == "G0":
            continue
        values = []
        for feature in ("H", "HB", "HF", "HBF"):
            delta = data["architecture_deltas_pp"][profile + "__" + feature + " minus G0__" + feature]
            values.append(f"{delta['mean']:+.2f} ({delta['positive_seeds']}/3)")
        lines.append("| " + profile + " | " + " | ".join(values) + " |")
    lines += ["", "## Thêm vùng vào H", "",
              "Δ mean Macro-F1, điểm phần trăm, ngoặc là số seeds tăng trên 3 seeds.", "",
              "| Profile | HB − H | HF − H | HBF − H |", "|---|---:|---:|---:|"]
    for profile in study.PROFILES:
        values = []
        for feature in ("HB", "HF", "HBF"):
            delta = data["region_deltas_pp"][profile + "__" + feature + " minus " + profile + "__H"]
            values.append(f"{delta['mean']:+.2f} ({delta['positive_seeds']}/3)")
        lines.append("| " + profile + " | " + " | ".join(values) + " |")
    lines += ["", "## Ba hướng nhầm đã theo dõi", "",
              "So sánh config tốt nhất của từng profile, đầu vào có thể khác nhau. Tỷ lệ là mean qua ba seeds, tính trên nhãn thật.", "",
              "| Config | awake → wake (%) | mouse → shhh (%) | pretend → mouse (%) |", "|---|---:|---:|---:|"]
    for row in best.values():
        pairs = data["prediction_diagnostics"][row["run_id"]]["targeted_confusions"]
        values = [f"{np.mean(pairs[a+'->'+b]['wrong'])/pairs[a+'->'+b]['support']*100:.2f}" for a, b in PAIRS]
        lines.append("| " + row["run_id"] + " | " + " | ".join(values) + " |")
    lines += ["", "Số lỗi pretend → mouse theo signer, theo thứ tự seeds 42/43/44.", "",
              "| Config | Signer 28656 | Signer 32319 | Signer 36257 |", "|---|---|---|---|"]
    for row in best.values():
        counts = data["prediction_diagnostics"][row["run_id"]]["targeted_confusions"]["pretend->mouse"]["signers"]
        lines.append("| " + row["run_id"] + " | " + " | ".join(
            "/".join(str(v) for v in counts[signer]["wrong"]) + f" trên {counts[signer]['support']}"
            for signer in ("28656", "32319", "36257")) + " |")
    lines += ["", "F1 từng gloss và recall của config được chọn.", "",
              "| Gloss | Clips | Mean F1 (%) | Mean recall (%) |", "|---|---:|---:|---:|"]
    class_metrics = details[data["winner"]]["class_metrics"]
    matrix = np.asarray(details[data["winner"]]["mean_confusion_matrix"])
    recalls = dict(zip(class_metrics, np.diag(matrix) / matrix.sum(axis=1)))
    for name, item in sorted(class_metrics.items(), key=lambda p: p[1]["f1"]["mean"]):
        lines.append(f"| {name} | {item['support']} | {item['f1']['mean']*100:.2f} | {recalls[name]*100:.2f} |")
    lines += ["", "## Nghiệm thu và giới hạn", "",
              "Cả 42 selected attempts pass canonical `study_v3.collect`, exact inventories, frozen source/config/split/cache identity, "
              "checkpoint hashes, successful stage logs, loss-selected histories và multi-seed summaries. "
              "Top-1/Top-3/Macro-F1 và confusion matrices được tính lại từ 126 bộ validation predictions.", "",
              "Ba configs HBF B cũ được đối chiếu từng artifact với ZIP B vòng đầu và giữ nguyên. "
              "ZIP B mới còn lưu lịch sử attempt split thất bại trước hotfix, queue hiện tại có failed rỗng.", "",
              "Rule xếp hạng dùng metrics làm tròn trong reports theo implementation đã freeze. Metrics chính xác từ predictions lưu riêng trong JSON. "
              "Mỗi seed dùng cùng validation cohort. Population std của ba seeds là thống kê mô tả.", "",
              "Các so sánh kiến trúc có số parameters khác nhau. Ablation vùng đồng thời thay đổi thành phần và trọng số fusion sau chuẩn hóa. "
              "HF dùng 87,5% H / 12,5% F, HB dùng 77,78% H / 22,22% B, HBF dùng 70/20/10.", "",
              "Nguyên nhân lỗi theo gloss/signer cần kiểm clip hoặc landmarks thực tế. "
              "Bước evaluation cuối dùng ba checkpoints đã chọn trên test cohort 1.141 clips.", "",
              "[Metrics, hashes và audit JSON](ALL_MEMBERS_FULL42_RESULTS.json) · [Lựa chọn config và checkpoints](WINNER_SELECTION.json)"]
    (output / "ALL_MEMBERS_FULL42_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for member in ("a", "b", "c"):
        parser.add_argument("--member-" + member, type=Path, required=True)
    args = parser.parse_args()
    sources = {m: getattr(args, "member_" + m.lower()).resolve() for m in ("A", "B", "C")}
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
        check(digest(target) == hashes[member], "Stored archive differs")
        archives[member], audits[member] = target, inspect(target, member, full_matrix=True)
        if member == "B":
            verify_reused_b(target)
            write_json(target.parent / "STORAGE.json", {"operation": "copy", "source": str(source), "destination": str(target),
                       "sha256": hashes[member], "bytes": target.stat().st_size, "stored_at_asia_saigon": timestamp,
                       "hash_verified": True, "source_removed": False})
            write_json(target.parent / "ZIP_CHECK.json", {k: v for k, v in audits[member].items() if k != "state"}
                       | {"sha256": hashes[member], "bytes": target.stat().st_size})
    temp_root = ROOT / "tmp/result_reviews"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v3-full42-", dir=temp_root) as temporary:
        workspace = Path(temporary).resolve()
        check(workspace.is_relative_to(temp_root.resolve()) and workspace != temp_root.resolve(), "Unsafe review workspace")
        attempts, locations = [], {}
        for member, archive in archives.items():
            extracted = workspace / member
            session_v3.extract(archive, extracted)
            state = audits[member]["state"]
            for run_id, prefix in state["selected"].items():
                attempts.append(session_v3.relative(extracted, prefix))
                locations[run_id] = (member, prefix, state)
        data = summarize_full_matrix(attempts, workspace / "ALL_MEMBERS_FULL42_RESULTS.md")
        check(data["status"] == "COMPLETED" and data["completed_cells"] == 42 and data["completed_fits"] == 126,
              "Full matrix is incomplete")
        details, prediction_data = {}, {}
        for run_id, row in data["records"].items():
            member, prefix, state = locations[run_id]
            check(row["owner"] == member and row["provenance"] == state["provenance"], "Selected owner/provenance differs")
            attempt = Path(row.pop("path"))
            details[run_id], prediction_data[run_id] = audit_attempt(attempt, row), diagnostics(attempt)
            row.update({"archive": str(archives[member]), "archive_prefix": prefix})
            print("Accepted", run_id, row["metrics"]["macro_f1"], flush=True)
        check(len({d["ground_truth_sha256"] for d in details.values()}) == 1, "Cross-model ground truth differs")
    data.update({"archive_sha256": hashes, "reviewed_at_asia_saigon": timestamp, "details": details,
                 "ranking": [r["run_id"] for r in study.rank(list(data["records"].values()))],
                 "prediction_diagnostics": prediction_data, "test_evaluated": False,
                 "training_executed_during_review": False, "inference_executed_during_review": False})
    data["region_deltas_pp"] = {key: paired(data, *key.split(" minus ")) for key in data["paired_region_deltas"]}
    data["architecture_deltas_pp"] = {p + "__" + f + " minus G0__" + f: paired(data, p + "__" + f, "G0__" + f)
                                      for p in study.PROFILES if p != "G0" for f in study.REGIONS}
    write_json(output / "ALL_MEMBERS_FULL42_RESULTS.json", data)
    winner = data["records"][data["winner"]]
    selection = {"scope": "v3-full42", "freeze_id": data["freeze_id"], "winner": data["winner"],
                 "selection_rule": "Mean Validation Macro-F1, lower population std, fewer parameters, run ID",
                 "provenance": winner["provenance"], "archive_sha256": hashes[winner["owner"]],
                 "archive": winner["archive"], "archive_prefix": winner["archive_prefix"],
                 "checkpoints": {"seed_" + str(e["seed"]): e["checkpoint_sha256"] for e in details[data["winner"]]["epoch_details"]},
                 "test_evaluated": False}
    write_json(output / "WINNER_SELECTION.json", selection)
    write_json(archives["B"].parent / "REVIEW.json", {"status": "ACCEPTED", "scope": "v3-full42", "member": "B",
               "configs": 21, "fits": 63, "archive_sha256": hashes["B"], "reviewed_at_asia_saigon": timestamp,
               "attempts": {k: r for k, r in data["records"].items() if r["owner"] == "B"},
               "original_three_configs_unchanged": True, "aggregate_report": str(output / "ALL_MEMBERS_FULL42_RESULTS.md")})
    plot_matrix(data, output)
    report(data, output)
    for member, source in sources.items():
        check(digest(source) == digest(archives[member]) == hashes[member], "Archive changed during review")
    print(json.dumps({"status": data["status"], "configs": 42, "fits": 126, "winner": data["winner"],
                      "report": str(output / "ALL_MEMBERS_FULL42_RESULTS.md")}, indent=2))


if __name__ == "__main__":
    main()
