"""Module đánh giá độc lập mô hình trên tập kiểm thử (Test Set).

Bao gồm:
- Tính Top-1 Accuracy, Top-3 Accuracy, Macro-F1.
- Vẽ và lưu ma trận nhầm lẫn (Confusion Matrix).
- Phân tích độ chính xác theo từng người tham gia (Per-Participant Breakdown).
- Xuất file báo cáo JSON hoàn chỉnh kèm git commit hash.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import (
    calculate_per_signer_accuracy,
    compute_confusion_matrix,
    compute_top_k_accuracy,
    plot_confusion_matrix,
)
from .train import evaluate_epoch


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    test_csv: str | Path,
    class_names: List[str],
    device: torch.device,
    plots_dir: Optional[str | Path] = None,
    report_json_path: Optional[str | Path] = None,
    git_commit_hash: Optional[str] = None,
    preprocess_identity: Optional[str] = None,
    split_identity: Optional[str] = None,
) -> Dict[str, Any]:
    """Đánh giá toàn diện trên tập test với zero hardcoding và đầy đủ provenance audit fields."""
    criterion = nn.CrossEntropyLoss()
    val_loss, top1, macro_f1, y_pred, y_probs, y_true = evaluate_epoch(
        model, test_loader, criterion, device, num_classes=model.num_classes
    )

    top3 = compute_top_k_accuracy(y_true, y_probs, k=3)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=model.num_classes)

    # Đọc metadata tập dữ liệu để phân tích signer / participant (ưu tiên samples đã nạp trong dataset)
    test_rows: List[dict] = []
    if hasattr(test_loader, "dataset") and hasattr(test_loader.dataset, "samples") and test_loader.dataset.samples:
        test_rows = test_loader.dataset.samples
    elif Path(test_csv).is_file():
        with open(test_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                test_rows.append(r)

    per_participant = calculate_per_signer_accuracy(test_rows, y_pred) if test_rows else {}
    sample_membership = [r.get("sequence_id", f"sample_{idx}") for idx, r in enumerate(test_rows)]

    # Vẽ và lưu Confusion Matrix
    if plots_dir:
        pdir = Path(plots_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        cm_path = pdir / "confusion_matrix.png"
        plot_confusion_matrix(cm, class_names, cm_path, normalize=True)

    derived_prep_id = preprocess_identity or getattr(getattr(test_loader, "dataset", None), "preprocess_identity", "unknown")
    derived_split_id = split_identity or Path(test_csv).stem

    report: Dict[str, Any] = {
        "num_test_samples": len(y_true),
        "num_classes": len(class_names),
        "class_names": class_names,
        "top1_accuracy": round(float(top1), 4),
        "top3_accuracy": round(float(top3), 4),
        "macro_f1": round(float(macro_f1), 4),
        "loss": round(float(val_loss), 4),
        "confusion_matrix": cm.tolist(),
        "sample_membership": sample_membership,
        "preprocess_identity": derived_prep_id,
        "split_identity": derived_split_id,
        "per_participant_accuracy": per_participant,
        "git_commit_hash": git_commit_hash or "unknown",
        "model_parameters": sum(p.numel() for p in model.parameters()),
    }
    # Aligned per-sequence predictions support paired comparisons and signer bootstrap.
    # Never label predictions with CSV identifiers if the loader order/length differs.
    from torch.utils.data import SequentialSampler
    if len(test_rows) == len(y_true) and isinstance(getattr(test_loader, "sampler", None), SequentialSampler):
        report["predictions"] = [
            {"sequence_id": str(row.get("sequence_id", f"sample_{idx}")),
             "participant_id": str(row.get("participant_id", "unknown")),
             "y_true": int(y_true[idx]), "y_pred": int(y_pred[idx]),
             "probabilities": y_probs[idx].tolist()}
            for idx, row in enumerate(test_rows)
        ]

    if report_json_path:
        rpath = Path(report_json_path)
        rpath.parent.mkdir(parents=True, exist_ok=True)
        rpath.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report
