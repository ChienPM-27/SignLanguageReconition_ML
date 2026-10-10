"""Module tính toán các độ đo đánh giá mô hình phân loại cử chỉ ASL.

Bao gồm:
- Top-1 Accuracy, Top-3 Accuracy.
- Macro-F1 score.
- Confusion Matrix và vẽ biểu đồ ma trận nhầm lẫn.
- Per-Signer Accuracy Breakdown.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def compute_top_k_accuracy(y_true: np.ndarray, y_probs: np.ndarray, k: int = 1) -> float:
    """Tính Top-K Accuracy từ ma trận xác suất (N, num_classes)."""
    if len(y_true) == 0:
        return 0.0
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = np.any(top_k_preds == y_true[:, np.newaxis], axis=1)
    return float(np.mean(correct))


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 20) -> float:
    """Tính Macro-F1 score trên toàn bộ các lớp."""
    try:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except ImportError:
        # Fallback tính toán thuần numpy nếu chưa cài sklearn
        f1_list = []
        for c in range(num_classes):
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_list.append(f1)
        return float(np.mean(f1_list))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 20) -> np.ndarray:
    """Tính ma trận nhầm lẫn (Confusion Matrix) shape (num_classes, num_classes)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_path: str | Path,
    normalize: bool = True
):
    """Vẽ và lưu ma trận nhầm lẫn ra file ảnh PNG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm_display = cm.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.divide(cm_display, row_sums, out=np.zeros_like(cm_display), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm_display, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=60, ha="left", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Gloss", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Gloss", fontweight="bold")
    ax.set_title("Confusion Matrix (20 Glosses)", fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)


def calculate_per_signer_accuracy(
    test_rows: List[dict],
    y_pred: np.ndarray
) -> Dict[str, dict]:
    """Phân tích độ chính xác theo từng người ký (Signer) trên tập test."""
    signer_data = {}
    for row, pred in zip(test_rows, y_pred):
        s_id = row.get("signer_id", "Unknown")
        label = int(row["label"])
        is_correct = int(pred == label)
        if s_id not in signer_data:
            signer_data[s_id] = {"correct": 0, "total": 0}
        signer_data[s_id]["correct"] += is_correct
        signer_data[s_id]["total"] += 1

    report = {}
    for s_id, d in sorted(signer_data.items()):
        acc = float(d["correct"] / d["total"]) if d["total"] > 0 else 0.0
        report[s_id] = {
            "videos": d["total"],
            "correct": d["correct"],
            "accuracy": round(acc, 4)
        }
    return report
