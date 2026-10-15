"""Evaluation script on sample videos for generalization diagnostic (Phase A).

Evaluates 20 sample videos against:
- Variant A: Default video pipeline (with dominance swap)
- Variant B: Video with horizontal mirror (x -> 1 - x)

Outputs top-1 and top-3 accuracies and writes structured evaluation JSON.
Based on outputs/diag/diag_samples.py without modifying the diagnostic file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add repo root to sys.path if not present
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.landmark_extractor import MediaPipeHandExtractor
from src.inference.predictor import ISLRPredictor
from src.utils.logging import get_logger

logger = get_logger("islr.eval_samples")


def extract_video_landmarks(
    video_path: Path,
    extractor: MediaPipeHandExtractor,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, float]]:
    """Extract per-frame hand coordinates and presence masks from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    coords_list: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        c, m = extractor.extract_from_frame(frame)
        coords_list.append(c)
        masks_list.append(m)

    cap.release()
    return (
        np.array(coords_list, dtype=np.float32),
        np.array(masks_list, dtype=bool),
        (width, height, fps),
    )


def apply_dominance_swap(
    coords: np.ndarray,
    masks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Swap hand slots if hand 1 demonstrates higher motion displacement than hand 0."""
    coords_swapped = coords.copy()
    masks_swapped = masks.copy()
    num_frames = len(coords)
    displacement = [0.0, 0.0]

    for s in range(2):
        for t in range(1, num_frames):
            if masks_swapped[t, s] and masks_swapped[t - 1, s]:
                wrist_diff = coords_swapped[t, s, 0] - coords_swapped[t - 1, s, 0]
                displacement[s] += float(np.linalg.norm(wrist_diff))

    if displacement[1] > displacement[0]:
        coords_swapped[:, [0, 1]] = coords_swapped[:, [1, 0]]
        masks_swapped[:, [0, 1]] = masks_swapped[:, [1, 0]]

    return coords_swapped, masks_swapped


def apply_mirror_x(
    coords: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Horizontally mirror landmarks along x-axis (x -> 1 - x) and re-sort slots by wrist x."""
    mirrored = coords.copy()
    # Mirror x coordinates for valid landmark points
    mirrored[..., 0] = np.where(masks[:, :, None], 1.0 - coords[..., 0], 0.0)

    # Re-sort hands so left hand in frame (smaller x) is slot 0 when 2 hands detected
    for t in range(len(mirrored)):
        if masks[t, 0] and masks[t, 1] and mirrored[t, 0, 0, 0] > mirrored[t, 1, 0, 0]:
            mirrored[t, [0, 1]] = mirrored[t, [1, 0]]

    return mirrored


def format_top_k(top_k: List[Tuple[str, float]]) -> str:
    """Format the top-1 prediction and confidence for terminal display."""
    if not top_k:
        return "N/A"
    gloss, conf = top_k[0]
    return f"{gloss:<8s} ({conf * 100:4.1f}%)"


def evaluate_samples(
    model_path: Optional[str | Path],
    glosses_csv: Optional[str | Path],
    videos_dir: Path,
    out_json_path: Path,
) -> Dict[str, Any]:
    """Evaluate sample videos on Variant A (default) and Variant B (mirrored)."""
    predictor = ISLRPredictor(model_path=model_path, glosses_csv=glosses_csv)
    extractor = MediaPipeHandExtractor()

    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No .mp4 video files found in {videos_dir}")

    print("\n========================================================")
    print("ISLR Sample Video Evaluation (Phase A Diagnostic)")
    print(f"Model: {predictor.model_path}")
    print(f"Glosses CSV: {predictor.glosses_csv}")
    print(f"Number of videos: {len(video_files)}")
    print("========================================================\n")

    video_results: List[Dict[str, Any]] = []
    correct_counts = {
        "A": {"top1": 0, "top3": 0},
        "B": {"top1": 0, "top3": 0},
    }

    for idx, v_path in enumerate(video_files, start=1):
        t0 = time.time()
        ground_truth = v_path.stem.lower()

        coords, masks, (w, h, fps) = extract_video_landmarks(v_path, extractor)
        det_rate = float(masks.any(axis=1).mean()) if len(masks) > 0 else 0.0
        two_hand_rate = float(masks.all(axis=1).mean()) if len(masks) > 0 else 0.0

        # Variant A: Standard video pipeline (dominance swap)
        c_a, m_a = apply_dominance_swap(coords, masks)
        pred_a = predictor.predict_raw_coordinates(c_a, m_a, top_k=3).top_k

        # Variant B: Mirrored video pipeline (mirror then dominance swap)
        c_mirrored = apply_mirror_x(coords, masks)
        c_b, m_b = apply_dominance_swap(c_mirrored, masks)
        pred_b = predictor.predict_raw_coordinates(c_b, m_b, top_k=3).top_k

        # Check correctness
        a_top1 = len(pred_a) > 0 and pred_a[0][0].lower() == ground_truth
        a_top3 = any(g.lower() == ground_truth for g, _ in pred_a[:3])
        b_top1 = len(pred_b) > 0 and pred_b[0][0].lower() == ground_truth
        b_top3 = any(g.lower() == ground_truth for g, _ in pred_b[:3])

        if a_top1:
            correct_counts["A"]["top1"] += 1
        if a_top3:
            correct_counts["A"]["top3"] += 1
        if b_top1:
            correct_counts["B"]["top1"] += 1
        if b_top3:
            correct_counts["B"]["top3"] += 1

        elapsed = time.time() - t0
        print(
            f"[{idx:02d}/{len(video_files):02d}] {ground_truth:<8s} ({w}x{h}@{fps:.0f}fps, n={len(coords):3d}, "
            f"det={det_rate:.2f}) | A: {format_top_k(pred_a)} [{'OK' if a_top1 else 'MISS'}] | "
            f"B: {format_top_k(pred_b)} [{'OK' if b_top1 else 'MISS'}] [{elapsed:.1f}s]"
        )

        video_results.append({
            "video_file": v_path.name,
            "ground_truth": ground_truth,
            "num_frames": len(coords),
            "detection_rate": det_rate,
            "two_hand_rate": two_hand_rate,
            "variant_A": {
                "top_k": pred_a,
                "top1_correct": bool(a_top1),
                "top3_correct": bool(a_top3),
            },
            "variant_B": {
                "top_k": pred_b,
                "top1_correct": bool(b_top1),
                "top3_correct": bool(b_top3),
            },
        })

    total = len(video_files)
    summary = {
        "variant_A": {
            "name": "A video (default)",
            "top1_acc": correct_counts["A"]["top1"] / total,
            "top3_acc": correct_counts["A"]["top3"] / total,
            "top1_correct": correct_counts["A"]["top1"],
            "top3_correct": correct_counts["A"]["top3"],
            "total": total,
        },
        "variant_B": {
            "name": "B video + mirror x",
            "top1_acc": correct_counts["B"]["top1"] / total,
            "top3_acc": correct_counts["B"]["top3"] / total,
            "top1_correct": correct_counts["B"]["top1"],
            "top3_correct": correct_counts["B"]["top3"],
            "total": total,
        },
    }

    print("\n" + "=" * 56)
    print("ACCURACY SUMMARY TREN 20 VIDEO MAU")
    print("=" * 56)
    print(
        f"A video (default)       top1: {summary['variant_A']['top1_acc'] * 100:5.1f}% ({summary['variant_A']['top1_correct']}/{total})  "
        f"top3: {summary['variant_A']['top3_acc'] * 100:5.1f}% ({summary['variant_A']['top3_correct']}/{total})"
    )
    print(
        f"B video + mirror x      top1: {summary['variant_B']['top1_acc'] * 100:5.1f}% ({summary['variant_B']['top1_correct']}/{total})  "
        f"top3: {summary['variant_B']['top3_acc'] * 100:5.1f}% ({summary['variant_B']['top3_correct']}/{total})"
    )
    print("=" * 56 + "\n")

    report_data = {
        "model_path": str(predictor.model_path),
        "glosses_csv": str(predictor.glosses_csv),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos": total,
        "summary": summary,
        "details": video_results,
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, mode="w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"Results saved to: {out_json_path}")
    return report_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISLR Sample Video Evaluation (Variant A default vs Variant B mirror x)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to best_model.pt checkpoint (defaults to ISLRPredictor default candidate)",
    )
    parser.add_argument(
        "--glosses",
        type=str,
        default=None,
        help="Path to glosses.csv mapping (defaults to ISLRPredictor default candidate)",
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="data/sample_videos",
        help="Directory containing sample .mp4 videos (default: data/sample_videos)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to save output JSON evaluation report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model) if args.model else None
    glosses_csv = Path(args.glosses) if args.glosses else None
    videos_dir = Path(args.videos_dir)
    out_json = Path(args.out)

    evaluate_samples(
        model_path=model_path,
        glosses_csv=glosses_csv,
        videos_dir=videos_dir,
        out_json_path=out_json,
    )


if __name__ == "__main__":
    main()
