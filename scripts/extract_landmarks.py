#!/usr/bin/env python3
"""Script CLI: Trích xuất toàn bộ Hand Landmarks từ video của 20 gloss mục tiêu.

Sử dụng:
    python scripts/extract_landmarks.py [--splits train val test] [--limit 0]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

# Thêm repo root vào sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.asl_citizen import resolve_paths
from src.features.landmark_extractor import MediaPipeHandExtractor
from src.features.normalization import process_video_landmarks


def main():
    parser = argparse.ArgumentParser(description="Trích xuất landmarks MediaPipe cho video 20 gloss")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Các split cần xử lý")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số video test thử (0 = xử lý toàn bộ)")
    parser.add_argument("--sequence-length", type=int, default=32, help="Độ dài chuỗi T (mặc định: 32)")
    args = parser.parse_args()

    P = resolve_paths(repo_root)
    videos_dir = P.videos
    landmarks_dir = repo_root / "data" / "landmarks"
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BẮT ĐẦU TRÍCH XUẤT HAND LANDMARKS (MEDIAPIPE HANDS)")
    print(f"Thư mục video: {videos_dir}")
    print(f"Thư mục lưu landmark: {landmarks_dir}")
    print(f"Target Sequence Length T = {args.sequence_length}")
    print("=" * 60)

    # Đọc danh sách video từ split CSVs
    videos_to_process = []
    for s in args.splits:
        sp_file = P.splits / f"{s}.csv"
        if not sp_file.exists():
            print(f"Cảnh báo: Không tìm thấy {sp_file}")
            continue
        with open(sp_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                videos_to_process.append(r)

    print(f"Tổng số video trong các tập split: {len(videos_to_process):,}")
    if args.limit > 0:
        videos_to_process = videos_to_process[:args.limit]
        print(f"Giới hạn xử lý: {len(videos_to_process)} video đầu tiên.")

    extractor = MediaPipeHandExtractor()
    gloss_stats = {}  # gloss -> list of det_rate
    processed_count, failed_count = 0, 0
    t0 = time.time()

    for item in tqdm(videos_to_process, desc="Extracting landmarks"):
        vid_id = item["video_id"]
        gloss = item["gloss"]
        vfile = item["video_file"]
        vpath = videos_dir / vfile
        out_npy = landmarks_dir / f"{vid_id}.npy"

        if out_npy.exists() and out_npy.stat().st_size > 0:
            processed_count += 1
            continue

        if not vpath.exists():
            failed_count += 1
            continue

        raw_coords, hand_masks, stats = extractor.extract_from_video(vpath)
        if not stats.get("ok", False) or len(raw_coords) == 0:
            failed_count += 1
            continue

        det_rate = stats.get("det_rate", 0.0)
        gloss_stats.setdefault(gloss, []).append(det_rate)

        # Chuẩn hóa về T=32 frames x 143 đặc trưng
        features_143, resampled_masks = process_video_landmarks(
            raw_coords, hand_masks, target_len=args.sequence_length
        )

        # Lưu file .npy
        np.save(str(out_npy), features_143.astype(np.float32))
        processed_count += 1

    extractor.close()
    elapsed = time.time() - t0

    print(f"\nHoàn thành trong {elapsed:.1f}s!")
    print(f" - Đã lưu thành công: {processed_count} files .npy")
    print(f" - Lỗi / thiếu file: {failed_count} files")

    # Tổng kết tỷ lệ phát hiện tay theo từng gloss
    if gloss_stats:
        summary_rows = []
        for g, rates in sorted(gloss_stats.items()):
            summary_rows.append({
                "gloss": g,
                "n_videos": len(rates),
                "mean_det_rate": round(float(np.mean(rates)), 4),
                "median_det_rate": round(float(np.median(rates)), 4),
            })
        summary_df = pd.DataFrame(summary_rows)
        out_csv = P.metrics / "detection_rate_by_gloss.csv"
        summary_df.to_csv(out_csv, index=False)
        print(f"\nĐã lưu thống kê phát hiện tay theo gloss tại: {out_csv}")
        print(summary_df.to_string(index=False))

    print("=" * 60)


if __name__ == "__main__":
    main()
