#!/usr/bin/env python3
"""Script CLI: Tải toàn bộ video cho 20 gloss mục tiêu qua HTTP Range từ ASL Citizen.

Sử dụng:
    python scripts/download_dataset.py [--workers 8] [--splits train val test]
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

from src.data.asl_citizen import DEFAULT_CFG, download_videos, resolve_paths


def main():
    parser = argparse.ArgumentParser(description="Tải video ASL-Citizen theo split")
    parser.add_argument("--workers", type=int, default=8, help="Số luồng tải song song (mặc định: 8)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Các split cần tải")
    parser.add_argument("--out-dir", type=str, default=None, help="Thư mục lưu video")
    args = parser.parse_args()

    P = resolve_paths(repo_root)
    out_dir = Path(args.out_dir) if args.out_dir else P.videos
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BẮT ĐẦU TẢI DỮ LIỆU VIDEO ASL-CITIZEN (HTTP RANGE)")
    print(f"Thư mục lưu: {out_dir}")
    print(f"Các split: {args.splits}")
    print("=" * 60)

    # 1. Đọc danh sách video cần tải từ data/splits/*.csv
    target_videos = set()
    splits_dir = P.splits
    for s in args.splits:
        sp_file = splits_dir / f"{s}.csv"
        if not sp_file.exists():
            print(f"Cảnh báo: Không tìm thấy {sp_file}")
            continue
        with open(sp_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                target_videos.add(r["video_file"])

    print(f"Tổng số video mục tiêu trong splits: {len(target_videos):,}")
    if not target_videos:
        print("Không có video nào cần tải. Hãy chạy scripts/create_splits.py trước.")
        sys.exit(1)

    # 2. Lấy metadata zip_name và file_size từ meta_full.csv
    meta_path = P.metrics / "full_eda_bundle" / "meta_full.csv"
    if not meta_path.exists():
        meta_path = repo_root / "outputs" / "metrics" / "full_eda_bundle" / "meta_full.csv"

    pool_rows = []
    if meta_path.exists():
        print(f"Nạp thông tin kích thước và chỉ mục từ: {meta_path}")
        with open(meta_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                vname = Path(r["video_file"]).name
                if vname in target_videos:
                    pool_rows.append({
                        "video_file": vname,
                        "zip_name": r["zip_name"],
                        "file_size": int(r["file_size"])
                    })
    else:
        print("Không có meta_full.csv local, nạp từ máy chủ ASL Citizen...")
        from src.data.asl_citizen import open_zip, index_zip
        zf = open_zip(DEFAULT_CFG["zip_src"])
        idx = index_zip(zf)
        idx_map = {Path(r["name"]).name: r for r in idx.to_dict("records")}
        for vname in target_videos:
            if vname in idx_map:
                pool_rows.append({
                    "video_file": vname,
                    "zip_name": idx_map[vname]["name"],
                    "file_size": int(idx_map[vname]["file_size"])
                })

    import pandas as pd
    pool_df = pd.DataFrame(pool_rows).drop_duplicates(subset=["video_file"])
    total_bytes = pool_df["file_size"].sum()
    print(f"Khớp {len(pool_df):,} video | Dung lượng nén ước tính: {total_bytes / 1e6:.2f} MB")

    # 3. Tải đa luồng
    log_path = P.metrics / "20gloss_download_log.csv"
    t0 = time.time()
    log_df = download_videos(
        src=DEFAULT_CFG["zip_src"],
        pool_df=pool_df,
        out_dir=out_dir,
        workers=args.workers,
        log_path=log_path
    )
    elapsed = time.time() - t0
    print(f"\nHoàn thành trong {elapsed:.1f}s!")

    # 4. Kiểm tra toàn vẹn
    downloaded_ok = sum(1 for v in target_videos if (out_dir / v).exists() and (out_dir / v).stat().st_size > 0)
    print(f"Kết quả kiểm tra file: {downloaded_ok}/{len(target_videos)} video sẵn sàng trên đĩa.")
    if downloaded_ok == len(target_videos):
        print(" TẤT CẢ 100% VIDEO ĐÃ CÓ SẴN!")
    else:
        print(f" Lưu ý: Còn thiếu {len(target_videos) - downloaded_ok} video. Hãy chạy lại để tải bổ sung.")


if __name__ == "__main__":
    main()
