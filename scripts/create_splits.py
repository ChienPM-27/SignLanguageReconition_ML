#!/usr/bin/env python3
"""Script CLI: Tạo và xác minh các file split train/val/test cho 20 gloss ASL-Citizen.

Sử dụng:
    python scripts/create_splits.py
"""
import sys
from pathlib import Path

# Thêm repo root vào sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.preprocessing import DEFAULT_20_GLOSSES, create_splits_from_csv, select_glosses, verify_splits


def main():
    print("=" * 60)
    print("BẮT ĐẦU TẠO SPLITS CHO 20 GLOSS ASL-CITIZEN")
    print("=" * 60)

    meta_path = repo_root / "outputs" / "metrics" / "full_eda_bundle" / "meta_full.csv"
    splits_dir = repo_root / "data" / "splits"

    if not meta_path.exists():
        print(f"Lỗi: Không tìm thấy {meta_path}. Hãy đảm bảo full_eda_bundle đã có sẵn.")
        sys.exit(1)

    print(f"Nạp metadata từ: {meta_path}")

    # Lọc 20 gloss
    glosses = select_glosses(k=20)
    print(f"\n20 Gloss được chọn ({len(glosses)} classes):")
    for i, g in enumerate(glosses):
        print(f"  {i:2d}: {g}")

    # Sinh các file split
    print(f"\nSinh các file split vào: {splits_dir}")
    records = create_splits_from_csv(meta_path, glosses, splits_dir)

    for s, rows in records.items():
        signers = {r['signer_id'] for r in rows}
        print(f" - {s}.csv: {len(rows)} video, {len(signers)} signers")

    # Xác minh tính toàn vẹn và leakage
    print("\nKiểm tra tính toàn vẹn (Verification & Signer Leakage Check)...")
    summary = verify_splits(splits_dir)
    print(" XÁC MINH THÀNH CÔNG: Tuyệt đối không rò rỉ signer!")
    print(f" - Train: {summary['train']['videos']} videos, {summary['train']['signers']} signers")
    print(f" - Val:   {summary['val']['videos']} videos, {summary['val']['signers']} signers")
    print(f" - Test:  {summary['test']['videos']} videos, {summary['test']['signers']} signers")
    print("=" * 60)


if __name__ == "__main__":
    main()
