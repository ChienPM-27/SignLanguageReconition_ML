"""Tiền xử lý và tạo phân chia tập dữ liệu (Splits) cho ASL-Citizen.

Chức năng chính:
- select_glosses: Lọc và chọn 20 gloss chất lượng cao nhất theo thống kê EDA.
- create_splits: Sinh các file split train/val/test/classes và ánh xạ nhãn.
- verify_splits: Kiểm tra tính toàn vẹn và đảm bảo tuyệt đối không rò rỉ người ký (Signer Leakage).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set


# 20 Gloss mặc định được chọn lọc kỹ từ Phương án A (n_variants == 1, sạch, cử chỉ rõ ràng, loại trừ OK)
DEFAULT_20_GLOSSES = [
    "BACON", "BAR", "BELL", "BOX", "BOY",
    "CONGRATULATIONS", "COUGH", "DEEP", "EMAIL", "LISTEN",
    "MANY", "MATH", "OOPS", "OSTRICH", "PAGE",
    "PERFECT", "TOILET", "TRUE", "WEDDING", "PASSOUT"
]


def select_glosses(
    meta_df=None,
    gloss_table_df=None,
    k: int = 20,
    override: Optional[List[str]] = None,
) -> List[str]:
    """Chọn K gloss mục tiêu thỏa mãn tiêu chí chất lượng cao nhất."""
    if override:
        return sorted(override)
    return sorted(DEFAULT_20_GLOSSES[:k])


def create_splits_from_csv(
    meta_csv_path: str | Path,
    selected_glosses: List[str],
    splits_dir: str | Path,
) -> Dict[str, list]:
    """Sinh các file split từ file CSV metadata bằng thư viện chuẩn (không bắt buộc pandas)."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    sorted_glosses = sorted(selected_glosses)
    label_map = {g: i for i, g in enumerate(sorted_glosses)}
    gloss_set = set(sorted_glosses)

    records = {"train": [], "val": [], "test": []}

    with open(meta_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = row.get("gloss")
            if g in gloss_set:
                sp = row.get("split")
                if sp in records:
                    vfile = Path(row.get("video_file", "")).name
                    records[sp].append({
                        "video_id": row.get("video_id", ""),
                        "gloss": g,
                        "split": sp,
                        "signer_id": row.get("signer_id", ""),
                        "video_file": vfile,
                        "label": label_map[g]
                    })

    # Ghi train.csv, val.csv, test.csv
    cols = ["video_id", "gloss", "split", "signer_id", "video_file", "label"]
    for sp in ["train", "val", "test"]:
        part = sorted(records[sp], key=lambda x: x["video_id"])
        out_path = splits_dir / f"{sp}.csv"
        with open(out_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(part)

    # Ghi classes.csv
    with open(splits_dir / "classes.csv", mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "gloss"])
        writer.writeheader()
        for g, idx in label_map.items():
            writer.writerow({"class_id": idx, "gloss": g})

    # Ghi label_map.json
    (splits_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    return records


def verify_splits(splits_dir: str | Path, videos_dir: Optional[str | Path] = None) -> Dict[str, dict]:
    """Kiểm tra rò rỉ người ký (Signer Leakage) và tính toàn vẹn của split."""
    splits_dir = Path(splits_dir)

    def read_split(name):
        rows = []
        with open(splits_dir / f"{name}.csv", mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows

    train_rows = read_split("train")
    val_rows = read_split("val")
    test_rows = read_split("test")

    train_signers = {r["signer_id"] for r in train_rows}
    val_signers = {r["signer_id"] for r in val_rows}
    test_signers = {r["signer_id"] for r in test_rows}

    # 1. Assert signer disjoint
    assert train_signers.isdisjoint(val_signers), (
        f"RÒ RỈ SIGNER: Train và Val trùng {train_signers.intersection(val_signers)}"
    )
    assert train_signers.isdisjoint(test_signers), (
        f"RÒ RỈ SIGNER: Train và Test trùng {train_signers.intersection(test_signers)}"
    )
    assert val_signers.isdisjoint(test_signers), (
        f"RÒ RỈ SIGNER: Val và Test trùng {val_signers.intersection(test_signers)}"
    )

    # 2. Assert all classes present in all splits
    train_glosses = {r["gloss"] for r in train_rows}
    val_glosses = {r["gloss"] for r in val_rows}
    test_glosses = {r["gloss"] for r in test_rows}
    assert train_glosses == val_glosses == test_glosses, (
        f"THIẾU CLASS: train={len(train_glosses)}, val={len(val_glosses)}, test={len(test_glosses)}"
    )

    # 3. Kiểm tra file video nếu có videos_dir
    missing_files = []
    if videos_dir is not None:
        vdir = Path(videos_dir)
        all_vids = [r["video_file"] for r in train_rows + val_rows + test_rows]
        missing_files = [f for f in all_vids if not (vdir / f).exists()]

    summary = {
        "train": {
            "videos": len(train_rows),
            "signers": len(train_signers),
            "glosses": len(train_glosses),
        },
        "val": {
            "videos": len(val_rows),
            "signers": len(val_signers),
            "glosses": len(val_glosses),
        },
        "test": {
            "videos": len(test_rows),
            "signers": len(test_signers),
            "glosses": len(test_glosses),
        },
        "missing_videos": len(missing_files),
    }
    return summary
