#!/usr/bin/env python3
"""Script CLI: Vòng huấn luyện và đánh giá an toàn cho mô hình nhận diện 20 gloss ASL-Citizen.

Các chế độ thực thi:
1. --overfit-check: Kiểm tra sanity check trên 20 video (Dropout=0, Aug=Tắt).
2. --baseline-rf: Huấn luyện Random Forest trên đặc trưng thống kê (mean/std/min/max).
3. Huấn luyện GRU chính thức với 3 seeds, Early Stopping, lưu Checkpoint và vẽ Confusion Matrix.
4. --ablation: Chạy thêm mô hình ablation nhỏ gọn (1 lớp, hidden 64).

Sử dụng:
    python scripts/train_model.py [--overfit-check] [--baseline-rf] [--num-seeds 3] [--ablation]
"""
import argparse
import json
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
import torch
from torch.utils.data import DataLoader, Subset

from src.data.asl_citizen import resolve_paths
from src.data.dataset import ASLLandmarkDataset
from src.models.gru import ASLGRUClassifier, create_model
from src.training.evaluate import evaluate_test_set
from src.training.train import train_gru


def run_overfit_sanity_check(train_dataset, device, num_classes=20):
    """Sanity Check: Huấn luyện trên 20 video với Dropout=0, Augmentation=False.

    Mục tiêu: Đảm bảo mô hình có thể overfit hoàn hảo (Loss -> 0, Acc -> 100%)
    để xác nhận pipeline, gradient flow và loss function hoàn toàn không có bug.
    """
    print("\n" + "=" * 60)
    print("CHẾ ĐỘ: OVERFIT SANITY CHECK (20 VIDEOS)")
    print("=" * 60)

    # Chọn 1 mẫu cho mỗi class (tổng 20 mẫu)
    class_indices = {}
    for idx, sample in enumerate(train_dataset.samples):
        c = sample["label"]
        if c not in class_indices:
            class_indices[c] = idx
        if len(class_indices) == num_classes:
            break

    mini_indices = list(class_indices.values())
    mini_dataset = Subset(train_dataset, mini_indices)
    mini_loader = DataLoader(mini_dataset, batch_size=len(mini_dataset), shuffle=False)

    # Model không dropout
    model = ASLGRUClassifier(
        input_size=143, hidden_size=64, num_layers=1, num_classes=num_classes, dropout=0.0
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    print(f"Đang huấn luyện mini-batch {len(mini_indices)} mẫu trong 40 epochs...")
    final_loss, final_acc = 1.0, 0.0

    for ep in range(1, 41):
        model.train()
        for x, y in mini_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            acc = float((preds == y).sum().item() / len(y))
            final_loss, final_acc = loss.item(), acc

        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:2d}/40 -> Loss: {final_loss:.4f} | Accuracy: {final_acc * 100:.1f}%")

    print(f"\nKết quả cuối: Loss = {final_loss:.4f}, Accuracy = {final_acc * 100:.1f}%")
    if final_acc >= 0.95 and final_loss < 0.15:
        print(" SANITY CHECK ĐẠT: Pipeline huấn luyện hoàn toàn lành mạnh!")
    else:
        print(" CẢNH BÁO: Mô hình chưa overfit hoàn hảo. Hãy kiểm tra lại gradient hoặc learning rate.")
    print("=" * 60 + "\n")


def run_random_forest_baseline(train_dataset, test_dataset, num_classes=20):
    """Huấn luyện Random Forest trên đặc trưng thống kê (mean/std/min/max qua thời gian)."""
    print("\n" + "=" * 60)
    print("CHẾ ĐỘ: BASELINE RANDOM FOREST (THỐNG KÊ THỜI GIAN)")
    print("=" * 60)

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        print("Cần cài đặt scikit-learn để chạy baseline Random Forest.")
        return

    def extract_summary_features(dataset):
        X_list, y_list = [], []
        for i in range(len(dataset)):
            x, y = dataset[i]
            arr = x.numpy()  # (T=32, 143)
            # Thống kê: mean, std, min, max (tổng 143 x 4 = 572 dims)
            feat_mean = np.mean(arr, axis=0)
            feat_std = np.std(arr, axis=0)
            feat_min = np.min(arr, axis=0)
            feat_max = np.max(arr, axis=0)
            summary_vec = np.concatenate([feat_mean, feat_std, feat_min, feat_max])
            X_list.append(summary_vec)
            y_list.append(int(y))
        return np.array(X_list), np.array(y_list)

    print("Trích xuất đặc trưng thống kê thời gian...")
    X_train, y_train = extract_summary_features(train_dataset)
    X_test, y_test = extract_summary_features(test_dataset)

    print(f"Shape dữ liệu: Train {X_train.shape}, Test {X_test.shape}")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\nKết quả Baseline Random Forest:")
    print(f" - Test Top-1 Accuracy: {acc * 100:.2f}% (Mức ngẫu nhiên: {100 / num_classes:.1f}%)")
    print(f" - Test Macro-F1:       {f1:.4f}")
    if acc > 0.20:
        print(" BASELINE TỐT: Mô hình học được tín hiệu cử chỉ vượt xa mức ngẫu nhiên!")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình GRU nhận diện ASL-Citizen")
    parser.add_argument("--epochs", type=int, default=60, help="Số epochs (mặc định: 60)")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch (mặc định: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (mặc định: 0.001)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (mặc định: 15)")
    parser.add_argument("--num-seeds", type=int, default=3, help="Số lần chạy với seed khác nhau (mặc định: 3)")
    parser.add_argument("--overfit-check", action="store_true", help="Chạy overfit sanity check trên 20 video")
    parser.add_argument("--baseline-rf", action="store_true", help="Chạy baseline Random Forest")
    parser.add_argument("--ablation", action="store_true", help="Chạy thêm mô hình ablation 1 lớp hidden 64")
    args = parser.parse_args()

    P = resolve_paths(repo_root)
    splits_dir = P.splits
    landmarks_dir = repo_root / "data" / "landmarks"
    checkpoints_dir = repo_root / "models" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị tính toán: {device}")

    # Đọc class names từ classes.csv
    classes_file = splits_dir / "classes.csv"
    if not classes_file.exists():
        print("Lỗi: Không tìm thấy data/splits/classes.csv. Hãy chạy scripts/create_splits.py trước.")
        sys.exit(1)

    import pandas as pd
    classes_df = pd.read_csv(classes_file)
    class_names = classes_df["gloss"].tolist()
    num_classes = len(class_names)
    print(f"Số lượng lớp: {num_classes} classes")

    # Tạo Datasets
    train_dataset = ASLLandmarkDataset(
        split_csv=splits_dir / "train.csv",
        landmarks_dir=landmarks_dir,
        is_train=True,
        augment=True,
    )
    val_dataset = ASLLandmarkDataset(
        split_csv=splits_dir / "val.csv",
        landmarks_dir=landmarks_dir,
        is_train=False,
        augment=False,
    )
    test_dataset = ASLLandmarkDataset(
        split_csv=splits_dir / "test.csv",
        landmarks_dir=landmarks_dir,
        is_train=False,
        augment=False,
    )

    print(f"Số lượng mẫu: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # 1. Chạy Overfit Check nếu yêu cầu
    if args.overfit_check:
        run_overfit_sanity_check(train_dataset, device, num_classes=num_classes)

    # 2. Chạy Baseline RF nếu yêu cầu
    if args.baseline_rf:
        run_random_forest_baseline(train_dataset, test_dataset, num_classes=num_classes)

    # 3. Huấn luyện GRU chính thức với multi-seed
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    seeds = [42, 123, 999][:args.num_seeds]
    results_per_seed = []

    print("\n" + "=" * 60)
    print(f"BẮT ĐẦU HUẤN LUYỆN GRU VỚI {len(seeds)} SEEDS: {seeds}")
    print("=" * 60)

    for s_idx, seed in enumerate(seeds):
        print(f"\n--- [Seed {seed} ({s_idx + 1}/{len(seeds)})] ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        model = create_model(
            input_size=143,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            is_ablation=False,
        )

        ckpt_path = checkpoints_dir / f"best_gru_seed{seed}.pth"
        train_gru(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=1e-4,
            label_smoothing=0.1,
            patience=args.patience,
            device=device,
            checkpoint_path=ckpt_path,
        )

        # Đánh giá trên test set
        report = evaluate_test_set(
            model=model,
            test_loader=test_loader,
            test_csv=splits_dir / "test.csv",
            class_names=class_names,
            device=device,
            plots_dir=P.plots if s_idx == 0 else None,
            report_json_path=P.metrics / f"test_report_seed{seed}.json",
        )
        print(f"Seed {seed} Test Results: Top-1 = {report['top1_accuracy'] * 100:.2f}% | Top-3 = {report['top3_accuracy'] * 100:.2f}% | Macro-F1 = {report['macro_f1']:.4f}")
        results_per_seed.append(report)

    # Tổng kết Multi-seed
    top1_list = [r["top1_accuracy"] for r in results_per_seed]
    top3_list = [r["top3_accuracy"] for r in results_per_seed]
    f1_list = [r["macro_f1"] for r in results_per_seed]

    print("\n" + "=" * 60)
    print(f"KẾT QUẢ TỔNG KẾT GRU TRÊN TEST SET ({len(seeds)} Seeds):")
    print(f" - Top-1 Accuracy: {np.mean(top1_list) * 100:.2f}% ± {np.std(top1_list) * 100:.2f}%")
    print(f" - Top-3 Accuracy: {np.mean(top3_list) * 100:.2f}% ± {np.std(top3_list) * 100:.2f}%")
    print(f" - Macro-F1:       {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print("=" * 60)

    # 4. Chạy Ablation (1 lớp, hidden 64) nếu có cờ --ablation
    if args.ablation:
        print("\n" + "=" * 60)
        print("CHẠY MÔ HÌNH ABLATION (GRU 1 LỚP, HIDDEN 64)")
        print("=" * 60)
        ablation_model = create_model(
            input_size=143, num_classes=num_classes, is_ablation=True
        )
        ablation_ckpt = checkpoints_dir / "ablation_gru_seed42.pth"
        train_gru(
            model=ablation_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=1e-4,
            label_smoothing=0.1,
            patience=args.patience,
            device=device,
            checkpoint_path=ablation_ckpt,
        )
        abl_report = evaluate_test_set(
            model=ablation_model,
            test_loader=test_loader,
            test_csv=splits_dir / "test.csv",
            class_names=class_names,
            device=device,
            plots_dir=None,
            report_json_path=P.metrics / "ablation_test_report.json",
        )
        print(f"Ablation Results: Top-1 = {abl_report['top1_accuracy'] * 100:.2f}% | Top-3 = {abl_report['top3_accuracy'] * 100:.2f}% | Macro-F1 = {abl_report['macro_f1']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
