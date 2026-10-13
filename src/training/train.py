"""Vòng huấn luyện mô hình GRU cho nhận diện ngôn ngữ ký hiệu ISLR.

Đặc điểm:
- Tích hợp Label Smoothing trong CrossEntropyLoss.
- Tích hợp Weight Decay chống overfit.
- Tùy biến Learning Rate Scheduler (ReduceLROnPlateau, CosineAnnealingLR, StepLR).
  Tương thích PyTorch hiện đại (không dùng verbose=False deprecated).
- Hỗ trợ Mixed Precision Training (AMP) khi bật cấu hình và GPU khả dụng.
- Gradient clipping chống bùng nổ gradient trong RNN.
- Early Stopping với patience cấu hình được.
- Lưu checkpoint best model dựa trên validation loss.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_macro_f1, compute_top_k_accuracy


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """Huấn luyện 1 epoch với hỗ trợ gradient clipping và optional AMP."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    use_amp = scaler is not None and device.type == "cuda"

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    epoch_loss = total_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 20,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Đánh giá 1 epoch trên validation hoặc test set. Trả về cả y_true đích thực từ dataloader."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            probs = torch.softmax(logits, dim=1)

            total_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    total = len(all_targets)
    val_loss = total_loss / total if total > 0 else 0.0

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    top1_acc = compute_top_k_accuracy(y_true, y_probs, k=1)
    macro_f1 = compute_macro_f1(y_true, y_pred, num_classes=num_classes)

    return val_loss, top1_acc, macro_f1, y_pred, y_probs, y_true


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: Optional[Dict[str, Any]],
    epochs: int,
) -> Tuple[Optional[Any], str]:
    """Khởi tạo learning rate scheduler tương thích PyTorch hiện đại."""
    if not scheduler_cfg:
        # Mặc định plateau
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return sched, "metric"

    s_type = str(scheduler_cfg.get("type", "plateau")).lower()
    if s_type == "plateau":
        factor = float(scheduler_cfg.get("factor", 0.5))
        patience = int(scheduler_cfg.get("patience", 5))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience
        )
        return sched, "metric"
    elif s_type in {"cosine", "cosine_annealing"}:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        return sched, "epoch"
    elif s_type == "step":
        step_size = int(scheduler_cfg.get("patience", 10))
        factor = float(scheduler_cfg.get("factor", 0.5))
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=factor)
        return sched, "epoch"
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return sched, "metric"


def train_gru(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 60,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    patience: int = 15,
    device: torch.device = torch.device("cpu"),
    checkpoint_path: Optional[str | Path] = None,
    verbose: bool = True,
    grad_clip: float = 1.0,
    scheduler_cfg: Optional[Dict[str, Any]] = None,
    amp: bool = False,
) -> Dict[str, list]:
    """Huấn luyện mô hình GRU với Early Stopping và lưu Checkpoint."""
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler, sched_mode = create_scheduler(optimizer, scheduler_cfg, epochs)

    scaler = None
    if amp and device.type == "cuda":
        scaler = torch.amp.GradScaler(device_type="cuda")

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    history: Dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "lr": [],
    }

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
            scaler=scaler,
        )
        val_loss, val_acc, val_f1, _, _, _ = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=model.num_classes,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        if sched_mode == "metric" and scheduler is not None:
            scheduler.step(val_loss)
        elif sched_mode == "epoch" and scheduler is not None:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_weights, str(checkpoint_path))
        else:
            patience_counter += 1

        if verbose and (ep % 5 == 0 or ep == 1 or is_best or patience_counter >= patience):
            star = " (*)" if is_best else ""
            print(f"Epoch {ep:2d}/{epochs} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} | lr: {current_lr:.6f}{star}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early Stopping tại epoch {ep} (patience={patience}). Best Val Loss: {best_val_loss:.4f}")
            break

    # Load lại trọng số tốt nhất
    model.load_state_dict(best_weights)
    elapsed = time.time() - t0
    history["training_time_sec"] = elapsed
    history["best_val_loss"] = best_val_loss
    history["best_val_acc"] = best_val_acc
    return history
