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
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.features.schema import PREPROCESS_SCHEMA_VERSION
from src.utils.checkpoint import validate_checkpoint_metadata, validate_model_architecture
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


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    best_val_loss: float,
    best_val_acc: float,
    patience_counter: int,
    early_stopping_best: float,
    history: Dict[str, list],
    resolved_config: Optional[Dict[str, Any]] = None,
    schema_version: int = PREPROCESS_SCHEMA_VERSION,
    preprocess_identity: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    label_map: Optional[Dict[str, int]] = None,
    seed: Optional[int] = None,
) -> Path:
    """Lưu stateful checkpoint phục vụ khả năng resume hoàn chỉnh."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
        "scaler_state_dict": copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "patience_counter": patience_counter,
        "early_stopping_best": early_stopping_best,
        "history": copy.deepcopy(history),
        "schema_version": schema_version,
        "preprocess_identity": preprocess_identity,
        "resolved_config": resolved_config,
        "class_names": class_names,
        "label_map": label_map,
        "seed": seed,
    }
    torch.save(payload, str(path))
    return path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Tải checkpoint (hỗ trợ cả stateful dict và raw state_dict)."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    target_device = device if device is not None else torch.device("cpu")
    ckpt = torch.load(str(path), map_location=target_device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        if model is not None:
            model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        return ckpt
    elif isinstance(ckpt, dict):
        if model is not None:
            model.load_state_dict(ckpt)
        return {
            "epoch": 0,
            "model_state_dict": ckpt,
            "best_val_loss": float("inf"),
            "best_val_acc": 0.0,
            "patience_counter": 0,
            "early_stopping_best": float("inf"),
            "history": {},
        }
    else:
        raise ValueError(f"Unrecognized checkpoint format at {path}")


def train_gru(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 60,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    patience: int = 15,
    early_stopping_min_delta: float = 0.0001,
    early_stopping_min_epochs: int = 0,
    device: torch.device = torch.device("cpu"),
    checkpoint_path: Optional[str | Path] = None,
    resume_checkpoint_path: Optional[str | Path] = None,
    metrics_csv_path: Optional[str | Path] = None,
    verbose: bool = True,
    grad_clip: float = 1.0,
    scheduler_cfg: Optional[Dict[str, Any]] = None,
    amp: bool = False,
    resolved_config: Optional[Dict[str, Any]] = None,
    preprocess_identity: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    label_map: Optional[Dict[str, int]] = None,
    seed: Optional[int] = None,
) -> Dict[str, list]:
    """Huấn luyện mô hình GRU với Stateful Checkpoint, Resume, CSV metric streaming và Early Stopping cải tiến."""
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler, sched_mode = create_scheduler(optimizer, scheduler_cfg, epochs)

    scaler = None
    if amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    start_epoch = 1
    early_stopping_best = float("inf")

    history: Dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "lr": [],
    }

    # Resolve class_names and label_map if not explicitly supplied
    if class_names is None:
        if train_loader is not None and hasattr(train_loader, "dataset") and getattr(train_loader.dataset, "class_names", None):
            class_names = list(train_loader.dataset.class_names)
        elif val_loader is not None and hasattr(val_loader, "dataset") and getattr(val_loader.dataset, "class_names", None):
            class_names = list(val_loader.dataset.class_names)
    if label_map is None:
        if train_loader is not None and hasattr(train_loader, "dataset") and getattr(train_loader.dataset, "label_map", None):
            label_map = dict(train_loader.dataset.label_map)
        elif val_loader is not None and hasattr(val_loader, "dataset") and getattr(val_loader.dataset, "label_map", None):
            label_map = dict(val_loader.dataset.label_map)
    if class_names is None and hasattr(model, "num_classes") and model.num_classes is not None:
        class_names = [f"class_{i}" for i in range(model.num_classes)]
    if label_map is None and class_names is not None:
        label_map = {name: idx for idx, name in enumerate(class_names)}

    # Phục hồi trạng thái huấn luyện nếu có resume_checkpoint_path
    if resume_checkpoint_path is not None:
        resume_p = Path(resume_checkpoint_path)
        if not resume_p.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint file not found: {resume_p}. Silent fresh start is strictly prohibited."
            )

        target_device = device if device is not None else torch.device("cpu")
        ckpt_payload = torch.load(str(resume_p), map_location=target_device, weights_only=False)

        if not isinstance(ckpt_payload, dict) or "model_state_dict" not in ckpt_payload:
            raise ValueError(f"Invalid checkpoint payload at {resume_p}: missing 'model_state_dict'.")

        # 1. Canonical checkpoint metadata validation
        saved_cfg, ckpt_classes, ckpt_label_map = validate_checkpoint_metadata(
            ckpt_payload=ckpt_payload,
            checkpoint_path=resume_p,
            expected_preprocess_identity=preprocess_identity,
            allow_legacy=False,
            require_optimizer=(optimizer is not None),
        )

        # 2. Strict classes and label_map verification against runtime expectation
        if class_names is not None and list(ckpt_classes) != list(class_names):
            raise ValueError(
                f"Checkpoint class names mismatch: checkpoint has {ckpt_classes}, expected {class_names}."
            )
        if label_map is not None and ckpt_label_map != label_map:
            raise ValueError(
                f"Checkpoint label_map mismatch: checkpoint has {ckpt_label_map}, expected {label_map}."
            )

        # 3. Model architecture semantics verification
        validate_model_architecture(saved_cfg, model, checkpoint_path=resume_p)

        # 4. Architecture and tensor shape verification (including unexpected extra keys)
        model_sd = model.state_dict()
        ckpt_sd = ckpt_payload["model_state_dict"]

        if set(ckpt_sd.keys()) != set(model_sd.keys()):
            unexpected_keys = set(ckpt_sd.keys()) - set(model_sd.keys())
            missing_keys = set(model_sd.keys()) - set(ckpt_sd.keys())
            raise ValueError(
                f"Checkpoint architecture mismatch: unexpected keys {sorted(unexpected_keys)}, "
                f"missing keys {sorted(missing_keys)}. Retrain required."
            )

        for param_name, param_tensor in model_sd.items():
            if ckpt_sd[param_name].shape != param_tensor.shape:
                raise ValueError(
                    f"Checkpoint parameter shape mismatch for '{param_name}': expected {param_tensor.shape}, "
                    f"got {ckpt_sd[param_name].shape}. Retrain required."
                )

        # 5. Pre-parse ALL training state scalars BEFORE mutating any weights or optimizer
        if "epoch" not in ckpt_payload:
            raise ValueError(f"Checkpoint at {resume_p} is missing required 'epoch'.")
        try:
            saved_epoch = int(ckpt_payload["epoch"])
        except Exception as e:
            raise ValueError(f"Invalid 'epoch' in checkpoint at {resume_p}: {ckpt_payload.get('epoch')}") from e

        try:
            parsed_val_loss = float(ckpt_payload.get("best_val_loss", float("inf")))
            parsed_val_acc = float(ckpt_payload.get("best_val_acc", 0.0))
            parsed_patience = int(ckpt_payload.get("patience_counter", 0))
            parsed_es_best = float(ckpt_payload.get("early_stopping_best", parsed_val_loss))
        except Exception as e:
            raise ValueError(f"Invalid metric/counter scalar in checkpoint at {resume_p}: {e}") from e

        # 6. Atomic state application: commit only if all operations succeed without error
        orig_model_state = copy.deepcopy(model.state_dict())
        orig_opt_state = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
        orig_sched_state = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
        orig_scaler_state = copy.deepcopy(scaler.state_dict()) if scaler is not None else None
        try:
            model.load_state_dict(ckpt_sd)
            if optimizer is not None and "optimizer_state_dict" in ckpt_payload and ckpt_payload["optimizer_state_dict"] is not None:
                optimizer.load_state_dict(ckpt_payload["optimizer_state_dict"])
            if scheduler is not None and ckpt_payload.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt_payload["scheduler_state_dict"])
            if scaler is not None and ckpt_payload.get("scaler_state_dict") is not None:
                scaler.load_state_dict(ckpt_payload["scaler_state_dict"])
        except Exception:
            # Atomic rollback on any failure to guarantee zero mutation
            model.load_state_dict(orig_model_state)
            if optimizer is not None and orig_opt_state is not None:
                optimizer.load_state_dict(orig_opt_state)
            if scheduler is not None and orig_sched_state is not None:
                scheduler.load_state_dict(orig_sched_state)
            if scaler is not None and orig_scaler_state is not None:
                scaler.load_state_dict(orig_scaler_state)
            raise

        start_epoch = saved_epoch + 1
        best_val_loss = parsed_val_loss
        best_val_acc = parsed_val_acc
        patience_counter = parsed_patience
        early_stopping_best = parsed_es_best
        best_weights = copy.deepcopy(model.state_dict())
        saved_hist = ckpt_payload.get("history")
        if isinstance(saved_hist, dict):
            for k in history:
                if k in saved_hist and isinstance(saved_hist[k], list):
                    history[k] = list(saved_hist[k])
        if verbose:
            print(f"Resumed from checkpoint: {resume_p} (epoch {saved_epoch}, best_val_loss={best_val_loss:.4f})")

    # Khởi tạo CSV Streaming cho metrics nếu có đường dẫn
    if metrics_csv_path is None and checkpoint_path is not None:
        metrics_csv_path = Path(checkpoint_path).parent / "train_metrics.csv"

    csv_file = None
    csv_writer = None
    if metrics_csv_path is not None:
        csv_p = Path(metrics_csv_path)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_p.is_file() and csv_p.stat().st_size > 0
        open_mode = "a" if (file_exists and resume_checkpoint_path is not None) else "w"
        csv_file = open(csv_p, mode=open_mode, encoding="utf-8", newline="")
        csv_writer = csv.writer(csv_file)
        if open_mode == "w" or not file_exists:
            csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr"])
            csv_file.flush()

    t0 = time.time()
    try:
        if start_epoch > epochs:
            if verbose:
                print(f"Training already completed up to epoch {start_epoch - 1} >= target epochs {epochs}.")
        else:
            for ep in range(start_epoch, epochs + 1):
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

                # Ghi metric real-time ra CSV và flush ngay lập tức
                if csv_writer is not None and csv_file is not None:
                    csv_writer.writerow([ep, f"{tr_loss:.6f}", f"{tr_acc:.4f}", f"{val_loss:.6f}", f"{val_acc:.4f}", f"{val_f1:.4f}", f"{current_lr:.8f}"])
                    csv_file.flush()

                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_weights = copy.deepcopy(model.state_dict())

                # Kiểm tra cải thiện theo min_delta cho early stopping
                if (early_stopping_best - val_loss) >= early_stopping_min_delta:
                    early_stopping_best = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if is_best and checkpoint_path:
                    save_checkpoint(
                        checkpoint_path=checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=ep,
                        best_val_loss=best_val_loss,
                        best_val_acc=best_val_acc,
                        patience_counter=patience_counter,
                        early_stopping_best=early_stopping_best,
                        history=history,
                        resolved_config=resolved_config,
                        schema_version=PREPROCESS_SCHEMA_VERSION,
                        preprocess_identity=preprocess_identity,
                        class_names=class_names,
                        label_map=label_map,
                        seed=seed,
                    )

                if verbose and (ep % 5 == 0 or ep == 1 or is_best or patience_counter >= patience):
                    star = " (*)" if is_best else ""
                    print(f"Epoch {ep:2d}/{epochs} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} | lr: {current_lr:.6f}{star}")

                # Kích hoạt Early Stopping nếu đã vượt qua early_stopping_min_epochs
                if patience_counter >= patience and ep >= early_stopping_min_epochs:
                    if verbose:
                        print(
                            f"Early Stopping tại epoch {ep} (patience={patience}, "
                            f"min_delta={early_stopping_min_delta}, min_epochs={early_stopping_min_epochs}). "
                            f"Best Val Loss: {best_val_loss:.4f}"
                        )
                    break
    finally:
        if csv_file is not None:
            csv_file.close()

    # Load lại trọng số tốt nhất
    model.load_state_dict(best_weights)
    elapsed = time.time() - t0
    history["training_time_sec"] = elapsed
    history["best_val_loss"] = best_val_loss
    history["best_val_acc"] = best_val_acc
    return history
