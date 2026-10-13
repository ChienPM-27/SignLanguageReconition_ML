"""Custom in-memory DataLoader for preloaded device tensors.

Phục vụ dữ liệu trực tiếp từ GPU/CPU memory:
- Không sử dụng đa tiến trình worker (tránh overhead IPC và memory copy).
- Hỗ trợ deterministic shuffling với torch.Generator độc lập (tái lập hoàn toàn theo seed).
- Duy trì thuộc tính .dataset và __len__ tương thích với giao diện PyTorch DataLoader chuẩn.
- Hỗ trợ drop_last, batch_size và batch-level augmentation hook.
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, Optional, Tuple
import torch


class DeviceDataLoader:
    """DataLoader phục vụ tensor từ bộ nhớ thiết bị (CPU/GPU)."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        augment: bool = False,
        augmentation_config: Optional[Any] = None,
        augmentation_fn: Optional[Callable[[torch.Tensor, Any, Optional[torch.Generator]], torch.Tensor]] = None,
    ):
        if getattr(dataset, "preloaded_features", None) is None or getattr(dataset, "preloaded_labels", None) is None:
            raise ValueError("DeviceDataLoader yêu cầu dataset phải có preloaded_features và preloaded_labels.")

        self.dataset = dataset
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.augment = augment
        self.augmentation_config = augmentation_config
        self.augmentation_fn = augmentation_fn

        if generator is not None:
            self.generator = generator
        elif seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            self.generator = g
        else:
            self.generator = None

    def __len__(self) -> int:
        n = len(self.dataset)
        if n == 0:
            return 0
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        n = len(self.dataset)
        if n == 0:
            return

        features: torch.Tensor = self.dataset.preloaded_features
        labels: torch.Tensor = self.dataset.preloaded_labels
        device = features.device

        if self.shuffle:
            if self.generator is not None:
                perm = torch.randperm(n, generator=self.generator, device="cpu").to(device)
            else:
                perm = torch.randperm(n, device=device)
        else:
            perm = torch.arange(n, device=device)

        total_samples = (n // self.batch_size) * self.batch_size if self.drop_last else n

        for start_idx in range(0, total_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n)
            if self.drop_last and (end_idx - start_idx) < self.batch_size:
                break
            batch_indices = perm[start_idx:end_idx]
            batch_x = features[batch_indices]
            batch_y = labels[batch_indices]

            if self.augment and self.augmentation_fn is not None:
                batch_x = self.augmentation_fn(batch_x, self.augmentation_config, self.generator)

            yield batch_x, batch_y
