"""Independent anatomical projections followed by fixed weighted early fusion."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.features.schema import FeatureLayout


class RegionFusionClassifier(nn.Module):
    """Keep temporal architecture fixed while removing entire input regions.

    All projections are registered in every subset so initialization and total
    parameter count are stable. Disabled branches are never read or evaluated.
    Weights are normalized over enabled regions only; missing observations are
    zeroed without redistributing their weight to other regions.
    """

    def __init__(self, core: nn.Module, regions: Sequence[str], weights: Mapping[str, float], epsilon: float):
        super().__init__()
        layout = FeatureLayout.build(feature_set="hands_pose", face=True)
        self.input_size = layout.total_dim
        self.core = core
        for name in ("hidden_size", "num_layers", "num_classes", "bidirectional", "pooling", "rnn_type"):
            setattr(self, name, getattr(core, name))
        for name in ("attention_heads", "feedforward_size"):
            if hasattr(core, name):
                setattr(self, name, getattr(core, name))
        self.region_fusion = True
        self.regions = tuple(regions)
        self.region_weights = dict(weights)
        self.region_normalization_epsilon = epsilon
        self.slices = {"hands": slice(0, layout.slice_pose_features.start),
                       "body": layout.slice_pose_features,
                       "face": slice(layout.slice_face_shapes.start, layout.total_dim)}
        self.support_slices = {"hands": layout.slice_hand_masks,
                               "body": layout.slice_pose_mask, "face": layout.slice_face_mask}
        self.projections = nn.ModuleDict({name: nn.Linear(block.stop - block.start, self.hidden_size, bias=False)
                                          for name, block in self.slices.items()})
        denominator = sum(weights[name] for name in self.regions)
        self.fusion_weights = {name: weights[name] / denominator for name in self.regions}

    def fuse(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_size or x.shape[1] == 0:
            raise ValueError(f"Expected nonempty (batch, time, {self.input_size}) region features.")
        fused = x.new_zeros((*x.shape[:2], self.hidden_size))
        for name in self.regions:
            supported = (x[..., self.support_slices[name]] > 0.5).any(dim=-1, keepdim=True)
            block = torch.where(supported, x[..., self.slices[name]], 0.0)
            projected = self.projections[name](block)
            normalized = F.normalize(projected, p=2, dim=-1, eps=self.region_normalization_epsilon)
            fused = fused + self.fusion_weights[name] * torch.where(supported, normalized, 0.0)
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(self.fuse(x))
