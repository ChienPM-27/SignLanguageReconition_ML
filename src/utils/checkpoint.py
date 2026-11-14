"""Canonical checkpoint validation and atomic loading utility.

Enforces strict provenance, metadata consistency, and atomic rollback across
training resume, inference, and standalone evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn as nn

from src.features.schema import PREPROCESS_SCHEMA_VERSION, compute_preprocess_identity
from src.utils.config import Config, build_config_from_dict


def validate_checkpoint_metadata(
    ckpt_payload: Dict[str, Any],
    checkpoint_path: Optional[Union[str, Path]] = None,
    expected_preprocess_identity: Optional[str] = None,
    allow_legacy: bool = False,
    require_optimizer: bool = False,
) -> Tuple[Config, List[str], Dict[str, int]]:
    """Validate canonical metadata within a checkpoint dictionary.

    Returns:
        Tuple of (saved_config, class_names, label_map)

    Raises:
        ValueError, KeyError: If any invariant is violated.
    """
    path_str = f" at {checkpoint_path}" if checkpoint_path else ""

    if not isinstance(ckpt_payload, dict):
        raise ValueError(f"Invalid checkpoint payload{path_str}: root must be a dict.")

    # 1. Schema version
    schema_version = ckpt_payload.get("schema_version")
    if schema_version != PREPROCESS_SCHEMA_VERSION:
        if allow_legacy and schema_version is None and "resolved_config" not in ckpt_payload:
            pass  # Legacy checkpoint
        else:
            raise ValueError(
                f"Checkpoint schema version mismatch (Unsupported checkpoint schema_version{path_str}: "
                f"expected {PREPROCESS_SCHEMA_VERSION}, got {schema_version})."
            )

    # 2. Resolved config
    if "resolved_config" not in ckpt_payload or not ckpt_payload["resolved_config"]:
        if allow_legacy:
            return None, [], {}  # type: ignore[return-value]
        raise ValueError(
            f"Checkpoint{path_str} is missing canonical 'resolved_config'. "
            "Provenanced training configuration is required."
        )

    try:
        saved_cfg = build_config_from_dict(ckpt_payload["resolved_config"])
    except KeyError as ke:
        raise KeyError(
            f"Checkpoint resolved_config contains unknown or deprecated keys: {ke}"
        ) from ke
    except Exception as ce:
        raise ValueError(f"Failed to parse checkpoint resolved_config: {ce}") from ce

    # 3. Preprocess identity verification
    ckpt_id = ckpt_payload.get("preprocess_identity")
    if not ckpt_id:
        raise ValueError(
            f"Checkpoint{path_str} is missing canonical 'preprocess_identity'. "
            "Cannot verify compatibility with preprocessed features."
        )

    recomputed_id, _ = compute_preprocess_identity(saved_cfg)
    if recomputed_id != ckpt_id:
        raise ValueError(
            f"Checkpoint metadata mismatch{path_str}: stored preprocess_identity '{ckpt_id}' "
            f"does not match identity recomputed from stored resolved_config '{recomputed_id}'."
        )

    if expected_preprocess_identity is not None and ckpt_id != expected_preprocess_identity:
        raise ValueError(
            f"Checkpoint preprocess_identity '{ckpt_id}' does not match "
            f"expected dataset preprocess_identity '{expected_preprocess_identity}'."
        )

    # 4. Classes and label map verification
    raw_classes = ckpt_payload.get("class_names")
    if raw_classes is None:
        raw_classes = ckpt_payload.get("classes")
    if raw_classes is None:
        raise ValueError(f"Checkpoint{path_str} is missing required 'class_names' list.")

    class_names = [str(c) for c in raw_classes]
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"Duplicate class names detected in checkpoint{path_str}: {class_names}.")

    raw_label_map = ckpt_payload.get("label_map")
    if raw_label_map is None:
        raise ValueError(f"Checkpoint{path_str} is missing required 'label_map' dictionary.")
    if not isinstance(raw_label_map, dict):
        raise ValueError(f"Invalid label_map in checkpoint{path_str}: must be a dictionary.")

    label_map = {str(k): int(v) for k, v in raw_label_map.items()}

    # Check contiguous label indices: 0 .. len(class_names) - 1
    expected_indices = list(range(len(class_names)))
    actual_indices = sorted(label_map.values())
    if actual_indices != expected_indices:
        raise ValueError(
            f"Non-contiguous label indices in checkpoint label_map{path_str}: "
            f"expected {expected_indices}, got {actual_indices}."
        )

    # Check bijectivity with class_names
    for idx, cname in enumerate(class_names):
        if cname not in label_map:
            raise ValueError(
                f"Class '{cname}' from class_names not found in label_map{path_str}."
            )
        if label_map[cname] != idx:
            raise ValueError(
                f"Class order / label map conflict in checkpoint{path_str}: "
                f"class_names[{idx}]='{cname}' but label_map['{cname}']={label_map[cname]}."
            )

    # 5. Optimizer state requirement
    if require_optimizer:
        if "optimizer_state_dict" not in ckpt_payload or ckpt_payload["optimizer_state_dict"] is None:
            raise ValueError(
                f"Checkpoint{path_str} is missing required 'optimizer_state_dict' for stateful resume."
            )
        opt_sd = ckpt_payload["optimizer_state_dict"]
        if not isinstance(opt_sd, dict) or "param_groups" not in opt_sd:
            raise ValueError(f"Invalid optimizer_state_dict in checkpoint{path_str}: missing 'param_groups'.")

    return saved_cfg, class_names, label_map


def validate_model_architecture(
    saved_cfg: Config,
    model: nn.Module,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> None:
    """Validate that in-memory model matches the architecture declared in saved_cfg."""
    path_str = f" at {checkpoint_path}" if checkpoint_path else ""

    # 1. Pooling
    if bool(getattr(model, "region_fusion", False)) != saved_cfg.model.region_fusion:
        raise ValueError(f"Model region_fusion mismatch{path_str}.")
    if saved_cfg.model.region_fusion:
        if (tuple(model.regions) != tuple(saved_cfg.model.regions)
                or model.region_weights != saved_cfg.model.region_weights
                or model.region_normalization_epsilon != saved_cfg.model.region_normalization_epsilon):
            raise ValueError(f"Model region settings mismatch{path_str}.")
    model_pooling = getattr(model, "pooling", None)
    if model_pooling is not None and model_pooling != saved_cfg.model.pooling:
        raise ValueError(
            f"Model pooling mismatch{path_str}: checkpoint was trained with pooling='{saved_cfg.model.pooling}', "
            f"but current model was initialized with pooling='{model_pooling}'."
        )

    # 2. RNN Type
    model_rnn_type = getattr(model, "rnn_type", None)
    if model_rnn_type is not None and model_rnn_type.upper() != saved_cfg.model.name.upper():
        raise ValueError(
            f"Model rnn_type mismatch{path_str}: checkpoint was trained with rnn_type='{saved_cfg.model.name}', "
            f"but current model was initialized with rnn_type='{model_rnn_type}'."
        )
    if saved_cfg.model.name.upper() == "TRANSFORMER":
        for key in ("attention_heads", "feedforward_size"):
            if getattr(model, key, None) != getattr(saved_cfg.model, key):
                raise ValueError(f"Model {key} mismatch{path_str}.")

    # 3. Hidden size
    model_hidden = getattr(model, "hidden_size", None)
    if model_hidden is not None and model_hidden != saved_cfg.model.hidden_size:
        raise ValueError(
            f"Model hidden_size mismatch{path_str}: checkpoint was trained with hidden_size={saved_cfg.model.hidden_size}, "
            f"but current model was initialized with hidden_size={model_hidden}."
        )

    # 4. Num layers
    model_layers = getattr(model, "num_layers", None)
    if model_layers is not None and model_layers != saved_cfg.model.num_layers:
        raise ValueError(
            f"Model num_layers mismatch{path_str}: checkpoint was trained with num_layers={saved_cfg.model.num_layers}, "
            f"but current model was initialized with num_layers={model_layers}."
        )

    # 5. Bidirectional
    model_bidi = getattr(model, "bidirectional", None)
    if model_bidi is not None and model_bidi != saved_cfg.model.bidirectional:
        raise ValueError(
            f"Model bidirectional mismatch{path_str}: checkpoint was trained with bidirectional={saved_cfg.model.bidirectional}, "
            f"but current model was initialized with bidirectional={model_bidi}."
        )
