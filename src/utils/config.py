"""Strictly typed configuration system for ISLR Phase 0.

Provides dataclass-based validation, YAML loading, arbitrary depth CLI override handling,
and configuration serialization with zero magic numbers.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import yaml

DEFAULT_CONFIG_PATH = "configs/islr.yaml"

ALLOWED_ENV_NAMES = frozenset({"auto", "local", "colab", "kaggle"})
ALLOWED_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
ALLOWED_SAMPLE_STRATEGIES = frozenset({"spread_over_participants", "random"})
ALLOWED_REQUIRE_HANDS = frozenset({"any", "dominant_only", "both"})
ALLOWED_DIVERSITY_METHODS = frozenset({"greedy_maxmin", "top_score"})

ALLOWED_WEIGHT_KEYS = frozenset({
    "samples",
    "participant_coverage",
    "participant_balance",
    "hand_coverage",
    "motion_energy",
    "separability",
    "handshape_diversity",
})

ALLOWED_CANDIDATE_PREFILTER_KEYS = frozenset({
    "min_samples",
    "min_participants",
    "max_participant_dominance",
})

MIN_SEED = 0
MIN_SAMPLE_BUDGET_FILES = 1
MIN_LANDMARK_PROBE_FILES = 1
MIN_EXPECTED_LANDMARKS = 1

REQUIRED_LANDMARK_COLUMNS = frozenset({
    "frame",
    "row_id",
    "type",
    "landmark_index",
    "x",
    "y",
    "z",
})


@dataclass
class EnvConfig:
    name: str = "auto"
    seed: int = 42
    use_drive: bool = False
    project_dir_name: str = "asl-signs"
    local_root: str = "."
    log_level: str = "INFO"

    def validate(self) -> None:
        if self.name not in ALLOWED_ENV_NAMES:
            raise ValueError(
                f"Invalid env.name '{self.name}'. Allowed values are: {sorted(ALLOWED_ENV_NAMES)}"
            )
        if type(self.seed) is bool or not isinstance(self.seed, int):
            raise TypeError(
                f"env.seed must be an integer, got {type(self.seed).__name__}"
            )
        if self.seed < MIN_SEED:
            raise ValueError(
                f"env.seed must be >= {MIN_SEED}, got {self.seed}"
            )
        if not isinstance(self.use_drive, bool):
            raise TypeError(
                f"env.use_drive must be a boolean, got {type(self.use_drive).__name__}"
            )
        if not isinstance(self.project_dir_name, str) or not self.project_dir_name.strip():
            raise ValueError("env.project_dir_name must be a non-empty string.")
        if not isinstance(self.local_root, str) or not self.local_root.strip():
            raise ValueError("env.local_root must be a non-empty string.")
        if not isinstance(self.log_level, str) or self.log_level.upper() not in ALLOWED_LOG_LEVELS:
            raise ValueError(
                f"Invalid env.log_level '{self.log_level}'. Allowed values are: {sorted(ALLOWED_LOG_LEVELS)}"
            )


@dataclass
class DataConfig:
    competition_id: str = "asl-signs"
    train_csv_name: str = "train.csv"
    parquet_remote_template: str = "train_landmark_files/{participant_id}/{sequence_id}.parquet"
    local_mirror_dir: Optional[str] = None
    cache_dir: str = "data/cache"
    download_budget_total_files: Optional[int] = None
    max_files_per_gloss: Optional[int] = None
    num_download_workers: int = 4
    retry: int = 3
    sample_budget_total_files: int = 5
    seqs_per_candidate_sign: int = 2
    sample_strategy: str = "spread_over_participants"
    sample_seed: int = 42
    parquet_hand_types: List[str] = field(default_factory=lambda: ["left_hand", "right_hand"])
    landmark_files_dir: str = "train_landmark_files"

    def validate(self) -> None:
        if not isinstance(self.competition_id, str) or not self.competition_id.strip():
            raise ValueError("data.competition_id must be a non-empty string.")
        if not isinstance(self.train_csv_name, str) or not self.train_csv_name.strip():
            raise ValueError("data.train_csv_name must be a non-empty string.")
        if not isinstance(self.parquet_remote_template, str) or not self.parquet_remote_template.strip():
            raise ValueError("data.parquet_remote_template must be a non-empty string.")
        if self.local_mirror_dir is not None:
            if not isinstance(self.local_mirror_dir, str) or not self.local_mirror_dir.strip():
                raise ValueError("data.local_mirror_dir must be None or a non-empty string.")
        if not isinstance(self.cache_dir, str) or not self.cache_dir.strip():
            raise ValueError("data.cache_dir must be a non-empty string.")
        if self.download_budget_total_files is not None:
            if type(self.download_budget_total_files) is bool or not isinstance(self.download_budget_total_files, int) or self.download_budget_total_files < 1:
                raise ValueError("data.download_budget_total_files must be None or an integer >= 1.")
        if self.max_files_per_gloss is not None:
            if type(self.max_files_per_gloss) is bool or not isinstance(self.max_files_per_gloss, int) or self.max_files_per_gloss < 1:
                raise ValueError("data.max_files_per_gloss must be None or an integer >= 1.")
        if type(self.num_download_workers) is bool or not isinstance(self.num_download_workers, int) or self.num_download_workers < 1:
            raise ValueError("data.num_download_workers must be an integer >= 1.")
        if type(self.retry) is bool or not isinstance(self.retry, int) or self.retry < 0:
            raise ValueError("data.retry must be an integer >= 0.")
        if type(self.sample_budget_total_files) is bool or not isinstance(self.sample_budget_total_files, int):
            raise TypeError(
                f"data.sample_budget_total_files must be an integer, got {type(self.sample_budget_total_files).__name__}"
            )
        if self.sample_budget_total_files < MIN_SAMPLE_BUDGET_FILES:
            raise ValueError(
                f"data.sample_budget_total_files must be >= {MIN_SAMPLE_BUDGET_FILES}, got {self.sample_budget_total_files}"
            )
        if type(self.seqs_per_candidate_sign) is bool or not isinstance(self.seqs_per_candidate_sign, int):
            raise TypeError("data.seqs_per_candidate_sign must be an integer.")
        if self.seqs_per_candidate_sign < 1:
            raise ValueError("data.seqs_per_candidate_sign must be >= 1.")
        if self.sample_strategy not in ALLOWED_SAMPLE_STRATEGIES:
            raise ValueError(
                f"Invalid data.sample_strategy '{self.sample_strategy}'. "
                f"Allowed values are: {sorted(ALLOWED_SAMPLE_STRATEGIES)}"
            )
        if type(self.sample_seed) is bool or not isinstance(self.sample_seed, int):
            raise TypeError("data.sample_seed must be an integer.")
        if not isinstance(self.parquet_hand_types, list) or not self.parquet_hand_types:
            raise ValueError("data.parquet_hand_types must be a non-empty list.")
        if not isinstance(self.landmark_files_dir, str) or not self.landmark_files_dir.strip():
            raise ValueError("data.landmark_files_dir must be a non-empty string.")


@dataclass
class EDAConfig:
    num_landmark_probe_files: int = 1
    num_expected_landmarks: int = 543
    top_n_plot: int = 20
    min_frames: int = 10
    max_frames: int = 200
    min_hand_coverage: float = 0.1
    expected_columns: List[str] = field(
        default_factory=lambda: [
            "frame",
            "row_id",
            "type",
            "landmark_index",
            "x",
            "y",
            "z",
        ]
    )

    def validate(self) -> None:
        if type(self.num_landmark_probe_files) is bool or not isinstance(self.num_landmark_probe_files, int):
            raise TypeError(
                f"eda.num_landmark_probe_files must be an integer, got {type(self.num_landmark_probe_files).__name__}"
            )
        if self.num_landmark_probe_files < MIN_LANDMARK_PROBE_FILES:
            raise ValueError(
                f"eda.num_landmark_probe_files must be >= {MIN_LANDMARK_PROBE_FILES}, got {self.num_landmark_probe_files}"
            )
        if type(self.num_expected_landmarks) is bool or not isinstance(self.num_expected_landmarks, int):
            raise TypeError(
                f"eda.num_expected_landmarks must be an integer, got {type(self.num_expected_landmarks).__name__}"
            )
        if self.num_expected_landmarks < MIN_EXPECTED_LANDMARKS:
            raise ValueError(
                f"eda.num_expected_landmarks must be >= {MIN_EXPECTED_LANDMARKS}, got {self.num_expected_landmarks}"
            )
        if type(self.top_n_plot) is bool or not isinstance(self.top_n_plot, int) or self.top_n_plot < 1:
            raise ValueError("eda.top_n_plot must be an integer >= 1.")
        if type(self.min_frames) is bool or not isinstance(self.min_frames, int) or self.min_frames < 1:
            raise ValueError("eda.min_frames must be an integer >= 1.")
        if type(self.max_frames) is bool or not isinstance(self.max_frames, int) or self.max_frames <= self.min_frames:
            raise ValueError("eda.max_frames must be an integer greater than eda.min_frames.")
        if not isinstance(self.min_hand_coverage, (int, float)) or not (0.0 <= self.min_hand_coverage <= 1.0):
            raise ValueError("eda.min_hand_coverage must be a float between 0.0 and 1.0.")
        if not isinstance(self.expected_columns, list) or not self.expected_columns:
            raise ValueError("eda.expected_columns must be a non-empty list of column names.")
        for col in self.expected_columns:
            if not isinstance(col, str) or not col.strip():
                raise ValueError("eda.expected_columns must contain non-empty string column names.")
        if len(set(self.expected_columns)) != len(self.expected_columns):
            raise ValueError("eda.expected_columns must not contain duplicate column names.")
        missing_required = REQUIRED_LANDMARK_COLUMNS - set(self.expected_columns)
        if missing_required:
            raise ValueError(
                f"eda.expected_columns missing required competition columns: {sorted(missing_required)}"
            )


@dataclass
class SelectionConfig:
    n_glosses: int = 20
    n_candidates: int = 60
    candidate_prefilter: Dict[str, Any] = field(
        default_factory=lambda: {
            "min_samples": 10,
            "min_participants": 5,
            "max_participant_dominance": 0.5,
        }
    )
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "samples": 1.0,
            "participant_coverage": 1.0,
            "participant_balance": 1.0,
            "hand_coverage": 1.0,
            "motion_energy": 1.0,
            "separability": 1.0,
            "handshape_diversity": 1.0,
        }
    )
    require_hands: str = "any"
    exclude_signs: List[str] = field(default_factory=list)
    force_include_signs: List[str] = field(default_factory=list)
    diversity_method: str = "greedy_maxmin"
    selection_seed: int = 42

    def validate(self) -> None:
        if type(self.n_glosses) is bool or not isinstance(self.n_glosses, int) or self.n_glosses < 1:
            raise ValueError("selection.n_glosses must be an integer >= 1.")
        if type(self.n_candidates) is bool or not isinstance(self.n_candidates, int) or self.n_candidates < self.n_glosses:
            raise ValueError("selection.n_candidates must be >= selection.n_glosses.")
        if not isinstance(self.candidate_prefilter, dict):
            raise TypeError("selection.candidate_prefilter must be a dictionary.")
        for k in self.candidate_prefilter.keys():
            if k not in ALLOWED_CANDIDATE_PREFILTER_KEYS:
                raise KeyError(
                    f"Unknown candidate_prefilter key '{k}'. "
                    f"Allowed keys: {sorted(ALLOWED_CANDIDATE_PREFILTER_KEYS)}"
                )
        if not isinstance(self.weights, dict):
            raise TypeError("selection.weights must be a dictionary.")
        for k, v in self.weights.items():
            if k not in ALLOWED_WEIGHT_KEYS:
                raise KeyError(
                    f"Unknown weight key '{k}' in selection.weights. "
                    f"Allowed keys: {sorted(ALLOWED_WEIGHT_KEYS)}"
                )
            if not isinstance(v, (int, float)):
                raise TypeError(f"Weight '{k}' must be a numeric value, got {type(v).__name__}")
            if v < 0:
                raise ValueError(f"Weight '{k}' must be non-negative, got {v}")
        if self.require_hands not in ALLOWED_REQUIRE_HANDS:
            raise ValueError(
                f"Invalid selection.require_hands '{self.require_hands}'. "
                f"Allowed values: {sorted(ALLOWED_REQUIRE_HANDS)}"
            )
        if not isinstance(self.exclude_signs, list):
            raise TypeError("selection.exclude_signs must be a list.")
        if not isinstance(self.force_include_signs, list):
            raise TypeError("selection.force_include_signs must be a list.")
        if self.diversity_method not in ALLOWED_DIVERSITY_METHODS:
            raise ValueError(
                f"Invalid selection.diversity_method '{self.diversity_method}'. "
                f"Allowed values: {sorted(ALLOWED_DIVERSITY_METHODS)}"
            )
        if type(self.selection_seed) is bool or not isinstance(self.selection_seed, int) or self.selection_seed < 0:
            raise ValueError("selection.selection_seed must be an integer >= 0.")


@dataclass
class OutputConfig:
    output_dir: str = "outputs"
    report_dir: str = "reports"
    selection_dir: str = "data/selection"
    save_resolved_config: bool = True
    overwrite: bool = True

    def validate(self) -> None:
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output.output_dir must be a non-empty string.")
        if not isinstance(self.report_dir, str) or not self.report_dir.strip():
            raise ValueError("output.report_dir must be a non-empty string.")
        if not isinstance(self.selection_dir, str) or not self.selection_dir.strip():
            raise ValueError("output.selection_dir must be a non-empty string.")
        if not isinstance(self.save_resolved_config, bool):
            raise TypeError(
                f"output.save_resolved_config must be a boolean, got {type(self.save_resolved_config).__name__}"
            )
        if not isinstance(self.overwrite, bool):
            raise TypeError(
                f"output.overwrite must be a boolean, got {type(self.overwrite).__name__}"
            )


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eda: EDAConfig = field(default_factory=EDAConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        self.env.validate()
        self.data.validate()
        self.eda.validate()
        self.selection.validate()
        self.output.validate()
        if self.eda.num_landmark_probe_files > self.data.sample_budget_total_files:
            raise ValueError(
                f"eda.num_landmark_probe_files ({self.eda.num_landmark_probe_files}) "
                f"cannot exceed data.sample_budget_total_files ({self.data.sample_budget_total_files})."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTION_CLASSES = {
    "env": EnvConfig,
    "data": DataConfig,
    "eda": EDAConfig,
    "selection": SelectionConfig,
    "output": OutputConfig,
}


def _get_allowed_keys_for_section(section_name: str) -> Set[str]:
    cls = _SECTION_CLASSES.get(section_name)
    if cls is None:
        return set()
    return {f.name for f in fields(cls)}


def _apply_override(cfg_dict: Dict[str, Any], override_str: str) -> None:
    """Apply an arbitrary depth key.path=val override to the raw dictionary."""
    if "=" not in override_str:
        raise ValueError(
            f"Invalid override format '{override_str}'. Expected 'key.path=val'."
        )

    key_path, val_str = override_str.split("=", 1)
    key_path = key_path.strip()
    if not key_path:
        raise ValueError(f"Empty key path in override '{override_str}'.")

    try:
        parsed_val = yaml.safe_load(val_str)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Failed to parse override value '{val_str}' for key '{key_path}': {exc}"
        ) from exc

    tokens = [t.strip() for t in key_path.split(".")]
    if len(tokens) < 2:
        raise ValueError(
            f"Override '{key_path}' must specify at least a section and field name (e.g. 'env.seed=42')."
        )

    section_name = tokens[0]
    if section_name not in _SECTION_CLASSES:
        raise KeyError(
            f"Unknown configuration section '{section_name}' in override '{key_path}'. "
            f"Valid sections are: {sorted(_SECTION_CLASSES.keys())}"
        )

    if section_name in cfg_dict and not isinstance(cfg_dict[section_name], dict):
        raise TypeError(
            f"Section '{section_name}' must be a dictionary, got {type(cfg_dict[section_name]).__name__}"
        )

    field_name = tokens[1]
    allowed_keys = _get_allowed_keys_for_section(section_name)
    if field_name not in allowed_keys:
        raise KeyError(
            f"Unknown configuration key '{field_name}' in section '{section_name}'. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )

    section_cls = _SECTION_CLASSES[section_name]
    target_field = next(f for f in fields(section_cls) if f.name == field_name)

    if len(tokens) > 2:
        # Check if target_field is typed as a dict
        origin = getattr(target_field.type, "__origin__", target_field.type)
        is_dict_field = (
            origin is dict
            or target_field.type is dict
            or (isinstance(target_field.type, str) and "Dict" in target_field.type)
        )
        if not is_dict_field:
            raise TypeError(
                f"Cannot access nested key on non-dict field '{section_name}.{field_name}' "
                f"(type is {target_field.type})."
            )

        if section_name not in cfg_dict or not isinstance(cfg_dict[section_name], dict):
            cfg_dict[section_name] = {}
        if field_name not in cfg_dict[section_name] or not isinstance(cfg_dict[section_name][field_name], dict):
            cfg_dict[section_name][field_name] = {}

        curr = cfg_dict[section_name][field_name]
        for inner_key in tokens[2:-1]:
            if inner_key not in curr or not isinstance(curr[inner_key], dict):
                curr[inner_key] = {}
            curr = curr[inner_key]
        curr[tokens[-1]] = parsed_val
    else:
        if section_name not in cfg_dict or not isinstance(cfg_dict[section_name], dict):
            cfg_dict[section_name] = {}
        cfg_dict[section_name][field_name] = parsed_val


def _build_config_from_dict(raw_dict: Dict[str, Any]) -> Config:
    """Build and strictly validate Config from a dictionary."""
    valid_sections = set(_SECTION_CLASSES.keys())
    for key in raw_dict.keys():
        if key not in valid_sections:
            raise KeyError(
                f"Unknown configuration section '{key}'. "
                f"Valid sections are: {sorted(valid_sections)}"
            )

    sections: Dict[str, Any] = {}
    for section_name, cls in _SECTION_CLASSES.items():
        section_data = raw_dict.get(section_name, {})
        if not isinstance(section_data, dict):
            raise TypeError(
                f"Section '{section_name}' must be a dictionary, got {type(section_data).__name__}"
            )
        allowed_keys = _get_allowed_keys_for_section(section_name)
        for field_name in section_data.keys():
            if field_name not in allowed_keys:
                raise KeyError(
                    f"Unknown configuration key '{field_name}' in section '{section_name}'. "
                    f"Allowed keys: {sorted(allowed_keys)}"
                )
        try:
            sections[section_name] = cls(**section_data)
        except TypeError as exc:
            raise TypeError(
                f"Failed to instantiate section '{section_name}': {exc}"
            ) from exc

    config = Config(**sections)
    config.validate()
    return config


def load_config(
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    overrides: Optional[Sequence[str]] = None,
) -> Config:
    """Load configuration from YAML file with strict validation and overrides.

    Args:
        config_path: Path to YAML configuration file.
        overrides: Optional list of override strings formatted as 'key.path=val'.

    Returns:
        Config: Validated Config dataclass instance.

    Raises:
        FileNotFoundError: If the config file cannot be found.
        ValueError: If YAML is invalid or values fail validation.
        KeyError: If unknown sections or keys are encountered.
    """
    path = Path(config_path)
    if not path.is_file():
        repo_candidate = (Path(__file__).resolve().parent.parent.parent / path).resolve()
        if repo_candidate.is_file():
            path = repo_candidate
        else:
            raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML configuration '{path}': {exc}") from exc

    if raw is None:
        raw = {}
    elif not isinstance(raw, dict):
        raise ValueError(f"Config YAML must define a mapping/dict, got {type(raw).__name__}")

    if overrides:
        for override in overrides:
            _apply_override(raw, override)

    return _build_config_from_dict(raw)


def save_config(cfg: Config, target_path: Union[str, Path]) -> Path:
    """Save resolved configuration dictionary to a YAML file.

    Args:
        cfg: Validated Config dataclass.
        target_path: Destination path for saved configuration.

    Returns:
        Path: Resolved Path to the saved file.
    """
    out_path = Path(target_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg.to_dict(), f, indent=2, sort_keys=False)
    return out_path


def parse_cli_args(
    args: Optional[Sequence[str]] = None,
) -> Tuple[Config, argparse.Namespace]:
    """Parse CLI arguments, resolving --config and --set overrides.

    Args:
        args: Optional list of command-line argument strings.

    Returns:
        Tuple[Config, argparse.Namespace]: Resolved Config and parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="ISLR Phase 0 Configuration and Setup Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        default=None,
        help="Override config value formatted as 'section.key=val' or 'section.field.inner=val'",
    )

    parsed_args = parser.parse_args(args)
    cfg = load_config(
        config_path=parsed_args.config,
        overrides=parsed_args.overrides,
    )
    return cfg, parsed_args
