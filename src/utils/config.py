"""Strictly typed configuration system for ISLR Phase 0.

Provides dataclass-based validation, YAML loading, CLI override handling,
and configuration serialization with zero magic numbers.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import yaml

DEFAULT_CONFIG_PATH = "configs/islr.yaml"
ALLOWED_ENV_NAMES = frozenset({"auto", "local", "colab", "kaggle"})
ALLOWED_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

DEFAULT_CONFIG_SEED = 42
DEFAULT_SAMPLE_BUDGET_TOTAL_FILES = 5
DEFAULT_NUM_LANDMARK_PROBE_FILES = 1
DEFAULT_NUM_EXPECTED_LANDMARKS = 543

DEFAULT_ENV_NAME = "auto"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_DRIVE_PROJECT_PATH = "asl-signs"
DEFAULT_LOCAL_ROOT_DIR = "."
DEFAULT_COMPETITION_ID = "asl-signs"
DEFAULT_LOCAL_MIRROR_DIR = "data/local_mirror"
DEFAULT_TRAIN_CSV_NAME = "train.csv"
DEFAULT_LANDMARK_FILES_DIR = "train_landmark_files"
DEFAULT_SELECTION_OUTPUT_DIR = "data/selection"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_REPORT_DIR = "reports"

MIN_SEED = 0
MIN_SAMPLE_BUDGET_FILES = 1
MIN_LANDMARK_PROBE_FILES = 1
MIN_EXPECTED_LANDMARKS = 1
MIN_EXPECTED_COLUMNS = 1

OVERRIDE_KV_DELIMITER = "="
KEY_PATH_DELIMITER = "."
MAX_SPLIT_OVERRIDE = 1
MAX_KEY_PATH_TOKENS = 2
SECTION_KEY_INDEX = 0
FIELD_KEY_INDEX = 1
SINGLE_SECTION_TOKEN_COUNT = 1
REPO_ROOT_PARENT_LEVEL = 2

CLI_CONFIG_FLAG = "--config"
CLI_SET_FLAG = "--set"

YAML_INDENT = 2

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
    name: str = DEFAULT_ENV_NAME
    seed: int = DEFAULT_CONFIG_SEED
    use_drive: bool = False
    drive_project_path: str = DEFAULT_DRIVE_PROJECT_PATH
    local_root_dir: str = DEFAULT_LOCAL_ROOT_DIR
    log_level: str = DEFAULT_LOG_LEVEL

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
        if not isinstance(self.drive_project_path, str) or not self.drive_project_path.strip():
            raise ValueError("env.drive_project_path must be a non-empty string.")
        if not isinstance(self.local_root_dir, str) or not self.local_root_dir.strip():
            raise ValueError("env.local_root_dir must be a non-empty string.")
        if not isinstance(self.log_level, str) or self.log_level.upper() not in ALLOWED_LOG_LEVELS:
            raise ValueError(
                f"Invalid env.log_level '{self.log_level}'. Allowed values are: {sorted(ALLOWED_LOG_LEVELS)}"
            )


@dataclass
class DataConfig:
    competition_id: str = DEFAULT_COMPETITION_ID
    local_mirror_dir: Optional[str] = DEFAULT_LOCAL_MIRROR_DIR
    sample_budget_total_files: int = DEFAULT_SAMPLE_BUDGET_TOTAL_FILES
    train_csv_name: str = DEFAULT_TRAIN_CSV_NAME
    landmark_files_dir: str = DEFAULT_LANDMARK_FILES_DIR

    def validate(self) -> None:
        if not isinstance(self.competition_id, str) or not self.competition_id.strip():
            raise ValueError("data.competition_id must be a non-empty string.")
        if self.local_mirror_dir is not None:
            if not isinstance(self.local_mirror_dir, str) or not self.local_mirror_dir.strip():
                raise ValueError("data.local_mirror_dir must be None or a non-empty string.")
        if type(self.sample_budget_total_files) is bool or not isinstance(self.sample_budget_total_files, int):
            raise TypeError(
                f"data.sample_budget_total_files must be an integer, got {type(self.sample_budget_total_files).__name__}"
            )
        if self.sample_budget_total_files < MIN_SAMPLE_BUDGET_FILES:
            raise ValueError(
                f"data.sample_budget_total_files must be >= {MIN_SAMPLE_BUDGET_FILES}, got {self.sample_budget_total_files}"
            )
        if not isinstance(self.train_csv_name, str) or not self.train_csv_name.strip():
            raise ValueError("data.train_csv_name must be a non-empty string.")
        if not isinstance(self.landmark_files_dir, str) or not self.landmark_files_dir.strip():
            raise ValueError("data.landmark_files_dir must be a non-empty string.")


@dataclass
class EDAConfig:
    num_landmark_probe_files: int = DEFAULT_NUM_LANDMARK_PROBE_FILES
    num_expected_landmarks: int = DEFAULT_NUM_EXPECTED_LANDMARKS
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
        if not isinstance(self.expected_columns, list) or len(self.expected_columns) < MIN_EXPECTED_COLUMNS:
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
    output_dir: str = DEFAULT_SELECTION_OUTPUT_DIR

    def validate(self) -> None:
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("selection.output_dir must be a non-empty string.")


@dataclass
class OutputConfig:
    output_dir: str = DEFAULT_OUTPUT_DIR
    report_dir: str = DEFAULT_REPORT_DIR
    save_resolved_config: bool = True

    def validate(self) -> None:
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output.output_dir must be a non-empty string.")
        if not isinstance(self.report_dir, str) or not self.report_dir.strip():
            raise ValueError("output.report_dir must be a non-empty string.")
        if not isinstance(self.save_resolved_config, bool):
            raise TypeError(
                f"output.save_resolved_config must be a boolean, got {type(self.save_resolved_config).__name__}"
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
    """Apply a single key.path=val override to the raw dictionary."""
    if OVERRIDE_KV_DELIMITER not in override_str:
        raise ValueError(
            f"Invalid override format '{override_str}'. Expected 'key.path=val'."
        )

    key_path, val_str = override_str.split(OVERRIDE_KV_DELIMITER, MAX_SPLIT_OVERRIDE)
    key_path = key_path.strip()
    if not key_path:
        raise ValueError(f"Empty key path in override '{override_str}'.")

    # Safely parse the value using PyYAML
    try:
        parsed_val = yaml.safe_load(val_str)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML value '{val_str}' in override '{override_str}': {exc}"
        ) from exc

    tokens = key_path.split(KEY_PATH_DELIMITER)
    section_name = tokens[SECTION_KEY_INDEX]
    if section_name not in _SECTION_CLASSES:
        raise KeyError(
            f"Unknown configuration section '{section_name}' in override '{key_path}'. "
            f"Valid sections are: {sorted(_SECTION_CLASSES.keys())}"
        )

    allowed_section_keys = _get_allowed_keys_for_section(section_name)
    if len(tokens) == SINGLE_SECTION_TOKEN_COUNT:
        raise ValueError(
            f"Override '{key_path}' targets an entire section. Specify a full path, e.g. 'env.seed=123'."
        )

    field_name = tokens[FIELD_KEY_INDEX]
    if field_name not in allowed_section_keys:
        raise KeyError(
            f"Unknown configuration key '{field_name}' in section '{section_name}'. "
            f"Allowed keys: {sorted(allowed_section_keys)}"
        )

    if len(tokens) > MAX_KEY_PATH_TOKENS:
        raise KeyError(
            f"Unknown nested configuration key '{key_path}'. "
            f"Section '{section_name}.{field_name}' does not contain nested keys."
        )

    if section_name in cfg_dict and not isinstance(cfg_dict[section_name], dict):
        raise TypeError(
            f"Section '{section_name}' must be a dictionary, got {type(cfg_dict[section_name]).__name__}"
        )
    if section_name not in cfg_dict:
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
        # Fallback to repo root if path is relative
        repo_candidate = (Path(__file__).resolve().parents[REPO_ROOT_PARENT_LEVEL] / path).resolve()
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
        yaml.dump(cfg.to_dict(), f, indent=YAML_INDENT, sort_keys=False)
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
        CLI_CONFIG_FLAG,
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        CLI_SET_FLAG,
        action="append",
        dest="overrides",
        default=None,
        help="Override config value formatted as 'section.key=val' (can be repeated)",
    )

    parsed_args = parser.parse_args(args)
    cfg = load_config(
        config_path=parsed_args.config,
        overrides=parsed_args.overrides,
    )
    return cfg, parsed_args
