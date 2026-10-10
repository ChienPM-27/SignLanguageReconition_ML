"""Tests for typed configuration system (R1).

Covers loading, typing, unknown key error, range validation, CLI overrides, and serialization.
"""

from pathlib import Path
import pytest
import yaml

from src.utils.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    load_config,
    parse_cli_args,
    save_config,
)

SAMPLE_SEED_OVERRIDE = 999
INVALID_NEGATIVE_SEED = -1
SAMPLE_TOTAL_BUDGET = 10
INVALID_ZERO_BUDGET = 0
EXPECTED_COLUMNS_COUNT = 7


def test_load_default_config() -> None:
    """Verify default islr.yaml loads with proper types and sections."""
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(cfg, Config)
    assert cfg.env.name in {"auto", "local", "colab", "kaggle"}
    assert isinstance(cfg.env.seed, int)
    assert isinstance(cfg.env.use_drive, bool)
    assert isinstance(cfg.data.competition_id, str)
    assert cfg.data.competition_id == "asl-signs"
    assert len(cfg.eda.expected_columns) == EXPECTED_COLUMNS_COUNT
    assert cfg.eda.num_expected_landmarks == 543


def test_unknown_top_level_section(tmp_path: Path) -> None:
    """Verify unknown top-level section in YAML raises KeyError."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("invalid_section:\n  foo: bar\n", encoding="utf-8")
    with pytest.raises(KeyError, match="Unknown configuration section 'invalid_section'"):
        load_config(bad_yaml)


def test_unknown_field_in_section(tmp_path: Path) -> None:
    """Verify unknown key inside a known section raises KeyError."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("env:\n  unexpected_key: 123\n", encoding="utf-8")
    with pytest.raises(KeyError, match="Unknown configuration key 'unexpected_key' in section 'env'"):
        load_config(bad_yaml)


def test_unknown_key_in_override() -> None:
    """Verify unknown section and key in --set override raises KeyError."""
    with pytest.raises(KeyError, match="Unknown configuration section 'bogus'"):
        load_config(overrides=["bogus.field=10"])

    with pytest.raises(KeyError, match="Unknown configuration key 'nonexistent'"):
        load_config(overrides=["env.nonexistent=10"])


def test_invalid_override_format() -> None:
    """Verify malformed override string raises ValueError."""
    with pytest.raises(ValueError, match="Expected 'key.path=val'"):
        load_config(overrides=["no_equals_sign"])


def test_range_validation_seed() -> None:
    """Verify negative seed raises ValueError."""
    with pytest.raises(ValueError, match="env.seed must be >= 0"):
        load_config(overrides=[f"env.seed={INVALID_NEGATIVE_SEED}"])


def test_range_validation_sample_budget() -> None:
    """Verify sample budget < 1 raises ValueError."""
    with pytest.raises(ValueError, match="data.sample_budget_total_files must be >= 1"):
        load_config(overrides=[f"data.sample_budget_total_files={INVALID_ZERO_BUDGET}"])


def test_invalid_env_name() -> None:
    """Verify unsupported environment name raises ValueError."""
    with pytest.raises(ValueError, match="Invalid env.name"):
        load_config(overrides=["env.name=invalid_os"])


def test_invalid_log_level() -> None:
    """Verify unsupported log level raises ValueError."""
    with pytest.raises(ValueError, match="Invalid env.log_level"):
        load_config(overrides=["env.log_level=VERBOSE"])


def test_cli_overrides_application() -> None:
    """Verify multiple CLI --set overrides are correctly typed and applied."""
    cfg = load_config(
        overrides=[
            f"env.seed={SAMPLE_SEED_OVERRIDE}",
            "env.log_level=DEBUG",
            f"data.sample_budget_total_files={SAMPLE_TOTAL_BUDGET}",
            "env.use_drive=true",
        ]
    )
    assert cfg.env.seed == SAMPLE_SEED_OVERRIDE
    assert cfg.env.log_level == "DEBUG"
    assert cfg.data.sample_budget_total_files == SAMPLE_TOTAL_BUDGET
    assert cfg.env.use_drive is True


def test_parse_cli_args() -> None:
    """Verify parse_cli_args correctly parses command-line arguments."""
    cli_args = [
        "--config",
        DEFAULT_CONFIG_PATH,
        "--set",
        f"env.seed={SAMPLE_SEED_OVERRIDE}",
    ]
    cfg, parsed = parse_cli_args(cli_args)
    assert parsed.config == DEFAULT_CONFIG_PATH
    assert cfg.env.seed == SAMPLE_SEED_OVERRIDE


def test_save_and_reload_config(tmp_path: Path) -> None:
    """Verify saving configuration to YAML and reloading produces identical content."""
    cfg = load_config(DEFAULT_CONFIG_PATH)
    target_file = tmp_path / "out_config.yaml"
    save_config(cfg, target_file)

    assert target_file.is_file()
    with open(target_file, "r", encoding="utf-8") as f:
        reloaded_dict = yaml.safe_load(f)

    assert reloaded_dict["data"]["competition_id"] == "asl-signs"
    assert reloaded_dict["env"]["seed"] == cfg.env.seed


def test_boundary_values_seed_zero() -> None:
    """Verify seed=0 is valid (non-negative lower bound)."""
    cfg = load_config(overrides=["env.seed=0"])
    assert cfg.env.seed == 0


def test_null_local_mirror_dir() -> None:
    """Verify data.local_mirror_dir can be explicitly set to null/None."""
    cfg = load_config(overrides=["data.local_mirror_dir=null"])
    assert cfg.data.local_mirror_dir is None


def test_empty_expected_columns_raises_error(tmp_path: Path) -> None:
    """Verify empty expected_columns raises ValueError."""
    bad_yaml = tmp_path / "empty_cols.yaml"
    bad_yaml.write_text("eda:\n  expected_columns: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="eda.expected_columns must be a non-empty list"):
        load_config(bad_yaml)


def test_empty_key_path_override() -> None:
    """Verify empty key path in override raises ValueError."""
    with pytest.raises(ValueError, match="Empty key path"):
        load_config(overrides=["=123"])


def test_nested_dict_override_works() -> None:
    """Verify nested override on dict field works, e.g. selection.weights.separability=2.0."""
    cfg = load_config(overrides=["selection.weights.separability=2.0"])
    assert cfg.selection.weights["separability"] == 2.0


def test_list_override_works() -> None:
    """Verify list override works, e.g. selection.exclude_signs=[a,b]."""
    cfg = load_config(overrides=["selection.exclude_signs=['sign_a', 'sign_b']"])
    assert cfg.selection.exclude_signs == ["sign_a", "sign_b"]


def test_unknown_inner_dict_key_raises() -> None:
    """Verify unknown inner key in a dict field raises KeyError."""
    with pytest.raises(KeyError, match="Unknown weight key 'nonexistent' in selection.weights"):
        load_config(overrides=["selection.weights.nonexistent=2.0"])


def test_override_non_dict_field_with_extra_tokens_raises() -> None:
    """Verify overriding non-dict field with extra tokens raises TypeError."""
    with pytest.raises(TypeError, match="Cannot access nested key on non-dict field 'env.seed'"):
        load_config(overrides=["env.seed.extra=10"])


def test_seed_boolean_type_rejection() -> None:
    """Verify boolean value for seed raises TypeError instead of being coerced to int."""
    with pytest.raises(TypeError, match="env.seed must be an integer, got bool"):
        load_config(overrides=["env.seed=true"])


def test_probe_files_exceeding_budget_raises_value_error() -> None:
    """Verify eda.num_landmark_probe_files > data.sample_budget_total_files raises ValueError."""
    with pytest.raises(ValueError, match="cannot exceed data.sample_budget_total_files"):
        load_config(
            overrides=[
                "eda.num_landmark_probe_files=10",
                "data.sample_budget_total_files=5",
            ]
        )


def test_local_mirror_dir_whitespace_raises_value_error() -> None:
    """Verify whitespace-only local_mirror_dir raises ValueError."""
    with pytest.raises(ValueError, match="data.local_mirror_dir must be None or a non-empty string"):
        load_config(overrides=["data.local_mirror_dir='   '"])


def test_missing_required_landmark_column_raises_error() -> None:
    """Verify omitting required competition columns from expected_columns raises ValueError."""
    with pytest.raises(ValueError, match="eda.expected_columns missing required competition columns"):
        load_config(overrides=["eda.expected_columns=['frame', 'row_id']"])


def test_invalid_yaml_syntax_in_override_raises_value_error() -> None:
    """Verify malformed YAML syntax in CLI override raises ValueError."""
    with pytest.raises(ValueError, match="Failed to parse override value"):
        load_config(overrides=["env.seed={unclosed"])


def test_invalid_yaml_syntax_in_file_raises_value_error(tmp_path: Path) -> None:
    """Verify malformed YAML syntax in config file raises ValueError."""
    corrupt_yaml = tmp_path / "corrupt.yaml"
    corrupt_yaml.write_text("env:\n  seed: {unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse YAML configuration"):
        load_config(corrupt_yaml)


def test_expected_columns_duplicate_raises_value_error() -> None:
    """Verify duplicate column names in eda.expected_columns raises ValueError."""
    with pytest.raises(ValueError, match="eda.expected_columns must not contain duplicate column names"):
        load_config(
            overrides=[
                "eda.expected_columns=['frame', 'frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z']"
            ]
        )


def test_override_on_non_dict_section_raises_type_error(tmp_path: Path) -> None:
    """Verify applying override to non-dictionary section in YAML raises TypeError."""
    bad_yaml = tmp_path / "scalar_section.yaml"
    bad_yaml.write_text("env: scalar_value_instead_of_dict\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Section 'env' must be a dictionary"):
        load_config(bad_yaml, overrides=["env.seed=10"])


@pytest.mark.parametrize(
    "removed_key",
    [
        "env.drive_project_path",
        "env.local_root_dir",
        "data.competition_slug",
        "data.train_csv_remote",
        "output.reports_dir",
        "selection.selection_dir",
        "selection.output_dir",
    ],
)
def test_removed_config_keys_raise_clear_unknown_key_error(removed_key: str) -> None:
    """Verify any config override or YAML specifying a removed alias raises KeyError."""
    with pytest.raises(KeyError, match="Unknown configuration key"):
        load_config(overrides=[f"{removed_key}=dummy_val"])


def test_every_kept_key_override_updates_code_values() -> None:
    """Verify overriding each single kept key changes the exact value used across modules.

    competition_id is used by BOTH env paths and kaggle_access.
    selection_dir drives resolved paths.
    """
    from src.data import kaggle_access
    from src.utils.env import ENV_KAGGLE, resolve_paths

    # Verify default competition_id matches DEFAULT_COMPETITION_ID
    assert load_config().data.competition_id == kaggle_access.DEFAULT_COMPETITION_ID

    cfg = load_config(
        overrides=[
            "env.project_dir_name=custom_proj_name",
            "env.local_root=custom_local_root",
            "data.competition_id=custom_comp_id",
            "data.train_csv_name=custom_train.csv",
            "output.report_dir=custom_reports",
            "output.selection_dir=custom_selection",
        ]
    )

    assert cfg.env.project_dir_name == "custom_proj_name"
    assert cfg.env.local_root == "custom_local_root"
    assert cfg.data.competition_id == "custom_comp_id"
    assert cfg.data.train_csv_name == "custom_train.csv"
    assert cfg.output.report_dir == "custom_reports"
    assert cfg.output.selection_dir == "custom_selection"

    # Verify competition_id drives env paths
    paths_kaggle = resolve_paths(ENV_KAGGLE, cfg, create_dirs=False)
    assert paths_kaggle.data_dir == Path("/kaggle/input/custom_comp_id")

    # Verify selection_dir drives resolved paths
    assert paths_kaggle.selection_dir == Path("/kaggle/working/custom_selection")

    # Verify report_dir drives resolved paths
    assert paths_kaggle.report_dir == Path("/kaggle/working/custom_reports")


