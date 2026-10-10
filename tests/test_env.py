"""Tests for environment and path abstraction (R2).

Covers environment detection mocking (Kaggle, Colab, Local) and path resolution
with separation of repo_root and persistent_root.
"""

from pathlib import Path
import sys
import pytest

from src.utils.config import load_config
from src.utils.env import (
    ENV_COLAB,
    ENV_KAGGLE,
    ENV_LOCAL,
    detect_environment,
    resolve_paths,
)


def test_detect_kaggle_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Kaggle environment is detected via environment variables."""
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert detect_environment() == ENV_KAGGLE


def test_detect_colab_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Colab environment is detected via environment variables and modules."""
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delenv("KAGGLE_URL_BASE", raising=False)
    monkeypatch.setenv("COLAB_GPU", "1")
    assert detect_environment() == ENV_COLAB


def test_detect_local_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify local environment fallback when neither Kaggle nor Colab is present."""
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delenv("KAGGLE_URL_BASE", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    monkeypatch.delenv("GCS_READ_CACHE", raising=False)
    # Remove google.colab from sys.modules if present
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)

    env = detect_environment()
    assert env == ENV_LOCAL


def test_resolve_paths_kaggle() -> None:
    """Verify Kaggle paths map directly to /kaggle/input and persistent /kaggle/working."""
    cfg = load_config()
    paths = resolve_paths(ENV_KAGGLE, cfg, create_dirs=False)

    assert paths.env_name == ENV_KAGGLE
    assert paths.persistent_root == Path("/kaggle/working")
    assert paths.data_dir == Path("/kaggle/input") / cfg.data.competition_id
    assert paths.cache_dir == Path("/kaggle/temp") / cfg.data.cache_dir
    assert paths.output_dir == Path("/kaggle/working") / cfg.output.output_dir
    assert paths.report_dir == Path("/kaggle/working") / cfg.output.report_dir
    assert paths.selection_dir == Path("/kaggle/working") / cfg.output.selection_dir
    assert paths.is_kaggle_input_readonly is True


def test_resolve_paths_colab_without_drive() -> None:
    """Verify standard Colab paths map under /content and separate repo_root."""
    cfg = load_config(overrides=["env.use_drive=false"])
    paths = resolve_paths(ENV_COLAB, cfg, create_dirs=False)

    expected_repo_root = Path(__file__).resolve().parent.parent
    assert paths.env_name == ENV_COLAB
    assert paths.repo_root == expected_repo_root
    assert paths.persistent_root == Path("/content")
    assert paths.data_dir == Path("/content/data")
    assert paths.cache_dir == Path("/content") / cfg.data.cache_dir
    assert paths.output_dir == Path("/content") / cfg.output.output_dir
    assert paths.report_dir == Path("/content") / cfg.output.report_dir
    assert paths.selection_dir == Path("/content") / cfg.output.selection_dir
    assert paths.is_kaggle_input_readonly is False


def test_resolve_paths_colab_with_drive() -> None:
    """Verify persistent Colab paths map under /content/drive/MyDrive/<project>."""
    cfg = load_config(overrides=["env.use_drive=true", "env.project_dir_name=my_test_proj"])
    paths = resolve_paths(ENV_COLAB, cfg, create_dirs=False)

    expected_repo_root = Path(__file__).resolve().parent.parent
    expected_persistent_root = Path("/content/drive/MyDrive/my_test_proj")
    assert paths.env_name == ENV_COLAB
    assert paths.repo_root == expected_repo_root
    assert paths.persistent_root == expected_persistent_root
    assert paths.data_dir == expected_persistent_root / "data"
    assert paths.cache_dir == expected_persistent_root / cfg.data.cache_dir
    assert paths.output_dir == expected_persistent_root / cfg.output.output_dir
    assert paths.report_dir == expected_persistent_root / cfg.output.report_dir
    assert paths.selection_dir == expected_persistent_root / cfg.output.selection_dir


def test_resolve_paths_local(tmp_path: Path) -> None:
    """Verify local paths respect configured root and local mirror directories."""
    mirror_name = "custom_mirror"
    cfg = load_config(
        overrides=[
            f"env.local_root={tmp_path}",
            f"data.local_mirror_dir={mirror_name}",
        ]
    )
    paths = resolve_paths(ENV_LOCAL, cfg, create_dirs=True)

    expected_repo_root = Path(__file__).resolve().parent.parent
    assert paths.env_name == ENV_LOCAL
    assert paths.repo_root == expected_repo_root
    assert paths.persistent_root == tmp_path.resolve()
    assert paths.data_dir == (tmp_path / mirror_name).resolve()
    assert paths.cache_dir == (expected_repo_root / cfg.data.cache_dir).resolve()
    assert paths.is_kaggle_input_readonly is False
    assert paths.output_dir.is_dir()
    assert paths.report_dir.is_dir()
    assert paths.selection_dir.is_dir()


def test_resolve_paths_cache_dir_absolute_and_relative() -> None:
    """Verify cache_dir resolution for relative and absolute paths in all environments."""
    expected_repo_root = Path(__file__).resolve().parent.parent

    # 1. Relative cache_dir across envs
    cfg_rel = load_config(overrides=["data.cache_dir=custom/cache"])
    paths_kaggle = resolve_paths(ENV_KAGGLE, cfg_rel, create_dirs=False)
    assert paths_kaggle.cache_dir == Path("/kaggle/temp/custom/cache")

    paths_colab_no_drive = resolve_paths(ENV_COLAB, cfg_rel, create_dirs=False)
    assert paths_colab_no_drive.cache_dir == Path("/content/custom/cache")

    cfg_drive = load_config(overrides=["env.use_drive=true", "env.project_dir_name=proj", "data.cache_dir=custom/cache"])
    paths_colab_drive = resolve_paths(ENV_COLAB, cfg_drive, create_dirs=False)
    assert paths_colab_drive.cache_dir == Path("/content/drive/MyDrive/proj/custom/cache")

    paths_local = resolve_paths(ENV_LOCAL, cfg_rel, create_dirs=False)
    assert paths_local.cache_dir == (expected_repo_root / "custom/cache").resolve()

    # 2. Absolute cache_dir across envs
    abs_cache = "/tmp/global_cache" if sys.platform != "win32" else "C:/temp/global_cache"
    cfg_abs = load_config(overrides=[f"data.cache_dir={abs_cache}"])

    paths_kaggle_abs = resolve_paths(ENV_KAGGLE, cfg_abs, create_dirs=False)
    assert paths_kaggle_abs.cache_dir == Path(abs_cache)

    paths_colab_no_drive_abs = resolve_paths(ENV_COLAB, cfg_abs, create_dirs=False)
    assert paths_colab_no_drive_abs.cache_dir == Path(abs_cache)

    paths_colab_drive_abs = resolve_paths(ENV_COLAB, cfg_abs, create_dirs=False)
    assert paths_colab_drive_abs.cache_dir == Path(abs_cache)

    paths_local_abs = resolve_paths(ENV_LOCAL, cfg_abs, create_dirs=False)
    assert paths_local_abs.cache_dir == Path(abs_cache).resolve()


def test_resolve_paths_invalid_environment() -> None:
    """Verify unknown environment name raises ValueError."""
    cfg = load_config()
    with pytest.raises(ValueError, match="Invalid environment 'unknown_os'"):
        resolve_paths("unknown_os", cfg, create_dirs=False)


def test_resolved_paths_to_dict() -> None:
    """Verify ResolvedPaths serialization to dict includes persistent_root."""
    cfg = load_config()
    paths = resolve_paths(ENV_LOCAL, cfg, create_dirs=False)
    paths_dict = paths.to_dict()
    assert "repo_root" in paths_dict
    assert "persistent_root" in paths_dict
    assert "data_dir" in paths_dict
    assert "selection_dir" in paths_dict
    assert paths_dict["env_name"] == ENV_LOCAL
