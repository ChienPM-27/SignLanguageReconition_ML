"""Tests for environment and path abstraction (R2).

Covers environment detection mocking (Kaggle, Colab, Local) and path resolution.
"""

from pathlib import Path
import sys
import pytest

from src.utils.config import Config, load_config
from src.utils.env import (
    ENV_COLAB,
    ENV_KAGGLE,
    ENV_LOCAL,
    ResolvedPaths,
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
    """Verify Kaggle paths map directly to /kaggle/input and /kaggle/working."""
    cfg = load_config()
    paths = resolve_paths(ENV_KAGGLE, cfg, create_dirs=False)

    assert paths.env_name == ENV_KAGGLE
    assert paths.repo_root == Path("/kaggle/working")
    assert paths.data_dir == Path("/kaggle/input") / cfg.data.competition_id
    assert paths.output_dir == Path("/kaggle/working") / cfg.output.output_dir
    assert paths.report_dir == Path("/kaggle/working") / cfg.output.report_dir
    assert paths.is_kaggle_input_readonly is True


def test_resolve_paths_colab_without_drive() -> None:
    """Verify standard Colab paths map under /content."""
    cfg = load_config(overrides=["env.use_drive=false"])
    paths = resolve_paths(ENV_COLAB, cfg, create_dirs=False)

    assert paths.env_name == ENV_COLAB
    assert paths.repo_root == Path("/content")
    assert paths.data_dir == Path("/content/data")
    assert paths.output_dir == Path("/content") / cfg.output.output_dir
    assert paths.is_kaggle_input_readonly is False


def test_resolve_paths_colab_with_drive() -> None:
    """Verify persistent Colab paths map under /content/drive/MyDrive/<project>."""
    cfg = load_config(overrides=["env.use_drive=true", "env.drive_project_path=my_test_proj"])
    paths = resolve_paths(ENV_COLAB, cfg, create_dirs=False)

    expected_persistent_root = Path("/content/drive/MyDrive/my_test_proj")
    assert paths.env_name == ENV_COLAB
    assert paths.repo_root == expected_persistent_root
    assert paths.data_dir == expected_persistent_root / "data"
    assert paths.output_dir == expected_persistent_root / cfg.output.output_dir


def test_resolve_paths_local(tmp_path: Path) -> None:
    """Verify local paths respect configured root and local mirror directories."""
    mirror_name = "custom_mirror"
    cfg = load_config(
        overrides=[
            f"env.local_root_dir={tmp_path}",
            f"data.local_mirror_dir={mirror_name}",
        ]
    )
    paths = resolve_paths(ENV_LOCAL, cfg, create_dirs=True)

    assert paths.env_name == ENV_LOCAL
    assert paths.repo_root == tmp_path.resolve()
    assert paths.data_dir == (tmp_path / mirror_name).resolve()
    assert paths.is_kaggle_input_readonly is False
    assert paths.output_dir.is_dir()
    assert paths.report_dir.is_dir()


def test_resolve_paths_invalid_environment() -> None:
    """Verify unknown environment name raises ValueError."""
    cfg = load_config()
    with pytest.raises(ValueError, match="Invalid environment 'unknown_os'"):
        resolve_paths("unknown_os", cfg, create_dirs=False)


def test_resolved_paths_to_dict() -> None:
    """Verify ResolvedPaths serialization to dict."""
    cfg = load_config()
    paths = resolve_paths(ENV_LOCAL, cfg, create_dirs=False)
    paths_dict = paths.to_dict()
    assert "repo_root" in paths_dict
    assert "data_dir" in paths_dict
    assert paths_dict["env_name"] == ENV_LOCAL


def test_mount_colab_drive_already_mounted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _mount_colab_drive_if_needed returns True if MyDrive is already accessible."""
    import os
    from src.utils import env
    monkeypatch.setattr(os.path, "exists", lambda p: p == env.COLAB_DRIVE_MYDRIVE_PREFIX)
    assert env._mount_colab_drive_if_needed() is True


def test_mount_colab_drive_mount_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _mount_colab_drive_if_needed invokes google.colab.drive.mount when unmounted."""
    import os
    from src.utils import env
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    monkeypatch.setattr(os.path, "ismount", lambda p: False)

    mount_called = False
    class MockDrive:
        @staticmethod
        def mount(mount_point: str) -> None:
            nonlocal mount_called
            mount_called = True

    mock_colab = type(sys)("google.colab")
    mock_colab.drive = MockDrive()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google.colab", mock_colab)
    monkeypatch.setitem(sys.modules, "google.colab.drive", MockDrive())

    assert env._mount_colab_drive_if_needed() is True
    assert mount_called is True


def test_mount_colab_drive_timeout_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _mount_colab_drive_if_needed raises RuntimeError when drive.mount throws or times out."""
    import os
    from src.utils import env
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    monkeypatch.setattr(os.path, "ismount", lambda p: False)

    class FailingDrive:
        @staticmethod
        def mount(mount_point: str) -> None:
            raise TimeoutError("Drive mount timed out waiting for authorization")

    mock_colab = type(sys)("google.colab")
    mock_colab.drive = FailingDrive()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google.colab", mock_colab)
    monkeypatch.setitem(sys.modules, "google.colab.drive", FailingDrive())

    with pytest.raises(RuntimeError, match="Google Drive mounting failed or timed out"):
        env._mount_colab_drive_if_needed()


def test_resolve_paths_colab_drive_project_path_leading_slash() -> None:
    """Verify drive_project_path with leading slash does not discard /content/drive/MyDrive prefix."""
    cfg = load_config(overrides=["env.use_drive=true", "env.drive_project_path=/asl-signs"])
    paths = resolve_paths(ENV_COLAB, cfg, create_dirs=False)
    expected_persistent_root = Path("/content/drive/MyDrive/asl-signs")
    assert paths.repo_root == expected_persistent_root
    assert paths.data_dir == expected_persistent_root / "data"
