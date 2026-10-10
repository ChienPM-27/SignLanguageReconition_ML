"""Environment and path resolution abstraction for ISLR Phase 0.

Detects runtime platform (local, colab, kaggle) and resolves canonical paths
for data, cache, output, reports, selection, repo root, and persistent root.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.config import Config

ENV_LOCAL = "local"
ENV_COLAB = "colab"
ENV_KAGGLE = "kaggle"
ENV_AUTO = "auto"
VALID_ENV_NAMES = frozenset({ENV_LOCAL, ENV_COLAB, ENV_KAGGLE})

KAGGLE_INPUT_DIR = "/kaggle/input"
KAGGLE_WORKING_DIR = "/kaggle/working"
KAGGLE_TEMP_DIR = "/kaggle/temp"

COLAB_ROOT_DIR = "/content"
COLAB_DRIVE_ROOT = "/content/drive"
COLAB_DRIVE_MYDRIVE_PREFIX = "/content/drive/MyDrive"

DEFAULT_DATA_DIR_NAME = "data"


@dataclass(frozen=True)
class ResolvedPaths:
    """Canonical resolved paths across runtime environments."""
    env_name: str
    repo_root: Path
    persistent_root: Path
    data_dir: Path
    cache_dir: Path
    output_dir: Path
    report_dir: Path
    selection_dir: Path
    is_kaggle_input_readonly: bool = False

    def to_dict(self) -> dict:
        return {
            "env_name": self.env_name,
            "repo_root": str(self.repo_root),
            "persistent_root": str(self.persistent_root),
            "data_dir": str(self.data_dir),
            "cache_dir": str(self.cache_dir),
            "output_dir": str(self.output_dir),
            "report_dir": str(self.report_dir),
            "selection_dir": str(self.selection_dir),
            "is_kaggle_input_readonly": self.is_kaggle_input_readonly,
        }


def detect_environment() -> str:
    """Detect current runtime execution environment.

    Returns:
        str: 'kaggle', 'colab', or 'local'.
    """
    # 1. Kaggle environment detection
    if (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or os.environ.get("KAGGLE_URL_BASE") is not None
        or os.path.exists(KAGGLE_WORKING_DIR)
    ):
        return ENV_KAGGLE

    # 2. Google Colab environment detection
    if (
        "google.colab" in sys.modules
        or os.environ.get("COLAB_GPU") is not None
        or os.environ.get("GCS_READ_CACHE") is not None
    ):
        return ENV_COLAB

    try:
        importlib.import_module("google.colab")
        return ENV_COLAB
    except (ImportError, ModuleNotFoundError):
        pass

    # 3. Default fallback to local environment
    return ENV_LOCAL


def _mount_colab_drive_if_needed(mount_point: str = COLAB_DRIVE_ROOT) -> bool:
    """Mount Google Drive in Colab if not already mounted.

    Returns:
        bool: True if Drive is already mounted or successfully mounted, False if not in Colab.

    Raises:
        RuntimeError: If Google Drive mount fails or times out in Google Colab.
    """
    if os.path.exists(COLAB_DRIVE_MYDRIVE_PREFIX) or os.path.ismount(mount_point):
        return True
    try:
        from google.colab import drive
    except ImportError:
        return False

    try:
        drive.mount(mount_point)
        return True
    except Exception as exc:
        raise RuntimeError(
            f"Google Drive mounting failed or timed out at '{mount_point}': {exc}. "
            "Please grant Google Drive permissions in Colab or set env.use_drive: false."
        ) from exc


def resolve_paths(
    env_name: str,
    cfg: Config,
    create_dirs: bool = True,
) -> ResolvedPaths:
    """Resolve canonical filesystem paths based on environment and configuration.

    Separates repo_root (the cloned codebase location) from persistent_root
    (where outputs and caches live across disconnections).

    Args:
        env_name: Environment identifier ('auto', 'local', 'colab', 'kaggle').
        cfg: Validated Config dataclass instance.
        create_dirs: Whether to create output/cache directories on disk.

    Returns:
        ResolvedPaths: Immutable container of canonical paths.

    Raises:
        ValueError: If specified env_name is invalid.
    """
    effective_env = detect_environment() if env_name == ENV_AUTO else env_name.lower()
    if effective_env not in VALID_ENV_NAMES:
        raise ValueError(
            f"Invalid environment '{effective_env}'. Must be one of: {sorted(VALID_ENV_NAMES)}"
        )

    # Detect cloned repo root via location of this package, never cwd, never Drive
    package_repo_root = Path(__file__).resolve().parent.parent.parent

    if effective_env == ENV_KAGGLE:
        repo_root = package_repo_root
        persistent_root = Path(KAGGLE_WORKING_DIR)
        comp_slug = getattr(cfg.data, "competition_slug", None) or getattr(cfg.data, "competition_id", "asl-signs")
        data_dir = Path(KAGGLE_INPUT_DIR) / comp_slug
        cache_dir = Path(KAGGLE_TEMP_DIR)
        output_dir = persistent_root / cfg.output.output_dir
        report_dir = persistent_root / getattr(cfg.output, "report_dir", getattr(cfg.output, "reports_dir", "reports"))
        selection_dir = persistent_root / getattr(cfg.selection, "selection_dir", getattr(cfg.selection, "output_dir", "data/selection"))
        is_readonly = True

    elif effective_env == ENV_COLAB:
        repo_root = package_repo_root
        proj_name = getattr(cfg.env, "project_dir_name", None) or getattr(cfg.env, "drive_project_path", "asl-signs")
        clean_proj_name = proj_name.strip().strip("/\\")

        if cfg.env.use_drive:
            if create_dirs:
                _mount_colab_drive_if_needed()
            persistent_root = Path(COLAB_DRIVE_MYDRIVE_PREFIX) / clean_proj_name
            data_dir = persistent_root / DEFAULT_DATA_DIR_NAME
            cache_dir = persistent_root / "cache"
            output_dir = persistent_root / cfg.output.output_dir
            report_dir = persistent_root / getattr(cfg.output, "report_dir", getattr(cfg.output, "reports_dir", "reports"))
            selection_dir = persistent_root / getattr(cfg.selection, "selection_dir", getattr(cfg.selection, "output_dir", "data/selection"))
        else:
            persistent_root = Path(COLAB_ROOT_DIR)
            data_dir = persistent_root / DEFAULT_DATA_DIR_NAME
            cache_dir = persistent_root / "cache"
            output_dir = persistent_root / cfg.output.output_dir
            report_dir = persistent_root / getattr(cfg.output, "report_dir", getattr(cfg.output, "reports_dir", "reports"))
            selection_dir = persistent_root / getattr(cfg.selection, "selection_dir", getattr(cfg.selection, "output_dir", "data/selection"))
        is_readonly = False

    else:  # local
        repo_root = package_repo_root
        conf_local_root = getattr(cfg.env, "local_root", None) or getattr(cfg.env, "local_root_dir", ".")
        local_path = Path(conf_local_root)
        persistent_root = local_path.resolve() if local_path.is_absolute() else (repo_root / local_path).resolve()

        if cfg.data.local_mirror_dir:
            mirror_path = Path(cfg.data.local_mirror_dir)
            data_dir = mirror_path.resolve() if mirror_path.is_absolute() else (persistent_root / mirror_path).resolve()
        else:
            data_dir = (persistent_root / DEFAULT_DATA_DIR_NAME).resolve()

        cache_dir = (persistent_root / getattr(cfg.data, "cache_dir", "data/cache")).resolve()
        output_dir = (persistent_root / cfg.output.output_dir).resolve()
        report_dir = (persistent_root / getattr(cfg.output, "report_dir", getattr(cfg.output, "reports_dir", "reports"))).resolve()
        selection_dir = (persistent_root / getattr(cfg.selection, "selection_dir", getattr(cfg.selection, "output_dir", "data/selection"))).resolve()
        is_readonly = False

    if create_dirs:
        for p in (output_dir, report_dir, cache_dir, selection_dir):
            try:
                p.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError):
                pass
        if not is_readonly:
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError):
                pass

    return ResolvedPaths(
        env_name=effective_env,
        repo_root=repo_root,
        persistent_root=persistent_root,
        data_dir=data_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        report_dir=report_dir,
        selection_dir=selection_dir,
        is_kaggle_input_readonly=is_readonly,
    )
