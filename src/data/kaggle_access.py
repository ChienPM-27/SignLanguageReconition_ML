"""Kaggle access abstraction and robust file downloader for ISLR Phase 0.

Provides multi-source credential resolution with strict precedence, HTTP 403 handling,
automatic zip extraction, and normalization of nested paths against Kaggle flattening.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Optional, Tuple, Union
import zipfile

ENV_KAGGLE_USERNAME = "KAGGLE_USERNAME"
ENV_KAGGLE_KEY = "KAGGLE_KEY"

SOURCE_COLAB_SECRETS = "colab_secrets"
SOURCE_ENV_VARS = "env_vars"
SOURCE_KAGGLE_JSON = "kaggle_json"
SOURCE_INTERACTIVE_PROMPT = "interactive_prompt"

STATUS_PRESENT = "present"
STATUS_ABSENT = "absent"

HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_FORBIDDEN = 403
ZIP_EXTENSION = ".zip"
EMPTY_FILE_SIZE_BYTES = 0

DEFAULT_LOGGER_NAME = "kaggle_access"
DEFAULT_COMPETITION_ID = "asl-signs"
KAGGLE_RULES_URL_TEMPLATE = "https://www.kaggle.com/competitions/{competition_id}/rules"
KAGGLE_CONFIG_DIR = ".kaggle"
KAGGLE_JSON_FILENAME = "kaggle.json"


class KaggleAccessError(Exception):
    """Base error for Kaggle access and download operations."""


class KaggleCredentialsNotFoundError(KaggleAccessError):
    """Raised when Kaggle credentials cannot be discovered."""


class KaggleRulesNotAcceptedError(KaggleAccessError):
    """Raised when Kaggle returns HTTP 403 due to unaccepted competition rules."""


def get_kaggle_credentials(allow_interactive: bool = True) -> Optional[Tuple[str, str, str]]:
    """Discover Kaggle credentials following strict fallback precedence.

    Precedence:
        1. Google Colab secrets (google.colab.userdata)
        2. Environment variables (KAGGLE_USERNAME, KAGGLE_KEY)
        3. Local configuration file (~/.kaggle/kaggle.json)
        4. Interactive terminal prompt (if stdin is a TTY and allow_interactive is True)

    Args:
        allow_interactive: Whether to prompt interactively if other sources are absent.

    Returns:
        Optional[Tuple[str, str, str]]: (username, key, source) or None if absent.
    """
    # 1. Colab user secrets
    try:
        from google.colab import userdata
        username = userdata.get(ENV_KAGGLE_USERNAME)
        key = userdata.get(ENV_KAGGLE_KEY)
        u_colab = str(username).strip() if username else ""
        k_colab = str(key).strip() if key else ""
        if u_colab and k_colab:
            return u_colab, k_colab, SOURCE_COLAB_SECRETS
    except (ImportError, Exception):
        pass

    # 2. Environment variables
    username_env = (os.environ.get(ENV_KAGGLE_USERNAME) or "").strip()
    key_env = (os.environ.get(ENV_KAGGLE_KEY) or "").strip()
    if username_env and key_env:
        return username_env, key_env, SOURCE_ENV_VARS

    # 3. Local configuration file (~/.kaggle/kaggle.json or $KAGGLE_CONFIG_DIR/kaggle.json)
    config_dir_env = os.environ.get("KAGGLE_CONFIG_DIR")
    if config_dir_env:
        kaggle_json_path = Path(config_dir_env) / KAGGLE_JSON_FILENAME
    else:
        kaggle_json_path = Path.home() / KAGGLE_CONFIG_DIR / KAGGLE_JSON_FILENAME

    if kaggle_json_path.is_file():
        try:
            with open(kaggle_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                u = str(data.get("username") or "").strip()
                k = str(data.get("key") or "").strip()
                if u and k:
                    return u, k, SOURCE_KAGGLE_JSON
        except (json.JSONDecodeError, OSError, AttributeError):
            pass

    # 4. Interactive prompt fallback (last resort, only when allowed and interactive TTY available)
    if allow_interactive and sys.stdin.isatty():
        try:
            import getpass
            sys.stdout.write("Enter Kaggle Username: ")
            sys.stdout.flush()
            user_input = sys.stdin.readline().strip()
            if not user_input:
                return None
            key_input = getpass.getpass("Enter Kaggle API Key: ").strip()
            if key_input:
                return user_input, key_input, SOURCE_INTERACTIVE_PROMPT
        except Exception:
            pass

    return None


def get_credential_status() -> str:
    """Return presence status of Kaggle credentials without exposing sensitive keys.

    Inspects saved credentials non-interactively.

    Returns:
        str: 'present' or 'absent'.
    """
    creds = get_kaggle_credentials(allow_interactive=False)
    return STATUS_PRESENT if creds is not None else STATUS_ABSENT


def authenticate_kaggle_api(
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Authenticate and return KaggleApi instance using discovered credentials.

    Args:
        logger: Optional logger for diagnostic messages (credentials are never logged).

    Returns:
        KaggleApi: Authenticated API client instance.

    Raises:
        KaggleCredentialsNotFoundError: If credentials cannot be found.
    """
    creds = get_kaggle_credentials(allow_interactive=True)
    if creds is None:
        raise KaggleCredentialsNotFoundError(
            "Kaggle credentials not found. Configure KAGGLE_USERNAME and KAGGLE_KEY "
            "as environment variables, in Google Colab secrets, or in ~/.kaggle/kaggle.json."
        )

    username, key, source = creds
    os.environ[ENV_KAGGLE_USERNAME] = username
    os.environ[ENV_KAGGLE_KEY] = key

    if logger:
        logger.info("Authenticating Kaggle API via %s.", source)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as exc:
        err_msg = str(exc)
        if (
            str(HTTP_STATUS_FORBIDDEN) in err_msg
            or "forbidden" in err_msg.lower()
            or getattr(exc, "status", None) == HTTP_STATUS_FORBIDDEN
            or getattr(getattr(exc, "response", None), "status_code", None) == HTTP_STATUS_FORBIDDEN
        ):
            rules_url = KAGGLE_RULES_URL_TEMPLATE.format(competition_id=DEFAULT_COMPETITION_ID)
            raise KaggleRulesNotAcceptedError(
                f"HTTP 403 Forbidden during Kaggle authentication. Please verify access at: {rules_url}"
            ) from exc
        if (
            str(HTTP_STATUS_UNAUTHORIZED) in err_msg
            or "unauthorized" in err_msg.lower()
            or getattr(exc, "status", None) == HTTP_STATUS_UNAUTHORIZED
            or getattr(getattr(exc, "response", None), "status_code", None) == HTTP_STATUS_UNAUTHORIZED
        ):
            raise KaggleAccessError(
                f"HTTP 401 Unauthorized: Kaggle credentials are invalid or expired. "
                f"Please verify KAGGLE_USERNAME and KAGGLE_KEY. Original error: {exc}"
            ) from exc
        raise KaggleAccessError(f"Kaggle API authentication failed: {exc}") from exc


def fetch_file(
    remote_path: str,
    dest: Optional[Union[str, Path]] = None,
    competition_id: str = DEFAULT_COMPETITION_ID,
    local_mirror_dir: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None,
    dest_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Fetch a single competition file, checking local mirror and caching idempotently.

    Empirically handles nested remote paths regardless of Kaggle CLI flattening behavior.

    Args:
        remote_path: Path to remote file (e.g. 'train.csv' or 'train_landmark_files/26734/100.parquet').
        dest: Destination directory or file path.
        competition_id: Kaggle competition name (default: 'asl-signs').
        local_mirror_dir: Optional local mirror directory to bypass download if file exists.
        logger: Optional logger for diagnostic messages.
        dest_dir: Backward-compatible alias for dest.

    Returns:
        Path: Canonical local path to the fetched and uncompressed file.

    Raises:
        KaggleRulesNotAcceptedError: On HTTP 403 forbidden response.
        KaggleAccessError: On download, extraction, or filesystem error.
    """
    effective_dest = dest if dest is not None else dest_dir
    if effective_dest is None:
        raise ValueError("Either 'dest' or 'dest_dir' must be provided to fetch_file.")

    if not isinstance(remote_path, str) or not remote_path.strip():
        raise ValueError("remote_path must be a non-empty string.")

    normalized_remote_path = remote_path.replace("\\", "/").strip()
    remote_path_obj = Path(normalized_remote_path)
    flat_filename = remote_path_obj.name
    dest_resolved = Path(effective_dest).resolve()

    # If effective_dest points directly to a file matching remote_path's name, treat parent as dest directory
    if dest_resolved.is_file() or (not dest_resolved.is_dir() and dest_resolved.name == flat_filename):
        target_path = dest_resolved
        posix_dest = dest_resolved.as_posix()
        if posix_dest.endswith(normalized_remote_path):
            base_str = posix_dest[:-len(normalized_remote_path)].rstrip("/")
            dest_base = Path(base_str) if base_str else dest_resolved.parent
        else:
            dest_base = dest_resolved.parent
    else:
        dest_base = dest_resolved
        target_path = dest_base / remote_path_obj

    # 1. Check local mirror bypass if configured
    if local_mirror_dir:
        mirror_base = Path(local_mirror_dir).resolve()
        mirror_candidate = mirror_base / remote_path_obj
        if mirror_candidate.is_file() and mirror_candidate.stat().st_size > EMPTY_FILE_SIZE_BYTES:
            if logger:
                logger.info(
                    "Local mirror hit for '%s': bypassing download.",
                    normalized_remote_path,
                )
            return mirror_candidate

    # 2. Check idempotent cache at target path
    if target_path.is_file() and target_path.stat().st_size > EMPTY_FILE_SIZE_BYTES:
        if logger:
            logger.info("Cache hit for '%s': %s.", normalized_remote_path, target_path)
        return target_path

    # Clean up empty/corrupted zero-byte target if left by prior failed attempt
    if target_path.is_file() and target_path.stat().st_size == EMPTY_FILE_SIZE_BYTES:
        target_path.unlink(missing_ok=True)

    # 3. Check if file is already present in flattened location inside dest
    flat_candidate = dest_base / flat_filename
    if flat_candidate.is_file() and flat_candidate.stat().st_size > EMPTY_FILE_SIZE_BYTES:
        if flat_candidate.resolve() != target_path.resolve():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(flat_candidate), str(target_path))
            if logger:
                logger.info("Relocated flattened file '%s' to '%s'.", flat_candidate, target_path)
        return target_path

    if flat_candidate.is_file() and flat_candidate.stat().st_size == EMPTY_FILE_SIZE_BYTES:
        flat_candidate.unlink(missing_ok=True)

    # 4. Authenticate and download via Kaggle API
    api = authenticate_kaggle_api(logger=logger)
    dest_base.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info(
            "Downloading '%s' from competition '%s' to '%s'...",
            normalized_remote_path,
            competition_id,
            dest_base,
        )

    try:
        api.competition_download_file(
            competition=competition_id,
            file_name=normalized_remote_path,
            path=str(dest_base),
            force=False,
            quiet=False,
        )
    except Exception as exc:
        err_msg = str(exc)
        if (
            str(HTTP_STATUS_FORBIDDEN) in err_msg
            or "forbidden" in err_msg.lower()
            or getattr(exc, "status", None) == HTTP_STATUS_FORBIDDEN
            or getattr(getattr(exc, "response", None), "status_code", None) == HTTP_STATUS_FORBIDDEN
        ):
            rules_url = KAGGLE_RULES_URL_TEMPLATE.format(competition_id=competition_id)
            raise KaggleRulesNotAcceptedError(
                f"HTTP 403 Forbidden: You must accept the competition rules on Kaggle before downloading data. "
                f"Please visit: {rules_url} and click 'I Understand and Accept'."
            ) from exc
        if (
            str(HTTP_STATUS_UNAUTHORIZED) in err_msg
            or "unauthorized" in err_msg.lower()
            or getattr(exc, "status", None) == HTTP_STATUS_UNAUTHORIZED
            or getattr(getattr(exc, "response", None), "status_code", None) == HTTP_STATUS_UNAUTHORIZED
        ):
            raise KaggleAccessError(
                f"HTTP 401 Unauthorized: Kaggle credentials are invalid or expired. "
                f"Please verify KAGGLE_USERNAME and KAGGLE_KEY. Original error: {exc}"
            ) from exc
        raise KaggleAccessError(
            f"Failed to download '{normalized_remote_path}' from Kaggle: {exc}"
        ) from exc

    # 5. Extract archives if Kaggle delivered a zip file
    flat_stem = Path(flat_filename).stem
    candidate_zips = [
        dest_base / f"{flat_filename}{ZIP_EXTENSION}",
        dest_base / f"{flat_stem}{ZIP_EXTENSION}",
        dest_base / f"{normalized_remote_path}{ZIP_EXTENSION}",
        target_path.with_suffix(ZIP_EXTENSION),
        target_path.with_name(f"{flat_filename}{ZIP_EXTENSION}"),
        target_path.with_name(f"{flat_stem}{ZIP_EXTENSION}"),
        target_path.parent / f"{flat_filename}{ZIP_EXTENSION}",
        target_path.parent / f"{flat_stem}{ZIP_EXTENSION}",
    ]
    seen_zips = set()
    possible_zips = []
    for pz in candidate_zips:
        if pz not in seen_zips:
            seen_zips.add(pz)
            possible_zips.append(pz)

    for zip_path in possible_zips:
        if zip_path.is_file():
            if logger:
                logger.info("Extracting downloaded zip archive '%s'...", zip_path)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(path=str(dest_base))
                zip_path.unlink(missing_ok=True)
            except Exception as zip_exc:
                zip_path.unlink(missing_ok=True)
                raise KaggleAccessError(
                    f"Failed to unzip archive '{zip_path}' (removed corrupted archive): {zip_exc}"
                ) from zip_exc

    # 6. Normalize nested paths against Kaggle flattening behavior
    if target_path.is_file() and target_path.stat().st_size > EMPTY_FILE_SIZE_BYTES:
        return target_path

    if flat_candidate.is_file() and flat_candidate.stat().st_size > EMPTY_FILE_SIZE_BYTES:
        if flat_candidate.resolve() != target_path.resolve():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(flat_candidate), str(target_path))
        return target_path

    # Search in destination for any matching file by name
    matches = list(dest_base.rglob(flat_filename))
    for match in matches:
        if match.is_file() and match.stat().st_size > EMPTY_FILE_SIZE_BYTES:
            if match.resolve() == target_path.resolve():
                return target_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(match), str(target_path))
            return target_path

    raise FileNotFoundError(
        f"File '{normalized_remote_path}' was downloaded but could not be located at '{target_path}'."
    )
