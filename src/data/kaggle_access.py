"""Kaggle access abstraction and robust file downloader for ISLR Phase 0.

Provides multi-source credential resolution with strict precedence, HTTP 403 handling,
automatic zip extraction, and normalization of nested paths against Kaggle flattening.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Optional, Union
import zipfile

ENV_KAGGLE_USERNAME = "KAGGLE_USERNAME"
ENV_KAGGLE_KEY = "KAGGLE_KEY"
ENV_KAGGLE_API_TOKEN = "KAGGLE_API_TOKEN"

SOURCE_COLAB_SECRETS = "colab_secrets"
SOURCE_ENV_VARS = "env_vars"
SOURCE_KAGGLE_JSON = "kaggle_json"
SOURCE_COLAB_UPLOAD = "colab_file_upload"
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


class KaggleCredentials(namedtuple("KaggleCredentials", ["username", "key", "source", "token"])):
    """Kaggle credentials container supporting username/key pairs and unified API tokens.

    Unpacks as a 3-tuple (username, key, source) for backward compatibility, while exposing
    .token, .username, .key, and .source attributes.
    """
    __slots__ = ()

    def __new__(
        cls,
        username: str = "",
        key: str = "",
        source: str = "",
        token: Optional[str] = None,
    ) -> KaggleCredentials:
        # If token is provided but username/key are not, provide token as key or preserve both
        return super().__new__(cls, username, key, source, token)

    def __iter__(self):
        # Backward-compatible unpacking: user, key, source = creds
        yield self.username
        yield self.key
        yield self.source


@dataclass(frozen=True)
class FetchResult:
    """Record of observed file fetch behavior."""
    path: Path
    source: str  # in {"local_mirror", "cache", "downloaded"}
    delivered_as: Optional[str] = None  # in {"nested", "flat", "zip", None}
    relocated: bool = False

    def __fspath__(self) -> str:
        """Allow FetchResult to be used directly in open(), Path(), etc."""
        return str(self.path)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, FetchResult):
            return (
                self.path == other.path
                and self.source == other.source
                and self.delivered_as == other.delivered_as
                and self.relocated == other.relocated
            )
        if isinstance(other, (Path, str)):
            return self.path == Path(other)
        return False

    def __hash__(self) -> int:
        return hash((self.path, self.source, self.delivered_as, self.relocated))

    def __str__(self) -> str:
        return str(self.path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.path, name)


class KaggleAccessError(Exception):
    """Base error for Kaggle access and download operations."""


class KaggleCredentialsNotFoundError(KaggleAccessError):
    """Raised when Kaggle credentials cannot be discovered."""


class KaggleRulesNotAcceptedError(KaggleAccessError):
    """Raised when Kaggle returns HTTP 403 due to unaccepted competition rules."""


def _import_kaggle_api() -> Any:
    """Import KaggleApi class from kaggle package, raising clear ImportError if missing.

    Helper function to enable mocking in test environments without requiring kaggle installed.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        return KaggleApi
    except ImportError as exc:
        raise ImportError(
            "The 'kaggle' package is not installed. Please run 'pip install kaggle' to enable Kaggle API access."
        ) from exc


def get_kaggle_credentials(allow_interactive: bool = True) -> Optional[KaggleCredentials]:
    """Discover Kaggle credentials following strict fallback precedence.

    Supports KAGGLE_API_TOKEN as well as KAGGLE_USERNAME / KAGGLE_KEY pairs. Never logs credentials.

    Precedence:
        1. Google Colab secrets (google.colab.userdata: KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY)
        2. Environment variables (KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY)
        3. Local configuration file (~/.kaggle/kaggle.json or $KAGGLE_CONFIG_DIR/kaggle.json)
        4. In Google Colab: interactive fallback via google.colab.files.upload() for kaggle.json
        5. Interactive terminal prompt (if stdin is a TTY and allow_interactive is True)

    Args:
        allow_interactive: Whether to prompt interactively if other sources are absent.

    Returns:
        Optional[KaggleCredentials]: Discovered credentials or None if absent.
    """
    # 1. Colab user secrets
    try:
        from google.colab import userdata
        token = userdata.get(ENV_KAGGLE_API_TOKEN)
        t_colab = str(token).strip() if token else ""
        if t_colab:
            return KaggleCredentials(username="", key="", source=SOURCE_COLAB_SECRETS, token=t_colab)

        username = userdata.get(ENV_KAGGLE_USERNAME)
        key = userdata.get(ENV_KAGGLE_KEY)
        u_colab = str(username).strip() if username else ""
        k_colab = str(key).strip() if key else ""
        if u_colab and k_colab:
            return KaggleCredentials(username=u_colab, key=k_colab, source=SOURCE_COLAB_SECRETS, token=None)
    except (ImportError, Exception):
        pass

    # 2. Environment variables
    token_env = (os.environ.get(ENV_KAGGLE_API_TOKEN) or "").strip()
    if token_env:
        return KaggleCredentials(username="", key="", source=SOURCE_ENV_VARS, token=token_env)

    username_env = (os.environ.get(ENV_KAGGLE_USERNAME) or "").strip()
    key_env = (os.environ.get(ENV_KAGGLE_KEY) or "").strip()
    if username_env and key_env:
        return KaggleCredentials(username=username_env, key=key_env, source=SOURCE_ENV_VARS, token=None)

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
                token = str(data.get("token") or "").strip()
                if token:
                    return KaggleCredentials(username="", key="", source=SOURCE_KAGGLE_JSON, token=token)
                u = str(data.get("username") or "").strip()
                k = str(data.get("key") or "").strip()
                if u and k:
                    return KaggleCredentials(username=u, key=k, source=SOURCE_KAGGLE_JSON, token=None)
        except (json.JSONDecodeError, OSError, AttributeError):
            pass

    # 4. Interactive fallback
    if allow_interactive:
        # Distinguish Kaggle from Colab (Kaggle also has GCS env vars)
        is_kaggle = (
            os.path.exists("/kaggle")
            or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
            or "kaggle_web_client" in sys.modules
        )
        is_colab = not is_kaggle and (
            "google.colab" in sys.modules
            or os.environ.get("COLAB_GPU") is not None
        )
        if is_colab:
            try:
                from google.colab import files
                sys.stdout.write("Upload your kaggle.json file:\n")
                sys.stdout.flush()
                uploaded = files.upload()
                for fname, content in uploaded.items():
                    if fname.endswith(KAGGLE_JSON_FILENAME):
                        text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            token = str(parsed.get("token") or "").strip()
                            u = str(parsed.get("username") or "").strip()
                            k = str(parsed.get("key") or "").strip()
                            if token or (u and k):
                                target_dir = Path.home() / KAGGLE_CONFIG_DIR
                                target_dir.mkdir(parents=True, exist_ok=True)
                                target_file = target_dir / KAGGLE_JSON_FILENAME
                                target_file.write_text(json.dumps(parsed), encoding="utf-8")
                                try:
                                    target_file.chmod(0o600)
                                except OSError:
                                    pass
                                if token:
                                    return KaggleCredentials(username="", key="", source=SOURCE_COLAB_UPLOAD, token=token)
                                return KaggleCredentials(username=u, key=k, source=SOURCE_COLAB_UPLOAD, token=None)
            except Exception:
                pass

        # Interactive terminal prompt fallback for local TTY only (never in cloud notebooks)
        if sys.stdin.isatty() and not is_colab and not is_kaggle:
            try:
                import getpass
                sys.stdout.write("Enter Kaggle Username (or press Enter if using token): ")
                sys.stdout.flush()
                user_input = sys.stdin.readline().strip()
                if not user_input:
                    token_input = getpass.getpass("Enter Kaggle API Token: ").strip()
                    if token_input:
                        return KaggleCredentials(username="", key="", source=SOURCE_INTERACTIVE_PROMPT, token=token_input)
                    return None
                key_input = getpass.getpass("Enter Kaggle API Key: ").strip()
                if key_input:
                    return KaggleCredentials(username=user_input, key=key_input, source=SOURCE_INTERACTIVE_PROMPT, token=None)
            except Exception:
                pass

    return None


def get_credential_status() -> str:
    """Return presence status of Kaggle credentials without exposing sensitive keys.

    Inspects saved credentials non-interactively. Never logs credentials.

    Returns:
        str: 'present' or 'absent'.
    """
    creds = get_kaggle_credentials(allow_interactive=False)
    return STATUS_PRESENT if creds is not None else STATUS_ABSENT


def authenticate_kaggle_api(
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Authenticate and return KaggleApi instance using discovered credentials.

    Checks credentials BEFORE importing the kaggle package. Never logs credentials.

    Args:
        logger: Optional logger for diagnostic messages (credentials are never logged).

    Returns:
        KaggleApi: Authenticated API client instance.

    Raises:
        KaggleCredentialsNotFoundError: If credentials cannot be found.
        ImportError: If the 'kaggle' package is not installed.
        KaggleRulesNotAcceptedError: If HTTP 403 rules acceptance error occurs.
        KaggleAccessError: If authentication fails.
    """
    # 1. Discover credentials first before importing kaggle
    creds = get_kaggle_credentials(allow_interactive=True)
    if creds is None:
        raise KaggleCredentialsNotFoundError(
            "Kaggle credentials not found. Configure KAGGLE_API_TOKEN or KAGGLE_USERNAME and KAGGLE_KEY "
            "as environment variables, in Google Colab secrets, or in ~/.kaggle/kaggle.json."
        )

    # 2. Import kaggle api class
    kaggle_api_cls = _import_kaggle_api()

    # 3. Configure environment variables for authentication
    if creds.token:
        os.environ[ENV_KAGGLE_API_TOKEN] = creds.token
    if creds.username and creds.key:
        os.environ[ENV_KAGGLE_USERNAME] = creds.username
        os.environ[ENV_KAGGLE_KEY] = creds.key

    if logger:
        logger.info("Authenticating Kaggle API via %s.", creds.source)

    try:
        api = kaggle_api_cls()
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
                f"Please verify KAGGLE_API_TOKEN or KAGGLE_USERNAME and KAGGLE_KEY. Original error: {exc}"
            ) from exc
        raise KaggleAccessError(f"Kaggle API authentication failed: {exc}") from exc


def fetch_file(
    remote_path: str,
    dest: Optional[Union[str, Path]] = None,
    competition_id: str = DEFAULT_COMPETITION_ID,
    local_mirror_dir: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None,
    dest_dir: Optional[Union[str, Path]] = None,
) -> FetchResult:
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
        FetchResult: Record containing canonical local path, source, delivered_as, and relocated.

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
            return FetchResult(
                path=mirror_candidate,
                source="local_mirror",
                delivered_as=None,
                relocated=False,
            )

    # 2. Check idempotent cache at target path
    if target_path.is_file() and target_path.stat().st_size > EMPTY_FILE_SIZE_BYTES:
        if logger:
            logger.info("Cache hit for '%s': %s.", normalized_remote_path, target_path)
        return FetchResult(
            path=target_path,
            source="cache",
            delivered_as=None,
            relocated=False,
        )

    # 2.1 Check alternative Kaggle mount paths (e.g. /kaggle/input/competitions/ vs /kaggle/input/)
    str_target = str(target_path)
    if "/kaggle/input/" in str_target:
        alt_paths = []
        if "/kaggle/input/competitions/" in str_target:
            alt_paths.append(Path(str_target.replace("/kaggle/input/competitions/", "/kaggle/input/")))
        else:
            alt_paths.append(Path(str_target.replace("/kaggle/input/", "/kaggle/input/competitions/")))
        for alt_path in alt_paths:
            if alt_path.is_file() and alt_path.stat().st_size > EMPTY_FILE_SIZE_BYTES:
                if logger:
                    logger.info("Cache hit for '%s' at alternative Kaggle path: %s.", normalized_remote_path, alt_path)
                return FetchResult(
                    path=alt_path,
                    source="cache",
                    delivered_as=None,
                    relocated=False,
                )

    # Clean up empty/corrupted zero-byte target if left by prior failed attempt
    if target_path.is_file() and target_path.stat().st_size == EMPTY_FILE_SIZE_BYTES:
        target_path.unlink(missing_ok=True)

    # 3. Check if file is already present in flattened location inside dest
    flat_candidate = dest_base / flat_filename
    if flat_candidate.is_file() and flat_candidate.stat().st_size > EMPTY_FILE_SIZE_BYTES:
        relocated = False
        if flat_candidate.resolve() != target_path.resolve():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(flat_candidate), str(target_path))
            relocated = True
            if logger:
                logger.info("Relocated flattened file '%s' to '%s'.", flat_candidate, target_path)
        return FetchResult(
            path=target_path,
            source="cache",
            delivered_as=None,
            relocated=relocated,
        )

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

    was_zip = False
    for zip_path in possible_zips:
        if zip_path.is_file():
            was_zip = True
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
    if was_zip:
        if target_path.is_file() and target_path.stat().st_size > EMPTY_FILE_SIZE_BYTES:
            return FetchResult(
                path=target_path,
                source="downloaded",
                delivered_as="zip",
                relocated=False,
            )
        if flat_candidate.is_file() and flat_candidate.stat().st_size > EMPTY_FILE_SIZE_BYTES:
            relocated = flat_candidate.resolve() != target_path.resolve()
            if relocated:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(flat_candidate), str(target_path))
            return FetchResult(
                path=target_path,
                source="downloaded",
                delivered_as="zip",
                relocated=relocated,
            )
        matches = list(dest_base.rglob(flat_filename))
        for match in matches:
            if match.is_file() and match.stat().st_size > EMPTY_FILE_SIZE_BYTES:
                relocated = match.resolve() != target_path.resolve()
                if relocated:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(match), str(target_path))
                return FetchResult(
                    path=target_path,
                    source="downloaded",
                    delivered_as="zip",
                    relocated=relocated,
                )
    else:
        if target_path.is_file() and target_path.stat().st_size > EMPTY_FILE_SIZE_BYTES:
            delivered_as = "nested" if remote_path_obj.parent != Path(".") else "flat"
            return FetchResult(
                path=target_path,
                source="downloaded",
                delivered_as=delivered_as,
                relocated=False,
            )
        if flat_candidate.is_file() and flat_candidate.stat().st_size > EMPTY_FILE_SIZE_BYTES:
            relocated = flat_candidate.resolve() != target_path.resolve()
            if relocated:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(flat_candidate), str(target_path))
            return FetchResult(
                path=target_path,
                source="downloaded",
                delivered_as="flat",
                relocated=relocated,
            )
        matches = list(dest_base.rglob(flat_filename))
        for match in matches:
            if match.is_file() and match.stat().st_size > EMPTY_FILE_SIZE_BYTES:
                relocated = match.resolve() != target_path.resolve()
                if relocated:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(match), str(target_path))
                return FetchResult(
                    path=target_path,
                    source="downloaded",
                    delivered_as="nested",
                    relocated=relocated,
                )

    raise FileNotFoundError(
        f"File '{normalized_remote_path}' was downloaded but could not be located at '{target_path}'."
    )
