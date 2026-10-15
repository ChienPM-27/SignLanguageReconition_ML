"""Tests for Kaggle access abstraction and loader (R3).

Covers authentication fallback precedence, status reporting, HTTP 403 handling,
idempotent caching, local mirror bypass, zip extraction, and path flattening normalization.
"""

import json
import os
from pathlib import Path
import sys
from typing import Any
import zipfile
import pytest

from src.data.kaggle_access import (
    KaggleAccessError,
    KaggleCredentialsNotFoundError,
    KaggleRulesNotAcceptedError,
    SOURCE_COLAB_SECRETS,
    SOURCE_ENV_VARS,
    SOURCE_INTERACTIVE_PROMPT,
    SOURCE_KAGGLE_JSON,
    STATUS_ABSENT,
    STATUS_PRESENT,
    fetch_file,
    get_credential_status,
    get_kaggle_credentials,
)

SAMPLE_COMPETITION = "asl-signs"
DUMMY_REMOTE_FILE = "train_landmark_files/26734/1000035562.parquet"


def test_credential_precedence_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    """Precedence 1: Colab user secrets supersede env vars and local files."""
    class MockUserData:
        @staticmethod
        def get(key: str) -> str:
            if key == "KAGGLE_USERNAME":
                return "colab_user"
            if key == "KAGGLE_KEY":
                return "colab_key"
            return ""

    mock_colab = type(sys)("google.colab")
    mock_colab.userdata = MockUserData()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google.colab", mock_colab)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", MockUserData())

    monkeypatch.setenv("KAGGLE_USERNAME", "env_user")
    monkeypatch.setenv("KAGGLE_KEY", "env_key")

    creds = get_kaggle_credentials()
    assert creds is not None
    user, key, source = creds
    assert user == "colab_user"
    assert key == "colab_key"
    assert source == SOURCE_COLAB_SECRETS


def test_credential_precedence_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Precedence 2: Environment variables are used when Colab secrets are absent."""
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.setenv("KAGGLE_USERNAME", "env_user")
    monkeypatch.setenv("KAGGLE_KEY", "env_key")

    creds = get_kaggle_credentials()
    assert creds is not None
    user, key, source = creds
    assert user == "env_user"
    assert key == "env_key"
    assert source == SOURCE_ENV_VARS


def test_credential_precedence_kaggle_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Precedence 3: ~/.kaggle/kaggle.json used when Colab and env vars are absent."""
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    fake_home = tmp_path / "home"
    fake_kaggle_dir = fake_home / ".kaggle"
    fake_kaggle_dir.mkdir(parents=True)
    fake_json = fake_kaggle_dir / "kaggle.json"
    fake_json.write_text(json.dumps({"username": "json_user", "key": "json_key"}), encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: fake_home)

    creds = get_kaggle_credentials()
    assert creds is not None
    user, key, source = creds
    assert user == "json_user"
    assert key == "json_key"
    assert source == SOURCE_KAGGLE_JSON


def test_credential_precedence_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Precedence 4: Interactive prompt used when all earlier sources are absent and stdin is TTY."""
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent/home"))

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdin, "readline", lambda: "prompt_user\n")

    import getpass
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "prompt_key")

    creds = get_kaggle_credentials()
    assert creds is not None
    user, key, source = creds
    assert user == "prompt_user"
    assert key == "prompt_key"
    assert source == SOURCE_INTERACTIVE_PROMPT


def test_credential_status_reporting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_credential_status reports 'present' or 'absent' without leaking secrets."""
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent/home"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    assert get_credential_status() == STATUS_ABSENT

    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    assert get_credential_status() == STATUS_PRESENT


def test_fetch_file_local_mirror_bypass(tmp_path: Path) -> None:
    """Verify pre-existing file in local mirror bypasses download and auth entirely."""
    mirror_dir = tmp_path / "mirror"
    mirror_file = mirror_dir / "train.csv"
    mirror_dir.mkdir(parents=True)
    mirror_file.write_text("dummy,content\n", encoding="utf-8")

    dest_dir = tmp_path / "dest"
    result = fetch_file(
        remote_path="train.csv",
        dest_dir=dest_dir,
        local_mirror_dir=mirror_dir,
    )
    assert result == mirror_file
    assert not (dest_dir / "train.csv").exists()


def test_fetch_file_idempotent_caching(tmp_path: Path) -> None:
    """Verify cached target file in dest_dir is reused without downloading."""
    dest_dir = tmp_path / "dest"
    cached_file = dest_dir / DUMMY_REMOTE_FILE
    cached_file.parent.mkdir(parents=True)
    cached_file.write_text("cached_parquet", encoding="utf-8")

    result = fetch_file(
        remote_path=DUMMY_REMOTE_FILE,
        dest_dir=dest_dir,
        competition_id=SAMPLE_COMPETITION,
    )
    assert result == cached_file
    assert result.read_text(encoding="utf-8") == "cached_parquet"


def test_fetch_file_http_403_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify HTTP 403 competition access error raises KaggleRulesNotAcceptedError with URL."""
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")

    class MockKaggleApi403:
        def authenticate(self) -> None:
            pass

        def competition_download_file(self, **kwargs: Any) -> None:
            raise Exception("403 - Forbidden: You must accept the competition rules")

    from src.data import kaggle_access
    monkeypatch.setattr(kaggle_access, "authenticate_kaggle_api", lambda logger=None: MockKaggleApi403())

    dest_dir = tmp_path / "dest"
    with pytest.raises(KaggleRulesNotAcceptedError, match="accept the competition rules on Kaggle"):
        fetch_file(
            remote_path="train.csv",
            dest_dir=dest_dir,
            competition_id=SAMPLE_COMPETITION,
        )


def test_fetch_file_zip_extraction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify downloaded zip archive is automatically extracted and cleaned up."""
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir(parents=True)

    class MockKaggleApiZip:
        def authenticate(self) -> None:
            pass

        def competition_download_file(self, **kwargs: Any) -> None:
            # Simulate kaggle downloading train.csv.zip into dest_dir
            zip_target = dest_dir / "train.csv.zip"
            with zipfile.ZipFile(zip_target, "w") as zf:
                zf.writestr("train.csv", "path,participant_id\n1,2\n")

    from src.data import kaggle_access
    monkeypatch.setattr(kaggle_access, "authenticate_kaggle_api", lambda logger=None: MockKaggleApiZip())

    result = fetch_file(
        remote_path="train.csv",
        dest_dir=dest_dir,
        competition_id=SAMPLE_COMPETITION,
    )
    assert result == dest_dir / "train.csv"
    assert result.is_file()
    assert not (dest_dir / "train.csv.zip").exists()


def test_fetch_file_flattening_normalization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify flattened download is moved into canonical nested directory structure."""
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir(parents=True)

    class MockKaggleApiFlat:
        def authenticate(self) -> None:
            pass

        def competition_download_file(self, **kwargs: Any) -> None:
            # Kaggle CLI flattens nested path to just the filename in dest_dir
            flat_target = dest_dir / "1000035562.parquet"
            flat_target.write_text("fake_parquet_content", encoding="utf-8")

    from src.data import kaggle_access
    monkeypatch.setattr(kaggle_access, "authenticate_kaggle_api", lambda logger=None: MockKaggleApiFlat())

    result = fetch_file(
        remote_path=DUMMY_REMOTE_FILE,
        dest_dir=dest_dir,
        competition_id=SAMPLE_COMPETITION,
    )
    assert result == dest_dir / DUMMY_REMOTE_FILE
    assert result.is_file()
    assert not (dest_dir / "1000035562.parquet").exists()


def test_fetch_file_absent_credentials_raises_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify fetch_file raises KaggleCredentialsNotFoundError when credentials are absent and file is un-cached."""
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent/home"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    dest_dir = tmp_path / "dest"
    with pytest.raises(KaggleCredentialsNotFoundError, match="Kaggle credentials not found"):
        fetch_file(
            remote_path="missing.csv",
            dest_dir=dest_dir,
            competition_id=SAMPLE_COMPETITION,
        )


def test_authenticate_kaggle_api_absent_credentials_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify authenticate_kaggle_api raises KaggleCredentialsNotFoundError when no creds exist."""
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent/home"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    from src.data.kaggle_access import authenticate_kaggle_api
    with pytest.raises(KaggleCredentialsNotFoundError, match="Kaggle credentials not found"):
        authenticate_kaggle_api()


def test_fetch_file_dest_keyword_argument(tmp_path: Path) -> None:
    """Verify fetch_file accepts 'dest' keyword argument as required by R3."""
    dest_dir = tmp_path / "dest"
    cached = dest_dir / "train.csv"
    dest_dir.mkdir(parents=True)
    cached.write_text("header,val\n1,2\n", encoding="utf-8")

    result = fetch_file(
        remote_path="train.csv",
        dest=dest_dir,
        competition_id=SAMPLE_COMPETITION,
    )
    assert result == cached


def test_fetch_file_direct_target_filepath(tmp_path: Path) -> None:
    """Verify fetch_file accepts destination pointing directly to target file path."""
    dest_file = tmp_path / "custom_data" / "train.csv"
    dest_file.parent.mkdir(parents=True)
    dest_file.write_text("dummy,content\n", encoding="utf-8")

    result = fetch_file(
        remote_path="train.csv",
        dest=dest_file,
        competition_id=SAMPLE_COMPETITION,
    )
    assert result == dest_file


def test_fetch_file_corrupt_zip_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify corrupted zip archive is cleaned up on extraction failure to prevent permanent poisoning."""
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir(parents=True)
    bad_zip = dest_dir / "train.csv.zip"
    bad_zip.write_bytes(b"corrupted partial zip bytes")

    class MockKaggleApiNoop:
        def authenticate(self) -> None:
            pass
        def competition_download_file(self, **kwargs: Any) -> None:
            pass

    from src.data import kaggle_access
    monkeypatch.setattr(kaggle_access, "authenticate_kaggle_api", lambda logger=None: MockKaggleApiNoop())

    with pytest.raises(KaggleAccessError, match="Failed to unzip archive"):
        fetch_file(
            remote_path="train.csv",
            dest=dest_dir,
            competition_id=SAMPLE_COMPETITION,
        )

    assert not bad_zip.exists(), "Corrupted zip file must be unlinked to allow future retry"


def test_fetch_file_windows_backslash_normalization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify remote path containing backslashes is normalized to POSIX forward slashes for Kaggle API."""
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir(parents=True)
    downloaded_filename = ""

    class MockKaggleApiCapture:
        def authenticate(self) -> None:
            pass
        def competition_download_file(self, **kwargs: Any) -> None:
            nonlocal downloaded_filename
            downloaded_filename = kwargs.get("file_name", "")
            # Place flat file to simulate download
            target = dest_dir / "1000035562.parquet"
            target.write_text("parquet_content", encoding="utf-8")

    from src.data import kaggle_access
    monkeypatch.setattr(kaggle_access, "authenticate_kaggle_api", lambda logger=None: MockKaggleApiCapture())

    backslash_remote = "train_landmark_files\\26734\\1000035562.parquet"
    result = fetch_file(
        remote_path=backslash_remote,
        dest=dest_dir,
        competition_id=SAMPLE_COMPETITION,
    )
    assert downloaded_filename == "train_landmark_files/26734/1000035562.parquet"
    assert result.is_file()


def test_credential_precedence_kaggle_config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify KAGGLE_CONFIG_DIR environment variable is respected for locating kaggle.json."""
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    custom_cfg_dir = tmp_path / "custom_kaggle"
    custom_cfg_dir.mkdir(parents=True)
    custom_json = custom_cfg_dir / "kaggle.json"
    custom_json.write_text(json.dumps({"username": "cfg_dir_user", "key": "cfg_dir_key"}), encoding="utf-8")

    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(custom_cfg_dir))
    monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent/home"))

    creds = get_kaggle_credentials()
    assert creds is not None
    user, key, source = creds
    assert user == "cfg_dir_user"
    assert key == "cfg_dir_key"
    assert source == SOURCE_KAGGLE_JSON


def test_get_credential_status_does_not_prompt_in_interactive_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_credential_status does NOT trigger interactive prompt when stdin is a TTY and creds are absent."""
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent/home"))
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)

    # Simulate interactive TTY
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    prompt_triggered = False

    def mock_readline() -> str:
        nonlocal prompt_triggered
        prompt_triggered = True
        return "user\n"

    monkeypatch.setattr(sys.stdin, "readline", mock_readline)

    status = get_credential_status()
    assert status == STATUS_ABSENT
    assert prompt_triggered is False, "get_credential_status must not interactively prompt the user"


def test_fetch_file_destination_with_dot_in_directory_name(tmp_path: Path) -> None:
    """Verify fetch_file correctly treats non-existent destination directory with a dot as a directory."""
    dotted_dest_dir = tmp_path / "custom.v1"
    # Destination directory does not exist yet
    assert not dotted_dest_dir.exists()

    mirror_dir = tmp_path / "mirror"
    mirror_file = mirror_dir / "train.csv"
    mirror_dir.mkdir(parents=True)
    mirror_file.write_text("dummy,content\n", encoding="utf-8")

    # Local mirror hit should resolve to mirror_file without misclassifying dotted_dest_dir as a file
    result = fetch_file(
        remote_path="train.csv",
        dest=dotted_dest_dir,
        local_mirror_dir=mirror_dir,
    )
    assert result == mirror_file

    # Now verify with an existing cached file in dotted_dest_dir / train.csv
    cached_file = dotted_dest_dir / "train.csv"
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    cached_file.write_text("cached,content\n", encoding="utf-8")

    result_cached = fetch_file(
        remote_path="train.csv",
        dest=dotted_dest_dir,
    )
    assert result_cached == cached_file


def test_fetch_file_stem_zip_extraction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify zip archive named with stem (e.g. 1000035562.zip) in dest_base is extracted and normalized."""
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir(parents=True)

    class MockKaggleApiStemZip:
        def authenticate(self) -> None:
            pass

        def competition_download_file(self, **kwargs: Any) -> None:
            # Simulate Kaggle downloading 1000035562.zip (stem without .parquet) into dest_dir
            zip_target = dest_dir / "1000035562.zip"
            with zipfile.ZipFile(zip_target, "w") as zf:
                zf.writestr("1000035562.parquet", "sample_parquet_bytes")

    from src.data import kaggle_access
    monkeypatch.setattr(kaggle_access, "authenticate_kaggle_api", lambda logger=None: MockKaggleApiStemZip())

    result = fetch_file(
        remote_path=DUMMY_REMOTE_FILE,
        dest=dest_dir,
        competition_id=SAMPLE_COMPETITION,
    )
    assert result == dest_dir / DUMMY_REMOTE_FILE
    assert result.is_file()
    assert not (dest_dir / "1000035562.zip").exists()


def test_fetch_file_empty_remote_path_raises_value_error(tmp_path: Path) -> None:
    """Verify empty or whitespace-only remote_path raises ValueError."""
    dest_dir = tmp_path / "dest"
    with pytest.raises(ValueError, match="remote_path must be a non-empty string"):
        fetch_file(
            remote_path="   ",
            dest=dest_dir,
            competition_id=SAMPLE_COMPETITION,
        )


def test_get_kaggle_credentials_non_dict_json_handled_safely(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify non-dictionary content in kaggle.json is ignored safely without raising AttributeError."""
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    fake_home = tmp_path / "home"
    fake_kaggle_dir = fake_home / ".kaggle"
    fake_kaggle_dir.mkdir(parents=True)
    fake_json = fake_kaggle_dir / "kaggle.json"
    fake_json.write_text("[\"not\", \"a\", \"dict\"]", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    creds = get_kaggle_credentials(allow_interactive=False)
    assert creds is None
    assert get_credential_status() == STATUS_ABSENT


def test_get_kaggle_credentials_whitespace_only_env_vars_treated_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify whitespace-only credentials in environment variables are treated as absent."""
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.setenv("KAGGLE_USERNAME", "   ")
    monkeypatch.setenv("KAGGLE_KEY", "   ")
    monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent/home"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    creds = get_kaggle_credentials(allow_interactive=False)
    assert creds is None
    assert get_credential_status() == STATUS_ABSENT


def test_fetch_file_destination_dir_named_same_as_file(tmp_path: Path) -> None:
    """Verify when dest is an existing directory matching remote filename, file is fetched inside it."""
    dest_dir = tmp_path / "train.csv"
    dest_dir.mkdir()
    cached_file = dest_dir / "train.csv"
    cached_file.write_text("header,col\n1,2\n", encoding="utf-8")

    result = fetch_file(
        remote_path="train.csv",
        dest=dest_dir,
    )
    assert result == cached_file
    assert result.is_file()


def test_fetch_file_http_401_unauthorized_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify HTTP 401 error raises KaggleAccessError with actionable credentials notice."""
    monkeypatch.setenv("KAGGLE_USERNAME", "bad_user")
    monkeypatch.setenv("KAGGLE_KEY", "bad_key")

    class MockKaggleApi401:
        def authenticate(self) -> None:
            pass

        def competition_download_file(self, **kwargs: Any) -> None:
            raise Exception("401 - Unauthorized")

    from src.data import kaggle_access
    monkeypatch.setattr(kaggle_access, "authenticate_kaggle_api", lambda logger=None: MockKaggleApi401())

    dest_dir = tmp_path / "dest"
    with pytest.raises(KaggleAccessError, match="HTTP 401 Unauthorized"):
        fetch_file(
            remote_path="train.csv",
            dest=dest_dir,
            competition_id=SAMPLE_COMPETITION,
        )


def test_missing_kaggle_package_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing kaggle package raises ImportError with clear installation instructions."""
    monkeypatch.setenv("KAGGLE_USERNAME", "some_user")
    monkeypatch.setenv("KAGGLE_KEY", "some_key")

    monkeypatch.setitem(sys.modules, "kaggle", None)
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", None)

    from src.data.kaggle_access import authenticate_kaggle_api
    with pytest.raises(ImportError, match="pip install kaggle"):
        authenticate_kaggle_api()


def test_colab_files_upload_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify in Colab with no credentials found, fallback triggers files.upload()."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    class MockUserData:
        @staticmethod
        def get(key: str) -> None:
            return None

    class MockFiles:
        @staticmethod
        def upload() -> dict:
            return {
                "kaggle.json": json.dumps({"username": "uploaded_user", "key": "uploaded_key"}).encode("utf-8")
            }

    mock_colab = type(sys)("google.colab")
    mock_colab.userdata = MockUserData()  # type: ignore[attr-defined]
    mock_colab.files = MockFiles()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google.colab", mock_colab)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", MockUserData())
    monkeypatch.setitem(sys.modules, "google.colab.files", MockFiles())

    creds = get_kaggle_credentials(allow_interactive=True)
    assert creds is not None
    user, key, source = creds
    assert user == "uploaded_user"
    assert key == "uploaded_key"
    assert source == "colab_file_upload"


def test_credential_precedence_colab_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify KAGGLE_API_TOKEN in Colab secrets is discovered and given highest precedence."""
    class MockUserDataToken:
        @staticmethod
        def get(key: str) -> str:
            if key == "KAGGLE_API_TOKEN":
                return "colab_secret_token_123"
            if key == "KAGGLE_USERNAME":
                return "colab_user"
            if key == "KAGGLE_KEY":
                return "colab_key"
            return ""

    mock_colab = type(sys)("google.colab")
    mock_colab.userdata = MockUserDataToken()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google.colab", mock_colab)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", MockUserDataToken())

    monkeypatch.setenv("KAGGLE_API_TOKEN", "env_token")
    monkeypatch.setenv("KAGGLE_USERNAME", "env_user")
    monkeypatch.setenv("KAGGLE_KEY", "env_key")

    creds = get_kaggle_credentials()
    assert creds is not None
    assert creds.source == SOURCE_COLAB_SECRETS
    assert creds.token == "colab_secret_token_123"


def test_credential_precedence_env_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify KAGGLE_API_TOKEN in environment variables is discovered over username/key."""
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "env_secret_token_456")
    monkeypatch.setenv("KAGGLE_USERNAME", "env_user")
    monkeypatch.setenv("KAGGLE_KEY", "env_key")

    creds = get_kaggle_credentials()
    assert creds is not None
    assert creds.source == SOURCE_ENV_VARS
    assert creds.token == "env_secret_token_456"


def test_authenticate_with_mocked_import_kaggle_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify authenticate_kaggle_api works with mocked _import_kaggle_api when kaggle is uninstalled."""
    monkeypatch.setenv("KAGGLE_API_TOKEN", "test_mock_token")

    authenticated_called = False

    class MockKaggleApiInstance:
        def authenticate(self) -> None:
            nonlocal authenticated_called
            authenticated_called = True

    from src.data import kaggle_access
    monkeypatch.setattr(kaggle_access, "_import_kaggle_api", lambda: MockKaggleApiInstance)

    api = kaggle_access.authenticate_kaggle_api()
    assert isinstance(api, MockKaggleApiInstance)
    assert authenticated_called is True
    assert os.environ.get("KAGGLE_API_TOKEN") == "test_mock_token"


def test_check_credentials_happens_before_kaggle_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify credentials check occurs strictly before importing kaggle (KaggleCredentialsNotFoundError raised without import)."""
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent/home"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    from src.data import kaggle_access

    import_called = False

    def mock_import() -> Any:
        nonlocal import_called
        import_called = True
        raise AssertionError("_import_kaggle_api should not be called when credentials are absent!")

    monkeypatch.setattr(kaggle_access, "_import_kaggle_api", mock_import)

    with pytest.raises(KaggleCredentialsNotFoundError):
        kaggle_access.authenticate_kaggle_api()

    assert import_called is False, "_import_kaggle_api must NOT be called when credentials are absent"


def test_fetch_result_metadata_mirror_and_cache(tmp_path: Path) -> None:
    """Verify FetchResult attributes for mirror bypass and cache hit."""
    mirror_dir = tmp_path / "mirror"
    mirror_file = mirror_dir / "train.csv"
    mirror_dir.mkdir(parents=True)
    mirror_file.write_text("a,b\n1,2\n", encoding="utf-8")

    dest_dir = tmp_path / "dest"
    res_mirror = fetch_file(
        remote_path="train.csv",
        dest_dir=dest_dir,
        local_mirror_dir=mirror_dir,
    )
    assert res_mirror.source == "local_mirror"
    assert res_mirror.delivered_as is None
    assert res_mirror.relocated is False
    assert res_mirror.path == mirror_file

    # Cache hit
    cached_file = dest_dir / DUMMY_REMOTE_FILE
    cached_file.parent.mkdir(parents=True)
    cached_file.write_text("data", encoding="utf-8")
    res_cache = fetch_file(
        remote_path=DUMMY_REMOTE_FILE,
        dest_dir=dest_dir,
        competition_id=SAMPLE_COMPETITION,
    )
    assert res_cache.source == "cache"
    assert res_cache.delivered_as is None
    assert res_cache.relocated is False
    assert res_cache.path == cached_file


def test_fetch_result_metadata_downloaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify FetchResult attributes for downloaded files (flat, zip, nested)."""
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir(parents=True)

    class MockKaggleApiFlat:
        def authenticate(self) -> None:
            pass

        def competition_download_file(self, **kwargs: Any) -> None:
            flat_target = dest_dir / "1000035562.parquet"
            flat_target.write_text("fake_parquet_content", encoding="utf-8")

    from src.data import kaggle_access
    monkeypatch.setattr(kaggle_access, "authenticate_kaggle_api", lambda logger=None: MockKaggleApiFlat())

    res = fetch_file(
        remote_path=DUMMY_REMOTE_FILE,
        dest_dir=dest_dir,
        competition_id=SAMPLE_COMPETITION,
    )
    assert res.source == "downloaded"
    assert res.delivered_as == "flat"
    assert res.relocated is True
    assert res.path == dest_dir / DUMMY_REMOTE_FILE



