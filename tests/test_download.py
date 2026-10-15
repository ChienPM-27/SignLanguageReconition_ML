"""Tests for ISLR data downloading module and CLI stage (Step 2).

Covers local mirror bypass, max_files_per_gloss cap, download budget,
retry resilience on single-file failures, clear authentication and rules error reporting,
and manifest CSV writing.
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
import pytest

from src.data.download import download_target_gloss_data
from src.data.kaggle_access import (
    FetchResult,
    KaggleCredentialsNotFoundError,
    KaggleRulesNotAcceptedError,
)
from src.utils.config import load_config
from src.utils.env import ENV_LOCAL, resolve_paths
from tests.test_selection import _create_synthetic_catalog

prepare_data_module = importlib.import_module("scripts.10_prepare_data")
run_pipeline = prepare_data_module.run_pipeline


def test_download_with_local_mirror_bypass(tmp_path: Path) -> None:
    mirror_dir = tmp_path / "mirror"
    catalog_path = mirror_dir / "train.csv"
    _create_synthetic_catalog(catalog_path, mirror_dir, num_signs=5, samples_per_sign=4)

    selection_dir = tmp_path / "selection"
    selection_dir.mkdir(parents=True)
    glosses_csv = selection_dir / "glosses.csv"
    glosses_csv.write_text(
        "gloss,label,n_samples,n_participants,max_dominance,hand_coverage,motion_energy,composite_score\n"
        "sign_00,0,4,4,0.25,1.0,0.1,0.9\n"
        "sign_01,1,4,4,0.25,1.0,0.1,0.85\n",
        encoding="utf-8",
    )

    overrides = [
        f"data.local_mirror_dir={mirror_dir}",
        f"output.selection_dir={selection_dir}",
        f"output.output_dir={tmp_path / 'out'}",
        "data.num_download_workers=2",
    ]
    cfg = load_config(overrides=overrides)
    paths = resolve_paths(ENV_LOCAL, cfg, create_dirs=True)

    manifest_path, records = download_target_gloss_data(cfg=cfg, paths=paths)

    assert manifest_path.is_file()
    assert len(records) == 8  # 2 signs * 4 samples each
    for r in records:
        assert r.status == "local_mirror"
        assert Path(r.local_path).is_file()

    with open(manifest_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 8
        assert rows[0]["status"] == "local_mirror"


def test_download_max_files_per_gloss(tmp_path: Path) -> None:
    mirror_dir = tmp_path / "mirror"
    catalog_path = mirror_dir / "train.csv"
    _create_synthetic_catalog(catalog_path, mirror_dir, num_signs=5, samples_per_sign=6)

    selection_dir = tmp_path / "selection"
    selection_dir.mkdir(parents=True)
    glosses_csv = selection_dir / "glosses.csv"
    glosses_csv.write_text(
        "gloss,label,n_samples,n_participants,max_dominance,hand_coverage,motion_energy,composite_score\n"
        "sign_00,0,6,4,0.25,1.0,0.1,0.9\n"
        "sign_01,1,6,4,0.25,1.0,0.1,0.85\n",
        encoding="utf-8",
    )

    overrides = [
        f"data.local_mirror_dir={mirror_dir}",
        f"output.selection_dir={selection_dir}",
        f"output.output_dir={tmp_path / 'out'}",
        "data.max_files_per_gloss=2",
    ]
    cfg = load_config(overrides=overrides)
    paths = resolve_paths(ENV_LOCAL, cfg, create_dirs=True)

    manifest_path, records = download_target_gloss_data(cfg=cfg, paths=paths)

    assert len(records) == 4  # 2 signs * 2 files capped


def test_download_budget_total_files(tmp_path: Path) -> None:
    mirror_dir = tmp_path / "mirror"
    catalog_path = mirror_dir / "train.csv"
    _create_synthetic_catalog(catalog_path, mirror_dir, num_signs=5, samples_per_sign=6)

    selection_dir = tmp_path / "selection"
    selection_dir.mkdir(parents=True)
    glosses_csv = selection_dir / "glosses.csv"
    glosses_csv.write_text(
        "gloss,label,n_samples,n_participants,max_dominance,hand_coverage,motion_energy,composite_score\n"
        "sign_00,0,6,4,0.25,1.0,0.1,0.9\n"
        "sign_01,1,6,4,0.25,1.0,0.1,0.85\n",
        encoding="utf-8",
    )

    overrides = [
        f"data.local_mirror_dir={mirror_dir}",
        f"output.selection_dir={selection_dir}",
        f"output.output_dir={tmp_path / 'out'}",
        "data.download_budget_total_files=3",
    ]
    cfg = load_config(overrides=overrides)
    paths = resolve_paths(ENV_LOCAL, cfg, create_dirs=True)

    manifest_path, records = download_target_gloss_data(cfg=cfg, paths=paths)

    assert len(records) == 3


def test_download_resilience_on_single_file_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mirror_dir = tmp_path / "mirror"
    catalog_path = mirror_dir / "train.csv"
    _create_synthetic_catalog(catalog_path, mirror_dir, num_signs=3, samples_per_sign=2)

    selection_dir = tmp_path / "selection"
    selection_dir.mkdir(parents=True)
    (selection_dir / "glosses.csv").write_text(
        "gloss,label,n_samples,n_participants,max_dominance,hand_coverage,motion_energy,composite_score\n"
        "sign_00,0,2,2,0.5,1.0,0.1,0.9\n",
        encoding="utf-8",
    )

    # Make the second file fail during download
    from src.data import download
    original_fetch = download.fetch_file

    def mock_fetch(remote_path, **kwargs):
        if "seq_0_1" in remote_path:
            raise IOError("Simulated network timeout for sequence 1")
        return original_fetch(remote_path, **kwargs)

    monkeypatch.setattr(download, "fetch_file", mock_fetch)

    overrides = [
        f"data.local_mirror_dir={mirror_dir}",
        f"output.selection_dir={selection_dir}",
        f"output.output_dir={tmp_path / 'out'}",
        "data.retry=1",
    ]
    cfg = load_config(overrides=overrides)
    paths = resolve_paths(ENV_LOCAL, cfg, create_dirs=True)

    manifest_path, records = download_target_gloss_data(cfg=cfg, paths=paths)

    assert len(records) == 2
    successes = [r for r in records if r.status != "failed"]
    failures = [r for r in records if r.status == "failed"]
    assert len(successes) == 1
    assert len(failures) == 1
    assert "Simulated network timeout" in failures[0].error


def test_download_missing_credentials_raises_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir(parents=True)
    (selection_dir / "glosses.csv").write_text(
        "gloss,label,n_samples,n_participants,max_dominance,hand_coverage,motion_energy,composite_score\n"
        "sign_00,0,2,2,0.5,1.0,0.1,0.9\n",
        encoding="utf-8",
    )

    # Mock fetch_file for train.csv
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "path,participant_id,sequence_id,sign\n"
        "train_landmark_files/1/100.parquet,1,100,sign_00\n",
        encoding="utf-8",
    )

    from src.data import download

    def mock_fetch(remote_path, **kwargs):
        if remote_path == "train.csv":
            return FetchResult(path=train_csv, source="cache")
        raise KaggleCredentialsNotFoundError("No credentials")

    monkeypatch.setattr(download, "fetch_file", mock_fetch)

    overrides = [
        f"output.selection_dir={selection_dir}",
        f"output.output_dir={tmp_path / 'out'}",
        "data.retry=0",
    ]
    cfg = load_config(overrides=overrides)
    paths = resolve_paths(ENV_LOCAL, cfg, create_dirs=True)

    with pytest.raises(KaggleCredentialsNotFoundError, match="KAGGLE CREDENTIALS NOT FOUND"):
        download_target_gloss_data(cfg=cfg, paths=paths)


def test_download_rules_not_accepted_raises_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir(parents=True)
    (selection_dir / "glosses.csv").write_text(
        "gloss,label,n_samples,n_participants,max_dominance,hand_coverage,motion_energy,composite_score\n"
        "sign_00,0,2,2,0.5,1.0,0.1,0.9\n",
        encoding="utf-8",
    )

    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "path,participant_id,sequence_id,sign\n"
        "train_landmark_files/1/100.parquet,1,100,sign_00\n",
        encoding="utf-8",
    )

    from src.data import download

    def mock_fetch(remote_path, **kwargs):
        if remote_path == "train.csv":
            return FetchResult(path=train_csv, source="cache")
        raise KaggleRulesNotAcceptedError("HTTP 403 Forbidden")

    monkeypatch.setattr(download, "fetch_file", mock_fetch)

    overrides = [
        f"output.selection_dir={selection_dir}",
        f"output.output_dir={tmp_path / 'out'}",
        "data.retry=0",
    ]
    cfg = load_config(overrides=overrides)
    paths = resolve_paths(ENV_LOCAL, cfg, create_dirs=True)

    with pytest.raises(KaggleRulesNotAcceptedError, match="KAGGLE COMPETITION RULES NOT ACCEPTED"):
        download_target_gloss_data(cfg=cfg, paths=paths)


def test_cli_prepare_data_download_stage(tmp_path: Path) -> None:
    mirror_dir = tmp_path / "mirror"
    catalog_path = mirror_dir / "train.csv"
    _create_synthetic_catalog(catalog_path, mirror_dir, num_signs=10, samples_per_sign=12, num_participants=6)

    overrides = [
        f"env.local_root={tmp_path}",
        f"data.local_mirror_dir={mirror_dir}",
        "selection.n_glosses=5",
        "selection.n_candidates=10",
        "data.max_files_per_gloss=2",
        f"output.output_dir={tmp_path / 'out'}",
        f"output.selection_dir={tmp_path / 'selection'}",
    ]

    ret = run_pipeline(stage="download", overrides=overrides)
    assert ret == 0
    assert (tmp_path / "selection" / "glosses.csv").is_file()
    assert (tmp_path / "out" / "resolved_config.yaml").is_file()
