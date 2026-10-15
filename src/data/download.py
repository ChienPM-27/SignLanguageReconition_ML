"""ISLR Dataset Downloader for target 20 glosses.

Downloads required parquet files for selected glosses with retry/backoff,
concurrent workers, idempotent local mirror and cache checking,
detailed manifest logging, and informative authentication failure messages.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Optional, Set, Tuple

from src.data.kaggle_access import (
    KaggleCredentialsNotFoundError,
    KaggleRulesNotAcceptedError,
    fetch_file,
)
from src.data.selection import read_catalog_records, select_20_glosses
from src.utils.config import Config
from src.utils.env import ResolvedPaths
from src.utils.logging import get_logger

logger = get_logger("islr.download")

HELP_MSG_CREDENTIALS = (
    "\n"
    "================================================================================\n"
    "KAGGLE CREDENTIALS NOT FOUND OR AUTHENTICATION FAILED:\n"
    "- Google Colab: Click the Key icon (Secrets) in the left panel,\n"
    "  add 'KAGGLE_USERNAME' and 'KAGGLE_KEY' (or 'KAGGLE_API_TOKEN'),\n"
    "  and toggle 'Notebook access' ON.\n"
    "- Kaggle Notebooks: Add credentials under Add-ons -> Secrets.\n"
    "- Local Environment: Place your credentials in ~/.kaggle/kaggle.json:\n"
    "  {\"username\": \"YOUR_USERNAME\", \"key\": \"YOUR_API_KEY\"}\n"
    "  or export KAGGLE_API_TOKEN / KAGGLE_USERNAME & KAGGLE_KEY in your shell.\n"
    "================================================================================\n"
)

HELP_MSG_RULES = (
    "\n"
    "================================================================================\n"
    "KAGGLE COMPETITION RULES NOT ACCEPTED (HTTP 403 Forbidden):\n"
    "You must accept the competition rules before Kaggle allows file downloads.\n"
    "Please open your browser to:\n"
    "  https://www.kaggle.com/competitions/{comp_id}/rules\n"
    "and click 'I Understand and Accept'.\n"
    "================================================================================\n"
)


@dataclass
class DownloadRecord:
    sequence_id: str
    participant_id: str
    gloss: str
    remote_path: str
    local_path: str
    size_bytes: int
    status: str
    error: str = ""


def _fetch_single_file_with_retry(
    record: Dict[str, str],
    dest_dir: Path,
    local_mirror_dir: Optional[str],
    competition_id: str,
    max_retries: int,
) -> DownloadRecord:
    remote_path = record["path"]
    seq_id = record["sequence_id"]
    part_id = record["participant_id"]
    gloss = record["sign"]

    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            fetch_res = fetch_file(
                remote_path=remote_path,
                dest_dir=dest_dir,
                local_mirror_dir=local_mirror_dir,
                competition_id=competition_id,
            )
            local_file = fetch_res.path
            file_size = local_file.stat().st_size if local_file.is_file() else 0
            return DownloadRecord(
                sequence_id=seq_id,
                participant_id=part_id,
                gloss=gloss,
                remote_path=remote_path,
                local_path=str(local_file),
                size_bytes=file_size,
                status=fetch_res.source,
                error="",
            )
        except (KaggleCredentialsNotFoundError, KaggleRulesNotAcceptedError) as auth_err:
            # Fatal authentication/rules error, do not retry blindly
            last_error = str(auth_err)
            break
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                backoff_sec = min(0.5 * (2 ** attempt), 4.0)
                time.sleep(backoff_sec)

    return DownloadRecord(
        sequence_id=seq_id,
        participant_id=part_id,
        gloss=gloss,
        remote_path=remote_path,
        local_path="",
        size_bytes=0,
        status="failed",
        error=last_error,
    )


def download_target_gloss_data(
    cfg: Config,
    paths: ResolvedPaths,
) -> Tuple[Path, List[DownloadRecord]]:
    """Download landmark parquet files for selected glosses according to config budget."""
    dest_data_dir = paths.data_dir
    if not paths.is_kaggle_input_readonly:
        dest_data_dir.mkdir(parents=True, exist_ok=True)
    selection_dir = paths.selection_dir
    selection_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure selection glosses.csv exists
    glosses_csv = selection_dir / "glosses.csv"
    if not glosses_csv.is_file():
        logger.info(f"glosses.csv not found at {glosses_csv}. Running selection first.")
        select_20_glosses(cfg=cfg, paths=paths)

    selected_glosses: Set[str] = set()
    with open(glosses_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            selected_glosses.add(row["gloss"].strip())

    logger.info(f"Targeting {len(selected_glosses)} selected glosses for data download.")

    # 2. Read catalog records
    train_csv_res = fetch_file(
        remote_path=cfg.data.train_csv_name,
        dest_dir=dest_data_dir,
        local_mirror_dir=cfg.data.local_mirror_dir,
        competition_id=cfg.data.competition_id,
    )
    all_records = read_catalog_records(train_csv_res.path)

    # 3. Filter for selected glosses
    target_records = [r for r in all_records if r["sign"] in selected_glosses]
    logger.info(f"Found {len(target_records)} sequences in catalog matching selected glosses.")

    # 4. Apply max_files_per_gloss cap if configured
    if cfg.data.max_files_per_gloss is not None:
        cap = cfg.data.max_files_per_gloss
        by_gloss: Dict[str, List[Dict[str, str]]] = {}
        for r in target_records:
            by_gloss.setdefault(r["sign"], []).append(r)
        capped_records: List[Dict[str, str]] = []
        for g in sorted(by_gloss.keys()):
            capped_records.extend(by_gloss[g][:cap])
        target_records = capped_records
        logger.info(f"Applied max_files_per_gloss={cap}: {len(target_records)} sequences remaining.")

    # 5. Apply download_budget_total_files cap if configured
    if cfg.data.download_budget_total_files is not None:
        budget = cfg.data.download_budget_total_files
        target_records = target_records[:budget]
        logger.info(f"Applied download_budget_total_files={budget}: {len(target_records)} sequences remaining.")

    # 6. Execute downloads (parallel or sequential)
    num_workers = max(1, cfg.data.num_download_workers)
    max_retries = max(0, cfg.data.retry)
    results: List[DownloadRecord] = []

    logger.info(f"Starting download of {len(target_records)} files using {num_workers} workers.")

    auth_rule_error_encountered: Optional[str] = None

    if num_workers > 1 and len(target_records) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_rec = {
                executor.submit(
                    _fetch_single_file_with_retry,
                    rec,
                    dest_data_dir,
                    cfg.data.local_mirror_dir,
                    cfg.data.competition_id,
                    max_retries,
                ): rec
                for rec in target_records
            }
            for future in as_completed(future_to_rec):
                res = future.result()
                results.append(res)
                if "HTTP 403" in res.error or "rules" in res.error.lower():
                    auth_rule_error_encountered = HELP_MSG_RULES.format(comp_id=cfg.data.competition_id)
                elif "credentials" in res.error.lower():
                    auth_rule_error_encountered = HELP_MSG_CREDENTIALS
    else:
        for rec in target_records:
            res = _fetch_single_file_with_retry(
                rec,
                dest_data_dir,
                cfg.data.local_mirror_dir,
                cfg.data.competition_id,
                max_retries,
            )
            results.append(res)
            if "HTTP 403" in res.error or "rules" in res.error.lower():
                auth_rule_error_encountered = HELP_MSG_RULES.format(comp_id=cfg.data.competition_id)
            elif "credentials" in res.error.lower():
                auth_rule_error_encountered = HELP_MSG_CREDENTIALS

    # Log summary
    success_count = sum(1 for r in results if r.status != "failed")
    failed_count = len(results) - success_count
    logger.info(f"Download complete: {success_count} succeeded, {failed_count} failed out of {len(results)}.")

    if auth_rule_error_encountered and success_count == 0:
        logger.error(auth_rule_error_encountered)
        if "HTTP 403" in auth_rule_error_encountered:
            raise KaggleRulesNotAcceptedError(auth_rule_error_encountered)
        raise KaggleCredentialsNotFoundError(auth_rule_error_encountered)

    # 7. Write download manifest
    manifest_dir = paths.persistent_root if paths.is_kaggle_input_readonly else paths.data_dir
    manifest_path = manifest_dir / "download_manifest.csv"
    with open(manifest_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sequence_id",
            "participant_id",
            "gloss",
            "remote_path",
            "local_path",
            "size_bytes",
            "status",
            "error",
        ])
        for r in sorted(results, key=lambda x: (x.gloss, x.sequence_id)):
            writer.writerow([
                r.sequence_id,
                r.participant_id,
                r.gloss,
                r.remote_path,
                r.local_path,
                r.size_bytes,
                r.status,
                r.error,
            ])

    logger.info(f"Saved download manifest to {manifest_path}")
    return manifest_path, results
