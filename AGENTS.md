# ISLR Project Invariants & Engineering Guidelines

## 1. Single Source of Truth & Zero Config Aliases
- Every configuration concept has exactly ONE canonical key. Do not maintain alias or duplicate keys (e.g. `competition_id` only, no `competition_slug`; `report_dir` only, no `reports_dir`).
- Every tunable must live in `configs/islr.yaml` and map directly to dataclasses in `src/utils/config.py`.
- Any removed, deprecated, or unrecognized key must fail loudly with `KeyError` during config loading and `--set` overrides.
- Overriding any canonical key must propagate deterministically across all downstream modules.

## 2. Credential Precedence & Import Isolation
- Credential precedence order: Colab user secrets (`google.colab.userdata`), environment variables, `~/.kaggle/kaggle.json`, interactive prompt fallback.
- Support modern authentication token formats (`KAGGLE_API_TOKEN`) alongside traditional credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`).
- Validate credential presence **before** importing external API packages. If credentials are missing, raise explicit exceptions (`KaggleCredentialsNotFoundError`) immediately.
- Never log, print, or leak raw keys or tokens.
- Wrap external third-party SDK imports in dedicated helper functions (e.g., `_import_kaggle_api()`) so unit tests can run in environments without third-party packages installed.

## 3. Canonical Path Resolution Semantics (`env.resolve_paths`)
- Canonical paths are resolved based on `detect_env()` (`local`, `colab`, `kaggle`):
  - Kaggle: Data from `/kaggle/input/<competition_id>`, outputs to `/kaggle/working`, cache under `/kaggle/temp/<cache_dir>` (or absolute as-is).
  - Colab with Drive: Persistent root at `/content/drive/MyDrive/<project>`, cache under `persistent_root/<cache_dir>`.
  - Colab without Drive: Persistent root at `/content`, cache under `/content/<cache_dir>`.
  - Local: Repo root detected via package location, cache under `repo_root/<cache_dir>`.
- If `data.cache_dir` or any configured directory path is absolute, honor it as-is without re-rooting.

## 4. Test Suite Integrity
- Tests must be completely self-contained. No test may depend on untracked/gitignored data or pre-existing files on disk.
- Unit tests must pass 100% in minimal environments where optional client libraries (like `kaggle`) are not installed.
- All code must pass `pyflakes` cleanly with zero unused imports or undefined variables.
