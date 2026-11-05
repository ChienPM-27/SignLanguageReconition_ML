# Graph Report - SignLanguageReconition_ML  (2026-11-05)

## Corpus Check
- 68 files · ~61,959 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 24 file(s) not represented in the graph (top: (none) 22, .ipynb 1, .ini 1)

## Summary
- 1167 nodes · 2896 edges · 85 communities (52 shown, 33 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 211 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `eb3c1790`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_augmentation.py
- csv
- test_islr_schema.py
- pytest
- test_setup_check.py
- pathlib
- config.py
- get_kaggle_credentials
- ISRLandmarkDataset
- pathlib
- numpy
- fetch_file
- augmentation
- evaluate.py
- train_gru
- set_seed
- 20_train.py
- torch
- test_train.py
- training
- resolve_paths
- 40_eval_samples.py
- data
- MediaPipeHandExtractor
- test_inference.py
- islr_schema.py
- load_config
- selection
- pathlib
- test_selection.py
- TemporalAttention
- ISLR Project Invariants & Engineering Guidelines
- eda
- preprocess
- weights
- split
- ISLRPredictor
- env
- model
- Path
- islr.yaml
- output
- Google Isolated Sign Language Recognition (ISLR)
- read_catalog_csv
- Báo cáo kỹ thuật: Sửa Augmentation & Cấu hình Huấn luyện Tổng quát hóa (Phase A)
- Priority 0 (P0) Validation & Verification Report
- batch_horizontal_flip
- test_fetch_file_flattening_normalization
- schema.py
- _is_string_type
- argparse
- test_config.py
- test_fetch_file_http_403_error
- test_fetch_file_http_401_unauthorized_error
- test_fetch_file_windows_backslash_normalization
- get_credential_status
- test_smoke_train_2_epochs_with_augmentation
- test_fetch_file_stem_zip_extraction
- authenticate_kaggle_api
- load_and_validate_feature_file
- Q: đề xuất cho tôi config training
- test_fetch_file_zip_extraction
- test_colab_files_upload_fallback
- 1. Tóm tắt Diff theo File
- test_augmentation_config_and_overrides
- test_removed_config_keys_raise_clear_unknown_key_error
- test_unknown_inner_dict_key_raises
- test_seed_boolean_type_rejection
- test_missing_required_landmark_column_raises_error
- test_expected_columns_duplicate_raises_value_error
- test_training_preload_config
- test_invalid_log_level

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 82 edges
2. `resolve_paths()` - 39 edges
3. `Config` - 37 edges
4. `fetch_file()` - 35 edges
5. `ISLRPredictor` - 34 edges
6. `ISRLandmarkDataset` - 33 edges
7. `apply_augmentation_batch()` - 26 edges
8. `get_logger()` - 23 edges
9. `apply_augmentation()` - 22 edges
10. `download_target_gloss_data()` - 22 edges

## Surprising Connections (you probably didn't know these)
- `3. Rủi ro & Điểm Chưa Chắc Chắn` --references--> `time_stretch()`  [INFERRED]
  reports/training_fix_notes.md → src/data/augmentation.py
- `Executive Summary` --references--> `prepare_landmark_sequence()`  [INFERRED]
  reports/p0_validation_report.md → src/features/pipeline.py
- `P0.1 — Freeze the feature contract and add synthetic fixtures` --references--> `get_feature_dim()`  [INFERRED]
  reports/p0_implementation_plan.md → src/features/schema.py
- `3.2. Migration Rules` --references--> `ISLRPredictor`  [INFERRED]
  reports/p0_validation_report.md → src/inference/predictor.py
- `3. Canonical Path Resolution Semantics (`env.resolve_paths`)` --references--> `resolve_paths()`  [INFERRED]
  AGENTS.md → src/utils/env.py

## Import Cycles
- None detected.

## Communities (85 total, 33 thin omitted)

### Community 0 - "test_augmentation.py"
Cohesion: 0.16
Nodes (18): P0.5 — Make spatial augmentation outputs internally consistent, Chi tiết các kiểm thử trong `tests/test_augmentation.py`:, P0.5: Đồng bộ Tính Nhất quán Động học cho Augmentation (Internal Motion Consistency), `src/data/augmentation.py` (Numpy Pipeline), add_jitter(), apply_augmentation(), apply_random_augmentation(), _get_aug_param() (+10 more)

### Community 1 - "csv"
Cohesion: 0.15
Nodes (11): create_signer_splits(), _find_valid_holdout_partition(), _generate_split_report(), _partition_group_kfold(), verify_no_participant_leakage(), _populate_features_manifest(), test_cli_prepare_data_all_stages(), test_create_signer_splits_group_kfold() (+3 more)

### Community 2 - "test_islr_schema.py"
Cohesion: 0.15
Nodes (13): generate_schema_markdown_report(), _create_mock_catalog(), _create_mock_parquet(), test_inspect_catalog_csv_column_mismatch(), test_inspect_catalog_csv_valid(), test_inspect_parquet_file_column_mismatch(), test_inspect_parquet_file_hand_presence_exact_fractions(), test_inspect_parquet_file_valid() (+5 more)

### Community 3 - "pytest"
Cohesion: 0.13
Nodes (7): test_empty_key_path_override(), test_invalid_env_name(), test_invalid_override_format(), test_local_mirror_dir_whitespace_raises_value_error(), test_range_validation_sample_budget(), test_training_extended_configs(), test_unknown_key_in_override()

### Community 4 - "test_setup_check.py"
Cohesion: 0.11
Nodes (18): verify_parquet_schema(), test_read_catalog_csv(), test_run_setup_check_end_to_end(), test_run_setup_check_probes_multiple_files_budget(), test_sample_catalog_records_random(), test_sample_catalog_records_spread_over_participants(), test_verify_parquet_schema_accepts_str_path(), test_verify_parquet_schema_dictionary_encoded_string_column() (+10 more)

### Community 5 - "pathlib"
Cohesion: 0.08
Nodes (12): detect_environment(), test_detect_colab_environment(), test_detect_kaggle_environment(), test_detect_local_environment(), test_resolve_paths_cache_dir_absolute_and_relative(), test_resolve_paths_colab_with_drive(), test_resolve_paths_colab_without_drive(), test_resolve_paths_invalid_environment() (+4 more)

### Community 6 - "config.py"
Cohesion: 0.07
Nodes (19): main(), run_setup_check(), sample_catalog_records(), _apply_override(), _build_config_from_dict(), Config, DataConfig, EDAConfig (+11 more)

### Community 7 - "get_kaggle_credentials"
Cohesion: 0.10
Nodes (8): get_kaggle_credentials(), KaggleCredentials, test_credential_precedence_colab(), test_credential_precedence_colab_api_token(), test_credential_precedence_env_vars(), test_credential_precedence_interactive(), test_credential_precedence_kaggle_config_dir(), test_credential_precedence_kaggle_json()

### Community 8 - "ISRLandmarkDataset"
Cohesion: 0.11
Nodes (14): extract_rf_features(), create_dataloader(), ISRLandmarkDataset, DeviceDataLoader, _create_synthetic_split_dataset(), test_dataset_dataloader_batching(), test_dataset_loading_clean(), test_dataset_non_strict_filters_with_count() (+6 more)

### Community 9 - "pathlib"
Cohesion: 0.10
Nodes (9): main(), parse_args(), ResolvedPaths, get_logger(), test_get_logger_case_insensitive_level(), test_get_logger_custom_name_and_level(), test_get_logger_default(), test_get_logger_handler_reuse() (+1 more)

### Community 10 - "numpy"
Cohesion: 0.05
Nodes (41): P0.1 — Freeze the feature contract and add synthetic fixtures, P0.2 — Repair bounded interpolation and temporal resampling, P0.3 — Gate derived motion on valid endpoints, P0.6 — Share preprocessing across training and inference, P0.8 — Regression checks and corrected-baseline handoff, Priority 0 implementation plan, P0.2 & P0.3: Sửa Bounded Interpolation, Temporal Resampling & Gating Vận tốc, _generate_preprocess_report() (+33 more)

### Community 11 - "fetch_file"
Cohesion: 0.09
Nodes (12): test_credential_precedence_env_api_token(), test_fetch_file_absent_credentials_raises_error(), test_fetch_file_corrupt_zip_cleanup(), test_fetch_file_dest_keyword_argument(), test_fetch_file_destination_dir_named_same_as_file(), test_fetch_file_destination_with_dot_in_directory_name(), test_fetch_file_direct_target_filepath(), test_fetch_file_empty_remote_path_raises_value_error() (+4 more)

### Community 12 - "augmentation"
Cohesion: 0.07
Nodes (27): augmentation, flip, jitter, rotate, scale_translate, shape_scale, time_stretch, enabled (+19 more)

### Community 13 - "evaluate.py"
Cohesion: 0.12
Nodes (7): evaluate_test_set(), calculate_per_signer_accuracy(), compute_confusion_matrix(), compute_macro_f1(), compute_top_k_accuracy(), plot_confusion_matrix(), plot_learning_curves()

### Community 14 - "train_gru"
Cohesion: 0.17
Nodes (7): P0.7 — Bind caches and checkpoints to preprocessing semantics, create_scheduler(), evaluate_epoch(), load_checkpoint(), save_checkpoint(), train_gru(), train_one_epoch()

### Community 15 - "set_seed"
Cohesion: 0.10
Nodes (10): get_predictor(), predict_video_with_dominance(), main(), set_seed(), test_set_seed_boolean_type_rejection(), test_set_seed_custom_value(), test_set_seed_default(), test_set_seed_negative_raises_value_error() (+2 more)

### Community 16 - "20_train.py"
Cohesion: 0.16
Nodes (9): device, get_git_commit_hash(), load_class_metadata(), main(), run_baseline_rf(), run_overfit_check(), run_training_pipeline(), select_device() (+1 more)

### Community 17 - "torch"
Cohesion: 0.11
Nodes (19): 5. Adversarial Review & Robustness Hardening, `src/data/augmentation_gpu.py` (PyTorch Batch GPU/CPU Pipeline), apply_augmentation_batch(), batch_add_jitter(), batch_random_rotate_2d(), batch_recompute_derived_features(), batch_scale_translate(), batch_shape_scale() (+11 more)

### Community 18 - "test_train.py"
Cohesion: 0.23
Nodes (13): get_feature_dim(), ASLGRUClassifier, _setup_synthetic_train_environment(), test_amp_gradscaler_initialization(), test_cli_train_pipeline_end_to_end(), test_create_scheduler_types(), test_device_loader_train_one_epoch_smoke(), test_early_stopping_min_delta_and_min_epochs() (+5 more)

### Community 19 - "training"
Cohesion: 0.10
Nodes (20): factor, patience, type, training, amp, batch_size, early_stopping_min_delta, early_stopping_min_epochs (+12 more)

### Community 20 - "resolve_paths"
Cohesion: 0.10
Nodes (21): run_pipeline(), download_target_gloss_data(), DownloadRecord, _fetch_single_file_with_retry(), fetch_file(), FetchResult, KaggleAccessError, KaggleCredentialsNotFoundError (+13 more)

### Community 21 - "40_eval_samples.py"
Cohesion: 0.18
Nodes (6): apply_mirror_x(), evaluate_samples(), extract_video_landmarks(), format_top_k(), main(), parse_args()

### Community 22 - "data"
Cohesion: 0.12
Nodes (16): data, cache_dir, competition_id, download_budget_total_files, landmark_files_dir, local_mirror_dir, max_files_per_gloss, num_download_workers (+8 more)

### Community 24 - "test_inference.py"
Cohesion: 0.14
Nodes (10): PredictionResult, build_config_from_dict(), _create_mock_model_and_glosses(), test_checkpoint_resolved_config_drives_parameters(), test_legacy_checkpoint_blocks_raw_coordinates_inference(), test_mismatched_model_input_size_raises(), test_predictor_160_dims_and_pose_support(), test_predictor_initialization_and_prediction() (+2 more)

### Community 25 - "islr_schema.py"
Cohesion: 0.14
Nodes (7): CatalogSummary, format_nested_path_behavior(), inspect_catalog_csv(), inspect_parquet_file(), ParquetSummary, SchemaVerificationReport, verify_schema()

### Community 26 - "load_config"
Cohesion: 0.13
Nodes (7): load_config(), test_boundary_values_seed_zero(), test_every_kept_key_override_updates_code_values(), test_list_override_works(), test_nested_dict_override_works(), test_null_local_mirror_dir(), test_override_non_dict_field_with_extra_tokens_raises()

### Community 27 - "selection"
Cohesion: 0.17
Nodes (12): max_participant_dominance, min_participants, min_samples, selection, candidate_prefilter, diversity_method, exclude_signs, force_include_signs (+4 more)

### Community 28 - "pathlib"
Cohesion: 0.16
Nodes (5): draw_hand_landmarks(), draw_hud(), run_realtime(), download_sample_videos(), main()

### Community 29 - "test_selection.py"
Cohesion: 0.20
Nodes (11): CandidateStats, compute_parquet_hand_metrics(), read_catalog_records(), select_20_glosses(), _write_selection_report(), _create_synthetic_catalog(), test_cli_prepare_data_selection_stage(), test_compute_parquet_hand_metrics() (+3 more)

### Community 30 - "TemporalAttention"
Cohesion: 0.13
Nodes (4): ASLLSTMClassifier, create_model(), TemporalAttention, test_model_pooling_and_lstm_variants()

### Community 31 - "ISLR Project Invariants & Engineering Guidelines"
Cohesion: 0.18
Nodes (6): 1. Single Source of Truth & Zero Config Aliases, 2. Credential Precedence & Import Isolation, 3. Canonical Path Resolution Semantics (`env.resolve_paths`), 4. Test Suite Integrity, ISLR Project Invariants & Engineering Guidelines, _import_kaggle_api()

### Community 32 - "eda"
Cohesion: 0.25
Nodes (8): eda, expected_columns, max_frames, min_frames, min_hand_coverage, num_expected_landmarks, num_landmark_probe_files, top_n_plot

### Community 33 - "preprocess"
Cohesion: 0.22
Nodes (9): preprocess, dominant_hand, max_gap_frames, min_valid_frames, normalization_epsilon, num_workers, pad_frames_around, sequence_length (+1 more)

### Community 34 - "weights"
Cohesion: 0.25
Nodes (8): weights, hand_coverage, handshape_diversity, motion_energy, participant_balance, participant_coverage, samples, separability

### Community 35 - "split"
Cohesion: 0.25
Nodes (8): split, method, min_per_gloss_per_split, n_folds, n_test, n_train, n_val, seed

### Community 36 - "ISLRPredictor"
Cohesion: 0.32
Nodes (3): 6.3. Quy tắc Di chuyển (Migration Rules) cho Caches & Checkpoints, P0.6: Dùng Chung Pipeline Tiền Xử lý giữa Huấn luyện và Suy luận, ISLRPredictor

### Community 37 - "env"
Cohesion: 0.29
Nodes (7): env, local_root, log_level, name, project_dir_name, seed, use_drive

### Community 38 - "model"
Cohesion: 0.29
Nodes (7): model, bidirectional, dropout, hidden_size, name, num_layers, pooling

### Community 39 - "Path"
Cohesion: 0.13
Nodes (7): test_deprecated_features_use_pose_raises_vietnamese_key_error(), test_empty_expected_columns_raises_error(), test_invalid_yaml_syntax_in_file_raises_value_error(), test_override_on_non_dict_section_raises_type_error(), test_save_and_reload_config(), test_unknown_field_in_section(), test_unknown_top_level_section()

### Community 40 - "islr.yaml"
Cohesion: 0.33
Nodes (5): features, set, run, name, output_dir

### Community 41 - "output"
Cohesion: 0.33
Nodes (6): output, output_dir, overwrite, report_dir, save_resolved_config, selection_dir

### Community 42 - "Google Isolated Sign Language Recognition (ISLR)"
Cohesion: 0.33
Nodes (5): 1. Mục tiêu và Kiến trúc, 2. Cấu trúc thư mục, 3. Cài đặt và Chạy kiểm thử, 4. Quy trình thực thi (Config-Driven Workflow), Google Isolated Sign Language Recognition (ISLR)

### Community 43 - "read_catalog_csv"
Cohesion: 0.18
Nodes (5): read_catalog_csv(), test_read_catalog_csv_accepts_str_path(), test_read_catalog_csv_empty_path_raises_value_error(), test_read_catalog_csv_missing_column(), test_read_catalog_csv_missing_metadata_field_raises_value_error()

### Community 44 - "Báo cáo kỹ thuật: Sửa Augmentation & Cấu hình Huấn luyện Tổng quát hóa (Phase A)"
Cohesion: 0.18
Nodes (10): 2. Kết quả Pytest & Pyflakes, 3. Rủi ro & Điểm Chưa Chắc Chắn, 4. Ba Lệnh Huấn luyện Thực nghiệm Trên Kaggle (3 Seeds), 5. Bảng Baseline Đối Chứng, 6.1. Mục tiêu và Nguyên nhân Thực hiện, 6. Giai đoạn P0: Chuẩn hóa Toàn diện Preprocessing, Augmentation & Inference Parity (P0.1 - P0.8), Báo cáo kỹ thuật: Sửa Augmentation & Cấu hình Huấn luyện Tổng quát hóa (Phase A), R1: Chỉ bật Flip (Flip-Only) (+2 more)

### Community 45 - "Priority 0 (P0) Validation & Verification Report"
Cohesion: 0.25
Nodes (7): 1.1. Targeted P0 Test Suite, 1.2. Complete Repository Test Suite, 1.3. Static Analysis & Code Cleanliness, 1. Test Execution & Regression Matrix, 4. Architectural Readiness & Next Steps, Executive Summary, Priority 0 (P0) Validation & Verification Report

### Community 46 - "batch_horizontal_flip"
Cohesion: 0.12
Nodes (13): P0.4 — Correct normalized-pose reflection in both backends, Scope and verified starting points, 2. Mathematical Acceptance Proofs & Verification, 6.2. Chi tiết các Thay đổi Kỹ thuật, P0.1: Đóng băng Contract Đặc trưng & Schema Versioning, P0.4: Chuẩn hóa Phép Lật Pose (Pose Reflection) trong NumPy & PyTorch, P0.7: Gắn Kết Caches và Checkpoints với Preprocessing Identity, batch_horizontal_flip() (+5 more)

### Community 48 - "schema.py"
Cohesion: 0.16
Nodes (5): 3.1. Preprocessing Identity Definition, 3.2. Migration Rules, 3. Schema & Artifact Migration Specification, compute_preprocess_hash(), compute_preprocess_identity()

### Community 64 - "test_config.py"
Cohesion: 0.14
Nodes (6): test_cli_overrides_application(), test_invalid_yaml_syntax_in_override_raises_value_error(), test_load_default_config(), test_parse_cli_args(), test_probe_files_exceeding_budget_raises_value_error(), test_range_validation_seed()

### Community 68 - "get_credential_status"
Cohesion: 0.18
Nodes (5): get_credential_status(), test_credential_status_reporting(), test_get_credential_status_does_not_prompt_in_interactive_tty(), test_get_kaggle_credentials_non_dict_json_handled_safely(), test_get_kaggle_credentials_whitespace_only_env_vars_treated_as_absent()

### Community 71 - "authenticate_kaggle_api"
Cohesion: 0.18
Nodes (5): authenticate_kaggle_api(), test_authenticate_kaggle_api_absent_credentials_raises(), test_authenticate_with_mocked_import_kaggle_api(), test_check_credentials_happens_before_kaggle_import(), test_missing_kaggle_package_raises_import_error()

### Community 73 - "Q: đề xuất cho tôi config training"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: đề xuất cho tôi config training, Source Nodes

### Community 76 - "1. Tóm tắt Diff theo File"
Cohesion: 0.67
Nodes (3): 1. Tóm tắt Diff theo File, `configs/islr.yaml`, `src/utils/config.py`

## Knowledge Gaps
- **33 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `training` connect `training` to `islr.yaml`, `20_train.py`?**
  _High betweenness centrality (0.206) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `resolve_paths()` (e.g. with `3. Canonical Path Resolution Semantics (`env.resolve_paths`)` and `Config`) actually correct?**
  _`resolve_paths()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `csv` be split into smaller, more focused modules?**
  _Cohesion score 0.14624505928853754 - nodes in this community are weakly interconnected._
- **Why does `device` connect `20_train.py` to `load_and_validate_feature_file`, `evaluate.py`, `train_gru`, `torch`, `training`?**
  _High betweenness centrality (0.204) - this node is a cross-community bridge._
- **Should `test_islr_schema.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1477832512315271 - nodes in this community are weakly interconnected._
- **Why does `load_config()` connect `load_config` to `test_augmentation.py`, `csv`, `pytest`, `test_setup_check.py`, `pathlib`, `config.py`, `pathlib`, `numpy`, `20_train.py`, `test_train.py`, `resolve_paths`, `test_selection.py`, `Path`, `Báo cáo kỹ thuật: Sửa Augmentation & Cấu hình Huấn luyện Tổng quát hóa (Phase A)`, `test_config.py`, `test_smoke_train_2_epochs_with_augmentation`, `test_augmentation_config_and_overrides`, `test_removed_config_keys_raise_clear_unknown_key_error`, `test_unknown_inner_dict_key_raises`, `test_seed_boolean_type_rejection`, `test_missing_required_landmark_column_raises_error`, `test_expected_columns_duplicate_raises_value_error`, `test_training_preload_config`, `test_invalid_log_level`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **Should `pytest` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._