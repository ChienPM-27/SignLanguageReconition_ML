---
type: "query"
date: "2026-11-05T02:34:36.850798+00:00"
question: "đề xuất cho tôi config training"
contributor: "graphify"
outcome: "useful"
source_nodes: ["train_gru()", "TrainingConfig", "ASLGRUClassifier", "run_preprocessing()"]
---

# Q: đề xuất cho tôi config training

## Answer

Expanded from original query via graph vocabulary: training, config, gru, preload, amp, scheduler, augmentation, baseline, schema, checkpoint. User target: Colab/Kaggle T4 with balanced speed and accuracy. Proposed starting configuration, not benchmarked: current 20-gloss hands_pose features, 32 frames, participant_holdout 70/15/15 with split seed 42; GRU hidden_size=128 num_layers=2 dropout=0.3 bidirectional=false pooling=mean_max (215572 parameters at 160 input features and 20 classes). Training: preload=device, device=cuda, batch_size=128, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, epochs=80, patience=15, early_stopping_min_delta=0.0001, early_stopping_min_epochs=15, scheduler plateau factor=0.5 patience=5, grad_clip=1.0, amp=false, num_workers=0, num_seeds=1, seeds=[42], resume_checkpoint=null, run.name=gru_t4_balanced. Keep current spatial augmentations; time_stretch.enabled=false and flip.swap_hands=false with dominant_hand=auto. Exact overrides validated through load_config. Runtime inspection confirmed torch.amp.GradScaler rejects device_type keyword at src/training/train.py:268, so AMP should remain off until that call is corrected; this was reproduced without running training. Rebuild legacy cache using scripts/10_prepare_data.py --stage preprocess then --stage split (the reports incorrectly mention nonexistent scripts/10_preprocess.py). Select using validation loss, accuracy and macro-F1; checkpoint selection is validation loss. After choosing config, rerun seeds [42,43,44] with num_seeds=3. If overfitting persists, compare dropout=0.4 and weight_decay=0.001 on identical validation split; do not tune on test or sample videos. No training was run and canonical config/source files were not edited.

## Outcome

- Signal: useful

## Source Nodes

- train_gru()
- TrainingConfig
- ASLGRUClassifier
- run_preprocessing()