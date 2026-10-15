"""CLI script for ISLR model prediction on video or dummy test sequences.

Usage:
    python scripts/30_predict.py --video path/to/sample.mp4
    python scripts/30_predict.py --test-synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.inference.predictor import ISLRPredictor
from src.utils.logging import get_logger

logger = get_logger("islr.predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISLR Sign Language Prediction CLI")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (.mp4, .mov, etc.)")
    parser.add_argument("--model", type=str, default=None, help="Path to best_model.pt checkpoint")
    parser.add_argument("--glosses", type=str, default=None, help="Path to glosses.csv mapping")
    parser.add_argument("--test-synthetic", action="store_true", help="Run sanity check on synthetic random sample")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictor = ISLRPredictor(model_path=args.model, glosses_csv=args.glosses)

    logger.info("Loaded ISLR Predictor successfully with %d classes.", predictor.num_classes)
    logger.info("Model: %s", predictor.model_path)

    if args.video:
        video_p = Path(args.video)
        if not video_p.is_file():
            logger.error("Video file not found: %s", video_p)
            return 1
        logger.info("Processing video '%s'...", video_p.name)
        result = predictor.predict_video(video_p)
        logger.info("=" * 50)
        logger.info("PREDICTED GLOSS: %s (Confidence: %.2f%%)", result.predicted_gloss.upper(), result.confidence * 100)
        logger.info("Top-3 Predictions:")
        for g, c in result.top_k:
            logger.info("  - %-12s: %.2f%%", g, c * 100)
        logger.info("=" * 50)
        return 0

    if args.test_synthetic:
        logger.info("Running synthetic prediction test...")
        dummy_coords = np.random.randn(40, 2, 21, 3).astype(np.float32)
        dummy_masks = np.ones((40, 2), dtype=bool)
        result = predictor.predict_raw_coordinates(dummy_coords, dummy_masks)
        logger.info("Synthetic result: %s (%.2f%%)", result.predicted_gloss, result.confidence * 100)
        logger.info("Top-3: %s", result.top_k)
        return 0

    logger.warning("No input provided. Use --video <path> or --test-synthetic. See --help for options.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
