"""Real-time Isolated Sign Language Recognition (ISLR) Webcam Demo.

Features:
- Live 30+ FPS webcam stream using MediaPipe Hands + PyTorch GRU.
- Dual-mode triggering:
    1. AUTO Mode: Detects hand movement in active zone, buffers 32 frames, predicts, applies cooldown.
    2. SPACEBAR Mode: Press & hold Spacebar while signing; release to predict immediately.
- On-screen visual feedback: landmarks, state indicator, confidence bar, prediction history.

Usage:
    python scripts/31_realtime_demo.py
    python scripts/31_realtime_demo.py --camera 0 --mode auto
    python scripts/31_realtime_demo.py --mode spacebar
"""
from __future__ import annotations

import argparse
from collections import deque
import logging
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from src.features.landmark_extractor import MediaPipeHandExtractor
from src.inference.predictor import ISLRPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("islr.realtime")


def draw_hud(
    frame: np.ndarray,
    state: str,
    fps: float,
    predicted_gloss: str,
    confidence: float,
    history: list[str],
    mode: str,
    buffer_len: int,
    max_buffer: int = 32,
) -> np.ndarray:
    """Draw modern HUD overlay on webcam frame."""
    h, w, _ = frame.shape

    # 1. Top status bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
    # Bottom history bar
    cv2.rectangle(overlay, (0, h - 50), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    # 2. Header text
    cv2.putText(
        frame,
        f"ISLR Realtime | Mode: {mode.upper()} | FPS: {fps:.1f}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    # 3. State indicator badge
    state_colors = {
        "WAITING": (160, 160, 160),    # Gray
        "RECORDING": (0, 140, 255),    # Orange
        "DETECTED": (50, 205, 50),     # Green
        "COOLDOWN": (255, 191, 0),     # Cyan
    }
    color = state_colors.get(state, (200, 200, 200))

    cv2.circle(frame, (25, 55), 8, color, -1)
    cv2.putText(
        frame,
        f"State: {state} ({buffer_len}/{max_buffer} frames)",
        (45, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )

    # 4. Prediction display (Top right)
    if predicted_gloss:
        conf_str = f"{confidence * 100:.1f}%"
        text = f"Sign: {predicted_gloss.upper()} ({conf_str})"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.putText(
            frame,
            text,
            (w - tw - 20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 5. History at bottom
    history_str = "History: " + " -> ".join(history[-4:]) if history else "History: (None yet)"
    cv2.putText(
        frame,
        history_str,
        (15, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 240, 180),
        1,
        cv2.LINE_AA,
    )

    return frame


def draw_hand_landmarks(frame: np.ndarray, coords_2_21_3: np.ndarray, mask: np.ndarray) -> None:
    """Draw simple landmark points on the frame for visual feedback."""
    h, w, _ = frame.shape
    for hand_idx in range(2):
        if not mask[hand_idx]:
            continue
        color = (0, 255, 0) if hand_idx == 0 else (255, 100, 0)
        pts = coords_2_21_3[hand_idx]
        for i in range(21):
            px, py = int(pts[i, 0] * w), int(pts[i, 1] * h)
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(frame, (px, py), 4, color, -1)


def run_realtime(
    camera_id: int = 0,
    mode: str = "auto",
    confidence_threshold: float = 0.70,
    cooldown_seconds: float = 1.5,
) -> None:
    """Main real-time loop."""
    logger.info("Initializing ISLR Predictor...")
    predictor = ISLRPredictor()
    extractor = MediaPipeHandExtractor()

    logger.info(f"Opening camera index {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_id}. Please check device connection.")
        return

    # Set camera resolution to 640x480 for ultra-fast processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    logger.info("=" * 60)
    logger.info("REALTIME ISLR STARTED!")
    logger.info(f"- Mode: {mode.upper()}")
    logger.info("  * AUTO: Buffer activates when hands move; predicts when buffer reaches 32 frames.")
    logger.info("  * SPACEBAR: Press & hold Spacebar to record, release to predict.")
    logger.info("- Press 'q' or 'ESC' to exit.")
    logger.info("- Press 'm' to toggle AUTO / SPACEBAR mode.")
    logger.info("- Press 'c' to clear history.")
    logger.info("=" * 60)

    buffer_coords: deque = deque(maxlen=32)
    buffer_masks: deque = deque(maxlen=32)

    state = "WAITING"
    last_predicted_gloss = ""
    last_confidence = 0.0
    history: list[str] = []
    last_predict_time = 0.0
    space_pressed = False

    fps_tracker = deque(maxlen=30)
    t_prev = time.time()

    try:
        while True:
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_tracker.append(1.0 / dt)
            fps = float(np.mean(fps_tracker)) if fps_tracker else 30.0

            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame from camera.")
                break

            # Mirror view for intuitive user experience
            frame = cv2.flip(frame, 1)

            # 1. Extract hand landmarks
            coords, mask = extractor.extract_from_frame(frame)
            has_hand = bool(mask.any())

            # Draw visual landmarks
            draw_hand_landmarks(frame, coords, mask)

            # 2. Logic based on Mode
            if mode == "auto":
                # Check cooldown
                if time.time() - last_predict_time < cooldown_seconds:
                    state = "COOLDOWN"
                    buffer_coords.clear()
                    buffer_masks.clear()
                elif has_hand:
                    state = "RECORDING"
                    buffer_coords.append(coords)
                    buffer_masks.append(mask)

                    # When buffer is full (32 frames ~ 1.0 second), run prediction
                    if len(buffer_coords) == 32:
                        raw_c = np.array(buffer_coords, dtype=np.float32)
                        raw_m = np.array(buffer_masks, dtype=bool)

                        # Check if hand detection rate in buffer is sufficient (> 50%)
                        det_rate = float(raw_m.any(axis=1).mean())
                        if det_rate >= 0.5:
                            result = predictor.predict_raw_coordinates(raw_c, raw_m, top_k=3)
                            if result.confidence >= confidence_threshold:
                                last_predicted_gloss = result.predicted_gloss
                                last_confidence = result.confidence
                                history.append(result.predicted_gloss.upper())
                                last_predict_time = time.time()
                                state = "DETECTED"
                                logger.info(
                                    f"Detected: {result.predicted_gloss.upper()} "
                                    f"({result.confidence*100:.1f}%) [det_rate={det_rate*100:.0f}%]"
                                )
                        buffer_coords.clear()
                        buffer_masks.clear()
                else:
                    if len(buffer_coords) > 0:
                        buffer_coords.popleft()
                        buffer_masks.popleft()
                    state = "WAITING"

            elif mode == "spacebar":
                if space_pressed:
                    state = "RECORDING"
                    buffer_coords.append(coords)
                    buffer_masks.append(mask)
                else:
                    state = "WAITING"

            # 3. Draw HUD
            frame = draw_hud(
                frame=frame,
                state=state,
                fps=fps,
                predicted_gloss=last_predicted_gloss,
                confidence=last_confidence,
                history=history,
                mode=mode,
                buffer_len=len(buffer_coords),
                max_buffer=32,
            )

            cv2.imshow("ISLR Realtime Sign Recognition", frame)

            # 4. Handle Key Inputs
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord("m"):
                mode = "spacebar" if mode == "auto" else "auto"
                buffer_coords.clear()
                buffer_masks.clear()
                logger.info(f"Switched mode to: {mode.upper()}")
            elif key == ord("c"):
                history.clear()
                last_predicted_gloss = ""
                last_confidence = 0.0
            elif key == 32:  # Spacebar pressed in spacebar mode
                if mode == "spacebar" and not space_pressed:
                    space_pressed = True
                    buffer_coords.clear()
                    buffer_masks.clear()
                    logger.info("Recording gesture started (Spacebar held)...")

            # Check if spacebar was released (in spacebar mode)
            if mode == "spacebar" and space_pressed and key != 32 and len(buffer_coords) >= 10:
                space_pressed = False
                logger.info(f"Evaluating {len(buffer_coords)} recorded frames...")
                raw_c = np.array(buffer_coords, dtype=np.float32)
                raw_m = np.array(buffer_masks, dtype=bool)
                result = predictor.predict_raw_coordinates(raw_c, raw_m, top_k=3)
                last_predicted_gloss = result.predicted_gloss
                last_confidence = result.confidence
                history.append(result.predicted_gloss.upper())
                logger.info(f"Result: {result.predicted_gloss.upper()} ({result.confidence*100:.1f}%)")
                buffer_coords.clear()
                buffer_masks.clear()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        predictor.close()
        logger.info("Realtime session ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time ISLR Webcam Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--mode", type=str, choices=["auto", "spacebar"], default="auto", help="Trigger mode")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold (default: 0.70)")
    parser.add_argument("--cooldown", type=float, default=1.5, help="Cooldown in seconds (default: 1.5)")
    args = parser.parse_args()

    run_realtime(
        camera_id=args.camera,
        mode=args.mode,
        confidence_threshold=args.threshold,
        cooldown_seconds=args.cooldown,
    )
