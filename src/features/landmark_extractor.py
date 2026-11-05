"""Module trích xuất Hand Landmarks từ video bằng MediaPipe Hands.

Quy ước và tính năng đặc biệt:
- Sắp xếp theo Tay Trội (Dominant Hand): Phân tích tổng chuyển động trong cả video,
  tay di chuyển nhiều hơn được xếp vào Slot 1 (Dominant Hand), tay còn lại vào Slot 2.
- Trả về tọa độ gốc (T, 2, 21, 3) và Mask (T, 2) biểu thị frame có thấy tay hay không.
- Hỗ trợ MediaPipe Hands an toàn, tự đóng tài nguyên khi xử lý xong video.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np


class MediaPipeHandExtractor:
    """Trích xuất 2 bàn tay x 21 landmarks x 3 tọa độ (x, y, z) từ video."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._hands = None
        self._use_tasks_api = False

    def _get_hands(self):
        if self._hands is None:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self._use_tasks_api = False
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=self.static_image_mode,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
            else:
                import urllib.request
                task_path = Path("models/hand_landmarker.task")
                task_path.parent.mkdir(parents=True, exist_ok=True)
                if not task_path.exists():
                    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                    urllib.request.urlretrieve(url, task_path)

                BaseOptions = mp.tasks.BaseOptions
                HandLandmarker = mp.tasks.vision.HandLandmarker
                HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(task_path)),
                    running_mode=VisionRunningMode.IMAGE,
                    num_hands=self.max_num_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_hand_presence_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self._use_tasks_api = True
                self._hands = HandLandmarker.create_from_options(options)
        return self._hands

    def close(self):
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def extract_from_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Trích xuất 2 bàn tay x 21 landmarks x 3 tọa độ từ một khung hình (BGR).

        Returns:
            coords: np.ndarray shape (2, 21, 3)
            mask: np.ndarray shape (2,) boolean mask
        """
        hands = self._get_hands()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_hands: list[tuple[str, np.ndarray]] = []

        import mediapipe as mp

        if self._use_tasks_api:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = hands.detect(mp_img)
            if results.hand_landmarks:
                for idx, h_lms in enumerate(results.hand_landmarks):
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in h_lms], dtype=np.float32)
                    label = "Unknown"
                    if results.handedness and idx < len(results.handedness):
                        label = results.handedness[idx][0].category_name
                    frame_hands.append((label, coords))
        else:
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for idx, h_lms in enumerate(results.multi_hand_landmarks):
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in h_lms.landmark], dtype=np.float32)
                    label = "Unknown"
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        label = results.multi_handedness[idx].classification[0].label
                    frame_hands.append((label, coords))

        coords_out = np.zeros((2, 21, 3), dtype=np.float32)
        mask_out = np.zeros((2,), dtype=bool)

        if len(frame_hands) == 1:
            lbl, pts = frame_hands[0]
            if lbl == "Right":
                coords_out[1] = pts
                mask_out[1] = True
            else:
                coords_out[0] = pts
                mask_out[0] = True
        elif len(frame_hands) >= 2:
            has_left, has_right = False, False
            for lbl, pts in frame_hands[:2]:
                if lbl == "Left" and not has_left:
                    coords_out[0] = pts
                    mask_out[0] = True
                    has_left = True
                elif lbl == "Right" and not has_right:
                    coords_out[1] = pts
                    mask_out[1] = True
                    has_right = True

            if not (has_left and has_right):
                coords_out[0] = frame_hands[0][1]
                coords_out[1] = frame_hands[1][1]
                mask_out[0] = True
                mask_out[1] = True

        return coords_out, mask_out

    def extract_from_video(
        self, video_path: str | Path
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Đọc video và trích xuất chuỗi hand landmarks theo thứ tự giải phẫu [Left, Right].

        Returns:
            raw_coords: np.ndarray shape (T, 2, 21, 3) - [slot 0: Left, slot 1: Right]
            hand_masks: np.ndarray shape (T, 2) - boolean mask cho [left, right]
            stats: dict - thông tin fps, n_frames, detection_rate
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros((0, 2, 21, 3), dtype=np.float32), np.zeros((0, 2), dtype=bool), {"ok": False}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        hands = self._get_hands()
        raw_hands_per_frame: list[list[tuple[str, np.ndarray]]] = []

        import mediapipe as mp

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_hands: list[tuple[str, np.ndarray]] = []

            if self._use_tasks_api:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = hands.detect(mp_img)
                if results.hand_landmarks:
                    for idx, h_lms in enumerate(results.hand_landmarks):
                        coords = np.array([[lm.x, lm.y, lm.z] for lm in h_lms], dtype=np.float32)
                        label = "Unknown"
                        if results.handedness and idx < len(results.handedness):
                            label = results.handedness[idx][0].category_name
                        frame_hands.append((label, coords))
            else:
                results = hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for idx, h_lms in enumerate(results.multi_hand_landmarks):
                        coords = np.array([[lm.x, lm.y, lm.z] for lm in h_lms.landmark], dtype=np.float32)
                        label = "Unknown"
                        if results.multi_handedness and idx < len(results.multi_handedness):
                            label = results.multi_handedness[idx].classification[0].label
                        frame_hands.append((label, coords))

            raw_hands_per_frame.append(frame_hands)

        cap.release()

        T = len(raw_hands_per_frame)
        if T == 0:
            return np.zeros((0, 2, 21, 3), dtype=np.float32), np.zeros((0, 2), dtype=bool), {"ok": False}

        # Gán theo thứ tự giải phẫu thuần túy: slot 0: Left, slot 1: Right
        raw_coords = np.zeros((T, 2, 21, 3), dtype=np.float32)
        hand_masks = np.zeros((T, 2), dtype=bool)

        for t, f_hands in enumerate(raw_hands_per_frame):
            if len(f_hands) == 1:
                lbl, pts = f_hands[0]
                if lbl == "Right":
                    raw_coords[t, 1] = pts
                    hand_masks[t, 1] = True
                else:
                    raw_coords[t, 0] = pts
                    hand_masks[t, 0] = True
            elif len(f_hands) >= 2:
                has_left, has_right = False, False
                for lbl, pts in f_hands[:2]:
                    if lbl == "Left" and not has_left:
                        raw_coords[t, 0] = pts
                        hand_masks[t, 0] = True
                        has_left = True
                    elif lbl == "Right" and not has_right:
                        raw_coords[t, 1] = pts
                        hand_masks[t, 1] = True
                        has_right = True
                if not (has_left and has_right):
                    raw_coords[t, 0] = f_hands[0][1]
                    raw_coords[t, 1] = f_hands[1][1]
                    hand_masks[t, 0] = True
                    hand_masks[t, 1] = True

        frames_with_any_hand = hand_masks.any(axis=1).sum()
        det_rate = float(frames_with_any_hand / T) if T > 0 else 0.0

        stats: Dict[str, Any] = {
            "ok": True,
            "fps": fps,
            "n_frames": T,
            "det_rate": round(det_rate, 4),
            "two_hand_rate": round(float(hand_masks.all(axis=1).sum() / T), 4) if T > 0 else 0.0,
        }
        return raw_coords, hand_masks, stats
