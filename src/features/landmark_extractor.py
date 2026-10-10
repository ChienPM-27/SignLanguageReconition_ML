"""Module trích xuất Hand Landmarks từ video bằng MediaPipe Hands.

Quy ước và tính năng đặc biệt:
- Sắp xếp theo Tay Trội (Dominant Hand): Phân tích tổng chuyển động trong cả video,
  tay di chuyển nhiều hơn được xếp vào Slot 1 (Dominant Hand), tay còn lại vào Slot 2.
- Trả về tọa độ gốc (T, 2, 21, 3) và Mask (T, 2) biểu thị frame có thấy tay hay không.
- Hỗ trợ MediaPipe Hands an toàn, tự đóng tài nguyên khi xử lý xong video.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

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

    def _get_hands(self):
        if self._hands is None:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        return self._hands

    def close(self):
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def extract_from_video(
        self, video_path: str | Path
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Đọc video và trích xuất chuỗi hand landmarks.

        Returns:
            raw_coords: np.ndarray shape (T, 2, 21, 3) - tọa độ gốc trong [0, 1]
            hand_masks: np.ndarray shape (T, 2) - boolean mask cho [hand1, hand2]
            stats: dict - thông tin fps, n_frames, detection_rate
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros((0, 2, 21, 3), dtype=np.float32), np.zeros((0, 2), dtype=bool), {"ok": False}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        hands = self._get_hands()
        raw_hands_per_frame = []  # List[List[np.ndarray(21, 3)]]

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            frame_hands = []
            if results.multi_hand_landmarks:
                for h_lms in results.multi_hand_landmarks:
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in h_lms.landmark], dtype=np.float32)
                    frame_hands.append(coords)
            raw_hands_per_frame.append(frame_hands)

        cap.release()

        T = len(raw_hands_per_frame)
        if T == 0:
            return np.zeros((0, 2, 21, 3), dtype=np.float32), np.zeros((0, 2), dtype=bool), {"ok": False}

        # -------------------------------------------------------------
        # Sắp xếp theo Tay Trội (Dominant Hand Sorting)
        # -------------------------------------------------------------
        # Nhặt tối đa 2 tay cho mỗi frame dựa trên độ mượt quỹ đạo và biên độ chuyển động
        raw_coords = np.zeros((T, 2, 21, 3), dtype=np.float32)
        hand_masks = np.zeros((T, 2), dtype=bool)

        # Gán sơ bộ: nếu có >= 1 tay, gán vào slot 0; nếu có 2 tay, gán vào slot 1
        for t, f_hands in enumerate(raw_hands_per_frame):
            if len(f_hands) == 1:
                raw_coords[t, 0] = f_hands[0]
                hand_masks[t, 0] = True
            elif len(f_hands) >= 2:
                # Sắp xếp theo vị trí hoành độ x để tạm tách 2 tay
                sorted_by_x = sorted(f_hands[:2], key=lambda h: h[0, 0])
                raw_coords[t, 0] = sorted_by_x[0]
                raw_coords[t, 1] = sorted_by_x[1]
                hand_masks[t, 0] = True
                hand_masks[t, 1] = True

        # Tính tổng quãng đường di chuyển của cổ tay (wrist = index 0)
        disp_0, disp_1 = 0.0, 0.0
        for t in range(1, T):
            if hand_masks[t, 0] and hand_masks[t - 1, 0]:
                disp_0 += np.linalg.norm(raw_coords[t, 0, 0] - raw_coords[t - 1, 0, 0])
            if hand_masks[t, 1] and hand_masks[t - 1, 1]:
                disp_1 += np.linalg.norm(raw_coords[t, 1, 0] - raw_coords[t - 1, 1, 0])

        # Nếu slot 1 chuyển động nhiều hơn slot 0, hoán đổi để slot 0 luôn là Tay Trội
        if disp_1 > disp_0:
            raw_coords[:, [0, 1]] = raw_coords[:, [1, 0]]
            hand_masks[:, [0, 1]] = hand_masks[:, [1, 0]]

        frames_with_any_hand = hand_masks.any(axis=1).sum()
        det_rate = float(frames_with_any_hand / T) if T > 0 else 0.0

        stats = {
            "ok": True,
            "fps": fps,
            "n_frames": T,
            "det_rate": round(det_rate, 4),
            "two_hand_rate": round(float(hand_masks.all(axis=1).sum() / T), 4) if T > 0 else 0.0,
        }
        return raw_coords, hand_masks, stats
