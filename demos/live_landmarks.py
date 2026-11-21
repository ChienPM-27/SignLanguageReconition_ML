"""Live camera landmarks and time-based isolated-gesture segmentation."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.features.landmark_extractor import MediaPipeHandExtractor
from src.features.schema import FACE_LANDMARK_INDICES
from src.features.video_auxiliary import _task_asset
from src.utils.config import CameraConfig

HAND_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))
POSE_EDGES = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


@dataclass(frozen=True)
class LandmarkFrame:
    captured_at: float
    hands: np.ndarray
    hand_masks: np.ndarray
    pose: np.ndarray
    pose_present: bool
    face: np.ndarray
    face_present: bool


class LiveLandmarkExtractor:
    """Owned by one worker, independently of the clip inference detectors."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.hands = MediaPipeHandExtractor(min_detection_confidence=cfg.detection_confidence,
                                            min_tracking_confidence=cfg.tracking_confidence)
        self.detectors = {}

    def _detector(self, kind: str):
        if kind not in self.detectors:
            import mediapipe as mp

            if hasattr(mp, "solutions"):
                self.detectors[kind] = (mp.solutions.face_mesh.FaceMesh(max_num_faces=1,
                                        min_detection_confidence=self.cfg.detection_confidence,
                                        min_tracking_confidence=self.cfg.tracking_confidence)
                                       if kind == "face" else mp.solutions.pose.Pose(
                                        min_detection_confidence=self.cfg.detection_confidence,
                                        min_tracking_confidence=self.cfg.tracking_confidence))
            else:
                vision = mp.tasks.vision
                base = mp.tasks.BaseOptions(model_asset_path=str(_task_asset(kind)))
                if kind == "face":
                    options = vision.FaceLandmarkerOptions(base_options=base, num_faces=1,
                        min_face_detection_confidence=self.cfg.detection_confidence,
                        min_face_presence_confidence=self.cfg.detection_confidence,
                        min_tracking_confidence=self.cfg.tracking_confidence)
                    self.detectors[kind] = vision.FaceLandmarker.create_from_options(options)
                else:
                    options = vision.PoseLandmarkerOptions(base_options=base, num_poses=1,
                        min_pose_detection_confidence=self.cfg.detection_confidence,
                        min_pose_presence_confidence=self.cfg.detection_confidence,
                        min_tracking_confidence=self.cfg.tracking_confidence)
                    self.detectors[kind] = vision.PoseLandmarker.create_from_options(options)
        return self.detectors[kind]

    def extract(self, frame: np.ndarray, captured_at: float) -> LandmarkFrame:
        import mediapipe as mp

        width = min(self.cfg.landmark_width, frame.shape[1])
        image = cv2.resize(frame, (width, max(1, round(frame.shape[0] * width / frame.shape[1]))))
        hands, masks = self.hands.extract_from_frame(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = {}
        for kind, count in (("pose", 33), ("face", 468)):
            detector = self._detector(kind)
            if hasattr(mp, "solutions"):
                result = detector.process(rgb)
                found = (result.multi_face_landmarks[0].landmark if result.multi_face_landmarks else None) if kind == "face" else (result.pose_landmarks.landmark if result.pose_landmarks else None)
            else:
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                sets = result.face_landmarks if kind == "face" else result.pose_landmarks
                found = sets[0] if sets else None
            coords = np.full((count, 3), np.nan, np.float32)
            if found is not None:
                coords[:] = np.asarray([(lm.x, lm.y, lm.z) for lm in found[:count]], np.float32)
            output[kind] = coords, found is not None
        return LandmarkFrame(captured_at, hands, masks, *output["pose"], *output["face"])

    def close(self) -> None:
        self.hands.close()
        for detector in self.detectors.values():
            detector.close()
        self.detectors.clear()


def draw_landmarks(frame: np.ndarray, landmarks: LandmarkFrame) -> np.ndarray:
    """Draw on a copy in original coordinates; the caller mirrors the final preview."""
    result = frame.copy()
    height, width = result.shape[:2]

    def point(coords, index):
        value = coords[index, :2]
        if not np.isfinite(value).all() or not (0 <= value[0] <= 1 and 0 <= value[1] <= 1):
            return None
        return round(value[0] * (width - 1)), round(value[1] * (height - 1))

    def skeleton(coords, indices, edges, color):
        for a, b in edges:
            first, second = point(coords, a), point(coords, b)
            if first is not None and second is not None:
                cv2.line(result, first, second, color, 2, cv2.LINE_AA)
        for index in indices:
            location = point(coords, index)
            if location is not None:
                cv2.circle(result, location, 3, color, -1, cv2.LINE_AA)

    for side, color in ((0, (70, 220, 90)), (1, (230, 160, 60))):
        if landmarks.hand_masks[side]:
            skeleton(landmarks.hands[side], range(21), HAND_EDGES, color)
    if landmarks.pose_present:
        skeleton(landmarks.pose, (11, 12, 13, 14, 15, 16), POSE_EDGES, (60, 200, 240))
    if landmarks.face_present:
        skeleton(landmarks.face, FACE_LANDMARK_INDICES, (), (220, 100, 220))
    return result


class AutoGestureSegmenter:
    """Start on hand arrival/motion, end on pause/loss, and suppress idle repeats."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.reset()

    def reset(self, now: float = 0.0) -> None:
        self.state = "waiting"
        self.previous = None
        self.pending_since = None
        self.started_at = 0.0
        self.last_motion_at = now
        self.last_hand_at = now
        self.cooldown_until = now
        self.motion = 0.0

    def observe(self, frame: LandmarkFrame) -> str | None:
        now = frame.captured_at
        if self.previous is not None and now <= self.previous.captured_at:
            return None
        visible = bool(frame.hand_masks.any())
        arrived = visible and (self.previous is None or not self.previous.hand_masks.any())
        self.motion = 0.0
        if self.previous is not None:
            common = frame.hand_masks & self.previous.hand_masks
            if common.any():
                dt = now - self.previous.captured_at
                palm = np.linalg.norm(frame.hands[common, 0, :2] - frame.hands[common, 9, :2], axis=-1).mean()
                if palm > np.finfo(np.float32).eps:
                    delta = np.linalg.norm(frame.hands[common, :, :2] - self.previous.hands[common, :, :2], axis=-1).mean()
                    self.motion = float(delta / palm / dt)
        self.previous = frame
        if visible:
            self.last_hand_at = now
        if arrived or self.motion >= self.cfg.stop_motion_threshold:
            self.last_motion_at = now
        if self.state == "cooldown":
            if now < self.cooldown_until:
                return None
            self.state = "waiting"
        if self.state == "waiting":
            if not visible:
                self.pending_since = None
            elif arrived or self.motion >= self.cfg.start_motion_threshold:
                if self.pending_since is None:
                    self.pending_since = now
            if self.pending_since is not None and visible and now - self.pending_since >= self.cfg.start_hold_seconds:
                self.state = "recording"
                self.started_at = self.pending_since
                self.pending_since = None
                return "start"
        elif self.state == "recording":
            duration = now - self.started_at
            ended = (now - self.last_hand_at >= self.cfg.lost_hand_seconds
                     or (visible and now - self.last_motion_at >= self.cfg.end_hold_seconds)
                     or duration >= self.cfg.max_gesture_seconds)
            if ended:
                self.state = "cooldown"
                self.cooldown_until = now + self.cfg.cooldown_seconds
                return "finish" if duration >= self.cfg.min_gesture_seconds else "discard"
        return None
