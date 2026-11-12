"""Lazy MediaPipe face/pose extraction on the same decoded video frame sequence."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np


def _task_asset(kind: str) -> Path:
    import urllib.request
    names = {"face": "face_landmarker", "pose": "pose_landmarker_lite"}
    name = names[kind]
    path = Path(__file__).resolve().parents[2] / "models" / (name + ".task")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://storage.googleapis.com/mediapipe-models/{kind}_landmarker/{name}/float16/1/{name}.task"
        urllib.request.urlretrieve(url, path)
    return path


def extract_auxiliary_video(video_path: str | Path, *, face: bool, pose: bool) -> dict[str, np.ndarray]:
    """Extract 468 face / 33 pose points; missing observations stay NaN with false masks."""
    import mediapipe as mp
    rows: dict[str, list[np.ndarray]] = {}
    masks: dict[str, list[bool]] = {}
    with ExitStack() as stack:
        detectors = {}
        legacy = hasattr(mp, "solutions")
        for kind, enabled in (("face", face), ("pose", pose)):
            if not enabled:
                continue
            if legacy:
                detector = (mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)
                            if kind == "face" else mp.solutions.pose.Pose())
            else:
                vision = mp.tasks.vision
                base = mp.tasks.BaseOptions(model_asset_path=str(_task_asset(kind)))
                if kind == "face":
                    options = vision.FaceLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.IMAGE, num_faces=1)
                    detector = vision.FaceLandmarker.create_from_options(options)
                else:
                    options = vision.PoseLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.IMAGE, num_poses=1)
                    detector = vision.PoseLandmarker.create_from_options(options)
            stack.callback(detector.close)
            detectors[kind] = detector
            rows[kind], masks[kind] = [], []
        cap = cv2.VideoCapture(str(video_path))
        stack.callback(cap.release)
        if not cap.isOpened():
            raise ValueError(f"Could not read video: {video_path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for kind, detector in detectors.items():
                count = 468 if kind == "face" else 33
                if legacy:
                    result = detector.process(rgb)
                    landmarks = (result.multi_face_landmarks[0].landmark if result.multi_face_landmarks else None) if kind == "face" else (result.pose_landmarks.landmark if result.pose_landmarks else None)
                else:
                    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    found = result.face_landmarks if kind == "face" else result.pose_landmarks
                    landmarks = found[0] if found else None
                coords = np.full((count, 3), np.nan, dtype=np.float32)
                if landmarks is not None:
                    coords[:] = np.asarray([[lm.x, lm.y, lm.z] for lm in landmarks[:count]], dtype=np.float32)
                rows[kind].append(coords)
                masks[kind].append(landmarks is not None)
    out = {}
    for kind, values in rows.items():
        count = 468 if kind == "face" else 33
        out[kind + "_coords"] = np.asarray(values, dtype=np.float32).reshape(-1, count, 3)
        out["face_mask" if kind == "face" else "pose_masks"] = np.asarray(masks[kind], dtype=bool)
    return out
