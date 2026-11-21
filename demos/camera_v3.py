"""Local full-gesture camera demo for the accepted V3 ablation checkpoints."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
import tempfile
import time
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.pipeline import prepare_landmark_sequence
from src.features.video_auxiliary import _task_asset, extract_auxiliary_video
from src.inference.predictor import ISLRPredictor, PredictionResult
from demos.v3_models import CameraModel, DEPLOYMENT, load_catalog, materialize_model
from demos.live_landmarks import AutoGestureSegmenter, LiveLandmarkExtractor, draw_landmarks
from demos.gpu_landmarks import GpuLandmarkExtractor, SharedGpuLandmarks, SharedGpuExtractor
from demos.ablation_table import build_chooser
from src.utils.config import CameraConfig, DEFAULT_CONFIG_PATH, load_config

LOGGER = logging.getLogger("islr.camera")


@dataclass(frozen=True)
class CapturedClip:
    path: Path
    frame_count: int
    duration_seconds: float


class GestureRecorder:
    """Write every captured frame until the user ends the gesture."""

    def __init__(self, folder: Path, fps: float):
        self.folder = folder
        self.fps = fps
        self.writer = None
        self.path = None
        self.frame_count = 0
        self.started_at = 0.0

    @property
    def active(self) -> bool:
        return self.writer is not None

    def start(self, frame: np.ndarray, pre_roll=None) -> None:
        if self.active:
            raise RuntimeError("Đang ghi một ký hiệu.")
        self.path = self.folder / (uuid.uuid4().hex + ".avi")
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (width, height))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError("Không tạo được đoạn video camera.")
        self.writer = writer
        self.frame_count = 0
        self.started_at = time.monotonic()
        if pre_roll:
            for prior in pre_roll:
                self.append(prior)
        else:
            self.append(frame)

    def append(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            self.writer.write(frame)
            self.frame_count += 1

    def finish(self) -> CapturedClip:
        if self.writer is None or self.path is None:
            raise RuntimeError("Chưa bắt đầu ghi ký hiệu.")
        self.writer.release()
        self.writer = None
        return CapturedClip(self.path, self.frame_count, time.monotonic() - self.started_at)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def load_runtime(model: CameraModel, camera_cfg: CameraConfig | None = None,
                 gpu_service: SharedGpuLandmarks | None = None) -> ISLRPredictor:
    import torch

    path, glosses = materialize_model(model)
    predictor = ISLRPredictor(model_path=path, glosses_csv=glosses,
                              device="cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Per-fit seed is checkpoint metadata, while resolved_config.env.seed
        # remains the campaign default. The accepted per-seed hash binds the fit.
        if (predictor.cfg.run.name != model.run_id
                or predictor.input_size != model.dimension
                or tuple(predictor.cfg.model.regions) != model.regions
                or not predictor.cfg.model.region_fusion):
            raise ValueError(f"Cấu hình checkpoint khác inventory: {model.label}.")
        if camera_cfg is not None and camera_cfg.landmark_backend == "webgl_gpu":
            predictor.extractor.close()
            if gpu_service is not None:
                predictor.extractor = SharedGpuExtractor(gpu_service)
                gpu_service.initialize()
            else:
                predictor.extractor = GpuLandmarkExtractor(camera_cfg)
                predictor.extractor.initialize()
        else:
            predictor.extractor._get_hands()
            if predictor.cfg.features.face:
                _task_asset("face")
            if predictor.cfg.features.set == "hands_pose":
                _task_asset("pose")
    except BaseException:
        predictor.close()
        raise
    return predictor


def replace_runtime(previous: ISLRPredictor | None, model: CameraModel, camera_cfg: CameraConfig | None = None,
                    gpu_service: SharedGpuLandmarks | None = None) -> ISLRPredictor:
    if previous is not None:
        previous.close()
    return load_runtime(model, camera_cfg, gpu_service)


def infer_clip(predictor: ISLRPredictor, path: Path) -> PredictionResult:
    extraction_started = time.perf_counter()
    if isinstance(predictor.extractor, (GpuLandmarkExtractor, SharedGpuExtractor)):
        complete = predictor.extractor.extract_complete_video(path)
        hands, hand_masks, stats = complete["hands"], complete["hand_masks"], complete["stats"]
        auxiliary = {name: complete[name] for name in ("pose_coords", "pose_masks", "face_coords", "face_mask")}
        stats["landmark_device"] = "GPU"
    else:
        hands, hand_masks, stats = predictor.extractor.extract_from_video(path)
        auxiliary = None
        stats["landmark_device"] = "CPU"
    if not stats.get("ok") or len(hands) == 0:
        raise ValueError("Không đọc được đoạn video camera.")
    if auxiliary is None:
        auxiliary = extract_auxiliary_video(path, face=predictor.cfg.features.face,
                                            pose=predictor.cfg.features.set == "hands_pose")
    if any(len(value) != len(hands) for value in auxiliary.values()):
        raise ValueError("Landmarks tay và mặt không cùng số frames.")
    stats["landmark_seconds"] = time.perf_counter() - extraction_started
    preprocessing_started = time.perf_counter()
    prepared = prepare_landmark_sequence(
        raw_coords=hands, hand_masks=hand_masks, pose_coords=auxiliary.get("pose_coords"),
        pose_masks=auxiliary.get("pose_masks"), face_coords=auxiliary.get("face_coords"),
        face_mask=auxiliary.get("face_mask"), cfg=predictor.cfg,
    )
    if not prepared.is_usable:
        raise ValueError("Chưa thấy đủ tay trong động tác. Đưa tay vào khung hình rồi ghi lại.")
    if "face" in predictor.cfg.model.regions and not prepared.provenance["usable_face"].any():
        raise ValueError("Chưa thấy rõ khuôn mặt. Đưa cả mặt và tay vào khung hình rồi ghi lại.")
    if "body" in predictor.cfg.model.regions and not prepared.provenance["usable_pose"].any():
        raise ValueError("Chưa thấy rõ thân trên. Đưa vai và khuỷu tay vào khung hình rồi ghi lại.")
    stats["face_detection_rate"] = float(prepared.provenance["usable_face"].mean())
    stats["pose_detection_rate"] = float(prepared.provenance["usable_pose"].mean())
    stats["feature_shape"] = list(prepared.features.shape)
    stats["preprocess_seconds"] = time.perf_counter() - preprocessing_started
    model_started = time.perf_counter()
    result = predictor.predict_features(prepared.features, stats=stats, top_k=3)
    # predict_features copies probabilities back to CPU, synchronizing CUDA work.
    stats["model_seconds"] = time.perf_counter() - model_started
    return result


class CameraApp:
    def __init__(self, root, catalog, initial_model: CameraModel, camera_cfg: CameraConfig | None = None):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.root.title("Camera ASL — Ablation V3")
        self.root.geometry("1080x720")
        self.root.minsize(900, 620)
        self.closed = False
        self.space_down = False
        self.latest_frame = None
        self.predictor = None
        self.catalog = catalog
        self.selected_model = initial_model
        self.saved_clip = None
        self.camera_cfg = camera_cfg or CameraConfig()
        self.segmenter = AutoGestureSegmenter(self.camera_cfg)
        self.gpu_service = SharedGpuLandmarks(self.camera_cfg) if self.camera_cfg.landmark_backend == "webgl_gpu" else None
        self.live_extractor = self.gpu_service if self.gpu_service is not None else LiveLandmarkExtractor(self.camera_cfg)
        self.live_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="islr-landmarks")
        self.live_future = None
        self.live_landmarks = None
        self.live_error = None
        self.next_landmark_at = 0.0
        self.pre_roll = deque()
        self.recording_auto = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="islr-inference")
        self.future = None
        self.job = "load"
        self.history = []
        self.last_clip = None
        self.job_started_at = time.monotonic()
        index = self.camera_cfg.camera_index
        self.capture = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("Không mở được camera mặc định. Kiểm tra kết nối và quyền camera của Windows.")
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or not 0 < fps <= 240:
            fps = 30.0
        self.temporary = tempfile.TemporaryDirectory(prefix="islr-camera-")
        self.temporary_path = Path(self.temporary.name).resolve()
        if self.temporary_path.parent != Path(tempfile.gettempdir()).resolve():
            raise ValueError("Camera temporary directory is outside the intended temp root.")
        self.recorder = GestureRecorder(self.temporary_path, fps)

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True)
        panel = ttk.Frame(self.tabs)
        ablation_panel = ttk.Frame(self.tabs)
        self.tabs.add(panel, text="Camera & landmarks")
        self.tabs.add(ablation_panel, text="Bảng ablation · 42 models")
        panel.columnconfigure(0, weight=3)
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(1, weight=1)
        self.camera_panel = panel
        self.camera_visible = True
        self.tabs.bind("<<NotebookTabChanged>>", self.change_tab)
        header = ttk.Frame(panel, padding=(18, 14))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(header, text="Nhận diện ký hiệu ASL", font=("Segoe UI", 19, "bold")).pack(anchor="w")
        ttk.Label(header, text="Auto detect: đưa tay vào hình, ký rồi dừng / hạ tay · Space để ghi thủ công").pack(anchor="w", pady=(5, 0))
        selectors = ttk.Frame(header)
        selectors.pack(fill="x", pady=(12, 0))
        ttk.Label(selectors, text="Model").pack(side="left", padx=(0, 6))
        self.model_text = tk.StringVar(value=initial_model.run_id)
        self.model_selector = ttk.Combobox(selectors, textvariable=self.model_text, width=16,
                                          values=list(dict.fromkeys(key[0] for key in catalog)), state="disabled")
        self.model_selector.pack(side="left", padx=(0, 14))
        ttk.Label(selectors, text="Seed").pack(side="left", padx=(0, 6))
        self.seed_text = tk.StringVar(value=str(initial_model.seed))
        self.seed_selector = ttk.Combobox(selectors, textvariable=self.seed_text, width=5,
                                         values=sorted(seed for run, seed in catalog if run == initial_model.run_id),
                                         state="disabled")
        self.seed_selector.pack(side="left", padx=(0, 14))
        self.model_info = tk.StringVar()
        ttk.Label(selectors, textvariable=self.model_info, wraplength=600).pack(side="left")
        self.model_selector.bind("<<ComboboxSelected>>", self.switch_model)
        self.seed_selector.bind("<<ComboboxSelected>>", self.switch_model)
        self.update_model_info()
        modes = ttk.Frame(header)
        modes.pack(fill="x", pady=(10, 0))
        self.auto_text = tk.BooleanVar(value=self.camera_cfg.auto_detect)
        self.overlay_text = tk.BooleanVar(value=self.camera_cfg.show_landmarks)
        ttk.Checkbutton(modes, text="Auto detect", variable=self.auto_text, command=self.change_auto_mode).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(modes, text="Hiện landmarks", variable=self.overlay_text, command=self.change_overlay).pack(side="left", padx=(0, 16))
        target = "GPU" if self.gpu_service is not None else "CPU"
        self.landmark_status = tk.StringVar(value=f"Đang khởi tạo landmarks {target} tay / body / mặt...")
        ttk.Label(modes, textvariable=self.landmark_status).pack(side="left")
        self.preview = ttk.Label(panel, anchor="center")
        self.preview.grid(row=1, column=0, sticky="nsew", padx=(18, 10), pady=8)
        results = ttk.Frame(panel, padding=(12, 14))
        results.grid(row=1, column=1, sticky="nsew", padx=(0, 18), pady=8)
        ttk.Label(results, text="Kết quả", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        result_tabs = ttk.Notebook(results)
        result_tabs.pack(fill="both", expand=True, pady=(8, 0))
        prediction_panel = ttk.Frame(result_tabs, padding=8)
        history_panel = ttk.Frame(result_tabs, padding=8)
        gloss_panel = ttk.Frame(result_tabs, padding=8)
        result_tabs.add(prediction_panel, text="Nhận diện")
        result_tabs.add(history_panel, text="Lịch sử")
        result_tabs.add(gloss_panel, text="20 gloss")
        self.result_text = tk.StringVar(value="Chờ ký hiệu")
        ttk.Label(prediction_panel, textvariable=self.result_text, font=("Segoe UI", 21, "bold"), wraplength=250).pack(anchor="w", pady=(8, 8))
        self.top_k_text = tk.StringVar(value="")
        ttk.Label(prediction_panel, text="Top-3 · điểm model").pack(anchor="w", pady=(8, 0))
        ttk.Label(prediction_panel, textvariable=self.top_k_text, font=("Segoe UI", 12), justify="left").pack(anchor="w", pady=8)
        self.detail_text = tk.StringVar(value="")
        ttk.Label(prediction_panel, textvariable=self.detail_text, wraplength=250).pack(anchor="w", pady=8)
        ttk.Label(history_panel, text="Các lần nhận diện gần nhất", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.history_text = tk.StringVar(value="")
        ttk.Label(history_panel, textvariable=self.history_text, wraplength=250, justify="left").pack(anchor="w", pady=8)
        self.classes_text = tk.StringVar(value="")
        ttk.Label(gloss_panel, textvariable=self.classes_text, wraplength=250).pack(anchor="w", pady=(8, 0))

        controls = ttk.Frame(panel, padding=(18, 12))
        controls.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.record_button = ttk.Button(controls, text="Ghi thủ công", command=self.toggle_record, state="disabled")
        self.record_button.pack(side="left", padx=(0, 10))
        self.replay_button = ttk.Button(controls, text="Chạy lại đoạn vừa ghi", command=self.replay_clip, state="disabled")
        self.replay_button.pack(side="left", padx=(0, 10))
        ttk.Button(controls, text="Xóa lịch sử", command=self.clear_history).pack(side="left", padx=(0, 10))
        ttk.Button(controls, text="Thoát", command=self.close).pack(side="right")
        self.status_text = tk.StringVar(value="Đang nạp model và detector tay / mặt...")
        ttk.Label(panel, textvariable=self.status_text, wraplength=1000).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=18, pady=(0, 12))
        root.bind("<space>", self.on_space)
        root.bind("<KeyRelease-space>", self.release_space)
        root.bind("<Escape>", lambda event: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.future = self.executor.submit(load_runtime, self.selected_model, self.camera_cfg, self.gpu_service)
        self.ablation_table = build_chooser(ablation_panel, catalog, self.choose_from_table)
        root.after(20, self.update_preview)
        root.after(100, self.poll_prediction)

    def choose_from_table(self, run_id: str) -> None:
        if self.future is not None or self.recorder.active:
            self.status_text.set("Đợi nhận diện xong hoặc kết thúc ghi trước khi đổi model.")
            self.tabs.select(self.camera_panel)
            return
        self.model_text.set(run_id)
        self.switch_model()
        self.tabs.select(self.camera_panel)

    def change_tab(self, event=None) -> None:
        self.camera_visible = self.tabs.select() == str(self.camera_panel)
        if not self.camera_visible and self.recorder.active:
            if self.recording_auto:
                self.recorder.finish().path.unlink(missing_ok=True)
                self.recording_auto = False
                self.record_button.configure(text="Ghi thủ công")
                self.set_controls()
            else:
                self.tabs.select(self.camera_panel)
                self.camera_visible = True
                self.status_text.set("Kết thúc ghi thủ công trước khi xem bảng ablation.")
        self.segmenter.reset(time.monotonic())

    def change_overlay(self) -> None:
        self.camera_cfg.show_landmarks = self.overlay_text.get()
        self.live_error = None

    def change_auto_mode(self) -> None:
        self.camera_cfg.auto_detect = self.auto_text.get()
        self.segmenter.reset(time.monotonic())
        self.live_error = None
        if self.recorder.active:
            self.recorder.finish().path.unlink(missing_ok=True)
        self.recording_auto = False
        self.record_button.configure(text="Ghi thủ công")
        self.status_text.set("Auto detect bật — ký rồi dừng / hạ tay" if self.camera_cfg.auto_detect else "Ghi thủ công — Space để bắt đầu / kết thúc")
        self.set_controls()

    def update_model_info(self) -> None:
        model = self.selected_model
        names = {"hands": "tay", "body": "thân trên", "face": "mặt"}
        regions = " + ".join(names[name] for name in model.regions)
        device = "Đang nạp" if self.predictor is None else ("GPU" if self.predictor.device.type == "cuda" else "CPU")
        self.model_info.set(f"{regions} · {device} · Validation Macro-F1 {model.macro_f1 * 100:.2f}% (mean 3 seeds)")

    def set_controls(self) -> None:
        idle = self.future is None and not self.recorder.active
        for selector in (self.model_selector, self.seed_selector):
            selector.configure(state="readonly" if idle else "disabled")
        self.replay_button.configure(state="normal" if idle and self.predictor and self.saved_clip else "disabled")

    def switch_model(self, event=None) -> None:
        if self.future is not None or self.recorder.active:
            return
        run_id = self.model_text.get()
        seeds = sorted(seed for run, seed in self.catalog if run == run_id)
        self.seed_selector.configure(values=seeds)
        seed = int(self.seed_text.get())
        if seed not in seeds:
            seed = seeds[0]
            self.seed_text.set(str(seed))
        self.selected_model = self.catalog[run_id, seed]
        self.segmenter.reset(time.monotonic())
        self.update_model_info()
        self.result_text.set("Chờ ký hiệu")
        self.top_k_text.set("")
        self.detail_text.set("")
        self.status_text.set(f"Đang nạp {self.selected_model.label}...")
        previous, self.predictor = self.predictor, None
        self.job = "load"
        self.future = self.executor.submit(replace_runtime, previous, self.selected_model, self.camera_cfg, self.gpu_service)
        self.record_button.configure(state="disabled")
        self.set_controls()

    def submit_clip(self, clip: CapturedClip) -> None:
        self.last_clip = clip
        self.record_button.configure(text="Đang nhận diện...", state="disabled")
        self.result_text.set("Đang nhận diện")
        self.top_k_text.set("")
        self.status_text.set(f"Đang phân tích với {self.selected_model.label}...")
        self.job = "predict"
        self.job_started_at = time.monotonic()
        self.future = self.executor.submit(infer_clip, self.predictor, clip.path)
        self.set_controls()

    def replay_clip(self) -> None:
        if self.predictor is not None and self.future is None and not self.recorder.active and self.saved_clip:
            self.submit_clip(self.saved_clip)

    def finish_recording(self) -> None:
        clip = self.recorder.finish()
        if self.saved_clip is not None:
            self.saved_clip.path.unlink(missing_ok=True)
        self.saved_clip = clip
        self.recording_auto = False
        self.submit_clip(clip)

    def update_live_landmarks(self, frame: np.ndarray, now: float) -> None:
        if self.live_future is not None and self.live_future.done():
            future, self.live_future = self.live_future, None
            try:
                packet = future.result()
                if self.live_landmarks is None:
                    hardware = getattr(getattr(self, "gpu_service", None), "hardware", None)
                    LOGGER.info("Live landmarks ready: %s", hardware if hardware else "CPU hands / upper body / face")
                self.live_landmarks = packet
                target = "GPU" if getattr(self, "gpu_service", None) is not None else "CPU"
                self.landmark_status.set(f"{target} · Tay: {int(packet.hand_masks.sum())}/2 · Body: {'có' if packet.pose_present else 'chưa thấy'} "
                                         f"· Mặt: {'có' if packet.face_present else 'chưa thấy'}")
                ready = (self.predictor is not None and self.future is None
                         and getattr(self, "camera_visible", True)
                         and now - packet.captured_at <= self.camera_cfg.overlay_max_age_seconds)
                if self.camera_cfg.auto_detect and ready:
                    event = self.segmenter.observe(packet)
                    if event == "start" and not self.recorder.active:
                        self.recorder.start(frame, [prior for _, prior in self.pre_roll])
                        self.recording_auto = True
                        self.record_button.configure(text="Kết thúc & nhận diện")
                        self.set_controls()
                        LOGGER.info("Auto gesture started: %s", self.selected_model.label)
                    elif event == "finish" and self.recorder.active:
                        self.finish_recording()
                        LOGGER.info("Auto gesture completed: %s", self.selected_model.label)
                    elif event == "discard" and self.recorder.active:
                        self.recorder.finish().path.unlink(missing_ok=True)
                        self.recording_auto = False
                        self.record_button.configure(text="Ghi thủ công")
                        self.status_text.set("Động tác quá ngắn — ký lại rồi dừng / hạ tay")
                        self.set_controls()
            except Exception as exc:
                LOGGER.exception("Live landmark extraction failed")
                self.live_error = str(exc)
                self.auto_text.set(False)
                self.camera_cfg.auto_detect = False
                if self.recorder.active and self.recording_auto:
                    self.recorder.finish().path.unlink(missing_ok=True)
                    self.recording_auto = False
                    self.record_button.configure(text="Ghi thủ công")
                    self.set_controls()
                self.landmark_status.set("Landmarks lỗi — tắt/bật Hiện landmarks để thử lại")
                self.status_text.set(str(exc))
        enabled = ((self.camera_cfg.auto_detect or self.camera_cfg.show_landmarks)
                   and getattr(self, "camera_visible", True))
        if enabled and not self.live_error and self.live_future is None and now >= self.next_landmark_at:
            self.live_future = self.live_executor.submit(self.live_extractor.extract, frame.copy(), now)
            self.next_landmark_at = now + 1.0 / self.camera_cfg.landmark_fps

    def update_preview(self) -> None:
        from PIL import Image, ImageTk

        if self.closed:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.latest_frame = None
            if self.recorder.active:
                self.recorder.finish().path.unlink(missing_ok=True)
                self.recording_auto = False
                self.segmenter.reset(time.monotonic())
                self.record_button.configure(text="Ghi thủ công")
                self.set_controls()
            self.status_text.set("Camera không trả hình. Kiểm tra kết nối camera.")
            self.record_button.configure(state="disabled")
            self.root.after(100, self.update_preview)
            return
        self.latest_frame = frame
        now = time.monotonic()
        self.pre_roll.append((now, frame.copy()))
        while self.pre_roll and now - self.pre_roll[0][0] > self.camera_cfg.pre_roll_seconds:
            self.pre_roll.popleft()
        if self.predictor is not None and self.future is None:
            self.record_button.configure(state="normal")
        self.recorder.append(frame)
        self.update_live_landmarks(frame, now)
        # Mirror the preview only. Stored frames keep the video pipeline's hand convention.
        shown = frame
        if (self.camera_cfg.show_landmarks and self.live_landmarks is not None
                and now - self.live_landmarks.captured_at <= self.camera_cfg.overlay_max_age_seconds):
            shown = draw_landmarks(frame, self.live_landmarks)
        rgb = cv2.cvtColor(cv2.flip(shown, 1), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((max(self.preview.winfo_width(), 1), max(self.preview.winfo_height(), 1)))
        photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=photo)
        self.preview.image = photo
        if self.recorder.active:
            ending = "dừng / hạ tay để tự nhận diện" if self.recording_auto else "bấm Kết thúc khi ký xong"
            self.status_text.set(f"{'Auto đang ghi' if self.recording_auto else 'Đang ghi'} {now - self.recorder.started_at:.1f} giây — {ending}")
        self.root.after(20, self.update_preview)

    def on_space(self, event):
        if not self.camera_visible:
            return None
        # Let native controls handle Space themselves, including checkboxes.
        if event.widget not in (self.root, self.preview):
            return None
        if not self.space_down:
            self.space_down = True
            self.toggle_record()
        return "break"

    def release_space(self, event):
        self.space_down = False
        return "break"

    def toggle_record(self) -> None:
        if self.predictor is None or self.future is not None or self.latest_frame is None:
            return
        self.root.focus_set()
        try:
            if self.recorder.active:
                self.segmenter.state = "cooldown"
                self.segmenter.cooldown_until = time.monotonic() + self.camera_cfg.cooldown_seconds
                self.finish_recording()
            else:
                if self.camera_cfg.auto_detect:
                    self.auto_text.set(False)
                    self.change_auto_mode()
                self.recorder.start(self.latest_frame)
                self.record_button.configure(text="Kết thúc & nhận diện")
                self.status_text.set("Đang ghi — thực hiện một ký hiệu rồi bấm Kết thúc")
                self.set_controls()
        except Exception as exc:
            LOGGER.exception("Camera recording failed")
            self.status_text.set(str(exc))

    def poll_prediction(self) -> None:
        if self.closed:
            return
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            try:
                value = future.result()
                if self.job == "load":
                    self.predictor = value
                    self.update_model_info()
                    self.classes_text.set(f"{len(self.predictor.class_names)} gloss được hỗ trợ\n" + ", ".join(self.predictor.class_names))
                    self.status_text.set("Auto sẵn sàng — ký rồi dừng / hạ tay" if self.camera_cfg.auto_detect else "Sẵn sàng — bấm Ghi thủ công hoặc Space")
                    LOGGER.info("Camera ready: %s on %s", self.selected_model.label, self.predictor.device)
                else:
                    self.result_text.set(value.predicted_gloss)
                    self.top_k_text.set("\n".join(f"{gloss}   {probability * 100:.1f}%" for gloss, probability in value.top_k))
                    elapsed = time.monotonic() - self.job_started_at
                    device = "GPU" if self.predictor.device.type == "cuda" else "CPU"
                    self.detail_text.set(f"{self.selected_model.label} · {device}\n"
                        f"Đoạn ký hiệu {self.last_clip.duration_seconds:.1f} giây\n"
                        f"Landmarks {value.stats['landmark_device']} {value.stats['landmark_seconds']:.2f} s · Model {value.stats['model_seconds'] * 1000:.1f} ms\n"
                        f"Tổng xử lý {elapsed:.1f} giây")
                    self.history.append(f"{self.selected_model.label}: {value.predicted_gloss}")
                    self.history_text.set("\n".join(self.history[-6:]))
                    self.status_text.set("Đã nhận diện — ghi ký hiệu tiếp theo")
                    LOGGER.info("Prediction [%s]: %s %.4f shape=%s face_coverage=%.3f", self.selected_model.label, value.predicted_gloss,
                                value.confidence, value.stats["feature_shape"], value.stats["face_detection_rate"])
            except Exception as exc:
                LOGGER.exception("Camera model job failed")
                self.result_text.set("Ghi lại ký hiệu")
                self.top_k_text.set("")
                self.status_text.set(str(exc))
            finally:
                self.last_clip = None
                self.record_button.configure(text="Ghi thủ công", state="normal" if self.predictor else "disabled")
                self.set_controls()
        self.root.after(100, self.poll_prediction)

    def clear_history(self) -> None:
        self.history.clear()
        self.history_text.set("")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.recorder.close()
        self.capture.release()
        self.root.destroy()
        if self.gpu_service is None:
            self.live_executor.submit(self.live_extractor.close)
        self.live_executor.shutdown(wait=False, cancel_futures=False)
        # Complete resource cleanup on the inference thread after any running clip finishes.
        self.executor.submit(self.close_runtime)
        self.executor.shutdown(wait=False, cancel_futures=False)

    def close_runtime(self) -> None:
        try:
            if self.predictor is not None:
                self.predictor.close()
            elif self.future is not None and self.job == "load":
                try:
                    self.future.result().close()
                except Exception:
                    LOGGER.exception("Model initialization failed during close")
        finally:
            try:
                if self.gpu_service is not None:
                    self.gpu_service.close()
            finally:
                self.temporary.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="T0__HF", help="Run ID trong ablation V3, ví dụ G-bi__HBF hoặc L0__H.")
    parser.add_argument("--seed", type=int, default=42, help="Checkpoint seed đã train: 42, 43 hoặc 44.")
    parser.add_argument("--list-models", action="store_true", help="Liệt kê các model và seeds đã nghiệm thu.")
    parser.add_argument("--video", type=Path, help="Run the same pipeline on a saved clip instead of opening the camera.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Canonical config; demo chỉ lấy camera settings.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override canonical camera settings.")
    args = parser.parse_args()
    camera_cfg = load_config(args.config, overrides=args.set).camera
    catalog = load_catalog()
    if args.list_models:
        for run_id in dict.fromkeys(key[0] for key in catalog):
            seeds = [seed for run, seed in catalog if run == run_id]
            model = catalog[run_id, seeds[0]]
            print(f"{run_id:14} seeds={seeds} regions={','.join(model.regions):16} validation_macro_f1={model.macro_f1:.4f}")
        return
    if (args.model, args.seed) not in catalog:
        parser.error("Model/seed không có trong inventory đã nghiệm thu. Dùng --list-models để xem danh sách.")
    model = catalog[args.model, args.seed]
    DEPLOYMENT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(DEPLOYMENT / "camera.log", encoding="utf-8"), logging.StreamHandler()])
    if args.video is not None:
        predictor = load_runtime(model, camera_cfg)
        try:
            result = infer_clip(predictor, args.video.resolve())
            print(json.dumps({"model": model.run_id, "seed": model.seed, "regions": model.regions, "device": str(predictor.device),
                              "gloss": result.predicted_gloss, "confidence": result.confidence,
                              "top_k": result.top_k, "frames": result.num_frames,
                              "feature_shape": result.stats["feature_shape"],
                              "face_detection_rate": result.stats["face_detection_rate"],
                              "landmark_seconds": result.stats["landmark_seconds"],
                              "landmark_device": result.stats["landmark_device"],
                              "model_seconds": result.stats["model_seconds"]}, ensure_ascii=False, indent=2))
        finally:
            predictor.close()
        return
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    try:
        CameraApp(root, catalog, model, camera_cfg)
    except Exception as exc:
        LOGGER.exception("Camera startup failed")
        messagebox.showerror("Không khởi chạy được camera", str(exc), parent=root)
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
