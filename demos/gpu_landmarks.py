"""Local WebGL GPU landmark extraction using official MediaPipe task assets."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
import os
from pathlib import Path
import threading
import time
import uuid
from urllib.parse import urlsplit

import cv2
import numpy as np

from demos.live_landmarks import LandmarkFrame
from demos.setup_gpu_landmarks import ROOT, FILES, ensure_sdk, ensure_task_assets
from src.utils.config import CameraConfig


class GPULandmarksUnavailableError(RuntimeError):
    pass


def _import_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise GPULandmarksUnavailableError("Thiếu Playwright. Cài requirements-inference.txt để dùng landmarks GPU.") from exc
    return sync_playwright


def browser_path(configured: str | None = None) -> Path:
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidates = [candidate]
    else:
        candidates = [Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
                      Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise GPULandmarksUnavailableError("Không tìm thấy Chrome/Edge để chạy WebGL. Đặt camera.landmark_browser_path.")


def packet_from_result(result: dict, captured_at: float) -> LandmarkFrame:
    hands = np.zeros((2, 21, 3), np.float32)
    masks = np.zeros(2, bool)
    found = [np.asarray(value, np.float32) for value in result["hands"]]
    if len(found) > 2 or any(value.shape != (21, 3) or not np.isfinite(value).all() for value in found):
        raise ValueError("GPU hand landmarks have an invalid shape or coordinates.")
    labels = result["handedness"]
    if len(found) == 1:
        slot = 1 if labels and labels[0] == "Right" else 0
        hands[slot], masks[slot] = found[0], True
    elif len(found) == 2:
        for label, coords in zip(labels, found):
            if label not in ("Left", "Right"):
                continue
            slot = 1 if label == "Right" else 0
            if not masks[slot]:
                hands[slot], masks[slot] = coords, True
        if not masks.all():
            hands[:] = found
            masks[:] = True
    output = {}
    for kind, count in (("pose", 33), ("face", 468)):
        value = result[kind]
        present = value is not None
        coords = np.full((count, 3), np.nan, np.float32) if not present else np.asarray(value, np.float32)
        if coords.shape != (count, 3) or (present and not np.isfinite(coords).all()):
            raise ValueError(f"GPU {kind} landmarks have an invalid shape or coordinates.")
        output[kind] = coords, present
    return LandmarkFrame(captured_at, hands, masks, *output["pose"], *output["face"])


def validate_hardware(hardware: dict) -> None:
    if set(hardware) != {"hands", "face", "pose"}:
        raise GPULandmarksUnavailableError("Cần đủ ba GPU detector tay / mặt / pose.")
    for info in hardware.values():
        renderer = info.get("renderer", "").lower()
        if info.get("delegate") != "GPU" or not renderer or any(word in renderer for word in
                ("swiftshader", "llvmpipe", "software", "basic render", "softpipe")):
            raise GPULandmarksUnavailableError("Landmarks cần GPU hardware, không dùng software renderer.")


def collect_video(path, extract_frame=None, *, extract_batch=None, batch_size=1) -> dict:
    """Decode every original frame once and stack synchronized anatomical outputs."""
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Không đọc được video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        ended = False
        while not ended:
            batch = []
            for _ in range(batch_size):
                ok, frame = capture.read()
                if not ok:
                    ended = True
                    break
                batch.append(frame)
            if not batch:
                break
            stamps = [(len(frames) + index) / fps for index in range(len(batch))]
            results = (extract_batch(batch, stamps) if extract_batch is not None
                       else [extract_frame(frame, stamp) for frame, stamp in zip(batch, stamps)])
            if len(results) != len(batch):
                raise ValueError("GPU backend dropped video frames.")
            frames.extend(results)
    finally:
        capture.release()
    if not frames:
        raise ValueError("Video không có frames.")
    masks = np.stack([packet.hand_masks for packet in frames])
    return {"hands": np.stack([packet.hands for packet in frames]), "hand_masks": masks,
            "pose_coords": np.stack([packet.pose for packet in frames]),
            "pose_masks": np.asarray([packet.pose_present for packet in frames]),
            "face_coords": np.stack([packet.face for packet in frames]),
            "face_mask": np.asarray([packet.face_present for packet in frames]),
            "stats": {"ok": True, "n_frames": len(frames), "fps": fps,
                      "det_rate": float(masks.any(axis=1).mean()), "landmark_backend": "webgl_gpu"}}


class GpuLandmarkExtractor:
    """One isolated headless GPU runtime, owned by the calling extraction thread."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.playwright = self.browser = self.page = self.server = self.server_thread = None
        self.owner = None
        self.hardware = None
        self.last_timing_ms = 0.0
        self.last_draw_calls = {}
        self.pending_images = {}

    def initialize(self) -> None:
        if self.page is not None:
            if self.owner != threading.get_ident():
                raise RuntimeError("GPU landmark runtime must stay on its owning thread.")
            return
        executable = browser_path(getattr(self.cfg, "landmark_browser_path", None))
        sdk = ensure_sdk()
        models = ensure_task_assets()
        paths = {"/runtime.html": ROOT / "demos/gpu_runtime.html", "/gpu_runtime.mjs": ROOT / "demos/gpu_runtime.mjs"}
        paths.update({"/" + name: sdk / name for name in FILES})
        paths.update({"/models/" + name: path for name, path in models.items()})
        if any(not path.is_file() for path in paths.values()):
            raise GPULandmarksUnavailableError("Thiếu model/runtime asset cho landmarks GPU.")

        pending_images = self.pending_images

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                resource = urlsplit(self.path).path
                if resource in pending_images:
                    payload, kind = pending_images[resource], "image/png"
                else:
                    path = paths.get(resource)
                    if path is None:
                        self.send_error(404)
                        return
                    payload = path.read_bytes()
                    kind = "text/javascript" if path.suffix in (".js", ".mjs") else mimetypes.guess_type(str(path))[0]
                self.send_response(200)
                self.send_header("Content-Type", kind or "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.owner = threading.get_ident()
        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="islr-gpu-assets")
            self.server_thread.start()
            base = f"http://127.0.0.1:{self.server.server_address[1]}"
            self.playwright = _import_playwright()().start()
            self.browser = self.playwright.chromium.launch(executable_path=str(executable), headless=True,
                ignore_default_args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--disable-gpu"],
                args=["--enable-gpu", "--enable-webgl", "--use-angle=d3d11", "--force_high_performance_gpu",
                      "--disable-software-rasterizer"])
            self.page = self.browser.new_page()
            self.page.route("**/*", lambda route: route.continue_() if route.request.url.startswith(base + "/") else route.abort())
            self.page.goto(base + "/runtime.html")
            self.page.wait_for_function("window.gpuRuntimeLoaded === true")
            info = self.page.evaluate("options => window.initializeGpuLandmarks(options)", {
                "detectionConfidence": self.cfg.detection_confidence, "trackingConfidence": self.cfg.tracking_confidence})
            self.hardware = info["hardware"]
            validate_hardware(self.hardware)
        except Exception as exc:
            self.close()
            raise GPULandmarksUnavailableError(f"Không khởi tạo được landmarks WebGL GPU: {exc}") from exc

    def _stage_image(self, frame):
        ok, png = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise ValueError("Không mã hóa được frame camera.")
        resource = "/frame/" + uuid.uuid4().hex + ".png"
        self.pending_images[resource] = png.tobytes()
        return {"imagePath": resource}

    def extract_batch(self, frames, captured_at, resize=False):
        self.initialize()
        staged = []
        try:
            for frame in frames:
                if resize and frame.shape[1] > self.cfg.landmark_width:
                    width = self.cfg.landmark_width
                    frame = cv2.resize(frame, (width, max(1, round(frame.shape[0] * width / frame.shape[1]))))
                staged.append(self._stage_image(frame))
            results = self.page.evaluate("frames => window.extractGpuBatch(frames)", staged)
            packets = []
            for result, stamp in zip(results, captured_at, strict=True):
                self.last_timing_ms = result["inferenceMs"]
                self.last_draw_calls = result["drawCalls"]
                for name, calls in result["drawCalls"].items():
                    self.hardware[name]["drawCalls"] += calls
                packets.append(packet_from_result(result, stamp))
            return packets
        finally:
            for image in staged:
                self.pending_images.pop(image["imagePath"], None)

    def verify_pixel_transfer(self, frame):
        self.initialize()
        image = self._stage_image(frame)
        try:
            actual = self.page.evaluate("frame => window.verifyPixelTransfer(frame)", image)
            expected = hashlib.sha256(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).tobytes()).hexdigest()
            return actual == expected
        finally:
            self.pending_images.pop(image["imagePath"], None)

    def _extract(self, frame: np.ndarray, captured_at: float, resize: bool) -> LandmarkFrame:
        return self.extract_batch([frame], [captured_at], resize=resize)[0]

    def extract(self, frame: np.ndarray, captured_at: float) -> LandmarkFrame:
        return self._extract(frame, captured_at, resize=True)

    def extract_complete_video(self, path: Path) -> dict:
        return collect_video(path, extract_batch=self.extract_batch, batch_size=self.cfg.landmark_batch_size)

    def extract_from_video(self, path):
        result = self.extract_complete_video(Path(path))
        return result["hands"], result["hand_masks"], result["stats"]

    def extract_from_frame(self, frame):
        packet = self.extract(frame, time.monotonic())
        return packet.hands, packet.hand_masks

    def close(self) -> None:
        if self.owner is not None and self.owner != threading.get_ident():
            raise RuntimeError("Close the GPU landmark runtime on its owning thread.")
        try:
            if self.page is not None:
                try:
                    self.page.evaluate("window.closeGpuLandmarks?.()")
                except Exception:
                    pass
            if self.browser is not None:
                try:
                    self.browser.close()
                except Exception:
                    pass
        finally:
            try:
                if self.playwright is not None:
                    self.playwright.stop()
            finally:
                if self.server is not None:
                    self.server.shutdown()
                    self.server.server_close()
                self.page = self.browser = self.playwright = self.server = None
                self.owner = None
                self.pending_images.clear()


class SharedGpuLandmarks:
    """Serialize GPU frame calls on one owner thread; reuse tasks across model changes."""

    def __init__(self, cfg: CameraConfig):
        self.core = GpuLandmarkExtractor(cfg)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="islr-webgl")
        self.closed = False

    @property
    def hardware(self):
        return self.core.hardware

    def initialize(self):
        if self.closed:
            raise RuntimeError("GPU landmarks đã đóng.")
        return self.executor.submit(self.core.initialize).result()

    def extract(self, frame, captured_at):
        if self.closed:
            raise RuntimeError("GPU landmarks đã đóng.")
        return self.executor.submit(self.core.extract, frame, captured_at).result()

    def extract_full_frame(self, frame, captured_at):
        if self.closed:
            raise RuntimeError("GPU landmarks đã đóng.")
        return self.executor.submit(self.core._extract, frame, captured_at, False).result()

    def extract_batch(self, frames, captured_at):
        if self.closed:
            raise RuntimeError("GPU landmarks đã đóng.")
        return self.executor.submit(self.core.extract_batch, frames, captured_at, False).result()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.executor.submit(self.core.close).result()
        finally:
            self.executor.shutdown(wait=True)


class SharedGpuExtractor:
    """Predictor adapter that borrows the camera's GPU runtime without closing it."""

    def __init__(self, service: SharedGpuLandmarks):
        self.service = service

    def extract_complete_video(self, path):
        return collect_video(path, extract_batch=self.service.extract_batch,
                             batch_size=self.service.core.cfg.landmark_batch_size)

    def extract_from_video(self, path):
        result = self.extract_complete_video(path)
        return result["hands"], result["hand_masks"], result["stats"]

    def close(self):
        pass
