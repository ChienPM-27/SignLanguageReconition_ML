"""Self-contained regressions for live overlays, auto boundaries, and camera config."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from demos.camera_v3 import CameraApp, GestureRecorder
from demos.live_landmarks import AutoGestureSegmenter, LandmarkFrame, draw_landmarks
from src.features.schema import compute_preprocess_identity
from src.utils.config import CameraConfig, build_config_from_dict, load_config


def packet(time: float, shift: float = 0.0, visible: bool = True) -> LandmarkFrame:
    hands = np.zeros((2, 21, 3), np.float32)
    hands[:, :, 0] = .2 + np.arange(21) * .005 + shift
    hands[:, :, 1] = .3 + np.arange(21) * .004
    return LandmarkFrame(time, hands, np.array([visible, False]),
                         np.full((33, 3), np.nan, np.float32), False,
                         np.full((468, 3), np.nan, np.float32), False)


def test_auto_detect_starts_on_arrival_ends_on_pause_and_does_not_repeat_idle():
    segmenter = AutoGestureSegmenter(CameraConfig())
    assert segmenter.observe(packet(0)) is None
    assert segmenter.observe(packet(.16)) == "start"
    assert segmenter.observe(packet(1.05)) == "finish"
    for time in (1.4, 2, 3, 4):
        assert segmenter.observe(packet(time)) is None
    assert segmenter.observe(packet(4.1, shift=.06)) is None
    assert segmenter.observe(packet(4.3, shift=.12)) == "start"


def test_auto_detect_ends_when_hands_leave_and_discards_short_clips():
    segmenter = AutoGestureSegmenter(CameraConfig())
    segmenter.observe(packet(0))
    assert segmenter.observe(packet(.16)) == "start"
    assert segmenter.observe(packet(.3, visible=False)) is None
    assert segmenter.observe(packet(.65, visible=False)) == "discard"
    segmenter.reset()
    segmenter.observe(packet(0))
    segmenter.observe(packet(.16))
    segmenter.observe(packet(.5, shift=.03))
    assert segmenter.observe(packet(.8, visible=False)) is None
    assert segmenter.observe(packet(1, visible=False)) == "finish"


def test_long_moving_gesture_is_bounded_and_stale_packets_are_ignored():
    segmenter = AutoGestureSegmenter(CameraConfig(max_gesture_seconds=2))
    segmenter.observe(packet(0))
    assert segmenter.observe(packet(.16)) == "start"
    assert segmenter.observe(packet(.1, visible=False)) is None
    assert segmenter.previous.captured_at == .16
    assert segmenter.observe(packet(1, shift=.5)) is None
    assert segmenter.observe(packet(2.1, shift=1)) == "finish"


def test_motion_threshold_uses_elapsed_time_not_detector_frame_count():
    def speed(dt):
        segmenter = AutoGestureSegmenter(CameraConfig())
        segmenter.observe(packet(0))
        segmenter.observe(packet(dt, shift=.1 * dt))
        return segmenter.motion
    assert speed(.05) == pytest.approx(speed(.2), rel=1e-4)


def test_overlay_does_not_modify_recorded_pixels_or_draw_missing_points():
    frame = np.zeros((160, 240, 3), np.uint8)
    annotated = draw_landmarks(frame, packet(0))
    assert annotated.any()
    assert not frame.any()
    absent = replace(packet(0), hand_masks=np.zeros(2, bool))
    np.testing.assert_array_equal(draw_landmarks(frame, absent), frame)


def test_pre_roll_records_original_full_frames_in_order(tmp_path, monkeypatch):
    from demos import camera_v3

    frames = []
    writer = SimpleNamespace(isOpened=lambda: True, write=lambda frame: frames.append(frame.copy()), release=lambda: None)
    monkeypatch.setattr(camera_v3.cv2, "VideoWriter", lambda *args: writer)
    recorder = GestureRecorder(tmp_path, 30)
    prior = [np.full((16, 24, 3), index, np.uint8) for index in range(3)]
    recorder.start(prior[-1], pre_roll=prior)
    recorder.append(np.full((16, 24, 3), 3, np.uint8))
    assert recorder.finish().frame_count == 4
    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 1, 2, 3]


def test_camera_config_is_canonical_and_does_not_change_checkpoint_feature_identity():
    cfg = load_config()
    identity = compute_preprocess_identity(cfg)[0]
    override = load_config(overrides=["camera.auto_detect=false", "camera.start_motion_threshold=1.2"])
    assert not override.camera.auto_detect
    assert override.camera.start_motion_threshold == 1.2
    assert compute_preprocess_identity(override)[0] == identity
    legacy_config = cfg.to_dict()
    legacy_config.pop("camera")
    assert compute_preprocess_identity(build_config_from_dict(legacy_config))[0] == identity
    with pytest.raises(KeyError, match="Unknown configuration key"):
        load_config(overrides=["camera.auto=true"])


@pytest.mark.parametrize("override", ["camera.landmark_fps=0", "camera.auto_detect=1",
    "camera.camera_index=-1", "camera.detection_confidence=2", "camera.start_hold_seconds=.nan",
    "camera.min_gesture_seconds=7", "camera.stop_motion_threshold=1"])
def test_invalid_camera_tuning_fails_loudly(override):
    with pytest.raises((ValueError, TypeError)):
        load_config(overrides=[override])


def test_table_selection_is_blocked_during_recording():
    app = CameraApp.__new__(CameraApp)
    app.future = None
    app.recorder = SimpleNamespace(active=True)
    messages, tabs = [], []
    app.status_text = SimpleNamespace(set=messages.append)
    app.tabs = SimpleNamespace(select=tabs.append)
    app.camera_panel = "camera"
    app.choose_from_table("T0__H")
    assert messages and tabs == ["camera"]


def test_auto_controller_records_pre_roll_and_submits_completed_clip(tmp_path, monkeypatch):
    from collections import deque
    from demos import camera_v3

    frames = []
    writer = SimpleNamespace(isOpened=lambda: True, write=lambda frame: frames.append(frame.copy()), release=lambda: None)
    monkeypatch.setattr(camera_v3.cv2, "VideoWriter", lambda *args: writer)
    app = CameraApp.__new__(CameraApp)
    app.camera_cfg = CameraConfig()
    app.segmenter = AutoGestureSegmenter(app.camera_cfg)
    app.recorder = GestureRecorder(tmp_path, 30)
    app.predictor = SimpleNamespace()
    app.selected_model = SimpleNamespace(label="T0__HF seed42")
    app.future = None
    app.live_landmarks = None
    app.live_error = None
    app.next_landmark_at = 999
    app.recording_auto = False
    app.saved_clip = None
    app.landmark_status = SimpleNamespace(set=lambda value: None)
    app.record_button = SimpleNamespace(configure=lambda **kwargs: None)
    app.set_controls = lambda: None
    submitted = []
    app.submit_clip = submitted.append
    raw = np.zeros((16, 24, 3), np.uint8)
    app.pre_roll = deque([(0, raw), (.1, raw + 1)])

    def observe(value):
        app.live_future = SimpleNamespace(done=lambda: True, result=lambda: value)
        app.update_live_landmarks(raw, value.captured_at)

    observe(packet(0))
    assert not app.recorder.active
    observe(packet(.16))
    assert app.recorder.active and app.recording_auto
    app.recorder.append(raw + 2)
    observe(packet(1.05))
    assert not app.recorder.active
    assert len(submitted) == 1 and submitted[0].frame_count == 3
    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 1, 2]


def test_stale_live_result_does_not_start_recording():
    app = CameraApp.__new__(CameraApp)
    app.camera_cfg = CameraConfig()
    app.predictor = SimpleNamespace()
    app.future = None
    app.live_landmarks = None
    app.live_future = SimpleNamespace(done=lambda: True, result=lambda: packet(0))
    app.landmark_status = SimpleNamespace(set=lambda value: None)
    app.live_error = None
    app.next_landmark_at = 999
    app.segmenter = SimpleNamespace(observe=lambda frame: pytest.fail("Stale detection must not trigger auto recording"))
    app.update_live_landmarks(np.zeros((16, 24, 3), np.uint8), 1)


def test_ablation_tab_pauses_auto_detection():
    app = CameraApp.__new__(CameraApp)
    app.camera_cfg = CameraConfig()
    app.camera_visible = False
    app.predictor = SimpleNamespace()
    app.future = None
    app.live_landmarks = None
    app.live_future = SimpleNamespace(done=lambda: True, result=lambda: packet(0))
    app.landmark_status = SimpleNamespace(set=lambda value: None)
    app.live_error = None
    app.next_landmark_at = 0
    app.segmenter = SimpleNamespace(observe=lambda frame: pytest.fail("Reading the table must not start recording"))
    # No live_executor: there must also be no new detector job while reading the table.
    app.update_live_landmarks(np.zeros((16, 24, 3), np.uint8), 0)
