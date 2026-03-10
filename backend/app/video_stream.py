from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .detection import DetectionPipeline


class VideoProcessor:
    def __init__(self, camera_index: int = 0, detect_every_n_frames: int = 2) -> None:
        self.camera_index = camera_index
        self.detect_every_n_frames = max(1, detect_every_n_frames)
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None
        self._latest_events: List[Dict[str, Any]] = []
        self._pipeline = DetectionPipeline()
        self._frame_count = 0
        self._last_vis_frame: Optional[np.ndarray] = None
        self._last_events: List[Dict[str, Any]] = []
        self._consecutive_failures = 0

    def _open_capture(self) -> cv2.VideoCapture:
        """
        Try a few camera indices with DirectShow first (Windows),
        then default backend. Returns the first working capture.
        """
        indices_to_try = [self.camera_index, 0, 1, 2, 3]
        tried: List[int] = []

        for idx in indices_to_try:
            if idx in tried:
                continue
            tried.append(idx)

            # Prefer DirectShow on Windows to reduce MSMF issues
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.camera_index = idx
                return cap
            cap.release()

            # Fallback: default backend (may be MSMF or others)
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                self.camera_index = idx
                return cap
            cap.release()

        raise RuntimeError(
            "Unable to open any webcam (tried indices 0–3). "
            "Check that a camera is connected, enabled in Windows privacy "
            "settings, and not in use by another app."
        )

    def start(self) -> None:
        if self._running:
            return
        self._capture = self._open_capture()
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._capture.set(cv2.CAP_PROP_FPS, 15)
        time.sleep(0.3)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def wait_for_first_frame(self, timeout: float = 5.0) -> None:
        """Block until at least one frame is available or timeout."""
        import time as _time
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    return
            _time.sleep(0.05)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._capture and self._capture.isOpened():
            self._capture.release()

    def _loop(self) -> None:
        while self._running and self._capture:
            ok, frame = self._capture.read()
            if not ok:
                self._consecutive_failures += 1
                # If we cannot grab any frame for a while, stop trying to avoid endless MSMF warnings.
                if self._consecutive_failures > 50:
                    break
                time.sleep(0.05)
                continue

            self._consecutive_failures = 0

            # Always push a raw frame first so the client sees video
            # even if detection is slow (e.g. YOLO model loading on CPU).
            raw_ret, raw_jpeg = cv2.imencode(".jpg", frame)
            if raw_ret:
                with self._lock:
                    self._latest_frame = raw_jpeg.tobytes()

            self._frame_count += 1
            run_detection = (self._frame_count % self.detect_every_n_frames) == 0

            if run_detection:
                events, vis_frame = self._pipeline.process_frame(frame)
                with self._lock:
                    self._last_events = events
                    self._last_vis_frame = vis_frame.copy()
            else:
                with self._lock:
                    events = list(self._last_events)
                vis_frame = frame.copy()
                if events:
                    for e in events:
                        bbox = e.get("bbox", [])
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = map(int, bbox)
                            label = e.get("label", "vehicle")
                            track_id = e.get("track_id", 0)
                            plate = (e.get("license_plate") or "").strip()
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                vis_frame,
                                f"{label} #{track_id}",
                                (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )
                            if plate:
                                cv2.putText(
                                    vis_frame,
                                    plate,
                                    (x1, y2 + 18),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 255, 255),
                                    1,
                                    cv2.LINE_AA,
                                )

            ret, jpeg = cv2.imencode(".jpg", vis_frame)
            if not ret:
                continue

            with self._lock:
                self._latest_frame = jpeg.tobytes()
                if run_detection:
                    self._latest_events = list(self._last_events)

        if self._capture and self._capture.isOpened():
            self._capture.release()

    def get_jpeg_frame(self) -> bytes:
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame

        # Fallback black frame
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        ret, jpeg = cv2.imencode(".jpg", blank)
        return jpeg.tobytes() if ret else b""

    def get_latest_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._last_events)

