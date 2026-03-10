from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

try:
    import easyocr
except ImportError:
    easyocr = None

VEHICLE_CLASS_NAMES = {"car", "motorbike", "bus", "truck"}
# When user shows a phone screen with car/plate image, run OCR on phone bbox
PHONE_CLASS_NAME = "cell phone"


def _clean_plate_text(text: str) -> str:
    """Keep only alphanumeric (and common plate chars), uppercase."""
    if not text or not text.strip():
        return ""
    s = re.sub(r"[^A-Za-z0-9]", "", text.strip().upper())
    return s if 2 <= len(s) <= 15 else ""


def _read_text_from_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, reader: Any, min_conf: float = 0.25) -> str:
    """Run OCR on a full region (e.g. phone screen) and return best plate-like text."""
    h, w = frame.shape[:2]
    crop_x1 = max(0, x1)
    crop_x2 = min(w, x2)
    crop_y1 = max(0, y1)
    crop_y2 = min(h, y2)
    if crop_x2 - crop_x1 < 20 or crop_y2 - crop_y1 < 15:
        return ""
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return ""
    ch, cw = crop.shape[:2]
    if ch < 30 or cw < 60:
        scale = max(2, 60 // cw, 30 // ch)
        crop = cv2.resize(crop, (cw * scale, ch * scale), interpolation=cv2.INTER_CUBIC)
    try:
        results = reader.readtext(crop)
        for (_bbox, text, conf) in (results or []):
            if conf < min_conf:
                continue
            cleaned = _clean_plate_text(text)
            if cleaned:
                return cleaned
    except Exception:
        pass
    return ""


def _read_plate_from_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, reader: Any) -> str:
    """Crop lower part of vehicle bbox (plate region) and run OCR. Handles small bboxes (e.g. car on phone)."""
    h, w = frame.shape[:2]
    if x2 <= x1 or y2 <= y1:
        return ""
    box_h = y2 - y1
    box_w = x2 - x1
    # Plate is usually in lower 25–45% of vehicle; for small boxes use full lower half
    crop_y1 = max(0, y2 - int(box_h * 0.5))
    crop_y2 = y2
    crop_x1 = max(0, x1)
    crop_x2 = min(w, x2)
    crop_h = crop_y2 - crop_y1
    crop_w = crop_x2 - crop_x1
    if crop_h < 12 or crop_w < 24:
        return ""
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return ""
    # Upscale small crops so OCR can read better (e.g. car on phone screen)
    if crop_h < 40 or crop_w < 80:
        scale = max(2, 40 // crop_h, 80 // crop_w)
        crop = cv2.resize(crop, (crop_w * scale, crop_h * scale), interpolation=cv2.INTER_CUBIC)
    try:
        results = reader.readtext(crop)
        for (_bbox, text, conf) in (results or []):
            if conf < 0.25:
                continue
            cleaned = _clean_plate_text(text)
            if cleaned:
                return cleaned
    except Exception:
        pass
    return ""


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class DetectionPipeline:
    def __init__(self) -> None:
        self._model = YOLO("yolov8n.pt")
        self._tracker = DeepSort(max_age=30, n_init=3)
        self._ocr_reader: Optional[Any] = None

    def _get_ocr_reader(self) -> Optional[Any]:
        if easyocr is None:
            return None
        if self._ocr_reader is None:
            try:
                self._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception:
                self._ocr_reader = False  # type: ignore
        return self._ocr_reader if self._ocr_reader else None

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Lower conf (0.2) so cars on phone screens / smaller in frame still get detected
        yolo_results = self._model(rgb, imgsz=640, conf=0.2, verbose=False)[0]

        detections: List[Detection] = []
        if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
            boxes_xyxy = yolo_results.boxes.xyxy.cpu().numpy()
            confs = yolo_results.boxes.conf.cpu().numpy()
            class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cid in zip(boxes_xyxy, confs, class_ids):
                name = self._model.names.get(int(cid), "object")
                if name not in VEHICLE_CLASS_NAMES:
                    continue
                detections.append(
                    Detection(bbox=(x1, y1, x2, y2), confidence=float(conf), class_id=int(cid), class_name=name)
                )

        tracker_inputs = [
            [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence, d.class_name] for d in detections
        ]
        tracks = self._tracker.update_tracks(tracker_inputs, frame=frame)

        timestamp = datetime.now(timezone.utc).isoformat()
        events: List[Dict[str, Any]] = []
        vis_frame = frame.copy()
        ocr_reader = self._get_ocr_reader()

        # If user holds a phone showing car/plate, detect phone and run OCR on full screen
        if ocr_reader and yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
            boxes_xyxy = yolo_results.boxes.xyxy.cpu().numpy()
            confs = yolo_results.boxes.conf.cpu().numpy()
            class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), conf, cid in zip(boxes_xyxy, confs, class_ids):
                name = self._model.names.get(int(cid), "object")
                if name != PHONE_CLASS_NAME:
                    continue
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                plate_text = _read_text_from_region(frame, x1, y1, x2, y2, ocr_reader)
                if plate_text:
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    cv2.putText(vis_frame, f"phone: {plate_text}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 1, cv2.LINE_AA)
                    cv2.putText(vis_frame, plate_text, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                    events.append({
                        "track_id": 0,
                        "label": "plate",
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2],
                        "license_plate": plate_text,
                        "camera": "webcam-0",
                        "timestamp": timestamp,
                    })

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            label = track.det_class or "vehicle"

            license_plate = ""
            if ocr_reader:
                license_plate = _read_plate_from_crop(frame, x1, y1, x2, y2, ocr_reader)

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
            if license_plate:
                cv2.putText(
                    vis_frame,
                    license_plate,
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            events.append(
                {
                    "track_id": int(track_id),
                    "label": label,
                    "confidence": float(track.det_conf or 0.0),
                    "bbox": [x1, y1, x2, y2],
                    "license_plate": license_plate,
                    "camera": "webcam-0",
                    "timestamp": timestamp,
                }
            )

        return events, vis_frame

