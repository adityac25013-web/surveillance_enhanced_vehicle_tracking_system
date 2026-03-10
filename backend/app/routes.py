import time
from typing import Dict, List

from flask import Flask, Response, current_app, jsonify, stream_with_context

from .db import crud
from .db.models import VehicleEvent


def _serialize_event(e: VehicleEvent) -> Dict:
    return {
        "id": e.id,
        "track_id": e.track_id,
        "label": e.label,
        "confidence": e.confidence,
        "license_plate": e.license_plate,
        "camera": e.camera,
        "timestamp": e.created_at.isoformat() if e.created_at else None,
    }


def register_routes(app: Flask) -> None:
    @app.get("/api/health")
    def health() -> Response:
        return jsonify({"status": "ok"})

    @app.get("/video_feed")
    def video_feed() -> Response:
        processor = current_app.config["VIDEO_PROCESSOR"]

        def generate():
            while True:
                frame = processor.get_jpeg_frame()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.05)

        return Response(
            stream_with_context(generate()),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/events/live")
    def live_events() -> Response:
        processor = current_app.config["VIDEO_PROCESSOR"]
        events = processor.get_latest_events()
        crud.log_vehicle_events(events)
        return jsonify({"events": events})

    @app.get("/api/events/recent")
    def recent_events() -> Response:
        items: List[VehicleEvent] = crud.get_recent_events(limit=50)
        return jsonify({"events": [_serialize_event(e) for e in items]})
