from __future__ import annotations

from typing import Iterable, List, Mapping

from sqlalchemy import select

from .models import VehicleEvent
from .session import get_session


def log_vehicle_events(events: Iterable[Mapping]) -> None:
    if not events:
        return
    with get_session() as session:
        for event in events:
            session.add(
                VehicleEvent(
                    track_id=event.get("track_id"),
                    label=event.get("label", "vehicle"),
                    confidence=float(event.get("confidence", 0.0)),
                    license_plate=event.get("license_plate"),
                    camera=event.get("camera", "webcam-0"),
                )
            )


def get_recent_events(limit: int = 50) -> List[VehicleEvent]:
    with get_session() as session:
        stmt = select(VehicleEvent).order_by(VehicleEvent.created_at.desc()).limit(limit)
        return list(session.scalars(stmt))

