from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from .session import Base


class VehicleEvent(Base):
    __tablename__ = "vehicle_events"

    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True, nullable=True)
    label = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    license_plate = Column(String(32), nullable=True)
    camera = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

