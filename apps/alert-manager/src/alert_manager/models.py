"""SQLAlchemy models for persisted alarm events."""

from __future__ import annotations

from sqlalchemy import BIGINT, Boolean, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for alert-manager tables."""


class AlarmEvent(Base):
    """Persisted warning/trip event emitted by the PLC runtime."""

    __tablename__ = "alarm_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alarm_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    source_service: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    parameter: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    topic: Mapped[str] = mapped_column(String(128), nullable=False)
    occurred_at_ms: Mapped[int] = mapped_column(BIGINT, nullable=False, index=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    acknowledged_at_ms: Mapped[int | None] = mapped_column(BIGINT, nullable=True)
    cleared: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
