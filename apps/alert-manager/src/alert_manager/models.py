"""
SQLAlchemy models of the alarm tables owned by alert-manager.

The schema itself is created by the Alembic chain (revision 0002, applied by the
`migrate` job); these models must stay in step with it. At most one alarm per condition
key is open at a time — a partial unique index enforces it in the database.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

OPEN_ALARM_PREDICATE = "state <> 'CLEARED'"


class Base(DeclarativeBase):
    """Declarative base for alert-manager tables."""


class AlarmEvent(Base):
    """One alarm: an occurrence of a condition, from activation until it is closed."""

    __tablename__ = "alarm_events"
    __table_args__ = (
        Index(
            "uq_alarm_events_open_key",
            "key",
            unique=True,
            postgresql_where=text(OPEN_ALARM_PREDICATE),
            sqlite_where=text(OPEN_ALARM_PREDICATE),
        ),
        Index("ix_alarm_events_state_raised", "state", "raised_at_ms"),
        CheckConstraint(
            "state IN ('ACTIVE_UNACK', 'ACTIVE_ACK', 'CLEARED_UNACK', 'CLEARED')",
            name="ck_alarm_events_state",
        ),
        CheckConstraint(
            "severity IN ('warning', 'critical')", name="ck_alarm_events_severity"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    source_service: Mapped[str] = mapped_column(String(64), nullable=False)
    parameter: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    unit: Mapped[str] = mapped_column(String(16), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    topic: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    raised_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    cleared_at_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    acknowledged_at_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ack_comment: Mapped[str | None] = mapped_column(String(500), nullable=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)


class AlarmTransition(Base):
    """One change of an alarm's state, with who or what caused it."""

    __tablename__ = "alarm_transitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alarm_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("alarm_events.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    to_state: Mapped[str] = mapped_column(String(16), nullable=False)
    at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    comment: Mapped[str | None] = mapped_column(String(500), nullable=True)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
