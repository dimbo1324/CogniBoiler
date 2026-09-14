"""Alarm endpoints backed by the alert-manager persistence table."""

from __future__ import annotations

from typing import Annotated

from alert_manager.models import AlarmEvent
from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, select

from api_gateway.auth.jwt_handler import TokenData
from api_gateway.auth.rbac import require_role
from api_gateway.dependencies import DbSession
from api_gateway.schemas.ops import AlarmResponse

router = APIRouter(prefix="/api/v1", tags=["alarms"])


@router.get("/alarms", response_model=list[AlarmResponse])
async def list_alarms(
    db: DbSession,
    _: Annotated[TokenData, Depends(require_role("viewer"))],
    active_only: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[AlarmResponse]:
    """Return recent alarm events persisted by alert-manager."""
    stmt = select(AlarmEvent).order_by(desc(AlarmEvent.occurred_at_ms)).limit(limit)
    if active_only:
        stmt = stmt.where(AlarmEvent.cleared.is_(False))
    rows = (await db.execute(stmt)).scalars().all()
    return [
        AlarmResponse(
            alarm_id=row.alarm_id,
            source_service=row.source_service,
            severity=row.severity,
            parameter=row.parameter,
            value=row.value,
            threshold=row.threshold,
            action=row.action,
            message=row.message,
            topic=row.topic,
            occurred_at_ms=row.occurred_at_ms,
            acknowledged=row.acknowledged,
            acknowledged_at_ms=row.acknowledged_at_ms,
            cleared=row.cleared,
        )
        for row in rows
    ]
