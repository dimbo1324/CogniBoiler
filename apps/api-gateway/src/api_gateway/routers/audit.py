"""Audit log endpoints backed by PostgreSQL."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, select

from api_gateway.auth.jwt_handler import TokenData
from api_gateway.auth.rbac import require_role
from api_gateway.dependencies import DbSession
from api_gateway.models.user import AuditLog
from api_gateway.schemas.ops import AuditResponse

router = APIRouter(prefix="/api/v1", tags=["audit"])


@router.get("/audit", response_model=list[AuditResponse])
async def list_audit_entries(
    db: DbSession,
    _: Annotated[TokenData, Depends(require_role("admin"))],
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[AuditResponse]:
    """Return recent immutable API audit entries."""
    rows = (
        (
            await db.execute(
                select(AuditLog).order_by(desc(AuditLog.timestamp_ms)).limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [
        AuditResponse(
            id=row.id,
            user_id=row.user_id,
            ip_address=row.ip_address,
            method=row.method,
            endpoint=row.endpoint,
            request_body_hash=row.request_body_hash,
            response_status=row.response_status,
            duration_ms=row.duration_ms,
            timestamp_ms=row.timestamp_ms,
            detail=row.detail,
        )
        for row in rows
    ]
