"""Audit log reads (admin only), filtered and paged, newest first."""

from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import ColumnElement, desc, func, select

from api_gateway.auth.rbac import AdminUser
from api_gateway.dependencies import DbSession
from api_gateway.models.user import AuditLog
from api_gateway.problems import ProblemError
from api_gateway.schemas.ops import AuditPageResponse, AuditResponse

router = APIRouter(prefix="/api/v1", tags=["audit"])


@router.get("/audit", response_model=AuditPageResponse)
async def list_audit_entries(
    db: DbSession,
    _: AdminUser,
    user_id: int | None = Query(default=None, ge=1),
    username: str | None = Query(default=None, max_length=64),
    method: str | None = Query(
        default=None, pattern="^(GET|POST|PUT|PATCH|DELETE|WS)$"
    ),
    endpoint: str | None = Query(
        default=None, max_length=256, description="Path prefix, e.g. /api/v1/commands."
    ),
    status: int | None = Query(default=None, ge=100, le=599),
    min_status: int | None = Query(
        default=None, ge=100, le=599, description="e.g. 400 for every refusal."
    ),
    from_ms: int | None = Query(default=None, ge=0, description="[UTC epoch ms]"),
    to_ms: int | None = Query(default=None, ge=0, description="[UTC epoch ms]"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0, le=1_000_000),
) -> AuditPageResponse:
    """Audit entries matching every given filter."""
    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise ProblemError(
            422, "request.invalid_range", "from_ms must not be after to_ms."
        )
    filters: list[ColumnElement[bool]] = []
    if user_id is not None:
        filters.append(AuditLog.user_id == user_id)
    if username:
        filters.append(AuditLog.username == username)
    if method:
        filters.append(AuditLog.method == method)
    if endpoint:
        escaped = endpoint.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        filters.append(AuditLog.endpoint.like(f"{escaped}%", escape="\\"))
    if status is not None:
        filters.append(AuditLog.response_status == status)
    if min_status is not None:
        filters.append(AuditLog.response_status >= min_status)
    if from_ms is not None:
        filters.append(AuditLog.timestamp_ms >= from_ms)
    if to_ms is not None:
        filters.append(AuditLog.timestamp_ms <= to_ms)

    total = await db.scalar(select(func.count(AuditLog.id)).where(*filters))
    rows = (
        (
            await db.execute(
                select(AuditLog)
                .where(*filters)
                .order_by(desc(AuditLog.timestamp_ms), desc(AuditLog.id))
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )
    return AuditPageResponse(
        items=[AuditResponse.model_validate(row, from_attributes=True) for row in rows],
        total=int(total or 0),
        limit=limit,
        offset=offset,
    )
