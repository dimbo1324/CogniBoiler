"""
Simulation control and the live plant snapshot.

Reading the plant and the simulation status needs viewer. Changing what the plant does —
pause, speed, stepping, scenarios, faults — needs engineer. None of these routes moves a
valve: actuators are reached only through the PLC. Loaded scenarios and injected or
cleared faults are also recorded in scenario_runs with the user who caused them.
"""

from __future__ import annotations

import logging
import time

import cogniboiler_pb2 as pb2
import grpc
from fastapi import APIRouter, Path, Query, Request
from sqlalchemy import desc, func, select
from sqlalchemy.exc import SQLAlchemyError

from api_gateway import plant_state
from api_gateway.audit import command_outcome, set_audit_detail, set_audit_outcome
from api_gateway.auth.identity import CurrentUser
from api_gateway.auth.rbac import EngineerUser, ViewerUser
from api_gateway.clients import PhysicsGatewayClient
from api_gateway.dependencies import DbSession
from api_gateway.models.user import ScenarioRun
from api_gateway.problems import ProblemError, upstream_unavailable
from api_gateway.schemas.plant import (
    FaultAckResponse,
    FaultRequest,
    PlantStateResponse,
    ScenarioListResponse,
    ScenarioRequest,
    ScenarioRunPageResponse,
    ScenarioRunResponse,
    SimulationAckResponse,
    SimulationStatusResponse,
    SpeedRequest,
    StepRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["simulation"])


def _physics(request: Request) -> PhysicsGatewayClient:
    return request.app.state.physics_client  # type: ignore[no-any-return]


def _unavailable(exc: grpc.RpcError) -> ProblemError:
    return upstream_unavailable("PhysicsService", exc)


async def _record_runs(
    db: DbSession,
    user: CurrentUser,
    kind: str,
    status: pb2.SimulationStatusMsg,
    faults: list[pb2.FaultMsg] | None = None,
) -> None:
    """Store who changed the run; a failure is logged, the action already happened."""
    now_ms = int(time.time() * 1000)
    rows = [
        ScenarioRun(
            kind=kind,
            scenario=status.scenario,
            run_id=status.run_id,
            fault_id=item.fault_id if item is not None else None,
            fault_label=item.label if item is not None else None,
            severity=item.severity if item is not None else None,
            simulation_time_s=status.simulation_time_s,
            user_id=user.id,
            username=user.username,
            at_ms=now_ms,
        )
        for item in (faults if faults is not None else [None])
    ]
    try:
        db.add_all(rows)
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.error(
            "scenario_runs NOT stored: %s by %s run_id=%d scenario=%s faults=%s",
            kind,
            user.username,
            status.run_id,
            status.scenario,
            [row.fault_label for row in rows],
            exc_info=True,
        )


def _simulation_ack(request: Request, ack: pb2.SimulationAck) -> SimulationAckResponse:
    set_audit_outcome(request, command_outcome(ack.accepted, ack.reason))
    return plant_state.simulation_ack(ack)


@router.get("/plant", response_model=PlantStateResponse)
async def get_plant_state(request: Request, _: ViewerUser) -> PlantStateResponse:
    """The latest plant step: measurements, valves, emissions, health, faults."""
    try:
        message = await _physics(request).get_system_state()
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    return plant_state.plant_state(message)


@router.get("/simulation", response_model=SimulationStatusResponse)
async def get_simulation_status(
    request: Request, _: ViewerUser
) -> SimulationStatusResponse:
    try:
        message = await _physics(request).get_simulation_status()
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    return plant_state.simulation_status(message)


@router.get("/simulation/scenarios", response_model=ScenarioListResponse)
async def list_scenarios(request: Request, _: ViewerUser) -> ScenarioListResponse:
    try:
        message = await _physics(request).list_scenarios()
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    return plant_state.scenario_list(message)


@router.post("/simulation/pause", response_model=SimulationAckResponse)
async def pause(request: Request, user: EngineerUser) -> SimulationAckResponse:
    try:
        ack = await _physics(request).pause(user.username)
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    return _simulation_ack(request, ack)


@router.post("/simulation/resume", response_model=SimulationAckResponse)
async def resume(request: Request, user: EngineerUser) -> SimulationAckResponse:
    try:
        ack = await _physics(request).resume(user.username)
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    return _simulation_ack(request, ack)


@router.post("/simulation/speed", response_model=SimulationAckResponse)
async def set_speed(
    request: Request, body: SpeedRequest, user: EngineerUser
) -> SimulationAckResponse:
    try:
        ack = await _physics(request).set_speed(body.speed_factor, user.username)
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    return _simulation_ack(request, ack)


@router.post("/simulation/step", response_model=SimulationAckResponse)
async def step(
    request: Request, body: StepRequest, user: EngineerUser
) -> SimulationAckResponse:
    """Advance a paused simulation by whole steps."""
    try:
        ack = await _physics(request).step(body.steps, user.username)
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    return _simulation_ack(request, ack)


@router.post("/simulation/scenario", response_model=SimulationAckResponse)
async def load_scenario(
    request: Request, body: ScenarioRequest, db: DbSession, user: EngineerUser
) -> SimulationAckResponse:
    """Restart the plant from a scenario; the PLC follows the new run."""
    set_audit_detail(request, f"scenario={body.name}")
    try:
        ack = await _physics(request).load_scenario(body.name, user.username)
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    if ack.accepted:
        await _record_runs(db, user, "scenario", ack.status)
    return _simulation_ack(request, ack)


@router.post("/simulation/faults", response_model=FaultAckResponse)
async def inject_fault(
    request: Request, body: FaultRequest, db: DbSession, user: EngineerUser
) -> FaultAckResponse:
    """Inject a labelled fault; the physics engine validates target and severity."""
    set_audit_detail(
        request,
        f"kind={body.kind} target={body.target} severity={body.severity:g} "
        f"ramp_s={body.ramp_s:g}",
    )
    physics = _physics(request)
    try:
        ack = await physics.inject_fault(
            pb2.FaultRequest(
                kind=plant_state.FAULT_KINDS_BY_NAME[body.kind],
                target=body.target,
                severity=body.severity,
                ramp_s=body.ramp_s,
                operator_id=user.username,
            )
        )
        status = await physics.get_simulation_status() if ack.accepted else None
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    if status is not None:
        await _record_runs(db, user, "fault_injected", status, list(ack.faults))
    return _fault_ack(request, ack)


@router.delete("/simulation/faults/{fault_id}", response_model=FaultAckResponse)
async def clear_fault(
    request: Request,
    db: DbSession,
    user: EngineerUser,
    fault_id: str = Path(..., min_length=1, max_length=64),
) -> FaultAckResponse:
    return await _clear(request, db, user, pb2.FaultClearRequest(fault_id=fault_id))


@router.delete("/simulation/faults", response_model=FaultAckResponse)
async def clear_all_faults(
    request: Request, db: DbSession, user: EngineerUser
) -> FaultAckResponse:
    return await _clear(request, db, user, pb2.FaultClearRequest(all=True))


async def _clear(
    request: Request,
    db: DbSession,
    user: CurrentUser,
    clear: pb2.FaultClearRequest,
) -> FaultAckResponse:
    clear.operator_id = user.username
    physics = _physics(request)
    try:
        ack = await physics.clear_fault(clear)
        status = (
            await physics.get_simulation_status()
            if ack.accepted and ack.faults
            else None
        )
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    if status is not None:
        await _record_runs(db, user, "fault_cleared", status, list(ack.faults))
    return _fault_ack(request, ack)


def _fault_ack(request: Request, ack: pb2.FaultAck) -> FaultAckResponse:
    set_audit_outcome(
        request,
        command_outcome(ack.accepted, ack.reason)
        + "".join(f" {item.label}" for item in ack.faults),
    )
    return FaultAckResponse(
        accepted=ack.accepted,
        reason=ack.reason,
        timestamp_ms=ack.timestamp_ms,
        faults=[plant_state.fault(item) for item in ack.faults],
    )


@router.get("/simulation/runs", response_model=ScenarioRunPageResponse)
async def list_scenario_runs(
    db: DbSession,
    _: ViewerUser,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> ScenarioRunPageResponse:
    """Scenario loads and fault changes, newest first, with who made them."""
    total = await db.scalar(select(func.count(ScenarioRun.id)))
    rows = (
        (
            await db.execute(
                select(ScenarioRun)
                .order_by(desc(ScenarioRun.at_ms), desc(ScenarioRun.id))
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )
    return ScenarioRunPageResponse(
        items=[
            ScenarioRunResponse.model_validate(row, from_attributes=True)
            for row in rows
        ],
        total=int(total or 0),
        limit=limit,
        offset=offset,
    )
