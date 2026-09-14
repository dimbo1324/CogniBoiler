"""WebSocket endpoint backed by the live PhysicsService stream."""

from __future__ import annotations

import json

import jwt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api_gateway.auth.jwt_handler import decode_access_token
from api_gateway.clients import PhysicsGatewayClient

router = APIRouter(tags=["websocket"])


def _physics_client(websocket: WebSocket) -> PhysicsGatewayClient:
    """Resolve the shared PhysicsService client from app state."""
    return websocket.app.state.physics_client  # type: ignore[no-any-return]


@router.websocket("/ws/realtime")
async def realtime_stream(websocket: WebSocket) -> None:
    """Stream real-time boiler/turbine updates from the live gRPC backbone."""
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Missing token.")
        return

    try:
        payload = decode_access_token(token)
    except jwt.PyJWTError as exc:
        await websocket.close(code=4001, reason=f"Invalid token: {exc}")
        return

    await websocket.accept()

    try:
        async for item in _physics_client(websocket).stream_system_state(
            interval_s=0.0
        ):
            message = {
                "ts_ms": item.boiler.timestamp_ms,
                "user": payload.get("sub"),
                "boiler": {
                    "pressure_pa": item.boiler.pressure_pa,
                    "water_level_m": item.boiler.water_level_m,
                    "water_temp_k": item.boiler.water_temp_k,
                    "flue_gas_temp_k": item.boiler.flue_gas_temp_k,
                    "internal_energy_j": item.boiler.internal_energy_j,
                    "quality": item.boiler.quality,
                },
                "turbine": {
                    "electrical_power_w": item.turbine.electrical_power_w,
                    "shaft_power_w": item.turbine.shaft_power_w,
                    "steam_flow_kg_s": item.turbine.steam_flow_kg_s,
                    "steam_temp_in_k": item.turbine.steam_temp_in_k,
                    "exhaust_pressure_pa": item.turbine.exhaust_pressure_pa,
                },
                "actuators": {
                    "fuel_valve_command": item.actuators.fuel_valve_command,
                    "fuel_valve_position": item.actuators.fuel_valve_position,
                    "feedwater_valve_command": item.actuators.feedwater_valve_command,
                    "feedwater_valve_position": item.actuators.feedwater_valve_position,
                    "steam_valve_command": item.actuators.steam_valve_command,
                    "steam_valve_position": item.actuators.steam_valve_position,
                },
            }
            await websocket.send_text(json.dumps(message))
    except WebSocketDisconnect:
        pass
