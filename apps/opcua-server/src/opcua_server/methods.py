"""
OPC UA methods: writes of a signed-in OPC UA user, performed through the API gateway.

Each call runs as the user of the calling session: the gateway checks the role, audits the
request under the user's name and forwards it to the PLC or the alarm service. The reply
is two outputs, Accepted (Boolean) and Reason (String), as the PLC or the alarm service
gave them. Refusals by the gateway become status codes:

    no signed-in user, 401, 403      BadUserAccessDenied
    400, 404, 409, 422               BadInvalidArgument
    gateway or upstream unavailable  BadCommunicationError
"""

from __future__ import annotations

import logging
from typing import Any

from asyncua import ua
from cogniboiler_observability import correlation_scope

from opcua_server.gateway import GatewayClient, GatewayReply, GatewayUnavailableError
from opcua_server.identity import CURRENT_USER, GatewayUser
from opcua_server.metrics import METHOD_CALLS
from opcua_server.ua_types import StatusCodes, node_id, status

logger = logging.getLogger(__name__)

# ua is untyped: a result is a list of ua.Variant outputs or a ua.StatusCode.
MethodResult = Any


def argument(name: str, variant_type: ua.VariantType, description: str) -> ua.Argument:
    arg = ua.Argument()
    arg.Name = name
    arg.DataType = node_id(variant_type.value)
    arg.ValueRank = -1
    arg.ArrayDimensions = []
    arg.Description = ua.LocalizedText(description)
    return arg


RESULT_ARGUMENTS = [
    argument("Accepted", ua.VariantType.Boolean, "The command was accepted."),
    argument("Reason", ua.VariantType.String, "Why it was refused; empty if accepted."),
]


def _status(code: int) -> Any:
    return status(code)


def _number(variant: Any) -> float | None:
    value = variant.Value if isinstance(variant, ua.Variant) else variant
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _text(variant: Any) -> str | None:
    value = variant.Value if isinstance(variant, ua.Variant) else variant
    return value if isinstance(value, str) else None


class MethodHandlers:
    def __init__(self, gateway: GatewayClient) -> None:
        self._gateway = gateway

    async def _forward(
        self, action: str, path: str, payload: dict[str, Any]
    ) -> MethodResult:
        """One method call under its own correlation id, counted by outcome."""
        with correlation_scope(None):
            result = await self._forward_as_user(action, path, payload)
        if isinstance(result, list):
            outcome = "accepted" if result and result[0].Value else "refused"
        else:
            outcome = "failed"
        METHOD_CALLS.labels(action, outcome).inc()
        return result

    async def _forward_as_user(
        self, action: str, path: str, payload: dict[str, Any]
    ) -> MethodResult:
        user = CURRENT_USER.get()
        if not isinstance(user, GatewayUser) or user.session is None:
            logger.warning(
                "OPC UA %s refused: the session has no signed-in user", action
            )
            return _status(StatusCodes.BadUserAccessDenied)
        try:
            reply = await self._call(user, path, payload)
        except GatewayUnavailableError as exc:
            logger.warning("OPC UA %s by %s failed: %s", action, user.name, exc)
            return _status(StatusCodes.BadCommunicationError)
        if reply is None or reply.status in (401, 403):
            logger.warning("OPC UA %s by %s refused by the gateway", action, user.name)
            return _status(StatusCodes.BadUserAccessDenied)
        if reply.status in (400, 404, 409, 422):
            return _status(StatusCodes.BadInvalidArgument)
        if reply.status != 200:
            logger.warning(
                "OPC UA %s by %s: gateway answered HTTP %d %s",
                action,
                user.name,
                reply.status,
                reply.detail,
            )
            return _status(StatusCodes.BadCommunicationError)
        accepted = bool(reply.body.get("accepted"))
        reason = str(reply.body.get("reason", ""))
        logger.info(
            "OPC UA %s by %s: %s", action, user.name, "accepted" if accepted else reason
        )
        return [
            ua.Variant(accepted, ua.VariantType.Boolean),
            ua.Variant(reason, ua.VariantType.String),
        ]

    async def _call(
        self, user: GatewayUser, path: str, payload: dict[str, Any]
    ) -> GatewayReply | None:
        assert user.session is not None
        tokens = await user.session.tokens()
        if tokens is None:
            return None
        reply = await self._gateway.request("POST", path, payload, tokens.access_token)
        if reply.status != 401:
            return reply
        await user.session.invalidate_access()
        tokens = await user.session.tokens()
        if tokens is None:
            return None
        return await self._gateway.request("POST", path, payload, tokens.access_token)

    async def set_load_demand(self, parent: Any, load_w: Any) -> MethodResult:
        value = _number(load_w)
        if value is None:
            return _status(StatusCodes.BadInvalidArgument)
        return await self._forward(
            "SetLoadDemand", "/api/v1/commands/load", {"load_w": value}
        )

    async def set_control_mode(self, parent: Any, mode: Any) -> MethodResult:
        value = _text(mode)
        if value is None:
            return _status(StatusCodes.BadInvalidArgument)
        return await self._forward(
            "SetControlMode", "/api/v1/commands/mode", {"mode": value.strip().lower()}
        )

    async def reset_emergency_stop(self, parent: Any) -> MethodResult:
        return await self._forward("ResetEmergencyStop", "/api/v1/commands/reset", {})

    async def apply_valve_command(
        self, parent: Any, fuel: Any, feedwater: Any, steam: Any
    ) -> MethodResult:
        values = [_number(fuel), _number(feedwater), _number(steam)]
        if any(value is None for value in values):
            return _status(StatusCodes.BadInvalidArgument)
        return await self._forward(
            "ApplyValveCommand",
            "/api/v1/commands/valve",
            {
                "fuel_valve": values[0],
                "feedwater_valve": values[1],
                "steam_valve": values[2],
            },
        )

    async def acknowledge_alarm(
        self, parent: Any, alarm_id: Any, comment: Any
    ) -> MethodResult:
        number = _number(alarm_id)
        text = _text(comment)
        if number is None or text is None or number < 1 or number != int(number):
            return _status(StatusCodes.BadInvalidArgument)
        return await self._forward(
            "AcknowledgeAlarm",
            f"/api/v1/alarms/{int(number)}/ack",
            {"comment": text},
        )

    async def acknowledge_all_alarms(self, parent: Any, comment: Any) -> MethodResult:
        text = _text(comment)
        if text is None:
            return _status(StatusCodes.BadInvalidArgument)
        return await self._forward(
            "AcknowledgeAllAlarms", "/api/v1/alarms/ack-all", {"comment": text}
        )
