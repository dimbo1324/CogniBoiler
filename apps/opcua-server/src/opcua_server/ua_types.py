"""Typed constructors for the asyncua values this server builds."""

from __future__ import annotations

from datetime import UTC, datetime

from asyncua import ua
from asyncua.ua.attribute_ids import AttributeIds as AttributeIds
from asyncua.ua.object_ids import ObjectIds as ObjectIds
from asyncua.ua.status_codes import StatusCodes as StatusCodes


def node_id(identifier: int, namespace: int = 0) -> ua.NodeId:
    return ua.NodeId(ua.Int32(identifier), ua.Int16(namespace))


def status(code: int) -> ua.StatusCode:
    return ua.StatusCode(ua.UInt32(code))


def timestamp(moment: datetime) -> ua.DateTime:
    return ua.DateTime.fromtimestamp(moment.timestamp(), UTC)
