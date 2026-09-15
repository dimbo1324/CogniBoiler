"""
Alarm lifecycle, simplified from ISA-18.2.

    (condition starts) ──► ACTIVE_UNACK ──acknowledge──► ACTIVE_ACK
                               │                            │
                        condition ends               condition ends
                               ▼                            ▼
                         CLEARED_UNACK ──acknowledge──► CLEARED (closed)
                               │
                        condition returns ──► ACTIVE_UNACK (the same alarm)

An alarm stays open until its condition has ended and someone has acknowledged it. A
condition that comes back while its alarm is still open re-activates that alarm instead
of opening a second one; once closed, the next occurrence opens a new alarm.
"""

from __future__ import annotations

from enum import StrEnum


class AlarmState(StrEnum):
    """Where an alarm is in its lifecycle."""

    ACTIVE_UNACK = "ACTIVE_UNACK"
    ACTIVE_ACK = "ACTIVE_ACK"
    CLEARED_UNACK = "CLEARED_UNACK"
    CLEARED = "CLEARED"


OPEN_STATES: frozenset[AlarmState] = frozenset(
    {AlarmState.ACTIVE_UNACK, AlarmState.ACTIVE_ACK, AlarmState.CLEARED_UNACK}
)
ACTIVE_STATES: frozenset[AlarmState] = frozenset(
    {AlarmState.ACTIVE_UNACK, AlarmState.ACTIVE_ACK}
)
UNACKNOWLEDGED_STATES: frozenset[AlarmState] = frozenset(
    {AlarmState.ACTIVE_UNACK, AlarmState.CLEARED_UNACK}
)


class LifecycleError(ValueError):
    """A transition the lifecycle does not allow."""


def after_condition_active(state: AlarmState) -> AlarmState:
    """State of an open alarm whose condition is reported active."""
    if state is AlarmState.CLEARED:
        raise LifecycleError("a closed alarm is not re-activated; a new alarm opens")
    if state is AlarmState.CLEARED_UNACK:
        return AlarmState.ACTIVE_UNACK
    return state


def after_condition_cleared(state: AlarmState) -> AlarmState:
    """State of an alarm whose condition has ended."""
    if state is AlarmState.ACTIVE_UNACK:
        return AlarmState.CLEARED_UNACK
    if state is AlarmState.ACTIVE_ACK:
        return AlarmState.CLEARED
    return state


def after_acknowledge(state: AlarmState) -> AlarmState:
    """State of an alarm someone acknowledged."""
    if state is AlarmState.ACTIVE_UNACK:
        return AlarmState.ACTIVE_ACK
    if state is AlarmState.CLEARED_UNACK:
        return AlarmState.CLEARED
    raise LifecycleError("alarm is already acknowledged")
