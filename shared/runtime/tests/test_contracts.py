"""The typed JSON payloads of the MQTT contract: what parses, what is refused, what is sent."""

from __future__ import annotations

import json
from typing import Any

import pytest
from cogniboiler_runtime.contracts import (
    ALARM_STATES,
    AlarmChangeMessage,
    AlarmConditionMessage,
    AlarmRecord,
    AlarmSnapshotMessage,
    AlarmTransitionRecord,
    ContractError,
    PlcEventMessage,
    decode_payload,
)
from cogniboiler_runtime.payloads import MAX_JSON_PAYLOAD_BYTES

AT_MS = 1_710_000_000_000


def condition() -> AlarmConditionMessage:
    return AlarmConditionMessage(
        alarm_id=f"plc-controller:water_level_m:low:critical:{AT_MS}",
        key="plc-controller:water_level_m:low:critical",
        state="active",
        source_service="plc-controller",
        severity="critical",
        parameter="water_level_m",
        direction="low",
        unit="m",
        value=3.1,
        threshold=3.5,
        action="emergency_stop",
        message="Drum level low-low",
        raised_at_ms=AT_MS - 1_000,
        timestamp_ms=AT_MS,
    )


def event() -> PlcEventMessage:
    return PlcEventMessage(
        event_id=f"manual_trip:{AT_MS}",
        kind="manual_trip",
        source_service="plc-controller",
        operator_id="operator1",
        detail={"reason": "drill"},
        timestamp_ms=AT_MS,
    )


def change() -> AlarmChangeMessage:
    return AlarmChangeMessage(
        alarm=AlarmRecord(
            id=7,
            key="plc-controller:water_level_m:low:critical",
            source_service="plc-controller",
            parameter="water_level_m",
            severity="critical",
            direction="low",
            unit="m",
            state="ACTIVE_ACK",
            message="Drum level low-low",
            action="emergency_stop",
            topic="alerts/critical",
            value=3.1,
            threshold=3.5,
            raised_at_ms=AT_MS - 1_000,
            cleared_at_ms=None,
            acknowledged_at_ms=AT_MS,
            acknowledged_by="operator1",
            ack_comment="seen",
            occurrence_count=2,
            updated_at_ms=AT_MS,
        ),
        transition=AlarmTransitionRecord(
            id=3,
            alarm_id=7,
            from_state="ACTIVE_UNACK",
            to_state="ACTIVE_ACK",
            at_ms=AT_MS,
            actor="operator1",
            comment="seen",
            value=None,
        ),
        timestamp_ms=AT_MS,
    )


def edited(raw: bytes, path: tuple[str, ...], value: object = ...) -> bytes:
    """`raw` with the field at `path` replaced by `value`, or removed without one."""
    payload: dict[str, Any] = json.loads(raw)
    target = payload
    for key in path[:-1]:
        target = target[key]
    if value is ...:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return json.dumps(payload).encode()


class TestWireFormat:
    """The keys, in the order the producers always sent them: the bytes did not change."""

    def test_an_alarm_condition(self) -> None:
        assert list(json.loads(condition().encode())) == [
            "alarm_id",
            "key",
            "state",
            "source_service",
            "severity",
            "parameter",
            "direction",
            "unit",
            "value",
            "threshold",
            "action",
            "message",
            "raised_at_ms",
            "timestamp_ms",
        ]

    def test_a_snapshot(self) -> None:
        snapshot = AlarmSnapshotMessage("plc-controller", ("a", "b"), AT_MS)
        assert snapshot.encode() == (
            b'{"source_service": "plc-controller", "active_keys": ["a", "b"], '
            b'"timestamp_ms": 1710000000000}'
        )

    def test_a_plc_event(self) -> None:
        assert event().encode() == (
            b'{"event_id": "manual_trip:1710000000000", "kind": "manual_trip", '
            b'"source_service": "plc-controller", "operator_id": "operator1", '
            b'"detail": {"reason": "drill"}, "timestamp_ms": 1710000000000}'
        )

    def test_an_alarm_change(self) -> None:
        payload = json.loads(change().encode())
        assert list(payload) == ["alarm", "transition", "timestamp_ms"]
        assert list(payload["alarm"]) == list(AlarmRecord.REQUIRED)
        assert list(payload["transition"]) == list(AlarmTransitionRecord.REQUIRED)
        assert payload["transition"]["from_state"] == "ACTIVE_UNACK"
        assert payload["alarm"]["cleared_at_ms"] is None


class TestRoundTrip:
    def test_each_model_parses_back_from_its_own_bytes(self) -> None:
        snapshot = AlarmSnapshotMessage("plc-controller", ("a",), AT_MS)
        assert AlarmConditionMessage.parse(condition().encode()) == condition()
        assert AlarmSnapshotMessage.parse(snapshot.encode()) == snapshot
        assert PlcEventMessage.parse(event().encode()) == event()
        assert AlarmChangeMessage.parse(change().encode()) == change()

    def test_a_decoded_mapping_parses_like_its_bytes(self) -> None:
        assert PlcEventMessage.parse(event().to_payload()) == event()

    def test_a_field_added_by_a_producer_is_ignored(self) -> None:
        raw = edited(event().encode(), ("schema_hint",), "future")
        assert PlcEventMessage.parse(raw) == event()
        raw = edited(change().encode(), ("alarm", "shelved"), True)
        assert AlarmChangeMessage.parse(raw) == change()


class TestAlarmCondition:
    def test_the_format_before_lifecycles_parses_with_defaults(self) -> None:
        raw = condition().encode()
        for field in ("alarm_id", "key", "state", "direction", "unit", "action"):
            raw = edited(raw, (field,))
        raw = edited(raw, ("raised_at_ms",))
        message = AlarmConditionMessage.parse(raw)
        assert (message.key, message.state, message.action) == ("", "active", "warn")
        assert (message.alarm_id, message.raised_at_ms) == ("", None)
        assert message.active

    @pytest.mark.parametrize(
        ("path", "value", "problem"),
        [
            (("value",), ..., "missing fields: value"),
            (("severity",), "info", "unknown severity 'info'"),
            (("state",), "flapping", "unknown state"),
            (("parameter",), "", "must not be empty"),
            (("value",), "3.1", "value must be a number"),
            (("value",), True, "value must be a number"),
            (("value",), 10**400, "value must be a finite number"),
            (("message",), 12, "message must be a string"),
            (("timestamp_ms",), -5, "timestamp_ms -5 is not a plausible time"),
            (("timestamp_ms",), 1.5, "timestamp_ms 1.5 is not a plausible time"),
            (("raised_at_ms",), "then", "raised_at_ms must be a number"),
        ],
    )
    def test_a_broken_condition_is_refused(
        self, path: tuple[str, ...], value: object, problem: str
    ) -> None:
        with pytest.raises(ContractError, match=problem):
            AlarmConditionMessage.parse(edited(condition().encode(), path, value))


class TestSnapshot:
    @pytest.mark.parametrize(
        ("payload", "problem"),
        [
            ({"source_service": "plc", "active_keys": "a", "timestamp_ms": 1}, "list"),
            ({"source_service": "plc", "active_keys": [1], "timestamp_ms": 1}, "list"),
            ({"source_service": "", "active_keys": [], "timestamp_ms": 1}, "empty"),
            ({"active_keys": [], "timestamp_ms": 1}, "empty"),
            ({"source_service": "plc", "active_keys": []}, "timestamp_ms"),
        ],
    )
    def test_a_broken_snapshot_is_refused(
        self, payload: dict[str, Any], problem: str
    ) -> None:
        with pytest.raises(ContractError, match=problem):
            AlarmSnapshotMessage.parse(json.dumps(payload).encode())


class TestPlcEvent:
    def test_an_automatic_event_has_an_empty_operator(self) -> None:
        raw = edited(event().encode(), ("operator_id",), "")
        assert PlcEventMessage.parse(raw).operator_id == ""

    @pytest.mark.parametrize(
        ("path", "value", "problem"),
        [
            (("operator_id",), ..., "missing fields: operator_id"),
            (("kind",), "", "kind must not be empty"),
            (("kind",), 3, "kind must be a string"),
            (("detail",), [1], "detail must be an object"),
            (("timestamp_ms",), "now", "timestamp_ms must be a number"),
        ],
    )
    def test_a_broken_event_is_refused(
        self, path: tuple[str, ...], value: object, problem: str
    ) -> None:
        with pytest.raises(ContractError, match=problem):
            PlcEventMessage.parse(edited(event().encode(), path, value))


class TestAlarmChange:
    @pytest.mark.parametrize("state", ALARM_STATES)
    def test_every_lifecycle_state_parses(self, state: str) -> None:
        raw = edited(change().encode(), ("alarm", "state"), state)
        assert AlarmChangeMessage.parse(raw).alarm.state == state

    def test_an_opening_transition_has_no_previous_state(self) -> None:
        raw = edited(change().encode(), ("transition", "from_state"), None)
        assert AlarmChangeMessage.parse(raw).transition.from_state is None

    @pytest.mark.parametrize(
        ("path", "value", "problem"),
        [
            (("transition",), ..., "missing fields: transition"),
            (("alarm",), [], "alarm must be an object"),
            (("transition", "to_state"), ..., "missing fields: transition.to_state"),
            (("transition", "to_state"), "GONE", "unknown transition.to_state 'GONE'"),
            (("transition", "from_state"), "x", "unknown transition.from_state"),
            (("alarm", "state"), ..., "missing fields: alarm.state"),
            (("alarm", "id"), "7", "alarm.id must be an integer"),
            (("alarm", "id"), True, "alarm.id must be an integer"),
            (("alarm", "severity"), "info", "unknown alarm.severity"),
            (("alarm", "cleared_at_ms"), -1, "alarm.cleared_at_ms -1"),
            (("alarm", "acknowledged_by"), 5, "alarm.acknowledged_by must be a string"),
            (("transition", "value"), "x", "transition.value must be a number"),
            (("timestamp_ms",), None, "timestamp_ms must be a number"),
        ],
    )
    def test_a_broken_change_is_refused(
        self, path: tuple[str, ...], value: object, problem: str
    ) -> None:
        with pytest.raises(ContractError, match=problem):
            AlarmChangeMessage.parse(edited(change().encode(), path, value))


class TestDecoding:
    @pytest.mark.parametrize(
        ("raw", "problem", "reason"),
        [
            (b"[1]", "not a JSON object", "invalid"),
            (b'{"value": NaN}', "not JSON", "invalid"),
            (b"{" + b" " * MAX_JSON_PAYLOAD_BYTES + b"}", "too large", "too_large"),
        ],
        ids=["array", "nan", "oversized"],
    )
    def test_what_is_not_a_json_object_is_refused_with_a_reason(
        self, raw: bytes, problem: str, reason: str
    ) -> None:
        with pytest.raises(ContractError, match=problem) as refused:
            decode_payload(raw)
        assert refused.value.reason == reason

    def test_a_mapping_passes_through(self) -> None:
        payload = {"kind": "x"}
        assert decode_payload(payload) is payload
