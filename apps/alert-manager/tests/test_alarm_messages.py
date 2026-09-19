"""MQTT payloads of the alarm intake and of alarms/changes."""

from __future__ import annotations

import json
from typing import Any

import pytest
from alarm_factories import condition_payload
from alert_manager.lifecycle import AlarmState
from alert_manager.payloads import (
    PayloadError,
    change_payload,
    parse_condition,
    parse_snapshot,
)
from alert_manager.views import AlarmView, TransitionView


def alarm_view(state: AlarmState = AlarmState.ACTIVE_UNACK) -> AlarmView:
    return AlarmView(
        id=7,
        key="plc-controller:water_level_m:low:critical",
        source_service="plc-controller",
        parameter="water_level_m",
        severity="critical",
        direction="low",
        unit="m",
        state=state,
        message="Drum level low-low",
        action="trip",
        topic="alerts/critical",
        value=3.1,
        threshold=3.5,
        raised_at_ms=1_000,
        cleared_at_ms=None,
        acknowledged_at_ms=None,
        acknowledged_by=None,
        ack_comment=None,
        occurrence_count=1,
        updated_at_ms=1_000,
    )


class TestConditions:
    def test_a_complete_condition(self) -> None:
        report = parse_condition("alerts/critical", condition_payload())
        assert report.key == "plc-controller:water_level_m:low:critical"
        assert (report.severity, report.direction, report.unit) == (
            "critical",
            "low",
            "m",
        )
        assert (report.value, report.threshold) == (3.1, 3.5)
        assert report.active is True
        assert report.topic == "alerts/critical"
        assert report.timestamp_ms == 1_700_000_000_000

    def test_a_cleared_condition(self) -> None:
        report = parse_condition("alerts/critical", condition_payload(state="cleared"))
        assert report.active is False

    def test_the_format_before_lifecycles_is_still_accepted(self) -> None:
        report = parse_condition(
            "alerts/warning",
            condition_payload(
                key=None,
                state=None,
                direction=None,
                unit=None,
                action=None,
                severity="warning",
            ),
        )
        assert report.key == "plc-controller:water_level_m:warning"
        assert report.active is True
        assert (report.direction, report.unit, report.action) == ("", "", "warn")

    @pytest.mark.parametrize(
        ("overrides", "problem"),
        [
            ({"value": None}, "missing fields: value"),
            (
                {"message": None, "threshold": None},
                "missing fields: message, threshold",
            ),
            ({"severity": "info"}, "unknown severity"),
            ({"state": "flapping"}, "unknown state"),
            ({"source_service": ""}, "must not be empty"),
            ({"parameter": ""}, "must not be empty"),
            ({"value": "3.1"}, "value must be a number"),
            ({"value": True}, "value must be a number"),
            ({"threshold": float("inf")}, "threshold must be finite"),
            ({"message": 12}, "message must be a string"),
        ],
    )
    def test_invalid_conditions_are_rejected(
        self, overrides: dict[str, Any], problem: str
    ) -> None:
        with pytest.raises(PayloadError, match=problem):
            parse_condition("alerts/critical", condition_payload(**overrides))

    @pytest.mark.parametrize(
        ("raw", "problem"),
        [
            (b"\xff\xfe", "not JSON"),
            (b"{broken", "not JSON"),
            (b"[1, 2, 3]", "not a JSON object"),
        ],
    )
    def test_bytes_that_are_not_a_json_object(self, raw: bytes, problem: str) -> None:
        with pytest.raises(PayloadError, match=problem):
            parse_condition("alerts/critical", raw)

    def test_text_fields_and_the_topic_are_bounded(self) -> None:
        report = parse_condition(
            "alerts/" + "t" * 300,
            condition_payload(message="m" * 5000, parameter="p" * 300, unit="u" * 40),
        )
        assert len(report.message) == 2000
        assert len(report.parameter) == 128
        assert len(report.unit) == 16
        assert len(report.topic) == 128

    def test_an_integer_value_is_accepted_as_a_number(self) -> None:
        report = parse_condition("alerts/critical", condition_payload(value=3))
        assert report.value == 3.0


class TestSnapshots:
    def test_a_snapshot_lists_the_active_keys(self) -> None:
        raw = json.dumps(
            {
                "source_service": "plc-controller",
                "active_keys": ["a", "b", "a"],
                "timestamp_ms": 5,
            }
        ).encode()
        report = parse_snapshot(raw)
        assert report.active_keys == frozenset({"a", "b"})
        assert (report.source_service, report.timestamp_ms) == ("plc-controller", 5)

    @pytest.mark.parametrize(
        ("payload", "problem"),
        [
            ({"source_service": "plc", "active_keys": "a", "timestamp_ms": 1}, "list"),
            ({"source_service": "plc", "active_keys": [1], "timestamp_ms": 1}, "list"),
            ({"source_service": "", "active_keys": [], "timestamp_ms": 1}, "empty"),
            ({"source_service": "plc", "active_keys": []}, "timestamp_ms"),
        ],
    )
    def test_invalid_snapshots_are_rejected(
        self, payload: dict[str, Any], problem: str
    ) -> None:
        with pytest.raises(PayloadError, match=problem):
            parse_snapshot(json.dumps(payload).encode())


class TestChanges:
    def test_a_change_carries_the_alarm_and_its_transition(self) -> None:
        transition = TransitionView(
            id=3,
            alarm_id=7,
            from_state=AlarmState.ACTIVE_UNACK,
            to_state=AlarmState.ACTIVE_ACK,
            at_ms=2_000,
            actor="operator1",
            comment="seen",
            value=3.1,
        )
        message = json.loads(
            change_payload(alarm_view(AlarmState.ACTIVE_ACK), transition)
        )
        assert message["timestamp_ms"] == 2_000
        assert message["alarm"]["state"] == "ACTIVE_ACK"
        assert message["alarm"]["id"] == 7
        assert message["transition"] == {
            "id": 3,
            "alarm_id": 7,
            "from_state": "ACTIVE_UNACK",
            "to_state": "ACTIVE_ACK",
            "at_ms": 2_000,
            "actor": "operator1",
            "comment": "seen",
            "value": 3.1,
        }

    def test_an_opening_transition_has_no_previous_state(self) -> None:
        transition = TransitionView(
            id=1,
            alarm_id=7,
            from_state=None,
            to_state=AlarmState.ACTIVE_UNACK,
            at_ms=1_000,
            actor="plc-controller",
            comment=None,
            value=None,
        )
        assert transition.to_dict()["from_state"] is None


class TestViews:
    @pytest.mark.parametrize(
        ("state", "is_open", "is_acknowledged"),
        [
            (AlarmState.ACTIVE_UNACK, True, False),
            (AlarmState.ACTIVE_ACK, True, True),
            (AlarmState.CLEARED_UNACK, True, False),
            (AlarmState.CLEARED, False, True),
        ],
    )
    def test_open_and_acknowledged_follow_the_state(
        self, state: AlarmState, is_open: bool, is_acknowledged: bool
    ) -> None:
        view = alarm_view(state)
        assert (view.is_open, view.is_acknowledged) == (is_open, is_acknowledged)
        assert view.to_dict()["state"] == state.value
