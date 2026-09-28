"""The topic names, and the broker ACL that has to grant each of them to its service."""

from __future__ import annotations

from pathlib import Path

import pytest
from cogniboiler_runtime import topics

ACL = Path(__file__).parents[3] / "infrastructure" / "docker" / "mosquitto" / "acl"

# Who publishes and who reads what, as the contract in the command reference lists it.
PUBLISHES: dict[str, tuple[str, ...]] = {
    "physics-engine": (
        topics.TOPIC_PLANT,
        topics.TOPIC_BOILER,
        topics.TOPIC_TURBINE,
        topics.TOPIC_HEARTBEAT,
        topics.TOPIC_STATUS_PHYSICS_ENGINE,
    ),
    "plc-controller": (
        topics.TOPIC_ALERT_WARNING,
        topics.TOPIC_ALERT_CRITICAL,
        topics.TOPIC_ALERT_SNAPSHOT,
        topics.TOPIC_PLC_EVENTS,
        topics.TOPIC_STATUS_PLC_CONTROLLER,
    ),
    "alert-manager": (topics.TOPIC_ALARM_CHANGES,),
}
SUBSCRIBES: dict[str, tuple[str, ...]] = {
    "alert-manager": (topics.FILTER_ALERTS,),
    "historian": (
        topics.FILTER_SENSORS,
        topics.TOPIC_ALARM_CHANGES,
        topics.TOPIC_PLC_EVENTS,
        topics.FILTER_STATUS,
    ),
    "api-gateway": (topics.TOPIC_PLC_EVENTS, topics.TOPIC_ALARM_CHANGES),
    "opcua-server": (topics.FILTER_SENSORS, topics.TOPIC_ALARM_CHANGES),
}


def grants() -> dict[str, list[tuple[str, str]]]:
    """(access, pattern) per account, as the ACL file states them."""
    accounts: dict[str, list[tuple[str, str]]] = {}
    user: str | None = None
    for line in ACL.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        if words[0] == "user":
            user = words[1]
            accounts[user] = []
        elif words[0] == "topic" and user is not None:
            accounts[user].append((words[1], words[2]))
    return accounts


def covers(pattern: str, topic: str) -> bool:
    """True when the ACL pattern `pattern` matches `topic` (itself possibly a filter)."""
    wanted = pattern.split("/")
    given = topic.split("/")
    for index, level in enumerate(wanted):
        if level == "#":
            return True
        if index >= len(given) or (level != "+" and level != given[index]):
            return False
    return len(wanted) == len(given)


def test_the_topic_strings_are_the_contract() -> None:
    assert (
        topics.TOPIC_PLANT,
        topics.TOPIC_BOILER,
        topics.TOPIC_TURBINE,
        topics.TOPIC_HEARTBEAT,
        topics.TOPIC_ALERT_WARNING,
        topics.TOPIC_ALERT_CRITICAL,
        topics.TOPIC_ALERT_SNAPSHOT,
        topics.TOPIC_PLC_EVENTS,
        topics.TOPIC_ALARM_CHANGES,
        topics.TOPIC_STATUS_PHYSICS_ENGINE,
        topics.TOPIC_STATUS_PLC_CONTROLLER,
    ) == (
        "sensors/plant",
        "sensors/boiler",
        "sensors/turbine",
        "sensors/system/heartbeat",
        "alerts/warning",
        "alerts/critical",
        "alerts/snapshot",
        "plc/events",
        "alarms/changes",
        "status/physics-engine",
        "status/plc-controller",
    )
    assert (topics.FILTER_SENSORS, topics.FILTER_ALERTS, topics.FILTER_STATUS) == (
        "sensors/#",
        "alerts/#",
        "status/+",
    )


def test_every_filter_covers_the_topics_it_is_named_for() -> None:
    for topic in (
        topics.TOPIC_PLANT,
        topics.TOPIC_BOILER,
        topics.TOPIC_TURBINE,
        topics.TOPIC_HEARTBEAT,
    ):
        assert covers(topics.FILTER_SENSORS, topic)
    for topic in (
        topics.TOPIC_ALERT_WARNING,
        topics.TOPIC_ALERT_CRITICAL,
        topics.TOPIC_ALERT_SNAPSHOT,
    ):
        assert covers(topics.FILTER_ALERTS, topic)
    for topic in (
        topics.TOPIC_STATUS_PHYSICS_ENGINE,
        topics.TOPIC_STATUS_PLC_CONTROLLER,
    ):
        assert covers(topics.FILTER_STATUS, topic)
    assert not covers(topics.FILTER_STATUS, "status/a/b")


@pytest.mark.parametrize(("account", "published"), sorted(PUBLISHES.items()))
def test_the_acl_lets_each_publisher_write_exactly_its_topics(
    account: str, published: tuple[str, ...]
) -> None:
    writable = [
        pattern
        for access, pattern in grants()[account]
        if access in ("write", "readwrite")
    ]
    for topic in published:
        assert any(covers(pattern, topic) for pattern in writable), topic
    # A write grant no published topic needs would let the account forge another's.
    for pattern in writable:
        assert any(covers(pattern, topic) for topic in published), pattern


@pytest.mark.parametrize(("account", "filters"), sorted(SUBSCRIBES.items()))
def test_the_acl_lets_each_subscriber_read_what_it_subscribes_to(
    account: str, filters: tuple[str, ...]
) -> None:
    readable = [
        pattern
        for access, pattern in grants()[account]
        if access in ("read", "readwrite")
    ]
    for topic_filter in filters:
        assert any(covers(pattern, topic_filter) for pattern in readable), topic_filter
