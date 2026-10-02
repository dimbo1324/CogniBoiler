"""The topic names, and the broker ACL that has to grant each of them to its service."""

from __future__ import annotations

from pathlib import Path

import pytest
from alert_manager.subscriber import SUBSCRIPTIONS as ALERT_MANAGER_SUBSCRIPTIONS
from cogniboiler_runtime import topics
from historian.subscriber import SUBSCRIPTIONS as HISTORIAN_SUBSCRIPTIONS
from opcua_server.subscriber import SUBSCRIPTIONS as OPCUA_SUBSCRIPTIONS

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
# What each service subscribes to, read from its code where the list is a constant (the
# gateway builds its two routes inside the realtime source).
SUBSCRIBES: dict[str, tuple[str, ...]] = {
    "alert-manager": tuple(topic for topic, _ in ALERT_MANAGER_SUBSCRIPTIONS),
    "historian": tuple(topic for topic, _ in HISTORIAN_SUBSCRIPTIONS),
    "api-gateway": (topics.TOPIC_PLC_EVENTS, topics.TOPIC_ALARM_CHANGES),
    "opcua-server": tuple(topic for topic, _ in OPCUA_SUBSCRIPTIONS),
}
WILDCARDS = ("#", "+")


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
    # Exact topics both ways: a wildcard read right would hand a topic added later under
    # the same prefix to this account without any ACL review.
    assert sorted(readable) == sorted(filters)
    assert not any(level in WILDCARDS for f in filters for level in f.split("/"))


def test_an_account_that_subscribes_to_nothing_reads_nothing() -> None:
    for account, granted in grants().items():
        if account in SUBSCRIBES or account == "monitor":
            continue
        assert [access for access, _ in granted if access != "write"] == [], account


def test_the_healthcheck_account_reads_only_the_broker_statistics() -> None:
    assert grants()["monitor"] == [("read", "$SYS/#")]


def test_every_service_account_is_listed_here() -> None:
    assert set(grants()) == set(PUBLISHES) | set(SUBSCRIBES) | {"monitor"}
