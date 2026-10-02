"""The MQTT topics of the platform, named once.

Topics are a contract: a publisher and every subscriber must agree on the exact string,
and a change updates all of them in one task. Six services used to spell the strings
out themselves; here each topic has one name every publisher and subscriber imports.
Every service subscribes to exact topics, and the broker's ACL
(`infrastructure/docker/mosquitto/acl`) grants these same topics per account, without
wildcards. The `FILTER_*` wildcards are what services subscribed to before; persistent
sessions unsubscribe them on connect.
"""

from __future__ import annotations

SENSORS_PREFIX = "sensors/"
TOPIC_PLANT = SENSORS_PREFIX + "plant"
TOPIC_BOILER = SENSORS_PREFIX + "boiler"
TOPIC_TURBINE = SENSORS_PREFIX + "turbine"
TOPIC_HEARTBEAT = SENSORS_PREFIX + "system/heartbeat"
FILTER_SENSORS = SENSORS_PREFIX + "#"

ALERTS_PREFIX = "alerts/"
TOPIC_ALERT_WARNING = ALERTS_PREFIX + "warning"
TOPIC_ALERT_CRITICAL = ALERTS_PREFIX + "critical"
TOPIC_ALERT_SNAPSHOT = ALERTS_PREFIX + "snapshot"
FILTER_ALERTS = ALERTS_PREFIX + "#"

TOPIC_PLC_EVENTS = "plc/events"
TOPIC_ALARM_CHANGES = "alarms/changes"

STATUS_PREFIX = "status/"
FILTER_STATUS = STATUS_PREFIX + "+"


def availability_topic(service: str) -> str:
    """The retained online/offline topic of one service."""
    return STATUS_PREFIX + service


TOPIC_STATUS_PHYSICS_ENGINE = availability_topic("physics-engine")
TOPIC_STATUS_PLC_CONTROLLER = availability_topic("plc-controller")
