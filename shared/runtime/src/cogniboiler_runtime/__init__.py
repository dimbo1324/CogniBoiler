"""Runtime pieces every CogniBoiler service needs and none of them owns.

`cogniboiler_observability` answers "what is this service saying"; this package answers
"is it still running, and is it still connected".
"""

from cogniboiler_runtime.clock import (
    MILLISECONDS_PER_DAY,
    MILLISECONDS_PER_MINUTE,
    MILLISECONDS_PER_SECOND,
    NANOSECONDS_PER_MILLISECOND,
    SECONDS_PER_DAY,
    now_ms,
)
from cogniboiler_runtime.liveness import (
    DEFAULT_INTERVAL_S,
    DEFAULT_MAX_AGE_S,
    LivenessFile,
    is_fresh,
)
from cogniboiler_runtime.mqtt import (
    DEFAULT_BROKER_ERRORS,
    DEFAULT_RECONNECT_DELAY_S,
    MqttSession,
    reconnect_jitter,
    subscribe_all,
)

__all__ = [
    "DEFAULT_BROKER_ERRORS",
    "DEFAULT_INTERVAL_S",
    "DEFAULT_MAX_AGE_S",
    "DEFAULT_RECONNECT_DELAY_S",
    "MILLISECONDS_PER_DAY",
    "MILLISECONDS_PER_MINUTE",
    "MILLISECONDS_PER_SECOND",
    "NANOSECONDS_PER_MILLISECOND",
    "SECONDS_PER_DAY",
    "LivenessFile",
    "MqttSession",
    "is_fresh",
    "now_ms",
    "reconnect_jitter",
    "subscribe_all",
]
