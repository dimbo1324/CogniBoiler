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
from cogniboiler_runtime.mqtt_queue import (
    DEFAULT_QUEUE_LIMIT,
    QueuedMessage,
    QueuedMqttPublisher,
)
from cogniboiler_runtime.outage import OutageLog
from cogniboiler_runtime.payloads import (
    MAX_JSON_PAYLOAD_BYTES,
    decode_json_object,
    finite_number,
)
from cogniboiler_runtime.service import (
    STOP_SIGNALS,
    run_service,
    run_until_signalled,
)

__all__ = [
    "DEFAULT_BROKER_ERRORS",
    "DEFAULT_INTERVAL_S",
    "DEFAULT_MAX_AGE_S",
    "DEFAULT_QUEUE_LIMIT",
    "DEFAULT_RECONNECT_DELAY_S",
    "MAX_JSON_PAYLOAD_BYTES",
    "MILLISECONDS_PER_DAY",
    "MILLISECONDS_PER_MINUTE",
    "MILLISECONDS_PER_SECOND",
    "NANOSECONDS_PER_MILLISECOND",
    "SECONDS_PER_DAY",
    "STOP_SIGNALS",
    "LivenessFile",
    "MqttSession",
    "OutageLog",
    "QueuedMessage",
    "QueuedMqttPublisher",
    "decode_json_object",
    "finite_number",
    "is_fresh",
    "now_ms",
    "reconnect_jitter",
    "run_service",
    "run_until_signalled",
    "subscribe_all",
]
