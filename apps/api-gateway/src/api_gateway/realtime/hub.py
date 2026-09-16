"""
Fan-out of live data to WebSocket subscribers.

One upstream per channel feeds the hub; the hub serializes each message once and hands
it to every subscriber of that channel through a bounded queue. A slow client never slows
the upstream or other clients:

- telemetry is a stream of snapshots: a subscriber gets at most its rate, and when its
  queue is full the oldest frame is dropped;
- PLC events and alarm changes must not be lost silently: a subscriber whose queue
  overflows is marked, and its connection is closed so the client reloads state over
  REST and resubscribes.

The latest frame of each kind is kept, so a new subscriber starts with current state.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Channel(StrEnum):
    TELEMETRY = "telemetry"
    PLC = "plc"
    ALARMS = "alarms"


# Frames kept as the latest state of a channel, sent to every new subscriber.
SNAPSHOT_TYPES: frozenset[tuple[Channel, str]] = frozenset(
    {(Channel.TELEMETRY, "state"), (Channel.PLC, "status")}
)


def encode(message: dict[str, Any]) -> str:
    return json.dumps(message, separators=(",", ":"), allow_nan=False)


@dataclass(eq=False)
class Subscriber:
    queue_size: int
    max_rate_hz: float
    channels: set[Channel] = field(default_factory=set)
    overflowed: bool = False
    dropped_frames: int = 0
    _last_telemetry_at: float = 0.0
    queue: asyncio.Queue[str] = field(init=False)

    def __post_init__(self) -> None:
        self.queue = asyncio.Queue(maxsize=self.queue_size)

    def telemetry_due(self, now: float) -> bool:
        if self.max_rate_hz <= 0:
            return True
        if now - self._last_telemetry_at >= 1.0 / self.max_rate_hz:
            self._last_telemetry_at = now
            return True
        return False

    def offer_frame(self, frame: str) -> None:
        """Queue a telemetry frame, dropping the oldest one if the queue is full."""
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.dropped_frames += 1
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(frame)

    def offer_event(self, frame: str) -> None:
        """Queue an event; on overflow mark the subscriber for disconnection."""
        try:
            self.queue.put_nowait(frame)
        except asyncio.QueueFull:
            self.overflowed = True

    def send_control(self, message: dict[str, Any]) -> None:
        """A reply to the client itself; never dropped silently."""
        self.offer_event(encode(message))


class RealtimeHub:
    def __init__(self, *, queue_size: int, max_rate_hz: float) -> None:
        self._queue_size = queue_size
        self._max_rate_hz = max_rate_hz
        self._subscribers: set[Subscriber] = set()
        self._latest: dict[tuple[Channel, str], str] = {}
        self._last_telemetry_monotonic: float | None = None

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def max_rate_hz(self) -> float:
        return self._max_rate_hz

    def telemetry_age_s(self) -> float | None:
        if self._last_telemetry_monotonic is None:
            return None
        return round(time.monotonic() - self._last_telemetry_monotonic, 3)

    def register(self, max_rate_hz: float | None = None) -> Subscriber:
        rate = self._max_rate_hz if max_rate_hz is None else max_rate_hz
        subscriber = Subscriber(
            queue_size=self._queue_size,
            max_rate_hz=min(max(rate, 0.1), self._max_rate_hz),
        )
        self._subscribers.add(subscriber)
        return subscriber

    def unregister(self, subscriber: Subscriber) -> None:
        self._subscribers.discard(subscriber)

    def subscribe(self, subscriber: Subscriber, channels: set[Channel]) -> None:
        added = channels - subscriber.channels
        subscriber.channels |= channels
        for (channel, _), frame in self._latest.items():
            if channel in added:
                subscriber.offer_event(frame)

    def unsubscribe(self, subscriber: Subscriber, channels: set[Channel]) -> None:
        subscriber.channels -= channels

    def publish(self, channel: Channel, kind: str, data: dict[str, Any]) -> None:
        frame = encode(
            {
                "type": "data",
                "channel": channel.value,
                "kind": kind,
                "ts_ms": int(time.time() * 1000),
                "data": data,
            }
        )
        if (channel, kind) in SNAPSHOT_TYPES:
            self._latest[(channel, kind)] = frame
        now = time.monotonic()
        if channel is Channel.TELEMETRY:
            self._last_telemetry_monotonic = now
        for subscriber in self._subscribers:
            if channel not in subscriber.channels:
                continue
            if channel is Channel.TELEMETRY:
                if subscriber.telemetry_due(now):
                    subscriber.offer_frame(frame)
            else:
                subscriber.offer_event(frame)
