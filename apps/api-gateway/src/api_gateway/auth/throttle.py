"""
Sign-in throttling.

Failed sign-ins are counted in a sliding window per account name and per client
address. The account name is counted whether or not such an account exists, so the
throttle answers the same for a real user and an invented one.

State lives in the gateway process: the stack runs one gateway worker. Several workers
or replicas would each count separately and need a shared store.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ThrottlePolicy:
    max_failures: int
    window_s: float


class _FailureLog:
    """Failure times per key, least recently used keys evicted once the bound is reached.

    "Used" means asked about as well as recorded: a key is asked about on exactly the
    attempts it is meant to refuse, so an account being attacked stays in the table while
    the attempt is being made. Without that, a flood of invented names would push the
    account that is actually locked out of the table and unlock it.

    The table is still bounded, so a flood of more than `max_keys` distinct names inside
    one window, between two attempts on an account, would forget the account's failures.
    Counting per account and per client address bounds how fast such a flood can go.
    """

    def __init__(self, policy: ThrottlePolicy, max_keys: int) -> None:
        self._policy = policy
        self._max_keys = max_keys
        self._failures: OrderedDict[str, deque[float]] = OrderedDict()

    def _prune(self, key: str, now: float) -> deque[float] | None:
        times = self._failures.get(key)
        if times is None:
            return None
        horizon = now - self._policy.window_s
        while times and times[0] <= horizon:
            times.popleft()
        if not times:
            del self._failures[key]
            return None
        self._failures.move_to_end(key)
        return times

    def retry_after_s(self, key: str, now: float) -> float:
        times = self._prune(key, now)
        if times is None or len(times) < self._policy.max_failures:
            return 0.0
        return times[-self._policy.max_failures] + self._policy.window_s - now

    def record(self, key: str, now: float) -> None:
        times = self._prune(key, now)
        if times is None:
            times = deque(maxlen=self._policy.max_failures)
            self._failures[key] = times
            while len(self._failures) > self._max_keys:
                self._failures.popitem(last=False)
        times.append(now)

    def clear(self, key: str) -> None:
        self._failures.pop(key, None)


class LoginThrottle:
    """Refuses sign-in attempts after too many recent failures."""

    def __init__(
        self,
        *,
        per_account: ThrottlePolicy,
        per_client: ThrottlePolicy,
        max_tracked_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._accounts = _FailureLog(per_account, max_tracked_keys)
        self._clients = _FailureLog(per_client, max_tracked_keys)
        self._clock = clock

    @staticmethod
    def _account_key(username: str) -> str:
        return username.strip().casefold()

    def retry_after_s(self, username: str, client: str) -> int:
        """Whole seconds until another attempt is allowed; 0 when it is allowed now."""
        now = self._clock()
        wait = max(
            self._accounts.retry_after_s(self._account_key(username), now),
            self._clients.retry_after_s(client, now),
        )
        return math.ceil(wait) if wait > 0 else 0

    def record_failure(self, username: str, client: str) -> None:
        now = self._clock()
        self._accounts.record(self._account_key(username), now)
        self._clients.record(client, now)

    def record_success(self, username: str) -> None:
        self._accounts.clear(self._account_key(username))
