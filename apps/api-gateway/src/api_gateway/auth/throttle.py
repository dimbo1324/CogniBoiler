"""
Sign-in throttling.

Failed sign-ins are counted in a sliding window per account name and per client
address. The account name is counted whether or not such an account exists, so the
throttle answers the same for a real user and an invented one.

An attempt is charged as a failure the moment it is let through, before the slow
password check decides it; a success then clears the account and takes the attempt's
charge off the address. Checking first and counting after the check would let a burst of
parallel attempts all find the counters empty.

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

    def discard(self, key: str, when: float) -> None:
        times = self._failures.get(key)
        if times is None:
            return
        try:
            times.remove(when)
        except ValueError:
            return
        if not times:
            del self._failures[key]


@dataclass(frozen=True, slots=True)
class Attempt:
    """One sign-in attempt; when it was let through, it is charged at `charged_at`."""

    account_key: str
    client: str
    retry_after_s: int
    charged_at: float | None

    @property
    def allowed(self) -> bool:
        return self.charged_at is not None


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
        return self._wait(self._account_key(username), client, self._clock())

    def _wait(self, account_key: str, client: str, now: float) -> int:
        wait = max(
            self._accounts.retry_after_s(account_key, now),
            self._clients.retry_after_s(client, now),
        )
        return math.ceil(wait) if wait > 0 else 0

    def begin_attempt(self, username: str, client: str) -> Attempt:
        """
        Let an attempt through and charge it as a failure, or refuse it uncharged.

        The check and the charge are one step with no await between them, so parallel
        attempts are counted in the order they arrive.
        """
        now = self._clock()
        account_key = self._account_key(username)
        wait = self._wait(account_key, client, now)
        if wait:
            return Attempt(account_key, client, wait, None)
        self._accounts.record(account_key, now)
        self._clients.record(client, now)
        return Attempt(account_key, client, 0, now)

    def record_success(self, attempt: Attempt) -> None:
        """The account proved its password: forget its failures and this charge."""
        self._accounts.clear(attempt.account_key)
        self.release(attempt)

    def release(self, attempt: Attempt) -> None:
        """Take back the charge of an attempt that turned out not to be a failure."""
        if attempt.charged_at is None:
            return
        self._accounts.discard(attempt.account_key, attempt.charged_at)
        self._clients.discard(attempt.client, attempt.charged_at)
