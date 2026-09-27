"""Sign-in throttling under the conditions that decide whether it can be worked around.

The throttle is the only thing between a password list and the account it is aimed at, so
the interesting cases are the ones an attacker would look for: a window that has just
slid, a name spelled differently, a flood of invented names to push the real one out of
the table, and a successful sign-in used to wipe the count.
"""

from __future__ import annotations

from api_gateway.auth.throttle import LoginThrottle, ThrottlePolicy

ACCOUNT = ThrottlePolicy(max_failures=3, window_s=60.0)
CLIENT = ThrottlePolicy(max_failures=5, window_s=60.0)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


def fail(guard: LoginThrottle, username: str, client: str) -> None:
    """An attempt that was let through and then failed: its charge stays."""
    assert guard.begin_attempt(username, client).allowed


def throttle(*, max_tracked_keys: int = 10_000) -> tuple[LoginThrottle, Clock]:
    clock = Clock()
    return (
        LoginThrottle(
            per_account=ACCOUNT,
            per_client=CLIENT,
            max_tracked_keys=max_tracked_keys,
            clock=clock,
        ),
        clock,
    )


class TestCounting:
    def test_the_last_allowed_failure_does_not_lock_the_account(self) -> None:
        guard, _ = throttle()
        for _ in range(ACCOUNT.max_failures - 1):
            fail(guard, "anna", "10.0.0.1")
        assert guard.retry_after_s("anna", "10.0.0.1") == 0

    def test_one_more_does(self) -> None:
        guard, _ = throttle()
        for _ in range(ACCOUNT.max_failures):
            fail(guard, "anna", "10.0.0.1")
        assert guard.retry_after_s("anna", "10.0.0.1") == ACCOUNT.window_s

    def test_the_wait_is_whole_seconds_and_never_negative(self) -> None:
        guard, clock = throttle()
        for _ in range(ACCOUNT.max_failures):
            fail(guard, "anna", "10.0.0.1")
        clock.tick(59.4)
        assert guard.retry_after_s("anna", "10.0.0.1") == 1
        clock.tick(0.7)
        assert guard.retry_after_s("anna", "10.0.0.1") == 0

    def test_the_window_slides_rather_than_resets(self) -> None:
        guard, clock = throttle()
        fail(guard, "anna", "10.0.0.1")
        clock.tick(59.0)
        fail(guard, "anna", "10.0.0.1")
        fail(guard, "anna", "10.0.0.1")
        # Three failures, but the first is about to leave the window.
        assert guard.retry_after_s("anna", "10.0.0.1") == 1
        clock.tick(2.0)
        # Two failures inside the window: allowed again, without any failure forgotten
        # that should still count.
        assert guard.retry_after_s("anna", "10.0.0.1") == 0
        fail(guard, "anna", "10.0.0.1")
        assert guard.retry_after_s("anna", "10.0.0.1") > 0


class TestTheKeysThemselves:
    def test_an_account_is_the_same_account_however_it_is_spelled(self) -> None:
        guard, _ = throttle()
        for name in ("Anna", " anna ", "ANNA"):
            fail(guard, name, "10.0.0.1")
        assert guard.retry_after_s("anna", "10.0.0.2") > 0

    def test_a_client_is_counted_across_the_names_it_tries(self) -> None:
        guard, _ = throttle()
        for index in range(CLIENT.max_failures):
            fail(guard, f"invented-{index}", "10.0.0.9")
        # No account reached its own limit, but the address did.
        assert guard.retry_after_s("invented-0", "10.0.0.9") > 0
        assert guard.retry_after_s("anna", "10.0.0.1") == 0

    def test_a_successful_sign_in_clears_the_account_but_not_the_address(self) -> None:
        guard, _ = throttle()
        for _ in range(ACCOUNT.max_failures - 1):
            fail(guard, "anna", "10.0.0.9")
        fail(guard, "bob", "10.0.0.9")
        fail(guard, "carl", "10.0.0.9")
        fail(guard, "dave", "10.0.0.9")
        attempt = guard.begin_attempt("anna", "10.0.0.9")
        assert not attempt.allowed
        clean = guard.begin_attempt("anna", "10.0.0.1")
        guard.record_success(clean)
        assert guard.retry_after_s("anna", "10.0.0.1") == 0
        # The address still carries five failures of its own.
        assert guard.retry_after_s("anna", "10.0.0.9") > 0

    def test_a_flood_of_invented_names_cannot_push_an_account_out_of_the_table(
        self,
    ) -> None:
        # The table is bounded, so it evicts; it must evict the least recently used,
        # not the one being attacked.
        guard, clock = throttle(max_tracked_keys=4)
        for _ in range(ACCOUNT.max_failures):
            fail(guard, "anna", "10.0.0.1")
        assert guard.retry_after_s("anna", "10.0.0.1") > 0
        for index in range(20):
            clock.tick(0.1)
            fail(guard, f"invented-{index}", f"10.0.1.{index}")
            # Anna stays the most recently used by being asked about.
            assert guard.retry_after_s("anna", "10.0.0.1") > 0

    def test_a_key_forgotten_after_its_window_costs_nothing(self) -> None:
        guard, clock = throttle()
        fail(guard, "anna", "10.0.0.1")
        clock.tick(ACCOUNT.window_s + 1.0)
        # Pruned on the next question; recording again starts a fresh count.
        assert guard.retry_after_s("anna", "10.0.0.1") == 0
        for _ in range(ACCOUNT.max_failures - 1):
            fail(guard, "anna", "10.0.0.1")
        assert guard.retry_after_s("anna", "10.0.0.1") == 0
        fail(guard, "anna", "10.0.0.1")
        assert guard.retry_after_s("anna", "10.0.0.1") > 0

    def test_clearing_an_account_that_was_never_seen_is_harmless(self) -> None:
        guard, clock = throttle()
        attempt = guard.begin_attempt("nobody", "10.0.0.1")
        clock.tick(ACCOUNT.window_s + 1.0)
        # Its charge has left the window already; nothing is left to clear.
        assert guard.retry_after_s("nobody", "10.0.0.1") == 0
        guard.record_success(attempt)
        guard.release(attempt)
        assert guard.retry_after_s("nobody", "10.0.0.1") == 0


class TestAttemptsInFlight:
    """An attempt is charged when it starts, before the slow password check decides it.

    Otherwise a burst of parallel attempts would all find the counters empty and all get
    a full password check: as many guesses per window as the attacker can send at once.
    """

    def test_attempts_started_together_are_counted_before_any_is_decided(
        self,
    ) -> None:
        guard, _ = throttle()
        started = [guard.begin_attempt("anna", "10.0.0.1") for _ in range(10)]
        assert [attempt.allowed for attempt in started] == [True] * 3 + [False] * 7
        assert all(attempt.retry_after_s > 0 for attempt in started[3:])

    def test_the_attempts_allowed_are_allowed(self) -> None:
        guard, _ = throttle()
        attempt = guard.begin_attempt("anna", "10.0.0.1")
        assert attempt.allowed
        assert attempt.retry_after_s == 0

    def test_a_refused_attempt_is_not_charged(self) -> None:
        guard, clock = throttle()
        for _ in range(ACCOUNT.max_failures):
            fail(guard, "anna", "10.0.0.1")
        for _ in range(10):
            assert not guard.begin_attempt("anna", "10.0.0.1").allowed
        clock.tick(ACCOUNT.window_s + 0.1)
        # Only the three real failures counted; they have all left the window.
        assert guard.begin_attempt("anna", "10.0.0.1").allowed

    def test_a_success_does_not_count_against_its_address(self) -> None:
        guard, _ = throttle()
        for index in range(CLIENT.max_failures * 2):
            attempt = guard.begin_attempt(f"user-{index}", "10.0.0.9")
            assert attempt.allowed
            guard.record_success(attempt)
        assert guard.retry_after_s("someone", "10.0.0.9") == 0

    def test_a_released_attempt_leaves_no_trace(self) -> None:
        guard, _ = throttle()
        for _ in range(ACCOUNT.max_failures * 2):
            attempt = guard.begin_attempt("anna", "10.0.0.1")
            assert attempt.allowed
            guard.release(attempt)
        assert guard.retry_after_s("anna", "10.0.0.1") == 0

    def test_a_success_clears_what_parallel_failures_charged_to_the_account(
        self,
    ) -> None:
        guard, _ = throttle()
        failing = guard.begin_attempt("anna", "10.0.0.1")
        succeeding = guard.begin_attempt("anna", "10.0.0.1")
        assert failing.allowed and succeeding.allowed
        guard.record_success(succeeding)
        assert guard.retry_after_s("anna", "10.0.0.2") == 0
