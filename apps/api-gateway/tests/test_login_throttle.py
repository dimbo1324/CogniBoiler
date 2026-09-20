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
            guard.record_failure("anna", "10.0.0.1")
        assert guard.retry_after_s("anna", "10.0.0.1") == 0

    def test_one_more_does(self) -> None:
        guard, _ = throttle()
        for _ in range(ACCOUNT.max_failures):
            guard.record_failure("anna", "10.0.0.1")
        assert guard.retry_after_s("anna", "10.0.0.1") == ACCOUNT.window_s

    def test_the_wait_is_whole_seconds_and_never_negative(self) -> None:
        guard, clock = throttle()
        for _ in range(ACCOUNT.max_failures):
            guard.record_failure("anna", "10.0.0.1")
        clock.tick(59.4)
        assert guard.retry_after_s("anna", "10.0.0.1") == 1
        clock.tick(0.7)
        assert guard.retry_after_s("anna", "10.0.0.1") == 0

    def test_the_window_slides_rather_than_resets(self) -> None:
        guard, clock = throttle()
        guard.record_failure("anna", "10.0.0.1")
        clock.tick(59.0)
        guard.record_failure("anna", "10.0.0.1")
        guard.record_failure("anna", "10.0.0.1")
        # Three failures, but the first is about to leave the window.
        assert guard.retry_after_s("anna", "10.0.0.1") == 1
        clock.tick(2.0)
        # Two failures inside the window: allowed again, without any failure forgotten
        # that should still count.
        assert guard.retry_after_s("anna", "10.0.0.1") == 0
        guard.record_failure("anna", "10.0.0.1")
        assert guard.retry_after_s("anna", "10.0.0.1") > 0


class TestTheKeysThemselves:
    def test_an_account_is_the_same_account_however_it_is_spelled(self) -> None:
        guard, _ = throttle()
        for name in ("Anna", " anna ", "ANNA"):
            guard.record_failure(name, "10.0.0.1")
        assert guard.retry_after_s("anna", "10.0.0.2") > 0

    def test_a_client_is_counted_across_the_names_it_tries(self) -> None:
        guard, _ = throttle()
        for index in range(CLIENT.max_failures):
            guard.record_failure(f"invented-{index}", "10.0.0.9")
        # No account reached its own limit, but the address did.
        assert guard.retry_after_s("invented-0", "10.0.0.9") > 0
        assert guard.retry_after_s("anna", "10.0.0.1") == 0

    def test_a_successful_sign_in_clears_the_account_but_not_the_address(self) -> None:
        guard, _ = throttle()
        for _ in range(ACCOUNT.max_failures):
            guard.record_failure("anna", "10.0.0.9")
        guard.record_failure("bob", "10.0.0.9")
        guard.record_failure("carl", "10.0.0.9")
        guard.record_success("anna")
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
            guard.record_failure("anna", "10.0.0.1")
        assert guard.retry_after_s("anna", "10.0.0.1") > 0
        for index in range(20):
            clock.tick(0.1)
            guard.record_failure(f"invented-{index}", f"10.0.1.{index}")
            # Anna stays the most recently used by being asked about.
            assert guard.retry_after_s("anna", "10.0.0.1") > 0

    def test_a_key_forgotten_after_its_window_costs_nothing(self) -> None:
        guard, clock = throttle()
        guard.record_failure("anna", "10.0.0.1")
        clock.tick(ACCOUNT.window_s + 1.0)
        # Pruned on the next question; recording again starts a fresh count.
        assert guard.retry_after_s("anna", "10.0.0.1") == 0
        for _ in range(ACCOUNT.max_failures - 1):
            guard.record_failure("anna", "10.0.0.1")
        assert guard.retry_after_s("anna", "10.0.0.1") == 0
        guard.record_failure("anna", "10.0.0.1")
        assert guard.retry_after_s("anna", "10.0.0.1") > 0

    def test_clearing_an_account_that_was_never_seen_is_harmless(self) -> None:
        guard, _ = throttle()
        guard.record_success("nobody")
        assert guard.retry_after_s("nobody", "10.0.0.1") == 0
