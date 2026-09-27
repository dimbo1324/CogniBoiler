"""Argon2 work: its parameters, its failures, and how much of it may run at once."""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest
from api_gateway import accounts
from api_gateway.auth import password
from api_gateway.auth.password import hash_password, verify_password
from api_gateway.config import settings
from api_gateway.routers import auth as auth_router


class GatedHash:
    """A stand-in for Argon2 that holds its worker thread until the test releases it."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.active = 0
        self.most_at_once = 0

    def __call__(self, *_: str) -> bool:
        with self._lock:
            self.active += 1
            self.most_at_once = max(self.most_at_once, self.active)
        try:
            self.release.wait(timeout=10.0)
        finally:
            with self._lock:
                self.active -= 1
        return False


async def wait_until(condition: GatedHash, active: int) -> None:
    for _ in range(500):
        if condition.active >= active:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"never reached {active} hashes at once")


class TestConcurrentHashing:
    async def test_no_more_hashes_than_the_limit_run_at_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Each Argon2 run holds 64 MiB: a burst of sign-ins must queue, not allocate.
        monkeypatch.setattr(settings, "login_max_concurrent_hashes", 2)
        gate = GatedHash()
        monkeypatch.setattr(accounts, "verify_password", gate)
        checks = [
            asyncio.create_task(accounts.verify_password_async("pw", "hash"))
            for _ in range(6)
        ]
        await wait_until(gate, 2)
        await asyncio.sleep(0.05)
        assert gate.active == 2
        gate.release.set()
        assert await asyncio.gather(*checks) == [False] * 6
        assert gate.most_at_once == 2

    async def test_hashing_and_verifying_share_the_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "login_max_concurrent_hashes", 1)
        gate = GatedHash()
        monkeypatch.setattr(accounts, "verify_password", gate)
        monkeypatch.setattr(accounts, "hash_password", gate)
        work = [
            asyncio.create_task(accounts.verify_password_async("pw", "hash")),
            asyncio.create_task(accounts.hash_password_async("pw")),
        ]
        await wait_until(gate, 1)
        await asyncio.sleep(0.05)
        gate.release.set()
        await asyncio.gather(*work)
        assert gate.most_at_once == 1

    async def test_up_to_the_limit_runs_in_parallel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "login_max_concurrent_hashes", 3)
        gate = GatedHash()
        monkeypatch.setattr(accounts, "verify_password", gate)
        checks = [
            asyncio.create_task(accounts.verify_password_async("pw", "hash"))
            for _ in range(3)
        ]
        await wait_until(gate, 3)
        gate.release.set()
        await asyncio.gather(*checks)
        assert gate.most_at_once == 3


class TestParameters:
    def test_the_argon2id_cost_is_pinned(self) -> None:
        # Pinned rather than taken from library defaults: an upgrade that changed them
        # would give the unknown-user dummy hash a different cost from stored hashes.
        assert hash_password("x").startswith("$argon2id$v=19$m=65536,t=3,p=4$")

    def test_the_dummy_hash_costs_what_a_stored_hash_costs(self) -> None:
        prefix = hash_password("x").rsplit("$", 2)[0]
        assert auth_router._DUMMY_HASH.startswith(prefix)


class TestVerificationFailures:
    def test_a_right_password_passes(self) -> None:
        assert verify_password("correct horse", hash_password("correct horse"))

    def test_a_hash_nobody_can_read_is_a_mismatch_and_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="api_gateway.auth.password"):
            assert verify_password("pw", "md5$not-a-hash-we-know") is False
        assert "malformed" in caplog.text
        assert "not-a-hash-we-know" not in caplog.text

    def test_a_wrong_password_is_a_mismatch_without_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="api_gateway.auth.password"):
            assert verify_password("wrong", hash_password("right")) is False
        assert caplog.text == ""

    def test_an_unexpected_fault_is_not_turned_into_a_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def out_of_memory(*_: object) -> bool:
            raise MemoryError

        monkeypatch.setattr(password._hasher, "verify", out_of_memory)
        with pytest.raises(MemoryError):
            verify_password("pw", hash_password("pw"))
