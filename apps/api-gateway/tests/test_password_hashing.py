"""Argon2 work: its parameters, its failures, and how much of it may run at once."""

from __future__ import annotations

import asyncio
import threading

import pytest
from api_gateway import accounts
from api_gateway.config import settings


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
