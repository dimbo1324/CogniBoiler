"""Gateway test fixtures that must not depend on a developer's .env."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator

import pytest
import pytest_asyncio
from api_gateway.auth.identity import load_account
from api_gateway.auth.sessions import ClientInfo, open_session
from api_gateway.config import settings
from api_gateway.dependencies import get_db
from api_gateway.main import create_app
from api_gateway.models.user import Base
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from gateway_fakes import (
    FakeAlarmClient,
    FakeHistorianClient,
    FakePhysicsClient,
    FakePLCClient,
    seed_accounts,
)
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


def _pem_key_pair() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private.decode("ascii"), public.decode("ascii")


@pytest.fixture(scope="session", autouse=True)
def ephemeral_jwt_keys() -> Iterator[None]:
    """Sign and verify tokens with a key pair that exists only for this test run."""
    original = (settings.jwt_private_key, settings.jwt_public_key)
    settings.jwt_private_key, settings.jwt_public_key = _pem_key_pair()
    yield
    settings.jwt_private_key, settings.jwt_public_key = original


@pytest_asyncio.fixture
async def app() -> AsyncGenerator[FastAPI]:
    """Fresh FastAPI application with an in-memory database and fake upstreams."""
    application = create_app()
    application.state.physics_client = FakePhysicsClient()
    application.state.plc_client = FakePLCClient()
    application.state.alarm_client = FakeAlarmClient()
    application.state.historian_client = FakeHistorianClient()

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def override_get_db() -> AsyncGenerator:
        async with session_factory() as session:
            yield session

    application.dependency_overrides[get_db] = override_get_db

    async with session_factory() as session:
        await seed_accounts(session)

    yield application

    application.dependency_overrides.clear()
    await engine.dispose()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient]:
    """HTTP client wired to the application through ASGI; no socket is opened."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


async def session_tokens(app: FastAPI, username: str) -> dict[str, str]:
    """A signed-in session of a seeded user, stored in the test database."""
    sessions = app.dependency_overrides[get_db]()
    db = await anext(sessions)
    try:
        account = await load_account(db, username=username)
        assert account is not None
        tokens = await open_session(
            db, account, ClientInfo(ip="127.0.0.1", user_agent="pytest")
        )
    finally:
        await sessions.aclose()
    return {"access": tokens.access.token, "refresh": tokens.refresh.token}


@pytest_asyncio.fixture
async def viewer_tokens(app: FastAPI) -> dict[str, str]:
    return await session_tokens(app, "viewer1")


@pytest_asyncio.fixture
async def operator_tokens(app: FastAPI) -> dict[str, str]:
    return await session_tokens(app, "operator1")


@pytest_asyncio.fixture
async def engineer_tokens(app: FastAPI) -> dict[str, str]:
    return await session_tokens(app, "engineer1")


@pytest_asyncio.fixture
async def admin_tokens(app: FastAPI) -> dict[str, str]:
    return await session_tokens(app, "admin1")
