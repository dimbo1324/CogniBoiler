"""The identity lifecycle of an OPC UA session: sign-in, re-activation and sign-out."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from asyncua import ua
from asyncua.common.utils import ServiceError
from asyncua.server.internal_session import InternalSession
from opcua_server.gateway import GatewaySession, GatewayTokens
from opcua_server.identity import GatewayUser, drain_sign_outs, install_identity
from opcua_server.ua_types import StatusCodes

PEER = ("192.0.2.10", 50123)
CERTIFICATE = b"client certificate"


class FakeGateway:
    """The sign-in side of GatewayClient, recording what the gateway was asked."""

    def __init__(self) -> None:
        self.logins: list[tuple[str, str | None]] = []
        self.logouts: list[str] = []

    async def login(
        self, username: str, password: str, client_address: str | None = None
    ) -> GatewayTokens | None:
        self.logins.append((username, client_address))
        if password != "pw":
            return None
        return GatewayTokens(
            access_token=f"a{len(self.logins)}",
            refresh_token=f"r{len(self.logins)}",
            access_expires_at_ms=int(time.time() * 1000) + 900_000,
            username=username,
            role="operator",
        )

    async def logout(
        self, refresh_token: str, client_address: str | None = None
    ) -> None:
        self.logouts.append(refresh_token)


class FakeSubscriptions:
    subscriptions: dict[int, Any] = {}

    async def delete_subscriptions(self, ids: list[int]) -> list[Any]:
        return []


class FakeInternalServer:
    """What InternalSession needs from asyncua's InternalServer, and nothing more."""

    supported_tokens = (ua.AnonymousIdentityToken, ua.UserNameIdentityToken)

    def __init__(self) -> None:
        self.user_manager: Any = None
        self.aspace = None
        self.subscription_service = FakeSubscriptions()
        self.sessions: dict[ua.NodeId, InternalSession] = {}

    def set_user_manager(self, user_manager: Any) -> None:
        self.user_manager = user_manager

    def register_external_session(self, session: InternalSession) -> None:
        self.sessions[session.auth_token] = session

    def unregister_external_session(self, session: InternalSession) -> None:
        self.sessions.pop(session.auth_token, None)

    def decrypt_user_token(self, session: Any, token: Any) -> tuple[str, str]:
        password = token.Password
        return token.UserName, (
            password.decode() if isinstance(password, bytes) else password
        )


class Harness:
    def __init__(self) -> None:
        self.gateway = FakeGateway()
        self.iserver = FakeInternalServer()
        install_identity(SimpleNamespace(iserver=self.iserver), self.gateway)  # type: ignore[arg-type]
        self.sessions: list[InternalSession] = []

    def session(self, peer: Any = PEER) -> InternalSession:
        created: InternalSession = self.iserver.create_session(peer, external=True)  # type: ignore[attr-defined]
        self.sessions.append(created)
        return created


@pytest_asyncio.fixture
async def harness() -> AsyncIterator[Harness]:
    found = Harness()
    try:
        yield found
    finally:
        for session in found.sessions:
            await session.close_session()
        await drain_sign_outs(5.0)


def username_token(
    password: bytes = b"pw", encryption: str | None = None
) -> ua.ActivateSessionParameters:
    params = ua.ActivateSessionParameters()
    params.UserIdentityToken = ua.UserNameIdentityToken(
        UserName="operator1", Password=password, EncryptionAlgorithm=encryption
    )
    return params


def anonymous_token() -> ua.ActivateSessionParameters:
    params = ua.ActivateSessionParameters()
    params.UserIdentityToken = ua.AnonymousIdentityToken()
    return params


async def signed_in(session: InternalSession) -> GatewaySession:
    user = session.user
    assert isinstance(user, GatewayUser) and user.session is not None
    assert await user.session.tokens() is not None
    return user.session


class TestClientAddress:
    @pytest.mark.parametrize(
        ("peer", "forwarded"),
        [
            (("192.0.2.10", 50123), "192.0.2.10"),
            (("2001:db8::7", 50123, 0, 0), "2001:db8::7"),
            (("not an address", 1), None),
            ("192.0.2.10", None),
            (None, None),
        ],
    )
    async def test_the_sign_in_names_the_peer_when_it_is_an_address(
        self, harness: Harness, peer: Any, forwarded: str | None
    ) -> None:
        session = harness.session(peer)
        session.activate_session(username_token(), CERTIFICATE)
        await signed_in(session)
        assert harness.gateway.logins == [("operator1", forwarded)]


class TestAuthenticationToken:
    async def test_tokens_are_random_secrets_registered_in_place_of_the_counter(
        self, harness: Harness
    ) -> None:
        first, second = harness.session(), harness.session()
        for session in (first, second):
            assert session.auth_token.NodeIdType == ua.NodeIdType.ByteString
            assert len(session.auth_token.Identifier) == 32
            assert harness.iserver.sessions[session.auth_token] is session
        assert first.auth_token != second.auth_token
        assert len(harness.iserver.sessions) == 2

    async def test_a_re_activation_from_the_same_channel_certificate_is_accepted(
        self, harness: Harness
    ) -> None:
        session = harness.session()
        session.activate_session(anonymous_token(), CERTIFICATE)
        session.activate_session(anonymous_token(), CERTIFICATE)
        assert session.is_activated()

    @pytest.mark.parametrize(
        ("first", "then"),
        [(CERTIFICATE, None), (None, CERTIFICATE), (CERTIFICATE, b"another client")],
    )
    async def test_a_re_activation_from_another_channel_certificate_is_refused(
        self, harness: Harness, first: bytes | None, then: bytes | None
    ) -> None:
        session = harness.session()
        session.activate_session(anonymous_token(), first)
        with pytest.raises(ServiceError) as refused:
            session.activate_session(anonymous_token(), then)
        assert refused.value.code == StatusCodes.BadSecurityChecksFailed


class TestPasswordEncryption:
    @pytest.mark.parametrize(
        "algorithm",
        [
            "http://www.w3.org/2001/04/xmlenc#rsa-oaep",
            "http://opcfoundation.org/UA/security/rsa-oaep-sha2-256",
        ],
    )
    async def test_an_oaep_encrypted_password_is_accepted(
        self, harness: Harness, algorithm: str
    ) -> None:
        session = harness.session()
        session.activate_session(username_token(encryption=algorithm), None)
        await signed_in(session)

    @pytest.mark.parametrize(
        "algorithm",
        ["http://www.w3.org/2001/04/xmlenc#rsa-1_5", "urn:unknown"],
    )
    async def test_legacy_or_unknown_password_encryption_is_refused(
        self, harness: Harness, algorithm: str
    ) -> None:
        session = harness.session()
        with pytest.raises(ServiceError) as refused:
            session.activate_session(username_token(encryption=algorithm), CERTIFICATE)
        assert refused.value.code == StatusCodes.BadIdentityTokenRejected
        assert harness.gateway.logins == []


class TestSignOut:
    async def test_closing_a_session_signs_its_user_out_once(
        self, harness: Harness
    ) -> None:
        session = harness.session()
        session.activate_session(username_token(), CERTIFICATE)
        await signed_in(session)
        await session.close_session()
        await session.close_session()
        await drain_sign_outs(5.0)
        assert harness.gateway.logouts == ["r1"]

    async def test_re_activation_signs_out_the_replaced_user(
        self, harness: Harness
    ) -> None:
        session = harness.session()
        session.activate_session(username_token(), CERTIFICATE)
        first = await signed_in(session)
        session.activate_session(username_token(), CERTIFICATE)
        await signed_in(session)
        await drain_sign_outs(5.0)
        assert harness.gateway.logouts == ["r1"]
        assert await first.tokens() is None
        await session.close_session()
        await drain_sign_outs(5.0)
        assert harness.gateway.logouts == ["r1", "r2"]

    async def test_re_activation_as_anonymous_signs_out_the_user(
        self, harness: Harness
    ) -> None:
        session = harness.session()
        session.activate_session(username_token(), CERTIFICATE)
        await signed_in(session)
        session.activate_session(anonymous_token(), CERTIFICATE)
        await drain_sign_outs(5.0)
        assert harness.gateway.logouts == ["r1"]
        assert not isinstance(session.user, GatewayUser)

    async def test_a_refused_re_activation_keeps_the_signed_in_user(
        self, harness: Harness
    ) -> None:
        session = harness.session()
        session.activate_session(username_token(), CERTIFICATE)
        kept = await signed_in(session)
        with pytest.raises(ServiceError) as refused:
            session.activate_session(username_token(password=b""), CERTIFICATE)
        assert refused.value.code == StatusCodes.BadUserAccessDenied
        await drain_sign_outs(5.0)
        assert harness.gateway.logouts == []
        assert await kept.tokens() is not None

    async def test_draining_waits_for_a_scheduled_sign_out(
        self, harness: Harness
    ) -> None:
        release = asyncio.Event()
        original = harness.gateway.logout

        async def slow_logout(
            refresh_token: str, client_address: str | None = None
        ) -> None:
            await release.wait()
            await original(refresh_token)

        harness.gateway.logout = slow_logout  # type: ignore[method-assign]
        session = harness.session()
        session.activate_session(username_token(), CERTIFICATE)
        await signed_in(session)
        await session.close_session()
        asyncio.get_running_loop().call_soon(release.set)
        await drain_sign_outs(5.0)
        assert harness.gateway.logouts == ["r1"]
