"""
OPC UA users: who opened a session and who calls a method.

asyncua checks credentials synchronously while it activates a session, so an HTTP call
to the gateway there would block the server's event loop. Instead a username session is
activated at once and its sign-in starts in the background; methods wait for it and are
refused with BadUserAccessDenied if it failed. Browsing and reading need no sign-in: the
same plant data is published on MQTT (authentication of MQTT is hardening work, S11).

asyncua does not tell a method callback which session called it. The server's session
factory is replaced with a subclass whose `call` puts the session's user into a context
variable for the duration of the call; the callbacks read it from there. When a session
closes, its gateway session is signed out.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from asyncua import ua
from asyncua.common.utils import ServiceError
from asyncua.crypto.permission_rules import User, UserRole
from asyncua.server.internal_session import InternalSession
from asyncua.server.server import Server
from asyncua.server.user_managers import UserManager

from opcua_server.gateway import GatewayClient, GatewaySession
from opcua_server.ua_types import StatusCodes

logger = logging.getLogger(__name__)

CURRENT_USER: ContextVar[User | None] = ContextVar("opcua_user", default=None)


@dataclass(eq=False)
class GatewayUser(User):
    """A username session whose identity the gateway verifies."""

    session: GatewaySession | None = None


class GatewayUserManager(UserManager):
    def __init__(self, gateway: GatewayClient) -> None:
        self._gateway = gateway

    def get_user(
        self,
        iserver: Any,
        username: str | None = None,
        password: str | None = None,
        certificate: Any = None,
    ) -> User | None:
        if username is None:
            return User(role=UserRole.User, name=None)
        if not username or not password:
            return None
        login = asyncio.get_running_loop().create_task(
            self._gateway.login(username, password), name=f"opcua-login-{username}"
        )
        return GatewayUser(
            role=UserRole.User,
            name=username,
            session=GatewaySession(self._gateway, login),
        )


def sends_password_in_clear(token: object, peer_certificate: bytes | None) -> bool:
    """
    A username token whose password is neither encrypted in the token nor on an encrypted
    channel. The server offers only `None` and `SignAndEncrypt` endpoints, so a channel
    with a client certificate is encrypted.
    """
    return (
        isinstance(token, ua.UserNameIdentityToken)
        and not token.EncryptionAlgorithm
        and not peer_certificate
    )


class _UserAwareSession(InternalSession):
    def activate_session(
        self, params: ua.ActivateSessionParameters, peer_certificate: bytes | None
    ) -> ua.ActivateSessionResult:
        if sends_password_in_clear(params.UserIdentityToken, peer_certificate):
            logger.warning(
                "OPC UA sign-in refused: a password sent in clear on an open channel"
            )
            raise ServiceError(StatusCodes.BadIdentityTokenRejected)
        return super().activate_session(params, peer_certificate)

    async def call(self, params: Any) -> Any:
        token = CURRENT_USER.set(self.user)
        try:
            return await super().call(params)
        finally:
            CURRENT_USER.reset(token)

    async def close_session(self, delete_subs: bool = True) -> None:
        await super().close_session(delete_subs)
        user = self.user
        if isinstance(user, GatewayUser) and user.session is not None:
            closing = asyncio.get_running_loop().create_task(user.session.close())
            _CLOSING.add(closing)
            closing.add_done_callback(_CLOSING.discard)


_CLOSING: set[asyncio.Task[None]] = set()


def install_identity(server: Server, gateway: GatewayClient) -> None:
    """Username and anonymous tokens, gateway-verified users, user-aware sessions."""
    iserver = server.iserver
    iserver.set_user_manager(GatewayUserManager(gateway))

    def create_session(
        name: str, user: User | None = None, external: bool = False
    ) -> InternalSession:
        return _UserAwareSession(
            iserver,
            iserver.aspace,
            iserver.subscription_service,
            name,
            user=user or User(role=UserRole.Anonymous),
            external=external,
        )

    setattr(iserver, "create_session", create_session)  # noqa: B010
