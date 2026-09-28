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
closes, or a new activation replaces its user, the gateway session of that user is
signed out, once.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import secrets
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from asyncua import ua
from asyncua.common.utils import ServiceError
from asyncua.crypto.permission_rules import User, UserRole
from asyncua.server.internal_session import InternalSession, SessionState
from asyncua.server.server import Server
from asyncua.server.user_managers import UserManager

from opcua_server.gateway import GatewayClient, GatewaySession
from opcua_server.ua_types import StatusCodes

logger = logging.getLogger(__name__)

CURRENT_USER: ContextVar[User | None] = ContextVar("opcua_user", default=None)
# asyncua hands the user manager no session, so the activating session's peer travels
# to get_user in a context variable, set only while activate_session runs.
_ACTIVATING_PEER: ContextVar[str | None] = ContextVar("opcua_peer", default=None)


def peer_address(name: object) -> str | None:
    """The IP address in asyncua's session name (the socket's peername), if any."""
    if not isinstance(name, tuple) or not name or not isinstance(name[0], str):
        return None
    try:
        return str(ipaddress.ip_address(name[0]))
    except ValueError:
        return None


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
        client_address = _ACTIVATING_PEER.get()
        login = asyncio.get_running_loop().create_task(
            self._gateway.login(username, password, client_address=client_address),
            name="opcua-login",
        )
        return GatewayUser(
            role=UserRole.User,
            name=username,
            session=GatewaySession(self._gateway, login, client_address),
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


AUTH_TOKEN_BYTES = 32
ACCEPTED_PASSWORD_ENCRYPTION = frozenset(
    {
        "http://www.w3.org/2001/04/xmlenc#rsa-oaep",
        "http://opcfoundation.org/UA/security/rsa-oaep-sha2-256",
    }
)


def uses_refused_password_encryption(token: object) -> bool:
    """
    A username token encrypted with anything but RSA-OAEP. asyncua would also decrypt
    PKCS#1 v1.5 (rsa-1_5), whose padding errors are the shape of a Bleichenbacher
    oracle; the advertised policy, Basic256Sha256, uses OAEP.
    """
    return (
        isinstance(token, ua.UserNameIdentityToken)
        and bool(token.EncryptionAlgorithm)
        and token.EncryptionAlgorithm not in ACCEPTED_PASSWORD_ENCRYPTION
    )


class _UserAwareSession(InternalSession):
    def __init__(
        self,
        internal_server: Any,
        aspace: Any,
        submgr: Any,
        name: str,
        user: User,
        external: bool = False,
    ) -> None:
        super().__init__(internal_server, aspace, submgr, name, user, external)
        # asyncua numbers AuthenticationTokens 1000, 1001, ... and lets any new
        # SecureChannel re-activate a live session by its token; OPC UA Part 4 wants a
        # secret. The session is registered under the counter first, so it moves.
        if external:
            internal_server.unregister_external_session(self)
        self.auth_token = ua.NodeId(
            ua.ByteString(secrets.token_bytes(AUTH_TOKEN_BYTES)),
            ua.Int16(0),
            ua.NodeIdType.ByteString,
        )
        if external:
            internal_server.register_external_session(self)
        self._channel_certificate: bytes | None = None

    def activate_session(
        self, params: ua.ActivateSessionParameters, peer_certificate: bytes | None
    ) -> ua.ActivateSessionResult:
        token = params.UserIdentityToken
        if sends_password_in_clear(token, peer_certificate):
            logger.warning(
                "OPC UA sign-in refused: a password sent in clear on an open channel"
            )
            raise ServiceError(StatusCodes.BadIdentityTokenRejected)
        if uses_refused_password_encryption(token):
            logger.warning(
                "OPC UA sign-in refused: password encryption %r is not accepted",
                str(token.EncryptionAlgorithm)[:100],
            )
            raise ServiceError(StatusCodes.BadIdentityTokenRejected)
        # Part 4 §5.6.3: a later activation, possibly on a new SecureChannel, must come
        # with the certificate of the channel the session was first activated on.
        if self.is_activated() and peer_certificate != self._channel_certificate:
            logger.warning(
                "OPC UA session re-activation refused: another channel certificate"
            )
            raise ServiceError(StatusCodes.BadSecurityChecksFailed)
        first_activation = not self.is_activated()
        previous = self.user
        peer = _ACTIVATING_PEER.set(peer_address(self.name))
        try:
            result = super().activate_session(params, peer_certificate)
        finally:
            _ACTIVATING_PEER.reset(peer)
        # OPC UA lets a client activate a live session again, as another user or the
        # same one; asyncua then simply replaces the user, and the gateway session of
        # the one replaced would stay open until its refresh token expired.
        if first_activation:
            self._channel_certificate = peer_certificate
        if self.user is not previous:
            _schedule_sign_out(previous)
        return result

    async def call(self, params: Any) -> Any:
        token = CURRENT_USER.set(self.user)
        try:
            return await super().call(params)
        finally:
            CURRENT_USER.reset(token)

    async def close_session(self, delete_subs: bool = True) -> None:
        # asyncua closes a session again when its transport goes away after a
        # CloseSession; only the first close ends the user's gateway session.
        was_open = self.state is not SessionState.Closed
        await super().close_session(delete_subs)
        if was_open:
            _schedule_sign_out(self.user)


_CLOSING: set[asyncio.Task[None]] = set()


def _schedule_sign_out(user: User | None) -> None:
    if not isinstance(user, GatewayUser) or user.session is None:
        return
    closing = asyncio.get_running_loop().create_task(
        user.session.close(), name="opcua-sign-out"
    )
    _CLOSING.add(closing)
    closing.add_done_callback(_signed_out)


def _signed_out(task: asyncio.Task[None]) -> None:
    _CLOSING.discard(task)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.warning(
            "Gateway sign-out failed: %s", type(error).__name__, exc_info=error
        )


async def drain_sign_outs(timeout_s: float) -> None:
    """Wait at most `timeout_s` for the sign-outs already scheduled to finish."""
    pending = set(_CLOSING)
    if not pending:
        return
    _, unfinished = await asyncio.wait(pending, timeout=timeout_s)
    if unfinished:
        logger.warning("%d gateway sign-outs did not finish in time", len(unfinished))


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
