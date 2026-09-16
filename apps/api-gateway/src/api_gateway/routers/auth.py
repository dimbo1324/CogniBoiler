"""
Authentication endpoints: sign-in, token refresh, sign-out, profile, password change.

The refresh token travels two ways. Every response that issues one also sets it as an
httpOnly, SameSite=Strict cookie scoped to /auth, which is what a browser should use:
script cannot read it and other sites cannot send it. The same value is in the JSON body
for non-browser clients (smoke checks, the OPC UA server). /auth/refresh and
/auth/logout accept the body field or, when it is absent, the cookie.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from fastapi.security.utils import get_authorization_scheme_param

from api_gateway.accounts import change_own_password, verify_password_async
from api_gateway.audit import (
    AuditActor,
    client_address,
    set_audit_actor,
    set_audit_detail,
    set_audit_outcome,
)
from api_gateway.auth.identity import (
    AuthenticationError,
    load_account,
    resolve_access_token,
)
from api_gateway.auth.password import hash_password
from api_gateway.auth.rbac import AuthenticatedUser
from api_gateway.auth.sessions import (
    ClientInfo,
    RefreshRejectedError,
    SessionTokens,
    open_session,
    revoke_family,
    rotate_session,
    session_id_of,
)
from api_gateway.auth.throttle import LoginThrottle
from api_gateway.config import settings
from api_gateway.dependencies import DbSession
from api_gateway.problems import ProblemError
from api_gateway.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    MessageResponse,
    PasswordChangeRequest,
    ProfileResponse,
    RefreshRequest,
    TokenResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])

# Verified against when the account does not exist, so an unknown username costs the
# same Argon2 time as a wrong password.
_DUMMY_HASH = hash_password("cogniboiler-dummy-password")


def _invalid_credentials() -> ProblemError:
    return ProblemError(
        401, "auth.invalid_credentials", "Invalid username or password."
    )


def _throttle(request: Request) -> LoginThrottle:
    return request.app.state.login_throttle  # type: ignore[no-any-return]


def _client(request: Request) -> ClientInfo:
    return ClientInfo(
        ip=client_address(request.scope)[:45],
        user_agent=request.headers.get("user-agent", ""),
    )


def _set_refresh_cookie(response: Response, tokens: SessionTokens) -> None:
    max_age = max(
        (tokens.refresh.expires_at_ms - tokens.refresh.issued_at_ms) // 1000, 0
    )
    response.set_cookie(
        settings.refresh_cookie_name,
        tokens.refresh.token,
        max_age=max_age,
        path=settings.refresh_cookie_path,
        secure=settings.refresh_cookie_secure,
        httponly=True,
        samesite="strict",
    )


def _cookie_clearing_headers() -> dict[str, str]:
    response = Response()
    response.delete_cookie(
        settings.refresh_cookie_name,
        path=settings.refresh_cookie_path,
        secure=settings.refresh_cookie_secure,
        httponly=True,
        samesite="strict",
    )
    return {"Set-Cookie": response.headers["set-cookie"]}


def _token_response(tokens: SessionTokens) -> TokenResponse:
    return TokenResponse(
        access_token=tokens.access.token,
        refresh_token=tokens.refresh.token,
        expires_in=max(
            (tokens.access.expires_at_ms - tokens.access.issued_at_ms) // 1000, 0
        ),
        access_expires_at_ms=tokens.access.expires_at_ms,
        session_expires_at_ms=tokens.refresh.expires_at_ms,
        username=tokens.account.username,
        role=tokens.account.role,
    )


def _signed_in(request: Request, tokens: SessionTokens, outcome: str) -> None:
    account = tokens.account
    set_audit_actor(request, AuditActor(account.id, account.username, account.role))
    set_audit_outcome(request, outcome)


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request, response: Response, body: LoginRequest, db: DbSession
) -> TokenResponse:
    """
    Sign in with a username and password.

    An unknown user, a wrong password and a blocked account get the same 401. After too
    many failures for one account name or one client the answer is 429 with Retry-After,
    again whether or not the account exists.
    """
    client = _client(request)
    set_audit_detail(request, f"username={body.username}")
    throttle = _throttle(request)
    retry_after = throttle.retry_after_s(body.username, client.ip)
    if retry_after:
        set_audit_outcome(request, "refused: too many failed attempts")
        raise ProblemError(
            429,
            "auth.too_many_attempts",
            "Too many failed sign-in attempts. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )

    account = await load_account(db, username=body.username)
    password_ok = await verify_password_async(
        body.password, account.hashed_password if account else _DUMMY_HASH
    )
    if account is None or not password_ok or not account.can_sign_in:
        throttle.record_failure(body.username, client.ip)
        set_audit_outcome(request, "refused: invalid credentials")
        raise _invalid_credentials()

    throttle.record_success(body.username)
    tokens = await open_session(db, account, client)
    _set_refresh_cookie(response, tokens)
    _signed_in(request, tokens, "signed in")
    return _token_response(tokens)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    request: Request,
    response: Response,
    db: DbSession,
    body: RefreshRequest | None = None,
) -> TokenResponse:
    """
    Exchange a refresh token for a new pair; the presented token becomes unusable.

    Presenting an exchanged token again closes the whole session.
    """
    token = (body.refresh_token if body else None) or request.cookies.get(
        settings.refresh_cookie_name
    )
    if not token:
        raise ProblemError(
            401, "auth.refresh_missing", "No refresh token was presented."
        )
    try:
        tokens = await rotate_session(db, token, _client(request))
    except RefreshRejectedError as exc:
        set_audit_outcome(
            request,
            "refused: token reuse, session closed"
            if exc.session_revoked
            else f"refused: {exc.code}",
        )
        raise ProblemError(
            401,
            exc.code,
            exc.detail,
            headers=_cookie_clearing_headers() if exc.session_revoked else None,
        ) from exc
    _set_refresh_cookie(response, tokens)
    _signed_in(request, tokens, "session refreshed")
    return _token_response(tokens)


@router.post("/logout", response_model=MessageResponse)
async def logout(
    request: Request,
    response: Response,
    db: DbSession,
    body: LogoutRequest | None = None,
) -> MessageResponse:
    """
    Close the session of the presented refresh token (body or cookie) or bearer token.

    Always answers 200, so it reveals nothing about the token.
    """
    sessions: set[str] = set()
    token = (body.refresh_token if body else None) or request.cookies.get(
        settings.refresh_cookie_name
    )
    if token and (found := session_id_of(token)) is not None:
        sessions.add(found[0])

    scheme, bearer = get_authorization_scheme_param(
        request.headers.get("Authorization")
    )
    if scheme.lower() == "bearer" and bearer:
        try:
            user = await resolve_access_token(db, bearer)
        except AuthenticationError:
            user = None
        if user is not None:
            sessions.add(user.session_id)
            set_audit_actor(request, AuditActor(user.id, user.username, user.role))

    revoked = 0
    for session_id in sessions:
        revoked += await revoke_family(db, session_id, "logout")
    await db.commit()
    response.delete_cookie(
        settings.refresh_cookie_name,
        path=settings.refresh_cookie_path,
        secure=settings.refresh_cookie_secure,
        httponly=True,
        samesite="strict",
    )
    set_audit_outcome(request, "signed out" if revoked else "no open session")
    return MessageResponse(message="Signed out.")


@router.get("/me", response_model=ProfileResponse)
async def me(user: AuthenticatedUser) -> ProfileResponse:
    """The signed-in user with their current role."""
    return ProfileResponse(
        id=user.id,
        username=user.username,
        role=user.role,
        session_id=user.session_id,
        access_expires_at_ms=user.token_expires_at_ms,
    )


@router.post("/password", response_model=TokenResponse)
async def change_password(
    request: Request,
    response: Response,
    body: PasswordChangeRequest,
    db: DbSession,
    user: AuthenticatedUser,
) -> TokenResponse:
    """
    Change the signed-in user's password.

    Every session of the user is closed, including this one; the response carries a
    fresh session. Wrong current passwords count toward the sign-in throttle.
    """
    client = _client(request)
    throttle = _throttle(request)
    retry_after = throttle.retry_after_s(user.username, client.ip)
    if retry_after:
        set_audit_outcome(request, "refused: too many failed attempts")
        raise ProblemError(
            429,
            "auth.too_many_attempts",
            "Too many failed attempts. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )
    try:
        account = await change_own_password(
            db, user, body.current_password, body.new_password
        )
    except ProblemError as exc:
        if exc.code == "auth.password_mismatch":
            throttle.record_failure(user.username, client.ip)
        set_audit_outcome(request, f"refused: {exc.code}")
        raise
    throttle.record_success(user.username)
    tokens = await open_session(db, account, client)
    _set_refresh_cookie(response, tokens)
    _signed_in(request, tokens, "password changed, sessions closed")
    return _token_response(tokens)
