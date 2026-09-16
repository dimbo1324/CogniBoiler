"""
Pydantic schemas for authentication endpoints.
These schemas define the exact shape of JSON request bodies and
response payloads for /auth/login, /auth/refresh, /auth/logout, /auth/me and
/auth/password. FastAPI uses them for validation and OpenAPI documentation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 128


class LoginRequest(BaseModel):
    """
    Request body for POST /auth/login.
    Both fields are required. Passwords are never logged or stored
    in plain text — only the Argon2id hash is persisted.
    """

    username: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Username registered in the system.",
        examples=["operator1"],
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Plain-text password (transmitted over TLS only).",
        examples=["s3cur3P@ssword"],
    )


class TokenResponse(BaseModel):
    """
    Response body for POST /auth/login, POST /auth/refresh and POST /auth/password.
    access_token  — short-lived (15 min), used in the Authorization header; keep it in
                    memory.
    refresh_token — valid until the session expires (7 days after sign-in), exchanged
                    once at /auth/refresh. Browsers should rely on the httpOnly cookie
                    set with the same value and never store this field.
    """

    access_token: str = Field(..., description="RS256-signed JWT access token.")
    refresh_token: str = Field(..., description="RS256-signed JWT refresh token.")
    token_type: str = Field(default="bearer", description="Token scheme.")
    expires_in: int = Field(..., description="Seconds until the access token expires.")
    access_expires_at_ms: int = Field(..., description="Access token expiry [UTC ms].")
    session_expires_at_ms: int = Field(
        ..., description="When the session ends and a new sign-in is needed [UTC ms]."
    )
    username: str
    role: str


class RefreshRequest(BaseModel):
    """Request body for POST /auth/refresh; omit it to use the refresh cookie."""

    refresh_token: str | None = Field(
        default=None, max_length=4096, description="Refresh token to exchange."
    )


class LogoutRequest(BaseModel):
    """
    Request body for POST /auth/logout; omit it to use the refresh cookie or the bearer
    token. The whole session of the token is closed.
    """

    refresh_token: str | None = Field(
        default=None, max_length=4096, description="Refresh token of the session."
    )


class PasswordChangeRequest(BaseModel):
    """Request body for POST /auth/password."""

    current_password: str = Field(..., min_length=1, max_length=PASSWORD_MAX_LENGTH)
    new_password: str = Field(
        ..., min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH
    )


class ProfileResponse(BaseModel):
    """The signed-in user, as the gateway sees them now."""

    id: int
    username: str
    role: str
    session_id: str
    access_expires_at_ms: int


class MessageResponse(BaseModel):
    """Generic single-message response used across multiple endpoints."""

    message: str = Field(..., description="Human-readable status message.")
