"""
Pydantic schemas for authentication endpoints.
These schemas define the exact shape of JSON request bodies and
response payloads for /auth/login, /auth/refresh, and /auth/logout.
FastAPI uses them for automatic validation and OpenAPI documentation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
    Response body for POST /auth/login and POST /auth/refresh.
    access_token  — short-lived (15 min), used in Authorization header.
    refresh_token — long-lived (7 days), used only at /auth/refresh.
    token_type    — always "bearer" per OAuth 2.0 convention.
    """

    access_token: str = Field(..., description="RS256-signed JWT access token.")
    refresh_token: str = Field(..., description="RS256-signed JWT refresh token.")
    token_type: str = Field(default="bearer", description="Token scheme.")


class RefreshRequest(BaseModel):
    """Request body for POST /auth/refresh."""

    refresh_token: str = Field(..., description="Valid refresh token.")


class LogoutRequest(BaseModel):
    """
    Request body for POST /auth/logout.
    The refresh token is added to the server-side blacklist so it
    cannot be used again even before its natural expiry.
    """

    refresh_token: str = Field(..., description="Refresh token to invalidate.")


class MessageResponse(BaseModel):
    """Generic single-message response used across multiple endpoints."""

    message: str = Field(..., description="Human-readable status message.")
