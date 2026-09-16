"""Schemas for user administration."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from api_gateway.schemas.auth import PASSWORD_MAX_LENGTH, PASSWORD_MIN_LENGTH

RoleNameField = Literal["viewer", "operator", "engineer", "admin"]

USERNAME_PATTERN = r"^[A-Za-z0-9._-]{3,64}$"


class UserResponse(BaseModel):
    id: int
    username: str
    role: RoleNameField | None = Field(
        ..., description="None if the account has no role."
    )
    is_active: bool
    created_at_ms: int
    last_login_at_ms: int | None
    open_sessions: int = Field(..., description="Sessions that can still be refreshed.")


class UserPageResponse(BaseModel):
    items: list[UserResponse]
    total: int
    limit: int
    offset: int


class UserCreateRequest(BaseModel):
    username: str = Field(..., pattern=USERNAME_PATTERN, examples=["shift.operator"])
    password: str = Field(
        ..., min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH
    )
    role: RoleNameField


class UserUpdateRequest(BaseModel):
    """Change the role, block or unblock. Either change closes the user's sessions."""

    role: RoleNameField | None = None
    is_active: bool | None = None

    @model_validator(mode="after")
    def _something_to_change(self) -> Self:
        if self.role is None and self.is_active is None:
            raise ValueError("Give a role, is_active, or both.")
        return self


class PasswordResetRequest(BaseModel):
    new_password: str = Field(
        ..., min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH
    )


class SessionsRevokedResponse(BaseModel):
    user_id: int
    revoked_tokens: int
