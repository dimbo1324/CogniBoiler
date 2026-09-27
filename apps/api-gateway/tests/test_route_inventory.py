"""Every route declares who may call it, and each declaration holds both ways.

The rule "every non-health route declares a minimum role" is enforced here, not by review:
a route added without one fails this file. For each declared role the lesser role is
refused and the declared role gets through, so a check that refused everyone would fail
as surely as a missing one.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from api_gateway.auth.identity import ROLE_HIERARCHY
from api_gateway.auth.rbac import get_current_user, required_role
from api_gateway.main import create_app
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute, iter_route_contexts
from httpx import AsyncClient

MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Sign-in, refresh and sign-out cannot require a session: they are how one is obtained
# or ended. A password change requires one, of any role.
UNGUARDED_MUTATIONS = frozenset(
    {("POST", "/auth/login"), ("POST", "/auth/refresh"), ("POST", "/auth/logout")}
)
ANY_SIGNED_IN_USER = frozenset({("POST", "/auth/password"), ("GET", "/auth/me")})
PUBLIC_READS = frozenset({"/health", "/ready", "/metrics"})

PATH_VALUES = {"{user_id}": "3", "{alarm_id}": "1", "{fault_id}": "feedwater_pump"}


@pytest.fixture
def tokens_by_role(
    viewer_tokens: dict[str, str],
    operator_tokens: dict[str, str],
    engineer_tokens: dict[str, str],
    admin_tokens: dict[str, str],
) -> dict[str, dict[str, str]]:
    return {
        "viewer": viewer_tokens,
        "operator": operator_tokens,
        "engineer": engineer_tokens,
        "admin": admin_tokens,
    }


@dataclass(frozen=True)
class Endpoint:
    method: str
    path: str
    dependant: Dependant
    include_in_schema: bool

    def __str__(self) -> str:
        return f"{self.method} {self.path}"


def walk(dependant: Dependant) -> Iterator[Dependant]:
    for dependency in dependant.dependencies:
        yield dependency
        yield from walk(dependency)


def minimum_role(endpoint: Endpoint) -> str | None:
    roles = [required_role(dep.call) for dep in walk(endpoint.dependant)]
    known = [role for role in roles if role is not None]
    return max(known, key=ROLE_HIERARCHY.index) if known else None


def needs_session(endpoint: Endpoint) -> bool:
    return any(dep.call is get_current_user for dep in walk(endpoint.dependant))


def endpoints() -> list[Endpoint]:
    """Every HTTP endpoint as the application serves it, included routers resolved."""
    found = []
    for context in iter_route_contexts(create_app().routes):
        route = context.original_route
        if not isinstance(route, APIRoute):
            continue
        dependant = getattr(context, "dependant", None) or route.dependant
        for method in sorted(route.methods):
            found.append(
                Endpoint(
                    method,
                    context.path or route.path,
                    dependant,
                    route.include_in_schema,
                )
            )
    return found


def concrete(path: str) -> str:
    for placeholder, value in PATH_VALUES.items():
        path = path.replace(placeholder, value)
    return path


GUARDED = [
    pytest.param(endpoint.method, endpoint.path, role, id=str(endpoint))
    for endpoint in endpoints()
    if (role := minimum_role(endpoint)) is not None
]


class TestInventory:
    def test_the_inventory_is_not_empty(self) -> None:
        assert len(GUARDED) >= 30

    @pytest.mark.parametrize(
        "endpoint",
        [
            pytest.param(endpoint, id=str(endpoint))
            for endpoint in endpoints()
            if endpoint.method in MUTATING
        ],
    )
    def test_every_mutating_route_declares_a_role(self, endpoint: Endpoint) -> None:
        key = (endpoint.method, endpoint.path)
        if key in UNGUARDED_MUTATIONS:
            assert minimum_role(endpoint) is None
            return
        if key in ANY_SIGNED_IN_USER:
            assert needs_session(endpoint)
            return
        assert minimum_role(endpoint) is not None, f"{endpoint} has no role"

    @pytest.mark.parametrize(
        "endpoint",
        [
            pytest.param(endpoint, id=str(endpoint))
            for endpoint in endpoints()
            if endpoint.method not in MUTATING
        ],
    )
    def test_every_read_declares_a_role_unless_it_is_public(
        self, endpoint: Endpoint
    ) -> None:
        if endpoint.path in PUBLIC_READS or not endpoint.include_in_schema:
            return
        if (endpoint.method, endpoint.path) in ANY_SIGNED_IN_USER:
            assert needs_session(endpoint)
            return
        assert minimum_role(endpoint) is not None, f"{endpoint} has no role"

    def test_the_allow_lists_name_routes_that_exist(self) -> None:
        present = {(endpoint.method, endpoint.path) for endpoint in endpoints()}
        assert UNGUARDED_MUTATIONS <= present
        assert ANY_SIGNED_IN_USER <= present
        assert {path for _, path in present} >= PUBLIC_READS

    def test_safety_actions_need_an_engineer(self) -> None:
        roles = {
            (endpoint.method, endpoint.path): minimum_role(endpoint)
            for endpoint in endpoints()
        }
        assert roles[("POST", "/api/v1/commands/reset")] == "engineer"
        assert roles[("POST", "/api/v1/commands/setpoint")] == "engineer"

    def test_accounts_and_the_audit_log_need_an_admin(self) -> None:
        for endpoint in endpoints():
            if endpoint.path.startswith(("/api/v1/users", "/api/v1/audit")):
                assert minimum_role(endpoint) == "admin", str(endpoint)


class TestBothOutcomes:
    """No body is sent: the role check runs before validation, so a refusal is 401 or
    403, while a caller who passes it gets a validation error or an answer instead."""

    @pytest.mark.parametrize(("method", "path", "role"), GUARDED)
    async def test_the_role_below_is_refused(
        self,
        tokens_by_role: dict[str, dict[str, str]],
        client: AsyncClient,
        method: str,
        path: str,
        role: str,
    ) -> None:
        level = ROLE_HIERARCHY.index(role)  # type: ignore[arg-type]
        if level == 0:
            response = await client.request(method, concrete(path))
            assert response.status_code == 401
            assert response.json()["code"] == "auth.token_missing"
            return
        lesser = tokens_by_role[ROLE_HIERARCHY[level - 1]]
        response = await client.request(
            method,
            concrete(path),
            headers={"Authorization": f"Bearer {lesser['access']}"},
        )
        assert response.status_code == 403
        assert response.json()["code"] == "auth.forbidden"
        assert response.json()["required_role"] == role

    @pytest.mark.parametrize(("method", "path", "role"), GUARDED)
    async def test_the_declared_role_gets_through(
        self,
        tokens_by_role: dict[str, dict[str, str]],
        client: AsyncClient,
        method: str,
        path: str,
        role: str,
    ) -> None:
        tokens = tokens_by_role[role]
        response = await client.request(
            method,
            concrete(path),
            headers={"Authorization": f"Bearer {tokens['access']}"},
        )
        assert response.status_code not in (401, 403), response.text
