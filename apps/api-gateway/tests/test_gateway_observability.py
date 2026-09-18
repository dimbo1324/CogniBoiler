"""The gateway's correlation ids and metrics endpoint."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest_asyncio
from api_gateway.main import create_app
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=create_app()), base_url="http://test"
    ) as c:
        yield c


async def test_a_caller_correlation_id_is_adopted_and_returned(
    client: AsyncClient,
) -> None:
    response = await client.get("/health", headers={"X-Correlation-ID": "demo-42"})
    assert response.headers["X-Correlation-ID"] == "demo-42"


async def test_an_unsafe_correlation_id_is_replaced(client: AsyncClient) -> None:
    response = await client.get(
        "/health", headers={"X-Correlation-ID": 'bad "id" with spaces'}
    )
    returned = response.headers["X-Correlation-ID"]
    assert returned != 'bad "id" with spaces'
    assert len(returned) == 32


async def test_every_request_gets_an_id_of_its_own(client: AsyncClient) -> None:
    first = await client.get("/health")
    second = await client.get("/health")
    assert first.headers["X-Correlation-ID"] != second.headers["X-Correlation-ID"]


async def test_metrics_count_requests_by_route_template(client: AsyncClient) -> None:
    await client.get("/health")
    await client.get("/api/v1/alarms/12345")
    body = (await client.get("/metrics")).text
    assert 'http_requests_total{method="GET",route="/health",status="200"}' in body
    assert 'route="/api/v1/alarms/{alarm_id}"' in body
    assert "12345" not in body
    assert "gateway_websocket_clients" in body


async def test_the_metrics_endpoint_is_not_part_of_the_api_contract(
    client: AsyncClient,
) -> None:
    schema = (await client.get("/openapi.json")).json()
    assert "/metrics" not in schema["paths"]
