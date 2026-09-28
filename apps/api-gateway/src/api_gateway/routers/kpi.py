"""
Key performance indicators over a time range, from the historian.

Every ratio is a ratio of means over the range, not a mean of instantaneous ratios, so
it weights each moment by the energy that flowed then: efficiency is mean electrical
output over mean fuel heat input. Samples are taken once per simulated step, so the
means are over simulated time whatever the simulation speed was. Ratios are null when
the unit did not generate (mean output below 1 MW) or when there is no data.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from api_gateway.auth.rbac import ViewerUser
from api_gateway.limits import MAX_EPOCH_MS
from api_gateway.problems import UPSTREAM_RESPONSES, historian_call
from api_gateway.routers.history import resolve_range
from api_gateway.schemas.ops import KpiResponse
from api_gateway.upstreams import HistorianClient

router = APIRouter(prefix="/api/v1", tags=["kpi"], responses=UPSTREAM_RESPONSES)

MIN_GENERATING_POWER_W = 1.0e6


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


@router.get("/kpi", response_model=KpiResponse)
async def get_kpis(
    _: ViewerUser,
    historian: HistorianClient,
    start_ms: int | None = Query(
        default=None, ge=0, le=MAX_EPOCH_MS, description="[UTC epoch ms]"
    ),
    end_ms: int | None = Query(
        default=None, ge=0, le=MAX_EPOCH_MS, description="[UTC epoch ms]"
    ),
) -> KpiResponse:
    """KPIs of the unit over a range: the last 15 minutes by default, at most 90 days."""
    start, end = resolve_range(start_ms, end_ms)
    async with historian_call():
        source, values = await asyncio.to_thread(
            historian.fetch_kpi_inputs, start_ms=start, end_ms=end
        )

    def mean(field: str) -> float | None:
        return values.get((field, "mean"))

    power = mean("electrical_power_w")
    generating = (
        power if power is not None and power >= MIN_GENERATING_POWER_W else None
    )
    fuel = mean("fuel_heat_input_w")
    heat_to_cycle = mean("heat_to_cycle_w")
    return KpiResponse(
        start_ms=start,
        end_ms=end,
        source=source.name,
        samples=int(values.get(("electrical_power_w", "count"), 0.0)),
        mean_electrical_power_w=power,
        mean_fuel_heat_input_w=fuel,
        net_efficiency=_ratio(generating, fuel),
        boiler_efficiency=_ratio(heat_to_cycle, fuel),
        turbine_heat_rate_j_per_j=_ratio(heat_to_cycle, generating),
        plant_heat_rate_j_per_j=_ratio(fuel, generating),
        co2_intensity_kg_per_j=_ratio(mean("co2_kg_s"), generating),
        mean_nox_ppmv=mean("nox_ppmv"),
        peak_nox_ppmv=values.get(("nox_ppmv", "max")),
        mean_health_pct=mean("overall_health_pct"),
        lowest_health_pct=values.get(("overall_health_pct", "min")),
    )
