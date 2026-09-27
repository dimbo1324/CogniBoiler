"""What the operator and the engineer ask the unit for: load demand and setpoints.

Each change is validated, attributed and published as a PLC event. The controller
never takes a target as a step: its working setpoints ramp toward it (`control.py`).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from cogniboiler_runtime import now_ms

from plc_controller.commands import (
    Setpoints,
    ValidationResult,
    check_load_demand,
    check_operator,
    check_setpoints,
    refuse,
)
from plc_controller.control import ControlTargets
from plc_controller.events import PlcEvent, PlcEventKind
from plc_controller.measurements import ProcessMeasurements


class OperatorTargets:
    """The load demand and process setpoints the unit is asked to hold."""

    def __init__(self, publish: Callable[[PlcEvent], None]) -> None:
        self._publish = publish
        self._setpoints = Setpoints()
        self._load_demand_w: float | None = None

    @property
    def setpoints(self) -> Setpoints:
        return self._setpoints.copy()

    @property
    def load_demand_w(self) -> float | None:
        """The load target; None until the PLC has seen the plant."""
        return self._load_demand_w

    def update_setpoints(
        self,
        pressure_pa: float,
        water_level_m: float,
        steam_temp_k: float,
        operator_id: str,
    ) -> ValidationResult:
        """Validate and store new setpoints."""
        operator = operator_id.strip()
        refusal = check_setpoints(pressure_pa, water_level_m, steam_temp_k)
        if refusal.accepted:
            refusal = check_operator(operator)
        if not refusal.accepted:
            return refuse(refusal.reason)
        self._setpoints = Setpoints(
            pressure_pa=pressure_pa,
            water_level_m=water_level_m,
            steam_temp_k=steam_temp_k,
            updated_at_ms=now_ms(),
        )
        self._publish(
            PlcEvent(
                PlcEventKind.SETPOINTS_CHANGED,
                operator,
                {
                    "pressure_pa": pressure_pa,
                    "water_level_m": water_level_m,
                    "steam_temp_k": steam_temp_k,
                },
            )
        )
        return ValidationResult(accepted=True)

    def set_load_demand(self, load_w: float, operator_id: str) -> ValidationResult:
        """Validate and store a new electrical load target."""
        operator = operator_id.strip()
        refusal = check_load_demand(load_w)
        if refusal.accepted:
            refusal = check_operator(operator)
        if not refusal.accepted:
            return refuse(refusal.reason)
        previous = self._load_demand_w
        self._load_demand_w = load_w
        self._publish(
            PlcEvent(
                PlcEventKind.LOAD_DEMAND_CHANGED,
                operator,
                {"load_w": load_w, "previous_load_w": previous},
            )
        )
        return ValidationResult(accepted=True)

    def seed_load_demand(self, power_w: float, *, keep_existing: bool) -> None:
        """Start the load demand at the plant's output, so a new run holds its load.

        A reading that is not a number never becomes the demand.
        """
        if keep_existing and self._load_demand_w is not None:
            return
        if math.isfinite(power_w):
            self._load_demand_w = power_w

    def for_scan(self, m: ProcessMeasurements) -> ControlTargets:
        setpoints = self._setpoints
        return ControlTargets(
            load_w=(
                self._load_demand_w
                if self._load_demand_w is not None
                else m.electrical_power_w
            ),
            pressure_pa=setpoints.pressure_pa,
            water_level_m=setpoints.water_level_m,
            steam_temp_k=setpoints.steam_temp_k,
        )
