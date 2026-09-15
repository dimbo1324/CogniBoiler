"""
Fault injection for the plant model.

A fault is a labelled deviation of the plant from its design: it changes the physics
(fouled burners, a steam leak, a failing feedwater pump), freezes an actuator, or corrupts
a measurement. Every active fault is visible in the plant state with its kind, target and
intensity, which is what makes recorded runs usable as labelled data later.

Faults develop over `ramp_s` seconds of simulation time, so a slow degradation and a
sudden failure are both expressible with one specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from physics_engine.sensors import SensorId


class FaultKind(StrEnum):
    """Kinds of faults the plant model can simulate."""

    BURNER_FOULING = "burner_fouling"
    STEAM_LEAK = "steam_leak"
    FEEDWATER_PUMP_FAILURE = "feedwater_pump_failure"
    VALVE_STUCK = "valve_stuck"
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_FAILURE = "sensor_failure"


class ValveId(StrEnum):
    """Actuators a stuck-valve fault can freeze."""

    FUEL = "fuel"
    FEEDWATER = "feedwater"
    STEAM = "steam"
    SPRAY = "spray"


# Severity meaning and admissible range per kind:
#   burner_fouling          — fraction of combustion efficiency lost
#   steam_leak              — leak flow as a fraction of rated steam flow at nominal pressure
#   feedwater_pump_failure  — fraction of pump capacity lost (1.0 = pump tripped)
#   sensor_drift            — drift rate as a fraction of the sensor span per minute
#   valve_stuck, sensor_failure — severity is not used
_SEVERITY_RANGES: dict[FaultKind, tuple[float, float]] = {
    FaultKind.BURNER_FOULING: (0.0, 0.5),
    FaultKind.STEAM_LEAK: (0.0, 0.3),
    FaultKind.FEEDWATER_PUMP_FAILURE: (0.0, 1.0),
    FaultKind.SENSOR_DRIFT: (-0.2, 0.2),
}

MAX_RAMP_S: float = 3600.0


class FaultError(ValueError):
    """A fault specification or request that the plant cannot accept."""


@dataclass(frozen=True)
class FaultSpec:
    """What to break, how badly, and how fast the fault develops."""

    kind: FaultKind
    target: str = ""
    severity: float = 1.0
    ramp_s: float = 0.0

    def validated(self) -> FaultSpec:
        """Return the spec with a normalized target, or raise FaultError."""
        if not 0.0 <= self.ramp_s <= MAX_RAMP_S:
            raise FaultError(f"ramp_s={self.ramp_s:g} outside [0, {MAX_RAMP_S:g}] s")

        target = self.target.strip().lower()
        if self.kind is FaultKind.VALVE_STUCK:
            if target not in {valve.value for valve in ValveId}:
                raise FaultError(
                    f"valve_stuck needs a valve target: {', '.join(ValveId)}"
                )
        elif self.kind in (FaultKind.SENSOR_DRIFT, FaultKind.SENSOR_FAILURE):
            if target not in {sensor.value for sensor in SensorId}:
                raise FaultError(
                    f"{self.kind.value} needs a sensor target: {', '.join(SensorId)}"
                )
        elif target:
            raise FaultError(f"{self.kind.value} does not take a target")

        bounds = _SEVERITY_RANGES.get(self.kind)
        if bounds is not None:
            low, high = bounds
            if not low <= self.severity <= high or self.severity == 0.0:
                raise FaultError(
                    f"{self.kind.value} severity {self.severity:g} outside "
                    f"[{low:g}, {high:g}] or zero"
                )

        return FaultSpec(
            kind=self.kind,
            target=target,
            severity=self.severity,
            ramp_s=self.ramp_s,
        )


@dataclass(frozen=True)
class ActiveFault:
    """A fault in effect since `started_at_s` of simulation time."""

    fault_id: str
    spec: FaultSpec
    started_at_s: float

    def intensity(self, simulation_time_s: float) -> float:
        """How developed the fault is, from 0 at onset to 1 after its ramp."""
        if self.spec.ramp_s <= 0.0:
            return 1.0
        elapsed = max(simulation_time_s - self.started_at_s, 0.0)
        return min(elapsed / self.spec.ramp_s, 1.0)

    @property
    def label(self) -> str:
        """Stable label for data annotation, e.g. `sensor_drift:drum_level`."""
        if self.spec.target:
            return f"{self.spec.kind.value}:{self.spec.target}"
        return self.spec.kind.value


@dataclass(frozen=True)
class PlantDisturbances:
    """Physical effect of all active faults on one integration step."""

    combustion_efficiency_factor: float = 1.0
    steam_leak_fraction: float = 0.0
    feedwater_capacity_factor: float = 1.0


NO_DISTURBANCES = PlantDisturbances()


@dataclass
class FaultRegistry:
    """The set of active faults; one fault per kind and target at a time."""

    _active: dict[str, ActiveFault] = field(default_factory=dict)
    _counter: int = 0

    def inject(self, spec: FaultSpec, simulation_time_s: float) -> ActiveFault:
        """Activate a fault, refusing a duplicate of an already active one."""
        normalized = spec.validated()
        for fault in self._active.values():
            if (fault.spec.kind, fault.spec.target) == (
                normalized.kind,
                normalized.target,
            ):
                raise FaultError(f"{fault.label} is already active as {fault.fault_id}")
        self._counter += 1
        fault = ActiveFault(
            fault_id=f"F{self._counter:04d}",
            spec=normalized,
            started_at_s=simulation_time_s,
        )
        self._active[fault.fault_id] = fault
        return fault

    def clear(self, fault_id: str) -> ActiveFault:
        """Deactivate one fault, raising FaultError when it is not active."""
        fault = self._active.pop(fault_id, None)
        if fault is None:
            raise FaultError(f"fault {fault_id!r} is not active")
        return fault

    def clear_all(self) -> tuple[ActiveFault, ...]:
        """Deactivate every fault and return what was active."""
        cleared = tuple(self._active.values())
        self._active.clear()
        return cleared

    def active(self) -> tuple[ActiveFault, ...]:
        """Active faults in injection order."""
        return tuple(self._active.values())

    def disturbances(self, simulation_time_s: float) -> PlantDisturbances:
        """Combine the physical faults into the disturbances of one step."""
        efficiency = 1.0
        leak = 0.0
        pump = 1.0
        for fault in self._active.values():
            amount = fault.spec.severity * fault.intensity(simulation_time_s)
            match fault.spec.kind:
                case FaultKind.BURNER_FOULING:
                    efficiency *= 1.0 - amount
                case FaultKind.STEAM_LEAK:
                    leak += amount
                case FaultKind.FEEDWATER_PUMP_FAILURE:
                    pump *= 1.0 - amount
                case _:
                    pass
        return PlantDisturbances(
            combustion_efficiency_factor=efficiency,
            steam_leak_fraction=leak,
            feedwater_capacity_factor=max(pump, 0.0),
        )

    def stuck_valves(self, simulation_time_s: float) -> frozenset[ValveId]:
        """Valves frozen by a fault that has fully developed."""
        return frozenset(
            ValveId(fault.spec.target)
            for fault in self._active.values()
            if fault.spec.kind is FaultKind.VALVE_STUCK
            and fault.intensity(simulation_time_s) >= 1.0
        )

    def sensor_faults(self) -> tuple[ActiveFault, ...]:
        """Active drift and failure faults, in injection order."""
        return tuple(
            fault
            for fault in self._active.values()
            if fault.spec.kind in (FaultKind.SENSOR_DRIFT, FaultKind.SENSOR_FAILURE)
        )
