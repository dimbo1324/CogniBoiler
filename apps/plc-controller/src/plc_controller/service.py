"""
PLC business logic: command validation, setpoint management.

Separates pure logic from gRPC transport so it can be tested
without a running gRPC server.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Valve position limits
VALVE_MIN: float = 0.0
VALVE_MAX: float = 1.0

# Safety hard limits
PRESSURE_SETPOINT_MIN_PA: float = 50.0e5  # 50 bar
PRESSURE_SETPOINT_MAX_PA: float = 160.0e5  # 160 bar
LEVEL_SETPOINT_MIN_M: float = 1.0
LEVEL_SETPOINT_MAX_M: float = 8.0
TEMP_SETPOINT_MIN_K: float = 400.0
TEMP_SETPOINT_MAX_K: float = 900.0


@dataclass
class Setpoints:
    """Current PLC setpoints for all control loops."""

    pressure_pa: float = 140.0e5  # 140 bar nominal
    water_level_m: float = 4.8
    steam_temp_k: float = 811.0  # ~538 °C nominal
    updated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class ValidationResult:
    """Result of a command or setpoint validation check."""

    accepted: bool
    reason: str = ""


class PLCService:
    """
    Virtual PLC business logic.

    Validates incoming commands against safety limits,
    manages setpoints, and tracks command history.
    """

    VERSION = "0.1.0"

    def __init__(self) -> None:
        self._setpoints = Setpoints()
        self._start_time = time.monotonic()
        self._commands_received: int = 0
        self._commands_rejected: int = 0

    # ─── Health ───────────────────────────────────────────────────────────────

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def stats(self) -> dict[str, int]:
        return {
            "commands_received": self._commands_received,
            "commands_rejected": self._commands_rejected,
        }

    # ─── Setpoints ────────────────────────────────────────────────────────────

    def get_setpoints(self) -> Setpoints:
        """Return current setpoints (copy)."""
        return Setpoints(
            pressure_pa=self._setpoints.pressure_pa,
            water_level_m=self._setpoints.water_level_m,
            steam_temp_k=self._setpoints.steam_temp_k,
            updated_at_ms=self._setpoints.updated_at_ms,
        )

    def update_setpoints(
        self,
        pressure_pa: float,
        water_level_m: float,
        steam_temp_k: float,
    ) -> ValidationResult:
        """Validate and apply new setpoints."""
        if not (PRESSURE_SETPOINT_MIN_PA <= pressure_pa <= PRESSURE_SETPOINT_MAX_PA):
            return ValidationResult(
                accepted=False,
                reason=(
                    f"Pressure setpoint {pressure_pa/1e5:.1f} bar "
                    f"outside [{PRESSURE_SETPOINT_MIN_PA/1e5:.0f}, "
                    f"{PRESSURE_SETPOINT_MAX_PA/1e5:.0f}] bar"
                ),
            )
        if not (LEVEL_SETPOINT_MIN_M <= water_level_m <= LEVEL_SETPOINT_MAX_M):
            return ValidationResult(
                accepted=False,
                reason=(
                    f"Level setpoint {water_level_m:.2f} m "
                    f"outside [{LEVEL_SETPOINT_MIN_M}, {LEVEL_SETPOINT_MAX_M}] m"
                ),
            )
        if not (TEMP_SETPOINT_MIN_K <= steam_temp_k <= TEMP_SETPOINT_MAX_K):
            return ValidationResult(
                accepted=False,
                reason=(
                    f"Steam temp setpoint {steam_temp_k:.1f} K "
                    f"outside [{TEMP_SETPOINT_MIN_K}, {TEMP_SETPOINT_MAX_K}] K"
                ),
            )

        self._setpoints = Setpoints(
            pressure_pa=pressure_pa,
            water_level_m=water_level_m,
            steam_temp_k=steam_temp_k,
            updated_at_ms=int(time.time() * 1000),
        )
        return ValidationResult(accepted=True)

    # ─── Commands ─────────────────────────────────────────────────────────────

    def validate_command(
        self,
        fuel_valve: float,
        feedwater_valve: float,
        steam_valve: float,
    ) -> ValidationResult:
        """
        Validate a control command against safety limits.

        All valve positions must be in [0.0, 1.0].
        """
        self._commands_received += 1

        for name, value in [
            ("fuel_valve", fuel_valve),
            ("feedwater_valve", feedwater_valve),
            ("steam_valve", steam_valve),
        ]:
            if not (VALVE_MIN <= value <= VALVE_MAX):
                self._commands_rejected += 1
                return ValidationResult(
                    accepted=False,
                    reason=f"{name}={value:.3f} outside [0.0, 1.0]",
                )

        return ValidationResult(accepted=True)
