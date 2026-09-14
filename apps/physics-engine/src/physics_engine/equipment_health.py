"""
Equipment health and wear tracking for the CogniBoiler digital twin.

Tracks cumulative damage to three major components:

    Turbine rotor:  thermal fatigue from temperature cycling + start-stop cycles.
    Boiler tubes:   creep damage from sustained high temperature and pressure.
    Feed pump:      bearing wear from operating hours and starts.

Damage model: Linear Damage Rule (Palmgren-Miner):
    D_total = Σ (n_i / N_i)
    D = 0.0 → new equipment
    D = 1.0 → design life exhausted (maintenance overdue)
    D > 1.0 → operating on borrowed time (failure risk rising)

Each damage contribution is dimensionless and additive.
The tracker is stateful: it accumulates damage across the full simulation.

Physical basis:
    - Cold start thermal fatigue: ΔT ~ 400°C in turbine rotor → high stress
    - Creep: exponential dependence on temperature above design value
    - Running hours: each hour consumes a tiny fraction of design life

References:
    IEC 60045-1: Steam turbines — performance test codes
    EPRI TR-107396: Life assessment of steam turbines (simplified Miner's rule)
    EN 12952-3: Water-tube boilers — design and calculation (creep rules)
"""

from __future__ import annotations

from dataclasses import dataclass

# ─── Design life parameters ───────────────────────────────────────────────────

TURBINE_DESIGN_HOURS: float = 200_000.0  # h — LP/IP turbine design running life
TURBINE_DESIGN_STARTS: float = 3_000.0  # starts — design start-stop cycles

BOILER_TUBE_DESIGN_HOURS: float = 150_000.0  # h — superheater tube design life
PUMP_DESIGN_HOURS: float = 100_000.0  # h — feed pump bearing design life

# Thermal fatigue damage per startup event (fraction of turbine design cycles)
COLD_START_DAMAGE: float = 1.0 / TURBINE_DESIGN_STARTS  # worst case
WARM_START_DAMAGE: float = 0.2 / TURBINE_DESIGN_STARTS  # metal above 200°C
HOT_START_DAMAGE: float = 0.05 / TURBINE_DESIGN_STARTS  # metal above 450°C

# Load cycle damage per MW-change [fraction of design cycles / MW / h]
LOAD_CYCLE_DAMAGE_RATE: float = 0.001 / TURBINE_DESIGN_STARTS  # per (MW·h)

# Running-hour damage rate for turbine [1/h of design life consumed per hour]
# At 200 000 h design life: 1/200000 = 5×10⁻⁶ per hour
TURBINE_HOUR_DAMAGE_RATE: float = 1.0 / TURBINE_DESIGN_HOURS  # per hour

# Boiler tube creep:
#   At design temperature → nominal rate
#   Above design → accelerated; uses power law: rate ∝ (T/T_design)^n
TUBE_DESIGN_TEMP: float = 823.15  # K — 550°C nominal superheater tube temp
TUBE_CREEP_EXPONENT: float = 5.0  # typical creep exponent for 9Cr-1Mo steel
TUBE_NOMINAL_DAMAGE_RATE: float = 1.0 / (
    BOILER_TUBE_DESIGN_HOURS * 3600.0
)  # per second

# Alarm / warning thresholds
HEALTH_WARNING_THRESHOLD: float = 0.80  # 80% → schedule maintenance window
HEALTH_ALARM_THRESHOLD: float = 0.95  # 95% → maintenance overdue
HEALTH_CRITICAL_THRESHOLD: float = 1.00  # 100% → failure risk; forced outage risk


# ─── Health state (immutable snapshot) ───────────────────────────────────────


@dataclass(frozen=True)
class EquipmentHealth:
    """
    Point-in-time snapshot of equipment health across all tracked components.

    All damage values are in [0, 1+]:
        0.0 = brand-new
        1.0 = design life consumed (maintenance overdue)
        > 1.0 = overdue — failure probability rising

    Attributes:
        turbine_hours:        Cumulative turbine running hours [h].
        turbine_starts:       Total cold + warm + hot starts.
        turbine_damage:       Total Miner damage fraction for turbine [-].
        boiler_tube_hours:    Cumulative boiler tube hot hours [h].
        boiler_tube_damage:   Creep damage fraction for boiler tubes [-].
        pump_hours:           Feed pump operating hours [h].
        maintenance_alarm:    True if any component ≥ WARNING_THRESHOLD.
        maintenance_critical: True if any component ≥ CRITICAL_THRESHOLD.
    """

    turbine_hours: float = 0.0
    turbine_starts: float = 0.0
    turbine_damage: float = 0.0
    boiler_tube_hours: float = 0.0
    boiler_tube_damage: float = 0.0
    pump_hours: float = 0.0
    maintenance_alarm: bool = False
    maintenance_critical: bool = False

    @property
    def turbine_health_pct(self) -> float:
        """Turbine remaining life [%]. Clamped to [0, 100]."""
        return max(0.0, min(100.0, (1.0 - self.turbine_damage) * 100.0))

    @property
    def boiler_tube_health_pct(self) -> float:
        """Boiler tube remaining life [%]. Clamped to [0, 100]."""
        return max(0.0, min(100.0, (1.0 - self.boiler_tube_damage) * 100.0))

    @property
    def overall_health_pct(self) -> float:
        """Overall plant health = worst component [%]."""
        return min(self.turbine_health_pct, self.boiler_tube_health_pct)

    @property
    def worst_damage(self) -> float:
        """Highest damage fraction across all components."""
        return max(self.turbine_damage, self.boiler_tube_damage)


# ─── Health tracker (stateful) ────────────────────────────────────────────────


class HealthTracker:
    """
    Stateful accumulator that tracks equipment damage over simulation time.

    Call update() at every simulation timestep with current process conditions.
    The tracker accumulates damage using Miner's linear damage rule.

    Usage:
        tracker = HealthTracker()

        # In the simulation loop:
        health = tracker.update(
            dt=1.0,              # s  — timestep
            power_mw=250.0,      # MW — current electrical output
            tube_temp_k=820.0,   # K  — superheater tube temperature
            is_running=True,
            startup_type="none", # "cold" | "warm" | "hot" | "none"
        )
        if health.maintenance_alarm:
            print(f"Maintenance due — turbine: {health.turbine_health_pct:.0f}%")

    Note: tube_temp_k is best approximated as the drum water_temp (which
    represents steam temperature entering the superheater). For a more
    accurate model, use superheater outlet temperature if available.
    """

    def __init__(
        self,
        initial_turbine_hours: float = 0.0,
        initial_turbine_starts: float = 0.0,
        initial_turbine_damage: float = 0.0,
        initial_tube_hours: float = 0.0,
        initial_tube_damage: float = 0.0,
        initial_pump_hours: float = 0.0,
    ) -> None:
        """
        Args:
            initial_turbine_hours:  Pre-existing turbine running hours.
                                    Used for mid-life simulation starts.
            initial_turbine_starts: Pre-existing turbine start count.
            initial_turbine_damage: Pre-existing turbine Miner damage (0 = new).
            initial_tube_hours:     Pre-existing boiler tube hot hours.
                                    Independent from turbine hours — a boiler
                                    can have more hot hours than a turbine if the
                                    turbine was offline during steam-raising.
            initial_tube_damage:    Pre-existing tube creep damage fraction.
            initial_pump_hours:     Pre-existing feed pump operating hours.
                                    Pump and turbine hours diverge when the pump
                                    runs during startup before the turbine comes on.
        """
        self._turbine_hours = initial_turbine_hours
        self._turbine_starts = initial_turbine_starts
        self._turbine_damage = initial_turbine_damage

        # BUG FIX: tube and pump hours have their own parameters and counters.
        # Original code incorrectly set all three to `initial_turbine_hours`,
        # making mid-life restarts produce wrong boiler and pump age estimates.
        self._tube_hours = initial_tube_hours
        self._tube_damage = initial_tube_damage
        self._pump_hours = initial_pump_hours

        self._last_power_mw = 0.0

    def update(
        self,
        dt: float,
        power_mw: float,
        tube_temp_k: float,
        is_running: bool,
        startup_type: str = "none",
    ) -> EquipmentHealth:
        """
        Advance health state by one simulation timestep.

        Args:
            dt:            Timestep [s].
            power_mw:      Current electrical output [MW].
            tube_temp_k:   Boiler tube / superheater temperature [K].
                           Proxy: use water_temp from BoilerState.
            is_running:    True if turbine is on-line and generating.
            startup_type:  "cold" | "warm" | "hot" | "none".
                           "cold" = turbine metal below 200°C.
                           "warm" = metal 200–450°C.
                           "hot"  = metal above 450°C (quick restart).

        Returns:
            EquipmentHealth snapshot (frozen dataclass).
        """
        dt_hours = dt / 3600.0

        if is_running:
            # ── Running-hour damage (turbine) ────────────────────────────────
            self._turbine_hours += dt_hours
            self._turbine_damage += TURBINE_HOUR_DAMAGE_RATE * dt_hours

            # ── Load cycling damage (turbine) ────────────────────────────────
            # Damage proportional to MW swing per unit time.
            delta_mw = abs(power_mw - self._last_power_mw)
            self._turbine_damage += delta_mw * LOAD_CYCLE_DAMAGE_RATE * dt_hours

        # ── Boiler tube creep damage (accumulates whenever boiler is hot) ────
        # The boiler tubes accumulate creep damage whenever they are at
        # temperature — regardless of whether the turbine is running.
        # This correctly handles startup phases where the boiler fires but
        # the turbine has not yet been admitted to steam.
        if tube_temp_k > 0.0:
            self._tube_hours += dt_hours
            t_ratio = tube_temp_k / TUBE_DESIGN_TEMP
            creep_rate = TUBE_NOMINAL_DAMAGE_RATE * (t_ratio**TUBE_CREEP_EXPONENT)
            self._tube_damage += creep_rate * dt

        # ── Feed pump (runs whenever feedwater flow is active) ───────────────
        # Pump operates during startup and whenever the boiler is running,
        # so it is always incremented together with tube hours.
        if tube_temp_k > 373.15:  # proxy: pump runs when water is hot
            self._pump_hours += dt_hours

        # ── Startup events (thermal fatigue spikes) ──────────────────────────
        if startup_type == "cold":
            self._turbine_damage += COLD_START_DAMAGE
            self._turbine_starts += 1.0
        elif startup_type == "warm":
            self._turbine_damage += WARM_START_DAMAGE
            self._turbine_starts += 1.0
        elif startup_type == "hot":
            self._turbine_damage += HOT_START_DAMAGE
            self._turbine_starts += 1.0

        self._last_power_mw = power_mw

        # ── Determine alarm state ─────────────────────────────────────────────
        worst = max(self._turbine_damage, self._tube_damage)
        alarm = worst >= HEALTH_WARNING_THRESHOLD
        critical = worst >= HEALTH_CRITICAL_THRESHOLD

        return EquipmentHealth(
            turbine_hours=self._turbine_hours,
            turbine_starts=self._turbine_starts,
            turbine_damage=self._turbine_damage,
            boiler_tube_hours=self._tube_hours,
            boiler_tube_damage=self._tube_damage,
            pump_hours=self._pump_hours,
            maintenance_alarm=alarm,
            maintenance_critical=critical,
        )

    # ── Inspection helpers ────────────────────────────────────────────────────

    @property
    def current_health(self) -> EquipmentHealth:
        """Read current health snapshot without advancing time."""
        worst = max(self._turbine_damage, self._tube_damage)
        alarm = worst >= HEALTH_WARNING_THRESHOLD
        critical = worst >= HEALTH_CRITICAL_THRESHOLD
        return EquipmentHealth(
            turbine_hours=self._turbine_hours,
            turbine_starts=self._turbine_starts,
            turbine_damage=self._turbine_damage,
            boiler_tube_hours=self._tube_hours,
            boiler_tube_damage=self._tube_damage,
            pump_hours=self._pump_hours,
            maintenance_alarm=alarm,
            maintenance_critical=critical,
        )

    def reset(self) -> None:
        """
        Reset tracker to zero (new equipment state).

        Use when restarting a simulation from scratch, not for mid-life starts
        (use the constructor parameters instead).
        """
        self._turbine_hours = 0.0
        self._turbine_starts = 0.0
        self._turbine_damage = 0.0
        self._tube_hours = 0.0
        self._tube_damage = 0.0
        self._pump_hours = 0.0
        self._last_power_mw = 0.0
