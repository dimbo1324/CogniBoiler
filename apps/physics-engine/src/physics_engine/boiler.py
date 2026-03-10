"""
Thermodynamic ODE model of a steam boiler.

Integrates all sub-models into a unified simulation:
    - CombustionModel    : heat release from fuel burning
    - SuperheaterModel   : saturated → superheated steam
    - EconomizerModel    : feedwater preheating from flue gas
    - steam_tables       : IAPWS-IF97 water/steam properties
    - ValveState         : actuator dynamics for all control valves

State vector y = [U, P, h, T_gas, T_water] — 5 dimensions.

ODE system:
    dU/dt      — internal energy balance
    dP/dt      — steam pressure dynamics via IAPWS-IF97
    dh/dt      — water level mass balance
    dT_gas/dt  — furnace flue gas temperature
    dT_w/dt    — bulk water temperature (variable-mass corrected)

Flue gas path through heat exchangers (temperature decreasing):
    Furnace → Superheater → Boiler drum tubes → Economizer → Stack
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from physics_engine import steam_tables
from physics_engine.combustion import CombustionModel
from physics_engine.constants import (
    PRESSURE_MAX,
    PRESSURE_MIN,
    TEMP_STEAM_MAX,
)
from physics_engine.heat_exchanger import EconomizerModel, SuperheaterModel
from physics_engine.models import BoilerParameters, BoilerState, ControlInputs

# ─── ODE solver configuration ─────────────────────────────────────────────────

ODE_METHOD: str = "Radau"
ODE_RTOL: float = 1e-4
ODE_ATOL: float = 1e-6
ODE_MAX_STEP: float = 5.0  # seconds

# Flue gas thermal mass in furnace [kg] and Cp [J/(kg·K)]
FURNACE_GAS_MASS: float = 5000.0
FURNACE_GAS_CP: float = 1100.0

# Pressure response time constant [s].
# How fast drum pressure tracks the saturation pressure derived from T_water.
# 30 s is physically consistent with typical boiler pressure response rates
# (~0.5–2 bar/min at part load; up to ~5 bar/min under extreme transients).
# A larger TAU (e.g. 300 s) was tested but caused two problems:
#   1. With full fuel + closed steam valve, pressure rose too slowly to reach
#      PRESSURE_MAX before T_water escaped the liquid region (> 647 K).
#   2. With max steam and no feedwater, pressure lagged so far below saturation
#      that it occasionally hit PRESSURE_MIN before the drum went dry.
TAU_PRESSURE: float = 30.0

# Reference temperature for the internal-energy state variable [K].
# U ≡ M · cp · (T − T_REF), consistent with IAPWS-IF97 whose enthalpy
# reference is 273.15 K (0 °C).
T_REF: float = 273.15

# Critical temperature of water [K] (IAPWS-IF97).
# water_temp is clamped to this value inside _derivatives so that the ODE
# never wanders into the supercritical region where IAPWS-IF97 functions
# behave non-monotonically and _event_temp_high would fire prematurely.
T_CRITICAL: float = 647.0

# Maximum cp used in the drum energy equations [J/(kg·K)].
# Near the saturation curve at high pressure the IAPWS-IF97 cp diverges toward
# the pseudo-critical peak (8–15 kJ/(kg·K)).  In a simplified single-phase drum
# model this spike over-amplifies the cold-dilution term.  5 000 J/(kg·K)
# suppresses the artefact while staying accurate across the rest of the range.
CP_WATER_MAX: float = 5_000.0


class BoilerModel:
    """
    Full thermodynamic ODE model of a steam boiler.

    Integrates combustion, heat exchangers, drum dynamics, and
    actuator dynamics into a single simulate() call.

    Usage:
        params = BoilerParameters()
        model  = BoilerModel(params)
        state0 = params.nominal_initial_state()
        ctrl   = ControlInputs(
            fuel_valve_command=0.7,
            feedwater_valve_command=0.5,
            steam_valve_command=0.6,
        )
        result = model.simulate(state0, ctrl, t_span=(0, 600), dt=1.0)
    """

    def __init__(
        self,
        params: BoilerParameters,
        excess_air_ratio: float = 1.1,
        air_preheat_temp: float | None = None,
    ) -> None:
        self.params = params
        self.combustion = CombustionModel(
            max_fuel_flow=params.max_fuel_flow,
            nominal_excess_air=excess_air_ratio,
            air_preheat_temp=air_preheat_temp or params.ambient_temp,
        )
        self.superheater = SuperheaterModel()
        self.economizer = EconomizerModel()

    # ─── Flow calculations ────────────────────────────────────────────────────

    def _steam_flow(self, pressure_pa: float, valve_position: float) -> float:
        """
        Steam mass flow through outlet valve [kg/s].

        Square-root pressure drop model:
            m = Cv · position · sqrt(P − P_downstream)
        Downstream pressure assumed 10 bar (turbine inlet).
        """
        p_downstream = 10.0e5  # Pa
        dp = max(pressure_pa - p_downstream, 0.0)
        flow = self.params.steam_valve_coeff * valve_position * np.sqrt(dp / 1.0e5)
        return float(np.clip(flow, 0.0, self.params.max_steam_flow))

    def _feedwater_flow(self, valve_position: float) -> float:
        """Feedwater mass flow into drum [kg/s]."""
        max_feedwater = 300.0  # kg/s
        return valve_position * max_feedwater

    # ─── Heat transfer ────────────────────────────────────────────────────────

    def _heat_to_water(self, flue_gas_temp: float, water_temp: float) -> float:
        """Heat transferred from flue gas to drum water [W]."""
        delta_t = max(flue_gas_temp - water_temp, 0.0)
        return self.params.heat_transfer_coeff * delta_t

    def _heat_loss(self, water_temp: float) -> float:
        """Heat lost through boiler walls to ambient [W]."""
        delta_t = max(water_temp - self.params.ambient_temp, 0.0)
        return self.params.heat_loss_coeff * delta_t

    def _heat_carried_by_steam(
        self,
        steam_flow: float,
        pressure_pa: float,
    ) -> float:
        """
        Enthalpy carried out of the drum by steam [W].

        Steam always leaves the drum as saturated vapour at drum pressure —
        regardless of where T_water sits relative to the saturation curve.
        Using saturated_vapor_enthalpy(P) instead of steam_enthalpy(T, P)
        prevents a numerical collapse: if T_water drifts even 0.1 K below
        T_sat (which can happen with the pressure lag model), steam_enthalpy
        falls back to compressed-liquid values (~1 440 kJ/kg vs ~2 785 kJ/kg),
        and the correction term in dT_water_dt explodes by 10–20 K/s.

        Q_steam_out = m_steam · h_g(P)
        """
        if steam_flow <= 0.0:
            return 0.0
        h_sat_vapor = steam_tables.saturated_vapor_enthalpy(pressure_pa)
        return steam_flow * h_sat_vapor

    # ─── Event functions for solve_ivp ───────────────────────────────────────

    @staticmethod
    def _event_pressure_high(t: float, y: list[float], *args: object) -> float:
        """Trigger when pressure rises above PRESSURE_MAX."""
        return y[1] - PRESSURE_MAX

    @staticmethod
    def _event_pressure_low(t: float, y: list[float], *args: object) -> float:
        """Trigger when pressure falls below PRESSURE_MIN."""
        return y[1] - PRESSURE_MIN

    @staticmethod
    def _event_water_empty(t: float, y: list[float], *args: object) -> float:
        """Trigger when water level falls below 5 cm safety margin."""
        return y[2] - 0.05

    @staticmethod
    def _event_water_overflow(t: float, y: list[float], *args: object) -> float:
        """Trigger when water level rises above drum limit."""
        from physics_engine.constants import DRUM_HEIGHT

        return y[2] - (DRUM_HEIGHT - 0.1)

    @staticmethod
    def _event_temp_high(t: float, y: list[float], *args: object) -> float:
        """
        Trigger when drum water temperature rises above TEMP_STEAM_MAX.

        y[4] is the drum water temperature.  Because water_temp is clamped at
        T_CRITICAL = 647 K inside _derivatives, this event only fires when the
        physical limit is approached — well above the normal operating range
        (~610 K at 140 bar) and below TEMP_STEAM_MAX (838 K).
        """
        return y[4] - TEMP_STEAM_MAX

    # ─── ODE right-hand side ─────────────────────────────────────────────────

    def _derivatives(
        self,
        t: float,  # noqa: ARG002
        y: list[float],
        controls: ControlInputs,
    ) -> list[float]:
        """
        Compute dy/dt for the ODE solver.

        State vector y = [U, P, h, T_gas, T_water]
        Returns [dU/dt, dP/dt, dh/dt, dT_gas/dt, dT_water/dt]
        """
        state = BoilerState.from_vector(y)

        # ── Clamp state to physical bounds ────────────────────────────────────
        water_level = max(state.water_level, 0.01)
        # Lower bound: keep above boiling point at 1 bar.
        # Upper bound: clamp at critical temperature of water (647 K).
        # Without the upper clamp the ODE can push water_temp into the
        # supercritical region where IAPWS-IF97 is non-monotonic and
        # _event_temp_high fires prematurely (before PRESSURE HIGH or DRUM DRY).
        water_temp = max(state.water_temp, 373.15)
        water_temp = min(water_temp, T_CRITICAL)
        pressure_pa = max(state.pressure, 1.0e5)

        # ── Water mass via IAPWS-IF97 density ────────────────────────────────
        rho_water = steam_tables.water_density(water_temp, pressure_pa)
        water_mass = rho_water * self.params.drum_cross_section * water_level
        water_mass = max(water_mass, 1.0)

        # ── Cp of water — capped to suppress the pseudo-critical divergence ──
        cp_water = min(
            steam_tables.water_specific_heat(water_temp, pressure_pa),
            CP_WATER_MAX,
        )

        # ── Valve positions ───────────────────────────────────────────────────
        fv = controls.fuel_valve.position
        wv = controls.feedwater_valve.position
        sv = controls.steam_valve.position

        # ── Combustion ────────────────────────────────────────────────────────
        comb = self.combustion.calculate(fuel_valve=fv)

        # ── Superheater (hottest flue gas first) ──────────────────────────────
        m_steam = self._steam_flow(pressure_pa, sv)
        sh = self.superheater.calculate(
            pressure_pa=pressure_pa,
            steam_flow=m_steam,
            flue_gas_temp_in=state.flue_gas_temp,
            flue_gas_flow=comb.flue_gas_flow,
        )

        # ── Economizer (flue gas already cooled by superheater) ───────────────
        m_feed_raw = self._feedwater_flow(wv)
        eco = self.economizer.calculate(
            feedwater_flow=m_feed_raw,
            feedwater_temp_in=self.params.feedwater_temp,
            pressure_pa=pressure_pa,
            flue_gas_temp_in=sh.flue_gas_temp_out,
            flue_gas_flow=comb.flue_gas_flow,
        )
        feedwater_temp_actual = eco.water_temp_out

        # ── Heat flows ────────────────────────────────────────────────────────
        q_gas_to_water = self._heat_to_water(state.flue_gas_temp, water_temp)
        q_loss = self._heat_loss(water_temp)

        # Steam leaves as saturated vapour — enthalpy depends only on pressure.
        q_steam_out = self._heat_carried_by_steam(m_steam, pressure_pa)

        # Feedwater energy referenced to T_REF = 273.15 K (IAPWS-IF97 zero).
        q_feedwater_in = m_feed_raw * cp_water * (feedwater_temp_actual - T_REF)

        # ── ODE 1: dU/dt ──────────────────────────────────────────────────────
        du_dt = q_gas_to_water + q_feedwater_in - q_steam_out - q_loss

        # ── ODE 2: dP/dt — pressure tracks saturation pressure ───────────────
        p_sat = steam_tables.saturation_pressure(water_temp)
        dp_dt = (p_sat - pressure_pa) / TAU_PRESSURE

        # ── ODE 3: dh/dt — liquid mass balance ───────────────────────────────
        # d(ρ·A·h)/dt = m_feed − m_steam
        # → dh/dt = (m_feed − m_steam) / (ρ_water · A)
        #
        # FIX: previous formula subtracted m_steam/ρ_steam (a volumetric flow
        # [m³/s]) from m_feed (a mass flow [kg/s]) — dimensionally inconsistent.
        # That made the level appear to rise even with zero feedwater.
        dh_dt = (m_feed_raw - m_steam) / (rho_water * self.params.drum_cross_section)

        # ── ODE 4: dT_gas/dt ──────────────────────────────────────────────────
        q_sh_absorbed = sh.heat_transferred
        dt_gas_dt = (comb.heat_available - q_gas_to_water - q_sh_absorbed) / (
            FURNACE_GAS_MASS * FURNACE_GAS_CP
        )

        # ── ODE 5: dT_water/dt — variable-mass corrected ──────────────────────
        # U = M · cp · (T − T_REF)  →  product rule:
        #   dU/dt = dM/dt · cp · (T − T_REF) + M · cp · dT/dt
        # Solving for dT/dt:
        #   dT/dt = [dU/dt − dM/dt · cp · (T − T_REF)] / (M · cp)
        #
        # FIX: previous code used bare water_temp instead of (water_temp − T_REF),
        # injecting a spurious ~200 MW cooling term at nominal conditions.
        dm_dt = m_feed_raw - m_steam
        dt_water_dt = (du_dt - dm_dt * cp_water * (water_temp - T_REF)) / (
            water_mass * cp_water
        )

        return [du_dt, dp_dt, dh_dt, dt_gas_dt, dt_water_dt]

    # ─── Public simulation interface ─────────────────────────────────────────

    def simulate(
        self,
        initial_state: BoilerState,
        controls: ControlInputs,
        t_span: tuple[float, float],
        dt: float = 1.0,
    ) -> OdeResult:
        """
        Integrate the ODE system over a time span with event detection.

        Args:
            initial_state: Starting state of the boiler.
            controls: Control inputs (valve commands + actuator states).
            t_span: (t_start, t_end) in seconds.
            dt: Output time step in seconds.

        Returns:
            scipy OdeResult. status:
                0 = reached t_end normally
               -1 = integration step failed
                1 = termination event triggered (alarm condition)
        """
        controls.update_valves(dt)

        t_eval = np.arange(t_span[0], t_span[1], dt)
        y0 = initial_state.to_vector()

        self._event_pressure_high.terminal = True  # type: ignore[attr-defined]
        self._event_pressure_high.direction = 1.0  # type: ignore[attr-defined]

        self._event_pressure_low.terminal = True  # type: ignore[attr-defined]
        self._event_pressure_low.direction = -1.0  # type: ignore[attr-defined]

        self._event_water_empty.terminal = True  # type: ignore[attr-defined]
        self._event_water_empty.direction = -1.0  # type: ignore[attr-defined]

        self._event_water_overflow.terminal = True  # type: ignore[attr-defined]
        self._event_water_overflow.direction = 1.0  # type: ignore[attr-defined]

        self._event_temp_high.terminal = True  # type: ignore[attr-defined]
        self._event_temp_high.direction = 1.0  # type: ignore[attr-defined]

        events = [
            self._event_pressure_high,
            self._event_pressure_low,
            self._event_water_empty,
            self._event_water_overflow,
            self._event_temp_high,
        ]

        result: OdeResult = solve_ivp(
            fun=lambda t, y: self._derivatives(t, y, controls),
            t_span=t_span,
            y0=y0,
            method=ODE_METHOD,
            t_eval=t_eval,
            events=events,
            rtol=ODE_RTOL,
            atol=ODE_ATOL,
            max_step=ODE_MAX_STEP,
        )

        return result

    def get_state_at(self, result: OdeResult, index: int) -> BoilerState:
        """Extract BoilerState from ODE result at a given time index."""
        y = [float(result.y[i, index]) for i in range(5)]
        return BoilerState.from_vector(y)

    def check_result(self, result: OdeResult) -> str:
        """Return human-readable simulation termination reason."""
        if result.status == 0:
            return "Simulation completed normally."
        if result.status == -1:
            return f"Solver failed: {result.message}"
        event_names = [
            "PRESSURE HIGH",
            "PRESSURE LOW",
            "DRUM DRY",
            "DRUM OVERFLOW",
            "STEAM TEMP HIGH",
        ]
        for i, t_event in enumerate(result.t_events):
            if len(t_event) > 0:
                return f"ALARM [{event_names[i]}] at t={t_event[0]:.1f}s"
        return "Terminated by unknown event."
