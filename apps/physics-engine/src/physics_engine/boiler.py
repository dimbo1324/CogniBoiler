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
    dh/dt      — water level mass balance with real steam density
    dT_gas/dt  — furnace flue gas temperature
    dT_w/dt    — bulk water temperature (variable-mass corrected)

Flue gas path through heat exchangers (temperature decreasing):
    Furnace → Superheater → Boiler drum tubes → Economizer → Stack
"""

import numpy as np
from scipy.integrate import OdeResult, solve_ivp

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

# Use Radau (implicit, stiff solver) instead of RK45.
# Our system is stiff: pressure responds in ~seconds,
# water temperature changes over ~minutes.
ODE_METHOD: str = "Radau"
ODE_RTOL: float = 1e-4
ODE_ATOL: float = 1e-6
ODE_MAX_STEP: float = 5.0  # seconds

# Flue gas thermal mass in furnace [kg] and Cp [J/(kg·K)]
FURNACE_GAS_MASS: float = 5000.0
FURNACE_GAS_CP: float = 1100.0

# Pressure response time constant [s]
# How fast drum pressure tracks saturation pressure
TAU_PRESSURE: float = 30.0


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
        """
        Args:
            params: Fixed boiler design parameters.
            excess_air_ratio: Combustion lambda (1.0=stoichiometric, 1.1=typical).
            air_preheat_temp: Combustion air inlet temperature [K].
                              Defaults to params.ambient_temp.
        """
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

        Uses square-root pressure drop model (more accurate than linear):
            m = Cv * position * sqrt(P - P_downstream)
        Assumes fixed downstream pressure of 10 bar (turbine inlet).
        """
        p_downstream = 10.0e5  # Pa — turbine inlet pressure
        dp = max(pressure_pa - p_downstream, 0.0)
        flow = self.params.steam_valve_coeff * valve_position * np.sqrt(dp / 1.0e5)
        return float(np.clip(flow, 0.0, self.params.max_steam_flow))

    def _feedwater_flow(self, valve_position: float) -> float:
        """Feedwater mass flow into drum [kg/s]."""
        max_feedwater = 300.0  # kg/s
        return valve_position * max_feedwater

    # ─── Heat transfer ────────────────────────────────────────────────────────

    def _heat_to_water(self, flue_gas_temp: float, water_temp: float) -> float:
        """
        Heat transferred from flue gas to drum water [W].

        UA coefficient is fixed by design — a more advanced model
        would make this a function of flow rates and fouling.
        """
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
        water_temp: float,
    ) -> float:
        """
        Enthalpy carried out of the drum by steam [W].

        This is an open thermodynamic system: steam physically leaves the
        control volume and carries its full specific enthalpy h_steam with it.
        We do NOT subtract h_liquid — that subtraction would be valid only in
        a closed system. Removing it corrects a 2.5–3x underestimate of energy
        leaving the drum, which previously caused unbounded pressure rise.

        Q_steam_out = m_steam * h_steam(T, P)
        """
        if steam_flow <= 0.0:
            return 0.0
        h_steam = steam_tables.steam_enthalpy(
            temp_k=water_temp,
            pressure_pa=pressure_pa,
        )
        return steam_flow * h_steam

    # ─── Event functions for solve_ivp ───────────────────────────────────────
    # Each event function must have .terminal and .direction set individually.
    # direction =  1.0 : trigger when crossing zero from below (value rising)
    # direction = -1.0 : trigger when crossing zero from above (value falling)

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
        """Trigger when steam temperature rises above TEMP_STEAM_MAX."""
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
        water_temp = max(state.water_temp, 373.15)
        pressure_pa = max(state.pressure, 1.0e5)

        # ── Water mass via IAPWS-IF97 density (not a fixed constant) ─────────
        rho_water = steam_tables.water_density(water_temp, pressure_pa)
        water_mass = rho_water * self.params.drum_cross_section * water_level
        water_mass = max(water_mass, 1.0)  # avoid div-by-zero

        # ── Cp of water at current conditions via IAPWS-IF97 ─────────────────
        cp_water = steam_tables.water_specific_heat(water_temp, pressure_pa)

        # ── Valve positions (actual, after actuator dynamics) ─────────────────
        fv = controls.fuel_valve.position
        wv = controls.feedwater_valve.position
        sv = controls.steam_valve.position

        # ── Combustion ────────────────────────────────────────────────────────
        comb = self.combustion.calculate(fuel_valve=fv)

        # ── Superheater FIRST — it sees the hottest flue gas directly ─────────
        # Must be calculated before economizer so we can pass sh.flue_gas_temp_out
        # to the economizer (correct flue gas path order).
        m_steam = self._steam_flow(pressure_pa, sv)
        sh = self.superheater.calculate(
            pressure_pa=pressure_pa,
            steam_flow=m_steam,
            flue_gas_temp_in=state.flue_gas_temp,
            flue_gas_flow=comb.flue_gas_flow,
        )

        # ── Economizer SECOND — uses flue gas cooled by the superheater ───────
        # sh.flue_gas_temp_out is the physically correct inlet temperature here.
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

        # Steam carries its full enthalpy out of the open control volume.
        q_steam_out = self._heat_carried_by_steam(m_steam, pressure_pa, water_temp)

        # Feedwater adds energy relative to absolute zero reference (273.15 K).
        # U is extensive: adding mass always increases U. The drum temperature
        # may drop (mixing effect) but that is captured in dT_water_dt below,
        # not by making q_feedwater_in negative.
        q_feedwater_in = m_feed_raw * cp_water * (feedwater_temp_actual - 273.15)

        # ── ODE 1: dU/dt — internal energy ────────────────────────────────────
        du_dt = q_gas_to_water + q_feedwater_in - q_steam_out - q_loss

        # ── ODE 2: dP/dt — pressure tracks IAPWS-IF97 saturation pressure ────
        p_sat = steam_tables.saturation_pressure(water_temp)
        dp_dt = (p_sat - pressure_pa) / TAU_PRESSURE

        # ── ODE 3: dh/dt — water level (mass balance) ─────────────────────────
        # Steam volume flow uses real steam density from IAPWS-IF97
        rho_steam = steam_tables.steam_density(water_temp, pressure_pa)
        rho_steam = max(rho_steam, 0.1)  # physical floor
        dh_dt = (m_feed_raw - m_steam / rho_steam) / (
            rho_water * self.params.drum_cross_section
        )

        # ── ODE 4: dT_gas/dt — furnace flue gas temperature ──────────────────
        # Available heat minus what drum tubes and superheater absorb
        q_sh_absorbed = sh.heat_transferred
        dt_gas_dt = (comb.heat_available - q_gas_to_water - q_sh_absorbed) / (
            FURNACE_GAS_MASS * FURNACE_GAS_CP
        )

        # ── ODE 5: dT_water/dt — variable-mass corrected ──────────────────────
        # For a system with changing mass U = M * cp * T, the product rule gives:
        #   dU/dt = dM/dt * cp * T + M * cp * dT/dt
        # Solving for dT/dt:
        #   dT/dt = (dU/dt - dM/dt * cp * T) / (M * cp)
        # This correctly accounts for the thermal dilution effect when cold
        # feedwater mixes with hotter drum water.
        dm_dt = m_feed_raw - m_steam  # net mass flow into drum [kg/s]
        dt_water_dt = (du_dt - dm_dt * cp_water * water_temp) / (water_mass * cp_water)

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

        Solver: Radau (implicit, handles stiff systems correctly).
        Events: stops automatically on pressure/level/temperature limits.

        Args:
            initial_state: Starting state of the boiler.
            controls: Control inputs (valve commands + actuator states).
            t_span: (t_start, t_end) in seconds.
            dt: Output time step in seconds.

        Returns:
            scipy OdeResult. Check result.status:
                0 = reached t_end normally
               -1 = integration step failed
                1 = termination event triggered (alarm condition)
        """
        controls.update_valves(dt)

        t_eval = np.arange(t_span[0], t_span[1], dt)
        y0 = initial_state.to_vector()

        # Set terminal flag and correct crossing direction per event.
        # direction= 1.0 : fires when function crosses zero going UP
        # direction=-1.0 : fires when function crosses zero going DOWN
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
        """
        Return human-readable simulation termination reason.

        Useful for detecting which alarm condition was triggered.
        """
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
