"""
Thermodynamic ODE model of a steam boiler.

The boiler state is described by 4 coupled ODEs:
    dU/dt    — internal energy balance (heat in - heat out)
    dP/dt    — steam pressure dynamics
    dh/dt    — water level dynamics (mass balance)
    dT_gas/dt — flue gas temperature dynamics
"""

import numpy as np
from scipy.integrate import (
    OdeResult,
    solve_ivp,
)

from physics_engine.constants import (
    ANTOINE_A,
    ANTOINE_B,
    ANTOINE_C,
    COMBUSTION_EFFICIENCY,
    FUEL_HEATING_VALUE,
    LATENT_HEAT_VAPORIZATION,
    SPECIFIC_HEAT_STEAM,
    SPECIFIC_HEAT_WATER,
)
from physics_engine.models import BoilerParameters, BoilerState, ControlInputs


class BoilerModel:
    """
    Numerical ODE model of a steam boiler using RK45 integration.

    Physics summary:
        - Energy balance: heat from combustion heats water, steam carries energy out
        - Pressure: derived from saturation temperature via Antoine equation
        - Water level: feedwater in minus steam out (mass balance)
        - Flue gas: heated by combustion, cooled by heat transfer to water

    Usage:
        params = BoilerParameters()
        model = BoilerModel(params)
        initial = params.nominal_initial_state()
        controls = ControlInputs(fuel_valve=0.7, feedwater_valve=0.5, steam_valve=0.6)
        result = model.simulate(initial, controls, t_span=(0, 300), dt=1.0)
    """

    def __init__(self, params: BoilerParameters) -> None:
        self.params = params

    # ─── Antoine equation ────────────────────────────────────────────────────

    def _saturation_pressure(self, temp_k: float) -> float:
        """
        Calculate saturation pressure from temperature using Antoine equation.

        Antoine: ln(P) = A - B / (T - C)
        Returns pressure in Pa.
        """
        temp_k = max(temp_k, 373.15)  # clamp to boiling point minimum
        ln_p = ANTOINE_A - ANTOINE_B / (temp_k - ANTOINE_C)
        return float(np.exp(ln_p))

    def _saturation_temp(self, pressure_pa: float) -> float:
        """
        Calculate saturation temperature from pressure (inverse Antoine).

        Returns temperature in K.
        """
        pressure_pa = max(pressure_pa, 1.0e4)  # avoid log(0)
        ln_p = float(np.log(pressure_pa))
        return ANTOINE_B / (ANTOINE_A - ln_p) + ANTOINE_C

    # ─── Steam flow ───────────────────────────────────────────────────────────

    def _steam_flow(self, pressure_pa: float, steam_valve: float) -> float:
        """
        Calculate steam mass flow through the outlet valve.

        Simple linear valve model: m_steam = Cv * valve_position * pressure_bar
        Returns flow in kg/s.
        """
        pressure_bar = pressure_pa / 1.0e5
        flow = self.params.steam_valve_coeff * steam_valve * pressure_bar
        return float(np.clip(flow, 0.0, self.params.max_steam_flow))

    def _feedwater_flow(self, feedwater_valve: float) -> float:
        """
        Calculate feedwater mass flow into the drum.

        Returns flow in kg/s.
        """
        max_feedwater = 300.0  # kg/s — maximum feedwater pump capacity
        return feedwater_valve * max_feedwater

    # ─── Heat flows ──────────────────────────────────────────────────────────

    def _heat_from_combustion(self, fuel_valve: float) -> float:
        """
        Heat released by burning fuel [W].

        Q_fuel = m_fuel * LHV * eta_combustion
        """
        m_fuel = fuel_valve * self.params.max_fuel_flow
        return m_fuel * FUEL_HEATING_VALUE * COMBUSTION_EFFICIENCY

    def _heat_to_water(self, flue_gas_temp: float, water_temp: float) -> float:
        """
        Heat transferred from flue gas to water [W].

        Q_transfer = UA * (T_gas - T_water)
        """
        delta_t = flue_gas_temp - water_temp
        return self.params.heat_transfer_coeff * max(delta_t, 0.0)

    def _heat_loss(self, water_temp: float) -> float:
        """
        Heat lost through boiler walls to ambient [W].

        Q_loss = UA_loss * (T_water - T_ambient)
        """
        delta_t = water_temp - self.params.ambient_temp
        return self.params.heat_loss_coeff * max(delta_t, 0.0)

    def _heat_carried_by_steam(self, steam_flow: float, water_temp: float) -> float:
        """
        Enthalpy carried out by steam [W].

        Includes latent heat of vaporization + sensible heat of superheated steam.
        """
        enthalpy_per_kg = LATENT_HEAT_VAPORIZATION + SPECIFIC_HEAT_STEAM * (
            water_temp - 373.15
        )
        return steam_flow * enthalpy_per_kg

    # ─── ODE right-hand side ─────────────────────────────────────────────────

    def _derivatives(
        self,
        t: float,  # noqa: ARG002  (time unused — autonomous system)
        y: list[float],
        controls: ControlInputs,
    ) -> list[float]:
        """
        Compute dy/dt for the ODE solver.

        State vector y = [U, P, h, T_gas]
        Returns derivative vector [dU/dt, dP/dt, dh/dt, dT_gas/dt]
        """
        state = BoilerState.from_vector(y)

        # ── Derived quantities ────────────────────────────────────────────────
        water_mass = (
            self.params.water_density
            * self.params.drum_cross_section
            * max(state.water_level, 0.01)
        )
        water_temp = state.internal_energy / (
            water_mass * SPECIFIC_HEAT_WATER + 1.0  # +1 avoids div-by-zero
        )
        water_temp = max(water_temp, 373.15)

        # ── Flow rates ────────────────────────────────────────────────────────
        m_steam = self._steam_flow(state.pressure, controls.steam_valve)
        m_feed = self._feedwater_flow(controls.feedwater_valve)

        # ── Heat flows ────────────────────────────────────────────────────────
        q_combustion = self._heat_from_combustion(controls.fuel_valve)
        q_gas_to_water = self._heat_to_water(state.flue_gas_temp, water_temp)
        q_loss = self._heat_loss(water_temp)
        q_steam_out = self._heat_carried_by_steam(m_steam, water_temp)
        q_feedwater_in = (
            m_feed * SPECIFIC_HEAT_WATER * (self.params.feedwater_temp - 273.15)
        )

        # ── ODE 1: dU/dt — internal energy balance ────────────────────────────
        # Energy in: heat from gas + feedwater enthalpy
        # Energy out: heat in steam + wall losses
        du_dt = q_gas_to_water + q_feedwater_in - q_steam_out - q_loss

        # ── ODE 2: dP/dt — pressure dynamics ──────────────────────────────────
        # Pressure tracks saturation pressure of current water temp
        # with a lag time constant (boiler thermal inertia ~30s)
        p_sat = self._saturation_pressure(water_temp)
        tau_pressure = 30.0  # seconds — pressure response time constant
        dp_dt = (p_sat - state.pressure) / tau_pressure

        # ── ODE 3: dh/dt — water level (mass balance) ─────────────────────────
        # Level rises with feedwater, drops with steam (converted to volume)
        m_steam_liquid_equiv = m_steam * 0.001  # steam density ~1 kg/m³ → volume
        dh_dt = (m_feed - m_steam_liquid_equiv) / (
            self.params.water_density * self.params.drum_cross_section
        )

        # ── ODE 4: dT_gas/dt — flue gas temperature ───────────────────────────
        # Gas mass in furnace (approximate)
        m_gas = 5000.0  # kg — approximate flue gas mass in furnace
        cp_gas = 1100.0  # J/(kg·K) — specific heat of flue gas
        dt_gas_dt = (q_combustion - q_gas_to_water) / (m_gas * cp_gas)

        return [du_dt, dp_dt, dh_dt, dt_gas_dt]

    # ─── Public simulation interface ─────────────────────────────────────────

    def simulate(
        self,
        initial_state: BoilerState,
        controls: ControlInputs,
        t_span: tuple[float, float],
        dt: float = 1.0,
    ) -> OdeResult:
        """
        Integrate the ODE system over a time span.

        Args:
            initial_state: Starting state of the boiler.
            controls: Constant control inputs during this simulation period.
            t_span: (t_start, t_end) in seconds.
            dt: Output time step in seconds (does not affect solver accuracy).

        Returns:
            scipy OdeResult with .t (time array) and .y (state array, shape 4×N).
        """
        t_eval = np.arange(t_span[0], t_span[1], dt)
        y0 = initial_state.to_vector()

        result: OdeResult = solve_ivp(
            fun=lambda t, y: self._derivatives(t, y, controls),
            t_span=t_span,
            y0=y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-4,  # relative tolerance
            atol=1e-6,  # absolute tolerance
            max_step=5.0,  # max solver step: 5 seconds
        )

        return result

    def get_state_at(self, result: OdeResult, index: int) -> BoilerState:
        """Extract BoilerState from ODE result at a given time index."""
        y = [float(result.y[i, index]) for i in range(4)]
        return BoilerState.from_vector(y)
