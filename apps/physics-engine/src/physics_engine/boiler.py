"""
Thermodynamic model of the drum boiler — lumped and energy-conserving.

State vector y = [U, P, h, T_gas, T_water]:
    U        — energy stored in the drum water [J] (bookkeeping, not fed back)
    P        — drum pressure [Pa]; follows saturation pressure of the water with a lag
    h        — drum water level [m]
    T_gas    — furnace exit gas temperature [K]
    T_water  — drum water temperature [K]

Heat path, in order of falling gas temperature:
    furnace (gas node, radiant water walls)
        → superheater → evaporator bank → economizer → stack

Balances:
    furnace:  M_g·cp_g·dT_gas/dt = Q_fuel − Q_walls − m_gas·cp_g·(T_gas − T_ambient)
    drum:     C·dT_water/dt = Q_walls + Q_bank − Q_loss
                               + m_fw·(h_eco − h_f) − (m_steam + m_leak)·(h_g − h_f)
              with C = M_water·cp_f + C_storage (pressure parts and circulating water)
    level:    ρ_f·A·dh/dt = m_fw − m_steam − m_leak

At steady state the fuel heat equals the heat carried into the steam and feedwater plus
the stack and wall losses: the model neither creates nor destroys energy. The turbine
admission valve is choked, so steam flow is proportional to valve opening and pressure.
Spray water for the attemperator is taken from the feedwater pump and bypasses the drum.

`step` advances the state with a fixed-step RK4 and is what the live runtime uses: it is
deterministic and fast. `simulate` integrates with an adaptive implicit solver and alarm
events for offline analysis.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from physics_engine import properties
from physics_engine.combustion import CombustionModel
from physics_engine.constants import (
    FUEL_HEATING_VALUE,
    PRESSURE_MAX,
    PRESSURE_MIN,
    PRESSURE_NOMINAL,
    RATED_STEAM_FLOW,
    SPRAY_MAX_FRACTION,
    TEMP_STEAM_MAX,
)
from physics_engine.faults import NO_DISTURBANCES, PlantDisturbances
from physics_engine.heat_exchanger import (
    CP_FLUE_GAS,
    EconomizerModel,
    EvaporatorBankModel,
    SuperheaterModel,
)
from physics_engine.models import BoilerParameters, BoilerState, ControlInputs

# ─── ODE solver configuration ─────────────────────────────────────────────────

ODE_METHOD: str = "Radau"
ODE_RTOL: float = 1e-4
ODE_ATOL: float = 1e-6
ODE_MAX_STEP: float = 5.0  # seconds

# Flue gas inventory of the furnace [kg] — about 10 000 m³ at ~0.21 kg/m³ — and its Cp
# [J/(kg·K)]. After a fuel trip this gas gives its heat to the water walls, so an
# oversized inventory would make the drum pressure climb long after the flame is out.
# Only the product mass x Cp matters for that; it is 2.75e6 J/K, as calibrated in S2.
FURNACE_GAS_MASS: float = 2115.0
FURNACE_GAS_CP: float = CP_FLUE_GAS

# Spring-loaded drum safety valves: mechanical protection that acts without the PLC.
# They start to lift at 172 bar and discharge their rated capacity at 178 bar, which
# holds a boiler with the turbine isolated and a hot furnace below the 185 bar trip.
SAFETY_VALVE_SET_PRESSURE: float = 172.0e5  # Pa
SAFETY_VALVE_FULL_LIFT_PRESSURE: float = 178.0e5  # Pa
SAFETY_VALVE_CAPACITY: float = 90.0  # kg/s

# Pressure response time constant [s]: how fast drum pressure tracks the saturation
# pressure of the water. It stands in for the steam-space dynamics the lumped drum
# does not resolve.
TAU_PRESSURE: float = 30.0

# The water temperature is kept inside the range of the property tables: below the
# critical point, where saturation properties exist, and above the boiling point at
# atmospheric pressure, the coldest state the drum model represents.
WATER_TEMP_MIN: float = 373.15
WATER_TEMP_MAX: float = properties.T_TABLE_MAX_K

# Maximum cp used in the drum energy equations [J/(kg·K)]. Near the critical point the
# IAPWS-IF97 cp of saturated liquid diverges; a lumped drum would over-amplify it.
CP_WATER_MAX: float = 5_000.0

MIN_MODEL_PRESSURE: float = 0.1e5  # Pa
MIN_MODEL_LEVEL: float = 0.01  # m

# Below this level the evaporator runs short of water and steam production collapses;
# within this margin of the drum top the feedwater pump can no longer push water in.
DRY_OUT_LEVEL: float = 0.2  # m
OVERFILL_MARGIN: float = 0.2  # m


@dataclass(frozen=True)
class BoilerBalance:
    """Flows, heat duties and derivatives of the boiler at one state."""

    fuel_flow: float  # kg/s
    flue_gas_flow: float  # kg/s
    excess_air_ratio: float  # —
    flame_temp: float  # K — adiabatic
    heat_release: float  # W — heat available in the furnace
    turbine_steam_flow: float  # kg/s — through the admission valve
    drum_steam_flow: float  # kg/s — raised in the drum
    spray_flow: float  # kg/s — attemperator water
    leak_flow: float  # kg/s — steam lost through a leak
    relief_flow: float  # kg/s — steam discharged by the drum safety valves
    feedwater_flow: float  # kg/s — into the drum
    furnace_to_water: float  # W
    superheater_heat: float  # W
    evaporator_bank_heat: float  # W
    economizer_heat: float  # W
    heat_loss: float  # W — through the boiler casing
    stack_loss: float  # W
    stack_temp: float  # K
    saturation_temp: float  # K
    superheater_outlet_temp: float  # K — before spray
    superheater_outlet_enthalpy: float  # J/kg — before spray
    economizer_outlet_temp: float  # K
    spray_water_enthalpy: float  # J/kg
    derivatives: tuple[float, float, float, float, float]

    @property
    def turbine_inlet_enthalpy(self) -> float:
        """Steam enthalpy after spray water has mixed in [J/kg]."""
        if self.turbine_steam_flow <= 0.0:
            return self.superheater_outlet_enthalpy
        return (
            self.drum_steam_flow * self.superheater_outlet_enthalpy
            + self.spray_flow * self.spray_water_enthalpy
        ) / self.turbine_steam_flow

    @property
    def boiler_efficiency(self) -> float:
        """Heat delivered to water and steam over fuel chemical energy (LHV) [—]."""
        if self.fuel_flow <= 0.0:
            return 0.0
        absorbed = (
            self.furnace_to_water
            + self.superheater_heat
            + self.evaporator_bank_heat
            + self.economizer_heat
            - self.heat_loss
        )
        return max(absorbed, 0.0) / (self.fuel_flow * FUEL_HEATING_VALUE)


class BoilerModel:
    """
    Full thermodynamic ODE model of a steam boiler.

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
        state1 = model.step(state0, ctrl, dt=1.0)
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
        self.evaporator_bank = EvaporatorBankModel()
        self.economizer = EconomizerModel()

    # ─── Flow calculations ────────────────────────────────────────────────────

    def _steam_flow(self, pressure_pa: float, valve_position: float) -> float:
        """
        Steam mass flow through the turbine admission valve [kg/s].

        The valve and turbine nozzles run choked, so flow is proportional to the
        valve opening and to drum pressure:
            m = Cv · position · P / P_nominal
        """
        flow = (
            self.params.steam_valve_coeff
            * valve_position
            * max(pressure_pa, 0.0)
            / PRESSURE_NOMINAL
        )
        return float(np.clip(flow, 0.0, self.params.max_steam_flow))

    def _feedwater_flow(
        self, valve_position: float, pump_capacity_factor: float = 1.0
    ) -> float:
        """Feedwater mass flow into drum [kg/s]."""
        return valve_position * self.params.max_feedwater_flow * pump_capacity_factor

    # ─── Balance ──────────────────────────────────────────────────────────────

    def balance(
        self,
        state: BoilerState,
        controls: ControlInputs,
        disturbances: PlantDisturbances = NO_DISTURBANCES,
    ) -> BoilerBalance:
        """Evaluate every flow, duty and derivative of the boiler at one state."""
        p = self.params
        pressure = max(state.pressure, MIN_MODEL_PRESSURE)
        level = min(max(state.water_level, MIN_MODEL_LEVEL), p.drum_height)
        water_temp = min(max(state.water_temp, WATER_TEMP_MIN), WATER_TEMP_MAX)
        gas_temp = state.flue_gas_temp

        rho_water = properties.liquid_density(water_temp)
        cp_water = min(properties.liquid_cp(water_temp), CP_WATER_MAX)
        h_liquid = properties.liquid_enthalpy(water_temp)
        h_vapor = properties.vapor_enthalpy_at_pressure(pressure)
        t_sat = properties.saturation_temperature(pressure)
        water_mass = max(rho_water * p.drum_cross_section * level, 1.0)

        # ── Combustion ────────────────────────────────────────────────────────
        comb = self.combustion.calculate(
            fuel_valve=controls.fuel_valve.position,
            efficiency_factor=disturbances.combustion_efficiency_factor,
        )
        gas_flow = comb.flue_gas_flow

        # ── Water and steam flows ─────────────────────────────────────────────
        water_availability = min(max(state.water_level / DRY_OUT_LEVEL, 0.0), 1.0)
        feed_headroom = min(
            max((p.drum_height - state.water_level) / OVERFILL_MARGIN, 0.0), 1.0
        )
        pump = disturbances.feedwater_capacity_factor

        turbine_flow = (
            self._steam_flow(pressure, controls.steam_valve.position)
            * water_availability
        )
        spray_flow = min(
            controls.spray_valve.position * p.max_spray_flow * pump,
            SPRAY_MAX_FRACTION * turbine_flow,
        )
        drum_steam = turbine_flow - spray_flow
        leak_flow = (
            disturbances.steam_leak_fraction
            * RATED_STEAM_FLOW
            * pressure
            / PRESSURE_NOMINAL
            * water_availability
        )
        feedwater = (
            self._feedwater_flow(controls.feedwater_valve.position, pump)
            * feed_headroom
        )
        lift = (pressure - SAFETY_VALVE_SET_PRESSURE) / (
            SAFETY_VALVE_FULL_LIFT_PRESSURE - SAFETY_VALVE_SET_PRESSURE
        )
        relief_flow = (
            SAFETY_VALVE_CAPACITY * min(max(lift, 0.0), 1.0) * water_availability
        )

        # ── Gas path ──────────────────────────────────────────────────────────
        q_walls = p.heat_transfer_coeff * (gas_temp - water_temp)
        q_gas_exit = gas_flow * FURNACE_GAS_CP * (gas_temp - p.ambient_temp)

        sh = self.superheater.calculate(
            pressure_pa=pressure,
            steam_flow=drum_steam,
            flue_gas_temp_in=gas_temp,
            flue_gas_flow=gas_flow,
        )
        bank = self.evaporator_bank.calculate(
            saturation_temp=t_sat,
            flue_gas_temp_in=sh.flue_gas_temp_out,
            flue_gas_flow=gas_flow,
        )
        eco = self.economizer.calculate(
            feedwater_flow=feedwater,
            feedwater_temp_in=p.feedwater_temp,
            pressure_pa=pressure,
            flue_gas_temp_in=bank.flue_gas_temp_out,
            flue_gas_flow=gas_flow,
        )
        h_feedwater_in = properties.liquid_enthalpy(p.feedwater_temp)
        h_economizer_out = h_feedwater_in + eco.water_enthalpy_gain
        stack_temp = eco.flue_gas_temp_out
        stack_loss = gas_flow * FURNACE_GAS_CP * max(stack_temp - p.ambient_temp, 0.0)
        q_loss = p.heat_loss_coeff * max(water_temp - p.ambient_temp, 0.0)

        # ── Derivatives ───────────────────────────────────────────────────────
        q_drum = q_walls + bank.heat_transferred - q_loss
        steam_out = drum_steam + leak_flow + relief_flow
        du_dt = q_drum + feedwater * h_economizer_out - steam_out * h_vapor
        heat_capacity = water_mass * cp_water + p.storage_heat_capacity
        dt_water_dt = (
            q_drum
            + feedwater * (h_economizer_out - h_liquid)
            - steam_out * (h_vapor - h_liquid)
        ) / heat_capacity
        dp_dt = (properties.saturation_pressure(water_temp) - pressure) / TAU_PRESSURE
        dh_dt = (feedwater - steam_out) / (rho_water * p.drum_cross_section)
        dt_gas_dt = (comb.heat_available - q_walls - q_gas_exit) / (
            FURNACE_GAS_MASS * FURNACE_GAS_CP
        )

        return BoilerBalance(
            fuel_flow=comb.fuel_flow,
            flue_gas_flow=gas_flow,
            excess_air_ratio=comb.excess_air_ratio,
            flame_temp=comb.flue_gas_temp_exit,
            heat_release=comb.heat_available,
            turbine_steam_flow=turbine_flow,
            drum_steam_flow=drum_steam,
            spray_flow=spray_flow,
            leak_flow=leak_flow,
            relief_flow=relief_flow,
            feedwater_flow=feedwater,
            furnace_to_water=q_walls,
            superheater_heat=sh.heat_transferred,
            evaporator_bank_heat=bank.heat_transferred,
            economizer_heat=eco.heat_transferred,
            heat_loss=q_loss,
            stack_loss=stack_loss,
            stack_temp=stack_temp,
            saturation_temp=t_sat,
            superheater_outlet_temp=sh.steam_temp_out,
            superheater_outlet_enthalpy=sh.steam_enthalpy_out,
            economizer_outlet_temp=eco.water_temp_out,
            spray_water_enthalpy=h_feedwater_in,
            derivatives=(du_dt, dp_dt, dh_dt, dt_gas_dt, dt_water_dt),
        )

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
        Last-resort safety event: triggers if drum temperature somehow exceeds
        TEMP_STEAM_MAX (838 K / 565°C).

        The water temperature used by the derivatives is clamped below the critical
        point, so this never fires in normal operation; it is a numerical safety net.
        """
        return y[4] - TEMP_STEAM_MAX

    # ─── ODE right-hand side ─────────────────────────────────────────────────

    def _derivatives(
        self,
        t: float,  # noqa: ARG002
        y: list[float],
        controls: ControlInputs,
        disturbances: PlantDisturbances = NO_DISTURBANCES,
    ) -> list[float]:
        """
        Compute dy/dt for the ODE solver.

        State vector y = [U, P, h, T_gas, T_water]
        Returns [dU/dt, dP/dt, dh/dt, dT_gas/dt, dT_water/dt]
        """
        balance = self.balance(BoilerState.from_vector(y), controls, disturbances)
        return list(balance.derivatives)

    # ─── Public simulation interface ─────────────────────────────────────────

    def step(
        self,
        state: BoilerState,
        controls: ControlInputs,
        dt: float,
        disturbances: PlantDisturbances = NO_DISTURBANCES,
    ) -> BoilerState:
        """
        Advance the boiler by one fixed step with classic Runge-Kutta 4.

        Valves move first, then the state is integrated with valve positions held
        over the step. The fastest dynamics (furnace gas, ~7 s) are well inside the
        RK4 stability region at the 1 s step the runtime uses.
        """
        controls.update_valves(dt)
        y0 = np.asarray(state.to_vector(), dtype=float)

        def rate(y: np.ndarray) -> np.ndarray:
            return np.asarray(
                self.balance(
                    BoilerState.from_vector([float(v) for v in y]),
                    controls,
                    disturbances,
                ).derivatives
            )

        k1 = rate(y0)
        k2 = rate(y0 + 0.5 * dt * k1)
        k3 = rate(y0 + 0.5 * dt * k2)
        k4 = rate(y0 + dt * k3)
        y1 = y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self._bounded(BoilerState.from_vector([float(v) for v in y1]))

    def _bounded(self, state: BoilerState) -> BoilerState:
        """Keep an integrated state inside the physical domain of the model."""
        return BoilerState(
            internal_energy=max(state.internal_energy, 0.0),
            pressure=max(state.pressure, MIN_MODEL_PRESSURE),
            water_level=min(max(state.water_level, 0.0), self.params.drum_height),
            flue_gas_temp=max(state.flue_gas_temp, self.params.ambient_temp),
            water_temp=min(max(state.water_temp, WATER_TEMP_MIN), WATER_TEMP_MAX),
        )

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
        # Valve dynamics are stepped once per discrete control interval.
        # Callers that want realistic actuator lag must reuse the same
        # ControlInputs instance across successive simulate() calls.
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
