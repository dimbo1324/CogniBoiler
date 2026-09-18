"""
Virtual PLC: the only path from an operator to the actuators.

Every scan follows one plant state published by PhysicsService. The interlocks and the
alarm monitor judge the measurements first; then, by mode, the unit controller computes
the valves (AUTO), the operator's last command stands (MANUAL), or the trip response
holds the unit safe (ESTOP). Accepted commands go to PhysicsService; alarm conditions and
PLC events go to MQTT.

An E-Stop latches. Only an explicit reset clears it, and only once no critical condition
is active any more; the reset hands the unit back to AUTO without a bump.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

import cogniboiler_pb2 as pb2

from plc_controller.alarms import AlarmConditionMonitor, AlarmTransition, Severity
from plc_controller.client import PhysicsClient
from plc_controller.control import (
    RATED_POWER_W,
    ControlTargets,
    UnitController,
)
from plc_controller.events import PlcEvent, PlcEventKind, PlcPublisher, now_ms
from plc_controller.measurements import (
    SENSOR_DRUM_LEVEL,
    SENSOR_DRUM_PRESSURE,
    SENSOR_DRUM_WATER_TEMP,
    SENSOR_FURNACE_GAS_TEMP,
    SENSOR_STEAM_TEMP,
    ProcessMeasurements,
)
from plc_controller.metrics import SCAN_SECONDS
from plc_controller.safety import (
    WATER_LEVEL_LIMITS,
    ArmingState,
    ArmingTracker,
    SafetyEvent,
    SafetyInterlock,
    trip_overrides,
)

logger = logging.getLogger(__name__)

# Valve position limits
VALVE_MIN: float = 0.0
VALVE_MAX: float = 1.0

# Setpoints an engineer may choose. They stay inside the alarm warning bands, so an
# accepted setpoint never parks the unit in a standing alarm: pressure below the
# 160 bar warning, level inside 2–7 m, steam below the 565 °C warning.
PRESSURE_SETPOINT_MIN_PA: float = 60.0e5
PRESSURE_SETPOINT_MAX_PA: float = 155.0e5
LEVEL_SETPOINT_MIN_M: float = 2.5
LEVEL_SETPOINT_MAX_M: float = 6.5
TEMP_SETPOINT_MIN_K: float = 700.0
TEMP_SETPOINT_MAX_K: float = 835.0

LOAD_DEMAND_MIN_W: float = 0.0
LOAD_DEMAND_MAX_W: float = RATED_POWER_W

DEFAULT_CONTROL_INTERVAL_S: float = 0.2
DEFAULT_ALERT_MQTT_HOST: str = "localhost"
DEFAULT_ALERT_MQTT_PORT: int = 1883

# Only people and schedules send commands from outside. PID and SAFETY name the PLC's
# own outputs; accepting them from a caller would let it pass for the interlock.
EXTERNAL_COMMAND_SOURCES: frozenset[int] = frozenset(
    {int(pb2.CommandSource.OPERATOR), int(pb2.CommandSource.SCHEDULER)}
)

# Interlock trip causes named differently from the alarm parameter they depend on.
TRIP_CAUSE_PARAMETERS: dict[str, str] = {"fuel_permissive": "water_level_m"}

# Measured values whose limits are alarmed, with the instrument that provides them.
ALARMED_VALUES: tuple[tuple[str, str], ...] = (
    ("pressure_pa", SENSOR_DRUM_PRESSURE),
    ("water_level_m", SENSOR_DRUM_LEVEL),
    ("water_temp_k", SENSOR_DRUM_WATER_TEMP),
    ("flue_gas_temp_k", SENSOR_FURNACE_GAS_TEMP),
    ("steam_temp_k", SENSOR_STEAM_TEMP),
)


class RuntimeMode(StrEnum):
    """PLC operating mode."""

    AUTO = "auto"
    MANUAL = "manual"
    ESTOP = "estop"


@dataclass
class Setpoints:
    """Engineer's targets for the process loops."""

    pressure_pa: float = 140.0e5  # 140 bar nominal
    water_level_m: float = 4.8
    steam_temp_k: float = 811.0  # turbine inlet design temperature
    updated_at_ms: int = field(default_factory=now_ms)


@dataclass
class ValidationResult:
    """Result of a command or setpoint validation check."""

    accepted: bool
    reason: str = ""


@dataclass
class CommandSnapshot:
    """A valve command the PLC sent or is about to send."""

    fuel_valve: float = 0.5
    feedwater_valve: float = 0.5
    steam_valve: float = 0.5
    spray_valve: float = 0.0
    source: int = pb2.CommandSource.PID
    operator_id: str = ""
    timestamp_ms: int = field(default_factory=now_ms)


@dataclass
class SafetySnapshot:
    """The event that latched the E-Stop."""

    timestamp_ms: int
    parameter: str
    value: float
    threshold: float
    level: str
    action: str

    @classmethod
    def from_event(cls, event: SafetyEvent) -> SafetySnapshot:
        return cls(
            timestamp_ms=event.timestamp_ms,
            parameter=event.parameter,
            value=event.value,
            threshold=event.threshold,
            level=event.level.value,
            action=event.action.value,
        )


class PLCService:
    """Virtual PLC business logic with an event-driven scan loop."""

    VERSION = "0.3.0"

    def __init__(
        self,
        physics_client: PhysicsClient | None = None,
        *,
        control_interval_s: float = DEFAULT_CONTROL_INTERVAL_S,
        mqtt_host: str = DEFAULT_ALERT_MQTT_HOST,
        mqtt_port: int = DEFAULT_ALERT_MQTT_PORT,
        enable_control_loop: bool = True,
        enable_alert_publishing: bool = True,
        mqtt_username: str | None = None,
        mqtt_password: str | None = None,
    ) -> None:
        self._physics = physics_client or PhysicsClient()
        self._retry_delay_s = max(control_interval_s, 0.05)
        self._enable_control_loop = enable_control_loop

        self._setpoints = Setpoints()
        self._load_demand_w: float | None = None
        self._mode = RuntimeMode.AUTO
        self._controller = UnitController()
        self._interlock = SafetyInterlock()
        self._arming = ArmingTracker()
        self._alarms = AlarmConditionMonitor()
        self._publisher = PlcPublisher(
            mqtt_host,
            mqtt_port,
            enabled=enable_alert_publishing,
            active_conditions=self._alarms.active,
            username=mqtt_username,
            password=mqtt_password,
        )

        self._lock = asyncio.Lock()
        self._start_time = time.monotonic()
        self._commands_received = 0
        self._commands_rejected = 0
        self._commands_forwarded = 0
        self._latest_command = CommandSnapshot()
        self._manual_command: CommandSnapshot | None = None
        self._last_sent_key: tuple[float, float, float, float, int, str] | None = None
        self._latest_state: pb2.SystemStateMsg | None = None
        self._latest_measurements: ProcessMeasurements | None = None
        self._trip_cause: SafetySnapshot | None = None
        self._run_id: int | None = None
        self._last_simulation_time_s: float | None = None
        self._scans_completed = 0
        self._last_scanned_step = -1

        self._control_task: asyncio.Task[None] | None = None
        self._task_error = ""
        self._stream_failing = False

    # ─── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start publishing and, if enabled, the scan loop."""
        self._publisher.start()
        if not self._enable_control_loop:
            return
        if self._control_task is not None and not self._control_task.done():
            return
        self._control_task = asyncio.create_task(
            self._run_control_loop(), name="plc-scan-loop"
        )

    async def close(self) -> None:
        """Stop background work and close remote connections."""
        task = self._control_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self._control_task = None
        await self._publisher.aclose()
        await self._physics.close()

    # ─── Health ─────────────────────────────────────────────────────────────

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def mode(self) -> RuntimeMode:
        return self._mode

    @property
    def stats(self) -> dict[str, int]:
        return {
            "commands_received": self._commands_received,
            "commands_rejected": self._commands_rejected,
            "commands_forwarded": self._commands_forwarded,
            "warnings": self._interlock.warning_count,
            "trips": self._interlock.trip_count,
            "scans": self._scans_completed,
        }

    @property
    def active_condition_count(self) -> int:
        return len(self._alarms.active())

    @property
    def last_scanned_step(self) -> int:
        """Plant step count of the latest state the scan loop finished; -1 before any."""
        return self._last_scanned_step

    async def physics_status(self) -> str:
        """Overall PLC status: degraded while tripped or cut off from the plant."""
        if self._task_error or self._mode is RuntimeMode.ESTOP:
            return "degraded"
        try:
            health = await self._physics.health()
        except Exception:
            return "degraded"
        return str(health.status)

    # ─── Targets ────────────────────────────────────────────────────────────

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
        operator_id: str = "",
    ) -> ValidationResult:
        """Validate and store new targets; the working setpoints ramp toward them."""
        if not (PRESSURE_SETPOINT_MIN_PA <= pressure_pa <= PRESSURE_SETPOINT_MAX_PA):
            return ValidationResult(
                accepted=False,
                reason=(
                    f"Pressure setpoint {pressure_pa / 1e5:.1f} bar "
                    f"outside [{PRESSURE_SETPOINT_MIN_PA / 1e5:.0f}, "
                    f"{PRESSURE_SETPOINT_MAX_PA / 1e5:.0f}] bar"
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
            updated_at_ms=now_ms(),
        )
        self._publisher.publish_event(
            PlcEvent(
                PlcEventKind.SETPOINTS_CHANGED,
                operator_id or "unknown",
                {
                    "pressure_pa": pressure_pa,
                    "water_level_m": water_level_m,
                    "steam_temp_k": steam_temp_k,
                },
            )
        )
        return ValidationResult(accepted=True)

    @property
    def load_demand_w(self) -> float | None:
        """Operator's load target; None until the PLC has seen the plant."""
        return self._load_demand_w

    def set_load_demand(self, load_w: float, operator_id: str = "") -> ValidationResult:
        """Set the electrical load target; the load setpoint ramps toward it."""
        if not LOAD_DEMAND_MIN_W <= load_w <= LOAD_DEMAND_MAX_W:
            return ValidationResult(
                accepted=False,
                reason=(
                    f"Load demand {load_w / 1e6:.1f} MW outside "
                    f"[{LOAD_DEMAND_MIN_W / 1e6:.0f}, {LOAD_DEMAND_MAX_W / 1e6:.0f}] MW"
                ),
            )
        previous = self._load_demand_w
        self._load_demand_w = load_w
        self._publisher.publish_event(
            PlcEvent(
                PlcEventKind.LOAD_DEMAND_CHANGED,
                operator_id or "unknown",
                {"load_w": load_w, "previous_load_w": previous},
            )
        )
        return ValidationResult(accepted=True)

    # ─── Commands ───────────────────────────────────────────────────────────

    def validate_command(
        self,
        fuel_valve: float,
        feedwater_valve: float,
        steam_valve: float,
        spray_valve: float | None = None,
    ) -> ValidationResult:
        """Validate a control command against hard valve bounds."""
        self._commands_received += 1

        values = [
            ("fuel_valve", fuel_valve),
            ("feedwater_valve", feedwater_valve),
            ("steam_valve", steam_valve),
        ]
        if spray_valve is not None:
            values.append(("spray_valve", spray_valve))
        for name, value in values:
            if not (VALVE_MIN <= value <= VALVE_MAX):
                self._commands_rejected += 1
                return ValidationResult(
                    accepted=False,
                    reason=f"{name}={value:.3f} outside [0.0, 1.0]",
                )

        return ValidationResult(accepted=True)

    def latest_command(self) -> CommandSnapshot:
        """Return the most recently accepted command."""
        latest = self._latest_command
        return CommandSnapshot(
            fuel_valve=latest.fuel_valve,
            feedwater_valve=latest.feedwater_valve,
            steam_valve=latest.steam_valve,
            spray_valve=latest.spray_valve,
            source=latest.source,
            operator_id=latest.operator_id,
            timestamp_ms=latest.timestamp_ms,
        )

    async def send_command(
        self,
        *,
        fuel_valve: float,
        feedwater_valve: float,
        steam_valve: float,
        source: int,
        operator_id: str,
        spray_valve: float | None = None,
    ) -> ValidationResult:
        """Validate an operator's valve command, switch to MANUAL and forward it."""
        result = self.validate_command(
            fuel_valve, feedwater_valve, steam_valve, spray_valve
        )
        if not result.accepted:
            return result
        if int(source) not in EXTERNAL_COMMAND_SOURCES:
            return self._reject(
                f"command source {pb2.CommandSource.Name(source)} is reserved for the PLC"
            )

        async with self._lock:
            if self._interlock.emergency_stop.is_active:
                return self._reject(
                    "Emergency stop is active. Reset required before commands."
                )
            measurements = self._latest_measurements
            if (
                fuel_valve > 0.0
                and measurements is not None
                and not self._interlock.fuel_permitted(measurements.water_level_m)
            ):
                return self._reject(
                    f"Fuel not permitted: drum level at or below "
                    f"{WATER_LEVEL_LIMITS.trip_low:.2f} m"
                )

            snapshot = CommandSnapshot(
                fuel_valve=fuel_valve,
                feedwater_valve=feedwater_valve,
                steam_valve=steam_valve,
                spray_valve=(
                    spray_valve
                    if spray_valve is not None
                    else self._current_spray_command()
                ),
                source=source,
                operator_id=operator_id,
                timestamp_ms=now_ms(),
            )
            self._change_mode(RuntimeMode.MANUAL, operator_id)
            self._manual_command = snapshot
            return await self._forward(snapshot)

    async def set_mode(self, mode: RuntimeMode, operator_id: str) -> ValidationResult:
        """AUTO or MANUAL on request; ESTOP is a manual trip that latches."""
        operator = operator_id or "unknown"
        async with self._lock:
            if mode is RuntimeMode.ESTOP:
                if not self._interlock.emergency_stop.is_active:
                    event = self._interlock.trip("manual_trip", 1.0, 1.0)
                    self._enter_estop(event, operator, PlcEventKind.MANUAL_TRIP)
                    if self._latest_measurements is not None:
                        await self._forward(
                            self._trip_command(self._latest_measurements, 0.0)
                        )
                return ValidationResult(accepted=True)

            if self._interlock.emergency_stop.is_active:
                return self._reject("Emergency stop is active. Reset it first.")
            if mode is RuntimeMode.MANUAL and self._mode is not RuntimeMode.MANUAL:
                self._manual_command = self.latest_command()
            if mode is RuntimeMode.AUTO:
                self._manual_command = None
                self._controller.invalidate()
            self._change_mode(mode, operator)
            return ValidationResult(accepted=True)

    async def reset_emergency_stop(self, operator_id: str) -> ValidationResult:
        """Clear the E-Stop latch once its cause is gone and return to AUTO."""
        operator = operator_id or "unknown"
        async with self._lock:
            if not self._interlock.emergency_stop.is_active:
                return ValidationResult(
                    accepted=True, reason="Emergency stop is not active."
                )
            blockers = self._reset_blockers()
            if blockers:
                self._publisher.publish_event(
                    PlcEvent(
                        PlcEventKind.ESTOP_RESET_REFUSED,
                        operator,
                        {"blockers": blockers},
                    )
                )
                return self._reject("Reset refused: " + "; ".join(blockers))

            cause = self._trip_cause
            self._interlock.reset(operator_id=operator)
            self._controller.invalidate()
            self._manual_command = None
            self._trip_cause = None
            self._mode = RuntimeMode.AUTO
            self._publisher.publish_event(
                PlcEvent(
                    PlcEventKind.ESTOP_RESET,
                    operator,
                    {"cause": cause.parameter if cause is not None else ""},
                )
            )
            logger.warning("E-Stop reset by %s; unit returned to AUTO", operator)
            return ValidationResult(accepted=True)

    # ─── Status ─────────────────────────────────────────────────────────────

    async def get_process_state(self) -> pb2.SystemStateMsg:
        """Fetch the current process state from the live PhysicsService."""
        return await self._physics.get_system_state()

    async def get_control_status(self) -> pb2.PLCStatusMsg:
        """The PLC's mode, targets, working setpoints, loops and alarm conditions."""
        latest = self.latest_command()
        setpoints = self.get_setpoints()
        cause = self._trip_cause
        output = self._controller.last_output
        working = (
            self._controller.working_setpoints() if self._controller.primed else None
        )
        blockers = (
            self._reset_blockers() if self._interlock.emergency_stop.is_active else []
        )
        load_demand = self._load_demand_w if self._load_demand_w is not None else 0.0
        return pb2.PLCStatusMsg(
            mode=self._mode_to_proto(self._mode),
            emergency_stop_active=self._interlock.emergency_stop.is_active,
            setpoints=pb2.SetpointsMsg(
                pressure_pa=setpoints.pressure_pa,
                water_level_m=setpoints.water_level_m,
                steam_temp_k=setpoints.steam_temp_k,
                timestamp_ms=setpoints.updated_at_ms,
            ),
            latest_command=pb2.ControlCommandMsg(
                fuel_valve=latest.fuel_valve,
                feedwater_valve=latest.feedwater_valve,
                steam_valve=latest.steam_valve,
                spray_valve=latest.spray_valve,
                timestamp_ms=latest.timestamp_ms,
                source=latest.source,
                operator_id=latest.operator_id,
            ),
            warning_count=self._interlock.warning_count,
            trip_count=self._interlock.trip_count,
            active_trip=(
                pb2.SafetyEventMsg(
                    timestamp_ms=cause.timestamp_ms,
                    parameter=cause.parameter,
                    value=cause.value,
                    threshold=cause.threshold,
                    level=cause.level,
                    action=cause.action,
                )
                if cause is not None
                else pb2.SafetyEventMsg()
            ),
            load_demand_w=load_demand,
            load_setpoint_w=working.load_w if working is not None else load_demand,
            active_setpoints=(
                pb2.SetpointsMsg(
                    pressure_pa=working.pressure_pa,
                    water_level_m=working.water_level_m,
                    steam_temp_k=working.steam_temp_k,
                )
                if working is not None
                else pb2.SetpointsMsg()
            ),
            reset_permitted=self._interlock.emergency_stop.is_active and not blockers,
            reset_blockers=blockers,
            active_conditions=[
                pb2.AlarmConditionMsg(
                    key=condition.key,
                    parameter=condition.rule.parameter,
                    severity=condition.rule.severity.value,
                    direction=condition.rule.direction.value,
                    value=condition.value,
                    threshold=condition.rule.threshold,
                    message=condition.message,
                    since_ms=condition.since_ms,
                )
                for condition in self._alarms.active()
            ],
            loops=[
                pb2.ControlLoopMsg(
                    name=loop.name,
                    setpoint=loop.setpoint,
                    measurement=loop.measurement,
                    output=loop.output,
                    unit=loop.unit,
                )
                for loop in (output.loops if output is not None else ())
            ],
            run_id=self._run_id or 0,
        )

    # ─── Scan ───────────────────────────────────────────────────────────────

    async def _run_control_loop(self) -> None:
        """Scan on every plant state; reconnect with a delay when the stream breaks."""
        while True:
            try:
                async for state in self._physics.stream_system_state():
                    started = time.perf_counter()
                    await self.process_state(state)
                    SCAN_SECONDS.observe(time.perf_counter() - started)
                    self._scans_completed += 1
                    self._last_scanned_step = state.simulation.step_count
                    if self._stream_failing:
                        logger.info("PLC scan stream restored")
                    self._stream_failing = False
                    self._task_error = ""
                raise ConnectionError("physics state stream ended")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._task_error = str(exc) or type(exc).__name__
                if not self._stream_failing:
                    logger.warning(
                        "PLC scan stream failed: %s — retrying every %.2fs",
                        self._task_error,
                        self._retry_delay_s,
                    )
                self._stream_failing = True
                await asyncio.sleep(self._retry_delay_s)

    async def process_state(self, state: pb2.SystemStateMsg) -> None:
        """One PLC scan for one published plant state."""
        measurements = ProcessMeasurements.from_proto(state)
        async with self._lock:
            now = now_ms()
            self._latest_state = state
            dt = self._advance(measurements, now)
            self._latest_measurements = measurements
            arming = self._arming.update(
                measurements.steam_flow_kg_s, measurements.fuel_flow_kg_s, dt or 0.0
            )

            was_latched = self._interlock.emergency_stop.is_active
            self._interlock.check(
                pressure=measurements.pressure_pa,
                water_level=measurements.water_level_m,
                water_temp=measurements.water_temp_k,
                flue_gas_temp=measurements.flue_gas_temp_k,
                dt=dt or 0.0,
                steam_temp=measurements.steam_temp_k,
                arming=arming,
                sensor_qualities=measurements.qualities,
            )
            trigger = self._interlock.emergency_stop.trigger_event
            if (
                not was_latched
                and trigger is not None
                and (self._interlock.emergency_stop.is_active)
            ):
                self._enter_estop(
                    trigger, "safety-interlock", PlcEventKind.INTERLOCK_TRIPPED
                )

            self._evaluate_alarms(measurements, arming, now)

            targets = self._targets(measurements)
            if self._mode is RuntimeMode.ESTOP:
                self._controller.track(measurements, targets, keep_level=True)
                await self._forward(self._trip_command(measurements, dt or 0.0))
                return
            if self._mode is RuntimeMode.MANUAL:
                self._controller.track(measurements, targets)
                return
            if dt is None or dt <= 0.0:
                return
            output = self._controller.scan(measurements, targets, dt)
            await self._forward(
                CommandSnapshot(
                    fuel_valve=output.valves.fuel,
                    feedwater_valve=output.valves.feedwater,
                    steam_valve=output.valves.steam,
                    spray_valve=output.valves.spray,
                    source=pb2.CommandSource.PID,
                    operator_id="plc-auto",
                    timestamp_ms=now,
                )
            )

    def _advance(self, m: ProcessMeasurements, now: int) -> float | None:
        """Scan interval in plant time, or None on the first scan of a plant run."""
        if self._run_id != m.run_id:
            first_run = self._run_id is None
            self._run_id = m.run_id
            self._last_simulation_time_s = m.simulation_time_s
            self._controller.invalidate()
            self._interlock.reset_rate_history()
            self._arming.reset()
            if not first_run:
                for transition in self._alarms.clear_all(now):
                    self._publisher.publish_alarm(transition)
                self._publisher.publish_event(
                    PlcEvent(
                        PlcEventKind.RUN_CHANGED, "physics-engine", {"run_id": m.run_id}
                    )
                )
            if not first_run or self._load_demand_w is None:
                self._load_demand_w = m.electrical_power_w
            return None
        previous = self._last_simulation_time_s
        self._last_simulation_time_s = m.simulation_time_s
        if previous is None:
            return None
        return max(m.simulation_time_s - previous, 0.0)

    def _targets(self, m: ProcessMeasurements) -> ControlTargets:
        return ControlTargets(
            load_w=(
                self._load_demand_w
                if self._load_demand_w is not None
                else m.electrical_power_w
            ),
            pressure_pa=self._setpoints.pressure_pa,
            water_level_m=self._setpoints.water_level_m,
            steam_temp_k=self._setpoints.steam_temp_k,
        )

    def _evaluate_alarms(
        self, m: ProcessMeasurements, arming: ArmingState, now: int
    ) -> None:
        values: dict[str, float] = {}
        for parameter, sensor in ALARMED_VALUES:
            if not m.is_bad(sensor):
                values[parameter] = getattr(m, parameter)
        rate = self._interlock.last_pressure_rate
        if rate is not None and not m.is_bad(SENSOR_DRUM_PRESSURE):
            values["pressure_rate_pa_s"] = rate
        for sensor, quality in m.qualities.items():
            values[f"{sensor}_quality"] = float(quality)
        for transition in self._alarms.evaluate(values, arming, now):
            self._log_alarm(transition)
            self._publisher.publish_alarm(transition)

    def _reset_blockers(self) -> list[str]:
        """
        Why a reset would be refused now.

        Every critical condition blocks, and so does any condition — warnings included —
        on the parameter that caused the trip: a drum that tripped on low level must be
        back above its low-level warning, not just above the trip limit.
        """
        if self._latest_measurements is None:
            return ["no plant state received yet"]
        cause = self._trip_cause.parameter if self._trip_cause is not None else ""
        cause = TRIP_CAUSE_PARAMETERS.get(cause, cause)
        return [
            condition.message
            for condition in self._alarms.active()
            if condition.rule.severity is Severity.CRITICAL
            or condition.rule.parameter == cause
        ]

    # ─── Mode and trip handling ─────────────────────────────────────────────

    def _change_mode(self, mode: RuntimeMode, operator_id: str) -> None:
        if mode is self._mode:
            return
        previous = self._mode
        self._mode = mode
        logger.info("PLC mode %s -> %s by %s", previous.value, mode.value, operator_id)
        self._publisher.publish_event(
            PlcEvent(
                PlcEventKind.MODE_CHANGED,
                operator_id,
                {"from": previous.value, "to": mode.value},
            )
        )

    def _enter_estop(
        self, event: SafetyEvent, operator_id: str, kind: PlcEventKind
    ) -> None:
        self._trip_cause = SafetySnapshot.from_event(event)
        self._manual_command = None
        self._change_mode(RuntimeMode.ESTOP, operator_id)
        logger.error(
            "E-Stop latched: %s=%.4g (limit %.4g) by %s",
            event.parameter,
            event.value,
            event.threshold,
            operator_id,
        )
        self._publisher.publish_event(
            PlcEvent(
                kind,
                operator_id,
                {
                    "parameter": event.parameter,
                    "value": event.value,
                    "threshold": event.threshold,
                },
            )
        )

    def _trip_command(self, m: ProcessMeasurements, dt: float) -> CommandSnapshot:
        """Fuel and spray shut, steam valve by cause, feedwater keeps the drum wet."""
        overrides = trip_overrides(self._interlock.emergency_stop.trigger_event)
        if overrides.feedwater is not None:
            feedwater = overrides.feedwater
        else:
            feedwater = self._controller.hold_level(
                m, self._setpoints.water_level_m, max(dt, 1.0e-3)
            )
        return CommandSnapshot(
            fuel_valve=overrides.fuel,
            feedwater_valve=feedwater,
            steam_valve=overrides.steam,
            spray_valve=overrides.spray,
            source=pb2.CommandSource.SAFETY,
            operator_id="safety-interlock",
            timestamp_ms=now_ms(),
        )

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _reject(self, reason: str) -> ValidationResult:
        self._commands_rejected += 1
        return ValidationResult(accepted=False, reason=reason)

    def _current_spray_command(self) -> float:
        if self._latest_measurements is not None:
            return self._latest_measurements.commands.spray
        return self._latest_command.spray_valve

    async def _forward(self, snapshot: CommandSnapshot) -> ValidationResult:
        """Send a command to PhysicsService, skipping exact repeats."""
        key = (
            round(snapshot.fuel_valve, 4),
            round(snapshot.feedwater_valve, 4),
            round(snapshot.steam_valve, 4),
            round(snapshot.spray_valve, 4),
            int(snapshot.source),
            snapshot.operator_id,
        )
        if self._last_sent_key == key:
            self._latest_command = snapshot
            return ValidationResult(accepted=True)

        ack = await self._physics.apply_command(
            pb2.ControlCommandMsg(
                fuel_valve=snapshot.fuel_valve,
                feedwater_valve=snapshot.feedwater_valve,
                steam_valve=snapshot.steam_valve,
                spray_valve=snapshot.spray_valve,
                timestamp_ms=snapshot.timestamp_ms,
                source=snapshot.source,
                operator_id=snapshot.operator_id,
            )
        )
        if not ack.accepted:
            logger.warning("PhysicsService refused a command: %s", ack.reason)
            return self._reject(ack.reason)

        self._commands_forwarded += 1
        self._latest_command = snapshot
        self._last_sent_key = key
        return ValidationResult(accepted=True)

    @staticmethod
    def _log_alarm(transition: AlarmTransition) -> None:
        condition = transition.condition
        if transition.active:
            logger.warning("Alarm raised: %s", condition.message)
        else:
            logger.info("Alarm cleared: %s", condition.key)

    @staticmethod
    def _mode_to_proto(mode: RuntimeMode) -> int:
        """Map internal runtime mode to protobuf enum."""
        if mode is RuntimeMode.MANUAL:
            return int(pb2.ControlMode.MANUAL)
        if mode is RuntimeMode.ESTOP:
            return int(pb2.ControlMode.ESTOP)
        return int(pb2.ControlMode.AUTO)
