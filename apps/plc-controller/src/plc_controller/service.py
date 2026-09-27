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
import contextlib
import logging
import math
import time
from enum import StrEnum

import cogniboiler_pb2 as pb2
import grpc.aio
from cogniboiler_runtime import now_ms

from plc_controller.alarms import (
    AlarmConditionMonitor,
    AlarmTransition,
    alarm_values,
    blocking_conditions,
)
from plc_controller.client import PhysicsClient
from plc_controller.commands import (
    CommandSnapshot,
    Setpoints,
    ValidationResult,
    check_load_demand,
    check_setpoints,
    check_valves,
)
from plc_controller.control import ControlTargets, UnitController
from plc_controller.events import PlcEvent, PlcEventKind, PlcPublisher
from plc_controller.measurements import ProcessMeasurements, ValveSet
from plc_controller.metrics import SCAN_SECONDS
from plc_controller.safety import (
    ArmingTracker,
    SafetyInterlock,
)
from plc_controller.safety_limits import (
    WATER_LEVEL_LIMITS,
    ArmingState,
    SafetyEvent,
    trip_overrides,
)
from plc_controller.status import SafetySnapshot, control_status

logger = logging.getLogger(__name__)

DEFAULT_CONTROL_INTERVAL_S: float = 0.2
DEFAULT_ALERT_MQTT_HOST: str = "localhost"
DEFAULT_ALERT_MQTT_PORT: int = 1883

# Only people and schedules send commands from outside. PID and SAFETY name the PLC's
# own outputs; accepting them from a caller would let it pass for the interlock.
EXTERNAL_COMMAND_SOURCES: frozenset[int] = frozenset(
    {int(pb2.CommandSource.OPERATOR), int(pb2.CommandSource.SCHEDULER)}
)


# Commands are compared at the precision they are deduplicated at.
COMMAND_TOLERANCE: float = 1.0e-4

# Failures of the plant link, as opposed to faults of the PLC itself.
LINK_ERRORS: tuple[type[Exception], ...] = (grpc.aio.AioRpcError, ConnectionError)
PLANT_DID_NOT_ACKNOWLEDGE: str = "plant did not acknowledge the command"
ESTOP_UNSENT: str = (
    "E-Stop latched; the plant has not yet acknowledged the trip command"
)


def link_error(exc: BaseException) -> str:
    """A one-line reason for a failed call to the plant."""
    if isinstance(exc, grpc.aio.AioRpcError):
        return f"{exc.code().name}: {exc.details()}"
    return str(exc) or type(exc).__name__


def _plant_holds(reported: ValveSet, command: CommandSnapshot) -> bool:
    """Does the plant report this command as the one in force?"""
    pairs = (
        (reported.fuel, command.fuel_valve),
        (reported.feedwater, command.feedwater_valve),
        (reported.steam, command.steam_valve),
        (reported.spray, command.spray_valve),
    )
    return all(abs(held - sent) <= COMMAND_TOLERANCE for held, sent in pairs)


class RuntimeMode(StrEnum):
    """PLC operating mode."""

    AUTO = "auto"
    MANUAL = "manual"
    ESTOP = "estop"


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
        self._forward_failing = False
        self._forward_failures = 0
        self._scan_failures = 0
        self._faults_logged: set[type[Exception]] = set()
        self._holding_non_finite = False

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
            "scan_failures": self._scan_failures,
            "forward_failures": self._forward_failures,
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
        return self._setpoints.copy()

    def update_setpoints(
        self,
        pressure_pa: float,
        water_level_m: float,
        steam_temp_k: float,
        operator_id: str = "",
    ) -> ValidationResult:
        """Validate and store new targets; the working setpoints ramp toward them."""
        refusal = check_setpoints(pressure_pa, water_level_m, steam_temp_k)
        if not refusal.accepted:
            return refusal
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
        refusal = check_load_demand(load_w)
        if not refusal.accepted:
            return refusal
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
        result = check_valves(fuel_valve, feedwater_valve, steam_valve, spray_valve)
        if not result.accepted:
            self._commands_rejected += 1
        return result

    def latest_command(self) -> CommandSnapshot:
        """Return the most recently accepted command."""
        return self._latest_command.copy()

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
            # The mode follows the command only once the plant has it: a command that
            # never arrived must not leave the unit in MANUAL with nobody driving it.
            result = await self._forward(snapshot)
            if result.accepted:
                self._change_mode(RuntimeMode.MANUAL, operator_id)
                self._manual_command = snapshot
            return result

    async def set_mode(self, mode: RuntimeMode, operator_id: str) -> ValidationResult:
        """AUTO or MANUAL on request; ESTOP is a manual trip that latches."""
        operator = operator_id or "unknown"
        async with self._lock:
            if mode is RuntimeMode.ESTOP:
                if not self._interlock.emergency_stop.is_active:
                    event = self._interlock.trip("manual_trip", 1.0, 1.0)
                    self._enter_estop(event, operator, PlcEventKind.MANUAL_TRIP)
                    if self._latest_measurements is not None:
                        sent = await self._forward(
                            self._trip_command(self._latest_measurements, 0.0)
                        )
                        if not sent.accepted:
                            # The latch stands either way; every scan re-sends the trip
                            # command until the plant holds it.
                            logger.warning(
                                "E-Stop by %s is latched, but the plant has not "
                                "acknowledged the trip command: %s",
                                operator,
                                sent.reason,
                            )
                            return ValidationResult(accepted=True, reason=ESTOP_UNSENT)
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
        output = self._controller.last_output
        working = (
            self._controller.working_setpoints() if self._controller.primed else None
        )
        return control_status(
            mode=self._mode_to_proto(self._mode),
            emergency_stop_active=self._interlock.emergency_stop.is_active,
            setpoints=self.get_setpoints(),
            latest_command=self.latest_command(),
            warning_count=self._interlock.warning_count,
            trip_count=self._interlock.trip_count,
            trip_cause=self._trip_cause,
            load_demand_w=self._load_demand_w or 0.0,
            working=working,
            reset_blockers=(
                self._reset_blockers()
                if self._interlock.emergency_stop.is_active
                else []
            ),
            conditions=self._alarms.active(),
            loops=output.loops if output is not None else (),
            run_id=self._run_id or 0,
        )

    # ─── Scan ───────────────────────────────────────────────────────────────

    async def _run_control_loop(self) -> None:
        """Scan on every plant state; reconnect with a delay when the stream breaks."""
        while True:
            # A new stream may reach a restarted plant, whose valves are not ours.
            self._last_sent_key = None
            try:
                stream = self._physics.stream_system_state()
                async with contextlib.aclosing(stream) as states:
                    async for state in states:
                        await self._scan(state)
                raise ConnectionError("physics state stream ended")
            except LINK_ERRORS as exc:
                self._stream_failed(link_error(exc))
            except Exception as exc:
                self._stream_failed(type(exc).__name__)
                self._log_fault_once("PLC scan stream broke", exc)
            await asyncio.sleep(self._retry_delay_s)

    async def _scan(self, state: pb2.SystemStateMsg) -> None:
        """One scan; a fault in it is logged and the stream carries on."""
        started = time.perf_counter()
        try:
            await self.process_state(state)
        except Exception as exc:
            self._scan_failures += 1
            self._task_error = f"scan failed: {type(exc).__name__}"
            self._log_fault_once(
                f"PLC scan failed on plant step {state.simulation.step_count}", exc
            )
            return
        SCAN_SECONDS.observe(time.perf_counter() - started)
        self._scans_completed += 1
        self._last_scanned_step = state.simulation.step_count
        if self._stream_failing:
            logger.info("PLC scan stream restored")
        self._stream_failing = False
        self._task_error = ""

    def _stream_failed(self, reason: str) -> None:
        self._task_error = reason
        if not self._stream_failing:
            logger.warning(
                "PLC scan stream failed: %s — retrying every %.2fs",
                reason,
                self._retry_delay_s,
            )
        self._stream_failing = True

    def _log_fault_once(self, message: str, exc: Exception) -> None:
        """A fault of the PLC itself: an error with its traceback, once per kind."""
        kind = type(exc)
        if kind in self._faults_logged:
            logger.debug("%s: %s (logged before)", message, kind.__name__)
            return
        self._faults_logged.add(kind)
        logger.error("%s", message, exc_info=exc)

    async def process_state(self, state: pb2.SystemStateMsg) -> None:
        """One PLC scan for one published plant state."""
        measurements = ProcessMeasurements.from_proto(state)
        async with self._lock:
            now = now_ms()
            self._note_non_finite(measurements)
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
                command = self._trip_command(measurements, dt or 0.0)
                if not _plant_holds(measurements.commands, command):
                    self._last_sent_key = None
                await self._forward(command)
                return
            if self._mode is RuntimeMode.MANUAL:
                self._controller.track(measurements, targets)
                return
            if dt is None or dt <= 0.0 or not measurements.finite:
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
        time_s = m.simulation_time_s if math.isfinite(m.simulation_time_s) else None
        if self._run_id != m.run_id:
            first_run = self._run_id is None
            self._run_id = m.run_id
            self._last_simulation_time_s = time_s
            # A new run starts from the scenario's own valves, not from ours.
            self._last_sent_key = None
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
            seed = not first_run or self._load_demand_w is None
            if seed and math.isfinite(m.electrical_power_w):
                self._load_demand_w = m.electrical_power_w
            return None
        if time_s is None:
            return None
        previous = self._last_simulation_time_s
        self._last_simulation_time_s = time_s
        if previous is None:
            return None
        return max(time_s - previous, 0.0)

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
        values = alarm_values(m, self._interlock.last_pressure_rate)
        for transition in self._alarms.evaluate(values, arming, now):
            self._log_alarm(transition)
            self._publisher.publish_alarm(transition)

    def _reset_blockers(self) -> list[str]:
        """Why a reset would be refused now; empty means the cause has cleared."""
        if self._latest_measurements is None:
            return ["no plant state received yet"]
        cause = self._trip_cause.parameter if self._trip_cause is not None else ""
        return blocking_conditions(self._alarms.active(), cause)

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
        logger.warning(
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
        elif m.finite:
            feedwater = self._controller.hold_level(
                m, self._setpoints.water_level_m, max(dt, 1.0e-3)
            )
        else:
            feedwater = self._latest_command.feedwater_valve
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
        measurements = self._latest_measurements
        if measurements is not None and math.isfinite(measurements.commands.spray):
            return measurements.commands.spray
        return self._latest_command.spray_valve

    def _note_non_finite(self, m: ProcessMeasurements) -> None:
        """Say once, when it starts, that a reading is not a number."""
        finite = m.finite
        if not finite and not self._holding_non_finite:
            logger.warning(
                "A plant reading is not a number: its instrument counts as failed and "
                "the control loops hold their valves until every reading is a number"
            )
        elif finite and self._holding_non_finite:
            logger.info("Every plant reading is a number again")
        self._holding_non_finite = not finite

    def _forward_failed(self, snapshot: CommandSnapshot, exc: Exception) -> None:
        """A person's command is reported every time; the PLC's own once per outage."""
        self._forward_failures += 1
        reason = link_error(exc)
        if int(snapshot.source) in EXTERNAL_COMMAND_SOURCES:
            logger.warning(
                "Command from %s did not reach the plant: %s",
                snapshot.operator_id,
                reason,
            )
        elif not self._forward_failing:
            logger.warning(
                "PhysicsService did not acknowledge the PLC's command: %s — "
                "the next scan sends it again",
                reason,
            )
        self._forward_failing = True

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

        try:
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
        except LINK_ERRORS as exc:
            self._forward_failed(snapshot, exc)
            return ValidationResult(accepted=False, reason=PLANT_DID_NOT_ACKNOWLEDGE)
        if self._forward_failing:
            logger.info("PhysicsService acknowledges commands again")
            self._forward_failing = False
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
