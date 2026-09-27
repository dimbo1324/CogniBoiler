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
import math
import time

import cogniboiler_pb2 as pb2
from cogniboiler_runtime import now_ms

from plc_controller.alarms import (
    AlarmConditionMonitor,
    AlarmTransition,
    alarm_values,
    blocking_conditions,
)
from plc_controller.client import PhysicsClient
from plc_controller.commands import (
    EXTERNAL_COMMAND_SOURCES,
    OPERATOR_AUTO,
    OPERATOR_INTERLOCK,
    OPERATOR_PHYSICS,
    OPERATOR_UNATTRIBUTED,
    CommandSnapshot,
    Setpoints,
    ValidationResult,
    check_operator,
    check_valves,
    refuse,
)
from plc_controller.control import UnitController
from plc_controller.events import (
    DEFAULT_MQTT_HOST,
    DEFAULT_MQTT_PORT,
    PlcEvent,
    PlcEventKind,
    PlcPublisher,
)
from plc_controller.forwarding import CommandForwarder, link_error, plant_holds
from plc_controller.measurements import ProcessMeasurements
from plc_controller.modes import RuntimeMode, to_proto
from plc_controller.runs import PlantRun
from plc_controller.safety import (
    ArmingTracker,
    SafetyInterlock,
)
from plc_controller.safety_limits import (
    WATER_LEVEL_LIMITS,
    ArmingState,
    SafetyEvent,
    SafetyLevel,
    trip_overrides,
)
from plc_controller.scan_loop import ScanLoop
from plc_controller.status import SafetySnapshot, control_status
from plc_controller.targets import OperatorTargets

logger = logging.getLogger(__name__)

__all__ = ["PLCService", "RuntimeMode"]

# The scan is event-driven (one per plant state); this is only the pause before a
# broken state stream is opened again.
STREAM_RETRY_DELAY_S: float = 0.2

NO_PLANT_STATE: str = "no plant state received yet"
ESTOP_UNSENT: str = (
    "E-Stop latched; the plant has not yet acknowledged the trip command"
)


class PLCService:
    """Virtual PLC business logic with an event-driven scan loop."""

    VERSION = "0.3.0"

    def __init__(
        self,
        physics_client: PhysicsClient | None = None,
        *,
        retry_delay_s: float = STREAM_RETRY_DELAY_S,
        mqtt_host: str = DEFAULT_MQTT_HOST,
        mqtt_port: int = DEFAULT_MQTT_PORT,
        enable_control_loop: bool = True,
        enable_alert_publishing: bool = True,
        mqtt_username: str | None = None,
        mqtt_password: str | None = None,
    ) -> None:
        self._physics = physics_client or PhysicsClient()
        self._enable_control_loop = enable_control_loop
        self._forwarder = CommandForwarder(self._physics)
        self._loop = ScanLoop(
            self._physics,
            self.process_state,
            retry_delay_s=max(retry_delay_s, 0.05),
            on_new_stream=self._forwarder.invalidate,
        )

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
        self._targets = OperatorTargets(
            lambda event: self._publisher.publish_event(event)
        )
        self._run = PlantRun()

        self._lock = asyncio.Lock()
        self._start_time = time.monotonic()
        self._commands_received = 0
        self._commands_rejected = 0
        self._latest_measurements: ProcessMeasurements | None = None
        self._trip_cause: SafetySnapshot | None = None

    # ─── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start publishing and, if enabled, the scan loop."""
        self._publisher.start()
        if self._enable_control_loop:
            self._loop.start()

    async def close(self) -> None:
        """Stop background work and close remote connections."""
        await self._loop.close()
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
            "commands_rejected": self._commands_rejected + self._forwarder.refused,
            "commands_forwarded": self._forwarder.forwarded,
            "warnings": self._interlock.warning_count,
            "trips": self._interlock.trip_count,
            "scans": self._loop.scans,
            "scan_failures": self._loop.scan_failures,
            "stream_failures": self._loop.stream_failures,
            "forward_failures": self._forwarder.failures,
        }

    @property
    def publisher_health(self) -> tuple[int, bool, int]:
        """Messages lost to a full queue, broker connected, broker failures."""
        publisher = self._publisher
        return publisher.dropped, publisher.connected, publisher.failures

    @property
    def plant_link_up(self) -> bool:
        """True while the plant's state stream is open and delivering."""
        return self._loop.link_up

    @property
    def active_condition_count(self) -> int:
        return len(self._alarms.active())

    @property
    def last_scanned_step(self) -> int:
        """Plant step count of the latest state the scan loop finished; -1 before any."""
        return self._loop.last_scanned_step

    async def physics_status(self) -> str:
        """Overall PLC status: degraded while tripped or cut off from the plant."""
        if self._loop.error or self._mode is RuntimeMode.ESTOP:
            return "degraded"
        try:
            health = await self._physics.health()
        except Exception as exc:
            logger.debug("PhysicsService health check failed: %s", link_error(exc))
            return "degraded"
        return str(health.status)

    # ─── Targets ────────────────────────────────────────────────────────────

    def get_setpoints(self) -> Setpoints:
        """Return current setpoints (copy)."""
        return self._targets.setpoints

    def update_setpoints(
        self,
        pressure_pa: float,
        water_level_m: float,
        steam_temp_k: float,
        operator_id: str,
    ) -> ValidationResult:
        """Validate and store new targets; the working setpoints ramp toward them."""
        return self._targets.update_setpoints(
            pressure_pa, water_level_m, steam_temp_k, operator_id
        )

    @property
    def load_demand_w(self) -> float | None:
        """Operator's load target; None until the PLC has seen the plant."""
        return self._targets.load_demand_w

    def set_load_demand(self, load_w: float, operator_id: str) -> ValidationResult:
        """Set the electrical load target; the load setpoint ramps toward it."""
        return self._targets.set_load_demand(load_w, operator_id)

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
            return self._reject(result.reason)
        return result

    def latest_command(self) -> CommandSnapshot:
        """Return the command most recently in force."""
        return self._forwarder.latest

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
            # A caller claiming the PID or SAFETY source tries to pass for the PLC.
            logger.warning(
                "Command from %r claims source %s, which is reserved for the PLC",
                operator_id,
                pb2.CommandSource.Name(source),
            )
            self._commands_rejected += 1
            return ValidationResult(
                accepted=False,
                reason=(
                    f"command source {pb2.CommandSource.Name(source)} "
                    "is reserved for the PLC"
                ),
            )
        operator = operator_id.strip()
        attributed = check_operator(operator)
        if not attributed.accepted:
            return self._reject(attributed.reason)

        async with self._lock:
            if self._interlock.emergency_stop.is_active:
                return self._reject(
                    "Emergency stop is active. Reset required before commands."
                )
            measurements = self._latest_measurements
            if measurements is None:
                # Without a plant state the permissives cannot be judged.
                return self._reject(NO_PLANT_STATE)
            if fuel_valve > 0.0 and not self._interlock.fuel_permitted(
                measurements.water_level_m
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
                operator_id=operator,
                timestamp_ms=now_ms(),
            )
            # The mode follows the command only once the plant has it: a command that
            # never arrived must not leave the unit in MANUAL with nobody driving it.
            result = await self._forwarder.forward(snapshot)
            if result.accepted:
                self._change_mode(RuntimeMode.MANUAL, operator)
            return result

    async def set_mode(self, mode: RuntimeMode, operator_id: str) -> ValidationResult:
        """AUTO or MANUAL on request; ESTOP is a manual trip that latches."""
        operator = operator_id.strip()
        attributed = check_operator(operator)
        async with self._lock:
            if mode is RuntimeMode.ESTOP:
                if not attributed.accepted:
                    # A trip request is honoured whoever sends it; only its record
                    # suffers from the missing name.
                    logger.warning("E-Stop requested without a valid operator_id")
                    operator = OPERATOR_UNATTRIBUTED
                if not self._interlock.emergency_stop.is_active:
                    event = self._interlock.trip("manual_trip", 1.0, 1.0)
                    self._enter_estop(event, operator, PlcEventKind.MANUAL_TRIP)
                    if self._latest_measurements is not None:
                        sent = await self._forwarder.forward(
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

            if not attributed.accepted:
                return self._reject(attributed.reason)
            if self._interlock.emergency_stop.is_active:
                return self._reject("Emergency stop is active. Reset it first.")
            if mode is RuntimeMode.AUTO:
                self._controller.invalidate()
            self._change_mode(mode, operator)
            return ValidationResult(accepted=True)

    async def reset_emergency_stop(self, operator_id: str) -> ValidationResult:
        """Clear the E-Stop latch once its cause is gone and return to AUTO."""
        operator = operator_id.strip()
        attributed = check_operator(operator)
        if not attributed.accepted:
            return refuse(attributed.reason)
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
                return refuse("Reset refused: " + "; ".join(blockers))

            cause = self._trip_cause
            self._interlock.reset(operator_id=operator)
            self._controller.invalidate()
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

    async def get_control_status(self) -> pb2.PLCStatusMsg:
        """The PLC's mode, targets, working setpoints, loops and alarm conditions."""
        output = self._controller.last_output
        working = (
            self._controller.working_setpoints() if self._controller.primed else None
        )
        return control_status(
            mode=to_proto(self._mode),
            emergency_stop_active=self._interlock.emergency_stop.is_active,
            setpoints=self.get_setpoints(),
            latest_command=self.latest_command(),
            warning_count=self._interlock.warning_count,
            trip_count=self._interlock.trip_count,
            trip_cause=self._trip_cause,
            load_demand_w=self._targets.load_demand_w or 0.0,
            working=working,
            reset_blockers=(
                self._reset_blockers()
                if self._interlock.emergency_stop.is_active
                else []
            ),
            conditions=self._alarms.active(),
            loops=output.loops if output is not None else (),
            run_id=self._run.run_id or 0,
        )

    # ─── Scan ───────────────────────────────────────────────────────────────

    async def process_state(self, state: pb2.SystemStateMsg) -> None:
        """One PLC scan for one published plant state."""
        measurements = ProcessMeasurements.from_proto(state)
        async with self._lock:
            now = now_ms()
            dt = self._advance(measurements, now)
            self._latest_measurements = measurements
            arming = self._arming.update(
                measurements.steam_flow_kg_s, measurements.fuel_flow_kg_s, dt or 0.0
            )

            was_latched = self._interlock.emergency_stop.is_active
            status = self._interlock.check(
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
                and status.level is SafetyLevel.TRIP
                and trigger is not None
            ):
                self._enter_estop(
                    trigger, OPERATOR_INTERLOCK, PlcEventKind.INTERLOCK_TRIPPED
                )

            self._evaluate_alarms(measurements, arming, now)

            targets = self._targets.for_scan(measurements)
            if self._mode is RuntimeMode.ESTOP:
                self._controller.track(measurements, targets, keep_level=True)
                command = self._trip_command(measurements, dt or 0.0)
                if not plant_holds(measurements.commands, command):
                    self._forwarder.invalidate()
                await self._forwarder.forward(command)
                return
            if self._mode is RuntimeMode.MANUAL:
                self._controller.track(measurements, targets)
                return
            if dt is None or dt <= 0.0 or not measurements.finite:
                return
            output = self._controller.scan(measurements, targets, dt)
            await self._forwarder.forward(
                CommandSnapshot(
                    fuel_valve=output.valves.fuel,
                    feedwater_valve=output.valves.feedwater,
                    steam_valve=output.valves.steam,
                    spray_valve=output.valves.spray,
                    source=pb2.CommandSource.PID,
                    operator_id=OPERATOR_AUTO,
                    timestamp_ms=now,
                )
            )

    def _advance(self, m: ProcessMeasurements, now: int) -> float | None:
        """Scan interval in plant time, or None on the first scan of a plant run."""
        step = self._run.advance(m)
        if step.new_run:
            # A new run starts from the scenario's own valves, not from ours.
            self._forwarder.invalidate()
            self._controller.invalidate()
            self._interlock.reset_rate_history()
            self._arming.reset()
            if not step.first_run:
                for transition in self._alarms.clear_all(now):
                    self._publisher.publish_alarm(transition)
                self._publisher.publish_event(
                    PlcEvent(
                        PlcEventKind.RUN_CHANGED, OPERATOR_PHYSICS, {"run_id": m.run_id}
                    )
                )
            self._targets.seed_load_demand(
                m.electrical_power_w, keep_existing=step.first_run
            )
        return step.interval_s

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
            return [NO_PLANT_STATE]
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
                m, self._targets.setpoints.water_level_m, dt
            )
        else:
            feedwater = self._forwarder.latest.feedwater_valve
        return CommandSnapshot(
            fuel_valve=overrides.fuel,
            feedwater_valve=feedwater,
            steam_valve=overrides.steam,
            spray_valve=overrides.spray,
            source=pb2.CommandSource.SAFETY,
            operator_id=OPERATOR_INTERLOCK,
            timestamp_ms=now_ms(),
        )

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _reject(self, reason: str) -> ValidationResult:
        """Refuse a command or mode change; it counts as a rejected command."""
        self._commands_rejected += 1
        return refuse(reason)

    def _current_spray_command(self) -> float:
        measurements = self._latest_measurements
        if measurements is not None and math.isfinite(measurements.commands.spray):
            return measurements.commands.spray
        return self._forwarder.latest.spray_valve

    @staticmethod
    def _log_alarm(transition: AlarmTransition) -> None:
        condition = transition.condition
        if transition.active:
            logger.warning("Alarm raised: %s", condition.message)
        else:
            logger.info("Alarm cleared: %s", condition.key)
