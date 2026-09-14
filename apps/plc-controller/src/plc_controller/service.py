"""
PLC business logic and live control runtime.

The PLC owns:
  - operator/manual commands
  - PID setpoints
  - automatic closed-loop control
  - safety interlocks and E-Stop latching
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

import cogniboiler_pb2 as pb2
from aiomqtt import Client, MqttError, Will
from physics_engine.boiler import BoilerModel
from physics_engine.combustion import CombustionModel
from physics_engine.controller import BoilerController, BoilerSetpoints
from physics_engine.models import BoilerParameters
from physics_engine.safety import SafetyEvent, SafetyInterlock, SafetyLevel

from plc_controller.client import PhysicsClient

logger = logging.getLogger(__name__)

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

DEFAULT_CONTROL_INTERVAL_S: float = 0.2
DEFAULT_ALERT_MQTT_HOST: str = "localhost"
DEFAULT_ALERT_MQTT_PORT: int = 1883
TOPIC_ALERT_WARNING: str = "alerts/warning"
TOPIC_ALERT_CRITICAL: str = "alerts/critical"
TOPIC_AVAILABILITY: str = "status/plc-controller"


def _now_ms() -> int:
    """Return current UTC epoch milliseconds."""
    return int(time.time() * 1000)


class RuntimeMode(StrEnum):
    """Internal PLC runtime mode."""

    AUTO = "auto"
    MANUAL = "manual"
    ESTOP = "estop"


@dataclass
class Setpoints:
    """Current PLC setpoints for all control loops."""

    pressure_pa: float = 140.0e5  # 140 bar nominal
    water_level_m: float = 4.8
    steam_temp_k: float = 811.0  # ~538 C nominal
    updated_at_ms: int = field(default_factory=_now_ms)


@dataclass
class ValidationResult:
    """Result of a command or setpoint validation check."""

    accepted: bool
    reason: str = ""


@dataclass
class CommandSnapshot:
    """Latest accepted PLC command."""

    fuel_valve: float = 0.5
    feedwater_valve: float = 0.5
    steam_valve: float = 0.5
    source: int = pb2.CommandSource.PID
    operator_id: str = ""
    timestamp_ms: int = field(default_factory=_now_ms)


@dataclass
class SafetySnapshot:
    """Most recent safety event exposed by the PLC runtime."""

    timestamp_ms: int
    parameter: str
    value: float
    threshold: float
    level: str
    action: str


class PLCService:
    """
    Virtual PLC business logic with a live closed-loop control task.

    The runtime polls PhysicsService state, computes PID outputs in AUTO,
    enforces safety overrides, and forwards validated commands back to the
    physics engine.
    """

    VERSION = "0.2.0"

    def __init__(
        self,
        physics_client: PhysicsClient | None = None,
        *,
        control_interval_s: float = DEFAULT_CONTROL_INTERVAL_S,
        mqtt_host: str = DEFAULT_ALERT_MQTT_HOST,
        mqtt_port: int = DEFAULT_ALERT_MQTT_PORT,
        enable_control_loop: bool = True,
        enable_alert_publishing: bool = True,
    ) -> None:
        self._setpoints = Setpoints()
        self._start_time = time.monotonic()
        self._commands_received: int = 0
        self._commands_rejected: int = 0
        self._commands_forwarded: int = 0
        self._latest_command = CommandSnapshot()
        self._latest_process_state: pb2.SystemStateMsg | None = None
        self._latest_safety_event: SafetySnapshot | None = None
        self._physics = physics_client or PhysicsClient()

        self._control_interval_s = max(control_interval_s, 0.05)
        self._mqtt_host = mqtt_host
        self._mqtt_port = mqtt_port
        self._enable_control_loop = enable_control_loop
        self._enable_alert_publishing = enable_alert_publishing

        self._control_mode = RuntimeMode.AUTO
        self._controller = BoilerController()
        self._interlock = SafetyInterlock()
        self._boiler_model = BoilerModel(BoilerParameters())
        self._combustion = CombustionModel(
            max_fuel_flow=self._boiler_model.params.max_fuel_flow
        )
        self._control_task: asyncio.Task[None] | None = None
        self._task_error: str = ""
        self._last_alarm_signature: tuple[str, str, float] | None = None
        self._last_sent_key: tuple[float, float, float, int, str] | None = None
        self._last_simulation_time_s: float | None = None
        self._controller_primed = False

    # ─── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the closed-loop runtime if it is enabled."""
        if not self._enable_control_loop:
            return
        if self._control_task is not None and not self._control_task.done():
            return
        self._control_task = asyncio.create_task(
            self._run_control_loop(),
            name="plc-control-loop",
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
        await self._physics.close()

    # ─── Health ─────────────────────────────────────────────────────────────

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def stats(self) -> dict[str, int]:
        return {
            "commands_received": self._commands_received,
            "commands_rejected": self._commands_rejected,
            "commands_forwarded": self._commands_forwarded,
            "warnings": self._interlock.warning_count,
            "trips": self._interlock.trip_count,
        }

    async def physics_status(self) -> str:
        """Return current overall PLC status."""
        if self._task_error:
            return "degraded"
        if self._control_mode == RuntimeMode.ESTOP:
            return "degraded"
        try:
            health = await self._physics.health()
        except Exception:
            return "degraded"
        return str(health.status)

    # ─── Setpoints ──────────────────────────────────────────────────────────

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
        """Validate and apply new setpoints, returning PLC to AUTO."""
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
            updated_at_ms=_now_ms(),
        )
        if self._control_mode != RuntimeMode.ESTOP:
            self._control_mode = RuntimeMode.AUTO
        return ValidationResult(accepted=True)

    # ─── Commands ───────────────────────────────────────────────────────────

    def validate_command(
        self,
        fuel_valve: float,
        feedwater_valve: float,
        steam_valve: float,
    ) -> ValidationResult:
        """Validate a control command against hard valve bounds."""
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

    def latest_command(self) -> CommandSnapshot:
        """Return the most recently accepted command."""
        return CommandSnapshot(
            fuel_valve=self._latest_command.fuel_valve,
            feedwater_valve=self._latest_command.feedwater_valve,
            steam_valve=self._latest_command.steam_valve,
            source=self._latest_command.source,
            operator_id=self._latest_command.operator_id,
            timestamp_ms=self._latest_command.timestamp_ms,
        )

    async def get_process_state(self) -> pb2.SystemStateMsg:
        """Fetch the current process state from the live PhysicsService."""
        state = await self._physics.get_system_state()
        self._latest_process_state = state
        return state

    async def get_control_status(self) -> pb2.PLCStatusMsg:
        """Return the current PLC runtime status."""
        active_trip = (
            pb2.SafetyEventMsg(
                timestamp_ms=self._latest_safety_event.timestamp_ms,
                parameter=self._latest_safety_event.parameter,
                value=self._latest_safety_event.value,
                threshold=self._latest_safety_event.threshold,
                level=self._latest_safety_event.level,
                action=self._latest_safety_event.action,
            )
            if self._latest_safety_event is not None
            else pb2.SafetyEventMsg()
        )
        latest = self.latest_command()
        setpoints = self.get_setpoints()
        return pb2.PLCStatusMsg(
            mode=self._mode_to_proto(self._control_mode),
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
                timestamp_ms=latest.timestamp_ms,
                source=latest.source,
                operator_id=latest.operator_id,
            ),
            warning_count=self._interlock.warning_count,
            trip_count=self._interlock.trip_count,
            active_trip=active_trip,
        )

    async def reset_emergency_stop(self, operator_id: str) -> ValidationResult:
        """Reset the safety latch and return the PLC to AUTO mode."""
        self._interlock.reset(operator_id=operator_id)
        self._controller.reset()
        self._controller_primed = False
        self._control_mode = RuntimeMode.AUTO
        self._latest_safety_event = None
        self._last_alarm_signature = None
        return ValidationResult(accepted=True)

    async def send_command(
        self,
        *,
        fuel_valve: float,
        feedwater_valve: float,
        steam_valve: float,
        source: int,
        operator_id: str,
    ) -> ValidationResult:
        """Validate, store mode intent, and forward a command to PhysicsService."""
        result = self.validate_command(
            fuel_valve=fuel_valve,
            feedwater_valve=feedwater_valve,
            steam_valve=steam_valve,
        )
        if not result.accepted:
            return result

        if (
            self._interlock.emergency_stop.is_active
            and source != pb2.CommandSource.SAFETY
        ):
            self._commands_rejected += 1
            return ValidationResult(
                accepted=False,
                reason="Emergency stop is active. Reset required before commands.",
            )

        snapshot = CommandSnapshot(
            fuel_valve=fuel_valve,
            feedwater_valve=feedwater_valve,
            steam_valve=steam_valve,
            source=source,
            operator_id=operator_id,
            timestamp_ms=_now_ms(),
        )

        if source in (pb2.CommandSource.OPERATOR, pb2.CommandSource.SCHEDULER):
            self._control_mode = RuntimeMode.MANUAL

        return await self._forward_command(snapshot)

    # ─── Closed-loop runtime ────────────────────────────────────────────────

    async def _run_control_loop(self) -> None:
        """Poll PhysicsService, compute control action, and enforce safety."""
        if self._enable_alert_publishing:
            await self._publish_availability("online")

        try:
            while True:
                try:
                    await self._control_step()
                    self._task_error = ""
                except Exception as exc:
                    self._task_error = str(exc)
                    logger.warning("PLC control loop step failed: %s", exc)
                    await asyncio.sleep(self._control_interval_s)
                    continue

                await asyncio.sleep(self._control_interval_s)
        except asyncio.CancelledError:
            if self._enable_alert_publishing:
                try:
                    await self._publish_availability("offline")
                except Exception:
                    logger.debug("PLC availability publish failed during shutdown.")
            raise

    async def _control_step(self) -> None:
        """Execute one PLC scan cycle."""
        state = await self.get_process_state()
        process_dt = self._process_dt(state.simulation_time_s)

        safety_status = self._interlock.check(
            pressure=state.boiler.pressure_pa,
            water_level=state.boiler.water_level_m,
            water_temp=state.boiler.water_temp_k,
            flue_gas_temp=state.boiler.flue_gas_temp_k,
            dt=process_dt,
        )
        await self._handle_safety_events(safety_status.events)

        if self._interlock.emergency_stop.is_active:
            self._control_mode = RuntimeMode.ESTOP
            snapshot = CommandSnapshot(
                fuel_valve=safety_status.fuel_valve_override or 0.0,
                feedwater_valve=state.actuators.feedwater_valve_command,
                steam_valve=safety_status.steam_valve_override or 1.0,
                source=pb2.CommandSource.SAFETY,
                operator_id="safety-interlock",
                timestamp_ms=_now_ms(),
            )
            self._latest_command = snapshot
            await self._forward_command(snapshot)
            return

        if self._control_mode == RuntimeMode.MANUAL:
            return

        self._control_mode = RuntimeMode.AUTO

        current_steam_temp = (
            state.turbine.steam_temp_in_k
            if state.turbine.steam_temp_in_k > 0.0
            else state.boiler.water_temp_k
        )
        fuel_flow = self._combustion.calculate(
            fuel_valve=state.actuators.fuel_valve_position
        ).fuel_flow
        feedwater_flow = self._boiler_model._feedwater_flow(
            state.actuators.feedwater_valve_position
        )

        if not self._controller_primed:
            self._prime_controller(
                fuel_flow=fuel_flow,
                feedwater_flow=feedwater_flow,
                steam_valve_command=state.actuators.steam_valve_command,
            )

        output = self._controller.step(
            setpoints=BoilerSetpoints(
                pressure=self._setpoints.pressure_pa,
                water_level=self._setpoints.water_level_m,
                steam_temp=self._setpoints.steam_temp_k,
            ),
            pressure=state.boiler.pressure_pa,
            water_level=state.boiler.water_level_m,
            steam_temp=current_steam_temp,
            fuel_flow=fuel_flow,
            feedwater_flow=feedwater_flow,
            dt=process_dt,
        )

        # The current plant model uses steam valve position as the primary load
        # throttle. Driving it from the temperature loop destabilizes pressure
        # control, so AUTO currently stabilizes pressure/level while holding the
        # existing process load until a dedicated temperature actuator exists.
        steam_valve_command = state.actuators.steam_valve_command
        snapshot = CommandSnapshot(
            fuel_valve=output.fuel_valve,
            feedwater_valve=output.feedwater_valve,
            steam_valve=steam_valve_command,
            source=pb2.CommandSource.PID,
            operator_id="plc-auto",
            timestamp_ms=_now_ms(),
        )
        await self._forward_command(snapshot)

    async def _forward_command(self, snapshot: CommandSnapshot) -> ValidationResult:
        """Send a command to PhysicsService, deduplicating identical repeats."""
        key = (
            round(snapshot.fuel_valve, 4),
            round(snapshot.feedwater_valve, 4),
            round(snapshot.steam_valve, 4),
            snapshot.source,
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
                timestamp_ms=snapshot.timestamp_ms,
                source=snapshot.source,
                operator_id=snapshot.operator_id,
            )
        )
        if not ack.accepted:
            self._commands_rejected += 1
            return ValidationResult(accepted=False, reason=ack.reason)

        self._commands_forwarded += 1
        self._latest_command = snapshot
        self._last_sent_key = key
        return ValidationResult(accepted=True)

    async def _handle_safety_events(self, events: list[SafetyEvent]) -> None:
        """Record and publish new safety events without flooding duplicates."""
        if not events:
            self._last_alarm_signature = None
            return

        primary = next(
            (event for event in events if event.level == SafetyLevel.TRIP), None
        )
        if primary is None:
            primary = events[0]

        signature = (primary.parameter, primary.level.value, primary.threshold)
        if signature == self._last_alarm_signature:
            return

        self._last_alarm_signature = signature
        self._latest_safety_event = SafetySnapshot(
            timestamp_ms=primary.timestamp_ms,
            parameter=primary.parameter,
            value=primary.value,
            threshold=primary.threshold,
            level=primary.level.value,
            action=primary.action.value,
        )

        if self._enable_alert_publishing:
            await self._publish_alarm(primary)

    async def _publish_alarm(self, event: SafetyEvent) -> None:
        """Publish one alarm event to MQTT for AlertManager ingestion."""
        severity_topic = (
            TOPIC_ALERT_CRITICAL
            if event.level == SafetyLevel.TRIP
            else TOPIC_ALERT_WARNING
        )
        payload = json.dumps(
            {
                "alarm_id": f"{event.parameter}:{event.timestamp_ms}",
                "source_service": "plc-controller",
                "severity": "critical"
                if event.level == SafetyLevel.TRIP
                else "warning",
                "parameter": event.parameter,
                "value": event.value,
                "threshold": event.threshold,
                "action": event.action.value,
                "message": (
                    f"{event.parameter} crossed threshold {event.threshold:.3f} "
                    f"with value {event.value:.3f}"
                ),
                "timestamp_ms": event.timestamp_ms,
            }
        ).encode("utf-8")

        try:
            async with Client(
                hostname=self._mqtt_host,
                port=self._mqtt_port,
                identifier="plc-alert-publisher",
            ) as client:
                await client.publish(severity_topic, payload, qos=1)
        except MqttError as exc:
            logger.warning("Alarm publish failed: %s", exc)

    async def _publish_availability(self, status: str) -> None:
        """Publish retained PLC availability status for operators."""
        try:
            async with Client(
                hostname=self._mqtt_host,
                port=self._mqtt_port,
                identifier="plc-availability-publisher",
                will=Will(TOPIC_AVAILABILITY, payload="offline", qos=1, retain=True),
            ) as client:
                await client.publish(TOPIC_AVAILABILITY, status, qos=1, retain=True)
        except MqttError as exc:
            logger.warning("Availability publish failed: %s", exc)

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _mode_to_proto(self, mode: RuntimeMode) -> int:
        """Map internal runtime mode to protobuf enum."""
        if mode == RuntimeMode.MANUAL:
            return int(pb2.ControlMode.MANUAL)
        if mode == RuntimeMode.ESTOP:
            return int(pb2.ControlMode.ESTOP)
        return int(pb2.ControlMode.AUTO)

    def _process_dt(self, simulation_time_s: float) -> float:
        """Return the elapsed process time between PLC scans."""
        if self._last_simulation_time_s is None:
            self._last_simulation_time_s = simulation_time_s
            return 1.0
        dt = max(simulation_time_s - self._last_simulation_time_s, 1.0e-6)
        self._last_simulation_time_s = simulation_time_s
        return dt

    def _prime_controller(
        self,
        *,
        fuel_flow: float,
        feedwater_flow: float,
        steam_valve_command: float,
    ) -> None:
        """Preload PID states from the current operating point."""
        self._controller.pressure_loop.master.reset(initial_output=fuel_flow)
        self._controller.pressure_loop.slave.reset(
            initial_output=self._latest_process_state.actuators.fuel_valve_command
            if self._latest_process_state is not None
            else 0.5
        )
        self._controller.level_loop.master.reset(initial_output=feedwater_flow)
        self._controller.level_loop.slave.reset(
            initial_output=self._latest_process_state.actuators.feedwater_valve_command
            if self._latest_process_state is not None
            else 0.5
        )
        self._controller.temp_loop.reset(initial_output=steam_valve_command)
        self._controller_primed = True
