"""Stand-ins for the gateway's upstream clients; each records what the routes sent."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from functools import cache
from typing import Any

import cogniboiler_pb2 as pb2
import grpc
import grpc.aio
from api_gateway.auth.password import hash_password
from api_gateway.clients import HistorySource
from api_gateway.models.user import Role, User, UserRole
from sqlalchemy.ext.asyncio import AsyncSession

TIMESTAMP_MS = 1_710_000_000_000

# Username, password and role of every seeded account.
SEEDED_USERS: tuple[tuple[str, str, str], ...] = (
    ("testuser", "test_password", "operator"),
    ("viewer1", "viewer_password", "viewer"),
    ("operator1", "operator_password", "operator"),
    ("engineer1", "engineer_password", "engineer"),
    ("admin1", "admin_password", "admin"),
)


@cache
def _hashed(password: str) -> str:
    """Argon2 is slow on purpose; one hash per seeded password is enough per run."""
    return hash_password(password)


async def seed_accounts(session: AsyncSession) -> None:
    """The four roles and one account per entry of SEEDED_USERS. Commits."""
    roles = {
        name: Role(name=name, description=name)
        for name in ("viewer", "operator", "engineer", "admin")
    }
    session.add_all(roles.values())
    await session.flush()
    users = {
        username: User(
            username=username,
            hashed_password=_hashed(password),
            is_active=True,
            created_at_ms=1,
        )
        for username, password, _ in SEEDED_USERS
    }
    session.add_all(users.values())
    await session.flush()
    session.add_all(
        UserRole(user_id=users[username].id, role_id=roles[role].id, granted_at_ms=1)
        for username, _, role in SEEDED_USERS
    )
    await session.commit()


class UpstreamDownError(grpc.RpcError):
    """What a stub raises when its service cannot be reached."""

    def code(self) -> grpc.StatusCode:
        return grpc.StatusCode.UNAVAILABLE

    def details(self) -> str:
        return "connection refused"


def not_found() -> grpc.aio.AioRpcError:
    return grpc.aio.AioRpcError(
        grpc.StatusCode.NOT_FOUND,
        grpc.aio.Metadata(),
        grpc.aio.Metadata(),
        details="no such alarm",
    )


def simulation_status(**overrides: Any) -> pb2.SimulationStatusMsg:
    values: dict[str, Any] = {
        "run_state": pb2.SimulationRunState.SIMULATION_RUNNING,
        "speed_factor": 1.0,
        "simulation_time_s": 120.0,
        "step_count": 120,
        "scenario": "nominal",
        "run_id": 3,
        "step_s": 1.0,
    }
    values.update(overrides)
    return pb2.SimulationStatusMsg(**values)


def system_state() -> pb2.SystemStateMsg:
    return pb2.SystemStateMsg(
        boiler=pb2.BoilerStateMsg(
            pressure_pa=140.0e5,
            water_level_m=4.8,
            water_temp_k=611.0,
            flue_gas_temp_k=1200.0,
            internal_energy_j=2.5e12,
            timestamp_ms=TIMESTAMP_MS,
            quality=pb2.SensorQuality.GOOD,
        ),
        turbine=pb2.TurbineStateMsg(
            electrical_power_w=200.0e6,
            shaft_power_w=205.0e6,
            enthalpy_in_j_kg=3_400_000.0,
            enthalpy_out_j_kg=2_200_000.0,
            exhaust_pressure_pa=7_000.0,
            steam_flow_kg_s=150.0,
            timestamp_ms=TIMESTAMP_MS,
            steam_temp_in_k=825.65,
        ),
        actuators=pb2.ActuatorStateMsg(
            fuel_valve_command=0.5,
            fuel_valve_position=0.5,
            feedwater_valve_command=0.5,
            feedwater_valve_position=0.5,
            steam_valve_command=0.5,
            steam_valve_position=0.5,
        ),
        active_faults=[
            pb2.FaultMsg(
                fault_id="f-1",
                kind=pb2.FaultKind.FAULT_SENSOR_DRIFT,
                target="drum_level",
                severity=0.2,
                label="sensor_drift:drum_level",
            )
        ],
        sensors=[
            pb2.SensorStatusMsg(
                sensor_id="drum_level",
                quality=pb2.SensorQuality.UNCERTAIN,
                measured_value=4.9,
            )
        ],
        simulation=simulation_status(),
    )


def plc_status(**overrides: Any) -> pb2.PLCStatusMsg:
    values: dict[str, Any] = {
        "mode": pb2.ControlMode.AUTO,
        "setpoints": pb2.SetpointsMsg(
            pressure_pa=140e5, water_level_m=4.8, steam_temp_k=811.0
        ),
        "active_setpoints": pb2.SetpointsMsg(
            pressure_pa=139e5, water_level_m=4.8, steam_temp_k=810.0
        ),
        "latest_command": pb2.ControlCommandMsg(
            fuel_valve=0.6,
            feedwater_valve=0.5,
            steam_valve=0.7,
            spray_valve=0.1,
            source=pb2.CommandSource.PID,
            timestamp_ms=TIMESTAMP_MS,
        ),
        "load_demand_w": 250e6,
        "load_setpoint_w": 240e6,
        "reset_permitted": False,
        "reset_blockers": ["not tripped"],
        "active_conditions": [
            pb2.AlarmConditionMsg(
                key="plc-controller:water_level_m:low:warning",
                parameter="water_level_m",
                severity="warning",
                direction="low",
                value=3.9,
                threshold=4.0,
                message="Drum level low",
                since_ms=TIMESTAMP_MS,
            )
        ],
        "loops": [
            pb2.ControlLoopMsg(
                name="pressure",
                setpoint=139e5,
                measurement=138e5,
                output=0.6,
                unit="Pa",
            )
        ],
        "warning_count": 2,
        "trip_count": 0,
        "run_id": 3,
    }
    values.update(overrides)
    return pb2.PLCStatusMsg(**values)


def alarm(alarm_id: int = 7, **overrides: Any) -> pb2.AlarmMsg:
    values: dict[str, Any] = {
        "alarm_id": alarm_id,
        "key": "plc-controller:water_level_m:low:critical",
        "source_service": "plc-controller",
        "parameter": "water_level_m",
        "severity": "critical",
        "direction": "low",
        "unit": "m",
        "state": pb2.AlarmState.ALARM_ACTIVE_UNACK,
        "message": "Drum level low-low",
        "action": "trip",
        "topic": "alerts/critical",
        "value": 3.1,
        "threshold": 3.5,
        "raised_at_ms": TIMESTAMP_MS,
        "occurrence_count": 1,
        "updated_at_ms": TIMESTAMP_MS,
    }
    values.update(overrides)
    return pb2.AlarmMsg(**values)


def _ack(accepted: bool, reason: str) -> pb2.CommandAck:
    return pb2.CommandAck(accepted=accepted, reason=reason, timestamp_ms=TIMESTAMP_MS)


class FakePhysicsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.down = False
        self.accept = True
        self.reason = ""
        self.status = simulation_status()
        self.streamed: list[pb2.SystemStateMsg] = []

    def _record(self, name: str, value: Any = None) -> None:
        if self.down:
            raise UpstreamDownError()
        self.calls.append((name, value))

    def _sim_ack(self) -> pb2.SimulationAck:
        return pb2.SimulationAck(
            accepted=self.accept,
            reason=self.reason,
            timestamp_ms=TIMESTAMP_MS,
            status=self.status,
        )

    async def close(self) -> None:
        return None

    async def health(self) -> pb2.HealthStatus:
        self._record("health")
        return pb2.HealthStatus(service="physics-engine", status="running")

    async def get_system_state(self) -> pb2.SystemStateMsg:
        self._record("get_system_state")
        return system_state()

    async def stream_system_state(
        self, *, interval_s: float = 0.0
    ) -> AsyncGenerator[pb2.SystemStateMsg]:
        self._record("stream_system_state", interval_s)
        for message in self.streamed:
            yield message

    async def get_simulation_status(self) -> pb2.SimulationStatusMsg:
        self._record("get_simulation_status")
        return self.status

    async def pause(self, operator_id: str) -> pb2.SimulationAck:
        self._record("pause", operator_id)
        return self._sim_ack()

    async def resume(self, operator_id: str) -> pb2.SimulationAck:
        self._record("resume", operator_id)
        return self._sim_ack()

    async def set_speed(
        self, speed_factor: float, operator_id: str
    ) -> pb2.SimulationAck:
        self._record("set_speed", (speed_factor, operator_id))
        return self._sim_ack()

    async def step(self, steps: int, operator_id: str) -> pb2.SimulationAck:
        self._record("step", (steps, operator_id))
        return self._sim_ack()

    async def list_scenarios(self) -> pb2.ScenarioListMsg:
        self._record("list_scenarios")
        return pb2.ScenarioListMsg(
            scenarios=[
                pb2.ScenarioMsg(
                    name="nominal", title="Nominal", description="Full load"
                ),
                pb2.ScenarioMsg(
                    name="hot_start", title="Hot start", description="From standby"
                ),
            ],
            current="nominal",
        )

    async def load_scenario(self, name: str, operator_id: str) -> pb2.SimulationAck:
        self._record("load_scenario", (name, operator_id))
        return self._sim_ack()

    async def inject_fault(self, request: pb2.FaultRequest) -> pb2.FaultAck:
        self._record("inject_fault", request)
        faults = (
            [
                pb2.FaultMsg(
                    fault_id="f-9",
                    kind=request.kind,
                    target=request.target,
                    severity=request.severity,
                    ramp_s=request.ramp_s,
                    label=f"fault:{request.target or 'plant'}",
                )
            ]
            if self.accept
            else []
        )
        return pb2.FaultAck(
            accepted=self.accept,
            reason=self.reason,
            timestamp_ms=TIMESTAMP_MS,
            faults=faults,
        )

    async def clear_fault(self, request: pb2.FaultClearRequest) -> pb2.FaultAck:
        self._record("clear_fault", request)
        faults = (
            [pb2.FaultMsg(fault_id=request.fault_id or "f-1", label="fault:cleared")]
            if self.accept
            else []
        )
        return pb2.FaultAck(
            accepted=self.accept,
            reason=self.reason,
            timestamp_ms=TIMESTAMP_MS,
            faults=faults,
        )


class FakePLCClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.down = False
        self.accept = True
        self.reason = ""
        self.status = plc_status()

    def _ack(self, name: str, value: Any) -> pb2.CommandAck:
        if self.down:
            raise UpstreamDownError()
        self.calls.append((name, value))
        return _ack(self.accept, self.reason)

    async def close(self) -> None:
        return None

    async def health(self) -> pb2.HealthStatus:
        if self.down:
            raise UpstreamDownError()
        return pb2.HealthStatus(service="plc-controller", status="running")

    async def send_command(self, command: pb2.ControlCommandMsg) -> pb2.CommandAck:
        return self._ack("send_command", command)

    async def update_setpoints(self, setpoints: pb2.SetpointsMsg) -> pb2.CommandAck:
        return self._ack("update_setpoints", setpoints)

    async def reset_emergency_stop(self, operator_id: str) -> pb2.CommandAck:
        return self._ack("reset_emergency_stop", operator_id)

    async def set_load_demand(self, load_w: float, operator_id: str) -> pb2.CommandAck:
        return self._ack("set_load_demand", (load_w, operator_id))

    async def set_control_mode(self, mode: int, operator_id: str) -> pb2.CommandAck:
        return self._ack("set_control_mode", (mode, operator_id))

    async def get_control_status(self) -> pb2.PLCStatusMsg:
        if self.down:
            raise UpstreamDownError()
        return self.status


class FakeAlarmClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.down = False
        self.accept = True
        self.reason = ""
        self.alarms = [alarm(7), alarm(8, severity="warning", state=2)]

    def _record(self, name: str, value: Any) -> None:
        if self.down:
            raise UpstreamDownError()
        self.calls.append((name, value))

    def _result(self, alarms: list[pb2.AlarmMsg]) -> pb2.AcknowledgeResult:
        return pb2.AcknowledgeResult(
            accepted=self.accept,
            reason=self.reason,
            timestamp_ms=TIMESTAMP_MS,
            alarms=alarms if self.accept else [],
        )

    async def close(self) -> None:
        return None

    async def health(self) -> pb2.HealthStatus:
        self._record("health", None)
        return pb2.HealthStatus(service="alert-manager", status="running")

    async def list_alarms(self, request: pb2.ListAlarmsRequest) -> pb2.AlarmListMsg:
        self._record("list_alarms", request)
        return pb2.AlarmListMsg(alarms=self.alarms, total=len(self.alarms) + 40)

    async def get_alarm(self, alarm_id: int) -> pb2.AlarmDetailMsg:
        self._record("get_alarm", alarm_id)
        if alarm_id != 7:
            raise not_found()
        return pb2.AlarmDetailMsg(
            alarm=self.alarms[0],
            transitions=[
                pb2.AlarmTransitionMsg(
                    transition_id=1,
                    alarm_id=7,
                    from_state=pb2.AlarmState.ALARM_STATE_UNSPECIFIED,
                    to_state=pb2.AlarmState.ALARM_ACTIVE_UNACK,
                    at_ms=TIMESTAMP_MS,
                    actor="plc-controller",
                    value=3.1,
                ),
                pb2.AlarmTransitionMsg(
                    transition_id=2,
                    alarm_id=7,
                    from_state=pb2.AlarmState.ALARM_ACTIVE_UNACK,
                    to_state=pb2.AlarmState.ALARM_ACTIVE_ACK,
                    at_ms=TIMESTAMP_MS + 1000,
                    actor="operator1",
                    comment="seen",
                    value=3.1,
                ),
            ],
        )

    async def acknowledge(
        self, alarm_id: int, operator_id: str, comment: str
    ) -> pb2.AcknowledgeResult:
        self._record("acknowledge", (alarm_id, operator_id, comment))
        return self._result([alarm(alarm_id, state=pb2.AlarmState.ALARM_ACTIVE_ACK)])

    async def acknowledge_all(
        self, operator_id: str, comment: str, severity: str
    ) -> pb2.AcknowledgeResult:
        self._record("acknowledge_all", (operator_id, comment, severity))
        return self._result(self.alarms)


class FakeHistorianClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.kpi_values: dict[tuple[str, str], float] = {}
        self.up = True

    def close(self) -> None:
        return None

    def ping(self) -> bool:
        return self.up

    def fetch_history(
        self,
        *,
        measurement: str,
        start_ms: int,
        end_ms: int,
        limit: int,
        window_s: int = 0,
        fields: tuple[str, ...] = (),
    ) -> list[dict[str, object]]:
        self.calls.append(("fetch_history", (measurement, start_ms, end_ms)))
        return []

    def fetch_kpi_inputs(
        self, *, start_ms: int, end_ms: int
    ) -> tuple[HistorySource, dict[tuple[str, str], float]]:
        if not self.up:
            raise ConnectionRefusedError("influxdb refused the connection")
        self.calls.append(("fetch_kpi_inputs", (start_ms, end_ms)))
        return HistorySource("sensors", aggregated=False), self.kpi_values
