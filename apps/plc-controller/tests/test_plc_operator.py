"""Every request that changes the unit is attributed to a person (invariant I3).

The gateway always sends the signed-in username; these tests pin what the PLC does
with a caller that does not: it refuses, except for an E-Stop, which is never refused.
"""

from __future__ import annotations

import cogniboiler_pb2 as pb2
import pytest
from plc_controller.commands import MAX_OPERATOR_ID_LENGTH, check_operator
from plc_controller.service import PLCService, RuntimeMode
from plc_fakes import RecordingPublisher, plc, state

UNATTRIBUTED = ["", "   ", "\t", "x" * (MAX_OPERATOR_ID_LENGTH + 1)]


async def seen() -> tuple[PLCService, RecordingPublisher]:
    svc, _ = plc()
    publisher = RecordingPublisher()
    svc._publisher = publisher  # type: ignore[assignment]
    await svc.process_state(state(step=0))
    return svc, publisher


class TestCheckOperator:
    def test_a_name_is_accepted_up_to_the_username_length(self) -> None:
        assert check_operator("anna").accepted
        assert check_operator("x" * MAX_OPERATOR_ID_LENGTH).accepted

    @pytest.mark.parametrize("operator_id", UNATTRIBUTED)
    def test_no_name_or_a_name_too_long_is_refused(self, operator_id: str) -> None:
        result = check_operator(operator_id)
        assert not result.accepted
        assert "operator_id" in result.reason


class TestEveryRequestIsAttributed:
    @pytest.mark.parametrize("operator_id", UNATTRIBUTED)
    async def test_a_valve_command(self, operator_id: str) -> None:
        svc, _ = await seen()
        result = await svc.send_command(
            fuel_valve=0.4,
            feedwater_valve=0.5,
            steam_valve=0.6,
            source=pb2.CommandSource.OPERATOR,
            operator_id=operator_id,
        )
        assert not result.accepted and "operator_id" in result.reason
        assert svc.mode is RuntimeMode.AUTO

    @pytest.mark.parametrize("mode", [RuntimeMode.AUTO, RuntimeMode.MANUAL])
    async def test_a_mode_change(self, mode: RuntimeMode) -> None:
        svc, _ = await seen()
        result = await svc.set_mode(mode, "")
        assert not result.accepted and "operator_id" in result.reason

    async def test_a_reset(self) -> None:
        svc, _ = await seen()
        await svc.set_mode(RuntimeMode.ESTOP, "eng")
        result = await svc.reset_emergency_stop(" ")
        assert not result.accepted and "operator_id" in result.reason
        assert svc.mode is RuntimeMode.ESTOP

    async def test_setpoints_and_load_demand(self) -> None:
        svc, _ = await seen()
        setpoints = svc.update_setpoints(130.0e5, 5.0, 800.0, operator_id="")
        load = svc.set_load_demand(200.0e6, operator_id="")
        assert not setpoints.accepted and "operator_id" in setpoints.reason
        assert not load.accepted and "operator_id" in load.reason
        assert svc.get_setpoints().pressure_pa == pytest.approx(140.0e5)

    async def test_an_e_stop_is_never_refused_for_want_of_a_name(self) -> None:
        svc, publisher = await seen()
        result = await svc.set_mode(RuntimeMode.ESTOP, "")
        assert result.accepted
        assert svc.mode is RuntimeMode.ESTOP
        assert publisher.events[-1].operator_id == "unattributed"

    async def test_the_name_is_recorded_without_surrounding_blanks(self) -> None:
        svc, publisher = await seen()
        assert svc.set_load_demand(200.0e6, operator_id="  anna ").accepted
        assert publisher.events[-1].operator_id == "anna"
