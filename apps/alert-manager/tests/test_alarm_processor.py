"""The alarm processor: conditions become alarms with a lifecycle."""

from __future__ import annotations

import asyncio
import logging

import pytest
from alarm_factories import (
    SOURCE,
    Recorder,
    condition,
    only_alarm,
    snapshot,
    wait_for_state,
)
from alert_manager import lifecycle
from alert_manager.lifecycle import AlarmState, LifecycleError
from alert_manager.models import AlarmEvent
from alert_manager.processor import (
    AlarmNotFoundError,
    AlarmProcessor,
    AlarmQuery,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

S = AlarmState


class TestLifecycleRules:
    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            (S.ACTIVE_UNACK, S.ACTIVE_UNACK),
            (S.ACTIVE_ACK, S.ACTIVE_ACK),
            (S.CLEARED_UNACK, S.ACTIVE_UNACK),
        ],
    )
    def test_a_condition_that_is_active(self, state: S, expected: S) -> None:
        assert lifecycle.after_condition_active(state) is expected

    def test_a_closed_alarm_is_never_reactivated(self) -> None:
        with pytest.raises(LifecycleError):
            lifecycle.after_condition_active(S.CLEARED)

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            (S.ACTIVE_UNACK, S.CLEARED_UNACK),
            (S.ACTIVE_ACK, S.CLEARED),
            (S.CLEARED_UNACK, S.CLEARED_UNACK),
            (S.CLEARED, S.CLEARED),
        ],
    )
    def test_a_condition_that_ends(self, state: S, expected: S) -> None:
        assert lifecycle.after_condition_cleared(state) is expected

    @pytest.mark.parametrize(
        ("state", "expected"),
        [(S.ACTIVE_UNACK, S.ACTIVE_ACK), (S.CLEARED_UNACK, S.CLEARED)],
    )
    def test_an_acknowledgement(self, state: S, expected: S) -> None:
        assert lifecycle.after_acknowledge(state) is expected

    @pytest.mark.parametrize("state", [S.ACTIVE_ACK, S.CLEARED])
    def test_acknowledging_twice_is_refused(self, state: S) -> None:
        with pytest.raises(LifecycleError, match="already acknowledged"):
            lifecycle.after_acknowledge(state)

    def test_the_state_groups(self) -> None:
        assert S.CLEARED not in lifecycle.OPEN_STATES
        assert lifecycle.ACTIVE_STATES < lifecycle.OPEN_STATES
        assert lifecycle.UNACKNOWLEDGED_STATES == {S.ACTIVE_UNACK, S.CLEARED_UNACK}


class TestActivation:
    async def test_a_new_condition_opens_an_unacknowledged_alarm(
        self, processor: AlarmProcessor, recorder: Recorder
    ) -> None:
        await processor.handle_condition(condition(timestamp_ms=1_234))
        alarm = await only_alarm(processor)
        assert alarm.state is S.ACTIVE_UNACK
        assert alarm.raised_at_ms == 1_234
        assert alarm.occurrence_count == 1
        assert alarm.is_open and not alarm.is_acknowledged
        assert recorder.states == [(None, "ACTIVE_UNACK")]
        transition = recorder.changes[0][1]
        assert (transition.actor, transition.value) == (SOURCE, 3.1)

    async def test_a_repeated_report_updates_the_value_without_a_transition(
        self, processor: AlarmProcessor, recorder: Recorder
    ) -> None:
        await processor.handle_condition(condition(value=3.1))
        await processor.handle_condition(condition(value=2.9))
        alarm = await only_alarm(processor)
        assert alarm.value == 2.9
        assert alarm.occurrence_count == 1
        assert len(recorder.changes) == 1

    async def test_different_conditions_are_different_alarms(
        self, processor: AlarmProcessor
    ) -> None:
        await processor.handle_condition(condition("water_level_m"))
        await processor.handle_condition(condition("pressure_pa", direction="high"))
        _, total = await processor.list_alarms(AlarmQuery())
        assert total == 2


class TestClearing:
    async def test_a_condition_that_stays_normal_clears_after_the_hold(
        self, processor: AlarmProcessor, recorder: Recorder
    ) -> None:
        await processor.handle_condition(condition(timestamp_ms=1_000))
        alarm = await only_alarm(processor)
        await processor.handle_condition(
            condition(active=False, value=4.2, timestamp_ms=2_000)
        )
        cleared = await wait_for_state(processor, alarm.id, S.CLEARED_UNACK)
        assert cleared.cleared_at_ms == 2_000
        assert cleared.value == 4.2
        assert cleared.is_open
        assert recorder.states == [
            (None, "ACTIVE_UNACK"),
            ("ACTIVE_UNACK", "CLEARED_UNACK"),
        ]

    async def test_a_flickering_condition_keeps_one_alarm(
        self,
        processor: AlarmProcessor,
        recorder: Recorder,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await processor.handle_condition(condition())
        with caplog.at_level(logging.INFO, logger="alert_manager.processor"):
            await processor.handle_condition(condition(active=False))
            await processor.handle_condition(condition())
        await asyncio.sleep(0.1)
        alarm = await only_alarm(processor)
        assert alarm.state is S.ACTIVE_UNACK
        assert len(recorder.changes) == 1
        assert "returned within its clear hold" in caplog.text

    async def test_a_second_clear_report_does_not_restart_the_hold(
        self, processor: AlarmProcessor, recorder: Recorder
    ) -> None:
        await processor.handle_condition(condition())
        alarm = await only_alarm(processor)
        await processor.handle_condition(condition(active=False, timestamp_ms=2_000))
        await processor.handle_condition(condition(active=False, timestamp_ms=3_000))
        cleared = await wait_for_state(processor, alarm.id, S.CLEARED_UNACK)
        assert cleared.cleared_at_ms == 2_000
        assert recorder.states.count(("ACTIVE_UNACK", "CLEARED_UNACK")) == 1

    async def test_a_clear_for_an_unknown_condition_does_nothing(
        self, processor: AlarmProcessor, recorder: Recorder
    ) -> None:
        await processor.handle_condition(condition(active=False))
        await asyncio.sleep(0.1)
        assert recorder.changes == []

    async def test_a_returning_condition_reactivates_the_same_open_alarm(
        self, processor: AlarmProcessor, recorder: Recorder
    ) -> None:
        await processor.handle_condition(condition())
        alarm = await only_alarm(processor)
        await processor.handle_condition(condition(active=False))
        await wait_for_state(processor, alarm.id, S.CLEARED_UNACK)
        await processor.handle_condition(condition(timestamp_ms=9_000))
        again = await only_alarm(processor)
        assert again.id == alarm.id
        assert again.state is S.ACTIVE_UNACK
        assert again.occurrence_count == 2
        assert again.cleared_at_ms is None
        assert recorder.states[-1] == ("CLEARED_UNACK", "ACTIVE_UNACK")

    async def test_closing_waits_for_nothing_and_cancels_pending_clears(
        self,
        sessions: async_sessionmaker[AsyncSession],
        recorder: Recorder,
    ) -> None:
        slow = AlarmProcessor(sessions, recorder, clear_hold_s=60.0)
        await slow.handle_condition(condition())
        await slow.handle_condition(condition(active=False))
        await slow.close()
        alarm = await only_alarm(slow)
        assert alarm.state is S.ACTIVE_UNACK

    async def test_a_failed_clear_is_logged(
        self,
        processor: AlarmProcessor,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await processor.handle_condition(condition())

        async def broken(*_: object) -> None:
            raise RuntimeError("database went away")

        monkeypatch.setattr(processor, "_apply_clear", broken)
        with caplog.at_level(logging.ERROR, logger="alert_manager.processor"):
            await processor.handle_condition(condition(active=False))
            async with asyncio.timeout(5.0):
                while "Clearing alarm" not in caplog.text:
                    await asyncio.sleep(0.005)
        assert "database went away" in caplog.text


class TestAcknowledgement:
    async def test_an_active_alarm_is_acknowledged_by_name_with_a_comment(
        self, processor: AlarmProcessor, recorder: Recorder
    ) -> None:
        await processor.handle_condition(condition())
        alarm = await only_alarm(processor)
        acknowledged = await processor.acknowledge(alarm.id, "  operator1 ", " seen ")
        assert acknowledged.state is S.ACTIVE_ACK
        assert acknowledged.acknowledged_by == "operator1"
        assert acknowledged.ack_comment == "seen"
        assert acknowledged.acknowledged_at_ms is not None
        transition = recorder.changes[-1][1]
        assert (transition.actor, transition.comment) == ("operator1", "seen")

    async def test_an_acknowledged_alarm_closes_when_its_condition_ends(
        self, processor: AlarmProcessor
    ) -> None:
        await processor.handle_condition(condition())
        alarm = await only_alarm(processor)
        await processor.acknowledge(alarm.id, "operator1")
        await processor.handle_condition(condition(active=False))
        closed = await wait_for_state(processor, alarm.id, S.CLEARED)
        assert not closed.is_open

    async def test_after_closing_the_next_occurrence_is_a_new_alarm(
        self, processor: AlarmProcessor
    ) -> None:
        await processor.handle_condition(condition())
        first = await only_alarm(processor)
        await processor.handle_condition(condition(active=False))
        await wait_for_state(processor, first.id, S.CLEARED_UNACK)
        await processor.acknowledge(first.id, "operator1")
        await processor.handle_condition(condition(timestamp_ms=7_000))
        alarms, total = await processor.list_alarms(AlarmQuery())
        assert total == 2
        assert {alarm.id for alarm in alarms} - {first.id}

    async def test_acknowledging_twice_is_refused(
        self, processor: AlarmProcessor
    ) -> None:
        await processor.handle_condition(condition())
        alarm = await only_alarm(processor)
        await processor.acknowledge(alarm.id, "operator1")
        with pytest.raises(LifecycleError):
            await processor.acknowledge(alarm.id, "operator2")

    async def test_an_unknown_alarm_is_not_found(
        self, processor: AlarmProcessor
    ) -> None:
        with pytest.raises(AlarmNotFoundError):
            await processor.acknowledge(404, "operator1")
        with pytest.raises(AlarmNotFoundError):
            await processor.get_alarm(404)

    async def test_names_and_comments_are_bounded(
        self, processor: AlarmProcessor
    ) -> None:
        await processor.handle_condition(condition())
        alarm = await only_alarm(processor)
        acknowledged = await processor.acknowledge(alarm.id, "   ", "x" * 900)
        assert acknowledged.acknowledged_by == "unknown"
        assert acknowledged.ack_comment == "x" * 500

    async def test_acknowledge_all_of_one_severity(
        self, processor: AlarmProcessor
    ) -> None:
        await processor.handle_condition(condition("water_level_m"))
        await processor.handle_condition(condition("stack_temp_k", severity="warning"))
        await processor.handle_condition(condition("nox_ppmv", severity="warning"))
        acknowledged = await processor.acknowledge_all("operator1", "shift", "warning")
        assert sorted(alarm.parameter for alarm in acknowledged) == [
            "nox_ppmv",
            "stack_temp_k",
        ]
        remaining = await processor.acknowledge_all("operator1")
        assert [alarm.parameter for alarm in remaining] == ["water_level_m"]
        assert await processor.acknowledge_all("operator1") == []


class TestSnapshots:
    async def test_active_alarms_missing_from_a_snapshot_are_cleared(
        self, processor: AlarmProcessor, caplog: pytest.LogCaptureFixture
    ) -> None:
        kept = condition("water_level_m")
        gone = condition("pressure_pa", direction="high")
        await processor.handle_condition(kept)
        await processor.handle_condition(gone)
        with caplog.at_level(logging.INFO, logger="alert_manager.processor"):
            await processor.handle_snapshot(snapshot(kept.key, timestamp_ms=5_000))
        alarms, _ = await processor.list_alarms(AlarmQuery())
        by_key = {alarm.key: alarm for alarm in alarms}
        cleared = await wait_for_state(processor, by_key[gone.key].id, S.CLEARED_UNACK)
        assert cleared.cleared_at_ms == 5_000
        still, _ = await processor.get_alarm(by_key[kept.key].id)
        assert still.state is S.ACTIVE_UNACK
        assert "no longer lists" in caplog.text

    async def test_a_snapshot_speaks_only_for_its_own_source(
        self, processor: AlarmProcessor
    ) -> None:
        await processor.handle_condition(condition(source="physics-engine"))
        await processor.handle_snapshot(snapshot(timestamp_ms=5_000))
        await asyncio.sleep(0.1)
        alarm = await only_alarm(processor)
        assert alarm.state is S.ACTIVE_UNACK

    async def test_alarms_raised_after_the_snapshot_are_kept(
        self, processor: AlarmProcessor
    ) -> None:
        await processor.handle_condition(condition(timestamp_ms=9_000))
        await processor.handle_snapshot(snapshot(timestamp_ms=5_000))
        await asyncio.sleep(0.1)
        assert (await only_alarm(processor)).state is S.ACTIVE_UNACK


class TestQueries:
    async def _three_alarms(self, processor: AlarmProcessor) -> dict[str, int]:
        await processor.handle_condition(
            condition("stack_temp_k", severity="warning", timestamp_ms=1_000)
        )
        await processor.handle_condition(
            condition("water_level_m", severity="critical", timestamp_ms=2_000)
        )
        await processor.handle_condition(
            condition("nox_ppmv", severity="warning", timestamp_ms=3_000)
        )
        alarms, _ = await processor.list_alarms(AlarmQuery())
        return {alarm.parameter: alarm.id for alarm in alarms}

    async def test_history_is_newest_first(self, processor: AlarmProcessor) -> None:
        await self._three_alarms(processor)
        alarms, total = await processor.list_alarms(AlarmQuery())
        assert total == 3
        assert [alarm.raised_at_ms for alarm in alarms] == [3_000, 2_000, 1_000]

    async def test_open_alarms_put_critical_then_unacknowledged_first(
        self, processor: AlarmProcessor
    ) -> None:
        ids = await self._three_alarms(processor)
        await processor.acknowledge(ids["nox_ppmv"], "operator1")
        alarms, _ = await processor.list_alarms(AlarmQuery(open_only=True))
        assert [alarm.parameter for alarm in alarms] == [
            "water_level_m",
            "stack_temp_k",
            "nox_ppmv",
        ]

    async def test_closed_alarms_leave_the_open_list(
        self, processor: AlarmProcessor
    ) -> None:
        ids = await self._three_alarms(processor)
        await processor.acknowledge(ids["stack_temp_k"], "operator1")
        await processor.handle_condition(
            condition("stack_temp_k", severity="warning", active=False)
        )
        await wait_for_state(processor, ids["stack_temp_k"], S.CLEARED)
        _, total = await processor.list_alarms(AlarmQuery(open_only=True))
        assert total == 2

    @pytest.mark.parametrize(
        ("query", "parameters"),
        [
            (AlarmQuery(severity="warning"), ["nox_ppmv", "stack_temp_k"]),
            (AlarmQuery(parameter="water_level_m"), ["water_level_m"]),
            (AlarmQuery(from_ms=2_000), ["nox_ppmv", "water_level_m"]),
            (AlarmQuery(to_ms=2_000), ["water_level_m", "stack_temp_k"]),
            (AlarmQuery(limit=1, offset=1), ["water_level_m"]),
        ],
    )
    async def test_filters_and_paging(
        self, processor: AlarmProcessor, query: AlarmQuery, parameters: list[str]
    ) -> None:
        await self._three_alarms(processor)
        alarms, _ = await processor.list_alarms(query)
        assert [alarm.parameter for alarm in alarms] == parameters

    async def test_out_of_range_paging_is_clamped(
        self, processor: AlarmProcessor
    ) -> None:
        await self._three_alarms(processor)
        alarms, total = await processor.list_alarms(AlarmQuery(limit=0, offset=-5))
        assert (len(alarms), total) == (3, 3)
        alarms, _ = await processor.list_alarms(AlarmQuery(limit=5_000))
        assert len(alarms) == 3

    async def test_a_detail_has_its_transitions_oldest_first(
        self, processor: AlarmProcessor
    ) -> None:
        await processor.handle_condition(condition(timestamp_ms=1_000))
        alarm = await only_alarm(processor)
        await processor.acknowledge(alarm.id, "operator1", "seen")
        _, transitions = await processor.get_alarm(alarm.id)
        assert [t.to_state for t in transitions] == [S.ACTIVE_UNACK, S.ACTIVE_ACK]
        assert transitions[0].from_state is None
        assert transitions[1].to_dict()["from_state"] == "ACTIVE_UNACK"


class TestDatabase:
    async def test_ping_reaches_the_database(self, processor: AlarmProcessor) -> None:
        await processor.ping()

    async def test_ping_fails_without_a_database(self, recorder: Recorder) -> None:
        engine = create_async_engine("sqlite+aiosqlite:///Z:/no/such/dir/alarms.db")
        broken = AlarmProcessor(async_sessionmaker(engine), recorder)
        with pytest.raises(Exception):  # noqa: B017 - the driver's own error
            await broken.ping()
        await engine.dispose()

    async def test_one_open_alarm_per_key_is_enforced_by_the_schema(
        self, sessions: async_sessionmaker[AsyncSession], processor: AlarmProcessor
    ) -> None:
        await processor.handle_condition(condition())
        alarm = await only_alarm(processor)
        async with sessions() as session:
            session.add(
                AlarmEvent(
                    key=alarm.key,
                    source_service=SOURCE,
                    parameter="water_level_m",
                    severity="critical",
                    direction="low",
                    unit="m",
                    state="ACTIVE_UNACK",
                    message="duplicate",
                    action="trip",
                    topic="alerts/critical",
                    value=1.0,
                    threshold=3.5,
                    raised_at_ms=1,
                    updated_at_ms=1,
                )
            )
            with pytest.raises(IntegrityError):
                await session.commit()

    async def test_a_processor_without_a_listener_still_works(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        silent = AlarmProcessor(sessions, clear_hold_s=0.01)
        await silent.handle_condition(condition())
        assert (await only_alarm(silent)).state is S.ACTIVE_UNACK
        await silent.close()
