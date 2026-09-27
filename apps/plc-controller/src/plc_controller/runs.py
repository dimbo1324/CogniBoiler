"""Which plant run the PLC is following, and how much plant time each scan covers.

A scenario load or a physics restart starts a new run: the PLC then re-primes its loops
and forgets the plant's history. The interval between scans is plant time, not wall
time, so the loops behave the same at any simulation speed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from plc_controller.measurements import ProcessMeasurements
from plc_controller.safety_limits import ON_LINE_STEAM_FLOW_KG_S

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunStep:
    """What one plant state means for the scan's timing."""

    interval_s: float | None
    new_run: bool
    first_run: bool


class PlantRun:
    """Follows the plant's run id and simulation time from one state to the next."""

    def __init__(self) -> None:
        self.run_id: int | None = None
        self._last_time_s: float | None = None
        self._non_finite = False

    def advance(self, m: ProcessMeasurements) -> RunStep:
        """The scan interval in plant time; None on a run's first state or a bad clock."""
        self._note_non_finite(m)
        time_s = m.simulation_time_s if math.isfinite(m.simulation_time_s) else None
        if self.run_id != m.run_id:
            first_run = self.run_id is None
            self.run_id = m.run_id
            self._last_time_s = time_s
            if first_run:
                _warn_if_tripped(m)
            return RunStep(interval_s=None, new_run=True, first_run=first_run)
        if time_s is None:
            return RunStep(interval_s=None, new_run=False, first_run=False)
        previous, self._last_time_s = self._last_time_s, time_s
        interval = None if previous is None else max(time_s - previous, 0.0)
        return RunStep(interval_s=interval, new_run=False, first_run=False)

    def _note_non_finite(self, m: ProcessMeasurements) -> None:
        """Say once, when it starts, that a reading is not a number."""
        finite = m.finite
        if not finite and not self._non_finite:
            logger.warning(
                "A plant reading is not a number: its instrument counts as failed and "
                "the control loops hold their valves until every reading is a number"
            )
        elif finite and self._non_finite:
            logger.info("Every plant reading is a number again")
        self._non_finite = not finite


def _warn_if_tripped(m: ProcessMeasurements) -> None:
    """The latch lives in memory only: say so when the PLC starts on a tripped unit."""
    if m.commands.fuel <= 0.0 and m.steam_flow_kg_s < ON_LINE_STEAM_FLOW_KG_S:
        logger.warning(
            "The first plant state looks tripped (fuel command 0, turbine off line). "
            "The PLC starts in AUTO without a latched E-Stop: a trip latched before a "
            "PLC restart is not restored"
        )
