"""The "log once per outage" rule, as one small state machine.

The gateway's realtime sources, both OPC UA projections and the PLC scan loop each kept
a hand-rolled `down` flag with its own wording, and the historian's InfluxDB writer kept
none and warned on every failed batch. A dependency that stays away for an hour should
cost one warning and one recovery line, not one line per retry.
"""

from __future__ import annotations

import logging


class OutageLog:
    """Warns at the first failure of an outage, says once when it is over."""

    def __init__(self, logger: logging.Logger, what: str) -> None:
        self._logger = logger
        self._what = what
        self._down = False

    @property
    def down(self) -> bool:
        """True from the first failure until the next recovery."""
        return self._down

    def failed(self, error: BaseException) -> bool:
        """Record a failure; True when it started a new outage (and was warned about)."""
        if self._down:
            self._logger.debug("%s still unavailable: %s", self._what, error)
            return False
        self._down = True
        self._logger.warning("%s unavailable: %s", self._what, error)
        return True

    def recovered(self) -> bool:
        """Record a success; True when it ended an outage (and said so)."""
        if not self._down:
            return False
        self._down = False
        self._logger.info("%s available again", self._what)
        return True
