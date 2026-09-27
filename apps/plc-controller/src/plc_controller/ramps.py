"""Rate-limited setpoints: an operator can ask for a new target, not for a step."""

from __future__ import annotations

import math


class RampedSetpoint:
    """A working setpoint that approaches its target at a bounded rate.

    A target, a measurement or an interval that is not a real number is ignored, so
    one bad reading cannot leave the working setpoint running off at full rate.
    """

    def __init__(self, rate_per_s: float, value: float = 0.0) -> None:
        if rate_per_s <= 0.0:
            raise ValueError("rate_per_s must be > 0")
        self.rate_per_s = rate_per_s
        self.target = value
        self.value = value

    def set_target(self, target: float) -> None:
        """Where the setpoint should go; it gets there at `rate_per_s`."""
        if math.isfinite(target):
            self.target = target

    def track(self, value: float) -> None:
        """Start the working setpoint from a measurement (bumpless), keeping the target."""
        if math.isfinite(value):
            self.value = value

    def step(self, dt: float) -> float:
        """Move toward the target by at most `rate_per_s · dt` and return the value."""
        if not math.isfinite(dt):
            return self.value
        max_move = self.rate_per_s * max(dt, 0.0)
        error = self.target - self.value
        self.value += max(-max_move, min(max_move, error))
        return self.value

    @property
    def settled(self) -> bool:
        return self.value == self.target
