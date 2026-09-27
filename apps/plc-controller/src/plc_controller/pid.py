"""
Discrete-time PID controller for the boiler control loops.

    - Anti-windup (integrator clamping)
    - Output clamping
    - Derivative filtering (low-pass)
    - Back-calculation when the caller limits the output further

Bumpless transfer is done by the caller: `reset(initial_output=...)` loads the
integrator with the valve position the loop takes over (see `control.py`).
"""

import math
from dataclasses import dataclass

from plc_controller.numeric import all_finite, clamp

# ─── PID tuning parameters ────────────────────────────────────────────────────


@dataclass
class PIDParameters:
    """
    Tuning parameters and constraints for a single PID controller.

    All time constants in seconds, all limits normalized unless noted.
    """

    kp: float  # Proportional gain
    ki: float  # Integral gain [1/s]
    kd: float  # Derivative gain [s]

    output_min: float = 0.0  # Lower clamp on controller output
    output_max: float = 1.0  # Upper clamp on controller output

    # Derivative low-pass filter coefficient [s].
    # Filters high-frequency noise in the derivative term.
    # tau_d = 0 disables filtering (pure derivative).
    # Typical: 0.1 × Td (derivative time constant).
    tau_d: float = 0.0

    # Anti-windup: integrator is frozen when output is saturated.
    # True  = clamp integrator when output hits output_min / output_max.
    # False = allow integrator to wind up (useful for feed-forward schemes).
    anti_windup: bool = True


# ─── PID state ────────────────────────────────────────────────────────────────


@dataclass
class PIDState:
    """
    Internal state of a PID controller between time steps.

    Preserved across calls to PIDController.step().
    """

    integral: float = 0.0  # Accumulated integral term
    prev_error: float = 0.0  # Error at previous time step (for derivative)
    prev_derivative: float = 0.0  # Filtered derivative at previous step
    prev_output: float = 0.0  # Output at previous step
    initialized: bool = False  # False until first step() call


# ─── Single PID controller ────────────────────────────────────────────────────


class PIDController:
    """
    Discrete-time PID controller with anti-windup and derivative filtering.

    Algorithm (velocity / positional form with clamping):

        error        = setpoint − measurement
        derivative   = (error − prev_error) / dt          [raw]
        d_filtered   = (tau_d·d_prev + dt·derivative) /   [filtered]
                       (tau_d + dt)
        integral    += ki · error · dt                    [with anti-windup]
        output       = kp·error + integral + kd·d_filtered
        output       = clamp(output, out_min, out_max)

    Anti-windup: if output saturates, the integral is back-calculated so
    that the unsaturated equivalent matches the clamped output.  This
    prevents the integrator from winding up during actuator saturation
    (e.g. valve fully open/closed).

    Usage:
        params = PIDParameters(kp=2.0, ki=0.1, kd=0.5, output_min=0.0, output_max=1.0)
        pid    = PIDController(params)
        # In control loop:
        output = pid.step(setpoint=140e5, measurement=138e5, dt=1.0)
    """

    def __init__(self, params: PIDParameters) -> None:
        self.params = params
        self.state = PIDState()

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(self, initial_output: float = 0.0) -> None:
        """
        Reset controller state.

        Args:
            initial_output: Pre-load the integrator to this output value.
                            Avoids a large transient on first AUTO step. A value
                            that is not a real number keeps the previous output.
        """
        if not math.isfinite(initial_output):
            initial_output = self.state.prev_output
        self.state = PIDState(integral=initial_output, prev_output=initial_output)

    def constrain(self, limited_output: float) -> None:
        """
        Back-calculate the integrator after the caller limited this step's output.

        Loops whose output is limited outside the PID — a feedforward added to it, a
        guard, an actuator range — call this so the integrator does not wind up.
        """
        if not math.isfinite(limited_output):
            return
        delta = limited_output - self.state.prev_output
        if delta != 0.0:
            self.state.integral += delta
            self.state.prev_output = limited_output

    # ─── Main step ────────────────────────────────────────────────────────────

    def step(
        self,
        setpoint: float,
        measurement: float,
        dt: float,
    ) -> float:
        """
        Compute one PID step.

        Args:
            setpoint:    Desired value (in process units).
            measurement: Current measured value (same units as setpoint).
            dt:          Time step [s]. Must be > 0.

        Returns:
            Controller output, clamped to [output_min, output_max]. With an input
            that is not a real number the step is skipped: the previous output is
            returned and the state is left untouched.
        """
        if not all_finite(setpoint, measurement, dt):
            return self.state.prev_output

        p = self.params

        # ── Initialise on first call ──────────────────────────────────────────
        if not self.state.initialized:
            self.state.prev_error = setpoint - measurement
            self.state.initialized = True

        error = setpoint - measurement

        # ── Derivative term with low-pass filtering ───────────────────────────
        raw_derivative = (error - self.state.prev_error) / dt if dt > 0.0 else 0.0
        if p.tau_d > 0.0:
            alpha = p.tau_d / (p.tau_d + dt)
            derivative = (
                alpha * self.state.prev_derivative + (1.0 - alpha) * raw_derivative
            )
        else:
            derivative = raw_derivative

        # ── Proportional + derivative (before integral) ───────────────────────
        output_pd = p.kp * error + p.kd * derivative

        # ── Integral with anti-windup ─────────────────────────────────────────
        self.state.integral += p.ki * error * dt

        output_raw = output_pd + self.state.integral

        # ── Output clamping ───────────────────────────────────────────────────
        output = clamp(output_raw, p.output_min, p.output_max)

        # ── Anti-windup: back-calculate integrator ────────────────────────────
        if p.anti_windup and output != output_raw:
            # Clamp integrator so that output_pd + integral == output
            self.state.integral = output - output_pd

        # ── Save state ────────────────────────────────────────────────────────
        self.state.prev_error = error
        self.state.prev_derivative = derivative
        self.state.prev_output = output

        return output
