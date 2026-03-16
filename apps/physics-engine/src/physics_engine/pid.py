"""
PID controller and cascade PID for boiler control loops.

Implements a discrete-time PID controller with:
    - Anti-windup (integrator clamping)
    - Output clamping
    - Derivative filtering (low-pass)
    - Bumpless transfer between AUTO and MANUAL modes

Cascade PID connects two PID controllers in series:
    - Master (outer): slow loop, computes setpoint for slave
    - Slave (inner):  fast loop, drives the actuator directly

Typical boiler control loops:
    Pressure    -> master: pressure PID   / slave: fuel flow PID
    Drum level  -> master: level PID      / slave: feedwater flow PID
    Steam temp  -> single PID             / output: spray valve position
"""

from dataclasses import dataclass

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
    prev_output: float = 0.0  # Output at previous step (for bumpless transfer)
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
        self._manual_output: float | None = None  # set when in MANUAL mode

    # ─── Mode control ─────────────────────────────────────────────────────────

    def set_manual(self, output: float) -> None:
        """
        Switch to MANUAL mode with a fixed output value.

        The integrator is back-calculated to match the manual output,
        ensuring bumpless transfer back to AUTO.

        Args:
            output: Fixed output value [output_min, output_max].
        """
        output = max(self.params.output_min, min(self.params.output_max, output))
        self._manual_output = output
        # Back-calculate integrator for bumpless AUTO resumption
        self.state.integral = output

    def set_auto(self) -> None:
        """Switch to AUTO mode (resume PID control)."""
        self._manual_output = None

    @property
    def is_manual(self) -> bool:
        """True if controller is in MANUAL mode."""
        return self._manual_output is not None

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(self, initial_output: float = 0.0) -> None:
        """
        Reset controller state.

        Args:
            initial_output: Pre-load the integrator to this output value.
                            Avoids a large transient on first AUTO step.
        """
        self.state = PIDState(integral=initial_output)
        self._manual_output = None

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
            Controller output, clamped to [output_min, output_max].
        """
        if self._manual_output is not None:
            self.state.prev_output = self._manual_output
            return self._manual_output

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
        output = max(p.output_min, min(p.output_max, output_raw))

        # ── Anti-windup: back-calculate integrator ────────────────────────────
        if p.anti_windup and output != output_raw:
            # Clamp integrator so that output_pd + integral == output
            self.state.integral = output - output_pd

        # ── Save state ────────────────────────────────────────────────────────
        self.state.prev_error = error
        self.state.prev_derivative = derivative
        self.state.prev_output = output

        return output


# ─── Cascade PID ──────────────────────────────────────────────────────────────


@dataclass
class CascadePIDParameters:
    """
    Tuning parameters for a cascade (master-slave) PID pair.

    Master (outer loop): controls process variable (e.g. pressure, level).
    Slave  (inner loop): controls intermediate variable (e.g. fuel flow, feed flow).
    """

    master: PIDParameters  # Outer (slow) loop
    slave: PIDParameters  # Inner (fast) loop

    # Setpoint limits for the slave loop output of the master [process units].
    # Master output is clamped to this range before being passed to the slave
    # as its setpoint.  Prevents the master from demanding physically impossible
    # intermediate setpoints.
    slave_setpoint_min: float = 0.0
    slave_setpoint_max: float = 1.0


class CascadePIDController:
    """
    Cascade (master-slave) PID controller.

    The master PID controls the primary process variable (e.g. drum pressure).
    Its output becomes the setpoint for the slave PID, which controls a faster
    inner variable (e.g. fuel flow rate) and drives the final actuator.

    Cascade control advantages over single-loop PID:
        1. Inner loop rejects disturbances before they reach the outer loop.
        2. Outer loop can be tuned more aggressively (inner loop is faster).
        3. Actuator nonlinearity is handled by the inner loop.

    Usage:
        params = CascadePIDParameters(
            master=PIDParameters(kp=0.5, ki=0.02, kd=1.0,
                                 output_min=0.0, output_max=10.0),
            slave=PIDParameters(kp=2.0,  ki=0.5,  kd=0.1,
                                output_min=0.0, output_max=1.0),
            slave_setpoint_min=0.0,
            slave_setpoint_max=10.0,
        )
        cascade = CascadePIDController(params)
        valve_cmd = cascade.step(
            primary_setpoint=140e5,    # desired drum pressure [Pa]
            primary_measurement=138e5, # actual drum pressure [Pa]
            inner_measurement=6.5,     # actual fuel flow [kg/s]
            dt=1.0,
        )
    """

    def __init__(self, params: CascadePIDParameters) -> None:
        self.params = params
        self.master = PIDController(params.master)
        self.slave = PIDController(params.slave)

    def step(
        self,
        primary_setpoint: float,
        primary_measurement: float,
        inner_measurement: float,
        dt: float,
    ) -> float:
        """
        Compute one cascade PID step.

        Args:
            primary_setpoint:     Desired value for outer loop (e.g. pressure [Pa]).
            primary_measurement:  Measured value for outer loop.
            inner_measurement:    Measured value for inner loop (e.g. fuel flow [kg/s]).
            dt:                   Time step [s].

        Returns:
            Final actuator command, clamped to slave output range [0, 1].
        """
        # ── Master: primary variable -> slave setpoint ─────────────────────────
        slave_setpoint_raw = self.master.step(
            setpoint=primary_setpoint,
            measurement=primary_measurement,
            dt=dt,
        )
        # Clamp master output to valid slave setpoint range
        slave_setpoint = max(
            self.params.slave_setpoint_min,
            min(self.params.slave_setpoint_max, slave_setpoint_raw),
        )

        # ── Slave: inner variable -> actuator command ──────────────────────────
        actuator_command = self.slave.step(
            setpoint=slave_setpoint,
            measurement=inner_measurement,
            dt=dt,
        )

        return actuator_command

    def set_manual(self, output: float) -> None:
        """Set both loops to MANUAL with bumpless transfer."""
        self.slave.set_manual(output)

    def set_auto(self) -> None:
        """Resume AUTO mode on both loops."""
        self.master.set_auto()
        self.slave.set_auto()

    def reset(self) -> None:
        """Reset both controllers."""
        self.master.reset()
        self.slave.reset()
