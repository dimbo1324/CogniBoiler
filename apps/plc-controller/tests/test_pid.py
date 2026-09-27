"""Unit tests for PIDController: correctness and edge cases."""

import pytest
from plc_controller.pid import PIDController, PIDParameters

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture  # type: ignore[misc]
def simple_pid() -> PIDController:
    """Basic PID with proportional-only tuning for predictable output."""
    params = PIDParameters(
        kp=1.0,
        ki=0.0,
        kd=0.0,
        output_min=0.0,
        output_max=10.0,
        anti_windup=True,
    )
    return PIDController(params)


@pytest.fixture  # type: ignore[misc]
def integrating_pid() -> PIDController:
    """PID with integral term for steady-state error elimination tests."""
    params = PIDParameters(
        kp=1.0,
        ki=0.5,
        kd=0.0,
        output_min=0.0,
        output_max=10.0,
        anti_windup=True,
    )
    return PIDController(params)


# ─── PID tests ────────────────────────────────────────────────────────────────


class TestPID:
    """
    Verify single PID controller correctness.
    """

    def test_proportional_output_matches_kp_times_error(
        self, simple_pid: PIDController
    ) -> None:
        """
        With ki=kd=0, output must equal Kp × error.

        Physics: u = Kp × (SP − PV) = 1.0 × (10.0 − 7.0) = 3.0
        """
        output = simple_pid.step(setpoint=10.0, measurement=7.0, dt=1.0)
        assert abs(output - 3.0) < 1e-9, (
            f"Proportional output wrong: expected 3.0, got {output:.6f}"
        )

    def test_output_clamped_at_max(self, simple_pid: PIDController) -> None:
        """
        Output must not exceed output_max even with large error.
        """
        output = simple_pid.step(setpoint=100.0, measurement=0.0, dt=1.0)
        assert output <= simple_pid.params.output_max, (
            f"Output exceeded max: {output:.3f} > {simple_pid.params.output_max}"
        )

    def test_output_clamped_at_min(self, simple_pid: PIDController) -> None:
        """
        Output must not go below output_min even with negative error.
        """
        output = simple_pid.step(setpoint=0.0, measurement=100.0, dt=1.0)
        assert output >= simple_pid.params.output_min, (
            f"Output below min: {output:.3f} < {simple_pid.params.output_min}"
        )

    def test_integral_eliminates_steady_state_error(
        self, integrating_pid: PIDController
    ) -> None:
        """
        With ki > 0, repeated steps at fixed error must drive output up.

        Physics: integral accumulates -> output grows until error = 0.
        """
        outputs = [
            integrating_pid.step(setpoint=5.0, measurement=4.0, dt=1.0)
            for _ in range(20)
        ]
        # Output must increase over time due to integral action
        assert outputs[-1] > outputs[0], (
            f"Integral did not accumulate: "
            f"first={outputs[0]:.3f}, last={outputs[-1]:.3f}"
        )

    def test_anti_windup_prevents_integrator_overflow(self) -> None:
        """
        With anti_windup=True, integrator must not grow beyond what
        is needed to reach output_max.
        """
        params = PIDParameters(
            kp=1.0,
            ki=1.0,
            kd=0.0,
            output_min=0.0,
            output_max=1.0,
            anti_windup=True,
        )
        pid = PIDController(params)

        # Apply large sustained error for many steps
        for _ in range(100):
            pid.step(setpoint=100.0, measurement=0.0, dt=1.0)

        # Integral must be clamped — not hundreds of accumulated error
        assert pid.state.integral <= params.output_max + 1.0, (
            f"Integrator wound up: integral={pid.state.integral:.1f}"
        )

    def test_zero_error_gives_zero_proportional_output(
        self, simple_pid: PIDController
    ) -> None:
        """
        When setpoint equals measurement, proportional output must be zero.
        """
        output = simple_pid.step(setpoint=5.0, measurement=5.0, dt=1.0)
        assert abs(output) < 1e-9, f"Non-zero output at zero error: {output:.6f}"
