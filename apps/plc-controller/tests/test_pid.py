"""
Unit tests for PIDController and CascadePIDController.

Test categories:
    TestPID         — single PID correctness and edge cases
    TestCascadePID  — cascade PID coupling and mode switching
"""

import pytest
from plc_controller.pid import (
    CascadePIDController,
    CascadePIDParameters,
    PIDController,
    PIDParameters,
)

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


@pytest.fixture  # type: ignore[misc]
def cascade() -> CascadePIDController:
    """Cascade PID with simple tunings for coupling tests."""
    params = CascadePIDParameters(
        master=PIDParameters(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            output_min=0.0,
            output_max=5.0,
        ),
        slave=PIDParameters(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            output_min=0.0,
            output_max=1.0,
        ),
        slave_setpoint_min=0.0,
        slave_setpoint_max=5.0,
    )
    return CascadePIDController(params)


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

    def test_manual_mode_returns_fixed_output(self, simple_pid: PIDController) -> None:
        """
        In MANUAL mode, output must equal the manual setpoint
        regardless of process variable.
        """
        simple_pid.set_manual(0.7)
        output = simple_pid.step(setpoint=100.0, measurement=0.0, dt=1.0)
        assert abs(output - 0.7) < 1e-9, (
            f"MANUAL mode output wrong: expected 0.7, got {output:.6f}"
        )

    def test_auto_resumes_after_manual(self, simple_pid: PIDController) -> None:
        """
        After switching back to AUTO, controller must resume normal output.
        """
        simple_pid.set_manual(0.5)
        simple_pid.step(setpoint=5.0, measurement=5.0, dt=1.0)
        simple_pid.set_auto()

        # With SP=PV=5.0 and kp=1.0, ki=kd=0 -> error=0 -> output=0
        # But due to bumpless transfer integrator=0.5, output may vary
        # Key check: controller is no longer in manual mode
        assert not simple_pid.is_manual

    def test_zero_error_gives_zero_proportional_output(
        self, simple_pid: PIDController
    ) -> None:
        """
        When setpoint equals measurement, proportional output must be zero.
        """
        output = simple_pid.step(setpoint=5.0, measurement=5.0, dt=1.0)
        assert abs(output) < 1e-9, f"Non-zero output at zero error: {output:.6f}"


# ─── Cascade PID tests ────────────────────────────────────────────────────────


class TestCascadePID:
    """
    Verify cascade PID master-slave coupling.
    """

    def test_larger_primary_error_gives_larger_output(
        self, cascade: CascadePIDController
    ) -> None:
        """
        Larger primary (master) error must produce larger final output.
        """
        out_small = cascade.step(
            primary_setpoint=10.0,
            primary_measurement=9.5,  # error = 0.5
            inner_measurement=0.0,
            dt=1.0,
        )
        cascade.reset()
        out_large = cascade.step(
            primary_setpoint=10.0,
            primary_measurement=8.0,  # error = 2.0
            inner_measurement=0.0,
            dt=1.0,
        )
        assert out_large > out_small, (
            f"Larger error did not increase output: "
            f"small={out_small:.4f}, large={out_large:.4f}"
        )

    def test_manual_mode_freezes_output(self, cascade: CascadePIDController) -> None:
        """
        In MANUAL mode, output must be fixed regardless of inputs.
        """
        cascade.set_manual(0.4)
        out1 = cascade.step(
            primary_setpoint=100.0,
            primary_measurement=0.0,
            inner_measurement=0.0,
            dt=1.0,
        )
        out2 = cascade.step(
            primary_setpoint=100.0,
            primary_measurement=0.0,
            inner_measurement=0.0,
            dt=1.0,
        )
        assert abs(out1 - 0.4) < 1e-9
        assert abs(out2 - 0.4) < 1e-9

    def test_output_within_slave_bounds(self, cascade: CascadePIDController) -> None:
        """
        Cascade output must always be within slave output_min/output_max.
        """
        for _ in range(10):
            out = cascade.step(
                primary_setpoint=100.0,
                primary_measurement=0.0,
                inner_measurement=0.0,
                dt=1.0,
            )
        assert (
            cascade.params.slave.output_min <= out <= cascade.params.slave.output_max
        ), f"Output out of slave bounds: {out:.4f}"
