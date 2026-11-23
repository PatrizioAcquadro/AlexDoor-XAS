"""Causal stopped-response input/lifecycle contracts; no simulation qualification."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.assets.purdue import ARM_JOINTS
from alexdoor_xas.perception.contracts import RobotFeedback
from alexdoor_xas.perception.load import LoadCalibration, RobotLoadSample
from alexdoor_xas.perception.stopped_response import CausalResidualStopController


@pytest.fixture
def inputs():
    calibration = LoadCalibration(
        np.full(7, 0.01),
        -np.ones(7),
        np.ones(7),
        np.ones(7),
        np.full(7, 2.0),
        -1,
        verified=True,
        stop_verified=True,
        velocity_residual_nm_per_rad_s=np.full(7, 0.02),
        acceleration_residual_nm_per_rad_s2=np.full(7, 0.03),
    )
    sample = RobotLoadSample(
        RobotFeedback(
            1.0,
            1.0,
            ARM_JOINTS,
            np.full(7, -0.1),
            "simulated_actuator_feedback",
            "joint_output",
            True,
        ),
        np.zeros(7),
        np.zeros((6, 7)),
        np.zeros(7),
        np.zeros(7),
        np.zeros(7),
    )
    controller = CausalResidualStopController(calibration)
    controller.reset(7)
    return calibration, sample, controller


def test_cold_start_completed_residual_one_coupled_budget_and_pure_preview(inputs):
    _, sample, controller = inputs
    assert not controller.preview(1.0, 7)[1]["valid"]
    sample.dq[0], sample.ddq[1] = 0.5, 1.0
    controller.observe(sample, 7)
    for _ in range(3):
        correction, receipt = controller.preview(1.0, 7)
        np.testing.assert_array_equal(receipt["component_bound_nm"], np.full(7, 0.05))
        np.testing.assert_array_equal(correction, np.full(7, -0.05))
    controller.observe(sample, 7)
    assert controller.observations == 1
    assert not controller.preview(1.0 + 1 / 240, 7)[1]["valid"]
    controller.reset(8)
    assert controller.last is None and not controller.preview(2.0, 8)[0].any()


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", "commanded_effort"),
        ("reason", "incomplete"),
        ("units", "N"),
        ("semantics", "motor_command"),
        ("sign_convention", "unknown"),
        ("joint_names", tuple(reversed(ARM_JOINTS))),
        ("healthy", False),
        ("torque_nm", None),
    ],
)
def test_unverified_feedback_never_becomes_correction(inputs, field, value):
    _, sample, controller = inputs
    sample = replace(sample, feedback=replace(sample.feedback, **{field: value}))
    controller.observe(sample, 7)
    correction, receipt = controller.preview(1.0, 7)
    assert not correction.any() and not receipt["valid"]


@pytest.mark.parametrize("field,value", [("q", 1.01), ("dq", 1.01), ("ddq", 2.01)])
def test_no_weak_or_outside_calibration_extrapolation(inputs, field, value):
    _, sample, controller = inputs
    getattr(sample, field)[0] = value
    controller.observe(sample, 7)
    correction, receipt = controller.preview(1.0, 7)
    assert not correction.any() and receipt["reason"] == "outside_existing_load_calibration"


def test_same_clock_conflict_reversed_clock_and_cross_generation_rejected(inputs):
    _, sample, controller = inputs
    controller.observe(sample, 7)
    conflict = replace(sample, feedback=replace(sample.feedback, torque_nm=np.ones(7)))
    with pytest.raises(ValueError, match="different_stop_feedback_at_same_time"):
        controller.observe(conflict, 7)
    with pytest.raises(ValueError, match="stop_feedback_time_reversed"):
        controller.observe(replace(sample, feedback=replace(sample.feedback, timestamp_s=0.9)), 7)
    with pytest.raises(ValueError, match="stop_feedback_generation_changed"):
        controller.observe(sample, 8)
    with pytest.raises(ValueError, match="stop_dispatch_generation_changed"):
        controller.preview(1.0, 8)


@pytest.mark.parametrize("stamp,available", [(np.nan, 1.0), (1.0, np.inf), (1.1, 1.0)])
def test_invalid_clocks_rejected(inputs, stamp, available):
    _, sample, controller = inputs
    with pytest.raises(ValueError, match="invalid_stop_feedback_clock"):
        controller.observe(
            replace(
                sample, feedback=replace(sample.feedback, timestamp_s=stamp, available_s=available)
            ),
            7,
        )


def test_late_feedback_and_confidence_deadzone(inputs):
    _, sample, controller = inputs
    sample = replace(
        sample, feedback=replace(sample.feedback, available_s=1.001, torque_nm=np.full(7, -0.005))
    )
    controller.observe(sample, 7)
    assert not controller.preview(1.0, 7)[1]["valid"]
    controller.reset(7)
    sample = replace(sample, feedback=replace(sample.feedback, available_s=1.0))
    controller.observe(sample, 7)
    assert controller.preview(1.0, 7)[1]["valid"]
    assert not controller.preview(1.0, 7)[0].any()


def test_existing_halo_requires_full_ten_newton_support(inputs):
    calibration, sample, _ = inputs
    tolerance = np.zeros((2, 7))
    tolerance[1, 0] = 0.1
    caps = np.full((2, 7), 10.0)
    calibration = replace(
        calibration, position_tolerance_rad=tolerance, position_halo_force_upper_max_n=caps
    )
    sample.q[0] = 1.05
    for cap, valid in [(10.0, True), (1.0, False)]:
        caps[1, 0] = cap
        controller = CausalResidualStopController(calibration)
        controller.reset(7)
        controller.observe(sample, 7)
        assert controller.preview(1.0, 7)[1]["valid"] is valid


@pytest.mark.parametrize(
    "change",
    [
        {"verified": False},
        {"stop_verified": False},
        {"external_sign": 0},
        {"residual_bound_nm": -np.ones(7)},
        {"velocity_residual_nm_per_rad_s": -np.ones(7)},
    ],
)
def test_unverified_or_invalid_calibration_rejected(inputs, change):
    calibration, _, _ = inputs
    with pytest.raises(ValueError):
        CausalResidualStopController(replace(calibration, **change))


def test_same_clock_origin_change_is_not_an_idempotent_duplicate(inputs):
    _, sample, controller = inputs
    controller.observe(sample, 7)
    changed_origin = replace(
        sample, feedback=replace(sample.feedback, source="reported_joint_state")
    )
    with pytest.raises(ValueError, match="different_stop_feedback_at_same_time"):
        controller.observe(changed_origin, 7)
