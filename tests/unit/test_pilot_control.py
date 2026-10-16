"""Axis-free admission and robot-only load/stop contracts; no physical qualification."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_local_contact import supported_patch

from alexdoor_xas.action.b1 import apply_pose_delta
from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA, A3_OBJ_REL_EE_DELTA
from alexdoor_xas.assets.purdue import ARM_JOINTS
from alexdoor_xas.perception.admission import (
    ActionLimits,
    ActionProposal,
    require_current_admission,
)
from alexdoor_xas.perception.contracts import (
    OPERATIONAL_V1,
    DoorEstimate,
    LocalContactSelection,
    LocalMaterialState,
    RobotFeedback,
)
from alexdoor_xas.perception.control import ObservedControlChecks
from alexdoor_xas.perception.load import LoadCalibration, ResidualLoadModel
from alexdoor_xas.perception.local_admission import LocalActionSupport, admit_local_action
from alexdoor_xas.policies.observations import PolicyObservation
from alexdoor_xas.policies.rollout import ActionAdapter


def local_inputs(space=A2_EE_DELTA):
    patch = replace(
        supported_patch("contact"), tracking_state="tracking", ownership="observed_rigid_material"
    )
    state = LocalMaterialState(
        0, "contact", (patch,), LocalContactSelection("s", "contact", 1, "policy")
    )
    estimate = DoorEstimate(1, False, "local_only", local=state)
    row = np.array([0.001, 0, 0, 0, 0, 0]) if space == A2_EE_DELTA else np.zeros(7)
    goal = apply_pose_delta(patch.world_pose, row) if space == A2_EE_DELTA else patch.world_pose
    proposal = ActionProposal(
        "policy", space, row, np.array([1, 1 + 1 / 60]), (patch.world_pose, goal), "s", True
    )
    evidence = LocalActionSupport(
        patch.pose_support, proposal.world_poses, (0.04, 0.04), 0.04, False, 0.001, 0.001, 0.05
    )
    limits = ActionLimits(0.2, 0.01, 0.1, 0.1, 0.001, 0.001, 0.01, 0.001)
    return estimate, proposal, evidence, limits


@pytest.mark.parametrize("space", [A1_JOINT_DELTA, A2_EE_DELTA])
def test_hingeless_admission_is_bounded_and_routes_actual_action(space):
    estimate, proposal, evidence, limits = local_inputs(space)
    receipt = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    assert receipt.admitted and not receipt.provisional
    assert estimate.frame is None and estimate.operational is None
    observation = PolicyObservation(1, 1, np.zeros(22), estimate, "local_only", OPERATIONAL_V1, 0)
    assert not observation.valid
    command = ActionAdapter(space, 0.15, profile=OPERATIONAL_V1).command(
        observation,
        proposal.world_poses[0],
        action=proposal.action,
        remaining_ticks=1,
        admission=receipt,
    )
    assert command.admission is receipt
    require_current_admission(receipt, estimate, 1 + 1 / 60)
    with pytest.raises(ValueError, match="outside_admitted_schedule"):
        require_current_admission(receipt, estimate, 1.1)
    patch = estimate.local.patches[0]
    changed = replace(
        patch, world_pose=replace(patch.world_pose, origin=patch.world_pose.origin + 0.2)
    )
    with pytest.raises(ValueError, match="response_exceeded"):
        require_current_admission(
            receipt, replace(estimate, local=replace(estimate.local, patches=(changed,))), 1
        )


@pytest.mark.parametrize(
    "defect", ["unknown", "clearance", "response", "stop", "source", "geometry", "A3", "separation"]
)
def test_local_admission_never_fills_missing_support(defect):
    estimate, proposal, evidence, limits = local_inputs()
    if defect == "unknown":
        evidence = replace(evidence, unknown_in_envelope=True)
    elif defect == "clearance":
        evidence = replace(evidence, finger_clearance_m=(None, 0.04))
    elif defect == "response":
        evidence = replace(evidence, response_position_bound_m=None)
    elif defect == "stop":
        limits = replace(limits, stop_travel_m=None)
    elif defect == "source":
        proposal = replace(proposal, source="diagnostic")
    elif defect == "geometry":
        patch = replace(estimate.local.patches[0], geometry_support=None)
        estimate = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    elif defect == "A3":
        proposal = replace(proposal, action_space=A3_OBJ_REL_EE_DELTA)
    else:
        proposal = replace(proposal, loaded=False)
    assert not admit_local_action(estimate, proposal, evidence, limits, 1, generation=0).admitted


def load_inputs(*, verified=True, stop=True):
    calibration = LoadCalibration(
        np.full(7, 0.05), -np.ones(7), np.ones(7), np.ones(7), np.ones(7), 1, verified, stop
    )
    feedback = RobotFeedback(
        1,
        1,
        ARM_JOINTS,
        np.array([-4.0, 0, 0, 0, 0, 0, 0]),
        "simulated_actuator_feedback",
        "joint_output",
        True,
    )
    jac = np.c_[np.eye(6), np.zeros(6)]
    model = ResidualLoadModel(calibration)
    estimate = model.estimate(
        feedback, np.zeros(7), jac, np.zeros(7), np.zeros(7), np.zeros(7), [1, 0, 0]
    )
    return model, jac, estimate


def test_robot_residual_bounds_force_and_requires_verified_model_and_stop():
    _, _, load = load_inputs()
    assert load.force_upper_n > 4 and load.normal_lower_n < 4
    assert load.admission(1, 10, require_contact=True, loaded_force_n=1).admitted
    assert not load.admission(1, 3).admitted
    assert not load.admission(1.1, 10).admitted
    assert not load_inputs(verified=False)[2].admission(1, 10).admitted
    assert not load_inputs(stop=False)[2].admission(1, 10).admitted
    model, jac, _ = load_inputs()
    with pytest.raises(ValueError, match="unobservable"):
        model.estimate(
            load.feedback, np.zeros(7), jac * 0, np.zeros(7), np.zeros(7), np.zeros(7), [1, 0, 0]
        )
    bad = replace(
        load.feedback, torque_nm=load.feedback.torque_nm + np.array([0, 0, 0, 0, 0, 0, 1])
    )
    with pytest.raises(ValueError, match="inconsistent"):
        model.estimate(bad, np.zeros(7), jac, np.zeros(7), np.zeros(7), np.zeros(7), [1, 0, 0])


def test_monitor_uses_robot_load_and_local_support_without_task_truth():
    estimate, proposal, evidence, limits = local_inputs()
    receipt = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    setup = SimpleNamespace(
        orientation_tolerance=0.1,
        force_limit_n=10,
        loaded_force_n=1,
        contact_force_window_s=0.1,
        contact_gap_tolerance_m=0.01,
    )
    from alexdoor_xas.perception.contact import ContactMonitorConfig, ObservedContactMonitor

    monitor = ObservedContactMonitor(
        ContactMonitorConfig(0.008, 1, 0.5, 0.1, 0.04, 3, 0.03),
        (np.array([[0, -0.01, 0]]), np.array([[0, 0.01, 0]])),
    )
    checks = ObservedControlChecks(setup, contact_monitor=monitor)
    sensor = dict(
        time_s=1,
        joint_position=np.zeros(9),
        joint_velocity=np.zeros(9),
        rgb=np.ones((1, 1, 3)),
        valid_depth=np.ones((1, 1), bool),
    )
    _, _, load = load_inputs()
    args = (
        sensor,
        estimate,
        proposal.world_poses[0],
        np.tile([-2, 2], (7, 1)),
        "contact",
        np.zeros(2),
    )
    assert (
        checks.check(
            *args, profile=OPERATIONAL_V1, feedback=load.feedback, admission=receipt, load=load
        )
        == ""
    )
    assert (
        checks.check(*args, profile=OPERATIONAL_V1, feedback=load.feedback, admission=receipt)
        == "load_control_unverified"
    )
    assert (
        checks.check(
            *args,
            profile=OPERATIONAL_V1,
            feedback=replace(load.feedback, timestamp_s=0.9),
            admission=receipt,
            load=load,
        )
        == "load_feedback_mismatch"
    )


def test_physics_peak_load_cannot_disappear_between_control_ticks(tmp_path):
    import h5py
    from test_b1_recording import obs

    from alexdoor_xas.perception.load import PurdueLoadSource, RobotLoadSample
    from alexdoor_xas.recording.b1 import B1Writer

    model, jac, load = load_inputs()
    first = replace(
        load.feedback,
        timestamp_s=1 - 1 / 120,
        available_s=1 - 1 / 120,
        torque_nm=load.feedback.torque_nm * 2,
    )
    samples = [
        RobotLoadSample(fb, np.zeros(7), jac, np.zeros(7), np.zeros(7), np.zeros(7))
        for fb in (first, load.feedback)
    ]
    estimate = local_inputs()[0]
    io = SimpleNamespace(
        admission=None,
        env=SimpleNamespace(load_samples=samples),
        safety=SimpleNamespace(provider=SimpleNamespace(last_estimate=estimate)),
    )
    peak = PurdueLoadSource(model.calibration)(io, load.feedback)
    assert peak.force_upper_n > 8
    assert not peak.admission(1, 6).admitted
    writer = B1Writer(tmp_path / "physics.hdf5", {}, dict(depth_interval_m=[0.1, 5]))
    writer.observe(obs(), {})
    writer.transition(dict(time_s=0, joint_target=np.zeros(7)), obs(1, 2), {})
    writer.physics_feedback(samples)
    writer.close()
    with h5py.File(tmp_path / "physics.hdf5") as source:
        assert len(source["physics_feedback/time_s"]) == 2
        np.testing.assert_array_equal(source["physics_feedback/torque_nm"][0], first.torque_nm)
