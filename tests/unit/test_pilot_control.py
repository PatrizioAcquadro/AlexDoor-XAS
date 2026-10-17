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
        max_joint_speed=0.5,
        orientation_tolerance=0.1,
        force_limit_n=10,
        loaded_force_n=1,
        contact_force_window_s=0.1,
        contact_gap_tolerance_m=0.01,
    )
    from alexdoor_xas.perception.contact import ContactMonitorConfig, ObservedContactMonitor

    monitor = ObservedContactMonitor(
        ContactMonitorConfig(0.008, 1, 0.5, 0.1, 0.04, 3, 0.03, 0.1),
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
    initial = (*args[:4], None, args[5])
    unloaded = replace(load, normal_lower_n=-0.05, force_upper_n=0.1)
    assert (
        checks.check(*initial, profile=OPERATIONAL_V1, feedback=load.feedback, load=unloaded) == ""
    )
    assert (
        checks.check(*initial, profile=OPERATIONAL_V1, feedback=load.feedback, load=load)
        == "unexpected_observed_load"
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
    peak = PurdueLoadSource(model.calibration, normal_angular_speed_rad_s=0.15)(io, load.feedback)
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


def test_measured_speed_guard_includes_both_native_substeps():
    from alexdoor_xas.perception.control import ObservedPurdueSafety

    setup = SimpleNamespace(max_joint_speed=0.5)
    checks = ObservedControlChecks(setup)
    sensor = dict(joint_position=np.zeros(9), joint_velocity=np.r_[0.52, np.zeros(8)])
    assert checks.check(sensor, None, None, None, None, None) == "actual_joint_speed_limit"
    io = SimpleNamespace(
        env=SimpleNamespace(
            load_samples=[
                SimpleNamespace(dq=np.r_[0.6, np.zeros(6)]),
                SimpleNamespace(dq=np.zeros(7)),
            ]
        )
    )
    safety = ObservedPurdueSafety(io, setup, None)
    assert safety.check(None) == "actual_joint_speed_limit"


def test_dynamic_residual_bounds_cover_coupling_without_inflating_static_load():
    model, jac, baseline = load_inputs()
    calibration = replace(
        model.calibration,
        velocity_residual_nm_per_rad_s=np.full(7, 0.1),
        acceleration_residual_nm_per_rad_s2=np.full(7, 0.02),
    )
    model = ResidualLoadModel(calibration)
    static = model.estimate(
        baseline.feedback, np.zeros(7), jac, np.zeros(7), np.zeros(7), np.zeros(7), [1, 0, 0]
    )
    moving = model.estimate(
        baseline.feedback,
        np.zeros(7),
        jac,
        np.zeros(7),
        np.r_[0, 0.5, np.zeros(5)],
        np.r_[0, 0, 0.2, np.zeros(4)],
        [1, 0, 0],
    )
    assert static.normal_lower_n == baseline.normal_lower_n
    np.testing.assert_allclose(moving.residual_bound_nm, np.full(7, 0.104))
    assert moving.normal_lower_n < static.normal_lower_n
    assert moving.force_upper_n > static.force_upper_n
    model = ResidualLoadModel(replace(calibration, velocity_residual_nm_per_rad_s=-np.ones(7)))
    with pytest.raises(ValueError, match="dynamic_residual"):
        model.estimate(
            baseline.feedback, np.zeros(7), jac, np.zeros(7), np.zeros(7), np.zeros(7), [1, 0, 0]
        )


def test_safety_uses_io_geometry_profile_with_an_independent_provider_recipe():
    from alexdoor_xas.perception.control import ObservedPurdueSafety

    estimate, proposal, _, _ = local_inputs()
    provider = SimpleNamespace(binding=SimpleNamespace(config={}), last_estimate=estimate)
    io = SimpleNamespace(
        profile=OPERATIONAL_V1,
        env=SimpleNamespace(
            load_samples=[],
            limits=np.zeros((1, 7, 2)),
            targets=np.zeros((1, 9)),
            neck_ids=[7, 8],
            robot=SimpleNamespace(
                data=SimpleNamespace(joint_pos=np.zeros((1, 4))),
                find_joints=lambda names, **_: (list(range(4)), names),
            ),
        ),
        load_source=None,
        robot_feedback=lambda: None,
        tool_pose=lambda: proposal.world_poses[0],
        observe=lambda: {},
        admission=None,
    )
    safety = ObservedPurdueSafety(io, SimpleNamespace(max_joint_speed=0.5), provider)
    safety.checks.check = lambda *args, **kwargs: kwargs["profile"]
    assert safety.check(None) == OPERATIONAL_V1


def test_axis_free_admission_spends_measured_error_against_the_actual_sweep():
    estimate, proposal, evidence, limits = local_inputs()
    patch = estimate.local.patches[0]
    patch = replace(patch, pose_support=replace(patch.pose_support, position_bound_m=0.025))
    estimate = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    # A large but finite frame error cannot fit inside this narrow clearance.
    evidence = replace(evidence, finger_clearance_m=(0.03, 0.03), collision_clearance_m=0.03)
    assert not admit_local_action(estimate, proposal, evidence, limits, 1, generation=0).admitted
    wide = replace(evidence, finger_clearance_m=(0.1, 0.1), collision_clearance_m=0.1)
    assert admit_local_action(estimate, proposal, wide, limits, 1, generation=0).admitted
    unknown = replace(patch, pose_support=replace(patch.pose_support, position_bound_m=None))
    estimate = replace(estimate, local=replace(estimate.local, patches=(unknown,)))
    assert not admit_local_action(estimate, proposal, wide, limits, 1, generation=0).admitted


@pytest.mark.parametrize("change", ["uncertainty", "clearance", "ownership"])
def test_local_receipt_spends_existing_margin_on_current_support(change):
    estimate, proposal, evidence, limits = local_inputs()
    receipt = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    patch = estimate.local.patches[0]
    if change == "uncertainty":
        patch = replace(patch, pose_support=replace(patch.pose_support, position_bound_m=0.0041))
    elif change == "clearance":
        patch = replace(patch, finger_clearance_m=(0.0399, 0.0399))
    else:
        patch = replace(patch, ownership="observed_moving_rigid_candidate")
    current = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    require_current_admission(receipt, current, 1 + 1 / 60)


@pytest.mark.parametrize("change", ["uncertainty", "clearance", "missing", "ownership"])
def test_local_receipt_refuses_exhausted_margin_or_missing_evidence(change):
    estimate, proposal, evidence, limits = local_inputs()
    receipt = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    patch = estimate.local.patches[0]
    if change == "uncertainty":
        patch = replace(patch, pose_support=replace(patch.pose_support, position_bound_m=0.05))
        reason = "margin_exhausted"
    elif change == "clearance":
        patch = replace(patch, finger_clearance_m=(0.001, 0.001))
        reason = "margin_exhausted"
    elif change == "missing":
        patch = replace(patch, pose_support=replace(patch.pose_support, position_bound_m=None))
        reason = "missing_or_excessive_field_bound"
    else:
        patch = replace(patch, ownership="unconfirmed")
        reason = "reference_changed"
    current = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    with pytest.raises(ValueError, match=reason):
        require_current_admission(receipt, current, 1 + 1 / 60)


def test_scalar_tool_normal_calibration_is_optional_and_weak_load_bounded():
    original_model, jacobian, original_load = load_inputs()
    calibration, feedback = original_model.calibration, original_load.feedback
    # Identity wrench coordinates with a redundant seventh measurement.
    q = dq = ddq = np.zeros(7)
    weak = replace(feedback, torque_nm=np.array([-0.1, 0, 0, 0, 0, 0, 0]))
    normal = np.array([1.0, 0, 0])
    base = ResidualLoadModel(calibration).estimate(weak, np.zeros(7), jacobian, q, dq, ddq, normal)
    calibrated = replace(
        calibration,
        tool_normal_residual_n=np.array([0.005, 0.1, 0.01]),
        tool_normal_force_upper_max_n=1.0,
    )
    model = ResidualLoadModel(calibrated)
    tight = model.estimate(
        weak, np.zeros(7), jacobian, q, dq, ddq, normal, tool_normal_world=normal
    )
    assert tight.normal_lower_n > base.normal_lower_n
    assert tight.force_upper_n == base.force_upper_n
    assert tight.model_verified == base.model_verified
    with pytest.raises(ValueError):
        model.estimate(weak, np.zeros(7), jacobian, q, dq, ddq, normal)
    strong = model.estimate(feedback, np.zeros(7), jacobian, q, dq, ddq, normal)
    original = ResidualLoadModel(calibration).estimate(
        feedback, np.zeros(7), jacobian, q, dq, ddq, normal
    )
    assert strong.normal_lower_n == original.normal_lower_n
    np.testing.assert_array_equal(tight.residual_error_nm, base.residual_error_nm)


@pytest.mark.parametrize("opening", [0.002, np.nan])
def test_observed_monitor_preserves_closed_gripper_requirement(opening):
    from alexdoor_xas.perception.control import ObservedPurdueSafety

    robot = SimpleNamespace(
        data=SimpleNamespace(joint_pos=np.array([[0.0, 0, opening, 0]])),
        find_joints=lambda names, **_: (list(range(4)), names),
    )
    io = SimpleNamespace(env=SimpleNamespace(robot=robot, load_samples=[]))
    safety = ObservedPurdueSafety(io, SimpleNamespace(max_joint_speed=0.5), None)
    assert safety.check(None) == "gripper_opened"
    missing = ObservedPurdueSafety(
        SimpleNamespace(env=SimpleNamespace(load_samples=[])),
        SimpleNamespace(max_joint_speed=0.5),
        None,
    )
    assert missing.check(None) == "missing_gripper_proprioception"


@pytest.mark.parametrize(
    "translation,rotation,violation", [(0.0005, 0.0, False), (0.002, 0.0, True), (0.0, 0.002, True)]
)
def test_actual_response_is_checked_after_real_dispatch_and_latched(
    translation, rotation, violation
):
    from scipy.spatial.transform import Rotation

    from alexdoor_xas.action.frames import ObjectFrame
    from alexdoor_xas.policies.purdue import PurdueIO
    from alexdoor_xas.policies.rollout import Command

    estimate, proposal, evidence, limits = local_inputs(A1_JOINT_DELTA)
    receipt = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    events = []
    env = SimpleNamespace(
        motion_stopped=False, capture=SimpleNamespace(sample=SimpleNamespace(time_s=1.0))
    )
    env.step_a1 = lambda _: events.append("executed")

    def stop():
        events.append("stop")
        env.motion_stopped = True

    env.stop_motion = stop
    io = PurdueIO.__new__(PurdueIO)
    io.env, io.dt, io.profile = env, 1 / 60, OPERATIONAL_V1
    io.response_violation = None
    io.safety = SimpleNamespace(
        provider=SimpleNamespace(last_estimate=estimate),
        before_command=lambda _: "",
        check=lambda _: "",
    )
    goal = proposal.world_poses[-1]
    io.preview = lambda _: goal
    io.tool_pose = lambda: ObjectFrame(
        goal.origin + [translation, 0, 0],
        Rotation.from_rotvec([0, 0, rotation]).as_matrix() @ goal.rot,
    )
    command = Command("joints", proposal.action, admission=receipt)
    io.step_recorder = lambda recorded, goal: events.append(("recorded", recorded, goal))
    io.execute(command)
    assert events[1] == ("recorded", command, None)
    if violation:
        assert events[2] == "stop"
    assert events.count("executed") == 1
    assert env.motion_stopped == violation
    assert io.safety_reason(None) == ("actual_robot_response_exceeded" if violation else "")
    if violation:
        with pytest.raises(RuntimeError, match="motion_stopped"):
            io.execute(command)
        assert events.count("executed") == 1


def test_loaded_alignment_uses_the_supported_normal_without_discarding_full_pose_error():
    estimate, proposal, evidence, limits = local_inputs()
    patch = estimate.local.patches[0]
    patch = replace(
        patch,
        pose_support=replace(patch.pose_support, rotation_bound_rad=0.2),
        normal_rotation_bound_rad=0.01,
    )
    estimate = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    wide = replace(evidence, finger_clearance_m=(0.1, 0.1), collision_clearance_m=0.1)
    receipt = admit_local_action(estimate, proposal, wide, limits, 1, generation=0)
    assert receipt.admitted
    setup = SimpleNamespace(max_joint_speed=0.5, orientation_tolerance=0.087, force_limit_n=10)
    checks = ObservedControlChecks(setup)
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
    # Normal alignment passes; the independent contact/load monitor is still required.
    assert (
        checks.check(
            *args, profile=OPERATIONAL_V1, feedback=load.feedback, admission=receipt, load=load
        )
        == "operational_contact_monitor_unconfigured"
    )
    ambiguous = replace(patch, normal_rotation_bound_rad=0.2)
    bad = replace(estimate, local=replace(estimate.local, patches=(ambiguous,)))
    assert (
        checks.check(
            sensor, bad, *args[2:], profile=OPERATIONAL_V1, feedback=load.feedback, load=load
        )
        == "observed_contact_orientation"
    )
    # Full pose error still spends finite footprint/sweep clearance.
    narrow = replace(wide, finger_clearance_m=(0.005, 0.005))
    assert not admit_local_action(estimate, proposal, narrow, limits, 1, generation=0).admitted


def test_native_normal_load_uses_its_bound_and_preserves_full_wrench_peak():
    from alexdoor_xas.perception.load import PurdueLoadSource, RobotLoadSample

    model, jac, load = load_inputs()
    samples = [
        RobotLoadSample(
            replace(load.feedback, timestamp_s=t, available_s=t),
            np.zeros(7),
            jac,
            np.zeros(7),
            np.zeros(7),
            np.zeros(7),
        )
        for t in (1 - 1 / 120, 1)
    ]
    estimate = local_inputs()[0]
    patch = estimate.local.patches[0]
    patch = replace(patch, pose_support=replace(patch.pose_support, rotation_bound_rad=0.2))
    estimate = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    provider = SimpleNamespace(last_estimate=estimate)
    io = SimpleNamespace(
        admission=None,
        env=SimpleNamespace(load_samples=samples),
        safety=SimpleNamespace(provider=provider),
    )
    source = PurdueLoadSource(model.calibration, normal_angular_speed_rad_s=0.15)
    conservative = source(io, load.feedback)
    supported = replace(patch, normal_rotation_bound_rad=0.01)
    provider.last_estimate = replace(estimate, local=replace(estimate.local, patches=(supported,)))
    projected = source(io, load.feedback)
    assert projected.normal_lower_n > conservative.normal_lower_n
    assert projected.force_upper_n == conservative.force_upper_n
    assert len(projected.normal_samples) == 2
    expected = [
        load.normal_lower_n
        - load.force_upper_n
        * 2
        * np.sin((0.01 + 0.15 * (load.feedback.timestamp_s - s.feedback.timestamp_s)) / 2)
        for s in samples
    ]
    assert projected.normal_lower_n == pytest.approx(min(expected))
    assert [v for _, _, v in projected.normal_samples] == pytest.approx(expected)
    assert provider.last_estimate.local.patches[0].pose_support.rotation_bound_rad == 0.2


def test_independent_robot_sweep_proof_keeps_its_own_residual_margin():
    estimate, proposal, evidence, limits = local_inputs()
    evidence = replace(evidence, collision_clearance_m=None, swept_collision_margin_m=0.0001)
    receipt = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    assert receipt.admitted
    assert dict(receipt.margins_m)["independent_robot_sweep"] == 0.0001
    require_current_admission(receipt, estimate, 1.0)
    for invalid in (0.0, -0.001, np.nan):
        bad = replace(evidence, swept_collision_margin_m=invalid)
        assert not admit_local_action(estimate, proposal, bad, limits, 1, generation=0).admitted
    # Neither a valid robot proof nor its margin can replace finite material coverage.
    narrow = replace(evidence, finger_clearance_m=(0.001, 0.001))
    assert not admit_local_action(estimate, proposal, narrow, limits, 1, generation=0).admitted
    unknown = replace(evidence, unknown_in_envelope=True)
    assert not admit_local_action(estimate, proposal, unknown, limits, 1, generation=0).admitted


def test_unloaded_full_robot_proof_does_not_require_an_irrelevant_contact_footprint():
    estimate, proposal, evidence, limits = local_inputs()
    patch = replace(
        estimate.local.patches[0], normal_position_bound_m=0.005, normal_rotation_bound_rad=0.01
    )
    estimate = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    proposal = replace(proposal, loaded=False)
    free = replace(
        evidence,
        finger_clearance_m=(None, None),
        collision_clearance_m=None,
        swept_collision_margin_m=0.02,
        unloaded_separation_margin_m=0.0001,
        unloaded_normal_lever_m=0.8,
        unloaded_temporal_bound_m_rad=(limits.relative_speed_bound_m_s / 60, 0.0),
    )
    receipt = admit_local_action(estimate, proposal, free, limits, 1, generation=0)
    assert receipt.admitted
    require_current_admission(receipt, estimate, 1 + 1 / 60)
    for bad in [
        replace(free, swept_collision_margin_m=None),
        replace(free, unloaded_normal_lever_m=None),
        replace(free, unloaded_separation_margin_m=0),
        replace(free, unknown_in_envelope=True),
        replace(free, unloaded_temporal_bound_m_rad=(-1.0, 0.0)),
        replace(free, unloaded_temporal_bound_m_rad=(float("nan"), 0.0)),
    ]:
        assert not admit_local_action(estimate, proposal, bad, limits, 1, generation=0).admitted
    assert not admit_local_action(
        estimate, replace(proposal, loaded=True), free, limits, 1, generation=0
    ).admitted
    changed = replace(patch, normal_position_bound_m=0.025)
    with pytest.raises(ValueError, match="unloaded_separation_exhausted"):
        require_current_admission(
            receipt,
            replace(estimate, local=replace(estimate.local, patches=(changed,))),
            1 + 1 / 60,
        )


@pytest.mark.parametrize("start", [25.216666666666665, 130.0, 140.0])
def test_control_receipt_tolerates_actual_float_clock_roundoff_without_an_extra_tick(start):
    estimate, proposal, evidence, limits = local_inputs()
    limits = replace(limits, duration_s=1 / 60)
    patch = estimate.local.patches[0]
    support = replace(patch.pose_support, acquired_s=start, supported_s=start, available_s=start)
    patch = replace(patch, geometry_support=support, identity_support=support, pose_support=support)
    state = replace(
        estimate.local,
        patches=(patch,),
        selection=replace(estimate.local.selection, selected_s=start),
    )
    estimate = replace(estimate, timestamp_s=start, local=state)
    proposal = replace(proposal, times_s=np.array([start, start + 1 / 60]))
    evidence = replace(evidence, support=support)
    receipt = admit_local_action(estimate, proposal, evidence, limits, start, generation=0)
    assert receipt.admitted
    endpoint = (round(start * 60) + 1) / 60
    require_current_admission(receipt, estimate, endpoint)
    with pytest.raises(ValueError, match="outside_admitted_schedule"):
        require_current_admission(receipt, estimate, endpoint + 1 / 60)
    extra_tick = replace(proposal, times_s=np.array([start, start + 2 / 60]))
    refused = admit_local_action(estimate, extra_tick, evidence, limits, start, generation=0)
    assert not refused.admitted and refused.reason == "action_exceeds_declared_limits"
