"""Axis-free admission and robot-only load/stop contracts; no physical qualification."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA, A3_OBJ_REL_EE_DELTA
from alexdoor_xas.perception.admission import (
    require_current_admission,
)
from alexdoor_xas.perception.contracts import (
    OPERATIONAL_V1,
)
from alexdoor_xas.perception.control import ObservedControlChecks
from alexdoor_xas.perception.load import ResidualLoadModel
from alexdoor_xas.perception.local_admission import admit_local_action
from alexdoor_xas.policies.observations import PolicyObservation
from alexdoor_xas.policies.rollout import ActionAdapter
from perception_helpers import load_inputs, local_inputs


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


@pytest.mark.parametrize("substeps", [2, 4])
def test_physics_peak_load_cannot_disappear_between_control_ticks(tmp_path, substeps):
    import h5py

    from alexdoor_xas.perception.load import PurdueLoadSource, RobotLoadSample
    from alexdoor_xas.recording.b1 import B1Writer
    from helpers import raw_observation as obs

    model, jac, load = load_inputs()
    first = replace(
        load.feedback,
        timestamp_s=1 - (substeps - 1) / (60 * substeps),
        available_s=1 - (substeps - 1) / (60 * substeps),
        torque_nm=load.feedback.torque_nm * 2,
    )
    samples = [
        RobotLoadSample(fb, np.zeros(7), jac, np.zeros(7), np.zeros(7), np.zeros(7))
        for fb in (
            first,
            *[
                replace(
                    load.feedback,
                    timestamp_s=1 - k / (60 * substeps),
                    available_s=1 - k / (60 * substeps),
                )
                for k in range(substeps - 2, -1, -1)
            ],
        )
    ]
    estimate = local_inputs()[0]
    io = SimpleNamespace(
        admission=None,
        env=SimpleNamespace(
            load_samples=samples,
            cfg=SimpleNamespace(
                sim=SimpleNamespace(dt=1 / (60 * len(samples))), decimation=len(samples)
            ),
        ),
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
        assert len(source["physics_feedback/time_s"]) == substeps
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
        env=SimpleNamespace(
            load_samples=samples,
            cfg=SimpleNamespace(
                sim=SimpleNamespace(dt=1 / (60 * len(samples))), decimation=len(samples)
            ),
        ),
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


@pytest.mark.parametrize(
    "force,acceleration,extension,accepted",
    [
        (0.1, 1.5, 2.0, True),
        (4.0, 1.5, 2.0, False),
        (0.1, 2.1, 2.0, False),
        (0.1, 1.5, None, False),
        (4.0, 0.5, 2.0, True),
    ],
)
def test_acceleration_extension_requires_bounded_weak_load(
    force, acceleration, extension, accepted
):
    original, jacobian, baseline = load_inputs()
    calibration = replace(
        original.calibration,
        weak_load_acceleration_max_rad_s2=None if extension is None else np.full(7, extension),
        tool_normal_force_upper_max_n=1.0,
        tool_normal_residual_n=np.array([0.005, 0.1, 0.01]),
    )
    model = ResidualLoadModel(calibration)
    feedback = replace(baseline.feedback, torque_nm=np.r_[-force, np.zeros(6)])
    arguments = (
        feedback,
        np.zeros(7),
        jacobian,
        np.zeros(7),
        np.zeros(7),
        np.full(7, acceleration),
        np.array([1.0, 0, 0]),
    )
    if accepted:
        result = model.estimate(*arguments, tool_normal_world=np.array([1.0, 0, 0]))
        assert result.model_verified and result.stop_verified
    else:
        with pytest.raises(ValueError, match="outside_verified_load_envelope"):
            model.estimate(*arguments, tool_normal_world=np.array([1.0, 0, 0]))


@pytest.mark.parametrize("extension", [np.full(7, 0.5), np.full(7, np.nan)])
def test_invalid_weak_load_acceleration_domain_is_rejected(extension):
    original, jacobian, baseline = load_inputs()
    model = ResidualLoadModel(
        replace(
            original.calibration,
            weak_load_acceleration_max_rad_s2=extension,
            tool_normal_force_upper_max_n=1.0,
        )
    )
    with pytest.raises(ValueError):
        model.estimate(
            baseline.feedback,
            np.zeros(7),
            jacobian,
            np.zeros(7),
            np.zeros(7),
            np.zeros(7),
            np.array([1.0, 0, 0]),
        )


def test_local_margin_charges_prepaid_capture_age_once_and_keeps_future_motion():
    from alexdoor_xas.perception.local_admission import _local_error

    estimate, _, evidence, limits = local_inputs()
    patch = estimate.local.patches[0]
    duration, age = 1 / 60, 4 / 60
    speed = limits.relative_speed_bound_m_s
    baseline = _local_error(patch, evidence, limits, 1, duration)
    aged = replace(
        patch,
        pose_support=replace(
            patch.pose_support,
            position_bound_m=patch.pose_support.position_bound_m + speed * age,
        ),
        pose_temporal_position_m=speed * age,
    )
    assert _local_error(aged, evidence, limits, 1 + age, duration) == pytest.approx(
        baseline + speed * age
    )
    assert _local_error(aged, evidence, limits, 1 + age + duration, duration) == pytest.approx(
        baseline + speed * (age + duration)
    )
    # Future material motion is supplied once, over the complete interval.
    longer = replace(
        evidence, response_position_bound_m=evidence.response_position_bound_m + speed * duration
    )
    assert _local_error(aged, longer, limits, 1 + age, 2 * duration) == pytest.approx(
        baseline + speed * (age + duration)
    )
    # Producers without a measured prepaid term retain the conservative old behavior.
    unknown = replace(aged, pose_temporal_position_m=None)
    assert _local_error(unknown, evidence, limits, 1 + age, duration) == pytest.approx(
        baseline + 2 * speed * age
    )
    for paid in [-1, np.nan, np.inf, aged.pose_support.position_bound_m + 1]:
        with pytest.raises(ValueError, match="invalid_prepaid_pose_growth"):
            _local_error(
                replace(aged, pose_temporal_position_m=paid), evidence, limits, 1 + age, duration
            )


@pytest.mark.parametrize(
    "force,position,acceleration,extension,accepted",
    [
        (0.1, -1.1, 0.5, -1.2, True),
        (4.0, -1.1, 0.5, -1.2, False),
        (0.1, -1.21, 0.5, -1.2, False),
        (0.1, 1.01, 0.5, -1.2, False),
        (0.1, -1.1, 1.5, -1.2, False),  # no combined position/acceleration evidence
        (0.1, -1.1, 0.5, None, False),
        (0.1, 0.0, 1.5, -1.2, True),  # original weak-acceleration behavior retained
        (4.0, 0.0, 0.5, -1.2, True),  # original strong-load domain retained
    ],
)
def test_position_extension_requires_weak_force_and_original_acceleration(
    force, position, acceleration, extension, accepted
):
    original, jacobian, baseline = load_inputs()
    calibration = replace(
        original.calibration,
        weak_load_q_min=None if extension is None else np.full(7, extension),
        weak_load_acceleration_max_rad_s2=np.full(7, 2.0),
        tool_normal_force_upper_max_n=1.0,
        tool_normal_residual_n=np.array([0.005, 0.1, 0.01]),
    )
    model = ResidualLoadModel(calibration)
    feedback = replace(baseline.feedback, torque_nm=np.r_[-force, np.zeros(6)])
    arguments = (
        feedback,
        np.zeros(7),
        jacobian,
        np.full(7, position),
        np.zeros(7),
        np.full(7, acceleration),
        np.array([1.0, 0, 0]),
    )
    if accepted:
        model.estimate(*arguments, tool_normal_world=np.array([1.0, 0, 0]))
    else:
        with pytest.raises(ValueError, match="outside_verified_load_envelope"):
            model.estimate(*arguments, tool_normal_world=np.array([1.0, 0, 0]))


@pytest.mark.parametrize("extension,cap", [(-0.5, 1.0), (np.nan, 1.0), (-1.2, None)])
def test_position_extension_rejects_invalid_or_unbounded_recipe(extension, cap):
    original, jacobian, baseline = load_inputs()
    model = ResidualLoadModel(
        replace(
            original.calibration,
            weak_load_q_min=np.full(7, extension),
            tool_normal_force_upper_max_n=cap,
        )
    )
    with pytest.raises(ValueError):
        model.estimate(
            baseline.feedback,
            np.zeros(7),
            jacobian,
            np.zeros(7),
            np.zeros(7),
            np.zeros(7),
            np.array([1.0, 0, 0]),
        )


@pytest.mark.parametrize("margin", [None, np.nan, np.inf, 0.0, -0.001])
def test_weak_position_rejects_contact_without_complete_free_stop_support(margin):
    model, _, _ = load_inputs()
    calibration = replace(model.calibration, weak_load_q_min=np.full(7, -1.2))
    with pytest.raises(ValueError, match="weak_position_requires_contact_free_motion_and_stop"):
        calibration.require_position_motion_support(
            np.full(7, -1.1), contact_free_stop_margin_m=margin
        )
    # Original position/contact domain does not depend on free-motion support.
    calibration.require_position_motion_support(np.zeros(7), contact_free_stop_margin_m=margin)


def test_weak_position_accepts_positive_future_free_stop_margin():
    model, _, _ = load_inputs()
    calibration = replace(model.calibration, weak_load_q_min=np.full(7, -1.2))
    calibration.require_position_motion_support(np.full(7, -1.1), contact_free_stop_margin_m=0.001)


def test_position_motion_guard_only_applies_to_the_extended_joint():
    model, _, _ = load_inputs()
    extended = np.full(7, -1.0)
    extended[3] = -1.2
    calibration = replace(model.calibration, weak_load_q_min=extended)
    lower = np.zeros(7)
    lower[0] = -1.1
    calibration.require_position_motion_support(lower, contact_free_stop_margin_m=None)
    lower[3] = -1.1
    with pytest.raises(ValueError, match="weak_position_requires_contact_free_motion_and_stop"):
        calibration.require_position_motion_support(lower, contact_free_stop_margin_m=None)


@pytest.mark.parametrize(
    "qvalue,accepted",
    [
        (1.0, True),
        (1.004, True),
        (1.00425, True),
        (1.004251, False),
        (-1.00425, True),
        (-1.004251, False),
    ],
)
def test_position_halo_boundary_preserves_unclipped_load_estimate(qvalue, accepted):
    from alexdoor_xas.perception.load import LoadEnvelopeError

    model, jac, nominal = load_inputs()
    model = ResidualLoadModel(
        replace(
            model.calibration,
            position_tolerance_rad=np.full((2, 7), 0.00425),
            position_halo_force_upper_max_n=5.0,
        )
    )
    q = np.full(7, qvalue)
    args = (nominal.feedback, np.zeros(7), jac, q, np.zeros(7), np.zeros(7), [1, 0, 0])
    if accepted:
        load = model.estimate(*args)
        np.testing.assert_array_equal(load.wrench, nominal.wrench)
        np.testing.assert_array_equal(load.residual_bound_nm, nominal.residual_bound_nm)
    else:
        with pytest.raises(LoadEnvelopeError) as caught:
            model.estimate(*args)
        value = caught.value.decision["violations"][0]
        assert value["measured"] == qvalue and value["tolerance"] == 0.00425
        assert value["quantity"] == "position_rad"


@pytest.mark.parametrize("quantity", ["speed", "acceleration"])
def test_position_halo_cannot_relax_motion_envelopes(quantity):
    model, jac, nominal = load_inputs()
    model = ResidualLoadModel(
        replace(
            model.calibration,
            position_tolerance_rad=np.full((2, 7), 0.00425),
            position_halo_force_upper_max_n=5.0,
        )
    )
    dq = np.full(7, 1.00001) if quantity == "speed" else np.zeros(7)
    ddq = np.full(7, 1.00001) if quantity == "acceleration" else np.zeros(7)
    with pytest.raises(ValueError, match="outside_verified_load_envelope"):
        model.estimate(nominal.feedback, np.zeros(7), jac, np.zeros(7), dq, ddq, [1, 0, 0])


def test_position_halo_does_not_extend_the_opposite_unverified_boundary():
    model, jac, nominal = load_inputs()
    allowance = np.zeros((2, 7))
    allowance[1, 6] = 0.00425
    model = ResidualLoadModel(
        replace(
            model.calibration, position_tolerance_rad=allowance, position_halo_force_upper_max_n=5.0
        )
    )
    q = np.zeros(7)
    q[6] = -1.000001
    with pytest.raises(ValueError, match="outside_verified_load_envelope"):
        model.estimate(nominal.feedback, np.zeros(7), jac, q, np.zeros(7), np.zeros(7), [1, 0, 0])


def test_position_halo_refuses_stronger_load_and_requires_future_free_stop():
    model, jac, nominal = load_inputs()
    allowance = np.zeros((2, 7))
    allowance[1, 6] = 0.00425
    calibration = replace(
        model.calibration, position_tolerance_rad=allowance, position_halo_force_upper_max_n=1.0
    )
    model = ResidualLoadModel(calibration)
    q = np.zeros(7)
    q[6] = 1.001
    with pytest.raises(ValueError, match="outside_verified_load_envelope"):
        model.estimate(nominal.feedback, np.zeros(7), jac, q, np.zeros(7), np.zeros(7), [1, 0, 0])
    with pytest.raises(ValueError, match="requires_contact_free_motion_and_stop"):
        calibration.require_position_motion_support(
            np.zeros(7), motion_q_upper=q, contact_free_stop_margin_m=0.0
        )
    calibration.require_position_motion_support(
        np.zeros(7), motion_q_upper=q, contact_free_stop_margin_m=0.001
    )


def test_position_halo_cannot_combine_an_independent_weak_acceleration_extension():
    model, jac, nominal = load_inputs()
    allowance = np.zeros((2, 7))
    allowance[1, 6] = 0.00425
    calibration = replace(
        model.calibration,
        position_tolerance_rad=allowance,
        position_halo_force_upper_max_n=1.0,
        weak_load_acceleration_max_rad_s2=np.full(7, 2.0),
        tool_normal_force_upper_max_n=1.0,
        tool_normal_residual_n=np.array([0.005, 0.1, 0.01]),
    )
    model = ResidualLoadModel(calibration)
    fb = replace(nominal.feedback, torque_nm=np.zeros(7))
    q = np.zeros(7)
    q[6] = 1.001
    with pytest.raises(ValueError, match="outside_verified_load_envelope") as caught:
        model.estimate(
            fb,
            np.zeros(7),
            jac,
            q,
            np.zeros(7),
            np.full(7, 1.5),
            [1, 0, 0],
            tool_normal_world=[1, 0, 0],
        )
    assert caught.value.decision["violations"][-1]["quantity"] == (
        "combined_weak_position_and_acceleration"
    )


@pytest.mark.parametrize("prepaid", [-1, np.nan, np.inf, 0.01])
def test_material_geometry_credit_refuses_invalid_or_uncovered_error(prepaid):
    estimate, proposal, evidence, limits = local_inputs()
    evidence = replace(evidence, geometry_position_prepaid_m=prepaid, swept_collision_margin_m=0.01)
    receipt = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    assert not receipt.admitted and receipt.reason == "invalid_prepaid_geometry_error"


def test_material_geometry_credit_requires_separate_collision_proof_and_preserves_boundary():
    from alexdoor_xas.perception.local_admission import _local_error

    estimate, proposal, evidence, limits = local_inputs()
    patch = estimate.local.patches[0]
    prepaid = patch.geometry_support.position_bound_m
    evidence = replace(evidence, geometry_position_prepaid_m=prepaid)
    receipt = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    assert not receipt.admitted and receipt.reason == "invalid_prepaid_geometry_error"
    evidence = replace(evidence, swept_collision_margin_m=0.01)
    error = _local_error(patch, evidence, limits, 1, 1 / 60)
    # The complete uncertainty is still required; zero numerical tolerance.
    for delta, accepted in [(-1e-6, False), (0, False), (1e-6, True)]:
        boundary = replace(evidence, finger_clearance_m=(error + delta, error + delta))
        receipt = admit_local_action(estimate, proposal, boundary, limits, 1, generation=0)
        assert receipt.admitted is accepted
        assert receipt.decision["geometry_position_prepaid_m"] == prepaid
        assert receipt.decision["numerical_tolerance_m"] == 0


@pytest.mark.parametrize("anchor", [0.0, 0.6, 1.2])
def test_pose_rotation_transports_each_primitive_directly_to_actual_footprint(anchor):
    from alexdoor_xas.perception.admission import rotational_travel
    from alexdoor_xas.perception.local_admission import _local_error_terms

    estimate, _, evidence, limits = local_inputs()
    patch = estimate.local.patches[0]
    origin = patch.world_pose.origin + [0, 0.2, 0]
    shifted = replace(patch.world_pose, origin=patch.world_pose.origin + [0, anchor, 0])
    patch = replace(
        patch,
        world_pose=shifted,
        pose_support=replace(
            patch.pose_support,
            position_bound_m=0.002 + rotational_travel(abs(anchor - 0.2), 0.1),
            rotation_bound_rad=0.1,
        ),
        pose_uncertainty_sources=(("dynamic_pose", 0.002, 0.1, abs(anchor - 0.2), tuple(origin)),),
    )
    terms = _local_error_terms(patch, evidence, limits, 1, 1 / 60)
    lever = evidence.footprint_radius_m + max(
        np.linalg.norm(p.origin - origin) for p in evidence.world_poses
    )
    assert terms["pose_position_m"] == 0.002
    assert terms["pose_rotation_m"] == pytest.approx(rotational_travel(lever, 0.1))
    # Changing the arbitrary material anchor cannot change this primitive's
    # displacement at the same actual footprint. A farther tool must cost more.
    farther = replace(
        evidence,
        world_poses=tuple(replace(p, origin=p.origin - [0, 0.2, 0]) for p in evidence.world_poses),
    )
    assert (
        _local_error_terms(patch, farther, limits, 1, 1 / 60)["pose_rotation_m"]
        > terms["pose_rotation_m"]
    )


def test_legacy_pose_bound_still_covers_offset_footprint_without_primitives():
    from alexdoor_xas.perception.local_admission import _local_error_terms

    estimate, _, evidence, limits = local_inputs()
    patch = estimate.local.patches[0]
    nominal = _local_error_terms(patch, evidence, limits, 1, 1 / 60)
    shifted = replace(
        patch, world_pose=replace(patch.world_pose, origin=patch.world_pose.origin + [0, 0.5, 0])
    )
    larger = _local_error_terms(shifted, evidence, limits, 1, 1 / 60)
    assert larger["pose_rotation_m"] > nominal["pose_rotation_m"]
    assert larger["robot_rotation_m"] == nominal["robot_rotation_m"]


def test_interval_material_response_is_counted_once_separately_from_robot_tracking():
    from alexdoor_xas.perception.local_admission import _local_error_terms

    estimate, _, evidence, limits = local_inputs()
    duration = 1 / 60
    evidence = replace(
        evidence, response_position_bound_m=limits.relative_speed_bound_m_s * duration
    )
    terms = _local_error_terms(estimate.local.patches[0], evidence, limits, 1, duration)
    assert terms["response_m"] == limits.relative_speed_bound_m_s * duration
    assert terms["temporal_m"] == 0
    assert terms["robot_position_m"] == limits.robot_position_bound_m


def test_incomplete_pose_primitives_cannot_reduce_a_published_bound():
    estimate, proposal, evidence, limits = local_inputs()
    patch = replace(
        estimate.local.patches[0],
        pose_uncertainty_sources=(("dynamic_pose", 0.0, 0.0, 0.0, (0.0, 0.0, 0.0)),),
    )
    estimate = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    receipt = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    assert not receipt.admitted and receipt.reason == "incomplete_pose_error_sources"


@pytest.mark.parametrize("side,joint", [(0, 4), (1, 2)])
def test_position_support_covers_the_future_stop_load_only_on_verified_sides(side, joint):
    model, _, _ = load_inputs()
    tolerance = np.full((2, 7), 0.00425)
    caps = np.ones((2, 7))
    tolerance[side, joint] = 0.0085
    caps[side, joint] = 10.0
    calibration = replace(
        model.calibration,
        position_tolerance_rad=tolerance,
        position_halo_force_upper_max_n=caps,
    )
    lower, upper = np.zeros(7), np.zeros(7)
    if side == 0:
        lower[joint] = -1.0085
    else:
        upper[joint] = 1.0085
    calibration.require_position_motion_support(
        lower,
        motion_q_upper=upper,
        contact_free_stop_margin_m=-0.2,
        force_upper_n=10.0,
    )
    # A weak measurement cannot certify a future stronger stop; the supplied
    # quantity is the complete required load domain, not current load.
    for unsupported_force in [10.0001, 20.0]:
        with pytest.raises(ValueError, match="requires_contact_free_motion_and_stop"):
            calibration.require_position_motion_support(
                lower,
                motion_q_upper=upper,
                contact_free_stop_margin_m=0.0,
                force_upper_n=unsupported_force,
            )
    if side == 0:
        lower[joint] -= 0.00001
    else:
        upper[joint] += 0.00001
    with pytest.raises(ValueError, match="requires_contact_free_motion_and_stop"):
        calibration.require_position_motion_support(
            lower,
            motion_q_upper=upper,
            contact_free_stop_margin_m=0.0,
            force_upper_n=10.0,
        )


def test_position_force_caps_do_not_promote_other_sides_or_clip_feedback():
    model, jac, nominal = load_inputs()
    caps = np.ones((2, 7))
    caps[0, 4] = 10.0
    c = replace(
        model.calibration,
        position_tolerance_rad=np.full((2, 7), 0.0085),
        position_halo_force_upper_max_n=caps,
    )
    model = ResidualLoadModel(c)
    q = np.zeros(7)
    q[4] = -1.008
    original = q.copy()
    estimate = model.estimate(
        nominal.feedback, np.zeros(7), jac, q, np.zeros(7), np.zeros(7), [1.0, 0.0, 0.0]
    )
    np.testing.assert_array_equal(q, original)
    np.testing.assert_array_equal(estimate.wrench, nominal.wrench)
    # Same posture distance at an independently weak side still refuses.
    q[4] = 1.008
    with pytest.raises(ValueError, match="outside_verified_load_envelope"):
        model.estimate(
            nominal.feedback, np.zeros(7), jac, q, np.zeros(7), np.zeros(7), [1.0, 0.0, 0.0]
        )
    lower, upper = np.zeros(7), np.zeros(7)
    upper[2] = 1.001
    with pytest.raises(ValueError, match="requires_contact_free_motion_and_stop"):
        c.require_position_motion_support(
            lower, motion_q_upper=upper, contact_free_stop_margin_m=0.0, force_upper_n=10.0
        )


@pytest.mark.parametrize("caps", [np.ones(7), np.full((2, 7), -1.0), np.full((2, 7), np.nan)])
def test_position_support_requires_a_valid_force_domain(caps):
    model, _, _ = load_inputs()
    c = replace(
        model.calibration,
        position_tolerance_rad=np.full((2, 7), 0.0085),
        position_halo_force_upper_max_n=caps,
    )
    with pytest.raises(ValueError, match="unverified_position_halo_force_domain"):
        c.position_support()


def test_unloaded_tracker_aging_spends_the_prepaid_interval_once():
    estimate, proposal, evidence, limits = local_inputs()
    limits = replace(limits, relative_speed_bound_m_s=0.1)
    dt = proposal.times_s[-1] - proposal.times_s[0]
    motion = limits.relative_speed_bound_m_s * dt
    patch = replace(
        estimate.local.patches[0],
        normal_position_bound_m=0.005,
        normal_rotation_bound_rad=0.01,
        pose_temporal_position_m=0.001,
    )
    patch = replace(patch, pose_support=replace(patch.pose_support, position_bound_m=0.01))
    estimate = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    proposal = replace(proposal, loaded=False)
    evidence = replace(
        evidence,
        finger_clearance_m=(None, None),
        collision_clearance_m=None,
        swept_collision_margin_m=0.02,
        unloaded_separation_margin_m=0.0004,
        unloaded_normal_lever_m=0.8,
        unloaded_temporal_bound_m_rad=(motion, 0.0),
    )
    receipt = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    assert receipt.admitted
    current = replace(
        patch,
        normal_position_bound_m=patch.normal_position_bound_m + motion,
        pose_temporal_position_m=patch.pose_temporal_position_m + motion,
    )

    def check(value, admission=receipt):
        require_current_admission(
            admission, replace(estimate, local=replace(estimate.local, patches=(value,))), 1 + dt
        )

    # Static geometry with only the already-eroded observation aging remains valid.
    check(current)
    # Real additional normal uncertainty still exhausts the unchanged small margin.
    with pytest.raises(ValueError, match="separation_exhausted"):
        check(replace(current, normal_position_bound_m=current.normal_position_bound_m + 0.00041))
    # Missing or invalid aging attribution cannot receive the credit.
    with pytest.raises(ValueError, match="separation_exhausted"):
        check(replace(current, pose_temporal_position_m=None))
    with pytest.raises(ValueError, match="invalid_prepaid"):
        check(replace(current, pose_temporal_position_m=0.02))
    unreserved = admit_local_action(
        estimate,
        proposal,
        replace(evidence, unloaded_temporal_bound_m_rad=None),
        limits,
        1,
        generation=0,
    )
    with pytest.raises(ValueError, match="separation_exhausted"):
        check(current, unreserved)


@pytest.mark.parametrize("offset", [np.zeros(3), np.array([3.0, -7.0, 2.0])])
def test_unloaded_receipt_revalidates_normal_growth_at_its_complete_spheres(offset):
    from alexdoor_xas.perception.admission import rotational_travel

    estimate, proposal, evidence, limits = local_inputs()
    anchor = offset + np.array([0.0, 10.0, 0.0])
    original = estimate.local.patches[0]

    def patch(angle):
        return replace(
            original,
            world_pose=replace(original.world_pose, origin=anchor),
            normal_position_bound_m=rotational_travel(10.0, angle),
            normal_rotation_bound_rad=angle,
            normal_uncertainty_sources=(("measured_plane", 0.0, angle, tuple(offset)),),
        )

    before = patch(0.01)
    estimate = replace(estimate, local=replace(estimate.local, patches=(before,)))
    proposal = replace(proposal, loaded=False)
    center = offset + np.array([0.1, 0.0, 0.0])
    evidence = replace(
        evidence,
        finger_clearance_m=(None, None),
        collision_clearance_m=None,
        swept_collision_margin_m=0.02,
        unloaded_separation_margin_m=0.005,
        unloaded_normal_lever_m=np.linalg.norm(center - anchor) + 0.1,
        unloaded_temporal_bound_m_rad=(limits.relative_speed_bound_m_s / 60, 0.0),
        unloaded_normal_spheres=((*center, 0.1),),
    )
    receipt = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    assert receipt.admitted

    def check(current, admission=receipt):
        require_current_admission(
            admission,
            replace(estimate, local=replace(estimate.local, patches=(current,))),
            1 + 1 / 60,
        )

    check(before)
    # At this complete 100mm-radius ball, the bound grows by about4mm,
    # while transport via the distant anchor falsely grows by about400mm.
    check(patch(0.03))
    with pytest.raises(ValueError, match="separation_exhausted"):
        check(patch(0.04))
    # Keeping only the ball center would miss growth at its outer extent.
    wide = replace(evidence, unloaded_normal_spheres=((*center, 0.3),))
    wide_receipt = admit_local_action(estimate, proposal, wide, limits, 1, generation=0)
    assert wide_receipt.admitted
    with pytest.raises(ValueError, match="separation_exhausted"):
        check(patch(0.03), wide_receipt)
    with pytest.raises(ValueError, match="missing_unloaded_normal_error_sources"):
        check(replace(patch(0.03), normal_uncertainty_sources=()))
    legacy = admit_local_action(
        estimate, proposal, replace(evidence, unloaded_normal_spheres=()), limits, 1, generation=0
    )
    with pytest.raises(ValueError, match="separation_exhausted"):
        check(patch(0.03), legacy)
    for invalid in [((0, 0, 0, -1),), ((0, 0, np.nan, 0.1),), ((0, 0, 0),)]:
        assert not admit_local_action(
            estimate,
            proposal,
            replace(evidence, unloaded_normal_spheres=invalid),
            limits,
            1,
            generation=0,
        ).admitted


@pytest.mark.parametrize("side", [0, 1])
def test_qualified_position_forecast_uses_native_boundary_resolution(side):
    model, _, _ = load_inputs()
    tolerance = np.zeros((2, 7))
    tolerance[side, 2] = 0.014842030560106245
    caps = np.ones((2, 7))
    caps[side, 2] = 10.0
    qmin, qmax = np.full(7, -1.0), np.full(7, 1.0)
    qmin[2], qmax[2] = -0.7854710817337036, 0.7854710817337036
    calibration = replace(
        model.calibration,
        q_min=qmin,
        q_max=qmax,
        position_tolerance_rad=tolerance,
        position_halo_force_upper_max_n=caps,
    )
    direction = -1 if side == 0 else 1
    boundary = direction * 0.8003131122938099
    lower, upper = np.zeros(7), np.zeros(7)
    interval = lower if side == 0 else upper
    # A retained native forecast differed from its qualified boundary by
    # 1.6e-12rad, far below a single native float32 joint-position step.
    interval[2] = boundary + direction * 1.6e-12

    def check():
        calibration.require_position_motion_support(
            lower, motion_q_upper=upper, contact_free_stop_margin_m=-0.2, force_upper_n=10.0
        )

    check()
    rounded = np.float32(boundary)
    outward = np.nextafter(rounded, np.float32(direction * np.inf))
    interval[2] = outward
    check()
    # No promotion of a weaker force-qualified halo, clipping, or added
    # physical/empirical travel beyond that representable boundary step.
    with pytest.raises(ValueError, match="requires_contact_free_motion_and_stop"):
        calibration.require_position_motion_support(
            lower, motion_q_upper=upper, contact_free_stop_margin_m=-0.2, force_upper_n=10.0001
        )
    interval[2] = np.nextafter(outward, np.float32(direction * np.inf))
    with pytest.raises(ValueError, match="requires_contact_free_motion_and_stop"):
        check()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("offset", [np.zeros(3), np.array([3.0, -7.0, 2.0])])
def test_unloaded_growth_spends_each_complete_spheres_own_margin(reverse, offset):
    from alexdoor_xas.perception.admission import rotational_travel

    estimate, proposal, evidence, limits = local_inputs()
    original = estimate.local.patches[0]
    anchor = offset + np.array([0.0, 10.0, 0.0])

    def patch(angle):
        return replace(
            original,
            world_pose=replace(original.world_pose, origin=anchor),
            normal_position_bound_m=rotational_travel(10.0, angle),
            normal_rotation_bound_rad=angle,
            normal_uncertainty_sources=(("measured_plane", 0.0, angle, tuple(offset)),),
        )

    before, current = patch(0.0), patch(0.02)
    estimate = replace(estimate, local=replace(estimate.local, patches=(before,)))
    proposal = replace(proposal, loaded=False)
    limits = replace(limits, relative_speed_bound_m_s=0.0)
    spheres = [(*offset, 0.1), (*(offset + [1, 0, 0]), 0.1)]
    margins = [0.003, 0.03]
    if reverse:
        spheres.reverse()
        margins.reverse()
    evidence = replace(
        evidence,
        finger_clearance_m=(None, None),
        collision_clearance_m=None,
        swept_collision_margin_m=0.02,
        unloaded_separation_margin_m=min(margins),
        unloaded_normal_lever_m=12.0,
        unloaded_temporal_bound_m_rad=(0.0, 0.0),
        unloaded_normal_spheres=tuple(spheres),
        unloaded_normal_margins_m=tuple(margins),
    )

    def admit(support):
        receipt = admit_local_action(estimate, proposal, support, limits, 1, generation=0)
        assert receipt.admitted
        return receipt

    def check(receipt, value=current):
        require_current_admission(
            receipt,
            replace(estimate, local=replace(estimate.local, patches=(value,))),
            proposal.times_s[-1],
        )

    # About2mm near the tightly cleared volume and22mm at the distant volume
    # each fit their original3mm/30mm bounds. All sphere radii remain charged.
    check(admit(evidence))
    # Old receipts retain their global minimum rule; swapping the association
    # also refuses. Neither missing evidence nor a different ball lends margin.
    with pytest.raises(ValueError, match="separation_exhausted"):
        check(admit(replace(evidence, unloaded_normal_margins_m=())))
    with pytest.raises(ValueError, match="separation_exhausted"):
        check(admit(replace(evidence, unloaded_normal_margins_m=tuple(margins[::-1]))))
    # Exact exhaustion, and growth beyond it, still refuse strictly.
    boundary = rotational_travel(0.1, 0.02)
    tight = [boundary if value == 0.003 else value for value in margins]
    with pytest.raises(ValueError, match="separation_exhausted") as caught:
        check(
            admit(
                replace(
                    evidence,
                    unloaded_separation_margin_m=boundary,
                    unloaded_normal_margins_m=tuple(tight),
                )
            )
        )
    assert caught.value.decision["normal_sphere_index"] == (1 if reverse else 0)
    with pytest.raises(ValueError, match="separation_exhausted"):
        check(admit(evidence), patch(0.05))


@pytest.mark.parametrize(
    "margins,spheres",
    [
        ((0.005,), ()),
        ((0.005, 0.01), ((0, 0, 0, 0.1),)),
        ((0.0,), ((0, 0, 0, 0.1),)),
        ((-0.005,), ((0, 0, 0, 0.1),)),
        ((np.nan,), ((0, 0, 0, 0.1),)),
        ((np.inf,), ((0, 0, 0, 0.1),)),
        ((0.006,), ((0, 0, 0, 0.1),)),
    ],
)
def test_unloaded_sphere_margins_require_complete_matching_original_proof(margins, spheres):
    estimate, proposal, evidence, limits = local_inputs()
    patch = replace(
        estimate.local.patches[0],
        normal_position_bound_m=0.0,
        normal_rotation_bound_rad=0.0,
        normal_uncertainty_sources=(("measured_plane", 0.0, 0.0, (0.0, 0.0, 0.0)),),
    )
    estimate = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    receipt = admit_local_action(
        estimate,
        replace(proposal, loaded=False),
        replace(
            evidence,
            swept_collision_margin_m=0.02,
            unloaded_separation_margin_m=0.005,
            unloaded_normal_lever_m=1.0,
            unloaded_normal_spheres=spheres,
            unloaded_normal_margins_m=margins,
        ),
        limits,
        1,
        generation=0,
    )
    assert not receipt.admitted
    assert receipt.reason == "invalid_unloaded_normal_margins"
