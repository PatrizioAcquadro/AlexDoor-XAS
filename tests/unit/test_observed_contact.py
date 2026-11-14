"""Software contact logic on numerical fixtures, not physical qualification."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.action.b1 import apply_pose_delta
from alexdoor_xas.perception.contact import ContactMonitorConfig, ObservedContactMonitor
from alexdoor_xas.perception.local_admission import admit_local_action
from perception_helpers import load_inputs, local_inputs


def monitor():
    return ObservedContactMonitor(
        ContactMonitorConfig(0.008, 1, 0.5, 0.1, 0.04, 0.5, 0.03, 0.1),
        (np.array([[0, -0.01, 0]]), np.array([[0, 0.01, 0]])),
    )


def inputs(t, *, force=4, x=0.003, withdrawal=False, rotation=0, step_m=0.001, bad=None):
    estimate, proposal, support, limits = local_inputs()
    patch = estimate.local.patches[0]
    patch = replace(
        patch,
        **{
            k: replace(getattr(patch, k), available_s=t, supported_s=t)
            for k in ("pose_support", "identity_support", "geometry_support")
        },
    )
    state = replace(estimate.local, patches=(patch,))
    estimate = replace(estimate, timestamp_s=t, local=state)
    tool = replace(patch.world_pose, origin=patch.world_pose.origin + [x, 0, 0])
    row = np.array([-step_m if withdrawal else step_m, 0, 0, 0, 0, rotation])
    proposal = replace(
        proposal,
        action=row,
        times_s=np.array([t, t + 1 / 60]),
        world_poses=(tool, apply_pose_delta(tool, row)),
    )
    support = replace(support, support=patch.pose_support, world_poses=proposal.world_poses)
    if bad == "unknown":
        support = replace(support, unknown_in_envelope=True)
    if bad == "ownership":
        estimate = replace(
            estimate,
            local=replace(state, patches=(replace(patch, ownership="ambiguous_leaf_candidate"),)),
        )
    receipt = admit_local_action(estimate, proposal, support, limits, t, generation=0)
    load = load_inputs()[2]
    feedback = replace(load.feedback, timestamp_s=t, available_s=t)
    load = replace(
        load,
        feedback=feedback,
        normal_lower_n=force - 0.05,
        force_upper_n=abs(force) + 0.15,
        normal_samples=((t - 1 / 60, t, force - 0.05),),
    )
    if bad == "unverified":
        load = replace(load, model_verified=False)
    if bad == "future":
        load = replace(load, feedback=replace(feedback, available_s=t + 0.01))
    return t, estimate, tool, load, receipt


def establish(m):
    for t in np.linspace(1, 1.2, 13):
        result = m.update(*inputs(float(t)))
    assert result.status == "contact"
    return result


def test_contact_is_indirect_temporal_evidence_with_explicit_mm_uncertainty():
    m = monitor()
    first = m.update(*inputs(1))
    assert first.status == "seeking"
    assert m.update(*inputs(1)).status == "seeking"
    result = establish(m)
    assert not result.directly_observed
    assert result.uncertainty_m > 0.004
    assert len(result.finger_separation_intervals_m) == 2
    assert all(lo < 0 < hi for lo, hi in result.finger_separation_intervals_m)


def test_positive_load_away_from_selected_material_never_confirms_contact():
    result = monitor().update(*inputs(1, x=-0.05))
    assert result.status == "unavailable"
    assert result.reason == "load_without_intended_surface"


def test_no_contact_and_contact_load_ambiguity_stay_distinct():
    result = monitor().update(*inputs(1, force=0, x=-0.05))
    assert result.status == "separated"
    uncertain = monitor().update(*inputs(1, force=0))
    assert uncertain.status == "seeking"
    assert uncertain.reason == "insufficient_contact_load_history"


@pytest.mark.parametrize("bad", ["unknown", "ownership", "unverified", "future"])
def test_uncalibrated_ambiguous_unknown_and_future_support_never_admits(bad):
    assert monitor().update(*inputs(1, bad=bad)).status == "unavailable"


def test_contact_loss_latches_and_reacquired_force_does_not_resume():
    m = monitor()
    establish(m)
    assert m.update(*inputs(1.2 + 1 / 60, force=0)).status == "contact"
    for t in np.arange(1.2 + 2 / 60, 1.36, 1 / 60):
        result = m.update(*inputs(float(t), force=0))
    assert result.status == "lost"
    assert m.update(*inputs(1.36)).status == "lost"
    m.reset()
    assert m.update(*inputs(1.4)).status == "seeking"


def test_observation_gap_and_geometry_loss_cannot_refresh_contact():
    m = monitor()
    establish(m)
    assert m.update(*inputs(1.4)).status == "unavailable"
    assert m.update(*inputs(1.41)).status == "lost"
    m.reset()
    establish(m)
    assert m.update(*inputs(1.22, x=-0.05)).status in ("lost", "unavailable")


def test_only_the_submitted_withdrawal_allows_expected_unloading():
    m = monitor()
    establish(m)
    assert m.update(*inputs(1.2 + 1 / 60, force=3, withdrawal=True)).status == "contact"
    for t in np.arange(1.2 + 2 / 60, 1.4, 1 / 60):
        released = m.update(*inputs(float(t), force=0, x=-0.001, withdrawal=True))
    assert released.status == "unloading"
    assert released.reason == "geometric_withdrawal"
    assert not m.latched_reason


def test_acquisition_deadline_is_a_stop_and_does_not_insert_motion():
    m = monitor()
    for t in np.arange(1, 1.53, 1 / 60):
        result = m.update(*inputs(float(t), force=0))
    assert result.status == "lost"
    assert result.reason == "contact_acquisition_budget"


def test_same_clock_cannot_supply_a_different_contact_pose():
    m = monitor()
    m.update(*inputs(1))
    result = m.update(*inputs(1, x=0.004))
    assert result.status == "unavailable"
    assert result.reason == "different_contact_pose_at_same_time"


def test_center_withdrawal_cannot_hide_a_finger_rotating_into_material():
    m = monitor()
    establish(m)
    for t in np.arange(1.2 + 1 / 60, 1.4, 1 / 60):
        result = m.update(
            *inputs(float(t), force=0, withdrawal=True, rotation=0.001, step_m=0.000001)
        )
    assert result.status == "lost"
    assert result.reason == "contact_load_lost_or_uncertain"


def test_a_larger_model_uncertainty_cannot_make_remote_contact_admissible():
    m = monitor()
    m.config = replace(m.config, normal_model_bound_m=1.0)
    result = m.update(*inputs(1, x=-0.05))
    assert result.status == "unavailable"
    assert result.reason == "contact_uncertainty_exceeds_admitted_margin"


def test_impulse_contact_uses_covered_native_mean_while_load_peak_is_preserved():
    m = monitor()
    for t in np.linspace(1, 1.2, 13):
        now, estimate, tool, load, receipt = inputs(float(t))
        load = replace(
            load,
            normal_lower_n=-0.05,
            force_upper_n=8.15,
            normal_samples=((now - 1 / 60, now - 1 / 120, 7.95), (now - 1 / 120, now, -0.05)),
        )
        result = m.update(now, estimate, tool, load, receipt)
    assert result.status == "contact"
    assert result.normal_force_lower_n == pytest.approx(3.95)
    assert load.force_upper_n == 8.15
    # A missing physics interval is unknown, even when the next sample is high.
    result = m.update(*inputs(1.2 + 2 / 60))
    assert result.status == "unavailable"
    assert result.reason == "contact_load_coverage_gap"


def test_contact_bound_includes_separately_published_material_geometry():
    base = inputs(1)
    now, estimate, tool, load, receipt = base
    patch = estimate.local.patches[0]
    changed = replace(
        patch,
        geometry_support=replace(
            patch.geometry_support, position_bound_m=patch.geometry_support.position_bound_m + 0.001
        ),
    )
    estimate = replace(estimate, local=replace(estimate.local, patches=(changed,)))
    updated = admit_local_action(
        estimate, receipt.proposal, receipt.local_evidence, receipt.limits, now, generation=0
    )
    original = monitor().update(*base)
    measured = monitor().update(now, estimate, tool, load, updated)
    assert measured.uncertainty_m == pytest.approx(original.uncertainty_m + 0.001)


def plane_inputs(t, *, force=4, x=0.001, reference_y=0, tool_y=0):
    now, estimate, tool, load, receipt = inputs(t, force=force, x=x)
    patch = estimate.local.patches[0]
    patch = replace(
        patch,
        world_pose=replace(patch.world_pose, origin=patch.world_pose.origin + [0, reference_y, 0]),
        pose_support=replace(patch.pose_support, position_bound_m=0.025, rotation_bound_rad=0.2),
        normal_position_bound_m=0.001,
        normal_rotation_bound_rad=0.003,
    )
    estimate = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    tool = replace(tool, origin=tool.origin + [0, tool_y, 0])
    proposal = replace(
        receipt.proposal, world_poses=(tool, apply_pose_delta(tool, receipt.proposal.action))
    )
    support = replace(
        receipt.local_evidence,
        world_poses=proposal.world_poses,
        finger_clearance_m=(0.1, 0.1),
        collision_clearance_m=0.1,
    )
    receipt = admit_local_action(estimate, proposal, support, receipt.limits, now, generation=0)
    assert receipt.admitted
    return now, estimate, tool, load, receipt


def plane_monitor():
    return ObservedContactMonitor(
        ContactMonitorConfig(
            0.001,
            1,
            0.5,
            0.1,
            0.04,
            0.5,
            0.03,
            0.1,
            robot_fk_position_bound_m=1e-5,
            robot_fk_rotation_bound_rad=1e-5,
        ),
        (np.array([[0, -0.01, 0]]), np.array([[0, 0.01, 0]])),
    )


def test_plane_normal_distance_is_separate_from_material_pose_and_command_tracking_error():
    m = plane_monitor()
    for t in np.linspace(1, 1.2, 13):
        result = m.update(*plane_inputs(float(t)))
    assert result.status == "contact" and not result.directly_observed
    assert 0.002 < result.uncertainty_m < 0.0021
    # A 25 mm material-coordinate uncertainty cannot make load 20 mm away
    # from a separately measured plane consistent with intended contact.
    distant = plane_monitor().update(*plane_inputs(1, x=-0.02))
    assert distant.status == "unavailable" and distant.reason == "load_without_intended_surface"
    now, estimate, tool, load, receipt = plane_inputs(1)
    narrow = replace(receipt.local_evidence, finger_clearance_m=(0.005, 0.005))
    refused = admit_local_action(
        estimate, receipt.proposal, narrow, receipt.limits, now, generation=0
    )
    assert not refused.admitted  # Complete pose error still protects finite material support.


def test_seek_travel_measures_robot_motion_without_material_anchor_jitter():
    m = plane_monitor()
    for i, t in enumerate(np.linspace(1, 1.2, 13)):
        result = m.update(*plane_inputs(float(t), force=0, reference_y=0.02 * (i % 2)))
    assert result.status == "seeking"
    moved = m.update(*plane_inputs(1.2 + 1 / 60, force=0, tool_y=0.031))
    assert moved.status == "lost" and moved.reason == "contact_acquisition_budget"


def test_partial_fk_error_configuration_never_supplies_contact_uncertainty():
    with pytest.raises(ValueError, match="partial_robot_fk"):
        replace(monitor().config, robot_fk_position_bound_m=1e-5)


def test_seek_budget_counts_measured_path_length_even_when_robot_returns_to_anchor():
    m = plane_monitor()
    assert m.update(*plane_inputs(1, force=0)).status == "seeking"
    assert m.update(*plane_inputs(1 + 1 / 60, force=0, tool_y=0.02)).status == "seeking"
    result = m.update(*plane_inputs(1 + 2 / 60, force=0, tool_y=0))
    assert result.status == "lost" and result.reason == "contact_acquisition_budget"


def test_contact_model_error_spends_material_margin_not_independent_static_robot_margin():
    values = inputs(1.0)
    receipt = values[-1]
    receipt = replace(
        receipt,
        local_evidence=replace(receipt.local_evidence, swept_collision_margin_m=0.0001),
        margins_m=receipt.margins_m + (("independent_robot_sweep", 0.0001),),
    )
    result = monitor().update(*values[:-1], receipt)
    assert result.status == "seeking"
    extra = result.decision["additional_tangent_uncertainty_m"]
    assert 0 < extra < monitor().config.normal_model_bound_m
    narrow = replace(receipt, margins_m=(("contact", extra / 2), ("independent_robot_sweep", 0.1)))
    result = monitor().update(*values[:-1], narrow)
    assert result.status == "unavailable"
    assert result.reason == "contact_uncertainty_exceeds_admitted_margin"


def test_bounded_relief_extends_only_possible_surface_interval_and_keeps_far_load_unavailable():
    m = monitor()
    t, estimate, tool, load, receipt = inputs(1, x=0.025)
    # The exact plane-only monitor cannot associate a surface25mm behind it.
    assert m.update(t, estimate, tool, load, receipt).reason == "load_without_intended_surface"
    m.reset()
    receipt = replace(
        receipt, local_evidence=replace(receipt.local_evidence, material_normal_range_m=(0.0, 0.02))
    )
    result = m.update(t, estimate, tool, load, receipt)
    assert result.status == "seeking" and not result.directly_observed
    assert result.finger_separation_intervals_m[0][1] > 0
    assert result.uncertainty_m > 0.02
    m.reset()
    t, estimate, tool, load, far = inputs(1, x=0.1)
    far = replace(
        far, local_evidence=replace(far.local_evidence, material_normal_range_m=(0.0, 0.02))
    )
    assert m.update(t, estimate, tool, load, far).reason == "load_without_intended_surface"


def test_complete_unloaded_proof_is_not_an_unspent_contact_patch_margin():
    now, estimate, tool, load, receipt = inputs(1, force=0, x=-0.05)
    patch = replace(
        estimate.local.patches[0], normal_position_bound_m=0.005, normal_rotation_bound_rad=0.01
    )
    estimate = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    proposal = replace(receipt.proposal, loaded=False)
    support = replace(
        receipt.local_evidence,
        finger_clearance_m=(None, None),
        collision_clearance_m=None,
        swept_collision_margin_m=0.002,
        unloaded_separation_margin_m=0.001,
        unloaded_normal_lever_m=0.1,
    )
    free = admit_local_action(estimate, proposal, support, receipt.limits, now, generation=0)
    assert free.admitted
    result = monitor().update(now, estimate, tool, load, free)
    assert result.status == "separated"
    # Contradictory positive load still cannot establish remote intended contact.
    positive = replace(
        load, normal_lower_n=4.0, force_upper_n=4.1, normal_samples=((now - 1 / 60, now, 4.0),)
    )
    result = monitor().update(now, estimate, tool, positive, free)
    assert result.status == "unavailable" and result.reason == "load_without_intended_surface"


def test_certain_separation_does_not_start_the_contact_acquisition_deadline():
    m = monitor()
    for t in np.arange(1.0, 2.0, 1 / 60):
        result = m.update(*inputs(float(t), force=0, x=-0.05))
    assert result.status == "separated"
    assert m.seek_started is None and not m.latched_reason
    identity = m.identity
    # Approach and withdraw continuously within the unchanged travel cap.
    for t in np.arange(2.0, 2.5, 1 / 60):
        result = m.update(
            *inputs(float(t), force=0, x=float(np.interp(t, [2.0, 2.5], [-0.05, -0.005])))
        )
    assert m.seek_started is not None and result.status == "seeking"
    for t in np.arange(2.5, 2.8, 1 / 60):
        result = m.update(
            *inputs(float(t), force=0, x=float(np.interp(t, [2.5, 2.8], [-0.005, -0.05])))
        )
    assert result.status == "separated" and m.seek_started is None
    assert m.identity == identity and not m.latched_reason and m.force_samples
    assert result.acquisition_started_s is None and result.acquisition_travel_m == 0
    # A later uninterrupted unresolved attempt still reaches the same deadline.
    for t in np.arange(2.8, 3.3, 1 / 60):
        result = m.update(
            *inputs(float(t), force=0, x=float(np.interp(t, [2.8, 3.3], [-0.05, 0.003])))
        )
    for t in np.arange(3.3, 3.83, 1 / 60):
        result = m.update(*inputs(float(t), force=0))
    assert result.status == "lost" and result.reason == "contact_acquisition_budget"


@pytest.mark.parametrize(
    "stage,x,expected",
    [
        ("policy", -0.05, ""),
        ("policy", 0.003, "observed_contact_orientation"),
        ("push", -0.05, "observed_contact_orientation"),
    ],
)
def test_contact_orientation_still_applies_to_possible_contact_not_separated_approach(
    stage, x, expected
):
    from types import SimpleNamespace

    from scipy.spatial.transform import Rotation

    from alexdoor_xas.perception.contracts import OPERATIONAL_V1
    from alexdoor_xas.perception.control import ObservedControlChecks

    now, estimate, tool, load, receipt = inputs(1, force=0, x=x)
    tool = replace(tool, rot=Rotation.from_euler("y", 0.2).as_matrix())
    proposal = replace(
        receipt.proposal, world_poses=(tool, apply_pose_delta(tool, receipt.proposal.action))
    )
    support = replace(receipt.local_evidence, world_poses=proposal.world_poses)
    receipt = admit_local_action(estimate, proposal, support, receipt.limits, now, generation=0)
    assert receipt.admitted
    checks = ObservedControlChecks(
        SimpleNamespace(max_joint_speed=0.5, orientation_tolerance=np.deg2rad(5), force_limit_n=6),
        contact_monitor=monitor(),
    )
    sensor = dict(
        time_s=now,
        joint_position=np.zeros(9),
        joint_velocity=np.zeros(9),
        valid_depth=np.ones((1, 1), bool),
        rgb=np.ones((1, 1, 3), np.uint8),
    )
    reason = checks.check(
        sensor,
        estimate,
        tool,
        np.zeros((7, 2)),
        stage,
        np.zeros(2),
        profile=OPERATIONAL_V1,
        feedback=load.feedback,
        admission=receipt,
        load=load,
    )
    assert reason == expected


def test_angular_evidence_is_used_at_the_observation_clock_and_keeps_five_degree_gate():
    from types import SimpleNamespace

    from scipy.spatial.transform import Rotation

    from alexdoor_xas.perception.contracts import OPERATIONAL_V1, NormalAlignmentEvidence
    from alexdoor_xas.perception.control import ObservedControlChecks

    now, estimate, tool, load, _ = inputs(1.0, x=0.003)
    patch = estimate.local.patches[0]
    tool = replace(tool, rot=Rotation.from_euler("z", 5.1, degrees=True).as_matrix())
    checks = ObservedControlChecks(
        SimpleNamespace(max_joint_speed=0.5, orientation_tolerance=np.deg2rad(5), force_limit_n=6),
        contact_monitor=monitor(),
    )
    sensor = dict(
        time_s=now,
        joint_position=np.zeros(9),
        joint_velocity=np.zeros(9),
        valid_depth=np.ones((1, 1), bool),
        rgb=np.ones((1, 1, 3), np.uint8),
    )

    def check(evidence):
        local = replace(estimate.local, patches=(replace(patch, normal_alignment=evidence),))
        return checks.check(
            sensor,
            replace(estimate, local=local),
            tool,
            np.tile([-2, 2], (7, 1)),
            "push",
            np.zeros(2),
            profile=OPERATIONAL_V1,
            feedback=load.feedback,
            load=load,
        )

    center = tuple(Rotation.from_euler("z", 1.1, degrees=True).as_matrix()[:, 0])
    assert check(None) == "observed_contact_orientation"
    assert check(NormalAlignmentEvidence(now, ((center, np.deg2rad(0.99)),))) == (
        "action_not_admitted"
    )
    assert check(NormalAlignmentEvidence(now, ((center, np.deg2rad(1.01)),))) == (
        "observed_contact_orientation"
    )
    assert check(NormalAlignmentEvidence(now - 1 / 60, ((center, 0.0),))) == "invalid_perception"


def test_zero_operational_off_threshold_still_requires_strictly_positive_bound():
    m = monitor()
    m.config = replace(m.config, force_off_n=0)
    establish(m)
    assert m.seek_started is None and m.seek_travel_m == 0
    # A small strictly positive lower bound maintains indirect contact, without
    # claiming that independent physical loaded-force requirements are met.
    for tick in range(13, 25):
        result = m.update(*inputs(1 + tick / 60, force=0.06))
    assert result.status == "contact"
    for tick in range(25, 37):
        result = m.update(*inputs(1 + tick / 60, force=0.05))
    assert result.status == "lost"
    assert result.reason == "contact_load_lost_or_uncertain"
    assert m.latched_reason == result.reason
    assert m.seek_started is None


@pytest.mark.parametrize(
    "angle,fk_bound,expected",
    [
        (0.04, 0.0001, ""),
        (0.04, None, "observed_contact_orientation"),
        (0.04, 0.07, "observed_contact_orientation"),
        (0.12, 0.0001, "observed_contact_orientation"),
    ],
)
def test_current_orientation_uses_measured_fk_error_not_future_controller_response(
    angle, fk_bound, expected
):
    from types import SimpleNamespace

    from scipy.spatial.transform import Rotation

    from alexdoor_xas.perception.contracts import OPERATIONAL_V1
    from alexdoor_xas.perception.control import ObservedControlChecks

    now, estimate, tool, load, receipt = inputs(1, force=0)
    tool = replace(tool, rot=Rotation.from_euler("y", angle).as_matrix())
    proposal = replace(
        receipt.proposal, world_poses=(tool, apply_pose_delta(tool, receipt.proposal.action))
    )
    support = replace(receipt.local_evidence, world_poses=proposal.world_poses)
    limits = replace(receipt.limits, robot_rotation_bound_rad=0.08)
    receipt = admit_local_action(estimate, proposal, support, limits, now, generation=0)
    assert receipt.admitted
    contact = monitor()
    contact.config = replace(
        contact.config,
        robot_fk_position_bound_m=None if fk_bound is None else 0.001,
        robot_fk_rotation_bound_rad=fk_bound,
    )
    checks = ObservedControlChecks(
        SimpleNamespace(max_joint_speed=0.5, orientation_tolerance=0.1, force_limit_n=6),
        contact_monitor=contact,
    )
    sensor = dict(
        time_s=now,
        joint_position=np.zeros(9),
        joint_velocity=np.zeros(9),
        valid_depth=np.ones((1, 1), bool),
        rgb=np.ones((1, 1, 3), np.uint8),
    )
    reason = checks.check(
        sensor,
        estimate,
        tool,
        np.zeros((7, 2)),
        "policy",
        np.zeros(2),
        profile=OPERATIONAL_V1,
        feedback=load.feedback,
        admission=receipt,
        load=load,
    )
    assert reason == expected
    assert receipt.limits.robot_rotation_bound_rad == 0.08
    if expected:
        assert checks.last_failure["alignment_rad"] == pytest.approx(angle)
        assert checks.last_failure["current_robot_rotation_bound_rad"] == (
            0.08 if fk_bound is None else fk_bound
        )


@pytest.mark.parametrize("fraction,blocked", [(0.99, True), (1.0, True), (1.01, False)])
def test_projected_normal_error_boundary_and_complete_normal_interval(fraction, blocked):
    values = inputs(1.0)
    receipt = values[-1]
    receipt = replace(
        receipt,
        local_evidence=replace(
            receipt.local_evidence,
            swept_collision_margin_m=0.0001,
        ),
    )
    m = monitor()
    nominal = m.update(*values[:-1], receipt)
    tangent = nominal.decision["additional_tangent_uncertainty_m"]
    projected = replace(receipt, margins_m=(("contact", tangent * fraction),))
    result = monitor().update(*values[:-1], projected)
    assert (result.status == "unavailable") is blocked
    if blocked:
        assert result.reason == "contact_uncertainty_exceeds_admitted_margin"
        assert result.decision["material_margin_m"] == tangent * fraction
    else:
        assert result.uncertainty_m == nominal.uncertainty_m
        assert result.uncertainty_m >= m.config.normal_model_bound_m


def test_missing_material_receipt_is_distinct_from_insufficient_margin():
    values = inputs(1.0)
    receipt = replace(values[-1], margins_m=(("independent_robot_sweep", 0.1),))
    result = monitor().update(*values[:-1], receipt)
    assert result.status == "unavailable"
    assert result.reason == "missing_local_material_margin"
    assert result.decision["material_margin_m"] is None


def test_normal_error_retains_isotropic_fallback_without_independent_collision_proof():
    values = inputs(1.0)
    receipt = replace(values[-1], margins_m=(("contact", 0.001),))
    result = monitor().update(*values[:-1], receipt)
    assert result.reason == "contact_uncertainty_exceeds_admitted_margin"
    assert result.decision["additional_tangent_uncertainty_m"] == 0.008


def test_directional_reserve_contains_normal_cone_offsets_without_spending_normal_twice():
    values = inputs(1.0)
    receipt = replace(
        values[-1],
        local_evidence=replace(values[-1].local_evidence, swept_collision_margin_m=0.001),
    )
    result = monitor().update(*values[:-1], receipt)
    decision = result.decision
    angle = decision["normal_direction_bound_rad"]
    radius = decision["normal_model_bound_m"]
    # Independently project offsets in every sampled azimuth of the normal cone.
    for theta in np.linspace(0.0, angle, 17):
        for azimuth in np.linspace(0.0, 2 * np.pi, 31):
            offset = radius * np.array(
                [np.cos(theta), np.sin(theta) * np.cos(azimuth), np.sin(theta) * np.sin(azimuth)]
            )
            assert np.linalg.norm(offset[1:]) <= (
                decision["additional_tangent_uncertainty_m"] + 1e-15
            )
    assert result.uncertainty_m >= radius


def test_supported_pause_suspends_only_seek_time_and_resumption_spends_it():
    m = monitor()
    m.config = replace(
        m.config,
        pause_aware_seek=True,
        robot_fk_position_bound_m=1.3e-5,
        robot_fk_rotation_bound_rad=2.7e-5,
    )
    for t in np.arange(1, 2.1, 1 / 60):
        result = m.update(*inputs(float(t), force=0, step_m=0))
    assert result.status == "seeking"
    assert result.acquisition_active_s == 0
    assert result.decision["supported_seek_pause"]
    result = m.update(*inputs(2.1, force=0, step_m=0.001))
    assert result.status == "seeking"
    assert result.acquisition_active_s < 0.04
    for i, t in enumerate(np.arange(2.1 + 1 / 60, 2.7, 1 / 60)):
        result = m.update(*inputs(float(t), force=0, step_m=0.0001, x=0.003 + i * 0.0001))
    assert result.status == "lost"
    assert result.reason == "contact_acquisition_budget"
    assert result.acquisition_active_s > m.config.max_seek_s


def test_measured_motion_is_not_a_pause_even_with_zero_submitted_motion():
    m = monitor()
    m.config = replace(
        m.config,
        pause_aware_seek=True,
        robot_fk_position_bound_m=1.3e-5,
        robot_fk_rotation_bound_rad=2.7e-5,
    )
    for i, t in enumerate(np.arange(1, 1.7, 1 / 60)):
        result = m.update(*inputs(float(t), force=0, step_m=0, x=0.003 + i * 0.00004))
    assert result.status == "lost"
    assert result.acquisition_active_s > m.config.max_seek_s


def test_pause_aware_seeking_requires_independent_robot_resolution():
    with pytest.raises(ValueError, match="independent robot FK"):
        replace(monitor().config, pause_aware_seek=True)


def test_slow_creep_cannot_renew_a_standstill_resolution_allowance_each_tick():
    m = monitor()
    m.config = replace(
        m.config,
        pause_aware_seek=True,
        robot_fk_position_bound_m=1.3e-5,
        robot_fk_rotation_bound_rad=2.7e-5,
    )
    for i, t in enumerate(np.arange(1, 1.7, 1 / 60)):
        result = m.update(*inputs(float(t), force=0, step_m=1e-6, x=0.003 + i * 1e-5))
    assert result.status == "lost"
    assert result.acquisition_active_s > m.config.max_seek_s


def test_arriving_load_confirms_contact_without_charging_a_prior_supported_pause():
    m = monitor()
    m.config = replace(
        m.config,
        pause_aware_seek=True,
        robot_fk_position_bound_m=1.3e-5,
        robot_fk_rotation_bound_rad=2.7e-5,
    )
    for t in np.arange(1, 2.0, 1 / 60):
        result = m.update(*inputs(float(t), force=0, step_m=0))
    for t in np.arange(2.0, 2.25, 1 / 60):
        result = m.update(*inputs(float(t), force=4, step_m=0))
    assert result.status == "contact"


def test_fixed_target_allows_tracking_variation_inside_qualified_response_bound():
    m = monitor()
    m.config = replace(
        m.config,
        pause_aware_seek=True,
        robot_fk_position_bound_m=1.3e-5,
        robot_fk_rotation_bound_rad=2.7e-5,
    )
    for i, t in enumerate(np.arange(1, 2.2, 1 / 60)):
        deviation = 0.0001 * np.sin(i)
        result = m.update(*inputs(float(t), force=0, x=0.003 + deviation, step_m=-deviation))
    assert result.status == "seeking"
    assert result.acquisition_active_s == 0
    assert result.acquisition_travel_m > 0  # Physical travel never disappears.


def test_before_after_same_clock_checks_preserve_the_submitted_motion_interval():
    m = monitor()
    m.config = replace(
        m.config,
        pause_aware_seek=True,
        robot_fk_position_bound_m=1.3e-5,
        robot_fk_rotation_bound_rad=2.7e-5,
    )
    for i, t in enumerate(np.arange(1, 1.7, 1 / 60)):
        # Post-step observation of the previous target, followed at the same
        # timestamp by the next target. Physical progress is100um per interval.
        x = 0.003 + i * 0.0001
        m.update(*inputs(float(t), force=0, x=x, step_m=0.0001))
        result = m.update(*inputs(float(t), force=0, x=x, step_m=0.0002))
    assert result.status == "lost"
    assert result.reason == "contact_acquisition_budget"
    assert result.acquisition_active_s > m.config.max_seek_s
    assert result.acquisition_travel_m < m.config.max_seek_travel_m


def test_small_resumption_after_long_hold_does_not_recharge_the_completed_hold():
    m = monitor()
    m.config = replace(
        m.config,
        pause_aware_seek=True,
        robot_fk_position_bound_m=1.3e-5,
        robot_fk_rotation_bound_rad=2.7e-5,
    )
    for t in np.arange(1, 2.1, 1 / 60):
        result = m.update(*inputs(float(t), force=0, step_m=0))
    for i, t in enumerate(np.arange(2.1, 2.4, 1 / 60)):
        result = m.update(*inputs(float(t), force=0, x=0.003 + i * 1e-5, step_m=1e-5))
    assert result.status == "seeking"
    assert result.acquisition_active_s < 0.35
    for i, t in enumerate(np.arange(2.4, 2.8, 1 / 60)):
        result = m.update(*inputs(float(t), force=0, x=0.00318 + i * 1e-5, step_m=1e-5))
    assert result.status == "lost"


def test_held_response_travel_is_logged_without_becoming_new_search():
    m = monitor()
    m.config = replace(
        m.config,
        pause_aware_seek=True,
        robot_fk_position_bound_m=1.3e-5,
        robot_fk_rotation_bound_rad=2.7e-5,
    )
    for i, t in enumerate(np.arange(1, 2.2, 1 / 60)):
        deviation = 0.0001 * np.sin(i)
        result = m.update(*inputs(float(t), force=0, x=0.003 + deviation, step_m=-deviation))
    assert result.status == "seeking"
    assert result.acquisition_travel_m > 0.001
    assert result.acquisition_active_travel_m == 0


def test_active_search_travel_still_stops_before_the_time_budget():
    m = monitor()
    m.config = replace(
        m.config,
        pause_aware_seek=True,
        max_seek_s=2.0,
        robot_fk_position_bound_m=1.3e-5,
        robot_fk_rotation_bound_rad=2.7e-5,
    )
    for i, t in enumerate(np.arange(1, 1.7, 1 / 60)):
        result = m.update(*inputs(float(t), force=0, x=-0.015 + i * 0.001, step_m=0.001))
        if result.status == "lost":
            break
    assert result.reason == "contact_acquisition_budget"
    assert result.acquisition_active_travel_m > m.config.max_seek_travel_m
    assert result.acquisition_active_s < m.config.max_seek_s


def test_repeated_callback_cannot_report_an_active_interval_as_a_supported_pause():
    m = monitor()
    m.config = replace(
        m.config,
        pause_aware_seek=True,
        robot_fk_position_bound_m=1.3e-5,
        robot_fk_rotation_bound_rad=2.7e-5,
    )
    m.update(*inputs(1, force=0, step_m=0))
    values = inputs(1 + 1 / 60, force=0, step_m=0.0001)
    m.update(*values)
    repeated = m.update(*values)
    assert repeated.decision["seek_interval_active"]
    assert not repeated.decision["supported_seek_pause"]
    # Only the completed preceding pause has elapsed; current motion is pending.
    assert repeated.acquisition_active_s == 0
    next_tick = m.update(*inputs(1 + 2 / 60, force=0, step_m=0.0001))
    assert next_tick.acquisition_active_s == pytest.approx(1 / 60)


@pytest.mark.parametrize("reference_y", [-0.02, -0.05])
def test_direct_normal_pivot_covers_rotations_independently_of_selected_anchor(reference_y):
    from scipy.spatial.transform import Rotation

    from alexdoor_xas.perception.admission import rotational_travel

    now, estimate, tool, load, receipt = plane_inputs(1.0, force=0, reference_y=reference_y)
    patch = estimate.local.patches[0]
    pivot = tool.origin.copy()
    angle, position = 0.03, 0.001
    patch = replace(
        patch,
        normal_position_bound_m=position
        + rotational_travel(np.linalg.norm(patch.world_pose.origin - pivot), angle),
        normal_rotation_bound_rad=angle,
    )

    def measured(changed):
        state = replace(estimate, local=replace(estimate.local, patches=(changed,)))
        admitted = admit_local_action(
            state, receipt.proposal, receipt.local_evidence, receipt.limits, now, generation=0
        )
        assert admitted.admitted
        return plane_monitor().update(now, state, tool, load, admitted)

    legacy = measured(patch)
    sources = (("measured_plane", position, angle, tuple(pivot)),)
    direct = measured(replace(patch, normal_uncertainty_sources=sources))
    assert direct.status != "unavailable"
    term = direct.decision["normal_primitive_terms_m"]["measured_plane"]
    points = np.concatenate([face @ tool.rot.T + tool.origin for face in plane_monitor().faces])
    for axis in np.eye(3):
        for rotation in np.linspace(-angle, angle, 9):
            changed = (points - pivot) @ Rotation.from_rotvec(axis * rotation).as_matrix().T + pivot
            # Adversarial normal translation can reinforce rotational error.
            assert np.max(np.linalg.norm(changed - points, axis=1)) + position <= term + 1e-14
    assert direct.uncertainty_m < legacy.uncertainty_m
    assert sum(
        direct.decision["normal_uncertainty_terms_m"][name]
        for name in (
            "model",
            "material_position",
            "material_rotation",
            "robot_position",
            "robot_rotation",
        )
    ) == pytest.approx(direct.decision["normal_uncertainty_terms_m"]["total"])
    # An omitted primitive cannot claim a tighter bound than its own producer.
    incomplete = replace(
        patch, normal_uncertainty_sources=(("measured_plane", 0.0, angle, tuple(pivot)),)
    )
    state = replace(estimate, local=replace(estimate.local, patches=(incomplete,)))
    refused = admit_local_action(
        state, receipt.proposal, receipt.local_evidence, receipt.limits, now, generation=0
    )
    assert not refused.admitted
    assert refused.reason == "incomplete_normal_error_sources"


@pytest.mark.parametrize(
    "request_step,target_step,rotation,expected",
    [
        (-0.0001, -0.00005, 0, True),
        (0.0001, -0.00005, 0, False),  # contrary push target increment
        (0, -0.00005, 0, False),
        (-0.0001, 0, 0, False),  # clipped to no retreat
        (-0.0001, 0.00005, 0, False),
        (-0.000001, -0.000001, 0.001, False),  # inward rotating distal finger
        (-1e-9, -1e-9, 0, False),  # existing numerical sign floor
        (-2e-9, -2e-9, 0, True),
    ],
)
def test_elastic_unloading_requires_both_requested_and_delivered_retreat(
    request_step,
    target_step,
    rotation,
    expected,
):
    m = monitor()
    establish(m)
    now, estimate, tool, _, receipt = inputs(1.2)
    previous = apply_pose_delta(tool, [0.002, 0, 0, 0, 0, 0])
    requested = apply_pose_delta(tool, [request_step, 0, 0, 0, 0, rotation])
    target = apply_pose_delta(previous, [target_step, 0, 0, 0, 0, rotation])
    intent = m.preview_unloading(now, estimate, tool, requested, previous, target, receipt)
    assert intent["unloading_after"] == expected
    assert not m.elastic_unloading  # previews never advance state
    m.commit_unloading(intent, target, now + 1 / 60)
    assert m.elastic_unloading == expected
    assert intent["geometric_withdrawal_margin_m"] < 0


def begin_elastic_unloading(m):
    establish(m)
    now, estimate, tool, _, receipt = inputs(1.2)
    previous = apply_pose_delta(tool, [0.002, 0, 0, 0, 0, 0])
    target = apply_pose_delta(previous, [-0.00005, 0, 0, 0, 0, 0])
    request = apply_pose_delta(tool, [-0.0001, 0, 0, 0, 0, 0])
    intent = m.preview_unloading(now, estimate, tool, request, previous, target, receipt)
    m.commit_unloading(intent, target, now + 1 / 60)
    return intent, target


def test_expected_elastic_loss_is_not_geometric_separation_and_requires_force_history():
    m = monitor()
    begin_elastic_unloading(m)
    for t in np.arange(1.2 + 1 / 60, 1.35, 1 / 60):
        result = m.update(*inputs(float(t), force=0))
    assert result.status == "unloading"
    assert not m.releasing and not m.established and not m.latched_reason
    assert result.decision["delivered_unloading"]["unloading_after"]
    values = list(inputs(1.35, force=0))
    values[3] = replace(values[3], normal_samples=())
    assert m.update(*values).reason == "insufficient_contact_load_history"


def test_elastic_unloading_requires_current_receipt_identity_and_actual_delivery():
    m = monitor()
    establish(m)
    now, estimate, tool, _, receipt = inputs(1.2)
    previous = apply_pose_delta(tool, [0.002, 0, 0, 0, 0, 0])
    target = apply_pose_delta(previous, [-0.00005, 0, 0, 0, 0, 0])
    request = apply_pose_delta(tool, [-0.0001, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        m.preview_unloading(
            now,
            replace(estimate, local=replace(estimate.local, generation=1)),
            tool,
            request,
            previous,
            target,
            receipt,
        )
    with pytest.raises(ValueError):
        m.preview_unloading(now + 0.1, estimate, tool, request, previous, target, receipt)
    intent = m.preview_unloading(now, estimate, tool, request, previous, target, receipt)
    with pytest.raises(ValueError, match="unloading_delivery_mismatch"):
        m.commit_unloading(intent, previous, now + 1 / 60)
    with pytest.raises(ValueError, match="unloading_delivery_mismatch"):
        m.commit_unloading(intent, target, now + 0.1)
    assert not m.elastic_unloading
    # An admitted but unexecuted preview must still fail accidental loss.
    for t in np.arange(1.2 + 1 / 60, 1.35, 1 / 60):
        result = m.update(*inputs(float(t), force=0))
    assert result.status == "lost"


def test_unloaded_elastic_continuation_cannot_resume_forward_without_support():
    m = monitor()
    begin_elastic_unloading(m)
    for t in np.arange(1.2 + 1 / 60, 1.35, 1 / 60):
        m.update(*inputs(float(t), force=0))
    now, estimate, tool, _, receipt = inputs(1.35, force=0)
    m.update(now, estimate, tool, inputs(now, force=0)[3], receipt)
    with pytest.raises(ValueError, match="contact_load_lost_or_uncertain"):
        m.preview_unloading(now, estimate, tool, tool, tool, tool, receipt)


def test_observed_separation_ends_elastic_state_and_allows_stationary_stabilization():
    m = monitor()
    begin_elastic_unloading(m)
    for t in np.arange(1.2 + 1 / 60, 1.35, 1 / 60):
        m.update(*inputs(float(t), force=0))
    # Move gradually under the unchanged continuity bound before separation.
    for i in range(1, 31):
        values = inputs(1.35 + i / 60, force=0, x=0.003 - i * 0.002)
        result = m.update(*values)
    assert result.status == "separated" and m.elastic_unloading
    now, estimate, tool, _, receipt = values
    intent = m.preview_unloading(now, estimate, tool, tool, tool, tool, receipt)
    assert not intent["unloading_after"]
    m.commit_unloading(intent, tool, now + 1 / 60)
    assert not m.elastic_unloading


def test_aging_after_first_positive_gap_preserves_certified_withdrawal():
    m = monitor()
    begin_elastic_unloading(m)
    for t in np.arange(1.2 + 1 / 60, 1.35, 1 / 60):
        m.update(*inputs(float(t), force=0))
    for i in range(1, 15):
        now = 1.35 + i / 60
        values = inputs(now, force=0, x=0.003 - i * 0.002)
        if m.update(*values).status == "separated":
            break
    assert m.result.status == "separated"
    values = inputs(now + 1 / 60, force=0, x=0.003, withdrawal=True)
    now, estimate, tool, _, receipt = values
    assert m.update(*values).status == "unloading"
    target = receipt.proposal.world_poses[-1]
    intent = m.preview_unloading(now, estimate, tool, target, tool, target, receipt)
    assert intent["unloading_after"]


def test_supported_forward_command_cancels_elastic_intent_before_later_accidental_loss():
    m = monitor()
    begin_elastic_unloading(m)
    now, estimate, tool, _, receipt = inputs(1.2)
    intent = m.preview_unloading(now, estimate, tool, tool, tool, tool, receipt)
    m.commit_unloading(intent, tool, now + 1 / 60)
    assert not m.elastic_unloading
    for t in np.arange(1.2 + 1 / 60, 1.35, 1 / 60):
        result = m.update(*inputs(float(t), force=0))
    assert result.status == "lost" and result.reason == "contact_load_lost_or_uncertain"
