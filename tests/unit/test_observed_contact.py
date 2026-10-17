"""Software contact logic on numerical fixtures, not physical qualification."""

from dataclasses import replace

import numpy as np
import pytest
from test_pilot_control import load_inputs, local_inputs

from alexdoor_xas.action.b1 import apply_pose_delta
from alexdoor_xas.perception.contact import ContactMonitorConfig, ObservedContactMonitor
from alexdoor_xas.perception.local_admission import admit_local_action


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
    assert released.status == "separated"
    assert released.reason == "commanded_withdrawal"
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
    narrow = replace(receipt, margins_m=(("contact", 0.001), ("independent_robot_sweep", 0.1)))
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
