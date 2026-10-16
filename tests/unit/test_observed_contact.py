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
        ContactMonitorConfig(0.008, 1, 0.5, 0.1, 0.04, 0.5, 0.03),
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
        load, feedback=feedback, normal_lower_n=force - 0.05, force_upper_n=abs(force) + 0.15
    )
    if bad == "unverified":
        load = replace(load, model_verified=False)
    if bad == "future":
        load = replace(load, feedback=replace(feedback, available_s=t + 0.01))
    return t, estimate, tool, load, receipt


def establish(m):
    for t in np.linspace(1, 1.1, 7):
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
    assert uncertain.reason == "normal_contact_not_established"


@pytest.mark.parametrize("bad", ["unknown", "ownership", "unverified", "future"])
def test_uncalibrated_ambiguous_unknown_and_future_support_never_admits(bad):
    assert monitor().update(*inputs(1, bad=bad)).status == "unavailable"


def test_contact_loss_latches_and_reacquired_force_does_not_resume():
    m = monitor()
    establish(m)
    assert m.update(*inputs(1.12, force=0)).status == "lost"
    assert m.update(*inputs(1.14)).status == "lost"
    m.reset()
    assert m.update(*inputs(1.14)).status == "seeking"


def test_observation_gap_and_geometry_loss_cannot_refresh_contact():
    m = monitor()
    establish(m)
    assert m.update(*inputs(1.2)).status == "unavailable"
    assert m.update(*inputs(1.21)).status == "lost"
    m.reset()
    establish(m)
    assert m.update(*inputs(1.12, x=-0.05)).status in ("lost", "unavailable")


def test_only_the_submitted_withdrawal_allows_expected_unloading():
    m = monitor()
    establish(m)
    assert m.update(*inputs(1.12, force=3, withdrawal=True)).status == "contact"
    released = m.update(*inputs(1.14, force=0, x=-0.001, withdrawal=True))
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
    result = m.update(*inputs(1.12, force=0, withdrawal=True, rotation=0.001, step_m=0.000001))
    assert result.status == "lost"
    assert result.reason == "contact_load_lost_or_uncertain"
