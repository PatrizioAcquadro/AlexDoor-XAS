"""Essential geometry, kinematics and qualification decision contracts."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alexdoor_xas.qualification.synthetic_probe import SustainedAngle


def test_push_reference_does_not_chase_coasting_panel_or_exceed_lead_or_stop():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, push_reference

    setup = ProbeSetup()
    assert push_reference(0.2, 0.3, setup, 1 / 60, 1.0) == pytest.approx(
        0.2 + setup.angular_speed / 60
    )
    assert push_reference(0.2, 0.2, setup, 1 / 60, 1.0) == pytest.approx(
        0.2 + setup.angular_speed / 60
    )
    assert push_reference(0.4, 0.3, setup, 1 / 60, 1.0) == 0.3 + setup.lead_angle
    assert push_reference(0.999, 0.999, setup, 1 / 60, 1.0) == 1.0


def test_hold_reference_preserves_material_lead_for_separate_pressure_control():
    from alexdoor_xas.qualification.synthetic_probe import hold_reference

    assert hold_reference(0.9, 0.003, 2) == pytest.approx(0.903)
    assert hold_reference(0.91, 0.003, 2) == pytest.approx(0.913)
    assert hold_reference(0.9, 0.003, 0.901) == 0.901


def test_tracking_reserve_anticipates_growth_but_does_not_amplify_settling():
    from alexdoor_xas.qualification.synthetic_probe import tracking_reserve

    assert tracking_reserve([(0, 0.001), (0.5, 0.002)], 3) == pytest.approx(0.008)
    assert tracking_reserve([(0, 0.003), (0.5, 0.002)], 3) == pytest.approx(0.002)
    assert tracking_reserve([(0, 0.002)], 3) == pytest.approx(0.002)


def test_normal_support_removes_excess_preload_and_is_bounded():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, normal_contact_support

    setup = ProbeSetup()
    initial = setup.compression_m
    assert normal_contact_support(initial, setup.contact_load_guard_n, setup, 1 / 60) == initial
    assert normal_contact_support(initial, 1.0, setup, 1 / 60) == pytest.approx(
        initial + setup.normal_admittance_m_n_s * (setup.contact_load_guard_n - 1.0) / 60
    )
    assert normal_contact_support(initial, 0.0, setup, 1 / 60) == pytest.approx(
        initial + setup.normal_admittance_m_n_s * setup.contact_load_guard_n / 60
    )
    maximum = setup.position_tolerance - setup.material_drift_guard_m
    assert normal_contact_support(maximum, 0.0, setup, 1 / 60) == maximum
    assert normal_contact_support(-maximum, 1.0, setup, 1 / 60) == -maximum


def test_tracking_guard_reserves_hold_margin_but_accepts_stable_bias_and_contact_chatter():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, tracking_margin_exhausted

    setup = ProbeSetup()
    stable = [(tick / 60, 0.00253) for tick in range(31)]
    assert not tracking_margin_exhausted(stable, setup)
    transient = stable[:-1] + [(0.5, 0.0035)]
    assert not tracking_margin_exhausted(transient, setup)
    growing = [(tick / 60, 0.003 + 0.002 * tick / 60) for tick in range(31)]
    assert tracking_margin_exhausted(growing, setup)
    assert not tracking_margin_exhausted(growing[-2:], setup)
    assert tracking_margin_exhausted([(0, 0.0075)], setup)


def test_angular_guard_accepts_stable_bias_but_stops_growth_before_invalid_hold():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, tracking_margin_exhausted

    setup = ProbeSetup()
    slow = [(tick / 60, np.deg2rad(2.6 + 0.026 * tick / 60)) for tick in range(31)]
    assert not tracking_margin_exhausted(slow, setup, orientation=True)
    stable = [(tick / 60, np.deg2rad(4.0)) for tick in range(31)]
    assert not tracking_margin_exhausted(stable, setup, orientation=True)
    growing = [(tick / 60, np.deg2rad(3.0 + tick / 60)) for tick in range(31)]
    assert tracking_margin_exhausted(growing, setup, orientation=True)
    assert not tracking_margin_exhausted(growing[-2:], setup, orientation=True)
    assert tracking_margin_exhausted([(0, setup.orientation_tolerance)], setup, orientation=True)


def test_tangential_bias_correction_freezes_when_disabled_and_limits_rate_and_extent():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, tangential_compensation

    setup = ProbeSetup()
    offset = np.zeros(3)
    np.testing.assert_array_equal(
        tangential_compensation(offset, [1, 1, 1], False, setup, 0.1), offset
    )
    offset = tangential_compensation(offset, [1, 1, 1], True, setup, 0.1)
    assert offset[0] == 0
    assert np.linalg.norm(offset) == pytest.approx(setup.compression_m * 0.1 / setup.sustain_s)
    for _ in range(100):
        offset = tangential_compensation(offset, [1, 1, 1], True, setup, 0.1)
    bound = setup.position_tolerance - setup.material_drift_guard_m
    assert np.linalg.norm(offset) == pytest.approx(bound)
    correction = tangential_compensation(offset, [1, -1, -1], True, setup, 0.1)
    assert np.linalg.norm(correction) < bound
    assert correction[0] == 0


@pytest.mark.parametrize("sign", [-1, 1])
def test_release_clears_surface_before_rotating_in_either_opening_direction(sign):
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, release_reference

    setup = ProbeSetup()
    start = np.array([0.2, -0.15 * sign, 1.09])
    rotation = Rotation.from_euler("z", sign * 60, degrees=True).as_matrix()
    retreat = start + rotation @ np.array([-0.05, 0.01, 0.0])
    retreat_rotation = Rotation.from_euler("z", sign * 55, degrees=True).as_matrix()
    for elapsed in np.linspace(0, setup.release_s / 2, 31):
        point, orientation = release_reference(
            start, rotation, retreat, retreat_rotation, elapsed, setup
        )
        displacement = (point - start) @ rotation
        np.testing.assert_allclose(displacement[1:], 0, atol=1e-12)
        assert -setup.precontact_m - 1e-12 <= displacement[0] <= 0
        np.testing.assert_allclose(orientation, rotation, atol=1e-12)
    assert displacement[0] == pytest.approx(-setup.precontact_m)
    point, orientation = release_reference(
        start, rotation, retreat, retreat_rotation, setup.release_s, setup
    )
    np.testing.assert_allclose(point, retreat, atol=1e-12)
    np.testing.assert_allclose(orientation, retreat_rotation, atol=1e-12)


def test_contact_load_tolerates_impulse_chatter_but_rejects_unloaded_or_detached_points():
    from alexdoor_xas.qualification.synthetic_probe import ContactLoad

    load = ContactLoad(0.1, 0.02, 0.0001)
    assert not load.update(0.0, 0.0, 0.0)
    assert load.update(0.02, 0.2, 0.0)
    assert load.update(0.04, 0.0, 0.00001)
    assert not load.update(0.06, 0.1, 0.001)
    assert not load.update(0.08, 0.1, None)
    for tick in range(5, 11):
        load.update(tick * 0.02, 0.0, 0.0)
    assert not load.update(0.22, 0.0, 0.0)


def test_sustain_requires_contiguous_loaded_hold_and_uses_lower_angle():
    window = SustainedAngle(0.5)
    for tick in range(5):
        window.update(tick * 0.1, 1.0, True)
    assert window.maximum is None
    window.update(0.5, 0.9, True)
    assert window.maximum == pytest.approx(0.9)
    window.update(0.6, 2.0, False)
    for tick in range(7, 12):
        window.update(tick * 0.1, 2.0, True)
    assert window.maximum == pytest.approx(0.9)
    window.update(1.2, 2.0, True)
    assert window.maximum == pytest.approx(2.0)


def test_final_hold_requires_a_valid_window_at_the_end_not_only_an_earlier_maximum():
    window = SustainedAngle(0.5)
    for tick in range(6):
        window.update(tick / 10, 1.0, True)
    assert window.current == 1.0
    window.update(0.6, 1.1, False)
    assert window.current is None and window.maximum == 1.0
    for tick in range(7, 13):
        window.update(tick / 10, 0.9, True)
    assert window.current == 0.9 and window.maximum == 1.0


def test_trial_qualification_rejects_unresolved_stops_and_inconsistent_causes():
    from alexdoor_xas.qualification.synthetic_probe import summarize_trials

    trial = dict(
        case="left-0.65",
        passed=True,
        released=True,
        angle_deg=60.0,
        stop_reason="safety_stop",
        safety_detail="tracking_margin",
        joint_margin=0.1,
        peak_force_n=5.0,
        clearance_m=0.02,
    )
    assert summarize_trials([trial, {**trial, "angle_deg": 62.0}])["angle_deg"] == 60.0
    assert not summarize_trials([trial, {**trial, "angle_deg": 62.01}])["passed"]
    assert not summarize_trials([trial, {**trial, "safety_detail": "declining_contact_load"}])[
        "passed"
    ]
    for cause in ("timeout", "solver_tracking_stall", "lost_contact", "invalid_physics"):
        assert not summarize_trials([{**trial, "stop_reason": cause}])["passed"]
    for cause in ("kinematic_limit", "mechanical_stop"):
        assert summarize_trials([{**trial, "stop_reason": cause}])["passed"]
    assert not summarize_trials([{**trial, "released": False}])["passed"]
    below = summarize_trials([{**trial, "angle_deg": 44.99}])
    assert below["passed"] and not below["meets_45_deg"]


def test_visibility_rejects_occlusion_invalid_depth_and_out_of_frame():
    from alexdoor_xas.qualification.visibility import visible_points

    intrinsics = np.array([[10.0, 0, 10], [0, 10.0, 10], [0, 0, 1]])
    depth = np.full((21, 21), 2.0)
    points = np.array([[0, 0, 2.0], [0, 0, 3.0], [10, 0, 2.0], [0, 0, -1.0]])
    assert visible_points(points, np.zeros(3), np.eye(3), intrinsics, depth).tolist() == [
        True,
        False,
        False,
        False,
    ]
    depth[:] = np.nan
    assert not visible_points(points, np.zeros(3), np.eye(3), intrinsics, depth).any()


def test_local_joint_limit_requires_blocked_direction():
    from alexdoor_xas.qualification.limits import constrained_step_evidence

    jac = np.c_[np.eye(6), np.zeros(6)]
    limits = np.array([[-1.0, 1.0]] * 7)
    q = np.zeros(7)
    q[0] = 1.0
    outward = constrained_step_evidence(jac, q, limits, np.array([0.02, 0, 0, 0, 0, 0]))
    inward = constrained_step_evidence(jac, q, limits, np.array([-0.02, 0, 0, 0, 0, 0]))
    assert outward["blocked"] and outward["active_joint_indices"] == [0]
    assert not inward["blocked"]


def test_conditioning_guard_reserves_hold_without_an_angle_threshold():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, conditioning_margin_exhausted

    setup = ProbeSetup()
    stable = [(t, 0.025) for t in np.linspace(0, 0.5, 31)]
    declining = [(t, 0.05 - 0.01 * t) for t in np.linspace(0, 0.5, 31)]
    assert not conditioning_margin_exhausted(stable, setup)
    assert conditioning_margin_exhausted(declining, setup)
    assert not conditioning_margin_exhausted(declining[-2:], setup)
    assert conditioning_margin_exhausted([(0, 2 * setup.ik_damping)], setup)


def test_pressure_hold_reduces_force_target_without_changing_physical_contact_score():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, normal_contact_support

    setup = ProbeSetup()
    assert setup.loaded_force_n == 0.02 and setup.contact_gap_tolerance_m == 0.0001
    assert (
        normal_contact_support(
            0.003, setup.contact_load_guard_n, setup, 1 / 60, target_force=setup.hold_force_n
        )
        < 0.003
    )


def test_hold_handoff_carries_completed_reference_velocity_without_extra_lead():
    from alexdoor_xas.qualification.synthetic_probe import hold_initial_lead, hold_reference

    reference, angle, step = 0.9, 0.899, 0.00015
    lead = hold_initial_lead(reference, angle, step, 0.005)
    assert hold_reference(angle, lead, 2) == pytest.approx(reference + step)
    assert hold_initial_lead(reference, angle, 0, 0.005) == pytest.approx(reference - angle)
    assert hold_initial_lead(reference, angle, 0.02, 0.005) == 0.005
    assert hold_reference(angle, lead, 0.9001) == 0.9001


def test_measured_separating_gap_has_bounded_support_before_contact_deadline():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, normal_contact_support

    setup = ProbeSetup()
    dt, initial = 1 / 60, 0.00023
    force_only = normal_contact_support(initial, 0, setup, dt)
    measured = normal_contact_support(initial, 0, setup, dt, gap_m=30e-6)
    assert measured - initial == pytest.approx(30e-6 + force_only - initial)
    assert measured > force_only
    for gap in (None, 0, -1e-4):
        assert normal_contact_support(initial, 0, setup, dt, gap_m=gap) == force_only
    # A contradictory positive gap never defeats unloading under actual force.
    assert normal_contact_support(initial, 1, setup, dt, gap_m=0.01) < initial
    limited = normal_contact_support(initial, 0, setup, dt, gap_m=0.1)
    assert limited - initial == pytest.approx(setup.compression_m / setup.sustain_s * dt)
    maximum = setup.position_tolerance - setup.material_drift_guard_m
    assert normal_contact_support(maximum, 0, setup, dt, gap_m=0.1) == maximum
    with pytest.raises(ValueError, match="contact gap"):
        normal_contact_support(initial, 0, setup, dt, gap_m=np.nan)


def test_achieved_contact_hold_reconstructs_position_in_rotated_material_frame():
    from types import SimpleNamespace

    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, achieved_contact_hold_state

    origin = np.array([0.3, -0.4, 1.09])
    rotation = Rotation.from_euler("z", 0.9).as_matrix()
    door = SimpleNamespace(contact_pose=lambda *args: (origin, rotation))
    local = np.array([-0.0002, 0.001, -0.0003])
    position = origin + rotation @ local
    row = dict(valid=True, physical_safe=True, angle=0.9, tool_position=position)
    angle, compression, tangent = achieved_contact_hold_state(door, ProbeSetup(), row)
    assert angle == 0.9
    np.testing.assert_allclose(origin + rotation @ (tangent + [compression, 0, 0]), position)
    row["physical_safe"] = False
    with pytest.raises(ValueError, match="Supported physical contact"):
        achieved_contact_hold_state(door, ProbeSetup(), row)
    row.update(physical_safe=True, tool_position=origin + rotation @ [0.008, 0, 0])
    with pytest.raises(ValueError, match="offset bounds"):
        achieved_contact_hold_state(door, ProbeSetup(), row)


def test_geometric_catchup_does_not_starve_positive_force_error():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, normal_contact_support

    setup = ProbeSetup()
    dt, gap = 1 / 60, 6e-6
    pressure = setup.normal_admittance_m_n_s * setup.contact_load_guard_n * dt
    moved = normal_contact_support(0, 0, setup, dt, gap_m=gap)
    assert moved == pytest.approx(gap + pressure)
    assert moved > gap  # touching geometry alone cannot establish the requested load
    # Existing force unloading wins even when geometry reports a separating gap.
    unloaded = normal_contact_support(0, 1, setup, dt, gap_m=gap)
    assert unloaded == normal_contact_support(0, 1, setup, dt)
    assert unloaded < 0


def test_contact_pause_reserves_its_physical_duration_without_changing_nominal_guard():
    from alexdoor_xas.qualification.collection_corrections import CollectionCorrection
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, conditioning_margin_exhausted

    setup = ProbeSetup()
    pause = CollectionCorrection("D", "correction-2", setup, 1 / 60)
    extra = pause.additional_contact_reserve_s()
    assert extra == 1.0 + pause.corrective_transition_s
    assert (
        CollectionCorrection("D", "correction-1", setup, 1 / 60).additional_contact_reserve_s() == 0
    )
    horizon = setup.hold_settle_s + setup.release_s
    end_sigma = 2 * setup.ik_damping + 0.003 * horizon + 0.0015
    samples = [(t, end_sigma + 0.003 * (0.5 - t)) for t in np.linspace(0, 0.5, 31)]
    assert not conditioning_margin_exhausted(samples, setup)
    assert conditioning_margin_exhausted(samples, setup, extra_contact_s=extra)
    with pytest.raises(ValueError, match="contact reserve"):
        conditioning_margin_exhausted(samples, setup, extra_contact_s=-1)


def test_default_teacher_contact_ramp_preserves_exact_existing_reference():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, contact_ramp_offset

    setup = ProbeSetup()
    for tick in range(round(setup.contact_s * 60)):
        elapsed = (tick + 1) / 60
        previous = -setup.precontact_m + (setup.precontact_m + setup.compression_m) * min(
            1.0, elapsed / (setup.contact_s * 0.7)
        )
        assert contact_ramp_offset(elapsed, setup) == previous


def test_eased_teacher_contact_ramp_reduces_reference_speed_at_first_contact():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, contact_ramp_offset

    setup = ProbeSetup(contact_ramp_profile="ease_out", contact_ramp_compression_m=0.001)
    offsets = [contact_ramp_offset(t / 60, setup) for t in range(181)]
    assert offsets[0] == -setup.precontact_m
    assert offsets[-1] == pytest.approx(0.001)
    assert all(b >= a for a, b in zip(offsets[:-1], offsets[1:], strict=True))
    duration = setup.contact_s * 0.7
    geometric_contact_time = duration * (1 - np.sqrt(0.001 / (setup.precontact_m + 0.001)))
    h = 1e-5
    contact_speed = (
        contact_ramp_offset(geometric_contact_time + h, setup)
        - contact_ramp_offset(geometric_contact_time - h, setup)
    ) / (2 * h)
    old_speed = (setup.precontact_m + setup.compression_m) / duration
    assert contact_speed < 0.35 * old_speed
    assert setup.compression_m == 0.003  # Existing admittance rate/extent stays fixed.
    assert contact_ramp_offset(geometric_contact_time, setup) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("endpoint", [0, -0.001, 0.004, float("nan")])
def test_teacher_contact_ramp_rejects_endpoint_outside_existing_budget(endpoint):
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup

    with pytest.raises(ValueError, match="compression budget"):
        ProbeSetup(contact_ramp_compression_m=endpoint)


def test_teacher_contact_ramp_rejects_unknown_profile_and_bad_clock():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, contact_ramp_offset

    with pytest.raises(ValueError, match="ramp profile"):
        ProbeSetup(contact_ramp_profile="unknown")
    with pytest.raises(ValueError, match="ramp time"):
        contact_ramp_offset(-1, ProbeSetup())


def test_slow_tail_preserves_linear_prefix_and_slows_before_contact():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, contact_ramp_offset

    old = ProbeSetup()
    setup = ProbeSetup(contact_ramp_profile="slow_tail", contact_ramp_compression_m=0.001)
    speed = (old.precontact_m + old.compression_m) / (old.contact_s * 0.7)
    junction = (old.precontact_m - 0.005) / speed
    for time in np.arange(0, junction, 1 / 60):
        assert contact_ramp_offset(time, setup) == contact_ramp_offset(time, old)
    assert contact_ramp_offset(junction, setup) == pytest.approx(-0.005)
    assert contact_ramp_offset(setup.contact_s * 0.9, setup) == pytest.approx(0.001)
    offsets = np.array([contact_ramp_offset(t / 60, setup) for t in range(181)])
    assert np.all(np.diff(offsets) >= 0)
    assert offsets[-1] == pytest.approx(0.001)
    # The fixed tail starts before physical contact, retaining a loading tail.
    tail_speed = 0.006 / (setup.contact_s * 0.9 - junction)
    assert tail_speed < 0.35 * speed
    contact_time = junction + 0.005 / tail_speed
    assert junction < contact_time < setup.contact_s - 0.4
    assert setup.compression_m == old.compression_m
    assert setup.normal_admittance_m_n_s == old.normal_admittance_m_n_s


def test_slow_tail_requires_separated_initial_ramp():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup

    with pytest.raises(ValueError, match="5mm slowdown"):
        ProbeSetup(contact_ramp_profile="slow_tail", precontact_m=0.004)


def test_full_contact_continuity_rejects_short_early_loss_but_allows_release():
    from alexdoor_xas.qualification.synthetic_probe import (
        ProbeSetup,
        pre_release_contact_continuity,
    )

    setup = ProbeSetup()
    rows = [
        dict(time=i / 60, angle=0.01, phase="contact", loaded=True, valid=True) for i in range(40)
    ]
    released = rows + [dict(time=40 / 60, angle=0.01, phase="release", loaded=False, valid=False)]
    assert pre_release_contact_continuity(released, setup)["passed"]
    lost = rows + [dict(time=40 / 60, angle=0.01, phase="push", loaded=False, valid=False)]
    result = pre_release_contact_continuity(lost, setup)
    assert not result["passed"]
    assert result["contact_loss_times_s"] == [40 / 60]
    # The old teacher tolerated this short loss for controller recovery. It
    # cannot qualify a continuous full task merely by reacquiring afterward.
    recovered = lost + [
        dict(time=i / 60, angle=0.02, phase="push", loaded=True, valid=True) for i in range(41, 90)
    ]
    assert not pre_release_contact_continuity(recovered, setup)["passed"]


def test_unresolved_acquisition_does_not_become_established_contact_loss():
    from alexdoor_xas.qualification.synthetic_probe import (
        ProbeSetup,
        pre_release_contact_continuity,
    )

    rows = [
        dict(time=0, angle=0, phase="contact", loaded=True, valid=True),
        dict(time=1 / 60, angle=0, phase="contact", loaded=False, valid=False),
    ]
    rows += [
        dict(time=i / 60, angle=0.01, phase="push", loaded=True, valid=True) for i in range(2, 40)
    ]
    result = pre_release_contact_continuity(rows, ProbeSetup())
    assert result["passed"] and not result["contact_loss_times_s"]


def test_teacher_push_ramp_preserves_speed_lead_and_mechanical_limits():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, push_reference

    setup = ProbeSetup(push_ramp_s=1)
    dt = 1 / 60
    original = min(0.2 + setup.angular_speed * dt, 0.3 + setup.lead_angle, 1)
    assert push_reference(0.2, 0.3, setup, dt, 1) == original
    first = push_reference(0.2, 0.3, setup, dt, 1, speed_scale=dt / setup.push_ramp_s)
    assert first - 0.2 == pytest.approx((original - 0.2) / 60)
    assert push_reference(0.4, 0.3, setup, dt, 1, speed_scale=0.5) == 0.3 + setup.lead_angle
    assert push_reference(0.999, 0.999, setup, dt, 1, speed_scale=1) == 1
    for value in [-1, float("nan")]:
        with pytest.raises(ValueError, match="push ramp"):
            ProbeSetup(push_ramp_s=value)
    with pytest.raises(ValueError, match="speed scale"):
        push_reference(0.2, 0.3, setup, dt, 1, speed_scale=2)


def test_ordinary_continuity_gate_is_independent_of_extended_hold_timing():
    from alexdoor_xas.qualification.synthetic_probe import (
        ProbeSetup,
        pre_release_contact_continuity,
    )

    rows = [dict(time=i / 60, angle=0.01, phase="push", loaded=True, valid=True) for i in range(40)]
    rows.append(dict(time=40 / 60, angle=0.01, phase="push", loaded=False, valid=False))
    assert not pre_release_contact_continuity(rows, ProbeSetup())["passed"]
    for family in ("A", "B", "C", "D"):
        assert pre_release_contact_continuity(rows, ProbeSetup(), phase_family=family) is None
    assert pre_release_contact_continuity(rows, ProbeSetup(), intermediate_start=True) is None


def test_teacher_native_support_feedback_does_not_change_qualification_or_acquisition():
    from alexdoor_xas.qualification.synthetic_probe import (
        ProbeSetup,
        normal_contact_support,
        support_force_feedback,
    )

    previous = dict(native_mean_normal_force_n=0.0)
    old = ProbeSetup()
    setup = ProbeSetup(support_force_source="native_interval_mean")
    for phase in ("contact", "push", "hold"):
        assert support_force_feedback(0.2, previous, old, phase) == (
            0.2,
            "contact_qualification_window",
        )
    assert support_force_feedback(0.2, previous, setup, "contact") == (
        0.2,
        "contact_qualification_window",
    )
    for phase in ("push", "hold"):
        force, source = support_force_feedback(0.2, previous, setup, phase)
        assert (force, source) == (0.0, "preceding_native_interval_mean")
        # Stale above-target feedback withdraws despite a completed zero-load
        # interval. Causal native feedback instead requests bounded catch-up.
        assert normal_contact_support(-0.001, force, setup, 1 / 60) > -0.001
        assert normal_contact_support(-0.001, 0.2, setup, 1 / 60) < -0.001
    assert setup.contact_force_window_s == old.contact_force_window_s
    assert setup.loaded_force_n == old.loaded_force_n
    assert setup.normal_admittance_m_n_s == old.normal_admittance_m_n_s
    assert setup.compression_m == old.compression_m
    assert support_force_feedback(0.2, previous, old, "hold", native_handoff=True) == (
        0.0,
        "preceding_native_interval_mean",
    )
    with pytest.raises(ValueError, match="support force source"):
        ProbeSetup(support_force_source="current_truth")


def test_pending_distal_motion_is_not_counted_again_as_gap_catchup():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, normal_contact_support

    setup = ProbeSetup()
    dt = 1 / 60
    pressure = setup.normal_admittance_m_n_s * setup.contact_load_guard_n * dt
    assert normal_contact_support(
        0.0, 0.0, setup, dt, gap_m=40e-6, pending_normal_m=80e-6
    ) == pytest.approx(pressure)
    assert normal_contact_support(
        0.0, 0.0, setup, dt, gap_m=40e-6, pending_normal_m=30e-6
    ) == pytest.approx(pressure + 10e-6)
    legacy = normal_contact_support(0.0, 0.0, setup, dt, gap_m=40e-6)
    assert (
        normal_contact_support(0.0, 0.0, setup, dt, gap_m=40e-6, pending_normal_m=-30e-6) == legacy
    )
    assert normal_contact_support(0.0, 1.0, setup, dt, gap_m=40e-6, pending_normal_m=80e-6) < 0
    with pytest.raises(ValueError, match="pending normal"):
        normal_contact_support(0, 0, setup, dt, pending_normal_m=np.nan)


def test_pending_advance_includes_rotating_distal_points_not_only_tool_center():
    from alexdoor_xas.qualification.synthetic_probe import distal_pending_advance

    vertices = np.array([[0, -0.02, 0], [0, 0.02, 0]])
    origin = np.zeros(3)
    target = np.array([0.001, 0.0, 0.0])
    rotation = Rotation.from_euler("z", 0.1).as_matrix()
    assert distal_pending_advance(
        vertices, origin, np.eye(3), target, np.eye(3), np.array([1, 0, 0])
    ) == pytest.approx(0.001)
    assert (
        distal_pending_advance(vertices, origin, np.eye(3), target, rotation, np.array([1, 0, 0]))
        < 0
    )


def test_push_ramp_can_match_initial_motion_without_chasing_future_coasting():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, push_speed_scale

    old = ProbeSetup(push_ramp_s=1)
    setup = ProbeSetup(push_ramp_s=1, push_ramp_from_motion=True)
    initial = 0.2 * setup.angular_speed
    assert push_speed_scale(0, setup, initial) == pytest.approx(0.2)
    assert push_speed_scale(0.5, setup, initial) == pytest.approx(0.6)
    assert push_speed_scale(1.5, setup, initial) == 1.0
    assert push_speed_scale(0, setup, -initial) == 0.0
    assert push_speed_scale(0.1, setup, 3 * initial / 0.2) == 1.0
    assert push_speed_scale(1 / 60, old, initial) == 1 / 60
    assert push_speed_scale(1 / 60, ProbeSetup(), initial) == 1.0
    with pytest.raises(ValueError, match="handoff speed"):
        push_speed_scale(0, setup, np.nan)
    with pytest.raises(ValueError, match="must be boolean"):
        ProbeSetup(support_gap_antiwindup="true")


def test_guarded_qualification_feedback_holds_only_contradicted_unloading():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, support_force_feedback

    setup = ProbeSetup(support_force_source="guarded_qualification_window")
    previous = dict(native_mean_normal_force_n=0.0)
    force, reason = support_force_feedback(0.2, previous, setup, "push")
    assert (
        force == setup.contact_load_guard_n
        and reason == "qualification_window_stale_unloading_held"
    )
    assert support_force_feedback(0.05, previous, setup, "push")[0] == 0.05
    previous["native_mean_normal_force_n"] = 0.3
    assert support_force_feedback(0.2, previous, setup, "push")[0] == 0.2
    previous["native_mean_normal_force_n"] = 0
    assert (
        support_force_feedback(0.2, previous, setup, "hold", target_force=setup.hold_force_n)[0]
        == setup.hold_force_n
    )
    assert support_force_feedback(0.2, previous, setup, "contact")[0] == 0.2


def test_gap_recovery_can_be_one_pulse_without_changing_pressure_gain_or_limits():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, normal_contact_support

    setup = ProbeSetup()
    dt = 1 / 60
    frozen = setup.contact_load_guard_n
    pulse = normal_contact_support(0, frozen, setup, dt, gap_m=20e-6, gap_catchup=True)
    assert pulse == pytest.approx(20e-6)
    assert normal_contact_support(pulse, frozen, setup, dt, gap_m=20e-6, gap_catchup=False) == pulse
    assert normal_contact_support(0, frozen, setup, dt, gap_m=1, gap_catchup=True) == pytest.approx(
        setup.compression_m / setup.sustain_s * dt
    )


def test_online_continuity_refusal_matches_posthoc_first_loss_and_keeps_release_scope():
    from alexdoor_xas.qualification.synthetic_probe import (
        ContactContinuity,
        ProbeSetup,
        pre_release_contact_continuity,
    )

    rows = [
        dict(time=i / 60, angle=0.01, phase="contact", loaded=True, valid=True) for i in range(40)
    ]
    rows += [
        dict(time=40 / 60, angle=0.01, phase="push", loaded=False, valid=False),
        dict(time=41 / 60, angle=0.01, phase="release", loaded=False, valid=False),
    ]
    monitor = ContactContinuity(ProbeSetup().sustain_s)
    refusals = [r["time"] for r in rows if monitor.update(r)]
    assert refusals == [40 / 60]
    assert monitor.result() == pre_release_contact_continuity(rows, ProbeSetup())
