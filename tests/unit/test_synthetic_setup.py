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
