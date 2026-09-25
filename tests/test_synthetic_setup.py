"""Essential geometry, kinematics and qualification decision contracts."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from alexdoor_xas.qualification.synthetic_probe import SustainedAngle, rank_candidates


def test_push_reference_follows_coasting_panel_without_exceeding_lead_or_stop():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, push_reference

    setup = ProbeSetup()
    assert push_reference(0.2, 0.3, setup, 1 / 60, 1.0) == 0.3
    assert push_reference(0.2, 0.2, setup, 1 / 60, 1.0) == pytest.approx(
        0.2 + setup.angular_speed / 60
    )
    assert push_reference(0.4, 0.3, setup, 1 / 60, 1.0) == 0.3 + setup.lead_angle
    assert push_reference(0.999, 0.999, setup, 1 / 60, 1.0) == 1.0


def test_hold_reference_removes_lead_smoothly_but_tracks_the_moving_surface():
    from alexdoor_xas.qualification.synthetic_probe import hold_reference

    start, lead, duration = 0.9, np.deg2rad(0.3), 3.0
    times = np.linspace(0, duration, 301)
    angles = np.array([hold_reference(start, lead, t, duration, 2.0) for t in times])
    velocities = np.diff(angles) / np.diff(times)
    assert angles[0] == pytest.approx(start + lead)
    assert np.all(velocities <= 0)
    assert abs(velocities[0]) < lead * 0.01
    assert abs(velocities[-1]) < lead * 0.01
    assert angles[-1] == start
    assert hold_reference(start + 0.01, lead, duration, duration, 2.0) == start + 0.01
    assert hold_reference(start, lead, 0, duration, 0.901) == 0.901


def test_tracking_reserve_anticipates_growth_but_does_not_amplify_settling():
    from alexdoor_xas.qualification.synthetic_probe import tracking_reserve

    assert tracking_reserve([(0, 0.001), (0.5, 0.002)], 3) == pytest.approx(0.008)
    assert tracking_reserve([(0, 0.003), (0.5, 0.002)], 3) == pytest.approx(0.002)
    assert tracking_reserve([(0, 0.002)], 3) == pytest.approx(0.002)


def test_hold_support_only_restores_low_load_and_is_bounded():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, hold_contact_support

    setup = ProbeSetup()
    initial = setup.compression_m
    assert hold_contact_support(initial, setup.contact_load_guard_n, setup, 1 / 60) == initial
    assert hold_contact_support(initial, 1.0, setup, 1 / 60) == initial
    assert hold_contact_support(initial, 0.0, setup, 1 / 60) == pytest.approx(initial + 0.0001)
    maximum = setup.position_tolerance - setup.material_drift_guard_m
    assert hold_contact_support(maximum, 0.0, setup, 1 / 60) == maximum


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


def test_minimax_excludes_failures_and_ties_use_joint_margin():
    def candidate(angle, margin, passed=True):
        return {
            "cases": [
                dict(
                    passed=passed,
                    angle_deg=a,
                    joint_margin=margin,
                    peak_force_n=10.0,
                    clearance_m=0.02,
                )
                for a in (angle, angle + 10, angle + 1, angle + 2)
            ]
        }

    a, b, bad = candidate(60, 0.1), candidate(59.8, 0.2), candidate(90, 0.4, False)
    assert rank_candidates([a, b, bad], 0.5) == [b, a]
    assert rank_candidates([bad], 0.5) == []
    stronger = candidate(60.6, 0.01)
    assert rank_candidates([a, stronger], 0.5) == [stronger, a]
    lower_force = candidate(60, -1e-8)
    for case in lower_force["cases"]:
        case["peak_force_n"] = 5.0
    at_limit = candidate(60, 0.0)
    assert rank_candidates([at_limit, lower_force], 0.5)[0] is lower_force
    more_clearance = candidate(60, 0.0)
    for case in more_clearance["cases"]:
        case["clearance_m"] = 0.03
    assert rank_candidates([at_limit, more_clearance], 0.5)[0] is more_clearance


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


def test_urdf_chain_jacobian_matches_finite_difference():
    from ihmc_alex_isaaclab._paths import REPOSITORY_ROOT

    from alexdoor_xas.kinematics.purdue_chain import PurdueChain

    urdf = (
        REPOSITORY_ROOT
        / "assets/robots/alex_purdue/urdf/baseline/alex_purdue_wsg32_umi_v1_full_convex.urdf"
    )
    chain = PurdueChain(urdf, "cpu")
    q = torch.tensor([[-0.5, -0.5, 0.2, -1.0, 0.1, 0.2, 0.1]])
    pose, jac = chain.forward(q)
    for index in range(7):
        plus, minus = q.clone(), q.clone()
        plus[0, index] += 0.001
        minus[0, index] -= 0.001
        tp, _ = chain.forward(plus)
        tm, _ = chain.forward(minus)
        linear = (tp[0, :3, 3] - tm[0, :3, 3]).numpy() / 0.002
        angular = (
            Rotation.from_matrix((tp[0, :3, :3] @ tm[0, :3, :3].T).numpy()).as_rotvec() / 0.002
        )
        np.testing.assert_allclose(jac[0, :, index], np.r_[linear, angular], atol=2e-4)
    recovered, pe, re = chain.solve(pose[:, :3, 3], pose[:, :3, :3], q + 0.05)
    assert pe.item() < 1e-4 and re.item() < 1e-3


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
