"""Essential geometry, kinematics and qualification decision contracts."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from alexdoor_xas.qualification.synthetic_probe import SustainedAngle, rank_candidates


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
