"""Operational math and observation boundaries, without claiming physics evidence."""

import numpy as np
import pytest
import torch

from alexdoor_xas.envs.door_task.purdue_contacts import has_forbidden_contact
from alexdoor_xas.kinematics.pose_control import (
    bounded_joint_step,
    bounded_pose_step,
    pose_target_limits,
)
from alexdoor_xas.recording.rgbd import RGBDCapture


def test_forbidden_contact_requires_touch_or_load():
    candidate = dict(category="forbidden", separation_m=0.0033186, force_n=0.0)
    assert not has_forbidden_contact([candidate])
    assert has_forbidden_contact([{**candidate, "force_n": 0.01}])
    assert has_forbidden_contact([{**candidate, "force_n": -0.01}])
    assert has_forbidden_contact([{**candidate, "separation_m": 0.0}])
    # Fixed obstacles may report penetration with zero reaction force.
    assert has_forbidden_contact([{**candidate, "separation_m": -0.03}])
    for category in ("positive", "negative", "support"):
        assert not has_forbidden_contact(
            [{**candidate, "category": category, "separation_m": -0.001, "force_n": 2.0}]
        )


def test_contact_cover_clips_sloped_mesh_edges_without_inventing_pad_width():
    from scipy.spatial import ConvexHull

    from alexdoor_xas.assets.purdue import PushGeometry

    # Forward ridge at x=0.02; the sloping sides widen to +/-0.02 at x=0.
    vertices = np.array(
        [[x, y, z] for x, y in ((0, -0.02), (0, 0.02), (0.02, 0)) for z in (-0.01, 0.01)]
    )
    mesh = vertices[ConvexHull(vertices).simplices].reshape(-1, 3)
    geometry = PushGeometry(np.array([0.02, 0, 0]), np.eye(4), {"finger": mesh})
    (cover,) = geometry.contact_covers(0.003)
    np.testing.assert_allclose(cover[:, 0], 0)
    np.testing.assert_allclose(cover[:, 1:].min(0), [-0.003, -0.01])
    np.testing.assert_allclose(cover[:, 1:].max(0), [0.003, 0.01])
    # A vertex-only crop would leave the ridge and miss the entire sloped band.
    assert np.linalg.matrix_rank(geometry.distal_faces[0]) == 1
    for band in (0, -0.001, float("nan")):
        with pytest.raises(ValueError, match="positive finite"):
            geometry.contact_covers(band)


def test_pose_control_rotates_and_centers_without_changing_primary_task():
    jac = torch.zeros(1, 6, 7)
    jac[0, :, :6] = torch.eye(6)
    q = torch.zeros(1, 7)
    q[0, 6] = 0.5
    limits = torch.tensor([[[-1.0, 1.0]] * 7])
    error = torch.tensor([[0.0, 0, 0, 0.02, -0.02, 0.01]])
    target, _ = bounded_pose_step(jac, error, q, limits, torch.ones_like(q), 0.1)
    assert target[0, 3] > 0 and target[0, 4] < 0 and target[0, 5] > 0
    assert target[0, 6] < q[0, 6]
    assert torch.equal(target[0, :3], q[0, :3])
    repeat, _ = bounded_pose_step(jac, error, q, limits, torch.ones_like(q), 0.1)
    assert torch.equal(target, repeat)


def test_pose_velocity_limit_preserves_nullspace_motion():
    jac = torch.zeros(1, 6, 7)
    jac[0, :, :6] = torch.eye(6)
    jac[0, 0, 6] = 1
    q = torch.zeros(1, 7)
    q[0, 0], q[0, 6] = 0.5, -0.5
    limits = torch.tensor([[[-1.0, 1.0]] * 7])
    velocities = torch.ones_like(q) * 0.2
    velocities[0, 0] = 0.1
    target, clipped = bounded_pose_step(
        jac, torch.zeros(1, 6), q, limits, velocities, 0.1, centering_gain=2.0
    )
    # Centering may move joints, but cannot move the hand when the pose error is zero.
    assert not torch.equal(target, q)
    torch.testing.assert_close(jac @ (target - q).unsqueeze(-1), torch.zeros(1, 6, 1))
    assert bool(((target - q).abs() <= velocities * 0.1 + 1e-7).all())
    assert float(clipped.max()) > 0


def test_joint_command_bounds_and_invalid_input():
    q = torch.ones(1, 7) * 0.99
    limits = torch.tensor([[[-1.0, 1.0]] * 7])
    target, clipped = bounded_joint_step(torch.ones_like(q), q, limits, torch.ones_like(q), 0.1)
    assert torch.equal(target, torch.ones_like(q))
    assert bool((clipped > 0).all())
    with pytest.raises(ValueError):
        bounded_joint_step(torch.zeros(1, 6), q, limits, torch.ones_like(q), 0.1)
    with pytest.raises(ValueError):
        bounded_joint_step(q * float("nan"), q, limits, torch.ones_like(q), 0.1)


def test_capture_mask_freshness_and_reset():
    capture = RGBDCapture(0.1, 5.0)
    rgb = torch.zeros(1, 1, 6, 3, dtype=torch.uint8)
    depth = torch.tensor([0.0, 0.05, 1.0, 6.0, float("inf"), float("nan")]).reshape(1, 1, 6, 1)
    q = torch.zeros(1, 9)
    sample = capture.capture(1, 0.0, rgb, depth, q, q)
    assert sample.valid_depth.flatten().tolist() == [False, False, True, False, False, False]
    rgb.fill_(255)
    assert sample.rgb.max() == 0
    with pytest.raises(ValueError, match="fresh"):
        capture.capture(1, 0.0, rgb, depth, q, q)
    capture.reset()
    assert capture.sample is None
    assert capture.capture(1, 0.0, rgb, depth, q, q).episode == sample.episode + 1


def test_target_speed_is_bounded_against_previous_command():
    q = torch.zeros(1, 7)
    previous = torch.ones(1, 7) * 0.1
    limits = torch.tensor([[[-1.0, 1.0]] * 7])
    targets, _ = bounded_joint_step(q + 0.2, q, limits, q + 0.5, 0.02, previous)
    torch.testing.assert_close(targets, previous + 0.01)


def test_fk_reuse_owns_joint_keys_and_results_and_keeps_calibration_current(monkeypatch):
    from alexdoor_xas.kinematics import purdue_chain
    from alexdoor_xas.policies.purdue import PurdueFK

    class Chain:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        @staticmethod
        def array(value):
            return torch.as_tensor(value, dtype=torch.float32)

        def forward(self, q, *, with_jacobian):
            assert not with_jacobian
            self.calls += 1
            pose = torch.eye(4)[None].clone()
            pose[:, :3, 3] = q[:, :3]
            return pose, None

    monkeypatch.setattr(purdue_chain, "PurdueChain", Chain)
    fk = PurdueFK("unused", "cpu")
    joints, root = np.zeros(9), np.eye(4)
    first = fk(joints, {"root_world": root})
    first.origin[:] = 99
    root[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    root[:3, 3] = [1, 2, 3]
    joints[7:] = 0.1
    current = fk(joints, {"root_world": root})
    np.testing.assert_array_equal(current.origin, root[:3, 3])
    np.testing.assert_array_equal(current.rot, root[:3, :3])
    assert fk.chain.calls == 1
    joints[0] = 0.1
    fk(joints, {"root_world": root})
    joints[0] = 0
    fk(joints, {"root_world": root})
    assert fk.chain.calls == 2
    # Bounded reuse cannot hide invalid current input, including ignored neck FK.
    joints[7] = np.nan
    with pytest.raises(ValueError):
        fk(joints, {"root_world": root})
    joints[7] = 0
    for value in (0.2, 0.3, 0.4, 0.5):
        joints[0] = value
        fk(joints, {"root_world": root})
    before = fk.chain.calls
    joints[0] = 0
    fk(joints, {"root_world": root})
    assert fk.chain.calls == before + 1


def test_pose_request_is_captured_before_position_and_rate_limits():
    jac = torch.zeros(1, 6, 7)
    jac[0, :, :6] = torch.eye(6)
    q = torch.zeros(1, 7)
    error = torch.ones(1, 6)
    limits = torch.tensor([[[-0.1, 0.1]] * 7])
    target, clipped, requested = bounded_pose_step(
        jac,
        error,
        q,
        limits,
        torch.ones_like(q),
        0.01,
        centering_gain=0,
        return_request=True,
    )
    ordinary = bounded_pose_step(jac, error, q, limits, torch.ones_like(q), 0.01, centering_gain=0)
    torch.testing.assert_close(target, ordinary[0], rtol=0, atol=0)
    torch.testing.assert_close(clipped, ordinary[1], rtol=0, atol=0)
    assert requested[0, :6].min() > 0.9
    assert target.max() <= 0.010001


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("coupling", [0.7, 0.73])
def test_pose_command_reserve_preserves_physical_bounds_and_target_rate(sign, coupling):
    jac = torch.zeros(1, 6, 7)
    jac[0, :, :6] = torch.eye(6)
    jac[0, 4, 6] = coupling
    limits = torch.tensor([[[-1.0, 1.0]] * 7])
    original = limits.clone()
    reserved = pose_target_limits(limits, [0, 0, 0, 0, 0.12, 0, 0.02])
    q = torch.zeros(1, 7)
    q[0, 4], q[0, 6] = sign * 0.88, sign * 0.4
    error = torch.zeros(1, 6)
    error[0, 4] = sign * 10
    speed = torch.ones_like(q) * 0.5
    target, _ = bounded_pose_step(jac, error, q, reserved, speed, 0.05, centering_gain=0)
    ordinary, _ = bounded_pose_step(jac, error, q, limits, speed, 0.05, centering_gain=0)
    assert bool((target >= reserved[..., 0]).all() and (target <= reserved[..., 1]).all())
    assert bool(((target - q).abs() <= speed * 0.05 + 1e-7).all())
    assert sign * ordinary[0, 4] > sign * target[0, 4]
    torch.testing.assert_close(limits, original, rtol=0, atol=0)
    # A1 commands keep their own original physical/rate contract.
    joint, _ = bounded_joint_step(sign * torch.ones_like(q), q, limits, speed, 0.05)
    assert sign * joint[0, 4] > sign * target[0, 4]
    default = bounded_pose_step(jac, error, q, pose_target_limits(limits), speed, 0.05)
    previous = bounded_pose_step(jac, error, q, limits, speed, 0.05)
    for actual, expected in zip(default, previous, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "margin", [[0] * 6, [-0.01] * 7, [float("nan")] * 7, [float("inf")] * 7, [1] * 7]
)
def test_pose_command_reserve_refuses_unmeasured_or_empty_domains(margin):
    limits = torch.tensor([[[-1.0, 1.0]] * 7])
    original = limits.clone()
    with pytest.raises(ValueError, match="pose_target_joint"):
        pose_target_limits(limits, margin)
    torch.testing.assert_close(limits, original, rtol=0, atol=0)


@pytest.mark.parametrize("margin", [None, [0.1] * 7])
def test_pose_target_domain_intersects_physics_before_reserving_targets(margin):
    limits = torch.tensor([[[-1.0, 1.0]] * 7, [[-0.8, 0.8]] * 7])
    original = limits.clone()
    domain = torch.tensor([[-0.9, 0.7]] * 7)
    saved_domain = domain.clone()
    actual = pose_target_limits(limits, margin, domain=domain)
    expected = torch.tensor([[[-0.9, 0.7]] * 7, [[-0.8, 0.7]] * 7])
    if margin is not None:
        expected[..., 0] += 0.1
        expected[..., 1] -= 0.1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(limits, original, rtol=0, atol=0)
    torch.testing.assert_close(domain, saved_domain, rtol=0, atol=0)
    enclosing = pose_target_limits(limits, domain=[[-2.0, 2.0]] * 7)
    torch.testing.assert_close(enclosing, limits, rtol=0, atol=0)


@pytest.mark.parametrize(
    "domain, margin",
    [
        ([[-0.5, 0.5]] * 6, None),
        ([[float("nan"), 0.5]] * 7, None),
        ([[-0.5, float("inf")]] * 7, None),
        ([[0.5, -0.5]] * 7, None),
        ([[1.0, 2.0]] * 7, None),
        ([[-0.05, 0.05]] * 7, [0.05] * 7),
    ],
)
def test_pose_target_domain_rejects_invalid_or_empty_intersections(domain, margin):
    limits = torch.tensor([[[-1.0, 1.0]] * 7])
    original = limits.clone()
    with pytest.raises(ValueError, match="pose_target_joint_domain"):
        pose_target_limits(limits, margin, domain=domain)
    torch.testing.assert_close(limits, original, rtol=0, atol=0)


def test_pose_domain_cannot_override_the_original_target_rate():
    jac = torch.zeros(1, 6, 7)
    jac[0, :, :6] = torch.eye(6)
    physical = torch.tensor([[[-1.0, 1.0]] * 7])
    domain = pose_target_limits(physical, domain=[[-0.5, 0.5]] * 7)
    q = torch.full((1, 7), 0.9)
    speed = torch.full_like(q, 0.5)
    with pytest.raises(ValueError, match="pose_target_domain_outside_rate_limits"):
        bounded_pose_step(jac, torch.zeros(1, 6), q, domain, speed, 0.05)
    # A nearby boundary can be entered without exceeding the original rate.
    q.fill_(0.51)
    target, _ = bounded_pose_step(jac, torch.zeros(1, 6), q, domain, speed, 0.05)
    assert bool(((target - q).abs() <= speed * 0.05).all())
    assert bool((target <= domain[..., 1]).all())


@pytest.mark.parametrize("coupling", [0.0, 0.4, -0.7])
def test_pose_feedback_balances_constant_deflection_and_keeps_preview_pure(coupling):
    # This linear equilibrium checks controller bias rejection, not native dynamics.
    jac = torch.zeros(1, 6, 7, dtype=torch.float64)
    jac[0, :, :6] = torch.eye(6, dtype=torch.float64)
    jac[0, 0, 6] = coupling
    limits = torch.tensor([[[-1.0, 1.0]] * 7], dtype=torch.float64)
    previous = torch.zeros(1, 7, dtype=torch.float64)
    speed = torch.ones_like(previous) * 0.5
    deflection = torch.tensor([[0.03, -0.02, 0.01, 0, 0.02, -0.01, 0.01]])
    goal = torch.tensor([[0.2, -0.1, 0.1, 0.02, -0.03, 0.01]], dtype=torch.float64)
    for _ in range(400):
        q = previous - deflection
        error = goal - (jac @ q.unsqueeze(-1)).squeeze(-1)
        before = previous.clone()
        arguments = dict(previous_targets=previous, feedback_time_constant_s=0.5, centering_gain=0)
        target, excess = bounded_pose_step(jac, error, q, limits, speed, 0.02, **arguments)
        repeat = bounded_pose_step(jac, error, q, limits, speed, 0.02, **arguments)
        torch.testing.assert_close(previous, before, rtol=0, atol=0)
        torch.testing.assert_close(target, repeat[0], rtol=0, atol=0)
        torch.testing.assert_close(excess, repeat[1], rtol=0, atol=0)
        assert bool(((target - previous).abs() <= speed * 0.02 + 1e-12).all())
        previous = target
    measured_pose = (jac @ (previous - deflection).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(measured_pose, goal, rtol=0, atol=1e-6)
    # At zero measured pose error, a nonzero balancing target offset is retained.
    target, _ = bounded_pose_step(
        jac,
        torch.zeros_like(goal),
        previous - deflection,
        limits,
        speed,
        0.02,
        previous_targets=previous,
        feedback_time_constant_s=0.5,
        centering_gain=0,
    )
    torch.testing.assert_close(target, previous, rtol=0, atol=0)


def test_pose_feedback_saturation_does_not_accumulate_hidden_windup():
    jac = torch.zeros(1, 6, 7, dtype=torch.float64)
    jac[0, :, :6] = torch.eye(6, dtype=torch.float64)
    q = torch.zeros(1, 7, dtype=torch.float64)
    previous = q.clone()
    previous[0, 0] = 0.2
    limits = torch.tensor([[[-0.2, 0.2]] * 7], dtype=torch.float64)
    speed = torch.ones_like(q)
    for _ in range(10):
        target, _, requested = bounded_pose_step(
            jac,
            torch.ones(1, 6, dtype=torch.float64),
            q,
            limits,
            speed,
            0.01,
            previous_targets=previous,
            feedback_time_constant_s=0.1,
            centering_gain=0,
            return_request=True,
        )
        assert target[0, 0] == previous[0, 0] and requested[0, 0] > 0.2
    reverse, _ = bounded_pose_step(
        jac,
        -torch.ones(1, 6, dtype=torch.float64),
        q,
        limits,
        speed,
        0.01,
        previous_targets=previous,
        feedback_time_constant_s=0.1,
        centering_gain=0,
    )
    assert reverse[0, 0] < previous[0, 0]
    assert bool((reverse >= limits[..., 0]).all() and (reverse <= limits[..., 1]).all())
    assert bool(((reverse - previous).abs() <= speed * 0.01 + 1e-12).all())


@pytest.mark.parametrize("tau", [0, -1, 0.009, float("nan"), float("inf"), True])
def test_pose_feedback_refuses_invalid_or_overunity_gain(tau):
    jac = torch.eye(6, 7)[None]
    q = torch.zeros(1, 7)
    limits = torch.tensor([[[-1.0, 1.0]] * 7])
    with pytest.raises(ValueError, match="invalid_pose_feedback_time_constant"):
        bounded_pose_step(
            jac,
            torch.zeros(1, 6),
            q,
            limits,
            torch.ones_like(q),
            0.01,
            previous_targets=q,
            feedback_time_constant_s=tau,
        )


def test_pose_feedback_requires_finite_previous_delivery_and_allows_unit_gain():
    jac = torch.eye(6, 7)[None]
    q = torch.zeros(1, 7)
    limits = torch.tensor([[[-1.0, 1.0]] * 7])
    arguments = (jac, torch.zeros(1, 6), q, limits, torch.ones_like(q), 0.01)
    with pytest.raises(ValueError, match="pose_feedback_requires_previous_targets"):
        bounded_pose_step(*arguments, feedback_time_constant_s=0.01)
    for previous in (q[:, :6], q + float("nan")):
        with pytest.raises(ValueError, match="invalid_pose_feedback_previous_targets"):
            bounded_pose_step(*arguments, previous_targets=previous, feedback_time_constant_s=0.01)
    target, _ = bounded_pose_step(
        *arguments, previous_targets=q, feedback_time_constant_s=0.01, centering_gain=0
    )
    torch.testing.assert_close(target, q, rtol=0, atol=0)


@pytest.mark.parametrize("tau", [0.02, 0.5])
def test_causal_reference_tracks_moving_goal_without_one_tick_lead(tau):
    # A linear kinematic plant with constant elastic deflection distinguishes
    # correct next-tick tracking from paying the reference increment twice.
    dtype = torch.float64
    jac = torch.eye(6, 7, dtype=dtype)[None]
    limits = torch.tensor([[[-1.0, 1.0]] * 7], dtype=dtype)
    deflection = torch.tensor([[0.03, -0.02, 0.01, 0, 0.02, -0.01, 0]], dtype=dtype)
    previous = deflection.clone()
    increment = torch.tensor([[0.0002, -0.0001, 0, 0.0001, 0, -0.0001]], dtype=dtype)
    goal = torch.zeros_like(increment)
    for _ in range(120):
        goal = goal + increment
        measured = previous - deflection
        error = goal - measured[:, :6]
        target, _ = bounded_pose_step(
            jac, error, measured, limits, torch.ones_like(previous), 0.02,
            damping=1e-6, centering_gain=0, previous_targets=previous,
            feedback_time_constant_s=tau, reference_delta=increment,
        )
        torch.testing.assert_close((target-deflection)[:, :6], goal, rtol=0, atol=1e-11)
        previous = target
    held, _ = bounded_pose_step(
        jac, goal-(previous-deflection)[:, :6], previous-deflection,
        limits, torch.ones_like(previous), 0.02, damping=1e-6, centering_gain=0,
        previous_targets=previous, feedback_time_constant_s=tau,
        reference_delta=torch.zeros_like(goal),
    )
    torch.testing.assert_close(held, previous, rtol=0, atol=1e-11)


def test_zero_reference_increment_preserves_feedback_and_pure_preview():
    jac = torch.eye(6, 7)[None]
    q = torch.tensor([[0.1, -0.2, 0.05, 0, 0.1, -0.1, 0.2]])
    previous = q + 0.01
    limits = torch.tensor([[[-1.0, 1.0]] * 7])
    error = torch.tensor([[0.01, -0.02, 0.03, 0.02, 0.01, -0.01]])
    args = (jac, error, q, limits, torch.ones_like(q), 0.02)
    kw = dict(previous_targets=previous, feedback_time_constant_s=0.5, return_request=True)
    original = bounded_pose_step(*args, **kw)
    saved = previous.clone()
    for _ in range(2):
        result = bounded_pose_step(*args, reference_delta=torch.zeros_like(error), **kw)
        for actual, expected in zip(result, original, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(previous, saved, rtol=0, atol=0)


def test_reference_step_preserves_target_reserve_rate_and_no_hidden_windup():
    jac = torch.eye(6, 7, dtype=torch.float64)[None]
    q = torch.zeros(1, 7, dtype=torch.float64)
    previous = q.clone()
    previous[0, 0] = 0.19
    limits = torch.tensor([[[-0.2, 0.2]] * 7], dtype=torch.float64)
    speed = torch.ones_like(q) * 0.5
    error = torch.ones(1, 6, dtype=torch.float64) * 0.05
    kw = dict(previous_targets=previous, feedback_time_constant_s=0.5, centering_gain=0)
    first = bounded_pose_step(jac, error, q, limits, speed, 0.02,
                             reference_delta=error*100, **kw)[0]
    repeated = bounded_pose_step(jac, error, q, limits, speed, 0.02,
                                reference_delta=error*100, **kw)[0]
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert bool(((first-previous).abs() <= speed*0.02+1e-12).all())
    assert bool((first >= limits[..., 0]).all() and (first <= limits[..., 1]).all())
    reversed_target = bounded_pose_step(jac, -error, q, limits, speed, 0.02,
                                       reference_delta=-error*100, **kw)[0]
    assert reversed_target[0, 0] < previous[0, 0]


@pytest.mark.parametrize("defect", ["feedback_missing", "shape", "nonfinite"])
def test_reference_increment_requires_feedback_and_finite_pose_delta(defect):
    jac = torch.eye(6, 7)[None]
    q = torch.zeros(1, 7)
    limits = torch.tensor([[[-1.0, 1.0]] * 7])
    reference = torch.zeros(1, 6)
    kw = dict(previous_targets=q, feedback_time_constant_s=0.5)
    reason = "invalid_pose_reference_delta"
    if defect == "feedback_missing":
        kw.pop("feedback_time_constant_s")
        reason = "pose_reference_requires_feedback"
    elif defect == "shape":
        reference = reference[:, :5]
    else:
        reference += float("nan")
    with pytest.raises(ValueError, match=reason):
        bounded_pose_step(jac, torch.zeros(1, 6), q, limits, torch.ones_like(q),
                          0.02, reference_delta=reference, **kw)


@pytest.mark.parametrize("tau", [0.02, 0.5])
def test_compliant_normal_keeps_finite_wall_impedance_and_other_pose_tracking(tau):
    jac = torch.eye(6, 7, dtype=torch.float64)[None]
    q = torch.zeros(1, 7, dtype=torch.float64)
    limits = torch.tensor([[[-1., 1.]] * 7], dtype=torch.float64)
    previous = q.clone()
    previous[0, 0] = .02
    previous[0, 1] = -.01
    error = torch.tensor([[.0001, .001, 0, 0, 0, .002]], dtype=torch.float64)
    reference = torch.tensor([[.0002, .0003, 0, 0, 0, -.0001]], dtype=torch.float64)
    normal = torch.tensor([[1., 0, 0]], dtype=torch.float64)
    for _ in range(30):
        target, _, requested = bounded_pose_step(
            jac, error, q, limits, torch.ones_like(q) * 100, .02,
            damping=.01, centering_gain=0, previous_targets=previous,
            feedback_time_constant_s=tau, reference_delta=reference,
            compliant_normal=normal, return_request=True,
        )
        # A fixed wall cannot satisfy a positive penetration goal. Finite
        # normal impedance must not integrate another elastic target each tick.
        assert requested[0, 0] == pytest.approx(.0001, abs=1e-15)
        expected_y = previous[0, 1] + (
            .02 / tau * error[0, 1] + (1-.02/tau) * reference[0, 1]
        ) / (1+.01**2)
        assert requested[0, 1] == pytest.approx(float(expected_y), abs=1e-15)
        previous = target


@pytest.mark.parametrize("normal_scale", [1., -1., 7.])
def test_compliant_normal_preserves_coupled_tangent_and_rotation_requests(normal_scale):
    dtype = torch.float64
    jac = torch.tensor([[[1, .2, 0, .1, 0, 0, .1],
                         [.1, 1, .2, 0, 0, 0, .2],
                         [0, .1, 1, 0, .2, 0, .1],
                         [.1, 0, 0, 1, .1, 0, .2],
                         [0, .1, 0, 0, 1, .1, .1],
                         [0, 0, .1, .1, 0, 1, .1]]], dtype=dtype)
    q = torch.zeros(1, 7, dtype=dtype)
    previous = torch.tensor([[.03, -.02, .01, 0, -.01, .02, 0]], dtype=dtype)
    limits = torch.tensor([[[-1., 1.]] * 7], dtype=dtype)
    error = torch.tensor([[.001, -.002, .003, -.001, .002, .001]], dtype=dtype)
    reference = error / 10
    normal = torch.tensor([[.3, .4, 3**.5/2]], dtype=dtype)
    options = dict(previous_targets=previous, feedback_time_constant_s=.5,
                   reference_delta=reference, centering_gain=.1, return_request=True)
    _, _, old = bounded_pose_step(jac, error, q, limits, torch.ones_like(q)*100, .02, **options)
    _, _, new = bounded_pose_step(jac, error, q, limits, torch.ones_like(q)*100, .02,
                                  compliant_normal=normal*normal_scale, **options)
    task_axis = torch.cat((normal, torch.zeros_like(normal)), dim=-1)
    normal_change = ((jac @ (new-q).unsqueeze(-1)).squeeze(-1) * task_axis).sum()
    assert normal_change == pytest.approx(float((normal*error[:, :3]).sum()), abs=1e-14)
    difference = (jac @ (new-old).unsqueeze(-1)).squeeze(-1)
    tangent_difference = difference - task_axis*(difference*task_axis).sum(-1, keepdim=True)
    torch.testing.assert_close(tangent_difference, torch.zeros_like(difference), rtol=0, atol=1e-14)


@pytest.mark.parametrize("normal", [[0., 0, 0], [float("nan"), 0, 1], [float("inf"), 0, 1]])
def test_compliant_normal_refuses_invalid_axes(normal):
    q = torch.zeros(1, 7)
    args = (torch.eye(6, 7)[None], torch.zeros(1, 6), q,
            torch.tensor([[[-1., 1.]]*7]), torch.ones_like(q), .02)
    with pytest.raises(ValueError, match="invalid_compliant_normal"):
        bounded_pose_step(*args, previous_targets=q, feedback_time_constant_s=.5,
                          compliant_normal=torch.tensor([normal]))


def test_compliant_normal_requires_feedback_and_reachable_normal():
    q = torch.zeros(1, 7)
    args = (torch.zeros(1, 6, 7), torch.zeros(1, 6), q,
            torch.tensor([[[-1., 1.]]*7]), torch.ones_like(q), .02)
    normal = torch.tensor([[1., 0, 0]])
    with pytest.raises(ValueError, match="compliant_normal_requires_pose_feedback"):
        bounded_pose_step(*args, compliant_normal=normal)
    with pytest.raises(ValueError, match="unreachable_compliant_normal"):
        bounded_pose_step(*args, previous_targets=q, feedback_time_constant_s=.5,
                          compliant_normal=normal)


def test_compliant_normal_refuses_normal_coupled_by_rank_deficiency():
    q = torch.zeros(1, 7)
    jac = torch.eye(6, 7)[None]
    jac[:, 1] = jac[:, 0]
    with pytest.raises(ValueError, match="unreachable_compliant_normal"):
        bounded_pose_step(
            jac, torch.zeros(1, 6), q, torch.tensor([[[-1., 1.]] * 7]),
            torch.ones_like(q), .02, previous_targets=q,
            feedback_time_constant_s=.5, compliant_normal=torch.tensor([[1., 0, 0]]),
        )


def test_compliant_normal_retains_physical_target_reserve_rate_and_reversal():
    q = torch.zeros(1, 7, dtype=torch.float64)
    previous = q.clone()
    previous[0, 0] = .19
    limits = torch.tensor([[[-.2, .2]]*7], dtype=q.dtype)
    speed = torch.ones_like(q)*.5
    normal = torch.tensor([[1., 0, 0]], dtype=q.dtype)
    for sign in [1., -1.]:
        target, _, request = bounded_pose_step(
            torch.eye(6, 7, dtype=q.dtype)[None], torch.ones(1, 6, dtype=q.dtype)*sign,
            q, limits, speed, .02, previous_targets=previous,
            feedback_time_constant_s=.5, compliant_normal=normal, return_request=True,
        )
        assert bool((target >= limits[..., 0]).all() and (target <= limits[..., 1]).all())
        assert bool(((target-previous).abs() <= speed*.02+1e-14).all())
        assert request[0, 0] == pytest.approx(sign, abs=1e-14)
