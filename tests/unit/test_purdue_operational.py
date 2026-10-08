"""Operational math and observation boundaries, without claiming physics evidence."""

import numpy as np
import pytest
import torch

from alexdoor_xas.envs.door_task.purdue_contacts import has_forbidden_contact
from alexdoor_xas.kinematics.pose_control import bounded_joint_step, bounded_pose_step
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
