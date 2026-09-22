"""Operational math and observation boundaries, without claiming physics evidence."""

import numpy as np
import pytest
import torch

from alexdoor_xas.assets.purdue import ARM_JOINTS, derive_push_geometry
from alexdoor_xas.envs.door_task.purdue_contacts import DistalSurface
from alexdoor_xas.kinematics.pose_control import bounded_joint_step, bounded_pose_step
from alexdoor_xas.recording.rgbd import RGBDCapture


def test_canonical_geometry():
    from ihmc_alex_isaaclab._paths import REPOSITORY_ROOT

    path = (
        REPOSITORY_ROOT
        / "assets/robots/alex_purdue/urdf/baseline/alex_purdue_wsg32_umi_v1_full_convex.urdf"
    )
    geometry = derive_push_geometry(path)
    np.testing.assert_allclose(geometry.translation, [0.202, 0, 0], atol=1e-5)
    assert len(ARM_JOINTS) == 7 and ARM_JOINTS[-1] == "RIGHT_GRIPPER_Y"
    for name, points in geometry.finger_vertices.items():
        surface = DistalSurface("actor", name, points, np.array([1.0, 0, 0]))
        face = points[np.abs(points[:, 0] - points[:, 0].max()) < 1e-6]
        point = (face.min(0) + face.max(0)) / 2
        assert surface.contains(point, np.array([1.0, 0, 0]))
        assert not surface.contains(point - [0.02, 0, 0], np.array([1.0, 0, 0]))
        assert not surface.contains(point, np.array([0.0, 1, 0]))


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
