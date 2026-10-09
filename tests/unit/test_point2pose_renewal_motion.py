"""Essential independent-motion, ambiguity, spatial support and elapsed-time behavior."""

import numpy as np
import pytest

from alexdoor_xas.perception.point2pose.renewal_motion import (
    background_camera_motion,
    motion_evidence,
    observed_rigid_motion,
    spatial_support,
    transform,
)


def test_camera_estimate_uses_measured_background_rgbd():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 255, (192, 192, 3), dtype=np.uint8)
    rgb = cv2.GaussianBlur(rgb, (5, 5), 1)
    shifted = cv2.warpAffine(rgb, np.array([[1, 0, 10], [0, 1, -6]], float), (192, 192))
    depth, mask = np.ones((192, 192)), np.zeros((192, 192), bool)
    k = np.array([[500, 0, 96], [0, 500, 96], [0, 0, 1]], float)
    pose, stats = background_camera_motion((rgb, depth, mask, k), (shifted, depth, mask, k))
    assert stats["reason"] == "supported"
    np.testing.assert_allclose(pose[:3, 3], [0.02, -0.012, 0], atol=0.002)


def test_camera_motion_is_compensated_for_both_hypotheses():
    camera, body = np.eye(4), np.eye(4)
    camera[0, 3], body[0, 3] = 0.03, 0.05
    point = np.array([0.2, 0.3, 1.0])
    args = dict(elapsed_s=0.2, deadline_s=0.25)
    assert (
        motion_evidence(point, transform(body, point), body, camera, **args)["decision"] == "body"
    )
    assert (
        motion_evidence(point, transform(camera, point), body, camera, **args)["decision"]
        == "static"
    )
    assert (
        motion_evidence(point, transform(camera, point), camera, camera, **args)["decision"]
        == "insufficient"
    )


def test_stationary_small_motion_unreliable_pose_and_elapsed_budget_do_not_pass():
    point, body = np.array([0.2, 0.3, 1.0]), np.eye(4)
    body[0, 3] = 0.003
    assert (
        motion_evidence(point, point, body, np.eye(4), elapsed_s=0.2, deadline_s=0.25)["reason"]
        == "indistinguishable_motion"
    )
    assert (
        motion_evidence(point, point, None, np.eye(4), elapsed_s=0.2, deadline_s=0.25)["reason"]
        == "unreliable_motion"
    )
    # A sparse second capture cannot turn a three-second wait into two valid frames.
    record = motion_evidence(point, point, body, np.eye(4), elapsed_s=3, deadline_s=0.25)
    assert record["expired"] and record["decision"] == "insufficient"


def test_independent_body_fit_rejects_outliers_and_narrow_support():
    source = np.array([[x, y, 1.0] for x in (-0.2, 0, 0.2) for y in (-0.2, 0, 0.2)])
    current = source + [0.02, 0, 0]
    current[-1] += [0.1, 0.2, 0]
    pose, stats = observed_rigid_motion(source, current)
    assert stats["count"] == 8
    np.testing.assert_allclose(pose[:3, 3], [0.02, 0, 0], atol=1e-12)
    # The candidate is an input only to the subsequent hypothesis comparison.
    candidate = np.array([8.0, 9.0, 1.0])
    motion_evidence(candidate, candidate, pose, np.eye(4), elapsed_s=0.2, deadline_s=0.25)
    np.testing.assert_allclose(pose[:3, 3], [0.02, 0, 0], atol=1e-12)
    source[:, 1] = 0
    assert observed_rigid_motion(source, source)[0] is None
    assert spatial_support([[np.nan, 0, 1]]) == dict(count=0, second_rms_m=0.0)
