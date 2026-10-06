"""Evaluator-only rigid transport errors; no hinge inference or action admission.

The expected zone is the observed seed zone transported by evaluator truth.
Targets are saved world poses, used only to measure conditional propagation:
their local coordinates are held fixed in that expected zone. This is not an
A4 rollout score or an estimate of an absent static A3 reference.
"""

import numpy as np

from alexdoor_xas.perception.point2pose_offline import finite_pose
from alexdoor_xas.perception.point2pose_replay import pose_error


def transport_impact(actual_zone, expected_zone, target_world=None):
    """Score +X normal, zone point and optional full-pose world target.

    Missing/invalid output stays missing. A finite lost pose may be scored by
    the caller, which must retain its support state separately. Seed alignment
    removes initial calibration/selection bias; it does not validate ownership.
    """
    if not finite_pose(expected_zone):
        raise ValueError("invalid_expected_zone")
    if target_world is not None and not finite_pose(target_world):
        raise ValueError("invalid_saved_target")
    if actual_zone is None or not finite_pose(actual_zone):
        return None
    actual, expected = np.asarray(actual_zone), np.asarray(expected_zone)
    delta = actual[:3, 3] - expected[:3, 3]
    # Recorded float32 rotations can have small unit-length roundoff.
    normal = expected[:3, 0] / np.linalg.norm(expected[:3, 0])
    actual_normal = actual[:3, 0] / np.linalg.norm(actual[:3, 0])
    normal_m = float(delta @ normal)
    position_m, rotation_deg = pose_error(actual, expected)
    result = dict(
        point_error_m=position_m,
        point_signed_normal_m=normal_m,
        point_tangential_m=float(np.linalg.norm(delta - normal_m * normal)),
        normal_error_deg=float(
            np.degrees(np.arccos(np.clip(actual_normal @ normal, -1.0, 1.0)))
        ),
        rotation_error_deg=rotation_deg,
    )
    if target_world is not None:
        target = np.asarray(target_world)
        transformed = actual @ np.linalg.inv(expected) @ target
        target_delta = transformed[:3, 3] - target[:3, 3]
        target_normal_m = float(target_delta @ normal)
        error = pose_error(transformed, target)
        result.update(
            target_position_error_m=error[0],
            target_rotation_error_deg=error[1],
            target_signed_normal_m=target_normal_m,
            target_tangential_m=float(
                np.linalg.norm(target_delta - target_normal_m * normal)
            ),
            target_lever_m=float(np.linalg.norm(target[:3, 3] - expected[:3, 3])),
        )
    return result
