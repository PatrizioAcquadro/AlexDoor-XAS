"""Informative vertical revolute fits in observed world poses; no closed-pose claim."""

import numpy as np
from scipy.spatial.transform import Rotation


def fit_motion_axis(initial, samples, floor_z):
    """Diagnostic axis/relative angle, only when motion exceeds its error bounds."""
    motions = []
    for pose, position_error, rotation_error in samples:
        delta = pose @ np.linalg.inv(initial)
        vector = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
        angle = np.linalg.norm(vector)
        if angle <= max(np.deg2rad(5), 2 * rotation_error):
            continue
        axis = vector / angle * np.sign(vector[2])
        if abs(axis[2]) < np.cos(np.deg2rad(5)):
            return None
        motions.append((delta, position_error, rotation_error))
    if len(motions) < 3:
        return None
    a = np.concatenate([(np.eye(3) - d[:3, :3])[:, :2] for d, _, _ in motions])
    b = np.concatenate([d[:3, 3] - (np.eye(3) - d[:3, :3])[:, 2] * floor_z for d, _, _ in motions])
    singular = np.linalg.svd(a, compute_uv=False)
    if singular[-1] <= 0 or singular[0] / singular[-1] > 10:
        return None
    xy, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    if rank != 2:
        return None
    origin = np.r_[xy, floor_z]
    residual = float(np.linalg.norm((a @ xy - b).reshape(-1, 3), axis=1).max())
    lever = float(np.linalg.norm(origin - initial[:3, 3]))
    envelope = max(p + 2 * lever * np.sin(r / 2) for _, p, r in motions)
    if residual > envelope:
        return None
    sensitivity = float(np.linalg.norm(np.abs(np.linalg.pinv(a)).sum(1)))
    latest = samples[-1][0] @ np.linalg.inv(initial)
    vector = Rotation.from_matrix(latest[:3, :3]).as_rotvec()
    return dict(
        origin=origin,
        direction=np.array([0.0, 0.0, 1.0]),
        relative_angle_rad=float(np.linalg.norm(vector) * np.sign(vector[2])),
        position_bound_m=sensitivity * envelope,
        residual_m=residual,
        condition=float(singular[0] / singular[-1]),
        closed_reference_validated=False,
    )
