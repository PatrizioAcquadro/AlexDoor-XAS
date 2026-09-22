"""Evidence for a local, controller-qualified joint limit; never global IK proof."""

import numpy as np
from scipy.optimize import lsq_linear


def constrained_step_evidence(jacobian, positions, limits, desired_delta):
    """Test the next small task-space step against actual remaining joint travel."""
    scale = np.array([0.003] * 3 + [np.deg2rad(1)] * 3)
    low = np.minimum(limits[:, 0] - positions, 0.0)
    high = np.maximum(limits[:, 1] - positions, 0.0)
    active = np.flatnonzero(np.minimum(positions - limits[:, 0], limits[:, 1] - positions) < 0.005)
    result = lsq_linear(
        jacobian / scale[:, None], desired_delta / scale, bounds=(low - 1e-9, high + 1e-9), tol=1e-8
    )
    residual = jacobian @ result.x - desired_delta
    relative_residual = float(
        np.linalg.norm(residual / scale) / max(np.linalg.norm(desired_delta / scale), 1e-9)
    )
    return dict(
        blocked=bool(result.success and len(active) and relative_residual > 0.1),
        active_joint_indices=active.tolist(),
        relative_residual=relative_residual,
        residual=residual.tolist(),
        weighted_residual=float(np.linalg.norm(residual / scale)),
        scope="local_linearized_joint_limit_not_global_reachability",
    )
