"""Seven-joint pose IK with deterministic joint centering and bounded targets."""

import torch


def bounded_pose_step(
    jacobian,
    error,
    positions,
    limits,
    velocities,
    dt,
    damping=0.05,
    centering_gain=0.1,
    previous_targets=None,
):
    """Damped primary solve plus exact-SVD nullspace centering, in world axes."""
    if jacobian.shape[-2:] != (6, 7) or error.shape[-1] != 6:
        raise ValueError("Purdue pose control requires a 6x7 Jacobian and six pose errors")
    if not all(bool(torch.isfinite(v).all()) for v in (jacobian, error, positions, limits)):
        raise ValueError("non-finite IK input")
    lo, hi = limits[..., 0], limits[..., 1]
    span = hi - lo
    if bool((span <= 0).any()) or dt <= 0 or damping <= 0:
        raise ValueError("invalid control limits, timestep or damping")
    eye = torch.eye(6, device=jacobian.device, dtype=jacobian.dtype)
    primary = jacobian.transpose(-1, -2) @ torch.linalg.solve(
        jacobian @ jacobian.transpose(-1, -2) + damping**2 * eye, error.unsqueeze(-1)
    )
    projector = torch.eye(7, device=jacobian.device, dtype=jacobian.dtype) - (
        torch.linalg.pinv(jacobian, rtol=1e-4) @ jacobian
    )
    center = ((lo + hi) / 2 - positions) / span.square()
    delta = primary.squeeze(-1) + centering_gain * dt * (projector @ center.unsqueeze(-1)).squeeze(
        -1
    )
    return bounded_joint_step(delta, positions, limits, velocities, dt, previous_targets)


def bounded_joint_step(delta, positions, limits, velocities, dt, previous_targets=None):
    """Bound target displacement by velocity and physical position limits."""
    if delta.shape != positions.shape or delta.shape[-1] != 7:
        raise ValueError("A1 requires seven deltas in canonical right-arm order")
    if not all(bool(torch.isfinite(v).all()) for v in (delta, positions, limits, velocities)):
        raise ValueError("non-finite joint command")
    if dt <= 0 or bool((velocities <= 0).any()):
        raise ValueError("invalid joint velocity limits or timestep")
    reference = positions if previous_targets is None else previous_targets
    raw = positions + delta
    bounded = torch.clamp(raw, reference - velocities * dt, reference + velocities * dt)
    targets = torch.clamp(bounded, limits[..., 0], limits[..., 1])
    return targets, (positions + delta - targets).abs()
