"""Coupled seven-joint posture preference with measured margins and bounded targets."""

import math

import torch


def joint_command_position_response(
    mass, basis, passive_damping, stiffness, damping, arm_ids, physics_dt, substeps
):
    """Fixed-geometry position derivative, including armature and mimic constraints.

    Callers include motor armature in mass. Explicit actuator damping and native
    passive damping are distinct. This approximate response guides preference;
    it does not certify a future state or use external-force truth.
    """
    reduced = basis.T @ mass @ basis + physics_dt * (basis.T @ torch.diag(passive_damping) @ basis)
    inverse_rhs = torch.linalg.solve(reduced, basis.T.expand(*mass.shape[:-2], -1, -1))
    momentum_response = (basis @ inverse_rhs) @ mass
    effort_response = physics_dt * (basis @ inverse_rhs)
    injection = torch.diag(stiffness)[:, arm_ids]
    position = torch.zeros(
        (*mass.shape[:-2], len(stiffness), len(arm_ids)), device=mass.device, dtype=mass.dtype
    )
    velocity = torch.zeros_like(position)
    for _ in range(substeps):
        velocity = momentum_response @ velocity + effort_response @ (
            injection - stiffness[:, None] * position - damping[:, None] * velocity
        )
        position = position + physics_dt * velocity
    return position[..., arm_ids, :]


def _maximize_worst_margin(intercept, slope, lower, upper):
    """Maximize a concave piecewise-affine score on one bounded coordinate."""
    pairs = torch.triu_indices(
        intercept.shape[-1], intercept.shape[-1], offset=1, device=intercept.device
    )
    denominator = slope[..., pairs[0]] - slope[..., pairs[1]]
    nonzero = denominator != 0
    crossings = (intercept[..., pairs[1]] - intercept[..., pairs[0]]) / torch.where(
        nonzero, denominator, torch.ones_like(denominator)
    )
    valid = nonzero & (crossings >= lower[..., None]) & (crossings <= upper[..., None])
    crossings = torch.where(valid, crossings, torch.zeros_like(crossings))
    candidates = torch.cat(
        (lower[..., None], upper[..., None], torch.zeros_like(lower[..., None]), crossings), -1
    )
    scores = (intercept[..., :, None] + slope[..., :, None] * candidates[..., None, :]).amin(-2)
    maximum = scores.amax(-1, keepdim=True)
    distance = torch.where(
        scores == maximum, abs(candidates), torch.full_like(candidates, torch.inf)
    )
    chosen = distance.argmin(-1, keepdim=True)
    return candidates.gather(-1, chosen).squeeze(-1), maximum.squeeze(-1)


def coupled_margin_preference(
    jacobian,
    positions,
    measured_velocity,
    primary_target,
    previous_target,
    limits,
    velocities,
    dt,
    support,
    stopping_reserve,
    position_response,
    gain,
):
    """Choose a soft all-face preference inside the primary command's residual bounds.

    The target correction stays in the kinematic Jacobian nullspace, as in the
    existing controller. Actual four-substep response is used for its margin
    score; target-null and measured-motion-null are not equated. Current measured
    motion supplies the soft forecast. No fixture force or future sensor input.
    The preference uses the measured velocity allowance once. Penalizing the
    proposed endpoint speed again can make inward motion no better than rest.
    Stopping reserve excludes velocity, which is paid once in the score below.
    Approximate score infeasibility never becomes a new admission refusal.
    """
    if not math.isfinite(gain) or gain < 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid_posture_preference_gain_or_dt")
    if gain == 0:
        return primary_target.clone(), dict(disabled=True)
    dtype = primary_target.dtype
    # Preserve the already bounded primary, including its existing float32 rate
    # representation. A posture correction cannot extend that accepted extent.
    hard_lower = torch.maximum(
        limits[..., 0], torch.minimum(previous_target - velocities * dt, primary_target)
    )
    hard_upper = torch.minimum(
        limits[..., 1], torch.maximum(previous_target + velocities * dt, primary_target)
    )
    values = [
        jacobian,
        positions,
        measured_velocity,
        primary_target,
        previous_target,
        support,
        stopping_reserve,
        position_response,
    ]
    if not all(bool(torch.isfinite(v).all()) for v in (*values, limits, velocities)):
        raise ValueError("invalid_posture_preference_input")
    if bool(((primary_target < hard_lower) | (primary_target > hard_upper)).any()):
        raise ValueError("primary_target_outside_original_bounds")
    J, q, v, primary, previous, bounds, reserve, X = [v.to(torch.float64) for v in values]
    hard_lower, hard_upper = hard_lower.to(torch.float64), hard_upper.to(torch.float64)
    widths = bounds[..., 1] - bounds[..., 0]
    if bool((widths <= 0).any()) or bool((reserve < 0).any()):
        raise ValueError("invalid_posture_support")
    _, _, vh = torch.linalg.svd(J, full_matrices=True)
    direction = vh[..., -1, :]
    moving_position = (X @ direction.unsqueeze(-1)).squeeze(-1)
    delta_primary = primary - previous
    q0 = q + (X @ delta_primary.unsqueeze(-1)).squeeze(-1)
    allowance = reserve + dt * abs(v)
    # This soft score starts at the same measured box as original admission.
    # The primary command's differential response is included, without treating
    # a model position or counterfactual velocity as a new safety certificate.
    intercept = torch.cat(
        ((q0 - bounds[..., 0] - allowance) / widths, (bounds[..., 1] - q0 - allowance) / widths), -1
    )
    slope = torch.cat((moving_position / widths, -moving_position / widths), -1)
    nonzero = direction != 0
    safe = torch.where(nonzero, direction, torch.ones_like(direction))
    a, b = (hard_lower - primary) / safe, (hard_upper - primary) / safe
    lower = torch.where(nonzero, torch.minimum(a, b), -torch.inf).amax(-1)
    upper = torch.where(nonzero, torch.maximum(a, b), torch.inf).amin(-1)
    coordinate, optimum = _maximize_worst_margin(intercept, slope, lower, upper)
    coordinate = coordinate * min(1.0, gain * dt)
    target = (primary + direction * coordinate[..., None]).to(dtype)
    target = torch.maximum(torch.minimum(target, hard_upper.to(dtype)), hard_lower.to(dtype))
    # Quantization may turn a mathematically flat preference into a worse one.
    # Falling back to the already bounded primary is preference-only, no refusal.
    correction = target.to(torch.float64) - primary
    q1 = q0 + (X @ correction.unsqueeze(-1)).squeeze(-1)
    before = intercept.amin(-1)
    after = torch.minimum(
        (q1 - bounds[..., 0] - allowance) / widths, (bounds[..., 1] - q1 - allowance) / widths
    ).amin(-1)
    improving = after >= before
    target = torch.where(improving[..., None], target, primary_target)
    after = torch.where(improving, after, before)
    return target, dict(
        score_before=before,
        score_after=after,
        optimum=optimum,
        unrelaxed_coordinate=coordinate / min(1.0, gain * dt),
        kinematic_null_direction=direction,
        physical_response_direction=moving_position,
        predicted_score_is_not_an_admission_gate=True,
    )
