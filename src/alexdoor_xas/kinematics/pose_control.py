"""Seven-joint pose IK with deterministic joint centering and bounded targets."""

import math

import torch


def pose_target_limits(limits, margin=None, *, domain=None):
    """Intersect a pose-command domain, then reserve targets within physical limits."""
    if (
        limits.shape[-2:] != (7, 2)
        or not bool(torch.isfinite(limits).all())
        or not bool((limits[..., 0] < limits[..., 1]).all())
    ):
        raise ValueError("invalid_pose_target_joint_domain")
    result = limits.clone()
    if domain is not None:
        bounds = torch.as_tensor(domain, device=limits.device, dtype=limits.dtype)
        if (
            bounds.shape != (7, 2)
            or not bool(torch.isfinite(bounds).all())
            or not bool((bounds[:, 0] < bounds[:, 1]).all())
        ):
            raise ValueError("invalid_pose_target_joint_domain")
        result[..., 0] = torch.maximum(result[..., 0], bounds[:, 0])
        result[..., 1] = torch.minimum(result[..., 1], bounds[:, 1])
    if margin is not None:
        reserve = torch.as_tensor(margin, device=limits.device, dtype=limits.dtype)
        if (
            reserve.shape != (7,)
            or not bool(torch.isfinite(reserve).all())
            or bool((reserve < 0).any())
        ):
            raise ValueError("invalid_pose_target_joint_margin")
        result[..., 0] += reserve
        result[..., 1] -= reserve
    if not bool((result[..., 0] < result[..., 1]).all()):
        raise ValueError("empty_pose_target_joint_domain")
    return result


def normal_reference_transport(normal_jacobian, stiffness, delivered_offset, previous=None):
    """Transport existing elastic support when its normal joint covector changes.

    This is a geometric increment from delivered targets and measured joints,
    not external-force estimation or cancellation. Fixed geometry gives exactly
    zero for any elastic offset. A fresh mode entry has no transport history.
    """
    values = (normal_jacobian, stiffness, delivered_offset)
    if previous is not None:
        values += (previous,)
    if (
        not isinstance(normal_jacobian, torch.Tensor)
        or normal_jacobian.shape[-1:] != (7,)
        or not all(
            isinstance(value, torch.Tensor)
            and value.shape == normal_jacobian.shape
            and value.device == normal_jacobian.device
            and value.dtype == normal_jacobian.dtype
            and bool(torch.isfinite(value).all())
            for value in values
        )
        or not bool((stiffness > 0).all())
    ):
        raise ValueError("invalid_normal_transport_state")
    compliance = (normal_jacobian.square() / stiffness).sum(dim=-1, keepdim=True)
    if not bool(torch.isfinite(compliance).all() and (compliance > 0).all()):
        raise ValueError("unreachable_normal_transport")
    if previous is None:
        return torch.zeros_like(compliance)
    elastic_support = (normal_jacobian * delivered_offset).sum(dim=-1, keepdim=True) / compliance
    increment = elastic_support * (normal_jacobian * (normal_jacobian - previous) / stiffness).sum(
        dim=-1, keepdim=True
    )
    if not bool(torch.isfinite(increment).all()):
        raise ValueError("invalid_normal_transport_increment")
    return increment


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
    *,
    return_request=False,
    feedback_time_constant_s=None,
    reference_delta=None,
    compliant_normal=None,
    normal_reference_only=False,
    normal_transport_m=None,
    finite_tangent_impedance=False,
):
    """Bounded pose solve with optional measured-error integration on prior targets.

    The default is the original absolute IK request. Feedback mode increments
    the last delivered target by dt/tau times the measured pose correction;
    a constant elastic deflection can then be balanced without changing inner
    impedance. Only the bounded delivered target is retained, so saturation
    cannot accumulate a hidden integral. A causal reference increment can
    supply the part of each new goal step not already paid by feedback.
    An explicit compliant normal keeps absolute position feedback on that axis
    while retaining integral/reference tracking on the other five pose axes.
    Reference-only normal mode instead retains elastic support and pays only
    the causal normal goal increment supplied by a force admittance.
    Optional geometric transport updates that existing elastic support as the
    normal joint covector changes. Optional finite tangential impedance removes
    elastic pose memory from the other five axes' error integration. Their causal
    reference increments remain, and the explicit normal channel retains load
    support. Preview calls remain pure.
    """
    if jacobian.shape[-2:] != (6, 7) or error.shape[-1] != 6:
        raise ValueError("Purdue pose control requires a 6x7 Jacobian and six pose errors")
    if not all(bool(torch.isfinite(v).all()) for v in (jacobian, error, positions, limits)):
        raise ValueError("non-finite IK input")
    feedback = feedback_time_constant_s is not None
    if feedback:
        if (
            isinstance(feedback_time_constant_s, bool)
            or not math.isfinite(feedback_time_constant_s)
            or feedback_time_constant_s < dt
        ):
            raise ValueError("invalid_pose_feedback_time_constant")
        if previous_targets is None:
            raise ValueError("pose_feedback_requires_previous_targets")
        if previous_targets.shape != positions.shape or not bool(
            torch.isfinite(previous_targets).all()
        ):
            raise ValueError("invalid_pose_feedback_previous_targets")
    if reference_delta is not None:
        if not feedback:
            raise ValueError("pose_reference_requires_feedback")
        if reference_delta.shape != error.shape or not bool(torch.isfinite(reference_delta).all()):
            raise ValueError("invalid_pose_reference_delta")
    if type(normal_reference_only) is not bool:
        raise ValueError("invalid_normal_reference_mode")
    if type(finite_tangent_impedance) is not bool:
        raise ValueError("invalid_finite_tangent_impedance")
    if finite_tangent_impedance and not normal_reference_only:
        raise ValueError("finite_tangent_requires_normal_reference")
    if normal_reference_only and compliant_normal is None:
        raise ValueError("normal_reference_requires_axis")
    if normal_transport_m is not None:
        if not normal_reference_only:
            raise ValueError("normal_transport_requires_reference_mode")
        if (
            not isinstance(normal_transport_m, torch.Tensor)
            or normal_transport_m.shape != error.shape[:-1] + (1,)
            or normal_transport_m.device != error.device
            or normal_transport_m.dtype != error.dtype
            or not bool(torch.isfinite(normal_transport_m).all())
        ):
            raise ValueError("invalid_normal_transport_increment")
    if compliant_normal is not None:
        if not feedback:
            raise ValueError("compliant_normal_requires_pose_feedback")
        if (
            not isinstance(compliant_normal, torch.Tensor)
            or compliant_normal.shape != error.shape[:-1] + (3,)
            or compliant_normal.device != error.device
            or compliant_normal.dtype != error.dtype
            or not bool(torch.isfinite(compliant_normal).all())
        ):
            raise ValueError("invalid_compliant_normal")
        length = torch.linalg.vector_norm(compliant_normal, dim=-1, keepdim=True)
        if not bool(torch.isfinite(length).all() and (length > 0).all()):
            raise ValueError("invalid_compliant_normal")
        normal = compliant_normal / length
        normal = torch.cat((normal, torch.zeros_like(normal)), dim=-1)
    lo, hi = limits[..., 0], limits[..., 1]
    span = hi - lo
    if bool((span <= 0).any()) or dt <= 0 or damping <= 0:
        raise ValueError("invalid control limits, timestep or damping")
    eye = torch.eye(6, device=jacobian.device, dtype=jacobian.dtype)
    pose_error = error
    if finite_tangent_impedance:
        # Position feedback should target a finite elastic pose, rather than
        # integrate a blocked tangential error into increasing contact effort.
        elastic_error = (jacobian @ (previous_targets - positions).unsqueeze(-1)).squeeze(-1)
        pose_error = error - elastic_error
        pose_error = pose_error - normal * (normal * pose_error).sum(dim=-1, keepdim=True)
    primary = jacobian.transpose(-1, -2) @ torch.linalg.solve(
        jacobian @ jacobian.transpose(-1, -2) + damping**2 * eye, pose_error.unsqueeze(-1)
    )
    inverse = torch.linalg.pinv(jacobian, rtol=1e-4)
    projector = torch.eye(7, device=jacobian.device, dtype=jacobian.dtype) - inverse @ jacobian
    normal_direction = None
    if compliant_normal is not None:
        normal_jacobian = (normal.unsqueeze(-2) @ jacobian).squeeze(-2)
        normal_direction = (inverse @ normal.unsqueeze(-1)).squeeze(-1)
        response = (normal_jacobian * normal_direction).sum(dim=-1, keepdim=True)
        if not bool(torch.isfinite(response).all() and (response > 0).all()):
            raise ValueError("unreachable_compliant_normal")
        normal_direction = normal_direction / response
        # A rank-deficient solve may couple the requested normal into another
        # pose axis. Refuse it at the existing pseudoinverse rank tolerance.
        achieved = (jacobian @ normal_direction.unsqueeze(-1)).squeeze(-1)
        if bool((torch.linalg.vector_norm(achieved - normal, dim=-1) > 1e-4).any()):
            raise ValueError("unreachable_compliant_normal")

        def normal_component(value):
            return normal_direction.unsqueeze(-1) * (normal_jacobian.unsqueeze(-2) @ value)

        # Remove normal elastic-target memory. Its absolute pose error supplies
        # finite impedance, rather than another integrator into a contact surface.
        absolute_normal = normal_direction * (normal * error).sum(dim=-1, keepdim=True)
        elastic_normal = normal_direction * (normal_jacobian * (previous_targets - positions)).sum(
            dim=-1, keepdim=True
        )
        primary = primary - normal_component(primary)
    center = ((lo + hi) / 2 - positions) / span.square()
    if feedback:
        gain = dt / feedback_time_constant_s
        primary = primary * gain
        if normal_direction is not None:
            if normal_reference_only:
                # A force admittance already integrates its normal correction.
                # Pay only the causal goal increment, retaining delivered elastic
                # load support without integrating normal pose error again.
                increment = (
                    torch.zeros_like(error[..., :1])
                    if reference_delta is None
                    else (normal * reference_delta).sum(dim=-1, keepdim=True)
                )
                if normal_transport_m is not None:
                    increment = increment + normal_transport_m
                primary = primary + (normal_direction * increment).unsqueeze(-1)
            else:
                primary = primary + (absolute_normal - elastic_normal).unsqueeze(-1)
        if reference_delta is not None:
            # The current pose error already includes this next-tick reference
            # increment. Add only its unpaid fraction, avoiding a one-tick lead.
            tangent_reference = reference_delta
            if finite_tangent_impedance:
                # The normal increment is already delivered by its explicit
                # reference channel; do not leak it through a damped pose solve.
                tangent_reference = reference_delta - normal * (normal * reference_delta).sum(
                    dim=-1, keepdim=True
                )
            reference_step = (
                (1 - gain)
                * jacobian.transpose(-1, -2)
                @ torch.linalg.solve(
                    jacobian @ jacobian.transpose(-1, -2) + damping**2 * eye,
                    tangent_reference.unsqueeze(-1),
                )
            )
            if normal_direction is not None:
                reference_step = reference_step - normal_component(reference_step)
            primary = primary + reference_step
    centering = centering_gain * dt * (projector @ center.unsqueeze(-1)).squeeze(-1)
    if normal_direction is not None:
        centering = centering - normal_component(centering.unsqueeze(-1)).squeeze(-1)
    delta = primary.squeeze(-1) + centering
    # Independent velocity clipping leaks nullspace motion into the hand pose.
    # Scale the whole target step to preserve its direction within the bounds.
    reference = positions if previous_targets is None else previous_targets
    if bool(((lo > reference + velocities * dt) | (hi < reference - velocities * dt)).any()):
        raise ValueError("pose_target_domain_outside_rate_limits")
    requested = (reference if feedback else positions) + delta
    step = torch.clamp(requested, lo, hi) - reference
    scale = (step.abs() / (velocities * dt)).amax(dim=-1, keepdim=True).clamp(min=1.0)
    targets, _ = bounded_joint_step(
        reference + step / scale - positions,
        positions,
        limits,
        velocities,
        dt,
        previous_targets,
    )
    result = (targets, (requested - targets).abs())
    return (*result, requested) if return_request else result


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
