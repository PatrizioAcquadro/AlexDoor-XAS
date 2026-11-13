"""Predict one bounded pose target from the last completed native interval."""

import xml.etree.ElementTree as ET

import numpy as np
import torch

from alexdoor_xas.kinematics.pose_control import bounded_joint_step
from alexdoor_xas.kinematics.purdue_chain import PurdueChain


def operators(mass, basis, passive_damping, h):
    reduced = basis.T @ mass @ basis + h * (basis.T @ torch.diag(passive_damping) @ basis)
    inverse_rhs = torch.linalg.solve(reduced, basis.T.expand(*mass.shape[:-2], -1, -1))
    effort_response = h * (basis @ inverse_rhs)
    momentum_response = (basis @ inverse_rhs) @ mass
    return reduced, momentum_response, effort_response


def command_response(momentum_response, effort_response, stiffness, damping, arm_ids, h, steps=4):
    injection = torch.diag(stiffness)[:, arm_ids]
    x = torch.zeros(
        (*momentum_response.shape[:-2], len(stiffness), len(arm_ids)),
        device=momentum_response.device,
        dtype=momentum_response.dtype,
    )
    velocity = torch.zeros_like(x)
    history = []
    for _ in range(steps):
        velocity = momentum_response @ velocity + effort_response @ (
            injection - stiffness[:, None] * x - damping[:, None] * velocity
        )
        x = x + h * velocity
        history.append((x.clone(), velocity.clone()))
    return x, velocity, history


def surface_support(rotation, normal, vertices):
    return (((rotation.transpose(-1, -2) @ normal[..., None]).squeeze(-1)) @ vertices.T).amax(-1)


def surface_target_step(
    target, jacobian, normal, increment, limits, velocities, native_dt, remaining
):
    """Pure bounded normal increment, with returned actual quantized pose payment."""
    F = torch.float64
    J = jacobian.to(F)
    n = normal.to(F)
    n = n / n.norm(dim=-1, keepdim=True)
    u = target.to(F)
    bounds = limits.to(F)
    speed = velocities.to(F)
    remaining = remaining.to(F)
    if not all(
        bool(torch.isfinite(x).all()) for x in [J, n, u, bounds, speed, increment, remaining]
    ):
        raise ValueError("invalid_surface_transport_state")
    n6 = torch.cat((n, torch.zeros_like(n)), -1)
    v = (torch.linalg.pinv(J, rtol=1e-4) @ n6[..., None]).squeeze(-1)
    response = ((J @ v[..., None]).squeeze(-1) * n6).sum(-1, keepdim=True)
    if bool((response <= 0).any()):
        raise ValueError("unreachable_surface_transport_normal")
    v /= response
    if bool((((J @ v[..., None]).squeeze(-1) - n6).norm(dim=-1) > 1e-4).any()):
        raise ValueError("unreachable_surface_transport_normal")
    delta = v * increment.to(F).reshape(*u.shape[:-1], 1)
    cap = torch.minimum(speed * native_dt, remaining.clamp(min=0))
    distance = torch.where(delta >= 0, bounds[..., 1] - u, u - bounds[..., 0]).clamp(min=0)
    allowed = torch.minimum(cap, distance)
    ratio = torch.where(
        abs(delta) > 0, allowed / abs(delta).clamp(min=torch.finfo(F).tiny), torch.ones_like(delta)
    )
    scale = ratio.amin(-1, keepdim=True).clamp(min=0, max=1)
    delivered = (u + delta * scale).to(target.dtype)
    actual_delta = delivered.to(F) - u
    paid = (J @ actual_delta[..., None]).squeeze(-1)
    return delivered, paid, scale.squeeze(-1) < 1, actual_delta.abs()


def quantize_held_target(target, previous, limits, speed, control_dt):
    """Round inward at actual domain/rate faces; no new tolerance is introduced."""
    lower = torch.maximum(limits[..., 0], previous - speed * control_dt)
    upper = torch.minimum(limits[..., 1], previous + speed * control_dt)
    lo, hi = lower.float(), upper.float()
    lo = torch.where(lo.double() < lower, torch.nextafter(lo, torch.full_like(lo, torch.inf)), lo)
    hi = torch.where(hi.double() > upper, torch.nextafter(hi, torch.full_like(hi, -torch.inf)), hi)
    if bool((lo > hi).any()):
        raise ValueError("no_representable_held_target_in_original_bounds")
    return target.float().clamp(lo, hi)


def canonical_A1_target(target, positions, previous, limits, speed, control_dt):
    """Return a representable held target whose ordinary A1 action replays exactly.

    Preserve the original domain/rate interval. At its faces, move the encoded
    delta inward by actual float32 neighbors, rather than adding a tolerance.
    """
    q = positions.float()
    prior = previous.float()
    bounds = limits.float()
    velocity = speed.float()
    lower = torch.maximum(limits[..., 0], previous - speed * control_dt)
    upper = torch.minimum(limits[..., 1], previous + speed * control_dt)
    delta = target.float() - q
    delivered, _ = bounded_joint_step(delta, q, bounds, velocity, control_dt, prior)
    below = delivered.double() < lower
    above = delivered.double() > upper
    while bool((below | above).any()):
        delta = torch.where(
            below,
            torch.nextafter(delta, torch.full_like(delta, torch.inf)),
            torch.where(above, torch.nextafter(delta, torch.full_like(delta, -torch.inf)), delta),
        )
        adjusted, _ = bounded_joint_step(delta, q, bounds, velocity, control_dt, prior)
        next_below = adjusted.double() < lower
        next_above = adjusted.double() > upper
        if bool(((below & next_above) | (above & next_below)).any()):
            raise ValueError("no_representable_A1_target_in_original_interval")
        delivered = adjusted
        below, above = next_below, next_above
    replay, _ = bounded_joint_step(delivered - q, q, bounds, velocity, control_dt, prior)
    if not torch.equal(replay, delivered):
        raise ValueError("noncanonical_A1_target")
    return delivered


def advance(
    q, v, target, momentum, response, stiffness, damping, velocity_residual, position_residual, h
):
    v = (
        (momentum @ v[..., None]).squeeze(-1)
        + (response @ (stiffness * (target - q) - damping * v)[..., None]).squeeze(-1)
        + velocity_residual
    )
    return q + h * v + position_residual, v


def held_surface_target(
    q,
    v,
    full_target,
    previous_arm_target,
    normal,
    momentum,
    response,
    stiffness,
    damping,
    velocity_residual,
    position_residual,
    arm_ids,
    limits,
    speed,
    vertices,
    forward,
    h,
    control_dt,
):
    """Pure preview; caller records returned target before any physics dispatch.

    forward maps the supplied arm positions to world tool pose and Jacobian.
    Returned payment is computed from the delivered, quantized target and current
    Jacobian; only this payment may update the caller's delivered goal memory.
    """
    n = normal / normal.norm(dim=-1, keepdim=True)
    n6 = torch.cat((n, torch.zeros_like(n)), -1)
    base = full_target[..., arm_ids]
    initial_pose, jacobian = forward(q[..., arm_ids])
    support_before = surface_support(initial_pose[..., :3, :3], n, vertices)
    micro_target = full_target.clone()
    micro_q, micro_v = q.clone(), v.clone()
    held_q, held_v = q.clone(), v.clone()
    used = (base - previous_arm_target).abs()
    for step in range(4):
        if step:
            pose, J = forward(micro_q[..., arm_ids])
            support_now = surface_support(pose[..., :3, :3], n, vertices)
            next_target, _, _, length = surface_target_step(
                micro_target[..., arm_ids],
                J,
                n,
                support_before - support_now,
                limits,
                speed,
                h,
                speed * control_dt - used,
            )
            micro_target[..., arm_ids] = next_target
            used += length
            support_before = support_now
        micro_q, micro_v = advance(
            micro_q,
            micro_v,
            micro_target,
            momentum,
            response,
            stiffness,
            damping,
            velocity_residual,
            position_residual,
            h,
        )
        held_q, held_v = advance(
            held_q,
            held_v,
            full_target,
            momentum,
            response,
            stiffness,
            damping,
            velocity_residual,
            position_residual,
            h,
        )
    mp, _ = forward(micro_q[..., arm_ids])
    hp, hJ = forward(held_q[..., arm_ids])
    desired = ((mp[..., :3, 3] - hp[..., :3, 3]) * n).sum(-1)
    desired += surface_support(mp[..., :3, :3], n, vertices) - surface_support(
        hp[..., :3, :3], n, vertices
    )
    # Derivative of n.p + max_vertex n.R.vertex at the held endpoint.
    projected = ((hp[..., :3, :3].transpose(-1, -2) @ n[..., None]).squeeze(-1)) @ vertices.T
    active_vertex = vertices[projected.argmax(-1)]
    lever = (hp[..., :3, :3] @ active_vertex[..., None]).squeeze(-1)
    surface_row = (n[..., None, :] @ hJ[..., :3, :]).squeeze(-2)
    surface_row += (torch.linalg.cross(lever, n)[..., None, :] @ hJ[..., 3:, :]).squeeze(-2)
    X, _, _ = command_response(momentum, response, stiffness, damping, arm_ids, h)
    endpoint_row = (surface_row[..., None, :] @ X[..., arm_ids, :]).squeeze(-2)
    # Preserve the other five virtual reference axes. Both the ordinary normal
    # direction and the robot's kinematic nullspace remain available. Select
    # the minimum motor-norm correction for the endpoint surface coordinate,
    # rather than inverting one possibly unresponsive kinematic direction.
    other_five = jacobian - n6[..., None] * (n6[..., None, :] @ jacobian)
    projection = (
        torch.eye(len(arm_ids), device=q.device, dtype=q.dtype)
        - torch.linalg.pinv(other_five, rtol=1e-4) @ other_five
    )
    projected = (projection @ endpoint_row[..., None]).squeeze(-1)
    length = projected.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(length).all() and (length > 0).all()):
        raise ValueError("unresponsive_other5_preserving_endpoint")
    direction = projected / length
    denominator = (endpoint_row * direction).sum(-1)
    if not bool(torch.isfinite(denominator).all() and (denominator > 0).all()):
        raise ValueError("unresponsive_other5_preserving_endpoint")
    if bool(
        (
            (other_five @ direction[..., None]).squeeze(-1).norm(dim=-1)
            > 1e-4 * jacobian.norm(dim=(-1, -2))
        ).any()
    ):
        raise ValueError("unreachable_other5_preserving_endpoint")
    requested = direction * (desired / denominator)[..., None]
    # There is exactly one target change, so constrain its actual final distance
    # from the preceding target, not a fictitious sum of two dispatched changes.
    lower = torch.maximum(limits[..., 0], previous_arm_target - speed * control_dt)
    upper = torch.minimum(limits[..., 1], previous_arm_target + speed * control_dt)
    available = torch.where(requested >= 0, upper - base, base - lower).clamp(min=0)
    scale = (
        torch.where(
            requested != 0,
            available / requested.abs().clamp(min=torch.finfo(q.dtype).tiny),
            torch.ones_like(requested),
        )
        .amin(-1)
        .clamp(min=0, max=1)
    )
    delivered = quantize_held_target(
        base + requested * scale[..., None], previous_arm_target, limits, speed, control_dt
    )
    delivered = canonical_A1_target(
        delivered, q[..., arm_ids], previous_arm_target, limits, speed, control_dt
    )
    actual_delta = delivered.to(jacobian.dtype) - base.to(jacobian.dtype)
    payment = (jacobian @ actual_delta[..., None]).squeeze(-1)
    return (
        delivered,
        payment,
        dict(
            desired_surface_endpoint_increment=desired,
            endpoint_response=denominator,
            limited=scale < 1,
            predicted_micro_final_target=micro_target[..., arm_ids],
            requested_motor_correction=requested,
            other5_reference_rows=other_five,
            endpoint_surface_command_row=endpoint_row,
            minimum_motor_norm_in_other5_nullspace=True,
            constant_target_profile=delivered[..., None, :].expand(
                *delivered.shape[:-1], 4, len(arm_ids)
            ),
        ),
    )


class HeldSurfaceController:
    """Pay the predicted support change with one held 60 Hz target."""

    def __init__(self, env):
        from isaaclab.utils.math import matrix_from_quat

        self.env, self.rotation = env, matrix_from_quat
        self.chain = PurdueChain(env.robot.cfg.spawn.asset_path, env.device, dtype=torch.float64)
        geometry = env.push_geometry
        self.vertices = self.tensor(
            np.unique(
                np.concatenate(list(geometry.finger_vertices.values())) - geometry.translation,
                axis=0,
            )
        )
        names = list(env.robot.joint_names)
        mimic = {}
        for joint in ET.parse(env.robot.cfg.spawn.asset_path).getroot().findall("joint"):
            relation = joint.find("mimic")
            if relation is not None:
                mimic[joint.get("name")] = (
                    relation.get("joint"),
                    float(relation.get("multiplier", "1")),
                    float(joint.find("dynamics").get("damping", "0")),
                )
        free = [name for name in names if name not in mimic]
        self.basis = self.tensor(np.zeros((len(names), len(free))))
        self.passive = self.tensor(np.zeros(len(names)))
        for i, name in enumerate(names):
            driver, multiplier, damping = mimic.get(name, (name, 1, 0))
            self.basis[i, free.index(driver)] = multiplier
            self.passive[i] = damping
        self.stiffness = torch.zeros_like(self.passive)
        self.damping = torch.zeros_like(self.passive)
        for actuator in env.robot.actuators.values():
            indices = [names.index(name) for name in actuator.joint_names]
            self.stiffness[indices] = self.tensor(actuator.stiffness)[0]
            self.damping[indices] = self.tensor(actuator.damping)[0]
        if env.cfg.decimation != 4 or self.basis.shape != (20, 18):
            raise ValueError("held_surface_requires_four_substeps_and_purdue_mimic")
        self.previous = None
        self.cache = None

    def tensor(self, value):
        if hasattr(value, "torch"):
            value = value.torch
        return torch.as_tensor(value, device=self.env.device, dtype=torch.float64)

    def mass(self):
        data = self.env.robot.data
        mass = self.tensor(data.mass_matrix)
        # The Newton public matrix already contains rotor armature.
        if self.env.steel_joint_load is None:
            mass = mass + torch.diag_embed(self.tensor(data.joint_armature))
        return mass

    def before_last_substep(self):
        env = self.env
        if env.motion_stopped or env._physics_actions != env.cfg.decimation:
            return
        data = env.robot.data
        self.previous = (
            self.tensor(data.joint_pos),
            self.tensor(data.joint_vel),
            self.tensor(env.targets),
            self.mass(),
            env._time_s + env.step_dt,
        )
        self.cache = None

    def preview(self, base, goal):
        env, h = self.env, self.env.cfg.sim.dt
        if self.previous is None or abs(self.previous[-1] - env._time_s) > 1e-9:
            raise ValueError("completed_shape_response_unavailable")
        if self.cache is not None:
            old_base, position, orientation, result = self.cache
            if (
                torch.equal(base, old_base)
                and torch.equal(goal[0], position)
                and torch.equal(goal[1], orientation)
            ):
                return result
        q0, v0, target0, mass0, _ = self.previous
        data = env.robot.data
        q, v = self.tensor(data.joint_pos), self.tensor(data.joint_vel)
        _, momentum, effort = operators(self.mass(), self.basis, self.passive, h)
        _, momentum0, effort0 = operators(mass0, self.basis, self.passive, h)
        pd0 = self.stiffness * (target0 - q0) - self.damping * v0
        residual_v = v - (momentum0 @ v0[..., None] + effort0 @ pd0[..., None]).squeeze(-1)
        residual_q = q - q0 - h * v
        root = torch.eye(4, dtype=q.dtype, device=q.device)[None]
        root[:, :3, :3] = self.rotation(self.tensor(data.root_link_quat_w))
        root[:, :3, 3] = self.tensor(data.root_link_pos_w)

        def fk(arm):
            pose, jacobian = self.chain.forward(arm)
            rotation = root[:, :3, :3]
            return root @ pose, torch.cat(
                (rotation @ jacobian[:, :3], rotation @ jacobian[:, 3:]), dim=-2
            )

        target = env.targets.clone()
        target[:, env.arm_ids] = base
        normal = self.rotation(self.tensor(goal[1]))[:, :, 0]
        delivered, payment, _ = held_surface_target(
            q,
            v,
            self.tensor(target),
            self.tensor(env.targets[:, env.arm_ids]),
            normal,
            momentum,
            effort,
            self.stiffness,
            self.damping,
            residual_v,
            residual_q,
            env.arm_ids,
            self.tensor(env.pose_limits),
            self.tensor(env.velocities),
            self.vertices,
            fk,
            h,
            env.step_dt,
        )
        result = delivered, (payment[:, :3] * normal).sum(dim=-1, keepdim=True) * normal
        self.cache = base.clone(), goal[0].clone(), goal[1].clone(), result
        return result
