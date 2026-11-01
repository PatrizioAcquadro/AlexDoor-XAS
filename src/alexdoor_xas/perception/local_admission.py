"""A1/A2 admission from observed local material and continuous swept-space evidence."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from alexdoor_xas.action.b1 import ACTION_DIMS, checked_pose, finite_vector
from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA
from alexdoor_xas.perception.admission import ActionAdmission, nonnegative, rotational_travel
from alexdoor_xas.perception.contracts import (
    TIME_ROUNDOFF_S,
    FieldSupport,
    same_pose,
    validate_local_contact,
    validate_local_transition,
)


@dataclass(frozen=True)
class LocalActionSupport:
    """Producer-owned minima over the actual sweep, two fingers and stop envelope.

    A hinge is unnecessary, but local identity, bounded pose, collision-free swept
    space and bounded material response remain necessary. None is unmeasured.
    """

    support: FieldSupport
    world_poses: tuple[ObjectFrame, ...]
    finger_clearance_m: tuple[float | None, float | None]
    collision_clearance_m: float | None
    unknown_in_envelope: bool
    # Complete material motion over the proposed interval, including its
    # response to actuation. Robot tracking response is a separate source.
    response_position_bound_m: float | None
    response_rotation_bound_rad: float | None
    footprint_radius_m: float | None
    unloaded_separation_m: float | None = None
    # Optional independently eroded whole-robot sweep/stop certificate. Its
    # producer already includes its own kinematic, scene and stopping errors.
    # Local material uncertainty still spends the finite finger margin below.
    swept_collision_margin_m: float | None = None
    # Independently eroded separation of the WHOLE robot/stop volume from
    # current material. With the free-space proof, no contact footprint is used.
    unloaded_separation_margin_m: float | None = None
    unloaded_normal_lever_m: float | None = None
    # Position error already eroded from BOTH finite material clearances.
    # Credit is valid only with an independent whole-robot sweep certificate.
    geometry_position_prepaid_m: float = 0.0
    # Measured possible material depth relative to the reference plane. This
    # affects indirect contact intervals, never the independent physical score.
    material_normal_range_m: tuple[float, float] | None = None
    # Full-interval position/angular growth ALREADY eroded by the producer,
    # including relative material motion. Only excess growth spends the residual.
    unloaded_temporal_bound_m_rad: tuple[float, float] | None = None
    # World-space enclosures for normal-error growth at each SAME proved volume.
    # A convex scene proof does not assert free space for its enclosing ball.
    unloaded_normal_spheres: tuple[tuple[float, float, float, float], ...] = ()
    # Original producer-eroded separation at EACH corresponding proved volume.
    # Empty preserves the conservative global-minimum rule for older receipts.
    unloaded_normal_margins_m: tuple[float, ...] = ()


def _local_error_terms(patch, evidence, limits, now, duration):
    """Same finite footprint, material-response and stopping bound for a receipt."""
    radius = nonnegative(evidence.footprint_radius_m, "unmeasured_footprint_radius")
    response = nonnegative(evidence.response_position_bound_m, "unmeasured_material_response")
    response_rotation = nonnegative(
        evidence.response_rotation_bound_rad, "unmeasured_material_response"
    )
    age = now - min(
        patch.pose_support.supported_s,
        patch.identity_support.supported_s,
        evidence.support.supported_s,
    )
    paid = patch.pose_temporal_position_m
    if paid is not None:
        paid = nonnegative(paid, "invalid_prepaid_pose_growth")
        if paid > patch.pose_support.position_bound_m:
            raise ValueError("invalid_prepaid_pose_growth")
    # The tracker already grows its current pose bound from the last capture.
    # Future material motion belongs to the independently supplied interval
    # response bound. Charging rate * duration here would spend it twice.
    temporal = limits.relative_speed_bound_m_s * age - min(
        limits.relative_speed_bound_m_s * age, paid or 0.0
    )
    prepaid = nonnegative(evidence.geometry_position_prepaid_m, "invalid_prepaid_geometry_error")
    if prepaid > patch.geometry_support.position_bound_m or (
        prepaid
        and (evidence.swept_collision_margin_m is None or evidence.swept_collision_margin_m <= 0)
    ):
        raise ValueError("invalid_prepaid_geometry_error")
    # Frame uncertainty rotates the entire offset footprint about the material
    # anchor, while robot tracking rotation acts about the tool origin.
    material_lever = radius + max(
        np.linalg.norm(pose.origin - patch.world_pose.origin) for pose in evidence.world_poses
    )
    pose_position = patch.pose_support.position_bound_m
    pose_rotation = rotational_travel(material_lever, patch.pose_support.rotation_bound_rad)
    geometry_lever = material_lever
    sources = getattr(patch, "pose_uncertainty_sources", ())
    if sources:
        pose_position = pose_rotation = 0.0
        names = set()
        frame_position = frame_rotation = 0.0
        for name, position, angle, frame_lever, origin in sources:
            origin = np.asarray(origin, dtype=float)
            if name in names or origin.shape != (3,) or not np.isfinite(origin).all():
                raise ValueError("invalid_pose_error_source")
            names.add(name)
            lever = radius + max(
                np.linalg.norm(pose.origin - origin) for pose in evidence.world_poses
            )
            position = nonnegative(position, "invalid_pose_error_source")
            angle = nonnegative(angle, "invalid_pose_error_source")
            frame_lever = nonnegative(frame_lever, "invalid_pose_error_source")
            frame_position += position + rotational_travel(frame_lever, angle)
            frame_rotation += angle
            pose_position += position
            pose_rotation += rotational_travel(lever, angle)
            if name != "calibration_fk":
                geometry_lever = max(geometry_lever, lever)
        # This is a floating-point consistency check on the SAME primitives,
        # not a physical acceptance tolerance or permission to omit a source.
        roundoff = 32 * np.finfo(float).eps
        if not np.allclose(
            [frame_position, frame_rotation],
            [patch.pose_support.position_bound_m, patch.pose_support.rotation_bound_rad],
            rtol=roundoff,
            atol=roundoff,
        ):
            raise ValueError("incomplete_pose_error_sources")
    return dict(
        geometry_position_m=patch.geometry_support.position_bound_m - prepaid,
        pose_position_m=pose_position,
        sweep_position_m=evidence.support.position_bound_m,
        robot_position_m=limits.robot_position_bound_m,
        geometry_rotation_m=rotational_travel(
            geometry_lever, patch.geometry_support.rotation_bound_rad
        ),
        pose_rotation_m=pose_rotation,
        sweep_rotation_m=rotational_travel(radius, evidence.support.rotation_bound_rad),
        robot_rotation_m=rotational_travel(radius, limits.robot_rotation_bound_rad),
        response_rotation_m=rotational_travel(material_lever, response_rotation),
        response_m=response,
        temporal_m=temporal,
        stop_m=limits.stop_travel_m,
    )


def _local_error(patch, evidence, limits, now, duration):
    return sum(_local_error_terms(patch, evidence, limits, now, duration).values())


def admit_local_action(estimate, proposal, evidence, limits, now, *, generation):
    decision = {}
    try:
        state = estimate.local
        patch = validate_local_contact(state, now, generation=generation)
        if patch.tracking_state != "tracking" or patch.ownership not in (
            "observed_rigid_material",
            "observed_moving_rigid_candidate",
        ):
            raise ValueError("unresolved_local_material_identity")
        if len(patch.finger_clearance_m) != 2:
            raise ValueError("missing_two_finger_support")
        for support in (patch.geometry_support, patch.pose_support):
            # A1/A2 consume these bounds in the actual footprint/sweep margin
            # below. Global A3/A4 frame qualification is a separate requirement.
            support.require_bounds(position=True, rotation=True)
        if proposal.action_space not in (A1_JOINT_DELTA, A2_EE_DELTA):
            raise ValueError("local_admission_requires_robot_world_action")
        if proposal.source != state.selection.source or proposal.source not in (
            "policy",
            "diagnostic",
        ):
            raise ValueError("local_action_source_mismatch")
        if (
            proposal.selection_id != state.selection.selection_id
            or type(proposal.loaded) is not bool
        ):
            raise ValueError("local_selection_or_load_mismatch")
        if proposal.required_dimensions:
            raise ValueError("local_action_requires_unavailable_dimensions")
        finite_vector(proposal.action, ACTION_DIMS[proposal.action_space])
        times = np.asarray(proposal.times_s)
        if (
            times.shape != (2,)
            or not np.isfinite(times).all()
            or times[0] != now
            or times[1] <= times[0]
            or len(proposal.world_poses) != 2
        ):
            raise ValueError("invalid_local_action_schedule")
        for pose in proposal.world_poses:
            checked_pose(pose)
        if len(evidence.world_poses) != 2 or any(
            not same_pose(a, b)
            for a, b in zip(evidence.world_poses, proposal.world_poses, strict=True)
        ):
            raise ValueError("different_world_trajectory")
        bounds = {k: nonnegative(v, f"unmeasured_{k}") for k, v in vars(limits).items()}
        duration = times[1] - times[0]
        a, b = proposal.world_poses
        travel = np.linalg.norm(b.origin - a.origin)
        rotation = Rotation.from_matrix(b.rot @ a.rot.T).magnitude()
        if (
            duration > bounds["duration_s"] + TIME_ROUNDOFF_S
            or travel > bounds["displacement_m"]
            or travel / duration > bounds["speed_m_s"]
            or rotation / duration > bounds["angular_speed_rad_s"]
        ):
            raise ValueError("action_exceeds_declared_limits")
        evidence.support.require(now, generation, dynamic=True)
        evidence.support.require_bounds(position=True, rotation=True)
        if evidence.unknown_in_envelope is not False:
            raise ValueError("unknown_relevant_swept_space")
        if len(evidence.finger_clearance_m) != 2:
            raise ValueError("missing_two_finger_support")
        if evidence.material_normal_range_m is not None:
            interval = np.asarray(evidence.material_normal_range_m)
            if (
                interval.shape != (2,)
                or not np.isfinite(interval).all()
                or interval[0] > interval[1]
            ):
                raise ValueError("invalid_material_normal_range")
        sweep_margin = evidence.swept_collision_margin_m
        free_margin = evidence.unloaded_separation_margin_m
        if free_margin is not None:
            if proposal.loaded or sweep_margin is None:
                raise ValueError("free_motion_requires_unloaded_full_robot_proof")
            clearance = nonnegative(free_margin, "invalid_unloaded_separation_margin")
            nonnegative(evidence.unloaded_normal_lever_m, "missing_unloaded_normal_lever")
            nonnegative(patch.normal_position_bound_m, "missing_unloaded_normal_bound")
            nonnegative(patch.normal_rotation_bound_rad, "missing_unloaded_normal_bound")
            if evidence.unloaded_normal_spheres:
                spheres = np.asarray(evidence.unloaded_normal_spheres)
                if (
                    spheres.ndim != 2
                    or spheres.shape[1] != 4
                    or not np.isfinite(spheres).all()
                    or np.any(spheres[:, 3] < 0)
                ):
                    raise ValueError("invalid_unloaded_normal_spheres")
                if not patch.normal_uncertainty_sources:
                    raise ValueError("missing_unloaded_normal_error_sources")
            if evidence.unloaded_normal_margins_m:
                margins = np.asarray(evidence.unloaded_normal_margins_m)
                if (
                    margins.shape != (len(evidence.unloaded_normal_spheres),)
                    or not np.isfinite(margins).all()
                    or (margins <= 0).any()
                    or margins.min() != clearance
                ):
                    raise ValueError("invalid_unloaded_normal_margins")
            if evidence.unloaded_temporal_bound_m_rad is not None:
                temporal = np.asarray(evidence.unloaded_temporal_bound_m_rad)
                if (
                    temporal.shape != (2,)
                    or not np.isfinite(temporal).all()
                    or (temporal < 0).any()
                ):
                    raise ValueError("invalid_unloaded_temporal_bound")
            if clearance <= 0 or nonnegative(sweep_margin, "invalid_robot_sweep_margin") <= 0:
                raise ValueError("insufficient_unloaded_full_robot_margin")
            error = 0.0
            decision = dict(
                margin_kind="unloaded_whole_robot_separation",
                separation_margin_m=clearance,
                independent_robot_sweep_margin_m=sweep_margin,
            )
        else:
            values = evidence.finger_clearance_m
            if sweep_margin is None:
                values = (*values, evidence.collision_clearance_m)
            elif nonnegative(sweep_margin, "invalid_robot_sweep_margin") <= 0:
                raise ValueError("insufficient_robot_sweep_margin")
            clearance = min(nonnegative(v, "missing_relevant_clearance") for v in values)
            terms = _local_error_terms(patch, evidence, limits, now, duration)
            error = sum(terms.values())
            decision = dict(
                material_id=patch.patch_id,
                finger_clearance_m=evidence.finger_clearance_m,
                applicable_clearance_m=clearance,
                uncertainty_terms_m=terms,
                total_uncertainty_m=error,
                material_margin_m=clearance - error,
                numerical_tolerance_m=0.0,
                geometry_position_prepaid_m=evidence.geometry_position_prepaid_m,
                frame_origin_position_bound_m=patch.pose_support.position_bound_m,
                frame_origin_rotation_bound_rad=patch.pose_support.rotation_bound_rad,
                pose_uncertainty_sources=getattr(patch, "pose_uncertainty_sources", ()),
                material_rotation_lever_m=evidence.footprint_radius_m
                + max(
                    np.linalg.norm(pose.origin - patch.world_pose.origin)
                    for pose in evidence.world_poses
                ),
            )
            if clearance <= error:
                raise ValueError("insufficient_local_action_margin")
            if (
                not proposal.loaded
                and nonnegative(evidence.unloaded_separation_m, "unmeasured_unloaded_separation")
                <= error
            ):
                raise ValueError("unloaded_contact_not_excluded")
    except (ValueError, TypeError, AttributeError, IndexError) as error:
        return ActionAdmission(False, False, str(error), decision=decision)
    return ActionAdmission(
        True,
        False,
        "admitted_local_geometry_only",
        ((patch.patch_id, clearance - error),)
        + (() if sweep_margin is None else (("independent_robot_sweep", sweep_margin),)),
        deepcopy(proposal),
        deepcopy(estimate),
        limits,
        deepcopy(evidence),
        decision=deepcopy(decision),
    )


def require_current_local_admission(admission, estimate, now):
    if admission.proposal.action_space not in (A1_JOINT_DELTA, A2_EE_DELTA):
        raise ValueError("local_admission_requires_robot_world_action")
    old = admission.estimate.local
    current = estimate.local
    before = validate_local_contact(old, admission.proposal.times_s[0])
    patch = validate_local_contact(current, now, generation=old.generation)
    validate_local_transition(old.selection, current.selection)
    if (
        current.selection != old.selection
        or patch.patch_id != before.patch_id
        or (
            patch.ownership != before.ownership
            and not (
                before.ownership == "observed_rigid_material"
                and patch.ownership == "observed_moving_rigid_candidate"
            )
        )
        or patch.tracking_state != "tracking"
    ):
        raise ValueError("admitted_local_reference_changed")
    for a, b in (
        (before.geometry_support, patch.geometry_support),
        (before.identity_support, patch.identity_support),
        (before.pose_support, patch.pose_support),
    ):
        if b.supported_s < a.supported_s:
            raise ValueError("admitted_support_backdated")
        for key in ("position_bound_m", "rotation_bound_rad"):
            old_bound, new_bound = getattr(a, key), getattr(b, key)
            if old_bound is not None and (
                new_bound is None or not np.isfinite(new_bound) or new_bound < 0
            ):
                raise ValueError("admitted_uncertainty_missing")
    if admission.local_evidence is None:
        raise ValueError("missing_local_sweep_evidence")
    evidence = admission.local_evidence
    elapsed = now - admission.proposal.times_s[0]
    displacement = np.linalg.norm(patch.world_pose.origin - before.world_pose.origin)
    rotation = Rotation.from_matrix(patch.world_pose.rot @ before.world_pose.rot.T).magnitude()
    if (
        displacement
        > evidence.response_position_bound_m
        + before.pose_support.position_bound_m
        + patch.pose_support.position_bound_m
        or rotation
        > evidence.response_rotation_bound_rad
        + before.pose_support.rotation_bound_rad
        + patch.pose_support.rotation_bound_rad
    ):
        raise ValueError("admitted_local_response_exceeded")
    # Spend the original proof's remaining margin on any new uncertainty or
    # local support loss. Small updates are valid only while that proof still fits.
    start, end = admission.proposal.times_s
    if evidence.unloaded_separation_margin_m is not None:
        # The original producer proved separation for the complete interval.
        # Spend any subsequent normal-bound growth; never substitute a finite
        # patch margin for a free-volume proof or silently assume zero motion.
        prepaid_p = prepaid_a = 0.0
        if evidence.unloaded_temporal_bound_m_rad is not None:
            fraction = min(1.0, max(0.0, elapsed / (end - start)))
            prepaid_p, prepaid_a = np.asarray(evidence.unloaded_temporal_bound_m_rad) * fraction
        # The maintained tracker includes the SAME linear aging primitive in
        # both its pose and normal-position bounds. Credit that identified
        # growth once, bounded by the interval's explicit relative-motion term.
        # Unidentified registration/calibration growth receives no such credit.
        linear_aging_credit = 0.0
        aging = (before.pose_temporal_position_m, patch.pose_temporal_position_m)
        if all(value is not None for value in aging):
            for state, value in zip((before, patch), aging, strict=True):
                if not np.isfinite(value) or not 0 <= value <= min(
                    state.normal_position_bound_m, state.pose_support.position_bound_m
                ):
                    raise ValueError("invalid_prepaid_pose_growth")
            linear_aging_credit = min(
                admission.limits.relative_speed_bound_m_s * elapsed,
                max(0.0, aging[1] - aging[0]),
            )
        growth = max(
            0.0,
            patch.normal_position_bound_m
            - before.normal_position_bound_m
            + admission.limits.relative_speed_bound_m_s * elapsed
            - linear_aging_credit
            - prepaid_p,
        ) + evidence.unloaded_normal_lever_m * max(
            0.0, patch.normal_rotation_bound_rad - before.normal_rotation_bound_rad - prepaid_a
        )
        if evidence.unloaded_normal_spheres:
            from alexdoor_xas.perception.material_zone import normal_uncertainty_terms

            if not (before.normal_uncertainty_sources and patch.normal_uncertainty_sources):
                raise ValueError("missing_unloaded_normal_error_sources")
            point_growth = []
            for sphere in evidence.unloaded_normal_spheres:
                center, radius = np.asarray(sphere[:3]), sphere[3]
                old_error, new_error = (
                    sum(
                        normal_uncertainty_terms(
                            state.normal_uncertainty_sources, center[None], radius
                        ).values()
                    )
                    for state in (before, patch)
                )
                lever = np.linalg.norm(center - before.world_pose.origin) + radius
                point_growth.append(
                    new_error
                    - old_error
                    + admission.limits.relative_speed_bound_m_s * elapsed
                    - linear_aging_credit
                    - prepaid_p
                    - lever * prepaid_a
                )
            # Spend growth against the SAME volume's original margin. Mixing the
            # largest distant-body growth with the smallest fingertip margin
            # loses the spatial association of otherwise complete proofs.
            margins = evidence.unloaded_normal_margins_m or (
                (evidence.unloaded_separation_margin_m,) * len(point_growth)
            )
            index = int(np.argmax(np.maximum(0.0, point_growth) - margins))
            growth = max(0.0, point_growth[index])
            remaining_margin = margins[index]
        else:
            remaining_margin = evidence.unloaded_separation_margin_m
        if growth >= remaining_margin:
            error = ValueError("admitted_unloaded_separation_exhausted")
            error.decision = dict(
                normal_position_growth_m=patch.normal_position_bound_m
                - before.normal_position_bound_m,
                explicit_relative_motion_m=admission.limits.relative_speed_bound_m_s * elapsed,
                identified_linear_aging_credit_m=linear_aging_credit,
                prepaid_interval_position_m=float(prepaid_p),
                prepaid_interval_rotation_rad=float(prepaid_a),
                excess_growth_m=float(growth),
                normal_error_domain="complete_proved_spheres"
                if evidence.unloaded_normal_spheres
                else "aggregate_anchor_and_lever",
                remaining_separation_margin_m=remaining_margin,
                normal_sphere_index=index if evidence.unloaded_normal_spheres else None,
            )
            raise error
        return
    old_error = _local_error(before, evidence, admission.limits, start, end - start)
    current_error = _local_error(patch, evidence, admission.limits, now, end - now)
    clearance_loss = max(
        max(0.0, old - new)
        for old, new in zip(before.finger_clearance_m, patch.finger_clearance_m, strict=True)
    )
    material_margins = [value for name, value in admission.margins_m if name == before.patch_id]
    if len(material_margins) != 1:
        raise ValueError("missing_local_material_margin")
    margin = material_margins[0]
    if not admission.proposal.loaded:
        margin = min(margin, evidence.unloaded_separation_m - old_error)
    if max(0.0, current_error - old_error) + clearance_loss >= margin:
        raise ValueError("admitted_local_margin_exhausted")
    # Pose changes may follow the bounded material response. The receipt has one
    # control interval only; next command needs freshly produced sweep evidence.
