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
    response_position_bound_m: float | None
    response_rotation_bound_rad: float | None
    footprint_radius_m: float | None
    unloaded_separation_m: float | None = None


def _local_error(patch, evidence, limits, now, duration):
    """Same finite footprint, material-response and stopping bound for a receipt."""
    radius = nonnegative(evidence.footprint_radius_m, "unmeasured_footprint_radius")
    response = nonnegative(evidence.response_position_bound_m, "unmeasured_material_response")
    response_rotation = nonnegative(
        evidence.response_rotation_bound_rad, "unmeasured_material_response"
    )
    supports = (patch.geometry_support, patch.pose_support, evidence.support)
    return (
        sum(s.position_bound_m for s in supports)
        + limits.robot_position_bound_m
        + sum(rotational_travel(radius, s.rotation_bound_rad) for s in supports)
        + rotational_travel(radius, limits.robot_rotation_bound_rad + response_rotation)
        + response
        + limits.relative_speed_bound_m_s
        * (
            now
            - min(
                patch.pose_support.supported_s,
                patch.identity_support.supported_s,
                evidence.support.supported_s,
            )
            + duration
        )
        + limits.stop_travel_m
    )


def admit_local_action(estimate, proposal, evidence, limits, now, *, generation):
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
            duration > bounds["duration_s"]
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
        clearance = min(
            nonnegative(v, "missing_relevant_clearance")
            for v in (*evidence.finger_clearance_m, evidence.collision_clearance_m)
        )
        error = _local_error(patch, evidence, limits, now, duration)
        if clearance <= error:
            raise ValueError("insufficient_local_action_margin")
        if (
            not proposal.loaded
            and nonnegative(evidence.unloaded_separation_m, "unmeasured_unloaded_separation")
            <= error
        ):
            raise ValueError("unloaded_contact_not_excluded")
    except (ValueError, TypeError, AttributeError, IndexError) as error:
        return ActionAdmission(False, False, str(error))
    return ActionAdmission(
        True,
        False,
        "admitted_local_geometry_only",
        ((patch.patch_id, clearance - error),),
        deepcopy(proposal),
        deepcopy(estimate),
        limits,
        deepcopy(evidence),
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
        + admission.limits.relative_speed_bound_m_s * elapsed
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
    old_error = _local_error(before, evidence, admission.limits, start, end - start)
    current_error = _local_error(patch, evidence, admission.limits, now, end - now)
    clearance_loss = max(
        max(0.0, old - new)
        for old, new in zip(before.finger_clearance_m, patch.finger_clearance_m, strict=True)
    )
    margin = min(value for _, value in admission.margins_m)
    if not admission.proposal.loaded:
        margin = min(margin, evidence.unloaded_separation_m - old_error)
    if max(0.0, current_error - old_error) + clearance_loss >= margin:
        raise ValueError("admitted_local_margin_exhausted")
    # Pose changes may follow the bounded material response. The receipt has one
    # control interval only; next command needs freshly produced sweep evidence.
