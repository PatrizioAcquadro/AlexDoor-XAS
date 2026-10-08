"""Numerical action/load admission; surface queries and measured bounds are producer-owned."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from alexdoor_xas.action.b1 import ACTION_DIMS, checked_pose, finite_vector, relative_pose
from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.perception.contracts import (
    OPERATIONAL_V1,
    ContactSelection,
    DoorEstimate,
    FieldSupport,
    RobotFeedback,
    geometric_admission,
    same_pose,
    validate_contact_transition,
    validate_reference,
)


def nonnegative(value, reason):
    if value is None or not np.isfinite(value) or value < 0:
        raise ValueError(reason)
    return float(value)


def rotational_travel(radius, angle):
    """Worst point displacement under a bounded rigid rotation, without small-angle assumptions."""
    return 2 * radius * np.sin(min(angle, np.pi) / 2)


@dataclass(frozen=True)
class ActionLimits:
    """Required action/error/stop bounds; unmeasured values remain None.

    Values come from the shared measured recipe, not the scoring thresholds.
    Producers cover continuous action semantics and tracking/interpolation error
    between command knots; finite differences below bound the submitted commands.
    """

    duration_s: float | None
    displacement_m: float | None
    speed_m_s: float | None
    angular_speed_rad_s: float | None
    robot_position_bound_m: float | None
    robot_rotation_bound_rad: float | None
    relative_speed_bound_m_s: float | None
    stop_travel_m: float | None


@dataclass(frozen=True)
class ActionProposal:
    """One actual command trajectory, N+1 poses/times including the initial tool pose."""

    source: str
    action_space: str
    action: np.ndarray
    times_s: np.ndarray
    world_poses: tuple[ObjectFrame, ...]
    selection_id: str
    loaded: bool
    required_dimensions: tuple[int, ...] = ()


@dataclass(frozen=True)
class RotationLevers:
    """Maximum radii over the relevant footprint/sweep for each rotation source."""

    hinge_m: float | None
    angle_m: float | None
    panel_m: float | None
    contact_m: float | None
    clearance_m: float | None
    robot_m: float | None
    response_m: float | None


@dataclass(frozen=True)
class HypothesisActionSupport:
    """Same physical patch/trajectory expressed and bounded for one hypothesis.

    Clearances are minima over the continuous proposed sweep AND stopping envelope,
    for both finite finger footprints and robot/leaf collision volumes. They must
    come from observed geometry, including interpolation uncertainty. Unknown
    space outside that envelope does not set unknown_in_envelope. Response bounds
    include leaf response/slip and disagreement with the proposed world path.
    """

    hypothesis_id: str
    contact_local: ObjectFrame
    world_poses: tuple[ObjectFrame, ...]
    support: FieldSupport
    finger_clearance_m: tuple[float | None, float | None]
    collision_clearance_m: float | None
    unknown_in_envelope: bool
    response_position_bound_m: float | None
    response_rotation_bound_rad: float | None
    rotation_levers: RotationLevers
    unloaded_separation_m: float | None = None


@dataclass(frozen=True)
class ActionAdmission:
    admitted: bool
    provisional: bool
    reason: str
    margins_m: tuple[tuple[str, float], ...] = ()
    proposal: ActionProposal | None = None
    estimate: DoorEstimate | None = None
    limits: ActionLimits | None = None


def admit_action(estimate, proposal, hypotheses, limits, now, *, generation):
    """Admit only the submitted trajectory; never choose/correct it or drop a hypothesis."""
    try:
        validate_reference(estimate, now, generation=generation)
        state = estimate.operational
        provisional = not geometric_admission(
            estimate, now, profile=OPERATIONAL_V1, generation=generation
        ).qualified
        if proposal.source not in ("policy", "diagnostic") or (
            provisional and proposal.source != "diagnostic"
        ):
            raise ValueError("provisional_requires_diagnostic_source")
        if proposal.action_space not in ACTION_DIMS:
            raise ValueError("unknown_action_space")
        finite_vector(proposal.action, ACTION_DIMS[proposal.action_space])
        if proposal.selection_id != state.contact.selection_id:
            raise ValueError("wrong_contact_selection")
        if type(proposal.loaded) is not bool:
            raise ValueError("undeclared_load_requirement")
        times = np.asarray(proposal.times_s)
        if (
            times.ndim != 1
            or len(times) < 2
            or len(times) != len(proposal.world_poses)
            or not np.isfinite(times).all()
            or times[0] != now
            or not (np.diff(times) > 0).all()
        ):
            raise ValueError("invalid_action_schedule")
        for pose in proposal.world_poses:
            checked_pose(pose)
        bounds = {k: nonnegative(v, f"unmeasured_{k}") for k, v in vars(limits).items()}
        displacement = [
            np.linalg.norm(b.origin - a.origin)
            for a, b in zip(proposal.world_poses[:-1], proposal.world_poses[1:], strict=True)
        ]
        rotations = [
            Rotation.from_matrix(b.rot @ a.rot.T).magnitude()
            for a, b in zip(proposal.world_poses[:-1], proposal.world_poses[1:], strict=True)
        ]
        if (
            times[-1] - times[0] > bounds["duration_s"]
            or sum(displacement) > bounds["displacement_m"]
            or np.any(np.array(displacement) / np.diff(times) > bounds["speed_m_s"])
            or np.any(np.array(rotations) / np.diff(times) > bounds["angular_speed_rad_s"])
        ):
            raise ValueError("action_exceeds_declared_limits")
        if proposal.required_dimensions:
            if state.dimensions_support is None:
                raise ValueError("required_dimensions_unavailable")
            state.dimensions_support.require(now, generation)
            state.dimensions_support.require_bounds(position=True)
            for index in proposal.required_dimensions:
                if type(index) is not int or index not in (0, 1, 2):
                    raise ValueError("invalid_dimension_requirement")
                if (
                    estimate.dimensions is None
                    or not np.isfinite(estimate.dimensions[index])
                    or (estimate.dimensions[index] <= 0)
                ):
                    raise ValueError("required_dimensions_unavailable")
        by_id = {h.hypothesis_id: h for h in hypotheses}
        if len(by_id) != len(hypotheses) or set(by_id) != {
            h.hypothesis_id for h in state.hypotheses
        }:
            raise ValueError("unbounded_hinge_hypotheses")
        margins = []
        for hinge in state.hypotheses:
            evidence = by_id[hinge.hypothesis_id]
            panel = ObjectFrame(hinge.frame.origin, estimate.panel_rotation)
            expected_local = relative_pose(state.contact.world_pose, panel)
            if not same_pose(evidence.contact_local, expected_local):
                raise ValueError("different_physical_contact")
            if len(evidence.world_poses) != len(proposal.world_poses) or any(
                not same_pose(a, b)
                for a, b in zip(evidence.world_poses, proposal.world_poses, strict=True)
            ):
                raise ValueError("different_world_trajectory")
            evidence.support.require(now, generation)
            evidence.support.require_bounds(position=True, rotation=True)
            if evidence.unknown_in_envelope is not False:
                raise ValueError("unknown_relevant_swept_space")
            if len(evidence.finger_clearance_m) != 2:
                raise ValueError("missing_two_finger_support")
            clearance = min(
                nonnegative(v, "missing_relevant_clearance")
                for v in (*evidence.finger_clearance_m, evidence.collision_clearance_m)
            )
            if evidence.rotation_levers is None:
                raise ValueError("unmeasured_rotation_levers")
            radii = {
                k: nonnegative(v, f"unmeasured_{k}_lever")
                for k, v in vars(evidence.rotation_levers).items()
            }
            offset = state.contact.world_pose.origin - hinge.frame.origin
            axis = hinge.frame.rot[:, 2]
            if radii["hinge_m"] < np.linalg.norm(offset) or radii["angle_m"] < np.linalg.norm(
                offset - axis * (offset @ axis)
            ):
                raise ValueError("underbounded_contact_lever_arm")
            response_p = nonnegative(evidence.response_position_bound_m, "unmeasured_leaf_response")
            response_r = nonnegative(
                evidence.response_rotation_bound_rad, "unmeasured_leaf_response"
            )
            supports = (
                hinge.support,
                state.contact.local_support,
                state.contact.world_support,
                evidence.support,
            )
            translation = sum(s.position_bound_m for s in supports)
            rotations = (
                (radii["hinge_m"], hinge.support.rotation_bound_rad),
                (radii["angle_m"], state.angle_support.rotation_bound_rad),
                (radii["panel_m"], state.panel_support.rotation_bound_rad),
                (radii["contact_m"], state.contact.local_support.rotation_bound_rad),
                (radii["contact_m"], state.contact.world_support.rotation_bound_rad),
                (radii["clearance_m"], evidence.support.rotation_bound_rad),
                (radii["robot_m"], bounds["robot_rotation_bound_rad"]),
                (radii["response_m"], response_r),
            )
            age = max(
                now - s.supported_s
                for s in (state.angle_support, state.panel_support, state.contact.world_support)
            )
            rotation_error = sum(rotational_travel(r, a) for r, a in rotations)
            error = (
                translation
                + rotation_error
                + response_p
                + (
                    bounds["robot_position_bound_m"]
                    + bounds["relative_speed_bound_m_s"] * (age + times[-1] - times[0])
                    + bounds["stop_travel_m"]
                )
            )
            margin = clearance - error
            if margin <= 0:
                raise ValueError("insufficient_action_margin")
            if (
                not proposal.loaded
                and nonnegative(evidence.unloaded_separation_m, "unmeasured_unloaded_separation")
                <= error
            ):
                raise ValueError("unloaded_contact_not_excluded")
            margins.append((hinge.hypothesis_id, float(margin)))
    except (ValueError, TypeError, AttributeError, IndexError) as error:
        return ActionAdmission(False, False, str(error))
    return ActionAdmission(
        True,
        provisional,
        "admitted_geometry_only",
        tuple(margins),
        deepcopy(proposal),
        deepcopy(estimate),
        limits,
    )


def require_current_admission(admission, estimate, now):
    """Validate receipt identity and bounds; segment coordinates remain anchored at admission."""
    if (
        admission is None
        or not admission.admitted
        or (admission.estimate is None or admission.proposal is None)
    ):
        raise ValueError("action_not_admitted")
    before = admission.estimate.operational
    if not admission.proposal.times_s[0] <= now <= admission.proposal.times_s[-1]:
        raise ValueError("outside_admitted_schedule")
    validate_reference(estimate, now, generation=before.generation)
    current = estimate.operational
    validate_contact_transition(before.contact, current.contact)
    identity = (
        current.leaf_id,
        current.reference_id,
        current.contact.selection_id,
        current.contact.patch_id,
    )
    if identity != (
        before.leaf_id,
        before.reference_id,
        before.contact.selection_id,
        before.contact.patch_id,
    ) or {h.hypothesis_id for h in current.hypotheses} != {
        h.hypothesis_id for h in before.hypotheses
    }:
        raise ValueError("admitted_reference_changed")
    by_id = {h.hypothesis_id: h for h in current.hypotheses}
    hinge_pairs = [(old, by_id[old.hypothesis_id]) for old in before.hypotheses]
    for old, new in hinge_pairs:
        if (
            np.linalg.norm(new.frame.origin - old.frame.origin) > old.support.position_bound_m
            or Rotation.from_matrix(new.frame.rot @ old.frame.rot.T).magnitude()
            > old.support.rotation_bound_rad
        ):
            raise ValueError("admitted_reference_changed")
    if (
        np.linalg.norm(current.contact.local_pose.origin - before.contact.local_pose.origin)
        > before.contact.local_support.position_bound_m
        or Rotation.from_matrix(
            current.contact.local_pose.rot @ before.contact.local_pose.rot.T
        ).magnitude()
        > before.contact.local_support.rotation_bound_rad
    ):
        raise ValueError("admitted_material_contact_changed")
    pairs = [(old.support, new.support) for old, new in hinge_pairs] + [
        (before.angle_support, current.angle_support),
        (before.panel_support, current.panel_support),
        (before.contact.local_support, current.contact.local_support),
        (before.contact.world_support, current.contact.world_support),
    ]
    if any(
        new_value is None or old_value is None or new_value > old_value
        for old, new in pairs
        for old_value, new_value in ((old.rotation_bound_rad, new.rotation_bound_rad),)
    ) or any(
        old.position_bound_m is not None
        and (new.position_bound_m is None or new.position_bound_m > old.position_bound_m)
        for old, new in pairs
    ):
        raise ValueError("admitted_uncertainty_increased")
    if any(new.supported_s < old.supported_s for old, new in pairs):
        raise ValueError("admitted_support_backdated")


@dataclass(frozen=True)
class ContactCandidate:
    selection: ContactSelection
    reachable: bool
    finger_clearance_m: tuple[float, float]
    hinge_distance_m: float | None


def select_contact(candidates, *, hinge_resolved):
    """Deterministic diagnostic ordering of producer-verified observed candidates."""
    usable = [
        c
        for c in candidates
        if c.reachable
        and len(c.finger_clearance_m) == 2
        and np.isfinite(c.finger_clearance_m).all()
        and min(c.finger_clearance_m) > 0
        and (
            not hinge_resolved
            or c.hinge_distance_m is not None
            and np.isfinite(c.hinge_distance_m)
            and c.hinge_distance_m >= 0
        )
    ]
    if not usable:
        raise ValueError("no_supported_contact_candidate")
    return min(
        usable,
        key=lambda c: (
            -min(c.finger_clearance_m),
            -c.hinge_distance_m if hinge_resolved else 0,
            c.selection.selection_id,
        ),
    ).selection


@dataclass(frozen=True)
class LoadAdmission:
    admitted: bool
    reason: str


def admit_load(
    feedback: RobotFeedback,
    now,
    joint_names,
    *,
    timeout_s,
    model_semantics,
    residual_bound_nm,
    load_upper_n,
    force_limit_n,
    model_verified,
    stop_verified,
    contact_consistent,
):
    """Robot-only producer evidence; anomaly detection alone cannot authorize loaded control."""
    try:
        invalid_names = any(not isinstance(name, str) or not name for name in joint_names)
        if not joint_names or invalid_names or (len(set(joint_names)) != len(joint_names)):
            raise ValueError("invalid_feedback_joint_order")
        if feedback.torque_nm is None:
            raise ValueError(feedback.reason or "force_feedback_unavailable")
        timeout = nonnegative(timeout_s, "unmeasured_feedback_timeout")
        times = [feedback.timestamp_s, feedback.available_s, now]
        if (
            timeout == 0
            or not np.isfinite(times).all()
            or (
                not 0 <= feedback.timestamp_s <= feedback.available_s <= now
                or now - feedback.timestamp_s > timeout
            )
        ):
            raise ValueError("stale_or_unavailable_feedback")
        if (
            feedback.joint_names != tuple(joint_names)
            or feedback.units != "N m"
            or feedback.sign_convention != "positive_joint_coordinate"
            or feedback.source not in ("reported_joint_state", "simulated_actuator_feedback")
            or feedback.semantics not in ("actuator_side", "joint_output")
            or model_semantics != feedback.semantics
        ):
            raise ValueError("unverified_torque_semantics")
        finite_vector(feedback.torque_nm, len(joint_names))
        if feedback.healthy is not True or feedback.reason:
            raise ValueError(feedback.reason or "unhealthy_robot_feedback")
        if model_verified is not True or residual_bound_nm is None:
            raise ValueError("unverified_load_model")
        if np.any(finite_vector(residual_bound_nm, len(joint_names)) < 0):
            raise ValueError("invalid_residual_bound")
        if nonnegative(load_upper_n, "unmeasured_load_bound") > nonnegative(
            force_limit_n, "missing_physical_force_limit"
        ):
            raise ValueError("load_limit")
        if stop_verified is not True or contact_consistent is not True:
            raise ValueError("unverified_stop_or_contact_load")
    except (ValueError, TypeError, AttributeError) as error:
        return LoadAdmission(False, str(error))
    return LoadAdmission(True, "admitted_load_model")
