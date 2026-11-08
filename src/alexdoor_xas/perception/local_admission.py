"""A1/A2 admission from observed local material and continuous swept-space evidence."""

from copy import deepcopy
from dataclasses import dataclass
from itertools import product

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


def _material_point_terms(patch, world_poses, radius):
    """Transport all published material primitives to the same finite domain."""
    material_lever = radius + max(
        np.linalg.norm(pose.origin - patch.world_pose.origin) for pose in world_poses
    )
    geometry_lever = material_lever
    pose_position = patch.pose_support.position_bound_m
    pose_rotation = rotational_travel(material_lever, patch.pose_support.rotation_bound_rad)
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
            lever = radius + max(np.linalg.norm(pose.origin - origin) for pose in world_poses)
            position = nonnegative(position, "invalid_pose_error_source")
            angle = nonnegative(angle, "invalid_pose_error_source")
            frame_lever = nonnegative(frame_lever, "invalid_pose_error_source")
            frame_position += position + rotational_travel(frame_lever, angle)
            frame_rotation += angle
            pose_position += position
            pose_rotation += rotational_travel(lever, angle)
            if name != "calibration_fk":
                geometry_lever = max(geometry_lever, lever)
        # Consistency of the same published primitives, not a physical tolerance.
        roundoff = 32 * np.finfo(float).eps
        if not np.allclose(
            [frame_position, frame_rotation],
            [patch.pose_support.position_bound_m, patch.pose_support.rotation_bound_rad],
            rtol=roundoff,
            atol=roundoff,
        ):
            raise ValueError("incomplete_pose_error_sources")
    return dict(
        geometry_position_m=nonnegative(
            patch.geometry_support.position_bound_m, "missing_geometry_position_bound"
        ),
        pose_position_m=pose_position,
        geometry_rotation_m=rotational_travel(
            geometry_lever, patch.geometry_support.rotation_bound_rad
        ),
        pose_rotation_m=pose_rotation,
    )


@dataclass(frozen=True)
class MaterialMotionSupport:
    """Finite observed material for complete physical contact-capable volumes.

    Spheres include measured-state velocity, acceleration, robot-model error and
    the independently calibrated full stop. Nominal target-tracking and generic
    stop allowances do not describe a second motion of those same volumes.
    Pose, geometry, calibration and target response are still spent here. This
    certificate proves tangent coverage only, never normal contact or free space.

    The grid is measured in the initial WORLD frame, not an ideal object frame.
    For T0=C0 M O and Tt=Ct R M O, T0 inv(Tt)=C0 inv(R) inv(Ct): the shared
    initial object/zone basis cancels. Its orientation error must not rotate the
    measured grid again. Native registration (including both measured bases),
    camera/FK error and aging remain in pose sources. The grid's measured depth
    columns and erosion pay local geometry; normal/interface bounds stay separate.
    """

    reference_pose: ObjectFrame
    tangent_basis: tuple[tuple[float, float], ...]
    lower_m: tuple[float, float]
    cell_m: float
    clearance_m: np.ndarray
    spheres: tuple[tuple[float, float, float, float], ...]
    interval_motion_m_rad: tuple[float, float]
    geometry_position_prepaid_m: float = 0.0
    boxes: tuple["TargetReliefBox | None", ...] = ()

    def margin(self, patch, *, remaining_fraction=1.0):
        from alexdoor_xas.perception.material_zone import pose_matrix

        checked_pose(self.reference_pose)
        checked_pose(patch.world_pose)
        basis = np.asarray(self.tangent_basis, float)
        lower = np.asarray(self.lower_m, float)
        grid = np.asarray(self.clearance_m, float)
        spheres = np.asarray(self.spheres, float)
        motion = np.asarray(self.interval_motion_m_rad, float)
        if (
            basis.shape != (3, 2)
            or not np.allclose(basis.T @ basis, np.eye(2), atol=32 * np.finfo(float).eps, rtol=0)
            or lower.shape != (2,)
            or grid.ndim != 2
            or not grid.size
            or spheres.ndim != 2
            or spheres.shape[1:] != (4,)
            or not len(spheres)
            or motion.shape != (2,)
            or not np.isfinite(
                np.r_[
                    basis.ravel(),
                    lower,
                    grid.ravel(),
                    spheres.ravel(),
                    motion,
                    self.cell_m,
                    remaining_fraction,
                ]
            ).all()
            or np.any(grid < 0)
            or np.any(spheres[:, 3] < 0)
            or np.any(motion < 0)
            or self.cell_m <= 0
            or not 0 <= remaining_fraction <= 1
        ):
            raise ValueError("invalid_material_motion_support")
        prepaid = nonnegative(self.geometry_position_prepaid_m, "invalid_prepaid_geometry_error")
        if prepaid > patch.geometry_support.position_bound_m:
            raise ValueError("invalid_prepaid_geometry_error")
        mapping = pose_matrix(self.reference_pose) @ np.linalg.inv(pose_matrix(patch.world_pose))
        scale = max(1.0, float(np.linalg.norm(mapping[:3, :3], ord=2)))
        if self.boxes and len(self.boxes) != len(spheres):
            raise ValueError("invalid_material_motion_boxes")
        margins = []
        for index, (center, radius) in enumerate(zip(spheres[:, :3], spheres[:, 3], strict=True)):
            terms = _material_point_terms(patch, (ObjectFrame(center, np.eye(3)),), radius)
            error = (
                sum(value for name, value in terms.items() if name != "geometry_rotation_m")
                - prepaid
            )
            lever = np.linalg.norm(center - patch.world_pose.origin) + radius
            future = remaining_fraction * (motion[0] + lever * motion[1])
            tangent = (mapping[:3, :3] @ center + mapping[:3, 3]) @ basis
            uv = np.rint((tangent - lower) / self.cell_m).astype(int)
            if (uv < 0).any() or (uv >= np.array(grid.shape)[::-1]).any():
                return -np.inf
            rounding = np.linalg.norm(tangent - (lower + uv * self.cell_m))
            margin = float(grid[uv[1], uv[0]] - rounding - scale * (radius + error + future))
            box = self.boxes[index] if self.boxes else None
            if margin <= 0 and box is not None:
                axes, lo, hi, reserve = box.parameters()
                midpoint, half = (lo + hi) / 2, (hi - lo) / 2
                center = axes @ midpoint
                radius = float(np.linalg.norm(axes, ord=2) * np.linalg.norm(half) + reserve)
                terms = _material_point_terms(patch, (ObjectFrame(center, np.eye(3)),), radius)
                error = (
                    sum(value for name, value in terms.items() if name != "geometry_rotation_m")
                    - prepaid
                )
                lever = np.linalg.norm(center - patch.world_pose.origin) + radius
                future = remaining_fraction * (motion[0] + lever * motion[1])
                project = basis.T @ mapping[:3, :3] @ axes
                tangent = project @ midpoint + basis.T @ mapping[:3, 3]
                span = np.abs(project) @ half
                first = np.floor((tangent - span - lower) / self.cell_m).astype(int)
                last = np.ceil((tangent + span - lower) / self.cell_m).astype(int)
                if (first >= 0).all() and (last < np.array(grid.shape)[::-1]).all():
                    # Every raster cell of the full projected box is retained,
                    # including interior holes. Grid clearance already pays its
                    # cell diagonal; no empty corners of the sphere are needed.
                    inside = grid[first[1] : last[1] + 1, first[0] : last[0] + 1].min()
                    margin = max(margin, float(inside - scale * (reserve + error + future)))
            margins.append(margin)
        return min(margins)

    def require(self, patch, *, remaining_fraction=1.0):
        margin = self.margin(patch, remaining_fraction=remaining_fraction)
        if margin <= 0:
            error = ValueError("material_motion_support_exhausted")
            error.decision = dict(material_motion_margin_m=margin)
            raise error
        return margin


@dataclass(frozen=True)
class TargetReliefBox:
    """Compact enclosure of the same complete physical motion and stop volume."""

    axes_world: tuple[tuple[float, float, float], ...]
    lower_m: tuple[float, float, float]
    upper_m: tuple[float, float, float]
    radius_m: float

    def parameters(self):
        axes = np.asarray(self.axes_world, dtype=float)
        lower, upper = np.asarray(self.lower_m, float), np.asarray(self.upper_m, float)
        if (
            axes.shape != (3, 3)
            or lower.shape != (3,)
            or upper.shape != (3,)
            or not np.isfinite(np.r_[axes.ravel(), lower, upper, self.radius_m]).all()
            or np.linalg.det(axes) == 0
            or np.any(lower > upper)
            or self.radius_m < 0
        ):
            raise ValueError("invalid_target_relief_box")
        return axes, lower, upper, self.radius_m


@dataclass(frozen=True)
class TargetReliefExclusion:
    """Complete physical enclosures outside the original padded target relief.

    Each optional box independently encloses the SAME physical volume as its
    sphere; background and normal proofs remain separate. Refine only spatial
    association of distance and point uncertainty, retaining every primitive.
    """

    reference_pose: ObjectFrame
    tangent_basis: tuple[tuple[float, float], ...]
    lower_m: tuple[float, float]
    upper_m: tuple[float, float]
    spheres: tuple[tuple[float, float, float, float], ...]
    interval_motion_m_rad: tuple[float, float]
    boxes: tuple[TargetReliefBox, ...] = ()
    minimum_cell_m: float | None = None
    maximum_queries: int | None = None
    # An optional height layer is qualified only inside this entire observed root.
    tangent_domain_m: tuple[tuple[float, float], tuple[float, float]] | None = None

    def margin(self, patch, *, remaining_fraction=1.0):
        checked_pose(self.reference_pose)
        checked_pose(patch.world_pose)
        basis = np.asarray(self.tangent_basis, dtype=float)
        lower, upper = np.asarray(self.lower_m), np.asarray(self.upper_m)
        spheres = np.asarray(self.spheres, dtype=float)
        motion = np.asarray(self.interval_motion_m_rad, dtype=float)
        if (
            basis.shape != (3, 2)
            or not np.isfinite(basis).all()
            or not np.allclose(basis.T @ basis, np.eye(2), atol=32 * np.finfo(float).eps, rtol=0)
            or lower.shape != (2,)
            or upper.shape != (2,)
            or not np.isfinite(np.r_[lower, upper]).all()
            or np.any(lower > upper)
            or spheres.ndim != 2
            or spheres.shape[1:] != (4,)
            or not len(spheres)
            or not np.isfinite(spheres).all()
            or np.any(spheres[:, 3] < 0)
            or motion.shape != (2,)
            or not np.isfinite(motion).all()
            or np.any(motion < 0)
            or not np.isfinite(remaining_fraction)
            or not 0 <= remaining_fraction <= 1
        ):
            raise ValueError("invalid_target_relief_exclusion")
        domain = getattr(self, "tangent_domain_m", None)
        if domain is not None:
            domain = np.asarray(domain, dtype=float)
            if (
                domain.shape != (2, 2)
                or not np.isfinite(domain).all()
                or np.any(domain[0] >= domain[1])
            ):
                raise ValueError("invalid_target_relief_domain")
        from alexdoor_xas.perception.material_zone import pose_matrix

        boxes = getattr(self, "boxes", ())
        if boxes and (
            len(boxes) != len(spheres)
            or self.minimum_cell_m is None
            or not np.isfinite(self.minimum_cell_m)
            or self.minimum_cell_m <= 0
            or type(self.maximum_queries) is not int
            or self.maximum_queries < 1
        ):
            raise ValueError("invalid_target_relief_box_limits")
        parameters = [box.parameters() for box in boxes]
        queries = 0

        mapping = pose_matrix(self.reference_pose) @ np.linalg.inv(pose_matrix(patch.world_pose))
        # Native float32 camera transports can be slightly non-isometric. Cover
        # that measured operator norm rather than assuming an exactly rigid map.
        scale = max(1.0, float(np.linalg.norm(mapping[:3, :3], ord=2)))
        margins = []
        for index, sphere in enumerate(spheres):
            center, radius = sphere[:3], sphere[3]
            point = ObjectFrame(center, np.eye(3))
            error = sum(_material_point_terms(patch, (point,), radius).values())
            lever = np.linalg.norm(center - patch.world_pose.origin) + radius
            future = remaining_fraction * (motion[0] + lever * motion[1])
            mapped = mapping[:3, :3] @ center + mapping[:3, 3]
            tangent = mapped @ basis
            distance = np.linalg.norm(np.maximum(lower - tangent, np.maximum(tangent - upper, 0)))
            margin = float(distance - scale * (radius + error + future))
            if domain is not None:
                inside = np.min(np.minimum(tangent - domain[0], domain[1] - tangent))
                margin = min(margin, float(inside - scale * (radius + error + future)))
            if margin <= 0 and boxes:
                axes, lo, hi, reserve = parameters[index]
                project = basis.T @ mapping[:3, :3] @ axes
                offset = mapping[:3, 3] @ basis
                axes_scale = float(np.linalg.norm(axes, ord=2))
                pending, refined_margin = [(lo, hi)], np.inf
                while pending:
                    if queries >= self.maximum_queries:
                        raise ValueError("target_relief_query_budget_exhausted")
                    queries += 1
                    lo, hi = pending.pop()
                    midpoint, half = (lo + hi) / 2, (hi - lo) / 2
                    center = axes @ midpoint
                    radius = axes_scale * np.linalg.norm(half) + reserve
                    point = ObjectFrame(center, np.eye(3))
                    error = sum(_material_point_terms(patch, (point,), radius).values())
                    lever = np.linalg.norm(center - patch.world_pose.origin) + radius
                    future = remaining_fraction * (motion[0] + lever * motion[1])
                    tangent, span = project @ midpoint + offset, np.abs(project) @ half
                    distance = np.linalg.norm(
                        np.maximum(lower - tangent - span, np.maximum(tangent - span - upper, 0))
                    )
                    cell_margin = float(distance - scale * (reserve + error + future))
                    if domain is not None:
                        inside = np.min(
                            np.minimum(tangent - span - domain[0], domain[1] - tangent - span)
                        )
                        cell_margin = min(
                            cell_margin, float(inside - scale * (reserve + error + future))
                        )
                    if not np.isfinite(cell_margin):
                        raise ValueError("invalid_target_relief_box")
                    if cell_margin > 0:
                        refined_margin = min(refined_margin, cell_margin)
                    elif axes_scale * np.max(hi - lo) <= self.minimum_cell_m:
                        refined_margin = cell_margin
                        break
                    else:
                        pending.extend(
                            (np.where(bits, midpoint, lo), np.where(bits, hi, midpoint))
                            for bits in product((False, True), repeat=3)
                        )
                margin = max(margin, refined_margin)
            margins.append(margin)
        return min(margins)

    def require(self, patch, *, remaining_fraction=1.0):
        margin = self.margin(patch, remaining_fraction=remaining_fraction)
        if margin <= 0:
            error = ValueError("target_relief_exclusion_exhausted")
            error.decision = dict(target_relief_margin_m=margin)
            raise error
        return margin


@dataclass(frozen=True)
class TargetFrontHeightExclusion:
    """One fixed observed height layer over the same complete physical volume.

    Recheck both finite spatial eligibility and normal separation, including on
    loaded receipts. A convex proof never establishes free space for its ball.
    """

    spatial: TargetReliefExclusion
    front_extent_m: float
    normal_model_bound_m: float
    vertices_world: tuple[tuple[float, float, float], ...] = ()
    radius_m: float = 0.0

    def margin(self, patch, *, remaining_fraction=1.0):
        spatial = self.spatial.margin(patch, remaining_fraction=remaining_fraction)
        front = nonnegative(self.front_extent_m, "invalid_target_front_height")
        model = nonnegative(self.normal_model_bound_m, "invalid_target_front_height")
        reserve = nonnegative(self.radius_m, "invalid_target_front_height")
        spheres = np.asarray(self.spatial.spheres, float)
        vertices = np.asarray(self.vertices_world, float)
        if self.spatial.tangent_domain_m is None or len(spheres) != 1:
            raise ValueError("missing_target_front_height_domain")
        center, radius = spheres[0, :3], spheres[0, 3]
        normal = patch.world_pose.rot[:, 0]
        if vertices.size:
            if (
                vertices.ndim != 2
                or vertices.shape[1:] != (3,)
                or not np.isfinite(vertices).all()
                or np.max(np.linalg.norm(vertices - center, axis=1)) + reserve
                > np.nextafter(radius, np.inf)
            ):
                raise ValueError("invalid_target_front_height_volume")
            foremost = float((vertices @ normal).max() + reserve)
        else:
            if reserve != 0 or vertices.shape != (0,):
                raise ValueError("invalid_target_front_height_volume")
            foremost = float(center @ normal + radius)
        lever = np.linalg.norm(center - patch.world_pose.origin) + radius
        if patch.normal_uncertainty_sources:
            from alexdoor_xas.perception.material_zone import normal_uncertainty_terms

            error = sum(
                normal_uncertainty_terms(
                    patch.normal_uncertainty_sources, center[None], radius
                ).values()
            )
        else:
            error = nonnegative(
                patch.normal_position_bound_m, "invalid_target_front_height_uncertainty"
            ) + lever * nonnegative(
                patch.normal_rotation_bound_rad, "invalid_target_front_height_uncertainty"
            )
        if not np.isfinite(error) or error < 0:
            raise ValueError("invalid_target_front_height_uncertainty")
        motion = self.spatial.interval_motion_m_rad
        future = remaining_fraction * (motion[0] + lever * motion[1])
        normal_margin = float(
            patch.world_pose.origin @ normal - foremost - error - model - front - future
        )
        return min(spatial, normal_margin)

    def require(self, patch, *, remaining_fraction=1.0):
        margin = self.margin(patch, remaining_fraction=remaining_fraction)
        if margin <= 0:
            error = ValueError("target_front_height_exhausted")
            error.decision = dict(target_front_height_margin_m=margin)
            raise error
        return margin


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
    # Re-prove spatial target exclusion for localized panel certificates,
    # independently of normal-only separation and finite contact receipts.
    target_relief_exclusion: TargetReliefExclusion | None = None
    target_height_exclusions: tuple[TargetFrontHeightExclusion, ...] = ()
    material_motion_support: MaterialMotionSupport | None = None


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
    material_lever = radius + max(
        np.linalg.norm(pose.origin - patch.world_pose.origin) for pose in evidence.world_poses
    )
    point = _material_point_terms(patch, evidence.world_poses, radius)
    return dict(
        geometry_position_m=point["geometry_position_m"] - prepaid,
        pose_position_m=point["pose_position_m"],
        sweep_position_m=evidence.support.position_bound_m,
        robot_position_m=limits.robot_position_bound_m,
        geometry_rotation_m=point["geometry_rotation_m"],
        pose_rotation_m=point["pose_rotation_m"],
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
        if evidence.target_relief_exclusion is not None:
            evidence.target_relief_exclusion.require(patch)
        for certificate in getattr(evidence, "target_height_exclusions", ()):
            certificate.require(patch)
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
        elif getattr(evidence, "material_motion_support", None) is not None:
            if not proposal.loaded or sweep_margin is None:
                raise ValueError("material_motion_requires_loaded_full_robot_proof")
            if nonnegative(sweep_margin, "invalid_robot_sweep_margin") <= 0:
                raise ValueError("insufficient_robot_sweep_margin")
            # Both nominal finite faces must exist. The continuous physical
            # motion proof, rather than nominal endpoint error, covers every
            # volume which can actually contact during this command and stop.
            if (
                min(
                    nonnegative(v, "missing_relevant_clearance")
                    for v in evidence.finger_clearance_m
                )
                <= 0
            ):
                raise ValueError("missing_two_finger_material_support")
            clearance = evidence.material_motion_support.require(patch)
            error = 0.0
            decision = dict(
                margin_kind="complete_physical_material_motion",
                material_id=patch.patch_id,
                material_margin_m=clearance,
                contact_capable_volumes=len(evidence.material_motion_support.spheres),
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
    start, end = admission.proposal.times_s
    fraction = min(1.0, max(0.0, (end - now) / (end - start)))
    if evidence.target_relief_exclusion is not None:
        evidence.target_relief_exclusion.require(patch, remaining_fraction=fraction)
    for certificate in getattr(evidence, "target_height_exclusions", ()):
        certificate.require(patch, remaining_fraction=fraction)
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
    if getattr(evidence, "material_motion_support", None) is not None:
        evidence.material_motion_support.require(patch, remaining_fraction=fraction)
        if any(value is None or value <= 0 for value in patch.finger_clearance_m):
            raise ValueError("admitted_material_support_lost")
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
