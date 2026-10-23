"""Observed A1/A2 motion evidence from actual robot state and measured workcell rays."""

import json
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import ConvexHull

from alexdoor_xas.action.b1 import apply_pose_delta
from alexdoor_xas.action.spaces import A1_JOINT_DELTA
from alexdoor_xas.assets.purdue import ARM_JOINTS, derive_push_geometry
from alexdoor_xas.kinematics.self_clearance import RobotSelfClearance
from alexdoor_xas.kinematics.urdf_geometry import RobotGeometry, stop_box_cover
from alexdoor_xas.perception.admission import ActionAdmission, ActionLimits, ActionProposal
from alexdoor_xas.perception.contact_geometry import distal_body_order, observed_front_extent
from alexdoor_xas.perception.contracts import FieldSupport, validate_local_contact
from alexdoor_xas.perception.evaluation import json_safe
from alexdoor_xas.perception.local_admission import LocalActionSupport, admit_local_action
from alexdoor_xas.perception.material_zone import object_frame, pose_matrix
from alexdoor_xas.perception.scan import ObservedSpace, footprint_support
from alexdoor_xas.perception.workspace import (
    CommonWorkspace,
    ObservedBackground,
    StaticWorkspaceSupport,
)
from alexdoor_xas.policies.rollout import Command
from alexdoor_xas.recording.b1_runtime import array

# Match the unchanged Purdue importer and authored USD rigid-body filters.
BODY_FILTERS = (
    ("HEAD_LINK", "TORSO_LINK"),
    ("LEFT_GRIPPER_Y_LINK", "LEFT_WRIST_Z_LINK"),
    ("RIGHT_GRIPPER_Y_LINK", "RIGHT_WRIST_Z_LINK"),
    ("left_WSG32_NEGATIVE_JAW_LINK", "left_WSG32_POSITIVE_JAW_LINK"),
    ("right_WSG32_NEGATIVE_JAW_LINK", "right_WSG32_POSITIVE_JAW_LINK"),
)


class PilotMotionEvidence:
    """One actual proposed command; no teacher actions, door assets or evaluator.

    Static calibration is common across conditions. Current target separation,
    full relative robot clearance, continuous motion and stopping are separate
    proofs. Unmeasured or unsupported volumes fail closed with retained reasons.
    """

    def __init__(self, io, provider, space, recipe, config, output, *, load_calibration):
        self.io, self.provider, self.space = io, provider, space
        self.config = config
        self.load_calibration = load_calibration
        self.model = RobotGeometry(io.env.robot.cfg.spawn.asset_path, ARM_JOINTS)
        self.self_clearance = RobotSelfClearance(self.model, BODY_FILTERS)
        self.faces = derive_push_geometry(io.env.robot.cfg.spawn.asset_path).contact_covers()
        self.background_mode = config.get("background_recording_groups") is not None
        self.workspace = (
            ObservedBackground.from_recording_groups(
                config["background_recording_groups"],
                recipe.config,
                config["camera_bound"],
                config["background_sample_times_s"],
            )
            if self.background_mode
            else CommonWorkspace.from_recordings(
                config["workspace_recordings"], recipe.config, config["camera_bound"]
            )
        )
        self.rigid_workspace = self.rigid_identity = None
        self.rigid_reference_workspace = (
            CommonWorkspace.from_recordings(
                config["workspace_recordings"], recipe.config, config["camera_bound"]
            )
            if self.background_mode
            else None
        )
        self.interface_model = None
        if self.background_mode:
            self.interface_model = json.loads(Path(config["interface_calibration"]).read_text())
            if self.interface_model["verified"] is not True:
                raise ValueError("unverified_rigid_interface_component")
            if (
                config["stop_duration_s"] != self.interface_model["stop_duration_s"]
                or config["stop_force_domain_n"] > self.interface_model["force_domain_n"]
                or config["max_joint_speed_rad_s"] > self.interface_model["max_joint_speed_rad_s"]
                or config["robot_stop_rotation_bound_rad"]
                < self.interface_model["robot_stop_rotation_bound_rad"]
            ):
                raise ValueError("outside_rigid_interface_calibration_domain")
            values = [
                self.interface_model[k]
                for k in (
                    "indentation_bound_m",
                    "planarity_bound_m",
                    "normal_model_bound_rad",
                    "panel_front_bound_m",
                )
            ] + [config["robot_stop_rotation_bound_rad"], config["stop_duration_s"]]
            if not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError("invalid_rigid_interface_domain")
        self.stop = np.asarray(config["stop_joint_bound_rad"], float)
        self.acceleration = np.asarray(config["acceleration_max_rad_s2"], float)
        self.shape_stop = np.asarray(config["shape_stop_bound_m"], float)
        if self.shape_stop.shape != (len(self.model.shapes),) or not (
            np.isfinite(self.shape_stop).all() and (self.shape_stop > 0).all()
        ):
            raise ValueError("missing_verified_full_link_stop_bounds")
        if any(
            a.shape != (7,) or not np.isfinite(a).all() or (a < 0).any()
            for a in (self.stop, self.acceleration)
        ):
            raise ValueError("invalid_measured_motion_domain")
        self.stop_boxes = None
        if boxes := config.get("shape_stop_boxes_m"):
            lower, upper = np.asarray(boxes["lower"], float), np.asarray(boxes["upper"], float)
            if (
                lower.shape != (len(self.model.shapes), 3)
                or upper.shape != lower.shape
                or not np.isfinite(
                    np.r_[lower.ravel(), upper.ravel(), config["stop_cover_radius_m"]]
                ).all()
                or (lower > 0).any()
                or (upper < 0).any()
                or config["stop_cover_radius_m"] <= 0
            ):
                raise ValueError("missing_verified_full_link_stop_boxes")
            self.stop_boxes = lower, upper
        self.cells = self.model.cells()
        self.log = Path(output).open("x")
        self.last = None
        Path(output).with_suffix(".background-sources.json").write_text(
            json.dumps(self.workspace.source_clocks) + "\n"
        )

    def close(self):
        self.log.close()

    def __call__(self, observation, tool, row, remaining):
        started = perf_counter()
        diagnostics = dict(time_s=float(observation.time_s), source="observed_robot_motion")
        try:
            result = self.produce(observation, tool, row, remaining, diagnostics)
        except (ValueError, TypeError, AttributeError, IndexError) as error:
            result = ActionAdmission(False, False, str(error))
        diagnostics.update(
            admitted=result.admitted,
            reason=result.reason,
            margins_m=result.margins_m,
            decision=result.decision,
            wall_s=perf_counter() - started,
        )
        self.last = json_safe(diagnostics)
        self.log.write(json.dumps(self.last, allow_nan=False) + "\n")
        self.log.flush()
        return result

    def produce(self, observation, tool, row, remaining, diagnostics):
        if remaining < 1:
            raise ValueError("no_remaining_control_interval")
        io, env, now = self.io, self.io.env, float(observation.time_s)
        if now != float(env.capture.sample.time_s):
            raise ValueError("unsynchronized_motion_state")
        patch = validate_local_contact(
            observation.estimate.local, now, generation=observation.generation
        )
        candidate = next(
            c for c in self.provider.tracking.candidates if c.candidate_id == patch.patch_id
        )
        cover = candidate.material_cover
        if self.provider.config.get("material_depth_cover") is not None and cover is None:
            raise ValueError("missing_observed_material_cover")
        if cover is not None:
            cover.support.require(now, observation.generation)
        normal_range = (0.0, 0.0) if cover is None else cover.normal_range_m
        kind = "joints" if self.space == A1_JOINT_DELTA else "delta"
        endpoint = (
            io.preview(Command(kind, row)) if kind == "joints" else apply_pose_delta(tool, row)
        )
        poses = (tool, endpoint)
        positions = dict(
            zip(env.robot.joint_names, array(env.robot.data.joint_pos)[0], strict=True)
        )
        dq = np.array(env.load_samples[-1].dq, copy=True)
        if np.max(abs(dq)) > self.config["max_joint_speed_rad_s"]:
            raise ValueError("outside_verified_motion_speed")
        matrices = self.model.transforms(positions, io.calibration["root_world"])
        joint_travel = dict(self.config["held_joint_travel"])
        joint_travel.update(
            zip(
                ARM_JOINTS,
                abs(dq) * io.dt + 0.5 * self.acceleration * io.dt**2 + self.stop,
                strict=True,
            )
        )
        self_margin, pair = self.self_clearance.query(
            matrices, joint_travel, relative_model_error_m=self.config["robot_model_bound_m"]
        )
        diagnostics.update(self_margin_m=self_margin, closest_robot_pair=pair)
        if self_margin <= 0:
            raise ValueError("insufficient_full_robot_self_clearance")
        normal = patch.world_pose.rot[:, 0]
        ball = self.model.velocity_sweep(
            positions, io.calibration["root_world"], dq, self.acceleration, io.dt
        )
        point_candidate = (
            self.provider.tracking._constrained_candidate(candidate, candidate.local_constraint)
            if candidate.local_constraint is not None
            else candidate
        )
        point_budget = self.provider.tracking.point_budget_query(point_candidate, now)
        map_from_current = candidate.initial_world @ np.linalg.inv(pose_matrix(patch.world_pose))
        if self.background_mode and self.rigid_identity != (observation.generation, patch.patch_id):
            views = self.provider.material_views
            if not views:
                raise ValueError("missing_causal_rigid_target_inspection")
            initial_space = ObservedSpace(self.provider.config)
            for view in views:
                if float(view["available_s"]) > now:
                    raise ValueError("future_rigid_target_inspection")
                initial_space.add(view, float(view["available_s"]))
            self.rigid_workspace = CommonWorkspace([initial_space], self.config["camera_bound"])
            self.target_front_model = observed_front_extent(
                candidate.surface, views, self.provider.config
            )
            Path(self.log.name).with_suffix(".target-front-model.json").write_text(
                json.dumps(self.target_front_model) + "\n"
            )
            Path(self.log.name).with_suffix(".initial-target-sources.json").write_text(
                json.dumps(
                    dict(
                        generation=observation.generation,
                        patch_id=patch.patch_id,
                        source_clocks=self.rigid_workspace.source_clocks,
                    )
                )
                + "\n"
            )
            self.rigid_identity = observation.generation, patch.patch_id
        pending, covered = list(self.cells), []
        free_margin, separation, lever = np.inf, np.inf, 0.0
        contact_free_stop_margin = np.inf
        checked = directional = 0
        control_angle = float(abs(dq).sum() * io.dt + 0.5 * self.acceleration.sum() * io.dt**2)

        order = {}
        if self.background_mode:
            # Absolute interface offset is shared and cancels. The two real CAD
            # distal lines constrain a rigid unilateral interface; the finite
            # observed material and all static volumes are still checked below.
            age = now - patch.pose_support.supported_s
            cone = (
                patch.normal_rotation_bound_rad
                + self.interface_model["normal_model_bound_rad"]
                + self.config["material_angular_speed_rad_s"]
                * (age + io.dt + self.config["stop_duration_s"])
                + control_angle
                + self.config["robot_stop_rotation_bound_rad"]
                + self.config["camera_bound"][1]
            )
            by_link = {}
            for shape in self.self_clearance.shapes:
                if shape["moving"] is None or "FINGER" not in shape["link"]:
                    continue
                transform = matrices[shape["link"]]
                world = shape["convex"].points @ transform[:3, :3].T + transform[:3, 3]
                local = (world - tool.origin) @ tool.rot
                by_link.setdefault(shape["link"], []).append((shape["moving"], local))
            for shapes in by_link.values():
                all_points = np.concatenate([points for _, points in shapes])
                forward = all_points[:, 0].max()
                front = all_points[abs(all_points[:, 0] - forward) < 1e-6]
                for index, points in shapes:
                    triangles = points[ConvexHull(points).simplices].reshape(-1, 3)
                    order[index] = distal_body_order(
                        triangles, front, tool.rot.T @ normal, cone, 0.003
                    ) - (
                        2 * self.config["robot_model_bound_m"]
                        + self.interface_model["planarity_bound_m"]
                        + self.interface_model["indentation_bound_m"]
                    )
            diagnostics.update(interface_normal_cone_rad=cone, non_distal_order_margins_m=order)

        # Actual robot motion is covered by velocity_sweep and the stop volume.
        # This is the target's remaining motion, not another robot sweep.
        temporal_p = self.config["relative_speed_bound_m_s"] * io.dt
        temporal_a = self.config["material_angular_speed_rad_s"] * io.dt
        foreground_by_key = {}
        surface_basis = None

        def query(center, radius, key, *, constructive=True):
            nonlocal checked, surface_basis
            checked += 1
            if checked > self.config["maximum_cover_queries"]:
                diagnostics.update(
                    exhausted_cover_key=key,
                    exhausted_center=center.tolist(),
                    exhausted_radius_m=radius,
                    robot_cover_queries=checked,
                    exhausted_link=self.model.shapes[key[0]]["link"],
                )
                raise ValueError("robot_cover_query_budget_exhausted")
            local_lever = float(np.linalg.norm(center - patch.world_pose.origin) + radius)
            error = (
                patch.normal_position_bound_m
                + local_lever * patch.normal_rotation_bound_rad
                + self.config["normal_model_bound_m"]
                + temporal_p
                + local_lever * temporal_a
            )
            gap = float(
                (patch.world_pose.origin - center) @ normal + normal_range[0] - radius - error
            )
            if self.background_mode:
                # Proper registration/calibration levers start at this actual
                # sphere, rather than treating the distant material anchor as
                # the tool center. The robot/stop volume is already in radius.
                point_pose = type(tool)(center, tool.rot)
                point_error = point_budget(radius, point_pose).displacement_m
                if point_error is None:
                    raise ValueError("unmeasured_current_target_point_bound")
                # Registration and target growth are independent of the actual
                # robot volume. Target-tracking error is not a second departure
                # from that measured-state velocity/acceleration/stop volume.
                point_error += temporal_p + local_lever * temporal_a
                mapped = map_from_current[:3, :3] @ center + map_from_current[:3, 3]
                enlarged = radius + point_error
                # This complete target envelope also covers attachments, unlike
                # a plane-only leaf test. The bound is observed, then separately
                # validated on the fixed pilot domain before execution.
                model = self.target_front_model
                lower, upper = (
                    model["protrusion_tangent_lower_m"],
                    model["protrusion_tangent_upper_m"],
                )
                protrusion_margin = None
                if lower is not None and upper is not None:
                    if surface_basis is None:
                        surface_basis = candidate.surface.basis
                    tangent = (mapped @ surface_basis)[1:]
                    distance = np.linalg.norm(
                        np.maximum(
                            np.asarray(lower) - tangent, np.maximum(tangent - np.asarray(upper), 0)
                        )
                    )
                    protrusion_margin = float(distance - enlarged)
                # A one-sided Panel front bound is distinct from contact
                # planarity: recessed surfaces remain behind it. The smaller
                # bound is valid only outside the ENTIRE observed relief/handle
                # envelope, with current pose, motion and stop uncertainty paid.
                front_extent = (
                    self.interface_model["panel_front_bound_m"]
                    if protrusion_margin is not None and protrusion_margin > 0
                    else self.target_front_model["front_extent_m"]
                )
                front_gap = (
                    float((patch.world_pose.origin - center) @ normal)
                    - radius
                    - error
                    - front_extent
                )
                foreground = front_gap if front_gap > 0 else None
                if foreground is not None:
                    gap = front_gap
                current = StaticWorkspaceSupport(None, True, False, ())
                if (
                    foreground is None
                    and key[0] in order
                    and order[key[0]] > 0
                    and cover is not None
                ):
                    reserve = max(0.0, 0.02 - self.shape_stop[key[0]])
                    finite = cover.projected_ball_margin(mapped, enlarged + reserve)
                    # Body order applies to the planar interface only. Exclude
                    # the complete observed handle/relief envelope independently;
                    # its finite tangent extent includes a calibrated reserve.
                    finite_protrusion_margin = (
                        None if protrusion_margin is None else protrusion_margin - reserve
                    )
                    if (
                        finite is not None
                        and finite > 0
                        and finite_protrusion_margin is not None
                        and finite_protrusion_margin > 0
                    ):
                        foreground = min(finite, order[key[0]], finite_protrusion_margin)
                        gap = max(0.0, gap)
                if foreground is None:
                    # Each fixed initial train target is included in this
                    # common calibration. No runtime asset lookup selects it.
                    current = self.rigid_reference_workspace.query_static(
                        mapped[None],
                        np.array([enlarged]),
                        cover_ids=[("initial_target", key)],
                        constructive=False,
                    )
                    if current.collision_clearance_m is None:
                        current = self.rigid_workspace.query_static(
                            mapped[None],
                            np.array([enlarged]),
                            cover_ids=[key],
                            constructive=False,
                        )
                    foreground = current.collision_clearance_m
                    gap = 0.0 if foreground is None else float(foreground)
                foreground_by_key[key] = foreground
                if foreground is not None and foreground > 0:
                    # Prefer the already measured world-space proof. Only a
                    # volume hidden by an old moving foreground needs the
                    # larger historical-background search.
                    support = self.workspace.query_cached_background(
                        center[None],
                        np.array([radius]),
                        current_foreground_margin_m=foreground,
                    )
                    if support.collision_clearance_m is None:
                        support = self.rigid_reference_workspace.query_static(
                            center[None],
                            np.array([radius]),
                            cover_ids=[("static_world", key)],
                            constructive=False,
                        )
                    if support.collision_clearance_m is None:
                        # Initial views of THIS scene also certify stationary
                        # background in world coordinates. Only the target may
                        # move; its current exclusion has already been proved.
                        support = self.rigid_workspace.query_static(
                            center[None],
                            np.array([radius]),
                            cover_ids=[("static_world", key)],
                            constructive=False,
                        )
                    if support.collision_clearance_m is None:
                        support = self.workspace.query_background(
                            center[None],
                            np.array([radius]),
                            current_foreground_margin_m=foreground,
                            cover_ids=[key],
                            constructive=constructive and key[2] == "box",
                        )
                    elif support.collision_clearance_m is not None:
                        support = StaticWorkspaceSupport(
                            min(foreground, support.collision_clearance_m),
                            False,
                            False,
                            support.source_clocks,
                        )
                else:
                    support = StaticWorkspaceSupport(
                        None, True, current.occupied, self.workspace.source_clocks
                    )
            else:
                support = self.workspace.query_static(
                    center[None],
                    np.array([radius]),
                    cover_ids=[key],
                    constructive=constructive,
                )
            return support, gap, local_lever

        while pending:
            cell = pending.pop()
            base_center, base_radius = ball(cell)
            link = self.model.shapes[cell.shape]["link"]
            extra = self.config["robot_model_bound_m"]
            if "JAW" in link or "FINGER" in link:
                extra += self.config["closed_jaw_bound_m"]
            center, radius = base_center, base_radius + self.shape_stop[cell.shape] + extra
            support, gap, local_lever = query(center, radius, (cell.shape, cell.path, "ball"))
            proofs = None
            # The radial and directional stop certificates independently cover
            # every physical point. Prove current foreground with the radial
            # cover, and background with the complete tighter oriented box.
            foreground = foreground_by_key.get((cell.shape, cell.path, "ball"))
            if (
                support.collision_clearance_m is None
                and self.background_mode
                and self.stop_boxes is not None
                and foreground is not None
                and foreground > 0
            ):
                travel = dict(
                    zip(
                        ARM_JOINTS,
                        abs(dq) * io.dt + 0.5 * self.acceleration * io.dt**2,
                        strict=True,
                    )
                )
                _, motion_radius = self.model.world_ball(cell, matrices, travel)
                motion_radius -= np.linalg.norm(cell.upper - cell.lower) / 2
                rotation_error = (
                    2 * self.shape_stop[cell.shape] * np.sin(min(control_angle, np.pi) / 2)
                )
                lo, hi = self.stop_boxes
                corners = np.array(list(product((False, True), repeat=3)))
                local = np.where(corners, cell.upper + hi[cell.shape], cell.lower + lo[cell.shape])
                world = local @ matrices[link][:3, :3].T + matrices[link][:3, 3]
                box_support = self.workspace.query_convex_background(
                    world,
                    float(motion_radius + extra + rotation_error),
                    current_foreground_margin_m=foreground,
                )
                if box_support.collision_clearance_m is not None:
                    support = box_support
                    diagnostics["complete_box_proofs"] = (
                        diagnostics.get("complete_box_proofs", 0) + 1
                    )
            if support.collision_clearance_m is not None:
                proofs = [(support.collision_clearance_m, gap, local_lever)]
            elif np.max(cell.upper - cell.lower) > self.config["minimum_cell_m"]:
                pending.extend(self.model.subdivide(cell))
                continue
            elif self.stop_boxes is not None:
                # Both stop certificates cover the complete physical domain.
                # Their intersection is covered without shrinking either model.
                # Stop axes belong to stop onset, AFTER this control interval.
                # Spend all possible intervening link rotation explicitly.
                rotation_reserve = (
                    2 * self.shape_stop[cell.shape] * np.sin(min(control_angle, np.pi) / 2)
                )
                proofs = []

                def certify(
                    offset,
                    stop_radius,
                    path,
                    *,
                    base_center=base_center,
                    base_radius=base_radius,
                    link=link,
                    extra=extra,
                    rotation_reserve=rotation_reserve,
                    cell=cell,
                    proofs=proofs,
                ):
                    nonlocal center, radius, support, directional
                    center = base_center + matrices[link][:3, :3] @ offset
                    radius = base_radius + stop_radius + extra + rotation_reserve
                    support, gap, local_lever = query(
                        center,
                        radius,
                        (cell.shape, cell.path, "box", path),
                        constructive=stop_radius <= self.config["stop_cover_radius_m"],
                    )
                    directional += 1
                    if support.collision_clearance_m is None:
                        return False
                    proofs.append((support.collision_clearance_m, gap, local_lever))
                    return True

                lower, upper = self.stop_boxes
                complete = stop_box_cover(
                    lower[cell.shape],
                    upper[cell.shape],
                    self.shape_stop[cell.shape],
                    self.config["stop_cover_radius_m"],
                    certify=certify,
                )
                if complete is None:
                    proofs = None
            if proofs is None:
                diagnostics.update(
                    unsupported_link=link,
                    unsupported_shape=cell.shape,
                    unsupported_cell_lower_m=cell.lower.tolist(),
                    unsupported_cell_upper_m=cell.upper.tolist(),
                    unsupported_base_center=base_center.tolist(),
                    unsupported_base_radius_m=base_radius,
                    unsupported_center=center.tolist(),
                    unsupported_radius_m=radius,
                    occupied=support.occupied,
                    unknown=support.unknown_in_envelope,
                    robot_cover_queries=checked,
                    directional_stop_queries=directional,
                )
                raise ValueError("unsupported_actual_robot_sweep_and_stop")
            for margin, gap, local_lever in proofs:
                free_margin = min(free_margin, margin)
                separation = min(separation, gap)
                # Robot stopping travel is already inside each proved volume.
                # Also pay target translation/normal growth through the entire
                # stop, beyond the control-interval growth paid in query().
                contact_free_stop_margin = min(
                    contact_free_stop_margin,
                    gap
                    - self.config["stop_duration_s"]
                    * (
                        self.config["relative_speed_bound_m_s"]
                        + local_lever * self.config["material_angular_speed_rad_s"]
                    ),
                )
                lever = max(lever, local_lever)
            covered.append(cell)
        self.cells = tuple(covered)
        diagnostics.update(
            robot_cells=len(covered),
            robot_cover_queries=checked,
            directional_stop_queries=directional,
            sweep_margin_m=float(free_margin),
            current_material_separation_margin_m=float(separation),
        )
        diagnostics["contact_free_stop_margin_m"] = float(contact_free_stop_margin)
        diagnostics["future_stop_force_domain_n"] = self.config["stop_force_domain_n"]
        # Apply position support before entry, not only after observing
        # an envelope exit. These are the same joint travel bounds used by the
        # independently checked full robot sweep and stopping volumes above.
        motion_q_lower = np.array([positions[name] - joint_travel[name] for name in ARM_JOINTS])
        motion_q_upper = np.array([positions[name] + joint_travel[name] for name in ARM_JOINTS])
        try:
            self.load_calibration.require_position_motion_support(
                motion_q_lower,
                contact_free_stop_margin_m=contact_free_stop_margin,
                motion_q_upper=motion_q_upper,
                force_upper_n=self.config["stop_force_domain_n"],
            )
        except ValueError:
            diagnostics.update(
                motion_joint_lower_rad=motion_q_lower.tolist(),
                motion_joint_upper_rad=motion_q_upper.tolist(),
                main_load_q_min_rad=list(self.load_calibration.q_min),
                main_load_q_max_rad=list(self.load_calibration.q_max),
                position_tolerance_rad=self.load_calibration.position_tolerance_rad,
                joint_stop_bound_rad=self.stop.tolist(),
                motion_acceleration_bound_rad_s2=self.acceleration.tolist(),
                required_contact_free_stop_margin_m=0.0,
                numerical_tolerance_m=0.0,
            )
            raise
        loaded = bool(separation <= 0)
        support = FieldSupport(now, now, now, observation.generation, *self.config["camera_bound"])
        limits = ActionLimits(
            io.dt, 0.02, 1.2, 3.0, 0.013, 0.04, self.config["relative_speed_bound_m_s"], 0.02
        )
        fingers = (None, None)
        if loaded:
            surface = candidate.local_surface or candidate.surface
            clearances = []
            for pose in poses:
                projected = type(tool)(
                    pose.origin + normal * ((patch.world_pose.origin - pose.origin) @ normal),
                    pose.rot,
                )
                original = candidate.initial_world @ np.linalg.inv(pose_matrix(patch.world_pose))
                original = original @ pose_matrix(projected)
                clearances.append(
                    (
                        cover.query(object_frame(original), self.faces)
                        if cover is not None
                        else footprint_support(
                            surface, object_frame(original), self.faces, self.provider.config
                        )
                    ).finger_clearance_m
                )
            fingers = tuple(
                None if any(c[j] is None for c in clearances) else min(c[j] for c in clearances)
                for j in range(2)
            )
        proposal = ActionProposal(
            "policy",
            self.space,
            np.asarray(row).copy(),
            np.array([now, now + io.dt]),
            poses,
            observation.estimate.local.selection.selection_id,
            loaded,
        )
        evidence = LocalActionSupport(
            support,
            poses,
            fingers,
            None,
            False,
            # Material motion over this interval uses its existing measured
            # rate envelope. The 13 mm / 0.04 rad limits above describe ROBOT
            # tracking response and are not a second material response source.
            self.config["relative_speed_bound_m_s"] * io.dt,
            self.config["material_angular_speed_rad_s"] * io.dt,
            max(np.linalg.norm(f, axis=1).max() for f in self.faces),
            geometry_position_prepaid_m=self.provider.config["plane_tolerance_m"],
            swept_collision_margin_m=min(float(free_margin), self_margin),
            unloaded_separation_margin_m=None if loaded else float(separation),
            unloaded_normal_lever_m=None if loaded else lever,
            unloaded_temporal_bound_m_rad=None if loaded else (temporal_p, temporal_a),
            material_normal_range_m=normal_range,
        )
        return admit_local_action(
            observation.estimate, proposal, evidence, limits, now, generation=observation.generation
        )


def loaded_observation_recipe(recipe, preroll_s):
    """Schedule genuine loaded-start captures; leave workspace calibration alone."""
    from alexdoor_xas.perception.provider import PrototypeRecipe

    if preroll_s is None:
        return recipe
    config = recipe.config
    count = len(config["inspection"]["sample_times_s"])
    first, last = 0.1, float(preroll_s) - 0.2
    if count < 2 or not np.isfinite(last) or last - first < (count - 1) * 0.1 - 1e-9:
        raise ValueError("Loaded pre-roll needs all inspection captures plus fresh tracking")
    neck = config["inspection"]["waypoints"][-1][1:]
    config["inspection"]["waypoints"] = [[0.0, *neck], [float(preroll_s), *neck]]
    config["inspection"]["sample_times_s"] = np.linspace(first, last, count).tolist()
    return PrototypeRecipe(json.dumps(config, sort_keys=True, allow_nan=False))


def live_observation_binding(binding, recipe):
    """Bind actual acquisition; the independent workspace retains its own recipe."""
    from alexdoor_xas.policies.common.b1_contract import PilotBinding

    inspection = recipe.config["inspection"]
    if inspection == binding.config["inspection"]:
        return binding
    if not isinstance(binding, PilotBinding):
        raise ValueError("Changed acquisition requires an explicit pilot binding")
    original = binding.config["inspection"]
    acquisition_keys = {"waypoints", "sample_times_s"}
    if {k: v for k, v in original.items() if k not in acquisition_keys} != {
        k: v for k, v in inspection.items() if k not in acquisition_keys
    }:
        raise ValueError("Loaded acquisition cannot change inspection calibration")
    payload = binding.to_dict()
    payload["config"]["inspection"] = inspection
    return PilotBinding.from_dict(payload)


def observed_pilot_runtime(
    env, binding, robot_asset, setup, config_path, output, action_space, *, initialization_s=None
):
    """Build real sensing/load/contact/control inputs from a measured recipe."""
    from alexdoor_xas.perception.contact import ContactMonitorConfig, ObservedContactMonitor
    from alexdoor_xas.perception.load import LoadCalibration, PurdueLoadSource
    from alexdoor_xas.perception.provider import load_recipe
    from alexdoor_xas.perception.session import session_provider
    from alexdoor_xas.policies.encoding import RGBDEncoder
    from alexdoor_xas.policies.observations import B1Observer
    from alexdoor_xas.policies.purdue import PurdueFK, PurdueIO
    from alexdoor_xas.recording.b1_runtime import camera_calibration

    config = json.loads(Path(config_path).read_text())
    if "stop_stiffness" in config and not np.array_equal(
        config["stop_stiffness"], env.cfg.stop_stiffness
    ):
        raise ValueError("stop_controller_calibration_mismatch")
    calibration = camera_calibration(env)
    recipe = load_recipe(config["geometry_recipe"], Path(__file__).resolve().parents[3])
    load_calibration = LoadCalibration(**json.loads(Path(config["load_calibration"]).read_text()))
    if load_calibration.verified is not True or load_calibration.stop_verified is not True:
        raise ValueError("unverified_robot_load_or_stop_recipe")
    robot = RobotGeometry(env.robot.cfg.spawn.asset_path, ARM_JOINTS)
    fk = PurdueFK(env.robot.cfg.spawn.asset_path, str(env.device))
    faces = derive_push_geometry(env.robot.cfg.spawn.asset_path).contact_covers()

    def exclude_robot(sensor, raw):
        if float(sensor["time_s"]) != float(env.capture.sample.time_s):
            raise ValueError("unsynchronized_full_robot_geometry_capture")
        positions = dict(
            zip(env.robot.joint_names, array(env.robot.data.joint_pos)[0], strict=True)
        )
        return robot.occluded_pixels(
            dict(sensor, depth_m=raw["depth_m"], valid_depth=raw["valid_depth"]),
            positions,
            calibration["root_world"],
            config["camera_bound"],
        )

    live_recipe = loaded_observation_recipe(recipe, initialization_s)
    live_binding = (
        binding if initialization_s is None else live_observation_binding(binding, live_recipe)
    )
    if initialization_s is not None:
        (Path(output) / "observed-initialization.json").write_text(
            json.dumps(
                dict(
                    scope="fresh_loaded_state_acquisition_only",
                    workspace_sample_times_s=recipe.config["inspection"]["sample_times_s"],
                    live_sample_times_s=live_recipe.config["inspection"]["sample_times_s"],
                    live_waypoints=live_recipe.config["inspection"]["waypoints"],
                    limits_and_calibration_unchanged=True,
                ),
                indent=2,
            )
            + "\n"
        )
    geometry = session_provider(
        live_recipe,
        config["perception_models"],
        calibration,
        Path(output) / "perception",
        replay=False,
        clock=lambda: float(env.capture.sample.time_s),
        synchronous_simulation=True,
        inference_period_s=0.1,
        inspection_samples_only=True,
        preload=False,
        # Single-run CLI: clear state without loading an unused next episode.
        preload_on_reset=False,
        tool_fk=fk,
        distal_faces=faces,
        calibration_bound=tuple(config["camera_bound"]),
        relative_speed_bound=(
            config["relative_speed_bound_m_s"],
            config["material_angular_speed_rad_s"],
        ),
        geometry_filter=exclude_robot,
        local_selection_source="policy",
    )
    observer = B1Observer(geometry, live_binding, encoder=RGBDEncoder(live_binding))
    contact = ObservedContactMonitor(ContactMonitorConfig(**config["contact_monitor"]), faces)
    measured_load = PurdueLoadSource(
        load_calibration, normal_angular_speed_rad_s=config["material_angular_speed_rad_s"]
    )
    load_log = (Path(output) / "operational-load.jsonl").open("x")

    def load_source(io, feedback):
        try:
            load = measured_load(io, feedback)
        except ValueError as error:
            load_log.write(
                json.dumps(
                    dict(
                        time_s=feedback.timestamp_s,
                        reason=str(error),
                        decision=getattr(error, "decision", None),
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
            load_log.flush()
            raise
        load_log.write(
            json.dumps(
                dict(
                    time_s=feedback.timestamp_s,
                    available_s=feedback.available_s,
                    force_upper_n=load.force_upper_n,
                    normal_lower_n=load.normal_lower_n,
                    normal_samples=load.normal_samples,
                    position_halo=load.position_halo,
                    stop_force_limit_n=config["stop_force_domain_n"],
                    model_verified=load.model_verified,
                    stop_verified=load.stop_verified,
                )
            )
            + "\n"
        )
        load_log.flush()
        if load.force_upper_n > config["stop_force_domain_n"]:
            raise ValueError("outside_verified_robot_stop_load_domain")
        return load

    io = PurdueIO(
        env,
        binding=live_binding,
        robot_asset=robot_asset,
        setup=setup,
        observed_provider=geometry,
        load_source=load_source,
        contact_monitor=contact,
    )
    from dataclasses import asdict

    contact_log = (Path(output) / "operational-contact.jsonl").open("x")

    def record_contact(evidence):
        contact_log.write(json.dumps(asdict(evidence), allow_nan=False) + "\n")
        contact_log.flush()

    io.safety.checks.on_contact_evidence = record_contact
    motion = PilotMotionEvidence(
        io,
        geometry,
        action_space,
        recipe,
        config,
        Path(output) / "operational-admission.jsonl",
        load_calibration=load_calibration,
    )
    return observer, io, geometry, motion, load_log, contact_log
