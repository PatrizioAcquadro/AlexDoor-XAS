"""Observed A1/A2 motion evidence from actual robot state and measured workcell rays."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from alexdoor_xas.action.b1 import apply_pose_delta
from alexdoor_xas.action.spaces import A1_JOINT_DELTA
from alexdoor_xas.assets.purdue import ARM_JOINTS, derive_push_geometry
from alexdoor_xas.kinematics.self_clearance import RobotSelfClearance
from alexdoor_xas.kinematics.urdf_geometry import RobotGeometry
from alexdoor_xas.perception.admission import ActionAdmission, ActionLimits, ActionProposal
from alexdoor_xas.perception.contracts import FieldSupport, validate_local_contact
from alexdoor_xas.perception.evaluation import json_safe
from alexdoor_xas.perception.local_admission import LocalActionSupport, admit_local_action
from alexdoor_xas.perception.material_zone import object_frame, pose_matrix
from alexdoor_xas.perception.scan import footprint_support
from alexdoor_xas.perception.workspace import CommonWorkspace
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

    def __init__(self, io, provider, space, recipe, config, output):
        self.io, self.provider, self.space = io, provider, space
        self.config = config
        self.model = RobotGeometry(io.env.robot.cfg.spawn.asset_path, ARM_JOINTS)
        self.self_clearance = RobotSelfClearance(self.model, BODY_FILTERS)
        self.faces = derive_push_geometry(io.env.robot.cfg.spawn.asset_path).contact_covers()
        self.workspace = CommonWorkspace.from_recordings(
            config["workspace_recordings"], recipe.config, config["camera_bound"]
        )
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
        self.cells = self.model.cells()
        self.log = Path(output).open("x")
        self.last = None

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
            admitted=result.admitted, reason=result.reason, wall_s=perf_counter() - started
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
        pending, covered = list(self.cells), []
        free_margin, separation, lever = np.inf, np.inf, 0.0
        checked = 0
        while pending:
            cell = pending.pop()
            center, radius = ball(cell)
            # Independent whole-link continuous stop calibration has already
            # covered every bounding-box corner, with held-jaw travel separate.
            radius += self.shape_stop[cell.shape]
            if (
                "JAW" in self.model.shapes[cell.shape]["link"]
                or "FINGER" in self.model.shapes[cell.shape]["link"]
            ):
                radius += self.config["closed_jaw_bound_m"]
            radius += self.config["robot_model_bound_m"]
            local_lever = float(np.linalg.norm(center - patch.world_pose.origin) + radius)
            normal_error = (
                patch.normal_position_bound_m
                + local_lever * patch.normal_rotation_bound_rad
                + self.config["normal_model_bound_m"]
                + self.config["relative_speed_bound_m_s"] * io.dt
            )
            cell_separation = float(
                (patch.world_pose.origin - center) @ normal
                + normal_range[0]
                - radius
                - normal_error
            )
            support = self.workspace.query_static(
                center[None], np.array([radius]), cover_ids=[(cell.shape, cell.path)]
            )
            checked += 1
            if support.collision_clearance_m is not None:
                lever = max(lever, local_lever)
                separation = min(separation, cell_separation)
                free_margin = min(free_margin, support.collision_clearance_m)
                covered.append(cell)
                continue
            if np.max(cell.upper - cell.lower) <= self.config["minimum_cell_m"]:
                diagnostics.update(
                    unsupported_link=self.model.shapes[cell.shape]["link"],
                    unsupported_center=center.tolist(),
                    unsupported_radius_m=radius,
                    occupied=support.occupied,
                    unknown=support.unknown_in_envelope,
                )
                raise ValueError("unsupported_actual_robot_sweep_and_stop")
            if checked > self.config["maximum_cover_queries"]:
                raise ValueError("robot_cover_query_budget_exhausted")
            pending.extend(self.model.subdivide(cell))
        self.cells = tuple(covered)
        diagnostics.update(
            robot_cells=checked,
            sweep_margin_m=float(free_margin),
            current_material_separation_margin_m=float(separation),
        )
        loaded = separation <= 0
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
            0.013,
            0.04,
            max(np.linalg.norm(f, axis=1).max() for f in self.faces),
            swept_collision_margin_m=min(float(free_margin), self_margin),
            unloaded_separation_margin_m=None if loaded else float(separation),
            unloaded_normal_lever_m=None if loaded else lever,
            material_normal_range_m=normal_range,
        )
        return admit_local_action(
            observation.estimate, proposal, evidence, limits, now, generation=observation.generation
        )


def observed_pilot_runtime(env, binding, robot_asset, setup, config_path, output, action_space):
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
    calibration = camera_calibration(env)
    recipe = load_recipe(config["geometry_recipe"], Path(__file__).resolve().parents[3])
    load_calibration = LoadCalibration(**json.loads(Path(config["load_calibration"]).read_text()))
    if load_calibration.verified is not True or load_calibration.stop_verified is not True:
        raise ValueError("unverified_robot_load_or_stop_recipe")
    robot = RobotGeometry(env.robot.cfg.spawn.asset_path, ARM_JOINTS)
    fk = PurdueFK(env.robot.cfg.spawn.asset_path, str(env.device))
    faces = derive_push_geometry(env.robot.cfg.spawn.asset_path).contact_covers()

    def exclude_robot(sensor, raw):
        del raw
        if float(sensor["time_s"]) != float(env.capture.sample.time_s):
            raise ValueError("unsynchronized_full_robot_geometry_capture")
        positions = dict(
            zip(env.robot.joint_names, array(env.robot.data.joint_pos)[0], strict=True)
        )
        return robot.occluded_pixels(
            sensor, positions, calibration["root_world"], config["camera_bound"]
        )

    geometry = session_provider(
        recipe,
        config["perception_models"],
        calibration,
        Path(output) / "perception",
        replay=False,
        clock=lambda: float(env.capture.sample.time_s),
        synchronous_simulation=True,
        inference_period_s=0.1,
        inspection_samples_only=True,
        preload=False,
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
    observer = B1Observer(geometry, binding, encoder=RGBDEncoder(binding))
    contact = ObservedContactMonitor(ContactMonitorConfig(**config["contact_monitor"]), faces)
    measured_load = PurdueLoadSource(
        load_calibration, normal_angular_speed_rad_s=config["material_angular_speed_rad_s"]
    )
    load_log = (Path(output) / "operational-load.jsonl").open("x")

    def load_source(io, feedback):
        load = measured_load(io, feedback)
        load_log.write(
            json.dumps(
                dict(
                    time_s=feedback.timestamp_s,
                    available_s=feedback.available_s,
                    force_upper_n=load.force_upper_n,
                    normal_lower_n=load.normal_lower_n,
                    normal_samples=load.normal_samples,
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
        binding=binding,
        robot_asset=robot_asset,
        setup=setup,
        observed_provider=geometry,
        load_source=load_source,
        contact_monitor=contact,
    )
    motion = PilotMotionEvidence(
        io, geometry, action_space, recipe, config, Path(output) / "operational-admission.jsonl"
    )
    return observer, io, geometry, motion, load_log
