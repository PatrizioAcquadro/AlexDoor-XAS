"""Single fixed-base Purdue runtime with full-pose control and observed RGB-D."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from ihmc_alex_isaaclab.end_effectors.weiss_wsg32 import alex_purdue_wsg32_targets
from ihmc_alex_isaaclab.platforms.purdue_alex003_pedestal import (
    load_purdue_alex003_pedestal_spec,
    make_purdue_alex003_pedestal_cfg,
)
from ihmc_alex_isaaclab.sensors.zed_x_mini import (
    author_alex_purdue_zed_x_mini_mount,
    make_zed_x_mini_cfgs,
)
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera
from isaaclab.sim import schemas
from isaaclab.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
)
from pxr import Usd, UsdGeom, UsdPhysics

from alexdoor_xas.action.frames import frame_delta_to_world, validate_object_frame
from alexdoor_xas.assets.purdue import ARM_JOINTS, NECK_JOINTS, derive_push_geometry
from alexdoor_xas.kinematics.point_jacobian import link_jacobian_to_point
from alexdoor_xas.kinematics.pose_control import (
    bounded_joint_step,
    bounded_pose_step,
    normal_reference_transport,
    pose_target_limits,
    tool_frame_elastic_reference,
)
from alexdoor_xas.kinematics.posture import (
    coupled_margin_preference,
    joint_command_position_response,
)
from alexdoor_xas.recording.rgbd import RGBDCapture
from isaac_physics_profile.camera import SteelMountedCamera

from .physics import configure_purdue_physics
from .purdue_contacts import ContactObserver

ROOT = "/World/envs/env_0"
ROBOT = ROOT + "/Robot"
PANEL = ROOT + "/Panel"
PEDESTAL = ROOT + "/Pedestal"


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


class DoorPushPurdueEnv(DirectRLEnv):
    """A1 joint deltas or A2 world pose deltas; A3 uses an explicitly supplied frame."""

    def __init__(self, cfg, **kwargs):
        if cfg.scene.num_envs != 1 or cfg.sim.device == "cpu":
            raise ValueError("Purdue commissioning requires one GPU environment")
        if cfg.action_mode not in ("A1", "A2"):
            raise ValueError("action_mode must be A1 or A2")
        if cfg.synthetic_door is not None and cfg.prepared_door is not None:
            raise ValueError("Select either a synthetic or a prepared door")
        cfg.action_space = 7 if cfg.action_mode == "A1" else 6
        self.event_sink = None
        self.generation = -1
        self.command_event = None
        self.capture = None
        self.contacts = None
        self._time_s = 0.0
        self._pending_pose = None
        self._pending_normal_reference_only = False
        self._pending_joints = None
        self._last_world_goal = None
        self._last_normal_jacobian = None
        self._held_surface = None

        self._finite_tangent_tool_reference = None
        self.motion_stopped = False
        self.stop_response = None
        self.stop_response_receipt = None
        stop_stiffness = np.asarray(cfg.stop_stiffness, dtype=float)
        if (
            stop_stiffness.shape != (7,)
            or not np.isfinite(stop_stiffness).all()
            or (stop_stiffness <= 0).any()
        ):
            raise ValueError("invalid_stop_impedance")
        if cfg.pose_reference_feedforward and cfg.pose_feedback_time_constant_s is None:
            raise ValueError("pose_reference_requires_feedback")
        if type(cfg.pose_normal_compliance) is not bool:
            raise ValueError("invalid_pose_normal_compliance")
        if cfg.pose_normal_compliance and cfg.pose_feedback_time_constant_s is None:
            raise ValueError("compliant_normal_requires_pose_feedback")
        if type(cfg.pose_normal_transport) is not bool:
            raise ValueError("invalid_pose_normal_transport")
        if cfg.pose_normal_transport and (
            cfg.pose_feedback_time_constant_s is None or not cfg.pose_reference_feedforward
        ):
            raise ValueError("normal_transport_requires_causal_pose_feedback")
        if type(cfg.pose_finite_tangent_impedance) is not bool:
            raise ValueError("invalid_finite_tangent_impedance")
        if cfg.pose_finite_tangent_impedance and (
            cfg.pose_feedback_time_constant_s is None or not cfg.pose_reference_feedforward
        ):
            raise ValueError("finite_tangent_requires_causal_pose_feedback")
        if (cfg.pose_posture_support_rad is None) != (
            cfg.pose_posture_stopping_reserve_rad is None
        ):
            raise ValueError("posture_support_requires_stopping_reserve")
        if cfg.pose_posture_support_rad is not None and cfg.pose_centering_domain_rad is not None:
            raise ValueError("select_one_posture_preference")
        self._motion_impedance = []
        self._native_arm_impedance = None
        self._mounted_camera = None
        self._native_contacts = None
        with configure_purdue_physics(cfg) as self._steel:
            super().__init__(cfg, **kwargs)
        if cfg.native_steel:
            from .native_steel import NativeSteelPurdueViews

            self.sim.backend_manager.set_decimation(1)
            self._physics_handles_decimation = False
            self._native_contacts = NativeSteelPurdueViews(self.sim, self.panel_path)
        self.arm_ids = self._resolve_joints(ARM_JOINTS)
        self.steel_joint_load = None
        if self._steel and cfg.observed_load_feedback:
            from isaac_physics_profile.feedback import SteelJointLoad

            self.steel_joint_load = SteelJointLoad(self.sim.physics_manager, self.robot.data)
        self.neck_ids = self._resolve_joints(NECK_JOINTS)
        self.observed_ids = self.arm_ids + self.neck_ids
        self.targets = tensor(self.robot.data.default_joint_pos).clone()
        # Preserve the task impedance across reset; only a latched stop changes it.
        if cfg.native_steel:
            from .native_steel import NativeArmImpedance

            self._native_arm_impedance = NativeArmImpedance(self.robot, ARM_JOINTS)
            self._normal_transport_stiffness = self._native_arm_impedance.stiffness
        else:
            for actuator in self.robot.actuators.values():
                for name in ARM_JOINTS:
                    if name not in actuator.joint_names:
                        continue
                    column = actuator.joint_names.index(name)
                    self._motion_impedance.append(
                        (
                            actuator,
                            column,
                            actuator.stiffness[:, column].clone(),
                            actuator.damping[:, column].clone(),
                        )
                    )
            if len(self._motion_impedance) != len(ARM_JOINTS):
                raise RuntimeError("stop impedance joints do not resolve uniquely")
            motion_stiffness = {
                actuator.joint_names[column]: stiffness
                for actuator, column, stiffness, _ in self._motion_impedance
            }
            self._normal_transport_stiffness = torch.stack(
                [motion_stiffness[name] for name in ARM_JOINTS], dim=-1
            )
        self.limits = tensor(self.robot.data.joint_pos_limits)[:, self.arm_ids].clone()
        self.pose_limits = pose_target_limits(
            self.limits, cfg.pose_target_margin_rad, domain=cfg.pose_target_domain_rad
        )
        self.pose_centering_limits = (
            None
            if cfg.pose_centering_domain_rad is None
            else pose_target_limits(self.pose_limits, domain=cfg.pose_centering_domain_rad)
        )
        self.pose_posture_support = None
        if cfg.pose_posture_support_rad is not None:
            self.pose_posture_support = pose_target_limits(
                self.limits.to(torch.float64), domain=cfg.pose_posture_support_rad
            )
            self._posture_stopping_reserve = torch.as_tensor(
                cfg.pose_posture_stopping_reserve_rad, device=self.device, dtype=torch.float64
            )
            if (
                self._posture_stopping_reserve.shape != (7,)
                or not bool(torch.isfinite(self._posture_stopping_reserve).all())
                or bool((self._posture_stopping_reserve < 0).any())
            ):
                raise ValueError("invalid_posture_stopping_reserve")
            names = self.robot.joint_names
            followers = {row[0] for row in self._posture_mimic_relations}
            free = [name for name in names if name not in followers]
            self._posture_basis = torch.zeros(
                len(names), len(free), device=self.device, dtype=torch.float64
            )
            self._posture_passive_damping = torch.zeros(
                len(names), device=self.device, dtype=torch.float64
            )
            for column, name in enumerate(free):
                self._posture_basis[names.index(name), column] = 1
            for follower, driver, multiplier, damping in self._posture_mimic_relations:
                if driver not in free:
                    raise ValueError("unsupported_chained_posture_mimic")
                self._posture_basis[names.index(follower), free.index(driver)] = multiplier
                self._posture_passive_damping[names.index(follower)] = damping
            self._posture_stiffness = torch.zeros_like(self._posture_passive_damping)
            self._posture_damping = torch.zeros_like(self._posture_passive_damping)
            for actuator in self.robot.actuators.values():
                for column, name in enumerate(actuator.joint_names):
                    self._posture_stiffness[names.index(name)] = tensor(actuator.stiffness)[
                        0, column
                    ]
                    self._posture_damping[names.index(name)] = tensor(actuator.damping)[0, column]
        self.velocities = tensor(self.robot.data.joint_vel_limits)[:, self.arm_ids].clone()
        # Target-rate clipping alone permits tracking overshoot: the pilot
        # measured 0.5183 rad/s under a 0.5 rad/s reference. Retain 10% reference
        # headroom; the independent actual-speed criterion is unchanged.
        self.velocities.clamp_(
            max=cfg.max_joint_speed * (0.9 if cfg.observed_load_feedback else 1.0)
        )
        self.clamp_excess = torch.zeros((1, 7), device=self.device)
        self.pose_error = torch.zeros((1, 6), device=self.device)
        contact_view = self._native_contacts
        if self._steel:
            contact_view = self.sim.physics_manager.contact_sim_view
        self.contacts = ContactObserver(
            self.sim,
            self.sim.stage,
            ROBOT,
            self.panel_path,
            PEDESTAL,
            "/World/Ground/CollisionPlane",
            self.push_geometry.wrist_from_base,
            physics_view=contact_view,
        )
        if cfg.cameras:
            if self._steel:
                self._mounted_camera = SteelMountedCamera(
                    self.sim, self.robot, self.camera, ROOT + "/Zed"
                )
                self._mounted_camera.bind()
            camera_prim = self.sim.stage.GetPrimAtPath(self.camera.cfg.prim_path)
            near, far = UsdGeom.Camera(camera_prim).GetClippingRangeAttr().Get()
            self.capture = RGBDCapture(float(near), float(far))
        self.contact_history = []
        if cfg.pose_held_surface:
            from alexdoor_xas.kinematics.held_surface import HeldSurfaceController

            self._held_surface = HeldSurfaceController(self)

    def _resolve_joints(self, names):
        indices, found = self.robot.find_joints(list(names), preserve_order=True)
        if tuple(found) != tuple(names) or len(set(indices)) != len(names):
            raise RuntimeError(f"Joint order does not resolve: {names}")
        return indices

    def _setup_scene(self):
        import isaaclab.sim as sim_utils

        spec = load_purdue_alex003_pedestal_spec()
        cache = (
            Path(self.cfg.urdf_cache_dir).expanduser().resolve()
            if self.cfg.urdf_cache_dir
            else None
        )
        if self.cfg.native_steel:
            from isaaclab import cloner
            from isaaclab.assets import AssetBaseCfg

            from .native_steel import PEDESTAL_ASSET, converted_usd, robot_config

            paths = (PEDESTAL, ROBOT, ROOT + "/Door", "/World/Ground", "/World/Light")
            if self.cfg.cameras:
                paths += (ROOT + "/Zed",)
            plan = cloner.clone_plan_from_env_0(
                self.scene.cloner_cfg,
                [AssetBaseCfg(prim_path=path) for path in paths],
                self.cfg.scene.num_envs,
                self.cfg.scene.env_spacing,
            )
            pedestal_spawn = converted_usd(cache, "pedestal", PEDESTAL_ASSET)
            robot_cfg, self.robot_urdf_path = robot_config(cache)
        else:
            from ihmc_alex_isaaclab.robots.alex_purdue import make_alex_purdue_cfg

            pedestal_spawn = make_purdue_alex003_pedestal_cfg(PEDESTAL).spawn
            robot_cfg = make_alex_purdue_cfg(
                fix_base=True, variant="full_convex", end_effector="wsg32_umi_v1"
            )
            if cache is not None:
                pedestal_spawn.usd_dir = str(cache / "pedestal")
                robot_cfg.spawn.usd_dir = str(cache / "robot")
            self.robot_urdf_path = Path(robot_cfg.spawn.asset_path)
        x, y, yaw = self.cfg.floor_pose
        floor_quat = (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))
        pedestal_spawn.func(
            PEDESTAL, pedestal_spawn, translation=(x, y, 0.0), orientation=floor_quat
        )
        schemas.activate_contact_sensors(PEDESTAL)
        if self.cfg.observed_load_feedback:
            # Task-local impedance; keep the shared Alex package and physical limits intact.
            # At the ready pose, 4 N reflects about 1.7/0.7/0.7/0.22 N m into
            # shoulder Y/Z, elbow and terminal Y. Allocate about 0.005 rad static
            # deflection to those joints; shoulder X already meets that budget.
            arm = robot_cfg.actuators["arms"]
            for key, selected in (
                ("stiffness", (320.0, 26.78, 160.0, 160.0, 20.0, 10.0, 50.0)),
                ("damping", (8.0, 8.0, 4.0, 4.0, 1.0, 1.0, 1.0)),
            ):
                original = getattr(arm, key)
                values = {
                    f"{side}_{pattern.removeprefix('.*')}": value
                    for side in ("LEFT", "RIGHT")
                    for pattern, value in original.items()
                }
                values.update(zip(ARM_JOINTS, selected, strict=True))
                setattr(arm, key, values)
        robot_cfg.prim_path = ROBOT
        robot_cfg.init_state.pos = (x, y, spec.alex_root_world_z_m)
        robot_cfg.init_state.rot = floor_quat
        joints = ET.parse(self.robot_urdf_path).getroot().findall("joint")
        if self.cfg.pose_posture_support_rad is not None:
            self._posture_mimic_relations = [
                (
                    joint.get("name"),
                    joint.find("mimic").get("joint"),
                    float(joint.find("mimic").get("multiplier", "1")),
                    float(joint.find("dynamics").get("damping", "0"))
                    if joint.find("dynamics") is not None
                    else 0.0,
                )
                for joint in joints
                if joint.find("mimic") is not None
            ]

        defaults = {j.get("name"): 0.0 for j in joints if j.get("type") != "fixed"}
        robot_cfg.init_state.joint_pos = {
            **defaults,
            **self.cfg.initial_joints,
            **alex_purdue_wsg32_targets(0.0, "left"),
            **alex_purdue_wsg32_targets(0.0, "right"),
        }
        self.push_geometry = derive_push_geometry(self.robot_urdf_path)
        self.robot = Articulation(robot_cfg)
        if self.cfg.native_steel:
            from ihmc_alex_isaaclab.end_effectors.weiss_wsg32 import (
                author_wsg32_finger_physics_material,
            )
            from ihmc_alex_isaaclab.robots.purdue_physics import (
                ALEX_PURDUE_WSG32_SELF_COLLISION_FILTER_PAIRS,
                author_alex_purdue_self_collision_filters,
            )

            author_alex_purdue_self_collision_filters(
                self.sim.stage, ROBOT, ALEX_PURDUE_WSG32_SELF_COLLISION_FILTER_PAIRS
            )
            author_wsg32_finger_physics_material(self.sim.stage, ROBOT)
        for prim in Usd.PrimRange(self.sim.stage.GetPrimAtPath(ROBOT)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                schemas.activate_contact_sensors(str(prim.GetPath()))
        self.scene.articulations["robot"] = self.robot
        sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
        light = sim_utils.DomeLightCfg(intensity=1800)
        light.func("/World/Light", light)
        self.door = None
        self.panel_path = PANEL
        if self.cfg.synthetic_door is not None or self.cfg.prepared_door is not None:
            from isaaclab.actuators import ImplicitActuatorCfg
            from isaaclab.assets import ArticulationCfg

            from alexdoor_xas.assets.synthetic_door import author_synthetic_door

            root = ROOT + "/Door"
            geometry = self.cfg.prepared_door or self.cfg.synthetic_door
            if self.cfg.prepared_door is not None:
                spawn = sim_utils.UsdFileCfg(usd_path=str(geometry.usd))
                spawn.func(root, spawn)
                if self.cfg.door_mass_scale != 1.0 or self.cfg.door_hinge_damping is not None:
                    from alexdoor_xas.qualification.collection_dynamics import author_door_dynamics

                    self.door_dynamics_override = author_door_dynamics(
                        self.sim.stage,
                        root,
                        mass_scale=self.cfg.door_mass_scale,
                        hinge_damping=self.cfg.door_hinge_damping,
                    )
                if self.cfg.diagnostic_hinge_stop_deg is not None:
                    joint = UsdPhysics.RevoluteJoint(self.sim.stage.GetPrimAtPath(root + "/Hinge"))
                    joint.GetUpperLimitAttr().Set(self.cfg.diagnostic_hinge_stop_deg)
                geometry.load_stage(self.sim.stage, root)
            else:
                author_synthetic_door(self.sim.stage, root, geometry)
            schemas.activate_contact_sensors(root)
            self.panel_path = root + "/Panel"
            self.door = Articulation(
                ArticulationCfg(
                    prim_path=root,
                    spawn=None,
                    actuators={
                        "hinge": ImplicitActuatorCfg(
                            joint_names_expr=["Hinge"],
                            stiffness=0.0,
                            damping=geometry.damping,
                        )
                    },
                )
            )
            self.scene.articulations["door"] = self.door
        else:
            # Movable commissioning fixture, not a corpus or benchmark door.
            for name, size, position, color in (
                ("Panel", (0.05, 0.12, 0.12), (1.5, -0.3, 1.1), (0.7, 0.3, 0.1)),
                ("Frame", (0.08, 0.08, 0.8), (1.5, 0.05, 1.1), (0.1, 0.4, 0.7)),
                ("Handle", (0.08, 0.12, 0.06), (1.43, -0.1, 1.1), (0.2, 0.8, 0.2)),
            ):
                fixture_cfg = RigidObjectCfg(
                    prim_path=ROOT + "/" + name,
                    spawn=sim_utils.CuboidCfg(
                        size=size,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=name != "Handle", disable_gravity=True
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=position),
                )
                self.scene.rigid_objects[name.lower()] = RigidObject(fixture_cfg)
        stage = self.sim.stage
        self._tool_body_name = "RIGHT_GRIPPER_Y_LINK"
        local = self.push_geometry.wrist_from_base
        self._tool_offset = local[:3, :3] @ self.push_geometry.translation + local[:3, 3]
        from scipy.spatial.transform import Rotation

        self._tool_quat = Rotation.from_matrix(local[:3, :3]).as_quat()
        if self.cfg.cameras:
            model, left, _ = make_zed_x_mini_cfgs(ROOT + "/Zed", resolution="SVGA")
            model.spawn.func(model.prim_path, model.spawn)
            mount = author_alex_purdue_zed_x_mini_mount(stage, ROBOT, model.prim_path)
            if self.cfg.camera_mount_pitch_rad:
                from alexdoor_xas.perception.inspection import tilt_mount

                tilt_mount(stage, mount, self.cfg.camera_mount_pitch_rad)
            schemas.activate_contact_sensors(model.prim_path)
            left.update_latest_camera_pose = True
            optic = stage.GetPrimAtPath(left.prim_path)
            body = stage.GetPrimAtPath(model.prim_path)
            if self.cfg.native_steel:
                from isaaclab_newton.renderers import NewtonWarpRendererCfg

                body = next(
                    p
                    for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT))
                    if p.GetName() == "HEAD_LINK"
                )
                left.renderer_cfg = NewtonWarpRendererCfg()
            world_body = np.array(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0)).T
            world_optic = np.array(UsdGeom.Xformable(optic).ComputeLocalToWorldTransform(0)).T
            if self.cfg.native_steel:
                world_body, world_optic = world_body.copy(), world_optic.copy()
                for transform in (world_body, world_optic):
                    transform[:3, :3] /= np.linalg.norm(transform[:3, :3], axis=0)
            local_optic = np.linalg.solve(world_body, world_optic)
            if self.cfg.native_steel:
                self.authored_camera_from_head = local_optic
            self._optical_translation = local_optic[:3, 3].copy()
            rotation = local_optic[:3, :3]
            rotation = rotation / np.linalg.norm(rotation, axis=0)
            self._optical_quaternion = Rotation.from_matrix(rotation).as_quat()
            self.camera = Camera(left)
            self.scene.sensors["head_rgbd"] = self.camera
        # This scene has one environment. Let Steel's isolated OpenUSD importer
        # build the composed stage instead of importing it in Kit's cloner.
        if self.cfg.native_steel:
            cloner.replicate(plan)
        else:
            self.scene.clone_environments(copy_from_source=self._steel)

    def tool_pose(self):
        index = self.robot.body_names.index(self._tool_body_name)
        position = tensor(self.robot.data.body_link_pos_w)[:, index]
        orientation = tensor(self.robot.data.body_link_quat_w)[:, index]
        offset = torch.tensor(self._tool_offset, device=self.device, dtype=torch.float32)[None]
        quat = torch.tensor(self._tool_quat, device=self.device, dtype=torch.float32)[None]
        return combine_frame_transforms(position, orientation, offset, quat)

    def tool_jacobian(self):
        index = self.robot.body_names.index(self._tool_body_name)
        jac = tensor(self.robot.data.body_link_jacobian_w)[:, index - 1, :, self.arm_ids]
        return link_jacobian_to_point(
            jac,
            tensor(self.robot.data.body_link_quat_w)[:, index],
            torch.tensor(self._tool_offset, device=self.device, dtype=torch.float32)[None],
        )

    def command_pose(self, position, orientation, *, normal_reference_only=False):
        """Set an absolute operational pose for the next A2 step (world, XYZW)."""
        if self.motion_stopped:
            raise RuntimeError("motion_stopped")
        if self.cfg.action_mode != "A2":
            raise ValueError("absolute pose requires A2 mode")
        if type(normal_reference_only) is not bool:
            raise ValueError("invalid_normal_reference_mode")
        if normal_reference_only and (
            self.cfg.pose_feedback_time_constant_s is None
            or not self.cfg.pose_reference_feedforward
        ):
            raise ValueError("normal_reference_requires_causal_pose_feedback")
        position = torch.as_tensor(position, dtype=torch.float32, device=self.device).reshape(1, 3)
        orientation = torch.as_tensor(orientation, dtype=torch.float32, device=self.device).reshape(
            1, 4
        )
        if not bool(torch.isfinite(position).all() and torch.isfinite(orientation).all()):
            raise ValueError("non-finite pose")
        norm = orientation.norm(dim=-1, keepdim=True)
        if bool((norm < 1e-8).any()):
            raise ValueError("zero quaternion")
        self._pending_pose = position, orientation / norm
        self._pending_normal_reference_only = normal_reference_only

    def set_neck_target(self, position):
        """Set a fixed neck pose within the external model's mechanical limits."""
        if self.motion_stopped:
            raise RuntimeError("motion_stopped")
        value = torch.as_tensor(position, device=self.device, dtype=torch.float32).reshape(1, 2)
        limits = tensor(self.robot.data.joint_pos_limits)[:, self.neck_ids]
        if not bool(torch.isfinite(value).all()) or bool(
            ((value < limits[..., 0]) | (value > limits[..., 1])).any()
        ):
            raise ValueError("neck target outside physical limits")
        self.targets[:, self.neck_ids] = value

    def step_a1(self, delta):
        """Execute seven joint deltas without changing the Gym action-space declaration."""
        if self.motion_stopped:
            raise RuntimeError("motion_stopped")
        value = torch.as_tensor(delta, device=self.device, dtype=torch.float32)
        if value.shape not in ((7,), (1, 7)) or not bool(torch.isfinite(value).all()):
            raise ValueError("A1 requires seven finite joint deltas")
        self._pending_joints = value.reshape(1, 7)
        return self.step(torch.zeros((1, self.cfg.action_space), device=self.device))

    def step_a3(self, delta, frame):
        if self.motion_stopped:
            raise RuntimeError("motion_stopped")
        if self.cfg.action_mode != "A2":
            raise ValueError("A3 requires A2 execution mode")
        if reason := validate_object_frame(frame):
            raise ValueError(reason)
        world = frame_delta_to_world(delta, frame)
        return self.step(torch.tensor(world, dtype=torch.float32, device=self.device)[None])

    def _coupled_posture_target(self, primary, jacobian, positions):
        """Pure preference on current robot state; original admission is external."""
        data = self.robot.data
        mass = tensor(data.mass_matrix).to(torch.float64) + torch.diag_embed(
            tensor(data.joint_armature).to(torch.float64)
        )
        response = joint_command_position_response(
            mass,
            self._posture_basis,
            self._posture_passive_damping,
            self._posture_stiffness,
            self._posture_damping,
            self.arm_ids,
            self.cfg.sim.dt,
            self.cfg.decimation,
        )
        target, _ = coupled_margin_preference(
            jacobian,
            positions,
            tensor(data.joint_vel)[:, self.arm_ids],
            primary,
            self.targets[:, self.arm_ids],
            self.pose_limits,
            self.velocities,
            self.step_dt,
            self.pose_posture_support,
            self._posture_stopping_reserve,
            response,
            self.cfg.centering_gain,
        )
        return target

    def preview_targets(
        self,
        *,
        joint_delta=None,
        world_goal=None,
        with_request=False,
        normal_reference_only=False,
    ):
        """Pure controller preview, reused verbatim before dispatch and by admission."""
        if type(normal_reference_only) is not bool:
            raise ValueError("invalid_normal_reference_mode")
        if normal_reference_only and (
            joint_delta is not None
            or self.cfg.pose_feedback_time_constant_s is None
            or not self.cfg.pose_reference_feedforward
        ):
            raise ValueError("normal_reference_requires_causal_pose_feedback")
        q = tensor(self.robot.data.joint_pos)[:, self.arm_ids]
        if (joint_delta is None) == (world_goal is None):
            raise ValueError("Supply one joint or world-pose command")
        error = torch.zeros((1, 6), device=self.device)
        if joint_delta is not None:
            target, excess = bounded_joint_step(
                joint_delta,
                q,
                self.limits,
                self.velocities,
                self.step_dt,
                self.targets[:, self.arm_ids],
            )
        else:
            pe, re = compute_pose_error(*self.tool_pose(), *world_goal, rot_error_type="axis_angle")
            error = torch.cat((pe, re), dim=-1)
            reference_delta = None
            if getattr(self.cfg, "pose_reference_feedforward", False):
                if self.cfg.pose_feedback_time_constant_s is None:
                    raise ValueError("pose_reference_requires_feedback")
                previous = self._last_world_goal
                if (
                    getattr(self.cfg, "pose_normal_transport", False)
                    and not normal_reference_only
                    and self._last_normal_jacobian is not None
                ):
                    # The new ordinary trajectory starts at the achieved pose.
                    # Pay its requested motion, not the old force-goal tracking
                    # error. Keep delivered elastic support through the handoff.
                    reference_delta = error.clone()
                elif previous is not None:
                    dp, dr = compute_pose_error(*previous, *world_goal, rot_error_type="axis_angle")
                    reference_delta = torch.cat((dp, dr), dim=-1)
            jacobian = self.tool_jacobian()
            normal_transport = None
            if normal_reference_only and getattr(self.cfg, "pose_normal_transport", False):
                normal_jacobian = (
                    matrix_from_quat(world_goal[1])[..., :, 0].unsqueeze(-2) @ jacobian[..., :3, :]
                ).squeeze(-2)
                normal_transport = normal_reference_transport(
                    normal_jacobian,
                    self._normal_transport_stiffness,
                    self.targets[:, self.arm_ids] - q,
                    self._last_normal_jacobian,
                )
            finite_tangent_reference_offset = None
            if normal_reference_only and getattr(self.cfg, "pose_tool_frame_reference", False):
                finite_tangent_reference_offset, _ = tool_frame_elastic_reference(
                    jacobian,
                    self._normal_transport_stiffness,
                    self.targets[:, self.arm_ids] - q,
                    matrix_from_quat(self.tool_pose()[1]),
                    getattr(self, "_finite_tangent_tool_reference", None),
                )
            result = bounded_pose_step(
                jacobian,
                error,
                q,
                self.pose_limits,
                self.velocities,
                self.step_dt,
                previous_targets=self.targets[:, self.arm_ids],
                centering_gain=(
                    0.0
                    if getattr(self, "pose_posture_support", None) is not None
                    else self.cfg.centering_gain
                ),
                centering_limits=getattr(self, "pose_centering_limits", None),
                damping=self.cfg.ik_damping,
                feedback_time_constant_s=getattr(self.cfg, "pose_feedback_time_constant_s", None),
                reference_delta=reference_delta,
                compliant_normal=(
                    matrix_from_quat(world_goal[1])[..., :, 0]
                    if normal_reference_only or getattr(self.cfg, "pose_normal_compliance", False)
                    else None
                ),
                normal_reference_only=normal_reference_only,
                normal_transport_m=normal_transport,
                finite_tangent_impedance=(
                    normal_reference_only
                    and getattr(self.cfg, "pose_finite_tangent_impedance", False)
                ),
                finite_tangent_reference_offset=finite_tangent_reference_offset,
                return_request=with_request,
            )
            target, excess = result[:2]
            if normal_reference_only and self._held_surface is not None:
                base = target
                target, _ = self._held_surface.preview(base, world_goal)
                if with_request:
                    result = (target, excess, result[2] + (target - base))

            if getattr(self, "pose_posture_support", None) is not None:
                primary = target
                target = self._coupled_posture_target(primary, jacobian, q)
                if with_request:
                    result = (target, excess, result[2] + (target - primary))
        if with_request:
            requested = q + joint_delta if joint_delta is not None else result[2]
            return target, excess, error, requested
        return target, excess, error

    def stop_motion(self):
        """Latch a measured joint anchor; remove old elastic effort, retain gravity.

        No physics tick, release trajectory or held disturbance torque. Repeated
        calls must not move the anchor as the loaded arm settles. Only reset
        re-enables motion. Zero Gym actions may advance physics while stopped.
        """
        self._pending_pose = self._pending_joints = None
        self._pending_normal_reference_only = False
        self._last_world_goal = None
        self._last_normal_jacobian = None
        self._finite_tangent_tool_reference = None
        if self.motion_stopped:
            return
        self.motion_stopped = True
        if self.stop_response is not None:
            self.stop_response.reset(self.generation)
        self.stop_response_receipt = None
        measured = tensor(self.robot.data.joint_pos).clone()
        if not bool(torch.isfinite(measured).all()):
            raise RuntimeError("invalid_stop_state")
        self.targets = measured
        # Allocate loaded-stop compliance across the complete arm. A shoulder-
        # only change leaves the low-stiffness wrist carrying most deflection.
        # Keep the motion gains, damping ratios and physical effort limits.
        if self._native_arm_impedance is not None:
            gains = self._native_arm_impedance
            stiffness = torch.as_tensor(self.cfg.stop_stiffness, device=self.device)[None]
            gains.write(stiffness, gains.damping * (stiffness / gains.stiffness).sqrt())
        for actuator, column, stiffness, damping in self._motion_impedance:
            name = actuator.joint_names[column]
            stop_stiffness = self.cfg.stop_stiffness[ARM_JOINTS.index(name)]
            ratio = stop_stiffness / stiffness
            actuator.stiffness[:, column] = stop_stiffness
            actuator.damping[:, column] = damping * ratio.sqrt()
        self.robot.set_joint_position_target_index(target=self.targets)
        self.robot.set_joint_velocity_target_index(target=torch.zeros_like(self.targets))
        self.robot.set_joint_effort_target_index(
            target=tensor(self.robot.data.gravity_compensation_forces)
        )

    def _pre_physics_step(self, actions):
        if self.cfg.cameras and not self.render_enabled:
            raise RuntimeError("RGB-D capture requires rendering on each control tick")
        if actions.shape != (1, self.cfg.action_space) or not bool(torch.isfinite(actions).all()):
            raise ValueError("invalid Purdue command")
        if self.motion_stopped:
            self._pending_pose = self._pending_joints = None
            self._pending_normal_reference_only = False
            self._last_world_goal = None
            self._last_normal_jacobian = None
            self._finite_tangent_tool_reference = None
            if bool((actions != 0).any()):
                raise RuntimeError("motion_stopped")
            self.contact_history = []
            self.load_samples = []
            self._physics_actions = 0
            return
        goal = None
        if self.cfg.action_mode == "A1" or self._pending_joints is not None:
            delta = actions if self._pending_joints is None else self._pending_joints
            request_kind, request = 1, delta.clone()
            target, excess, error, requested = self.preview_targets(
                joint_delta=delta, with_request=True
            )
        else:
            request_kind = 2 if self._pending_pose is not None else 3
            request = (
                torch.cat(self._pending_pose, dim=-1)
                if self._pending_pose is not None
                else torch.cat((actions, torch.zeros((1, 1), device=self.device)), dim=-1)
            )
            delta = actions.clone()
            delta[:, :3].clamp_(-self.cfg.max_pos_delta_m, self.cfg.max_pos_delta_m)
            delta[:, 3:].clamp_(-self.cfg.max_rot_delta_rad, self.cfg.max_rot_delta_rad)
            goal = self._pending_pose or apply_delta_pose(*self.tool_pose(), delta)
            target, excess, error, requested = self.preview_targets(
                world_goal=goal,
                with_request=True,
                normal_reference_only=getattr(self, "_pending_normal_reference_only", False),
            )
        delivered_normal_reference_only = bool(
            goal is not None
            and self._pending_pose is not None
            and getattr(self, "_pending_normal_reference_only", False)
        )
        delivered_normal_reference_handoff = bool(
            goal is not None
            and not delivered_normal_reference_only
            and getattr(self.cfg, "pose_normal_transport", False)
            and self._last_normal_jacobian is not None
        )
        dispatched_tangent_tool_reference = None
        if delivered_normal_reference_only and getattr(
            self.cfg, "pose_tool_frame_reference", False
        ):
            dispatched_tangent_tool_reference = getattr(
                self, "_finite_tangent_tool_reference", None
            )
            if dispatched_tangent_tool_reference is None:
                _, dispatched_tangent_tool_reference = tool_frame_elastic_reference(
                    self.tool_jacobian(),
                    self._normal_transport_stiffness,
                    self.targets[:, self.arm_ids]
                    - tensor(self.robot.data.joint_pos)[:, self.arm_ids],
                    matrix_from_quat(self.tool_pose()[1]),
                )
            dispatched_tangent_tool_reference = dispatched_tangent_tool_reference.clone()
        dispatched_normal_jacobian = None
        if delivered_normal_reference_only and getattr(self.cfg, "pose_normal_transport", False):
            dispatched_normal_jacobian = (
                (
                    matrix_from_quat(goal[1])[..., :, 0].unsqueeze(-2)
                    @ self.tool_jacobian()[..., :3, :]
                )
                .squeeze(-2)
                .clone()
            )
        self._pending_pose = self._pending_joints = None
        self._pending_normal_reference_only = False
        self.command_event = dict(
            normal_reference_only=delivered_normal_reference_only,
            normal_reference_handoff=delivered_normal_reference_handoff,
            request_kind=request_kind,
            request=request.clone(),
            requested_target=requested.clone(),
            q_before=tensor(self.robot.data.joint_pos)[:, self.arm_ids].clone(),
            request_s=self._time_s,
        )
        self.clamp_excess, self.pose_error = excess, error
        self.targets[:, self.arm_ids] = target
        # Commit only a dispatched pose goal; previews and rejected commands
        # cannot advance reference history. Joint commands end that history.
        self._last_world_goal = None if goal is None else tuple(value.clone() for value in goal)
        if delivered_normal_reference_only and self._held_surface is not None:
            _, _, _, (_, shift) = self._held_surface.cache
            position, orientation = self._last_world_goal
            self._last_world_goal = (position.double() + shift).to(position.dtype), orientation
        self._last_normal_jacobian = dispatched_normal_jacobian
        self._finite_tangent_tool_reference = dispatched_tangent_tool_reference
        self.contact_history = []
        self.load_samples = []
        self._physics_actions = 0

    def _apply_action(self):
        if self.cfg.observed_load_feedback and self._physics_actions > 0:
            from alexdoor_xas.perception.load import simulated_load_sample

            sample = simulated_load_sample(
                self, self._time_s + self.cfg.sim.dt * self._physics_actions
            )
            self.load_samples.append(sample)
            if self.motion_stopped and self.stop_response is not None:
                self.stop_response.observe(sample, self.generation)
            if self.event_sink is not None:
                self.event_sink("feedback", sample)
        self.robot.set_joint_position_target_index(target=self.targets)
        gravity = tensor(self.robot.data.gravity_compensation_forces)
        if self.motion_stopped and self.stop_response is not None:
            now = self._time_s + self.cfg.sim.dt * self._physics_actions
            correction, self.stop_response_receipt = self.stop_response.preview(
                now, self.generation
            )
            gravity = gravity.clone()
            gravity[:, self.arm_ids] += torch.as_tensor(
                correction, device=self.device, dtype=gravity.dtype
            )
        self.robot.set_joint_effort_target_index(target=gravity)
        if self.event_sink is not None and self._physics_actions == 0 and not self.motion_stopped:
            self.event_sink("command", self.command_event)
        if self.contacts is not None and self._physics_actions > 0:
            self.contact_history.append(self.contacts.read())
        self._physics_actions += 1
        if self._held_surface is not None:
            self._held_surface.before_last_substep()

    def _get_observations(self):
        q = tensor(self.robot.data.joint_pos)[:, self.observed_ids]
        dq = tensor(self.robot.data.joint_vel)[:, self.observed_ids]
        self._time_s = float(self.episode_length_buf[0]) * self.step_dt
        if self.cfg.observed_load_feedback:
            from alexdoor_xas.perception.load import simulated_load_sample

            self.load_samples = getattr(self, "load_samples", [])
            sample = simulated_load_sample(self, self._time_s)
            self.load_samples.append(sample)
            if self.motion_stopped and self.stop_response is not None:
                self.stop_response.observe(sample, self.generation)
            if self.event_sink is not None:
                self.event_sink("feedback", sample)
        if self.contacts is not None:
            self.contact_history.append(self.contacts.read())
        if self.capture is not None:
            # Fabric does not propagate the attached ZED's camera descendant pose.
            # Bridge the canonical optical transform from the measured rigid-body pose.
            if self._mounted_camera is not None:
                self._mounted_camera.sync()
            else:
                if not self.cfg.native_steel:
                    self.sim.forward()
                index = self.robot.body_names.index("HEAD_LINK" if self.cfg.native_steel else "Zed")
                position, orientation = combine_frame_transforms(
                    tensor(self.robot.data.body_link_pos_w)[:, index],
                    tensor(self.robot.data.body_link_quat_w)[:, index],
                    torch.as_tensor(
                        self._optical_translation, device=self.device, dtype=torch.float32
                    )[None],
                    torch.as_tensor(
                        self._optical_quaternion, device=self.device, dtype=torch.float32
                    )[None],
                )
                self.camera.set_world_poses(position, orientation, convention="opengl")
            if self.capture.sample is None:
                # Settle renderer history at the reset state without stepping physics.
                for _ in range(self.cfg.reset_camera_frames):
                    self.sim.render()
                    self.camera.update(0.0, force_recompute=True)
            data = self.camera.data
            frame = int(tensor(self.camera.frame)[0])
            self.capture.capture(
                frame,
                self._time_s,
                tensor(data.output["rgb"]),
                tensor(data.output["distance_to_image_plane"]),
                q,
                dq,
            )
        if self.event_sink is not None:
            self.event_sink("state", self)
        return {"policy": torch.cat((q, dq), dim=-1)}

    def _get_rewards(self):
        return torch.zeros(1, device=self.device)

    def _get_dones(self):
        return torch.zeros(1, dtype=torch.bool, device=self.device), torch.zeros(
            1, dtype=torch.bool, device=self.device
        )

    def _reset_idx(self, env_ids):
        if self._steel:
            self.sim.physics_manager.contact_sim_view.reset()
            self.contact_history = []
        if self._native_contacts is not None:
            self._native_contacts.reset()
            self.contact_history = []
        super()._reset_idx(env_ids)
        self.generation += 1
        if self.stop_response is not None:
            self.stop_response.reset(self.generation)
        self.stop_response_receipt = None
        self.command_event = None
        if self.motion_stopped and self._native_arm_impedance is not None:
            gains = self._native_arm_impedance
            gains.write(gains.stiffness, gains.damping)
        self.motion_stopped = False
        for actuator, column, stiffness, damping in self._motion_impedance:
            actuator.stiffness[:, column] = stiffness
            actuator.damping[:, column] = damping
        for fixture in self.scene.rigid_objects.values():
            state = tensor(fixture.data.default_root_state).clone()
            fixture.write_root_pose_to_sim_index(root_pose=state[:, :7])
            fixture.write_root_velocity_to_sim_index(root_velocity=state[:, 7:])
        if self.door is not None:
            door_q = tensor(self.door.data.default_joint_pos).clone()
            self.door.write_joint_position_to_sim_index(position=door_q)
            self.door.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(door_q))
            self.door.set_joint_effort_target_index(target=torch.zeros_like(door_q))
        q = tensor(self.robot.data.default_joint_pos).clone()
        self.robot.write_joint_position_to_sim_index(position=q)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        self.robot.set_joint_position_target_index(target=q)
        self.targets = q.clone()
        self._last_world_goal = None
        self._last_normal_jacobian = None
        self._finite_tangent_tool_reference = None
        self._pending_pose = None
        self._pending_normal_reference_only = False
        self._pending_joints = None
        self.load_samples = []
        if self._held_surface is not None:
            self._held_surface.previous = self._held_surface.cache = None
        if self.capture is not None:
            self.capture.reset()
