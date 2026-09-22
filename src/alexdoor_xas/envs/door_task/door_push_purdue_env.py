"""Single fixed-base Purdue runtime with full-pose control and observed RGB-D."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import torch
from ihmc_alex_isaaclab.end_effectors.weiss_wsg32 import alex_purdue_wsg32_targets
from ihmc_alex_isaaclab.platforms.purdue_alex003_pedestal import (
    load_purdue_alex003_pedestal_spec,
    make_purdue_alex003_pedestal_cfg,
)
from ihmc_alex_isaaclab.robots.alex_purdue import make_alex_purdue_cfg
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
)
from pxr import Usd, UsdGeom, UsdPhysics

from alexdoor_xas.action.frames import frame_delta_to_world
from alexdoor_xas.assets.purdue import ARM_JOINTS, NECK_JOINTS, derive_push_geometry
from alexdoor_xas.kinematics.point_jacobian import link_jacobian_to_point
from alexdoor_xas.kinematics.pose_control import bounded_joint_step, bounded_pose_step
from alexdoor_xas.recording.rgbd import RGBDCapture

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
        cfg.action_space = 7 if cfg.action_mode == "A1" else 6
        self.capture = None
        self.contacts = None
        self._time_s = 0.0
        self._pending_pose = None
        self._pending_joints = None
        super().__init__(cfg, **kwargs)
        self.arm_ids = self._resolve_joints(ARM_JOINTS)
        self.neck_ids = self._resolve_joints(NECK_JOINTS)
        self.observed_ids = self.arm_ids + self.neck_ids
        self.targets = tensor(self.robot.data.default_joint_pos).clone()
        self.limits = tensor(self.robot.data.joint_pos_limits)[:, self.arm_ids].clone()
        self.velocities = tensor(self.robot.data.joint_vel_limits)[:, self.arm_ids].clone()
        self.velocities.clamp_(max=cfg.max_joint_speed)
        self.clamp_excess = torch.zeros((1, 7), device=self.device)
        self.pose_error = torch.zeros((1, 6), device=self.device)
        self.contacts = ContactObserver(
            self.sim,
            self.sim.stage,
            ROBOT,
            self.panel_path,
            PEDESTAL,
            "/World/Ground/CollisionPlane",
            self.push_geometry.wrist_from_base,
        )
        if cfg.cameras:
            camera_prim = self.sim.stage.GetPrimAtPath(self.camera.cfg.prim_path)
            near, far = UsdGeom.Camera(camera_prim).GetClippingRangeAttr().Get()
            self.capture = RGBDCapture(float(near), float(far))
        self.contact_history = []

    def _resolve_joints(self, names):
        indices, found = self.robot.find_joints(list(names), preserve_order=True)
        if tuple(found) != tuple(names) or len(set(indices)) != len(names):
            raise RuntimeError(f"Joint order does not resolve: {names}")
        return indices

    def _setup_scene(self):
        import isaaclab.sim as sim_utils

        spec = load_purdue_alex003_pedestal_spec()
        pedestal = make_purdue_alex003_pedestal_cfg(PEDESTAL)
        x, y, yaw = self.cfg.floor_pose
        floor_quat = (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))
        pedestal.spawn.func(
            PEDESTAL, pedestal.spawn, translation=(x, y, 0.0), orientation=floor_quat
        )
        schemas.activate_contact_sensors(PEDESTAL)
        robot_cfg = make_alex_purdue_cfg(
            fix_base=True, variant="full_convex", end_effector="wsg32_umi_v1"
        )
        robot_cfg.prim_path = ROBOT
        robot_cfg.init_state.pos = (x, y, spec.alex_root_world_z_m)
        robot_cfg.init_state.rot = floor_quat
        joints = ET.parse(robot_cfg.spawn.asset_path).getroot().findall("joint")
        defaults = {j.get("name"): 0.0 for j in joints if j.get("type") != "fixed"}
        robot_cfg.init_state.joint_pos = {
            **defaults,
            **self.cfg.initial_joints,
            **alex_purdue_wsg32_targets(0.0, "left"),
            **alex_purdue_wsg32_targets(0.0, "right"),
        }
        self.push_geometry = derive_push_geometry(robot_cfg.spawn.asset_path)
        self.robot = Articulation(robot_cfg)
        for prim in Usd.PrimRange(self.sim.stage.GetPrimAtPath(ROBOT)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                schemas.activate_contact_sensors(str(prim.GetPath()))
        self.scene.articulations["robot"] = self.robot
        sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
        light = sim_utils.DomeLightCfg(intensity=1800)
        light.func("/World/Light", light)
        self.door = None
        self.panel_path = PANEL
        if self.cfg.synthetic_door is not None:
            from isaaclab.actuators import ImplicitActuatorCfg
            from isaaclab.assets import ArticulationCfg

            from alexdoor_xas.assets.synthetic_door import author_synthetic_door

            root = ROOT + "/Door"
            author_synthetic_door(self.sim.stage, root, self.cfg.synthetic_door)
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
                            damping=self.cfg.synthetic_door.damping,
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
            author_alex_purdue_zed_x_mini_mount(stage, ROBOT, model.prim_path)
            schemas.activate_contact_sensors(model.prim_path)
            left.update_latest_camera_pose = True
            optic = stage.GetPrimAtPath(left.prim_path)
            body = stage.GetPrimAtPath(model.prim_path)
            world_body = np.array(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0)).T
            world_optic = np.array(UsdGeom.Xformable(optic).ComputeLocalToWorldTransform(0)).T
            local_optic = np.linalg.inv(world_body) @ world_optic
            self._optical_translation = local_optic[:3, 3].copy()
            rotation = local_optic[:3, :3]
            rotation = rotation / np.linalg.norm(rotation, axis=0)
            self._optical_quaternion = Rotation.from_matrix(rotation).as_quat()
            self.camera = Camera(left)
            self.scene.sensors["head_rgbd"] = self.camera
        self.scene.clone_environments(copy_from_source=False)

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

    def command_pose(self, position, orientation):
        """Set an absolute operational pose for the next A2 step (world, XYZW)."""
        if self.cfg.action_mode != "A2":
            raise ValueError("absolute pose requires A2 mode")
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

    def set_neck_target(self, position):
        """Set a fixed neck pose within the external model's mechanical limits."""
        value = torch.as_tensor(position, device=self.device, dtype=torch.float32).reshape(1, 2)
        limits = tensor(self.robot.data.joint_pos_limits)[:, self.neck_ids]
        if not bool(torch.isfinite(value).all()) or bool(
            ((value < limits[..., 0]) | (value > limits[..., 1])).any()
        ):
            raise ValueError("neck target outside physical limits")
        self.targets[:, self.neck_ids] = value

    def step_a1(self, delta):
        """Execute seven joint deltas without changing the Gym action-space declaration."""
        value = torch.as_tensor(delta, device=self.device, dtype=torch.float32)
        if value.shape not in ((7,), (1, 7)) or not bool(torch.isfinite(value).all()):
            raise ValueError("A1 requires seven finite joint deltas")
        self._pending_joints = value.reshape(1, 7)
        return self.step(torch.zeros((1, self.cfg.action_space), device=self.device))

    def step_a3(self, delta, frame):
        if self.cfg.action_mode != "A2":
            raise ValueError("A3 requires A2 execution mode")
        from alexdoor_xas.adapters.a3 import validate_object_frame

        if reason := validate_object_frame(frame):
            raise ValueError(reason)
        world = frame_delta_to_world(delta, frame)
        return self.step(torch.tensor(world, dtype=torch.float32, device=self.device)[None])

    def _pre_physics_step(self, actions):
        if self.cfg.cameras and not self.render_enabled:
            raise RuntimeError("RGB-D capture requires rendering on each control tick")
        if actions.shape != (1, self.cfg.action_space) or not bool(torch.isfinite(actions).all()):
            raise ValueError("invalid Purdue command")
        q = tensor(self.robot.data.joint_pos)[:, self.arm_ids]
        if self.cfg.action_mode == "A1" or self._pending_joints is not None:
            delta = actions if self._pending_joints is None else self._pending_joints
            self._pending_joints = None
            target, self.clamp_excess = bounded_joint_step(
                delta,
                q,
                self.limits,
                self.velocities,
                self.step_dt,
                self.targets[:, self.arm_ids],
            )
        else:
            position, orientation = self.tool_pose()
            delta = actions.clone()
            delta[:, :3].clamp_(-self.cfg.max_pos_delta_m, self.cfg.max_pos_delta_m)
            delta[:, 3:].clamp_(-self.cfg.max_rot_delta_rad, self.cfg.max_rot_delta_rad)
            goal = self._pending_pose or apply_delta_pose(position, orientation, delta)
            self._pending_pose = None
            pe, re = compute_pose_error(position, orientation, *goal, rot_error_type="axis_angle")
            self.pose_error = torch.cat((pe, re), dim=-1)
            target, self.clamp_excess = bounded_pose_step(
                self.tool_jacobian(),
                self.pose_error,
                q,
                self.limits,
                self.velocities,
                self.step_dt,
                previous_targets=self.targets[:, self.arm_ids],
                centering_gain=self.cfg.centering_gain,
                damping=self.cfg.ik_damping,
            )
        self.targets[:, self.arm_ids] = target
        self.contact_history = []
        self._physics_actions = 0

    def _apply_action(self):
        self.robot.set_joint_position_target_index(target=self.targets)
        gravity = tensor(self.robot.data.gravity_compensation_forces)
        self.robot.set_joint_effort_target_index(target=gravity)
        if self.contacts is not None and self._physics_actions > 0:
            self.contact_history.append(self.contacts.read())
        self._physics_actions += 1

    def _get_observations(self):
        q = tensor(self.robot.data.joint_pos)[:, self.observed_ids]
        dq = tensor(self.robot.data.joint_vel)[:, self.observed_ids]
        self._time_s = float(self.episode_length_buf[0]) * self.step_dt
        if self.contacts is not None:
            self.contact_history.append(self.contacts.read())
        if self.capture is not None:
            # Fabric does not propagate the attached ZED's camera descendant pose.
            # Bridge the canonical optical transform from the measured rigid-body pose.
            self.sim.forward()
            index = self.robot.body_names.index("Zed")
            position, orientation = combine_frame_transforms(
                tensor(self.robot.data.body_link_pos_w)[:, index],
                tensor(self.robot.data.body_link_quat_w)[:, index],
                torch.as_tensor(self._optical_translation, device=self.device, dtype=torch.float32)[
                    None
                ],
                torch.as_tensor(self._optical_quaternion, device=self.device, dtype=torch.float32)[
                    None
                ],
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
        return {"policy": torch.cat((q, dq), dim=-1)}

    def _get_rewards(self):
        return torch.zeros(1, device=self.device)

    def _get_dones(self):
        return torch.zeros(1, dtype=torch.bool, device=self.device), torch.zeros(
            1, dtype=torch.bool, device=self.device
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
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
        self._pending_pose = None
        self._pending_joints = None
        if self.capture is not None:
            self.capture.reset()
