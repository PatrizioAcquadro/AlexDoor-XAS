"""Common articulated gantry and free block used by the short benchmarks."""

from pathlib import Path

import numpy as np
import torch
from isaaclab import sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

JOINTS = ("slide_x", "slide_z", "finger_positive", "finger_negative")
FINGERS = ("Positive", "Negative")


def author_robot(stage, path, offset, robot_cfg, material, collision):
    """Author the same fixed-base four-DOF articulation for both solvers."""
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    finger_offset = robot_cfg["finger_offset_m"]
    joints = (
        ("Carriage", "Base", JOINTS[0], "X", (0, 0, 0), (-0.25, 0.3)),
        ("Hand", "Carriage", JOINTS[1], "Z", (0, 0, -0.23), (-0.18, 0.13)),
        ("Positive", "Hand", JOINTS[2], "Y", (0, finger_offset, -0.09), (-0.023, 0.008)),
        ("Negative", "Hand", JOINTS[3], "Y", (0, -finger_offset, -0.09), (-0.008, 0.023)),
    )
    poses = {"Base": np.asarray((offset, 0, 0.50))}
    for child, parent, _, _, delta, _ in joints:
        poses[child] = poses[parent] + delta
    for name, position in poses.items():
        body = UsdGeom.Xform.Define(stage, f"{path}/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(*position))
        prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = 0.15 if name in FINGERS else 0.5
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        if name in ("Carriage", "Base"):
            # Explicit inertia for links without a collider.
            UsdPhysics.MassAPI(prim).CreateDiagonalInertiaAttr(Gf.Vec3f(0.003))
        if name in ("Hand", *FINGERS):
            dims = robot_cfg["finger_size_m"] if name in FINGERS else (0.10, 0.135, 0.025)
            box = UsdGeom.Cube.Define(stage, f"{path}/{name}/Collider")
            box.CreateSizeAttr(1)
            box.AddScaleOp().Set(Gf.Vec3f(*dims))
            box.CreateDisplayColorAttr([(0.23, 0.30, 0.36)])
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            sim_utils.schemas.modify_collision_properties(str(box.GetPath()), collision, stage)
            UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
                material, materialPurpose="physics"
            )
        if name in FINGERS:
            sim_utils.schemas.activate_contact_sensors(str(body.GetPath()))

    anchor = UsdPhysics.FixedJoint.Define(stage, f"{path}/fixed_base")
    anchor.CreateBody1Rel().SetTargets([f"{path}/Base"])
    anchor.CreateLocalPos0Attr(Gf.Vec3f(*poses["Base"]))
    for child, parent, name, axis, delta, limits in joints:
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/{name}")
        joint.CreateBody0Rel().SetTargets([f"{path}/{parent}"])
        joint.CreateBody1Rel().SetTargets([f"{path}/{child}"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*delta))
        joint.CreateAxisAttr(axis)
        joint.CreateLowerLimitAttr(limits[0])
        joint.CreateUpperLimitAttr(limits[1])
        joint.CreateCollisionEnabledAttr(False)


class BenchFixture:
    def __init__(self, simulation_cfg, cfg, case, backend, env_roots):
        sim = sim_utils.SimulationContext(simulation_cfg)
        self.sim, self.cfg, self.case = sim, cfg, case
        self.block = self.camera = None
        self.backend = backend
        stage = sim.stage
        scene = cfg["scene"]
        material = UsdShade.Material.Define(stage, "/World/GripMaterial")
        properties = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        properties.CreateStaticFrictionAttr(scene["friction"])
        properties.CreateDynamicFrictionAttr(scene["friction"])
        collision = sim_utils.CollisionPropertiesCfg(
            contact_offset=scene["contact_offset_m"],
            rest_offset=0.0,
        )
        origins = np.zeros((len(env_roots), 3), dtype=np.float32)
        origins[:, 0] = np.arange(len(env_roots)) * scene["spacing_m"]
        effort = ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0, damping=0)
        if backend == "steel":
            from isaaclab import cloner
            from isaaclab.assets import AssetBaseCfg
            from isaaclab.cloner import CloneCfg

            paths = [f"{env_roots[0]}/Robot"]
            if case["task"] == "block":
                paths.extend(f"{env_roots[0]}/{name}" for name in ("Table", "Block"))
            if case.get("rgbd"):
                paths.extend(("/World/Camera", "/World/Light"))
            plan = cloner.clone_plan_from_env_0(
                CloneCfg(),
                [AssetBaseCfg(prim_path=path) for path in paths],
                len(env_roots),
                scene["spacing_m"],
                positions=origins,
            )
            effort.joint_effort_limit = 100
            author_robot(stage, f"{env_roots[0]}/Robot", 0.0, cfg["robot"], material, collision)
        else:
            effort.effort_limit_sim = 100
            for root, offset in zip(env_roots, origins[:, 0], strict=True):
                author_robot(
                    stage, f"{root}/Robot", float(offset), cfg["robot"], material, collision
                )
        self.robot = Articulation(
            ArticulationCfg(
                prim_path="/World/envs/env_.*/Robot",
                spawn=None,
                actuators={"effort": effort},
            )
        )
        if case["task"] == "block":
            if len(env_roots) != 1:
                raise ValueError("The common block fixture uses one environment")
            self.make_block(env_roots[0], scene, collision)
        if case.get("rgbd"):
            self.make_camera(cfg["camera"])
        if backend == "steel":
            cloner.replicate(plan)
        self.gain = torch.tensor(cfg["robot"]["stiffness"], device=sim.device)
        self.damping = torch.tensor(cfg["robot"]["damping"], device=sim.device)
        self.limit = torch.tensor(cfg["robot"]["effort_limits"], device=sim.device)
        self.amplitudes = torch.tensor(cfg["tracking"]["amplitudes_m"], device=sim.device)
        self.rates = torch.tensor(cfg["tracking"]["rates_rad_s"], device=sim.device)
        self.waypoints = np.asarray(cfg["block_program"]["waypoints"]).T

    def make_block(self, root, scene, collision):
        surface = sim_utils.RigidBodyMaterialCfg(
            static_friction=scene["friction"],
            dynamic_friction=scene["friction"],
            restitution=0,
        )
        table = sim_utils.CuboidCfg(
            size=(0.80, 0.65, 0.06),
            collision_props=collision,
            physics_material=surface,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.37, 0.40)),
        )
        table.func(f"{root}/Table", table, translation=(0.05, 0, -0.03))
        size = scene["block_size_m"]
        self.block_path = f"{root}/Block"
        self.finger_paths = [f"{root}/Robot/{finger}" for finger in FINGERS]
        self.block = RigidObject(
            RigidObjectCfg(
                prim_path=self.block_path,
                spawn=sim_utils.CuboidCfg(
                    size=(size,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=collision,
                    mass_props=sim_utils.MassPropertiesCfg(mass=scene["block_mass_kg"]),
                    physics_material=surface,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.78, 0.033, 0.024)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, size / 2 + 0.001)),
            )
        )

    def make_camera(self, camera):
        from scipy.spatial.transform import Rotation

        if self.backend == "steel":
            from isaaclab_ov.renderers import OVRTXRendererCfg

            renderer = OVRTXRendererCfg()
        else:
            from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

            renderer = IsaacRtxRendererCfg()
        eye = np.array(camera["eye_m"])
        back = eye - np.asarray(camera["look_at_m"])
        back /= np.linalg.norm(back)
        right = np.cross([0, 0, 1], back)
        right /= np.linalg.norm(right)
        orientation = Rotation.from_matrix(np.column_stack((right, np.cross(back, right), back)))
        light = sim_utils.DomeLightCfg(intensity=2200)
        light.func("/World/Light", light)
        self.camera = Camera(
            CameraCfg(
                prim_path="/World/Camera",
                height=camera["height"],
                width=camera["width"],
                data_types=["rgb", "distance_to_image_plane"],
                update_period=0,
                renderer_cfg=renderer,
                spawn=sim_utils.PinholeCameraCfg(focal_length=22, clipping_range=(0.1, 4.0)),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(eye), rot=tuple(orientation.as_quat()), convention="opengl"
                ),
            )
        )

    def start(self):
        self.sim.reset()
        self.ids, _ = self.robot.find_joints(list(JOINTS), preserve_order=True)
        if len(self.ids) != 4:
            raise RuntimeError("Common articulation joint layout differs between backends")
        self.hand = self.robot.body_names.index("Hand")
        if self.block is None:
            return
        if self.backend == "physx":
            self.contact = self.sim.physics_sim_view.create_rigid_contact_view(
                self.finger_paths,
                max_contact_data_count=64,
                filter_patterns=[[self.block_path] for _ in self.finger_paths],
            )
            return
        manager = self.sim.backend_manager
        manager.add_contact_sensor(body_names_expr=self.block_path)
        model = manager.backend.model
        owners = np.array(model.body_label, dtype=object)
        body = model.shape_body.numpy()
        self.shape_owner = np.full(len(body), "", dtype=object)
        self.shape_owner[body >= 0] = owners[body[body >= 0]]
        if not all(path in self.shape_owner for path in (*self.finger_paths, self.block_path)):
            raise RuntimeError("Steel contact shapes must resolve to both fingers and the block")

    def reset(self, seed):
        random = np.random.default_rng(seed)
        q = self.robot.data.default_joint_pos.torch.clone()
        self.robot.write_joint_position_to_sim_index(position=q)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        self.robot.reset()
        self.last_effort = torch.zeros_like(q)
        self.phase = torch.as_tensor(
            random.uniform(0, 0.2, size=(len(q), 1)), dtype=torch.float32, device=q.device
        )
        self.rows, self.frames = [], []
        if self.block is not None:
            state = self.block.data.default_root_state.torch.clone()
            state[0, 1] = float(random.uniform(-0.001, 0.001))
            self.block_initial = state[0, :3].cpu().numpy().copy()
            self.block.write_root_pose_to_sim_index(root_pose=state[:, :7])
            self.block.write_root_velocity_to_sim_index(root_velocity=state[:, 7:])
            self.block.reset()
        self.sim.forward()

    def control(self, t):
        q = self.robot.data.joint_pos.torch[:, self.ids]
        dq = self.robot.data.joint_vel.torch[:, self.ids]
        if self.block is None:
            target = self.amplitudes * torch.sin(self.rates * t + self.phase)
        else:
            times, xs, zs, grips = self.waypoints
            x, z, grip = (np.interp(t, times, values) for values in (xs, zs, grips))
            target = torch.tensor([[x, z, -grip, grip]], device=q.device, dtype=q.dtype)
        self.target = target
        self.last_effort[:, self.ids] = ((target - q) * self.gain - dq * self.damping).clamp(
            -self.limit,
            self.limit,
        )

    def dispatch(self):
        self.robot.set_joint_effort_target_index(target=self.last_effort)
        self.robot.write_data_to_sim()

    def advance(self, dt):
        self.robot.update(dt)
        if self.camera is not None:
            self.camera.update(dt)
        if self.block is not None:
            self.block.update(dt)

    def contacts(self):
        dt = self.sim.get_physics_dt()
        if self.backend == "physx":
            return np.linalg.norm(self.contact.get_contact_force_matrix(dt).numpy()[:, 0], axis=-1)
        backend = self.sim.backend_manager.backend
        buf = backend.contacts
        count = int(buf.rigid_contact_count.numpy()[0])
        if count >= buf.rigid_contact_max:
            raise RuntimeError("Steel contact buffer saturated")
        a = self.shape_owner[buf.rigid_contact_shape0.numpy()[:count]]
        b = self.shape_owner[buf.rigid_contact_shape1.numpy()[:count]]
        force = backend.contact_observables.contact_f.numpy()[:count, :3]
        normals = buf.rigid_contact_normal.numpy()[:count]
        normal = np.maximum(-np.sum(force * normals, axis=-1), 0)
        return np.asarray(
            [
                normal[
                    ((a == finger) & (b == self.block_path))
                    | ((b == finger) & (a == self.block_path))
                ].sum()
                for finger in self.finger_paths
            ]
        )

    def observe(self, t):
        q = self.robot.data.joint_pos.torch[:, self.ids]
        error = (q - self.target).cpu().numpy()
        if self.block is None:
            self.rows.append((t, error))
            return
        pose = self.block.data.root_pos_w.torch[0].cpu().numpy().copy()
        speed = np.linalg.norm(self.block.data.root_lin_vel_w.torch[0].cpu().numpy())
        hand = self.robot.data.body_link_pos_w.torch[0, self.hand].cpu().numpy().copy()
        self.rows.append((t, error, pose, speed, hand, self.contacts()))

    def capture(self):
        self.sim.render()
        self.camera.update(0, force_recompute=True)
        data = self.camera.data
        return (
            int(self.camera.frame.torch[0]),
            data.output["rgb"].torch[0].cpu().numpy().copy(),
            data.output["distance_to_image_plane"].torch[0, ..., 0].cpu().numpy().copy(),
        )

    def perceive(self, frame):
        _, rgb, depth = frame
        rgb = rgb[..., :3].astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0.1) & (depth < 4.0)
        r, g, b = rgb.transpose(2, 0, 1)
        mask = valid & (r > g * 1.4) & (r > b * 1.4) & (r > 55)
        yy, xx = np.nonzero(mask)
        detection = None if len(xx) < 8 else (xx.mean(), yy.mean(), np.median(depth[mask]))
        return float(valid.mean()), detection, bool(np.ptp(rgb) > 0)

    def record(self, frame, detection, time_s):
        self.frames.append((frame, detection, time_s))

    def finish_recording(self, output):
        if not self.frames or output is None:
            return
        with Path(output).open("xb") as file:
            np.savez_compressed(
                file,
                frame_ids=[f[0][0] for f in self.frames],
                sim_time_s=[f[2] for f in self.frames],
                rgb=np.stack([f[0][1] for f in self.frames]),
                depth_m=np.stack([f[0][2] for f in self.frames]),
                detections=[f[1][1] if f[1][1] is not None else (np.nan,) * 3 for f in self.frames],
            )

    def evaluate(self):
        errors = np.stack([row[1] for row in self.rows])
        result = dict(
            joint_names=JOINTS,
            tracking_rmse_mm=(
                1000
                * np.sqrt(
                    np.mean(errors**2, axis=(0, 1)),
                )
            ).tolist(),
            tracking_max_mm=(1000 * np.max(np.abs(errors), axis=(0, 1))).tolist(),
        )
        if self.block is None:
            return result
        t, _, pose, speed, hand, contact = zip(*self.rows, strict=True)
        t, pose, speed, hand, contact = map(np.asarray, (t, pose, speed, hand, contact))
        cfg = self.cfg["criteria"]
        program = self.cfg["block_program"]
        hold_start, hold_end = program["lift_and_carry_s"]
        hold = (t >= hold_start) & (t < hold_end)
        tail = t >= t[-1] - program["settle_window_s"]
        relative = (pose - hand)[hold]
        slip = float(np.linalg.norm(relative - relative[0], axis=-1).max())
        lift = float(np.max(pose[:, 2]) - self.block_initial[2])
        drift = float(np.linalg.norm(pose[tail] - pose[tail][0], axis=-1).max())
        xy_error = float(np.linalg.norm(pose[-1, :2] - [self.waypoints[1, -1], 0]))
        support_error = abs(float(pose[-1, 2]) - self.cfg["scene"]["block_size_m"] / 2)
        bilateral = np.all(contact >= cfg["minimum_contact_n"], axis=1)
        contact_ticks = int(bilateral.sum())
        held = bool(np.any(bilateral & hold))
        release = held and not bool(np.any(contact[tail] >= cfg["minimum_contact_n"]))
        last_contact = (
            int(np.flatnonzero(np.any(contact >= cfg["minimum_contact_n"], axis=1))[-1])
            if release
            else None
        )
        settled = bool(
            drift <= cfg["maximum_settle_drift_m"]
            and speed[tail].max() <= cfg["maximum_final_speed_m_s"]
        )
        result.update(
            success=bool(
                held
                and lift >= cfg["minimum_lift_m"]
                and slip <= cfg["maximum_grip_slip_m"]
                and release
                and settled
                and xy_error <= cfg["maximum_final_xy_error_m"]
                and support_error <= cfg["maximum_final_support_error_m"]
            ),
            bilateral_contact_ticks=contact_ticks,
            first_bilateral_contact_s=float(t[bilateral][0]) if contact_ticks else None,
            peak_finger_force_n=np.max(contact, axis=0).tolist(),
            lift_m=lift,
            grip_slip_m=slip,
            released=release,
            release_time_s=float(t[last_contact + 1]) if release else None,
            settled=settled,
            settle_drift_m=drift,
            settle_peak_speed_m_s=float(speed[tail].max()),
            final_xy_error_m=xy_error,
            final_support_error_m=support_error,
        )
        if self.frames:
            ids = np.array([f[0][0] for f in self.frames])
            depth, detection, nonflat = zip(*(f[1] for f in self.frames), strict=True)
            detection_fraction = sum(d is not None for d in detection) / len(ids)
            centroids = [d[:2] for d in detection if d is not None]
            valid_frames = bool(
                np.all(np.diff(ids) > 0)
                and all(nonflat)
                and min(depth) >= cfg["minimum_frame_valid_depth_fraction"]
            )
            result.update(
                frames=len(ids),
                frames_valid=valid_frames,
                min_valid_depth_fraction=min(depth),
                detection_fraction=detection_fraction,
                detection_span_px=float(np.linalg.norm(np.ptp(centroids, axis=0)))
                if centroids
                else 0.0,
            )
            result["success"] &= (
                valid_frames and detection_fraction >= cfg["minimum_detection_fraction"]
            )
        return result
