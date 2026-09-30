"""Purdue IO with legacy physical stops or observed-only prototype checks."""

import numpy as np
from scipy.spatial.transform import Rotation

from alexdoor_xas.action.b1 import apply_pose_delta, finite_vector
from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.assets.purdue import ARM_JOINTS, NECK_JOINTS
from alexdoor_xas.qualification.synthetic_probe import ContactLoad


def contact_status(history, force_limit):
    """Reuse exact distal/panel classifications at every physics substep."""
    if not history:
        return "missing_contact_diagnostics", 0.0, None
    forces, gaps = [], []
    for sample in history:
        if sample.get("components") != "normal_only":
            return "unsupported_contact_diagnostics", 0.0, None
        if sample["forbidden"]:
            return "forbidden_contact", 0.0, None
        contacts = [c for c in sample["contacts"] if c["category"] in ("positive", "negative")]
        values = [v for c in sample["contacts"] for v in (c["force_n"], c["separation_m"])]
        if not np.isfinite(values).all():
            return "invalid_contact_diagnostics", 0.0, None
        forces.append(sum(abs(c["force_n"]) for c in contacts))
        gaps.append(min((c["separation_m"] for c in contacts), default=None))
    if max(forces) > force_limit:
        return "force_limit", float(np.mean(forces)), None
    gap = max(gaps) if all(g is not None for g in gaps) else None
    return "", float(np.mean(forces)), gap


class PurdueFK:
    """Observed joint FK with the same derived distal tool offset as Phase 4."""

    def __init__(self, urdf, device="cuda:0"):
        from alexdoor_xas.kinematics.purdue_chain import PurdueChain

        self.chain = PurdueChain(urdf, device=device)

    def __call__(self, joints, calibration):
        import torch

        joints = finite_vector(joints, 9)
        with torch.no_grad():
            transform, _ = self.chain.forward(self.chain.array(joints[:7])[None])
        world = np.asarray(calibration["root_world"]) @ transform[0].cpu().numpy()
        return ObjectFrame(world[:3, 3], world[:3, :3])


class PurdueSafety:
    """Physical stops only. No expert point, angle target or corrective commands.

    Contact quality is evaluated at the actual tool location, allowing a policy
    to choose a different safe panel point from the qualification teacher.
    """

    def __init__(self, env, door, setup):
        self.env, self.door, self.setup = env, door, setup
        names = [
            f"{side}_WSG32_{joint}"
            for side in ("left", "right")
            for joint in ("JAW_OPENING", "JAW_FOLLOWER")
        ]
        self.gripper_ids, found = env.robot.find_joints(names, preserve_order=True)
        if tuple(found) != tuple(names):
            raise ValueError("Expected canonical closed finger joints")
        self.reset()

    def reset(self):
        self.load = ContactLoad(
            self.setup.contact_force_window_s,
            self.setup.loaded_force_n,
            self.setup.contact_gap_tolerance_m,
        )
        self.last_time = None
        self.loaded = False
        self.initial_tool = None
        self.initial_angle = None

    def check(self, stage):
        from alexdoor_xas.recording.b1_runtime import array

        env, setup, door = self.env, self.setup, self.door
        q = array(env.robot.data.joint_pos)[0]
        limits = array(env.limits)[0]
        tool_p, tool_q = [array(v)[0] for v in env.tool_pose()]
        angle = float(array(env.door.data.joint_pos)[0, 0])
        if not np.isfinite(np.r_[q, tool_p, tool_q, angle]).all():
            return "invalid_physics"
        if np.any(q[env.arm_ids] < limits[:, 0] - 0.005) or np.any(
            q[env.arm_ids] > limits[:, 1] + 0.005
        ):
            return "joint_limit_violation"
        if np.abs(q[self.gripper_ids]).max() > 0.001:
            return "gripper_opened"
        reason, force, gap = contact_status(env.contact_history, setup.force_limit_n)
        if reason:
            return reason
        t = env.capture.sample.time_s
        if self.last_time != t:
            self.loaded = self.load.update(t, force, gap)
            self.last_time = t
        if self.initial_tool is None:
            self.initial_tool, self.initial_angle = tool_p.copy(), angle
        if stage == "inspect":
            neck_error = np.abs(q[env.neck_ids] - array(env.targets)[0, env.neck_ids]).max()
            if (
                np.linalg.norm(tool_p - self.initial_tool) > 0.01
                or abs(angle - self.initial_angle) > 0.01
                or neck_error > 0.1
            ):
                return "unsafe_inspection"
        rotation = Rotation.from_quat(tool_q).as_matrix()
        if self.loaded:
            footprint = np.concatenate(env.push_geometry.distal_faces) @ rotation.T + tool_p
            if not door.footprint_inside(footprint, angle):
                return "contact_outside_panel"
            panel_rotation = door.rotation(angle)
            closed_tool = panel_rotation.T @ (tool_p - door.hinge) + door.hinge
            try:
                front_x, outward = door.front_surface(float(closed_tool[1]), float(closed_tool[2]))
            except ValueError:
                return "contact_outside_panel"
            forward = panel_rotation @ (-outward / np.linalg.norm(outward))
            error = np.arccos(np.clip(np.dot(rotation[:, 0], forward), -1, 1))
            if error > setup.orientation_tolerance:
                return "contact_orientation"
            if abs(closed_tool[0] - front_x) > setup.position_tolerance:
                return "contact_position"
        if stage in ("push", "hold") and not self.loaded:
            return "uncontrolled_contact"
        return ""


class PurdueIO:
    """Adapt an already-created, camera-enabled Purdue environment; never launch one."""

    def __init__(self, env, *, binding, robot_asset, setup, observed_provider=None):
        from alexdoor_xas.recording.b1_runtime import camera_calibration

        if env.cfg.action_mode != "A2" or not env.cfg.cameras or env.cfg.prepared_door is None:
            raise ValueError("B1 rollout requires a prepared-door A2 Purdue environment with RGB-D")
        if not str(env.device).startswith("cuda"):
            raise ValueError("Purdue B1 execution requires CUDA")
        if not np.isclose(
            env.cfg.camera_mount_pitch_rad, binding.config["inspection"]["mount_pitch_rad"]
        ):
            raise ValueError("Runtime camera mount differs from frozen perception")
        if tuple(env.cfg.floor_pose) != tuple(setup.floor_pose) or any(
            getattr(env.cfg, key) != getattr(setup, key)
            for key in ("max_joint_speed", "centering_gain", "ik_damping")
        ):
            raise ValueError("Runtime differs from the common setup/controller")
        if any(env.cfg.initial_joints[k] != v for k, v in setup.initial_joints.items()):
            raise ValueError("Runtime ready pose differs from the common setup")
        self.env, self.binding, self.robot_asset = env, binding, robot_asset
        self.dt = env.step_dt
        self.max_ticks = int(round(setup.horizon_s / self.dt))
        self.calibration = camera_calibration(env)
        if self.calibration["joint_names"] != list(ARM_JOINTS + NECK_JOINTS):
            raise ValueError("Runtime joint order differs")
        self.tool_fk = PurdueFK(env.robot.cfg.spawn.asset_path, str(env.device))
        if observed_provider is None:
            self.safety = PurdueSafety(env, env.cfg.prepared_door, setup)
        else:
            from alexdoor_xas.perception.control import ObservedPurdueSafety

            self.safety = ObservedPurdueSafety(self, setup, observed_provider)

    def reset(self):
        self.env.reset()
        self.safety.reset()
        self.parked_tool = self.tool_pose()

    def observe(self):
        import torch

        from alexdoor_xas.recording.b1_runtime import array, camera_from_joints

        env = self.env
        sample = env.capture.sample
        q = sample.joint_position[0]
        camera = camera_from_joints(self.calibration, array(q))
        return dict(
            time_s=sample.time_s,
            frame=sample.frame,
            rgb=sample.rgb[0],
            depth_m=sample.depth_m[0],
            valid_depth=sample.valid_depth[0],
            joint_position=q,
            joint_velocity=sample.joint_velocity[0],
            camera_world=torch.as_tensor(camera, device=env.device, dtype=torch.float32),
            intrinsics=torch.as_tensor(
                array(env.camera.data.intrinsic_matrices)[0], device=env.device, dtype=torch.float32
            ),
        )

    def tool_pose(self):
        from alexdoor_xas.recording.b1_runtime import array

        return self.tool_fk(array(self.env.capture.sample.joint_position)[0], self.calibration)

    def set_neck_target(self, value):
        self.env.set_neck_target(value)

    def hold_step(self):
        import torch

        env = self.env
        env.command_pose(
            self.parked_tool.origin, Rotation.from_matrix(self.parked_tool.rot).as_quat()
        )
        env.step(torch.zeros((1, 6), device=env.device))

    def execute(self, command):
        if before := getattr(self.safety, "before_command", None):
            if reason := before(command.stage):
                raise RuntimeError(reason)
        import torch

        env = self.env
        if command.kind == "joints":
            env.step_a1(command.value)
        elif command.kind in ("delta", "pose"):
            # Recorded teacher goals use command_pose. Reconstruct that same goal
            # rather than applying the raw Gym delta's additional component clips.
            # A2/A3/A4 share the existing pose IK and bounded seven-joint targets.
            goal = (
                apply_pose_delta(self.tool_pose(), command.value)
                if command.kind == "delta"
                else command.value
            )
            env.command_pose(goal.origin, Rotation.from_matrix(goal.rot).as_quat())
            env.step(torch.zeros((1, 6), device=env.device))
        else:
            raise ValueError("Unknown execution command")

    def safety_reason(self, stage):
        return self.safety.check(stage)

    def stop(self):
        # Do not add a release, IK correction or extra physics step on failure.
        self.env._pending_pose = None
        self.env._pending_joints = None
        self.env.robot.set_joint_position_target_index(target=self.env.targets)
