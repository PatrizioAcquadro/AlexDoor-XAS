"""Isaac-side observer for B1 recording; privileged labels remain separate."""

import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from alexdoor_xas.assets.purdue import ARM_JOINTS, NECK_JOINTS, origin_matrix
from alexdoor_xas.recording.b1 import PHASES, B1Writer


def array(value):
    import torch
    import warp as wp

    value = value.torch if hasattr(value, "torch") else value
    if isinstance(value, wp.array):
        value = wp.to_torch(value)
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def joint_si_units(urdf, names):
    """Native robot diagnostics include rotational joints and prismatic WSG jaws."""
    kinds = {
        joint.get("name"): joint.get("type") for joint in ET.parse(urdf).getroot().findall("joint")
    }
    selected = [kinds[name] for name in names]
    if any(kind not in ("revolute", "continuous", "prismatic") for kind in selected):
        raise ValueError("Unknown active-joint units")
    return dict(
        position=["m" if kind == "prismatic" else "rad" for kind in selected],
        velocity=["m/s" if kind == "prismatic" else "rad/s" for kind in selected],
        effort=["N" if kind == "prismatic" else "N m" for kind in selected],
    )


def annotate_native_units(file, urdf, names):
    """Add explicit SI units to diagnostics; never change samples or outcomes.

    The legacy effort_target_nm container holds full native feedforward effort,
    including force for prismatic jaws. It is separate from measured arm7 tau.
    """
    units = joint_si_units(urdf, names)
    for field, key in (("full_joint_position", "position"), ("full_joint_velocity", "velocity")):
        path = "control/" + field
        if path in file:
            file[path].attrs.update(
                joint_names=json.dumps(list(names)), units=json.dumps(units[key])
            )
    if "commands/effort_target_nm" in file:
        file["commands/effort_target_nm"].attrs.update(
            joint_names=json.dumps(list(names)),
            units=json.dumps(units["effort"]),
            semantics="native_feedforward_effort_not_measured_torque; legacy_container_name",
        )
    file["metadata"].attrs["full_joint_units"] = json.dumps(units)


def camera_calibration(env):
    """Serialize fixed mount + URDF neck chain, never a per-door camera reset cache."""
    from pxr import UsdGeom

    root = ET.parse(env.robot.cfg.spawn.asset_path).getroot()
    parents = {j.find("child").get("link"): j for j in root.findall("joint")}
    joints, link = [], "HEAD_LINK"
    while link in parents:
        joint = parents[link]
        name = joint.get("name")
        if joint.get("type") != "fixed" and name not in NECK_JOINTS:
            raise ValueError(f"Unobserved active camera ancestor: {name}")
        joints.append(
            dict(
                origin=origin_matrix(joint.find("origin")).tolist(),
                index=NECK_JOINTS.index(name) if name in NECK_JOINTS else None,
                axis=np.fromstring(joint.find("axis").get("xyz"), sep=" ").tolist()
                if name in NECK_JOINTS
                else [0, 0, 0],
            )
        )
        link = joint.find("parent").get("link")
    stage = env.sim.stage
    # The authored rigid mount is constant, independent of door identity/state.
    head = next(p for p in stage.Traverse() if p.GetName() == "HEAD_LINK")
    camera = stage.GetPrimAtPath(env.camera.cfg.prim_path)
    head_w = np.array(UsdGeom.Xformable(head).ComputeLocalToWorldTransform(0)).T
    camera_w = np.array(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(0)).T
    head_camera = np.linalg.inv(head_w) @ camera_w
    head_camera[:3, :3] /= np.linalg.norm(head_camera[:3, :3], axis=0)
    head_camera = head_camera @ np.diag([1, -1, -1, 1])  # USD optical -> ROS optical
    state = array(env.robot.data.default_root_state)[0]
    root_world = np.eye(4)
    root_world[:3, :3] = Rotation.from_quat(state[3:7]).as_matrix()
    root_world[:3, 3] = state[:3]
    return dict(
        joints=list(reversed(joints)),
        head_camera=head_camera.tolist(),
        root_world=root_world.tolist(),
        joint_names=list(ARM_JOINTS + NECK_JOINTS),
        depth_interval_m=[env.capture.near_m, env.capture.far_m],
        pixel_center_offset_xy=[0.5, 0.5],  # RTX sample centers; raw SDK K is retained.
        optical_convention="ROS",
        depth_kind="distance_to_image_plane_m",
    )


def camera_from_joints(calibration, q):
    transform = np.asarray(calibration["root_world"]).copy()
    for joint in calibration["joints"]:
        transform = transform @ np.asarray(joint["origin"])
        if joint["index"] is not None:
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(
                np.asarray(joint["axis"]) * q[7 + joint["index"]]
            ).as_matrix()
            transform = transform @ motion
    return transform @ np.asarray(calibration["head_camera"])


def record_robot_state(writer, env):
    """Retain actual calibrated tool-link pose and full robot joints, without door quantities."""
    position, quaternion = [array(v)[0] for v in env.tool_pose()]
    writer.robot_state(
        env.capture.sample.time_s,
        joint_names=env.robot.joint_names,
        q=array(env.robot.data.joint_pos)[0],
        dq=array(env.robot.data.joint_vel)[0],
        tool_position=position,
        tool_rotation=Rotation.from_quat(quaternion).as_matrix(),
    )


class ExpertRecorder:
    def __init__(
        self, path, metadata, *, purpose="perception_engineering_not_matched_policy_dataset"
    ):
        if purpose not in (
            "perception_engineering_not_matched_policy_dataset",
            "b1_partial_policy_pilot",
        ):
            raise ValueError("Unsupported expert recording purpose")
        self.purpose = purpose
        self.path, self.metadata = path, metadata
        self.writer = None
        self.calibration = None
        self.max_camera_error_m = 0.0
        self.max_camera_error_rad = 0.0

    def observation(self, env):
        sample = env.capture.sample
        q = array(sample.joint_position)[0]
        camera = camera_from_joints(self.calibration, q)
        data = env.camera.data
        p, quat = array(data.pos_w)[0], array(data.quat_w_ros)[0]
        pe = float(np.linalg.norm(camera[:3, 3] - p))
        re = float(
            Rotation.from_matrix(
                camera[:3, :3].T @ Rotation.from_quat(quat).as_matrix()
            ).magnitude()
        )
        self.max_camera_error_m = max(self.max_camera_error_m, pe)
        self.max_camera_error_rad = max(self.max_camera_error_rad, re)
        if pe > 0.002 or re > 0.005:
            raise ValueError(f"Camera FK mismatch: {pe} m, {re} rad")
        return dict(
            time_s=np.float64(sample.time_s),
            frame=np.int64(sample.frame),
            rgb=array(sample.rgb)[0],
            depth_m=array(sample.depth_m)[0],
            valid_depth=array(sample.valid_depth)[0],
            joint_position=q,
            joint_velocity=array(sample.joint_velocity)[0],
            camera_world=camera,
            intrinsics=array(data.intrinsic_matrices)[0],
        )

    def annotation(self, env, phase, valid=True):
        angle = float(array(env.door.data.joint_pos)[0, 0])
        position, rotation = self.door.contact_pose(
            angle, self.setup.contact_fraction, self.setup.contact_height
        )
        closed, closed_rotation = self.door.contact_pose(
            0, self.setup.contact_fraction, self.setup.contact_height
        )
        return dict(
            hinge_origin=self.door.hinge.astype(np.float32),
            hinge_rotation=np.eye(3, dtype=np.float32),
            signed_angle=np.float32(self.door.sign * angle),
            dimensions=np.asarray(
                [self.door.width, self.door.height, self.door.thickness], np.float32
            ),
            contact_local=(closed - self.door.hinge).astype(np.float32),
            contact_rotation_local=closed_rotation.astype(np.float32),
            contact_position=position.astype(np.float32),
            contact_rotation=rotation.astype(np.float32),
            phase=np.int8(PHASES.index(phase)),
            physical_valid=np.bool_(valid),
        )

    def start(self, env, door, setup, phase="approach"):
        self.door, self.setup = door, setup
        if self.calibration is None:
            raise RuntimeError("Prepare camera calibration before the first reset")
        self.metadata.update(
            control_dt=env.step_dt,
            sim_dt=env.cfg.sim.dt,
            purpose=self.purpose,
        )
        self.writer = B1Writer(self.path, self.metadata, self.calibration)
        from alexdoor_xas.perception.load import simulated_feedback

        self.writer.observe(
            self.observation(env), self.annotation(env, phase), feedback=simulated_feedback(env)
        )

        record_robot_state(self.writer, env)

    def transition(self, env, goal_p, goal_r, phase, trace):
        command = dict(
            time_s=np.float64(self.writer.last_time),
            tool_position=np.asarray(goal_p),
            tool_rotation=np.asarray(goal_r),
            joint_target=array(env.targets)[0, env.arm_ids],
            neck_target=array(env.targets)[0, env.neck_ids],
            effort_target_nm=array(env.robot.data.joint_effort_target)[0, env.observed_ids],
            phase=np.int8(PHASES.index(phase)),
        )
        from alexdoor_xas.perception.load import simulated_feedback

        self.writer.transition(
            command,
            self.observation(env),
            self.annotation(env, phase, trace["valid"]),
            feedback=simulated_feedback(env),
        )
        if env.load_samples:
            self.writer.physics_feedback(env.load_samples)
        record_robot_state(self.writer, env)

    def finish(self, result):
        self.writer.file["metadata"].attrs["camera_fk_error"] = json.dumps(
            dict(position_m=self.max_camera_error_m, rotation_rad=self.max_camera_error_rad)
        )
        if self.purpose == "b1_partial_policy_pilot":
            result = dict(result, termination=result["stop_reason"])
        self.writer.finish(result)

    def close(self):
        if self.writer is not None:
            self.writer.close()
