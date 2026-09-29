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


class ExpertRecorder:
    def __init__(self, path, metadata):
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
            purpose="perception_engineering_not_matched_policy_dataset",
        )
        self.writer = B1Writer(self.path, self.metadata, self.calibration)
        self.writer.observe(self.observation(env), self.annotation(env, phase))

    def transition(self, env, goal_p, goal_r, phase, trace):
        command = dict(
            time_s=np.float64(self.writer.last_time),
            tool_position=np.asarray(goal_p),
            tool_rotation=np.asarray(goal_r),
            joint_target=array(env.targets)[0, env.arm_ids],
            neck_target=array(env.targets)[0, env.neck_ids],
            phase=np.int8(PHASES.index(phase)),
        )
        self.writer.transition(
            command, self.observation(env), self.annotation(env, phase, trace["valid"])
        )

    def finish(self, result):
        self.writer.file["metadata"].attrs["camera_fk_error"] = json.dumps(
            dict(position_m=self.max_camera_error_m, rotation_rad=self.max_camera_error_rad)
        )
        self.writer.finish(result)

    def close(self):
        if self.writer is not None:
            self.writer.close()
