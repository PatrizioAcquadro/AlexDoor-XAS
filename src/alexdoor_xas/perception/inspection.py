"""Common observed inspection motion; door truth is used only in the offline audit."""

import json

import numpy as np


def load_inspection(path):
    config = json.loads(path.read_text())
    points = np.asarray(config["waypoints"], dtype=float)
    times = np.asarray(config["sample_times_s"], dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points[0, 0] != 0
        or not np.isfinite(points).all()
        or not (np.diff(points[:, 0]) > 0).all()
        or not (np.diff(times) > 0).all()
        or times[0] <= 0
        or times[-1] > points[-1, 0]
    ):
        raise ValueError("Invalid inspection trajectory or sample times")
    speed = np.abs(np.diff(points[:, 1:], axis=0) / np.diff(points[:, :1], axis=0))
    if speed.max() > config["max_neck_speed_rad_s"]:
        raise ValueError("Inspection exceeds the common neck speed")
    if np.abs(points[:, 1]).max() > 1.22173 or np.abs(points[:, 2]).max() > 0.488692:
        raise ValueError("Inspection exceeds physical neck limits")
    return config


def tilt_mount(stage, joint_path, pitch_rad):
    """Rotate the rigid camera mount about HEAD_LINK Y; preserve its origin and mass."""
    from pxr import Gf, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation

    if not np.isfinite(pitch_rad) or abs(pitch_rad) > np.deg2rad(15):
        raise ValueError("Camera mount study is bounded to 15 degrees")
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(joint_path))
    head = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
    body = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
    q = joint.GetLocalRot0Attr().Get()
    rotation = Rotation.from_euler("y", pitch_rad) * Rotation.from_quat(
        [*q.GetImaginary(), q.GetReal()]
    )
    xyzw = rotation.as_quat()
    joint.GetLocalRot0Attr().Set(Gf.Quatf(float(xyzw[3]), Gf.Vec3f(*xyzw[:3])))
    mount = np.eye(4)
    mount[:3, :3] = rotation.as_matrix()
    mount[:3, 3] = joint.GetLocalPos0Attr().Get()
    head_world = np.array(UsdGeom.Xformable(head).ComputeLocalToWorldTransform(0)).T
    parent_world = np.array(UsdGeom.Xformable(body.GetParent()).ComputeLocalToWorldTransform(0)).T
    local = np.linalg.inv(parent_world) @ head_world @ mount
    xform = UsdGeom.Xformable(body)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(Gf.Matrix4d(local.T.tolist()))


def run_inspection(env, config, recorder=None):
    """Hold the parked arm while scanning; abort on contacts, motion or tracking errors."""
    import torch
    from scipy.spatial.transform import Rotation

    from alexdoor_xas.recording.b1_runtime import array

    points = np.asarray(config["waypoints"])
    p, q = [array(v)[0] for v in env.tool_pose()]
    rotation = Rotation.from_quat(q).as_matrix()
    zero = torch.zeros((1, 6), device=env.device)
    maximum = dict(tool_drift_m=0.0, door_angle_rad=0.0, neck_error_rad=0.0)
    for i in range(1, round(points[-1, 0] / env.step_dt) + 1):
        time_s = i * env.step_dt
        neck = np.array([np.interp(time_s, points[:, 0], points[:, j]) for j in (1, 2)])
        env.set_neck_target(neck)
        env.command_pose(p, q)
        env.step(zero)
        drift = float(np.linalg.norm(array(env.tool_pose()[0])[0] - p))
        angle = float(np.abs(array(env.door.data.joint_pos)[0, 0]))
        neck_error = float(np.max(np.abs(array(env.robot.data.joint_pos)[0, env.neck_ids] - neck)))
        if (
            not np.isfinite([drift, angle, neck_error]).all()
            or drift > 0.01
            or angle > 0.01
            or neck_error > 0.1
            or any(s["forbidden"] for s in env.contact_history)
        ):
            raise RuntimeError(f"Unsafe inspection: drift={drift}, door={angle}, neck={neck_error}")
        for key, value in zip(maximum, (drift, angle, neck_error), strict=True):
            maximum[key] = max(maximum[key], value)
        if recorder is not None:
            recorder.transition(env, p, rotation, "inspect", dict(valid=True))
    return dict(duration_s=float(points[-1, 0]), **maximum)
