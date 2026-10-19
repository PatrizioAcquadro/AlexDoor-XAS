"""Task-specific loaded hold reconstruction, followed by fresh physical qualification."""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from alexdoor_xas.recording.b1_runtime import array


def save_hold_state(path, env, *, compression, compensation, retreat_p, retreat_r, source):
    gains = [
        (array(a.stiffness)[:, c], array(a.damping)[:, c]) for a, c, _, _ in env._motion_impedance
    ]
    np.savez_compressed(
        path,
        q=array(env.robot.data.joint_pos),
        dq=array(env.robot.data.joint_vel),
        targets=array(env.targets),
        door_q=array(env.door.data.joint_pos),
        door_dq=array(env.door.data.joint_vel),
        stiffness=np.array([v[0] for v in gains]),
        damping=np.array([v[1] for v in gains]),
        tool_position=array(env.tool_pose()[0])[0],
        tool_quaternion=array(env.tool_pose()[1])[0],
        compression=compression,
        compensation=compensation,
        retreat_p=retreat_p,
        retreat_r=retreat_r,
        metadata=json.dumps(source),
    )


def restore_hold_state(env, state, setup):
    """Restore all moving joints, panel velocity, targets and task impedance.

    Solver warm-start/contact caches are deliberately rebuilt by actual physics;
    the caller must acquire a new sustained loaded window before continuation.
    """
    import torch

    def t(key):
        return torch.as_tensor(state[key], dtype=torch.float32, device=env.device)

    env.robot.write_joint_position_to_sim_index(position=t("q"))
    env.robot.write_joint_velocity_to_sim_index(velocity=t("dq"))
    env.door.write_joint_position_to_sim_index(position=t("door_q"))
    env.door.write_joint_velocity_to_sim_index(velocity=t("door_dq"))
    env.targets = t("targets").clone()
    env.robot.set_joint_position_target_index(target=env.targets)
    env.robot.set_joint_velocity_target_index(target=torch.zeros_like(env.targets))
    for i, (actuator, column, _, _) in enumerate(env._motion_impedance):
        actuator.stiffness[:, column] = t("stiffness")[i]
        actuator.damping[:, column] = t("damping")[i]
    env.sim.forward()  # refresh articulation/collision transforms before camera read
    env.capture.reset()
    env.load_samples = []
    env._get_observations()  # fresh local t=0 image; no source prefix
    p, q = [array(v)[0] for v in env.tool_pose()]
    errors = dict(
        joint_position_rad=float(np.max(np.abs(array(env.robot.data.joint_pos) - state["q"]))),
        joint_velocity_rad_s=float(np.max(np.abs(array(env.robot.data.joint_vel) - state["dq"]))),
        door_angle_rad=float(np.max(np.abs(array(env.door.data.joint_pos) - state["door_q"]))),
        tool_position_m=float(np.linalg.norm(p - state["tool_position"])),
        tool_rotation_rad=float(
            (Rotation.from_quat(q) * Rotation.from_quat(state["tool_quaternion"]).inv()).magnitude()
        ),
    )
    bounds = dict(
        joint_position_rad=0.01,
        joint_velocity_rad_s=0.05,
        door_angle_rad=np.deg2rad(setup.tie_deg),
        tool_position_m=setup.position_tolerance,
        tool_rotation_rad=setup.orientation_tolerance,
    )
    if any(not np.isfinite(v) or v > bounds[key] for key, v in errors.items()):
        raise ValueError(f"intermediate_restore_outside_task_tolerances: {errors}")
    return dict(
        errors=errors,
        tolerances=bounds,
        solver_state="reconstructed; requires fresh sustained physical contact qualification",
    )
