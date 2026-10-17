"""Passive measured-stop continuation, separate from policy/teacher commands."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def verify_stop_continuation(env, output, *, duration_s=2.0, force_limit_n=80.0):
    """Latch once and continue actual physics; save partial evidence on failure."""
    import torch

    from alexdoor_xas.policies.purdue import contact_status
    from alexdoor_xas.recording.b1_runtime import array

    output = Path(output)
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Invalid continuing-physics duration")
    env.stop_motion()
    anchor, orientation = [array(v)[0].copy() for v in env.tool_pose()]
    rotation = Rotation.from_quat(orientation).as_matrix()
    targets = array(env.targets).copy()
    anchor_joints = array(env.robot.data.joint_pos)[0].copy()
    anchor_time_s = float(env._time_s)
    joints, positions, quaternions, native, evaluator = [], [], [], [], []
    rows, violations = [], []
    try:
        for tick in range(round(duration_s / env.step_dt)):
            env.step(torch.zeros((1, env.cfg.action_space), device=env.device))
            position, quaternion = [array(v)[0] for v in env.tool_pose()]
            joints.append(array(env.robot.data.joint_pos)[0].copy())
            positions.append(position.copy())
            quaternions.append(quaternion.copy())
            for sample, contact in zip(env.load_samples, env.contact_history, strict=True):
                fb = sample.feedback
                native.append(
                    dict(
                        time_s=fb.timestamp_s,
                        available_s=fb.available_s,
                        joint_position=sample.q.copy(),
                        joint_velocity=sample.dq.copy(),
                        joint_acceleration=sample.ddq.copy(),
                        torque_nm=np.full(7, np.nan)
                        if fb.torque_nm is None
                        else np.array(fb.torque_nm),
                        feedback_present=fb.torque_nm is not None,
                        feedback_healthy=fb.healthy,
                        expected_torque_nm=sample.expected_torque_nm.copy(),
                        jacobian_world=sample.jacobian_world.copy(),
                    )
                )
                intended = [
                    c for c in contact["contacts"] if c["category"] in ("positive", "negative")
                ]
                evaluator.append(
                    dict(
                        time_s=fb.timestamp_s,
                        forbidden=contact["forbidden"],
                        normal_force_n=sum(c["force_n"] for c in intended),
                        minimum_gap_m=min((c["separation_m"] for c in intended), default=np.nan),
                    )
                )
            dq = np.stack([sample.dq for sample in env.load_samples])
            q = array(env.robot.data.joint_pos)[0, env.arm_ids]
            limits = array(env.limits)[0]
            reason, force, _ = contact_status(env.contact_history, force_limit_n)
            if reason:
                violations.append(reason)
            if np.max(np.abs(dq)) > 0.51:
                violations.append("actual_joint_speed_limit")
            if np.any(q < limits[:, 0] - 0.005) or np.any(q > limits[:, 1] + 0.005):
                violations.append("joint_limit_violation")
            if not np.array_equal(array(env.targets), targets):
                violations.append("stop_target_changed")
            rows.append(
                (
                    (tick + 1) * env.step_dt,
                    np.linalg.norm(position - anchor),
                    Rotation.from_matrix(
                        Rotation.from_quat(quaternion).as_matrix() @ rotation.T
                    ).magnitude(),
                    np.max(np.abs(dq)),
                    np.max(np.abs(dq[-1])),
                    force,
                )
            )
    finally:
        streams = (
            {"native_" + key: np.asarray([x[key] for x in native]) for key in native[0]}
            if native
            else {}
        )
        streams.update(
            {"evaluator_" + key: np.asarray([x[key] for x in evaluator]) for key in evaluator[0]}
            if evaluator
            else {}
        )
        np.savez_compressed(
            output,
            values=np.asarray(rows),
            stopped_targets=targets,
            anchor_time_s=anchor_time_s,
            anchor_joint_position=anchor_joints,
            joint_names=np.asarray(env.robot.joint_names),
            joint_position=np.asarray(joints),
            tool_position=np.asarray(positions),
            tool_quaternion=np.asarray(quaternions),
            **streams,
        )
    values = np.asarray(rows)
    passed = bool(
        len(rows) == round(duration_s / env.step_dt)
        and np.isfinite(values).all()
        and not violations
        and values[:, 1].max() <= 0.01
        and values[:, 2].max() <= np.deg2rad(3)
        and values[-1, 4] <= 0.01
    )
    return dict(
        passed=passed,
        duration_s=len(rows) * env.step_dt,
        position_travel_max_m=float(values[:, 1].max()),
        rotation_travel_max_rad=float(values[:, 2].max()),
        actual_joint_speed_peak_rad_s=float(values[:, 3].max()),
        final_joint_speed_rad_s=float(values[-1, 4]),
        normal_force_mean_tick_peak_n=float(values[:, 5].max()),
        violations=sorted(set(violations)),
        scope="Measured continuing-physics stop; no corrective or release trajectory",
    )
