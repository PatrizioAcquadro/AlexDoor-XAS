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
    rows, violations = [], []
    try:
        for tick in range(round(duration_s / env.step_dt)):
            env.step(torch.zeros((1, env.cfg.action_space), device=env.device))
            position, quaternion = [array(v)[0] for v in env.tool_pose()]
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
        np.savez_compressed(output, values=np.asarray(rows), stopped_targets=targets)
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
