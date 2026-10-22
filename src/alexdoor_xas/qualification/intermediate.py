"""Source phase reconstruction with fresh physical qualification and legacy compatibility."""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from alexdoor_xas.recording.b1_runtime import array


def save_hold_state(
    path, env, *, compression, compensation, retreat_p, retreat_r, source, initial_lead
):
    np.savez_compressed(
        path,
        **capture_phase_state(
            env,
            phase="hold",
            compression=compression,
            compensation=compensation,
            retreat_p=retreat_p,
            retreat_r=retreat_r,
            source=source,
            initial_lead=initial_lead,
        ),
    )


def capture_phase_state(
    env,
    *,
    phase,
    compression,
    compensation,
    retreat_p,
    retreat_r,
    source,
    initial_lead,
    reference_angle=0.0,
):
    """Complete native state; contact caches require new physical evolution."""
    gains = [
        (array(a.stiffness)[:, c], array(a.damping)[:, c]) for a, c, _, _ in env._motion_impedance
    ]
    state = dict(
        phase=phase,
        reference_angle=reference_angle,
        q=array(env.robot.data.joint_pos),
        dq=array(env.robot.data.joint_vel),
        targets=array(env.targets),
        door_q=array(env.door.data.joint_pos),
        door_dq=array(env.door.data.joint_vel),
        stiffness=np.array([v[0] for v in gains]),
        damping=np.array([v[1] for v in gains]),
        tool_position=array(env.tool_pose()[0])[0],
        tool_quaternion=array(env.tool_pose()[1])[0],
        initial_lead=initial_lead,
        source_generation=env.generation,
        compression=compression,
        compensation=compensation,
        retreat_p=retreat_p,
        retreat_r=retreat_r,
        metadata=json.dumps(source),
    )

    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in state.items()
    }


def restore_hold_state(env, state, setup, audit_path, *, continue_controller=False):
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
    env.sim.forward()
    env.capture.reset()
    conditioning = []
    anchor = env.targets[:, env.arm_ids].clone()
    door = env.cfg.prepared_door if continue_controller else None
    phase = str(state.get("phase", "hold"))
    compression = float(state["compression"])
    compensation = np.array(state["compensation"], copy=True)
    reference = float(state.get("reference_angle", 0.0))
    initial_lead = float(state["initial_lead"])
    force = 0.0
    contact_gap = None
    reconstruction = dict(
        continue_controller=continue_controller,
        gain_joint_names=[a.joint_names[c] for a, c, _, _ in env._motion_impedance],
        stiffness=[array(a.stiffness)[:, c].tolist() for a, c, _, _ in env._motion_impedance],
        damping=[array(a.damping)[:, c].tolist() for a, c, _, _ in env._motion_impedance],
        controller_input=dict(
            phase=phase,
            compression=compression,
            compensation=compensation.tolist(),
            reference_angle=reference,
            initial_lead=initial_lead,
        ),
    )
    from alexdoor_xas.qualification.synthetic_probe import (
        hold_reference,
        normal_contact_support,
        push_reference,
        tangential_compensation,
    )

    for _ in range(6):
        measured = torch.as_tensor(
            array(env.robot.data.joint_pos)[:, env.arm_ids], dtype=torch.float32, device=env.device
        )
        if continue_controller:
            angle = float(array(env.door.data.joint_pos)[0, 0])
            compression = normal_contact_support(
                compression,
                force,
                setup,
                env.step_dt,
                target_force=setup.hold_force_n if phase == "hold" else setup.contact_load_guard_n,
                gap_m=contact_gap,
            )
            if phase == "push":
                reference = push_reference(
                    reference, angle, setup, env.step_dt, door.mechanical_stop
                )
            else:
                reference = hold_reference(angle, initial_lead, door.mechanical_stop)
            goal_p, goal_r = door.contact_pose(
                reference,
                setup.contact_fraction,
                setup.contact_height,
                compression,
            )
            goal_p = goal_p + goal_r @ compensation
            env.command_pose(goal_p, Rotation.from_matrix(goal_r).as_quat())
            env.step(torch.zeros((1, 6), device=env.device))
        else:
            env.step_a1(anchor - measured)
        force = max(
            sum(c["force_n"] for c in row["contacts"] if c["category"] in ("positive", "negative"))
            for row in env.contact_history
        )
        gaps = [
            min(
                (
                    c["separation_m"]
                    for c in row["contacts"]
                    if c["category"] in ("positive", "negative")
                ),
                default=None,
            )
            for row in env.contact_history
        ]
        contact_gap = max(gaps) if gaps and all(g is not None for g in gaps) else None
        speed = max(float(np.max(abs(sample.dq))) for sample in env.load_samples)
        if continue_controller:
            actual_p = array(env.tool_pose()[0])[0]
            material_p, material_r = door.contact_pose(
                float(array(env.door.data.joint_pos)[0, 0]),
                setup.contact_fraction,
                setup.contact_height,
            )
            compensation = tangential_compensation(
                compensation,
                (material_p - actual_p) @ material_r,
                force >= setup.loaded_force_n,
                setup,
                env.step_dt,
            )
        conditioning.append(
            dict(
                time_s=float(env._time_s),
                source_target_delta=(anchor - measured).cpu().tolist(),
                commanded_tool_position=goal_p.tolist() if continue_controller else None,
                commanded_tool_quaternion=Rotation.from_matrix(goal_r).as_quat().tolist()
                if continue_controller
                else None,
                phase_controller=phase if continue_controller else "legacy_joint_anchor",
                delivered_target=array(env.targets)[:, env.arm_ids].tolist(),
                joint_position=array(env.robot.data.joint_pos).tolist(),
                joint_velocity=array(env.robot.data.joint_vel).tolist(),
                door_position=array(env.door.data.joint_pos).tolist(),
                door_velocity=array(env.door.data.joint_vel).tolist(),
                force_n=force,
                native_force_n=[
                    sum(
                        c["force_n"]
                        for c in row["contacts"]
                        if c["category"] in ("positive", "negative")
                    )
                    for row in env.contact_history
                ],
                native_gap_m=gaps,
                joint_speed_rad_s=speed,
                torque_nm=[sample.feedback.torque_nm.tolist() for sample in env.load_samples],
            )
        )
        audit_path.write_text(
            json.dumps(
                dict(
                    scope="real_reset_conditioning_excluded_from_new_observation_history",
                    generation=env.generation,
                    anchor_target=anchor.cpu().tolist(),
                    reconstruction=reconstruction,
                    measured_steps=conditioning,
                ),
                indent=2,
            )
            + "\n"
        )
        if (
            any(row["forbidden"] for row in env.contact_history)
            or force > setup.force_limit_n
            or speed > setup.max_joint_speed * 1.02
        ):
            raise ValueError("unsafe_intermediate_reset_conditioning")
    # Start a new local observation generation after real reset-conditioning
    # physics. Global simulator time continues; no old prefix is attached.
    env.generation = max(env.generation, int(state["source_generation"])) + 1
    env.episode_length_buf.zero_()
    env._time_s = 0.0
    env.capture.reset()
    env.load_samples = []
    env._get_observations()
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
        reconstruction=reconstruction,
        conditioning=conditioning,
        conditioning_sim_s=6 * env.step_dt,
        local_clock_origin=(
            "after six real reset-conditioning control steps; global physics clock continues"
        ),
        generation=env.generation,
        solver_state="reconstructed; requires fresh sustained physical contact qualification",
        conditioned_controller=dict(
            compression=compression,
            compensation=compensation.tolist(),
            reference_angle=reference,
            initial_lead=initial_lead,
        ),
        conditioning_contact_force_n=[row["force_n"] for row in conditioning],
    )
