#!/usr/bin/env python
"""Bounded robot-only Phase 6.0D probe; no geometry or learned command source."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PROTOCOL = dict(
    scope="robot-only simulated diagnostic; no door/contact/policy admission",
    physics_hz=120,
    control_hz=60,
    samples_per_tick=2,
    seed=6100,
    calibration=["120 settle", "120 unloaded cosine cycle", "60 hold", "60 known -2N X"],
    verification=[
        "60 settle",
        "120 unloaded cosine",
        "120 -4N X cosine",
        "30 -4N X acceleration + 60 frozen-target stop",
        "60 unloaded settle",
        "30 unloaded acceleration + 60 frozen-target stop",
        "60 A2 settle + 120 A2 unloaded 4mm cosine cycle",
    ],
    amplitude_rad=[0.025, -0.020, 0.030, 0.025, -0.020, 0.020, 0.015],
    verification_amplitude_fraction=0.8,
    calibration_residual_allowance_nm=0.02,
    provisional_state_box=dict(q_radius_rad=0.08, speed_rad_s=0.25, acceleration_rad_s2=5.0),
    error_model="native deterministic PhysX feedback/model; no added hardware noise/friction",
    force_reference="external wrench composer in evaluator only; never passed to estimator",
    calibration_flags=dict(verified=False, stop_verified=False),
    interpretation="empirical local checks only; observed maxima are not general safety bounds",
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", PROTOCOL)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    from alexdoor_xas.envs.door_task.door_push_purdue_env import DoorPushPurdueEnv, tensor
    from alexdoor_xas.envs.door_task.door_push_purdue_env_cfg import DoorPushPurdueEnvCfg
    from alexdoor_xas.perception.load import LoadCalibration, ResidualLoadModel
    from alexdoor_xas.policies.purdue import PurdueIO
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup
    from alexdoor_xas.recording.b1_runtime import array

    report = dict(
        completed=False,
        phase_6_0D_qualified=False,
        contact_rollout_ready=False,
        protocol=PROTOCOL,
        stops={},
        phases={},
    )
    assert app.is_running()
    samples, ticks = [], []
    io = None
    env = None
    try:
        assert torch.cuda.is_available()
        cfg = DoorPushPurdueEnvCfg()
        setup = ProbeSetup(**json.loads((REPO / "configs/purdue_synthetic_probe.json").read_text()))
        inspection = json.loads((REPO / "configs/perception_inspection.json").read_text())
        cfg.floor_pose = tuple(setup.floor_pose)
        cfg.initial_joints.update(setup.initial_joints)
        cfg.initial_joints.update(
            zip(("NECK_Z", "NECK_Y"), inspection["waypoints"][-1][1:], strict=True)
        )
        cfg.max_joint_speed, cfg.centering_gain = setup.max_joint_speed, setup.centering_gain
        cfg.ik_damping = setup.ik_damping
        cfg.camera_mount_pitch_rad = inspection["mount_pitch_rad"]
        cfg.seed = PROTOCOL["seed"]

        class DiagnosticEnv(DoorPushPurdueEnv):
            # Separate evaluator telemetry. It never enters RobotLoadSample/model inputs.
            def __init__(self, config):
                self.evaluator_poses = {}
                super().__init__(config)

            def record_evaluator_pose(self):
                if getattr(self, "load_samples", []):
                    time = self.load_samples[-1].feedback.timestamp_s
                    self.evaluator_poses[time] = tuple(array(v)[0].copy() for v in self.tool_pose())

            def _apply_action(self):
                super()._apply_action()
                self.record_evaluator_pose()

            def _get_observations(self):
                result = super()._get_observations()
                self.record_evaluator_pose()
                return result

        env = DiagnosticEnv(cfg)
        env.reset()
        assert env.step_dt == 1 / 60 and cfg.sim.dt == 1 / 120
        for i, name in enumerate(("panel", "frame", "handle")):
            obj = env.scene.rigid_objects[name]
            pose = torch.tensor([[3.0, float(i), 1.0, 0.0, 0.0, 0.0, 1.0]], device=env.device)
            obj.write_root_pose_to_sim_index(root_pose=pose)
            obj.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), device=env.device)
            )
        q0 = array(env.robot.data.joint_pos)[0, env.arm_ids].copy()
        amplitude = np.array(PROTOCOL["amplitude_rad"])
        body = env.robot.body_names.index(env._tool_body_name)
        io = PurdueIO.__new__(PurdueIO)
        io.env, io.admission = env, None

        def force_reference(force_n):
            # Evaluator injection at the tool point; estimator has no access to this function.
            env.robot.permanent_wrench_composer.reset()
            if force_n:
                env.robot.permanent_wrench_composer.set_forces_and_torques(
                    forces=torch.tensor([[[-force_n, 0.0, 0.0]]], device=env.device),
                    torques=torch.zeros((1, 1, 3), device=env.device),
                    positions=env.tool_pose()[0].reshape(1, 1, 3),
                    body_ids=torch.tensor([body], device=env.device),
                    is_global=True,
                )

        def step(phase, goal, force_n=0.0, a2=None):
            if not ticks or ticks[-1]["phase"] != phase:
                print(json.dumps(dict(phase=phase, time_s=env._time_s)), flush=True)
            application_point = array(env.tool_pose()[0])[0].copy()
            force_reference(force_n)
            before = env._time_s
            if a2 is None:
                measured = tensor(env.robot.data.joint_pos)[:, env.arm_ids]
                env.step_a1(
                    torch.tensor(goal, device=env.device, dtype=torch.float32)[None] - measured
                )
            else:
                env.command_pose(*a2)
                env.step(torch.zeros((1, 6), device=env.device))
            assert len(env.load_samples) == 2
            times = [s.feedback.timestamp_s for s in env.load_samples]
            np.testing.assert_allclose(
                times, [before + 1 / 120, before + 1 / 60], atol=1e-9, rtol=0
            )
            # Independent references are appended only after estimator inputs have been captured.
            for substep, sample in enumerate(env.load_samples):
                assert sample.feedback.healthy and sample.feedback.torque_nm is not None
                force = np.array([-force_n, 0.0, 0.0])
                tool_position = env.evaluator_poses[sample.feedback.timestamp_s][0]
                reference = np.r_[force, np.cross(application_point - tool_position, force)]
                samples.append((phase, substep, sample, reference))
            position, quaternion = (array(v)[0] for v in env.tool_pose())
            target = array(env.targets)[0, env.arm_ids]
            measured = array(env.robot.data.joint_pos)[0, env.arm_ids]
            contacts = [c for group in env.contact_history for c in group["contacts"]]
            # A robot-only trial ends on any measured robot contact.
            active = [c for c in contacts if c["separation_m"] <= 0 or abs(c["force_n"]) > 0]
            if active:
                raise RuntimeError(f"unexpected_robot_contact: {active}")
            ticks.append(
                dict(
                    phase=phase,
                    time_s=env._time_s,
                    position=position,
                    quaternion=quaternion,
                    target=target,
                    measured=measured,
                    desired=np.array(goal),
                    contacts=len(active),
                )
            )

        def hold(phase, n, force_n=0.0):
            for _ in range(n):
                step(phase, q0, force_n)

        def motion(phase, n=120, scale=1.0, force_n=0.0):
            for t in range(n):
                weight = (1 - np.cos(2 * np.pi * (t + 1) / 120)) / 2
                step(phase, q0 + amplitude * scale * weight, force_n)

        hold("calibration_settle", 120)
        motion("calibration_unloaded")
        hold("calibration_hold", 60)
        hold("calibration_known_2N", 60, 2.0)
        unloaded = [
            s
            for p, _, s, _ in samples
            if p.startswith("calibration") and p != "calibration_known_2N"
        ]
        residuals = np.array([s.feedback.torque_nm - s.expected_torque_nm for s in unloaded])
        sign_samples = [row for row in samples if row[0] == "calibration_known_2N"][-60:]
        sign_errors = {
            sign: float(
                np.mean(
                    [
                        np.linalg.norm(
                            sign * (s.feedback.torque_nm - s.expected_torque_nm)
                            - s.jacobian_world.T @ ref
                        )
                        for _, _, s, ref in sign_samples
                    ]
                )
            )
            for sign in (-1, 1)
        }
        sign = min(sign_errors, key=sign_errors.get)
        assert sign_errors[sign] < sign_errors[-sign]
        bound = np.abs(residuals).max(axis=0) + PROTOCOL["calibration_residual_allowance_nm"]
        limits = array(env.limits)[0]
        calibration = LoadCalibration(
            bound,
            np.maximum(q0 - 0.08, limits[:, 0]),
            np.minimum(q0 + 0.08, limits[:, 1]),
            np.full(7, 0.25),
            np.full(7, 5.0),
            sign,
        )
        model = ResidualLoadModel(calibration)
        report["calibration"] = dict(
            residual_bound_nm=bound.tolist(),
            sign=sign,
            sign_reference_error_nm=sign_errors,
            q_min=calibration.q_min.tolist(),
            q_max=calibration.q_max.tolist(),
            speed_max_rad_s=calibration.speed_max_rad_s.tolist(),
            acceleration_max_rad_s2=calibration.acceleration_max_rad_s2.tolist(),
            verified=False,
            stop_verified=False,
        )
        write(args.output / "calibration.json", report["calibration"])
        calibration_end = len(samples)
        hold("verification_settle", 60)
        motion("verification_unloaded", scale=0.8)
        motion("verification_known_4N", scale=0.8, force_n=4.0)

        def stop_trial(name, force_n):
            motion(name + "_acceleration", n=30, scale=0.8, force_n=force_n)
            time_before = env._time_s
            stop_pose = tuple(array(v)[0].copy() for v in env.tool_pose())
            speed = array(env.robot.data.joint_vel)[0, env.arm_ids].copy()
            io.stop()
            assert env._time_s == time_before
            assert env._pending_pose is None and env._pending_joints is None
            frozen = array(env.targets)[0, env.arm_ids].copy()
            np.testing.assert_array_equal(frozen, array(env.robot.data.joint_pos)[0, env.arm_ids])
            for _ in range(60):
                step(name + "_stopped", frozen, force_n)
                np.testing.assert_array_equal(array(env.targets)[0, env.arm_ids], frozen)
            stop_poses = [
                env.evaluator_poses[s.feedback.timestamp_s]
                for p, _, s, _ in samples
                if p == name + "_stopped"
            ]
            travel = [np.linalg.norm(p - stop_pose[0]) for p, _ in stop_poses]
            angle = [
                (Rotation.from_quat(q) * Rotation.from_quat(stop_pose[1]).inv()).magnitude()
                for _, q in stop_poses
            ]
            report["stops"][name] = dict(
                start_s=time_before,
                duration_s=env._time_s - time_before,
                joint_speed_at_stop_rad_s=speed.tolist(),
                force_reference_n=force_n,
                travel_max_mm=float(max(travel) * 1000),
                travel_final_mm=float(travel[-1] * 1000),
                rotation_max_deg=float(np.rad2deg(max(angle))),
                targets_unchanged=True,
                final_joint_speed_rad_s=array(env.robot.data.joint_vel)[0, env.arm_ids].tolist(),
            )

        stop_trial("verification_loaded_stop", 4.0)
        hold("verification_return", 60)
        stop_trial("verification_unloaded_stop", 0.0)
        hold("verification_A2_settle", 60)
        pose = tuple(v.clone() for v in env.tool_pose())
        for t in range(120):
            weight = (1 - np.cos(2 * np.pi * (t + 1) / 120)) / 2
            position = pose[0] + torch.tensor([[0.004 * weight, 0.0, 0.0]], device=env.device)
            step("verification_A2", q0, a2=(position, pose[1]))
            ticks[-1]["world_goal"] = array(position)[0]

        estimates, rejection = [], []
        for p, substep, sample, ref in samples[calibration_end:]:
            # Only robot feedback/model/state and a fixed world normal enter estimation.
            try:
                estimate = model.estimate(
                    sample.feedback,
                    sample.expected_torque_nm,
                    sample.jacobian_world,
                    sample.q,
                    sample.dq,
                    sample.ddq,
                    [1.0, 0.0, 0.0],
                )
                estimates.append((p, substep, estimate, ref))
                assert not estimate.admission(
                    sample.feedback.timestamp_s, setup.force_limit_n
                ).admitted
                rejection.append("")
            except ValueError as error:
                estimates.append((p, substep, None, ref))
                rejection.append(str(error))
        for phase in sorted({p for p, _, _, _ in samples[calibration_end:]}):
            rows = [row for row in estimates if row[0] == phase]
            accepted = [(s, e, ref) for _, s, e, ref in rows if e is not None]
            trace = [r for r in ticks if r["phase"] == phase]
            robot = [s for p, _, s, _ in samples[calibration_end:] if p == phase]
            report["phases"][phase] = dict(
                samples=len(rows),
                estimates=len(accepted),
                rejected=len(rows) - len(accepted),
                evaluator_corrected_residual_max_nm=np.max(
                    [
                        np.abs(
                            sign * (s.feedback.torque_nm - s.expected_torque_nm)
                            - s.jacobian_world.T @ ref
                        )
                        for p, _, s, ref in samples[calibration_end:]
                        if p == phase
                    ],
                    axis=0,
                ).tolist(),
                jacobian_min_singular_value=float(
                    min(np.linalg.svd(s.jacobian_world, compute_uv=False)[-1] for s in robot)
                ),
                residual_abs_max_nm=np.max(
                    [np.abs(s.feedback.torque_nm - s.expected_torque_nm) for s in robot], axis=0
                ).tolist(),
                desired_joint_error_max_rad=float(
                    max(np.max(np.abs(r["desired"] - r["measured"])) for r in trace)
                ),
                applied_joint_tracking_max_rad=float(
                    max(np.max(np.abs(r["target"] - r["measured"])) for r in trace)
                ),
                speed_max_rad_s=np.max([np.abs(s.dq) for s in robot], axis=0).tolist(),
                acceleration_max_rad_s2=np.max([np.abs(s.ddq) for s in robot], axis=0).tolist(),
                by_substep={
                    str(k): dict(
                        count=sum(s == k for s, _, _ in accepted),
                        force_error_max_n=max(
                            [
                                float(np.linalg.norm(e.wrench[:3] - ref[:3]))
                                for s, e, ref in accepted
                                if s == k
                            ],
                            default=None,
                        ),
                        upper_max_n=max(
                            [e.force_upper_n for s, e, _ in accepted if s == k], default=None
                        ),
                        upper_covers_reference=(
                            all(
                                e.force_upper_n >= np.linalg.norm(ref[:3])
                                for s, e, ref in accepted
                                if s == k
                            )
                            if any(s == k for s, _, _ in accepted)
                            else None
                        ),
                    )
                    for k in (0, 1)
                },
            )
            if phase == "verification_A2":
                report["phases"][phase]["world_tracking_max_mm"] = float(
                    max(np.linalg.norm(r["position"] - r["world_goal"]) for r in trace) * 1000
                )
        report.update(
            completed=True,
            device=torch.cuda.get_device_name(),
            samples=len(samples),
            ticks=len(ticks),
            sim_duration_s=env._time_s,
            rejection_counts={
                reason: rejection.count(reason) for reason in sorted(set(rejection)) if reason
            },
        )
        np.savez_compressed(
            args.output / "estimates.npz",
            phase=[p for p, _, _, _ in estimates],
            rejection=rejection,
            wrench=[np.full(6, np.nan) if e is None else e.wrench for _, _, e, _ in estimates],
            force_upper_n=[np.nan if e is None else e.force_upper_n for _, _, e, _ in estimates],
            normal_lower_n=[np.nan if e is None else e.normal_lower_n for _, _, e, _ in estimates],
        )
    except Exception as error:
        import traceback

        traceback.print_exc()
        report["error"] = repr(error)
    finally:
        if io is not None:
            io.stop()
            env.robot.permanent_wrench_composer.reset()
        if samples:
            np.savez_compressed(
                args.output / "feedback.npz",
                phase=[p for p, _, _, _ in samples],
                substep=[i for _, i, _, _ in samples],
                time_s=[s.feedback.timestamp_s for _, _, s, _ in samples],
                torque=[s.feedback.torque_nm for _, _, s, _ in samples],
                expected=[s.expected_torque_nm for _, _, s, _ in samples],
                jacobian=[s.jacobian_world for _, _, s, _ in samples],
                q=[s.q for _, _, s, _ in samples],
                dq=[s.dq for _, _, s, _ in samples],
                ddq=[s.ddq for _, _, s, _ in samples],
                evaluator_wrench=[ref for _, _, _, ref in samples],
            )
        if ticks:
            np.savez_compressed(
                args.output / "tracking.npz",
                **{
                    key: np.array([r[key] for r in ticks])
                    for key in (
                        "phase",
                        "time_s",
                        "position",
                        "quaternion",
                        "target",
                        "measured",
                        "desired",
                        "contacts",
                    )
                },
            )
        write(args.output / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        # Isaac shutdown is not part of the physics result; flush evidence first.
        os._exit(0 if report["completed"] else 1)


if __name__ == "__main__":
    main()
