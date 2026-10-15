#!/usr/bin/env python
"""Robot-only hold/stop isolation; force truth is evaluator-only."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    gpu_lock = (Path(tempfile.gettempdir()) / "alexdoor-xas-gpu.lock").open("a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
    ).strip()
    if active:
        raise RuntimeError(f"GPU occupied; no simulator started: {active}")
    args.output.mkdir(exist_ok=False)
    shutil.copy2(__file__, args.output / "probe-source.py")
    protocol = dict(
        scope="robot-only, 120 Hz physics/60 Hz control, cameras disabled, native PhysX",
        hold="2s unloaded, 3s 4N -world X, 2s removed, fixed ready target",
        stop=(
            "independent resets; 2s settle, 2s preload, .5s cosine motion, "
            "stop, 2s maintained, 2s removed"
        ),
        criteria=dict(
            steady_joint_error_rad=0.02,
            hold_travel_mm=10.0,
            rotation_deg=3.0,
            stop_travel_mm=10.0,
            final_speed_rad_s=0.01,
            removal_final_mm=2.0,
        ),
        calibration="reuse supplied saved calibration unchanged; no qualification flags",
        reference="evaluator-only injection, not passed to control or residual model",
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    import sys

    sys.path.insert(0, str(args.repo / "src"))
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device="cuda:0").app
    import numpy as np
    import torch

    from alexdoor_xas.envs.door_task.door_push_purdue_env import DoorPushPurdueEnv
    from alexdoor_xas.envs.door_task.door_push_purdue_env_cfg import DoorPushPurdueEnvCfg
    from alexdoor_xas.perception.load import LoadCalibration, ResidualLoadModel
    from alexdoor_xas.policies.purdue import PurdueIO
    from alexdoor_xas.recording.b1_runtime import array

    rows, events = [], []
    report = dict(completed=False, qualification=False, events=events)
    env = None
    try:
        assert app.is_running() and torch.cuda.is_available()
        cfg = DoorPushPurdueEnvCfg()
        setup = json.loads((args.repo / "configs/purdue_synthetic_probe.json").read_text())
        cfg.floor_pose = tuple(setup["floor_pose"])
        cfg.initial_joints.update(setup["initial_joints"])
        cfg.max_joint_speed = setup["max_joint_speed"]
        cfg.centering_gain = setup["centering_gain"]
        cfg.ik_damping = setup["ik_damping"]
        cfg.cameras = False
        cfg.seed = 6100
        saved = json.loads(args.calibration.read_text())
        calibration = LoadCalibration(
            *[
                np.array(saved[k])
                for k in (
                    "residual_bound_nm",
                    "q_min",
                    "q_max",
                    "speed_max_rad_s",
                    "acceleration_max_rad_s2",
                )
            ],
            saved["sign"],
        )
        model = ResidualLoadModel(calibration)

        class DiagnosticEnv(DoorPushPurdueEnv):
            phase = None
            force_n = 0.0
            application_point = None

            def record(self):
                if self.phase is None or not getattr(self, "load_samples", []):
                    return
                s = self.load_samples[-1]
                p, quat = (array(v)[0].copy() for v in self.tool_pose())
                # Reference appended after robot-only inputs; never used by controller/model.
                force = np.array([-self.force_n, 0.0, 0.0])
                ref = np.r_[force, np.cross(self.application_point - p, force)]
                try:
                    e = model.estimate(
                        s.feedback,
                        s.expected_torque_nm,
                        s.jacobian_world,
                        s.q,
                        s.dq,
                        s.ddq,
                        [1.0, 0.0, 0.0],
                    )
                    reason, wrench, upper = "", e.wrench, e.force_upper_n
                except ValueError as error:
                    reason, wrench, upper = str(error), np.full(6, np.nan), np.nan
                ids = self.arm_ids
                rows.append(
                    dict(
                        phase=self.phase,
                        time=s.feedback.timestamp_s,
                        q=s.q,
                        dq=s.dq,
                        ddq=s.ddq,
                        target=array(self.targets)[0, ids].copy(),
                        torque=s.feedback.torque_nm,
                        expected=s.expected_torque_nm,
                        jacobian=s.jacobian_world,
                        position=p,
                        quaternion=quat,
                        gravity=array(self.robot.data.gravity_compensation_forces)[0, ids].copy(),
                        feedforward=array(self.robot.data.joint_effort_target)[0, ids].copy(),
                        computed=array(self.robot.data.computed_torque)[0, ids].copy(),
                        applied=array(self.robot.data.applied_torque)[0, ids].copy(),
                        reference=ref,
                        wrench=wrench,
                        upper=upper,
                        rejection=reason,
                    )
                )

            def _apply_action(self):
                super()._apply_action()
                if self._physics_actions > 1:
                    self.record()

            def _get_observations(self):
                result = super()._get_observations()
                self.record()
                return result

        env = DiagnosticEnv(cfg)
        env.render_enabled = False
        io = PurdueIO.__new__(PurdueIO)
        io.env, io.admission = env, None
        body = env.robot.body_names.index(env._tool_body_name)

        def reset():
            env.phase = None
            env.robot.permanent_wrench_composer.reset()
            env.reset()
            for i, name in enumerate(("panel", "frame", "handle")):
                obj = env.scene.rigid_objects[name]
                obj.write_root_pose_to_sim_index(
                    root_pose=torch.tensor(
                        [[3.0, float(i), 1.0, 0.0, 0.0, 0.0, 1.0]], device=env.device
                    )
                )
                obj.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros((1, 6), device=env.device)
                )
            return array(env.robot.data.joint_pos)[0, env.arm_ids].copy()

        def step(phase, goal, force=0.0, stopped=False):
            if env.phase != phase:
                print(json.dumps(dict(phase=phase, time=env._time_s)), flush=True)
            env.phase, env.force_n = phase, force
            env.application_point = array(env.tool_pose()[0])[0].copy()
            env.robot.permanent_wrench_composer.reset()
            if force:
                env.robot.permanent_wrench_composer.set_forces_and_torques_index(
                    forces=torch.tensor([[[-force, 0.0, 0.0]]], device=env.device),
                    torques=torch.zeros((1, 1, 3), device=env.device),
                    positions=env.tool_pose()[0].reshape(1, 1, 3),
                    body_ids=torch.tensor([body], device=env.device),
                    is_global=True,
                )
            if stopped and hasattr(env, "stop_motion"):
                env.step(torch.zeros((1, 6), device=env.device))
            else:
                measured = array(env.robot.data.joint_pos)[0, env.arm_ids]
                env.step_a1(np.asarray(goal) - measured)
            assert len(env.load_samples) == 2
            active = [
                c
                for g in env.contact_history
                for c in g["contacts"]
                if c["separation_m"] <= 0 or abs(c["force_n"]) > 0
            ]
            assert not active, f"unexpected_robot_contact: {active}"

        def run(phase, n, goal, force=0.0, stopped=False):
            start = len(rows)
            for _ in range(n):
                step(phase, goal, force, stopped)
            return rows[start:]

        q0 = reset()
        ids = env.arm_ids
        report["runtime"] = dict(
            device=torch.cuda.get_device_name(),
            physics_dt=cfg.sim.dt,
            control_dt=env.step_dt,
            cameras=False,
            joints=[env.robot.joint_names[i] for i in ids],
            stiffness=array(env.robot.root_view.get_dof_stiffnesses())[0, ids].tolist(),
            damping=array(env.robot.root_view.get_dof_dampings())[0, ids].tolist(),
            effort_limits=array(env.robot.root_view.get_dof_max_forces())[0, ids].tolist(),
            mass_diagonal=np.diag(array(env.robot.data.mass_matrix)[0])[ids].tolist(),
            gravity_enabled=not env.robot.cfg.spawn.rigid_props.disable_gravity,
            calibration=saved,
            actuators={
                name: dict(
                    type=type(a).__name__,
                    joints=a.joint_names,
                    stiffness=array(a.stiffness)[0].tolist(),
                    damping=array(a.damping)[0].tolist(),
                    effort_limit=array(a.effort_limit)[0].tolist(),
                )
                for name, a in env.robot.actuators.items()
            },
            effort_semantics=(
                "IdealPD computed is commanded torque; applied is clipped torque sent to solver; "
                "torque is separate solver joint-output projection"
            ),
        )
        run("hold_settle", 120, q0)
        anchor = tuple(array(v)[0].copy() for v in env.tool_pose())
        report["hold_anchor"] = [v.tolist() for v in anchor]
        run("hold_loaded", 180, q0, 4.0)
        run("hold_removed", 120, q0)
        for name, force in (("loaded", 4.0), ("unloaded", 0.0)):
            q0 = reset()
            run(name + "_settle", 120, q0)
            run(name + "_preload", 120, q0, force)
            amplitude = np.array([0.025, -0.020, 0.030, 0.025, -0.020, 0.020, 0.015]) * 0.8
            for t in range(30):
                step(
                    name + "_motion",
                    q0 + amplitude * (1 - np.cos(2 * np.pi * (t + 1) / 120)) / 2,
                    force,
                )
            before = dict(
                time=env._time_s,
                target=array(env.targets)[0, ids].copy(),
                q=array(env.robot.data.joint_pos)[0, ids].copy(),
            )
            pose = tuple(array(v)[0].copy() for v in env.tool_pose())
            io.stop()
            assert before["time"] == env._time_s
            frozen = array(env.targets).copy()
            event = dict(
                case=name,
                time=env._time_s,
                before={k: v.tolist() if hasattr(v, "tolist") else v for k, v in before.items()},
                stop_anchor=[v.tolist() for v in pose],
                target_after=frozen[0, ids].tolist(),
            )
            events.append(event)
            run(name + "_stopped_loaded", 120, frozen[0, ids], force, True)
            run(name + "_stopped_removed", 120, frozen[0, ids], 0.0, True)
            np.testing.assert_array_equal(array(env.targets), frozen)
            if hasattr(env, "stop_motion"):
                io.stop()
                np.testing.assert_array_equal(array(env.targets), frozen)
                refused = []
                for command in (
                    lambda: env.step_a1(np.ones(7) * 0.001),
                    lambda: env.command_pose(*env.tool_pose()),
                    lambda: env.step_a3(np.zeros(6), None),
                    lambda: env.set_neck_target([0.0, 0.0]),
                    lambda: env.step(torch.ones((1, 6), device=env.device) * 0.001),
                    lambda: io.execute(None),
                ):
                    try:
                        command()
                    except RuntimeError as error:
                        assert str(error) == "motion_stopped"
                        refused.append(True)
                    else:
                        refused.append(False)
                event["command_refusal"] = refused
                event["stop_idempotent"] = True
                assert all(refused)
        report["completed"] = True
    except Exception as error:
        import traceback

        traceback.print_exc()
        report["error"] = repr(error)
    finally:
        if rows:
            np.savez_compressed(
                args.output / "trace.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]}
            )
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(dict(completed=report["completed"], samples=len(rows))), flush=True)
        os._exit(0 if report["completed"] else 1)


if __name__ == "__main__":
    main()
