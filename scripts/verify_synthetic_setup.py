#!/usr/bin/env python
"""Subphase 4.1 synthetic physics, expert and common-setup verification."""

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("command", choices=("physics", "probe", "search"))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--config", type=Path)
parser.add_argument("--case", type=int, choices=range(4))
parser.add_argument("--cameras", action="store_true")
parser.add_argument("--candidates", type=Path, help="Kinematic screening report for search")
parser.add_argument("--candidate-limit", type=int, default=3)
parser.add_argument("--repeats", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.cameras
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from alexdoor_xas.assets.synthetic_door import CASES  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_purdue_env import DoorPushPurdueEnv, tensor  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_purdue_env_cfg import DoorPushPurdueEnvCfg  # noqa: E402


def make_env(door, setup=None):
    cfg = DoorPushPurdueEnvCfg()
    cfg.sim.device = args.device
    cfg.cameras = args.cameras
    cfg.synthetic_door = door
    cfg.floor_pose = (-1.5, 0.0, 0.0) if setup is None else tuple(setup["floor_pose"])
    if setup is not None:
        cfg.initial_joints = {**cfg.initial_joints, **setup.get("initial_joints", {})}
        cfg.max_joint_speed = setup.get("max_joint_speed", cfg.max_joint_speed)
        cfg.centering_gain = setup["centering_gain"]
        cfg.ik_damping = setup["ik_damping"]
        cfg.initial_joints.update(NECK_Z=setup["neck"][0], NECK_Y=setup["neck"][1])
    return DoorPushPurdueEnv(cfg)


def physics(door):
    env = make_env(door)
    try:
        env.reset()
        args.output.mkdir(parents=True, exist_ok=True)
        env.sim.stage.Export(str(args.output / f"{door.name}.usda"))
        panel = env.sim.stage.GetPrimAtPath(env.panel_path)
        dimensions = np.array(
            env.sim.stage.GetPrimAtPath(env.panel_path + "/Collider")
            .GetAttribute("xformOp:scale")
            .Get()
        )
        inertia = np.array(panel.GetAttribute("physics:diagonalInertia").Get())
        expected_size = np.array([door.thickness, door.width, door.height])
        expected_inertia = (
            door.mass * (np.dot(expected_size, expected_size) - expected_size**2) / 12
        )
        zero = torch.zeros((1, 6), device=env.device)
        angle, speed, frame_positions = [], [], []
        for _ in range(120):
            env.step(zero)
            angle.append(float(tensor(env.door.data.joint_pos)[0, 0]))
            speed.append(float(tensor(env.door.data.joint_vel)[0, 0]))
            frame_positions.append(tensor(env.door.data.body_link_pos_w)[0, 0].cpu().numpy())
        effort = torch.tensor([[15.0]], device=env.device)
        torque_angles = []
        for _ in range(600):
            env.door.set_joint_effort_target_index(target=effort)
            env.step(zero)
            torque_angles.append(float(tensor(env.door.data.joint_pos)[0, 0]))
        env.reset()
        reset = float(tensor(env.door.data.joint_pos)[0, 0])
        result = dict(
            case=door.name,
            passive_drift_deg=float(np.rad2deg(np.abs(angle).max())),
            frame_drift_m=float(
                np.linalg.norm(np.array(frame_positions) - frame_positions[0], axis=1).max()
            ),
            maximum_deg=float(np.rad2deg(np.max(torque_angles))),
            mechanical_stop_deg=float(np.rad2deg(door.mechanical_stop)),
            reset_deg=float(np.rad2deg(reset)),
            bodies=env.door.body_names,
            device=str(env.device),
            panel_dimensions_m=dimensions.tolist(),
            panel_inertia_kg_m2=inertia.tolist(),
        )
        result["passed"] = bool(
            result["passive_drift_deg"] < 0.25
            and result["frame_drift_m"] < 1e-4
            and abs(reset) < np.deg2rad(0.1)
            and abs(max(torque_angles) - door.mechanical_stop) < np.deg2rad(1)
            and np.allclose(dimensions, expected_size, atol=1e-7)
            and np.allclose(inertia, expected_inertia, atol=1e-6)
        )
        np.savez_compressed(
            args.output / f"{door.name}-physics.npz",
            angle=angle,
            speed=speed,
            torque_angles=torque_angles,
        )
        return result
    finally:
        env.close()


def probe_candidate(setup, output, cases, repeats, stop_on_failure=False):
    from alexdoor_xas.qualification.synthetic_probe import run_probe, summarize_trials

    results = []
    for door in cases:
        env = make_env(door, setup.to_dict())
        try:
            trials = []
            for i in range(repeats):
                trial = run_probe(env, door, setup, output / door.name / f"repeat-{i + 1}")
                trials.append(trial)
                if not trial["passed"] or trial["angle_deg"] < 45.0:
                    break
        finally:
            env.close()
        result = summarize_trials(trials)
        results.append(result)
        print(json.dumps(result), flush=True)
        report_name = "report.json" if len(cases) == 4 else f"report-{cases[0].name}.json"
        (output / report_name).write_text(json.dumps(results, indent=2) + "\n")
        if stop_on_failure and not result["meets_45_deg"]:
            break
    return dict(setup=setup.to_dict(), cases=results)


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    cases = CASES if args.case is None else (CASES[args.case],)
    if args.command == "physics":
        results = []
        for door in cases:
            results.append(physics(door))
            print(json.dumps(results[-1]), flush=True)
            (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
        return 0 if all(r["passed"] for r in results) else 1
    from alexdoor_xas.assets.purdue import ARM_JOINTS
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, rank_candidates

    if args.config is None or args.repeats < 1:
        raise ValueError("--config and at least one repeat are required")
    setup = ProbeSetup(**json.loads(args.config.read_text()))
    if args.command == "probe":
        result = probe_candidate(setup, args.output, cases, args.repeats)
        passed = all(r["meets_45_deg"] for r in result["cases"])
        if args.cameras:
            passed = passed and all(
                trial["visibility"]["passed"]
                for case in result["cases"]
                for trial in case["trials"]
            )
        return 0 if passed else 1
    if args.candidates is None or args.case is not None or args.candidate_limit < 1:
        raise ValueError("Search requires --candidates, a positive limit and all four cases")
    screen = json.loads(args.candidates.read_text())
    if screen["scope"] != "kinematic_screen_only":
        raise ValueError("Expected a kinematic screening report")
    if screen["fraction"] != setup.contact_fraction or screen["height"] != setup.contact_height:
        raise ValueError("Screen and probe contact configuration differ")
    (args.output / "screening.json").write_text(json.dumps(screen, indent=2) + "\n")
    candidates = []
    for index, candidate in enumerate(screen["candidates"][: args.candidate_limit]):
        values = setup.to_dict()
        values["floor_pose"] = candidate["floor_pose"]
        values["initial_joints"] = {
            **setup.initial_joints,
            **dict(zip(ARM_JOINTS, candidate["contact_joints"][0], strict=True)),
        }
        trial_setup = ProbeSetup(**values)
        candidates.append(
            probe_candidate(
                trial_setup,
                args.output / f"candidate-{index:03d}",
                CASES,
                args.repeats,
                stop_on_failure=True,
            )
        )
        (args.output / "candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    ranked = rank_candidates(candidates, setup.tie_deg)
    domain_passed = bool(ranked and min(c["angle_deg"] for c in ranked[0]["cases"]) >= 45.0)
    report = dict(
        passed=domain_passed,
        candidates_tested=len(candidates),
        selected=ranked[0] if ranked else None,
        status="physical_candidate_requires_visibility_review"
        if domain_passed
        else "unresolved_common_setup",
        frozen=False,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if domain_passed else 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        import traceback

        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
