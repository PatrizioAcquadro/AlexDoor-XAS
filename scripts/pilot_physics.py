#!/usr/bin/env python
"""Bounded teacher collection, command replay and operational policy startup."""

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "replay", "rollout"))
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/b1_policy_pilot.json")
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--setup", type=Path, help="Explicit prospective pilot setup; defaults to corpus setup"
    )
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--space", choices=("A1_joint_delta", "A2_ee_delta"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--family", choices=("act", "diffusion"))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    selected = [e for e in protocol["episodes"] if e["asset_id"] == args.asset_id]
    if len(selected) != 1 or selected[0]["split"] not in ("train", "development"):
        parser.error("Only explicitly declared train/development identities are allowed")
    if args.mode == "replay" and (args.recording is None or args.space is None):
        parser.error("Replay needs a recording and action space")
    if args.mode == "rollout" and (
        args.checkpoint is None or args.family is None or selected[0]["split"] != "development"
    ):
        parser.error("Rollout requires one declared development identity and checkpoint family")
    args.output.mkdir(parents=True, exist_ok=False)
    from alexdoor_xas.qualification.corpus import load_corpus

    corpus = load_corpus(REPO / "datasets/doors/b1/corpus.json", REPO)
    entry = next(e for e in corpus["doors"] if e["asset_id"] == args.asset_id)
    if entry["split"] != selected[0]["split"]:
        raise ValueError("Pilot identity differs from frozen corpus split")
    folder = args.asset_root / args.asset_id
    for name in ("prepared.json", "recipe.json", "candidate.json"):
        if (folder / name).read_bytes() != (
            REPO / "datasets/doors/b1" / args.asset_id / name
        ).read_bytes():
            raise ValueError(f"Different prepared input: {name}")
    from isaaclab.app import AppLauncher

    app = AppLauncher(
        headless=True, enable_cameras=True, device="cuda:0", width=1280, height=720
    ).app
    import torch

    from alexdoor_xas.assets.identity import RobotAssetRef
    from alexdoor_xas.envs.door_task.door_push_purdue_env import DoorPushPurdueEnv
    from alexdoor_xas.perception.inspection import load_inspection
    from alexdoor_xas.qualification.door_geometry import PreparedDoor
    from alexdoor_xas.qualification.expert import probe_config
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, run_probe
    from alexdoor_xas.recording.b1_runtime import ExpertRecorder, camera_calibration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    inspection = load_inspection(REPO / "configs/perception_inspection.json")
    setup_path = args.setup or REPO / corpus["setup"]["path"]
    setup = ProbeSetup(**json.loads(setup_path.read_text()))
    setup.neck = tuple(inspection["waypoints"][-1][1:])
    door = PreparedDoor(
        folder,
        json.loads((folder / "prepared.json").read_text()),
        json.loads((folder / "recipe.json").read_text()),
    )
    cfg = probe_config(door, setup, "cuda:0")
    cfg.seed = selected[0]["seed"]
    cfg.camera_mount_pitch_rad = inspection["mount_pitch_rad"]
    env = DoorPushPurdueEnv(cfg)
    robot = RobotAssetRef(
        "purdue_alex003_full_convex_wsg32_umi_v1",
        hashlib.sha256(Path(env.robot.cfg.spawn.asset_path).read_bytes()).hexdigest(),
    )
    write_json(
        args.output / "runtime.json",
        dict(
            robot_asset=robot.to_dict(),
            urdf=env.robot.cfg.spawn.asset_path,
            device=torch.cuda.get_device_name(),
            setup=setup.to_dict(),
            setup_path=str(setup_path.resolve()),
        ),
    )
    started = perf_counter()
    try:
        if args.mode == "collect":
            metadata = dict(
                entry,
                episode_id=f"{args.asset_id}-nominal-pilot",
                condition="nominal",
                seed=cfg.seed,
                setup=setup.to_dict(),
                teacher="frozen_arm_expert_with_inspection",
                inspection=inspection,
                robot_asset=robot.to_dict(),
            )
            recorder = ExpertRecorder(
                args.output / "episode.hdf5", metadata, purpose="b1_partial_policy_pilot"
            )
            recorder.calibration = camera_calibration(env)
            try:
                result = run_probe(
                    env,
                    door,
                    setup,
                    args.output / "teacher",
                    recorder=recorder,
                    capture_evidence=False,
                    inspection=inspection,
                )
            finally:
                recorder.close()
            from alexdoor_xas.recording.b1 import validate_episode

            if result["passed"]:
                result["recording_validation"] = validate_episode(args.output / "episode.hdf5")
        elif args.mode == "replay":
            result = replay(env, args, protocol)
        else:
            result = rollout(env, args, protocol, robot, setup, inspection)
        result["wall_s"] = perf_counter() - started
        write_json(args.output / "report.json", result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        write_json(
            args.output / "failure.json",
            dict(
                error=repr(error), traceback=traceback.format_exc(), wall_s=perf_counter() - started
            ),
        )
        raise
    finally:
        env.close()
        del app
    return 0 if result.get("passed") else 1


def replay(env, args, protocol):
    """Diagnostic replays stop before loaded contact; source labels use measured state."""
    import h5py
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    from alexdoor_xas.action.b1 import apply_pose_delta, robot_actions
    from alexdoor_xas.action.frames import ObjectFrame
    from alexdoor_xas.policies.purdue import PurdueFK, contact_status
    from alexdoor_xas.recording.b1 import PHASES
    from alexdoor_xas.recording.b1_runtime import array

    env.reset()
    fk = PurdueFK(env.robot.cfg.spawn.asset_path)
    costs, rows = [], []
    failure = None
    with h5py.File(args.recording) as source:
        calibration = json.loads(source["metadata"].attrs["calibration"])
        obs, commands = source["observations"], source["commands"]
        start = next(i for i, p in enumerate(commands["phase"][:]) if PHASES[int(p)] != "inspect")
        end = min(start + protocol["replay_approach_ticks"], len(commands["time_s"]))
        for tick in range(end):
            begin = perf_counter()
            q = array(env.capture.sample.joint_position)[0]
            tool = fk(q, calibration)
            original = fk(obs["joint_position"][tick], calibration)
            goal = ObjectFrame(commands["tool_position"][tick], commands["tool_rotation"][tick])
            labels = robot_actions(
                obs["joint_position"][tick, :7], commands["joint_target"][tick], original, goal
            )
            env.set_neck_target(commands["neck_target"][tick])
            if args.space == "A1_joint_delta":
                env.step_a1(labels[args.space])
            else:
                predicted = apply_pose_delta(tool, labels[args.space])
                env.command_pose(predicted.origin, Rotation.from_matrix(predicted.rot).as_quat())
                env.step(torch.zeros((1, 6), device=env.device))
            observed = fk(array(env.capture.sample.joint_position)[0], calibration)
            expected = fk(obs["joint_position"][tick + 1], calibration)
            reason, force, gap = contact_status(env.contact_history, 80)
            rows.append(
                [
                    tick,
                    np.max(
                        np.abs(
                            array(env.capture.sample.joint_position)[0, :7]
                            - obs["joint_position"][tick + 1, :7]
                        )
                    ),
                    np.linalg.norm(observed.origin - expected.origin),
                    Rotation.from_matrix(observed.rot @ expected.rot.T).magnitude(),
                    force,
                    float("nan") if gap is None else gap,
                ]
            )
            costs.append((perf_counter() - begin) * 1000)
            if reason or force > 0:
                failure = reason or "unexpected_loaded_replay"
                break
    values = np.asarray(rows)
    np.savez_compressed(args.output / "trace.npz", values=values, cycle_ms=costs)
    arm = values[start:] if len(values) > start else values
    passed = (
        failure is None
        and len(values) == end
        and arm[:, 2].max() <= protocol["criteria"]["position_m"]
        and arm[:, 3].max() <= protocol["criteria"]["rotation_rad"]
    )
    env.sim.stage.Export(str(args.output / "scene.usda"))
    return dict(
        passed=bool(passed),
        scope="unloaded_command_replay_including_inspection_and_approach_rotation",
        space=args.space,
        ticks=len(values),
        approach_ticks=max(0, len(values) - start),
        failure=failure,
        max_joint_error_rad=float(arm[:, 1].max()),
        max_tool_position_error_m=float(arm[:, 2].max()),
        max_tool_rotation_error_rad=float(arm[:, 3].max()),
        peak_force_n=float(values[:, 4].max()),
        cycle_p50_ms=float(np.median(costs)),
        cycle_p95_ms=float(np.percentile(costs, 95)),
        cycle_max_ms=max(costs),
        contact_rollout_ready=False,
    )


def rollout(env, args, protocol, robot, setup, inspection):
    """Run the operational runner with honestly unavailable geometry/admission."""
    from dataclasses import asdict
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image

    from alexdoor_xas.policies.b1 import B1Policy
    from alexdoor_xas.policies.encoding import RGBDEncoder, pilot_binding
    from alexdoor_xas.policies.observations import B1Observer
    from alexdoor_xas.policies.purdue import PurdueIO, contact_status
    from alexdoor_xas.policies.rollout import B1Runner
    from alexdoor_xas.recording.b1_runtime import array

    binding = pilot_binding(inspection, depth_interval_m=(env.capture.near_m, env.capture.far_m))
    policy = B1Policy.from_checkpoint(
        args.checkpoint,
        args.family,
        binding=binding,
        runtime_asset=robot,
        device="cuda:0",
        num_inference_steps=protocol["ddim_steps"],
    )
    observer = B1Observer(None, binding, encoder=RGBDEncoder(binding))
    geometry = SimpleNamespace(binding=binding, last_estimate=None)
    io = PurdueIO(env, binding=binding, robot_asset=robot, setup=setup, observed_provider=geometry)
    # Passive evaluator telemetry is never returned to the observer, policy or monitor.
    rows, costs, cycle_ms, encoding_ms = [], [], [], []
    hold = io.hold_step
    update = observer.update
    last_completed = None

    def measured_update(sensor):
        nonlocal last_completed
        begin = perf_counter()
        result = update(sensor)
        end = perf_counter()
        encoding_ms.append((end - begin) * 1000)
        if last_completed is not None:
            cycle_ms.append((end - last_completed) * 1000)
        last_completed = end
        return result

    def measured_hold():
        begin = perf_counter()
        hold()
        costs.append((perf_counter() - begin) * 1000)
        _, force, _ = contact_status(env.contact_history, setup.force_limit_n)
        rows.append(
            [
                env.capture.sample.time_s,
                float(array(env.door.data.joint_pos)[0, 0]),
                force,
                float(np.max(np.abs(io.robot_feedback().torque_nm))),
            ]
        )

    io.hold_step = measured_hold
    observer.update = measured_update
    runner = B1Runner(policy, observer, io, n_action_steps=protocol["replan_ticks"])
    result = runner.run(max_ticks=protocol["rollout_max_ticks"], seed=protocol["inference_seed"])
    stop_time = env._time_s
    assert io.env._pending_pose is None and io.env._pending_joints is None
    np.testing.assert_array_equal(array(env.robot.data.joint_pos), array(env.targets))
    env.sim.stage.Export(str(args.output / "scene.usda"))
    Image.fromarray(array(env.capture.sample.rgb)[0]).save(args.output / "terminal.png")
    np.savez_compressed(
        args.output / "trace.npz",
        values=rows,
        hold_cycle_ms=costs,
        inspection_cycle_ms=cycle_ms,
        observer_update_ms=encoding_ms,
    )
    return dict(
        passed=False,
        runner=asdict(result),
        family=args.family,
        action_space=policy.action_space,
        checkpoint=str(args.checkpoint),
        scope="operational_rollout_attempt_no_admission_bypass",
        inspection_ticks=len(rows),
        policy_commands=result.control_steps,
        observed_geometry_available=False,
        load_source_available=False,
        action_admission_source_available=False,
        frozen_target_stop=True,
        stop_sim_time_s=stop_time,
        progress_max_deg=float(np.rad2deg(np.max(np.abs(np.asarray(rows)[:, 1])))) if rows else 0.0,
        evaluator_peak_normal_force_n=max((r[2] for r in rows), default=0.0),
        measured_joint_torque_abs_max_nm=max((r[3] for r in rows), default=0.0),
        released=False,
        release_attempted=False,
        hold_cycle_p50_ms=float(np.median(costs)) if costs else None,
        hold_cycle_p95_ms=float(np.percentile(costs, 95)) if costs else None,
        hold_cycle_max_ms=max(costs, default=None),
        inspection_cycle_p50_ms=float(np.median(cycle_ms)) if cycle_ms else None,
        inspection_cycle_p95_ms=float(np.percentile(cycle_ms, 95)) if cycle_ms else None,
        inspection_cycle_max_ms=max(cycle_ms, default=None),
        contact_rollout_ready=False,
    )


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
