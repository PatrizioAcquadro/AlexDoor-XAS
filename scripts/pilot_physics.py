#!/usr/bin/env python
"""Bounded teacher collection, command replay and operational policy startup."""

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import traceback
from pathlib import Path
from time import perf_counter, time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def interrupt_run(signum, _frame):
    raise KeyboardInterrupt(f"Pilot interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "replay", "rollout"))
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/b1_policy_pilot.json")
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--setup", type=Path, help="Explicit prospective pilot setup; defaults to corpus setup"
    )
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--condition", help="One condition explicitly declared by the protocol")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--space", choices=("A1_joint_delta", "A2_ee_delta"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--operational-recipe",
        type=Path,
        help="Measured observed perception/load/motion recipe for real rollout",
    )
    parser.add_argument("--family", choices=("act", "diffusion"))
    parser.add_argument(
        "--expert-reference",
        type=Path,
        help="Frozen matched reference JSON, read only by independent rollout scoring",
    )
    parser.add_argument(
        "--recipe",
        choices=("rgbd-grid-v1", "rgbd-grid-fixed-units-v2", "rgbd-conv-v1"),
        default="rgbd-grid-v1",
        help="Rollout input recipe; must match the checkpoint and current sensor",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    declared = (
        protocol.get("evaluation_panel", protocol["episodes"])
        if args.mode == "rollout"
        else protocol["episodes"]
    )
    condition = args.condition or protocol.get("condition", "nominal")
    selected = [
        e
        for e in declared
        if e["asset_id"] == args.asset_id
        and e.get("condition", protocol.get("condition", "nominal")) == condition
    ]
    if len(selected) != 1 or selected[0]["split"] not in ("train", "development"):
        parser.error("Only explicitly declared train/development identities are allowed")
    if args.mode == "replay" and (args.recording is None or args.space is None):
        parser.error("Replay needs a recording and action space")
    if args.mode == "rollout" and (
        args.checkpoint is None or args.family is None or args.operational_recipe is None
    ):
        parser.error(
            "Rollout requires a declared train/development condition, checkpoint family "
            "and operational recipe"
        )
    lighting_scale = float(selected[0].get("lighting_scale", 1.0))
    if not math.isfinite(lighting_scale) or lighting_scale <= 0:
        parser.error("Declared simulated lighting scale must be finite and positive")
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
    signal.signal(signal.SIGTERM, interrupt_run)
    write_json(
        args.output / "physical-case-start.json",
        dict(mode=args.mode, condition=condition, wall_time=time()),
    )
    from pxr import UsdLux

    light = UsdLux.DomeLight(env.sim.stage.GetPrimAtPath("/World/Light"))
    intensity = light.GetIntensityAttr()
    base_intensity = float(intensity.Get())
    intensity.Set(base_intensity * lighting_scale)
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
            condition=condition,
            lighting=dict(
                scope="simulated_dome_light",
                scale=lighting_scale,
                base_intensity=base_intensity,
                intensity=base_intensity * lighting_scale,
            ),
        ),
    )
    started = perf_counter()
    stop_attempted = False
    try:
        if args.mode == "collect":
            metadata = dict(
                entry,
                episode_id=selected[0].get("episode_id", f"{args.asset_id}-{condition}-pilot"),
                condition=condition,
                lighting_scale=lighting_scale,
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
            result = None
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
                from alexdoor_xas.qualification.stop import verify_stop_continuation

                stop_attempted = True
                result["teacher_cycle_passed"] = result["passed"]
                result["post_stop"] = verify_stop_continuation(
                    env, args.output / "post-stop.npz", force_limit_n=setup.force_limit_n
                )
                if not result["post_stop"]["passed"]:
                    result["passed"] = False
                    result["stop_reason"] = "post_stop_verification_failed"
                recorder.finish(result)
            except BaseException:
                if result is not None:
                    result["passed"] = False
                    result["stop_reason"] = "collection_continuation_interrupted"
                    recorder.finish(result)
                raise
            finally:
                recorder.close()
            from alexdoor_xas.recording.b1 import validate_episode

            if result["passed"]:
                result["recording_validation"] = validate_episode(args.output / "episode.hdf5")
        elif args.mode == "replay":
            result = replay(env, args, protocol)
        else:
            result = rollout(env, args, protocol, robot, setup, inspection)
        if not stop_attempted:
            from alexdoor_xas.qualification.stop import verify_stop_continuation

            stop_attempted = True
            result["post_stop"] = verify_stop_continuation(
                env, args.output / "post-stop.npz", force_limit_n=setup.force_limit_n
            )
            if not result["post_stop"]["passed"]:
                result["passed"] = False
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
        if not stop_attempted:
            from alexdoor_xas.qualification.stop import verify_stop_continuation

            try:
                verify_stop_continuation(
                    env, args.output / "interrupted-stop.npz", force_limit_n=setup.force_limit_n
                )
            except Exception as stop_error:
                write_json(args.output / "stop-failure.json", dict(error=repr(stop_error)))
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
    """Run learned commands with observed geometry and actual-motion admission."""
    from dataclasses import asdict
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation

    from alexdoor_xas.perception.evaluation import json_safe
    from alexdoor_xas.policies.b1 import B1Policy
    from alexdoor_xas.policies.encoding import pilot_binding
    from alexdoor_xas.policies.pilot_runtime import observed_pilot_runtime
    from alexdoor_xas.policies.purdue import contact_status
    from alexdoor_xas.policies.rollout import B1Runner
    from alexdoor_xas.qualification.pilot_evaluation import PilotEvaluation, physical_row
    from alexdoor_xas.recording.b1 import PHASES
    from alexdoor_xas.recording.b1_runtime import array
    from alexdoor_xas.recording.rollout import RolloutWriter

    binding = pilot_binding(
        inspection,
        depth_interval_m=(env.capture.near_m, env.capture.far_m),
        recipe=args.recipe,
    )
    policy = B1Policy.from_checkpoint(
        args.checkpoint,
        args.family,
        binding=binding,
        runtime_asset=robot,
        device="cuda:0",
        num_inference_steps=protocol["ddim_steps"],
    )
    observer, io, geometry, motion, load_log = observed_pilot_runtime(
        env, binding, robot, setup, args.operational_recipe, args.output, policy.action_space
    )
    recording = RolloutWriter(
        args.output / "rollout.hdf5",
        dict(
            purpose="learned_rollout_evaluation",
            family=args.family,
            checkpoint=str(args.checkpoint),
            action_space=policy.action_space,
            binding=binding.to_dict(),
            robot_asset=robot.to_dict(),
            robot_joint_names=list(env.robot.joint_names),
        ),
        io.calibration,
        control_dt=io.dt,
        image_period_s=binding.config["encoding"]["period_s"],
    )
    native_contacts = (args.output / "evaluator-native-contacts.jsonl").open("x")
    evaluator_states = (args.output / "evaluator-control-state.jsonl").open("x")
    pending = None
    applied_evaluation_goal = None
    evaluation = None
    if args.expert_reference is not None:
        reference = json.loads(args.expert_reference.read_text())
        if (
            reference["asset_id"] != args.asset_id
            or reference["status"] != "qualified"
            or reference.get("criteria_unchanged") is not True
        ):
            raise ValueError("Independent reference differs from the frozen matched pilot")
        evaluation = PilotEvaluation(setup, reference["theta_expert_d"], io.dt)

    recording.file["commands"].attrs["kind_codes"] = json.dumps(
        dict(inspection_hold=0, joints=1, delta=2, pose=3)
    )

    def record_step(command, goal):
        nonlocal pending, applied_evaluation_goal
        if pending is not None:
            raise RuntimeError("unrecorded_control_step")
        request = np.full(12, np.nan)
        kind, width, stage = 0, 0, "inspect"
        if command is not None:
            kind = {"joints": 1, "delta": 2, "pose": 3}[command.kind]
            value = (
                np.r_[command.value.origin, command.value.rot.flatten()]
                if command.kind == "pose"
                else np.asarray(command.value)
            )
            width, stage = len(value), command.stage
            request[:width] = value
        # FK of the targets actually submitted, including A1. This is a robot
        # model quantity for recording/evaluation, never a replacement command.
        applied_joints = array(env.capture.sample.joint_position)[0].copy()
        applied_joints[:7] = array(env.targets)[0, env.arm_ids]
        applied_evaluation_goal = io.tool_fk(applied_joints, io.calibration)
        pending = dict(
            time_s=np.float64(env.capture.sample.time_s - io.dt),
            kind=np.int8(kind),
            request=request,
            request_width=np.int8(width),
            learned=np.bool_(command is not None),
            submitted_tool_position=np.full(3, np.nan) if goal is None else goal.origin.copy(),
            submitted_tool_rotation=np.full((3, 3), np.nan) if goal is None else goal.rot.copy(),
            stage=np.int8(-1 if stage is None else PHASES.index(stage)),
            # Snapshot while these are still the applied controller targets.
            joint_target=array(env.targets)[0].copy(),
            effort_target_nm=array(env.robot.data.joint_effort_target)[0].copy(),
            applied_tool_position=applied_evaluation_goal.origin.copy(),
            applied_tool_rotation=applied_evaluation_goal.rot.copy(),
        )

    def record_observation(sensor, observed):
        nonlocal pending, applied_evaluation_goal
        position, quaternion = [array(v)[0] for v in env.tool_pose()]
        recording.observe(
            sensor,
            observed,
            dict(
                robot_joint_position=array(env.robot.data.joint_pos)[0],
                robot_joint_velocity=array(env.robot.data.joint_vel)[0],
                actual_tool_position=position,
                actual_tool_rotation=Rotation.from_quat(quaternion).as_matrix(),
            ),
        )
        # Independent truth is persisted only here, never passed back to IO or
        # the observer. Native contact records keep their original categories.
        independent = dict(
            time_s=float(sensor["time_s"]),
            native_door_angle_rad=float(array(env.door.data.joint_pos)[0, 0]),
            native_door_speed_rad_s=float(array(env.door.data.joint_vel)[0, 0]),
        )
        if pending is not None:
            physical = physical_row(
                env,
                env.cfg.prepared_door,
                setup,
                float(sensor["time_s"]),
                applied_evaluation_goal,
                learned_command=bool(pending["learned"]),
            )
            independent.update(evaluation.update(physical) if evaluation is not None else physical)
        evaluator_states.write(json.dumps(json_safe(independent), allow_nan=False) + "\n")
        if pending is not None:
            recording.command(pending)
            robot_rows, evaluator_rows = [], []
            for sample, contact in zip(env.load_samples, env.contact_history, strict=True):
                fb = sample.feedback
                robot_rows.append(
                    dict(
                        time_s=np.float64(fb.timestamp_s),
                        available_s=np.float64(fb.available_s),
                        present=np.bool_(fb.torque_nm is not None),
                        healthy=np.bool_(fb.healthy),
                        torque_nm=np.full(7, np.nan)
                        if fb.torque_nm is None
                        else np.asarray(fb.torque_nm, dtype=np.float64),
                        expected_torque_nm=sample.expected_torque_nm,
                        jacobian_world=sample.jacobian_world,
                        joint_position=sample.q,
                        joint_velocity=sample.dq,
                        joint_acceleration=sample.ddq,
                    )
                )
                intended = [
                    c for c in contact["contacts"] if c["category"] in ("positive", "negative")
                ]
                evaluator_rows.append(
                    dict(
                        time_s=np.float64(fb.timestamp_s),
                        forbidden=np.bool_(contact["forbidden"]),
                        normal_force_n=np.float64(sum(abs(c["force_n"]) for c in intended)),
                        minimum_separation_m=np.float64(
                            min((c["separation_m"] for c in intended), default=np.nan)
                        ),
                    )
                )
                native_contacts.write(
                    json.dumps(
                        dict(
                            time_s=fb.timestamp_s,
                            contact=contact,
                            robot_feedback=dict(
                                source=fb.source,
                                semantics=fb.semantics,
                                reason=fb.reason,
                                joint_names=fb.joint_names,
                                units=fb.units,
                                sign_convention=fb.sign_convention,
                            ),
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
            recording.feedback(robot_rows, evaluator_rows)
            pending = applied_evaluation_goal = None

    io.step_recorder = record_step
    # Passive evaluator telemetry is never returned to the observer, policy or monitor.
    rows, costs, cycle_ms, encoding_ms, recording_ms = [], [], [], [], []
    hold = io.hold_step
    update = observer.update
    last_completed = None

    def measured_update(sensor):
        nonlocal last_completed
        begin = perf_counter()
        try:
            result = update(sensor)
        except BaseException:
            # Preserve the observation after a real step even if preprocessing
            # fails; no feature support is invented for that state.
            record_observation(
                sensor,
                SimpleNamespace(
                    time_s=float(sensor["time_s"]),
                    frame=int(sensor["frame"]),
                    generation=observer.generation,
                    feature_support=None,
                ),
            )
            raise
        observer_end = perf_counter()
        record_observation(sensor, result)
        end = perf_counter()
        encoding_ms.append((observer_end - begin) * 1000)
        recording_ms.append((end - observer_end) * 1000)
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
    runner = B1Runner(
        policy, observer, io, n_action_steps=protocol["replan_ticks"], admission_source=motion
    )
    try:
        result = runner.run(
            max_ticks=protocol["rollout_max_ticks"], seed=protocol["inference_seed"]
        )
        physical_result = None if evaluation is None else evaluation.result()
        recording.finish(
            dict(
                passed=False if physical_result is None else physical_result["task_success"],
                termination=result.reason,
                learned_commands=result.control_steps,
                physical_evaluation=(
                    "unavailable_frozen_reference" if physical_result is None else physical_result
                ),
                qualification=False,
            )
        )
    finally:
        recording.close()
        native_contacts.close()
        evaluator_states.close()
        motion.close()
        load_log.close()
        geometry.close()
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
        recording_ms=recording_ms,
    )
    return dict(
        passed=False if physical_result is None else physical_result["task_success"],
        physical_evaluation=physical_result,
        runner=asdict(result),
        family=args.family,
        action_space=policy.action_space,
        checkpoint=str(args.checkpoint),
        scope="operational_rollout_attempt_no_admission_bypass",
        inspection_ticks=len(rows),
        policy_commands=result.control_steps,
        retained_inputs="rollout.hdf5",
        retained_native_contacts="evaluator-native-contacts.jsonl",
        retained_evaluator_states="evaluator-control-state.jsonl",
        observed_geometry_available=True,
        load_source_available=True,
        action_admission_source_available=True,
        safety_detail=io.safety.checks.last_failure,
        operational_recipe=str(args.operational_recipe),
        operational_admission="operational-admission.jsonl",
        operational_load="operational-load.jsonl",
        frozen_target_stop=True,
        stop_sim_time_s=stop_time,
        progress_max_deg=float(np.rad2deg(np.max(np.abs(np.asarray(rows)[:, 1])))) if rows else 0.0,
        evaluator_peak_normal_force_n=max((r[2] for r in rows), default=0.0),
        measured_joint_torque_abs_max_nm=max((r[3] for r in rows), default=0.0),
        released=False if physical_result is None else physical_result["released"],
        release_attempted=False
        if physical_result is None
        else physical_result["release_attempted"],
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
