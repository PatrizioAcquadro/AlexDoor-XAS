#!/usr/bin/env python
"""Bounded solver-feedback/hold/stop probe. Synthetic fixture; no policy contact rollout."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--cameras", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.cameras
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from alexdoor_xas.envs.door_task.door_push_purdue_env import DoorPushPurdueEnv, tensor  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_purdue_env_cfg import DoorPushPurdueEnvCfg  # noqa: E402
from alexdoor_xas.perception.load import (  # noqa: E402
    LoadCalibration,
    ResidualLoadModel,
    simulated_feedback,
)
from alexdoor_xas.policies.purdue import PurdueFK, PurdueIO  # noqa: E402
from alexdoor_xas.recording.b1_runtime import (  # noqa: E402
    array,
    camera_calibration,
    camera_from_joints,
)

report = dict(passed=False, contact_rollout_ready=False)
args.output.mkdir(parents=True, exist_ok=True)
try:
    cfg = DoorPushPurdueEnvCfg()
    cfg.sim.device = args.device
    cfg.cameras = args.cameras
    env = DoorPushPurdueEnv(cfg)
    env.reset()
    # Move commissioning obstacles away, then hold only the initial robot pose.
    for index, name in enumerate(("panel", "frame", "handle")):
        obj = env.scene.rigid_objects[name]
        pose = torch.tensor([[2.0, float(index), 1.0, 0.0, 0.0, 0.0, 1.0]], device=env.device)
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=env.device))
    goal = tuple(x.clone() for x in env.tool_pose())
    rows, costs = [], []
    inference_costs = {}
    if args.cameras:
        from alexdoor_xas.dataset.normalize import DatasetNormStats, NormStats
        from alexdoor_xas.policies.act.config import ActModelCfg
        from alexdoor_xas.policies.act.model import ACTModel
        from alexdoor_xas.policies.act.policy import ActPolicy
        from alexdoor_xas.policies.common.b1_contract import OBS_KEYS
        from alexdoor_xas.policies.diffusion.config import DiffusionModelCfg
        from alexdoor_xas.policies.diffusion.model import DiffusionTransformer
        from alexdoor_xas.policies.diffusion.policy import DiffusionPolicy
        from alexdoor_xas.policies.encoding import RGBDEncoder, pilot_binding
        from alexdoor_xas.policies.observations import B1Observer

        binding = pilot_binding(
            json.loads(Path("configs/perception_inspection.json").read_text()),
            depth_interval_m=(env.capture.near_m, env.capture.far_m),
        )
        calibration = camera_calibration(env)
        tool_fk = PurdueFK(env.robot.cfg.spawn.asset_path)

        def norm(width):
            return NormStats(np.zeros(width), np.ones(width), -np.ones(width), np.ones(width), 1)

        stats = DatasetNormStats(
            norm(6), norm(binding.obs_dim), OBS_KEYS, ("probe",), "A2_ee_delta"
        )
        policies = {
            "act": ActPolicy(
                ACTModel(binding.obs_dim, 6, ActModelCfg(chunk_size=12)), stats, device="cuda"
            ),
            "diffusion": DiffusionPolicy(
                DiffusionTransformer(binding.obs_dim, 6, DiffusionModelCfg(horizon=12)),
                stats,
                device="cuda",
                sampler="ddim",
                num_inference_steps=10,
            ),
        }
        for family, policy in policies.items():
            observer = B1Observer(None, binding, encoder=RGBDEncoder(binding))
            samples = []
            for tick in range(36):
                torch.cuda.synchronize()
                begin = perf_counter()
                env.command_pose(*goal)
                env.step(torch.zeros((1, 6), device=env.device))
                sample = env.capture.sample
                q = array(sample.joint_position)[0]
                sensor = dict(
                    time_s=sample.time_s,
                    frame=sample.frame,
                    rgb=sample.rgb[0],
                    depth_m=sample.depth_m[0],
                    valid_depth=sample.valid_depth[0],
                    joint_position=q,
                    joint_velocity=sample.joint_velocity[0],
                    camera_world=camera_from_joints(calibration, q),
                    intrinsics=array(env.camera.data.intrinsic_matrices)[0],
                )
                observed = observer.update(sensor)
                feedback = simulated_feedback(env)
                _ = env.tool_jacobian()
                _ = tool_fk(q, calibration)
                if tick % 6 == 0:
                    prediction = policy.predict(observed.features)
                    assert np.isfinite(prediction).all()  # output is never actuated
                torch.cuda.synchronize()
                if tick >= 6:
                    samples.append([(perf_counter() - begin) * 1000, tick % 6 == 0])
            queried = [value for value, query in samples if query]
            inference_costs[family] = dict(
                cycle_p50_ms=float(np.median(np.asarray(samples)[:, 0])),
                cycle_p95_ms=float(np.percentile(np.asarray(samples)[:, 0], 95)),
                query_max_ms=max(queried),
                query_p50_ms=float(np.median(queried)),
            )
    # Known 4 N distal perturbation is evaluator input only; the residual model
    # receives robot dynamics, measured torque and tool Jacobian, never this force.
    body = env.robot.body_names.index(env._tool_body_name)
    force = -tensor(env.tool_pose()[1]).new_tensor([1.0, 0.0, 0.0]) * 4
    env.robot.permanent_wrench_composer.set_forces_and_torques_index(
        forces=force.reshape(1, 1, 3),
        torques=torch.zeros((1, 1, 3), device=env.device),
        positions=env.tool_pose()[0].reshape(1, 1, 3),
        body_ids=torch.tensor([body], device=env.device),
        is_global=True,
    )
    load_rows = []
    for _ in range(36):
        env.command_pose(*goal)
        env.step(torch.zeros((1, 6), device=env.device))
        feedback = simulated_feedback(env)
        acceleration = array(env.robot.data.joint_acc)[0]
        expected = (
            array(env.robot.data.mass_matrix)[0] @ acceleration
            + array(env.robot.data.gravity_compensation_forces)[0]
            + array(env.robot.root_view.get_coriolis_and_centrifugal_compensation_forces())[0]
        )
        jac = array(env.tool_jacobian())[0]
        residual = feedback.torque_nm - expected[env.arm_ids]
        projected = jac.T @ np.r_[array(force), np.zeros(3)]
        load_rows.append(np.r_[residual, projected])
    env.robot.permanent_wrench_composer.reset()
    load_rows = np.asarray(load_rows)
    report["known_force_probe"] = dict(
        force_world_n=array(force).tolist(),
        residual_plus_projected_max_nm=float(
            np.abs(load_rows[-10:, :7] + load_rows[-10:, 7:]).max()
        ),
        residual_minus_projected_max_nm=float(
            np.abs(load_rows[-10:, :7] - load_rows[-10:, 7:]).max()
        ),
        jacobian_singular_values=np.linalg.svd(jac, compute_uv=False).tolist(),
    )
    calibration = LoadCalibration(
        np.full(7, 0.02), -np.full(7, 10.0), np.full(7, 10.0), np.full(7, 1.0), np.full(7, 10.0), -1
    )
    try:
        bounded = ResidualLoadModel(calibration).estimate(
            feedback,
            expected[env.arm_ids],
            jac,
            array(env.robot.data.joint_pos)[0, env.arm_ids],
            array(env.robot.data.joint_vel)[0, env.arm_ids],
            acceleration[env.arm_ids],
            [1, 0, 0],
        )
        report["known_force_probe"]["estimated_wrench"] = bounded.wrench.tolist()
        report["known_force_probe"]["force_upper_n"] = bounded.force_upper_n
        assert not bounded.admission(feedback.timestamp_s, 80).admitted
    except ValueError as error:
        report["known_force_probe"]["model_rejection"] = str(error)
    report["inference_cycles"] = inference_costs
    np.savez_compressed(args.output / "load-probe.npz", values=load_rows)
    for _tick in range(90):
        begin = perf_counter()
        env.command_pose(*goal)
        env.step(torch.zeros((1, 6), device=env.device))
        feedback = simulated_feedback(env)
        torch.cuda.synchronize()
        costs.append((perf_counter() - begin) * 1000)
        assert feedback.torque_nm is not None and feedback.healthy
        gravity = tensor(env.robot.data.gravity_compensation_forces)[0, env.arm_ids].cpu().numpy()
        position = tensor(env.robot.data.joint_pos)[0, env.arm_ids].cpu().numpy()
        rows.append(np.r_[feedback.timestamp_s, feedback.torque_nm, gravity, position])
    io = PurdueIO.__new__(PurdueIO)
    io.env, io.admission = env, None
    # Preview is the same pure function used by dispatch, including clipping/IK.
    delta = torch.full((1, 7), 0.0001, device=env.device)
    preview, _, _ = env.preview_targets(joint_delta=delta)
    env.step_a1(delta)
    np.testing.assert_allclose(
        preview.cpu(), env.targets[:, env.arm_ids].cpu(), atol=32 * np.finfo(np.float32).eps, rtol=0
    )
    p, quat = env.tool_pose()
    world_goal = (p + torch.tensor([[0.0001, 0.0, 0.0]], device=env.device), quat)
    preview, _, _ = env.preview_targets(world_goal=world_goal)
    env.command_pose(*world_goal)
    env.step(torch.zeros((1, 6), device=env.device))
    np.testing.assert_allclose(
        preview.cpu(), env.targets[:, env.arm_ids].cpu(), atol=32 * np.finfo(np.float32).eps, rtol=0
    )
    report["controller_preview_matches_applied_targets"] = True
    before = env._time_s
    io.stop()
    assert env._time_s == before and env._pending_pose is None and env._pending_joints is None
    np.testing.assert_array_equal(tensor(env.robot.data.joint_pos).cpu(), env.targets.cpu())
    # Advance under frozen targets externally to measure residual braking motion.
    stop_position = env.tool_pose()[0].clone()
    frozen_target = env.targets[:, env.arm_ids].clone()
    stop_travel = []
    for _ in range(12):
        env.step_a1(frozen_target - tensor(env.robot.data.joint_pos)[:, env.arm_ids])
        stop_travel.append(float((env.tool_pose()[0] - stop_position).norm()))
    values = np.asarray(rows)
    report.update(
        passed=True,
        feedback_source=feedback.source,
        feedback_semantics=feedback.semantics,
        torque_range_nm=[values[:, 1:8].min(axis=0).tolist(), values[:, 1:8].max(axis=0).tolist()],
        torque_minus_gravity_max_nm=np.abs(values[-30:, 1:8] - values[-30:, 8:15])
        .max(axis=0)
        .tolist(),
        torque_plus_gravity_max_nm=np.abs(values[-30:, 1:8] + values[-30:, 8:15])
        .max(axis=0)
        .tolist(),
        step_feedback_p50_ms=float(np.median(costs[10:])),
        step_feedback_p95_ms=float(np.percentile(costs[10:], 95)),
        post_probe_stop_travel_m=max(stop_travel),
        limit=(
            "Robot-only hold and injected 4 N probe. No verified dynamic load calibration "
            "or moving/contact stop envelope; no task contact rollout."
        ),
    )
    np.savez_compressed(args.output / "feedback.npz", values=values)
    env.sim.stage.Export(str(args.output / "probe.usda"))
except Exception as error:
    import traceback

    traceback.print_exc()
    report["error"] = repr(error)
finally:
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    os._exit(0 if report["passed"] else 1)
