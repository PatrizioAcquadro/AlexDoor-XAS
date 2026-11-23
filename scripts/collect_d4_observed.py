#!/usr/bin/env python
"""Collect one observed D4-N Steel simulation episode."""

import argparse
import hashlib
import json
import os
import resource
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from alexdoor_xas.qualification.b1_runtime import (  # noqa: E402
    NATIVE_STEEL_PROFILE,
    native_selected,
)
from alexdoor_xas.qualification.interruption import (  # noqa: E402
    CollectionBudgetExceeded,
    install_interruption,
)


def screen_initial_state(env, door, setup, output):
    """Audit a parked nominal reset, then measure its unloaded stop before collection."""
    import numpy as np
    from PIL import Image

    from alexdoor_xas.perception.evaluation import write_json
    from alexdoor_xas.qualification.clearance import DoorClearance
    from alexdoor_xas.qualification.limits import within_float32_joint_limits
    from alexdoor_xas.qualification.stop import verify_stop_continuation
    from alexdoor_xas.qualification.visibility import measure_visibility
    from alexdoor_xas.recording.b1_runtime import array

    q = array(env.robot.data.joint_pos)[0, env.arm_ids]
    limits = array(env.limits)[0]
    soft_limits = array(env.robot.data.soft_joint_pos_limits)[0, env.arm_ids]
    contacts = env.contacts.read()
    visibility = measure_visibility(env, door, 0.0, setup.contact_fraction, setup.contact_height)
    position, quaternion = [array(v)[0] for v in env.tool_pose()]
    audit = dict(
        joint_names=[env.robot.joint_names[i] for i in env.arm_ids],
        measured_q=q.tolist(),
        runtime_limits=limits.tolist(),
        runtime_soft_limits=soft_limits.tolist(),
        joint_limit_numerical_policy="one_outward_float32_bound_step_for_hard_and_soft_reset_limits",
        minimum_joint_margin_rad=float(np.minimum(q - limits[:, 0], limits[:, 1] - q).min()),
        minimum_soft_joint_margin_rad=float(
            np.minimum(q - soft_limits[:, 0], soft_limits[:, 1] - q).min()
        ),
        tool_position=position.tolist(),
        tool_quaternion=quaternion.tolist(),
        clearance_lower_bound_m=DoorClearance(env).measure(door, 0.0),
        contacts=contacts,
        visibility=visibility,
        scope="Parked reset only; no continuous posture range or loaded stop qualification",
    )
    audit["passed"] = bool(
        np.isfinite(q).all()
        and within_float32_joint_limits(q, limits)
        and within_float32_joint_limits(q, soft_limits)
        and not contacts["forbidden"]
        and not contacts["contacts"]
        and visibility["passed"]
    )
    write_json(output / "initial-state.json", audit)
    Image.fromarray(array(env.capture.sample.rgb)[0]).save(output / "initial-rgb.png")
    if not audit["passed"]:
        raise ValueError("unsupported_initial_state")
    stopping = verify_stop_continuation(
        env, output / "initial-stop.npz", force_limit_n=setup.force_limit_n
    )
    diagnostic_s = getattr(env.cfg, "stop_diagnostic_continuation_s", 0.0)
    if diagnostic_s:
        stopping["two_second_gate_passed"] = stopping["passed"]
        stopping["diagnostic_continuation"] = verify_stop_continuation(
            env,
            output / "initial-stop-diagnostic.npz",
            duration_s=diagnostic_s,
            force_limit_n=setup.force_limit_n,
        )
        stopping["passed"] &= stopping["diagnostic_continuation"]["passed"]
    write_json(output / "initial-stop.json", stopping)
    if not stopping["passed"]:
        raise ValueError("initial_stop_failed")
    # run_probe resets the complete configured state and ordinary impedance.
    # This detached diagnostic stop never enters the recorded teacher history.


def graph_active(env):
    if env.cfg.native_steel:
        return bool(env.sim.backend_manager.backend.graphs)
    return env.sim.physics_manager._graph is not None


def main():
    from alexdoor_xas.perception.evaluation import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--operational-recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf-cache-dir", type=Path)
    parser.add_argument("--record-tracker-stream", action="store_true")
    parser.add_argument("--component-opening-deg", type=float)
    parser.add_argument("--component-max-sim-s", type=float)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    (episode,) = protocol["episodes"]
    asset_id = "psx-interior-wood-002-02e442c"
    if (episode["condition"], episode["asset_id"], episode["split"]) != ("D4-N", asset_id, "train"):
        parser.error("This entry point accepts the single D4-N condition")
    if episode.get("recovery_plan") is not None or episode.get("collection_case") is not None:
        parser.error("D4-N requires an ordinary closed-door start")
    if protocol["recipe"] != "rgbd-conv-v1" or episode.get("launch_ready") is not True:
        parser.error("Expected an enabled RGB-D D4-N protocol")
    component = args.component_opening_deg is not None or args.component_max_sim_s is not None
    required_opening = protocol["collection_required_opening_deg"][asset_id]
    if (
        args.component_opening_deg is not None
        and not 0 < args.component_opening_deg < required_opening
    ):
        parser.error("Component opening must be positive and below the full task")
    if args.component_max_sim_s is not None and not 0 < args.component_max_sim_s < 90:
        parser.error("Component interval must be positive and below 90 s")
    recipe = json.loads(args.operational_recipe.read_text())
    if recipe.get("condition_scope") != "D4-N":
        parser.error("Operational recipe must declare D4-N")
    if (
        "solve_articulation_contact_last" in recipe
        or "enable_external_forces_every_iteration" in recipe
    ):
        parser.error("Use the observed Steel recipe without PhysX solver overrides")
    if not component and not protocol.get("demonstration_qualification_allowed"):
        parser.error("The protocol does not declare an ordinary full-cycle attempt")
    native = native_selected()
    native_rtx = os.environ.get("ISAAC_PHYSICS_PROFILE") == NATIVE_STEEL_PROFILE
    if native and not native_rtx and not component:
        parser.error("The native Warp D4 path supports component diagnostics only")
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    runtime = ExitStack()
    try:
        if not native:
            from isaaclab.app import AppLauncher

            app = AppLauncher(
                headless=True, enable_cameras=True, device="cuda:0", width=1280, height=720
            ).app
            runtime.callback(app.close)
        import torch

        from alexdoor_xas.perception.inspection import acquisition_inspection, load_inspection
        from alexdoor_xas.qualification.collection_dynamics import active_door_dynamics
        from alexdoor_xas.qualification.door_geometry import PreparedDoor
        from alexdoor_xas.qualification.expert import configure_physics_substeps, probe_config
        from alexdoor_xas.qualification.synthetic_probe import ProbeSetup
        from alexdoor_xas.simulation_profile import DEFAULT_PHYSICS_PROFILE
        from isaac_physics_profile import STEEL, selected_profile

        if not torch.cuda.is_available() or (
            not native and selected_profile(default=DEFAULT_PHYSICS_PROFILE) != STEEL
        ):
            raise RuntimeError("Observed D4-N requires CUDA and the Newton Steel profile")
        inspection = acquisition_inspection(
            load_inspection(REPO / "configs/perception_inspection.json"),
            recipe.get("inspection_acquisition"),
        )
        setup = ProbeSetup(**json.loads(args.setup.read_text()))
        setup.neck = tuple(inspection["waypoints"][-1][1:])
        folder = REPO / "datasets/doors/b1" / asset_id
        door = PreparedDoor(
            folder,
            json.loads((folder / "prepared.json").read_text()),
            json.loads((folder / "recipe.json").read_text()),
        )
        cfg = probe_config(door, setup, "cuda:0")
        cfg.seed = episode["seed"]
        cfg.camera_mount_pitch_rad = inspection["mount_pitch_rad"]
        if args.urdf_cache_dir is not None:
            cfg.urdf_cache_dir = str(args.urdf_cache_dir.resolve())
        cfg.observed_load_feedback = True
        cfg.pose_held_surface = protocol.get("minimum_motor_norm_surface_control") is True
        cfg.stop_stiffness = tuple(recipe["stop_stiffness"])
        cfg.pose_target_margin_rad = recipe.get("pose_target_margin_rad")
        cfg.pose_target_domain_rad = recipe.get("pose_target_domain_rad")
        cfg.pose_feedback_time_constant_s = recipe.get("pose_feedback_time_constant_s")
        cfg.pose_reference_feedforward = recipe.get("pose_reference_feedforward", False)
        cfg.pose_normal_compliance = recipe.get("pose_normal_compliance", False)
        cfg.pose_normal_transport = recipe.get("pose_normal_transport", False)
        cfg.pose_finite_tangent_impedance = recipe.get("pose_finite_tangent_impedance", False)
        cfg.stop_diagnostic_continuation_s = float(recipe["stop_diagnostic_continuation_s"])
        configure_physics_substeps(cfg, recipe["physics_substeps"])
        if native:
            from alexdoor_xas.qualification.b1_runtime import launch_native_steel

            if native_rtx:
                from isaaclab_ov.renderers import OVRTXRendererCfg

                cfg.native_renderer_cfg = OVRTXRendererCfg(
                    log_file_path=str((args.output / "ovrtx.log").resolve()), log_level="info"
                )
            runtime.enter_context(launch_native_steel(cfg))
        from alexdoor_xas.envs.door_task.door_push_purdue_env import DoorPushPurdueEnv

        env = DoorPushPurdueEnv(cfg)
        runtime.callback(env.close)
        env.reset()
        write_json(
            args.output / "runtime.json",
            dict(
                backend=os.environ.get("ISAAC_PHYSICS_PROFILE", STEEL),
                renderer=type(env.camera._renderer).__name__ if native else "Isaac RTX",
                solver=type(
                    env.sim.backend_manager.backend.solver
                    if native
                    else env.sim.physics_manager._solver
                ).__name__,
                gpu=torch.cuda.get_device_name(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                graph_requested=cfg.sim.physics.use_cuda_graph,
                record_tracker_stream=args.record_tracker_stream,
                control_dt=env.step_dt,
                physics_dt=cfg.sim.dt,
                component=component,
                dynamics=active_door_dynamics(env),
                setup=setup.to_dict(),
            ),
        )
        result = collect(args, env, door, setup, recipe, episode, inspection, required_opening)
        result["component_only"] = component
        result["wall_s"] = perf_counter() - started
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["graph_active"] = graph_active(env)
        admission_path = args.output / "operational/operational-admission.jsonl"
        if admission_path.exists():
            with admission_path.open() as stream:
                count, total = 0, 0.0
                for line in stream:
                    count += 1
                    total += json.loads(line)["wall_s"]
            result["motion_admission"] = dict(calls=count, wall_s=total)
        recording = args.output / "episode.hdf5"
        result["recording_bytes"] = recording.stat().st_size if recording.exists() else 0
        write_json(args.output / "report.json", result)
        print(json.dumps(result), flush=True)
        return 0 if result["passed"] else 1
    except Exception as error:
        write_json(
            args.output / "failure.json", dict(error=repr(error), traceback=traceback.format_exc())
        )
        raise
    finally:
        runtime.close()


def collect(args, env, door, setup, recipe, episode, inspection, required_opening):
    from uuid import uuid4

    import numpy as np
    from pxr import UsdLux

    from alexdoor_xas.assets.identity import RobotAssetRef
    from alexdoor_xas.envs.door_task.door_push_purdue_env import PEDESTAL
    from alexdoor_xas.qualification.collection_phase import qualify_opening_requirement
    from alexdoor_xas.qualification.stop import verify_stop_continuation
    from alexdoor_xas.qualification.synthetic_probe import run_probe
    from alexdoor_xas.recording.b1_runtime import camera_calibration
    from alexdoor_xas.recording.guarded_teacher import GuardedTeacherRecorder
    from alexdoor_xas.recording.quality import review_native_collection

    light = UsdLux.DomeLight(env.sim.stage.GetPrimAtPath("/World/Light"))
    intensity = light.GetIntensityAttr()
    intensity.Set(float(intensity.Get()) * episode.get("lighting_scale", 1.0))
    material, rotation = door.contact_pose(0.0, setup.contact_fraction, setup.contact_height)
    faces = [face @ rotation.T + material for face in env.push_geometry.distal_faces]
    if (
        door.pedestal_intersections(env.sim.stage, PEDESTAL)
        or not door.footprint_inside(np.concatenate(faces), 0.0)
        or door.footprint_obstructions(faces, setup.position_tolerance, rotation[:, 0])
    ):
        raise ValueError("unsupported_prescribed_contact_geometry")

    screen_initial_state(env, door, setup, args.output)
    if not graph_active(env):
        raise RuntimeError("Steel CUDA Graph was not captured during initial screening")
    robot = RobotAssetRef(
        "purdue_alex003_full_convex_wsg32_umi_v1",
        hashlib.sha256(env.robot_urdf_path.read_bytes()).hexdigest(),
    )
    metadata = dict(
        asset_id=episode["asset_id"],
        split=episode["split"],
        condition=episode["condition"],
        episode_family_id=episode["episode_id"],
        episode_id=f"{episode['episode_id']}:{uuid4()}",
        seed=episode["seed"],
        robot_asset=robot.to_dict(),
        setup=setup.to_dict(),
        inspection=inspection,
        component_only=args.component_opening_deg is not None
        or args.component_max_sim_s is not None,
    )
    opening = args.component_opening_deg or required_opening
    task_hold = dict(
        required_opening_deg=opening,
        minimum_s=setup.hold_settle_s,
        maximum_s=float(episode["completion_hold_max_s"]),
        handoff_at_required_opening=episode["completion_opening_handoff"],
    )
    budget = episode["collection_budget"]
    continuation_s = float(recipe["stop_diagnostic_continuation_s"])
    restore = install_interruption(
        env,
        max_wall_s=budget.get("wall_s_per_case"),
        max_sim_s=args.component_max_sim_s or (budget["sim_s_per_case"] - 2.0 - continuation_s),
    )
    recorder = GuardedTeacherRecorder(
        args.output / "episode.hdf5",
        metadata,
        operational_recipe=args.operational_recipe,
        robot_asset=robot,
        record_tracker_stream=args.record_tracker_stream,
        cache_temporal_frames=True,
        image_compression="lzf",
    )
    recorder.calibration = camera_calibration(env)
    try:
        try:
            result = run_probe(
                env,
                door,
                setup,
                args.output / "teacher",
                recorder=recorder,
                capture_evidence=False,
                inspection=inspection,
                task_hold=task_hold,
                controller=recorder.execute,
                rotate_at_withdrawal=args.component_opening_deg is not None,
            )
            result = qualify_opening_requirement(result, opening)
            if result["passed"] and (reason := recorder.io.safety_reason("stabilize")):
                result.update(passed=False, stop_reason=reason)
            recorder.finish(result)
            result["recording_complete"] = True
        except (Exception, KeyboardInterrupt) as error:
            result = dict(
                passed=False,
                recording_complete=False,
                stop_reason="budget_termination"
                if isinstance(error, CollectionBudgetExceeded)
                else "collection_exception",
                error=repr(error),
                traceback=traceback.format_exc(),
                task_failure_before_exception=getattr(error, "probe_task_failure", None),
                exception_phase=getattr(error, "probe_exception_phase", None),
            )
            recorder.interrupt(result)
        finally:
            restore()
            recorder.close_recording()
        result["post_stop"] = verify_stop_continuation(
            env, args.output / "post-stop.npz", force_limit_n=setup.force_limit_n
        )
        result["post_stop_diagnostic"] = verify_stop_continuation(
            env,
            args.output / "post-stop-diagnostic.npz",
            duration_s=continuation_s,
            force_limit_n=setup.force_limit_n,
        )
    finally:
        recorder.close()
    result["task_success"] = bool(result["passed"])
    result["safety_stop_passed"] = (
        result["post_stop"]["passed"] and result["post_stop_diagnostic"]["passed"]
    )
    result["recording_validation"] = (
        review_native_collection(recorder.path)
        if recorder.path.exists()
        else dict(integrity_passed=False, complete=False, data_usable=False)
    )
    component = metadata["component_only"]
    recording_ok = result["recording_validation"].get(
        "integrity_passed" if component else "required_scope_coverage"
    )
    result["passed"] = bool(
        result["task_success"] and result["safety_stop_passed"] and recording_ok
    )
    result["demonstration_qualified"] = bool(
        result["passed"] and args.component_opening_deg is None and args.component_max_sim_s is None
    )
    result["simulated_s"] = env.collection_physics_s + 2.0 + continuation_s
    return result


if __name__ == "__main__":
    sys.exit(main())
