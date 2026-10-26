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


class CollectionBudgetExceeded(TimeoutError):
    pass


def install_interruption(env, *, max_wall_s=None, max_sim_s=None):
    """Raise outside Kit callbacks, then restore unguarded stepping for the stop."""
    started = perf_counter()
    pending = []
    env.collection_physics_s = 0.0
    original_step = env.step

    def request(signum, _frame):
        pending.append(signum)

    def check():
        if max_wall_s is not None and perf_counter() - started >= max_wall_s:
            raise CollectionBudgetExceeded("collection_wall_budget")
        if max_sim_s is not None and env.collection_physics_s >= max_sim_s:
            raise CollectionBudgetExceeded("collection_simulation_budget")
        if pending and pending[0] == signal.SIGUSR1:
            raise CollectionBudgetExceeded("collection_resource_budget")
        if pending:
            raise KeyboardInterrupt(f"Pilot interrupted by signal {pending[0]}")

    def step(*args, **kwargs):
        check()
        result = original_step(*args, **kwargs)
        env.collection_physics_s += getattr(env, "step_dt", 0.0)
        check()
        return result

    env.step = step
    signal.signal(signal.SIGTERM, request)
    signal.signal(signal.SIGUSR1, request)
    return lambda: setattr(env, "step", original_step)


def screen_initial_state(env, door, setup, output):
    """Audit a parked nominal reset, then measure its unloaded stop before collection."""
    import numpy as np
    from PIL import Image

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "qualify", "replay", "rollout"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--setup", type=Path, help="Explicit prospective pilot setup; defaults to corpus setup"
    )
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--condition", help="One condition explicitly declared by the protocol")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--screen-initial-state",
        action="store_true",
        help="Audit the parked reset and its unloaded stop before a nominal collection",
    )
    parser.add_argument(
        "--intermediate-state",
        type=Path,
        help="Qualified source state; collection or separate temporal ACT phase diagnostic",
    )
    parser.add_argument(
        "--interrupt-after-commands",
        type=int,
        help="Controlled collection interruption after a real dispatch, before post-state",
    )
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--space", choices=("A1_joint_delta", "A2_ee_delta"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--temporal-act", action="store_true", help="Native causal temporal ACT A1 checkpoint"
    )
    parser.add_argument(
        "--reference-recording",
        type=Path,
        help="Diagnostic A1 command replay through the same operational checks; no learned success",
    )
    parser.add_argument(
        "--reference-prefix-recording",
        type=Path,
        help="Diagnostic observed prefix followed by temporal ACT; never full learned success",
    )
    parser.add_argument(
        "--reference-prefix-commands",
        type=int,
        help="Positive multiple of six staging commands, followed permanently by ACT",
    )
    parser.add_argument(
        "--rollout-compression",
        choices=("none", "lzf", "gzip"),
        default="none",
        help="Optional lossless RGB-D compression for retained rollout evidence",
    )
    parser.add_argument(
        "--collection-compression",
        choices=("none", "lzf", "gzip"),
        default="none",
        help="Optional lossless RGB-D compression for new teacher recordings",
    )
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
        help="Historical recipe identity; must agree with the protocol and checkpoint",
    )
    args = parser.parse_args(argv)
    if args.temporal_act and (
        args.mode != "rollout" or args.family != "act" or args.reference_recording
    ):
        parser.error("Temporal ACT requires learned ACT rollout without reference replay")
    if (args.reference_prefix_recording is None) != (args.reference_prefix_commands is None):
        parser.error("Reference prefix requires both recording and command count")
    if args.reference_prefix_recording is not None and (
        not args.temporal_act
        or args.expert_reference is None
        or args.reference_recording
        or args.intermediate_state
        or args.reference_prefix_commands <= 0
        or args.reference_prefix_commands % 6
    ):
        parser.error("Reference prefix requires normal temporal ACT and six-row boundaries")
    if args.rollout_compression != "none" and args.mode != "rollout":
        parser.error("Rollout compression is only available for rollout evidence")
    if args.collection_compression != "none" and args.mode != "collect":
        parser.error("Collection compression is only available for teacher collection")
    if args.screen_initial_state and (args.mode != "collect" or args.intermediate_state):
        parser.error("Initial screening requires closed-door collection")
    if args.reference_recording is not None and (args.mode != "rollout" or args.family != "act"):
        parser.error("Reference diagnostic requires rollout with the fixed ACT A1 checkpoint")
    if args.intermediate_state and not (args.mode == "collect" or args.temporal_act):
        parser.error("Intermediate start requires collection or temporal ACT phase diagnostic")
    if args.interrupt_after_commands is not None and args.mode != "collect":
        parser.error("Interruption is collection-only")
    if args.interrupt_after_commands is not None and args.interrupt_after_commands < 1:
        parser.error("Interruption command count must be positive")
    protocol = json.loads(args.protocol.read_text())
    from alexdoor_xas.policies.encoding import LEARNED_RECIPE, protocol_recipe

    try:
        args.recipe = protocol_recipe(protocol, args.recipe)
    except ValueError as error:
        parser.error(str(error))
    if args.mode in ("collect", "qualify") and args.recipe != LEARNED_RECIPE:
        parser.error("New pilot collection requires rgbd-conv-v1; grid recipes are historical")
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
        args.checkpoint is None
        and args.reference_recording is None
        or args.family is None
        or args.operational_recipe is None
    ):
        parser.error(
            "Rollout requires a declared train/development condition, checkpoint family "
            "and operational recipe"
        )
    campaign = selected[0].get("collection_case")
    if campaign is not None and (
        args.mode != "collect" or campaign.get("family") not in ("A", "B", "C", "D")
    ):
        parser.error("A short campaign cell requires collect and family A/B/C/D")
    if (
        campaign is not None
        and campaign.get("record_inspection_history", False)
        and (campaign["family"] != "A" or args.intermediate_state is not None)
    ):
        parser.error("Common inspection history requires a normal-start A collection")
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
    from alexdoor_xas.qualification.expert import configure_physics_substeps, probe_config
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, run_probe
    from alexdoor_xas.recording.b1_runtime import camera_calibration
    from alexdoor_xas.recording.teacher import TemporalExpertRecorder

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
    stop_diagnostic_s = 0.0
    if args.operational_recipe is not None:
        recipe = json.loads(args.operational_recipe.read_text())
        cfg.stop_stiffness = tuple(recipe.get("stop_stiffness", cfg.stop_stiffness))
        configure_physics_substeps(cfg, recipe.get("physics_substeps", cfg.decimation))
        stop_diagnostic_s = float(recipe.get("stop_diagnostic_continuation_s", 0.0))
        if not math.isfinite(stop_diagnostic_s) or stop_diagnostic_s < 0:
            raise ValueError("Invalid stop diagnostic duration")
        cfg.stop_diagnostic_continuation_s = stop_diagnostic_s
    env = DoorPushPurdueEnv(cfg)
    env.reset()  # Establish the configured robot state before any fallible setup.
    budget = (
        selected[0].get("collection_budget", protocol.get("collection_budget", {}))
        if args.mode in ("collect", "qualify")
        else {}
    )
    sim_guard = budget.get("sim_s_per_case")
    restore_interruption = install_interruption(
        env,
        max_wall_s=budget.get("wall_s_per_case"),
        max_sim_s=None if sim_guard is None else sim_guard - 2.0 - stop_diagnostic_s,
    )
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
            stop_stiffness=list(cfg.stop_stiffness),
            physics_dt=cfg.sim.dt,
            control_dt=env.step_dt,
            physics_substeps=cfg.decimation,
            render_interval=cfg.sim.render_interval,
            lighting=dict(
                scope="simulated_dome_light",
                scale=lighting_scale,
                base_intensity=base_intensity,
                intensity=base_intensity * lighting_scale,
            ),
        ),
    )
    from alexdoor_xas.qualification.collection_dynamics import active_door_dynamics

    write_json(
        args.output / "active-dynamics.json",
        dict(
            active=active_door_dynamics(env),
            authored=getattr(env, "door_dynamics_override", None),
        ),
    )
    started = perf_counter()
    stop_attempted = False
    try:
        if args.mode in ("collect", "qualify"):
            # These geometric checks supplement every-row physical contact checks.
            import numpy as np

            from alexdoor_xas.envs.door_task.door_push_purdue_env import PEDESTAL

            intersections = door.pedestal_intersections(env.sim.stage, PEDESTAL)
            contact, rotation = door.contact_pose(0.0, setup.contact_fraction, setup.contact_height)
            faces = [face @ rotation.T + contact for face in env.push_geometry.distal_faces]
            support = door.footprint_inside(np.concatenate(faces), 0.0)
            obstructions = door.footprint_obstructions(
                faces, setup.position_tolerance, rotation[:, 0]
            )
            write_json(
                args.output / "geometry-support.json",
                dict(
                    pedestal_intersections=intersections,
                    footprint_inside=bool(support),
                    footprint_obstructions=obstructions,
                ),
            )
            if intersections or not support or obstructions:
                raise ValueError("unsupported_prescribed_contact_geometry")
        if args.screen_initial_state:
            screen_initial_state(env, door, setup, args.output)
        if args.mode == "qualify":
            # One current-setup applicability check of a prospectively fixed
            # requirement. This does not publish a new maximum-opening reference
            # or produce expert supervision; corpus reference pairing is unchanged.
            result = run_probe(
                env,
                door,
                setup,
                args.output / "teacher",
                capture_evidence=False,
                inspection=inspection,
            )
            required = protocol["collection_required_opening_deg"][args.asset_id]
            result.update(
                required_opening_deg=required,
                opening_requirement_passed=bool(
                    result["angle_deg"] is not None and result["angle_deg"] >= required
                ),
                scope="fixed_requirement_current_setup_applicability",
                expert_supervision=False,
                publishes_expert_reference=False,
            )
            result["passed"] = bool(result["passed"] and result["opening_requirement_passed"])
        elif args.mode == "collect":
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
            if campaign is not None:
                from alexdoor_xas.qualification.collection_phase import collection_supervised_phases

                required_phases = {
                    "A": ["approach", "contact"],
                    "B": ["approach", "contact", "push"],
                    "C": ["push", "hold", "release", "stabilize"],
                    "D": ["hold", "release", "stabilize"],
                }[campaign["family"]]
                metadata.update(
                    subtask=dict(
                        name=campaign["family"],
                        required_phases=required_phases,
                        supervised_phases=collection_supervised_phases(campaign["family"]),
                    ),
                    collection_case=campaign,
                    static_reference_time_s=(
                        float(inspection["waypoints"][-1][0])
                        if campaign.get("record_inspection_history", False)
                        else 0.0
                    ),
                    action_roles={"expert": 0, "injected_excluded": 1, "corrective": 2},
                )
                if campaign["family"] in ("C", "D") and args.intermediate_state is None:
                    raise ValueError(
                        "Late short phases require a qualified actual-condition source"
                    )
            intermediate = None
            if args.intermediate_state is not None:
                import numpy as np

                intermediate = dict(np.load(args.intermediate_state, allow_pickle=False))
                origin = json.loads(str(intermediate["metadata"]))
                source_result = json.loads(
                    Path(
                        origin.get("source_result", args.intermediate_state.parent / "result.json")
                    ).read_text()
                )
                origin_setup = ProbeSetup(**origin["setup"]).to_dict()
                actual_setup = setup.to_dict()
                timing_fields = {
                    "approach_s",
                    "contact_s",
                    "release_s",
                    "preroll_s",
                    "horizon_s",
                    "intermediate_preroll_s",
                }
                if campaign is None:
                    timing_fields = set()
                same_condition = all(
                    json.loads(json.dumps(value)) == json.loads(json.dumps(actual_setup[key]))
                    for key, value in origin_setup.items()
                    if key not in timing_fields
                )
                expected_phase = (
                    "push" if campaign is not None and campaign["family"] in ("C", "D") else "hold"
                )
                if (
                    str(intermediate.get("phase", "hold")) != expected_phase
                    or origin["asset_id"] != args.asset_id
                    or origin["split"] != entry["split"]
                    or not same_condition
                    or not source_result["passed"]
                    or not source_result["released"]
                    or not source_result["stabilized"]
                ):
                    raise ValueError(
                        "Intermediate start requires a qualified same-group source/setup"
                    )
                metadata.update(
                    intermediate_start=True,
                    source_episode_id=origin["episode_id"],
                    source_state=str(args.intermediate_state.resolve()),
                    episode_id=metadata["episode_id"] + "-intermediate",
                )
            elif args.interrupt_after_commands is not None:
                metadata["episode_id"] += "-interrupted"
            # Distinct attempts can later coexist in one train normalization set.
            # Keep source linkage separately; never rename original recordings.
            from uuid import uuid4

            metadata["episode_family_id"] = metadata["episode_id"]
            metadata["episode_id"] += ":" + str(uuid4())
            task_hold = None
            if selected[0].get("completion_hold_max_s") is not None:
                if campaign is not None or intermediate is not None:
                    raise ValueError("Task-goal hold requires complete collection")
                task_hold = dict(
                    required_opening_deg=protocol["collection_required_opening_deg"][args.asset_id],
                    minimum_s=setup.hold_settle_s,
                    maximum_s=float(selected[0]["completion_hold_max_s"]),
                )
                metadata["task_hold_completion"] = task_hold
            recorder = TemporalExpertRecorder(
                args.output / "episode.hdf5",
                metadata,
                interrupt_after_commands=args.interrupt_after_commands,
                image_compression=(
                    None if args.collection_compression == "none" else args.collection_compression
                ),
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
                    inspection=(
                        inspection
                        if campaign is None or campaign.get("record_inspection_history", False)
                        else None
                    ),
                    intermediate=intermediate,
                    campaign=campaign,
                    task_hold=task_hold,
                )
                required = (
                    None
                    if campaign is not None
                    else protocol.get("collection_required_opening_deg", {}).get(args.asset_id)
                )
                if required is not None:
                    from alexdoor_xas.qualification.collection_phase import (
                        qualify_opening_requirement,
                    )

                    result = qualify_opening_requirement(result, required)
                # One finalization owner, after the ordinary physical tail.
                # Safety-stop verification happens after detaching the recorder.
                recorder.finish(result)
                result.update(
                    task_success=bool(result["passed"] and campaign is None),
                    recording_complete=True,
                )
            except BaseException as error:
                if not isinstance(error, (Exception, KeyboardInterrupt)):
                    raise
                termination = (
                    "budget_termination"
                    if isinstance(error, CollectionBudgetExceeded)
                    else "interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "collection_exception"
                )
                result = dict(
                    passed=False,
                    task_success=False,
                    recording_complete=False,
                    stop_reason=termination,
                    termination=termination,
                    error=repr(error),
                    traceback=traceback.format_exc(),
                )
                recorder.interrupt(result)
            finally:
                recorder.close()
            result["online_checks"] = recorder.online_checks
        elif args.mode == "replay":
            result = replay(env, args, protocol)
        else:
            result = rollout(env, args, protocol, robot, setup, inspection)
        restore_interruption()
        if not stop_attempted:
            from alexdoor_xas.qualification.stop import verify_stop_continuation

            stop_attempted = True
            result["post_stop"] = verify_stop_continuation(
                env, args.output / "post-stop.npz", force_limit_n=setup.force_limit_n
            )
            if not result["post_stop"]["passed"]:
                result["passed"] = False
        if stop_diagnostic_s:
            result["post_stop_diagnostic"] = verify_stop_continuation(
                env,
                args.output / "post-stop-diagnostic.npz",
                duration_s=stop_diagnostic_s,
                force_limit_n=setup.force_limit_n,
            )
        result["safety_stop_passed"] = bool(
            result["post_stop"]["passed"]
            and result.get("post_stop_diagnostic", {"passed": True})["passed"]
        )
        if args.temporal_act:
            result["passed"] = bool(result["passed"] and result["safety_stop_passed"])
            if "intermediate_phase_success" in result:
                result["intermediate_phase_success"] &= result["safety_stop_passed"]
        if args.mode == "collect":
            from alexdoor_xas.recording.quality import review_native_collection

            path = args.output / "episode.hdf5"
            result["recording_validation"] = (
                review_native_collection(path)
                if path.exists()
                else dict(
                    integrity_passed=False,
                    complete=False,
                    data_usable=False,
                    reason="recording_not_started",
                )
            )
            result["data_usable"] = result["recording_validation"]["data_usable"]
            result["full_task_success"] = result["recording_validation"].get(
                "full_task_success", False
            )
            result["demonstration_qualified"] = bool(
                (
                    result.get("subtask_success", False)
                    if campaign is not None
                    else result["task_success"]
                )
                and result["recording_validation"].get("required_scope_coverage", False)
            )
            result["passed"] = bool(
                result["demonstration_qualified"] and result["safety_stop_passed"]
            )

        result["accounted_simulation_s"] = (
            env.collection_physics_s
            + result["post_stop"]["duration_s"]
            + result.get("post_stop_diagnostic", {}).get("duration_s", 0.0)
        )
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
        restore_interruption()
        if not stop_attempted:
            from alexdoor_xas.qualification.stop import verify_stop_continuation

            try:
                stopping = verify_stop_continuation(
                    env, args.output / "interrupted-stop.npz", force_limit_n=setup.force_limit_n
                )
                write_json(args.output / "interrupted-stop.json", stopping)
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
    import torch
    from PIL import Image
    from scipy.spatial.transform import Rotation

    from alexdoor_xas.perception.evaluation import json_safe
    from alexdoor_xas.policies.b1 import B1Policy
    from alexdoor_xas.policies.encoding import pilot_binding
    from alexdoor_xas.policies.pilot_runtime import observed_pilot_runtime
    from alexdoor_xas.policies.purdue import contact_status
    from alexdoor_xas.policies.rollout import B1Runner, MatchedReplay
    from alexdoor_xas.qualification.pilot_evaluation import PilotEvaluation, physical_row
    from alexdoor_xas.recording.b1 import PHASES
    from alexdoor_xas.recording.b1_runtime import array
    from alexdoor_xas.recording.rollout import RolloutWriter

    binding = pilot_binding(
        inspection,
        depth_interval_m=(env.capture.near_m, env.capture.far_m),
        recipe=args.recipe,
    )
    if args.temporal_act:
        from alexdoor_xas.action.spaces import A1_JOINT_DELTA
        from alexdoor_xas.policies.act.execution import temporal_a1_runner
        from alexdoor_xas.policies.act.native_temporal import NativeTemporalEvents
        from alexdoor_xas.policies.temporal import TemporalConfig
        from alexdoor_xas.recording.teacher import simulated_feedback_contract

        config = TemporalConfig(
            static_time_s=0.0 if args.intermediate_state else float(inspection["waypoints"][-1][0]),
            depth_interval_m=(env.capture.near_m, env.capture.far_m),
        )
        startup = None
        if args.intermediate_state:
            from alexdoor_xas.qualification.temporal_loaded_start import loaded_phase_startup

            startup = loaded_phase_startup(
                env,
                setup,
                args.intermediate_state,
                asset_id=args.asset_id,
                output=args.output / "loaded-initialization",
            )
        if env.cfg.decimation != 4:
            raise ValueError("Temporal learned evaluation requires qualified 240 Hz physics")
        observer, io, geometry, motion, load_log, contact_log = observed_pilot_runtime(
            env,
            binding,
            robot,
            setup,
            args.operational_recipe,
            args.output,
            A1_JOINT_DELTA,
            initialization_s=setup.intermediate_preroll_s if args.intermediate_state else None,
        )
        binding = observer.binding
        events = NativeTemporalEvents(env, config)
        diagnostic_prefix = None
        if args.reference_prefix_recording is not None:
            import h5py

            from alexdoor_xas.policies.act.execution import DiagnosticA1Prefix

            with h5py.File(args.reference_prefix_recording, "r") as source:
                metadata = json.loads(source["metadata"].attrs["episode"])
                if (
                    not bool(source.attrs.get("complete", False))
                    or bool(source.attrs.get("full_task_eligible", True))
                    or metadata["asset_id"] != args.asset_id
                    or metadata.get("sim_dt") != env.cfg.sim.dt
                ):
                    raise ValueError("Prefix requires a complete matched diagnostic source")
                times = source["commands/time_s"][:]
                targets = source["commands/joint_target"][:]
                measured = source["observations/joint_position"][: len(times), :7]
                diagnostic_prefix = DiagnosticA1Prefix(
                    times, targets - measured, args.reference_prefix_commands, io.dt
                )
            if protocol["rollout_max_ticks"] <= args.reference_prefix_commands:
                raise ValueError("Phase budget must include actual ACT commands after the prefix")
        runner = temporal_a1_runner(
            args.checkpoint,
            observer,
            io,
            config=config,
            episode_id=args.output.name,
            events=events,
            feedback_contract=simulated_feedback_contract(env),
            admission_source=motion,
            startup=startup,
            diagnostic_prefix=diagnostic_prefix,
        )
        observer, policy = runner.observer, runner.policy
    else:
        if args.reference_recording is not None:
            from alexdoor_xas.action.spaces import A1_JOINT_DELTA

            policy = SimpleNamespace(action_space=A1_JOINT_DELTA)
        else:
            policy = B1Policy.from_checkpoint(
                args.checkpoint,
                args.family,
                binding=binding,
                runtime_asset=robot,
                device="cuda:0",
                num_inference_steps=protocol["ddim_steps"],
            )
        observer, io, geometry, motion, load_log, contact_log = observed_pilot_runtime(
            env, binding, robot, setup, args.operational_recipe, args.output, policy.action_space
        )
    reference_mode = args.reference_recording is not None
    prefix_mode = args.reference_prefix_recording is not None
    if reference_mode:
        import h5py

        from alexdoor_xas.action.spaces import A1_JOINT_DELTA

        if policy.action_space != A1_JOINT_DELTA:
            raise ValueError("Reference diagnostic requires fixed ACT A1")
        with h5py.File(args.reference_recording, "r") as source:
            if not bool(source.attrs.get("complete", False)):
                raise ValueError("Incomplete reference recording")
            metadata = json.loads(source["metadata"].attrs["episode"])
            if metadata["asset_id"] != args.asset_id:
                raise ValueError("Reference asset differs from the declared diagnostic")
            commands = source["commands"]
            times = commands["time_s"][:]
            # The recording contains inspection targets too. Reference actions
            # start after the runner's unchanged common inspection schedule.
            start = inspection["waypoints"][-1][0] + binding.config["warmup_s"]
            use = times >= start - 1e-8
            targets = commands["joint_target"][:][use]
            if targets.shape != (int(use.sum()), 7):
                raise ValueError("Reference commands require seven ordered arm targets")
            absolute_targets = source.attrs.get("action_semantics") == "absolute_joint_targets"
            actions = targets
            if not absolute_targets:
                measured = source["observations/joint_position"][: len(times)][use, :7]
                actions = targets - measured
            record = SimpleNamespace(times=times[use], actions=actions)
        dataset = SimpleNamespace(
            by_id=lambda _: record,
            binding=binding,
            robot_asset=robot,
            action_space=A1_JOINT_DELTA,
        )
        policy = MatchedReplay(dataset, "reference", absolute_joint_targets=absolute_targets)
    recording = RolloutWriter(
        args.output / "rollout.hdf5",
        dict(
            episode_id=args.output.name,
            asset_id=args.asset_id,
            split=next(
                e["split"] for e in protocol["evaluation_panel"] if e["asset_id"] == args.asset_id
            ),
            joint_names=list(env.robot.joint_names),
            clock_semantics="native_synchronous_simulation",
            sim_dt=env.cfg.sim.dt,
            purpose="operational_reference_diagnostic"
            if reference_mode
            else "observed_prefix_phase_diagnostic"
            if prefix_mode
            else "intermediate_phase_diagnostic"
            if args.intermediate_state
            else "learned_rollout_evaluation",
            intermediate_state=str(args.intermediate_state) if args.intermediate_state else None,
            full_task_eligible=not bool(args.intermediate_state or reference_mode or prefix_mode),
            reference_recording=str(args.reference_recording) if reference_mode else None,
            reference_prefix_recording=str(args.reference_prefix_recording)
            if prefix_mode
            else None,
            reference_prefix_commands=args.reference_prefix_commands if prefix_mode else None,
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
        image_compression=None if args.rollout_compression == "none" else args.rollout_compression,
    )
    from alexdoor_xas.recording.teacher import simulated_feedback_contract

    recording.file["metadata"].attrs["feedback_contract"] = json.dumps(
        simulated_feedback_contract(env)
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
        evaluation = PilotEvaluation(
            setup,
            reference["theta_expert_d"],
            io.dt,
            native_samples=env.cfg.decimation,
            require_stabilization=args.temporal_act or reference_mode,
        )

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
            # Retain the setter's actual pre-dispatch clock. Subtracting dt
            # from the post-step clock can change strict previous-target joins.
            time_s=np.float64(env.command_event["request_s"]),
            kind=np.int8(kind),
            request=request,
            request_width=np.int8(width),
            learned=np.bool_(
                command is not None
                and not reference_mode
                and not getattr(policy, "reference_command", False)
                and not getattr(io, "initializing_phase", False)
            ),
            reference=np.bool_(
                command is not None
                and (reference_mode or getattr(policy, "reference_command", False))
            ),
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
                reference_command=bool(pending["reference"]),
            )
            independent.update(evaluation.update(physical) if evaluation is not None else physical)
        evaluator_states.write(json.dumps(json_safe(independent), allow_nan=False) + "\n")
        if pending is not None:
            recording.command({k: v for k, v in pending.items() if k != "reference"})
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
    last_perception = last_perception_clock = None

    def measured_update(sensor):
        nonlocal last_completed, last_perception, last_perception_clock
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
        publication = geometry.last_publication
        if publication is not None:
            clock = (observer.generation, int(publication[1]["frame"]))
            if clock != last_perception_clock:
                from copy import deepcopy

                # The runner invalidates/reset geometry when it stops. Keep the
                # last published diagnostic, without touching policy observations.
                last_perception = deepcopy(geometry.diagnostics)
                last_perception_clock = clock
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
    prediction_policy = policy.policy if args.temporal_act else policy
    predict, prediction_ms = (None if reference_mode else prediction_policy.predict), []

    def measured_predict(features, **options):
        # Preserve the actual causal vector before inference, even if it fails.
        # These diagnostic files never feed the policy, admission or controller.
        first = not prediction_ms
        if first:
            sample = env.capture.sample
            np.savez_compressed(
                args.output / "first-policy-input.npz",
                **(features["obs"] if args.temporal_act else dict(features=features)),
                time_s=np.float64(sample.time_s),
                frame=np.int64(sample.frame),
            )
            write_json(
                args.output / "policy-backend.json",
                dict(
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,
                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                    mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
                    inference_mode=torch.is_inference_mode_enabled(),
                    grad_enabled=torch.is_grad_enabled(),
                    cuda_autocast=torch.is_autocast_enabled("cuda"),
                    features_dtype="temporal_dict"
                    if args.temporal_act
                    else str(np.asarray(features).dtype),
                    rgb_stride=list(sample.rgb.stride()),
                    depth_stride=list(sample.depth_m.stride()),
                ),
            )
        begin = perf_counter()
        chunk = predict(features, **options)
        prediction_ms.append((perf_counter() - begin) * 1000)
        if first:
            np.save(args.output / "first-policy-prediction.npy", chunk, allow_pickle=False)
        return chunk

    if not reference_mode:
        prediction_policy.predict = measured_predict
    if not args.temporal_act:
        runner = B1Runner(
            policy,
            observer,
            io,
            n_action_steps=None if reference_mode else protocol["replan_ticks"],
            admission_source=motion,
        )
    if args.temporal_act:
        # Independent terminal scoring only; no teacher action, takeover or
        # privileged feature reaches the policy. Ordinary stabilization has
        # already been physically commanded before this predicate can pass.
        runner.completion_source = lambda: (
            evaluation is not None and evaluation.result()["task_success"]
        )
    try:
        result = runner.run(
            max_ticks=(policy.control_ticks if reference_mode else protocol["rollout_max_ticks"]),
            seed=protocol["inference_seed"],
        )
        physical_result = None if evaluation is None else evaluation.result()
        success_key = "reference_cycle_success" if reference_mode else "task_success"
        episode_passed = False if physical_result is None else physical_result[success_key]
        if args.temporal_act:
            episode_passed &= result.reason == "evaluated_task_complete"
        phase_passed = bool(episode_passed and args.intermediate_state)
        episode_passed &= not prefix_mode
        episode_passed &= not bool(args.intermediate_state)
        outcome = dict(
            passed=episode_passed,
            intermediate_phase_success=phase_passed,
            full_task_eligible=not bool(args.intermediate_state or reference_mode or prefix_mode),
            termination=result.reason,
            learned_commands=(
                physical_result["learned_commands"]
                if physical_result is not None
                else 0
                if reference_mode
                else result.control_steps
            ),
            reference_commands=(
                physical_result["reference_commands"]
                if physical_result is not None
                else result.control_steps
                if reference_mode
                else 0
            ),
            physical_evaluation=(
                "unavailable_frozen_reference" if physical_result is None else physical_result
            ),
            qualification=False,
        )
        write_json(
            args.output / "runner-result.json",
            dict(
                runner=asdict(result),
                physical_evaluation=physical_result,
                recording_observations=recording.count,
            ),
        )
        if recording.count:
            recording.finish(outcome)
        else:
            recording.finish_unavailable(outcome)
        recording_complete = bool(recording.file.attrs["complete"])

    finally:
        recording.close()
        native_contacts.close()
        evaluator_states.close()
        motion.close()
        load_log.close()
        contact_log.close()
        try:
            from alexdoor_xas.perception.evaluation import write_json as write_perception_json

            write_perception_json(
                args.output / "operational-perception-terminal.json",
                geometry.diagnostics if last_perception is None else last_perception,
            )
        except (TypeError, ValueError) as error:
            write_json(
                args.output / "operational-perception-terminal.json",
                dict(available=False, reason=repr(error)),
            )
        finally:
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
        passed=episode_passed,
        recording_complete=recording_complete,
        intermediate_phase_success=phase_passed,
        full_task_eligible=not bool(args.intermediate_state or reference_mode or prefix_mode),
        intermediate_state=str(args.intermediate_state) if args.intermediate_state else None,
        physical_evaluation=physical_result,
        runner=asdict(result),
        family=args.family,
        action_space=policy.action_space,
        checkpoint=str(args.checkpoint),
        scope=(
            "operational_reference_diagnostic"
            if reference_mode
            else "observed_prefix_phase_diagnostic"
            if prefix_mode
            else "intermediate_phase_diagnostic"
            if args.intermediate_state
            else "operational_rollout_attempt_no_admission_bypass"
        ),
        reference_recording=str(args.reference_recording) if reference_mode else None,
        reference_commands=(
            physical_result["reference_commands"]
            if physical_result is not None
            else result.control_steps
            if reference_mode
            else 0
        ),
        reference_prefix_recording=str(args.reference_prefix_recording) if prefix_mode else None,
        reference_prefix_commands=args.reference_prefix_commands if prefix_mode else None,
        reference_cycle_success=(
            False if physical_result is None else physical_result["reference_cycle_success"]
        ),
        inspection_ticks=len(rows),
        policy_commands=(
            physical_result["learned_commands"]
            if physical_result is not None
            else 0
            if reference_mode
            else result.control_steps
        ),
        retained_inputs="rollout.hdf5",
        rollout_image_compression=args.rollout_compression,
        first_policy_input="first-policy-input.npz" if prediction_ms else None,
        first_policy_prediction="first-policy-prediction.npy" if prediction_ms else None,
        policy_backend="policy-backend.json" if prediction_ms else None,
        prediction_queries=len(prediction_ms),
        prediction_total_s=sum(prediction_ms) / 1000,
        prediction_p50_ms=float(np.median(prediction_ms)) if prediction_ms else None,
        prediction_p95_ms=float(np.percentile(prediction_ms, 95)) if prediction_ms else None,
        retained_native_contacts="evaluator-native-contacts.jsonl",
        retained_evaluator_states="evaluator-control-state.jsonl",
        observed_geometry_available=True,
        load_source_available=True,
        action_admission_source_available=True,
        safety_detail=io.safety.checks.last_failure,
        operational_recipe=str(args.operational_recipe),
        operational_admission="operational-admission.jsonl",
        operational_load="operational-load.jsonl",
        operational_perception_terminal="operational-perception-terminal.json",
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
    except KeyboardInterrupt:
        traceback.print_exc()
        code = 130
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
