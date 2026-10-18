"""Fresh-process observer diagnostics; scenario controls never enter perception IPC."""

import json
import time
from pathlib import Path

import numpy as np


def run_live(recipe, output, models, asset_id, *, case, selection=None, duration_s=65):
    from isaaclab.app import AppLauncher

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
    env = None
    try:
        from alexdoor_xas.envs.door_task.door_push_purdue_env import DoorPushPurdueEnv
        from alexdoor_xas.qualification.door_geometry import PreparedDoor
        from alexdoor_xas.qualification.expert import probe_config
        from alexdoor_xas.qualification.synthetic_probe import ProbeSetup
        from alexdoor_xas.recording.b1_runtime import array, camera_calibration, camera_from_joints

        root = Path(__file__).resolve().parents[4]
        folder = root / "datasets/doors/b1" / asset_id
        door = PreparedDoor(
            folder,
            json.loads((folder / "prepared.json").read_text()),
            json.loads((folder / "recipe.json").read_text()),
        )
        setup = ProbeSetup(**json.loads((root / "configs/purdue_synthetic_probe.json").read_text()))
        inspection = recipe.config["inspection"]
        completed = inspection["waypoints"][0][0]
        setup.neck = tuple(
            next(w[1:] for w in reversed(inspection["waypoints"]) if w[0] <= completed)
        )
        cfg = probe_config(door, setup, "cuda:0")
        cfg.camera_mount_pitch_rad = recipe.config["inspection"]["mount_pitch_rad"]
        env = DoorPushPurdueEnv(cfg)
        env.reset()
        calibration = camera_calibration(env)

        epoch = [time.perf_counter()]
        acquisition = [0.0]
        native_capture = env.capture.capture

        def capture(*args, **kwargs):
            acquisition[0] = time.perf_counter() - epoch[0]
            return native_capture(*args, **kwargs)

        def start_clock():
            epoch[0], acquisition[0] = time.perf_counter(), 0.0
            return epoch[0]

        env.capture.capture = capture

        def observe():
            sample = env.capture.sample
            q = array(sample.joint_position)[0]
            return dict(
                time_s=np.asarray(acquisition[0]),
                frame=np.asarray(sample.frame),
                rgb=array(sample.rgb)[0],
                depth_m=array(sample.depth_m)[0],
                valid_depth=array(sample.valid_depth)[0],
                joint_position=q,
                joint_velocity=array(sample.joint_velocity)[0],
                camera_world=camera_from_joints(calibration, q),
                intrinsics=array(env.camera.data.intrinsic_matrices)[0],
            )

        return observer_case(
            env,
            door,
            setup,
            calibration,
            observe,
            recipe,
            models,
            output,
            case,
            start_clock,
            selection,
            duration_s,
        )
    except Exception as error:
        from alexdoor_xas.perception.evaluation import write_json

        write_json(
            output / "failure.json", dict(error=f"{type(error).__name__}: {error}", qualified=False)
        )
        raise
    finally:
        if env:
            env.close()
        app.close()


def observer_case(
    env,
    door,
    setup,
    calibration,
    observe,
    recipe,
    models,
    output,
    case,
    start_clock,
    selection,
    duration_s,
):
    """Bounded observer motion/fault cases; truth and fault labels stay in this evaluator."""
    import torch

    from alexdoor_xas.action.frames import rot_z
    from alexdoor_xas.perception.diagnostics.replay import (
        pose_error,
        support_timeline,
        visible_references,
    )
    from alexdoor_xas.perception.evaluation import quantiles, write_json
    from alexdoor_xas.perception.material_zone import pose_matrix
    from alexdoor_xas.perception.point2pose.runtime import tracking_provider
    from alexdoor_xas.recording.b1_runtime import array

    if case not in ("stationary", "camera", "panel", "combined", "visibility"):
        raise ValueError("unknown_observer_case")
    provider = tracking_provider(
        recipe, models, calibration, output, replay=False, selection=selection
    )
    clock_start = start_clock()
    provider.engine.clock = lambda: time.perf_counter() - clock_start
    from alexdoor_xas.perception.diagnostics.operational import OperationalRecorder

    operational = OperationalRecorder(output)
    zero = torch.zeros((1, 6), device=env.device)
    records, events, reference, previous = [], [], None, None
    captured_truth = {}
    captured_cameras = {}
    motion_start = None
    neck = np.array(setup.neck)

    def truth():
        angle = door.sign * float(array(env.door.data.joint_pos)[0, 0])
        transform = np.eye(4)
        transform[:3, :3] = rot_z(angle)
        transform[:3, 3] = door.hinge
        return transform

    try:
        while time.perf_counter() - clock_start < duration_s:
            phase = 0.0 if motion_start is None else time.perf_counter() - motion_start
            if motion_start is None:
                elapsed = time.perf_counter() - clock_start
                target = next(
                    w[1:]
                    for w in reversed(recipe.config["inspection"]["waypoints"])
                    if w[0] <= elapsed
                )
                env.set_neck_target(np.asarray(target))
            elif case in ("camera", "combined"):
                limits = array(env.robot.data.joint_pos_limits)[0, env.neck_ids]
                target = neck + [0.025 * (1 - np.cos(phase)), 0.015 * (1 - np.cos(phase * 0.7))]
                env.set_neck_target(np.clip(target, limits[:, 0], limits[:, 1]))
            if case in ("panel", "combined"):
                angle = 0.10 * (1 - np.cos(min(phase, 8.0) * np.pi / 8))
                q = torch.full((1, 1), angle, device=env.device)
                env.door.write_joint_position_to_sim_index(position=q)
                env.door.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            env.step(zero)
            sensor = observe()
            if previous == int(sensor["frame"]):
                continue
            previous = int(sensor["frame"])
            expected_truth = truth()
            captured_truth[previous] = expected_truth.copy()
            captured_cameras[previous] = sensor["camera_world"].copy()
            fault = "none"
            # Controlled input faults are diagnostics, not physical occluder qualification.
            if case == "visibility" and reference is not None:
                if 3 <= phase < 5:
                    fault = "zone_covered"
                    from alexdoor_xas.perception.geometry import project

                    zone = expected_truth @ np.linalg.inv(reference[0]) @ reference[1]
                    uv, z = project(zone[None, :3, 3], sensor["intrinsics"], sensor["camera_world"])
                    if z[0] > 0:
                        u, v = np.rint(uv[0]).astype(int)
                        h, w = sensor["rgb"].shape[:2]
                        radius = max(1, int(0.05 * sensor["intrinsics"][0, 0] / z[0]))
                        ys = slice(max(0, v - radius), min(h, v + radius))
                        xs = slice(max(0, u - radius), min(w, u + radius))
                        sensor["rgb"] = sensor["rgb"].copy()
                        sensor["depth_m"] = sensor["depth_m"].copy()
                        sensor["valid_depth"] = sensor["valid_depth"].copy()
                        sensor["rgb"][ys, xs] = 127
                        sensor["depth_m"][ys, xs] = 0
                        sensor["valid_depth"][ys, xs] = False
                elif 6 <= phase < 7:
                    fault = "total_occlusion"
                    sensor["rgb"] = np.zeros_like(sensor["rgb"])
                    sensor["valid_depth"] = np.zeros_like(sensor["valid_depth"])
            received = provider.engine.clock()
            update_started = time.perf_counter()
            estimate = provider.update(sensor, now=received)
            operational.input(sensor, received, time.perf_counter() - update_started)
            operational.publication(provider)
            if provider.engine.failed:
                raise RuntimeError(provider.diagnostics.get("error", "perception_session_failed"))
            tracker = provider.tracking
            if tracker.candidates and reference is None:
                candidate = tracker.candidates[0]
                tracker.select_zone(
                    candidate.candidate_id, float(sensor["time_s"]), source="diagnostic"
                )
                # The map seed predates asynchronous initialization. The door was stationary then.
                reference = (
                    expected_truth.copy(),
                    candidate.initial_world.copy(),
                    candidate.surface.points[:: max(1, len(candidate.surface.points) // 32)].copy(),
                )
                seed_sensor, _ = provider.last_cue
                np.savez_compressed(
                    output / "seed.npz",
                    **seed_sensor,
                    mask=candidate.surface.observations[0].mask(),
                )
            result = tracker.last_result
            if result is not None and (not events or result["frame"] != events[-1]["frame"]):
                capture_error = None
                if reference is not None and tracker.candidates[0].state == "tracked":
                    expected = (
                        captured_truth[result["frame"]] @ np.linalg.inv(reference[0]) @ reference[1]
                    )
                    capture_error = pose_error(tracker.candidates[0].pose, expected)
                events.append(
                    dict(
                        frame=result["frame"],
                        capture_s=result["capture_s"],
                        available_s=tracker.diagnostics["available_s"],
                        latency_s=tracker.diagnostics["available_s"] - result["capture_s"],
                        inference_latency_s=result["latency_s"],
                        adapter_latency_s=result["adapter_latency_s"],
                        capture_error=capture_error,
                        gpu_free_bytes=result["gpu_free_bytes"],
                        gpu_resident_bytes=result.get("gpu_resident_bytes"),
                        gpu_sampled_peak_bytes=result.get("gpu_sampled_peak_bytes"),
                        torch_peak_bytes=result["torch_peak_bytes"],
                    )
                )
                if motion_start is None:
                    motion_start = time.perf_counter()
            observable = available = False
            error = None
            patch = estimate.local.patches[0] if estimate.local and estimate.local.patches else None
            if reference is not None:
                motion = expected_truth @ np.linalg.inv(reference[0])
                points = reference[2] @ motion[:3, :3].T + motion[:3, 3]
                observable = visible_references(points, sensor, recipe.config["plane_tolerance_m"])
                available = patch is not None and patch.world_pose is not None
                if available:
                    error = pose_error(pose_matrix(patch.world_pose), motion @ reference[1])
            records.append(
                dict(
                    frame=previous,
                    time_s=float(sensor["time_s"]),
                    phase_s=phase,
                    fault=fault,
                    observable=observable,
                    available=available,
                    error=error,
                    reason=patch.reason if patch else estimate.reason,
                    candidates=len(tracker.candidates),
                    generation=provider.generation,
                )
            )
            if len(records) % 30 == 0:
                write_json(
                    output / "status.json",
                    dict(frames=len(records), phase_s=phase, tracking=tracker.diagnostics),
                )
            if case != "stationary" and motion_start is not None and phase >= 9:
                break
        drain_start = time.perf_counter()
        while provider.engine.pending is not None and time.perf_counter() - drain_start < 30:
            provider.consume_results(provider.engine.clock())
            operational.publication(provider)
            time.sleep(0.005)
        operational_report = operational.summary(provider)
        operational_report["complete"] = provider.engine.pending is None
        operational_report["duration_s"] = duration_s
        operational_report["single_isaac"] = True
        # Only processed captures are scored; skipped acquisitions remain explicit.
        processed = []
        for event in operational.events:
            error = None
            if reference is not None and event["objects"]:
                obj = event["objects"][0]
                candidate = tracker.candidates[0]
                actual = captured_truth[event["frame"]] @ np.linalg.inv(reference[0]) @ reference[1]
                predicted = (
                    captured_cameras[event["frame"]]
                    @ np.asarray(obj["camera_from_map"])
                    @ candidate.map_from_object
                )
                error = pose_error(predicted, actual)
            processed.append(
                dict(
                    frame=event["frame"],
                    time_s=event["time_s"],
                    error=error,
                    is_seed=event["is_seed"],
                    accepted=bool(event["objects"] and event["objects"][0]["integration_accepted"]),
                )
            )
        operational_report["processed_quality"] = dict(
            scored_nonseed=sum(e["error"] is not None and not e["is_seed"] for e in processed),
            accepted_nonseed=sum(e["accepted"] and not e["is_seed"] for e in processed),
            accepted_position_error_m=quantiles(
                np.asarray([e["error"][0] for e in processed if e["accepted"] and not e["is_seed"]])
            ),
            accepted_rotation_error_deg=quantiles(
                np.asarray([e["error"][1] for e in processed if e["accepted"] and not e["is_seed"]])
            ),
            correct_10mm_5deg=sum(
                e["accepted"] and not e["is_seed"] and e["error"][0] <= 0.01 and e["error"][1] <= 5
                for e in processed
            ),
            scope=(
                f"{case} observed zone relative to its own seed; "
                "no absolute or opening quality claim"
            ),
        )
        write_json(output / "operational-report.json", operational_report)
        write_json(output / "processed-quality.json", processed)
        observed = [r for r in records if r["observable"]]
        errors = np.asarray([r["error"] for r in observed if r["available"]]).reshape(-1, 2)
        availability = sum(r["available"] for r in observed) / len(observed) if observed else 0.0
        p95 = np.quantile(errors, 0.95, axis=0) if len(errors) else None
        report = dict(
            complete=True,
            case=case,
            asset_id=door.name,
            frames=len(records),
            observable_frames=len(observed),
            useful_availability=availability,
            position_error_m=quantiles(errors[:, 0]),
            rotation_error_deg=quantiles(errors[:, 1]),
            latency_s=quantiles(np.asarray([e["latency_s"] for e in events])),
            diagnostic_passed=bool(
                availability >= 0.95 and p95 is not None and p95[0] <= 0.01 and p95[1] <= 5
            ),
            faults="controlled RGB-D faults" if case == "visibility" else "none",
            single_isaac=True,
            fresh_process=True,
            qualified=False,
            loaded_contact_admitted=False,
            final_tracking=tracker.diagnostics,
            sampled_support=support_timeline(records),
        )
        capture_errors = np.asarray(
            [e["capture_error"] for e in events if e["capture_error"] is not None]
        ).reshape(-1, 2)
        report.update(
            capture_position_error_m=quantiles(capture_errors[:, 0]),
            capture_rotation_error_deg=quantiles(capture_errors[:, 1]),
            gpu_sampled_peak_bytes=max(
                (e["gpu_sampled_peak_bytes"] or 0 for e in events), default=None
            ),
            torch_peak_bytes=max((e["torch_peak_bytes"] for e in events), default=None),
            gpu_min_free_bytes=min((e["gpu_free_bytes"] for e in events), default=None),
        )
        report["by_fault"] = {}
        for label in ("none", "zone_covered", "total_occlusion"):
            subset = [r for r in records if r["fault"] == label]
            supported = [r for r in subset if r["observable"]]
            report["by_fault"][label] = dict(
                frames=len(subset),
                observable_frames=len(supported),
                useful_frames=sum(r["available"] for r in supported),
                available_frames=sum(r["available"] for r in subset),
            )
        # A real process reset is inspected after the run, separately from motion scores.
        old_worker = provider.engine.worker
        old_pids = [w.process.pid for w in (old_worker.sam, old_worker.native)]
        old_generation = provider.generation
        provider.reset()
        report["reset"] = dict(
            old_pids=old_pids,
            new_pids=[
                w.process.pid for w in (provider.engine.worker.sam, provider.engine.worker.native)
            ],
            old_process_exited=all(
                w.process.poll() is not None for w in (old_worker.sam, old_worker.native)
            ),
            generation_incremented=provider.generation == old_generation + 1,
            empty_candidates=not tracker.candidates,
            queue=provider.engine.queue_stats,
            resources=provider.diagnostics.get("resources"),
            empty_selection=tracker.selection is None,
            empty_queue=provider.engine.pending is provider.engine.latest is None,
        )
        write_json(output / "report.json", report)
        return report
    finally:
        operational.close()
        write_json(output / "frames.json", records)
        write_json(output / "events.json", events)
        provider.close()
