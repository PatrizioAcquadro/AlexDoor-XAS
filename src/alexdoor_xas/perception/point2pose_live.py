"""Fresh-process observer diagnostics; scenario controls never enter perception IPC."""

import json
import time
from pathlib import Path

import numpy as np


def run_live_smoke(recipe, output, models, asset_id, *, case=None):
    from isaaclab.app import AppLauncher

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
    env = worker = None
    try:
        import torch

        from alexdoor_xas.envs.door_task.door_push_purdue_env import DoorPushPurdueEnv
        from alexdoor_xas.perception.evaluation import quantiles, write_json
        from alexdoor_xas.perception.geometry import surfaces
        from alexdoor_xas.perception.point2pose_runtime import Point2PoseWorker
        from alexdoor_xas.perception.provider import ModelWorker
        from alexdoor_xas.qualification.door_geometry import PreparedDoor
        from alexdoor_xas.qualification.expert import probe_config
        from alexdoor_xas.qualification.synthetic_probe import ProbeSetup
        from alexdoor_xas.recording.b1_runtime import array, camera_calibration, camera_from_joints

        root = Path(__file__).resolve().parents[3]
        folder = root / "assets/doors/b1" / asset_id
        door = PreparedDoor(
            folder,
            json.loads((folder / "prepared.json").read_text()),
            json.loads((folder / "recipe.json").read_text()),
        )
        setup = ProbeSetup(**json.loads((root / "configs/purdue_synthetic_probe.json").read_text()))
        inspection = recipe.config["inspection"]
        completed = inspection["sample_times_s"][0]
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

        if case is not None:
            env.capture.capture = capture

        def observe():
            sample = env.capture.sample
            q = array(sample.joint_position)[0]
            return dict(
                time_s=np.asarray(sample.time_s if case is None else acquisition[0]),
                frame=np.asarray(sample.frame),
                rgb=array(sample.rgb)[0],
                depth_m=array(sample.depth_m)[0],
                valid_depth=array(sample.valid_depth)[0],
                joint_position=q,
                joint_velocity=array(sample.joint_velocity)[0],
                camera_world=camera_from_joints(calibration, q),
                intrinsics=array(env.camera.data.intrinsic_matrices)[0],
            )

        if case is not None:
            return observer_case(
                env, door, setup, calibration, observe, recipe, models, output, case, start_clock
            )
        sensor = observe()
        with (output / "automatic-candidates.log").open("w") as log:
            visual = ModelWorker(models, recipe.config, log)
            try:
                cue = visual.infer(sensor["rgb"])
            finally:
                visual.close()
        candidates = surfaces(cue, sensor, recipe.config)
        if not candidates:
            raise ValueError("no_automatic_live_candidate")
        masks = [candidates[0].observations[0].mask()]
        np.savez_compressed(output / "seed.npz", **sensor, mask=masks[0])
        worker = Point2PoseWorker(
            models / "point2pose",
            calibration["depth_interval_m"],
            recipe.config["plane_tolerance_m"],
            1,
            output,
        )
        write_json(output / "runtime.json", worker.runtime)
        worker.infer(sensor, masks)
        records = []
        # The simulator actively steps/renders while CUDA inference is in flight.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            for _ in range(12):
                env.step(torch.zeros((1, 6), device=env.device))
                sensor = observe()
                started = time.perf_counter()
                pending = pool.submit(worker.infer, sensor)
                render_ticks = 0
                while not pending.done():
                    env.step(torch.zeros((1, 6), device=env.device))
                    render_ticks += 1
                result = pending.result()
                objects = result.pop("objects")
                result.pop("masks")
                result.update(
                    render_ticks=render_ticks,
                    completion_wall_s=time.perf_counter() - started,
                    lost=any(o["lost"] for o in objects),
                    correspondences=sum(o["current_points"]["shape"][0] for o in objects),
                )
                records.append(result)
                write_json(output / "frames.json", records)
        latency = np.asarray([r["completion_wall_s"] for r in records])
        report = dict(
            mode="concurrent-live-smoke",
            asset_id=asset_id,
            single_isaac=True,
            automatic_candidate=True,
            latency_s=quantiles(latency),
            timely_tracking_fraction=float(
                np.mean([r["completion_wall_s"] <= 0.15 and not r["lost"] for r in records])
            ),
            torch_peak_bytes=max(r["torch_peak_bytes"] for r in records),
            accuracy_validated=False,
            qualified=False,
            loaded_contact_admitted=False,
        )
        write_json(output / "smoke.json", report)
        return report
    except Exception as error:
        from alexdoor_xas.perception.evaluation import write_json

        write_json(
            output / "failure.json", dict(error=f"{type(error).__name__}: {error}", qualified=False)
        )
        raise
    finally:
        if worker:
            worker.close()
        if env:
            env.close()
        app.close()


def observer_case(
    env, door, setup, calibration, observe, recipe, models, output, case, start_clock
):
    """Bounded observer motion/fault cases; truth and fault labels stay in this evaluator."""
    import torch

    from alexdoor_xas.action.frames import rot_z
    from alexdoor_xas.perception.evaluation import quantiles, write_json
    from alexdoor_xas.perception.material_zone import pose_matrix
    from alexdoor_xas.perception.point2pose_replay import (
        pose_error,
        support_timeline,
        visible_references,
    )
    from alexdoor_xas.perception.point2pose_runtime import tracking_provider
    from alexdoor_xas.recording.b1_runtime import array

    if case not in ("camera", "panel", "combined", "visibility"):
        raise ValueError("unknown_observer_case")
    provider = tracking_provider(recipe, models, calibration, output, replay=False)
    start_clock()
    zero = torch.zeros((1, 6), device=env.device)
    records, events, reference, previous = [], [], None, None
    captured_truth = {}
    motion_start = None
    clock_start = time.perf_counter()
    neck = np.array(setup.neck)

    def truth():
        angle = door.sign * float(array(env.door.data.joint_pos)[0, 0])
        transform = np.eye(4)
        transform[:3, :3] = rot_z(angle)
        transform[:3, 3] = door.hinge
        return transform

    try:
        while time.perf_counter() - clock_start < 24:
            phase = 0.0 if motion_start is None else time.perf_counter() - motion_start
            if case in ("camera", "combined"):
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
            estimate = provider.update(sensor)
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
            if motion_start is not None and phase >= 9:
                break
        observed = [r for r in records if r["observable"]]
        errors = np.asarray([r["error"] for r in observed if r["available"]]).reshape(-1, 2)
        availability = sum(r["available"] for r in observed) / len(observed) if observed else 0.0
        p95 = np.quantile(errors, 0.95, axis=0) if len(errors) else None
        report = dict(
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
        old_worker = tracker.engine.worker
        old_pid = old_worker.process.pid if old_worker else None
        old_generation = provider.generation
        provider.reset()
        report["reset"] = dict(
            old_pid=old_pid,
            new_pid=tracker.engine.worker.process.pid if tracker.engine.worker else None,
            old_process_exited=old_worker is None or old_worker.process.poll() is not None,
            generation_incremented=provider.generation == old_generation + 1,
            empty_candidates=not tracker.candidates,
            empty_selection=tracker.selection is None,
            empty_queue=tracker.engine.pending is tracker.engine.latest is None,
        )
        write_json(output / "report.json", report)
        return report
    finally:
        write_json(output / "frames.json", records)
        write_json(output / "events.json", events)
        provider.close()
