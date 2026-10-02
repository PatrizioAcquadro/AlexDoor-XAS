"""Fresh serial parked-arm material observation checks; no teacher or loaded action."""

import json
import subprocess
import sys


def reacquisition_schedule(inspection, dt):
    """Visit an existing upper view and return, at the existing common neck speed."""
    import numpy as np

    points = np.array(inspection["waypoints"], float)
    final = points[-1, 1:]
    target = points[np.flatnonzero(points[:, 0] == 7)[0], 1:]
    travel = np.ceil(np.max(abs(target - final)) / inspection["max_neck_speed_rad_s"] / dt) * dt
    start = points[-1, 0]
    return np.r_[
        points,
        [
            [start + travel, *target],
            [start + travel + 2, *target],
            [start + 2 * travel + 2, *final],
            [start + 2 * travel + 4, *final],
        ],
    ]


def depth_visibility(sensor, candidate, tolerance):
    """Independent RGB-D audit of the saved static region; never tracker input."""
    import numpy as np

    from alexdoor_xas.perception.geometry import project

    grid = np.linspace(-0.08, 0.08, 5)
    yz = np.array([(y, z) for y in grid for z in grid])
    points = np.c_[np.zeros(len(yz)), yz] @ np.asarray(candidate["rotation"]).T
    points += np.asarray(candidate["position"])
    pixels, expected = project(points, sensor["intrinsics"], sensor["camera_world"])
    depth = sensor["depth_m"].squeeze(-1)
    valid = sensor["valid_depth"].squeeze(-1)
    visible = supported = 0
    gaps = []
    for pixel, z in zip(pixels, expected, strict=True):
        if not np.isfinite(pixel).all() or z <= 0:
            continue
        x, y = np.rint(pixel).astype(int)
        if not (1 <= x < depth.shape[1] - 1 and 1 <= y < depth.shape[0] - 1):
            continue
        visible += 1
        values = depth[y - 1 : y + 2, x - 1 : x + 2]
        usable = valid[y - 1 : y + 2, x - 1 : x + 2] & np.isfinite(values) & (values > 0)
        if usable.any():
            gap = float(abs(np.median(values[usable]) - z))
            gaps.append(gap)
            supported += gap <= tolerance
    return dict(
        samples=len(points),
        in_view=visible,
        depth_consistent=supported,
        max_depth_gap_m=max(gaps) if gaps else None,
        material_identity_verified=False,
    )


def run_live(args, repo):
    if args.asset is None:
        from alexdoor_xas.perception.evaluation import write_json

        args.output.mkdir(parents=True, exist_ok=False)
        reports = []
        for asset in ("door-2738468b94d74c5f", "animated-door-1-88abf40"):
            for condition in ("nominal", "light"):
                output = args.output / asset / condition
                output.parent.mkdir(parents=True, exist_ok=True)
                with (output.parent / f"{condition}.log").open("w") as log:
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(repo / "scripts/diagnose_material_tracking.py"),
                            "live",
                            "--asset",
                            asset,
                            "--condition",
                            condition,
                            "--output",
                            str(output),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                report = output / "report.json"
                reports.append(
                    json.loads(report.read_text())
                    if report.exists()
                    else dict(
                        asset=asset,
                        condition=condition,
                        complete=False,
                        error=f"worker_exit_{result.returncode}",
                    )
                )
                print(asset, condition, "worker_exit", result.returncode, flush=True)
        write_json(
            args.output / "summary.json",
            dict(
                reports=reports,
                complete=len(reports) == 4 and all(r.get("complete") for r in reports),
                offline_passed=False,
                dynamic_passed=False,
                mode="observation_only",
            ),
        )
        return
    if args.condition is None:
        raise ValueError("Live worker requires a lighting condition")
    args.output.mkdir(parents=True, exist_ok=False)
    from isaaclab.app import AppLauncher

    app = AppLauncher(
        headless=True, enable_cameras=True, device="cuda:0", width=1280, height=720
    ).app
    env = worker = engine = None
    report = dict(
        asset=args.asset,
        condition=args.condition,
        complete=False,
        interaction_started=False,
        offline_passed=False,
        dynamic_passed=False,
        training_started=False,
        collection_started=False,
        sealed_test_evaluated=False,
        mode="observation_only",
    )
    try:
        import h5py
        import numpy as np
        import torch
        from ihmc_alex_isaaclab._paths import REPOSITORY_ROOT as alex_root
        from PIL import Image, ImageDraw
        from pxr import UsdLux

        from alexdoor_xas.assets.purdue import derive_push_geometry
        from alexdoor_xas.envs.door_task.door_push_purdue_env import DoorPushPurdueEnv
        from alexdoor_xas.kinematics.purdue_chain import PurdueChain
        from alexdoor_xas.perception.evaluation import json_safe, write_json
        from alexdoor_xas.perception.geometry import project
        from alexdoor_xas.perception.inspection import (
            INSPECTION_LIMITS,
            inspection_tolerances,
            inspection_within_limits,
        )
        from alexdoor_xas.perception.provider import (
            CueEngine,
            GeometryProvider,
            ModelWorker,
            load_recipe,
        )
        from alexdoor_xas.policies.purdue import PurdueIO
        from alexdoor_xas.qualification.door_geometry import PreparedDoor
        from alexdoor_xas.qualification.expert import probe_config
        from alexdoor_xas.qualification.synthetic_probe import ProbeSetup
        from alexdoor_xas.recording.b1_runtime import array

        # Driver helpers contain diagnostic requests, never inference labels.
        sys.path.insert(0, str(repo / "scripts"))
        from diagnose_material_tracking import compare_candidates, configure, trace_record

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        corpus = json.loads((repo / "assets/doors/b1/corpus.json").read_text())
        entry = next(e for e in corpus["doors"] if e["asset_id"] == args.asset)
        if entry["split"] != "train":
            raise ValueError("Only the two train pilots are authorized")
        folder = repo / "assets/doors/b1" / args.asset
        door = PreparedDoor(
            folder,
            json.loads((folder / "prepared.json").read_text()),
            json.loads((folder / "recipe.json").read_text()),
        )
        setup = ProbeSetup(**json.loads((repo / corpus["setup"]["path"]).read_text()))
        binding = load_recipe(repo / "configs/perception_geometry.json", repo)
        inspection = binding.config["inspection"]
        setup.neck = tuple(inspection["waypoints"][-1][1:])
        cfg = probe_config(door, setup, "cuda:0")
        cfg.camera_mount_pitch_rad = inspection["mount_pitch_rad"]
        env = DoorPushPurdueEnv(cfg)
        if args.condition == "light":
            rng = np.random.default_rng(6100)
            scale, color = float(rng.uniform(0.8, 1.2)), rng.uniform(0.9, 1, 3).tolist()
            for prim in env.sim.stage.Traverse():
                if prim.IsA(UsdLux.DomeLight) or prim.IsA(UsdLux.DistantLight):
                    intensity = prim.GetAttribute("inputs:intensity")
                    intensity.Set(intensity.Get() * scale)
                    prim.GetAttribute("inputs:color").Set(tuple(color))
        urdf = (
            alex_root
            / "assets/robots/alex_purdue/urdf/baseline/alex_purdue_wsg32_umi_v1_full_convex.urdf"
        )
        covers = derive_push_geometry(urdf).contact_covers()
        log = (args.output / "models.log").open("w")
        worker = ModelWorker(repo / "models/perception", binding.config, log)
        engine = CueEngine(worker, replay=False)
        provider = GeometryProvider(binding, engine)
        io = PurdueIO(
            env, binding=binding, robot_asset=None, setup=setup, observed_provider=provider
        )
        io.reset()
        tracker, mapping, contact_id, selection_type = configure(
            provider, covers, args.asset, args.condition
        )
        with h5py.File(
            repo
            / f"datasets/b1/perception/engineering-v2/{args.asset}/{args.condition}/episode.hdf5"
        ) as h:
            calibration = json.loads(h["metadata"].attrs["calibration"])
            comparisons = compare_candidates(
                mapping, calibration, h["observations/joint_position"][0], PurdueChain(urdf)
            )
        write_json(args.output / "candidates.json", comparisons)
        write_json(args.output / "runtime.json", worker.runtime)
        write_json(
            args.output / "protocol.json",
            dict(
                source_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip(),
                recipe=binding.to_dict(),
                mode="observation_only",
                explicit_diagnostic_contact_tangent=[0.0, 1.1],
                contact_is_definitive=False,
                visibility_audit="raw depth at saved static candidate regions; not tracker input",
                offline_passed=False,
                dynamic_passed=False,
            ),
        )
        schedule = reacquisition_schedule(inspection, io.dt)
        write_json(
            args.output / "schedule.json",
            dict(
                waypoints=schedule,
                arm="parked",
                max_neck_speed_rad_s=inspection["max_neck_speed_rad_s"],
                hold_s=2,
                inspection_limits=INSPECTION_LIMITS,
                inspection_tolerances=inspection_tolerances(
                    io.dt, inspection["max_neck_speed_rad_s"]
                ),
            ),
        )
        maximum = dict(tool_drift_m=0.0, door_angle_rad=0.0, neck_error_rad=0.0)
        neck_speed = 0.0
        visibility = {k: dict(in_view_ticks=0, depth_supported_ticks=0) for k in mapping}
        snapshots = [0, 7, 25, float(schedule[-3, 0]), float(schedule[-1, 0])]
        selected_ticks = 0
        with (args.output / "trace.jsonl").open("w") as trace:
            for row in range(round(schedule[-1, 0] / io.dt) + 1):
                if row:
                    t = row * io.dt
                    neck = np.array([np.interp(t, schedule[:, 0], schedule[:, j]) for j in (1, 2)])
                    io.set_neck_target(neck)
                    io.hold_step()
                    drift = float(np.linalg.norm(io.tool_pose().origin - io.parked_tool.origin))
                    angle = float(abs(array(env.door.data.joint_pos)[0, 0]))
                    error = float(
                        abs(array(env.robot.data.joint_pos)[0, env.neck_ids] - neck).max()
                    )
                    neck_speed = max(
                        neck_speed,
                        float(abs(array(env.robot.data.joint_vel)[0, env.neck_ids]).max()),
                    )
                    for key, value in zip(maximum, (drift, angle, error), strict=True):
                        maximum[key] = max(maximum[key], value)
                    # Independent observation audit; door/contact truth never enters the provider.
                    if not inspection_within_limits(
                        maximum, io.dt, inspection["max_neck_speed_rad_s"]
                    ) or any(c["forbidden"] for c in env.contact_history):
                        raise RuntimeError(
                            f"Unsafe inspection: drift={drift},door={angle},neck={error}"
                        )
                sensor = {k: array(v) for k, v in io.observe().items()}
                estimate = provider.update(sensor)
                if (
                    tracker.selection is None
                    and tracker.tracks[contact_id].geometry is not None
                    and tracker.tracks[contact_id].geometry.available_s <= float(sensor["time_s"])
                ):
                    now = float(sensor["time_s"])
                    tracker.select(
                        selection_type("diagnostic-selection-1", contact_id, now, "diagnostic"), now
                    )
                record = trace_record(estimate, provider, sensor, row)
                record["visibility_audit"] = {
                    k: depth_visibility(sensor, candidate, 2 * binding.config["plane_tolerance_m"])
                    for k, candidate in mapping.items()
                }
                for k, audit in record["visibility_audit"].items():
                    visibility[k]["in_view_ticks"] += audit["in_view"] == audit["samples"]
                    visibility[k]["depth_supported_ticks"] += (
                        audit["depth_consistent"] == audit["samples"]
                    )
                selected_ticks += record["selected_geometry_supported"]
                trace.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
                if snapshots and float(sensor["time_s"]) >= snapshots[0]:
                    snapshots.pop(0)
                    image = Image.fromarray(sensor["rgb"])
                    draw = ImageDraw.Draw(image)
                    for patch_id, track in tracker.tracks.items():
                        if track.pose is not None:
                            xy, z = project(
                                track.pose.origin[None],
                                sensor["intrinsics"],
                                sensor["camera_world"],
                            )
                            if z[0] > 0:
                                x, y = xy[0]
                                draw.ellipse(
                                    (x - 12, y - 12, x + 12, y + 12), outline="yellow", width=2
                                )
                                draw.text((x + 15, y), f"{patch_id}: {track.reason}", fill="yellow")
                    image.save(args.output / f"frame-{int(sensor['frame']):06d}.png")
                if row % 300 == 0:
                    write_json(
                        args.output / "status.json",
                        dict(row=row, time_s=float(sensor["time_s"]), state="running"),
                    )
        report.update(
            complete=True,
            rows=row + 1,
            input_end_s=float(sensor["time_s"]),
            selected_supported_ticks=selected_ticks,
            physics_audit=maximum,
            measured_max_neck_speed_rad_s=neck_speed,
            visibility_audit=visibility,
            tracks={
                k: dict(
                    initialized=t.geometry is not None,
                    losses=t.losses,
                    recoveries=t.recoveries,
                    last_support_s=(t.dynamic.supported_s if t.dynamic else None),
                )
                for k, t in tracker.tracks.items()
            },
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        from alexdoor_xas.perception.evaluation import write_json

        if "maximum" in locals():
            report.update(
                physics_audit=maximum,
                measured_max_neck_speed_rad_s=neck_speed,
                visibility_audit=visibility,
                selected_supported_ticks=selected_ticks,
                input_end_s=float(sensor["time_s"]),
                rows=row + 1 if report["complete"] else row,
                tracks={
                    k: dict(
                        initialized=t.geometry is not None,
                        losses=t.losses,
                        recoveries=t.recoveries,
                        last_support_s=(t.dynamic.supported_s if t.dynamic else None),
                    )
                    for k, t in tracker.tracks.items()
                },
            )
        write_json(args.output / "report.json", report)
        if engine is not None:
            engine.close()
        if worker is not None:
            worker.close()
        if env is not None:
            env.close()
        app.close()
