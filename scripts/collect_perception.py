#!/usr/bin/env python
"""Collect fresh train/development B1 episodes without publishing qualification records."""

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from alexdoor_xas.qualification.corpus import load_corpus  # noqa: E402
from alexdoor_xas.recording.b1 import validate_episode  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-id")
    parser.add_argument("--condition", choices=("nominal", "light"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--inspection", type=Path, help="Common camera mount and neck scan config")
    parser.add_argument(
        "--inspection-only", action="store_true", help="Camera diagnostic, no expert"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip only validated complete episodes"
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.inspection_only and (not args.inspection or args.resume):
        parser.error("Inspection-only requires --inspection and fresh recorded output")
    if not args.device.startswith("cuda"):
        parser.error("Collection requires CUDA")
    corpus = load_corpus(REPO / "assets/doors/b1/corpus.json", REPO)
    entries = [e for e in corpus["doors"] if e["split"] in ("train", "development")]
    if args.asset_id:
        entries = [e for e in entries if e["asset_id"] == args.asset_id]
        if not entries:
            parser.error("Asset is not in the frozen train/development corpus")
    if args.smoke:
        entries = [
            next(e for e in entries if e["split"] == "train" and e["handedness"] == side)
            for side in ("left", "right")
        ]
    if not args._worker:
        args.output.mkdir(parents=True, exist_ok=True)
        pending_tasks = []
        for entry in entries:
            for condition in [args.condition] if args.condition else ["nominal", "light"]:
                path = args.output / entry["asset_id"] / condition / "episode.hdf5"
                if args.resume and path.exists():
                    summary = validate_episode(path)
                    report = path.parent / "validation.json"
                    if not report.exists():
                        report.write_text(json.dumps(summary, indent=2) + "\n")
                    continue
                pending_tasks.append((entry, condition, path))
        logs = args.output / "worker-logs"
        logs.mkdir(exist_ok=True)

        def collect(task):
            entry, condition, path = task
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--_worker",
                "--output",
                str(args.output),
                "--asset-id",
                entry["asset_id"],
                "--condition",
                condition,
                "--device",
                args.device,
            ]
            if args.inspection:
                command.extend(["--inspection", str(args.inspection.resolve())])
            if args.inspection_only:
                command.append("--inspection-only")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            log = logs / f"{stamp}-{entry['asset_id']}-{condition}.log"
            print(
                json.dumps(dict(start=entry["asset_id"], condition=condition, log=str(log))),
                flush=True,
            )
            with log.open("x") as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            if args.inspection_only:
                if not (path.parent / "expert/inspection.json").is_file():
                    raise RuntimeError("Worker did not finish inspection")
            else:
                validate_episode(path)
            print(json.dumps(dict(completed=entry["asset_id"], condition=condition)), flush=True)

        for task in pending_tasks:
            collect(task)
        return 0
    if len(entries) != 1 or not args.condition:
        parser.error("Worker needs one asset and condition")
    entry = entries[0]
    output = args.output / entry["asset_id"] / args.condition
    output.mkdir(parents=True, exist_ok=False)
    from isaaclab.app import AppLauncher

    app = AppLauncher(
        headless=True, enable_cameras=True, device=args.device, width=1280, height=720
    ).app
    from alexdoor_xas.envs.door_task.door_push_purdue_env import DoorPushPurdueEnv
    from alexdoor_xas.qualification.door_geometry import PreparedDoor
    from alexdoor_xas.qualification.expert import probe_config
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup, run_probe
    from alexdoor_xas.recording.b1_runtime import ExpertRecorder, camera_calibration

    folder = REPO / "assets/doors/b1" / entry["asset_id"]
    door = PreparedDoor(
        folder,
        json.loads((folder / "prepared.json").read_text()),
        json.loads((folder / "recipe.json").read_text()),
    )
    setup = ProbeSetup(**json.loads((REPO / corpus["setup"]["path"]).read_text()))
    inspection = None
    if args.inspection:
        from alexdoor_xas.perception.inspection import load_inspection

        inspection = load_inspection(args.inspection)
        setup.neck = tuple(inspection["waypoints"][-1][1:])
    env_config = probe_config(door, setup, args.device)
    if inspection:
        env_config.camera_mount_pitch_rad = inspection["mount_pitch_rad"]
    env = DoorPushPurdueEnv(env_config)
    seed = 6100 if entry["split"] == "train" else 6200
    lighting = dict(intensity_scale=1.0, color=[1.0, 1.0, 1.0])
    if args.condition == "light":
        import numpy as np
        import omni.graph.core as og
        import omni.replicator.core as rep
        from pxr import UsdLux

        rng = np.random.default_rng(seed)
        lighting = dict(
            intensity_scale=float(rng.uniform(0.8, 1.2)), color=rng.uniform(0.9, 1.0, 3).tolist()
        )
        lights = [
            prim
            for prim in env.sim.stage.Traverse()
            if prim.IsA(UsdLux.DomeLight) or prim.IsA(UsdLux.DistantLight)
        ]
        if not lights:
            raise RuntimeError("No existing lights to vary")
        expected = []
        for prim in lights:
            intensity = prim.GetAttribute("inputs:intensity").Get() * lighting["intensity_scale"]
            item = rep.get.prim_at_path(str(prim.GetPath()))
            with item:
                rep.modify.attribute("inputs:intensity", intensity)
                rep.modify.attribute("inputs:color", tuple(lighting["color"]))
            expected.append((prim, intensity))
        # Evaluate only the randomization graph. The existing Isaac camera owns capture;
        # Replicator orchestrator.step would introduce a second renderer/timeline driver.
        og.Controller.evaluate_sync(graph_id=item.node.get_graph())
        for prim, intensity in expected:
            if not np.isclose(prim.GetAttribute("inputs:intensity").Get(), intensity):
                raise RuntimeError("Replicator did not apply the requested light intensity")
            if not np.allclose(prim.GetAttribute("inputs:color").Get(), lighting["color"]):
                raise RuntimeError("Replicator did not apply the requested light color")
        print(json.dumps(dict(lighting=lighting, applied_lights=len(lights))), flush=True)

    metadata = dict(
        entry,
        condition=args.condition,
        seed=seed,
        lighting=lighting,
        setup=setup.to_dict(),
        teacher="frozen_arm_expert_with_inspection" if inspection else "frozen_phase5_expert",
        inspection=inspection,
    )
    recorder = ExpertRecorder(output / "episode.hdf5", metadata)
    recorder.calibration = camera_calibration(env)
    try:
        if args.inspection_only:
            from alexdoor_xas.perception.inspection import run_inspection

            env.reset()
            recorder.start(env, door, setup, phase="inspect")
            audit = run_inspection(env, inspection, recorder)
            result = dict(
                passed=True,
                released=False,
                hold_angle_deg=None,
                scope="inspection_only_not_training_episode",
            )
            recorder.finish(result)
            (output / "expert").mkdir()
            (output / "expert/inspection.json").write_text(json.dumps(audit, indent=2) + "\n")
        else:
            result = run_probe(
                env,
                door,
                setup,
                output / "expert",
                recorder=recorder,
                capture_evidence=False,
                inspection=inspection,
            )
    finally:
        recorder.close()
    if not args.inspection_only:
        summary = validate_episode(output / "episode.hdf5")
        (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(asset_id=door.name, condition=args.condition, result=result)), flush=True)
    env.close()
    del app
    return 0 if result["passed"] and (args.inspection_only or result["released"]) else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
