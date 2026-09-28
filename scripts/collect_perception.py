#!/usr/bin/env python
"""Collect fresh train/development B1 episodes without publishing qualification records."""

import argparse
import json
import os
import subprocess
import sys
import traceback
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
    parser.add_argument(
        "--without-recorder", action="store_true", help="Physical equivalence check"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip only validated complete episodes"
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
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
        for entry in entries:
            for condition in [args.condition] if args.condition else ["nominal", "light"]:
                path = args.output / entry["asset_id"] / condition / "episode.hdf5"
                if args.resume and path.exists():
                    validate_episode(path)
                    continue
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
                if args.without_recorder:
                    command.append("--without-recorder")
                subprocess.run(command, check=True)
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
    env = DoorPushPurdueEnv(probe_config(door, setup, args.device))
    seed = 6100 if entry["split"] == "train" else 6200
    lighting = dict(intensity_scale=1.0, color=[1.0, 1.0, 1.0])
    if args.condition == "light":
        import numpy as np
        import omni.replicator.core as rep
        from pxr import UsdLux

        rng = np.random.default_rng(seed)
        lighting = dict(
            intensity_scale=float(rng.uniform(0.8, 1.2)), color=rng.uniform(0.9, 1.0, 3).tolist()
        )
        for prim in env.sim.stage.Traverse():
            if prim.IsA(UsdLux.DomeLight) or prim.IsA(UsdLux.DistantLight):
                intensity = prim.GetAttribute("inputs:intensity").Get()
                with rep.get.prim_at_path(str(prim.GetPath())):
                    rep.modify.attribute(
                        "inputs:intensity", intensity * lighting["intensity_scale"]
                    )
                    rep.modify.attribute("inputs:color", tuple(lighting["color"]))
        # Evaluate the Replicator graph before reset; no physics is advanced.
        rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
    metadata = dict(
        entry,
        condition=args.condition,
        seed=seed,
        lighting=lighting,
        setup=setup.to_dict(),
        teacher="frozen_phase5_expert",
    )
    recorder = None if args.without_recorder else ExpertRecorder(output / "episode.hdf5", metadata)
    if recorder is not None:
        recorder.calibration = camera_calibration(env)
    try:
        result = run_probe(
            env, door, setup, output / "expert", recorder=recorder, capture_evidence=False
        )
    finally:
        if recorder is not None:
            recorder.close()
    if recorder is not None:
        summary = validate_episode(output / "episode.hdf5")
        (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(asset_id=door.name, condition=args.condition, result=result)), flush=True)
    env.close()
    del app
    return 0 if result["passed"] and result["released"] else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
