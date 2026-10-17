#!/usr/bin/env python
"""Acquire observed pre-task workspace calibration with the existing head ZED."""

import argparse
import json
import signal
import sys
from pathlib import Path
from time import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def interrupted(signum, _frame):
    raise KeyboardInterrupt(f"Survey interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--view", choices=("front", "back"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from alexdoor_xas.perception.inspection import (
        load_inspection,
        run_inspection,
        validate_inspection,
    )
    from alexdoor_xas.qualification.corpus import load_corpus
    from alexdoor_xas.qualification.door_geometry import PreparedDoor
    from alexdoor_xas.qualification.expert import probe_config
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup

    protocol = json.loads((REPO / "configs/b1_workspace_survey.json").read_text())
    corpus = load_corpus(REPO / "datasets/doors/b1/corpus.json", REPO)
    entry = next(e for e in corpus["doors"] if e["asset_id"] == args.asset_id)
    if args.asset_id not in protocol["domain"] or entry["split"] not in ("train", "development"):
        parser.error("Survey only the declared pilot train/development domain")
    folder = args.asset_root / args.asset_id
    for name in ("candidate.json", "prepared.json", "recipe.json"):
        if (folder / name).read_bytes() != (
            REPO / "datasets/doors/b1" / args.asset_id / name
        ).read_bytes():
            raise ValueError("Survey input differs from the frozen corpus")
    setup = ProbeSetup(**json.loads((REPO / protocol["task_setup"]).read_text()))
    nominal_floor = tuple(setup.floor_pose)
    view = protocol["views"][args.view]
    setup.floor_pose = (view["floor_x_m"], nominal_floor[1], view["floor_yaw_rad"])
    inspection = load_inspection(REPO / "configs/perception_inspection.json")
    for row in inspection["waypoints"]:
        row[1] *= view["neck_yaw_sign"]
    validate_inspection(inspection)
    setup.neck = tuple(inspection["waypoints"][-1][1:])
    args.output.mkdir(parents=True, exist_ok=False)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
    import torch

    from alexdoor_xas.envs.door_task.door_push_purdue_env import DoorPushPurdueEnv
    from alexdoor_xas.qualification.stop import verify_stop_continuation
    from alexdoor_xas.recording.b1_runtime import ExpertRecorder, camera_calibration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    door = PreparedDoor(
        folder,
        json.loads((folder / "prepared.json").read_text()),
        json.loads((folder / "recipe.json").read_text()),
    )
    cfg = probe_config(door, setup, "cuda:0")
    cfg.camera_mount_pitch_rad = inspection["mount_pitch_rad"]
    cfg.seed = 6100
    env = DoorPushPurdueEnv(cfg)
    signal.signal(signal.SIGTERM, interrupted)
    write(args.output / "physical-case-start.json", dict(role="workspace_survey", wall_time=time()))
    metadata = dict(
        scope=protocol["scope"],
        asset_id=args.asset_id,
        split=entry["split"],
        survey_view=args.view,
        survey_floor_pose=setup.floor_pose,
        accepted_task_floor_pose=nominal_floor,
        inspection=inspection,
        phase="inspect_only",
        policy_inputs=False,
        operational_qualification=False,
    )
    recorder = ExpertRecorder(args.output / "survey.hdf5", metadata)
    recorder.calibration = camera_calibration(env)
    stopped = False
    try:
        env.reset()
        recorder.start(env, door, setup, phase="inspect")
        result = dict(inspection=run_inspection(env, inspection, recorder))
        stopped = True
        result["post_stop"] = verify_stop_continuation(env, args.output / "post-stop.npz")
        result.update(
            completed=True,
            passed=result["post_stop"]["passed"],
            qualified=False,
            scope=protocol["scope"],
            task_manipulation=False,
            imitation_eligible=False,
        )
        recorder.finish(result)
        write(args.output / "report.json", result)
    finally:
        if not stopped:
            try:
                verify_stop_continuation(env, args.output / "interrupted-stop.npz")
            except Exception as error:
                write(args.output / "stop-failure.json", dict(error=repr(error)))
        recorder.close()
        env.close()
        del app
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
