#!/usr/bin/env python
"""Check the frozen starting setup, then qualify one prepared door with two trials."""

import argparse
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from alexdoor_xas.qualification.door_geometry import PreparedAssetError  # noqa: E402
from alexdoor_xas.qualification.expert import (  # noqa: E402
    probe_config,
    publish_expert,
    qualify_pair,
    trial_summary,
    valid_reference_inputs,
    write_result,
)
from alexdoor_xas.qualification.synthetic_probe import ProbeSetup  # noqa: E402


def capture_initial(env, output):
    from PIL import Image

    env.reset()
    env.sim.stage.Export(str(output / "initial-scene.usda"))
    Image.fromarray(env.capture.sample.rgb[0].cpu().numpy()[..., :3]).save(
        output / "initial-rgb.png"
    )


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--rerun", action="store_true", help="Recheck an already qualified door")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if Path(args.asset_id).name != args.asset_id or args.asset_id in (".", ".."):
        parser.error("asset-id must identify one published door")
    if not str(args.device).startswith("cuda"):
        parser.error("Qualification requires CUDA")
    folder = REPO / "assets/doors/b1" / args.asset_id
    original = json.loads((folder / "prepared.json").read_text())
    if not valid_reference_inputs(original):
        parser.error("Door does not have completed preparation; do not qualify it")
    previous = original.get("expert_qualification", "not_run")
    if isinstance(previous, dict) and previous.get("status") == "qualified" and not args.rerun:
        parser.error("Door already qualified; use --rerun only for a justified recheck")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = Path.home() / ".cache/alexdoor-xas/verification/expert" / args.asset_id / stamp
    output.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {output}", flush=True)
    for name in ("prepared.json", "recipe.json", "candidate.json"):
        (output / name).write_bytes((folder / name).read_bytes())
    config = REPO / "configs/purdue_synthetic_probe.json"
    (output / "setup.json").write_bytes(config.read_bytes())
    setup = ProbeSetup(**json.loads(config.read_text()))
    report = dict(
        asset_id=args.asset_id,
        status="unresolved",
        theta_expert_d=None,
        reason="incomplete_execution",
        trials=[],
        device=args.device,
        scope="nominal_simulated_expert_reference_not_learning_data",
        previous_evidence=previous.get("evidence") if isinstance(previous, dict) else None,
    )
    write_result(output / "report.json", report)
    app, env = None, None
    trials = []
    try:
        args.enable_cameras = True
        app = AppLauncher(args).app
        from alexdoor_xas.envs.door_task.door_push_purdue_env import DoorPushPurdueEnv
        from alexdoor_xas.qualification.door_geometry import PreparedDoor
        from alexdoor_xas.qualification.synthetic_probe import run_probe

        door = PreparedDoor(folder, original, json.loads((folder / "recipe.json").read_text()))
        env = DoorPushPurdueEnv(probe_config(door, setup, args.device))
        from alexdoor_xas.envs.door_task.door_push_purdue_env import PEDESTAL

        intersections = door.pedestal_intersections(env.sim.stage, PEDESTAL)
        if intersections:
            capture_initial(env, output)
            report.update(
                status="out_of_domain",
                reason="closed_door_pedestal_intersection",
                geometry_evidence=intersections,
                qualification_scope="initial_setup_infeasible_no_expert_angle",
            )
            return 2
        # Missing prescribed surface is diagnosed, never silently relocated.
        contact, _ = door.contact_pose(0.0, setup.contact_fraction, setup.contact_height)
        faces = [face + contact for face in env.push_geometry.distal_faces]
        import numpy as np

        if not door.footprint_inside(np.concatenate(faces), 0.0):
            raise ValueError("Prescribed fingertip footprint has missing leaf surface")
        obstructions = door.footprint_obstructions(faces, setup.position_tolerance)
        if obstructions:
            capture_initial(env, output)
            report.update(
                status="out_of_domain",
                reason="prescribed_footprint_obstructed",
                geometry_evidence=obstructions,
                qualification_scope="nominal_contact_pose_infeasible_no_expert_angle",
            )
            return 2
        for number in (1, 2):
            trial = run_probe(env, door, setup, output / f"repeat-{number}")
            trials.append(trial)
            report["trials"].append(trial_summary(trial, number))
            write_result(output / "report.json", report)
            print(json.dumps(report["trials"][-1]), flush=True)
        report.update(qualify_pair(trials))
    except Exception as exc:
        traceback.print_exc()
        report.update(
            status="invalid_asset" if isinstance(exc, PreparedAssetError) else "unresolved",
            theta_expert_d=None,
            reason="invalid_prepared_asset"
            if isinstance(exc, PreparedAssetError)
            else "execution_error",
            error=f"{type(exc).__name__}: {exc}",
        )
        (output / "error.txt").write_text(traceback.format_exc())
    finally:
        write_result(output / "report.json", report)
        publish_expert(folder, original, report, output)
        print(
            json.dumps({k: v for k, v in report.items() if k != "geometry_evidence"}, indent=2),
            flush=True,
        )
        if env is not None:
            env.close()
        # Match maintained Kit CLIs: flush and exit without unreliable teardown.
        del app
    return {"qualified": 0, "out_of_domain": 2, "invalid_asset": 3}.get(report["status"], 1)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
