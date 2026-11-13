#!/usr/bin/env python
"""Qualify a prepared door with two trials, or diagnose it with one unpublished trial."""

import argparse
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from alexdoor_xas.qualification.b1_runtime import (  # noqa: E402
    configure_native_light,
    launch_b1,
    load_probe,
    native_selected,
    native_state,
)
from alexdoor_xas.qualification.door_geometry import PreparedAssetError  # noqa: E402
from alexdoor_xas.qualification.expert import (  # noqa: E402
    continue_pair,
    publish_expert,
    qualify_pair,
    trial_summary,
    valid_reference_inputs,
    write_result,
)


def capture_initial(env, output):
    from PIL import Image

    env.reset()
    env.sim.stage.Export(str(output / "initial-scene.usda"))
    Image.fromarray(env.capture.sample.rgb[0].cpu().numpy()[..., :3]).save(
        output / "initial-rgb.png"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--rerun", action="store_true", help="Recheck an already qualified door")
    parser.add_argument(
        "--diagnostic", action="store_true", help="Run one cycle without changing prepared.json"
    )
    parser.add_argument("--_continue-pair", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--condition", choices=("nominal", "light"), default="nominal")
    parser.add_argument(
        "--bounded-diagnostic",
        type=Path,
        metavar="SETUP",
        help="Run two short unpublished cycles and evaluate the pair",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path.home() / ".cache/alexdoor-xas/verification/expert"
    )
    if native_selected():
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument(
            "--headless", action="store_true", help="Run without a viewer (default)"
        )
    else:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if Path(args.asset_id).name != args.asset_id or args.asset_id in (".", ".."):
        parser.error("asset-id must identify one published door")
    if not str(args.device).startswith("cuda"):
        parser.error("Qualification requires CUDA")
    bounded = args.bounded_diagnostic is not None
    if bounded and not args.diagnostic:
        parser.error("A bounded diagnostic requires --diagnostic")
    if args.condition == "light" and not (args.diagnostic and native_selected()):
        parser.error("Varied-light evaluation requires a native diagnostic")

    folder = REPO / "datasets/doors/b1" / args.asset_id
    original = json.loads((folder / "prepared.json").read_text())
    if not valid_reference_inputs(original):
        parser.error("Door does not have completed preparation; do not qualify it")
    previous = original.get("expert_qualification", "not_run")
    if (
        isinstance(previous, dict)
        and previous.get("status") == "qualified"
        and not (args.rerun or args.diagnostic)
    ):
        parser.error("Door already qualified; use --rerun only for a justified recheck")
    config = REPO / "configs/purdue_synthetic_probe.json"
    if args._continue_pair is not None:
        if args.diagnostic:
            parser.error("A diagnostic cannot continue a formal pair")
        output = args._continue_pair
        report, trials = continue_pair(output, folder, config, args.device)
        print(f"Continue pair: {output}", flush=True)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        output = args.output_root / args.asset_id / stamp
        output.mkdir(parents=True, exist_ok=False)
        print(f"Evidence: {output}", flush=True)
        for name in ("prepared.json", "recipe.json", "candidate.json"):
            (output / name).write_bytes((folder / name).read_bytes())
        (output / "setup.json").write_bytes(config.read_bytes())
        if bounded:
            (output / "diagnostic-setup.json").write_bytes(args.bounded_diagnostic.read_bytes())
        report = dict(
            asset_id=args.asset_id,
            status="unresolved",
            theta_expert_d=None,
            reason="incomplete_execution",
            trials=[],
            device=args.device,
            scope="single_diagnostic_not_learning_data"
            if args.diagnostic
            else "nominal_simulated_expert_reference_not_learning_data",
            previous_evidence=previous.get("evidence") if isinstance(previous, dict) else None,
            diagnostic_only=args.diagnostic,
            trial_isolation="shared_session" if bounded else "fresh_process",
            bounded_diagnostic=bounded,
            condition=args.condition,
            physics_profile=os.environ.get("ISAAC_PHYSICS_PROFILE"),
        )
        write_result(output / "report.json", report)
        trials = []
    try:
        from alexdoor_xas.qualification.synthetic_probe import run_probe

        door, setup, runtime = load_probe(
            folder,
            original,
            config,
            args.bounded_diagnostic,
            seed=6100 if args.condition == "light" else None,
        )
        with launch_b1(door, setup, args, **runtime) as env:
            if native_selected():
                report["lighting"] = configure_native_light(env, args.condition, runtime["seed"])
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
            contact, rotation = door.contact_pose(0.0, setup.contact_fraction, setup.contact_height)
            faces = [face @ rotation.T + contact for face in env.push_geometry.distal_faces]
            import numpy as np

            if not door.footprint_inside(np.concatenate(faces), 0.0):
                raise ValueError("Prescribed fingertip footprint has missing leaf surface")
            obstructions = door.footprint_obstructions(
                faces, setup.position_tolerance, rotation[:, 0]
            )
            if obstructions:
                capture_initial(env, output)
                report.update(
                    status="out_of_domain",
                    reason="prescribed_footprint_obstructed",
                    geometry_evidence=obstructions,
                    qualification_scope="nominal_contact_pose_infeasible_no_expert_angle",
                )
                return 2
            graphs = None
            repeats = 1 if args.diagnostic and not bounded else 2
            for number in range(len(trials) + 1, repeats + 1):
                trial = run_probe(env, door, setup, output / f"repeat-{number}")
                if native_selected():
                    state = native_state(env, stepped=True)
                    if bounded:
                        current = dict(env.sim.backend_manager.backend.graphs)
                        if graphs is not None and (
                            current.keys() != graphs.keys()
                            or any(current[key] is not graph for key, graph in graphs.items())
                        ):
                            raise RuntimeError("Steel CUDA Graph changed after reset")
                        graphs = current
                    report.setdefault("runtime", []).append(state)
                trials.append(trial)
                report["trials"].append(trial_summary(trial, number))
                write_result(output / "report.json", report)
                print(json.dumps(report["trials"][-1]), flush=True)
                if number == 1 and not args.diagnostic:
                    # Isolate contact history and renderer state between formal trials.
                    env.close()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os.execv(
                        sys.executable,
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            *sys.argv[1:],
                            "--_continue-pair",
                            str(output),
                        ],
                    )
            report.update(qualify_pair(trials))
            if bounded:
                report["graph_reused_across_reset"] = graphs is not None
            elif args.diagnostic:
                report["reason"] = "single_diagnostic_no_expert_reference"
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
        if not args.diagnostic:
            publish_expert(folder, original, report, output)
        print(
            json.dumps({k: v for k, v in report.items() if k != "geometry_evidence"}, indent=2),
            flush=True,
        )
    if bounded:
        return (
            0
            if (
                report["status"] == "out_of_domain"
                and len(trials) == 2
                and all(t["visibility"]["passed"] for t in trials)
            )
            else 1
        )
    if args.diagnostic and len(trials) == 1:
        trial = trials[0]
        if trial["passed"] and trial["released"] and trial["hold_angle_deg"] is not None:
            return 0 if trial["angle_deg"] >= 45 else 2
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
