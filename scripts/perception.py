#!/usr/bin/env python
"""Frozen-model, static B1 and Point2Pose smoke/replay diagnostics."""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

# Geometry stays in Isaac; native SAM3 uses the isolated worker.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def run_live_matrix(args, recipe):
    from alexdoor_xas.perception.diagnostics.scan import PILOTS
    from alexdoor_xas.perception.evaluation import write_json

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "protocol.json",
        dict(
            mode="point2pose-observer-matrix",
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            assets=[args.asset] if args.asset else list(PILOTS),
            cases=["camera", "panel", "combined", "visibility"],
            recipe=recipe.to_dict(),
            useful_availability_min=0.95,
            p95_position_limit_m=0.01,
            p95_rotation_limit_deg=5.0,
            max_dynamic_age_s=0.15,
            single_isaac=True,
            fresh_process=True,
            loaded_contact_admitted=False,
            official_qualification_evaluated=False,
        ),
    )
    reports = []
    for asset in (args.asset,) if args.asset else PILOTS:
        for case in ("camera", "panel", "combined", "visibility"):
            folder = args.output / asset / case
            with (args.output / f"{asset}-{case}.log").open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "point2pose-live",
                        "--output",
                        str(folder),
                        "--models",
                        str(args.models),
                        "--config",
                        str(args.config),
                        "--case",
                        case,
                        "--asset",
                        asset,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            reports.append(
                dict(
                    asset=asset,
                    case=case,
                    returncode=result.returncode,
                    report=str(folder / "report.json"),
                    complete=(folder / "report.json").exists(),
                    failed=(folder / "failure.json").exists(),
                )
            )
            write_json(args.output / "runs.json", reports)
    return int(any(r["returncode"] or not r["complete"] or r["failed"] for r in reports))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "smoke",
            "diagnose-scan",
            "point2pose-smoke",
            "point2pose-live-smoke",
            "point2pose-replay",
            "point2pose-offline",
            "point2pose-live",
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=REPO / "datasets/b1/perception/engineering-v2")
    parser.add_argument("--config", type=Path, default=REPO / "configs/perception_geometry.json")
    parser.add_argument("--models", type=Path, default=REPO / "models/perception")
    parser.add_argument(
        "--seed", type=Path, help="Reuse an automatic smoke seed from its source capture"
    )
    parser.add_argument("--case", choices=("camera", "panel", "combined", "visibility"))
    parser.add_argument("--asset", help="Authorized pilot asset for a live case")
    args = parser.parse_args(argv)
    import h5py

    from alexdoor_xas.perception.diagnostics.scan import PILOTS, run_scan_diagnostics
    from alexdoor_xas.perception.evaluation import write_json
    from alexdoor_xas.perception.provider import ModelWorker, load_recipe
    from alexdoor_xas.qualification.corpus import load_corpus
    from alexdoor_xas.recording.b1 import episode_paths

    recipe = load_recipe(args.config, REPO)
    if args.command == "point2pose-live":
        if args.asset and args.asset not in PILOTS:
            parser.error("Live diagnostics are restricted to the two authorized pilots")
        if args.case:
            from alexdoor_xas.perception.diagnostics.live import run_live_smoke

            run_live_smoke(
                recipe, args.output, args.models, args.asset or PILOTS[1], case=args.case
            )
        else:
            return run_live_matrix(args, recipe)
        return 0
    if args.command == "point2pose-live-smoke":
        from alexdoor_xas.perception.diagnostics.live import run_live_smoke

        print(run_live_smoke(recipe, args.output, args.models, PILOTS[1]), flush=True)
        return 0
    paths = episode_paths(args.data, load_corpus(REPO / "assets/doors/b1/corpus.json", REPO))
    paths = [p for p in paths if p.parent.parent.name in PILOTS]
    if args.command == "point2pose-offline":
        from alexdoor_xas.perception.diagnostics.offline import run_offline

        reports = run_offline(paths, recipe, args.output, args.models)
        return 0 if all(r["complete"] for r in reports) else 1
    if args.command == "point2pose-replay":
        from alexdoor_xas.perception.diagnostics.replay import run_replay

        run_replay(paths, recipe, args.output, args.models)
        return 0
    if args.command == "diagnose-scan":
        return run_scan_diagnostics(paths, recipe, args.output, args.models)
    if args.command == "point2pose-smoke":
        from alexdoor_xas.perception.diagnostics.smoke import run_point2pose_smoke

        print(
            run_point2pose_smoke(paths[0], recipe, args.output, args.models, seed=args.seed),
            flush=True,
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "protocol.json",
        dict(
            recipe=recipe.to_dict(),
            data=str(args.data),
            selected_episode=str(paths[0]),
            source_row=240,
            mode="smoke",
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            recipe_sha256=hashlib.sha256(recipe.recipe_json.encode()).hexdigest(),
            training_started=False,
            collection_started=False,
            sealed_test_evaluated=False,
            qualified=False,
        ),
    )
    with (args.output / "models.log").open("w") as log:
        worker = ModelWorker(args.models, recipe.config, log=log)
        try:
            write_json(args.output / "runtime.json", worker.runtime)
            with h5py.File(paths[0], "r") as h5:
                rgb = h5["observations/rgb"][240]
            first = worker.infer(rgb)
            result = worker.infer(rgb)
            write_json(
                args.output / "smoke.json",
                dict(
                    **worker.runtime,
                    masks=len(result["masks"]),
                    shape=result["shape"],
                    cold_latency_s=first["latency_s"],
                    warm_latency_s=result["latency_s"],
                    error=result.get("error"),
                    qualified=False,
                ),
            )
            print((args.output / "smoke.json").read_text(), flush=True)
            return 0
        finally:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
