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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "smoke",
            "diagnose-scan",
            "point2pose-smoke",
            "point2pose-replay",
            "point2pose-offline",
            "point2pose-live",
            "point2pose-operational",
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--data", type=Path, default=REPO / "datasets/recordings/b1/perception/engineering-v2"
    )
    parser.add_argument("--config", type=Path, default=REPO / "configs/perception_geometry.json")
    parser.add_argument("--models", type=Path, default=REPO / "models/perception")
    parser.add_argument(
        "--seed", type=Path, help="Reuse an automatic smoke seed from its source capture"
    )
    parser.add_argument(
        "--case", choices=("stationary", "camera", "panel", "combined", "visibility")
    )
    parser.add_argument("--asset", help="Authorized pilot asset for a live case")
    parser.add_argument("--selection", type=Path, help="Explicit observed geometric reference JSON")
    parser.add_argument(
        "--episode",
        type=Path,
        help="Single authorized recording for offline, replay or operational measurement",
    )
    parser.add_argument("--mode", choices=("wall-clock", "replay"), default="wall-clock")
    parser.add_argument(
        "--duration-s", type=float, default=65, help="Live stationary observation window"
    )
    args = parser.parse_args(argv)
    from alexdoor_xas.perception.selection import load_selection

    selection = load_selection(args.selection)
    import h5py

    from alexdoor_xas.perception.diagnostics.scan import PILOTS, run_scan_diagnostics
    from alexdoor_xas.perception.evaluation import write_json
    from alexdoor_xas.perception.provider import load_recipe
    from alexdoor_xas.perception.sam3.runtime import Sam3Worker
    from alexdoor_xas.qualification.corpus import load_corpus
    from alexdoor_xas.recording.b1 import episode_paths

    recipe = load_recipe(args.config, REPO)
    if args.command in ("point2pose-offline", "point2pose-replay", "point2pose-operational"):
        if args.episode is None or args.episode.parent.parent.name not in PILOTS:
            parser.error("This diagnostic requires one authorized pilot --episode")
        args.episode = episode_paths(
            args.episode,
            load_corpus(REPO / "datasets/doors/b1/corpus.json", REPO),
            complete_campaign=False,
        )[0]
        if args.command == "point2pose-offline":
            from alexdoor_xas.perception.diagnostics.offline import offline_episode

            report = offline_episode(
                args.episode, recipe, args.output, args.models, selection=selection
            )
        elif args.command == "point2pose-replay":
            from alexdoor_xas.perception.diagnostics.replay import run_replay

            report = run_replay(args.episode, recipe, args.output, args.models, selection=selection)
        else:
            from alexdoor_xas.perception.diagnostics.operational import recorded_operational

            report = recorded_operational(
                args.episode,
                recipe,
                args.output,
                args.models,
                selection=selection,
                replay=args.mode == "replay",
            )
        # Process completion is independent of scientific diagnostic_passed/qualification.
        return int(
            not report.get("complete", False)
            or bool(report.get("failure"))
            or (args.output / "failure.json").exists()
        )
    if args.command == "point2pose-live":
        if args.asset not in PILOTS or args.case is None:
            parser.error("Live diagnostics require an authorized --asset and an explicit --case")
        from alexdoor_xas.perception.diagnostics.live import run_live

        report = run_live(
            recipe,
            args.output,
            args.models,
            args.asset,
            case=args.case,
            selection=selection,
            duration_s=args.duration_s,
        )
        return int(not report.get("complete", False) or (args.output / "failure.json").exists())
    paths = episode_paths(args.data, load_corpus(REPO / "datasets/doors/b1/corpus.json", REPO))
    paths = [p for p in paths if p.parent.parent.name in PILOTS]
    if args.command == "diagnose-scan":
        return run_scan_diagnostics(paths, recipe, args.output, args.models)
    if args.command == "point2pose-smoke":
        from alexdoor_xas.perception.diagnostics.smoke import run_point2pose_smoke

        print(
            run_point2pose_smoke(
                paths[0], recipe, args.output, args.models, seed=args.seed, selection=selection
            ),
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
        worker = Sam3Worker(
            args.models, recipe.config["point2pose_selected"]["sam3_frontend"], log=log
        )
        try:
            write_json(args.output / "runtime.json", worker.runtime)
            with h5py.File(paths[0], "r") as h5:
                first = worker.infer(
                    h5["observations/rgb"][240],
                    h5["observations/frame"][240],
                    h5["observations/time_s"][240],
                )
                result = worker.infer(
                    h5["observations/rgb"][243],
                    h5["observations/frame"][243],
                    h5["observations/time_s"][243],
                )
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
