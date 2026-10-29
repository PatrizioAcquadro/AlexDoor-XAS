#!/usr/bin/env python
"""One explicit frozen-model perception diagnostic per invocation."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main(argv=None):
    from alexdoor_xas.perception.diagnostics.scan import PILOTS

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recorded = (
        "smoke",
        "diagnose-scan",
        "point2pose-smoke",
        "point2pose-replay",
        "point2pose-offline",
        "point2pose-operational",
    )
    for command in (*recorded, "point2pose-live"):
        sub = commands.add_parser(command)
        sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--config", type=Path, default=REPO / "configs/perception_geometry.json")
        sub.add_argument("--models", type=Path, default=REPO / "models/perception")
        if command in recorded:
            sub.add_argument(
                "--episode", required=True, type=Path, help="Single authorized pilot recording"
            )
        else:
            sub.add_argument("--asset", required=True, choices=PILOTS)
            sub.add_argument(
                "--case",
                required=True,
                choices=("stationary", "camera", "panel", "combined", "visibility"),
            )
            sub.add_argument(
                "--duration-s", type=float, default=65, help="Live stationary observation window"
            )
        if command.startswith("point2pose-"):
            sub.add_argument("--selection", type=Path, help="Observed reference JSON")
        if command == "point2pose-smoke":
            sub.add_argument("--seed", type=Path, help="Verify a cached source capture")
        if command == "point2pose-operational":
            sub.add_argument("--mode", choices=("wall-clock", "replay"), default="wall-clock")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output must be a fresh directory")

    from alexdoor_xas.perception.provider import load_recipe
    from alexdoor_xas.perception.selection import load_selection

    recipe = load_recipe(args.config, REPO)
    selection = load_selection(getattr(args, "selection", None))
    if args.command == "point2pose-live":
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
    else:
        from alexdoor_xas.qualification.corpus import load_corpus
        from alexdoor_xas.recording.b1 import episode_paths

        if not args.episode.is_file() or args.episode.parent.parent.name not in PILOTS:
            parser.error("This diagnostic requires one existing authorized pilot --episode")
        episode = episode_paths(
            args.episode,
            load_corpus(REPO / "datasets/doors/b1/corpus.json", REPO),
            complete_campaign=False,
        )[0]
        if args.command == "smoke":
            from alexdoor_xas.perception.diagnostics.smoke import run_sam3_smoke

            report = run_sam3_smoke(episode, recipe, args.output, args.models)
        elif args.command == "diagnose-scan":
            from alexdoor_xas.perception.diagnostics.scan import run_scan_diagnostic

            report = run_scan_diagnostic(episode, recipe, args.output, args.models)
        elif args.command == "point2pose-smoke":
            from alexdoor_xas.perception.diagnostics.smoke import run_point2pose_smoke

            report = run_point2pose_smoke(
                episode, recipe, args.output, args.models, seed=args.seed, selection=selection
            )
        elif args.command == "point2pose-offline":
            from alexdoor_xas.perception.diagnostics.offline import offline_episode

            report = offline_episode(episode, recipe, args.output, args.models, selection=selection)
        elif args.command == "point2pose-replay":
            from alexdoor_xas.perception.diagnostics.replay import run_replay

            report = run_replay(episode, recipe, args.output, args.models, selection=selection)
        else:
            from alexdoor_xas.perception.diagnostics.operational import recorded_operational

            report = recorded_operational(
                episode,
                recipe,
                args.output,
                args.models,
                selection=selection,
                replay=args.mode == "replay",
            )
    # Execution completion is independent of scientific quality and qualification.
    return int(
        not report.get("complete", False)
        or bool(report.get("failure"))
        or (args.output / "failure.json").exists()
    )


if __name__ == "__main__":
    raise SystemExit(main())
