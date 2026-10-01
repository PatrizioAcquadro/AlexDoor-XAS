#!/usr/bin/env python
"""Frozen-model geometric perception smoke and train/development causal evaluation."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Numerical geometry stays in the Isaac runtime; native SAM3 stays in its worker.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "evaluate"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=REPO / "datasets/b1/perception/engineering-v2")
    parser.add_argument("--config", type=Path, default=REPO / "configs/perception_geometry.json")
    parser.add_argument("--models", type=Path, default=REPO / "models/perception")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    from alexdoor_xas.perception.contracts import LEGACY_FULL_STATE, geometry_profile
    from alexdoor_xas.perception.evaluation import campaign_summary, evaluate_episode, write_json
    from alexdoor_xas.perception.provider import (
        CueEngine,
        GeometryProvider,
        ModelWorker,
        load_recipe,
    )
    from alexdoor_xas.recording.b1 import episode_paths

    recipe = load_recipe(args.config, REPO)
    if args.command == "evaluate" and geometry_profile(recipe) != LEGACY_FULL_STATE:
        raise ValueError("Operational scoring requires the independent 6.0E evaluator")
    corpus = json.loads((REPO / "assets/doors/b1/corpus.json").read_text())
    paths = episode_paths(args.data, corpus)
    if args.pilot or args.command == "smoke":
        ids = {"door-2738468b94d74c5f", "animated-door-1-88abf40"}
        paths = [p for p in paths if p.parent.parent.name in ids]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "protocol.json",
        dict(
            recipe=recipe.to_dict(),
            data=str(args.data),
            selected_episodes=[str(p) for p in paths],
            mode=args.command,
            pilot=args.pilot,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            recipe_sha256=hashlib.sha256(recipe.recipe_json.encode()).hexdigest(),
            provider_sources={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (REPO / "src/alexdoor_xas/perception").glob("*.py")
            },
            error_limits=dict(position_m=0.01, orientation_deg=5, coverage=0.95, precision=0.95),
            source_rows="HDF5 row; observed frame counter stored separately",
            truth_boundary="annotations and phase read by evaluator only",
            training_started=False,
            collection_started=False,
            sealed_test_evaluated=False,
        ),
    )
    with (args.output / "models.log").open("w") as log:
        worker = ModelWorker(args.models, recipe.config, log=log)
        engine = CueEngine(worker, replay=True)
        try:
            write_json(args.output / "runtime.json", worker.runtime)
            if args.command == "smoke":
                import h5py

                with h5py.File(paths[0], "r") as h:
                    row = 240
                    rgb = h["observations/rgb"][row]
                    first = worker.infer(rgb)
                    result = worker.infer(rgb)
                write_json(
                    args.output / "smoke.json",
                    dict(
                        **worker.runtime,
                        masks=len(result["masks"]),
                        shape=result["shape"],
                        token_shape=result["token_shape"],
                        cold_latency_s=first["latency_s"],
                        warm_latency_s=result["latency_s"],
                        error=result.get("error"),
                    ),
                )
                print((args.output / "smoke.json").read_text(), flush=True)
                return 0
            # Warm GPU kernels on an authorized train observation; discard its outputs.
            import h5py

            with h5py.File(paths[0], "r") as h:
                worker.infer(h["observations/rgb"][240])
            provider = GeometryProvider(recipe, engine)
            reports = []
            for path in paths:
                try:
                    report = evaluate_episode(
                        path, provider, args.output / path.parent.parent.name / path.parent.name
                    )
                except Exception as error:
                    write_json(
                        args.output / "failure.json",
                        dict(
                            path=str(path),
                            completed_episodes=len(reports),
                            error=f"{type(error).__name__}: {error}",
                            complete=False,
                            offline_passed=False,
                            dynamic_status="not_run_incomplete_campaign",
                        ),
                    )
                    raise
                reports.append(report)
                summary = campaign_summary(reports, args.output, len(paths))
                print(
                    json.dumps(
                        dict(
                            completed=len(reports),
                            episodes=len(paths),
                            passed_doors=summary["passed_doors"],
                        )
                    ),
                    flush=True,
                )
            return 0 if summary["offline_passed"] else 2
        finally:
            engine.close()
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
