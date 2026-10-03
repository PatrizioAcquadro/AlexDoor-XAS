#!/usr/bin/env python
"""Frozen image-model smoke and bounded 6.0B static scan diagnostics."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "diagnose-scan"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=REPO / "datasets/b1/perception/engineering-v2")
    parser.add_argument("--config", type=Path, default=REPO / "configs/perception_geometry.json")
    parser.add_argument("--models", type=Path, default=REPO / "models/perception")
    args = parser.parse_args()
    import h5py

    from alexdoor_xas.perception.evaluation import write_json
    from alexdoor_xas.perception.provider import ModelWorker, load_recipe
    from alexdoor_xas.perception.scan_diagnostics import PILOTS, run_scan_diagnostics
    from alexdoor_xas.qualification.corpus import load_corpus
    from alexdoor_xas.recording.b1 import episode_paths

    recipe = load_recipe(args.config, REPO)
    paths = episode_paths(args.data, load_corpus(REPO / "assets/doors/b1/corpus.json", REPO))
    paths = [p for p in paths if p.parent.parent.name in PILOTS]
    if args.command == "diagnose-scan":
        return run_scan_diagnostics(paths, recipe, args.output, args.models)
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
                    token_shape=result["token_shape"],
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
