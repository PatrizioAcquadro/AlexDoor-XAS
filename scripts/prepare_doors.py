#!/usr/bin/env python
"""B1 door intake: review, inspect, normalize, static, physics and promote.

Run through Isaac Lab. Each inspection/normalization creates a new attempt.
Sources and published candidates are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from alexdoor_xas.qualification.preparation import (
    PreparationError,
    new_attempt,
    promote,
    remote_review,
    write_json,
)
from alexdoor_xas.qualification.prepared import prepared_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("review", "inspect", "normalize", "static", "physics", "preview", "promote"),
    )
    parser.add_argument("--root", type=Path, default=Path("assets/doors/b1"))
    parser.add_argument("--asset-id")
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--dependency",
        type=Path,
        action="append",
        default=[],
        help="Additional source sidecar for inspect; normalize reads source_dependencies in recipe",
    )
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--attempt", type=Path)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = args.command == "preview"
    _app, output = None, args.attempt
    protected = False
    try:
        if output:
            resolved = output.resolve()
            protected = any(
                (parent / "prepared.json").is_file() and resolved.is_relative_to(parent / name)
                for parent in resolved.parents
                for name in ("source", "prepared")
            )
            if protected:
                raise PreparationError("Published payload is preserved; create a new attempt")
        if args.command == "review":
            if not args.candidate:
                parser.error("review requires --candidate")
            record = json.loads(args.candidate.read_text())
            existing = [record for record, _ in prepared_candidates(args.root)]
            result = remote_review(record, existing)
            target = args.root / record["asset_id"]
            target.mkdir(parents=True, exist_ok=True)
            destination = target / "candidate.json"
            if destination.exists() and json.loads(destination.read_text()) != record:
                raise PreparationError("Candidate record already exists; review/edit it explicitly")
            write_json(destination, record)
        elif args.command == "promote":
            if not args.attempt or not args.candidate:
                parser.error("promote requires --attempt and --candidate")
            distribution_scope = promote(args.attempt, args.candidate)
            result = {
                "status": "pass",
                "scope": "ready_for_5.1",
                "distribution_scope": distribution_scope,
            }
        else:
            if args.command in {"inspect", "normalize"}:
                if not args.asset_id or not args.source:
                    parser.error("inspect/normalize require --asset-id and --source")
                if args.command == "normalize" and not args.recipe:
                    parser.error("normalize requires --recipe")
                output = new_attempt(args.root, args.asset_id)
                print(f"Attempt: {output.resolve()}", flush=True)
            elif not output:
                parser.error("static/physics require --attempt")
            if args.command == "physics" and not str(args.device).startswith("cuda"):
                raise PreparationError(
                    "Physics readiness requires CUDA", category="runtime", status="unresolved"
                )
            _app = AppLauncher(args).app
            from alexdoor_xas.qualification.prepare_usd import load_source, normalize

            if args.command == "inspect":
                _, result = load_source(args.source, output, args.dependency)
            elif args.command == "normalize":
                recipe = json.loads(args.recipe.read_text())
                if "source_dependencies" in recipe:
                    recipe["source_dependencies"] = [
                        str((args.recipe.parent / path).resolve())
                        for path in recipe["source_dependencies"]
                    ]
                result = normalize(args.source, recipe, output)
            elif args.command == "preview":
                from alexdoor_xas.qualification.preview_prepared import preview

                result = preview(output)
            else:
                from alexdoor_xas.qualification.verify_prepared import physics_check, static_check

                result = (
                    static_check(output)
                    if args.command == "static"
                    else physics_check(output, args.device)
                )
            write_json(output / f"{args.command}.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except Exception as exc:
        result = {
            "status": getattr(exc, "status", "unresolved"),
            "category": getattr(exc, "category", "tool"),
            "reason": str(exc),
        }
        if output and not protected and args.command != "promote":
            write_json(output / f"{args.command}.json", result)
        print(json.dumps(result), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    finally:
        # Kit shutdown can override exit status; process cleanup is done by the launcher.
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    os._exit(main())
