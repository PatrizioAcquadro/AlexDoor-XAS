"""Publish a compact, relocatable door from completed preparation results."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .preparation import require, write_json


def prepared_candidates(root):
    """Read the canonical identity and readiness records without loading payloads."""
    for path in sorted(Path(root).glob("*/prepared.json")):
        yield (
            json.loads((path.parent / "candidate.json").read_text()),
            json.loads(path.read_text()),
        )


def _copy_sources(inspected, destination):
    """Localize USD asset paths while preserving geometry and original downloads."""
    source = Path(inspected["source"])
    targets = {}
    used = set()
    for index, item in enumerate(inspected["files"]):
        original = Path(item["path"])
        relative = (
            original.relative_to(source.parent)
            if original.is_relative_to(source.parent)
            else Path("dependencies") / original.name
        )
        if relative in used:
            relative = Path("dependencies") / str(index) / original.name
        used.add(relative)
        targets[str(original)] = destination / relative
        target = targets[str(original)]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["snapshot"], target)
    for original, target in targets.items():
        if target.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            continue
        from pxr import Sdf, UsdUtils

        layer = Sdf.Layer.FindOrOpen(str(target))

        def localize(asset, original=original, target=target):
            outer, separator, inner = asset.partition("[")
            resolved = str((Path(original).parent / outer).resolve())
            if resolved not in targets:
                return asset
            relative = Path(os.path.relpath(targets[resolved], target.parent)).as_posix()
            return relative + separator + inner

        UsdUtils.ModifyAssetPaths(layer, localize)
        layer.Save()
    return targets[str(source)].relative_to(destination)


def publish_prepared(attempt, candidate):
    """Package existing results; admission checks belong to ``promote``."""
    attempt = Path(attempt).resolve()
    root = attempt.parent.parent
    require(not (root / "prepared.json").exists(), "Prepared candidate already exists; preserve it")
    require(
        not any((root / name).exists() for name in ("source", "prepared", ".publish")),
        "Publication destination already exists",
    )
    inspected = json.loads((attempt / "inspect.json").read_text())
    normalized = json.loads((attempt / "normalize.json").read_text())
    checked = json.loads((attempt / "static.json").read_text())
    recipe = json.loads((attempt / "recipe.json").read_text())
    recipe.pop("preparation_status", None)
    recipe.pop("review_notes", None)
    record = {
        key: candidate[key]
        for key in (
            "asset_id",
            "source_url",
            "source_uid",
            "source_part",
            "author",
            "license",
            "license_evidence",
            "attribution",
            "retrieval_date",
            "door_type",
            "usage_notes",
        )
        if key in candidate
    }
    record["distribution_scope"] = candidate.get("distribution_scope", "redistributable")
    record["reviewed_source_sha256"] = inspected["source_sha256"]
    licenses = sorted({item["license"] for item in candidate.get("dependencies", [])})
    if licenses:
        record["dependency_licenses"] = licenses
    ready = {
        "status": "ready_for_5.1",
        "usd": "prepared/door.usda",
        **{
            key: normalized[key]
            for key in (
                "dimensions_m",
                "handedness",
                "initial_state",
                "hinge_m",
                "panel_center_m",
                "opening_to_hinge",
                "opening_to_panel_center_closed",
                "mechanical_limit_deg",
                "geometry_fingerprint",
            )
        },
        "checks": {name: "pass" for name in ("normalization", "static", "visual", "physics")},
        "validation_scope": "Existing isolated-door preparation; no robot/expert qualification.",
        "expert_qualification": "not_run",
    }
    staging = root / ".publish"
    staging.mkdir()
    try:
        source_path = _copy_sources(inspected, staging / "source")
        record["source"] = (Path("source") / source_path).as_posix()
        payloads = [Path(item["path"]) for item in checked["release_files"]]
        payloads += [attempt / f"preview-{side}.png" for side in ("front", "rear")]
        for path in payloads:
            target = staging / "prepared" / path.relative_to(attempt)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        for name in ("source", "prepared"):
            (staging / name).rename(root / name)
        write_json(root / "recipe.json", recipe)
        write_json(root / "candidate.json", record)
        write_json(root / "prepared.json", ready)
    finally:
        shutil.rmtree(staging)
    return ready
