"""B1 source admission, normalization recipes and preparation attempts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .contracts import (
    ACCEPTED_LICENSES,
    MAX_TEXTURE_EDGE_PX,
    MAX_TRIANGLES,
    DoorDimensions,
    sha256_file,
)

FORMATS = {".usd", ".usda", ".usdc", ".usdz", ".glb", ".gltf", ".fbx", ".obj"}
GROUPS = ("Frame", "Panel", "Handle")
LOCAL_ONLY_LICENSES = {"Sketchfab-Free-Standard"}
PRIVATE_NONCOMMERCIAL_LICENSES = {"CC-BY-NC-ND-4.0"}


class PreparationError(ValueError):
    def __init__(self, message, *, status="fail", category="recipe"):
        super().__init__(message)
        self.status, self.category = status, category


def require(condition, message, *, category="recipe", status="fail"):
    if not condition:
        raise PreparationError(message, category=category, status=status)


def write_json(path, value):
    """Replace a small record atomically, after its outputs have been completed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def candidate_id(value):
    require(bool(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value)), "Invalid candidate ID")
    return value


def remote_review(record, existing=()):
    """Record evidence, allowing unknown geometry to proceed to local inspection."""
    candidate_id(record["asset_id"])
    for name in (
        "source_url",
        "source_uid",
        "author",
        "license",
        "license_evidence",
        "attribution",
        "retrieval_date",
        "door_type",
    ):
        require(bool(record.get(name)), f"Missing {name}", status="unresolved", category="source")
    distribution_scope = record.get("distribution_scope", "redistributable")
    extra_licenses = {
        "redistributable": set(),
        "local_only": LOCAL_ONLY_LICENSES,
        "private_noncommercial": PRIVATE_NONCOMMERCIAL_LICENSES,
    }
    require(distribution_scope in extra_licenses, "Unknown distribution scope", category="source")
    permitted_licenses = ACCEPTED_LICENSES | extra_licenses[distribution_scope]
    require(
        record["license"] in permitted_licenses,
        "License is outside B1 admission",
        category="source",
    )
    require(
        record["door_type"] in {"interior", "exterior", "industrial"},
        "Expected a full-size single-leaf push door",
        category="asset",
    )
    for name in ("single_leaf", "frame_panel_separable"):
        require(record.get(name) is not False, f"Candidate fails {name}", category="asset")
    # Source latch state is prepared locally; it is not an identity rejection.
    for name in ("license_scope_review", "custom_terms_review", "duplicate_review"):
        require(record.get(name) != "fail", f"Review failed: {name}", category="source")
        require(
            record.get(name) == "pass",
            f"Review needed: {name}",
            category="source",
            status="unresolved",
        )
    for dep in record.get("dependencies", []):
        require(
            dep.get("license") in permitted_licenses and dep.get("license_evidence"),
            "Dependency license evidence missing or incompatible",
            category="source",
            status="unresolved",
        )
    require(
        record.get("selected_format", "").lower() in FORMATS,
        "Select a supported input format; export .blend first",
        category="tool",
        status="unresolved",
    )
    for other in existing:
        source_part = record.get("source_part")
        other_part = other.get("source_part")
        distinct_pack_parts = bool(source_part and other_part and source_part != other_part)
        require(
            other["asset_id"] == record["asset_id"]
            or not (
                other["source_uid"] == record["source_uid"]
                or other["source_url"] == record["source_url"]
            )
            or distinct_pack_parts,
            "Duplicate source identity",
            category="asset",
        )
    for key, limit in (
        ("reported_triangles", MAX_TRIANGLES),
        ("reported_texture_max_px", MAX_TEXTURE_EDGE_PX),
    ):
        require(
            record.get(key) is None or 0 < record[key] <= limit,
            f"Outside admission: {key}",
            category="asset",
        )
    return {
        "status": "pass",
        "scope": "remote_review_only",
        "distribution_scope": distribution_scope,
        "record": record,
    }


def validate_recipe(recipe, component_count):
    require(recipe.get("handedness") in {"left", "right"}, "Specify original handedness")
    rotation = np.asarray(recipe["rotation"], dtype=float)
    scale = float(recipe["scale"])
    translation = np.asarray(recipe["translation_m"], dtype=float)
    moving_translation = np.asarray(recipe.get("moving_translation_m", [0, 0, 0]), dtype=float)
    hinge = np.asarray(recipe["hinge_m"], dtype=float)
    require(rotation.shape == (3, 3) and np.isfinite(rotation).all(), "Invalid rotation")
    require(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        and abs(np.linalg.det(rotation) - 1) < 1e-7,
        "Rotation must be proper; no reflection",
    )
    require(np.isfinite(scale) and scale > 0, "Scale must be positive and uniform")
    require(
        translation.shape == hinge.shape == (3,) and np.isfinite([translation, hinge]).all(),
        "Invalid translation/hinge",
    )
    require(
        moving_translation.shape == (3,) and np.isfinite(moving_translation).all(),
        "Invalid moving assembly translation",
    )
    moving_scale = float(recipe.get("moving_scale", 1))
    moving_center = np.asarray(recipe.get("moving_scale_center_m", [0, 0, 0]), dtype=float)
    require(
        np.isfinite(moving_scale)
        and 0.98 <= moving_scale <= 1
        and moving_center.shape == (3,)
        and np.isfinite(moving_center).all(),
        "Moving clearance scale must be uniform, between 0.98 and 1, with a finite center",
    )
    require(
        moving_scale == 1 or bool(recipe.get("clearance_review")),
        "Document the source clearance defect and resulting gaps in clearance_review",
    )
    origin = np.asarray(recipe["opening_center_source"], dtype=float)
    require(
        origin.shape == (3,)
        and np.isfinite(origin).all()
        and np.allclose(rotation @ origin * scale + translation, 0, atol=1e-7),
        "Opening center must map to the canonical floor origin",
    )
    groups = recipe["components"]
    require(set(groups) == set(GROUPS), "Classify Frame, Panel and Handle (possibly empty)")
    indices = [i for group in GROUPS for i in groups[group]]
    require(
        all(type(i) is int for i in indices) and sorted(indices) == list(range(component_count)),
        "Classify every component exactly once",
    )
    require(bool(groups["Frame"]) and bool(groups["Panel"]), "Frame and Panel are required")
    leaf = recipe.get("leaf_components", groups["Panel"])
    require(
        isinstance(leaf, list)
        and bool(leaf)
        and len(set(leaf)) == len(leaf)
        and all(type(i) is int and i in groups["Panel"] for i in leaf),
        "Leaf measurement components must be a nonempty subset of Panel",
    )
    unlatched = recipe.get("unlatched_components", [])
    require(
        isinstance(unlatched, list)
        and all(type(i) is int and i in groups["Panel"] and i not in leaf for i in unlatched)
        and len(set(unlatched)) == len(unlatched),
        "Unlatched components must be panel latch/lock parts, never leaf, frame or handles",
    )
    require(
        not unlatched or bool(recipe.get("unlatched_review")),
        "Identify the disengaged latch/lock components and explain their exclusion",
    )
    axis_parts = recipe.get("hinge_axis_components", [])
    require(
        isinstance(axis_parts, list)
        and all(type(i) is int for i in axis_parts)
        and len(set(axis_parts)) == len(axis_parts)
        and (not axis_parts or len(axis_parts) >= 2)
        and all(i in groups["Panel"] and i not in leaf + unlatched for i in axis_parts),
        "Hinge axis components must be distinct moving hinge hardware, never leaf or latch",
    )
    require(
        not axis_parts or bool(recipe.get("hinge_axis_review")),
        "Review the hinge hardware used to locate the axis",
    )
    pairs = recipe.get("hinge_contact_exclusions", [])
    require(
        isinstance(pairs, list)
        and all(
            isinstance(pair, list)
            and len(pair) == 2
            and all(type(i) is int for i in pair)
            and pair[0] in groups["Panel"]
            and pair[0] not in leaf + unlatched
            and pair[1] in groups["Frame"]
            for pair in pairs
        )
        and len({tuple(pair) for pair in pairs}) == len(pairs),
        "Hinge contact exclusions must pair moving/fixed hinge hardware, never leaf or handles",
    )
    require(
        not pairs or bool(recipe.get("hinge_contact_review")),
        "Review the internal hinge interfaces represented by the revolute joint",
    )
    jamb = recipe.get("hinge_jamb_components", [])
    require(
        isinstance(jamb, list)
        and len(set(jamb)) == len(jamb)
        and all(type(i) is int and i in groups["Frame"] for i in jamb)
        and set(jamb) <= {fixed for _, fixed in pairs},
        "Hinge jamb components must be reviewed fixed participants in contact exclusions",
    )
    require(
        not jamb or bool(recipe.get("hinge_jamb_review")),
        "Review local hinge contact with the modeled jamb",
    )
    grouped = set()
    for batch in recipe.get("collider_groups", []):
        members = batch["components"]
        require(
            isinstance(members, list)
            and len(members) >= 2
            and all(type(i) is int for i in members)
            and len(set(members)) == len(members)
            and any(set(members) <= set(groups[name]) for name in GROUPS)
            and not set(members) & (grouped | set(unlatched)),
            "Collider groups must contain distinct colliding components of one rigid body",
        )
        require(bool(batch.get("review")), "Review the physical solid represented by each group")
        require(
            batch.get("approximation") in {"convexHull", "convexDecomposition"},
            "Specify a collider group approximation",
        )
        require(
            not set(map(str, members)) & set(recipe.get("colliders", {})),
            "Grouped components cannot also have individual collider recipes",
        )
        grouped.update(members)
    DoorDimensions.from_mapping(recipe["dimensions_m"])
    require(bool(recipe.get("modifications")), "Record normalization modifications")
    return rotation, scale, translation, hinge


def hinge_collision_pairs(recipe, collision, components):
    """Resolve reviewed bearing interfaces to hull pairs; keep all other contacts."""
    excluded = {tuple(pair) for pair in recipe.get("hinge_contact_exclusions", [])}
    jamb = set(recipe.get("hinge_jamb_components", []))
    result = set()
    if not excluded:
        return result
    hardware = {i for pair in excluded for i in pair}
    found = set()
    local_jamb_hulls = set()
    for name in GROUPS:
        for index, (points, ids) in enumerate(zip(collision[name], components[name], strict=True)):
            if hardware.intersection(ids):
                require(len(ids) == 1, "Filtered hinge hardware cannot share a collider group")
                # Only reviewed local jamb cells may join compact bearing contacts.
                radius = np.linalg.norm(
                    np.asarray(points)[:, :2] - np.asarray(recipe["hinge_m"])[:2], axis=1
                )
                if name == "Frame" and ids[0] in jamb and radius.max() > 0.1:
                    continue
                require(
                    radius.max() <= 0.1, "Filtered hardware must lie within 10 cm of hinge axis"
                )
                found.update(ids)
                if name == "Frame" and ids[0] in jamb:
                    local_jamb_hulls.add(index)
    require(found == hardware, "Missing filtered hinge collider")
    for i, moving in enumerate(components["Panel"]):
        for j, fixed in enumerate(components["Frame"]):
            if (
                len(moving) == len(fixed) == 1
                and (moving[0], fixed[0]) in excluded
                and (fixed[0] not in jamb or j in local_jamb_hulls)
            ):
                result.add((i, j))
    return result


def component_transform(recipe, name):
    """Apply global normalization, then the same clearance repair to leaf and handles."""
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(recipe["rotation"]) * recipe["scale"]
    matrix[:3, 3] = recipe["translation_m"]
    if name != "Frame":
        scale = recipe.get("moving_scale", 1)
        center = np.asarray(recipe.get("moving_scale_center_m", [0, 0, 0]))
        matrix[:3, :3] *= scale
        matrix[:3, 3] = (
            (matrix[:3, 3] - center) * scale
            + center
            + np.asarray(recipe.get("moving_translation_m", [0, 0, 0]))
        )
    return matrix


def check_hinge_edge(recipe, edge_y):
    """A fixed source hinge may stay in place while clearance fitting retreats the leaf."""
    center = recipe.get("moving_scale_center_m", [0, 0, 0])[1]
    shift = recipe.get("moving_translation_m", [0, 0, 0])[1]
    original_edge = (edge_y - center - shift) / recipe.get("moving_scale", 1) + center + shift
    require(
        min(abs(recipe["hinge_m"][1] - edge) for edge in (edge_y, original_edge)) <= 0.01,
        "Hinge does not match original handedness/panel edge",
    )


def check_hinge_axis_components(recipe, bounds_by_component, leaf_bounds):
    """Check a reviewed axis against two compact, vertically separated hinge barrels."""
    axis = np.asarray(recipe["hinge_m"])[:2]
    bounds = [np.asarray(bounds_by_component[i]) for i in recipe["hinge_axis_components"]]
    sign = 1 if recipe["handedness"] == "left" else -1
    require(
        sign * (axis[1] - np.asarray(leaf_bounds)[:, 1].mean()) > 0,
        "Hinge axis is on the wrong side of the leaf",
    )
    for box in bounds:
        center = box[:, :2].mean(axis=0)
        require(
            np.all(box[0, :2] <= axis)
            and np.all(axis <= box[1, :2])
            and np.all(box[1, :2] - box[0, :2] <= 0.1)
            and np.linalg.norm(axis - center) <= 0.01,
            "Hinge axis does not match compact hinge barrel centers",
        )
    require(
        max(box[:, 2].mean() for box in bounds) - min(box[:, 2].mean() for box in bounds) >= 0.1,
        "Hinge axis components must be vertically separated",
    )


def collider_batches(recipe, name):
    """Material-disconnected surfaces may share a collider for their physical solid."""
    remaining = set(recipe["components"][name]) - set(recipe.get("unlatched_components", []))
    for batch in recipe.get("collider_groups", []):
        if set(batch["components"]) <= remaining:
            yield batch["components"], batch["approximation"]
            remaining.difference_update(batch["components"])
    for index in sorted(remaining):
        yield [index], recipe.get("colliders", {}).get(str(index), "auto")


def new_attempt(root, asset_id):
    """Never reuse a failed or accepted output directory."""
    parent = Path(root) / candidate_id(asset_id) / "attempts"
    parent.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1_000_000):
        target = parent / f"{index:06d}"
        try:
            target.mkdir()
            return target
        except FileExistsError:
            continue
    raise PreparationError("Too many attempts")


def file_inventory(files):
    return [
        {"path": str(Path(p).resolve()), "sha256": sha256_file(p), "bytes": Path(p).stat().st_size}
        for p in sorted(set(map(Path, files)))
    ]


def verify_inventory(inventory):
    for item in inventory:
        path = Path(item["path"])
        require(
            path.is_file() and sha256_file(path) == item["sha256"],
            f"Evidence no longer matches {path}",
            status="unresolved",
            category="evidence",
        )


def promote(attempt, candidate):
    """Only technically valid, licensed candidates become ready for expert qualification."""
    attempt, candidate = Path(attempt).resolve(), Path(candidate)
    require(attempt.parent.name == "attempts", "Expected an attempt directory")
    stages = {}
    for name in ("normalize", "static", "physics"):
        result = json.loads((attempt / f"{name}.json").read_text())
        require(result["status"] == "pass", f"{name} has not passed")
        verify_inventory(result["release_files"])
        stages[name] = result
    require(
        stages["static"]["release_files"] == stages["physics"]["release_files"],
        "Static and physics evidence refer to different assets",
        category="evidence",
    )
    review = json.loads(candidate.read_text())
    from .prepared import prepared_candidates, publish_prepared

    accepted = list(prepared_candidates(attempt.parent.parent.parent))
    distribution_scope = remote_review(review, [record for record, _ in accepted])[
        "distribution_scope"
    ]
    inspected = json.loads((attempt / "inspect.json").read_text())
    require(
        review.get("reviewed_source_sha256") == inspected["source_sha256"],
        "Bind local review to the inspected source checksum",
        status="unresolved",
        category="source",
    )
    suspects = []
    for record, other in accepted:
        require(
            record["asset_id"] == review["asset_id"]
            or record["reviewed_source_sha256"] != inspected["source_sha256"],
            "Identical source payload",
            category="asset",
        )
        if record["asset_id"] != review["asset_id"] and (
            other["geometry_fingerprint"] == inspected["geometry_fingerprint"]
        ):
            suspects.append(record["asset_id"])
    require(
        not suspects or bool(review.get("duplicate_resolution")),
        f"Geometry requires duplicate review against {suspects}",
        status="unresolved",
        category="source",
    )
    require(review["asset_id"] == attempt.parent.parent.name, "Candidate identity mismatch")
    require(
        review.get("local_dependency_review") == "pass"
        and review.get("local_duplicate_review") == "pass"
        and review.get("local_visual_review") == "pass",
        "Complete local dependency, identity and visual review",
        category="source",
        status="unresolved",
    )
    publish_prepared(attempt, review)
    return distribution_scope
