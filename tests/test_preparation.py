"""Essential B1 recipe, admission and preservation contracts."""

import copy
import json

import numpy as np
import pytest

from alexdoor_xas.qualification.convex_geometry import Convex, mechanical_limit, overlap
from alexdoor_xas.qualification.preparation import (
    PreparationError,
    check_hinge_axis_components,
    check_hinge_edge,
    collider_batches,
    component_transform,
    file_inventory,
    hinge_collision_pairs,
    new_attempt,
    promote,
    remote_review,
    validate_recipe,
    write_json,
)


def recipe():
    return dict(
        handedness="left",
        scale=1,
        rotation=np.eye(3).tolist(),
        translation_m=[0, 0, 0],
        opening_center_source=[0, 0, 0],
        hinge_m=[0.085, 0.325, 0],
        components={"Panel": [0], "Frame": [1], "Handle": []},
        dimensions_m=dict(width_m=0.65, height_m=2.1, thickness_m=0.04),
        modifications=["rigid normalization"],
    )


def remote():
    return dict(
        asset_id="one",
        source_url="https://example.org/one",
        source_uid="one",
        author="Author",
        license="CC-BY-4.0",
        license_evidence="page screenshot",
        attribution="Author / one / CC BY 4.0",
        retrieval_date="2026-09-23",
        door_type="interior",
        license_scope_review="pass",
        custom_terms_review="pass",
        duplicate_review="pass",
        selected_format=".glb",
    )


def test_unknown_remote_geometry_is_deferred_but_license_must_be_known():
    data = remote()
    assert remote_review(data)["status"] == "pass"
    data["license"] = "CC-BY-NC-4.0"
    with pytest.raises(PreparationError, match="License"):
        remote_review(data)
    data = remote()
    data["frame_panel_separable"] = False
    with pytest.raises(PreparationError, match="separable"):
        remote_review(data)


def test_standard_license_is_local_only_including_dependencies():
    data = remote()
    data["license"] = "Sketchfab-Free-Standard"
    data["dependencies"] = [
        {"path": "texture.jpg", "license": "Sketchfab-Free-Standard", "license_evidence": "page"}
    ]
    with pytest.raises(PreparationError, match="License"):
        remote_review(data)
    data["distribution_scope"] = "local_only"
    assert remote_review(data)["distribution_scope"] == "local_only"
    data["dependencies"][0]["license"] = "unknown"
    with pytest.raises(PreparationError, match="Dependency"):
        remote_review(data)


def test_no_derivatives_license_requires_private_noncommercial_scope():
    data = remote()
    data["license"] = "CC-BY-NC-ND-4.0"
    for scope in (None, "local_only"):
        data.pop("distribution_scope", None)
        if scope:
            data["distribution_scope"] = scope
        with pytest.raises(PreparationError, match="License"):
            remote_review(data)
    data["distribution_scope"] = "private_noncommercial"
    data["dependencies"] = [
        {"path": "material", "license": "CC-BY-NC-ND-4.0", "license_evidence": "page"}
    ]
    assert remote_review(data)["distribution_scope"] == "private_noncommercial"
    data["dependencies"][0]["license"] = "Sketchfab-Free-Standard"
    with pytest.raises(PreparationError, match="Dependency"):
        remote_review(data)


@pytest.mark.parametrize(
    "field", ["license_scope_review", "custom_terms_review", "duplicate_review"]
)
def test_explicit_failed_source_review_is_rejected(field):
    data = remote()
    data[field] = "fail"
    with pytest.raises(PreparationError, match=f"Review failed: {field}") as failure:
        remote_review(data)
    assert failure.value.status == "fail"
    assert failure.value.category == "source"

    del data[field]
    with pytest.raises(PreparationError, match=f"Review needed: {field}") as failure:
        remote_review(data)
    assert failure.value.status == "unresolved"


def test_duplicate_identity_and_unsupported_format_are_explicit():
    data = remote()
    other = {**data, "asset_id": "two"}
    with pytest.raises(PreparationError, match="Duplicate"):
        remote_review(data, [other])
    data["selected_format"] = ".blend"
    with pytest.raises(PreparationError) as failure:
        remote_review(data)
    assert failure.value.status == "unresolved"


@pytest.mark.parametrize("change", ["reflection", "duplicate", "nonfinite", "scale"])
def test_recipe_rejects_invalid_transforms_and_selection(change):
    data = copy.deepcopy(recipe())
    if change == "reflection":
        data["rotation"][0][0] = -1
    elif change == "duplicate":
        data["components"]["Frame"] = [0]
    elif change == "nonfinite":
        data["hinge_m"][0] = float("nan")
    else:
        data["scale"] = 0
    with pytest.raises(PreparationError):
        validate_recipe(data, 2)


def test_moving_assembly_translation_is_explicit_and_finite():
    data = recipe()
    data["moving_translation_m"] = [0, -0.0001, 0]
    validate_recipe(data, 2)
    data["moving_translation_m"] = [0, float("nan"), 0]
    with pytest.raises(PreparationError, match="moving assembly translation"):
        validate_recipe(data, 2)


@pytest.mark.parametrize("moving_scale", [0.996, 0.98])
def test_clearance_repair_preserves_handle_attachment_and_fixed_frame(moving_scale):
    data = recipe()
    data.update(
        moving_scale=moving_scale,
        moving_scale_center_m=[0, 0, 1.05],
        clearance_review="Zero side gap; introduce 1.3 mm per side.",
    )
    validate_recipe(data, 2)
    panel = component_transform(data, "Panel")
    assert np.array_equal(panel, component_transform(data, "Handle"))
    assert np.array_equal(component_transform(data, "Frame"), np.eye(4))
    assert np.allclose(panel @ [0, 0, 1.05, 1], [0, 0, 1.05, 1])
    for scale in [0.979, 1.01, float("nan")]:
        data["moving_scale"] = scale
        with pytest.raises(PreparationError, match="clearance scale"):
            validate_recipe(data, 2)
    data["moving_scale"] = 0.996
    del data["clearance_review"]
    with pytest.raises(PreparationError, match="clearance_review"):
        validate_recipe(data, 2)


def test_clearance_fitting_can_retain_the_source_hinge_but_not_the_wrong_edge():
    data = recipe()
    data.update(
        handedness="right",
        moving_scale=0.98,
        moving_scale_center_m=[0, 0, 1],
        moving_translation_m=[0, -0.001, 0],
        hinge_m=[-0.08, -0.459, 0],
    )
    # The leaf edge retreats 9 mm; the source hinge remains 8 mm outside it.
    check_hinge_edge(data, -0.45 * 0.98 - 0.001)
    with pytest.raises(PreparationError, match="Hinge"):
        check_hinge_edge(data, 0.45 * 0.98 - 0.001)
    data["moving_scale"] = 1
    with pytest.raises(PreparationError, match="Hinge"):
        check_hinge_edge(data, -0.442)


def test_measured_wide_hinge_barrels_locate_axis_without_relaxing_edge_guard():
    data = recipe()
    data.update(
        hinge_m=[0.05, 0.52, 0],
        components={"Panel": [0, 2, 3], "Frame": [1], "Handle": []},
        leaf_components=[0],
        hinge_axis_components=[2, 3],
        hinge_axis_review="Two source hinge barrels share the measured rotation axis.",
    )
    validate_recipe(data, 4)
    leaf = np.array([[-0.05, -0.45, 0], [0.05, 0.45, 2]])
    barrels = {
        2: np.array([[0.03, 0.50, 0.3], [0.07, 0.54, 0.5]]),
        3: np.array([[0.03, 0.50, 1.5], [0.07, 0.54, 1.7]]),
    }
    with pytest.raises(PreparationError, match="Hinge"):
        check_hinge_edge(data, 0.54)
    check_hinge_axis_components(data, barrels, leaf)
    data["hinge_m"] = [0.05, 0.55, 0]
    with pytest.raises(PreparationError, match="hinge barrel centers"):
        check_hinge_axis_components(data, barrels, leaf)
    data["hinge_m"] = [0.05, 0.52, 0]
    data["handedness"] = "right"
    with pytest.raises(PreparationError, match="wrong side"):
        check_hinge_axis_components(data, barrels, leaf)
    data["handedness"] = "left"
    data["hinge_axis_components"] = [0, 3]
    with pytest.raises(PreparationError, match="Hinge axis components"):
        validate_recipe(data, 4)


def test_collider_groups_cover_surfaces_without_merging_bodies_or_latches():
    data = recipe()
    data["components"] = {"Panel": [0, 2, 3], "Frame": [1], "Handle": [4]}
    data["leaf_components"] = [0]
    data["unlatched_components"] = [3]
    data["unlatched_review"] = "Separate latch bolt."
    batch = dict(components=[0, 2], approximation="convexHull", review="Faces of one solid leaf.")
    data["collider_groups"] = [batch]
    validate_recipe(data, 5)
    assert list(collider_batches(data, "Panel")) == [([0, 2], "convexHull")]
    assert list(collider_batches(data, "Handle")) == [([4], "auto")]
    for members in [[0, 1], [0, 4], [0, 3], [0, 0]]:
        batch["components"] = members
        with pytest.raises(PreparationError, match="Collider groups"):
            validate_recipe(data, 5)
    batch["components"] = [0, 2]
    data["collider_groups"].append(dict(batch))
    with pytest.raises(PreparationError, match="Collider groups"):
        validate_recipe(data, 5)


def test_zero_gap_leaf_needs_clearance_and_an_opening_face_pivot():
    from itertools import product

    frame = [
        np.array(list(product([-0.08, 0.08], ys, [0, 2.23]))) for ys in ([-0.53, -0.5], [0.5, 0.53])
    ]
    leaf = np.array(list(product([-0.03, 0.03], [-0.5, 0.5], [0.01, 2.21])))
    groups = dict(Frame=frame, Panel=[leaf], Handle=[])
    with pytest.raises(PreparationError, match="immediately"):
        mechanical_limit(groups, np.array([0, -0.5, 0]), "right")
    center = np.array([0, 0, 1.11])
    groups["Panel"] = [(leaf - center) * 0.996 + center]
    pivot = np.array([0.03 * 0.996, -0.5 * 0.996, 0])
    limit = mechanical_limit(groups, pivot, "right")
    assert 90 < limit < 100


def test_hinge_interfaces_preserve_leaf_contacts_and_the_geometric_stop():
    from itertools import product

    data = recipe()
    data.update(
        components={"Panel": [0, 2], "Frame": [1, 3], "Handle": [4]},
        leaf_components=[0],
        hinge_contact_exclusions=[[2, 3]],
        hinge_contact_review="Solid pin inside moving barrel; revolute joint models bearing.",
        hinge_m=[0.02988, -0.498, 0],
    )
    validate_recipe(data, 5)
    for pair in ([0, 3], [4, 3], [2, 0]):
        with pytest.raises(PreparationError, match="Hinge contact"):
            validate_recipe({**data, "hinge_contact_exclusions": [pair]}, 5)
    frame = [
        np.array(list(product([-0.08, 0.08], ys, [0, 2.23]))) for ys in ([-0.53, -0.5], [0.5, 0.53])
    ]
    leaf = np.array(list(product([-0.03, 0.03], [-0.5, 0.5], [0.01, 2.21])))
    leaf = (leaf - [0, 0, 1.11]) * 0.996 + [0, 0, 1.11]
    bearing = np.array(list(product([0.08, 0.09], [-0.55, -0.54], [0.3, 0.4])))
    groups = dict(Frame=[*frame, bearing], Panel=[leaf, bearing], Handle=[])
    ids = dict(Frame=[[1], [1], [3]], Panel=[[0], [2]], Handle=[])
    pairs = hinge_collision_pairs(data, groups, ids)
    assert pairs == {(1, 2)}
    with pytest.raises(PreparationError, match="intersects"):
        mechanical_limit(groups, data["hinge_m"], "right")
    limit = mechanical_limit(groups, np.array(data["hinge_m"]), "right", pairs)
    assert 0 < limit < 100
    blocked = {**groups, "Panel": [frame[0], bearing]}
    with pytest.raises(PreparationError, match="intersects"):
        mechanical_limit(blocked, np.array(data["hinge_m"]), "right", pairs)
    blocked = {**groups, "Handle": [frame[0]]}
    with pytest.raises(PreparationError, match="intersects"):
        mechanical_limit(blocked, np.array(data["hinge_m"]), "right", pairs)
    # Reclassifying a complete jamb as bearing hardware must not hide it.
    ids["Frame"][0] = [3]
    with pytest.raises(PreparationError, match="10 cm"):
        hinge_collision_pairs(data, groups, ids)


def test_attempts_never_overwrite_sources_or_previous_outputs(tmp_path):
    a = new_attempt(tmp_path, "door")
    (a / "door.usda").write_text("accepted")
    b = new_attempt(tmp_path, "door")
    assert a != b and (a / "door.usda").read_text() == "accepted"
    with pytest.raises(PreparationError):
        new_attempt(tmp_path, "../escape")


def test_collision_test_does_not_fill_the_frame_opening():
    from itertools import product

    box = np.array(list(product([-0.02, 0.02], [-0.325, 0.325], [0.01, 2.11])))
    jamb = np.array(list(product([-0.06, 0.06], [0.33, 0.41], [0, 2.1])))
    assert not overlap(Convex(box), Convex(jamb))
    assert overlap(Convex(box), Convex(box + [0.01, 0, 0]))
    with pytest.raises(PreparationError, match="intersects"):
        mechanical_limit(
            {"Frame": [box], "Panel": [box], "Handle": []}, np.array([0, 0.325, 0]), "left"
        )


def promotion_fixture(tmp_path):
    attempt = new_attempt(tmp_path, "one")
    payload = attempt / "door.usda"
    payload.write_text("validated payload")
    digest = file_inventory([payload])
    for name in ("normalize", "static", "physics"):
        write_json(attempt / f"{name}.json", {"status": "pass", "release_files": digest})
    write_json(
        attempt / "inspect.json",
        {"source_sha256": "source-checksum", "geometry_fingerprint": "geometry"},
    )
    candidate = attempt.parent.parent / "candidate.json"
    write_json(
        candidate,
        {
            **remote(),
            "local_dependency_review": "pass",
            "local_duplicate_review": "pass",
            "local_visual_review": "pass",
            "reviewed_source_sha256": "source-checksum",
        },
    )
    return attempt, candidate, payload


def test_promotion_requires_current_evidence_and_preserves_prepared_candidate(tmp_path):
    attempt, candidate, payload = promotion_fixture(tmp_path)
    payload.write_text("changed after physics")
    with pytest.raises(PreparationError, match="Evidence no longer matches"):
        promote(attempt, candidate)
    pointer = attempt.parent.parent / "prepared.json"
    assert not pointer.exists()
    payload.write_text("validated payload")
    promote(attempt, candidate)
    accepted = pointer.read_bytes()
    assert b'"distribution_scope": "redistributable"' in accepted
    with pytest.raises(PreparationError):
        promote(attempt, candidate)
    assert pointer.read_bytes() == accepted


def test_local_only_promotion_retains_distribution_scope(tmp_path):
    attempt, candidate, _ = promotion_fixture(tmp_path)
    review = json.loads(candidate.read_text())
    review.update(license="Sketchfab-Free-Standard", distribution_scope="local_only")
    write_json(candidate, review)
    assert promote(attempt, candidate) == "local_only"
    pointer = json.loads((attempt.parent.parent / "prepared.json").read_text())
    assert pointer["distribution_scope"] == "local_only"


def test_leaf_measurement_can_exclude_attached_hardware_but_not_select_frame():
    data = recipe()
    data["components"]["Panel"].append(2)
    data["leaf_components"] = [0]
    validate_recipe(data, 3)
    for invalid in [[], [1], [0, 0]]:
        data["leaf_components"] = invalid
        with pytest.raises(PreparationError, match="Leaf measurement"):
            validate_recipe(data, 3)


@pytest.mark.parametrize("tilt", [0, 1e-7])
def test_partition_preserves_rebate_and_rejects_obstructed_aperture(tilt):
    import trimesh

    from alexdoor_xas.qualification.convex_geometry import clear_opening, partition_hulls

    # A stepped jamb: convexifying the entire mesh would fill its rebate.
    outer = trimesh.creation.box(extents=[0.04, 0.10, 2.1])
    outer.apply_translation([0.02, 0.45, 1.05])
    stop = trimesh.creation.box(extents=[0.02, 0.12, 2.1])
    stop.apply_translation([-0.01, 0.44, 1.05])
    mesh = trimesh.util.concatenate([outer, stop])
    mesh.vertices[:, 0] += tilt * mesh.vertices[:, 2]
    parts = partition_hulls(mesh.vertices, mesh.faces, {"x": [0], "z": [1.0]})
    probe = trimesh.creation.box(extents=[0.01, 0.01, 0.1])
    probe.apply_translation([0.01, 0.392, 1.05])
    assert overlap(Convex(mesh.vertices), Convex(probe.vertices))
    assert not any(overlap(Convex(p), Convex(probe.vertices)) for p in parts)
    assert np.allclose(np.concatenate(parts).min(0), mesh.bounds[0])
    assert np.allclose(np.concatenate(parts).max(0), mesh.bounds[1])
    panel = np.array([[-0.02, -0.4, 0], [0.04, 0.4, 2.1]])
    with pytest.raises(PreparationError, match="obstructs"):
        clear_opening(parts, panel)
    clear_opening(parts, panel, [[-0.379, 0.001], [0.379, 2.099]])
    with pytest.raises(PreparationError, match="aperture"):
        clear_opening(parts, panel, [[-0.01, 0.001], [0.01, 2.099]])


def test_original_surface_crossings_distinguish_interpenetration_from_touching():
    import trimesh

    from alexdoor_xas.qualification.convex_geometry import surface_crossings

    a = trimesh.creation.box()
    b = a.copy()
    b.apply_translation([0.2, 0.3, 0.4])
    assert len(surface_crossings(a.vertices, a.faces, b.vertices, b.faces)) > 0
    for shift in [1.0, 1.001]:
        b = a.copy()
        b.apply_translation([shift, 0, 0])
        assert len(surface_crossings(a.vertices, a.faces, b.vertices, b.faces)) == 0


def test_unlatched_model_excludes_only_reviewed_lock_parts():
    data = recipe()
    data["components"]["Panel"].append(2)
    data["components"]["Handle"] = [3]
    data["leaf_components"] = [0]
    data["unlatched_components"] = [2]
    data["unlatched_review"] = "Component 2 is the latch bolt; task starts disengaged."
    validate_recipe(data, 4)
    for invalid in [[0], [1], [3], [2, 2]]:
        data["unlatched_components"] = invalid
        with pytest.raises(PreparationError, match="Unlatched components"):
            validate_recipe(data, 4)
    data["unlatched_components"] = [2]
    del data["unlatched_review"]
    with pytest.raises(PreparationError, match="disengaged"):
        validate_recipe(data, 4)
    candidate = remote()
    candidate["no_latch_operation"] = False
    assert remote_review(candidate)["status"] == "pass"
