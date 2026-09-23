"""Essential B1 recipe, admission and preservation contracts."""

import copy

import numpy as np
import pytest

from alexdoor_xas.qualification.convex_geometry import Convex, mechanical_limit, overlap
from alexdoor_xas.qualification.preparation import (
    PreparationError,
    file_inventory,
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
    with pytest.raises(PreparationError):
        promote(attempt, candidate)
    assert pointer.read_bytes() == accepted
