"""Pure Phase 4.1 qualification and manifest contract tests."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from alexdoor_xas.door_qualification import (
    SCHEMA,
    DoorDimensions,
    QualificationError,
    align_angle_curves_deg,
    bootstrap_n_qual,
    connected_face_components,
    connected_mesh_face_components,
    cuboid_inertia_kg_m2,
    geometry_fingerprint,
    maximum_sustained_angle_deg,
    repeatability_metrics,
    validate_diagnostic_count,
    validate_manifest,
    validate_remote_candidate,
)


def _remote(slot: int = 1, handedness: str = "left") -> dict:
    return {
        "slot": slot,
        "source_url": f"https://example.com/models/door-{slot}",
        "source_uid": f"source-{slot}",
        "author": "Example Author",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": f"Door {slot} by Example Author, CC BY 4.0",
        "dependencies": [],
        "selected_format": "glb",
        "archive_size_bytes": 1024,
        "reported_triangles": 2000,
        "reported_texture_max_px": 2048,
        "reported_dimensions_m": None,
        "door_type": "interior",
        "handedness": handedness,
        "frame_panel_separable": True,
        "visual_duplicate_check": "pass",
        "ownership_dispute_check": "pass",
        "custom_terms_check": "pass",
        "retrieval_date": "2026-08-13",
    }


def _asset(slot: int) -> dict:
    handedness = "left" if slot <= 12 else "right"
    return {
        **_remote(slot, handedness),
        "asset_id": f"door_{slot:02d}",
        "source_path": f"assets/doors/phase4_1/source/door_{slot:02d}/source.glb",
        "source_size_bytes": 1024,
        "source_sha256": f"{slot:064x}",
        "geometry_fingerprint": f"{slot + 100:064x}",
        "recipe_path": f"assets/doors/phase4_1/recipes/door_{slot:02d}.json",
        "modifications": ["uniform_scale"],
        "normalized": {
            "path": f"assets/doors/phase4_1/normalized/door_{slot:02d}/door.usda",
            "dimensions_m": {"width_m": 0.8, "height_m": 2.0, "thickness_m": 0.04},
            "triangles": 2000,
            "texture_max_px": 2048,
            "checksums_sha256": {"door.usda": f"{slot + 200:064x}"},
        },
        "qualification": {
            "static": {"passed": True},
            "physics": {"passed": True},
            "nominal": {"passed": True},
            "repeatability": {"passed": True},
        },
        "evidence_sha256": f"{slot + 300:064x}",
        "final_status": "provisional_for_phase4_2",
    }


def _manifest() -> dict:
    return {
        "schema": SCHEMA,
        "n_qual_recommendation": {"recommended_n_qual": 10},
        "assets": [_asset(slot) for slot in range(1, 25)],
    }


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"width_m": 0.64, "height_m": 2.0, "thickness_m": 0.04}, "width"),
        ({"width_m": 0.8, "height_m": 2.41, "thickness_m": 0.04}, "height"),
        ({"width_m": 0.8, "height_m": 2.0, "thickness_m": 0.101}, "thickness"),
    ],
)
def test_dimension_contract_rejects_out_of_range(values, message) -> None:
    with pytest.raises(QualificationError, match=message):
        DoorDimensions.from_mapping(values)


def test_cuboid_inertia_uses_panel_axis_convention() -> None:
    dimensions = DoorDimensions(0.8, 2.0, 0.04)
    ix, iy, iz = cuboid_inertia_kg_m2(dimensions)
    assert ix == pytest.approx(25.0 * (0.8**2 + 2.0**2) / 12.0)
    assert iy == pytest.approx(25.0 * (0.04**2 + 2.0**2) / 12.0)
    assert iz == pytest.approx(25.0 * (0.04**2 + 0.8**2) / 12.0)


def test_geometry_fingerprint_ignores_transform_order_material_and_mirror() -> None:
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    first = geometry_fingerprint(vertices, faces)
    transform = np.diag([-4.0, 4.0, 4.0])
    transformed = vertices @ transform + np.array([8.0, -3.0, 2.0])
    second = geometry_fingerprint(
        transformed[[2, 0, 3, 1]],
        np.array([[1, 3, 0], [1, 3, 2], [1, 0, 2], [3, 0, 2]]),
    )
    assert second == first


def test_connected_face_components_are_linear_and_bounded() -> None:
    faces = np.array([[0, 1, 2], [2, 1, 3], [4, 5, 6]], dtype=np.int64)
    components = connected_face_components(faces, 7)
    assert [component.tolist() for component in components] == [[0, 1], [2]]

    with pytest.raises(QualificationError, match="limit is 1"):
        connected_face_components(faces, 7, max_components=1)


def test_connected_mesh_components_weld_duplicate_seam_vertices() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
    components = connected_mesh_face_components(vertices, faces)
    assert [component.tolist() for component in components] == [[0, 1], [2]]


@pytest.mark.parametrize("license_name", ["Free Standard", "CC-BY-NC-4.0", "CC-BY-SA-4.0"])
def test_remote_gate_rejects_non_redistributable_license(license_name) -> None:
    record = _remote()
    record["license"] = license_name
    with pytest.raises(QualificationError, match="license"):
        validate_remote_candidate(record)


def test_remote_gate_rejects_duplicate_uid_and_custom_terms() -> None:
    record = _remote(2)
    record["custom_terms_check"] = "NoAI"
    with pytest.raises(QualificationError, match="custom_terms"):
        validate_remote_candidate(record, [_remote(1)])
    record["custom_terms_check"] = "pass"
    record["source_uid"] = _remote(1)["source_uid"]
    with pytest.raises(QualificationError, match="duplicates"):
        validate_remote_candidate(record, [_remote(1)])


def test_maximum_sustained_angle_is_max_of_window_minima() -> None:
    degrees = [10.0] * 10 + [50.0] * 29 + [44.0] + [48.0] * 30
    assert maximum_sustained_angle_deg(np.radians(degrees)) == pytest.approx(48.0)


def test_curve_alignment_holds_last_valid_angle() -> None:
    first, second = align_angle_curves_deg(np.radians([0.0, 1.0]), np.radians([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(first, [0.0, 1.0, 1.0])
    np.testing.assert_allclose(second, [0.0, 1.0, 2.0])


def _rollout(offset_deg: float = 0.0, termination: str = "controller_done") -> dict:
    curve = np.radians(np.linspace(0.0, 50.0 + offset_deg, 80))
    return {
        "passed": True,
        "termination": termination,
        "angle_curve_rad": curve.tolist(),
    }


def test_repeatability_requires_outcome_termination_sustained_and_curve_tolerances() -> None:
    passing = repeatability_metrics(_rollout(), _rollout(1.0))
    assert passing["passed"] is True
    failing = repeatability_metrics(_rollout(), _rollout(4.0))
    assert failing["passed"] is False
    assert failing["max_time_aligned_curve_error_deg"] == pytest.approx(4.0)


def test_bootstrap_is_deterministic_and_selects_minimum_candidate_for_stable_pairs() -> None:
    values = {f"door_{index:02d}": [51.0, 51.0] for index in range(1, 25)}
    first = bootstrap_n_qual(values)
    second = bootstrap_n_qual(values)
    assert first == second
    assert first["recommended_n_qual"] == 10
    assert first["p95_absolute_error_deg_by_n"] == {"10": 0.0}


def test_complete_manifest_requires_24_unique_assets_and_12_per_handedness() -> None:
    manifest = _manifest()
    validate_manifest(manifest)

    duplicate = copy.deepcopy(manifest)
    duplicate["assets"][1]["geometry_fingerprint"] = duplicate["assets"][0][
        "geometry_fingerprint"
    ]
    with pytest.raises(QualificationError, match="geometry_fingerprint"):
        validate_manifest(duplicate)

    wrong_balance = copy.deepcopy(manifest)
    wrong_balance["assets"][-1]["handedness"] = "left"
    with pytest.raises(QualificationError, match="12/12"):
        validate_manifest(wrong_balance)


def test_manifest_forbids_subphase_4_2_fields() -> None:
    manifest = _manifest()
    manifest["assets"][0]["split"] = "train"
    with pytest.raises(QualificationError, match="Subphase 4.2"):
        validate_manifest(manifest)


def test_repeatability_diagnostics_contract_is_exactly_zero_or_three() -> None:
    validate_diagnostic_count(True, [])
    validate_diagnostic_count(False, [{}, {}, {}])
    with pytest.raises(QualificationError, match="requires 3 diagnostics"):
        validate_diagnostic_count(False, [{}, {}])
    assert math.isclose(math.degrees(math.pi / 4.0), 45.0)
