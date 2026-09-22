"""Reusable door preparation and raw measurement tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from alexdoor_xas.door_qualification import (
    DoorDimensions,
    QualificationError,
    connected_face_components,
    connected_mesh_face_components,
    cuboid_inertia_kg_m2,
    geometry_fingerprint,
    maximum_sustained_angle_deg,
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
    assert maximum_sustained_angle_deg(np.radians(degrees), window_ticks=30) == pytest.approx(48.0)
    assert maximum_sustained_angle_deg(np.radians(degrees), window_ticks=20) == pytest.approx(50.0)


@pytest.mark.parametrize("angles, window", [([0.0], 0), ([0.0], 2), ([np.nan], 1)])
def test_raw_sustained_measurement_rejects_invalid_trace_or_window(angles, window) -> None:
    with pytest.raises(QualificationError):
        maximum_sustained_angle_deg(angles, window_ticks=window)


def test_legacy_ingest_preserves_source_and_existing_payload(tmp_path, monkeypatch) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_phase4_1_assets.py"
    spec = importlib.util.spec_from_file_location("door_preparation", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module.paths, "PHASE4_1_SOURCE_DIR", tmp_path / "source")
    monkeypatch.setattr(module, "WORKLIST", tmp_path / "worklist.json")
    worklist = {"slots": [{"slot": 1, "state": "remote_pass", "candidate": _remote()}]}
    module.dump_json(module.WORKLIST, worklist)
    source = tmp_path / "download.glb"
    source.write_bytes(b"original source")

    module._ingest(1, source)

    target = tmp_path / "source" / "door_01" / "source.glb"
    assert target.read_bytes() == source.read_bytes() == b"original source"
    module.dump_json(module.WORKLIST, worklist)
    source.write_bytes(b"replacement source")
    with pytest.raises(QualificationError, match="already has a local source payload"):
        module._ingest(1, source)
    assert target.read_bytes() == b"original source"
    assert source.read_bytes() == b"replacement source"
