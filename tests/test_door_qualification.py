"""Reusable door preparation and raw measurement tests."""

from __future__ import annotations

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
)


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
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float64)
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


def test_maximum_sustained_angle_is_max_of_window_minima() -> None:
    degrees = [10.0] * 10 + [50.0] * 29 + [44.0] + [48.0] * 30
    assert maximum_sustained_angle_deg(np.radians(degrees), window_ticks=30) == pytest.approx(48.0)
    assert maximum_sustained_angle_deg(np.radians(degrees), window_ticks=20) == pytest.approx(50.0)


@pytest.mark.parametrize("angles, window", [([0.0], 0), ([0.0], 2), ([np.nan], 1)])
def test_raw_sustained_measurement_rejects_invalid_trace_or_window(angles, window) -> None:
    with pytest.raises(QualificationError):
        maximum_sustained_angle_deg(angles, window_ticks=window)
