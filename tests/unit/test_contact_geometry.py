"""A physical distal line must lead the complete clipped body under every normal."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from alexdoor_xas.perception.contact_geometry import distal_body_order


def wedge():
    points = np.array(
        [[x, y, z] for x, z in [(0.0, 0.02), (-0.1, -0.02), (-0.1, 0.02)] for y in [-0.02, 0.02]]
    )
    return points[ConvexHull(points).simplices].reshape(-1, 3), points[:2]


def test_actual_line_support_covers_clipped_wedge_interiors_and_normal_cone():
    triangles, front = wedge()
    cone = 0.2
    bound = distal_body_order(triangles, front, [1, 0, 0], cone, 0.003)
    expected = 0.003 * np.cos(cone) - 0.0012 * np.sin(cone)
    assert bound == pytest.approx(expected, abs=1e-12)
    # Independent dense normals and the complete clipped triangle cross section.
    normals = np.array(
        [
            [np.cos(a), np.sin(a) * np.cos(b), np.sin(a) * np.sin(b)]
            for a in np.linspace(0, cone, 20)
            for b in np.linspace(0, 2 * np.pi, 61)
        ]
    )
    clipped = np.array(
        [
            [x, y, z]
            for x, z in [(-0.003, 0.0188), (-0.003, 0.02), (-0.1, -0.02), (-0.1, 0.02)]
            for y in [-0.02, 0.02]
        ]
    )
    margins = (front @ normals.T).max(0) - (clipped @ normals.T).max(0)
    assert bound <= margins.min() + 1e-12
    assert distal_body_order(triangles, front, [-1, 0, 0], cone, 0.003) < 0


def test_planar_body_order_requires_real_finite_geometry_and_a_bounded_normal():
    triangles, front = wedge()
    for normal, cone, band in [
        ([2, 0, 0], 0.1, 0.003),
        ([1, 0, 0], np.pi, 0.003),
        ([1, 0, 0], 0.1, 0),
    ]:
        with pytest.raises(ValueError, match="invalid_finite"):
            distal_body_order(triangles, front, normal, cone, band)
    with pytest.raises(ValueError, match="missing_non_distal"):
        distal_body_order(triangles, front, [1, 0, 0], 0.1, 1.0)
