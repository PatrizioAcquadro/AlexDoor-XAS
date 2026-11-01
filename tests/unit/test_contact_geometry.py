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


def test_front_envelope_retains_relief_with_hidden_connections_and_ambiguous_foreground():
    from alexdoor_xas.perception.contact_geometry import observed_front_extent
    from perception_helpers import measured_surface, recipe

    surface, sensor, _ = measured_surface()
    config = recipe().config
    view = dict(sensor, available_s=float(sensor["time_s"]))
    # The visible handle is separated from the planar returns by more than the
    # association distance. Missing connection pixels cannot establish detachment.
    sensor["depth_m"][110:130, 110:130, 0] = 0.94
    detached = observed_front_extent(surface, [view], config)
    assert detached["rigid_connected_voxels"] < detached["observed_voxels"]
    assert detached["observed_extent_m"] == pytest.approx(0.06)
    assert detached["observed_protrusion_voxels"] > 0
    assert np.all(
        np.array(detached["protrusion_tangent_upper_m"])
        > detached["protrusion_tangent_lower_m"]
    )
    # A closer isolated object remains a possible attachment without independent
    # exclusion evidence; retaining it must not discard the first handle region.
    sensor["depth_m"][60:70, 80:90, 0] = 0.4
    ambiguous = observed_front_extent(surface, [view], config)
    assert ambiguous["observed_extent_m"] == pytest.approx(0.6)
    assert ambiguous["front_extent_m"] > detached["front_extent_m"]
    assert np.all(
        np.array(ambiguous["protrusion_tangent_lower_m"])
        <= detached["protrusion_tangent_lower_m"]
    )
    assert np.all(
        np.array(ambiguous["protrusion_tangent_upper_m"])
        >= detached["protrusion_tangent_upper_m"]
    )
    # Connecting more measured relief to the root cannot remove that obstacle.
    sensor["depth_m"][75:105, 80:110, 0] = np.linspace(1.0, 0.94, 30)[:, None]
    connected = observed_front_extent(surface, [view], config)
    assert connected["observed_extent_m"] == pytest.approx(0.6)
    assert connected["front_extent_m"] >= ambiguous["front_extent_m"]
    assert connected["source_clocks"] == [(0.0, 0.0)]


def test_front_envelope_without_measured_protrusion_keeps_spatial_bounds_unavailable():
    from alexdoor_xas.perception.contact_geometry import observed_front_extent
    from perception_helpers import measured_surface, recipe

    surface, sensor, _ = measured_surface()
    result = observed_front_extent(
        surface, [dict(sensor, available_s=float(sensor["time_s"]))], recipe().config
    )
    assert result["observed_extent_m"] < 0.001
    assert result["protrusion_tangent_lower_m"] is None
    assert result["protrusion_tangent_upper_m"] is None
    assert result["front_extent_m"] > 0
