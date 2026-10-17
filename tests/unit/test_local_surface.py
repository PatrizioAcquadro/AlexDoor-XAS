"""Measured plane/edge constraints retain finite conditioning and pair residuals."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alexdoor_xas.perception.local_surface import constrained_surface_pose, observed_basis


def surface(rotation, translation, noise=None):
    y, z = np.meshgrid(np.linspace(-0.5, 0.5, 17), np.linspace(-0.8, 0.8, 23))
    points = np.c_[y.ravel() * 0, y.ravel(), z.ravel()]
    points = points @ rotation.T + translation
    if noise is not None:
        points += noise
    edge = points.reshape(23, 17, 3)[:, 0]
    return SimpleNamespace(
        points=points,
        normal=rotation[:, 0],
        offset=translation @ rotation[:, 0],
        edge_points={"width_0": edge},
    )


def test_conditioned_pose_uses_observed_orientation_and_all_pair_residuals():
    initial = surface(np.eye(3), np.array([1, 0, 0]))
    rotation = Rotation.from_euler("z", 0.4).as_matrix()
    shift = np.array([0.02, -0.1, 0.03])
    current = surface(rotation, np.array([1, 0, 0]) @ rotation.T + shift)
    source = initial.points[::20]
    target = source @ rotation.T + shift
    result = constrained_surface_pose(
        initial,
        current,
        np.eye(4),
        np.eye(4),
        source,
        target,
        0.004,
        map_anchor=initial.points.mean(0),
    )
    np.testing.assert_allclose(result.camera_from_map[:3, :3], rotation, atol=1e-14)
    np.testing.assert_allclose(result.camera_from_map[:3, 3], shift, atol=1e-14)
    assert result.position_bound_m == pytest.approx(0.004)
    assert result.rotation_bound_rad > 0
    target[-1] += [0, 0.05, 0]
    changed = constrained_surface_pose(
        initial,
        current,
        np.eye(4),
        np.eye(4),
        source,
        target,
        0.004,
        map_anchor=initial.points.mean(0),
    )
    assert changed.pairs == len(source)
    assert changed.position_bound_m > 0.05
    assert changed.rotation_bound_rad == result.rotation_bound_rad


def test_basis_covers_bounded_point_and_line_perturbations_and_refuses_missing_edges():
    rng = np.random.default_rng(24)
    for angle in np.linspace(-0.6, 0.6, 9):
        rotation = Rotation.from_euler("zy", [angle, 0.05]).as_matrix()
        original = surface(rotation, np.array([1, 0, 0]))
        noise = rng.uniform(-0.001, 0.001, original.points.shape)
        measured = surface(rotation, np.array([1, 0, 0]), noise)
        basis, bound, normal_bound = observed_basis(measured, 0.004)
        assert normal_bound <= bound
        assert Rotation.from_matrix(basis @ rotation.T).magnitude() <= bound
    measured.edge_points = {}
    with pytest.raises(ValueError, match="missing_conditioned"):
        observed_basis(measured, 0.004)
    with pytest.raises(ValueError, match="missing_metric"):
        observed_basis(original, 0)
    line = surface(np.eye(3), np.array([1, 0, 0]))
    line.points[:, 1] = 0
    with pytest.raises(ValueError, match="unconditioned_observed_plane"):
        observed_basis(line, 0.004)


def test_local_constraint_does_not_rewrite_native_pose_or_hide_missing_geometry(monkeypatch):
    from alexdoor_xas.perception.contracts import FieldSupport
    from alexdoor_xas.perception.point2pose.tracking import PanelCandidate, PanelTracking

    initial = surface(np.eye(3), np.array([1, 0, 0]))
    rotation = Rotation.from_euler("z", 0.2).as_matrix()
    current = surface(rotation, np.array([1, 0, 0]) @ rotation.T)
    current.observations = [
        SimpleNamespace(semantic_id=3, frame=0, acquired_s=0.0, mask=lambda: np.ones((4, 4), bool))
    ]
    engine = SimpleNamespace(reset=lambda: None)
    tracker = PanelTracking(
        engine,
        {"local_surface_constraint": {"point_error_m": 0.004, "normal_model_bound_rad": 0.001}},
    )
    anchor_frame = np.eye(4)
    anchor_frame[:3, 3] = [1, 0, 0]
    candidate = PanelCandidate(
        "material",
        initial,
        anchor_frame,
        anchor_frame.copy(),
        np.eye(4),
        FieldSupport(0, 0, 0, 0, 0.004, 0.01),
        pose=np.eye(4),
        state="tracked",
        camera_world=np.eye(4),
        sam3_id=3,
    )
    tracker.candidates = [candidate]
    source = initial.points[::20]
    obj = dict(
        source_points=source,
        current_points=source @ rotation.T,
        measured_correspondences=np.ones(len(source), bool),
        inliers=np.ones(len(source), bool),
    )
    captured = dict(frame=0, time_s=0.0, camera_world=np.eye(4))
    result = dict(objects=[obj], local_surfaces=[current])
    monkeypatch.setattr(tracker, "root_consistent", lambda *args: True)
    tracker.constrain_local(captured, result)
    assert candidate.local_constraint is not None
    np.testing.assert_array_equal(candidate.pose, np.eye(4))
    local = tracker._constrained_candidate(candidate, candidate.local_constraint)
    np.testing.assert_allclose(local.pose[:3, :3], rotation, atol=1e-14)
    result["local_surfaces"] = []
    tracker.constrain_local(captured, result)
    assert candidate.local_constraint is None
    assert candidate.local_constraint_reason == "missing_or_ambiguous_current_surface"
    np.testing.assert_array_equal(candidate.pose, np.eye(4))


def test_measured_plane_fixes_anchor_normal_and_retains_contradictory_pair_error():
    initial = surface(np.eye(3), np.array([1, 0, 0]))
    current = surface(np.eye(3), np.array([1.1, 0, 0]))
    source = initial.points[::20]
    # Every native pair proposes an extra 2 cm normal translation.
    target = source + [0.12, 0, 0]
    anchor = initial.points.mean(0)
    result = constrained_surface_pose(
        initial, current, np.eye(4), np.eye(4), source, target, 0.001, map_anchor=anchor
    )
    point = result.camera_from_map[:3, :3] @ anchor + result.camera_from_map[:3, 3]
    assert current.normal @ point == pytest.approx(current.offset)
    assert result.position_bound_m >= 0.021 - 1e-12
    assert result.pairs == len(source)
    with pytest.raises(ValueError, match="anchor_outside"):
        constrained_surface_pose(
            initial,
            current,
            np.eye(4),
            np.eye(4),
            source,
            target,
            0.001,
            map_anchor=anchor + [0.1, 0, 0],
        )


def test_current_normal_bounds_do_not_invent_between_capture_speed():
    from alexdoor_xas.perception.contracts import FieldSupport
    from alexdoor_xas.perception.point2pose.tracking import PanelCandidate, PanelTracking

    observed = surface(np.eye(3), np.array([1, 0, 0]))
    points = observed.points[::20]
    constraint = constrained_surface_pose(
        observed,
        observed,
        np.eye(4),
        np.eye(4),
        points,
        points,
        0.001,
        map_anchor=observed.points.mean(0),
    )
    anchor = np.eye(4)
    anchor[:3, 3] = observed.points.mean(0)
    candidate = PanelCandidate(
        "m",
        observed,
        anchor,
        anchor.copy(),
        np.eye(4),
        FieldSupport(0, 0, 0, 0, 0.004, 0.01),
        pose=anchor.copy(),
        support=FieldSupport(1, 1, 1, 0),
        camera_world=np.eye(4),
        state="tracked",
        ownership="observed_rigid_material",
        local_constraint=constraint,
    )
    tracking = PanelTracking(
        SimpleNamespace(reset=lambda: None),
        {"local_surface_constraint": {"point_error_m": 0.001, "normal_model_bound_rad": 0.001}},
        calibration_bound=(0.0001, 0.001),
        relative_speed_bound=None,
    )
    tracking.candidates = [candidate]
    current = tracking.local_state(1).patches[0]
    assert current.normal_position_bound_m > 0
    assert current.normal_rotation_bound_rad > 0
    later = tracking.local_state(1.01).patches[0]
    assert later.normal_position_bound_m is None
    assert later.normal_rotation_bound_rad is None
    assert later.pose_support.position_bound_m is None
