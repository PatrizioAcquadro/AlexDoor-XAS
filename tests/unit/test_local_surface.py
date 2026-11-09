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
    monkeypatch.setattr(tracker, "reobserved_material_points", lambda *args: current.points)
    tracker.constrain_local(captured, result)
    assert candidate.local_constraint is not None
    np.testing.assert_array_equal(candidate.pose, np.eye(4))
    candidate.local_surface = surface(np.eye(3), np.array([2, 0, 0]))
    monkeypatch.setattr(tracker, "root_consistent", lambda c, _: c.surface is initial)
    tracker.constrain_local(captured, result)
    assert candidate.local_constraint is not None
    candidate.local_surface = None
    local = tracker._constrained_candidate(candidate, candidate.local_constraint)
    np.testing.assert_allclose(local.pose[:3, :3], rotation, atol=1e-14)
    wrong = surface(rotation, np.array([1, 0, 0]) @ rotation.T + 0.03 * rotation[:, 0])
    wrong.observations = [
        SimpleNamespace(semantic_id=3, frame=0, acquired_s=0.0, mask=lambda: np.ones((6, 6), bool))
    ]
    expected = local.pose[:3, 3].copy()
    monkeypatch.setattr(
        tracker, "root_consistent", lambda c, _: np.linalg.norm(c.pose[:3, 3] - expected) < 0.001
    )
    result["local_surfaces"] = [wrong, current]
    tracker.constrain_local(captured, result)
    assert candidate.local_constraint is not None
    recovered = tracker._constrained_candidate(candidate, candidate.local_constraint)
    np.testing.assert_allclose(recovered.pose[:3, 3], expected, atol=1e-14)
    # A larger but tilted component has a weaker finite-contact certificate.
    # Both pass the same material-identity check; rank their measured bounds.
    coarse_rotation = Rotation.from_euler("z", 0.2025).as_matrix()
    coarse = surface(coarse_rotation, np.array([1, 0, 0]) @ coarse_rotation.T)
    coarse.observations = [
        SimpleNamespace(semantic_id=3, frame=0, acquired_s=0.0, mask=lambda: np.ones((8, 8), bool))
    ]
    monkeypatch.setattr(tracker, "root_consistent", lambda *args: True)
    for choices in ([coarse, current], [current, coarse]):
        result["local_surfaces"] = choices
        tracker.constrain_local(captured, result)
        assert candidate.local_constraint is not None
        recovered = tracker._constrained_candidate(candidate, candidate.local_constraint)
        np.testing.assert_allclose(recovered.pose[:3, :3], rotation, atol=1e-14)
    # Equal compatible support is still ambiguous; no favorable first choice.
    result["local_surfaces"] = [current, current]
    tracker.constrain_local(captured, result)
    assert candidate.local_constraint is None
    assert candidate.local_constraint_reason == "missing_or_ambiguous_current_surface"
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
    assert current.pose_rotation_axis_bounds is not None
    np.testing.assert_allclose(
        current.pose_rotation_axis_bounds.basis_world, constraint.rotation_basis_world
    )
    assert current.pose_rotation_axis_bounds.rotation_bound_rad == constraint.rotation_bound_rad
    assert current.normal_position_bound_m > 0
    assert current.normal_rotation_bound_rad > 0
    later = tracking.local_state(1.01).patches[0]
    assert later.normal_position_bound_m is None
    assert later.normal_rotation_bound_rad is None
    assert later.pose_support.position_bound_m is None


def test_planar_core_conditions_orientation_without_rewriting_relief_geometry():
    original = surface(np.eye(3), np.array([1, 0, 0]))
    original.points[::9, 0] += 0.003
    retained = original.points.copy()
    basis, bound, normal_bound = observed_basis(original, 0.001)
    assert 0 < normal_bound <= bound < 0.03
    np.testing.assert_allclose(basis, np.eye(3), atol=1e-14)
    np.testing.assert_array_equal(original.points, retained)
    original.points[:, 0] += 0.01
    with pytest.raises(ValueError, match="missing_measured_planar_core"):
        observed_basis(original, 0.001)


def test_normal_certificate_covers_known_plane_under_bounded_points_and_clipping():
    rng = np.random.default_rng(71)
    error = 0.001
    for width, angle in [(0.12, -0.5), (0.4, 0.0), (1.0, 0.6)]:
        true_rotation = Rotation.from_euler("zy", [angle, 0.03]).as_matrix()
        y, z = np.meshgrid(np.linspace(-width / 2, width / 2, 23), np.linspace(-0.4, 0.4, 31))
        ideal = np.c_[np.zeros(y.size), y.ravel(), z.ravel()]
        # A clipped irregular region; retain bounded metric noise at every point.
        ideal = ideal[(ideal[:, 1] + 0.2 * ideal[:, 2]) < width * 0.4]
        world = ideal @ true_rotation.T + [1, 0, 0]
        noise = rng.normal(size=world.shape)
        noise *= (error / np.linalg.norm(noise, axis=1))[:, None]
        measured = world + noise
        for tilt in [-0.005, 0.0, 0.005]:
            normal = Rotation.from_rotvec([0, tilt, 0]).apply(true_rotation[:, 0])
            observed = SimpleNamespace(
                points=measured,
                normal=normal,
                offset=float(normal @ np.array([1, 0, 0])),
                edge_points={"width_0": world[np.argsort(ideal[:, 1])[:31]]},
            )
            expected = np.arccos(np.clip(normal @ true_rotation[:, 0], -1, 1))
            for order in [np.arange(len(measured)), rng.permutation(len(measured))]:
                observed.points = measured[order]
                _, _, bound = observed_basis(observed, error)
                assert expected <= bound
    # Insufficient transverse support remains unavailable at the same error.
    observed.points = np.c_[np.ones(21), np.zeros(21), np.linspace(-0.4, 0.4, 21)]
    observed.normal = np.array([1.0, 0.0, 0.0])
    observed.offset = 1.0
    with pytest.raises(ValueError, match="unconditioned_observed_plane"):
        observed_basis(observed, error)


@pytest.mark.parametrize("point_error", [0.001, 0.004])
@pytest.mark.parametrize("tool_y", [-2.0, 2.0])
def test_surface_selection_uses_complete_observed_root_conditioning(
    monkeypatch, point_error, tool_y
):
    from copy import deepcopy

    from alexdoor_xas.action.frames import ObjectFrame
    from alexdoor_xas.perception.contracts import FieldSupport
    from alexdoor_xas.perception.point2pose.tracking import PanelCandidate, PanelTracking

    initial = surface(np.eye(3), np.array([1, 0, 0]))
    rotation = Rotation.from_euler("z", 0.2).as_matrix()
    good = surface(rotation, np.array([1, 0, 0]) @ rotation.T)
    good.observations = [
        SimpleNamespace(semantic_id=3, frame=0, acquired_s=0.0, mask=lambda: np.ones((4, 4), bool))
    ]
    weak = deepcopy(good)
    weak.edge_points = {"width_0": good.edge_points["width_0"][:5].copy()}
    anchor = np.eye(4)
    anchor[:3, 3] = [1, 0, 0]
    candidate = PanelCandidate(
        "material",
        initial,
        anchor,
        anchor.copy(),
        np.eye(4),
        FieldSupport(0, 0, 0, 0, 0.004, 0.01),
        pose=np.eye(4),
        state="tracked",
        camera_world=np.eye(4),
        sam3_id=3,
    )
    tracker = PanelTracking(
        SimpleNamespace(reset=lambda: None),
        {
            "local_surface_constraint": {
                "point_error_m": point_error,
                "normal_model_bound_rad": 0.001,
            }
        },
        calibration={},
        tool_fk=lambda *_: ObjectFrame(np.array([1.0, tool_y, 0.0]), np.eye(3)),
        distal_faces=(np.zeros((4, 3)), np.zeros((4, 3))),
    )
    tracker.candidates = [candidate]
    source = initial.points[::20]
    obj = dict(
        source_points=source,
        current_points=source @ rotation.T,
        measured_correspondences=np.ones(len(source), bool),
        inliers=np.ones(len(source), bool),
    )
    captured = dict(frame=0, time_s=0.0, camera_world=np.eye(4), joint_position=np.zeros(9))
    monkeypatch.setattr(tracker, "root_consistent", lambda *args: True)
    monkeypatch.setattr(tracker, "reobserved_material_points", lambda *args: good.points)
    expected = constrained_surface_pose(
        initial,
        good,
        np.eye(4),
        np.eye(4),
        source,
        obj["current_points"],
        point_error,
        map_anchor=np.array([1, 0, 0]),
        current_normal_points=good.points,
    )
    worse = constrained_surface_pose(
        initial,
        weak,
        np.eye(4),
        np.eye(4),
        source,
        obj["current_points"],
        point_error,
        map_anchor=np.array([1, 0, 0]),
        current_normal_points=good.points,
    )
    assert expected.normal_rotation_bound_rad == worse.normal_rotation_bound_rad
    assert expected.normal_point_bound_m == worse.normal_point_bound_m
    assert expected.rotation_bound_rad < worse.rotation_bound_rad
    for choices in ([weak, good], [good, weak]):
        tracker.constrain_local(captured, dict(objects=[obj], local_surfaces=choices))
        selected = candidate.local_constraint
        assert selected is not None and selected.pairs == len(source)
        assert selected.rotation_bound_rad == expected.rotation_bound_rad
        assert selected.position_bound_m == expected.position_bound_m
        assert selected.normal_rotation_bound_rad == expected.normal_rotation_bound_rad
        np.testing.assert_array_equal(candidate.pose, np.eye(4))


@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 2, 3]])
@pytest.mark.parametrize("angle", [0.0, 0.02, 0.4, np.pi / 2, np.pi])
def test_two_orthogonal_axis_certificate_covers_full_rotation(axis, angle):
    from alexdoor_xas.perception.local_surface import _orthogonal_rotation_bound

    axis = np.asarray(axis, float)
    rotation = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
    first = np.arccos(np.clip(rotation[0, 0], -1, 1))
    second = np.arccos(np.clip(rotation[2, 2], -1, 1))
    bound = _orthogonal_rotation_bound(first, second)
    if angle == np.pi:
        # The live basis rejects bounds >= pi/2; inversion is singular at pi.
        assert bound >= np.pi / 2
        return
    assert angle <= bound + 32 * np.finfo(float).eps
    # With the rotation axis in the x/z plane, the two constraints are tight.
    if axis[1] == 0:
        assert bound == pytest.approx(angle, abs=32 * np.finfo(float).eps)


@pytest.mark.parametrize("angle", [-0.5, 0.0, 0.6])
@pytest.mark.parametrize("point_error", [0.001, 0.004, 0.012])
def test_basis_certificate_covers_joint_normal_and_edge_measurement_errors(angle, point_error):
    truth = Rotation.from_euler("zyx", [angle, -0.08, 0.13]).as_matrix()
    measured = surface(truth, np.array([1.0, -0.2, 0.3]))
    rng = np.random.default_rng(37)
    noise = rng.normal(size=measured.points.shape)
    noise *= point_error / np.linalg.norm(noise, axis=1).max()
    measured.points += noise
    measured.edge_points = {"width_0": measured.points.reshape(23, 17, 3)[:, 0]}
    center = measured.points.mean(0)
    _, _, vt = np.linalg.svd(measured.points - center, full_matrices=False)
    normal = vt[-1]
    if normal @ truth[:, 0] < 0:
        normal = -normal
    measured.normal, measured.offset = normal, float(center @ normal)
    basis, bound, normal_bound = observed_basis(measured, point_error)
    assert 0 < normal_bound <= bound < np.pi / 2
    assert Rotation.from_matrix(basis @ truth.T).magnitude() <= bound
    # A native float32-scale normal changes no physical plane or metric error.
    scaled = SimpleNamespace(**vars(measured))
    scaled.normal = measured.normal * (1 + np.finfo(np.float32).eps)
    scaled.offset = measured.offset * (1 + np.finfo(np.float32).eps)
    other, other_bound, other_normal = observed_basis(scaled, point_error)
    np.testing.assert_allclose(other, basis, atol=1e-12, rtol=0)
    assert 0 < other_normal <= other_bound < np.pi / 2
    assert Rotation.from_matrix(other @ truth.T).magnitude() <= other_bound


@pytest.mark.parametrize("width", [0.12, 0.4, 1.0])
@pytest.mark.parametrize("error", [0.001, 0.004])
def test_normal_spans_cover_adversarial_metric_errors_without_statistical_credit(width, error):
    truth = Rotation.from_euler("zyx", [0.3, -0.1, 0.07]).as_matrix()
    y, z = np.meshgrid(np.linspace(-width / 2, width / 2, 23), np.linspace(-0.4, 0.4, 31))
    ideal = np.c_[np.zeros(y.size), y.ravel(), z.ravel()]
    # Every error may be correlated and at the metric limit. More points do not
    # supply a sqrt(N) reduction, and the measured fitted normal need not be true.
    noise = error * np.sign(ideal[:, 1] + 0.4 * ideal[:, 2])
    measured = (ideal + noise[:, None] * [1, 0, 0]) @ truth.T + [1, 0, 0]
    center = measured.mean(0)
    _, _, vt = np.linalg.svd(measured - center, full_matrices=False)
    normal = vt[-1]
    normal *= 1 if normal @ truth[:, 0] > 0 else -1
    observed = SimpleNamespace(
        points=measured,
        normal=normal,
        offset=float(center @ normal),
        edge_points={"width_0": measured.reshape(31, 23, 3)[:, 0]},
    )
    _, _, bound = observed_basis(observed, error)
    expected = np.arccos(np.clip(normal @ truth[:, 0], -1, 1))
    assert expected <= bound
    original = observed.points.copy()
    # A conservative span certificate also covers repeated, fully correlated
    # points; neither density nor order constitutes independent evidence.
    observed.points = np.repeat(original[::-1], 3, axis=0)
    _, _, repeated = observed_basis(observed, error)
    assert expected <= repeated
    assert repeated == pytest.approx(bound)


def test_normal_spans_tighten_conditioned_plane_without_erasing_point_error():
    observed = surface(np.eye(3), np.array([1, 0, 0]))
    _, _, bound = observed_basis(observed, 0.001)
    assert 0.001 < bound < 0.003
    _, _, larger = observed_basis(observed, 0.004)
    assert larger > bound


def test_measured_axis_bounds_cover_noisy_plane_edge_inverse_transport():
    from alexdoor_xas.perception.contracts import AxisRotationBounds

    rng = np.random.default_rng(48)
    for angle in np.linspace(-0.5, 0.5, 9):
        true_initial = Rotation.from_euler("zy", [0.2, -0.1]).as_matrix()
        true_current = Rotation.from_euler("zy", [angle, 0.1]).as_matrix()
        measured = []
        for rotation in (true_initial, true_current):
            observed = surface(rotation, np.array([1, 0, 0]))
            observed.points += rng.uniform(-0.0005, 0.0005, observed.points.shape)
            # The fitted normal is also imperfect; all point errors remain bounded.
            observed.normal = Rotation.from_rotvec([0, 0.0002, -0.0003]).apply(rotation[:, 0])
            observed.offset = observed.points.mean(0) @ observed.normal
            measured.append(observed)
        initial, current = measured
        source, target = initial.points[::20], current.points[::20]
        pose = constrained_surface_pose(
            initial,
            current,
            np.eye(4),
            np.eye(4),
            source,
            target,
            0.004,
            map_anchor=initial.points.mean(0),
        )
        bi, ai, _ = observed_basis(initial, 0.004)
        bc, ac, _ = observed_basis(current, 0.004)
        assert pose.rotation_bound_rad == ai + ac
        np.testing.assert_allclose(pose.camera_from_map[:3, :3], bc @ bi.T)
        axes = AxisRotationBounds(
            tuple(map(tuple, pose.rotation_basis_world)),
            tuple(pose.rotation_axis_chord_bounds),
            pose.rotation_bound_rad,
        )
        vectors = rng.normal(size=(50, 3))
        error = true_initial @ true_current.T - bi @ bc.T
        for v in vectors:
            assert np.linalg.norm(error @ v) <= axes.displacement_bound([v], 0, ai + ac)


@pytest.mark.parametrize("plane_tilt", [-0.002, 0.0, 0.002])
def test_measured_registration_anchor_covers_complete_root_with_unchanged_errors(plane_tilt):
    initial = surface(np.eye(3), np.array([1.0, 0.0, 0.0]))
    rotation = Rotation.from_euler("z", 0.3).as_matrix()
    shift = np.array([0.02, -0.03, 0.01])
    measured_rotation = Rotation.from_euler("y", plane_tilt).as_matrix() @ rotation
    current = surface(measured_rotation, np.array([1.0, 0.0, 0.0]) @ rotation.T + shift)
    cameras = []
    for angles, position in [
        ([0.1, -0.2, 0.3], [0.2, -0.1, 0.4]),
        ([-0.2, 0.1, -0.1], [-0.1, 0.2, 0.3]),
    ]:
        camera = np.eye(4)
        camera[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
        camera[:3, 3] = position
        cameras.append(camera)
    # Uneven native coverage near the top; the whole material root stays present.
    world = initial.points[initial.points[:, 2] > 0.55][::4]
    world = np.vstack((world, [1.0, 0.0, 0.2]))
    source = (world - cameras[0][:3, 3]) @ cameras[0][:3, :3]
    true_current = world @ rotation.T + shift
    target = (true_current - cameras[1][:3, 3]) @ cameras[1][:3, :3]
    target[:, 1] += np.linspace(-0.0005, 0.0005, len(target))
    anchor = (initial.points.mean(0) - cameras[0][:3, 3]) @ cameras[0][:3, :3]
    result = constrained_surface_pose(
        initial, current, *cameras, source, target, 0.004, map_anchor=anchor
    )
    transform = result.camera_from_map
    residual = np.linalg.norm(source @ transform[:3, :3].T + transform[:3, 3] - target, axis=1)
    assert result.position_bound_m == pytest.approx(residual.max() + 0.004)
    assert result.pairs == len(source)
    reference = (initial.points - cameras[0][:3, 3]) @ cameras[0][:3, :3]
    old_radius = np.linalg.norm(reference - source.mean(0), axis=1).max()
    radius = np.linalg.norm(reference - result.registration_origin_map, axis=1).max()
    assert radius < old_radius
    assert any(np.array_equal(result.registration_origin_map, p) for p in source)
    estimated = reference @ transform[:3, :3].T + transform[:3, 3]
    actual_world = initial.points @ rotation.T + shift
    actual = (actual_world - cameras[1][:3, 3]) @ cameras[1][:3, :3]
    bound = result.position_bound_m + 2 * np.linalg.norm(
        reference - result.registration_origin_map, axis=1
    ) * np.sin(result.rotation_bound_rad / 2)
    assert np.all(np.linalg.norm(estimated - actual, axis=1) <= bound)
