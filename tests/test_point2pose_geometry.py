"""Essential numerical 6.0C contracts; no CUDA inference substitute."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alexdoor_xas.perception.material_zone import (
    BoundSource,
    ContactBudget,
    transported_zone,
)
from alexdoor_xas.perception.point2pose_scale import observed_bounds, voxel_count
from alexdoor_xas.perception.point2pose_seed import candidate_references, verify_candidate
from alexdoor_xas.perception.point2pose_worker import pack_array, unpack_array


def test_camera_motion_cancels_in_world_and_map_object_alignment_is_once():
    initial = np.eye(4)
    initial[:3, 3] = [1, 2, 3]
    current = np.eye(4)
    current[:3, :3] = Rotation.from_euler("zy", [0.3, -0.2]).as_matrix()
    current[:3, 3] = [-0.3, 0.6, 2]
    mo = np.eye(4)
    mo[:3, :3] = Rotation.from_euler("y", 0.2).as_matrix()
    mo[:3, 3] = [0.5, -0.4, 1]
    oz = np.eye(4)
    oz[:3, 3] = [0, 0.3, -0.2]
    cm = np.linalg.inv(current) @ initial
    actual = transported_zone(current, cm, mo, oz)
    np.testing.assert_allclose(actual, initial @ mo @ oz, atol=1e-12)
    np.testing.assert_allclose(np.linalg.inv(actual) @ actual, np.eye(4), atol=1e-12)
    motion = np.eye(4)
    motion[:3, :3] = Rotation.from_euler("z", 0.4).as_matrix()
    np.testing.assert_allclose(
        transported_zone(current, np.linalg.inv(current) @ motion @ initial, mo, oz),
        motion @ initial @ mo @ oz,
        atol=1e-12,
    )


def test_tsdf_rule_retains_door_scale_and_finite_memory_count():
    points = np.array([[0, 0, 1], [1.1, 0, 1], [0, 2.2, 1], [1.1, 2.2, 1]])
    bounds = observed_bounds(points, 0.05)
    assert ((points >= bounds[:, 0]) & (points <= bounds[:, 1])).all()
    assert voxel_count(bounds, 0.005) > 20000
    for invalid in (0, np.nan, -1):
        with pytest.raises(ValueError):
            voxel_count(bounds, invalid)


def test_budget_uses_rotation_lever_and_refuses_duplicate_or_missing_sources():
    source = BoundSource("pose", 0.002, np.deg2rad(5), 0.8)
    budget = ContactBudget((source, BoundSource("local_geometry", 0.004, 0.0, 0.0)))
    assert budget.displacement_m == pytest.approx(0.006 + 1.6 * np.sin(np.deg2rad(2.5)))
    with pytest.raises(ValueError, match="duplicate"):
        ContactBudget((source, source))
    assert ContactBudget((BoundSource("camera_fk", None, None, 0.8),)).displacement_m is None


def test_candidate_preservation_accepts_relief_and_one_uncertain_edge_pixel():
    mask = np.zeros((80, 90), bool)
    mask[10:70, 10:65] = True
    depth = np.ones((*mask.shape, 1))
    k = np.array([[200, 0, 45], [0, 200, 40], [0, 0, 1.0]])
    points = candidate_references(mask, depth, depth > 0, k, 0.004)
    actual = mask.copy()
    actual[69, 64] = False
    actual[10:70, 65:70] = True
    verify_candidate(mask, actual, points)
    actual[points[0, 0], points[0, 1]] = False
    with pytest.raises(ValueError, match="candidate_mismatch"):
        verify_candidate(mask, actual, points)


def test_primitive_ipc_preserves_scalar_shape_and_numpy_abi_boundary():
    for value in (np.asarray(2.0), np.arange(6, dtype=np.float32).reshape(2, 3)):
        packed = pack_array(value)
        assert type(packed["data"]) is bytes
        np.testing.assert_array_equal(unpack_array(packed), value)
        assert unpack_array(packed).shape == value.shape


def test_lifting_optimization_matches_official_measured_and_completed_depth():
    path = Path(__file__).resolve().parents[1] / "models/perception/point2pose/upstream"
    if not path.exists():
        pytest.skip("Official runtime not installed; pure contracts remain covered")
    import sys

    sys.path.insert(0, str(path))
    from point2pose.utils.camera import convert_pixel_to_world

    from alexdoor_xas.perception.point2pose_compat import measured_depth_lifting

    optimized = measured_depth_lifting(convert_pixel_to_world, (0.1, 4.0))
    depth = np.ones((32, 32), dtype=np.float32)
    pixels = np.array([[2, 2], [15, 15], [0, 31], [-1, 10], [31, 0]])
    k = np.array([[200, 0, 16], [0, 200, 16], [0, 0, 1.0]])
    for missing in (False, True):
        depth[15, 15] = np.nan if missing else 1
        for uncertainty in (False, True):
            args = dict(
                pixel=pixels,
                depth_image=depth,
                cam_intrinsics=k,
                min_depth=0.1,
                max_depth=4.0,
                fill_missing_depth=True,
                window_size=5,
                compute_depth_uncertainty=uncertainty,
            )
            actual, expected = optimized(**args), convert_pixel_to_world(**args)
            for a, e in zip(actual, expected, strict=True):
                if a is None:
                    assert e is None
                else:
                    np.testing.assert_allclose(a, e, atol=0, rtol=0, equal_nan=True)


def test_informative_motion_fits_axis_and_stationary_world_does_not():
    from alexdoor_xas.perception.contracts import FieldSupport
    from alexdoor_xas.perception.observed_articulation import (
        MotionObservation,
        StaticReference,
        current_angle,
        fit_motion_axis,
    )

    initial = np.eye(4)
    initial[:3, 3] = [1, 0.5, 1.1]
    reference = StaticReference.observed(initial, FieldSupport(0, 0.1, 0, 0, 0.004, 0.001))
    pivot = np.array([-0.2, 0.3, 0.0])
    samples = []
    for tick, angle in enumerate((-0.2, -0.4, -0.6), 1):
        motion = np.eye(4)
        motion[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
        motion[:3, 3] = pivot - motion[:3, :3] @ pivot
        samples.append(
            MotionObservation(
                motion @ initial,
                FieldSupport(tick, tick + 0.1, tick, 0, 0.005, 0.001),
                initial[:3, 3].copy(),
            )
        )
    fit = fit_motion_axis(reference, samples, 0.0)
    np.testing.assert_allclose(fit["origin"], pivot, atol=1e-12)
    assert not fit["closed_reference_validated"]
    assert fit["position_bound_m"] is None and "floor" in fit["missing_bound_sources"]
    assert fit["conditional_position_bound_m"] > fit["bound_before_feedback_m"]
    assert fit["supported_s"] == 3 and reference.support.supported_s == 0
    latest = MotionObservation(
        samples[-1].pose, FieldSupport(3.1, 3.2, 3.1, 0, 0.005, 0.001), initial[:3, 3]
    )
    angle = current_angle(reference, latest)
    assert angle["relative_angle_rad"] == pytest.approx(-0.6)
    assert angle["supported_s"] == 3.1  # current accepted pose, no new motion sample
    assert fit_motion_axis(reference, samples[:1], 0.0) is None
    stationary = MotionObservation(
        initial, FieldSupport(1, 1.1, 1, 0, 0.005, 0.001), initial[:3, 3]
    )
    assert fit_motion_axis(reference, [stationary] * 4, 0.0) is None
    shifted = np.array([3, -2, 0.0])
    translated = initial.copy()
    translated[:3, 3] += shifted
    other = StaticReference.observed(translated, reference.support)
    moved = []
    for sample in samples:
        pose = sample.pose.copy()
        pose[:3, 3] += shifted
        moved.append(MotionObservation(pose, sample.support, sample.anchor + shifted))
    result = fit_motion_axis(other, moved, 0.0)
    np.testing.assert_allclose(result["origin"], pivot + shifted, atol=1e-12)
    assert result["conditional_position_bound_m"] == pytest.approx(
        fit["conditional_position_bound_m"]
    )


def test_vertical_consensus_preserves_noisy_sample_but_rejects_inclined_motion():
    from alexdoor_xas.perception.contracts import FieldSupport
    from alexdoor_xas.perception.observed_articulation import (
        MotionObservation,
        StaticReference,
        fit_motion_axis,
    )

    initial = np.eye(4)
    initial[:3, 3] = [1, 0.5, 1]
    reference = StaticReference.observed(initial, FieldSupport(0, 0, 0, 0, 0.001, 0.001))
    samples = []
    for i, angle in enumerate((0.2, 0.3, 0.4, 0.5)):
        rotation = Rotation.from_rotvec(np.array([0.2 if i == 0 else 0, 0, 1]) * angle)
        pose = initial.copy()
        pose[:3, :3] = rotation.as_matrix()
        pose[:3, 3] = rotation.apply(initial[:3, 3])
        samples.append(
            MotionObservation(
                pose, FieldSupport(i + 1, i + 1, i + 1, 0, 0.001, 0.001), initial[:3, 3]
            )
        )
    detail = {}
    fit = fit_motion_axis(reference, samples, 0, diagnostics=detail)
    assert fit is not None and len(samples) == 4
    assert detail["informative_samples"] == 4 and detail["compatible_samples"] == 3
    assert detail["rejected_sample_times_s"] == [1]
    assert fit_motion_axis(reference, [samples[0]] * 4, 0) is None
    # Correlated angular uncertainty may make an inverse-motion bound unbounded.
    wide = StaticReference.observed(initial, FieldSupport(0, 0, 0, 0, 0.001, 0.06))
    short = []
    for i, yaw in enumerate((0.13, 0.135, 0.14)):
        pose = initial.copy()
        pose[:3, :3] = Rotation.from_rotvec([0.03, 0, yaw]).as_matrix()
        pose[:3, 3] = pose[:3, :3] @ initial[:3, 3]
        short.append(
            MotionObservation(
                pose, FieldSupport(i + 1, i + 1, i + 1, 0, 0.001, 0.001), initial[:3, 3]
            )
        )
    result = fit_motion_axis(wide, short, 0)
    assert result["conditional_position_bound_m"] is None
    assert result["reason"] == "unbounded_rotation_feedback"


def test_finite_finger_slide_preserves_holes_and_can_exceed_one_centimeter():
    from scipy.ndimage import binary_erosion

    from alexdoor_xas.action.frames import ObjectFrame
    from alexdoor_xas.perception.geometry import deproject
    from alexdoor_xas.perception.material_zone import query_slip
    from test_point2pose_temporal import make_surface, sensor

    surface = make_surface(sensor())
    observation = surface.observations[0]
    mask = observation.mask()
    mask[36:44, 36:44] = False
    observation.support_mask = np.packbits(mask).tobytes()
    v, u = np.nonzero(mask & ~binary_erosion(mask))
    observation.boundary_points = deproject(
        np.ones(len(u)), np.c_[u, v], observation.intrinsics, observation.camera_world
    )
    faces = tuple(
        np.array(
            [
                [0.0, y - 0.01, -0.025],
                [0.0, y + 0.01, -0.025],
                [0.0, y + 0.01, 0.025],
                [0.0, y - 0.01, 0.025],
            ]
        )
        for y in (-0.025, 0.025)
    )
    budget = ContactBudget(
        (BoundSource("local_geometry", 0.004, 0.0, 0.04), BoundSource("pose", 0.001, 0.0, 0.04))
    )
    zone = np.eye(4)
    zone[:3, 3] = [1, 0, 0]
    config = dict(plane_tolerance_m=0.004)
    hole = query_slip(
        surface, zone, zone, ObjectFrame(np.array([1.0, 0, 0]), np.eye(3)), faces, budget, config
    )
    assert not hole.compatible
    slide = query_slip(
        surface, zone, zone, ObjectFrame(np.array([1.0, 0.2, 0]), np.eye(3)), faces, budget, config
    )
    assert slide.compatible
    assert slide.tangential_m[0] == pytest.approx(0.2)
    assert not slide.loaded_contact_admitted
    np.testing.assert_array_equal(zone[:3, 3], [1, 0, 0])


def test_contact_cover_can_be_localized_while_current_contact_depth_is_unknown():
    from alexdoor_xas.action.frames import ObjectFrame
    from alexdoor_xas.perception.material_zone import contact_depth_supported
    from test_point2pose_temporal import sensor

    captured = sensor()
    zone = np.eye(4)
    zone[:3, 3] = [1, 0, 0]
    tool = ObjectFrame(zone[:3, 3], np.eye(3))
    faces = tuple(
        np.array(
            [[0, y - 0.02, -0.04], [0, y + 0.02, -0.04], [0, y + 0.02, 0.04], [0, y - 0.02, 0.04]]
        )
        for y in (-0.06, 0.06)
    )
    assert contact_depth_supported(captured, zone, tool, faces, 0.008)
    captured["valid_depth"][38:42, 34:46] = False
    assert not contact_depth_supported(captured, zone, tool, faces, 0.008)
    assert not contact_depth_supported(sensor(), zone, tool, faces, None)
