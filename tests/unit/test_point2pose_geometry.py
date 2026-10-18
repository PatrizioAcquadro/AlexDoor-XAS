"""Essential numerical 6.0C contracts; no CUDA inference substitute."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alexdoor_xas.perception.ipc import pack_array, unpack_array
from alexdoor_xas.perception.material_zone import (
    BoundSource,
    ContactBudget,
    transported_zone,
)
from alexdoor_xas.perception.point2pose.scale import observed_bounds, voxel_count
from alexdoor_xas.perception.point2pose.seed import candidate_references, verify_candidate


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
    from perception_helpers import tracking_sensor as sensor
    from perception_helpers import tracking_surface as make_surface

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
    from perception_helpers import tracking_sensor as sensor

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


@pytest.mark.parametrize("eroded_clearance", [0.03, -0.02])
@pytest.mark.parametrize("tool_x", [0.0, -0.2])
def test_occluded_or_parked_tool_does_not_erase_selected_material_support(
    monkeypatch, tool_x, eroded_clearance
):
    from types import SimpleNamespace

    import alexdoor_xas.perception.point2pose.tracking as module
    from alexdoor_xas.perception.contracts import FieldSupport
    from alexdoor_xas.perception.material_zone import BoundSource, ContactBudget

    tracking = module.PanelTracking(
        SimpleNamespace(reset=lambda: None),
        {},
        calibration={},
        tool_fk=lambda *_: module.object_frame(
            np.array([[1, 0, 0, tool_x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ),
        distal_faces=(np.zeros((4, 3)), np.zeros((4, 3))),
    )
    tracking.candidates = [
        SimpleNamespace(
            candidate_id="material",
            state="tracked",
            registration_origin_map=None,
            support=FieldSupport(1, 1, 1, 0),
            geometry=FieldSupport(0, 0, 0, 0, 0.004, 0.001),
            material_recovery_unverified=False,
            ownership="observed_rigid_material",
            pose=np.eye(4),
            initial_world=np.eye(4),
            surface=None,
        )
    ]
    monkeypatch.setattr(
        tracking,
        "budget",
        lambda *_args, **_kw: ContactBudget((BoundSource("local_geometry", 0.004, 0.001, 0.05),)),
    )
    monkeypatch.setattr(
        module,
        "query_slip",
        lambda _surface, _initial, _zone, tool, *_args: SimpleNamespace(
            finger_clearance_m=(None, None)
            if abs(tool.origin[0]) > 0.1
            else (eroded_clearance, eroded_clearance),
            loaded_contact_admitted=False,
        ),
    )
    monkeypatch.setattr(
        module,
        "footprint_support",
        lambda *_: SimpleNamespace(finger_clearance_m=(0.03, 0.03)),
    )
    monkeypatch.setattr(module, "contact_depth_supported", lambda *_args: False)
    monkeypatch.setattr(
        tracking,
        "frame_budget",
        lambda *_: ContactBudget((BoundSource("dynamic_pose", 0.001, 0.001, 0),)),
    )
    state = tracking.local_state(1, dict(joint_position=np.zeros(9)))
    assert state.patches[0].finger_clearance_m == (0.03, 0.03)
    if abs(tool_x) > 0.1:
        assert tracking.diagnostics["tool_relative_zone"]["finger_clearance_m"] == (None, None)
    visibility = tracking.diagnostics["contact_visibility"]["material"]
    assert not visibility["direct_surface_depth_supported"]
    assert not visibility["contact_directly_measured"]
    assert not visibility["contact_admitted"]
    assert state.patches[0].geometry_support.position_bound_m == 0.004
    assert state.patches[0].pose_support.position_bound_m == 0.001


def test_frame_origin_budget_keeps_geometry_and_fingertip_rotation_separate():
    from types import SimpleNamespace

    from alexdoor_xas.perception.contracts import FieldSupport
    from alexdoor_xas.perception.point2pose.tracking import PanelTracking

    tracking = PanelTracking(
        SimpleNamespace(reset=lambda: None),
        {},
        calibration_bound=(0.0001, 0.001),
        distal_faces=(np.array([[0, -0.01, 0]]), np.array([[0, 0.01, 0]])),
    )
    candidate = SimpleNamespace(
        geometry=FieldSupport(0, 0, 0, 0, 0.004, 0.01),
        support=FieldSupport(1, 1, 1, 0),
        registration_origin_map=np.zeros(3),
        map_from_object=np.eye(4),
        object_from_zone=np.eye(4),
        registration_position_m=0.002,
        registration_rotation_rad=0.02,
    )
    frame, finger = tracking.frame_budget(candidate, 1), tracking.budget(candidate, 1)
    assert frame.displacement_m == pytest.approx(0.0021)
    assert frame.rotation_rad == pytest.approx(0.021)
    assert finger.displacement_m > 0.004 + frame.displacement_m
    assert finger.rotation_rad == pytest.approx(0.01 + frame.rotation_rad)
    assert "local_geometry" not in [source.name for source in frame.sources]


def test_explicit_rtx_pixel_centers_preserve_raw_k_and_integer_geometry():
    from alexdoor_xas.perception.geometry import (
        deproject,
        integer_center_intrinsics,
        pixel_center_offset,
        project,
    )

    raw = np.array([[368.0, 2.0, 480], [0, 368.0, 300], [0, 0, 1]])
    saved = raw.copy()
    offset = pixel_center_offset(
        {"camera_pixel_center_offset_xy": [0.5, 0.5]}, {"pixel_center_offset_xy": [0.5, 0.5]}
    )
    converted = integer_center_intrinsics(raw, offset)
    pixels = np.array([[10, 50], [400, 250]])
    expected = deproject([1.0, 2.0], pixels + offset, raw, np.eye(4))
    actual = deproject([1.0, 2.0], pixels, converted, np.eye(4))
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_allclose(project(actual, converted, np.eye(4))[0], pixels, atol=1e-13)
    np.testing.assert_array_equal(raw, saved)
    np.testing.assert_array_equal(pixel_center_offset({}), [0, 0])
    with pytest.raises(ValueError, match="mismatch"):
        pixel_center_offset(
            {"camera_pixel_center_offset_xy": [0.5, 0.5]}, {"pixel_center_offset_xy": [0, 0]}
        )
    with pytest.raises(ValueError, match="invalid_pixel_center"):
        pixel_center_offset({"camera_pixel_center_offset_xy": [np.nan, 0]})
    with pytest.raises(ValueError, match="invalid_camera"):
        integer_center_intrinsics(np.zeros((3, 3)), offset)


def test_point_budget_query_reuses_only_one_current_candidate_snapshot():
    from types import SimpleNamespace

    from alexdoor_xas.action.frames import ObjectFrame
    from alexdoor_xas.perception.contracts import FieldSupport
    from alexdoor_xas.perception.point2pose.tracking import PanelTracking

    tracking = PanelTracking(
        SimpleNamespace(reset=lambda: None),
        {},
        calibration_bound=(0.0001, 0.001),
        relative_speed_bound=(0.1, 0.2),
    )
    candidate = SimpleNamespace(
        geometry=FieldSupport(0, 0, 0, 0, 0.004, 0.01),
        support=FieldSupport(1, 1, 1, 0),
        registration_origin_map=np.zeros(3),
        map_from_object=np.eye(4),
        object_from_zone=np.eye(4),
        pose=np.eye(4),
        camera_world=np.eye(4),
        registration_position_m=0.002,
        registration_rotation_rad=0.02,
    )
    for now in (1.0, 1.05):
        query = tracking.point_budget_query(candidate, now)
        for center in ([0, 0, 0], [0, 0.4, 0], [1, 0.2, 0.5]):
            tool = ObjectFrame(np.asarray(center), np.eye(3))
            for radius in (0, 0.01, 0.3):
                faces = (np.array([[radius, 0.0, 0.0]]),) * 2
                expected = tracking.budget(candidate, now, distal_faces=faces, tool_world=tool)
                actual = query(radius, tool)
                assert actual.sources == expected.sources
                assert actual.displacement_m == expected.displacement_m
        # The next admission creates its own query after changing the state.
        candidate.pose[:3, 3] += [0.01, 0, 0]
        candidate.map_from_object[:3, 3] += [0, 0.02, 0]
