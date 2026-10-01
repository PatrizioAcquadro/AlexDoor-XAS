"""Observed object ownership, static axes, finite contact and relevant space."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.action.frames import ObjectFrame, rot_z
from alexdoor_xas.perception.geometry import Surface, SurfaceObservation, fuse_surface, surfaces
from alexdoor_xas.perception.provider import CueEngine, GeometryProvider
from alexdoor_xas.perception.scan import (
    ObservedSpace,
    ScanMemory,
    footprint_support,
    object_members,
    registered_support,
    static_hinges,
)
from alexdoor_xas.perception.tracking import moved_surface
from test_geometric_perception import EmptyWorker, recipe


def camera_sensor(time=0, frame=8):
    h, w = 240, 240
    return dict(
        rgb=np.ones((h, w, 3), np.uint8),
        depth_m=np.full((h, w, 1), 2.0),
        valid_depth=np.ones((h, w, 1), bool),
        intrinsics=np.array([[120, 0, 120], [0, 120, 120], [0, 0, 1]], float),
        camera_world=np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 1], [0, 0, 0, 1]], float),
        time_s=time,
        frame=frame,
        joint_position=np.zeros(9),
        joint_velocity=np.zeros(9),
    )


def measured_surface(sensor=None, mask=None):
    sample = camera_sensor() if sensor is None else sensor
    mask = np.zeros((240, 240), bool) if mask is None else mask.copy()
    if not mask.any():
        mask[30:211, 60:181] = True
    sample["depth_m"][mask] = 1
    cue = EmptyWorker().infer(sample["rgb"])
    cue["masks"] = [np.packbits(mask).tobytes()]
    cue["tokens"] = np.ones((196, 384), np.float32).tobytes()
    return surfaces(cue, sample, recipe().config)[0], sample, cue


def test_components_split_coplanar_frame_and_preserve_clipped_evidence():
    mask = np.zeros((240, 240), bool)
    mask[20:240, 60:181] = True
    mask[20:240, 190:220] = True
    _, sample, cue = measured_surface(mask=mask)
    parts = surfaces(cue, sample, recipe().config)
    assert len(parts) == 2
    assert all(p.observations and len(p.extent_points) for p in parts)
    assert all(p.observations[0].edge_status["height_0"] == "clipped" for p in parts)
    assert all("height_0" not in p.edge_points for p in parts)
    assert parts[0].bounds[1, 1] < parts[1].bounds[0, 1]


def test_dense_residual_support_survives_global_sampling_and_three_plane_limit():
    sample = camera_sensor()
    # Four small measured faces have fewer than min_points in a global sample,
    # but ample support after the large surrounding plane has been removed.
    for i, depth in enumerate((1.0, 1.1, 1.2, 1.3)):
        sample["depth_m"][40:70, 20 + i * 50 : 50 + i * 50] = depth
    cue = EmptyWorker().infer(sample["rgb"])
    cue["masks"] = [np.packbits(np.ones((240, 240), bool)).tobytes()]
    cue["tokens"] = np.ones((196, 384), np.float32).tobytes()
    parts = surfaces(cue, sample, recipe().config)
    assert all(any(abs(s.offset - depth) < 0.001 for s in parts) for depth in (1, 1.1, 1.2, 1.3))


def test_multiview_retains_original_boundaries_support_and_generation():
    memory = ScanMemory(recipe().config, 7)
    first, sample, _ = measured_surface()
    memory.add([first], sample, 8)
    following = camera_sensor(0.2, 20)
    # Move the calibrated view so its clipped upper boundary remains within
    # the original leaf, rather than measuring material beyond its top edge.
    following["camera_world"][2, 3] -= 0.25
    mask = np.zeros((240, 240), bool)
    mask[:180, 60:181] = True
    second, following, _ = measured_surface(following, mask)
    memory.add([second], following, 20)
    fused = memory.surfaces[0]
    assert len(memory.surfaces) == 1
    assert len(fused.observations) == 2
    assert fused.observations[1].edge_status["height_1"] == "clipped"
    assert "height_1" in fused.edge_points
    assert memory.state.generation == 7
    assert memory.state.objects[0].support.available_s >= 0.2


def test_missing_depth_beyond_silhouette_is_not_an_observed_edge():
    sample = camera_sensor()
    sample["valid_depth"][:] = False
    sample["valid_depth"][30:211, 60:181] = True
    part, _, _ = measured_surface(sample)
    assert not part.edge_points
    assert set(part.observations[0].edge_status.values()) == {"unobserved"}


def test_partial_view_boundary_cannot_trim_or_bound_previously_measured_material():
    first, _, _ = measured_surface()
    # A later clipped view sees an internal step as its bottom boundary.
    partial = replace(
        first,
        points=first.points[first.points[:, 2] > 1.0],
        extent_points=np.empty((0, 3)),
        edge_points={"height_0": np.array([[1, -0.2, 1], [1, 0.2, 1]])},
        views={2},
    )
    clipped = replace(first, edge_points={"height_1": first.edge_points["height_1"]})
    fused = fuse_surface(clipped, partial, recipe().config, 2)
    assert fused.bounds[0, 2] < 0.3
    assert "height_0" not in fused.edge_points
    assert "height_1" in fused.edge_points
    # Even an individual inconsistent edge record cannot contract measured bounds.
    inconsistent = replace(clipped, edge_points=partial.edge_points)
    assert inconsistent.bounds[0, 2] < 0.3


def test_reprojection_uses_the_calibrated_camera_instead_of_image_overlap():
    surface, sample, cue = measured_surface()
    shifted = camera_sensor(0.2, 20)
    shifted["camera_world"][:3, 3] += [0, 0.15, 0]
    # A fixed world plane shifts by 18 pixels under this calibrated translation.
    mask = np.zeros((240, 240), bool)
    mask[30:211, 42:163] = True
    other, _, _ = measured_surface(shifted, mask)
    assert registered_support(surface, other.observations[0])
    incorrect = replace(other.observations[0], camera_world=sample["camera_world"])
    narrow = other.observations[0].mask()
    narrow[:, 60:] = False
    observation = replace(other.observations[0], support_mask=np.packbits(narrow).tobytes())
    assert registered_support(surface, observation)
    assert not registered_support(
        surface, replace(observation, camera_world=incorrect.camera_world)
    )


def test_repeated_overlapping_proposals_do_not_multiply_static_geometry():
    memory = ScanMemory(recipe().config, 0)
    for frame in range(8, 28):
        sample = camera_sensor((frame - 8) * 0.2, frame)
        mask = np.zeros((240, 240), bool)
        mask[30:211, 60:181] = True
        first, sample, _ = measured_surface(sample, mask)
        smaller = mask.copy()
        smaller[30:80] = False
        second, sample, _ = measured_surface(sample, smaller)
        memory.add([first, second], sample, frame)
    assert len(memory.surfaces) <= 2
    assert sum(len(s.observations) for s in memory.surfaces) == 40
    assert not memory.ambiguous_associations


@pytest.mark.parametrize("bottom_frame", [False, True])
def test_object_perimeter_keeps_leaf_bottom_and_excludes_surrounding_support(bottom_frame):
    memory = ScanMemory(recipe().config, 0)
    sample = camera_sensor()
    mask = np.zeros((240, 240), bool)
    mask[30:211, 60:181] = True
    sample["depth_m"][mask] = 1
    if bottom_frame:
        mask[215:230, 55:186] = True
        sample["depth_m"][215:230, 55:186] = 1.02
    cue = EmptyWorker().infer(sample["rgb"])
    cue["masks"] = [np.packbits(mask).tobytes()]
    cue["tokens"] = np.ones((196, 384), np.float32).tobytes()
    for frame, time_s in ((8, 0), (20, 0.2)):
        sample.update(frame=frame, time_s=time_s)
        memory.add(surfaces(cue, sample, recipe().config), sample, frame)
    assert len(memory.state.objects) == 1
    root = memory.state.objects[0].root
    assert root.edges["height_0"] == pytest.approx(0.25, abs=0.01)
    assert bool(memory.state.fixed_surfaces) == bottom_frame


def fixture_surface(points, normal, identifier, boundary, *, mask_index=0):
    sample = camera_sensor()
    observation = SurfaceObservation(
        8,
        0,
        0.1,
        mask_index,
        (240, 240),
        np.packbits(np.ones((240, 240), bool)).tobytes(),
        sample["intrinsics"],
        sample["camera_world"],
        boundary,
        {},
    )
    return Surface(
        points,
        np.array(normal, float),
        float(points.mean(0) @ normal),
        np.zeros(384),
        np.empty((0, 3)),
        np.empty((0, 384)),
        0.001,
        1.0,
        {0, 1},
        surface_id=identifier,
        observations=(observation,),
    )


def test_surrounding_support_needs_only_its_measured_separating_border():
    root, _, _ = measured_surface()
    root.surface_id = "root"
    root.views = {0, 1}
    root.edge_points.pop("height_0")
    root.edges.pop("height_0")
    y, z = np.meshgrid(np.linspace(0.6, 0.7, 20), np.linspace(0.4, 1.5, 20))
    points = np.c_[np.full(y.size, 1.02), y.ravel(), z.ravel()]
    outside = fixture_surface(points, [1, 0, 0], "side-support", points)
    unknown = replace(outside, surface_id="unknown-bottom", points=points - [0, 0.65, 1.7])
    memory = ScanMemory(recipe().config, 0)
    memory.surfaces = [root, outside, unknown]
    memory.assemble()
    assert [s.surface_id for s in memory.state.fixed_surfaces] == ["side-support"]
    assert [s.surface_id for s in memory.state.unresolved_surfaces] == ["unknown-bottom"]


def test_relief_requires_an_observed_connector_not_parallelism_or_color():
    y, z = np.meshgrid(np.linspace(-0.5, 0.5, 31), np.linspace(0.1, 1.9, 41))
    root = fixture_surface(
        np.c_[np.ones(y.size), y.ravel(), z.ravel()],
        [1, 0, 0],
        "root",
        np.c_[np.ones(31), np.linspace(-0.25, 0.25, 31), np.full(31, 0.8)],
    )
    y, z = np.meshgrid(np.linspace(-0.25, 0.25, 31), np.linspace(0.8, 1.3, 31))
    relief = fixture_surface(
        np.c_[np.full(y.size, 1.02), y.ravel(), z.ravel()],
        [1, 0, 0],
        "relief",
        np.c_[np.full(31, 1.02), np.linspace(-0.25, 0.25, 31), np.full(31, 0.8)],
    )
    x, y = np.meshgrid(np.linspace(1, 1.02, 5), np.linspace(-0.25, 0.25, 31))
    connector = fixture_surface(
        np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.8)],
        [0, 0, 1],
        "seam",
        np.r_[root.observations[0].boundary_points, relief.observations[0].boundary_points],
    )
    frame = replace(relief, surface_id="parallel-frame", points=relief.points + [0, 0.8, 0])
    assert len(object_members(root, [root, relief, frame], recipe().config)) == 1
    members = object_members(root, [root, relief, connector, frame], recipe().config)
    assert {s.surface_id for s in members} == {"root", "relief", "seam"}
    different_mask = replace(
        connector, observations=(replace(connector.observations[0], mask_index=1),)
    )
    assert len(object_members(root, [root, relief, different_mask], recipe().config)) == 1


def test_horizontal_connector_fusion_preserves_material_and_observations():
    x, y = np.meshgrid(np.linspace(1, 1.02, 5), np.linspace(-0.25, 0.25, 31))
    points = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.8)]
    connector = fixture_surface(points, [0, 0, 1], "seam", points)
    fused = fuse_surface(connector, connector, recipe().config, 2)
    assert fused.surface_id == "seam" and len(fused.observations) == 2
    np.testing.assert_allclose(fused.basis.T @ fused.basis, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("side", [0, 1])
def test_static_hardware_axes_both_sides_and_edge_is_not_axis(side):
    panel, _, _ = measured_surface()
    panel.surface_id = "observed"
    y = panel.edges[f"width_{side}"]
    theta, z = np.meshgrid(np.linspace(-np.pi, np.pi, 100, endpoint=False), [0.45, 0.85, 1.25])
    center = np.array([0.985, y])
    cloud = np.c_[
        center[0] + 0.012 * np.cos(theta.ravel()),
        center[1] + 0.012 * np.sin(theta.ravel()),
        z.ravel(),
    ]
    candidates = static_hinges(panel, np.r_[panel.points, cloud], recipe().config, 3)
    axes = [h for h in candidates if h.hypothesis is not None]
    assert len(axes) == 1
    np.testing.assert_allclose(axes[0].origin, [*center, 0], atol=0.0003)
    axes[0].hypothesis.support.require(1, 3)
    assert axes[0].hypothesis.support.position_bound_m > 0
    assert all(h.hypothesis is None for h in candidates if h.evidence == "panel_edge")
    # Hidden hardware and a nearly straight tiny arc never manufacture a physical axis.
    assert not any(h.hypothesis for h in static_hinges(panel, np.empty((0, 3)), recipe().config, 3))
    tiny = cloud.reshape(3, 100, 3)[:, :4].reshape(-1, 3)
    assert not any(h.hypothesis for h in static_hinges(panel, tiny, recipe().config, 3))
    arc, height = np.meshgrid(np.linspace(-0.2, 0.2, 60), [0.45, 0.85, 1.25])
    conditioned_poorly = np.c_[
        center[0] + 0.012 * np.cos(arc.ravel()),
        center[1] + 0.012 * np.sin(arc.ravel()),
        height.ravel(),
    ]
    assert not any(
        h.hypothesis for h in static_hinges(panel, conditioned_poorly, recipe().config, 3)
    )


def test_indistinguishable_static_hardware_keeps_both_axis_alternatives():
    panel, _, _ = measured_surface()
    clouds = []
    theta, z = np.meshgrid(np.linspace(-np.pi, np.pi, 100, endpoint=False), [0.45, 0.85, 1.25])
    for side in (0, 1):
        clouds.append(
            np.c_[
                0.985 + 0.012 * np.cos(theta.ravel()),
                panel.edges[f"width_{side}"] + 0.012 * np.sin(theta.ravel()),
                z.ravel(),
            ]
        )
    axes = [
        h
        for h in static_hinges(panel, np.concatenate(clouds), recipe().config, 0)
        if h.hypothesis is not None
    ]
    assert len(axes) == 2 and axes[0].candidate_id != axes[1].candidate_id


def finger_faces():
    return tuple(
        np.array(
            [
                [0, y + dy, dz]
                for dy, dz in ((-0.01, -0.01), (-0.01, 0.01), (0.01, -0.01), (0.01, 0.01))
            ]
        )
        for y in (-0.02, 0.02)
    )


def test_entire_two_finger_footprint_rejects_holes_and_edge_clipping():
    surface, _, _ = measured_surface()
    pose = ObjectFrame(np.array([1, 0, 1]), np.eye(3))
    result = footprint_support(surface, pose, finger_faces(), recipe().config)
    assert result.supported and min(result.finger_clearance_m) > 0.1
    mask = surface.observations[0].mask()
    mask[119:123, 116:119] = False
    damaged = replace(
        surface,
        observations=(replace(surface.observations[0], support_mask=np.packbits(mask).tobytes()),),
    )
    assert not footprint_support(damaged, pose, finger_faces(), recipe().config).supported
    clipped = ObjectFrame(np.array([1, 0.49, 1]), np.eye(3))
    assert not footprint_support(surface, clipped, finger_faces(), recipe().config).supported


def test_repeated_vertices_on_a_line_are_not_a_finite_finger_face():
    surface, _, _ = measured_surface()
    pose = ObjectFrame(np.array([1, 0, 1]), np.eye(3))
    # The current collision extremum produces nine vertices but only two
    # distinct positions; filling its projected line must not certify a face.
    lines = tuple(np.array([[0, y, -0.02], [0, y, 0.02], [0, y, -0.02]] * 3) for y in (-0.02, 0.02))
    result = footprint_support(surface, pose, lines, recipe().config)
    assert not result.supported and result.reason == "degenerate_finger_face"
    assert result.finger_clearance_m == (None, None)


def test_ray_space_queries_only_requested_cover_and_do_not_fill_unknown():
    space = ObservedSpace(recipe().config)
    sample = camera_sensor()
    space.add(sample)
    free = space.query([[1, 0, 1]], [0.02])
    assert not free.unknown_in_envelope and not free.occupied and free.collision_clearance_m > 0
    assert space.query([[2, 0, 1]], [0.02]).occupied
    assert space.query([[3, 0, 1]], [0.02]).unknown_in_envelope
    assert space.query([[1, 10, 1]], [0.02]).unknown_in_envelope
    sample["valid_depth"][120, 120] = False
    space.add(sample)
    assert space.query([[1, 0, 1]], [0.02]).unknown_in_envelope
    assert space.query([], []).unknown_in_envelope


def test_free_space_margin_accounts_for_lateral_obstacles_and_unknown_boundaries():
    sample = camera_sensor()
    # Outside the requested sphere's rays, but much closer than the far background.
    sample["depth_m"][120, 126] = 1.0
    space = ObservedSpace(recipe().config)
    space.add(sample)
    result = space.query([[1, 0, 1]], [0.02])
    assert not result.unknown_in_envelope and not result.occupied
    assert 0 < result.collision_clearance_m < 0.03
    # An unrelated missing pixel cannot invalidate a supported local volume.
    sample["valid_depth"][0, 0] = False
    space.add(sample)
    assert not space.query([[1, 0, 1]], [0.02]).unknown_in_envelope


def test_scan_fusion_does_not_pick_area_winner_or_erase_material_references():
    provider = GeometryProvider(recipe(), CueEngine(EmptyWorker(), replay=True))
    surface, sample, cue = measured_surface()
    provider._fuse(sample, 8, cue)
    second = dict(sample, frame=20, time_s=0.2)
    provider._fuse(second, 20, cue)
    assert provider.scan_state.objects
    assert provider.last_estimate is None  # fusion is not a qualified runtime estimate
    reference = provider.static[0]
    moved = moved_surface(reference, (rot_z(0.1), np.array([0.1, 0, 0]), 0.001))
    assert moved.surface_id == reference.surface_id
    assert moved.observations == reference.observations
    assert moved.edge_points.keys() == reference.edge_points.keys()
    previous = provider.scan_state.generation
    provider.reset()
    assert provider.scan_state.generation == previous + 1 and not provider.scan_state.objects
