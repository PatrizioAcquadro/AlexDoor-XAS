"""Finite material union must join real overlap while preserving common holes."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.perception.material_zone import MaterialCover
from perception_helpers import measured_surface, recipe


def test_material_union_preserves_holes_and_joins_partial_views():
    surface, _, _ = measured_surface()
    observation = surface.observations[0]
    first, second = observation.mask(), observation.mask()
    first[:, 125:] = False
    second[:, :115] = False
    # Neither view alone can contain the wide footprint below.
    faces = tuple(
        np.array([[0, y, z] for y in (-0.3, 0.3) for z in (-0.06, 0.06)]) for _ in range(2)
    )
    pose = ObjectFrame(np.array([1.0, 0.0, 1.0]), np.eye(3))

    def cover(masks):
        observations = tuple(
            replace(observation, support_mask=np.packbits(mask).tobytes()) for mask in masks
        )
        return MaterialCover(
            replace(surface, observations=observations), recipe().config, cell_m=0.005
        )

    assert not cover([first]).query(pose, faces).supported
    assert not cover([second]).query(pose, faces).supported
    assert cover([first, second]).query(pose, faces).supported
    # A common internal hole remains unknown, even though all outer corners fit.
    first[110:131, 117:124] = False
    second[110:131, 117:124] = False
    assert not cover([first, second]).query(pose, faces).supported
    # Neither a pose beyond the measured plane nor outside the atlas is supported.
    assert (
        not cover([observation.mask()])
        .query(ObjectFrame(np.array([1.05, 0.0, 1.0]), np.eye(3)), faces)
        .supported
    )


def test_material_union_rejects_tilted_relief_with_similar_plane_offset():
    from alexdoor_xas.perception.scan import observed_material_surface

    surface, _, _ = measured_surface()
    surface.observations[0].semantic_id = 7
    tilted = replace(surface, points=surface.points.copy(), observations=list(surface.observations))
    tilted.points[:, 0] += 0.04 * (tilted.points[:, 2] - 1.0)
    tilted.normal = np.array([1.0, 0.0, -0.04])
    tilted.normal /= np.linalg.norm(tilted.normal)
    tilted.offset = float(np.median(tilted.points @ tilted.normal))
    fused = observed_material_surface(surface, [tilted], 7, recipe().config)
    assert fused is surface


def test_coverage_atlas_keeps_the_registration_plane_and_seed_unchanged():
    from alexdoor_xas.perception.scan import observed_material_surface

    seed, _, _ = measured_surface()
    seed.observations[0].semantic_id = 7
    other = replace(
        seed,
        points=seed.points + [0.002, 0, 0],
        observations=[replace(seed.observations[0], frame=99)],
    )
    other.normal = np.array([1.0, 0.0, 0.001])
    other.normal /= np.linalg.norm(other.normal)
    other.offset = seed.offset + 0.002
    atlas = observed_material_surface(seed, [other], 7, recipe().config)
    np.testing.assert_array_equal(atlas.normal, seed.normal)
    assert atlas.offset == seed.offset
    assert len(seed.observations) == 1
    assert len(atlas.observations) == 2


def test_dense_material_retains_measured_relief_and_rejects_separate_background():
    from alexdoor_xas.perception.geometry import surfaces

    seed, sensor, cue = measured_surface()
    cue["ids"] = [7]
    identity = 7
    cue["shape"] = sensor["depth_m"].shape[:2]
    mask = seed.observations[0].mask()
    # A measured shallow interior relief remains material, with explicit depth.
    sensor["depth_m"][90:150, 90:150, 0] = 1.006
    reference = max(surfaces(cue, sensor, recipe().config), key=lambda s: len(s.points))
    cover = MaterialCover.from_depth(
        reference, sensor, cue, identity, recipe().config, generation=0, available_s=0.0
    )
    face = np.array([[0.0, a, b] for a in (-0.02, 0.02) for b in (-0.02, 0.02)])
    pose = ObjectFrame(np.array([1.0, 0.0, 1.0]), np.eye(3))
    assert cover.query(pose, (face, face)).supported
    assert cover.normal_range_m[1] >= 0.006
    # A separate plane behind an aperture is not connected material support.
    sensor["depth_m"][90:150, 90:150, 0] = 1.1
    reference = max(surfaces(cue, sensor, recipe().config), key=lambda s: len(s.points))
    cover = MaterialCover.from_depth(
        reference, sensor, cue, identity, recipe().config, generation=0, available_s=0.0
    )
    assert not cover.query(pose, (face, face)).supported
    assert cover.normal_range_m[1] < 0.01
    # A semantic hole at a different depth remains missing, even when nearby.
    sensor["depth_m"][90:150, 90:150, 0] = 1.006
    mask[90:150, 90:150] = False
    cue["masks"] = [np.packbits(mask).tobytes()]
    reference = max(surfaces(cue, sensor, recipe().config), key=lambda s: len(s.points))
    cover = MaterialCover.from_depth(
        reference, sensor, cue, identity, recipe().config, generation=0, available_s=0.0
    )
    assert not cover.query(pose, (face, face)).supported


def test_complete_normal_slab_spends_local_parallax_and_keeps_inspection_clocks():
    surface, sensor, cue = measured_surface()
    face = np.array([[0, y, z] for y in (-0.02, 0.02) for z in (-0.02, 0.02)])
    pose = ObjectFrame(np.array([1.0, 0.42, 1.0]), np.eye(3))
    assert MaterialCover(surface, recipe().config).query(pose, (face, face)).supported
    slab = MaterialCover(surface, recipe().config, normal_range_m=(-0.2, 0.2))
    assert not slab.query(pose, (face, face)).supported
    cue["ids"] = [7]
    cue["shape"] = sensor["depth_m"].shape[:2]
    view = dict(sensor, semantic=cue, available_s=0.0)
    result = MaterialCover.from_inspection(surface, [view], 7, recipe().config, generation=0)
    assert result.support.acquired_s == result.support.available_s == 0.0
    future = dict(view, time_s=0.1, available_s=0.1)
    with pytest.raises(ValueError, match="inspection_after_initialization"):
        MaterialCover.from_inspection(surface, [future], 7, recipe().config, generation=0)


def test_projected_ball_margin_is_only_finite_tangent_support_and_preserves_holes():
    surface, _, _ = measured_surface()
    cover = MaterialCover(surface, recipe().config, cell_m=0.002)
    center = np.array([1.0, 0.0, 1.0])
    margin = cover.projected_ball_margin(center, 0.02)
    assert margin > 0
    assert cover.projected_ball_margin(center + [0.5, 0, 0], 0.02) == margin
    assert cover.projected_ball_margin(center, 1.0) < 0
    assert cover.projected_ball_margin(center + [0, 2, 0], 0.02) is None
    observation = surface.observations[0]
    mask = observation.mask()
    mask[110:131, 117:124] = False
    hole = MaterialCover(
        replace(
            surface, observations=(replace(observation, support_mask=np.packbits(mask).tobytes()),)
        ),
        recipe().config,
        cell_m=0.002,
    )
    assert hole.projected_ball_margin(center, 0.02) <= 0


def test_material_clearance_already_spends_the_declared_geometry_tolerance():
    from alexdoor_xas.perception.scan import footprint_support

    surface, _, _ = measured_surface()
    pose = ObjectFrame(np.array([1.0, 0.0, 1.0]), np.eye(3))
    face = np.array([[0, y, z] for y in (-0.02, 0.02) for z in (-0.02, 0.02)])
    config = recipe().config
    enlarged = dict(config, plane_tolerance_m=config["plane_tolerance_m"] + 0.001)
    for query in (
        lambda cfg: MaterialCover(surface, cfg).query(pose, (face, face)),
        lambda cfg: footprint_support(surface, pose, (face, face), cfg),
    ):
        before, after = query(config), query(enlarged)
        np.testing.assert_allclose(
            np.array(before.finger_clearance_m) - after.finger_clearance_m, 0.001, atol=1e-14
        )


@pytest.mark.parametrize("cell_m", [0.001, 0.0005])
def test_finer_inspection_grid_keeps_holes_depth_error_and_clocks(cell_m):
    from perception_helpers import camera_sensor

    sensor = camera_sensor()
    # A bounded local pane keeps both resolutions below the existing grid cap.
    sensor["intrinsics"][0, 0] *= 5
    sensor["intrinsics"][1, 1] *= 5
    surface, sensor, cue = measured_surface(sensor=sensor)
    mask = surface.observations[0].mask()
    mask[:, 115:126] = False
    sensor["depth_m"][:, 115:126, 0] = 1.1
    cue.update(ids=[7], shape=mask.shape, masks=[np.packbits(mask).tobytes()])
    config = dict(recipe().config, material_depth_cover={"cell_m": cell_m})
    view = dict(sensor, semantic=cue, available_s=float(sensor["time_s"]))
    cover = MaterialCover.from_inspection(surface, [view], 7, config, generation=0)
    face = np.array([[0, y, z] for y in (-0.005, 0.005) for z in (-0.005, 0.005)])

    def supported(y, x=1.0):
        return cover.query(ObjectFrame(np.array([x, y, 1.0]), np.eye(3)), (face, face)).supported

    assert supported(-0.045) and supported(0.045)
    assert not supported(0.0)  # The measured background strip is not material.
    assert not supported(0.13)  # Neither resolution extends the measured perimeter.
    assert not supported(0.045, x=1.02)
    tolerance = config["plane_tolerance_m"]
    assert cover.support.position_bound_m == tolerance
    np.testing.assert_allclose(cover.normal_range_m, [-tolerance, tolerance], atol=1e-14)
    assert cover.support.acquired_s == cover.support.available_s == float(sensor["time_s"])
