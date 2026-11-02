"""Independent coplanar seeds retain measured components without bridging gaps."""

import numpy as np
import pytest

from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.perception.material_zone import MaterialCover
from perception_helpers import measured_surface, recipe


def disconnected_inputs():
    surface, sensor, cue = measured_surface()
    surface.observations[0].semantic_id = 7
    mask = surface.observations[0].mask()
    mask[:, 115:126] = False
    cue.update(ids=[7], shape=mask.shape, masks=[np.packbits(mask).tobytes()])
    return surface, sensor, cue, mask


def make_cover(surface, sensor, cue, *, config=None, seed_mask=None):
    return MaterialCover.from_depth(
        surface,
        sensor,
        cue,
        7,
        recipe().config if config is None else config,
        generation=0,
        available_s=0,
        seed_mask=seed_mask,
    )


def supported(cover, y):
    face = np.array([[0, a, b] for a in (-0.025, 0.025) for b in (-0.025, 0.025)])
    return cover.query(ObjectFrame(np.array([1.0, y, 1.0]), np.eye(3)), (face, face)).supported


def test_independently_seeded_parts_preserve_the_missing_strip():
    surface, sensor, cue, _ = disconnected_inputs()
    cover = make_cover(surface, sensor, cue)
    assert supported(cover, -0.22)
    assert supported(cover, 0.22)
    assert not supported(cover, 0.0)


def test_disconnected_background_without_coplanar_seed_stays_unknown():
    surface, sensor, cue, _ = disconnected_inputs()
    sensor["depth_m"][:, 126:, 0] = 1.1
    cover = make_cover(surface, sensor, cue)
    assert not supported(cover, 0.22)
    assert cover.normal_range_m[1] < 0.01


def test_each_component_requires_the_original_minimum_seed():
    surface, sensor, cue, mask = disconnected_inputs()
    seed = np.zeros(mask.shape, bool)
    seed[70:80, 80:100] = True
    seed[70:80, 145:155] = True
    config = dict(recipe().config, min_points=100)
    at = make_cover(surface, sensor, cue, config=config, seed_mask=seed)
    above = make_cover(surface, sensor, cue, config=dict(config, min_points=101), seed_mask=seed)
    assert supported(at, 0.22)
    assert not supported(above, 0.22)
    assert supported(above, -0.22)


def test_duplicate_leaf_identity_is_rejected():
    surface, sensor, cue, _ = disconnected_inputs()
    cue["ids"] = [7, 7]
    cue["masks"] *= 2
    with pytest.raises(ValueError, match="unavailable_material_semantic_identity"):
        make_cover(surface, sensor, cue)


def test_independent_depth_columns_preserve_every_previous_cell_bound():
    surface, sensor, cue, mask = disconnected_inputs()
    sensor["depth_m"][90:150, 140:160, 0] = 1.01
    seed = mask & (abs(sensor["depth_m"][..., 0] - 1.0) <= recipe().config["plane_tolerance_m"])
    left_seed = seed.copy()
    left_seed[:, 126:] = False
    baseline = make_cover(surface, sensor, cue, seed_mask=left_seed)
    combined = make_cover(surface, sensor, cue, seed_mask=seed)
    assert baseline.clearance.max() > 0
    assert np.all(combined.clearance >= baseline.clearance)
    assert combined.diagnostics["independently_seeded_components"] == 2
    assert combined.normal_range_m[1] >= 0.01


@pytest.mark.parametrize("relief_m", [0.012, 0.024, 0.048])
def test_local_depth_columns_keep_parent_bounds_and_complete_measured_error(relief_m, monkeypatch):
    from alexdoor_xas.perception.geometry import deproject

    surface, sensor, cue = measured_surface()
    surface.observations[0].semantic_id = 7
    mask = surface.observations[0].mask()
    v, u = np.indices(mask.shape)
    sensor["depth_m"][mask, 0] = 1 + relief_m * ((u[mask] - 120) / 60) ** 2
    cue.update(ids=[7], shape=mask.shape)
    config = recipe().config
    initialize = MaterialCover.__init__
    columns = []

    def inspect_column(self, observed, cfg, **kwargs):
        initialize(self, observed, cfg, **kwargs)
        columns.append(
            (observed.observations[0].mask(), kwargs["normal_range_m"], self.clearance.copy())
        )

    monkeypatch.setattr(MaterialCover, "__init__", inspect_column)
    cover = make_cover(surface, sensor, cue)
    parent_mask, parent_range, parent_clearance = columns[0]
    assert len(columns) > 1
    assert np.all(cover.clearance >= parent_clearance)
    assert np.any(cover.clearance > parent_clearance)
    assert cover.normal_range_m == parent_range
    tolerance = config["plane_tolerance_m"]
    for part, bounds, _ in columns[1:]:
        assert not np.any(part & ~parent_mask)
        vv, uu = np.nonzero(part)
        cloud = deproject(
            sensor["depth_m"][vv, uu, 0],
            np.c_[uu, vv],
            sensor["intrinsics"],
            sensor["camera_world"],
        )
        depth = cloud @ surface.normal - surface.offset
        assert bounds[0] <= depth.min() - tolerance
        assert bounds[1] >= depth.max() + tolerance
    # A measured relief cannot restore an unmeasured central aperture or relax
    # the original normal-plane test, even when local tangent coverage improves.
    sensor["valid_depth"][110:131, 110:131] = False
    hole = make_cover(surface, sensor, cue)
    assert not supported(hole, 0.0)
    face = np.array([[0, a, b] for a in (-0.02, 0.02) for b in (-0.02, 0.02)])
    assert not cover.query(
        ObjectFrame(np.array([1.03, 0.0, 1.0]), np.eye(3)), (face, face)
    ).supported


def test_constant_depth_keeps_the_original_single_component_proof():
    surface, sensor, cue = measured_surface()
    surface.observations[0].semantic_id = 7
    cue.update(ids=[7], shape=surface.observations[0].shape)
    cover = make_cover(surface, sensor, cue)
    assert cover.diagnostics["independently_seeded_components"] == 1
    assert cover.diagnostics["local_normal_column_patches"] == 0
    assert supported(cover, 0.0)
