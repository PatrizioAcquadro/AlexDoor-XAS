"""Finite material union must join real overlap while preserving common holes."""

from dataclasses import replace

import numpy as np

from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.perception.material_zone import MaterialCover
from perception_helpers import recipe
from test_scan_fusion import measured_surface


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
