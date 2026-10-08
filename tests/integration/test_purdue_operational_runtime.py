"""Runtime-specific checks separated from portable behavioral tests."""

import numpy as np

from alexdoor_xas.assets.purdue import ARM_JOINTS, derive_push_geometry
from alexdoor_xas.envs.door_task.purdue_contacts import DistalSurface


def test_canonical_geometry():
    from ihmc_alex_isaaclab._paths import REPOSITORY_ROOT

    path = (
        REPOSITORY_ROOT
        / "assets/robots/alex_purdue/urdf/baseline/alex_purdue_wsg32_umi_v1_full_convex.urdf"
    )
    geometry = derive_push_geometry(path)
    np.testing.assert_allclose(geometry.translation, [0.202, 0, 0], atol=1e-5)
    for extremum, cover in zip(geometry.distal_faces, geometry.contact_covers(), strict=True):
        assert np.linalg.matrix_rank(extremum - extremum.mean(0)) == 1
        assert np.linalg.matrix_rank(cover - cover.mean(0)) == 2
        np.testing.assert_allclose(cover[:, 0], extremum[0, 0], atol=1e-9)
    assert len(ARM_JOINTS) == 7 and ARM_JOINTS[-1] == "RIGHT_GRIPPER_Y"
    for name, points in geometry.finger_vertices.items():
        surface = DistalSurface("actor", name, points, np.array([1.0, 0, 0]))
        face = points[np.abs(points[:, 0] - points[:, 0].max()) < 1e-6]
        point = (face.min(0) + face.max(0)) / 2
        assert surface.contains(point, np.array([1.0, 0, 0]))
        assert not surface.contains(point - [0.02, 0, 0], np.array([1.0, 0, 0]))
        assert not surface.contains(point, np.array([0.0, 1, 0]))
