"""Synthetic door geometry contracts."""

import numpy as np
import pytest

from alexdoor_xas.assets.synthetic_door import CASES, SyntheticDoor, rectangles_overlap


def test_handedness_uses_proper_frames_and_same_floor_reference():
    for width in (0.65, 1.2):
        left, right = SyntheticDoor(width, "left"), SyntheticDoor(width, "right")
        for angle in (0.0, 0.5, 1.0):
            lp, lr = left.contact_pose(angle, 0.9, 1.2)
            rp, rr = right.contact_pose(angle, 0.9, 1.2)
            np.testing.assert_allclose(lp * [1, -1, 1], rp)
            for rotation in (lr, rr):
                assert np.linalg.det(rotation) == pytest.approx(1.0)
                np.testing.assert_allclose(rotation[:, 2], [0, 0, 1])
        assert left.mechanical_stop == pytest.approx(right.mechanical_stop)


def test_geometric_stop_and_panel_extent():
    for door in CASES:
        assert np.rad2deg(door.mechanical_stop) > 90
        near = door.panel_rectangle(0.0)
        np.testing.assert_allclose(near[:, 1].max() - near[:, 1].min(), door.width)
        jamb = np.array(
            [[x, y + door.width / 2 + 0.045] for x in (-0.06, 0.06) for y in (-0.04, 0.04)]
        )
        if door.sign == 1:
            assert not rectangles_overlap(door.panel_rectangle(door.mechanical_stop), jamb)
            assert rectangles_overlap(
                door.panel_rectangle(door.mechanical_stop + np.deg2rad(0.3)), jamb
            )

