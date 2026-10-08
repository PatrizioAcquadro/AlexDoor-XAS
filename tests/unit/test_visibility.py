"""Visibility uses rendered surfaces while retaining real occlusion checks."""

import subprocess
import sys
from textwrap import dedent
from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.assets.synthetic_door import SyntheticDoor
from alexdoor_xas.qualification.visibility import (
    front_mesh_points,
    visibility_groups,
    visible_points,
)


def rectangle(x, y0, y1, z0, z1):
    points = np.array([[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]])
    return points[[[0, 1, 2], [0, 2, 3]]]


def test_visual_intersections_preserve_recesses_holes_and_nearest_surface():
    triangles = np.concatenate([rectangle(0, -1, 0, 0, 2), rectangle(0.04, 0.2, 1, 0, 2)])
    points = front_mesh_points(triangles, [[-0.5, 1], [0.5, 1], [0.1, 1]])
    np.testing.assert_allclose(points[:2, 0], [0, 0.04])
    assert np.isnan(points[2, 0])
    combined = np.concatenate([triangles, rectangle(-0.1, 0.2, 1, 0, 2)])
    assert front_mesh_points(combined, [[0.5, 1]])[0, 0] == pytest.approx(-0.1)
    assert np.isnan(front_mesh_points(np.empty((0, 3, 3)), [[0, 1]])[0, 0])
    # A rendered recess matches its depth, while the enclosing convex proxy does not.
    depth = np.full((40, 40), 1.04)
    k = np.array([[10, 0, 20], [0, 10, 20], [0, 0, 1]])
    assert visible_points([[0, 0, 1.04]], np.zeros(3), np.eye(3), k, depth)[0]
    assert not visible_points([[0, 0, 1]], np.zeros(3), np.eye(3), k, depth)[0]


@pytest.mark.skipif(sys.platform != "linux", reason="Uses Linux process memory limits")
def test_many_visual_intersections_fit_bounded_memory():
    # The full Cartesian array would need >1 GiB before further temporaries.
    # Run in a child so a regression fails safely without exhausting host memory.
    subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""\
                import resource
                from pathlib import Path

                import numpy as np

                from alexdoor_xas.qualification.visibility import front_mesh_points

                triangles = np.tile([
                    [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[-0.1, 0, 0], [-0.1, 1, 0], [-0.1, 0, 1]],
                ], (3000, 1, 1))
                yz = np.tile([[0.25, 0.25], [0.75, 0.75], [0, 0], [0.5, 0.5]], (3000, 1))
                status = Path('/proc/self/status').read_text().splitlines()
                virtual = int(next(s.split()[1] for s in status if s.startswith('VmSize:'))) * 1024
                limit = virtual + 128 * 1024**2
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                points = front_mesh_points(triangles, yz)
                np.testing.assert_allclose(points[:, 0], np.tile([-0.1, np.nan, -0.1, -0.1], 3000))
                np.testing.assert_array_equal(points[:, 1:], yz)
                """),
        ],
        check=True,
        timeout=60,
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_visual_samples_rotate_with_leaf_and_cover_frame_near_contact_height(sign):
    door = SimpleNamespace(
        hinge=np.array([0.1, sign * 0.45, 0]),
        width=0.9,
        sign=sign,
        visual_triangles={
            "Panel": rectangle(0.04, -0.45, 0.45, 0, 2),
            "Frame": np.concatenate([rectangle(-0.05, y, y + 0.05, 0, 2) for y in [-0.5, 0.45]]),
        },
        _visibility_samples={},
    )
    door.rotation = lambda angle: SyntheticDoor.rotation(door, angle)
    door.contact_pose = lambda angle, fraction, height: (
        np.array([0, door.hinge[1] - sign * fraction * door.width, height]),
        np.eye(3),
    )
    closed = visibility_groups(door, 0, 0.295, 1.09)
    assert len(closed["panel"]) == 25 and len(closed["contact_surround"]) == 8
    assert np.all(closed["panel"][:, 0] == pytest.approx(0.04))
    assert len(np.unique(closed["frame"][:, 2])) == 7
    assert closed["frame"][:, 2].min() == pytest.approx(0.79)
    opened = visibility_groups(door, 0.8, 0.295, 1.09)
    np.testing.assert_allclose(opened["frame"], closed["frame"])
    np.testing.assert_allclose(
        (opened["panel"] - door.hinge) @ door.rotation(0.8) + door.hinge,
        closed["panel"],
        atol=1e-12,
    )
