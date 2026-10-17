"""Conservative robot volume/occlusion math; no simulation or task truth."""

from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alexdoor_xas.kinematics.urdf_geometry import RobotGeometry


@pytest.fixture
def model(tmp_path):
    vertices = np.array(list(product((-0.1, 0.1), repeat=3)))
    # Convex cube triangle connectivity; the production reader accepts ASCII STL.
    from scipy.spatial import ConvexHull

    triangles = ConvexHull(vertices).simplices
    lines = ["solid cube"]
    for triangle in vertices[triangles]:
        lines += ["facet normal 0 0 0", "outer loop"]
        lines += ["vertex " + " ".join(map(str, point)) for point in triangle]
        lines += ["endloop", "endfacet"]
    (tmp_path / "cube.stl").write_text("\n".join(lines + ["endsolid cube"]))
    (tmp_path / "cube.obj").write_text("\n".join("v " + " ".join(map(str, v)) for v in vertices))
    path = tmp_path / "robot.urdf"
    path.write_text("""<robot name="calibrated"><link name="base"/>
<link name="tool"><collision><geometry><mesh filename="cube.stl"/></geometry></collision>
<visual><geometry><mesh filename="cube.obj"/></geometry></visual></link>
<joint name="q" type="revolute"><parent link="base"/><child link="tool"/>
<origin xyz="0 0 1"/><axis xyz="0 0 1"/></joint></robot>""")
    return RobotGeometry(path, ("q",))


def test_subdivided_cells_cover_full_rotating_robot_volume(model):
    matrices = model.transforms({"q": 0.0}, np.eye(4))
    cells = [child for cell in model.cells() for child in model.subdivide(cell)]
    assert len(cells) == 8
    balls = [model.world_ball(cell, matrices, {"q": 0.1}) for cell in cells]
    points = np.array(list(product(np.linspace(-0.1, 0.1, 5), repeat=3)))
    for angle in np.linspace(-0.1, 0.1, 9):
        world = points @ Rotation.from_rotvec([0, 0, angle]).as_matrix().T + [0, 0, 1]
        covered = np.zeros(len(world), bool)
        for center, radius in balls:
            covered |= np.linalg.norm(world - center, axis=1) <= radius + 1e-12
        assert covered.all()
    with pytest.raises(ValueError, match="sweep travel"):
        model.world_ball(cells[0], matrices, {})
    with pytest.raises(ValueError, match="joint state"):
        model.transforms({}, np.eye(4))


def test_robot_returns_are_removed_without_claiming_new_material(model):
    depth = np.ones((3, 3, 1))
    depth[0, 0] = 2.0
    sensor = dict(
        depth_m=depth,
        valid_depth=np.ones_like(depth, bool),
        intrinsics=np.array([[100.0, 0, 1], [0, 100.0, 1], [0, 0, 1]]),
        camera_world=np.eye(4),
    )
    removed = model.occluded_pixels(sensor, {"q": 0.0}, np.eye(4), (0.001, 0.001))
    assert removed.sum() == 8 and not removed[0, 0]
    assert sensor["valid_depth"].all()
    with pytest.raises(ValueError, match="occlusion error"):
        model.occluded_pixels(sensor, {"q": 0.0}, np.eye(4), (None, None))


def test_pixel_prefilter_matches_full_robot_volume_membership(model):
    from alexdoor_xas.perception.geometry import deproject

    root = np.eye(4)
    root[:3, :3] = Rotation.from_euler("y", 0.3).as_matrix()
    root[:3, 3] = [0.02, -0.01, 0.2]
    q = {"q": 0.25}
    k = np.array([[50.0, 0, 60], [0, 50.0, 50], [0, 0, 1]])
    depth = np.full((100, 120, 1), 1.15)
    sensor = dict(
        depth_m=depth, valid_depth=np.ones_like(depth, bool), intrinsics=k, camera_world=np.eye(4)
    )
    v, u = np.indices(depth.shape[:2]).reshape(2, -1)
    points = deproject(depth[v, u, 0], np.c_[u, v], k, np.eye(4))
    error = 0.001 + 2 * np.linalg.norm(points, axis=1) * np.sin(0.02 / 2)
    truth = np.zeros(len(points), bool)
    matrices = model.transforms(q, root)
    for name, lo, hi in model.occluders:
        transform = matrices[name]
        local = (points - transform[:3, 3]) @ transform[:3, :3]
        truth |= ((local >= lo - error[:, None]) & (local <= hi + error[:, None])).all(1)
    result = model.occluded_pixels(sensor, q, root, (0.001, 0.02))
    np.testing.assert_array_equal(result, truth.reshape(depth.shape[:2]))
    assert result.any()
    domain = np.zeros(depth.shape[:2], bool)
    domain[:, :60] = True
    selected = model.occluded_pixels(sensor, q, root, (0.001, 0.02), pixel_domain=domain)
    np.testing.assert_array_equal(selected, result & domain)


def test_cell_levers_cover_coupled_bent_chain_and_prismatic_motion(model):
    path = model.urdf
    path.write_text(
        path.read_text()
        .replace(
            '<joint name="q"',
            '<link name="elbow"/><link name="slider"/>'
            '<joint name="shoulder" type="revolute"><parent link="base"/>'
            '<child link="elbow"/><origin xyz="0 0 0"/><axis xyz="0 1 0"/></joint>'
            '<joint name="extension" type="prismatic"><parent link="elbow"/>'
            '<child link="slider"/><origin xyz="1 0 0"/><axis xyz="1 0 0"/></joint>'
            '<joint name="q"',
        )
        .replace(
            '<parent link="base"/><child link="tool"/>',
            '<parent link="slider"/><child link="tool"/>',
        )
    )
    model = RobotGeometry(path, ("shoulder", "extension", "q"))
    initial = {"shoulder": 0.8, "extension": 0.2, "q": -0.7}
    travel = {"shoulder": 1.4, "extension": 0.15, "q": 2.0}
    matrices = model.transforms(initial, np.eye(4))
    cells = [child for cell in model.cells() for child in model.subdivide(cell)]
    vertices = np.array(list(product((-0.1, 0.0, 0.1), repeat=3)))
    balls = [model.world_ball(cell, matrices, travel) for cell in cells]
    for fractions in product(np.linspace(-1, 1, 5), repeat=3):
        q = {
            name: initial[name] + f * travel[name]
            for name, f in zip(travel, fractions, strict=True)
        }
        transform = model.transforms(q, np.eye(4))["tool"]
        world = vertices @ transform[:3, :3].T + transform[:3, 3]
        assert np.any(
            [np.linalg.norm(world - center, axis=1) <= radius + 1e-12 for center, radius in balls],
            axis=0,
        ).all()
    cell = cells[0]
    _, radius = model.world_ball(cell, matrices, {"shoulder": 0, "extension": 0, "q": 0.1})
    old_extra = model.shapes[cell.shape]["levers"]["q"] * 0.1
    assert radius < np.linalg.norm(cell.upper - cell.lower) / 2 + old_extra


def test_velocity_cover_preserves_coupled_motion_and_acceleration_departures(model):
    path = model.urdf
    path.write_text("""<robot name="coupled"><link name="base"/><link name="elbow"/>
<link name="tool"><collision><origin xyz="0.3 0 0"/>
<geometry><mesh filename="cube.stl"/></geometry></collision></link>
<joint name="shoulder" type="revolute"><parent link="base"/><child link="elbow"/>
<origin xyz="0 0 1"/><axis xyz="0 0 1"/></joint>
<joint name="q" type="revolute"><parent link="elbow"/><child link="tool"/>
<origin xyz="0.5 0 0"/><axis xyz="0 1 0"/></joint></robot>""")
    robot = RobotGeometry(path, ("shoulder", "q"))
    positions = dict(shoulder=0.6, q=-0.4)
    velocity, acceleration, dt = np.array([0.2, -0.3]), np.array([5.0, 3.0]), 0.05
    cell = robot.cells()[0]
    center, radius = robot.velocity_sweep(positions, np.eye(4), velocity, acceleration, dt)(cell)
    vertices = np.array(list(product(*zip(cell.lower, cell.upper, strict=True))))
    for fraction in np.linspace(0, 1, 31):
        t = fraction * dt
        for signs in product((-1, 0, 1), repeat=2):
            q = (
                np.array(list(positions.values()))
                + velocity * t
                + 0.5 * acceleration * signs * t**2
            )
            matrices = robot.transforms(dict(zip(robot.moving_joints, q, strict=True)), np.eye(4))
            transform = matrices["tool"]
            points = vertices @ transform[:3, :3].T + transform[:3, 3]
            assert (np.linalg.norm(points - center, axis=1) <= radius + 1e-12).all()
    with pytest.raises(ValueError, match="acceleration"):
        robot.velocity_sweep(positions, np.eye(4), velocity, [-1, 1], dt)


def test_self_clearance_filters_fixed_descendants_by_actual_rigid_body(model):
    from alexdoor_xas.kinematics.self_clearance import RobotSelfClearance

    path = model.urdf
    path.write_text("""<robot name="groups"><link name="base"/><link name="jawA"/>
<link name="jawB"/><link name="fingerA"><collision><geometry>
<mesh filename="cube.stl"/></geometry></collision></link>
<link name="fingerB"><collision><geometry><mesh filename="cube.stl"/>
</geometry></collision></link>
<joint name="q" type="prismatic"><parent link="base"/><child link="jawA"/>
<axis xyz="1 0 0"/></joint>
<joint name="other" type="prismatic"><parent link="base"/><child link="jawB"/>
<axis xyz="1 0 0"/></joint>
<joint name="fixedA" type="fixed"><parent link="jawA"/><child link="fingerA"/></joint>
<joint name="fixedB" type="fixed"><parent link="jawB"/><child link="fingerB"/></joint>
</robot>""")
    robot = RobotGeometry(path, ("q",))
    matrices = robot.transforms(dict(q=0, other=0), np.eye(4))
    unfiltered = RobotSelfClearance(robot, ())
    with pytest.raises(ValueError, match="full-robot joint travel"):
        unfiltered.query(matrices, {"q": 0}, relative_model_error_m=0)
    assert unfiltered.query(matrices, {"q": 0, "other": 0}, relative_model_error_m=0)[0] < 0
    filtered = RobotSelfClearance(robot, (("jawA", "jawB"),))
    assert not filtered.pairs
    assert np.isinf(filtered.query(matrices, {"q": 0}, relative_model_error_m=0)[0])
    separated = robot.transforms(dict(q=0, other=1), np.eye(4))
    margin, pair = unfiltered.query(separated, {"q": 0.1, "other": 0}, relative_model_error_m=0.01)
    assert margin == pytest.approx(0.69)
    assert set(pair) == {"fingerA", "fingerB"}

    reduced, _ = unfiltered.query(separated, {"q": 0.1, "other": 0.2}, relative_model_error_m=0.01)
    assert reduced == pytest.approx(0.49)


def test_directional_stop_cover_keeps_complete_box_sphere_intersection():
    from alexdoor_xas.kinematics.urdf_geometry import stop_box_cover

    lower, upper = np.array([-0.013, -0.003, -0.004]), np.array([0.004, 0.009, 0.006])
    balls = stop_box_cover(lower, upper, 0.014, 0.003)
    rng = np.random.default_rng(37)
    points = np.r_[rng.uniform(lower, upper, (2000, 3)), np.zeros((1, 3))]
    points = points[np.linalg.norm(points, axis=1) <= 0.014]
    covered = np.zeros(len(points), bool)
    for center, radius in balls:
        assert radius <= 0.003
        covered |= np.linalg.norm(points - center, axis=1) <= radius + 1e-12
    assert covered.all() and len(balls) > 1
    with pytest.raises(ValueError, match="invalid_continuous"):
        stop_box_cover([0.001, 0, 0], upper, 0.014, 0.003)


def test_adaptive_stop_cover_keeps_whole_volume_and_refuses_missing_interior():
    from alexdoor_xas.kinematics.urdf_geometry import stop_box_cover

    lo, hi = np.full(3, -0.01), np.full(3, 0.01)
    whole = stop_box_cover(lo, hi, 0.02, 0.003, certify=lambda c, r, p: True)
    assert len(whole) == 1 and whole[0][1] == pytest.approx(np.sqrt(3) * 0.01)
    # A missing center remains inside the requested physical stop volume.
    missing = stop_box_cover(lo, hi, 0.02, 0.003, certify=lambda c, r, p: np.linalg.norm(c) > r)
    assert missing is None
