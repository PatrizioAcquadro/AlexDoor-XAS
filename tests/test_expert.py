"""Prepared geometry and the two-cycle expert admission contract."""

import json
from itertools import product

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from alexdoor_xas.qualification.door_geometry import PreparedDoor
from alexdoor_xas.qualification.expert import publish_expert, qualify_pair
from alexdoor_xas.qualification.synthetic_probe import SustainedAngle


@pytest.fixture
def trial():
    return dict(
        case="door",
        passed=True,
        released=True,
        angle_deg=50.0,
        hold_angle_deg=49.9,
        stop_reason="safety_stop",
        safety_detail="tracking_margin",
        joint_margin=0.01,
        peak_force_n=2.0,
        clearance_m=0.0,
    )


def test_pair_requires_two_complete_consistent_cycles(trial):
    assert qualify_pair([trial])["status"] == "unresolved"
    assert qualify_pair([trial] * 3)["status"] == "unresolved"
    assert qualify_pair([trial, {**trial, "angle_deg": 52}])["theta_expert_d"] == 50
    for replacement in (
        {"angle_deg": 52.01},
        {"stop_reason": "timeout"},
        {"stop_reason": "solver_tracking_stall"},
        {"safety_detail": "high_normal_force"},
        {"released": False},
        {"hold_angle_deg": None},
        {"hold_angle_deg": float("nan")},
        {"passed": False},
    ):
        assert qualify_pair([trial, {**trial, **replacement}])["status"] == "unresolved"


def test_admission_is_inclusive_and_uses_lower_maximum(trial):
    low = {**trial, "angle_deg": 45.0}
    result = qualify_pair([low, {**low, "angle_deg": 46.0}])
    assert result["status"] == "qualified" and result["theta_expert_d"] == 45.0
    low["angle_deg"] = 44.99
    assert qualify_pair([low, low])["status"] == "out_of_domain"


def test_sustained_maximum_survives_lower_hold_and_release():
    window = SustainedAngle(0.5)
    for tick in range(11):
        window.update(tick / 10, 0.9 if tick < 6 else 0.8, True)
    for tick in range(11, 20):
        window.update(tick / 10, 0.8, True)
    window.update(2.0, 1.5, False)
    assert window.maximum == 0.9


def make_door(tmp_path, hand):
    folder = tmp_path / hand
    (folder / "prepared").mkdir(parents=True)
    (folder / "prepared/door.usda").touch()
    sign = 1 if hand == "left" else -1
    record = dict(
        handedness=hand,
        dimensions_m=dict(width_m=0.9, height_m=2, thickness_m=0.06),
        hinge_m=[0.11, sign * 0.46, 0],
        panel_center_m=[0.07, 0, 1.03],
        mechanical_limit_deg=74.5,
        usd="prepared/door.usda",
    )
    door = PreparedDoor(folder, record, {"components": {"Panel": [0]}})
    points = np.array(list(product([0.04, 0.10], [-0.45, 0.45], [0.03, 2.03])))
    door._leaf_planes = [ConvexHull(points).equations]
    door.shapes = {"Panel": [points], "Frame": [points + [0.1, 0.5, 0]], "Handle": []}
    return door


@pytest.mark.parametrize("hand", ["left", "right"])
def test_prepared_contact_uses_surface_and_actual_hinge(tmp_path, hand):
    door = make_door(tmp_path, hand)
    closed, _ = door.contact_pose(0, 0.4, 1)
    assert closed[0] == pytest.approx(0.04)
    assert closed[1] == pytest.approx(door.hinge[1] - door.sign * 0.36)
    opened, rotation = door.contact_pose(0.7, 0.4, 1)
    np.testing.assert_allclose((opened - door.hinge) @ rotation + door.hinge, closed)
    assert door.mechanical_stop == pytest.approx(np.deg2rad(74.5))
    assert door.footprint_inside(np.array([opened]), 0.7)
    assert not door.footprint_inside(np.array([[0, 4, 1]]), 0)
    with pytest.raises(ValueError, match="no collidable"):
        door.front_x(0.5, 1)
    assert all(name != "handle" for name, _ in door.collision_bounds(0.7))
    assert np.isfinite(door.frame_points).all()


def test_publication_preserves_preparation_and_detects_concurrent_edit(tmp_path):
    original = {
        "status": "ready_for_5.1",
        "expert_qualification": "not_run",
        "checks": {"physics": "pass"},
        "usd": "prepared/door.usda",
    }
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(original))
    report = dict(status="unresolved", theta_expert_d=None, reason="timeout", trials=[])
    publish_expert(tmp_path, original, report, tmp_path / "evidence")
    updated = json.loads(path.read_text())
    assert updated["checks"] == original["checks"]
    assert updated["usd"] == original["usd"]
    assert updated["expert_qualification"]["status"] == "unresolved"
    with pytest.raises(RuntimeError, match="changed"):
        publish_expert(tmp_path, original, report, tmp_path / "other")


def test_pedestal_gate_requires_positive_volume_not_touching(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    door = make_door(tmp_path, "left")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Pedestal")
    cube = UsdGeom.Cube.Define(stage, "/Pedestal/Collider")
    cube.CreateSizeAttr(0.1)
    shift = cube.AddTranslateOp()
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    shift.Set((-0.02, 0, 0.1))  # 1 cm gap from the leaf at X=0.04.
    assert door.pedestal_intersections(stage, "/Pedestal") == []
    shift.Set((-0.01, 0, 0.1))  # Touching is not a positive-volume intersection.
    assert door.pedestal_intersections(stage, "/Pedestal") == []
    shift.Set((0.01, 0, 0.1))
    witnesses = door.pedestal_intersections(stage, "/Pedestal")
    assert len(witnesses) == 1 and witnesses[0]["door_body"] == "Panel"


@pytest.mark.parametrize("metadata", ["scalar", "array", "both"])
def test_composed_leaf_accepts_published_component_metadata(tmp_path, metadata):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    door = make_door(tmp_path, "right")
    stage = Usd.Stage.CreateInMemory()
    for body in ("Panel", "Frame"):
        mesh = UsdGeom.Mesh.Define(stage, f"/Door/{body}/Collision")
        mesh.CreatePointsAttr(door.shapes[body][0].tolist())
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        if metadata in ("scalar", "both"):
            prim.CreateAttribute("b1:sourceComponent", Sdf.ValueTypeNames.Int).Set(0)
        if metadata in ("array", "both"):
            prim.CreateAttribute("b1:sourceComponents", Sdf.ValueTypeNames.IntArray).Set(
                [1] if metadata == "both" else [0]
            )
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Door/Hinge")
    hinge.CreateUpperLimitAttr(74.5)
    UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular").CreateDampingAttr(4.0)
    door.load_stage(stage, "/Door")
    assert len(door.leaf_shapes) == 1
    assert door.closed_contact(0.4, 1)[0] == pytest.approx(0.04)
