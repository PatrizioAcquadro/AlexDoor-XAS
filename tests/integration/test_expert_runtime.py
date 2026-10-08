"""Runtime-specific checks separated from portable behavioral tests."""

from itertools import product

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from alexdoor_xas.qualification.door_geometry import PreparedDoor


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
