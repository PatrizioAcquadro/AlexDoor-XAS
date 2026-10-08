"""Runtime-specific checks separated from portable behavioral tests."""

import numpy as np

from alexdoor_xas.qualification.visibility import (
    read_visual_triangles,
)


def rectangle(x, y0, y1, z0, z1):
    points = np.array([[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]])
    return points[[[0, 1, 2], [0, 2, 3]]]


def test_visual_loading_excludes_guide_and_hidden_meshes():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Door")
    UsdGeom.Mesh.Define(stage, "/Door/Empty")
    for name, x in [("Visual", 0.04), ("Collision", 0), ("Hidden", -0.1)]:
        mesh = UsdGeom.Mesh.Define(stage, "/Door/" + name)
        mesh.CreatePointsAttr(rectangle(x, -1, 1, 0, 2).reshape(-1, 3).tolist())
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr(list(range(6)))
        if name == "Collision":
            mesh.CreatePurposeAttr("guide")
        elif name == "Hidden":
            mesh.CreateVisibilityAttr("invisible")
    triangles = read_visual_triangles(stage, "/Door")
    assert len(triangles) == 2
    np.testing.assert_allclose(triangles[:, :, 0], 0.04)
