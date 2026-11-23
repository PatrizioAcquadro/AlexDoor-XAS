"""Uniform-density authoring preserves geometry-derived inertia and mass distribution."""

import numpy as np
import pytest

from alexdoor_xas.qualification.collection_dynamics import author_door_dynamics


@pytest.mark.parametrize("scale,damping", [(0.7, None), (1.0, 6.0), (1.3, 6.0)])
def test_density_scales_mass_and_inertia_together(scale, damping):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for name, kg in (("Panel", 25), ("Handle", 0.5)):
        prim = UsdGeom.Xform.Define(stage, "/Door/" + name).GetPrim()
        body = UsdPhysics.MassAPI.Apply(prim)
        body.CreateMassAttr(kg)
        body.CreateDiagonalInertiaAttr(Gf.Vec3f(1, 2, 2.5))
        body.CreateCenterOfMassAttr(Gf.Vec3f(0.1, 0.2, 1))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Door/Hinge")
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateDampingAttr(4)
    result = author_door_dynamics(stage, "/Door", mass_scale=scale, hinge_damping=damping)
    for name, kg in (("Panel", 25), ("Handle", 0.5)):
        values = result["bodies"][name]
        assert np.isclose(values["mass_kg"], kg * scale)
        np.testing.assert_allclose(values["diagonal_inertia_kg_m2"], np.array([1, 2, 2.5]) * scale)
        np.testing.assert_allclose(values["center_of_mass"], [0.1, 0.2, 1])
    assert result["hinge_damping"] == (4 if damping is None else damping)
