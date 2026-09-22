"""Four commissioning doors in a right-handed opening-centered floor frame."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticDoor:
    width: float
    handedness: str
    height: float = 2.10
    thickness: float = 0.04
    mass: float = 25.0
    damping: float = 4.0
    friction: float = 0.5
    gap: float = 0.015

    def __post_init__(self):
        if self.handedness not in ("left", "right") or self.width not in (0.65, 1.20):
            raise ValueError("Expected one of the four synthetic cases")
        if not np.isfinite([self.height, self.thickness, self.mass, self.damping]).all():
            raise ValueError("Non-finite door template")
        if min(self.height, self.thickness, self.mass) <= 0 or self.damping < 0:
            raise ValueError("Invalid door template")

    @property
    def sign(self):
        return 1 if self.handedness == "left" else -1

    @property
    def name(self):
        return f"{self.handedness}-{self.width:.2f}"

    @property
    def hinge(self):
        return np.array([0.085, self.sign * self.width / 2, 0.0])

    def rotation(self, angle):
        c, s = np.cos(self.sign * angle), np.sin(self.sign * angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    def contact_pose(self, angle, fraction, height, normal_offset=0.0):
        if not 0 < fraction < 1 or not 0 < height < self.height:
            raise ValueError("Contact must lie inside the panel")
        rotation = self.rotation(angle)
        local = np.array(
            [-self.thickness / 2 + normal_offset, -self.sign * fraction * self.width, height]
        )
        return self.hinge + rotation @ local, rotation

    def panel_rectangle(self, angle):
        points = np.array(
            [
                [x, -self.sign * y, 0.0]
                for x in (-self.thickness / 2, self.thickness / 2)
                for y in (self.gap, self.width - self.gap)
            ]
        )
        return (points @ self.rotation(angle).T + self.hinge)[:, :2]

    @property
    def mechanical_stop(self):
        """First panel/jamb intersection, resolved to 0.1 degree with a 0.2-degree gap."""
        jambs = [
            np.array(
                [
                    [x, y + side * (self.width / 2 + 0.045)]
                    for x in (-0.06, 0.06)
                    for y in (-0.04, 0.04)
                ]
            )
            for side in (-1, 1)
        ]
        for degrees in np.arange(1.0, 270.01, 0.1):
            polygon = self.panel_rectangle(np.deg2rad(degrees))
            if any(rectangles_overlap(polygon, jamb) for jamb in jambs):
                return float(np.deg2rad(degrees - 0.2))
        raise ValueError("Synthetic geometry has no demonstrated stop below 270 degrees")


def rectangles_overlap(a, b):
    """Separating axes for four corners, ordered as a Cartesian product."""
    for polygon in (a, b):
        for edge in (polygon[1] - polygon[0], polygon[2] - polygon[0]):
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return False
    return True


CASES = tuple(SyntheticDoor(w, h) for w in (0.65, 1.20) for h in ("left", "right"))


def author_synthetic_door(stage, root, door):
    """Author one fixed-frame articulation; handle remains a distinct contact actor."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    root_prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    physics = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    physics.CreateEnabledSelfCollisionsAttr(True)
    material = UsdShade.Material.Define(stage, root + "/Material")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(door.friction)
    api.CreateDynamicFrictionAttr(door.friction)
    api.CreateRestitutionAttr(0.0)

    def body(name, mass, center, inertia):
        obj = UsdGeom.Xform.Define(stage, root + "/" + name)
        obj.AddTranslateOp().Set(Gf.Vec3d(*door.hinge))
        UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
        m = UsdPhysics.MassAPI.Apply(obj.GetPrim())
        m.CreateMassAttr(mass)
        m.CreateCenterOfMassAttr(Gf.Vec3f(*center))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        p = PhysxSchema.PhysxRigidBodyAPI.Apply(obj.GetPrim())
        p.CreateSolverPositionIterationCountAttr(16)
        p.CreateSolverVelocityIterationCountAttr(4)
        return obj

    def box(parent, name, center, size, color):
        cube = UsdGeom.Cube.Define(stage, str(parent.GetPath()) + "/" + name)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        contact = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        contact.CreateContactOffsetAttr(0.002)
        contact.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose="physics")

    frame = body("Frame", 50.0, (0, 0, 1), (10, 10, 10))
    width = door.width - 2 * door.gap
    t, h, m = door.thickness, door.height, door.mass
    panel = body(
        "Panel",
        m,
        (0, -door.sign * door.width / 2, h / 2 + 0.01),
        (m * (width**2 + h**2) / 12, m * (t * t + h * h) / 12, m * (t * t + width * width) / 12),
    )
    box(
        panel,
        "Collider",
        (0, -door.sign * door.width / 2, h / 2 + 0.01),
        (t, width, h),
        (0.55, 0.30, 0.15),
    )
    for side in (-1, 1):
        y = side * (door.width / 2 + 0.045) - door.hinge[1]
        box(
            frame,
            "JambLeft" if side > 0 else "JambRight",
            (-door.hinge[0], y, h / 2),
            (0.12, 0.08, h),
            (0.3, 0.4, 0.5),
        )
    box(
        frame,
        "Header",
        (-door.hinge[0], -door.hinge[1], h + 0.06),
        (0.12, door.width + 0.17, 0.08),
        (0.3, 0.4, 0.5),
    )
    handle_center = (-0.06, -door.sign * 0.85 * door.width, 1.05)
    handle = body("Handle", 0.5, handle_center, (0.002, 0.002, 0.002))
    box(handle, "Collider", handle_center, (0.08, 0.12, 0.035), (0.7, 0.7, 0.7))
    fixed = UsdPhysics.FixedJoint.Define(stage, root + "/FixFrame")
    fixed.CreateBody1Rel().SetTargets([frame.GetPath()])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(*door.hinge))
    fixed.CreateLocalPos1Attr(Gf.Vec3f(0))
    hinge = UsdPhysics.RevoluteJoint.Define(stage, root + "/Hinge")
    hinge.CreateBody0Rel().SetTargets([frame.GetPath()])
    hinge.CreateBody1Rel().SetTargets([panel.GetPath()])
    hinge.CreateAxisAttr("Z")
    rotation = Gf.Quatf(1, 0, 0, 0) if door.sign > 0 else Gf.Quatf(0, 1, 0, 0)
    hinge.CreateLocalRot0Attr(rotation)
    hinge.CreateLocalRot1Attr(rotation)
    hinge.CreateLowerLimitAttr(0.0)
    hinge.CreateUpperLimitAttr(float(np.rad2deg(door.mechanical_stop)))
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(door.damping)
    joint = UsdPhysics.FixedJoint.Define(stage, root + "/FixHandle")
    joint.CreateBody0Rel().SetTargets([panel.GetPath()])
    joint.CreateBody1Rel().SetTargets([handle.GetPath()])
    # Attached bodies overlap intentionally; all robot/handle collisions stay active.
    joint.CreateCollisionEnabledAttr(False)
    return root_prim
