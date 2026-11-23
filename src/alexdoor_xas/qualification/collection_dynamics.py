"""Session-layer door variations with coherent density and active-physics audit."""

import numpy as np


def author_door_dynamics(stage, root, *, mass_scale=1.0, hinge_damping=None):
    """Uniform leaf/handle density; preserve geometry, COM and principal axes.

    Call before the articulation enters physics. Original prepared assets are
    untouched. Damping is a synthetic viscous model, not identified hardware drag.
    """
    from pxr import Gf, UsdPhysics

    if not np.isfinite(mass_scale) or mass_scale <= 0:
        raise ValueError("Require positive finite uniform density scale")
    if hinge_damping is not None and (not np.isfinite(hinge_damping) or hinge_damping < 0):
        raise ValueError("Require finite nonnegative hinge damping")
    bodies = {}
    for name in ("Panel", "Handle"):
        prim = stage.GetPrimAtPath(root + "/" + name)
        if not prim:
            if name == "Handle":
                continue
            raise ValueError("Missing physical panel")
        mass = UsdPhysics.MassAPI(prim)
        kg = mass.GetMassAttr().Get()
        inertia = mass.GetDiagonalInertiaAttr().Get()
        if kg is None or kg <= 0 or inertia is None or min(inertia) <= 0:
            raise ValueError("Density variation requires explicit positive mass and inertia")
        mass.GetMassAttr().Set(float(kg * mass_scale))
        mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*(float(x * mass_scale) for x in inertia)))
        bodies[name] = dict(
            baseline_mass_kg=float(kg),
            mass_kg=float(mass.GetMassAttr().Get()),
            baseline_diagonal_inertia_kg_m2=list(inertia),
            diagonal_inertia_kg_m2=list(mass.GetDiagonalInertiaAttr().Get()),
            center_of_mass=list(mass.GetCenterOfMassAttr().Get()),
        )
    drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(root + "/Hinge"), "angular")
    original_damping = float(drive.GetDampingAttr().Get())
    if hinge_damping is not None:
        drive.GetDampingAttr().Set(float(hinge_damping))
    return dict(
        bodies=bodies,
        uniform_mass_inertia_scale=mass_scale,
        baseline_hinge_damping=original_damping,
        hinge_damping=float(drive.GetDampingAttr().Get()),
        provenance="synthetic_density_and_viscous_hinge_model_not_hardware_identification",
    )


def active_door_dynamics(env):
    """Read active body mass, inertia and runtime actuator damping."""
    from alexdoor_xas.recording.b1_runtime import array

    joint_damping = array(env.door.data.joint_damping)
    if env.cfg.native_steel:
        # The door uses an implicit hinge drive. Its damping is in the joint model.
        actuator_damping = {
            "hinge": joint_damping[:, [env.door.joint_names.index("Hinge")]].tolist()
        }
    else:
        actuator_damping = {k: array(v.damping).tolist() for k, v in env.door.actuators.items()}
    return dict(
        body_names=list(env.door.body_names),
        mass_kg=array(env.door.data.body_mass).tolist(),
        inertia_kg_m2=array(env.door.data.body_inertia).tolist(),
        joint_damping=joint_damping.tolist(),
        actuator_damping=actuator_damping,
    )
