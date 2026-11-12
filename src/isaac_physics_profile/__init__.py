"""Shared physics profile selection for Isaac Lab consumers."""

import os
from pathlib import Path

PHYSX = "isaacsim_physx"
STEEL = "newton_steel"


def selected_profile(*, default=PHYSX):
    profile = os.environ.get("ISAAC_PHYSICS_PROFILE", default)
    if profile not in (PHYSX, STEEL):
        raise ValueError(f"Unknown Isaac physics profile: {profile}")
    return profile


def steel_config(env_roots, *, measure_contact=True, constraint_capacity=512, rigid_gap=None):
    from .steel import config, install_wheel

    site = Path(os.environ["NEWTON_SITE"]).resolve(strict=True)
    install_wheel(site)
    return config(
        site,
        env_roots,
        measure_contact=measure_contact,
        constraint_capacity=constraint_capacity,
        velocity_limits=True,
        rigid_gap=rigid_gap,
    )
