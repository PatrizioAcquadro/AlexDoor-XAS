"""Shared physics profile selection for Isaac Lab consumers."""

import os

PHYSX = "isaacsim_physx"
STEEL = "newton_steel"


def selected_profile(*, default=PHYSX):
    profile = os.environ.get("ISAAC_PHYSICS_PROFILE", default)
    if profile not in (PHYSX, STEEL):
        raise ValueError(f"Unknown Isaac physics profile: {profile}")
    return profile
