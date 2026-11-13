"""Physics selection at the Purdue environment boundary."""

import os
from pathlib import Path

from isaac_physics_profile import STEEL, selected_profile


def configure_purdue_physics(cfg):
    """Use the project profile unless a caller supplies a physics configuration."""
    if cfg.physics_profile is None:
        return False
    profile = selected_profile(default=cfg.physics_profile)
    if profile != STEEL:
        return False

    from isaac_physics_profile.steel import config, install_wheel

    if not (wheel_site := os.environ.get("NEWTON_SITE")):
        raise RuntimeError("Start AlexDoor with scripts/isaac-run.sh to load the Steel runtime")
    site = Path(wheel_site).resolve(strict=True)
    install_wheel(site)
    from .steel_contacts import SteelPurdueViews

    cfg.sim.physics = config(
        site, (), measure_contact=True, constraint_capacity=512, contact_view=SteelPurdueViews
    )
    return True
