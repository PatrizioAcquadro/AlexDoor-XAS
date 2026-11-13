"""Physics selection at the Purdue environment boundary."""

import os
from contextlib import contextmanager
from pathlib import Path

from isaac_physics_profile import STEEL, selected_profile


@contextmanager
def configure_purdue_physics(cfg):
    """Use the project profile unless a caller supplies a physics configuration."""
    if cfg.physics_profile is None:
        yield False
        return
    profile = selected_profile(default=cfg.physics_profile)
    if profile != STEEL:
        yield False
        return

    from isaac_physics_profile.steel import config, install_wheel, prepare_import_worker

    if not (wheel_site := os.environ.get("NEWTON_SITE")):
        raise RuntimeError("Start AlexDoor with scripts/isaac-run.sh to load the Steel runtime")
    site = Path(wheel_site).resolve(strict=True)
    with prepare_import_worker(site) as worker:
        install_wheel(site, defer_optional=True)
        from .steel_contacts import SteelPurdueViews

        cfg.sim.physics = config(
            site,
            (),
            measure_contact=True,
            constraint_capacity=512,
            contact_view=SteelPurdueViews,
            import_worker=worker,
        )
        yield True
