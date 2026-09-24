"""Configuration shared by learned policy families."""

from dataclasses import dataclass

from alexdoor_xas.action.spaces import A2_EE_DELTA
from alexdoor_xas.dataset.loader import validate_obs_keys


@dataclass(frozen=True)
class PolicyDatasetCfg:
    task: str
    version: str
    obs_keys: tuple[str, ...]
    space: str = A2_EE_DELTA
    view_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "obs_keys", validate_obs_keys(self.obs_keys))
