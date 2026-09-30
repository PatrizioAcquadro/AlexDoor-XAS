"""Observed door geometry in world coordinates; independent of any estimator."""

from dataclasses import dataclass

import numpy as np

from alexdoor_xas.action.frames import ObjectFrame


@dataclass(frozen=True)
class DoorEstimate:
    timestamp_s: float
    valid: bool
    reason: str
    confidence: float = 0.0
    frame: ObjectFrame | None = None
    panel_rotation: np.ndarray | None = None
    signed_angle: float | None = None
    dimensions: np.ndarray | None = None
    contact_position: np.ndarray | None = None
    contact_rotation: np.ndarray | None = None

    def fresh(self, now_s, max_age_s=0.15):
        return self.valid and 0 <= now_s - self.timestamp_s <= max_age_s
