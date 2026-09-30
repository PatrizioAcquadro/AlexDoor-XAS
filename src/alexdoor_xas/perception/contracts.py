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


def validate_complete(estimate):
    """Reject incomplete or inconsistent geometry before a controller consumes it."""
    from alexdoor_xas.action.b1 import checked_pose, finite_vector, panel_pose

    checked_pose(estimate.frame)
    panel = panel_pose(estimate.frame, estimate.signed_angle)
    rotation = np.asarray(estimate.panel_rotation)
    if rotation.shape != (3, 3) or not np.allclose(rotation, panel.rot, atol=1e-5):
        raise ValueError("Inconsistent observed panel rotation")
    if np.any(finite_vector(estimate.dimensions, 3) <= 0):
        raise ValueError("Nonpositive observed dimensions")
    checked_pose(ObjectFrame(estimate.contact_position, estimate.contact_rotation))
    if not np.isfinite(estimate.confidence) or not 0 <= estimate.confidence <= 1:
        raise ValueError("Invalid observed confidence")
