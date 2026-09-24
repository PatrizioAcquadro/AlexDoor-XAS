"""Adapter limits and door-panel geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorkspaceSphere:
    """Reachable shell around a fixed point (for example, the shoulder)."""

    center_w: tuple[float, float, float]
    min_reach_m: float
    max_reach_m: float

    def distance(self, point_w: np.ndarray) -> float:
        center = np.asarray(self.center_w, dtype=np.float64)
        return float(np.linalg.norm(np.asarray(point_w, dtype=np.float64).reshape(3) - center))

    def beyond_max_reach(self, point_w: np.ndarray, margin_m: float = 0.0) -> bool:
        return self.distance(point_w) > self.max_reach_m + margin_m

    def within_min_reach(self, point_w: np.ndarray) -> bool:
        return self.distance(point_w) < self.min_reach_m


@dataclass(frozen=True)
class RobotLimitsCfg:
    """Limits enforced and logged before the environment's hard clamps."""

    max_pos_delta_m: float = 0.02
    max_rot_delta_rad: float = 0.05
    workspace: WorkspaceSphere | None = None
    reach_margin_m: float = 0.02  # accommodates measured PD tracking overshoot
    contact_surface_x_m: float | None = None
    contact_approach_start_clearance_m: float | None = None
    contact_approach_max_step_m: float | None = None


@dataclass(frozen=True)
class DoorPanelGeometry:
    """Panel-frame geometry: hinge origin, Z axis, and +X push face."""

    panel_width_m: float = 0.83
    panel_height_m: float = 2.0
    panel_thickness_m: float = 0.036
    handle_band_y_m: tuple[float, float] = (0.63, 0.80)
    handle_band_z_m: tuple[float, float] = (0.0, 0.09)
    contact_eps_m: float = 0.002

    def surface_x_m(self, clearance_m: float) -> float:
        """Panel-frame x of the tool point off the +X face."""
        return self.panel_thickness_m + clearance_m

    def on_panel(self, point_panel: np.ndarray, tol_m: float = 0.0) -> bool:
        """Whether a panel-frame contact point lies on the panel face (±tol)."""
        point = np.asarray(point_panel, dtype=np.float64).reshape(3)
        half_height = self.panel_height_m / 2.0
        return bool(
            -tol_m <= point[1] <= self.panel_width_m + tol_m
            and -half_height - tol_m <= point[2] <= half_height + tol_m
        )

    def clamp_to_panel(self, point_panel: np.ndarray) -> np.ndarray:
        """Nudge a panel-frame point onto the panel face (y/z bounds only)."""
        point = np.asarray(point_panel, dtype=np.float64).reshape(3).copy()
        half_height = self.panel_height_m / 2.0
        point[1] = min(max(point[1], 0.0), self.panel_width_m)
        point[2] = min(max(point[2], -half_height), half_height)
        return point

    def in_handle_band(self, point_panel: np.ndarray) -> bool:
        point = np.asarray(point_panel, dtype=np.float64).reshape(3)
        return bool(
            self.handle_band_y_m[0] <= point[1] <= self.handle_band_y_m[1]
            and self.handle_band_z_m[0] <= point[2] <= self.handle_band_z_m[1]
        )

    def geometric_contact(self, ee_panel: np.ndarray) -> bool:
        """Tool-point-on-panel-face inference (the scripted controller's rule)."""
        point = np.asarray(ee_panel, dtype=np.float64).reshape(3)
        on_face = point[0] <= self.surface_x_m(self.contact_eps_m)
        half_height = self.panel_height_m / 2.0
        within_panel = (
            0.0 <= point[1] <= self.panel_width_m
            and -half_height <= point[2] <= half_height
            and point[0] >= 0.0
        )
        return bool(on_face and within_panel)


# The door stops against the wall at 90 degrees.
MAX_HINGE_ANGLE_RAD = math.pi / 2.0
