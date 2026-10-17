"""Observed static workcell calibration, separate from current dynamic evidence."""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from alexdoor_xas.perception.admission import rotational_travel
from alexdoor_xas.perception.scan import ObservedSpace


@dataclass(frozen=True)
class StaticWorkspaceSupport:
    collision_clearance_m: float | None
    unknown_in_envelope: bool
    occupied: bool
    source_clocks: tuple


class CommonWorkspace:
    """Intersect observed free volumes across every supplied calibration condition.

    These static rays keep their acquisition clocks. They cannot replace current
    target identity/pose, dynamic-object exclusion, robot response or stop evidence.
    No asset identity or evaluator annotation is read to answer a query.
    """

    def __init__(self, spaces, calibration_bound):
        self.spaces = tuple(spaces)
        self.calibration_bound = np.asarray(calibration_bound, dtype=float)
        if (
            not self.spaces
            or self.calibration_bound.shape != (2,)
            or not np.isfinite(self.calibration_bound).all()
            or (self.calibration_bound < 0).any()
            or any(not space.views for space in self.spaces)
        ):
            raise ValueError("missing_observed_workspace_calibration")
        self.source_clocks = tuple(
            tuple((float(v["time_s"]), float(v["available_s"])) for v in space.views.values())
            for space in self.spaces
        )

    @classmethod
    def from_recordings(cls, paths, config, calibration_bound):
        spaces = []
        for path in paths:
            with h5py.File(Path(path), "r") as source:
                if not bool(source.attrs.get("complete", False)):
                    raise ValueError("incomplete_workspace_recording")
                observed = source["observations"]
                times = observed["time_s"][:]
                if not len(times) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
                    raise ValueError("invalid_workspace_capture_clock")
                calibration = json.loads(source["metadata"].attrs["calibration"])
                measured = json.loads(source["metadata"].attrs["camera_fk_error"])
                if any(
                    not np.isfinite(measured[key]) or measured[key] > bound
                    for key, bound in zip(
                        ("position_m", "rotation_rad"), calibration_bound, strict=True
                    )
                ):
                    raise ValueError("workspace_camera_calibration_exceeded")
                space = ObservedSpace(config)
                for timestamp in config["inspection"]["sample_times_s"]:
                    i = int(np.argmin(np.abs(times - timestamp)))
                    if abs(times[i] - timestamp) > 1e-8:
                        raise ValueError("missing_workspace_inspection_view")
                    sensor = {
                        key: observed[key][i]
                        for key in (
                            "time_s",
                            "frame",
                            "depth_m",
                            "valid_depth",
                            "intrinsics",
                            "camera_world",
                        )
                    }
                    depth = sensor["depth_m"]
                    near, far = calibration["depth_interval_m"]
                    if not np.array_equal(
                        sensor["valid_depth"], np.isfinite(depth) & (depth >= near) & (depth <= far)
                    ):
                        raise ValueError("workspace_depth_validity_mismatch")
                    space.add(sensor)
                spaces.append(space)
        return cls(spaces, calibration_bound)

    def query_static(self, centers, radii):
        centers, radii = np.asarray(centers, float), np.asarray(radii, float)
        if (
            centers.ndim != 2
            or centers.shape[1:] != (3,)
            or not len(centers)
            or radii.shape != (len(centers),)
            or not np.isfinite(centers).all()
            or not np.isfinite(radii).all()
            or (radii < 0).any()
        ):
            return StaticWorkspaceSupport(None, True, False, self.source_clocks)
        supports = []
        for space in self.spaces:
            lever = np.max(
                np.stack(
                    [
                        np.linalg.norm(centers - view["camera_world"][:3, 3], axis=1)
                        for view in space.views.values()
                    ]
                ),
                axis=0,
            )
            enlarged = (
                radii
                + self.calibration_bound[0]
                + rotational_travel(lever, self.calibration_bound[1])
            )
            supports.append(space.query(centers, enlarged))
        unknown = any(s.unknown_in_envelope for s in supports)
        occupied = any(s.occupied for s in supports)
        clearance = None if unknown or occupied else min(s.collision_clearance_m for s in supports)
        return StaticWorkspaceSupport(clearance, unknown, occupied, self.source_clocks)
