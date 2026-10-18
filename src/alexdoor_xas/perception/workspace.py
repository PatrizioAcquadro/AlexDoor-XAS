"""Observed static workcell calibration, separate from current dynamic evidence."""

import json
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import h5py
import numpy as np

from alexdoor_xas.perception.admission import rotational_travel
from alexdoor_xas.perception.geometry import integer_center_intrinsics, pixel_center_offset
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
        self.spaces = tuple(deepcopy(space) for space in spaces)
        self._free_covers = {}
        self._subcovers = {}
        self._nearby_free = []
        self.cache_hits = 0
        self.calibration_bound = np.asarray(calibration_bound, dtype=float)
        if (
            not self.spaces
            or self.calibration_bound.shape != (2,)
            or not np.isfinite(self.calibration_bound).all()
            or (self.calibration_bound < 0).any()
            or any(not space.views for space in self.spaces)
        ):
            raise ValueError("missing_observed_workspace_calibration")
        for space in self.spaces:
            space.index_static_points()
        self.source_clocks = tuple(
            tuple((float(v["time_s"]), float(v["available_s"])) for v in space.views.values())
            for space in self.spaces
        )

    @classmethod
    def from_recordings(cls, paths, config, calibration_bound):
        return cls(cls._recorded_spaces(paths, config, calibration_bound), calibration_bound)

    @staticmethod
    def _recorded_spaces(paths, config, calibration_bound):
        """Read validated rays; only the final owned workspace builds static indexes."""
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
                offset = pixel_center_offset(config, calibration)
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
                    sensor["intrinsics"] = integer_center_intrinsics(sensor["intrinsics"], offset)
                    depth = sensor["depth_m"]
                    near, far = calibration["depth_interval_m"]
                    if not np.array_equal(
                        sensor["valid_depth"], np.isfinite(depth) & (depth >= near) & (depth <= far)
                    ):
                        raise ValueError("workspace_depth_validity_mismatch")
                    space.add(sensor)
                spaces.append(space)
        return spaces

    def query_static(self, centers, radii, *, cover_ids=None, constructive=True):
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
        if cover_ids is None:
            return self._query_uncached(centers, radii)
        if len(cover_ids) != len(centers) or len(set(cover_ids)) != len(cover_ids):
            raise ValueError("invalid_static_cover_identity")
        margins = []
        for key, center, radius in zip(cover_ids, centers, radii, strict=True):
            previous = self._free_covers.get(key)
            margin = None
            if previous is not None:
                old_center, old_radius, old_margin = previous
                # The original radial certificate is a fixed free volume, not a
                # renewed observation. Include calibration-lever growth as well.
                growth = (np.linalg.norm(center - old_center) + radius - old_radius) * (
                    1 + 2 * np.sin(min(self.calibration_bound[1], np.pi) / 2)
                )
                margin = old_margin - growth
                if margin > 0:
                    self.cache_hits += 1
                else:
                    margin = None
            if margin is None:
                margin = self._nearby_margin(center, radius)
                if margin is not None:
                    self.cache_hits += 1
                    self._free_covers[key] = (center.copy(), float(radius), float(margin))
            if margin is None and constructive and key in self._subcovers:
                margin = self._subcover_margin(center, radius, key=key)
                if margin is not None:
                    self.cache_hits += 1
                    self._free_covers[key] = (center.copy(), float(radius), float(margin))
            if margin is None:
                support = self._query_uncached(center[None], np.array([radius]))
                if support.collision_clearance_m is None and not support.occupied and constructive:
                    # Raster padding can make a larger query inconclusive even
                    # when a nearby ray pyramid certifies its entire volume.
                    # Spend a smaller seed's measured radial margin on ALL of
                    # the original radius growth; never shrink the requested
                    # sweep or override observed occupancy.
                    for fraction in (0.95, 0.9, 0.75, 0.5):
                        seed_radius = radius * fraction
                        seed = self._query_uncached(center[None], np.array([seed_radius]))
                        if seed.collision_clearance_m is None:
                            continue
                        remaining = seed.collision_clearance_m - (radius - seed_radius) * (
                            1 + 2 * np.sin(min(self.calibration_bound[1], np.pi) / 2)
                        )
                        if remaining > 0:
                            support = StaticWorkspaceSupport(
                                remaining, False, False, self.source_clocks
                            )
                            break
                if support.collision_clearance_m is None and not support.occupied and constructive:
                    margin = self._subcover_margin(center, radius, key=key)
                    if margin is not None:
                        support = StaticWorkspaceSupport(margin, False, False, self.source_clocks)
                if support.collision_clearance_m is None:
                    return support
                margin = support.collision_clearance_m
                self._free_covers[key] = (center.copy(), float(radius), float(margin))
                self._remember_free(center, radius, margin)
            margins.append(margin)
        return StaticWorkspaceSupport(min(margins), False, False, self.source_clocks)

    def _remember_free(self, center, radius, margin):
        self._nearby_free.append((center.copy(), float(radius), float(margin)))
        del self._nearby_free[:-256]

    def _nearby_margin(self, center, radius):
        if not self._nearby_free:
            return None
        centers = np.array([row[0] for row in self._nearby_free])
        factor = 1 + 2 * np.sin(min(self.calibration_bound[1], np.pi) / 2)
        growth = factor * (np.linalg.norm(centers - center, axis=1) + radius)
        margins = np.array([factor * r + m for _, r, m in self._nearby_free]) - growth
        best = float(margins.max())
        return best if best > 0 else None

    def _subcover_margin(self, center, radius, *, key=None):
        """Cover the entire requested ball; retain reusable octree leaf proofs.

        Unit boxes partition the complete sphere before translation/scaling.
        Reuse pays ALL leaf-center/radius and calibration-lever growth. Unknown
        leaves are refined as before; exhausted/failed trees are never adopted.
        Static capture clocks and the original physical envelope are unchanged.
        """
        previous = self._subcovers.get(key) if key is not None else None
        pending = list(previous) if previous else [(np.full(3, -1.0), np.full(3, 1.0), 0, None)]
        margins, leaves = [], []
        queries = 0
        while pending:
            lower, upper, level, certificate = pending.pop()
            nearest = np.maximum(lower, np.minimum(np.zeros(3), upper))
            if np.linalg.norm(nearest) > 1:
                continue
            midpoint = center + radius * (lower + upper) / 2
            leaf_radius = radius * np.linalg.norm(upper - lower) / 2
            margin = None
            if certificate is not None:
                old_center, old_radius, old_margin = certificate
                growth = (np.linalg.norm(midpoint - old_center) + leaf_radius - old_radius) * (
                    1 + 2 * np.sin(min(self.calibration_bound[1], np.pi) / 2)
                )
                if old_margin > growth:
                    margin = old_margin - growth
            if margin is None:
                if queries >= 1024:
                    return None
                support = self._query_uncached(midpoint[None], np.array([leaf_radius]))
                queries += 1
                margin = support.collision_clearance_m
                if margin is not None:
                    self._remember_free(midpoint, leaf_radius, margin)
            if margin is not None:
                margins.append(margin)
                leaves.append(
                    (lower, upper, level, (midpoint.copy(), float(leaf_radius), float(margin)))
                )
                continue
            if level >= 6:
                return None
            middle = (lower + upper) / 2
            for octant in product((False, True), repeat=3):
                pending.append(
                    (
                        np.where(octant, middle, lower),
                        np.where(octant, upper, middle),
                        level + 1,
                        None,
                    )
                )
        if margins and key is not None:
            self._subcovers[key] = tuple(leaves)
        return min(margins) if margins else None

    def _query_uncached(self, centers, radii):
        supports = []
        for space in self.spaces:
            lever = (
                np.max(
                    np.stack(
                        [
                            np.linalg.norm(centers - view["camera_world"][:3, 3], axis=1)
                            for view in space.views.values()
                        ]
                    ),
                    axis=0,
                )
                + radii
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


class ObservedBackground(CommonWorkspace):
    """Historical free-ray volumes for static background ONLY.

    A current producer must independently exclude every forbidden moving volume.
    Source clocks belong to the named calibration records, never the current
    episode. Old occupied foreground is not an assertion of current occupancy.
    """

    def __init__(self, spaces, calibration_bound):
        super().__init__(spaces, calibration_bound)
        self.views_by_condition = []
        for space in self.spaces:
            views = []
            for key, view in reversed(tuple(space.views.items())):
                single = ObservedSpace(space.config)
                single.views = {key: view}
                single._static_points = {key: space._static_points[key]}
                views.append(single)
            self.views_by_condition.append(tuple(views))
        self.source_clocks = tuple(
            tuple(
                (
                    str(key[0]) if isinstance(key, tuple) else f"calibration-{i}",
                    float(view["time_s"]),
                    float(view["available_s"]),
                )
                for key, view in space.views.items()
            )
            for i, space in enumerate(self.spaces)
        )

    @classmethod
    def from_recording_groups(cls, groups, config, calibration_bound, sample_times_s):
        config = deepcopy(config)
        config["inspection"]["sample_times_s"] = list(sample_times_s)
        spaces = []
        for paths in groups:
            bank = ObservedSpace(config)
            for item in paths:
                path = item["path"] if isinstance(item, dict) else item
                record_config = deepcopy(config)
                if isinstance(item, dict):
                    record_config["inspection"]["sample_times_s"] = item["sample_times_s"]
                recorded = CommonWorkspace._recorded_spaces(
                    [path], record_config, calibration_bound
                )[0]
                bank.views.update(
                    {(str(Path(path).resolve()), key): view for key, view in recorded.views.items()}
                )
            spaces.append(bank)
        return cls(spaces, calibration_bound)

    def query_static(self, *args, **kwargs):
        raise ValueError("historical_background_requires_current_foreground_exclusion")

    def query_background(
        self, centers, radii, *, current_foreground_margin_m, cover_ids=None, constructive=True
    ):
        margin = current_foreground_margin_m
        if margin is None or not np.isfinite(margin) or margin <= 0:
            raise ValueError("missing_current_foreground_exclusion")
        result = super().query_static(
            centers, radii, cover_ids=cover_ids, constructive=constructive
        )
        return StaticWorkspaceSupport(
            None
            if result.collision_clearance_m is None
            else min(margin, result.collision_clearance_m),
            result.unknown_in_envelope,
            result.occupied,
            result.source_clocks,
        )

    def _query_uncached(self, centers, radii):
        margins = []
        for center, radius in zip(centers, radii, strict=True):
            for views in self.views_by_condition:
                best = None
                for observed in views:
                    view = next(iter(observed.views.values()))
                    lever = np.linalg.norm(center - view["camera_world"][:3, 3]) + radius
                    enlarged = (
                        radius
                        + self.calibration_bound[0]
                        + rotational_travel(lever, self.calibration_bound[1])
                    )
                    result = observed.query(center[None], np.array([enlarged]))
                    if result.collision_clearance_m is not None:
                        best = (
                            result.collision_clearance_m
                            if best is None
                            else max(best, result.collision_clearance_m)
                        )
                if best is None:
                    return StaticWorkspaceSupport(None, True, False, self.source_clocks)
                margins.append(best)
        return StaticWorkspaceSupport(min(margins), False, False, self.source_clocks)

    def _cached_pyramid_margin(self, vertices, radius, condition):
        bank = getattr(self, "_background_pyramids", {}).get(condition, [])
        if not bank:
            return None
        normals = np.array([row[0] for row in bank])
        offsets = np.array([row[1] for row in bank])
        cameras = np.array([row[2] for row in bank])
        distances = np.einsum("ki,pji->pkj", vertices, normals) + offsets[:, None, :]
        levers = np.linalg.norm(vertices[None, :, :] - cameras[:, None, :], axis=2).max(1) + radius
        errors = (
            radius
            + self.calibration_bound[0]
            + rotational_travel(levers, self.calibration_bound[1])
        )
        depth_bound = self.spaces[condition].config["plane_tolerance_m"]
        margins = distances.min((1, 2)) - errors - depth_bound
        best = float(margins.max())
        return best if best > 0 else None

    def query_cached_background(self, centers, radii, *, current_foreground_margin_m):
        """Reuse immutable complete ray pyramids, paying every new point/error lever."""
        guard = current_foreground_margin_m
        if guard is None or not np.isfinite(guard) or guard <= 0:
            raise ValueError("missing_current_foreground_exclusion")
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
        margins = []
        for center, radius in zip(centers, radii, strict=True):
            remembered = self._nearby_margin(center, radius)
            if remembered is not None:
                margins.append(remembered)
                continue
            for condition in range(len(self.spaces)):
                margin = self._cached_pyramid_margin(np.asarray(center)[None], radius, condition)
                if margin is None:
                    return StaticWorkspaceSupport(None, True, False, self.source_clocks)
                margins.append(margin)
        return StaticWorkspaceSupport(min(guard, min(margins)), False, False, self.source_clocks)

    def query_convex_background(self, vertices, radius, *, current_foreground_margin_m):
        """Cover the ENTIRE convex hull plus a radial uncertainty/response bound.

        Every pyramid plane is tested at every vertex, covering the full interior.
        Radius and model/calibration errors erode all planes. A separately proved
        CURRENT moving-foreground exclusion is mandatory for historical background.
        """
        guard = current_foreground_margin_m
        if guard is None or not np.isfinite(guard) or guard <= 0:
            raise ValueError("missing_current_foreground_exclusion")
        vertices = np.asarray(vertices, float)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not len(vertices)
            or not np.isfinite(vertices).all()
            or not np.isfinite(radius)
            or radius < 0
        ):
            return StaticWorkspaceSupport(None, True, False, self.source_clocks)
        if not hasattr(self, "_background_pyramids"):
            self._background_pyramids = {}
        margins = []
        signs = np.array(list(product((-1, 1), repeat=3)))
        for condition, space in enumerate(self.spaces):
            best = self._cached_pyramid_margin(vertices, radius, condition)
            if best is not None:
                margins.append(best)
                continue
            witness = None
            for view in space.views.values():
                camera, k = view["camera_world"], view["intrinsics"]
                optical = (vertices - camera[:3, 3]) @ camera[:3, :3]
                enlarged = (
                    radius
                    + self.calibration_bound[0]
                    + rotational_travel(
                        np.linalg.norm(optical, axis=1).max() + radius, self.calibration_bound[1]
                    )
                )
                if optical[:, 2].min() <= enlarged:
                    continue
                corners = (optical[:, None, :] + enlarged * signs).reshape(-1, 3)
                uv = corners @ k.T
                uv = uv[:, :2] / uv[:, 2, None]
                lo, hi = np.floor(uv.min(0)).astype(int) - 1, np.ceil(uv.max(0)).astype(int) + 1
                depth, valid = view["depth_m"][..., 0], view["valid_depth"][..., 0]
                h, w = depth.shape
                if (lo < 0).any() or hi[0] >= w or hi[1] >= h:
                    continue
                for padding in (0, 1, 2, 4, 16, 64):
                    low, up = np.maximum(lo - padding, 0), np.minimum(hi + padding, [w - 1, h - 1])
                    region = depth[low[1] : up[1] + 1, low[0] : up[0] + 1]
                    known = valid[low[1] : up[1] + 1, low[0] : up[0] + 1]
                    if not (known & np.isfinite(region) & (region > 0)).all():
                        continue
                    planes = np.array(
                        [
                            k[0] - low[0] * k[2],
                            up[0] * k[2] - k[0],
                            k[1] - low[1] * k[2],
                            up[1] * k[2] - k[1],
                            [0, 0, 1],
                            [0, 0, -1],
                        ]
                    )
                    norm = np.linalg.norm(planes, axis=1)
                    offsets = np.array([0, 0, 0, 0, 0, float(region.min())]) / norm
                    planes = planes / norm[:, None]
                    clearance = (
                        float((optical @ planes.T + offsets).min())
                        - enlarged
                        - space.config["plane_tolerance_m"]
                    )
                    if clearance > 0 and (best is None or clearance > best):
                        best = clearance
                        world_normals = planes @ camera[:3, :3].T
                        witness = (
                            world_normals,
                            offsets - world_normals @ camera[:3, 3],
                            camera[:3, 3].copy(),
                        )
            if best is None:
                return StaticWorkspaceSupport(None, True, False, self.source_clocks)
            bank = self._background_pyramids.setdefault(condition, [])
            bank.append(witness)
            del bank[:-64]
            margins.append(best)
        return StaticWorkspaceSupport(min(guard, min(margins)), False, False, self.source_clocks)
