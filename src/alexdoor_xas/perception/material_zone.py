"""Panel-fixed zones, primitive uncertainty accounting and finite-finger slip queries."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.perception.admission import rotational_travel
from alexdoor_xas.perception.scan import footprint_support


def pose_matrix(pose):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = pose.rot, pose.origin
    return result


def object_frame(matrix):
    return ObjectFrame(np.array(matrix[:3, 3], copy=True), np.array(matrix[:3, :3], copy=True))


def transported_zone(camera_world, camera_from_map, map_from_object, object_from_zone):
    """W <- Ct <- M <- O <- Z; M remains the first optical camera reference."""
    return camera_world @ camera_from_map @ map_from_object @ object_from_zone


def normal_uncertainty_terms(sources, points, radius_m=0.0):
    """Transport validated normal-error primitives to points or complete spheres."""
    points = np.asarray(points, float)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or not len(points)
        or not np.isfinite(points).all()
        or not np.isfinite(radius_m)
        or radius_m < 0
    ):
        raise ValueError("invalid_normal_query_geometry")
    return {
        name: position
        + rotational_travel(np.linalg.norm(points - origin, axis=1).max() + radius_m, angle)
        for name, position, angle, origin in sources
    }


@dataclass(frozen=True)
class BoundSource:
    name: str
    position_m: float | None
    rotation_rad: float | None
    lever_m: float | None

    def displacement(self):
        values = (self.position_m, self.rotation_rad, self.lever_m)
        if any(v is None for v in values):
            return None
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("invalid_primitive_uncertainty")
        return self.position_m + rotational_travel(self.lever_m, self.rotation_rad)


@dataclass(frozen=True)
class ContactBudget:
    sources: tuple[BoundSource, ...]

    def __post_init__(self):
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)) or any(not name for name in names):
            raise ValueError("duplicate_uncertainty_source")

    @property
    def displacement_m(self):
        terms = [s.displacement() for s in self.sources]
        return None if any(t is None for t in terms) else sum(terms)

    @property
    def rotation_rad(self):
        terms = [s.rotation_rad for s in self.sources]
        return None if any(t is None for t in terms) else sum(terms)


@dataclass(frozen=True)
class SlipQuery:
    tangential_m: tuple[float, float]
    normal_separation_m: float
    orientation_rad: float
    compatible: bool
    finger_clearance_m: tuple[float | None, float | None]
    reason: str
    loaded_contact_admitted: bool = False


def query_slip(surface, initial_zone, zone_world, tool_world, distal_faces, budget, config):
    """Evaluate membership in the eroded measured region without moving its anchor.

    The existing footprint query retains mask holes, finite covers and local depth
    tolerance. Its spent local tolerance is deducted once from the composite
    budget; every other primitive and angular lever is still propagated.
    """
    relative = np.linalg.inv(zone_world) @ pose_matrix(tool_world)
    original_tool = initial_zone @ relative
    footprint = footprint_support(surface, object_frame(original_tool), distal_faces, config)
    displacement = budget.displacement_m
    local = [s for s in budget.sources if s.name == "local_geometry"]
    if len(local) != 1 or local[0].position_m is None:
        displacement = None
    elif displacement is not None:
        if local[0].position_m < config["plane_tolerance_m"]:
            raise ValueError("local_geometry_budget_omits_depth_tolerance")
        displacement -= config["plane_tolerance_m"]
    clearances = tuple(
        None if c is None or displacement is None else c - displacement
        for c in footprint.finger_clearance_m
    )
    compatible = footprint.supported and all(c is not None and c > 0 for c in clearances)
    return SlipQuery(
        tuple(relative[1:3, 3]),
        float(relative[0, 3]),
        float(Rotation.from_matrix(relative[:3, :3]).magnitude()),
        bool(compatible),
        clearances,
        "geometrically_compatible_slide"
        if compatible
        else "missing_relative_contact_bound"
        if displacement is None
        else footprint.reason
        if not footprint.supported
        else "outside_uncertainty_eroded_region",
    )


class MaterialCover:
    """Conservative union of measured finite planar cells across real views.

    A cell belongs only if its entire projected square, plus raster rounding,
    fits within one retained measured mask. The union joins overlapping views;
    it cannot fill an unobserved hole. Distances spend grid and depth tolerance.
    Pose, calibration, motion and stop errors belong to the action producer.
    """

    def __init__(self, surface, config, *, cell_m=0.002, normal_range_m=(0.0, 0.0)):
        import cv2

        from alexdoor_xas.perception.geometry import contact_frame, project

        if not np.isfinite(cell_m) or cell_m <= 0:
            raise ValueError("invalid_material_cell")
        normal_range = np.asarray(normal_range_m, float)
        if (
            normal_range.shape != (2,)
            or not np.isfinite(normal_range).all()
            or normal_range[0] > normal_range[1]
        ):
            raise ValueError("invalid_material_normal_range")
        self.surface, self.config, self.cell_m = surface, config, cell_m
        self.basis = contact_frame(surface.normal)
        points = surface.points @ self.basis
        self.lower = np.floor(points[:, 1:].min(0) / cell_m) * cell_m - 2 * cell_m
        upper = np.ceil(points[:, 1:].max(0) / cell_m) * cell_m + 2 * cell_m
        self.shape = tuple((np.ceil((upper - self.lower) / cell_m).astype(int) + 1)[::-1])
        if np.prod(self.shape) > 4_000_000:
            raise ValueError("excessive_observed_material_extent")
        v, u = np.indices(self.shape)
        local = np.column_stack(
            (
                np.full(u.size, surface.offset),
                self.lower[0] + u.ravel() * cell_m,
                self.lower[1] + v.ravel() * cell_m,
            )
        )
        world = local @ self.basis.T
        offsets = (
            np.array(
                [
                    [n, a, b]
                    for n in normal_range
                    for a in (-cell_m / 2, cell_m / 2)
                    for b in (-cell_m / 2, cell_m / 2)
                ]
            )
            @ self.basis.T
        )
        known = np.zeros(len(world), bool)
        for observation in surface.observations:
            pending = np.flatnonzero(~known)
            if not len(pending):
                break
            center, z = project(world[pending], observation.intrinsics, observation.camera_world)
            radius = np.zeros(len(pending))
            good = z > 0
            for offset in offsets:
                corner, cz = project(
                    world[pending] + offset, observation.intrinsics, observation.camera_world
                )
                good &= cz > 0
                radius = np.maximum(radius, np.linalg.norm(corner - center, axis=1))
            h, w = observation.shape
            pixel = np.rint(center).astype(int)
            good &= (pixel >= 0).all(1) & (pixel[:, 0] < w) & (pixel[:, 1] < h)
            distance = cv2.distanceTransform(
                np.pad(observation.mask().astype(np.uint8), 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )[1:-1, 1:-1]
            ids = np.flatnonzero(good)
            # Two pixel diagonals cover center rounding, measured pixel cells
            # and projected polygon rasterization. No interpolation of masks.
            covered = distance[pixel[ids, 1], pixel[ids, 0]] > radius[ids] + 2 * np.sqrt(2)
            known[pending[ids[covered]]] = True
        mask = known.reshape(self.shape).astype(np.uint8)
        distance = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[
            1:-1, 1:-1
        ]
        self.clearance = np.maximum(
            0, distance * cell_m - np.sqrt(2) * cell_m - config["plane_tolerance_m"]
        )
        self.clearance.setflags(write=False)

    @classmethod
    def from_depth(
        cls, surface, sensor, cue, semantic_id, config, *, generation, available_s, seed_mask=None
    ):
        """Observed interior RGB-D mesh, retaining relief and unresolved holes.

        Only a measured component connected to the reference plane is retained.
        The existing metric tolerance bounds local mesh interpolation. The finite
        root perimeter excludes adjacent frame material. Registration geometry
        stays independent: this mesh enlarges material coverage only.
        """
        from scipy.ndimage import binary_fill_holes, label

        from alexdoor_xas.perception.contracts import FieldSupport
        from alexdoor_xas.perception.geometry import contact_frame, deproject

        if not np.isfinite([sensor["time_s"], available_s]).all() or available_s < float(
            sensor["time_s"]
        ):
            raise ValueError("invalid_material_publication_clock")
        depth = np.asarray(sensor["depth_m"])[..., 0]
        valid = np.asarray(sensor["valid_depth"])[..., 0]
        height, width = depth.shape
        if tuple(cue["shape"]) != depth.shape or cue["ids"].count(semantic_id) != 1:
            raise ValueError("unavailable_material_semantic_identity")
        if seed_mask is None and not any(
            observation.frame == int(sensor["frame"])
            and observation.acquired_s == float(sensor["time_s"])
            and observation.semantic_id == semantic_id
            for observation in surface.observations
        ):
            raise ValueError("unsynchronized_material_depth_seed")
        index = cue["ids"].index(semantic_id)
        semantic = (
            np.unpackbits(np.frombuffer(cue["masks"][index], np.uint8), count=depth.size)
            .reshape(depth.shape)
            .astype(bool)
        )
        v, u = np.indices(depth.shape)
        safe_depth = np.where(valid & np.isfinite(depth) & (depth > 0), depth, 0)
        cloud = deproject(
            safe_depth.ravel(),
            np.c_[u.ravel(), v.ravel()],
            sensor["intrinsics"],
            sensor["camera_world"],
        ).reshape(height, width, 3)
        basis = contact_frame(surface.normal)
        local = cloud @ basis
        tolerance = config["plane_tolerance_m"]
        # Same measured coplanar-omission rule as surface extraction. A semantic
        # hole containing a different depth is not silently filled.
        semantic |= binary_fill_holes(semantic) & (abs(local[..., 0] - surface.offset) <= tolerance)
        lower, upper = surface.bounds
        allowed = valid & (safe_depth > 0) & semantic
        allowed &= (
            (local[..., 1:] >= lower[1:] + 2 * tolerance)
            & (local[..., 1:] <= upper[1:] - 2 * tolerance)
        ).all(2)
        pitch = safe_depth / min(sensor["intrinsics"][0, 0], sensor["intrinsics"][1, 1])
        horizontal_gap = np.linalg.norm(np.diff(cloud, axis=1), axis=2)
        vertical_gap = np.linalg.norm(np.diff(cloud, axis=0), axis=2)
        horizontal = allowed[:, :-1] & allowed[:, 1:]
        horizontal &= horizontal_gap <= np.maximum(
            2 * tolerance, 2 * np.maximum(pitch[:, :-1], pitch[:, 1:])
        )
        vertical = allowed[:-1] & allowed[1:]
        vertical &= vertical_gap <= np.maximum(2 * tolerance, 2 * np.maximum(pitch[:-1], pitch[1:]))
        graph = np.zeros((2 * height - 1, 2 * width - 1), bool)
        graph[::2, ::2], graph[::2, 1::2], graph[1::2, ::2] = allowed, horizontal, vertical
        labels, _ = label(graph)
        labels = labels[::2, ::2]
        if seed_mask is None:
            seed = np.zeros(depth.shape, bool)
            for observation in surface.observations:
                if observation.frame == int(sensor["frame"]):
                    seed |= observation.mask()
        else:
            seed = np.asarray(seed_mask)
            if (
                seed.dtype != bool
                or seed.shape != depth.shape
                or not (
                    (
                        ~seed
                        | (valid & semantic & (abs(local[..., 0] - surface.offset) <= tolerance))
                    ).all()
                )
            ):
                raise ValueError("unmeasured_static_material_reference")
        counts = np.bincount(labels[seed & allowed].ravel())
        if not len(counts):
            raise ValueError("missing_measured_material_seed")
        counts[0] = 0
        winners = np.flatnonzero(counts == counts.max())
        if len(winners) != 1 or counts[winners[0]] < config["min_points"]:
            raise ValueError("ambiguous_measured_material_component")
        region = allowed & (labels == winners[0])
        # A bounded shallow relief projects to a finite silhouette on the
        # reference plane. Keep the measured connected mask (including holes)
        # and charge its complete ray/plane parallax. This is a bounded relief
        # model, not a claim that the hidden interface is a measured flat plane.
        from dataclasses import replace

        from alexdoor_xas.perception.geometry import SurfaceObservation

        rays = cloud[region] - sensor["camera_world"][:3, 3]
        denominator = abs(rays @ surface.normal)
        if not len(rays) or (denominator <= tolerance).any():
            raise ValueError("unconditioned_material_view")
        points = local[region]
        parallax = float(
            np.max(
                (abs(points[:, 0] - surface.offset) + tolerance)
                * np.linalg.norm((rays @ basis)[:, 1:], axis=1)
                / denominator
            )
        )
        observation = SurfaceObservation(
            int(sensor["frame"]),
            float(sensor["time_s"]),
            float(available_s),
            index,
            depth.shape,
            np.packbits(region).tobytes(),
            np.array(sensor["intrinsics"], copy=True),
            np.array(sensor["camera_world"], copy=True),
            np.empty((0, 3)),
            {},
            semantic_id,
        )
        normal_range = (
            float(points[:, 0].min() - surface.offset - tolerance),
            float(points[:, 0].max() - surface.offset + tolerance),
        )
        # Project every corner of each complete normal-depth column. Local
        # parallax is paid by the same measured-mask coverage test, without
        # charging a distant grazing-view point to every cell of the surface.
        cover = cls(
            replace(surface, observations=(observation,)),
            config,
            cell_m=config.get("material_depth_cover", {}).get("cell_m", 0.002),
            normal_range_m=normal_range,
        )
        cover.normal_range_m = normal_range
        timestamp = float(sensor["time_s"])
        cover.support = FieldSupport(timestamp, available_s, timestamp, generation, tolerance, None)
        cover.diagnostics = dict(
            measured_pixels=int(region.sum()),
            reference_pixels=int((seed & region).sum()),
            silhouette_parallax_bound_m=parallax,
            normal_range_m=cover.normal_range_m,
        )
        return cover

    @classmethod
    def from_inspection(cls, surface, views, semantic_id, config, *, generation):
        """Union independently measured material cells from the frozen inspection.

        Each source capture keeps its original clock and semantic identity. A
        prior reference is seeded only by currently measured coplanar RGB-D.
        Coverage does not change the original registration plane or fill holes.
        """
        import cv2

        from alexdoor_xas.perception.contracts import FieldSupport
        from alexdoor_xas.perception.geometry import deproject

        covers = []
        failures = []
        capture_cutoff = max(o.acquired_s for o in surface.observations)
        publication_cutoff = max(o.available_s for o in surface.observations)
        for view in views:
            if (
                float(view["time_s"]) > capture_cutoff
                or float(view["available_s"]) > publication_cutoff
                or float(view["available_s"]) < float(view["time_s"])
            ):
                raise ValueError("material_inspection_after_initialization")
            cue = view.get("semantic")
            if cue is None or semantic_id not in cue["ids"]:
                failures.append("missing_inspection_semantic_identity")
                continue
            depth, valid = view["depth_m"][..., 0], view["valid_depth"][..., 0]
            v, u = np.nonzero(valid & np.isfinite(depth) & (depth > 0))
            points = deproject(depth[v, u], np.c_[u, v], view["intrinsics"], view["camera_world"])
            selected = (
                np.unpackbits(
                    np.frombuffer(cue["masks"][cue["ids"].index(semantic_id)], np.uint8),
                    count=depth.size,
                )
                .reshape(depth.shape)
                .astype(bool)
            )
            seed = np.zeros(depth.shape, bool)
            seed[v, u] = (
                abs(points @ surface.normal - surface.offset) <= config["plane_tolerance_m"]
            )
            seed &= selected
            try:
                covers.append(
                    cls.from_depth(
                        surface,
                        view,
                        cue,
                        semantic_id,
                        config,
                        generation=generation,
                        available_s=float(view["available_s"]),
                        seed_mask=seed,
                    )
                )
            except ValueError as error:
                failures.append(str(error))
        if not covers:
            raise ValueError("missing_measured_material_inspection")
        cover = cls.__new__(cls)
        cover.surface, cover.config = surface, config
        cover.basis, cover.cell_m = covers[0].basis, covers[0].cell_m
        cover.lower = np.min([c.lower for c in covers], axis=0)
        high = np.max([c.lower + (np.array(c.shape)[::-1] - 1) * c.cell_m for c in covers], axis=0)
        cover.shape = tuple(
            int(n) for n in (np.rint((high - cover.lower) / cover.cell_m).astype(int) + 1)[::-1]
        )
        if np.prod(cover.shape) > 4_000_000:
            raise ValueError("excessive_observed_material_extent")
        known = np.zeros(cover.shape, np.uint8)
        for source in covers:
            u, v = np.rint((source.lower - cover.lower) / cover.cell_m).astype(int)
            known[v : v + source.shape[0], u : u + source.shape[1]] |= (
                source.clearance > 0
            ).astype(np.uint8)
        distance = cv2.distanceTransform(np.pad(known, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[
            1:-1, 1:-1
        ]
        cover.clearance = np.maximum(0, distance * cover.cell_m - np.sqrt(2) * cover.cell_m)
        cover.clearance.setflags(write=False)
        cover.normal_range_m = (
            min(c.normal_range_m[0] for c in covers),
            max(c.normal_range_m[1] for c in covers),
        )
        cover.support = FieldSupport(
            min(c.support.acquired_s for c in covers),
            max(c.support.available_s for c in covers),
            max(c.support.supported_s for c in covers),
            generation,
            config["plane_tolerance_m"],
            None,
        )
        cover.diagnostics = dict(
            views=len(covers),
            unmeasured_views=failures,
            captures=[vars(c.support) for c in covers],
            normal_range_m=cover.normal_range_m,
        )
        return cover

    def projected_ball_margin(self, center, radius_m):
        """Finite tangent-circle support only; normal/contact evidence is separate."""
        center = np.asarray(center, float)
        if center.shape != (3,) or not np.isfinite(np.r_[center, radius_m]).all() or radius_m < 0:
            raise ValueError("invalid_projected_material_ball")
        local = (center @ self.basis)[1:]
        uv = np.rint((local - self.lower) / self.cell_m).astype(int)
        if (uv < 0).any() or (uv >= np.array(self.shape)[::-1]).any():
            return None
        rounding = np.linalg.norm(local - (self.lower + uv * self.cell_m))
        return float(self.clearance[uv[1], uv[0]] - rounding - radius_m)

    def query(self, pose, distal_faces):
        from alexdoor_xas.perception.scan import FootprintSupport

        margins = []
        for face in distal_faces:
            world = np.asarray(face) @ pose.rot.T + pose.origin
            if np.max(abs(world @ self.surface.normal - self.surface.offset)) > (
                2 * self.config["plane_tolerance_m"]
            ):
                margins.append(None)
                continue
            local = (world @ self.basis)[:, 1:]
            lower = np.floor((local.min(0) - self.lower) / self.cell_m).astype(int)
            upper = np.ceil((local.max(0) - self.lower) / self.cell_m).astype(int)
            if (lower < 0).any() or (upper >= np.array(self.shape)[::-1]).any():
                margins.append(None)
                continue
            margins.append(
                float(self.clearance[lower[1] : upper[1] + 1, lower[0] : upper[0] + 1].min())
            )
        supported = len(margins) == 2 and all(m is not None and m > 0 for m in margins)
        return FootprintSupport(
            tuple(margins),
            supported,
            "measured_material_union" if supported else "insufficient_measured_material_union",
        )


def contact_depth_supported(sensor, zone_world, tool_world, distal_faces, displacement_m):
    """Require current measured plane support across both full finite footprints."""
    if displacement_m is None or not np.isfinite(displacement_m) or displacement_m < 0:
        return False
    import cv2

    from alexdoor_xas.perception.geometry import deproject, project

    h, w = sensor["rgb"].shape[:2]
    normal, origin = zone_world[:3, 0], zone_world[:3, 3]
    for face in distal_faces:
        world = np.asarray(face) @ tool_world.rot.T + tool_world.origin
        pixels, z = project(world, sensor["intrinsics"], sensor["camera_world"])
        if (
            (z <= 0).any()
            or (pixels < 1).any()
            or (pixels[:, 0] >= w - 1).any()
            or (pixels[:, 1] >= h - 1).any()
        ):
            return False
        raster = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(
            raster, np.rint(cv2.convexHull(pixels.astype(np.float32))).astype(np.int32), 1
        )
        v, u = np.nonzero(raster)
        if not len(u):
            return False
        depth = sensor["depth_m"][v, u, 0]
        if not (sensor["valid_depth"][v, u, 0] & np.isfinite(depth) & (depth > 0)).all():
            return False
        measured = deproject(depth, np.c_[u, v], sensor["intrinsics"], sensor["camera_world"])
        if np.max(abs((measured - origin) @ normal)) > displacement_m:
            return False
    return True
