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

    def __init__(self, surface, config, *, cell_m=0.002):
        import cv2

        from alexdoor_xas.perception.geometry import contact_frame, project

        if not np.isfinite(cell_m) or cell_m <= 0:
            raise ValueError("invalid_material_cell")
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
                [[0, a, b] for a in (-cell_m / 2, cell_m / 2) for b in (-cell_m / 2, cell_m / 2)]
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
