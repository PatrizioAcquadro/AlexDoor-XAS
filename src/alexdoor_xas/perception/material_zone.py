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
