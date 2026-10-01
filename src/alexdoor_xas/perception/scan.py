"""Static object evidence and local queries; no door assets, actions or truth inputs."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial import cKDTree

from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.perception.contracts import FieldSupport, HingeHypothesis
from alexdoor_xas.perception.geometry import (
    contact_frame,
    deproject,
    fuse_surface,
    project,
    similar_surface,
    voxel_points,
)


def surface_support(surface, generation, config):
    observations = surface.observations
    if not observations:
        return FieldSupport(0, 0, 0, generation, reason="unobserved_surface")
    latest = max(observations, key=lambda o: o.acquired_s)
    span = np.ptp(surface.points @ contact_frame(surface.normal), axis=0)[1:]
    error = config["plane_tolerance_m"] + surface.residual_m
    return FieldSupport(
        min(o.acquired_s for o in observations),
        max(o.available_s for o in observations),
        latest.acquired_s,
        generation,
        error,
        float(np.arctan2(2 * error, max(min(span), config["voxel_m"]))),
    )


def registered_support(surface, observation):
    """Previously measured material must reproject into current measured membership."""
    pixels, z = project(surface.points, observation.intrinsics, observation.camera_world)
    pixels = np.rint(pixels).astype(int)
    h, w = observation.shape
    inside = (z > 0) & (pixels >= 0).all(1) & (pixels[:, 0] < w) & (pixels[:, 1] < h)
    mask = observation.mask()
    mv, mu = np.nonzero(mask)
    if not len(mu):
        return False
    inside &= (pixels[:, 0] >= mu.min()) & (pixels[:, 0] <= mu.max())
    inside &= (pixels[:, 1] >= mv.min()) & (pixels[:, 1] <= mv.max())
    if inside.sum() < 12:
        return False
    u, v = pixels[inside].T
    return bool(mask[v, u].mean() >= 0.6)


def attached_inside(root, other, config):
    """An observed internal seam can connect relief; proximity alone cannot connect a frame."""
    if not root.observations or not other.observations:
        return False
    if abs(root.normal @ other.normal) > np.cos(np.deg2rad(config["association_angle_deg"])):
        return False  # parallel faces need an observed intervening side face
    local = other.points @ root.basis
    bounds = root.bounds
    margin = 2 * config["plane_tolerance_m"]
    # A connector at the outer perimeter may connect the leaf to the fixed frame.
    if not (
        (local[:, 1:] >= bounds[0, 1:] + margin) & (local[:, 1:] <= bounds[1, 1:] - margin)
    ).all():
        return False
    common = {o.frame for o in root.observations} & {o.frame for o in other.observations}
    for frame in sorted(common):
        a = next(o for o in root.observations if o.frame == frame)
        b = next(o for o in other.observations if o.frame == frame)
        if a.mask_index != b.mask_index:
            continue
        if not len(a.boundary_points) or not len(b.boundary_points):
            continue
        distance, _ = cKDTree(a.boundary_points).query(b.boundary_points)
        seam = b.boundary_points[distance <= 3 * config["plane_tolerance_m"]]
        if len(seam) >= 10 and np.linalg.norm(np.ptp(seam, axis=0)) > 4 * config["voxel_m"]:
            return True
    return False


def object_members(root, surfaces, config):
    members = [root]
    connectors = [s for s in surfaces if s is not root and attached_inside(root, s, config)]
    members.extend(connectors)
    for face in surfaces:
        if any(face is s for s in members):
            continue
        local = face.points @ root.basis
        if not ((local[:, 1:] >= root.bounds[0, 1:]) & (local[:, 1:] <= root.bounds[1, 1:])).all():
            continue
        # Joint seam support through a measured side face, not similar color/normal.
        for connector in connectors:
            frames = {o.frame for o in connector.observations} & {
                o.frame for o in face.observations
            }
            for frame in frames:
                a = next(o for o in connector.observations if o.frame == frame)
                b = next(o for o in face.observations if o.frame == frame)
                if (
                    a.mask_index != b.mask_index
                    or not len(a.boundary_points)
                    or not len(b.boundary_points)
                ):
                    continue
                d, _ = cKDTree(a.boundary_points).query(b.boundary_points)
                seam = b.boundary_points[d <= 3 * config["plane_tolerance_m"]]
                if len(seam) >= 10 and np.linalg.norm(np.ptp(seam, axis=0)) > 4 * config["voxel_m"]:
                    members.append(face)
                    break
            if any(face is s for s in members):
                break
    return tuple(members)


@dataclass(frozen=True)
class StaticHingeCandidate:
    candidate_id: str
    origin: np.ndarray
    direction: np.ndarray
    evidence: str
    reason: str
    hypothesis: HingeHypothesis | None = None
    fit_residual_m: float | None = None
    condition: float | None = None


def static_hinges(surface, scene, config, generation, scene_support=None):
    """Fit separated cylinder arcs; panel borders remain unsupported axis alternatives."""
    basis, bounds = surface.basis, surface.bounds
    local = scene @ basis
    output = []
    for side in (0, 1):
        edge = f"width_{side}"
        if edge not in surface.edge_points:
            continue
        y = float(np.median(surface.edge_points[edge] @ basis[:, 1]))
        assumed = np.array([bounds.mean(0)[0], y, config["floor_z_m"]]) @ basis.T
        output.append(
            StaticHingeCandidate(
                f"{surface.surface_id}:edge:{side}",
                assumed,
                basis[:, 2],
                "panel_edge",
                "panel_edge_is_not_physical_axis",
            )
        )
        points = local[
            (abs(local[:, 1] - y) < 0.06) & (abs(local[:, 0] - bounds.mean(0)[0]) < 0.12)
        ]
        arcs = []
        for z in np.arange(bounds[0, 2] + 0.2, bounds[1, 2] - 0.2, 0.1):
            xy = np.unique(np.round(points[abs(points[:, 2] - z) < 0.025, :2], 4), axis=0)
            if len(xy) < 30:
                continue
            mean = xy.mean(0)
            relative = xy - mean
            fitted, _, rank, _ = np.linalg.lstsq(
                np.c_[2 * relative, np.ones(len(xy))], np.sum(relative**2, axis=1), rcond=None
            )
            if rank != 3:
                continue
            center = fitted[:2] + mean
            radii = np.linalg.norm(xy - center, axis=1)
            radius = float(np.median(radii))
            residual = float(np.max(abs(radii - radius)))
            if not 0.003 <= radius <= 0.03 or residual > 0.002:
                continue
            directions = (xy - center) / radii[:, None]
            jacobian = np.c_[directions, np.ones(len(xy))]
            singular = np.linalg.svd(jacobian, compute_uv=False)
            condition = float(singular[0] / singular[-1])
            if condition > 10 or np.linalg.norm(directions.mean(0)) >= 0.75:
                continue
            # Worst coordinate sensitivity to bounded optical-depth/fit error.
            sensitivity = np.abs(np.linalg.pinv(jacobian)[:2]).sum(1)
            error = float(np.linalg.norm(sensitivity) * (config["plane_tolerance_m"] + residual))
            arcs.append((center, z, error, residual, condition))
        used = set()
        for index, arc in enumerate(arcs):
            if index in used:
                continue
            matches = [j for j, a in enumerate(arcs) if np.linalg.norm(a[0] - arc[0]) < 0.004]
            if len(matches) < 2 or np.ptp([arcs[j][1] for j in matches]) <= 0.2:
                continue
            used.update(matches)
            centers = np.array([arcs[j][0] for j in matches])
            xy = centers.mean(0)
            spread = float(np.linalg.norm(centers - xy, axis=1).max())
            uncertainty = max(arcs[j][2] for j in matches) + spread
            separation = np.ptp([arcs[j][1] for j in matches])
            angle = float(np.arctan2(2 * uncertainty, separation))
            origin = np.r_[xy, config["floor_z_m"]] @ basis.T
            support = surface_support(surface, generation, config)
            support = FieldSupport(
                min(support.acquired_s, scene_support.acquired_s)
                if scene_support
                else support.acquired_s,
                max(support.available_s, scene_support.available_s)
                if scene_support
                else support.available_s,
                max(support.supported_s, scene_support.supported_s)
                if scene_support
                else support.supported_s,
                generation,
                uncertainty,
                angle,
            )
            identifier = f"{surface.surface_id}:hardware:{side}:{len(output)}"
            hypothesis = HingeHypothesis(
                identifier, ObjectFrame(origin, basis.copy()), config["floor_z_m"], support
            )
            output.append(
                StaticHingeCandidate(
                    identifier,
                    origin,
                    basis[:, 2],
                    "separated_cylinder_arcs",
                    "observed_axis",
                    hypothesis,
                    max(arcs[j][3] for j in matches),
                    max(arcs[j][4] for j in matches),
                )
            )
    return tuple(output)


@dataclass(frozen=True)
class ContactPatch:
    patch_id: str
    surface_id: str
    pose: ObjectFrame
    support: FieldSupport
    ownership: str


@dataclass(frozen=True)
class FootprintSupport:
    finger_clearance_m: tuple[float | None, float | None]
    supported: bool
    reason: str


def footprint_support(surface, pose, distal_faces, config):
    """Cover both entire finite faces in observed rasters; never fill a hole with a hull."""
    if len(distal_faces) != 2:
        return FootprintSupport((None, None), False, "missing_two_finger_geometry")
    import cv2

    clearances = []
    for face in distal_faces:
        world = np.asarray(face) @ pose.rot.T + pose.origin
        if world.ndim != 2 or len(world) < 3 or not np.isfinite(world).all():
            return FootprintSupport((None, None), False, "invalid_finger_geometry")
        if np.max(abs(world @ surface.normal - surface.offset)) > 2 * config["plane_tolerance_m"]:
            clearances.append(None)
            continue
        best = None
        for observation in surface.observations:
            pixels, z = project(world, observation.intrinsics, observation.camera_world)
            h, w = observation.shape
            if (
                (z <= 0).any()
                or (pixels < 1).any()
                or (pixels[:, 0] >= w - 1).any()
                or (pixels[:, 1] >= h - 1).any()
            ):
                continue
            hull = cv2.convexHull(pixels.astype(np.float32))
            footprint = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(footprint, np.rint(hull).astype(np.int32), 1)
            if not footprint.any() or not observation.mask()[footprint.astype(bool)].all():
                continue
            local_face = world @ contact_frame(surface.normal)
            boundary = observation.boundary_points @ contact_frame(surface.normal)
            lo, hi = local_face[:, 1:].min(0), local_face[:, 1:].max(0)
            # Distance to the enclosing rectangle is a conservative footprint clearance.
            distance = np.linalg.norm(
                np.maximum(np.maximum(lo - boundary[:, 1:], boundary[:, 1:] - hi), 0), axis=1
            )
            pixel_error = float(
                z.max() / min(observation.intrinsics[0, 0], observation.intrinsics[1, 1])
            )
            clearance = max(0.0, float(distance.min()) - pixel_error - config["plane_tolerance_m"])
            best = clearance if best is None else max(best, clearance)
        clearances.append(best)
    supported = all(c is not None and c > 0 for c in clearances)
    return FootprintSupport(
        tuple(clearances),
        supported,
        "observed_footprint" if supported else "insufficient_observed_footprint",
    )


@dataclass(frozen=True)
class SpaceSupport:
    collision_clearance_m: float | None
    unknown_in_envelope: bool
    occupied: bool
    reason: str
    support: FieldSupport | None = None


class ObservedSpace:
    """Covering-ball queries over a bounded bank of calibrated RGB-D rays.

    The caller supplies a cover of the entire relevant sweep and stop envelope.
    Empty/invalid queries fail. This does not construct a motion or stopping bound.
    """

    def __init__(self, config, generation=0):
        self.config, self.generation, self.views = config, generation, {}

    def add(self, sensor, available_s=None):
        slot = int(
            np.searchsorted(self.config["inspection"]["sample_times_s"], float(sensor["time_s"]))
        )
        self.views[slot] = {
            k: np.array(sensor[k], copy=True)
            for k in ("depth_m", "valid_depth", "intrinsics", "camera_world", "time_s", "frame")
        }
        self.views[slot]["available_s"] = (
            float(sensor["time_s"]) if available_s is None else available_s
        )

    def query(self, centers, radii):
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
            return SpaceSupport(None, True, False, "invalid_envelope_cover")
        margins, unknown, occupied = [], False, False
        for center, radius in zip(centers, radii, strict=True):
            free = []
            for sensor in self.views.values():
                camera, k = sensor["camera_world"], sensor["intrinsics"]
                optical = (center - camera[:3, 3]) @ camera[:3, :3]
                if optical[2] <= radius:
                    continue
                corners = optical + radius * np.array(list(product((-1, 1), repeat=3)))
                projected = corners @ k.T
                uv = projected[:, :2] / projected[:, 2, None]
                lo, hi = np.floor(uv.min(0)).astype(int) - 1, np.ceil(uv.max(0)).astype(int) + 1
                depth, valid = sensor["depth_m"].squeeze(-1), sensor["valid_depth"].squeeze(-1)
                h, w = depth.shape
                if (lo < 0).any() or hi[0] >= w or hi[1] >= h:
                    continue
                region = depth[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
                good = (
                    valid[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1] & np.isfinite(region) & (region > 0)
                )
                if good.any():
                    v, u = np.nonzero(good)
                    cloud = deproject(region[v, u], np.c_[u + lo[0], v + lo[1]], k, camera)
                    occupied |= bool(
                        (
                            np.linalg.norm(cloud - center, axis=1)
                            <= radius + self.config["plane_tolerance_m"]
                        ).any()
                    )
                if good.all():
                    # A free depth gap alone is not radial clearance: lateral
                    # unseen/occupied rays may be closer. Certify a truncated
                    # ray pyramid, including its side planes, around the ball.
                    for padding in (0, 4, 16, 64):
                        lower = np.maximum(lo - padding, 0)
                        upper = np.minimum(hi + padding, [w - 1, h - 1])
                        region = depth[lower[1] : upper[1] + 1, lower[0] : upper[0] + 1]
                        known = valid[lower[1] : upper[1] + 1, lower[0] : upper[0] + 1]
                        if not (known & np.isfinite(region) & (region > 0)).all():
                            continue
                        planes = np.array(
                            [
                                k[0] - lower[0] * k[2],
                                upper[0] * k[2] - k[0],
                                k[1] - lower[1] * k[2],
                                upper[1] * k[2] - k[1],
                                [0, 0, 1],
                            ]
                        )
                        lateral = np.min((planes @ optical) / np.linalg.norm(planes, axis=1))
                        margin = float(
                            min(lateral, region.min() - optical[2])
                            - radius
                            - self.config["plane_tolerance_m"]
                        )
                        if margin > 0:
                            free.append(margin)
            if free:
                margins.append(max(free))
            else:
                unknown = True
        return SpaceSupport(
            min(margins) if margins and not unknown and not occupied else None,
            unknown,
            occupied,
            "observed_occupied_space"
            if occupied
            else "unknown_relevant_space"
            if unknown
            else "observed_free_cover",
            FieldSupport(
                min(float(s["time_s"]) for s in self.views.values()),
                max(float(s["available_s"]) for s in self.views.values()),
                max(float(s["time_s"]) for s in self.views.values()),
                self.generation,
                self.config["plane_tolerance_m"],
            )
            if self.views
            else None,
        )


@dataclass
class LeafObject:
    leaf_id: str
    surfaces: tuple
    root: object
    support: FieldSupport
    ownership: str
    hinges: tuple[StaticHingeCandidate, ...] = ()
    patches: tuple[ContactPatch, ...] = ()


@dataclass
class ScanState:
    generation: int
    objects: tuple[LeafObject, ...] = ()
    fixed_surfaces: tuple = ()
    unresolved_surfaces: tuple = ()
    reason: str = "acquiring_object"
    space: ObservedSpace | None = None


class ScanMemory:
    def __init__(self, config, generation):
        self.config, self.generation = config, generation
        self.surfaces, self.scene = [], np.empty((0, 3))
        self.ambiguous_associations = {}
        self.space = ObservedSpace(config, generation)
        self.scene_support = None
        self.state = ScanState(generation, space=self.space)
        self.next_id = 0

    def add(self, candidates, sensor, view, *, available_s=None):
        for surface in candidates:
            geometric = [
                i
                for i, old in enumerate(self.surfaces)
                if similar_surface(old, surface, self.config)
            ]
            matches = [
                i
                for i in geometric
                if surface.observations
                and registered_support(self.surfaces[i], surface.observations[-1])
            ]
            duplicate_geometry = len(matches) > 1 and all(
                similar_surface(self.surfaces[a], self.surfaces[b], self.config)
                and similar_surface(self.surfaces[b], self.surfaces[a], self.config)
                for a in matches
                for b in matches
                if a != b
            )
            if duplicate_geometry:
                # Consolidate redundant measured patches, not competing object ownership.
                i = matches[0]
                for j in matches[1:]:
                    self.surfaces[i] = fuse_surface(
                        self.surfaces[i], self.surfaces[j], self.config, view
                    )
                for j in reversed(matches[1:]):
                    del self.surfaces[j]
                matches = [i]
            if len(matches) == 1:
                i = matches[0]
                self.surfaces[i] = fuse_surface(self.surfaces[i], surface, self.config, view)
            elif geometric:
                # Retain ambiguous observations once per competing association, not a
                # fresh independent object for every semantic result.
                key = tuple(self.surfaces[i].surface_id for i in geometric)
                previous = self.ambiguous_associations.get(key)
                surface.surface_id = "unassigned:" + ":".join(key)
                self.ambiguous_associations[key] = (
                    fuse_surface(previous, surface, self.config, view)
                    if previous is not None
                    else surface
                )
            else:
                surface.surface_id = f"surface-{self.next_id}"
                self.next_id += 1
                surface.views.add(view)
                self.surfaces.append(surface)
        time_s = float(sensor["time_s"])
        available_s = (
            max((o.available_s for s in candidates for o in s.observations), default=time_s)
            if available_s is None
            else available_s
        )
        self.space.add(sensor, available_s)
        self.scene_support = FieldSupport(
            self.scene_support.acquired_s if self.scene_support else time_s,
            available_s,
            time_s,
            self.generation,
            self.config["plane_tolerance_m"],
        )
        depth, valid = sensor["depth_m"].squeeze(-1), sensor["valid_depth"].squeeze(-1)
        v, u = np.nonzero(valid[::2, ::2] & np.isfinite(depth[::2, ::2]) & (depth[::2, ::2] > 0))
        cloud = deproject(
            depth[v * 2, u * 2], np.c_[u * 2, v * 2], sensor["intrinsics"], sensor["camera_world"]
        )
        self.scene = voxel_points(np.r_[self.scene, cloud], self.config["voxel_m"], 60000)
        self.assemble()

    def assemble(self):
        roots = [
            s
            for s in self.surfaces
            if abs(s.normal[2]) < 0.3
            and len(s.views) >= 2
            and np.ptp(s.points @ s.basis, axis=0)[1] > 0.2
            and np.ptp(s.points @ s.basis, axis=0)[2] > 0.3
        ]
        groups = [(root, object_members(root, self.surfaces, self.config)) for root in roots]
        # An attached internal face is part of its enclosing object, not a rival leaf.
        groups = [
            (r, members)
            for r, members in groups
            if not any(r is not other and any(r is s for s in others) for other, others in groups)
        ]
        objects = []
        for root, members in groups:
            supported_perimeter = all(
                k in root.edge_points for k in ("width_0", "width_1")
            ) and any(k in root.edge_points for k in ("height_0", "height_1"))
            ownership = (
                "supported_leaf_candidate"
                if len(groups) == 1 and supported_perimeter
                else "ambiguous_leaf_candidate"
            )
            support = surface_support(root, self.generation, self.config)
            hinges = static_hinges(
                root, self.scene, self.config, self.generation, self.scene_support
            )
            patches = []
            for surface in members:
                surface.ownership = ownership
                # Actual observed points, selected deterministically; no dimension fraction.
                for index in np.linspace(
                    0, len(surface.points) - 1, min(9, len(surface.points)), dtype=int
                ):
                    patches.append(
                        ContactPatch(
                            f"{surface.surface_id}:patch:{index}",
                            surface.surface_id,
                            ObjectFrame(
                                surface.points[index].copy(), contact_frame(surface.normal)
                            ),
                            surface_support(surface, self.generation, self.config),
                            ownership,
                        )
                    )
            objects.append(
                LeafObject(
                    f"leaf-{root.surface_id}",
                    members,
                    root,
                    support,
                    ownership,
                    hinges,
                    tuple(patches),
                )
            )
        fixed, unresolved = [], []
        assigned = {id(s) for o in objects for s in o.surfaces}
        for surface in self.surfaces:
            if id(surface) in assigned:
                continue
            outside = bool(objects)
            for obj in objects:
                local = (surface.points @ obj.root.basis)[:, 1:]
                inside = (
                    (local >= obj.root.bounds[0, 1:]) & (local <= obj.root.bounds[1, 1:])
                ).all(1)
                outside &= (
                    all(
                        k in obj.root.edge_points
                        for k in ("width_0", "width_1", "height_0", "height_1")
                    )
                    and not inside.any()
                )
            surface.ownership = "observed_surrounding_support" if outside else "unresolved"
            (fixed if outside else unresolved).append(surface)
        unresolved.extend(self.ambiguous_associations.values())
        reason = (
            "supported_static_candidate"
            if len(objects) == 1
            and objects[0].ownership == "supported_leaf_candidate"
            and not unresolved
            else "ambiguous_object_ownership"
            if objects
            else "acquiring_object"
        )
        self.state = ScanState(
            self.generation, tuple(objects), tuple(fixed), tuple(unresolved), reason, self.space
        )
