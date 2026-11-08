"""Static object evidence and local queries; no door assets, actions or truth inputs."""

from dataclasses import dataclass, replace
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


def observed_seam(first, second, config):
    """Keep every component/box observation belonging to the same captured mask."""
    by_mask = {}
    for observation in first.observations:
        by_mask.setdefault((observation.frame, observation.mask_index), []).append(observation)
    for b in second.observations:
        for a in by_mask.get((b.frame, b.mask_index), ()):
            if not len(a.boundary_points) or not len(b.boundary_points):
                continue
            distance, _ = cKDTree(a.boundary_points).query(b.boundary_points)
            seam = b.boundary_points[distance <= 3 * config["plane_tolerance_m"]]
            if len(seam) >= 10 and np.linalg.norm(np.ptp(seam, axis=0)) > 4 * config["voxel_m"]:
                return True
    return False


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
    return observed_seam(root, other, config)


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
            if observed_seam(connector, face, config):
                members.append(face)
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
    # The scene necessarily contains the leaf itself. Its planar points must
    # not dominate the hardware-arc fit or create a spurious straight arc.
    hardware = scene[abs(scene @ surface.normal - surface.offset) > 2 * config["plane_tolerance_m"]]
    local = hardware @ basis
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
    """Check two finite contact covers; observed holes remain holes, ownership is separate."""
    if len(distal_faces) != 2:
        return FootprintSupport((None, None), False, "missing_two_finger_geometry")
    import cv2

    clearances = []
    for face in distal_faces:
        face = np.asarray(face)
        if face.ndim != 2 or face.shape[1:] != (3,) or len(face) < 3 or not np.isfinite(face).all():
            return FootprintSupport((None, None), False, "invalid_finger_geometry")
        if np.linalg.matrix_rank(face - face.mean(0)) < 2:
            return FootprintSupport((None, None), False, "degenerate_finger_face")
        world = face @ pose.rot.T + pose.origin
        if not np.isfinite(world).all():
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
            hull = np.rint(cv2.convexHull(pixels.astype(np.float32))).astype(np.int32)
            lower, upper = hull.reshape(-1, 2).min(0), hull.reshape(-1, 2).max(0)
            footprint = np.zeros((upper[1] - lower[1] + 1, upper[0] - lower[0] + 1), np.uint8)
            cv2.fillConvexPoly(footprint, hull - lower, 1)
            rows, columns = np.nonzero(footprint)
            # Query exactly the same rasterized pixels directly in immutable
            # packed evidence. Avoid decoding/allocating a complete camera image
            # for each tiny footprint in every retained inspection view.
            packed = np.frombuffer(observation.support_mask, np.uint8)
            indices = (rows + lower[1]) * w + columns + lower[0]
            if (
                not len(indices)
                or len(packed) * 8 < h * w
                or not ((packed[indices // 8] >> (7 - indices % 8)) & 1).all()
            ):
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


def observed_material_surface(reference, surfaces, semantic_id, config):
    """Retain all measured coplanar support for one observed material identity.

    Each footprint still requires complete pixels from an actual observation;
    geometric fusion never fills holes or invents support between visible parts.
    The native seed surface remains unchanged.
    """
    result = reference
    seen = {(o.frame, o.mask_index, o.support_mask) for o in reference.observations}
    for surface in surfaces:
        if (
            surface.normal @ reference.normal < np.cos(np.deg2rad(config["association_angle_deg"]))
            or abs(surface.offset - reference.offset) > 2 * config["plane_tolerance_m"]
            # Similar normals/offsets alone admit long tilted relief faces.
            # Every measured point must belong to the reference plane envelope.
            or np.max(abs(surface.points @ reference.normal - reference.offset))
            > 2 * config["plane_tolerance_m"]
            or not surface.observations
            or any(o.semantic_id != semantic_id for o in surface.observations)
        ):
            continue
        observations = tuple(
            o for o in surface.observations if (o.frame, o.mask_index, o.support_mask) not in seen
        )
        if not observations:
            continue
        seen.update((o.frame, o.mask_index, o.support_mask) for o in observations)
        points = voxel_points(np.r_[result.points, surface.points], config["voxel_m"])
        # This is a coverage atlas, not a new registration reference. Averaging
        # nearby relief normals rotates the original material frame and can
        # invalidate the independently measured normal. Keep its plane intact.
        result = replace(
            result,
            points=points,
            observations=tuple(result.observations) + observations,
            residual_m=max(
                result.residual_m,
                float(np.max(abs(points @ reference.normal - reference.offset))),
            ),
        )
    return result


def supported_material_zone(surface, distal_faces, config, *, support_filter=None):
    """Choose a measured reference with maximum two-finger boundary clearance.

    This is a material-coordinate selection, not a robot goal or ownership proof.
    Holes and missing support retain their original footprint-query semantics.
    """
    best, score = None, -np.inf
    rotation = contact_frame(surface.normal)
    indices = np.linspace(0, len(surface.points) - 1, min(128, len(surface.points)), dtype=int)
    for index in indices:
        pose = ObjectFrame(surface.points[index].copy(), rotation)
        support = footprint_support(surface, pose, distal_faces, config)
        if support.supported and min(support.finger_clearance_m) > score:
            if support_filter is not None and not support_filter(pose):
                continue
            best = pose
            score = min(support.finger_clearance_m)
    return best


def static_material_witnesses(surfaces, zone, distal_faces, config, semantic_id, error_m):
    """Retain same-identity finite support from geometrically separated views.

    These witness one selected local rigid material region, not a physical hinge
    or the ownership of every part inside a semantic door mask.
    """
    from dataclasses import replace

    if not np.isfinite(error_m) or error_m < config["plane_tolerance_m"]:
        raise ValueError("missing_static_material_error_bound")
    observations = []
    for surface in surfaces:
        if surface.normal @ zone.rot[:, 0] < np.cos(np.deg2rad(config["association_angle_deg"])):
            continue
        if abs(zone.origin @ surface.normal - surface.offset) > 2 * config["plane_tolerance_m"]:
            continue
        for observation in surface.observations:
            if observation.semantic_id != semantic_id:
                continue
            one = replace(surface, observations=[observation])
            support = footprint_support(one, zone, distal_faces, config)
            if support.supported:
                observations.append((observation, support))
    witnesses = []
    for observation, support in sorted(observations, key=lambda pair: pair[0].acquired_s):
        origin = observation.camera_world[:3, 3]
        if all(
            np.linalg.norm(origin - np.asarray(w["camera_origin_m"])) > 2 * error_m
            for w in witnesses
        ):
            witnesses.append(
                dict(
                    frame=int(observation.frame),
                    acquired_s=float(observation.acquired_s),
                    available_s=float(observation.available_s),
                    semantic_id=int(semantic_id),
                    camera_origin_m=origin.tolist(),
                    finger_clearance_m=support.finger_clearance_m,
                    position_error_m=float(error_m),
                )
            )
    return tuple(witnesses)


@dataclass(frozen=True)
class SpaceSupport:
    collision_clearance_m: float | None
    unknown_in_envelope: bool
    occupied: bool
    reason: str
    support: FieldSupport | None = None


class _DepthRectangles:
    """Exact known-pixel counts over immutable depth rectangles."""

    def __init__(self, depth, valid):
        self.depth = depth
        invalid = ~valid | ~np.isfinite(depth) | (depth <= 0)
        dtype = np.int32 if depth.size < np.iinfo(np.int32).max else np.int64
        self.invalid = (
            np.pad(invalid, ((1, 0), (1, 0))).cumsum(0, dtype=dtype).cumsum(1, dtype=dtype)
        )

    def query(self, lower, upper, *, minimum=True):
        x0, y0 = lower
        x1, y1 = upper
        x1, y1 = x1 + 1, y1 + 1
        counts = self.invalid
        bad = int(counts[y1, x1] - counts[y0, x1] - counts[y1, x0] + counts[y0, x0])
        count = (x1 - x0) * (y1 - y0) - bad
        value = self.depth[y0:y1, x0:x1].min() if minimum and not bad else None
        return count, value


class ObservedSpace:
    """Covering-ball queries over a bounded bank of calibrated RGB-D rays.

    The caller supplies a cover of the entire relevant sweep and stop envelope.
    Empty/invalid queries fail. This does not construct a motion or stopping bound.
    """

    def __init__(self, config, generation=0):
        self.config, self.generation, self.views = config, generation, {}

    def index_static_points(self):
        """Index owned immutable returns with at most four independent builders."""
        from concurrent.futures import ThreadPoolExecutor

        from scipy.spatial import cKDTree

        self._static_points = {}
        self._depth_rectangles = {}
        if not self.views:
            return

        def build(item):
            key, view = item
            depth, valid = view["depth_m"][..., 0], view["valid_depth"][..., 0]
            v, u = np.nonzero(valid & np.isfinite(depth) & (depth > 0))
            cloud = deproject(depth[v, u], np.c_[u, v], view["intrinsics"], view["camera_world"])
            return key, (cKDTree(cloud), np.c_[u, v]), _DepthRectangles(depth, valid)

        with ThreadPoolExecutor(max_workers=min(4, len(self.views))) as executor:
            indexed = list(executor.map(build, self.views.items()))
        self._static_points = {key: points for key, points, _ in indexed}
        self._depth_rectangles = {key: rectangles for key, _, rectangles in indexed}
        for view in self.views.values():
            for value in view.values():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)

    def add(self, sensor, available_s=None):
        if hasattr(self, "_static_points"):
            del self._static_points
            del self._depth_rectangles
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
            for view_key, sensor in self.views.items():
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
                rectangles = getattr(self, "_depth_rectangles", {}).get(view_key)
                if rectangles is None:
                    good = (
                        valid[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
                        & np.isfinite(region)
                        & (region > 0)
                    )
                    any_good, all_good = good.any(), good.all()
                else:
                    count, _ = rectangles.query(lo, hi, minimum=False)
                    any_good, all_good = count > 0, count == region.size
                # Occupancy is an OR over all views and balls. Once established,
                # only free/unknown support still needs evaluation below.
                if any_good and not occupied:
                    indexed = getattr(self, "_static_points", {}).get(view_key)
                    if indexed is not None:
                        tree, pixels = indexed
                        distance = radius + self.config["plane_tolerance_m"]
                        matches = tree.query_ball_point(center, np.nextafter(distance, np.inf))
                        matched = pixels[matches]
                        exact = np.linalg.norm(tree.data[matches] - center, axis=1) <= distance
                        occupied |= bool((exact & ((matched >= lo) & (matched <= hi)).all(1)).any())
                    else:
                        v, u = np.nonzero(good)
                        cloud = deproject(region[v, u], np.c_[u + lo[0], v + lo[1]], k, camera)
                        occupied |= bool(
                            (
                                np.linalg.norm(cloud - center, axis=1)
                                <= radius + self.config["plane_tolerance_m"]
                            ).any()
                        )
                if all_good:
                    # A free depth gap alone is not radial clearance: lateral
                    # unseen/occupied rays may be closer. Certify a truncated
                    # ray pyramid, including its side planes, around the ball.
                    for padding in (0, 4, 16, 64):
                        lower = np.maximum(lo - padding, 0)
                        upper = np.minimum(hi + padding, [w - 1, h - 1])
                        region = depth[lower[1] : upper[1] + 1, lower[0] : upper[0] + 1]
                        if rectangles is None:
                            known = valid[lower[1] : upper[1] + 1, lower[0] : upper[0] + 1]
                            if not (known & np.isfinite(region) & (region > 0)).all():
                                continue
                            minimum = region.min()
                        else:
                            count, minimum = rectangles.query(lower, upper)
                            if count != region.size:
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
                            min(lateral, minimum - optical[2])
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

    @property
    def observed_surfaces(self):
        """All retained measurements; global associations remain unresolved."""
        return (*self.surfaces, *self.ambiguous_associations.values())

    def add(self, candidates, sensor, view, *, available_s=None):
        for surface in candidates:
            geometric = [
                i
                for i, old in enumerate(self.surfaces)
                if similar_surface(old, surface, self.config)
                and (
                    not surface.observations
                    or not old.observations
                    or surface.observations[-1].semantic_id == old.observations[-1].semantic_id
                )
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
                # One measured separating border is sufficient. Missing bottom
                # support cannot invalidate exclusion beyond a measured side.
                outside &= any(
                    (
                        (local[:, 0 if key.startswith("width") else 1] - coordinate)
                        * (-1 if key.endswith("_0") else 1)
                        > 3 * self.config["plane_tolerance_m"]
                    ).all()
                    for key, coordinate in obj.root.edges.items()
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
