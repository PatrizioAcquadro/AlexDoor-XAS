"""Metric RGB-D surface and rigid-motion fits; no simulator or asset dependencies."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from alexdoor_xas.action.frames import rot_z


def deproject(depth, pixels, intrinsics, camera):
    """Optical-axis depth in ROS optical coordinates, transformed into calibrated world."""
    pixels = np.asarray(pixels, dtype=float)
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(intrinsics).T
    return (rays * np.asarray(depth)[:, None]) @ camera[:3, :3].T + camera[:3, 3]


def project(points, intrinsics, camera):
    optical = (points - camera[:3, 3]) @ camera[:3, :3]
    image = optical @ intrinsics.T
    return image[:, :2] / image[:, 2, None], optical[:, 2]


def voxel_points(points, size, limit=40000):
    _, indices = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    indices = np.sort(indices)
    if len(indices) > limit:
        indices = indices[np.linspace(0, len(indices) - 1, limit, dtype=int)]
    return points[indices]


def normal_frame(normal):
    x = np.asarray(normal).copy()
    x[2] = 0
    if np.linalg.norm(x) < 1e-8:
        raise ValueError("Horizontal panel required for a vertical revolute axis")
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, 1.0])
    return np.column_stack((x, np.cross(z, x), z))


def plane_fit(points, tolerance, minimum, rng):
    """RANSAC followed by a robust SVD; retain the measured support and residual."""
    if len(points) < minimum:
        return None
    triples = points[rng.integers(len(points), size=(64, 3))]
    normals = np.cross(triples[:, 1] - triples[:, 0], triples[:, 2] - triples[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 1e-9
    if not keep.any():
        return None
    normals = normals[keep] / lengths[keep, None]
    offsets = np.sum(normals * triples[keep, 0], axis=1)
    distances = np.abs(sum(points[:, i, None] * normals[:, i] for i in range(3)) - offsets)
    index = (distances <= tolerance).sum(0).argmax()
    support = distances[:, index] <= tolerance
    if support.sum() < minimum:
        return None
    for _ in range(2):
        center = points[support].mean(0)
        _, _, vt = np.linalg.svd(points[support] - center, full_matrices=False)
        normal = vt[-1]
        residual = np.abs((points - center) @ normal)
        support = residual <= tolerance
        if support.sum() < minimum:
            return None
    return normal, float(center @ normal), support, float(np.quantile(residual[support], 0.95))


@dataclass
class Surface:
    points: np.ndarray
    normal: np.ndarray
    offset: float
    descriptor: np.ndarray
    anchors: np.ndarray
    features: np.ndarray
    residual_m: float
    support_fraction: float
    views: set = field(default_factory=set)
    edges: dict = field(default_factory=dict)
    edge_points: dict = field(default_factory=dict)
    extent_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    @property
    def basis(self):
        return normal_frame(self.normal)

    @property
    def bounds(self):
        local = self.points @ self.basis
        bounds = np.quantile(local, [0.002, 0.998], axis=0)
        if len(self.extent_points):
            dense_bounds = np.quantile(self.extent_points @ self.basis, [0.002, 0.998], axis=0)
            bounds[0] = np.minimum(bounds[0], dense_bounds[0])
            bounds[1] = np.maximum(bounds[1], dense_bounds[1])
        for axis, name in ((1, "width"), (2, "height")):
            for side in (0, 1):
                key = f"{name}_{side}"
                if key in self.edge_points:
                    bounds[side, axis] = np.median(self.edge_points[key] @ self.basis[:, axis])
        return bounds

    @property
    def center(self):
        return self.bounds.mean(0) @ self.basis.T


def extent_edges(
    points, normal, mask, sensor, tolerance, *, return_points=False, return_support=False
):
    """Keep measured RGB-D silhouette lines independently of interior point sampling.

    The SAM mask can include a frame: plane membership and a measured depth
    discontinuity establish the silhouette. Image clipping never establishes an edge.
    """
    from scipy.ndimage import binary_erosion, binary_opening

    basis = normal_frame(normal)
    depth = np.asarray(sensor["depth_m"]).squeeze(-1)
    valid = np.asarray(sensor["valid_depth"]).squeeze(-1)
    h, w = depth.shape
    v, u = np.nonzero(mask & valid & np.isfinite(depth) & (depth > 0))
    if not len(u):
        return ({}, {}, np.empty((0, 3))) if return_support else ({}, {}) if return_points else {}
    cloud = deproject(depth[v, u], np.c_[u, v], sensor["intrinsics"], sensor["camera_world"])
    offset = np.median(points @ normal)
    on_plane = abs(cloud @ normal - offset) <= tolerance
    support = np.zeros((h, w), bool)
    support[v[on_plane], u[on_plane]] = True
    # A floor/frame can intersect the fitted plane in a thin line. It is not
    # two-dimensional leaf support, even when its pixels fall inside a SAM mask.
    support = binary_opening(support, iterations=2)
    boundary = support & ~binary_erosion(support)
    v, u = np.nonzero(boundary)
    cloud = deproject(depth[v, u], np.c_[u, v], sensor["intrinsics"], sensor["camera_world"])
    local = cloud @ basis
    dense_extents = []
    for axis in (1, 2):
        for side in (0, 1):
            extreme = (min if side == 0 else max)(local[:, axis]) if len(local) else 0.0
            take = np.flatnonzero(abs(local[:, axis] - extreme) < 3 * tolerance)
            if len(take) > 512:
                take = take[np.linspace(0, len(take) - 1, 512, dtype=int)]
            dense_extents.append(cloud[take])
    extent_points = np.concatenate(dense_extents)
    edges, edge_points = {}, {}
    for axis, name in ((1, "width"), (2, "height")):
        for side, sign in ((0, -1), (1, 1)):
            outside, optical = project(
                cloud + sign * 0.025 * basis[:, axis],
                sensor["intrinsics"],
                sensor["camera_world"],
            )
            outside = np.rint(outside).astype(int)
            inside = (optical > 0) & (u >= 4) & (v >= 4) & (u < w - 4) & (v < h - 4)
            inside &= (outside >= 4).all(1) & (outside[:, 0] < w - 4) & (outside[:, 1] < h - 4)
            indices = np.flatnonzero(inside)
            uu, vv = outside[inside].T
            good = valid[vv, uu] & np.isfinite(depth[vv, uu]) & (depth[vv, uu] > 0)
            indices, uu, vv = indices[good], uu[good], vv[good]
            beyond = deproject(
                depth[vv, uu], np.c_[uu, vv], sensor["intrinsics"], sensor["camera_world"]
            )
            indices = indices[abs(beyond @ normal - offset) > 2 * tolerance]
            # Horizontal/vertical metric edge lines must span observable support.
            # Search the outermost supported line, rather than an area-weighted quantile.
            coordinates = local[indices, axis]
            bins = np.floor(coordinates / (2 * tolerance)).astype(int)
            other = 2 if axis == 1 else 1
            extreme = (min if side == 0 else max)(local[:, axis]) if len(local) else 0.0
            for cell in sorted(set(bins), reverse=side == 1):
                chosen = indices[abs(coordinates - (cell + 0.5) * 2 * tolerance) <= 2 * tolerance]
                if len(chosen) < 10 or np.ptp(local[chosen, other]) < 0.2:
                    continue
                center = np.median(local[chosen, axis])
                if abs(center - extreme) > 3 * tolerance:
                    continue
                chosen = chosen[abs(local[chosen, axis] - center) <= tolerance]
                if len(chosen) >= 10 and np.ptp(local[chosen, other]) >= 0.2:
                    key = f"{name}_{side}"
                    edges[key] = float(np.median(local[chosen, axis]))
                    edge_points[key] = cloud[chosen]
                    break
    return (
        (edges, edge_points, extent_points)
        if return_support
        else (edges, edge_points)
        if return_points
        else edges
    )


def dimensions_supported(surface, tolerance):
    bounds = surface.bounds
    return all(
        key in surface.edge_points
        and len(surface.edge_points[key]) >= 10
        and np.quantile(
            abs(surface.edge_points[key] @ surface.basis[:, axis] - bounds[side, axis]), 0.95
        )
        <= tolerance
        for axis, name in ((1, "width"), (2, "height"))
        for side in (0, 1)
        for key in (f"{name}_{side}",)
    )


def contact_frame(normal):
    x = np.asarray(normal) / np.linalg.norm(normal)
    z = np.array([0.0, 0.0, 1.0]) - x[2] * x
    z /= np.linalg.norm(z)
    return np.column_stack((x, np.cross(z, x), z))


def patch_anchors(cue, sensor, mask):
    tokens = np.frombuffer(cue["tokens"], np.float32).reshape(cue["token_shape"])
    grid = int(np.sqrt(len(tokens)))
    if grid * grid != len(tokens):
        raise ValueError("DINOv3 patch grid differs from the recorded mapping")
    yy, xx = np.mgrid[:grid, :grid]
    mapping = cue["pixel_mapping"]
    uv = np.c_[(xx.ravel() + 0.5) * 16, (yy.ravel() + 0.5) * 16]
    uv = (uv - [mapping["pad_x"], mapping["pad_y"]]) / mapping["scale"]
    pixels = np.rint(uv).astype(int)
    h, w = mask.shape
    keep = (pixels >= 0).all(1) & (pixels[:, 0] < w) & (pixels[:, 1] < h)
    indices = np.flatnonzero(keep)
    u, v = pixels[keep].T
    depth = np.asarray(sensor["depth_m"]).reshape(h, w)
    valid = np.asarray(sensor["valid_depth"]).reshape(h, w)
    good = mask[v, u] & valid[v, u] & np.isfinite(depth[v, u]) & (depth[v, u] > 0)
    indices, u, v = indices[good], u[good], v[good]
    anchors = deproject(depth[v, u], np.c_[u, v], sensor["intrinsics"], sensor["camera_world"])
    features = tokens[indices]
    descriptor = features.mean(0) if len(features) else np.zeros(tokens.shape[1])
    descriptor /= max(np.linalg.norm(descriptor), 1e-8)
    return anchors, features, descriptor


def surfaces(cue, sensor, config):
    from scipy.ndimage import binary_erosion

    h, w = cue["shape"]
    depth = np.asarray(sensor["depth_m"]).reshape(h, w)
    valid = np.asarray(sensor["valid_depth"]).reshape(h, w)
    rng = np.random.default_rng(int(sensor["frame"]))
    result = []
    for packed in cue["masks"]:
        mask = (
            np.unpackbits(np.frombuffer(packed, np.uint8), count=h * w).reshape(h, w).astype(bool)
        )
        interior = binary_erosion(mask, iterations=1)
        v, u = np.nonzero(interior & valid & np.isfinite(depth) & (depth > 0))
        if len(u) < config["min_points"]:
            continue
        take = np.linspace(0, len(u) - 1, min(len(u), config["max_points"]), dtype=int)
        u, v = u[take], v[take]
        cloud = deproject(depth[v, u], np.c_[u, v], sensor["intrinsics"], sensor["camera_world"])
        anchors, features, _ = patch_anchors(cue, sensor, interior)
        remaining = cloud
        for _ in range(3):
            fitted = plane_fit(remaining, config["plane_tolerance_m"], config["min_points"], rng)
            if fitted is None:
                break
            normal, offset, support, residual = fitted
            points = remaining[support]
            if normal @ (points.mean(0) - sensor["camera_world"][:3, 3]) < 0:
                normal, offset = -normal, -offset
            near = np.abs(anchors @ normal - offset) <= config["plane_tolerance_m"] * 2
            descriptor = features[near].mean(0) if near.any() else np.zeros(features.shape[1])
            descriptor /= max(np.linalg.norm(descriptor), 1e-8)
            edges, edge_points, extent_points = (
                extent_edges(
                    points, normal, mask, sensor, config["plane_tolerance_m"], return_support=True
                )
                if abs(normal[2]) < 0.3
                else ({}, {}, np.empty((0, 3)))
            )
            result.append(
                Surface(
                    points,
                    normal,
                    offset,
                    descriptor,
                    anchors[near],
                    features[near],
                    residual,
                    float(support.sum() / len(cloud)),
                    edges=edges,
                    edge_points=edge_points,
                    extent_points=extent_points,
                )
            )
            remaining = remaining[~support]
    return result


def similar_surface(a, b, config):
    cosine = float(a.normal @ b.normal)
    if cosine < np.cos(np.deg2rad(config["association_angle_deg"])):
        return False
    if abs(b.points.mean(0) @ a.normal - a.offset) > config["association_distance_m"]:
        return False
    aa, bb = a.bounds[:, 1:], np.quantile(b.points @ a.basis, [0.002, 0.998], axis=0)[:, 1:]
    lo, hi = np.maximum(aa[0], bb[0]), np.minimum(aa[1], bb[1])
    overlap = np.maximum(0, hi - lo)
    if np.prod(overlap) <= 0.1 * min(np.prod(aa[1] - aa[0]), np.prod(bb[1] - bb[0])):
        return False
    # Registered depth overlap can establish identity even when disjoint visual
    # regions (window, plain lower face) have different averaged DINO descriptors.
    local = b.points @ a.basis
    common = ((local[:, 1:] >= lo) & (local[:, 1:] <= hi)).all(1)
    if common.sum() < config["min_points"]:
        return False
    distance, index = cKDTree(a.points).query(b.points[common])
    normal_distance = abs((a.points[index] - b.points[common]) @ a.normal)
    return bool(
        np.mean(
            (distance <= config["association_distance_m"])
            & (normal_distance <= 2 * config["plane_tolerance_m"])
        )
        >= 0.6
    )


def fuse_surface(a, b, config, view):
    points = voxel_points(np.r_[a.points, b.points], config["voxel_m"])
    if len(a.points) + len(b.points):
        normal = a.normal * len(a.points) + b.normal * len(b.points)
        normal /= np.linalg.norm(normal)
    descriptor = a.descriptor + b.descriptor
    descriptor /= max(np.linalg.norm(descriptor), 1e-8)
    anchors = np.r_[a.anchors, b.anchors]
    features = np.r_[a.features, b.features]
    if len(anchors) > 2048:
        take = np.linspace(0, len(anchors) - 1, 2048, dtype=int)
        anchors, features = anchors[take], features[take]
    basis = normal_frame(normal)
    edge_points = dict(a.edge_points)
    for key, observed in b.edge_points.items():
        axis = 1 if key.startswith("width") else 2
        old = edge_points.get(key)
        if (
            old is None
            or (np.median(observed @ basis[:, axis]) - np.median(old @ basis[:, axis]))
            * (-1 if key.endswith("_0") else 1)
            > 0
        ):
            edge_points[key] = observed
    edges = {
        key: float(np.median(value @ basis[:, 1 if key.startswith("width") else 2]))
        for key, value in edge_points.items()
    }
    return Surface(
        points,
        normal,
        float(np.median(points @ normal)),
        descriptor,
        anchors,
        features,
        max(a.residual_m, b.residual_m),
        min(a.support_fraction, b.support_fraction),
        a.views | {view},
        edges,
        edge_points,
        voxel_points(np.r_[a.extent_points, b.extent_points], config["voxel_m"]),
    )


def track_surface(reference, sensor, config):
    """Validate projected surface support against *current* depth, never refresh by copying."""
    depth = np.asarray(sensor["depth_m"]).squeeze(-1)
    valid = np.asarray(sensor["valid_depth"]).squeeze(-1)
    h, w = depth.shape
    cloud = reference.points
    take = np.linspace(0, len(cloud) - 1, min(len(cloud), 800), dtype=int)
    pixels, optical = project(cloud[take], sensor["intrinsics"], sensor["camera_world"])
    pixels = np.rint(pixels).astype(int)
    good = (optical > 0) & (pixels >= 0).all(1) & (pixels[:, 0] < w) & (pixels[:, 1] < h)
    u, v = pixels[good].T
    good_depth = valid[v, u] & np.isfinite(depth[v, u]) & (depth[v, u] > 0)
    u, v = u[good_depth], v[good_depth]
    measured = deproject(depth[v, u], np.c_[u, v], sensor["intrinsics"], sensor["camera_world"])
    # Nearby expected plane points avoid jumping onto a distant wall after an occlusion.
    near = np.abs(measured @ reference.normal - reference.offset) < 0.03
    measured = measured[near]
    if len(measured) < config["min_points"]:
        return None
    fitted = plane_fit(
        measured,
        config["plane_tolerance_m"],
        config["min_points"],
        np.random.default_rng(int(sensor["frame"])),
    )
    if fitted is None:
        return None
    normal, offset, support, residual = fitted
    if normal @ reference.normal < 0:
        normal, offset = -normal, -offset
    return Surface(
        measured[support],
        normal,
        offset,
        reference.descriptor,
        reference.anchors,
        reference.features,
        residual,
        float(support.sum() / max(len(u), 1)),
        reference.views.copy(),
    )


def rigid_fit(source, target):
    if len(source) < 4 or len(source) != len(target):
        return None
    a, b = source.mean(0), target.mean(0)
    u, singular, vt = np.linalg.svd((source - a).T @ (target - b))
    if singular[1] < 1e-7:
        return None
    rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    translation = b - rotation @ a
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    return rotation, translation, float(np.quantile(residual, 0.95))


def matched_motion(reference, current, tolerance=0.01):
    """Mutual DINO matches propose motion; metric cloud alignment verifies it."""
    if min(len(reference.features), len(current.features)) < 4:
        return None
    similarity = reference.features @ current.features.T
    target = similarity.argmax(1)
    source = np.arange(len(target))
    keep = (similarity[source, target] > 0.8) & (similarity.argmax(0)[target] == source)
    source, target = reference.anchors[keep], current.anchors[target[keep]]
    if len(source) < 4:
        return None
    rng = np.random.default_rng(6100)
    best = None
    for _ in range(32):
        indices = rng.choice(len(source), 4, replace=False)
        motion = rigid_fit(source[indices], target[indices])
        if motion is None:
            continue
        rotation, translation, _ = motion
        errors = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        keep = errors < tolerance * 2
        if keep.sum() >= 4 and (best is None or keep.sum() > best.sum()):
            best = keep
    if best is None:
        return None
    fitted = rigid_fit(source[best], target[best])
    if fitted is None:
        return None
    rotation, translation, residual = fitted
    tree = cKDTree(current.points)
    points = reference.points[:: max(1, len(reference.points) // 2000)]
    for _ in range(3):
        moved = points @ rotation.T + translation
        distance, index = tree.query(moved)
        keep = distance < tolerance * 2
        if keep.sum() < 20:
            return None
        correction = rigid_fit(moved[keep], current.points[index[keep]])
        if correction is None:
            return None
        r, t, fit_residual = correction
        translation, rotation = r @ translation + t, r @ rotation
    if max(residual, fit_residual) > tolerance:
        return None
    return rotation, translation, max(residual, fit_residual)


def hinge_from_motion(motions, minimum_angle, floor_z):
    matrices, targets, residuals = [], [], []
    angles = []
    for rotation, translation, residual in motions:
        angle = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        if abs(angle) < minimum_angle or np.linalg.norm(rotation[:, 2] - [0, 0, 1]) > 0.05:
            continue
        matrices.append(np.eye(2) - rot_z(angle)[:2, :2])
        targets.append(translation[:2])
        residuals.append(residual)
        angles.append(abs(angle))
    if len(matrices) < 3:
        return None
    matrix, target = np.concatenate(matrices), np.concatenate(targets)
    hinge, _, _, singular = np.linalg.lstsq(matrix, target, rcond=None)
    if singular[-1] < 0.02 or singular[0] / singular[-1] > 100:
        return None
    residual = np.linalg.norm((matrix @ hinge - target).reshape(-1, 2), axis=1)
    uncertainty = (max(residuals) + np.quantile(residual, 0.95)) / (2 * np.sin(max(angles) / 2))
    return np.r_[hinge, floor_z], float(uncertainty)


def observed_thickness(panel, candidates, config):
    """Require a second attached parallel face or a measured side face; no nominal fill."""
    bounds = panel.bounds
    for side in candidates:
        if side is panel or side.descriptor @ panel.descriptor < config["descriptor_similarity"]:
            continue
        local = side.points @ panel.basis
        cosine = abs(side.normal @ panel.normal)
        lateral_overlap = min(local[:, 1].max(), bounds[1, 1]) - max(
            local[:, 1].min(), bounds[0, 1]
        )
        if cosine > 0.99 and lateral_overlap > 0.8 * (bounds[1, 1] - bounds[0, 1]):
            # A parallel jamb/wall can mimic a rear face: require joint measured motion first.
            continue
        if cosine < 0.2 and min(abs(np.median(local[:, 1]) - bounds[j, 1]) for j in (0, 1)) < 0.015:
            lo, hi = np.quantile(local[:, 0], [0.02, 0.98])
            front = float(np.median(panel.points @ panel.basis[:, 0]))
            if min(abs(lo - front), abs(hi - front)) < config["plane_tolerance_m"] * 2:
                thickness = float(hi - lo)
                if thickness > 2 * config["plane_tolerance_m"]:
                    return thickness, side.residual_m
    return None


def visible_hinge(panel, scene, config):
    """Look for vertically separated compact cylinder arcs at one panel edge."""
    local = scene @ panel.basis
    bounds = panel.bounds
    hypotheses = []
    for edge in bounds[:, 1]:
        near = (abs(local[:, 1] - edge) < 0.06) & (abs(local[:, 0] - bounds.mean(0)[0]) < 0.12)
        points = local[near]
        centers = []
        for z in np.arange(bounds[0, 2] + 0.2, bounds[1, 2] - 0.2, 0.1):
            xy = points[abs(points[:, 2] - z) < 0.025, :2]
            if len(xy) < 30:
                continue
            xy = np.unique(np.round(xy, 4), axis=0)
            if len(xy) < 20:
                continue
            center, _, _, singular = np.linalg.lstsq(
                np.c_[2 * xy, np.ones(len(xy))], np.sum(xy * xy, axis=1), rcond=None
            )
            if singular[-1] < 0.005:
                continue
            radius = np.linalg.norm(xy - center[:2], axis=1)
            r = float(np.median(radius))
            angles = np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0])
            if 0.003 <= r <= 0.03 and np.quantile(abs(radius - r), 0.95) < 0.002:
                if abs(np.mean(np.exp(1j * angles))) < 0.75:
                    centers.append((center[:2], z))
        for center, _z in centers:
            agrees = [(c, h) for c, h in centers if np.linalg.norm(c - center) < 0.004]
            if len(agrees) >= 2 and max(h for _, h in agrees) - min(h for _, h in agrees) > 0.2:
                xy = np.mean([c for c, _ in agrees], axis=0)
                hinge = np.r_[xy, config["floor_z_m"]] @ panel.basis.T
                hypotheses.append(hinge)
                break
    return hypotheses
