"""Collision clearance on the convex hulls actually authored for PhysX."""

import numpy as np
from scipy.spatial import ConvexHull

from .preparation import PreparationError, require


def clear_opening(frame_hulls, panel_bounds, aperture=None):
    from itertools import product

    frame = np.concatenate(frame_hulls)
    lo, hi = np.asarray(panel_bounds).copy()
    lo[0], hi[0] = frame[:, 0].min(), frame[:, 0].max()
    lo[1:] += 0.001
    hi[1:] -= 0.001
    if aperture is not None:
        aperture = np.asarray(aperture, dtype=float)
        require(
            aperture.shape == (2, 2)
            and np.isfinite(aperture).all()
            and np.all(aperture[1] > aperture[0])
            and np.all(aperture[0] >= np.asarray(panel_bounds)[0, 1:] - 0.001)
            and np.all(aperture[1] <= np.asarray(panel_bounds)[1, 1:] + 0.001)
            and np.all(np.diff(aperture, axis=0)[0] >= 0.9 * np.diff(panel_bounds, axis=0)[0, 1:])
            and abs(aperture[:, 0].mean()) <= 0.001,
            "Clear aperture must be centered, inside the leaf envelope and cover at least 90%",
        )
        lo[1:], hi[1:] = aperture
    opening = Convex(np.array(list(product(*zip(lo, hi, strict=True)))))
    require(
        not any(overlap(opening, Convex(p)) for p in frame_hulls),
        "Frame collider obstructs the rectangular opening",
        category="geometry_or_recipe",
    )


def hull(points):
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    require(np.isfinite(points).all(), "Non-finite collision geometry", category="asset")
    try:
        result = ConvexHull(points)
    except Exception as exc:
        raise PreparationError("Degenerate collision geometry", category="asset") from exc
    return points[result.vertices]


def _directions(vectors):
    lengths = np.linalg.norm(vectors, axis=1)
    vectors = vectors[lengths > 1e-9] / lengths[lengths > 1e-9, None]
    # Opposite axes define the same interval test.
    for row in vectors:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return np.unique(np.round(vectors, 8), axis=0)


class Convex:
    def __init__(self, points):
        self.points = hull(points)
        shape = ConvexHull(self.points)
        self.normals = _directions(shape.equations[:, :3])
        tri = self.points[shape.simplices]
        self.edges = _directions(np.concatenate([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]]))

    def transformed(self, rotation, origin):
        result = object.__new__(Convex)
        result.points = (self.points - origin) @ rotation.T + origin
        result.normals = self.normals @ rotation.T
        result.edges = self.edges @ rotation.T
        return result


def overlap(a, b, tolerance=1e-6):
    if np.any(
        np.minimum(a.points.max(0), b.points.max(0)) - np.maximum(a.points.min(0), b.points.min(0))
        <= tolerance
    ):
        return False
    axes = np.concatenate(
        [a.normals, b.normals, _directions(np.cross(a.edges[:, None], b.edges).reshape(-1, 3))]
    )
    pa, pb = a.points @ axes.T, b.points @ axes.T
    return bool(
        np.all(np.minimum(pa.max(0), pb.max(0)) - np.maximum(pa.min(0), pb.min(0)) > tolerance)
    )


def partition_hulls(vertices, faces, cuts):
    """Clip every source triangle at recipe planes, then convexify each occupied cell.

    No surface is moved or discarded. Cuts are canonical X/Y/Z coordinates in
    meters. Unlike a voxel decomposition, this preserves narrow rebates exactly
    when their concavities are separated by the supplied planes.
    """
    require(isinstance(cuts, dict) and set(cuts) <= {"x", "y", "z"}, "Invalid partition axes")
    cells = [[p for p in np.asarray(vertices)[np.asarray(faces)]]]
    for key, values in cuts.items():
        require(
            isinstance(values, list)
            and len(values) <= 32
            and np.isfinite(values).all()
            and all(a < b for a, b in zip(values, values[1:], strict=False)),
            "Partition cuts must be finite, strictly increasing and bounded",
        )
        axis = "xyz".index(key)
        for value in values:
            divided = []
            for cell in cells:
                sides = [[], []]
                for polygon in cell:
                    polygon = polygon.copy()
                    distance = polygon[:, axis] - value
                    # USD float transforms leave nominally coplanar vertices a
                    # fraction of a micron apart; do not turn that into a wedge.
                    near = np.abs(distance) <= 1e-6
                    polygon[near, axis] = value
                    distance[near] = 0
                    if np.all(near):
                        # A face on a cut bounds only its material side. Adding it
                        # to both cells would create a wedge across a rebate.
                        normal = np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(0)
                        sides[0 if normal[axis] > 0 else 1].append(polygon)
                        continue
                    for side, sign in enumerate((-1, 1)):
                        clipped = []
                        for i, point in enumerate(polygon):
                            previous = polygon[i - 1]
                            a, b = distance[i - 1] * sign, distance[i] * sign
                            if (a < 0 < b) or (b < 0 < a):
                                clipped.append(previous + a / (a - b) * (point - previous))
                            if b >= 0:
                                clipped.append(point)
                        if len(clipped) >= 3:
                            sides[side].append(np.asarray(clipped))
                divided.extend(s for s in sides if s)
            cells = divided
            require(len(cells) <= 256, "Too many occupied convex partition cells")
    result = []
    for cell in cells:
        points = np.unique(np.concatenate(cell), axis=0)
        if len(points) >= 4 and np.linalg.matrix_rank(points - points.mean(0), tol=1e-9) == 3:
            result.append(hull(points))
    require(bool(result), "No volumetric collision partitions", category="geometry_or_recipe")
    return result


def surface_crossings(vertices_a, faces_a, vertices_b, faces_b, maximum=8):
    """Witness proper edge/triangle crossings on original surfaces, without voxels.

    Empty output is not proof of clearance: containment and coplanar contact are
    deliberately excluded. Returned points demonstrate intersecting source faces.
    """
    triangles = [np.asarray(vertices_a)[faces_a], np.asarray(vertices_b)[faces_b]]
    if np.any(
        np.minimum(triangles[0].max((0, 1)), triangles[1].max((0, 1)))
        < np.maximum(triangles[0].min((0, 1)), triangles[1].min((0, 1)))
    ):
        return np.empty((0, 3))
    witnesses = []
    for source, target in (triangles, triangles[::-1]):
        v0 = target[:, 0]
        e1, e2 = target[:, 1] - v0, target[:, 2] - v0
        target_lo, target_hi = target.min(1), target.max(1)
        for triangle in source:
            if not np.any(
                np.all((target_hi >= triangle.min(0)) & (target_lo <= triangle.max(0)), axis=1)
            ):
                continue
            for i in range(3):
                start, end = triangle[i], triangle[(i + 1) % 3]
                direction = end - start
                h = np.cross(direction, e2)
                determinant = np.einsum("ij,ij->i", e1, h)
                valid = np.abs(determinant) > 1e-14
                inverse = np.divide(1, determinant, out=np.zeros_like(determinant), where=valid)
                offset = start - v0
                u = inverse * np.einsum("ij,ij->i", offset, h)
                q = np.cross(offset, e1)
                v = inverse * (q @ direction)
                t = inverse * np.einsum("ij,ij->i", e2, q)
                valid &= (u > 1e-7) & (v > 1e-7) & (u + v < 1 - 1e-7)
                valid &= (t > 1e-7) & (t < 1 - 1e-7)
                for fraction in t[valid]:
                    point = start + fraction * direction
                    if all(np.linalg.norm(point - p) > 1e-6 for p in witnesses):
                        witnesses.append(point)
                    if len(witnesses) >= maximum:
                        return np.asarray(witnesses)
    return np.asarray(witnesses).reshape(-1, 3)


def opening_rotation(degrees, handedness):
    angle = np.deg2rad(degrees) * (1 if handedness == "left" else -1)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mechanical_limit(groups, hinge, handedness):
    fixed = [Convex(p) for p in groups["Frame"]]
    moving = [Convex(p) for name in ("Panel", "Handle") for p in groups[name]]
    fixed_lo = np.array([p.points.min(0) for p in fixed])
    fixed_hi = np.array([p.points.max(0) for p in fixed])

    def intersects(shapes):
        for a in shapes:
            candidates = np.flatnonzero(
                np.all(
                    np.minimum(a.points.max(0), fixed_hi) - np.maximum(a.points.min(0), fixed_lo)
                    > 1e-6,
                    axis=1,
                )
            )
            if any(overlap(a, fixed[i]) for i in candidates):
                return True
        return False

    require(
        not intersects(moving),
        "Closed door intersects its frame",
        category="geometry_or_recipe",
    )
    for degrees in np.arange(0.1, 270.01, 0.1):
        rotation = opening_rotation(degrees, handedness)
        if intersects(a.transformed(rotation, hinge) for a in moving):
            require(
                degrees > 0.3, "Opening is obstructed immediately", category="geometry_or_recipe"
            )
            return round(float(degrees - 0.2), 4)
    raise PreparationError(
        "No evidenced geometric stop within 270 degrees",
        status="unresolved",
        category="geometry_or_recipe",
    )
