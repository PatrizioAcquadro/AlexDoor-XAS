"""Collision clearance on the convex hulls actually authored for PhysX."""

import numpy as np
from scipy.spatial import ConvexHull

from .preparation import PreparationError, require


def clear_opening(frame_hulls, panel_bounds):
    from itertools import product

    frame = np.concatenate(frame_hulls)
    lo, hi = np.asarray(panel_bounds).copy()
    lo[0], hi[0] = frame[:, 0].min(), frame[:, 0].max()
    lo[1:] += 0.001
    hi[1:] -= 0.001
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


def overlap(a, b, tolerance=1e-7):
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


def opening_rotation(degrees, handedness):
    angle = np.deg2rad(degrees) * (1 if handedness == "left" else -1)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mechanical_limit(groups, hinge, handedness):
    fixed = [Convex(p) for p in groups["Frame"]]
    moving = [Convex(p) for name in ("Panel", "Handle") for p in groups[name]]
    require(
        not any(overlap(a, b) for a in moving for b in fixed),
        "Closed door intersects its frame",
        category="geometry_or_recipe",
    )
    for degrees in np.arange(0.1, 270.01, 0.1):
        rotation = opening_rotation(degrees, handedness)
        if any(overlap(a.transformed(rotation, hinge), b) for a in moving for b in fixed):
            require(
                degrees > 0.3, "Opening is obstructed immediately", category="geometry_or_recipe"
            )
            return round(float(degrees - 0.2), 4)
    raise PreparationError(
        "No evidenced geometric stop within 270 degrees",
        status="unresolved",
        category="geometry_or_recipe",
    )
