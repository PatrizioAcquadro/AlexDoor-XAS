"""Relative rigid-finger order for a separately calibrated unilateral planar interface."""

import numpy as np


def distal_body_order(triangles, face, normal, cone_rad, band_m):
    """Lower-bound how far every non-distal mesh point trails a distal support.

    The front support must consist of actual coplanar mesh extrema, never a
    projected footprint cover. A finite line of extrema is also supported.
    Mesh triangles are clipped at the band boundary; complete triangle interiors
    follow by convexity. For each retained vertex a real point in the finite face
    supplies a separating support for EVERY normal in the supplied cone. This
    geometric ordering alone does not establish material identity, finite support,
    contact, a penetration bound, background clearance, or stop qualification.
    """
    points, face, normal = (np.asarray(v, float) for v in (triangles, face, normal))
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) % 3
        or face.ndim != 2
        or face.shape[1:] != (3,)
        or len(face) < 2
        or normal.shape != (3,)
        or not np.isfinite(np.r_[points.ravel(), face.ravel(), normal, cone_rad, band_m]).all()
        or not np.isclose(np.linalg.norm(normal), 1)
        or not 0 <= cone_rad < np.pi / 2
        or band_m <= 0
    ):
        raise ValueError("invalid_finite_distal_order_geometry")
    cutoff = face[:, 0].max() - band_m
    edges = points.reshape(-1, 3, 3)[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2, 3)
    a, b = edges[:, 0], edges[:, 1]
    crossing = (a[:, 0] < cutoff) != (b[:, 0] < cutoff)
    a, b = a[crossing], b[crossing]
    intersections = a + (b - a) * ((cutoff - a[:, 0]) / (b[:, 0] - a[:, 0]))[:, None]
    body = np.unique(np.r_[points[points[:, 0] <= cutoff], intersections], axis=0)
    if not len(body):
        raise ValueError("missing_non_distal_finger_body")
    # Every chosen support is an actual point in the convex distal mesh.
    # Endpoint segments also cover a numerically tilted line of CAD extrema;
    # never project an artificial finite footprint onto the leading plane.
    candidates = [np.broadcast_to(point, body.shape) for point in face]
    for i, a in enumerate(face):
        for b in face[i + 1 :]:
            edge = b - a
            if edge @ edge <= 1e-20:
                continue
            fraction = np.clip(((body - a) @ edge) / (edge @ edge), 0, 1)
            candidates.append(a + fraction[:, None] * edge)
    support = np.stack(candidates, axis=1)
    delta = support - body[:, None]
    parallel = delta @ normal
    perpendicular = np.linalg.norm(delta - parallel[..., None] * normal, axis=2)
    lower = parallel * np.cos(cone_rad) - perpendicular * np.sin(cone_rad)
    opposite_inside = -parallel >= np.linalg.norm(delta, axis=2) * np.cos(cone_rad)
    lower[opposite_inside] = -np.linalg.norm(delta[opposite_inside], axis=1)
    # For a triangle interior interpolate its three chosen support points. Their
    # convex combination is still in the same real distal convex mesh, so the
    # vertex minima bound the complete clipped body, not just sampled vertices.
    return float(lower.max(1).min())
