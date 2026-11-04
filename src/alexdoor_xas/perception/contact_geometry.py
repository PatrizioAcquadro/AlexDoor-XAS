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


def observed_front_extent(surface, views, config):
    """Observed conservative front envelope for the fixed pilot rigid target.

    All non-robot RGB-D returns in the root's finite tangent extent contribute,
    including handles/relief omitted by a planar semantic mask. A visible gap
    does not prove that a foreground component is mechanically detached.
    This bounded geometric model requires independent coverage validation on
    the pilot domain; it does not certify arbitrary hidden attachments or new objects.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    from alexdoor_xas.perception.geometry import deproject

    basis = surface.basis
    lower, upper = surface.bounds
    clouds, clocks = [], []
    for view in views:
        depth = np.asarray(view["depth_m"])[..., 0]
        valid = np.asarray(view["valid_depth"])[..., 0]
        v, u = np.nonzero(valid & np.isfinite(depth) & (depth > 0))
        points = deproject(depth[v, u], np.c_[u, v], view["intrinsics"], view["camera_world"])
        local = points @ basis
        inset = 2 * config["plane_tolerance_m"]
        selected = ((local[:, 1:] >= lower[1:] + inset) & (local[:, 1:] <= upper[1:] - inset)).all(
            1
        )
        if selected.any():
            clouds.append(points[selected])
            clocks.append((float(view["time_s"]), float(view["available_s"])))
    if not clouds:
        raise ValueError("missing_observed_target_front_extent")
    cloud = np.concatenate(clouds)
    # Keep each voxel's foremost measured return. All foreground in the finite
    # root extent remains potentially attached until independently excluded;
    # a handle's connection may be hidden even when its front is observed.
    order = np.argsort(cloud @ surface.normal)
    _, ids = np.unique(
        np.floor(cloud[order] / config["voxel_m"]).astype(np.int64), axis=0, return_index=True
    )
    points = cloud[order[ids]]
    pairs = cKDTree(points).query_pairs(config["association_distance_m"], output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points))
    )
    _, labels = connected_components(graph, directed=False)
    coordinate = points @ surface.normal
    seeds = abs(coordinate - surface.offset) <= config["plane_tolerance_m"]
    if seeds.sum() < config["min_points"]:
        raise ValueError("missing_observed_front_extent_root")
    connected = np.isin(labels, np.unique(labels[seeds]))
    measured = max(0.0, float(surface.offset - coordinate.min()))
    bound = 1.25 * measured + 0.01 + config["plane_tolerance_m"]
    protruding = surface.offset - coordinate > 2 * config["plane_tolerance_m"]
    tangent_lower = tangent_upper = None
    if protruding.any():
        tangent = (points[protruding] @ basis)[:, 1:]
        reserve = (
            config["association_distance_m"]
            + config["plane_tolerance_m"]
            + 0.01
            + 0.25 * np.ptp(tangent, axis=0)
        )
        tangent_lower = (tangent.min(0) - reserve).tolist()
        tangent_upper = (tangent.max(0) + reserve).tolist()
    model = dict(
        protrusion_tangent_lower_m=tangent_lower,
        protrusion_tangent_upper_m=tangent_upper,
        observed_protrusion_voxels=int(protruding.sum()),
        front_extent_m=bound,
        observed_extent_m=measured,
        source_clocks=clocks,
        observed_voxels=len(points),
        rigid_connected_voxels=int(connected.sum()),
    )

    enabled = config.get("front_height_layers", False)
    if type(enabled) is not bool:
        raise ValueError("invalid_front_height_layers_option")
    if enabled:
        # Every level keeps the ORIGINAL global height reserve and tangent pad.
        # All higher returns, including disconnected foreground, remain present.
        layers = []
        if protruding.any():
            heights = surface.offset - coordinate
            tangent = (points @ basis)[:, 1:]
            height_reserve = bound - measured
            for level in np.arange(config["voxel_m"], measured, config["voxel_m"]):
                higher = heights > level
                layers.append(
                    dict(
                        observed_height_level_m=float(level),
                        front_extent_m=float(level + height_reserve),
                        higher_return_voxels=int(higher.sum()),
                        lower_m=(tangent[higher].min(0) - reserve).tolist(),
                        upper_m=(tangent[higher].max(0) + reserve).tolist(),
                    )
                )
        model["height_layers"] = layers
        model["height_layer_tangent_domain_m"] = [lower[1:].tolist(), upper[1:].tolist()]
    return model
